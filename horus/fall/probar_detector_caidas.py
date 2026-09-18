# -*- coding: utf-8 -*-
"""
probar_detector_caidas.py · HORUS — el adaptador de caídas, sin cámara y sin GPU.

    python probar_detector_caidas.py
    python probar_detector_caidas.py --detalle
    python probar_detector_caidas.py --caso latch

Por qué se puede probar sin torch ni mediapipe
----------------------------------------------
`DetectorCaidas` recibe sus dos backends por parámetro. Acá se le inyectan:

  - una **pose guionada**: devuelve un esqueleto de 33 puntos con el torso al
    ángulo que pida el escenario, así se puede escribir "de pie 5 s, se cae en
    0,3 s, queda en el piso 30 s" y que sea repetible.
  - un **clasificador geométrico**: P(caída) = 1 − verticalidad del torso en el
    último frame del clip. No es el ST-GCN, pero se comporta como él en lo
    único que estos casos miran: sube cuando el torso se acuesta.

Lo que se verifica, entonces, no es la calidad del modelo — eso ya está medido
en `resultados_cv_subclases.txt` (bal_acc 89,0%, recall caída 89,4%, leave-one-
group-out) — sino el **cableado**: que el buffer sea por persona, que las
puertas de pico y verticalidad hagan lo que dicen, que la etiqueta se sostenga
mientras la persona no se levanta, y que la fusión termine abriendo el evento.

Un escenario de 30 segundos tarda milisegundos y no depende de que alguien se
tire al piso delante de una cámara.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
for _p in (_AQUI,
           os.path.join(_RAIZ, "05_tracking"),
           os.path.join(_RAIZ, "05_tracking", "local"),
           os.path.join(_RAIZ, "06_fusion_decision")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contratos import (  # noqa: E402
    CABEZA_ACCION, CABEZA_OBJETOS, CABEZA_POSE, ObservacionCamara, Severidad,
)
from detector_caidas import (  # noqa: E402
    IDX_CD, IDX_CI, IDX_HD, IDX_HI, IDX_TORSO, MP_A_COCO,
    ConfigCaidas, DetectorCaidas, kp_a_coco_px, normalizar_frame,
)
from motor_fusion import ConfigFusion, MotorFusion  # noqa: E402
from tracker_local import TrackLocal  # noqa: E402

FPS = 10.0
DT = 1.0 / FPS
ALTO, ANCHO = 1080, 1920
T0 = 1_700_000_000.0


# --------------------------------------------------------------------------- #
# Esqueleto sintético
# --------------------------------------------------------------------------- #
def esqueleto(theta_grados: float, paso: int = 0, visibilidad: float = 0.9,
              jitter: float = 0.005) -> np.ndarray:
    """(33, 4) como los devuelve MediaPipe: x, y normalizados al recorte.

    `theta` es el ángulo del torso contra la vertical: 0° de pie, 90° tirado.
    El jitter es una oscilación suave, no ruido blanco: una persona quieta
    frente a una cámara real nunca da velocidad exactamente cero, y con cero
    exacto el baseline de velocidad se queda clavado en 0 y ningún pico se
    puede medir nunca (`baseline > 1e-6` no se cumple jamás). Suave y no
    aleatorio para que no aparezcan picos espurios y el test sea repetible.
    """
    th = np.radians(float(theta_grados))
    d = np.array([np.sin(th), -np.cos(th)])          # dirección cadera->hombro
    perp = np.array([-d[1], d[0]])
    cad = np.array([0.5, 0.6])
    largo_torso, medio_ancho = 0.25, 0.06

    kp = np.zeros((33, 4), dtype=np.float32)
    kp[:, 3] = float(visibilidad)
    hom = cad + largo_torso * d
    kp[IDX_CI, :2] = cad + perp * medio_ancho
    kp[IDX_CD, :2] = cad - perp * medio_ancho
    kp[IDX_HI, :2] = hom + perp * medio_ancho
    kp[IDX_HD, :2] = hom - perp * medio_ancho
    for i in range(33):
        if i in IDX_TORSO:
            continue
        kp[i, :2] = (cad + d * (largo_torso * (0.15 + 0.035 * i))
                     + perp * (0.012 * ((i % 5) - 2)))
    if jitter:
        fases = np.arange(33) * 0.37
        kp[:, 0] += jitter * np.sin(paso * 0.8 + fases)
        kp[:, 1] += jitter * np.cos(paso * 0.8 + fases)
    return kp


def verticalidad_de(kp33: np.ndarray) -> float:
    n = normalizar_frame(kp33)
    return -float((n[IDX_HI, 1] + n[IDX_HD, 1]) / 2.0)


# --------------------------------------------------------------------------- #
# Backends de mentira
# --------------------------------------------------------------------------- #
class PoseGuionada:
    """Devuelve la pose que el guion diga para ESA persona.

    El recorte trae la identidad pintada en el píxel del centro: el detector
    solo le pasa una imagen, así que es la forma de que un backend sintético
    sepa a quién está mirando sin cambiar la interfaz real.
    """

    def __init__(self) -> None:
        self.guion: Dict[int, List[Optional[Tuple[float, float]]]] = {}
        self.paso = 0
        self.llamadas = 0

    def __call__(self, recorte: np.ndarray) -> Optional[np.ndarray]:
        self.llamadas += 1
        tid = int(recorte[recorte.shape[0] // 2, recorte.shape[1] // 2, 0])
        sec = self.guion.get(tid)
        if not sec:
            return None
        item = sec[min(self.paso, len(sec) - 1)]
        if item is None:
            return None
        theta, vis = item
        return esqueleto(theta, paso=self.paso, visibilidad=vis)


class ClfGeometrico:
    """P(caída) = 1 − verticalidad del último frame del clip.

    Se comporta como el ST-GCN en lo que estos casos miden y no necesita
    torch. Cuenta las llamadas para poder verificar que las personas del
    frame se clasifican en un solo forward.
    """

    def __init__(self) -> None:
        self.llamadas = 0
        self.tam_lote: List[int] = []

    def __call__(self, clips: np.ndarray) -> np.ndarray:
        self.llamadas += 1
        self.tam_lote.append(int(clips.shape[0]))
        v = -(clips[:, -1, IDX_HI, 1] + clips[:, -1, IDX_HD, 1]) / 2.0
        return np.clip(1.0 - v, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Fábricas
# --------------------------------------------------------------------------- #
def frame_con(cajas: Dict[int, Sequence[float]]) -> np.ndarray:
    """Un frame negro con cada caja pintada del valor de su `track_id`."""
    f = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    for tid, (x1, y1, x2, y2) in cajas.items():
        f[int(y1):int(y2), int(x1):int(x2)] = np.uint8(tid)
    return f


def track(tid: int, caja: Sequence[float], ts: float, *,
          clase: str = "persona", quieto: bool = False, quieto_s: float = 0.0,
          estado: str = "confirmado", cam: str = "cam-deposito",
          visto_s: float = 6.0) -> TrackLocal:
    return TrackLocal(
        track_id=tid, clase=clase, class_id=0, bbox_xyxy=tuple(caja),
        score=0.7, camera_id=cam, frame_idx=int((ts - T0) * FPS), ts=ts,
        estado=estado, hits=int(visto_s * FPS), edad=int(visto_s * FPS),
        frames_sin_ver=0, ts_nacimiento=ts - visto_s, visto_s=visto_s,
        quieto=quieto, quieto_s=quieto_s)


def caja_para(theta: float, tid: int) -> Tuple[float, float, float, float]:
    """Caja de persona coherente con la postura: parada es alta y angosta,
    tirada es baja y ancha. Importa porque el recorte cambia de forma y el
    modelo ve otra distorsión de aspecto."""
    cx = 400.0 + 500.0 * (tid % 3)
    cy = 700.0
    de_pie = (160.0, 360.0)
    tirado = (360.0, 160.0)
    f = np.sin(np.radians(theta))
    w = de_pie[0] + (tirado[0] - de_pie[0]) * f
    h = de_pie[1] + (tirado[1] - de_pie[1]) * f
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def detector(**kw: Any) -> Tuple[DetectorCaidas, PoseGuionada, ClfGeometrico]:
    pose, clf = PoseGuionada(), ClfGeometrico()
    cfg = ConfigCaidas(**kw)
    return DetectorCaidas(cfg, backend_pose=pose, clasificador=clf,
                          verboso=False), pose, clf


def correr(det: DetectorCaidas, pose: PoseGuionada,
           guion: Dict[int, List[Optional[Tuple[float, float]]]],
           n: int, *, cam: str = "cam-deposito",
           quieto_desde: Optional[Dict[int, int]] = None,
           ) -> List[Tuple[float, List[Any], List[Any]]]:
    """Corre `n` frames y devuelve (ts, tracks, acciones) por frame."""
    pose.guion = guion
    quieto_desde = quieto_desde or {}
    salida = []
    for i in range(n):
        ts = T0 + i * DT
        pose.paso = i
        cajas, tracks = {}, []
        for tid, sec in guion.items():
            item = sec[min(i, len(sec) - 1)]
            if item is None:
                continue
            theta = item[0]
            c = caja_para(theta, tid)
            cajas[tid] = c
            desde = quieto_desde.get(tid)
            q = desde is not None and i >= desde
            tracks.append(track(tid, c, ts, quieto=q,
                                quieto_s=(i - desde) * DT if q else 0.0,
                                cam=cam))
        acs = det.procesar(cam, frame_con(cajas), tracks, ts=ts)
        salida.append((ts, tracks, acs))
    return salida


def guion_caida(n_pie: int, n_transicion: int, n_piso: int,
                theta_final: float = 88.0) -> List[Tuple[float, float]]:
    sec: List[Tuple[float, float]] = [(2.0, 0.9)] * n_pie
    for k in range(1, n_transicion + 1):
        sec.append((2.0 + (theta_final - 2.0) * k / n_transicion, 0.9))
    sec += [(theta_final, 0.9)] * n_piso
    return sec


def etiquetas(salida, tid: int) -> List[str]:
    out = []
    for _, _, acs in salida:
        for a in acs:
            if a.track_id == tid:
                out.append(a.accion)
    return out


# --------------------------------------------------------------------------- #
# Casos
# --------------------------------------------------------------------------- #
def caso_de_pie_no_reporta() -> Tuple[bool, str]:
    """Una persona parada seis segundos no es una caída."""
    det, pose, _ = detector()
    sal = correr(det, pose, {1: [(2.0, 0.9)] * 60}, 60)
    et = etiquetas(sal, 1)
    ok = et and not any(e == "caida" for e in et)
    return ok, f"{len(et)} observaciones, ninguna etiquetada caída"


def caso_cabeza_siempre_declarada() -> Tuple[bool, str]:
    """Aunque no pase nada, cada persona mirada emite una `AccionObs`.

    Es el punto de todo el diseño: sin esto, un frame tranquilo no declara la
    cabeza, `ReglaCaida` figura dormida, y "no hubo caídas" se vuelve
    indistinguible de "no hay modelo de caídas".
    """
    det, pose, _ = detector()
    sal = correr(det, pose, {1: [(2.0, 0.9)] * 40}, 40)
    fuentes = {a.fuente for _, _, acs in sal for a in acs}
    todos = all(len(acs) == 1 for _, _, acs in sal)
    ok = todos and fuentes <= {CABEZA_ACCION, CABEZA_POSE} and fuentes
    return ok, f"1 obs/frame en {len(sal)} frames, fuentes {sorted(fuentes)}"


def caso_caida_dispara() -> Tuple[bool, str]:
    """De pie, se cae en 0,3 s, queda en el piso. Tiene que etiquetar."""
    det, pose, _ = detector()
    sal = correr(det, pose, {1: guion_caida(60, 3, 120)}, 183,
                 quieto_desde={1: 64})
    et = etiquetas(sal, 1)
    n = sum(1 for e in et if e == "caida")
    ok = n > 0 and det.stats["reportes"] > 0
    return ok, f"{n}/{len(et)} observaciones etiquetadas caída"


def caso_sentarse_no_dispara() -> Tuple[bool, str]:
    """Sentarse en el piso: el torso queda casi vertical.

    Dos puertas lo frenan y a propósito son independientes: la verticalidad
    no baja lo suficiente para validar el pico, y la probabilidad tampoco
    llega al umbral.
    """
    det, pose, _ = detector()
    guion = [(2.0, 0.9)] * 60 + [(25.0, 0.9)] * 120
    sal = correr(det, pose, {1: guion}, 180, quieto_desde={1: 62})
    et = etiquetas(sal, 1)
    ok = not any(e == "caida" for e in et)
    return ok, f"{len(et)} observaciones, 0 caídas (sentado a 25°)"


def caso_acostarse_despacio_no_dispara() -> Tuple[bool, str]:
    """Acostarse a propósito, despacio: no hay pico de velocidad.

    Es la diferencia física entre caerse y acostarse, y es lo único que separa
    las dos cosas cuando la postura final es idéntica — el clasificador solo
    diría caída en los dos casos.

    El margen NO es holgado, y conviene saberlo antes de la prueba de campo.
    El pico se mide contra un baseline que, con la persona quieta, es puro
    jitter de los landmarks. Con el jitter de este escenario el baseline queda
    en ~0,016 por frame; un descenso de 12 s da ~0,012 (no dispara) pero uno de
    4 s da ~0,037, que son 2,3x — a un pelo del umbral de 3,0. O sea que
    `factor_pico_velocidad` hay que calibrarlo contra el jitter real de la
    instalación, no dejarlo en el default y confiar.
    """
    det, pose, _ = detector()
    guion = ([(2.0, 0.9)] * 60
             + [(2.0 + 86.0 * k / 120, 0.9) for k in range(1, 121)]
             + [(88.0, 0.9)] * 100)
    sal = correr(det, pose, {1: guion}, 280, quieto_desde={1: 182})
    et = etiquetas(sal, 1)
    n = sum(1 for e in et if e == "caida")
    ok = n == 0 and det.stats["bloqueadas_sin_pico"] > 0
    return ok, (f"{n} caídas, {det.stats['bloqueadas_sin_pico']} bloqueadas "
                f"por falta de pico")


def caso_latch_sostiene() -> Tuple[bool, str]:
    """A los 25 s de estar en el piso la etiqueta sigue puesta.

    Si se exigiera un pico reciente en cada frame, la etiqueta se apagaría a
    los 4 s, el evento cerraría a los 5 s de silencio, y la severidad nunca
    llegaría a CRÍTICO — que es justo lo que `ReglaCaida` promete a los 15 s
    sin levantarse.
    """
    det, pose, _ = detector()
    n = 60 + 3 + 250                       # 25 s en el piso
    sal = correr(det, pose, {1: guion_caida(60, 3, 250)}, n,
                 quieto_desde={1: 64})
    ultimas = etiquetas(sal, 1)[-20:]
    ok = all(e == "caida" for e in ultimas)
    return ok, f"últimas 20 observaciones: {ultimas.count('caida')} caída"


def caso_se_levanta_apaga() -> Tuple[bool, str]:
    """Se cae, se levanta: la etiqueta se apaga sola."""
    det, pose, _ = detector()
    guion = guion_caida(60, 3, 100) + [(2.0, 0.9)] * 60
    sal = correr(det, pose, {1: guion}, len(guion), quieto_desde={1: 64})
    et = etiquetas(sal, 1)
    ok = "caida" in et and all(e != "caida" for e in et[-20:])
    return ok, f"caída detectada y apagada al levantarse ({len(et)} obs)"


def caso_dos_personas_no_se_mezclan() -> Tuple[bool, str]:
    """Una se cae, la otra camina. El buffer es por `track_id`.

    Con `num_poses=1` sobre el frame entero — como hacía el script de webcam —
    esto no se podía ni plantear: había una sola persona por cámara, y la que
    se cae suele no ser la de mayor confianza.
    """
    det, pose, _ = detector()
    guion = {1: guion_caida(60, 3, 120), 2: [(2.0, 0.9)] * 183}
    sal = correr(det, pose, guion, 183, quieto_desde={1: 64})
    e1, e2 = etiquetas(sal, 1), etiquetas(sal, 2)
    ok = any(e == "caida" for e in e1) and not any(e == "caida" for e in e2)
    return ok, (f"track 1: {e1.count('caida')} caídas · "
                f"track 2: {e2.count('caida')} caídas")


def caso_lote_unico_por_frame() -> Tuple[bool, str]:
    """Tres personas listas se clasifican en un solo forward.

    Misma lección que el rebuild de la galería en `tracker_global`: lo que se
    puede juntar por frame se junta. Tres forwards de lote 1 pagan tres veces
    el overhead de lanzamiento, que en una red de este tamaño es casi todo el
    costo.
    """
    det, pose, clf = detector()
    guion = {1: [(2.0, 0.9)] * 60, 2: [(2.0, 0.9)] * 60, 3: [(2.0, 0.9)] * 60}
    correr(det, pose, guion, 60)
    ok = bool(clf.tam_lote) and max(clf.tam_lote) == 3
    return ok, f"lotes {sorted(set(clf.tam_lote))}, {clf.llamadas} forwards"


def caso_persona_chica_se_descarta() -> Tuple[bool, str]:
    """Una persona de 40 px de alto no se mira.

    MediaPipe devuelve 33 landmarks igual, pero son inventados, y una pose
    inventada no se queda quieta: entra al buffer y contamina 32 frames.
    Misma regla que el filtro de calidad del ReID — es preferible no opinar.
    """
    det, pose, _ = detector()
    pose.guion = {7: [(2.0, 0.9)] * 40}
    caja = (900.0, 700.0, 920.0, 740.0)          # 40 px de alto
    acs_total = 0
    for i in range(40):
        pose.paso = i
        ts = T0 + i * DT
        acs = det.procesar("cam-deposito", frame_con({7: caja}),
                           [track(7, caja, ts)], ts=ts)
        acs_total += len(acs)
    ok = acs_total == 0 and det.stats["descarte_chica"] == 40
    return ok, f"{det.stats['descarte_chica']} descartes, {acs_total} observaciones"


def caso_visibilidad_baja_se_descarta() -> Tuple[bool, str]:
    """Un recorte cortado por el borde devuelve landmarks con visibility baja.

    La pose sale igual y es geométricamente falsa. Se tira antes de que entre
    al buffer.
    """
    det, pose, _ = detector()
    sal = correr(det, pose, {1: [(88.0, 0.2)] * 60}, 60, quieto_desde={1: 0})
    et = etiquetas(sal, 1)
    ok = (det.stats["descarte_visibilidad"] == 60
          and not any(e == "caida" for e in et))
    return ok, f"{det.stats['descarte_visibilidad']} descartes por visibilidad"


def caso_hueco_largo_reinicia() -> Tuple[bool, str]:
    """Un segundo sin pose confiable: se corta, no se sigue con pose vieja.

    Arrastrar la última pose durante un hueco largo es clasificar un frame
    congelado. Y el pico viejo tiene que morir con él: si sobreviviera, al
    reaparecer validaría una alarma con un movimiento de hace un minuto.
    """
    det, pose, _ = detector()
    guion = guion_caida(60, 3, 40) + [None] * 12 + [(88.0, 0.9)] * 40
    sal = correr(det, pose, {1: guion}, len(guion), quieto_desde={1: 64})
    et = etiquetas(sal, 1)
    tras_hueco = et[-30:]
    ok = "caida" in et[:110] and not any(e == "caida" for e in tras_hueco[:5])
    return ok, f"cortó en el hueco y volvió a llenar buffer ({len(et)} obs)"


def caso_remapeo_coco() -> Tuple[bool, str]:
    """MediaPipe 33 -> COCO 17, y de recorte a píxeles del frame.

    Los dos pasos van juntos porque separarlos deja keypoints en [0,1]
    conviviendo con una `bbox_xyxy` en píxeles, y la regla los compara.
    """
    kp = esqueleto(88.0, jitter=0.0)
    caja = (100.0, 200.0, 500.0, 400.0)
    coco = kp_a_coco_px(kp, caja)
    esperado = [(0, 0), (5, 11), (6, 12), (11, 23), (12, 24), (16, 28)]
    bien = all(
        abs(coco[ic, 0] - (caja[0] + kp[imp, 0] * (caja[2] - caja[0]))) < 1e-3
        and abs(coco[ic, 1] - (caja[1] + kp[imp, 1] * (caja[3] - caja[1]))) < 1e-3
        for ic, imp in esperado)
    dentro = (coco[:, 0].min() >= caja[0] - 60 and coco[:, 0].max() <= caja[2] + 60
              and coco[:, 1].min() >= caja[1] - 60 and coco[:, 1].max() <= caja[3] + 60)
    ok = coco.shape == (17, 3) and bien and dentro and len(MP_A_COCO) == 17
    return ok, f"(17,3) en píxeles del frame, {len(esperado)} puntos verificados"


def caso_calentamiento_manda_pose() -> Tuple[bool, str]:
    """Mientras el ST-GCN llena su ventana (3,2 s a 10 FPS) manda keypoints.

    Alguien que entra a cuadro ya tirado en el piso es exactamente el caso que
    no se puede perder esperando 32 frames.
    """
    det, pose, _ = detector()
    sal = correr(det, pose, {1: [(88.0, 0.9)] * 40}, 40, quieto_desde={1: 0})
    primeras = [a for _, _, acs in sal[:20] for a in acs]
    ultimas = [a for _, _, acs in sal[35:] for a in acs]
    ok = (primeras and all(a.fuente == CABEZA_POSE
                           and a.keypoints is not None
                           and a.keypoints.shape == (17, 3) for a in primeras)
          and ultimas and all(a.fuente == CABEZA_ACCION for a in ultimas))
    return ok, (f"{len(primeras)} obs de pose en calentamiento, "
                f"después {len(ultimas)} de clasificador")


def caso_calentamiento_apagable() -> Tuple[bool, str]:
    """Con `emitir_en_calentamiento=False` no se manda pose, pero la cabeza
    se sigue declarando: la observación existe, vacía."""
    det, pose, _ = detector(emitir_en_calentamiento=False)
    sal = correr(det, pose, {1: [(88.0, 0.9)] * 20}, 20, quieto_desde={1: 0})
    acs = [a for _, _, x in sal for a in x]
    ok = (acs and all(a.fuente == CABEZA_ACCION and a.keypoints is None
                      and a.accion == "" for a in acs))
    return ok, f"{len(acs)} observaciones vacías, cabeza declarada igual"


def caso_integracion_fusion() -> Tuple[bool, str]:
    """El camino entero: detector -> AccionObs -> ReglaCaida -> Evento.

    Y `reglas_dormidas()` ya no tiene que nombrar a `caida`: la cabeza existe.
    """
    det, pose, _ = detector()
    motor = MotorFusion(cfg=ConfigFusion(), fps=FPS)
    guion = {1: guion_caida(60, 3, 250)}
    pose.guion = guion
    eventos = []
    for i in range(313):
        ts = T0 + i * DT
        pose.paso = i
        theta = guion[1][min(i, len(guion[1]) - 1)][0]
        caja = caja_para(theta, 1)
        quieto = i >= 64
        t = track(1, caja, ts, quieto=quieto,
                  quieto_s=(i - 64) * DT if quieto else 0.0)
        acs = det.procesar("cam-deposito", frame_con({1: caja}), [t], ts=ts)
        cabezas = {CABEZA_OBJETOS, CABEZA_ACCION} | {a.fuente for a in acs}
        obs = ObservacionCamara(camera_id="cam-deposito", frame_idx=i, ts=ts,
                                tracks=[t], acciones=acs,
                                tam_frame=(ALTO, ANCHO),
                                cabezas=frozenset(cabezas))
        eventos.extend(motor.procesar([obs], ts=ts))

    caidas = [e for e in eventos if e.tipo == "caida"]
    ids = {e.evento_id for e in caidas}
    peor = max((e.severidad for e in caidas), default=Severidad.INFO)
    dormidas = motor.reglas_dormidas()
    ok = (len(ids) == 1 and peor == Severidad.CRITICO
          and "caida" not in dormidas)
    return ok, (f"{len(ids)} evento(s), severidad {peor.etiqueta}, "
                f"dormidas: {sorted(dormidas) or 'ninguna'}")


def _pipeline_simulado(det: Any) -> Any:
    """`PipelineHorus` con el motor simulado del servicio, para probar el
    cableado sin GPU ni pesos."""
    sys.path.insert(0, os.path.join(_RAIZ, "00_servicio"))
    from pipeline import PipelineHorus                      # noqa: E402
    from servicio import MotorSimulado                      # noqa: E402
    return PipelineHorus(motor_objetos=MotorSimulado(FPS), fps=FPS,
                         detector_caidas=det, verboso=False)


class PoseFija:
    """Siempre la misma pose, sin importar a quién mire. Para los casos donde
    lo que se prueba es el cableado y no la postura."""

    def __init__(self, theta: float = 2.0) -> None:
        self.theta, self.paso, self.llamadas = theta, 0, 0

    def __call__(self, recorte: np.ndarray) -> np.ndarray:
        self.llamadas += 1
        self.paso += 1
        return esqueleto(self.theta, paso=self.paso)


def caso_pipeline_cablea() -> Tuple[bool, str]:
    """`PipelineHorus(detector_caidas=...)` lo corre adentro, con los tracks.

    No se puede inyectar desde afuera: el recorte de cada persona sale del
    `TrackLocal`, y quien llama a `procesar()` todavía no tiene los tracks.
    """
    pose, clf = PoseFija(), ClfGeometrico()
    det = DetectorCaidas(ConfigCaidas(), backend_pose=pose, clasificador=clf,
                         verboso=False)
    pipe = _pipeline_simulado(det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(40):
        pipe.procesar({"cam-1": frame}, ts=T0 + i * DT)
    ok = det.stats["frames"] == 40 and det.stats["personas"] > 0
    return ok, (f"{det.stats['frames']} frames, {det.stats['personas']} "
                f"personas miradas, {pose.llamadas} poses")


def caso_pipeline_declara_sin_personas() -> Tuple[bool, str]:
    """Un frame sin una sola persona mirada NO deja la regla dormida.

    Es el caso que obliga a declarar la cabeza en el pipeline y no a partir de
    las `AccionObs` emitidas: acá el detector no emite ni una (todas las
    personas quedan por debajo del alto mínimo) y aun así `caida` tiene que
    estar despierta. Si no, "no hubo caídas" y "no hay modelo de caídas"
    vuelven a ser lo mismo.
    """
    pose, clf = PoseFija(), ClfGeometrico()
    det = DetectorCaidas(ConfigCaidas(min_alto_px=10_000),
                         backend_pose=pose, clasificador=clf, verboso=False)
    pipe = _pipeline_simulado(det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(30):
        pipe.procesar({"cam-1": frame}, ts=T0 + i * DT)
    dormidas = pipe.fusion.reglas_dormidas()
    ok = ("caida" not in dormidas and det.stats["descarte_chica"] > 0
          and pose.llamadas == 0)
    return ok, (f"0 observaciones emitidas, caida despierta · "
                f"dormidas: {sorted(dormidas) or 'ninguna'}")


CASOS = [
    ("de_pie", caso_de_pie_no_reporta),
    ("cabeza_declarada", caso_cabeza_siempre_declarada),
    ("caida", caso_caida_dispara),
    ("sentarse", caso_sentarse_no_dispara),
    ("acostarse", caso_acostarse_despacio_no_dispara),
    ("latch", caso_latch_sostiene),
    ("levantarse", caso_se_levanta_apaga),
    ("dos_personas", caso_dos_personas_no_se_mezclan),
    ("lote", caso_lote_unico_por_frame),
    ("persona_chica", caso_persona_chica_se_descarta),
    ("visibilidad", caso_visibilidad_baja_se_descarta),
    ("hueco", caso_hueco_largo_reinicia),
    ("remapeo", caso_remapeo_coco),
    ("calentamiento", caso_calentamiento_manda_pose),
    ("calentamiento_off", caso_calentamiento_apagable),
    ("fusion", caso_integracion_fusion),
    ("pipeline", caso_pipeline_cablea),
    ("pipeline_vacio", caso_pipeline_declara_sin_personas),
]


# --------------------------------------------------------------------------- #
# Medición del costo real (necesita mediapipe y torch)
# --------------------------------------------------------------------------- #
def medir(ruta_imagen: Optional[str] = None, repeticiones: int = 30,
          fps: float = 10.0) -> int:
    """Cuánto cuesta esta cabeza en ESTA máquina.

    Se separa del autotest porque no es una prueba: no hay nada que pueda dar
    OK o FALLA, es un número que depende del hardware. Y es el número que
    hace falta para rehacer la cuenta de `dimensionamiento-camaras-por-gpu`
    con la cabeza de caídas prendida.

    MediaPipe corre UNA VEZ POR PERSONA y es el costo dominante; el ST-GCN va
    batcheado y por eso se mide a varios tamaños de lote.
    """
    try:
        import cv2
        from detector_caidas import BackendPoseMediaPipe, ClasificadorSTGCN
    except ImportError as e:
        print(f"\n  falta una dependencia: {e}\n  pip install mediapipe torch "
              f"opencv-python\n")
        return 2

    cfg = ConfigCaidas()
    print(f"\n  midiendo en esta máquina · {repeticiones} repeticiones\n")

    if ruta_imagen:
        img = cv2.imread(ruta_imagen)
        if img is None:
            print(f"  no pude leer {ruta_imagen}")
            return 2
        origen = os.path.basename(ruta_imagen)
    else:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (360, 160, 3), dtype=np.uint8)
        origen = "recorte sintético"

    try:
        pose = BackendPoseMediaPipe(cfg.modelo_pose)
    except Exception as e:                                  # noqa: BLE001
        print(f"  no pude cargar MediaPipe: {type(e).__name__}: {e}")
        return 2

    kp = pose(img)                                          # calentar
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        kp = pose(img)
    ms_pose = (time.perf_counter() - t0) / repeticiones * 1000

    print(f"  pose (MediaPipe) sobre {origen} ({img.shape[1]}x{img.shape[0]}): "
          f"{ms_pose:.1f} ms por persona")
    if kp is None:
        print("  AVISO: no encontró ninguna persona en esa imagen. MediaPipe "
              "se saltea\n         la etapa de landmarks cuando no detecta "
              "nada, así que este número\n         es un PISO, no el costo "
              "real. Pasá --imagen con un recorte de\n         persona de "
              "verdad.")

    try:
        clf = ClasificadorSTGCN(cfg.checkpoint, canales=cfg.canales,
                                device=cfg.device)
    except Exception as e:                                  # noqa: BLE001
        print(f"\n  no pude cargar el ST-GCN: {type(e).__name__}: {e}")
        return 2

    print(f"\n  ST-GCN en {clf.device}:")
    ms_clf = {}
    for lote in (1, 2, 4, 8):
        clip = np.zeros((lote, cfg.ventana, 33, cfg.canales), dtype=np.float32)
        clf(clip)
        t0 = time.perf_counter()
        for _ in range(repeticiones):
            clf(clip)
        ms = (time.perf_counter() - t0) / repeticiones * 1000
        ms_clf[lote] = ms
        print(f"    lote {lote}: {ms:6.2f} ms  ({ms / lote:5.2f} ms por persona)")

    print(f"\n  costo por cámara y segundo, a {fps:.0f} FPS:")
    print(f"    {'personas':>10}  {'pose':>9}  {'ST-GCN':>9}  {'total':>9}")
    for n in (1, 2, 4):
        # el ST-GCN corre 1 de cada `paso` frames; la pose, todos
        p = n * ms_pose * fps
        c = ms_clf.get(min(n, 8), ms_clf[1]) * fps / cfg.paso
        print(f"    {n:>10}  {p:>7.0f} ms  {c:>7.0f} ms  {p + c:>7.0f} ms")
    print(f"\n  1000 ms por segundo es un núcleo entero. Con `max_personas="
          f"{cfg.max_personas}`\n  el techo por cámara es la fila de arriba "
          f"del todo multiplicada por {cfg.max_personas}.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caso", default=None, help="corre uno solo")
    ap.add_argument("--detalle", action="store_true")
    ap.add_argument("--medir", action="store_true",
                    help="medir el costo real de esta cabeza en esta máquina "
                         "(necesita mediapipe y torch)")
    ap.add_argument("--imagen", help="recorte de una persona, para que la "
                                     "medición de pose sea la real")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    if args.medir:
        return medir(args.imagen, fps=args.fps)

    casos = [c for c in CASOS if args.caso is None or c[0] == args.caso]
    if not casos:
        print(f"no existe el caso '{args.caso}'. Hay: "
              f"{', '.join(n for n, _ in CASOS)}")
        return 2

    print(f"\n{'':2}{'caso':<20}{'':2}detalle")
    print("  " + "-" * 76)
    bien = 0
    t0 = time.perf_counter()
    for nombre, fn in casos:
        try:
            ok, detalle = fn()
        except Exception as e:                       # noqa: BLE001
            ok, detalle = False, f"EXCEPCIÓN {type(e).__name__}: {e}"
            if args.detalle:
                import traceback
                traceback.print_exc()
        bien += bool(ok)
        print(f"  {'OK' if ok else 'FALLA':<6}{nombre:<20}{detalle}")

    ms = (time.perf_counter() - t0) * 1000
    print("  " + "-" * 76)
    print(f"  {bien}/{len(casos)} en {ms:.0f} ms\n")
    return 0 if bien == len(casos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
