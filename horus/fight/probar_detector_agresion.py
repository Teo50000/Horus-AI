# -*- coding: utf-8 -*-
"""
probar_detector_agresion.py · autotest del adaptador de agresión.

**Sin torch, sin pesos, sin GPU y sin cámara.** El clasificador se inyecta,
igual que en la suite de caídas. Que esta suite no necesite torch no es
comodidad: es la propiedad que hace que toda la capa de decisión se pueda
probar en cualquier máquina, y agregar una cabeza no la podía romper.

No se verifica la calidad del modelo —eso está medido: 88,75 % de balanced
accuracy sobre RWF-2000— sino el **cableado**: el muestreo por tiempo, la
compuerta, el descarte del clip viejo, que dos cámaras no se mezclen, la caja
de participantes, y que la fusión termine abriendo un evento.

    python probar_detector_agresion.py
    python probar_detector_agresion.py --caso compuerta
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
for _p in (_AQUI, os.path.join(_AQUI, "src"),
           os.path.join(_RAIZ, "06_fusion_decision"),
           os.path.join(_RAIZ, "05_tracking"),
           os.path.join(_RAIZ, "05_tracking", "local")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contratos import (  # noqa: E402
    CABEZA_ACCION, CABEZA_OBJETOS, ObservacionCamara,
)
from detector_agresion import (  # noqa: E402
    MEDIA_KINETICS, STD_KINETICS, ConfigAgresion, DetectorAgresion,
)
from motor_fusion import MotorFusion  # noqa: E402
from tracker_local import TrackLocal  # noqa: E402


# --------------------------------------------------------------------------- #
# Fábricas
# --------------------------------------------------------------------------- #
ALTO, ANCHO = 480, 640


def persona(tid: int, x: float, alto: float = 200.0, ts: float = 0.0,
            cam: str = "cam-1", estado: str = "confirmado") -> TrackLocal:
    return TrackLocal(
        track_id=tid, clase="persona", class_id=0,
        bbox_xyxy=(x, 100.0, x + 80.0, 100.0 + alto), score=0.7,
        camera_id=cam, frame_idx=0, ts=ts, estado=estado,
        hits=40, edad=40, frames_sin_ver=0, ts_nacimiento=ts - 4.0,
        visto_s=4.0, radio_permanencia=5.0, quieto=False, quieto_s=0.0)


def frame(v: int = 128) -> np.ndarray:
    return np.full((ALTO, ANCHO, 3), v, dtype=np.uint8)


def det(prob: Callable[[np.ndarray], Sequence[float]] | float = 0.9,
        **kw: Any) -> DetectorAgresion:
    f = prob if callable(prob) else (lambda clips, _p=prob: [_p] * len(clips))
    return DetectorAgresion(ConfigAgresion(**kw), clasificador=f)


def correr(d: DetectorAgresion, pasos: int, fps: float, gente: int,
           cam: str = "cam-1", t0: float = 1000.0) -> List[Any]:
    """Alimenta `pasos` frames a `fps` con `gente` personas en cuadro."""
    salida = []
    for i in range(pasos):
        t = t0 + i / fps
        tracks = [persona(30 + k, 100.0 + 120 * k, ts=t, cam=cam)
                  for k in range(gente)]
        salida.append(d.procesar(cam, frame(), tracks, ts=t)[0])
    return salida


# --------------------------------------------------------------------------- #
# Casos
# --------------------------------------------------------------------------- #
def caso_emite_siempre() -> Tuple[bool, str]:
    """Una observación por cámara mirada SIEMPRE, aunque no haya nada.

    Si solo emitiera al ver una pelea, la regla figuraría dormida en los
    frames tranquilos y 'no hubo peleas' sería indistinguible de 'no hay
    modelo de peleas'."""
    d = det(0.05)
    acs = correr(d, 60, 10.0, gente=2)
    ok = (len(acs) == 60 and all(a.fuente == CABEZA_ACCION for a in acs)
          and all(a.accion == "" for a in acs))
    return ok, f"{len(acs)} observaciones, todas con acción vacía: {ok}"


def caso_compuerta() -> Tuple[bool, str]:
    """Con una sola persona no se clasifica; con dos, sí."""
    d1 = det(0.9)
    correr(d1, 80, 10.0, gente=1)
    d2 = det(0.9)
    acs = correr(d2, 80, 10.0, gente=2)
    solas = d1.resumen()["clasificaciones"]
    dos = d2.resumen()["clasificaciones"]
    etiquetadas = sum(1 for a in acs if a.accion == "pelea")
    ok = solas == 0 and dos > 0 and etiquetadas > 0
    return ok, f"1 persona: {solas} clasificaciones · 2 personas: {dos}, {etiquetadas} etiquetadas"


def caso_buffer_se_llena_con_uno() -> Tuple[bool, str]:
    """El clip se llena con UNA persona, así la segunda no cuesta 5 s más.

    Es la diferencia con `03_webcam.py`, que tira el buffer con la compuerta
    cerrada: ahí los 5 s que hay que esperar son justo los del principio de la
    pelea, que son los que importan."""
    d = det(0.9)
    correr(d, 80, 10.0, gente=1)            # 8 s con una sola persona
    antes = d.resumen()["listas"]
    acs = correr(d, 3, 10.0, gente=2, t0=1000.0 + 8.0)
    ok = antes == 1 and any(a.accion == "pelea" for a in acs)
    return ok, (f"clip lleno con 1 persona: {antes == 1} · "
                f"etiqueta al aparecer la 2ª: {any(a.accion == 'pelea' for a in acs)}")


def caso_muestreo_por_tiempo() -> Tuple[bool, str]:
    """El clip cubre ~5 s a 30 FPS y a 8 FPS. Si se muestreara por frame,
    a 30 FPS duraría 1 s y el modelo vería el movimiento acelerado."""
    medidos = {}
    for fps in (30.0, 8.0):
        d = det(0.9)
        correr(d, int(fps * 12), fps, gente=2)
        st = d._cams["cam-1"]
        medidos[fps] = (len(st.buffer) - 1) * d.cfg.intervalo_buffer()
    ok = all(abs(v - 5.0) < 0.35 for v in medidos.values())
    return ok, " · ".join(f"{k:g} FPS -> clip de {v:.2f} s" for k, v in medidos.items())


def caso_caja_es_la_union() -> Tuple[bool, str]:
    """La caja de la observación es la unión de los participantes, no el
    cuadro: es lo que le permite a la fusión saber quién peleó y, con eso,
    correlacionar con el arma de uno o la caída del otro."""
    d = det(0.9)
    acs = correr(d, 80, 10.0, gente=2)
    caja = acs[-1].bbox_xyxy
    esperada = (100.0, 100.0, 300.0, 300.0)
    ok = (tuple(round(v, 1) for v in caja) == esperada
          and caja[2] - caja[0] < ANCHO)
    return ok, f"caja {tuple(round(v) for v in caja)} (esperada {tuple(map(int, esperada))})"


def caso_personas_chicas_no_cuentan() -> Tuple[bool, str]:
    """Dos personas de 20 px son 2 px después del resize a 128: no abren la
    compuerta. Misma lección que el filtro de calidad del ReID — es preferible
    no opinar que opinar sobre ruido."""
    d = det(0.9)
    for i in range(80):
        t = 1000.0 + i * 0.1
        chicas = [persona(30 + k, 100.0 + 120 * k, alto=20.0, ts=t)
                  for k in range(2)]
        d.procesar("cam-1", frame(), chicas, ts=t)
    ok = d.resumen()["clasificaciones"] == 0
    return ok, f"{d.resumen()['clasificaciones']} clasificaciones (esperado 0)"


def caso_clip_viejo_se_tira() -> Tuple[bool, str]:
    """Si la cámara queda vacía más que la tolerancia, el clip se descarta.

    Mezclarlo con lo que venga después arma un clip mitad de hace un minuto y
    mitad de ahora — el mismo bug del hueco que apareció en caídas."""
    d = det(0.9)
    correr(d, 80, 10.0, gente=2)
    for i in range(40):                      # 4 s de cámara vacía
        t = 1000.0 + 8.0 + i * 0.1
        d.procesar("cam-1", frame(), [], ts=t)
    vacio = len(d._cams["cam-1"].buffer)
    ok = vacio == 0 and d.resumen()["clips_tirados"] >= 1
    return ok, f"buffer tras 4 s sin nadie: {vacio} frames, {d.resumen()['clips_tirados']} clip(s) tirado(s)"


def caso_camaras_no_se_mezclan() -> Tuple[bool, str]:
    """Dos cámaras, dos clips. Un estado global haría que la pelea de una
    contamine el clip de la otra."""
    d = det(lambda clips: [0.95 if c[..., 0].mean() > 0 else 0.0 for c in clips])
    for i in range(80):
        t = 1000.0 + i * 0.1
        g2 = [persona(30 + k, 100.0 + 120 * k, ts=t, cam="cam-A") for k in range(2)]
        g1 = [persona(40, 100.0, ts=t, cam="cam-B")]
        d.procesar("cam-A", frame(), g2, ts=t)
        d.procesar("cam-B", frame(), g1, ts=t)
    a = d._cams["cam-A"]
    b = d._cams["cam-B"]
    ok = a.etiquetado and not b.etiquetado and len(a.buffer) == len(b.buffer)
    return ok, f"cam-A etiquetada={a.etiquetado} · cam-B etiquetada={b.etiquetado}"


def caso_normalizacion() -> Tuple[bool, str]:
    """El clip sale con la forma y la normalización con las que se entrenó, y
    las constantes son las mismas que las de `modelo_fight.py`."""
    d = det(0.9)
    capturados: List[np.ndarray] = []
    d._clasificador = lambda clips: (capturados.append(clips), [0.9] * len(clips))[1]
    correr(d, 80, 10.0, gente=2)
    clip = capturados[0]
    forma_ok = clip.shape == (1, 32, 112, 112, 3)
    gris = (128.0 / 255.0 - np.asarray(MEDIA_KINETICS)) / np.asarray(STD_KINETICS)
    valor_ok = np.allclose(clip[0, 0, 0, 0], gris, atol=1e-5)
    try:
        from modelo_fight import MEDIA_KINETICS as M2, STD_KINETICS as S2
        constantes_ok = (np.allclose(MEDIA_KINETICS, M2)
                         and np.allclose(STD_KINETICS, S2))
        nota = "constantes verificadas contra modelo_fight"
    except Exception:                                # torch no instalado
        constantes_ok, nota = True, "modelo_fight no importable (sin torch)"
    ok = forma_ok and valor_ok and constantes_ok
    return ok, f"clip {clip.shape}, normalización ok={valor_ok} · {nota}"


def caso_checkpoint_incompleto() -> Tuple[bool, str]:
    """Un state_dict que no es de este modelo tiene que CORTAR, no cargarse a
    medias. Un modelo que carga sin excepción y devuelve probabilidades no es
    un modelo que anda — costó un mes averiguarlo con la cabeza de objetos."""
    from detector_agresion import ErrorCheckpointAgresion
    try:
        DetectorAgresion(ConfigAgresion(checkpoint=os.path.join(_AQUI, "no_existe.pt")))
    except ErrorCheckpointAgresion as e:
        return "No existe" in str(e), "corta con mensaje explicado"
    except Exception as e:                            # pragma: no cover
        return False, f"cortó con otra excepción: {type(e).__name__}"
    return False, "NO cortó: arrancó sin checkpoint"


def caso_punta_a_punta() -> Tuple[bool, str]:
    """El adaptador -> `observacion()` -> motor de fusión: un evento de pelea.

    Es el caso que prueba que el contrato entre las dos capas cierra de
    verdad, no que cada una anda por su lado."""
    d = det(0.88)
    motor = MotorFusion(fps=10.0)
    eventos = []
    for i in range(120):                              # 12 s
        t = 1000.0 + i * 0.1
        tracks = [persona(30 + k, 100.0 + 120 * k, ts=t) for k in range(2)]
        acs = d.procesar("cam-1", frame(), tracks, ts=t)
        obs = ObservacionCamara(
            camera_id="cam-1", frame_idx=i, ts=t, tracks=tracks, acciones=acs,
            tam_frame=(ALTO, ANCHO),
            cabezas=frozenset({CABEZA_OBJETOS, CABEZA_ACCION}))
        eventos.extend(motor.procesar([obs]))
    peleas = [e for e in eventos if e.tipo == "pelea"]
    participantes = sorted(peleas[0].track_ids) if peleas else []
    ok = len(peleas) == 1 and participantes == [30, 31]
    return ok, f"{len(peleas)} evento(s) de pelea, participantes {participantes}"


CASOS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "emite_siempre": caso_emite_siempre,
    "compuerta": caso_compuerta,
    "buffer_con_uno": caso_buffer_se_llena_con_uno,
    "muestreo_tiempo": caso_muestreo_por_tiempo,
    "caja_union": caso_caja_es_la_union,
    "personas_chicas": caso_personas_chicas_no_cuentan,
    "clip_viejo": caso_clip_viejo_se_tira,
    "camaras": caso_camaras_no_se_mezclan,
    "normalizacion": caso_normalizacion,
    "checkpoint": caso_checkpoint_incompleto,
    "punta_a_punta": caso_punta_a_punta,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caso")
    ap.add_argument("--listar", action="store_true")
    args = ap.parse_args()

    if args.listar:
        for k, f in CASOS.items():
            print(f"  {k:<18} {(f.__doc__ or '').strip().splitlines()[0]}")
        return 0

    casos = {args.caso: CASOS[args.caso]} if args.caso else CASOS
    ancho = "=" * 76
    print(ancho)
    print("HORUS · adaptador de agresión — autotest (sin torch, sin pesos)")
    print(ancho)

    fallas = 0
    for nombre, f in casos.items():
        t0 = time.perf_counter()
        try:
            ok, detalle = f()
        except Exception as e:                        # pragma: no cover
            ok, detalle = False, f"EXCEPCIÓN {type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {'OK   ' if ok else 'FALLA'}  {nombre:<18} {detalle:<52} {ms:6.1f} ms")
        fallas += not ok

    print("-" * 76)
    print(f"{len(casos) - fallas}/{len(casos)} pruebas OK")
    return 1 if fallas else 0


if __name__ == "__main__":
    raise SystemExit(main())
