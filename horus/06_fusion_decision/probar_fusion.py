# -*- coding: utf-8 -*-
"""
probar_fusion.py · HORUS — escenarios sintéticos para la capa de fusión.

    python probar_fusion.py                 # corre todos los escenarios
    python probar_fusion.py --detalle       # además muestra cada evento
    python probar_fusion.py --caso incendio

Por qué sintético y no con cámara: la capa de fusión no mira píxeles, mira
tracks. Entonces se le pueden fabricar los tracks a mano y verificar que
decide bien SIN GPU, sin modelos y sin esperar a que pase algo raro delante
de una cámara. Un escenario que tarda 40 segundos en la vida real acá tarda
milisegundos, y encima es repetible.

Esto es lo que permite tener la capa terminada y probada antes de que existan
la cabeza de pose y el clasificador de acciones: sus escenarios ya están
escritos y pasan; el día que lleguen los modelos, lo único que cambia es de
dónde salen los `AccionObs`.

Cada caso devuelve OK/FALLA. Cero dependencias fuera de numpy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
for _p in (_AQUI,
           os.path.join(_RAIZ, "05_tracking"),
           os.path.join(_RAIZ, "05_tracking", "local")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contratos import (  # noqa: E402
    CABEZA_ACCION, CABEZA_OBJETOS, CABEZA_POSE, CABEZA_SEGMENTACION,
    AccionObs, ObservacionCamara, SegObs, Severidad,
)
from motor_fusion import ConfigFusion, MotorFusion  # noqa: E402
from topologia import Topologia  # noqa: E402
from tracker_local import TrackLocal  # noqa: E402


# Lunes 24 de agosto de 2026. Se fija la hora en dos momentos porque media
# capa depende de si es horario laboral o no.
def _ts(hora: int, minuto: int = 0, dia: int = 24) -> float:
    return time.mktime((2026, 8, dia, hora, minuto, 0, 0, 0, -1))


DIA = _ts(12, 0)        # lunes 12:00, depósito abierto
NOCHE = _ts(3, 0)       # lunes 03:00, depósito cerrado


TOPO = Topologia.desde_dict({
    "camaras": {
        "cam-porton": {
            "nombre": "Portón", "tam_frame": [1080, 1920],
            "vecinos": [{"destino": "cam-deposito", "t_min_s": 4.0, "t_max_s": 90.0}],
            "zonas": [
                {"nombre": "vereda", "tipo": "transito",
                 "puntos": [[0, 500], [1920, 500], [1920, 1080], [0, 1080]],
                 "merodeo_s": 20.0},
                {"nombre": "acceso", "tipo": "acceso", "prioridad": 1,
                 "puntos": [[700, 400], [1250, 400], [1250, 950], [700, 950]],
                 "merodeo_s": 12.0}]},
        "cam-deposito": {
            "nombre": "Depósito", "tam_frame": [1080, 1920],
            "zonas": [
                {"nombre": "deposito", "tipo": "restringida", "prioridad": 1,
                 "puntos": [[100, 300], [1800, 300], [1800, 1050], [100, 1050]],
                 "merodeo_s": 10.0,
                 "horario_permitido": {"dias": ["lun", "mar", "mie", "jue", "vie"],
                                       "desde": "07:00", "hasta": "19:00"}}]},
    }})


# --------------------------------------------------------------------------- #
# Fábricas
# --------------------------------------------------------------------------- #
def track(clase: str, caja: Sequence[float], *, tid: int = 1,
          score: float = 0.7, cam: str = "cam-deposito", ts: float = 0.0,
          visto_s: float = 5.0, quieto: bool = False, quieto_s: float = 0.0,
          radio: float = 5.0, gid: Optional[int] = None,
          estado: str = "confirmado") -> TrackLocal:
    """Un `TrackLocal` real, no un mock: si el contrato cambia, esto rompe."""
    return TrackLocal(
        track_id=tid, clase=clase, class_id=0, bbox_xyxy=tuple(caja),
        score=score, camera_id=cam, frame_idx=0, ts=ts, estado=estado,
        hits=int(visto_s * 10), edad=int(visto_s * 10), frames_sin_ver=0,
        ts_nacimiento=ts - visto_s, visto_s=visto_s,
        radio_permanencia=radio, quieto=quieto, quieto_s=quieto_s,
        global_id=gid)


def obs(cam: str, ts: float, tracks: Sequence[TrackLocal] = (),
        seg: Optional[Dict[str, float]] = None,
        acciones: Sequence[AccionObs] = (),
        cabezas: Sequence[str] = (CABEZA_OBJETOS,),
        novelty: float = 0.0) -> ObservacionCamara:
    s = None
    if seg is not None:
        s = SegObs(area=dict(seg),
                   score={k: 0.9 for k in seg}, camera_id=cam, ts=ts)
    return ObservacionCamara(
        camera_id=cam, frame_idx=int(ts * 10), ts=ts, tracks=list(tracks),
        seg=s, acciones=list(acciones), novelty=novelty,
        tam_frame=(1080, 1920), cabezas=frozenset(cabezas))


def correr(pasos: Sequence[Sequence[ObservacionCamara]],
           motor: Optional[MotorFusion] = None,
           cfg: Optional[ConfigFusion] = None) -> Tuple[List[Any], MotorFusion]:
    motor = motor or MotorFusion(topologia=TOPO,
                                 cfg=cfg or ConfigFusion())
    eventos: List[Any] = []
    for grupo in pasos:
        eventos.extend(motor.procesar(grupo))
    return eventos, motor


def tipos(eventos) -> List[str]:
    return [e.tipo for e in eventos]


# --------------------------------------------------------------------------- #
# Escenarios
# --------------------------------------------------------------------------- #
def caso_incendio_confirmado() -> Tuple[bool, str, List[Any]]:
    """Llama en objetos + fuego en segmentación = crítico, y rápido."""
    pasos = []
    for i in range(12):
        t = DIA + i * 0.1
        pasos.append([obs("cam-deposito", t,
                          [track("llama", (800, 400, 900, 520), tid=1,
                                 score=0.72, ts=t, visto_s=2.0 + i * 0.1)],
                          seg={"fuego": 0.018},
                          cabezas=(CABEZA_OBJETOS, CABEZA_SEGMENTACION))])
    ev, _ = correr(pasos)
    incendios = [e for e in ev if e.tipo == "incendio"]
    ok = (len(incendios) == 1
          and incendios[0].severidad == Severidad.CRITICO
          and incendios[0].confianza >= 0.90)
    return ok, f"{len(incendios)} evento(s), sev={incendios[0].severidad.etiqueta if incendios else '-'}", ev


def caso_humo_solo() -> Tuple[bool, str, List[Any]]:
    """Humo sin fuego: aviso y al VLM, NO alerta. Puede ser vapor o neblina —
    está documentado que a 384 px el modelo no los separa."""
    pasos = []
    for i in range(12):
        t = DIA + i * 0.1
        pasos.append([obs("cam-deposito", t,
                          [track("humo", (600, 300, 900, 600), tid=2,
                                 score=0.45, ts=t, visto_s=2.0)],
                          seg={"humo": 0.03},
                          cabezas=(CABEZA_OBJETOS, CABEZA_SEGMENTACION))])
    ev, _ = correr(pasos)
    inc = [e for e in ev if e.tipo == "incendio"]
    ok = len(inc) == 1 and inc[0].severidad <= Severidad.ALERTA and inc[0].necesita_vlm
    return ok, (f"sev={inc[0].severidad.etiqueta if inc else '-'} "
                f"vlm={inc[0].necesita_vlm if inc else '-'}"), ev


def caso_incendio_crece() -> Tuple[bool, str, List[Any]]:
    """Un área que crece sube la severidad; un reflejo estable no."""
    def corrida(crece: bool):
        pasos = []
        for i in range(80):
            t = DIA + i * 0.1
            a = 0.004 * (1.0 + (0.06 * i if crece else 0.0))
            pasos.append([obs("cam-porton", t, [], seg={"humo": a},
                              cabezas=(CABEZA_OBJETOS, CABEZA_SEGMENTACION))])
        ev, _ = correr(pasos)
        return [e for e in ev if e.tipo == "incendio"]

    con = corrida(True)
    sin = corrida(False)
    sev_con = max((e.severidad for e in con), default=Severidad.INFO)
    sev_sin = max((e.severidad for e in sin), default=Severidad.INFO)
    ok = sev_con > sev_sin
    return ok, f"creciendo={sev_con.etiqueta} estable={sev_sin.etiqueta}", con


def caso_intrusion_nocturna() -> Tuple[bool, str, List[Any]]:
    """Persona en el depósito a las 3 AM."""
    pasos = []
    for i in range(20):
        t = NOCHE + i * 0.1
        pasos.append([obs("cam-deposito", t,
                          [track("persona", (900, 500, 1000, 900), tid=3,
                                 score=0.55, ts=t, visto_s=2.0 + i * 0.1,
                                 radio=120.0, gid=7)])])
    ev, _ = correr(pasos)
    intr = [e for e in ev if e.tipo == "intrusion"]
    ok = len(intr) == 1 and intr[0].zona == "deposito" and intr[0].global_id == 7
    return ok, f"{len(intr)} intrusión(es), zona={intr[0].zona if intr else '-'}", ev


def caso_horario_laboral() -> Tuple[bool, str, List[Any]]:
    """La misma persona, el mismo lugar, un lunes al mediodía: NO es evento.
    Es la mitad del valor de la topología."""
    pasos = []
    for i in range(20):
        t = DIA + i * 0.1
        pasos.append([obs("cam-deposito", t,
                          [track("persona", (900, 500, 1000, 900), tid=4,
                                 score=0.55, ts=t, visto_s=2.0, radio=120.0)])])
    ev, _ = correr(pasos)
    ok = not [e for e in ev if e.tipo == "intrusion"]
    return ok, f"{len(ev)} evento(s) (esperado: ninguna intrusión)", ev


def caso_robo_correlacionado() -> Tuple[bool, str, List[Any]]:
    """Arma + intrusión sobre la misma persona = robo crítico, y las dos
    piezas sueltas se callan."""
    pasos = []
    for i in range(20):
        t = NOCHE + i * 0.1
        persona = track("persona", (900, 500, 1000, 900), tid=5, score=0.6,
                        ts=t, visto_s=2.0 + i * 0.1, radio=120.0, gid=9)
        arma = track("pistola", (940, 640, 985, 690), tid=6, score=0.55,
                     ts=t, visto_s=1.0 + i * 0.1)
        pasos.append([obs("cam-deposito", t, [persona, arma])])
    ev, _ = correr(pasos)
    robos = [e for e in ev if e.tipo == "robo"]
    sueltos = [e for e in ev if e.tipo in ("arma", "intrusion")]
    ok = (len(robos) == 1 and robos[0].severidad == Severidad.CRITICO
          and robos[0].global_id == 9 and not sueltos)
    return ok, (f"{len(robos)} robo(s), {len(sueltos)} componente(s) sin "
                f"suprimir"), ev


def caso_arma_sin_zona() -> Tuple[bool, str, List[Any]]:
    """Arma sola: alerta y siempre al VLM, nunca crítico por sí sola."""
    pasos = []
    for i in range(20):
        t = DIA + i * 0.1
        pasos.append([obs("cam-porton", t,
                          [track("persona", (900, 500, 1000, 900), tid=7,
                                 cam="cam-porton", score=0.6, ts=t, visto_s=3.0),
                           track("cuchillo", (940, 640, 980, 680), tid=8,
                                 cam="cam-porton", score=0.5, ts=t, visto_s=1.5)])])
    ev, _ = correr(pasos)
    armas = [e for e in ev if e.tipo == "arma"]
    ok = (len(armas) == 1 and armas[0].severidad == Severidad.ALERTA
          and armas[0].necesita_vlm)
    return ok, (f"sev={armas[0].severidad.etiqueta if armas else '-'} "
                f"vlm={armas[0].necesita_vlm if armas else '-'}"), ev


def caso_merodeo() -> Tuple[bool, str, List[Any]]:
    """Alguien 20 s frente al acceso sin irse (la zona pide 12 s)."""
    pasos = []
    for i in range(30):
        t = DIA + i * 0.5
        pasos.append([obs("cam-porton", t,
                          [track("persona", (950, 550, 1050, 900), tid=9,
                                 cam="cam-porton", score=0.6, ts=t,
                                 visto_s=1.0 + i * 0.5, radio=40.0)])])
    ev, _ = correr(pasos)
    mer = [e for e in ev if e.tipo == "merodeo"]
    ok = len(mer) == 1 and mer[0].zona == "acceso"
    return ok, f"{len(mer)} merodeo(s), zona={mer[0].zona if mer else '-'}", ev


def caso_persona_de_paso() -> Tuple[bool, str, List[Any]]:
    """La misma permanencia pero caminando: no es merodeo."""
    pasos = []
    for i in range(30):
        t = DIA + i * 0.5
        x = 750 + i * 15
        pasos.append([obs("cam-porton", t,
                          [track("persona", (x, 550, x + 100, 900), tid=10,
                                 cam="cam-porton", score=0.6, ts=t,
                                 visto_s=1.0 + i * 0.5, radio=600.0)])])
    ev, _ = correr(pasos)
    ok = not [e for e in ev if e.tipo == "merodeo"]
    return ok, f"{len(ev)} evento(s) (esperado: ningún merodeo)", ev


def caso_paquete_abandonado() -> Tuple[bool, str, List[Any]]:
    """Alguien deja un bulto y se va. El evento sale recién cuando se fue."""
    pasos, t = [], DIA
    caja = (700, 800, 780, 880)
    # 1) llega con el paquete
    for i in range(10):
        t = DIA + i * 0.5
        pasos.append([obs("cam-porton", t, [
            track("paquete", caja, tid=11, cam="cam-porton", score=0.5, ts=t,
                  visto_s=1.0 + i * 0.5, quieto=True, quieto_s=1.0 + i * 0.5),
            track("persona", (760, 500, 860, 880), tid=12, cam="cam-porton",
                  score=0.6, ts=t, visto_s=1.0 + i * 0.5)])])
    # 2) se va; el paquete se queda
    for i in range(60):
        t = DIA + 5.0 + i * 0.5
        pasos.append([obs("cam-porton", t, [
            track("paquete", caja, tid=11, cam="cam-porton", score=0.5, ts=t,
                  visto_s=6.0 + i * 0.5, quieto=True, quieto_s=6.0 + i * 0.5)])])
    ev, _ = correr(pasos)
    paq = [e for e in ev if e.tipo == "paquete_abandonado"]
    ok = len(paq) == 1 and paq[0].evidencia.get("sin_dueno_s", 0) >= 10.0
    return ok, (f"{len(paq)} paquete(s), sin dueño "
                f"{paq[0].evidencia.get('sin_dueno_s') if paq else '-'} s"), ev


def caso_paquete_con_dueno() -> Tuple[bool, str, List[Any]]:
    """El mismo paquete, pero el dueño se queda al lado: no es evento."""
    pasos = []
    for i in range(70):
        t = DIA + i * 0.5
        pasos.append([obs("cam-porton", t, [
            track("paquete", (700, 800, 780, 880), tid=13, cam="cam-porton",
                  score=0.5, ts=t, visto_s=1.0 + i * 0.5, quieto=True,
                  quieto_s=1.0 + i * 0.5),
            track("persona", (760, 500, 860, 880), tid=14, cam="cam-porton",
                  score=0.6, ts=t, visto_s=1.0 + i * 0.5, radio=30.0)])])
    ev, _ = correr(pasos)
    ok = not [e for e in ev if e.tipo == "paquete_abandonado"]
    return ok, f"{len(ev)} evento(s) (esperado: ningún paquete abandonado)", ev


def caso_caida_dormida() -> Tuple[bool, str, List[Any]]:
    """SIN cabeza de pose: la regla no corre y lo dice. No devuelve 'no hubo
    caídas', que sería mentira."""
    pasos = []
    for i in range(20):
        t = DIA + i * 0.1
        pasos.append([obs("cam-deposito", t,
                          [track("persona", (900, 700, 1200, 900), tid=15,
                                 score=0.6, ts=t, visto_s=5.0, quieto=True,
                                 quieto_s=5.0)])])
    ev, motor = correr(pasos)
    dormidas = motor.reglas_dormidas()
    ok = (not [e for e in ev if e.tipo == "caida"]
          and dormidas.get("caida", 0) == 20)
    return ok, f"reglas dormidas: {dormidas}", ev


def caso_caida_por_pose() -> Tuple[bool, str, List[Any]]:
    """CON keypoints: torso horizontal y la persona no se levanta.

    20 s tumbada: tiene que abrir como ALERTA y escalar a CRÍTICO al pasar
    `critico_s`. Que nadie la levante en 15 s es lo que separa un tropezón de
    una urgencia.
    """
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.9
    kp[5] = (900, 850, 0.9)      # hombro izq
    kp[6] = (900, 880, 0.9)      # hombro der
    kp[11] = (1150, 855, 0.9)    # cadera izq   -> torso horizontal
    kp[12] = (1150, 885, 0.9)    # cadera der
    kp[0] = (870, 865, 0.9)      # nariz

    pasos = []
    for i in range(100):                      # 20 s
        t = DIA + i * 0.2
        p = track("persona", (860, 820, 1180, 920), tid=16, score=0.65, ts=t,
                  visto_s=8.0, quieto=True, quieto_s=1.0 + i * 0.2)
        a = AccionObs(bbox_xyxy=(860, 820, 1180, 920), keypoints=kp,
                      track_id=16, camera_id="cam-deposito", ts=t,
                      fuente=CABEZA_POSE, score=0.8)
        pasos.append([obs("cam-deposito", t, [p], acciones=[a],
                          cabezas=(CABEZA_OBJETOS, CABEZA_POSE))])
    ev, _ = correr(pasos)
    caidas = [e for e in ev if e.tipo == "caida"]
    ids = {e.evento_id for e in caidas}
    ok = (len(ids) == 1
          and max((e.severidad for e in caidas), default=Severidad.INFO)
          == Severidad.CRITICO)
    return ok, (f"{len(ids)} caída(s), escaló a "
                f"{max((e.severidad.etiqueta for e in caidas), default='-')}"), ev


def caso_caida_por_clasificador() -> Tuple[bool, str, List[Any]]:
    """CON un clasificador de acción en vez de pose: mismo resultado."""
    pasos = []
    for i in range(30):
        t = DIA + i * 0.2
        p = track("persona", (860, 820, 1180, 920), tid=17, score=0.65, ts=t,
                  visto_s=6.0, quieto=True, quieto_s=1.0 + i * 0.2)
        a = AccionObs(bbox_xyxy=(860, 820, 1180, 920), accion="caida",
                      score=0.81, track_id=17, camera_id="cam-deposito",
                      ts=t, fuente=CABEZA_ACCION)
        pasos.append([obs("cam-deposito", t, [p], acciones=[a],
                          cabezas=(CABEZA_OBJETOS, CABEZA_ACCION))])
    ev, _ = correr(pasos)
    caidas = [e for e in ev if e.tipo == "caida"]
    ok = len(caidas) >= 1
    return ok, f"{len(caidas)} evento(s) de caída", ev


def caso_sentarse_no_es_caida() -> Tuple[bool, str, List[Any]]:
    """Postura de caída pero la persona se mueve: no alerta. Lo que separa
    caerse de agacharse es lo que pasa después."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.9
    kp[5] = (900, 850, 0.9); kp[6] = (900, 880, 0.9)
    kp[11] = (1150, 855, 0.9); kp[12] = (1150, 885, 0.9)
    pasos = []
    for i in range(30):
        t = DIA + i * 0.2
        p = track("persona", (860, 820, 1180, 920), tid=18, score=0.65, ts=t,
                  visto_s=6.0, quieto=False, quieto_s=0.0)
        a = AccionObs(bbox_xyxy=(860, 820, 1180, 920), keypoints=kp,
                      track_id=18, camera_id="cam-deposito", ts=t,
                      fuente=CABEZA_POSE)
        pasos.append([obs("cam-deposito", t, [p], acciones=[a],
                          cabezas=(CABEZA_OBJETOS, CABEZA_POSE))])
    ev, _ = correr(pasos)
    ok = not [e for e in ev if e.tipo == "caida"]
    return ok, f"{len(ev)} evento(s) (esperado: ninguna caída)", ev


def caso_accion_externa() -> Tuple[bool, str, List[Any]]:
    """El modelo del compañero manda 'robo' directamente. Entra por
    ACCIONES_EXTERNAS sin tocar el motor."""
    pasos = []
    for i in range(30):
        t = DIA + i * 0.2
        p = track("persona", (900, 500, 1000, 900), tid=19, score=0.6, ts=t,
                  visto_s=4.0, gid=21)
        a = AccionObs(bbox_xyxy=(900, 500, 1000, 900), accion="robo",
                      score=0.77, track_id=19, camera_id="cam-deposito",
                      ts=t, fuente=CABEZA_ACCION)
        pasos.append([obs("cam-deposito", t, [p], acciones=[a],
                          cabezas=(CABEZA_OBJETOS, CABEZA_ACCION))])
    ev, _ = correr(pasos)
    robos = [e for e in ev if e.tipo == "robo"]
    ok = (len(robos) == 1 and robos[0].severidad == Severidad.CRITICO
          and robos[0].global_id == 21)
    return ok, f"{len(robos)} robo(s) por clasificador", ev


def caso_un_evento_no_trescientos() -> Tuple[bool, str, List[Any]]:
    """30 s de incendio a 10 FPS son 300 frames. Tienen que salir UN evento
    abierto y unas pocas re-emisiones, no 300 alertas."""
    pasos = []
    for i in range(300):
        t = DIA + i * 0.1
        pasos.append([obs("cam-deposito", t,
                          [track("llama", (800, 400, 900, 520), tid=20,
                                 score=0.7, ts=t, visto_s=2.0 + i * 0.1)],
                          seg={"fuego": 0.02},
                          cabezas=(CABEZA_OBJETOS, CABEZA_SEGMENTACION))])
    ev, motor = correr(pasos)
    inc = [e for e in ev if e.tipo == "incendio"]
    distintos = {e.evento_id for e in inc}
    ok = len(distintos) == 1 and len(inc) <= 4
    return ok, (f"{len(distintos)} evento(s) distinto(s), {len(inc)} avisos "
                f"en 300 frames"), ev


def caso_seguimiento_entre_camaras() -> Tuple[bool, str, List[Any]]:
    """La misma persona (global_id 33) merodea el portón y sigue en el
    depósito: UN evento que cambia de cámara, no dos."""
    pasos = []
    for i in range(40):
        t = NOCHE + i * 0.5
        pasos.append([obs("cam-porton", t,
                          [track("persona", (950, 550, 1050, 900), tid=30,
                                 cam="cam-porton", score=0.6, ts=t,
                                 visto_s=1.0 + i * 0.5, radio=40.0, gid=33)])])
    for i in range(20):
        t = NOCHE + 20.0 + i * 0.5
        pasos.append([obs("cam-deposito", t,
                          [track("persona", (900, 500, 1000, 900), tid=31,
                                 score=0.6, ts=t, visto_s=25.0 + i * 0.5,
                                 radio=40.0, gid=33)])])
    ev, motor = correr(pasos)
    mer = [e for e in ev if e.tipo == "merodeo"]
    ids = {e.evento_id for e in mer}
    cams = set()
    for e in mer:
        cams.update(e.camaras)
    ok = len(ids) == 1 and len(cams) == 2
    return ok, f"{len(ids)} evento(s) de merodeo sobre {len(cams)} cámara(s)", ev


def caso_gate_vlm() -> Tuple[bool, str, List[Any]]:
    """Novedad alta de escena empuja al VLM aunque la regla no lo pida."""
    pasos = []
    for i in range(30):
        t = NOCHE + i * 0.5
        pasos.append([obs("cam-porton", t,
                          [track("persona", (950, 550, 1050, 900), tid=40,
                                 cam="cam-porton", score=0.6, ts=t,
                                 visto_s=1.0 + i * 0.5, radio=40.0)],
                          novelty=4.5)])
    ev, _ = correr(pasos)
    mer = [e for e in ev if e.tipo == "merodeo"]
    ok = bool(mer) and mer[0].necesita_vlm and mer[0].vlm_motivo == "novedad"
    return ok, (f"vlm={mer[0].necesita_vlm if mer else '-'} "
                f"motivo={mer[0].vlm_motivo if mer else '-'}"), ev


def caso_escena_vacia() -> Tuple[bool, str, List[Any]]:
    """Nada delante de la cámara durante un minuto: cero eventos. Suena
    trivial y es el que más se rompe cuando se agrega una regla."""
    pasos = [[obs("cam-deposito", NOCHE + i * 0.1, [], seg={},
                  cabezas=(CABEZA_OBJETOS, CABEZA_SEGMENTACION))]
             for i in range(600)]
    ev, _ = correr(pasos)
    return not ev, f"{len(ev)} evento(s)", ev


CASOS: Dict[str, Callable[[], Tuple[bool, str, List[Any]]]] = {
    "incendio": caso_incendio_confirmado,
    "humo_solo": caso_humo_solo,
    "incendio_crece": caso_incendio_crece,
    "intrusion": caso_intrusion_nocturna,
    "horario_laboral": caso_horario_laboral,
    "robo": caso_robo_correlacionado,
    "arma": caso_arma_sin_zona,
    "merodeo": caso_merodeo,
    "de_paso": caso_persona_de_paso,
    "paquete": caso_paquete_abandonado,
    "paquete_con_dueno": caso_paquete_con_dueno,
    "caida_dormida": caso_caida_dormida,
    "caida_pose": caso_caida_por_pose,
    "caida_clasificador": caso_caida_por_clasificador,
    "sentarse": caso_sentarse_no_es_caida,
    "accion_externa": caso_accion_externa,
    "dedup": caso_un_evento_no_trescientos,
    "entre_camaras": caso_seguimiento_entre_camaras,
    "gate_vlm": caso_gate_vlm,
    "vacio": caso_escena_vacia,
}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caso", help="correr uno solo")
    ap.add_argument("--detalle", action="store_true",
                    help="mostrar cada evento generado")
    ap.add_argument("--listar", action="store_true")
    args = ap.parse_args()

    if args.listar:
        for k, f in CASOS.items():
            print(f"  {k:<22} {(f.__doc__ or '').strip().splitlines()[0]}")
        return 0

    avisos = TOPO.verificar()
    if avisos:
        print("Avisos de la topología de prueba:")
        for a in avisos:
            print("  ·", a)
        print()

    casos = {args.caso: CASOS[args.caso]} if args.caso else CASOS
    if args.caso and args.caso not in CASOS:
        print(f"caso desconocido: {args.caso}. Probá --listar")
        return 2

    print("=" * 74)
    print("HORUS · capa de fusión — escenarios sintéticos")
    print("=" * 74)

    fallos = 0
    for nombre, fn in casos.items():
        t0 = time.perf_counter()
        try:
            ok, detalle, eventos = fn()
        except Exception as exc:            # pragma: no cover
            import traceback
            print(f"  {'ROTO':<6} {nombre:<22} {type(exc).__name__}: {exc}")
            traceback.print_exc()
            fallos += 1
            continue
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"  {'OK' if ok else 'FALLA':<6} {nombre:<22} {detalle:<52} "
              f"{dt:6.1f} ms")
        if not ok:
            fallos += 1
        if args.detalle:
            for e in eventos:
                print(f"           {e.linea()}")
            print()

    print("-" * 74)
    print(f"{len(casos) - fallos}/{len(casos)} escenarios OK")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
