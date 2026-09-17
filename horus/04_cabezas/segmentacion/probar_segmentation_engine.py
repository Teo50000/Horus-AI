# -*- coding: utf-8 -*-
"""
probar_segmentation_engine.py · el camino de segmentación, de punta a punta.

**Sin torch, sin pesos, sin GPU y sin cámara.** El motor se inyecta, igual que
en las otras suites. No se verifica la calidad del modelo —eso está medido:
F1 agua 95,4 % · humo 95,6 % · fuego 99,1 %— sino el **cableado**: que el
pipeline corra sin la cabeza de objetos, que el fuego solo abra un evento, que
el humo solo no alcance para crítico, que las reglas que necesitan tracks se
declaren DORMIDAS en vez de devolver vacío, y que el evento llegue hasta un
mensaje válido para el backend.

Lo que prueba de verdad: que hay una alerta real posible **hoy**, sin el
`backbone.pt` de objetos, que sigue perdido.

    python probar_segmentation_engine.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.normpath(os.path.join(_AQUI, "..", ".."))
for _p in (_AQUI,
           os.path.join(_RAIZ, "06_fusion_decision"),
           os.path.join(_RAIZ, "07_alerta"),
           os.path.join(_RAIZ, "05_tracking"),
           os.path.join(_RAIZ, "05_tracking", "local")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contratos import CABEZA_SEGMENTACION, Severidad  # noqa: E402
from pipeline import PipelineHorus  # noqa: E402
from segmentation_engine import (  # noqa: E402
    ConfigSegmentacion, ErrorCheckpointSegmentacion, MotorSegmentacion,
)


# --------------------------------------------------------------------------- #
# Un `SegResult` de mentira: duck typing, igual que lo consume la fusión.
# --------------------------------------------------------------------------- #
@dataclass
class SegFalso:
    area: Dict[str, float] = field(default_factory=dict)
    score: Dict[str, float] = field(default_factory=dict)
    camera_id: str = "cam-0"
    frame_idx: int = -1
    ts: float = 0.0
    needs_vlm: bool = False

    @property
    def clases_activas(self) -> List[str]:
        return [k for k, v in self.area.items() if v > 0]


def motor(areas: Callable[[int], Dict[str, float]]) -> MotorSegmentacion:
    """Motor inyectado: `areas(i)` devuelve las áreas del frame i."""
    estado = {"i": 0}

    def falso(frames: Sequence[Any], camera_ids: Optional[Sequence[str]]):
        a = areas(estado["i"])
        estado["i"] += 1
        return [SegFalso(area=dict(a), score={k: 0.93 for k in a},
                         camera_id=(camera_ids[j] if camera_ids else "cam-0"),
                         frame_idx=estado["i"],
                         needs_vlm=any(0.005 <= v <= 0.02 for v in a.values()))
                for j in range(len(frames))]

    return MotorSegmentacion(ConfigSegmentacion(verboso=False), motor_falso=falso)


def frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def correr(m: MotorSegmentacion, pasos: int, fps: float = 10.0,
           cam: str = "cam-0") -> Tuple[List[Any], PipelineHorus]:
    pipe = PipelineHorus(motor_segmentacion=m, fps=fps, verboso=False)
    eventos = []
    t0 = time.time()
    for i in range(pasos):
        eventos.extend(pipe.procesar({cam: frame()}, ts=t0 + i / fps))
    return eventos, pipe


# --------------------------------------------------------------------------- #
# Casos
# --------------------------------------------------------------------------- #
def caso_fuego_solo() -> Tuple[bool, str]:
    """Fuego de segmentación SOLO abre un evento, sin cabeza de objetos.

    Es el camino completo que no depende del backbone perdido: cámara ->
    segmentación -> fusión -> evento."""
    m = motor(lambda i: {"fuego": 0.02 + i * 0.0005})
    ev, _ = correr(m, 40)
    inc = [e for e in ev if e.tipo == "incendio"]
    ok = len(inc) >= 1 and inc[0].severidad >= Severidad.ALERTA
    return ok, (f"{len(inc)} evento(s) de incendio, sev="
                f"{inc[0].severidad.etiqueta if inc else '-'}")


def caso_humo_solo_no_es_critico() -> Tuple[bool, str]:
    """Humo solo no alcanza para crítico y pide VLM.

    Neblina y vapor son humo a 384 px y está documentado que el modelo no los
    separa; por eso el humo solo nunca dispara lo irreversible."""
    m = motor(lambda i: {"humo": 0.05})
    ev, _ = correr(m, 40)
    inc = [e for e in ev if e.tipo == "incendio"]
    ok = bool(inc) and all(e.severidad < Severidad.CRITICO for e in inc) \
        and all(e.necesita_vlm for e in inc)
    return ok, (f"{len(inc)} evento(s), sev="
                f"{inc[0].severidad.etiqueta if inc else '-'}, "
                f"vlm={inc[0].necesita_vlm if inc else '-'}")


def caso_escena_limpia() -> Tuple[bool, str]:
    """Sin fuego ni humo no pasa nada. Un motor que alerta de más se apaga
    solo, porque el operador deja de mirarlo."""
    m = motor(lambda i: {})
    ev, _ = correr(m, 60)
    return not ev, f"{len(ev)} evento(s) (esperado 0)"


def caso_reglas_dormidas() -> Tuple[bool, str]:
    """Sin cabeza de objetos, las reglas que necesitan tracks se declaran
    DORMIDAS — no devuelven "no pasó nada".

    "No hubo intrusiones" y "no hay quien las mire" no pueden parecer lo mismo
    en un sistema cuyo objetivo es no perderse nada."""
    m = motor(lambda i: {"fuego": 0.02})
    _, pipe = correr(m, 30)
    dormidas = set(pipe.fusion.reglas_dormidas())
    esperadas = {"intrusion", "merodeo", "paquete_abandonado", "arma"}
    ok = esperadas <= dormidas and "incendio" not in dormidas
    return ok, f"dormidas: {sorted(dormidas)}"


def caso_sin_ninguna_cabeza() -> Tuple[bool, str]:
    """Un pipeline sin objetos y sin segmentación no arranca.

    Arrancar igual daría un sistema que no mira nada y no lo dice."""
    try:
        PipelineHorus(fps=10.0, verboso=False)
    except ValueError as e:
        return "al menos una cabeza" in str(e), "corta con mensaje explicado"
    except Exception as e:                            # pragma: no cover
        return False, f"cortó con otra excepción: {type(e).__name__}"
    return False, "NO cortó: arrancó sin ninguna cabeza"


def caso_checkpoint_ausente() -> Tuple[bool, str]:
    """Sin el checkpoint, el motor corta y dice dónde viven los pesos.

    Son los que se rescataron de un stash el 27/08 y existen en un solo disco
    del mundo."""
    try:
        MotorSegmentacion(ConfigSegmentacion(
            pesos=os.path.join(_AQUI, "no_existe.pt"), verboso=False))
    except ErrorCheckpointSegmentacion as e:
        return "No existe" in str(e), "corta con mensaje explicado"
    except Exception as e:                            # pragma: no cover
        return False, f"cortó con otra excepción: {type(e).__name__}"
    return False, "NO cortó: arrancó sin pesos"


def caso_seg_inyectada_tiene_prioridad() -> Tuple[bool, str]:
    """Una `seg` pasada a mano gana sobre el motor: el motor no la pisa ni la
    recalcula. Es lo que permite reproducir un caso guardado."""
    m = motor(lambda i: {"fuego": 0.02})
    pipe = PipelineHorus(motor_segmentacion=m, fps=10.0, verboso=False)
    t0 = time.time()
    for i in range(30):
        pipe.procesar({"cam-0": frame()},
                      seg={"cam-0": SegFalso(area={}, score={}, camera_id="cam-0")},
                      ts=t0 + i * 0.1)
    ok = m.stats["frames"] == 0
    return ok, f"el motor corrió {m.stats['frames']} frame(s) (esperado 0)"


def caso_hasta_el_backend() -> Tuple[bool, str]:
    """El evento llega hasta un mensaje válido para el backend, con
    `segmentacion` entre los modelos que aportaron."""
    from alerta import armar_payload, validar_payload
    m = motor(lambda i: {"fuego": 0.02 + i * 0.0005})
    ev, _ = correr(m, 40)
    inc = [e for e in ev if e.tipo == "incendio"]
    if not inc:
        return False, "no se abrió ningún evento"
    p = armar_payload(inc[0], 1, sitio="sucursal-centro", nodo="caja-01")
    problemas = validar_payload(p)
    ok = not problemas and "segmentacion" in p["modelos"]
    return ok, (f"modelos={p['modelos']}, "
                f"validación={'OK' if not problemas else problemas[:2]}")


CASOS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "fuego_solo": caso_fuego_solo,
    "humo_no_critico": caso_humo_solo_no_es_critico,
    "escena_limpia": caso_escena_limpia,
    "reglas_dormidas": caso_reglas_dormidas,
    "sin_cabezas": caso_sin_ninguna_cabeza,
    "checkpoint": caso_checkpoint_ausente,
    "seg_inyectada": caso_seg_inyectada_tiene_prioridad,
    "hasta_el_backend": caso_hasta_el_backend,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caso")
    args = ap.parse_args()
    casos = {args.caso: CASOS[args.caso]} if args.caso else CASOS

    print("=" * 78)
    print("HORUS · camino de segmentación — autotest (sin torch, sin pesos)")
    print("=" * 78)
    fallas = 0
    for nombre, f in casos.items():
        t0 = time.perf_counter()
        try:
            ok, detalle = f()
        except Exception as e:                        # pragma: no cover
            ok, detalle = False, f"EXCEPCIÓN {type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {'OK   ' if ok else 'FALLA'}  {nombre:<18} {detalle:<50} {ms:7.1f} ms")
        fallas += not ok
    print("-" * 78)
    print(f"{len(casos) - fallas}/{len(casos)} pruebas OK")
    return 1 if fallas else 0


if __name__ == "__main__":
    raise SystemExit(main())
