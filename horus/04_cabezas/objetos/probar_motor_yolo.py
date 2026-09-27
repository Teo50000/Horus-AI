# -*- coding: utf-8 -*-
"""
probar_motor_yolo.py · el detector preentrenado respeta el contrato del motor.

    python probar_motor_yolo.py

Necesita `pip install ultralytics onnxruntime` y los pesos en modelos/
(yolo11n_416.onnx o yolo11n.pt). No necesita cámara ni GPU.
"""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AQUI)

from objects_head import DEFAULT_CLASSES, Detection  # noqa: E402
from motor_yolo import (ConfigYolo, MotorMixto, MotorYolo,  # noqa: E402
                        pesos_por_defecto)


def _persona_sintetica() -> np.ndarray:
    """Una figura humana de palitos no alcanza para un detector serio, así que
    se prueba el contrato con un cuadro vacío y la mezcla con un motor falso."""
    return np.full((480, 640, 3), 120, dtype=np.uint8)


class _PropioFalso:
    """Devuelve una persona (que la mezcla tiene que tirar) y una pistola."""

    def infer_batch(self, frames, camera_ids=None):
        out = []
        for cam in (camera_ids or ["cam-0"]):
            out.append(SimpleNamespace(
                detections=[
                    Detection((10, 10, 50, 120), 0.9, DEFAULT_CLASSES.index("persona"), "persona", cam),
                    Detection((60, 60, 80, 70), 0.7, DEFAULT_CLASSES.index("pistola"), "pistola", cam),
                ],
                camera_id=cam, frame_idx=0, novelty=0.33, latencia_ms=5.0,
                original_size=(480, 640)))
        return out


def main() -> int:
    casos = []
    pesos = pesos_por_defecto()
    if pesos is None:
        print("FALTAN los pesos: modelos/yolo11n_416.onnx o modelos/yolo11n.pt")
        return 1
    m = MotorYolo(ConfigYolo(verboso=False))

    t0 = time.perf_counter()
    r = m.infer_batch([_persona_sintetica(), _persona_sintetica()], ["cam-1", "cam-2"])
    ms = (time.perf_counter() - t0) * 1000 / 2
    casos.append(("contrato", len(r) == 2 and r[0].camera_id == "cam-1"
                  and r[1].original_size == (480, 640) and hasattr(r[0], "novelty"),
                  f"{len(r)} resultados · {ms:.0f} ms por cuadro"))
    casos.append(("clases_horus", set(m.mapa.values()) <= set(DEFAULT_CLASSES)
                  and "paquete" not in m.mapa.values(),
                  f"usa {sorted(set(m.mapa.values()))} (bolsos apagados)"))
    ids_ok = all(DEFAULT_CLASSES[i] == c for i, c in
                 ((DEFAULT_CLASSES.index(c), c) for c in m.mapa.values()))
    casos.append(("ids_consistentes", ids_ok, "class_id = posición en DEFAULT_CLASSES"))

    mx = MotorMixto(m, _PropioFalso())
    rr = mx.infer_batch([_persona_sintetica()], ["cam-1"])[0]
    etiquetas = sorted(d.label for d in rr.detections)
    casos.append(("mezcla", "pistola" in etiquetas and etiquetas.count("persona") == 0
                  and rr.novelty == 0.33,
                  f"quedan {etiquetas} · la persona de la cabeza propia se tira"))

    ok = 0
    for nombre, bien, detalle in casos:
        print(f"  {'OK   ' if bien else 'FALLA'}  {nombre:<18} {detalle}")
        ok += bien
    print(f"{ok}/{len(casos)} pruebas OK")
    return 0 if ok == len(casos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
