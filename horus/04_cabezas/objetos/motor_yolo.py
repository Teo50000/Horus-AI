# -*- coding: utf-8 -*-
"""
motor_yolo.py · detector preentrenado (YOLO11) con el mismo contrato que
`ObjectsEngine`.

26/09 — por qué existe
----------------------
"Los modelos solo sacan conclusiones en condiciones perfectas." Medido sobre
293 cuadros del cuarto vacío de Teo:

    cuadros con una persona que no existe (confianza ≥ 0,25)
      nuestra cabeza v3 / v5 ........ 239 / 234
      YOLO11n preentrenado ..........  23        (sin mostrarle nada de la casa)
      YOLO11s preentrenado ..........  15

Nuestra cabeza se entrenó de cero con ~14.000 fotos arriba de un backbone de
clasificación: encuentra personas con AP50 0,25. YOLO11 viene entrenado en COCO,
con cientos de miles de personas en todas las poses, luces y cortes posibles.
Y todo lo demás se apoya en las cajas de persona: caídas (recorta la persona),
agresión (compuerta de 2+ personas), "arma con portador", merodeo.

Qué hace cada uno
-----------------
- `MotorYolo`: YOLO11 de COCO. De sus 80 clases se usan las que Horus ya
  tiene: persona, cuchillo, celular. Mochila/cartera/valija se pueden mapear a
  `paquete` con `bolsos=True`, pero vienen APAGADAS: en la casa de Teo la silla
  con ropa sale como "valija" en 64 de 293 cuadros, y en una casa hay bolsos
  "abandonados" todo el día.
- `MotorMixto`: YOLO para esas clases + la cabeza propia SOLO para lo que COCO
  no sabe (pistola, humo, llama, paquete). Es el puente hasta que el reentreno
  de YOLO11 con esas cuatro clases esté listo; después la cabeza vieja se
  jubila.

Corre en PyTorch (`.pt`) o en ONNX Runtime (`.onnx`, ~1,8x más rápido en CPU,
medido: 32 ms contra 52 ms a 416 px). Licencia de YOLO11: AGPL-3.0.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from objects_head import DEFAULT_CLASSES, Detection

_AQUI = os.path.dirname(os.path.abspath(__file__))
MODELOS = os.path.join(_AQUI, "modelos")

# COCO -> Horus. Los ids de Horus son las posiciones en DEFAULT_CLASSES, así
# el tracker y la fusión no se enteran de que cambió el detector.
COCO_A_HORUS: Dict[int, str] = {0: "persona", 43: "cuchillo", 67: "celular"}
COCO_BOLSOS: Dict[int, str] = {24: "paquete", 26: "paquete", 28: "paquete"}
ID_HORUS = {c: i for i, c in enumerate(DEFAULT_CLASSES)}

# Qué clases le quedan a la cabeza propia en el modo mixto.
CLASES_CABEZA_PROPIA = ("pistola", "humo", "llama", "paquete")


def pesos_por_defecto() -> Optional[str]:
    """El primero que exista: ONNX (más rápido) y si no el .pt."""
    for nombre in ("yolo11n_416.onnx", "yolo11n.onnx", "yolo11n.pt"):
        ruta = os.path.join(MODELOS, nombre)
        if os.path.exists(ruta):
            return ruta
    return None


def pesos_horus_por_defecto() -> Optional[str]:
    """El YOLO reentrenado con pistola, humo, llama y paquete
    (`entrenar_yolo.py`). Si está, reemplaza a la cabeza propia entera."""
    for nombre in ("horus_yolo4_416.onnx", "horus_yolo4.onnx", "horus_yolo4.pt"):
        ruta = os.path.join(MODELOS, nombre)
        if os.path.exists(ruta):
            return ruta
    return None


@dataclass
class ConfigYolo:
    pesos: Optional[str] = None             # None = pesos_por_defecto()
    tam: int = 416                          # lado de entrada
    conf: float = 0.10                      # piso: el tracker aplica los suyos
    iou: float = 0.50
    device: str = "cpu"
    bolsos: bool = False
    # Clases a usar de ESTE modelo. Para un YOLO reentrenado con clases de
    # Horus (pistola, humo...) se deja en None y se usan sus nombres tal cual.
    mapa: Optional[Dict[int, str]] = None
    verboso: bool = True


@dataclass
class ResultadoYolo:
    """Mismo contrato que `objects_engine.FrameResult` (duck typing)."""
    detections: List[Detection]
    camera_id: str
    frame_idx: int
    novelty: float = 0.0
    input_size: Tuple[int, int] = (0, 0)
    original_size: Tuple[int, int] = (0, 0)
    latencia_ms: float = 0.0
    embedding: Any = None
    incertidumbre: float = 0.0

    @property
    def criticas(self) -> List[Detection]:
        from objects_head import CRITICAL_CLASSES
        return [d for d in self.detections if d.label in CRITICAL_CLASSES]


class MotorYolo:
    def __init__(self, cfg: Optional[ConfigYolo] = None) -> None:
        self.cfg = cfg or ConfigYolo()
        c = self.cfg
        if not c.pesos:
            c.pesos = pesos_por_defecto()
        if not c.pesos or not os.path.exists(c.pesos):
            raise FileNotFoundError(
                f"no están los pesos de YOLO ({c.pesos or 'modelos/yolo11n*.onnx|pt'})")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("falta `pip install ultralytics` (y `onnxruntime` "
                              "para el .onnx)") from e
        if c.pesos.endswith(".onnx"):
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                base = os.path.basename(c.pesos).split("_")[0].split(".")[0]
                alt = os.path.join(os.path.dirname(c.pesos),
                                   "horus_yolo4.pt" if base == "horus" else "yolo11n.pt")
                if not os.path.exists(alt):
                    raise ImportError("el .onnx necesita `pip install onnxruntime`")
                if c.verboso:
                    print("[yolo] sin onnxruntime: uso el .pt (más lento). "
                          "pip install onnxruntime")
                c.pesos = alt
        self.modelo = YOLO(c.pesos, task="detect")
        nombres = getattr(self.modelo, "names", None) or {}
        if c.mapa is not None:
            self.mapa = dict(c.mapa)
        elif any(str(v) in DEFAULT_CLASSES for v in dict(nombres).values()):
            # Un YOLO reentrenado con los nombres de Horus.
            self.mapa = {int(k): str(v) for k, v in dict(nombres).items()
                         if str(v) in DEFAULT_CLASSES}
        else:
            self.mapa = dict(COCO_A_HORUS)
            if c.bolsos:
                self.mapa.update(COCO_BOLSOS)
        self._frames: Dict[str, int] = {}
        # Primera pasada fuera del camino caliente (ONNX arma la sesión acá).
        self.modelo.predict(np.zeros((c.tam, c.tam, 3), np.uint8), imgsz=c.tam,
                            conf=c.conf, verbose=False, device=c.device)
        if c.verboso:
            print(f"[yolo] {os.path.basename(c.pesos)} a {c.tam} px · clases: "
                  f"{sorted(set(self.mapa.values()))}")

    def infer_batch(self, frames: Sequence[np.ndarray],
                    camera_ids: Optional[Sequence[str]] = None) -> List[ResultadoYolo]:
        c = self.cfg
        cams = list(camera_ids or [f"cam-{i}" for i in range(len(frames))])
        salida: List[ResultadoYolo] = []
        for frame, cam in zip(frames, cams):
            t0 = time.perf_counter()
            n = self._frames.get(cam, 0)
            self._frames[cam] = n + 1
            r = self.modelo.predict(frame, imgsz=c.tam, conf=c.conf, iou=c.iou,
                                    verbose=False, device=c.device)[0]
            dets: List[Detection] = []
            if r.boxes is not None and len(r.boxes):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                xyxy = r.boxes.xyxy.cpu().numpy()
                for k, s, b in zip(cls, conf, xyxy):
                    etiqueta = self.mapa.get(int(k))
                    if etiqueta is None:
                        continue
                    dets.append(Detection(
                        bbox_xyxy=tuple(float(v) for v in b), score=float(s),
                        class_id=ID_HORUS[etiqueta], label=etiqueta,
                        camera_id=cam, frame_idx=n))
            h, w = np.asarray(frame).shape[:2]
            salida.append(ResultadoYolo(
                detections=dets, camera_id=cam, frame_idx=n,
                input_size=(c.tam, c.tam), original_size=(h, w),
                latencia_ms=(time.perf_counter() - t0) * 1000))
        return salida

    def infer(self, frame: np.ndarray, camera_id: str = "cam-0") -> ResultadoYolo:
        return self.infer_batch([frame], [camera_id])[0]


class MotorMixto:
    """YOLO para lo que sabe + la cabeza propia para lo que COCO no tiene.

    Se queda con la `novelty` y el `embedding` de la cabeza propia (los usan
    el gate del VLM y el ReID), y con las detecciones de cada uno SOLO en sus
    clases: la persona de la cabeza propia se descarta, que es justamente la
    que inventaba gente en la silla.
    """

    def __init__(self, yolo: MotorYolo, propio: Any,
                 clases_propio: Sequence[str] = CLASES_CABEZA_PROPIA) -> None:
        self.yolo = yolo
        self.propio = propio
        self.clases_propio = set(clases_propio) - set(yolo.mapa.values())

    def infer_batch(self, frames: Sequence[np.ndarray],
                    camera_ids: Optional[Sequence[str]] = None) -> List[Any]:
        ry = self.yolo.infer_batch(frames, camera_ids)
        rp = self.propio.infer_batch(frames, camera_ids=camera_ids)
        salida = []
        for a, b in zip(ry, rp):
            b.detections = ([d for d in b.detections if d.label in self.clases_propio]
                            + list(a.detections))
            b.latencia_ms = float(getattr(b, "latencia_ms", 0.0)) + a.latencia_ms
            salida.append(b)
        return salida

    def infer(self, frame: np.ndarray, camera_id: str = "cam-0") -> Any:
        return self.infer_batch([frame], [camera_id])[0]
