# -*- coding: utf-8 -*-
"""
pipeline.py · HORUS — el cableado del camino caliente.

Junta las piezas en el orden del diagrama:

    cámaras -> objects_engine -> tracking local -> tracking global (ReID)
            -> fusión y decisión -> eventos listos para 07_alerta

Uso como librería
-----------------
    from pipeline import PipelineHorus

    pipe = PipelineHorus(pesos="modelos/objetos_v2.pt",
                         topologia="topologia.json",
                         fps=10.0)

    eventos = pipe.procesar({"cam-1": frame1, "cam-2": frame2})
    for e in eventos:
        print(e.linea())            # y de acá va a 07_alerta

Uso como programa
-----------------
    python pipeline.py 0 --ver --pesos ../04_cabezas/objetos/modelos/objetos_v2.pt
    python pipeline.py rtsp://... --topologia topologia.json

Cabezas opcionales
------------------
Hoy solo la de objetos entra sola. Las demás se inyectan por frame, para no
atar este archivo a modelos que todavía no existen ni obligar a que estén
todos para poder correr:

    pipe.procesar(frames,
                  seg={"cam-1": seg_result},        # SegResult
                  embeddings={"cam-1": [BodyEmbedding, ...]},
                  acciones={"cam-1": [AccionObs, ...]})

Lo que falta para cerrar el camino completo (una sola cosa)
-----------------------------------------------------------
`FusedObjectsNet.forward()` calcula la pirámide FPN y la descarta después de
usarla. Las cabezas de segmentación y de ReID necesitan exactamente esa
pirámide, y volver a calcularla sería pagar el backbone dos veces — que es
justo lo que el diseño de backbone compartido existe para evitar (backbone +
FPN son el 53% del costo por frame, según `dimensionamiento-camaras-por-gpu`).

El cambio es devolver `pyr` junto con `(logits, deltas, emb)` y guardarlo en
el `FrameResult`. Hay que tocar tres lugares y ninguno es grande:
`FusedObjectsNet.forward`, la captura de CUDA Graphs (`_capturar_graph`, que
fija las salidas estáticas) y el exportador a ONNX. Mientras tanto, este
archivo acepta la segmentación y los embeddings inyectados desde afuera, así
que la fusión se puede probar y desplegar sin esperar ese refactor.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
for _p in (_AQUI,
           os.path.join(_RAIZ, "05_tracking"),
           os.path.join(_RAIZ, "05_tracking", "local"),
           os.path.join(_RAIZ, "05_tracking", "global_reid"),
           os.path.join(_RAIZ, "04_cabezas", "objetos")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contratos import (  # noqa: E402
    CABEZA_ACCION, CABEZA_OBJETOS, CABEZA_REID, CABEZA_SEGMENTACION,
    AccionObs, Evento, ObservacionCamara, seg_desde_resultado,
)
from motor_fusion import ConfigFusion, MotorFusion  # noqa: E402
from topologia import Topologia  # noqa: E402
from tracker_local import (  # noqa: E402
    ConfigTracker, TrackerMultiCamara, config_motor_para_tracking,
)
from tracker_global import ConfigGlobal, TrackerGlobal  # noqa: E402


# --------------------------------------------------------------------------- #
class PipelineHorus:
    """Una instancia por caja. Mantiene el estado de todas las cámaras."""

    def __init__(self,
                 pesos: Optional[str] = None,
                 pesos_backbone: Optional[str] = None,
                 topologia: Optional[Any] = None,
                 fps: float = 10.0,
                 max_batch: int = 4,
                 cfg_tracker: Optional[ConfigTracker] = None,
                 cfg_global: Optional[ConfigGlobal] = None,
                 cfg_fusion: Optional[ConfigFusion] = None,
                 motor_objetos: Any = None,
                 device: Optional[str] = None,
                 verboso: bool = True) -> None:
        if isinstance(topologia, str):
            topologia = Topologia.cargar(topologia)
        self.topo: Optional[Topologia] = topologia
        self.fps = float(fps)

        self.objetos = motor_objetos
        if self.objetos is None and pesos is not None:
            self.objetos = self._crear_motor(pesos, pesos_backbone, max_batch,
                                             device, verboso)

        self.tracking = TrackerMultiCamara(cfg_tracker or ConfigTracker(fps=fps))
        self.reid = TrackerGlobal(cfg_global or ConfigGlobal(),
                                  topologia=topologia)
        self.fusion = MotorFusion(topologia=topologia,
                                  cfg=cfg_fusion or ConfigFusion(), fps=fps)

        self.frames = 0
        self.latencias: List[float] = []
        if verboso:
            for aviso in self.fusion.verificar():
                print(f"[pipeline] aviso: {aviso}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _crear_motor(pesos: str, pesos_backbone: Optional[str],
                     max_batch: int, device: Optional[str],
                     verboso: bool) -> Any:
        from objects_engine import EngineConfig, ObjectsEngine
        cfg = EngineConfig(**config_motor_para_tracking())
        cfg.pesos = pesos
        cfg.pesos_backbone = pesos_backbone
        cfg.max_batch = max_batch
        cfg.verboso = verboso
        if device:
            cfg.device = device
            if device == "cpu":
                cfg.precision = "fp32"
                cfg.cuda_graphs = False
        return ObjectsEngine(cfg)

    # ------------------------------------------------------------------ #
    def procesar(self,
                 frames: Dict[str, np.ndarray],
                 seg: Optional[Dict[str, Any]] = None,
                 embeddings: Optional[Dict[str, Sequence[Any]]] = None,
                 acciones: Optional[Dict[str, Sequence[AccionObs]]] = None,
                 ts: Optional[float] = None) -> List[Evento]:
        """Un tick: N cámaras entran, los eventos que hay que avisar salen."""
        t0 = time.perf_counter()
        cams = list(frames)
        ts = time.time() if ts is None else float(ts)

        # 1. Detección — las N cámaras en un solo forward. El batch
        #    multi-cámara es lo que hace que la GPU rinda; a batch 1 no se
        #    llega ni al 20% del pico.
        resultados = self.objetos.infer_batch([frames[c] for c in cams],
                                              camera_ids=cams)

        observaciones: List[ObservacionCamara] = []
        for cam, res in zip(cams, resultados):
            # 2. Tracking local.
            tracks = self.tracking.actualizar_desde_resultado(res)

            # 3. Tracking global, solo si hay embeddings de ReID por persona.
            embs = (embeddings or {}).get(cam)
            hay_reid = False
            if embs:
                self.reid.actualizar(tracks, embs,
                                     tam_frame=res.original_size, ts=ts)
                hay_reid = True

            # 4. Armar la observación para la fusión.
            cabezas = {CABEZA_OBJETOS}
            sr = (seg or {}).get(cam)
            if sr is not None:
                cabezas.add(CABEZA_SEGMENTACION)
            acs = list((acciones or {}).get(cam) or ())
            for a in acs:
                cabezas.add(a.fuente if a.fuente in (CABEZA_ACCION,) else a.fuente)
            if hay_reid:
                cabezas.add(CABEZA_REID)

            observaciones.append(ObservacionCamara(
                camera_id=cam, frame_idx=res.frame_idx, ts=ts,
                tracks=tracks,
                seg=seg_desde_resultado(sr) if sr is not None else None,
                acciones=acs,
                novelty=res.novelty, incertidumbre=res.incertidumbre,
                tam_frame=res.original_size,
                cabezas=frozenset(cabezas)))

        # 5. Fusión y decisión.
        eventos = self.fusion.procesar(observaciones, ts=ts)

        self.frames += 1
        self.latencias.append((time.perf_counter() - t0) * 1000.0)
        if len(self.latencias) > 600:
            del self.latencias[:300]
        return eventos

    # ------------------------------------------------------------------ #
    @property
    def abiertos(self) -> List[Evento]:
        return self.fusion.abiertos

    def resumen(self) -> str:
        lat = np.asarray(self.latencias[-300:]) if self.latencias else np.zeros(1)
        return "\n".join([
            f"[pipeline] {self.frames} ticks · latencia mediana "
            f"{np.median(lat):.1f} ms · p95 {np.percentile(lat, 95):.1f} ms",
            self.tracking.resumen(),
            self.reid.resumen(),
            self.fusion.resumen(),
        ])

    def reset(self) -> None:
        self.tracking.reset()
        self.reid.reset()
        self.fusion.reset()


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fuente", help="índice de webcam, archivo de video o RTSP")
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--pesos-backbone", dest="pesos_backbone")
    ap.add_argument("--topologia", help="topologia.json")
    ap.add_argument("--camara", default="cam-0")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--ver", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        print("falta opencv-python")
        return 2

    if args.topologia and not os.path.exists(args.topologia):
        print(f"no existe {args.topologia}")
        return 2
    if args.topologia is None:
        print("[pipeline] sin topología: intrusión y merodeo por zona no van "
              "a poder evaluarse. Copiá topologia.example.json y editalo.")

    pipe = PipelineHorus(pesos=args.pesos, pesos_backbone=args.pesos_backbone,
                         topologia=args.topologia, fps=args.fps,
                         device="cpu" if args.cpu else None)

    fuente = int(args.fuente) if str(args.fuente).isdigit() else args.fuente
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"no se pudo abrir {args.fuente!r}")
        return 2

    cam = args.camara
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for ev in pipe.procesar({cam: frame}):
                print(ev.linea())

            if args.ver:
                for t in pipe.tracking.tracker(cam).tracks:
                    if t.estado == "tentativo":
                        continue
                    x1, y1, x2, y2 = (int(v) for v in t.bbox_xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
                    cv2.putText(frame, f"#{t.track_id} {t.clase}",
                                (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (60, 220, 60), 1, cv2.LINE_AA)
                y = 40
                for ev in pipe.abiertos[:6]:
                    cv2.putText(frame, ev.linea()[:110], (8, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1,
                                cv2.LINE_AA)
                    y += 20
                cv2.imshow("HORUS · pipeline", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
    finally:
        cap.release()
        if args.ver:
            cv2.destroyAllWindows()

    print()
    print(pipe.resumen())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
