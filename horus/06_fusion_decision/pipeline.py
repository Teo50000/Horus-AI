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
    CABEZA_ACCION, CABEZA_OBJETOS, CABEZA_POSE, CABEZA_REID, CABEZA_SEGMENTACION,
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
                 motor_segmentacion: Any = None,
                 detector_agresion: Any = None,
                 detector_caidas: Any = None,
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

        # Cabeza de segmentación. Se puede correr SOLA: `ReglaIncendio` declara
        # `requiere_alguna = {objetos, segmentacion}`, y el fuego de
        # segmentación por sí solo ya es evidencia fuerte (F1 99,1 % medido).
        # Es el único camino completo a una alerta real que no depende del
        # `backbone.pt` de objetos, que sigue perdido.
        self.segmentacion = motor_segmentacion
        if self.objetos is None and self.segmentacion is None:
            raise ValueError(
                "el pipeline necesita al menos una cabeza: pasá `pesos` (objetos) "
                "o `motor_segmentacion`. Arrancar sin ninguna daría un sistema "
                "que no mira nada y no lo dice.")

        self.tracking = TrackerMultiCamara(cfg_tracker or ConfigTracker(fps=fps))
        self.reid = TrackerGlobal(cfg_global or ConfigGlobal(),
                                  topologia=topologia)
        self.fusion = MotorFusion(topologia=topologia,
                                  cfg=cfg_fusion or ConfigFusion(), fps=fps)

        # Cabeza de agresión (fight/). Apagada por defecto: cuesta una pasada
        # de MC3-18 cada 0,75 s por cámara con dos personas en cuadro.
        self.agresion = self._crear_agresion(detector_agresion)

        # Cabeza de caídas (fall/). Mismo trato que agresión: apagada por
        # defecto, y cuando está se declara por estar INSTALADA. Los pesos son
        # un ST-GCN de 103.074 parámetros sobre los 33 puntos de MediaPipe.
        self.caidas = self._crear_caidas(detector_caidas)

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
    @staticmethod
    def _crear_agresion(pedido: Any) -> Any:
        """Si se pide y no se puede cargar, **el pipeline no arranca**.

        Arrancar "igual pero sin agresión" haría que el operador crea que el
        sistema la está mirando, que es la misma razón por la que una regla
        sin su cabeza se declara dormida en vez de devolver "no pasó nada".
        """
        if not pedido:
            return None
        import sys as _sys
        _fight = os.path.normpath(os.path.join(_AQUI, "..", "fight"))
        if _fight not in _sys.path:
            _sys.path.insert(0, _fight)
        from detector_agresion import ConfigAgresion, DetectorAgresion
        cfg = pedido if isinstance(pedido, ConfigAgresion) else ConfigAgresion()
        return DetectorAgresion(cfg)

    @staticmethod
    def _crear_caidas(pedido: Any):
        """Igual que `_crear_agresion`: se acepta el detector ya armado, una
        `ConfigCaidas`, o True para el default.

        A diferencia de agresión, acá NO hay `procesar_lote`: el recorte de
        cada persona sale de su `TrackLocal`, así que se llama por cámara y
        después del tracking. Es la misma razón por la que no se puede
        inyectar desde afuera — quien llama al pipeline todavía no tiene los
        tracks.
        """
        if not pedido:
            return None
        import sys as _sys
        _fall = os.path.normpath(os.path.join(_AQUI, "..", "fall"))
        for _p in (_fall, os.path.join(_fall, "src")):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
        from detector_caidas import ConfigCaidas, DetectorCaidas
        if isinstance(pedido, DetectorCaidas):
            return pedido
        cfg = pedido if isinstance(pedido, ConfigCaidas) else ConfigCaidas()
        return DetectorCaidas(cfg)

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
        if self.objetos is not None:
            resultados = self.objetos.infer_batch([frames[c] for c in cams],
                                                  camera_ids=cams)
        else:
            resultados = [None] * len(cams)

        # 1.b Segmentación — también batcheada, y también antes de la fusión.
        seg = dict(seg or {})
        if self.segmentacion is not None:
            faltan = [c for c in cams if c not in seg]
            if faltan:
                for cam, r in zip(faltan, self.segmentacion.infer_batch(
                        [frames[c] for c in faltan], camera_ids=faltan, ts=ts)):
                    seg[cam] = r

        por_cam: Dict[str, Any] = {}
        for cam, res in zip(cams, resultados):
            # 2. Tracking local. Sin cabeza de objetos no hay nada que trackear,
            #    y eso NO es lo mismo que "no había nadie": las reglas que
            #    necesitan tracks quedan dormidas y se declaran como tales.
            tracks = (self.tracking.actualizar_desde_resultado(res)
                      if res is not None else [])
            tam = (res.original_size if res is not None
                   else tuple(np.asarray(frames[cam]).shape[:2]))

            # 3. Tracking global, solo si hay embeddings de ReID por persona.
            embs = (embeddings or {}).get(cam)
            hay_reid = False
            if embs and tracks:
                self.reid.actualizar(tracks, embs, tam_frame=tam, ts=ts)
                hay_reid = True
            por_cam[cam] = (res, tracks, hay_reid, tam)

        # 3.b Agresión — DESPUÉS del tracking, porque la compuerta se calcula
        #     sobre los tracks, y con todas las cámaras en un solo forward.
        acciones_agresion: Dict[str, List[AccionObs]] = {}
        if self.agresion is not None:
            acciones_agresion = self.agresion.procesar_lote(
                [(cam, frames[cam], por_cam[cam][1]) for cam in cams], ts=ts)

        # 3.c Caídas — también después del tracking, pero por cámara: el
        #     recorte de cada persona sale de su caja, no del frame entero.
        acciones_caidas: Dict[str, List[AccionObs]] = {}
        if self.caidas is not None:
            for cam in cams:
                acciones_caidas[cam] = self.caidas.procesar(
                    cam, frames[cam], por_cam[cam][1], ts=ts)

        observaciones: List[ObservacionCamara] = []
        for cam in cams:
            res, tracks, hay_reid, tam = por_cam[cam]

            # 4. Armar la observación para la fusión.
            cabezas = set()
            if self.objetos is not None:
                cabezas.add(CABEZA_OBJETOS)
            sr = seg.get(cam)
            if sr is not None:
                cabezas.add(CABEZA_SEGMENTACION)
            acs = list((acciones or {}).get(cam) or ())
            acs.extend(acciones_agresion.get(cam) or ())
            acs.extend(acciones_caidas.get(cam) or ())
            for a in acs:
                cabezas.add(a.fuente if a.fuente in (CABEZA_ACCION,) else a.fuente)
            if self.agresion is not None:
                # Se declara por tener el detector INSTALADO, no por haber
                # emitido algo: un frame sin dos personas en cuadro no produce
                # etiqueta, y sin esto la regla figuraría dormida justo en los
                # ratos tranquilos.
                cabezas.add(CABEZA_ACCION)
            if self.caidas is not None:
                # Mismo criterio: instalada, no "emitió". El detector de caídas
                # habla por dos cabezas — declara CABEZA_POSE siempre y emite
                # CABEZA_ACCION cuando además clasifica.
                cabezas.add(CABEZA_POSE)
                cabezas.add(CABEZA_ACCION)
            if hay_reid:
                cabezas.add(CABEZA_REID)

            observaciones.append(ObservacionCamara(
                camera_id=cam,
                frame_idx=res.frame_idx if res is not None else self.frames,
                ts=ts,
                tracks=tracks,
                seg=seg_desde_resultado(sr) if sr is not None else None,
                acciones=acs,
                novelty=getattr(res, "novelty", 0.0) if res is not None else 0.0,
                incertidumbre=(getattr(res, "incertidumbre", 0.0)
                               if res is not None else 0.0),
                tam_frame=tam,
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
    ap.add_argument("--pesos", help="pesos de la cabeza de objetos")
    ap.add_argument("--pesos-backbone", dest="pesos_backbone")
    ap.add_argument("--topologia", help="topologia.json")
    ap.add_argument("--camara", default="cam-0")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--ver", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--agresion", action="store_true",
                    help="prender la cabeza de agresión (fight/, MC3-18)")
    ap.add_argument("--segmentacion", nargs="?", const="auto", default=None,
                    metavar="PESOS",
                    help="prender la cabeza de segmentación (fuego/humo/agua). "
                         "Sin valor usa el checkpoint por defecto.")
    ap.add_argument("--sin-objetos", action="store_true",
                    help="correr SOLO con segmentación: no necesita el "
                         "backbone.pt de objetos")
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

    if not args.pesos and not args.segmentacion:
        ap.error("hace falta --pesos (objetos) o --segmentacion")

    motor_seg = None
    if args.segmentacion:
        sys.path.insert(0, os.path.normpath(
            os.path.join(_AQUI, "..", "04_cabezas", "segmentacion")))
        from segmentation_engine import ConfigSegmentacion, MotorSegmentacion
        cfg_seg = ConfigSegmentacion(device="cpu" if args.cpu else None,
                                     half=not args.cpu)
        if args.segmentacion != "auto":
            cfg_seg.pesos = args.segmentacion
        motor_seg = MotorSegmentacion(cfg_seg)

    pipe = PipelineHorus(detector_agresion=args.agresion,
                         motor_segmentacion=motor_seg,
                         pesos=None if args.sin_objetos else args.pesos,
                         pesos_backbone=args.pesos_backbone,
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
