# -*- coding: utf-8 -*-
"""
segmentation_engine.py · HORUS — la cabeza de segmentación en producción.

Lo que faltaba para que fuego, humo y agua dejen de estar dormidos. El modelo
está entrenado y medido desde hace tres semanas (F1 a nivel alerta: agua
95,4 % · humo 95,6 % · **fuego 99,1 %**), los pesos están en el disco, y la
regla de incendio está escrita y probada. Lo único que no existía era el
pedazo que produce la `SegObs` que la regla come: `pipeline.py` recibe `seg`
como argumento y nadie lo llenaba.

Por qué esto importa más de lo que parece
-----------------------------------------
`ReglaIncendio` declara `requiere = {}` y `requiere_alguna = {objetos,
segmentacion}`: **corre con segmentación sola**. Y el fuego de segmentación
por sí solo ya es evidencia fuerte. O sea que este archivo abre el único
camino completo —cámara → modelo → fusión → alerta → backend— que **no
depende del `backbone.pt` de objetos**, que sigue perdido.

Las cuatro decisiones que no son obvias
---------------------------------------
**1 · No se llama a `SharedBackbone.forward()`.** Calcula el embedding, el
GRU temporal y la novedad, y la novedad hace `float(...norm())`, que es un
sync GPU→CPU por batch. Se usa `backbone.fpn(backbone.extractor(x))`, igual
que en objetos.

**2 · El argmax y el upsample final quedan afuera del motor.** Por la misma
razón que el NMS en objetos: meterlos adentro ata los umbrales al build del
engine de TensorRT, y en vigilancia los umbrales se tocan seguido.
`set_umbrales()` existe justamente porque quedaron del lado barato.

**3 · Los umbrales son POR CLASE y no por gusto.** `(fondo 0, agua 0.005,
humo 0.020, fuego 0.002)`, medidos sobre 4.097 imágenes de validación. Un
solo número deja ~9 puntos de F1 sobre la mesa: el fuego vive en regiones
chicas (1,6 % del cuadro de mediana) y un piso de 0,02 se come un tercio de
los incendios reales; el humo es al revés, subir el piso filtra la neblina y
gana 7,5 puntos.

**4 · Si el backbone no es el suyo, corta.** La cabeza está entrenada contra
ESE backbone exacto. Con otro no falla: predice mal, en silencio, que es
exactamente lo que dejó a la cabeza de objetos alucinando un mes entero.

Uso
---
    from segmentation_engine import MotorSegmentacion, ConfigSegmentacion

    seg = MotorSegmentacion(ConfigSegmentacion(
        pesos="checkpoints/head_v4_produccion.pt"))
    resultados = seg.infer_batch([frame1, frame2], camera_ids=["cam-1", "cam-2"])

    # y a la fusión:
    pipe.procesar(frames, seg={"cam-1": resultados[0]})

Verificación: `python probar_segmentation_engine.py` — sin torch ni pesos.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.normpath(os.path.join(_AQUI, "..", ".."))
for _p in (_AQUI, os.path.join(_RAIZ, "03_backbone")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLASES_POR_DEFECTO = ("fondo", "agua", "humo", "fuego")


class ErrorCheckpointSegmentacion(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
@dataclass
class ConfigSegmentacion:
    pesos: str = os.path.join(_AQUI, "checkpoints", "head_v4_produccion.pt")
    pesos_backbone: Optional[str] = None   # por defecto, el que anota el checkpoint
    device: Optional[str] = None           # None = cuda si hay
    half: bool = True                      # fp16 en el camino caliente
    channels_last: bool = True
    tf32: bool = True
    max_batch: int = 4

    # Overrides opcionales. None = los calibrados que trae el checkpoint.
    area_thresh: Optional[Sequence[float]] = None
    prob_thresh: Optional[float] = None

    verboso: bool = True


# --------------------------------------------------------------------------- #
class MotorSegmentacion:
    """Frames -> `SegResult`, que es lo que come `seg_desde_resultado()`.

    `motor_falso` permite inyectar una función `(frames, camera_ids) ->
    List[SegResult]` para probar todo el cableado sin torch, sin pesos y sin
    GPU — la misma propiedad que hace que la fusión se pruebe en cualquier
    máquina.
    """

    def __init__(self, cfg: Optional[ConfigSegmentacion] = None,
                 motor_falso: Any = None) -> None:
        self.cfg = cfg or ConfigSegmentacion()
        self._falso = motor_falso
        self.clases: Tuple[str, ...] = CLASES_POR_DEFECTO
        self.tam = 384
        self.stats = {"frames": 0, "lotes": 0, "ms_total": 0.0}
        self._torch = None
        if motor_falso is None:
            self._cargar()

    # ------------------------------------------------------------------ #
    def _cargar(self) -> None:
        ruta = self.cfg.pesos
        if not os.path.exists(ruta):
            raise ErrorCheckpointSegmentacion(
                f"No existe {ruta}.\n"
                f"Los pesos de segmentación se rescataron el 27/08 de un stash "
                f"y viven SOLO en este disco: head_v4_produccion.pt (18,9 MB) y "
                f"backbone.pt (108 MB), en 04_cabezas/segmentacion/checkpoints/.")

        try:
            import torch  # noqa: WPS433
            from shared_backbone import BackboneConfig, SharedBackbone
            from segmentation_head import SegmentationHead, SegmentationHeadConfig
        except ImportError as e:  # pragma: no cover
            raise ErrorCheckpointSegmentacion(
                f"falta una dependencia para correr segmentación ({e}). "
                f"Instalá torch/torchvision, o inyectá `motor_falso=` para "
                f"probar el cableado sin el modelo.") from e

        dev = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ruta, map_location="cpu", weights_only=False)

        self.clases = tuple(ck.get("clases", CLASES_POR_DEFECTO))
        self.tam = int(ck.get("tam", 384))

        # --- backbone: el suyo, no otro -------------------------------- #
        ruta_bb = self.cfg.pesos_backbone or os.path.join(
            os.path.dirname(os.path.abspath(ruta)),
            ck.get("backbone_file", "backbone.pt"))
        if not os.path.exists(ruta_bb):
            raise ErrorCheckpointSegmentacion(
                f"Falta {ruta_bb}.\nLa cabeza está entrenada contra ESE backbone "
                f"exacto; con otro no falla, predice mal en silencio. Es la "
                f"misma trampa que dejó a la cabeza de objetos alucinando un mes.")

        backbone = SharedBackbone(BackboneConfig(
            pretrained=False, input_size=self.tam, freeze_encoder=True))
        estado_bb = torch.load(ruta_bb, map_location="cpu", weights_only=False)
        self._verificar(backbone, estado_bb, os.path.basename(ruta_bb))
        backbone = backbone.to(dev).eval()

        # --- cabeza ---------------------------------------------------- #
        cfg_head = SegmentationHeadConfig(
            classes=self.clases, in_channels=backbone.cfg.fpn_dim)
        if self.cfg.area_thresh is not None:
            cfg_head.area_thresh = tuple(float(v) for v in self.cfg.area_thresh)
        if self.cfg.prob_thresh is not None:
            cfg_head.prob_thresh = float(self.cfg.prob_thresh)
        head = SegmentationHead(cfg_head)
        estado_h = ck.get("head") or ck.get("head_raw")
        if estado_h is None:
            raise ErrorCheckpointSegmentacion(
                f"{ruta} no trae ni 'head' ni 'head_raw': no es un checkpoint "
                f"de esta cabeza.")
        self._verificar(head, estado_h, os.path.basename(ruta))
        head = head.to(dev).eval()

        # --- parte afinada (--afinar privado) -------------------------- #
        # Copia propia de layer4 y del FPN: +39 % de cómputo en el camino
        # caliente en vez de +100 %, porque el tronco se sigue compartiendo.
        l4 = fpn_priv = None
        afin = ck.get("afinado")
        if afin:
            import copy
            from torchvision.models import resnet50
            l4 = resnet50(weights=None).layer4
            l4.load_state_dict(afin["layer4"])
            l4 = l4.to(dev).eval()
            fpn_priv = copy.deepcopy(backbone.fpn)
            fpn_priv.load_state_dict(afin["fpn"])
            fpn_priv = fpn_priv.to(dev).eval()

        # --- precisión y layout ---------------------------------------- #
        if self.cfg.tf32 and dev.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self._half = bool(self.cfg.half and dev.startswith("cuda"))
        if self.cfg.channels_last:
            backbone = backbone.to(memory_format=torch.channels_last)
            if l4 is not None:
                l4 = l4.to(memory_format=torch.channels_last)

        self._torch, self._dev = torch, dev
        self.backbone, self.head, self._l4, self._fpn_priv = backbone, head, l4, fpn_priv
        self._media = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self._desvio = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

        if self.cfg.verboso:
            print(f"[seg] {os.path.basename(ruta)} · época {ck.get('epoca')} · "
                  f"mIoU {ck.get('mejor_miou')} · clases {list(self.clases)} · {dev}"
                  + ("  (+layer4/FPN afinados)" if afin else ""))
            print("[seg] umbrales de alerta: " + "  ".join(
                f"{self.clases[k]} {100 * head.umbral(k):.2f}%"
                for k in range(1, len(self.clases))))

    @staticmethod
    def _verificar(modulo: Any, estado: Dict[str, Any], nombre: str) -> None:
        """Carga estricta. Un `.pt` que entra con `strict=False` y devuelve
        máscaras no es un `.pt` que anda."""
        faltan, sobran = modulo.load_state_dict(estado, strict=False)
        if faltan or sobran:
            raise ErrorCheckpointSegmentacion(
                f"{nombre} no encaja con {type(modulo).__name__}: faltan "
                f"{len(faltan)} tensores y sobran {len(sobran)}.\n"
                f"  faltan: {list(faltan)[:4]}\n  sobran: {list(sobran)[:4]}")

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def infer_batch(self, frames: Sequence[np.ndarray],
                    camera_ids: Optional[Sequence[str]] = None,
                    frame_idxs: Optional[Sequence[int]] = None,
                    ts: Optional[float] = None) -> List[Any]:
        """N cámaras en un solo forward.

        A batch 1 no se llega ni al 20 % del pico de la placa: el batch
        multi-cámara es lo que hace que la cuenta de dimensionamiento cierre.
        """
        if not frames:
            return []
        ts = time.time() if ts is None else float(ts)
        t0 = time.perf_counter()

        if self._falso is not None:
            out = list(self._falso(frames, camera_ids))
        else:
            out = []
            for i in range(0, len(frames), self.cfg.max_batch):
                lote = frames[i:i + self.cfg.max_batch]
                cams = (list(camera_ids[i:i + self.cfg.max_batch])
                        if camera_ids else None)
                idxs = (list(frame_idxs[i:i + self.cfg.max_batch])
                        if frame_idxs else None)
                out.extend(self._inferir_lote(lote, cams, idxs))

        for r in out:
            if not getattr(r, "ts", 0):
                r.ts = ts
        self.stats["frames"] += len(frames)
        self.stats["lotes"] += 1
        self.stats["ms_total"] += (time.perf_counter() - t0) * 1000.0
        return out

    def infer(self, frame: np.ndarray, camera_id: str = "cam-0",
              frame_idx: int = -1, ts: Optional[float] = None) -> Any:
        return self.infer_batch([frame], [camera_id], [frame_idx], ts)[0]

    def set_umbrales(self, area: Optional[Dict[str, float]] = None,
                     prob: Optional[float] = None) -> None:
        """Cambia los umbrales en caliente, sin reconstruir nada.

        Es la contrapartida práctica de haber dejado el argmax y el umbral
        fuera del engine: en una instalación real esto se toca contra la
        escena, no se recompila.
        """
        if prob is not None:
            self.head.cfg.prob_thresh = float(prob)
        if area:
            actual = list(self.head.cfg.area_thresh)
            for clase, v in area.items():
                if clase in self.clases:
                    actual[self.clases.index(clase)] = float(v)
            self.head.cfg.area_thresh = tuple(actual)

    def resumen(self) -> Dict[str, Any]:
        n = max(1, self.stats["frames"])
        return dict(self.stats, ms_por_frame=round(self.stats["ms_total"] / n, 2),
                    clases=list(self.clases), tam=self.tam)

    # ------------------------------------------------------------------ #
    def _inferir_lote(self, frames: Sequence[np.ndarray],
                      camera_ids: Optional[Sequence[str]],
                      frame_idxs: Optional[Sequence[int]]) -> List[Any]:
        torch = self._torch
        x, tam_orig = self._preprocesar(frames)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=self._half):
                feats = self.backbone.extractor(x)
                if self._l4 is not None:
                    pir = self._fpn_priv({"p3": feats["p3"], "p4": feats["p4"],
                                          "p5": self._l4(feats["p4"])})
                else:
                    pir = self.backbone.fpn(feats)
            # El predict de la cabeza va en fp32: el softmax y el conteo de
            # área en fp16 pierden precisión justo en las regiones chicas, que
            # es donde vive el fuego.
            pir = {k: v.float() for k, v in pir.items()}
            return self.head.predict(
                pir, image_sizes=[tam_orig] * len(frames),
                camera_ids=list(camera_ids) if camera_ids else None,
                frame_idxs=list(frame_idxs) if frame_idxs else None)

    def _preprocesar(self, frames: Sequence[np.ndarray]):
        """uint8 al bus, float en la GPU.

        Normalizar en el host y subir float32 cuesta 4x más bytes por PCIe
        que subir uint8 y normalizar del otro lado. Es el mismo arreglo que
        ya tenía el motor de objetos.
        """
        torch = self._torch
        arr = np.stack([np.ascontiguousarray(f) for f in frames])
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"se esperaban frames HxWx3, llegó {arr.shape}")
        tam_orig = (int(arr.shape[1]), int(arr.shape[2]))

        t = torch.from_numpy(arr).to(self._dev, non_blocking=True)
        t = t.permute(0, 3, 1, 2).float()
        if t.max() > 1.5:
            t = t / 255.0
        t = torch.nn.functional.interpolate(
            t, size=(self.tam, self.tam), mode="bilinear", align_corners=False)
        t = (t - self._media) / self._desvio
        if self.cfg.channels_last:
            t = t.contiguous(memory_format=torch.channels_last)
        return t, tam_orig


# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Motor de segmentación de Horus")
    ap.add_argument("fuente", nargs="?", default="0",
                    help="índice de webcam, archivo de video o imagen")
    ap.add_argument("--pesos", default=ConfigSegmentacion.pesos)
    ap.add_argument("--pesos-backbone", dest="pesos_backbone")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()

    motor = MotorSegmentacion(ConfigSegmentacion(
        pesos=args.pesos, pesos_backbone=args.pesos_backbone,
        device="cpu" if args.cpu else None, half=not args.cpu))

    import cv2
    fuente = int(args.fuente) if str(args.fuente).isdigit() else args.fuente
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"No pude abrir {args.fuente}")
        return 1
    try:
        for i in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                break
            r = motor.infer(frame[:, :, ::-1], camera_id="cam-0", frame_idx=i)
            activas = r.clases_activas
            print(f"  frame {i:>3}  {activas or '-'}  "
                  + "  ".join(f"{k} area={v:.4f} score={r.score[k]:.2f}"
                              for k, v in r.area.items())
                  + ("  [VLM]" if r.needs_vlm else ""))
    finally:
        cap.release()
    print(motor.resumen())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
