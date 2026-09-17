# -*- coding: utf-8 -*-
"""
objects_engine.py · HORUS — motor de inferencia CUDA de la cabeza de objetos.

Este archivo es el "camino caliente": lo que corre 24/7 contra las cámaras.
`probar_objetos.py` sirve para mirar si el modelo detecta; ESTE sirve para
correrlo en producción lo más rápido que la GPU permita.

Qué hace distinto al camino de probar_objetos.py
------------------------------------------------
1. Preproceso en GPU. El frame viaja a la placa como uint8 (4x menos bytes
   por PCIe que float32) y el resize + normalización se hacen con kernels
   CUDA en vez de numpy/cv2 en el CPU.
2. Un solo módulo fusionado. backbone.extractor -> fpn -> torres de la cabeza
   quedan en un nn.Module sin lógica Python adentro, así se puede capturar
   con CUDA Graphs y compilar con torch.compile.
3. Anchors cacheados. El predict() original los regeneraba en cada frame y
   además alocaba un tensor (B,3,H,W) de ceros solo para engañar al
   AnchorGenerator. Acá se calculan una vez y quedan en VRAM.
4. Decode + NMS vectorizados y batcheados. El original hacía un bucle Python
   por imagen y un .tolist() por detección: cada uno de esos es un sync
   GPU->CPU que frena la placa. Acá hay UN solo sync por batch.
5. Umbral en espacio logit. En vez de sigmoid sobre (B, anchors, clases) y
   después comparar, se compara contra logit(umbral) y se hace sigmoid solo
   sobre los sobrevivientes. Da exactamente el mismo resultado.
6. Batch multi-cámara. N cámaras entran en un solo forward.
7. FP16/BF16, channels_last, TF32, cuDNN benchmark, CUDA Graphs y
   (opcional) un engine TensorRT en lugar de PyTorch.

Uso como librería
-----------------
    from objects_engine import ObjectsEngine, EngineConfig

    eng = ObjectsEngine(EngineConfig(pesos="checkpoints/head_best.pt"))

    res = eng.infer(frame_bgr, camera_id="cam-0")      # 1 cámara
    for d in res.detections:
        print(d.label, d.score, d.bbox_xyxy)           # caja en coords del frame original

    lote = eng.infer_batch([f0, f1, f2, f3],           # 4 cámaras de una
                           camera_ids=["c0","c1","c2","c3"])

Uso como reemplazo de probar_objetos.py
---------------------------------------
    python objects_engine.py 0 --ver --pesos checkpoints/head_best.pt
    python objects_engine.py video.mp4 --precision fp16 --cuda-graphs

Autotest (sin cámara, valida que da lo mismo que el predict() original):
    python objects_engine.py --autotest
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import boxes as box_ops

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DIR_BACKBONE = os.path.join(_RAIZ, "03_backbone")
if _DIR_BACKBONE not in sys.path:
    sys.path.insert(0, _DIR_BACKBONE)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_backbone import (  # noqa: E402
    BackboneConfig, SharedBackbone, huella_backbone, pesos_no_imagenet,
)
from objects_head import (  # noqa: E402
    CRITICAL_CLASSES,
    DEFAULT_CLASSES,
    Detection,
    ObjectsHead,
    ObjectsHeadConfig,
)

# Mismo clip que usa torchvision.models.detection._utils.BoxCoder, para que el
# decode de acá dé bit a bit lo mismo que el de entrenamiento.
_BBOX_XFORM_CLIP = math.log(1000.0 / 16)

# Niveles/anclas por defecto: los que usa entrenar_objetos.py. Si entrenás con
# otros, pasálos por EngineConfig o el checkpoint no va a cargar.
_FPN_LEVELS_ENTRENADOS: Tuple[str, ...] = ("p3", "p4", "p5")
# 2026-09-15: alineado con el entrenador. Un checkpoint que trae
# "anchor_sizes" pisa esto (ver _aplicar_checkpoint); el default solo se usa
# con checkpoints viejos que no lo anotan, y ahi un preset equivocado da
# detecciones corridas sin tirar ninguna excepcion.
_ANCHOR_SIZES_ENTRENADOS: Tuple[Tuple[int, ...], ...] = (
    (16, 20, 25), (32, 40, 51), (64, 81, 102))


class ErrorBackbone(RuntimeError):
    """El backbone no es el que entrenó la cabeza, o no está."""


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
@dataclass
class EngineConfig:
    """Todo lo que se puede tocar del motor. Los defaults son para una GPU
    NVIDIA moderna (RTX 30/40/50, Ada) y ya vienen optimizados."""

    # --- modelo -----------------------------------------------------------
    pesos: Optional[str] = None                # state_dict de la cabeza (.pt)
    pesos_backbone: Optional[str] = None       # state_dict del backbone (.pt)
    # Escape explicito: arrancar con un backbone que NO es el que entreno la
    # cabeza. Solo para diagnostico -- las detecciones no valen nada.
    sin_backbone: bool = False
    clases: Tuple[str, ...] = DEFAULT_CLASSES
    fpn_levels: Tuple[str, ...] = _FPN_LEVELS_ENTRENADOS
    anchor_sizes: Tuple[Tuple[int, ...], ...] = _ANCHOR_SIZES_ENTRENADOS
    aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)
    backbone_pretrained: bool = False

    # --- entrada ----------------------------------------------------------
    input_size: int = 384
    max_batch: int = 4                         # cuántas cámaras juntas
    entrada_bgr: bool = True                   # cv2 entrega BGR

    # --- optimización CUDA ------------------------------------------------
    device: str = "auto"                       # "auto" | "cuda" | "cpu"
    precision: str = "fp16"                    # "fp16" | "bf16" | "fp32" | "fp16-full"
    channels_last: bool = True
    cuda_graphs: bool = True
    compile: bool = False                      # torch.compile (1er frame lento)
    compile_mode: str = "max-autotune"
    tf32: bool = True
    cudnn_benchmark: bool = True
    warmup: int = 12

    # --- TensorRT (opcional) ---------------------------------------------
    trt_engine: Optional[str] = None           # .plan de exportar_objetos.py

    # --- post-proceso -----------------------------------------------------
    score_thresh: float = 0.30
    nms_thresh: float = 0.50
    detections_per_img: int = 100
    topk_candidates: int = 1000
    vlm_gate_band: Tuple[float, float] = (0.30, 0.60)
    coords_originales: bool = True             # devolver cajas en px del frame real

    # --- señales para la capa de fusión ----------------------------------
    calcular_novedad: bool = True              # alimenta 06_fusion_decision/vlm_gate.py
    devolver_embedding: bool = False           # embedding 512-d en el resultado

    # --- varios -----------------------------------------------------------
    verboso: bool = True                       # False = no imprime el setup

    @property
    def num_classes(self) -> int:
        return len(self.clases)

    def resolver_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FrameResult:
    """Resultado de un frame. Expone `.novelty` para que VLMGate.decide() lo
    coma tal cual (usa duck typing sobre SharedFeatures)."""

    detections: List[Detection]
    camera_id: str
    frame_idx: int
    novelty: float = 0.0
    input_size: Tuple[int, int] = (0, 0)
    original_size: Tuple[int, int] = (0, 0)
    latencia_ms: float = 0.0
    embedding: Optional[Tensor] = None
    incertidumbre: float = 0.0                 # head_uncertainty para el gate

    @property
    def criticas(self) -> List[Detection]:
        return [d for d in self.detections if d.label in CRITICAL_CLASSES]


# --------------------------------------------------------------------------- #
# Módulo fusionado: la parte estática, capturable por CUDA Graphs
# --------------------------------------------------------------------------- #
class FusedObjectsNet(nn.Module):
    """backbone.extractor -> FPN -> torres de la cabeza -> (logits, deltas, emb).

    Sin diccionarios de salida, sin bucles dependientes de datos, sin
    `.item()`: todo shape estático. Eso es lo que permite capturarlo en un
    CUDA Graph y exportarlo a ONNX/TensorRT sin sorpresas.
    """

    def __init__(self, backbone: SharedBackbone, head: ObjectsHead,
                 levels: Sequence[str], con_embedding: bool = True) -> None:
        super().__init__()
        self.extractor = backbone.extractor
        self.fpn = backbone.fpn
        self.embed_head = backbone.embed_head
        self.cls_tower = head.cls_tower
        self.box_tower = head.box_tower
        self.cls_logits = head.cls_logits
        self.bbox_pred = head.bbox_pred
        self.levels = tuple(levels)
        self.num_classes = head.cfg.num_classes
        self.con_embedding = con_embedding

    @staticmethod
    def _flatten(t: Tensor, last_dim: int) -> Tensor:
        # Mismo orden que ObjectsHead._flatten: (B, H, W, anchors, dim).
        # Tiene que coincidir con el orden en que AnchorGenerator emite anclas.
        b, _, h, w = t.shape
        t = t.view(b, -1, last_dim, h, w).permute(0, 3, 4, 1, 2)
        return t.reshape(b, -1, last_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        feats = self.extractor(x)
        pyr = self.fpn(feats)

        logits_l: List[Tensor] = []
        deltas_l: List[Tensor] = []
        for lvl in self.levels:
            f = pyr[lvl]
            logits_l.append(self._flatten(self.cls_logits(self.cls_tower(f)),
                                          self.num_classes))
            deltas_l.append(self._flatten(self.bbox_pred(self.box_tower(f)), 4))

        logits = torch.cat(logits_l, dim=1)     # (B, A, C) crudos (sin sigmoid)
        deltas = torch.cat(deltas_l, dim=1)     # (B, A, 4)

        if self.con_embedding:
            emb = self.embed_head(pyr[self.levels[-1]])
        else:
            emb = logits.new_zeros((x.shape[0], 1))
        return logits, deltas, emb


# --------------------------------------------------------------------------- #
# Preproceso en GPU
# --------------------------------------------------------------------------- #
class GpuPreproc:
    """uint8 HWC (lo que sale de cv2) -> tensor normalizado listo para la red.

    El resize y la normalización se hacen en la GPU. Por PCIe viaja el frame
    en uint8: 1920x1080x3 = 6 MB en vez de 25 MB en float32.
    """

    def __init__(self, size: int, device: torch.device, dtype: torch.dtype,
                 channels_last: bool = True, bgr: bool = True) -> None:
        self.size = (size, size)
        self.device = device
        self.dtype = dtype
        self.channels_last = channels_last
        self.bgr = bgr
        self.mean = torch.tensor([0.485, 0.456, 0.406],
                                 device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225],
                                device=device).view(1, 3, 1, 1)
        self._pin: Dict[Tuple[int, int, int], Tensor] = {}

    def _buffer(self, b: int, h: int, w: int) -> Tensor:
        """Buffer pinned (page-locked) reusado: habilita copias H2D asíncronas."""
        key = (b, h, w)
        buf = self._pin.get(key)
        if buf is None:
            buf = torch.empty((b, h, w, 3), dtype=torch.uint8,
                              pin_memory=(self.device.type == "cuda"))
            self._pin[key] = buf
        return buf

    @torch.no_grad()
    def __call__(self, frames: Sequence[np.ndarray]) -> Tuple[Tensor, Tuple[int, int]]:
        shapes = {f.shape[:2] for f in frames}
        if len(shapes) != 1:
            raise ValueError(
                "Todos los frames del batch tienen que tener el mismo alto/ancho. "
                f"Llegaron: {sorted(shapes)}. Redimensioná antes o mandá batches "
                "separados por resolución."
            )
        h, w = frames[0].shape[:2]

        buf = self._buffer(len(frames), h, w)
        for i, f in enumerate(frames):
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            buf[i].copy_(torch.from_numpy(np.ascontiguousarray(f)))

        gpu = buf.to(self.device, non_blocking=True)          # (B,H,W,3) uint8
        x = gpu.permute(0, 3, 1, 2)                           # ya es channels_last
        if self.bgr:
            x = x[:, [2, 1, 0]]                               # BGR -> RGB
        x = x.float()
        x = F.interpolate(x, size=self.size, mode="bilinear",
                          align_corners=False, antialias=False)
        x = x.div_(255.0).sub_(self.mean).div_(self.std)

        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x.to(self.dtype), (h, w)

    @torch.no_grad()
    def desde_tensor(self, x: Tensor) -> Tensor:
        """Para batches que ya vienen como tensor float 0-1 o 0-255 (B,3,H,W)."""
        x = x.to(self.device, non_blocking=True).float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[-2:] != self.size:
            x = F.interpolate(x, size=self.size, mode="bilinear",
                              align_corners=False)
        x = (x - self.mean) / self.std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x.to(self.dtype)


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #
class ObjectsEngine:
    def __init__(self, cfg: Optional[EngineConfig] = None) -> None:
        self.cfg = cfg or EngineConfig()
        c = self.cfg

        self.device = torch.device(c.resolver_device())
        self.es_cuda = self.device.type == "cuda"

        if c.precision not in ("fp16", "bf16", "fp32", "fp16-full"):
            raise ValueError(f"precision desconocida: {c.precision!r}")
        if not self.es_cuda and c.precision != "fp32":
            if c.verboso:
                print(f"[motor] sin CUDA: bajo precision {c.precision} -> fp32")
            c.precision = "fp32"
            c.cuda_graphs = False
            c.channels_last = False

        self._aplicar_flags_backend()

        # --- modelo ---------------------------------------------------------
        # El checkpoint se lee ANTES de construir el backbone: adentro dice con
        # qué backbone se entrenó la cabeza. Si el backbone de inferencia no es
        # el mismo que el de entrenamiento, la cabeza ve features que nunca vio
        # y las detecciones son ruido — el error más caro y más silencioso de
        # todo el pipeline.
        ck: Dict[str, Any] = {}
        if c.pesos:
            ck = torch.load(c.pesos, map_location="cpu")
            if not isinstance(ck, dict) or "head" not in ck:
                ck = {"head": ck}          # .pt viejo de entrenar_objetos.py

        # La geometría (anclas, niveles, clases, tamaño) sale del CHECKPOINT,
        # no de constantes de este archivo. Si se hardcodea acá, cualquier
        # cambio en el entrenamiento rompe la carga con un "size mismatch"
        # después de horas de GPU — o peor, carga algo incoherente.
        if ck:
            self._adoptar_geometria(ck)

        # head_cfg se arma DESPUÉS de adoptar: si se arma antes, se congela con
        # las anclas por defecto y el checkpoint no entra.
        self.head_cfg = ObjectsHeadConfig(
            classes=c.clases,
            fpn_levels=c.fpn_levels,
            anchor_sizes=c.anchor_sizes,
            aspect_ratios=c.aspect_ratios,
            score_thresh=c.score_thresh,
            nms_thresh=c.nms_thresh,
            detections_per_img=c.detections_per_img,
            topk_candidates=c.topk_candidates,
            vlm_gate_band=c.vlm_gate_band,
        )

        # Si los tensores vienen adentro, no hay que buscar ningún archivo al
        # lado — ni avisar de que falta.
        _parcial = bool(ck.get("backbone_parcial") and ck.get("backbone_tensores"))
        pesos_bb = c.pesos_backbone or (None if _parcial
                                        else self._resolver_backbone(ck))
        pretrained = c.backbone_pretrained
        if pesos_bb is None and ck.get("backbone_pretrained") and not pretrained:
            pretrained = True
            self._log("[motor] el checkpoint se entrenó con backbone ImageNet: "
                      "lo activo (descarga la primera vez)")

        self.backbone = SharedBackbone(
            BackboneConfig(pretrained=pretrained,
                           input_size=c.input_size, freeze_encoder=True)
        ).eval()
        # --- 1. tensores incrustados en el propio checkpoint --------------
        incrustados = ck.get("backbone_tensores") if ck.get("backbone_parcial") else None
        if incrustados:
            faltan, sobran = self.backbone.load_state_dict(incrustados, strict=False)
            if sobran:
                raise ErrorBackbone(
                    f"el checkpoint trae {len(sobran)} tensores que este "
                    f"backbone no conoce (ej: {sobran[0]}).")
            self._log(f"[motor] backbone incrustado en el checkpoint "
                      f"({len(incrustados)} tensores) — no depende de ningún "
                      f"archivo al lado")
        # --- 2. o el archivo de al lado -----------------------------------
        elif pesos_bb:
            sd_bb = torch.load(pesos_bb, map_location="cpu")
            faltan, sobran = self.backbone.load_state_dict(sd_bb, strict=False)
            self._log(f"[motor] backbone cargado de {pesos_bb}")
            if faltan:
                print(f"[motor] ⚠ al backbone le faltaron {len(faltan)} tensores "
                      f"(ej: {faltan[0]}). ¿Es el backbone correcto?")

        # --- 3. la huella: lo que convierte "detecta mal" en un error ------
        esperada = ck.get("backbone_huella")
        if c.pesos and esperada and not c.sin_backbone:
            actual = huella_backbone(self.backbone)
            if actual != esperada:
                raise ErrorBackbone(
                    f"El backbone NO es el que entrenó esta cabeza.\n"
                    f"  esperada: {esperada}\n  actual:   {actual}\n"
                    f"El FPN y el embed_head se sortean al azar al construir "
                    f"SharedBackbone y se congelan ahí: una cabeza entrenada "
                    f"contra un sorteo no funciona sobre otro, y NO falla — "
                    f"detecta mal en silencio. Traé el backbone.pt correcto, o "
                    f"pasá sin_backbone=True si estás diagnosticando.")
            self._log(f"[motor] huella del backbone verificada: {actual}")
        elif c.pesos and esperada and c.sin_backbone:
            print(f"[motor] ⚠ --sin-backbone: NO se verificó la huella "
                  f"({esperada}). Las detecciones no valen nada.")
        elif c.pesos and not esperada and not incrustados and not pesos_bb:
            # Checkpoint viejo, sin huella y sin backbone: es exactamente el
            # caso de objetos_v2.pt. No se puede verificar, así que se corta.
            if not c.sin_backbone:
                raise ErrorBackbone(
                    "Este checkpoint no trae huella y no se encontró su "
                    "backbone.pt.\n"
                    "Con ImageNet se restaura el ResNet-50, pero el FPN queda "
                    "en un sorteo NUEVO y distinto del que entrenó la cabeza: "
                    "medido sobre ruido puro, según el sorteo salen 'persona "
                    "0.41' o 20 detecciones fantasma. Eso es peor que no "
                    "arrancar.\n"
                    "Traé el backbone.pt, reentrená (el entrenador lo guarda y "
                    "le pone huella), o pasá sin_backbone=True para "
                    "diagnosticar.")
            print("[motor] ⚠ --sin-backbone con un checkpoint sin huella: "
                  "el FPN es un sorteo al azar. Solo para diagnóstico.")

        self.head = ObjectsHead(self.head_cfg).eval()
        if c.pesos:
            self.head.load_state_dict(ck["head"])
            self._log(f"[motor] cabeza cargada de {c.pesos}")
            m = ck.get("metricas") or {}
            if m.get("mAP@0.50") is not None:
                self._log(f"[motor] checkpoint: época {ck.get('epoca', '?')}, "
                          f"mAP@0.5={m['mAP@0.50']:.4f}")
            clases_ck = ck.get("clases")
            if clases_ck and tuple(clases_ck) != tuple(c.clases):
                print(f"[motor] ⚠ pediste clases {c.clases} pero el checkpoint "
                      f"se entrenó con {tuple(clases_ck)}. Las etiquetas van a "
                      "salir cambiadas.")
        else:
            self._log("[motor] SIN pesos entrenados: esperá 0 detecciones "
                      "(para ver el post-proceso andar usá --umbral 0.005)")

        self.model = FusedObjectsNet(
            self.backbone, self.head, c.fpn_levels,
            con_embedding=(c.calcular_novedad or c.devolver_embedding),
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.dtype_in = torch.float32
        if c.precision == "fp16-full":
            self.model.half()
            self.dtype_in = torch.float16
        if c.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # --- TensorRT opcional ---------------------------------------------
        self.trt = None
        if c.trt_engine:
            self.trt = self._cargar_trt(c.trt_engine)

        # --- torch.compile --------------------------------------------------
        if c.compile and self.trt is None:
            if c.cuda_graphs:
                self._log("[motor] torch.compile activo -> desactivo los CUDA "
                          "Graphs manuales (inductor ya hace cudagraphs)")
                c.cuda_graphs = False
            self.model = torch.compile(self.model, mode=c.compile_mode)

        # --- preproceso + anchors ------------------------------------------
        self.pre = GpuPreproc(c.input_size, self.device, self.dtype_in,
                              c.channels_last, c.entrada_bgr)
        self.anchors = self._anchors_cacheados()      # (A,4) fp32 en VRAM
        self._logit_thresh = self._a_logit(c.score_thresh)

        self._graphs: Dict[int, Tuple[Any, Tensor, Tuple[Tensor, ...]]] = {}
        self._contadores: Dict[str, int] = {}

        self._calentar()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _log(self, *a) -> None:
        if self.cfg.verboso:
            print(*a)


    def _adoptar_geometria(self, ck: Dict[str, Any]) -> None:
        """Toma del checkpoint cómo se entrenó la cabeza. Lo explícito en
        EngineConfig gana, para poder forzar algo a mano si hace falta."""
        c = self.cfg
        por_defecto = EngineConfig()

        if c.anchor_sizes == por_defecto.anchor_sizes and ck.get("anchor_sizes"):
            nuevas = tuple(tuple(x) for x in ck["anchor_sizes"])
            if nuevas != c.anchor_sizes:
                self._log(f"[motor] anclas del checkpoint: {nuevas}")
                c.anchor_sizes = nuevas

        if c.fpn_levels == por_defecto.fpn_levels and ck.get("fpn_levels"):
            c.fpn_levels = tuple(ck["fpn_levels"])

        if c.clases == por_defecto.clases and ck.get("clases"):
            c.clases = tuple(ck["clases"])

        if c.input_size == por_defecto.input_size and ck.get("tam"):
            if int(ck["tam"]) != c.input_size:
                self._log(f"[motor] input_size del checkpoint: {ck['tam']}px")
                c.input_size = int(ck["tam"])

    def _resolver_backbone(self, ck: Dict[str, Any]) -> Optional[str]:
        """El entrenador guarda el backbone congelado al lado del checkpoint y
        anota el nombre adentro. Acá se resuelve relativo a esa carpeta."""
        nombre = ck.get("backbone_file")
        if not nombre or not self.cfg.pesos:
            return None
        ruta = os.path.join(os.path.dirname(os.path.abspath(self.cfg.pesos)),
                            str(nombre))
        if os.path.exists(ruta):
            return ruta
        print(f"[motor] ⚠ el checkpoint apunta a '{nombre}' pero no está en "
              f"{os.path.dirname(ruta)}. Copiálo ahí o pasá --pesos-backbone: "
              "sin el backbone correcto las detecciones no valen nada.")
        return None

    def _aplicar_flags_backend(self) -> None:
        c = self.cfg
        if not self.es_cuda:
            return
        if c.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        torch.backends.cudnn.benchmark = c.cudnn_benchmark

    def _autocast(self, cache_enabled: bool = True):
        c = self.cfg
        if not self.es_cuda or c.precision in ("fp32", "fp16-full"):
            return torch.autocast("cuda", enabled=False)
        dt = torch.float16 if c.precision == "fp16" else torch.bfloat16
        return torch.autocast("cuda", dtype=dt, cache_enabled=cache_enabled)

    @staticmethod
    def _a_logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1.0 - p))

    @torch.no_grad()
    def _anchors_cacheados(self) -> Tensor:
        """Las anclas dependen solo del input_size y de los strides: se calculan
        una vez y viven en VRAM. El predict() original las regeneraba por frame."""
        s = self.cfg.input_size
        x = torch.zeros(1, 3, s, s, device=self.device)
        feats = self.backbone.extractor(x)
        pyr = self.backbone.fpn(feats)
        lista = [pyr[l] for l in self.cfg.fpn_levels]

        class _ImgList:
            def __init__(self, sizes, tensors):
                self.image_sizes = sizes
                self.tensors = tensors

        il = _ImgList([(s, s)], x)
        anchors = self.head.anchor_generator(il, lista)[0]
        return anchors.to(self.device).float().contiguous()

    def _cargar_trt(self, ruta: str):
        try:
            from exportar_objetos import TRTRunner
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "No pude importar TRTRunner de exportar_objetos.py "
                f"({e}). ¿Está el archivo al lado de este?"
            )
        self._log(f"[motor] usando engine TensorRT: {ruta}")
        return TRTRunner(ruta)

    @torch.no_grad()
    def _calentar(self) -> None:
        if self.cfg.warmup <= 0:
            return
        s = self.cfg.input_size
        for b in range(1, self.cfg.max_batch + 1):
            x = torch.zeros(b, 3, s, s, device=self.device, dtype=self.dtype_in)
            if self.cfg.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            for _ in range(self.cfg.warmup if b == 1 else 3):
                self._forward(x)
        if self.es_cuda:
            torch.cuda.synchronize()

    # ------------------------------------------------------------------ #
    # Forward (con CUDA Graphs si están activos)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.trt is not None:
            return self.trt(x)

        b = x.shape[0]
        if self.cfg.cuda_graphs and self.es_cuda:
            g = self._graphs.get(b)
            if g is None:
                g = self._capturar_graph(b)
            if g is not None:
                _, static_in, static_out = g
                static_in.copy_(x, non_blocking=True)
                g[0].replay()
                return static_out

        with self._autocast():
            return self.model(x)

    def _capturar_graph(self, b: int):
        """Captura el forward para batch `b`. Si falla, sigue sin graphs."""
        s = self.cfg.input_size
        try:
            static_in = torch.zeros(b, 3, s, s, device=self.device,
                                    dtype=self.dtype_in)
            if self.cfg.channels_last:
                static_in = static_in.contiguous(memory_format=torch.channels_last)

            # Warmup obligatorio en un stream aparte antes de capturar.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    with self._autocast(cache_enabled=False):
                        self.model(static_in)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with self._autocast(cache_enabled=False):
                    static_out = self.model(static_in)

            self._graphs[b] = (graph, static_in, static_out)
            return self._graphs[b]
        except Exception as e:  # pragma: no cover
            print(f"[motor] no pude capturar el CUDA Graph para batch {b} ({e}). "
                  "Sigo sin graphs.")
            self.cfg.cuda_graphs = False
            self._graphs.clear()
            return None

    # ------------------------------------------------------------------ #
    # Post-proceso: decode + NMS, todo en GPU, un solo sync
    # ------------------------------------------------------------------ #
    @staticmethod
    def _decode(deltas: Tensor, anchors: Tensor) -> Tensor:
        """Idéntico a torchvision BoxCoder(weights=(1,1,1,1)) pero vectorizado
        sobre (B, K, 4) en vez de un bucle por imagen."""
        widths = anchors[..., 2] - anchors[..., 0]
        heights = anchors[..., 3] - anchors[..., 1]
        ctr_x = anchors[..., 0] + 0.5 * widths
        ctr_y = anchors[..., 1] + 0.5 * heights

        dx, dy, dw, dh = deltas.unbind(-1)
        dw = dw.clamp(max=_BBOX_XFORM_CLIP)
        dh = dh.clamp(max=_BBOX_XFORM_CLIP)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        # c_to_c_* replica el 0.5*(w-1) implícito de torchvision (usa w exacto).
        return torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ], dim=-1)

    @torch.no_grad()
    def _postproceso(self, logits: Tensor, deltas: Tensor) -> Tensor:
        """(B,A,C) + (B,A,4) -> tensor (N,7): [img, clase, score, x1,y1,x2,y2].

        Sin bucles Python, sin nonzero (que sincroniza), sin .item().
        """
        c = self.cfg
        B, A, C = logits.shape
        logits = logits.float()
        deltas = deltas.float()

        flat = logits.reshape(B, A * C)
        k = min(c.topk_candidates, flat.shape[1])
        top_logits, top_idx = flat.topk(k, dim=1)             # (B,k)

        anchor_idx = torch.div(top_idx, C, rounding_mode="floor")
        class_idx = top_idx - anchor_idx * C

        d = torch.gather(deltas, 1,
                         anchor_idx.unsqueeze(-1).expand(-1, -1, 4))   # (B,k,4)
        a = self.anchors[anchor_idx]                                    # (B,k,4)
        boxes = self._decode(d, a)

        s = float(c.input_size)
        boxes[..., 0::2] = boxes[..., 0::2].clamp(0.0, s)
        boxes[..., 1::2] = boxes[..., 1::2].clamp(0.0, s)

        scores = top_logits.sigmoid()
        valido = top_logits > self._logit_thresh

        img_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand(-1, k)

        boxes_f = boxes.reshape(-1, 4)
        scores_f = scores.reshape(-1)
        clases_f = class_idx.reshape(-1)
        img_f = img_idx.reshape(-1)
        valido_f = valido.reshape(-1)

        # Los inválidos entran a NMS con score -1: quedan últimos y se filtran
        # después. Mantener el shape estático evita un sync acá.
        scores_nms = torch.where(valido_f, scores_f,
                                 torch.full_like(scores_f, -1.0))
        # idxs separa por (imagen, clase): NMS de todo el batch en una llamada.
        idxs = img_f * C + clases_f
        keep = box_ops.batched_nms(boxes_f, scores_nms, idxs, c.nms_thresh)

        keep = keep[valido_f[keep]]
        out = torch.cat([
            img_f[keep].float().unsqueeze(1),
            clases_f[keep].float().unsqueeze(1),
            scores_f[keep].unsqueeze(1),
            boxes_f[keep],
        ], dim=1)
        return out                                            # (N,7) en GPU

    def _a_detecciones(
        self,
        res: np.ndarray,
        camera_ids: Sequence[str],
        frame_idxs: Sequence[int],
        escala: Tuple[float, float],
        ts: float,
    ) -> List[List[Detection]]:
        c = self.cfg
        sx, sy = escala
        salida: List[List[Detection]] = [[] for _ in camera_ids]
        lo, hi = c.vlm_gate_band

        # res ya viene ordenado por score descendente dentro de cada grupo NMS;
        # reordenamos global por score para cortar detections_per_img bien.
        if res.size:
            res = res[np.argsort(-res[:, 2])]

        for fila in res:
            i = int(fila[0])
            if len(salida[i]) >= c.detections_per_img:
                continue
            cid = int(fila[1])
            score = float(fila[2])
            x1, y1, x2, y2 = fila[3:7]
            label = c.clases[cid]
            salida[i].append(Detection(
                bbox_xyxy=(float(x1 * sx), float(y1 * sy),
                           float(x2 * sx), float(y2 * sy)),
                score=score,
                class_id=cid,
                label=label,
                camera_id=camera_ids[i],
                frame_idx=frame_idxs[i],
                ts=ts,
                needs_vlm=(label in CRITICAL_CLASSES and lo <= score <= hi),
            ))
        return salida

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def infer_batch(
        self,
        frames: Sequence[np.ndarray],
        camera_ids: Optional[Sequence[str]] = None,
        frame_idxs: Optional[Sequence[int]] = None,
    ) -> List[FrameResult]:
        c = self.cfg
        if len(frames) == 0:
            return []
        if len(frames) > c.max_batch:
            raise ValueError(
                f"llegaron {len(frames)} frames pero max_batch={c.max_batch}. "
                "Subí max_batch en EngineConfig o partí el lote."
            )

        cams = list(camera_ids) if camera_ids else [f"cam-{i}" for i in range(len(frames))]
        if frame_idxs is None:
            frame_idxs = []
            for cam in cams:
                n = self._contadores.get(cam, 0)
                frame_idxs.append(n)
                self._contadores[cam] = n + 1

        t0 = time.perf_counter()
        x, (h_org, w_org) = self.pre(frames)
        logits, deltas, emb = self._forward(x)
        res_gpu = self._postproceso(logits, deltas)

        # ÚNICO sync GPU->CPU de todo el frame.
        res = res_gpu.cpu().numpy()
        if self.es_cuda:
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        s = c.input_size
        escala = (w_org / s, h_org / s) if c.coords_originales else (1.0, 1.0)
        ts = time.time()
        dets = self._a_detecciones(res, cams, frame_idxs, escala, ts)

        salida: List[FrameResult] = []
        for i, cam in enumerate(cams):
            nov = 0.0
            if c.calcular_novedad:
                nov = self._novedad(cam, emb[i:i + 1])
            scores = [d.score for d in dets[i]]
            inc = float(max(0.0, 1.0 - max(scores))) if scores else 0.0
            salida.append(FrameResult(
                detections=dets[i],
                camera_id=cam,
                frame_idx=frame_idxs[i],
                novelty=nov,
                input_size=(s, s),
                original_size=(h_org, w_org),
                latencia_ms=dt_ms / len(frames),
                # .clone() obligatorio: con CUDA Graphs `emb` es el buffer
                # estático del grafo y el próximo frame lo pisa.
                embedding=emb[i:i + 1].detach().clone() if c.devolver_embedding else None,
                incertidumbre=inc,
            ))
        return salida

    def infer(self, frame: np.ndarray, camera_id: str = "cam-0",
              frame_idx: Optional[int] = None) -> FrameResult:
        idxs = [frame_idx] if frame_idx is not None else None
        return self.infer_batch([frame], [camera_id], idxs)[0]

    def _novedad(self, camera_id: str, emb: Tensor) -> float:
        """Reusa la estadística de novedad del backbone para que el gate del
        VLM (06_fusion_decision/vlm_gate.py) siga viendo los mismos números."""
        try:
            return float(self.backbone._update_novelty(camera_id, emb.float()))
        except Exception:
            return 0.0

    def reset(self, camera_id: Optional[str] = None) -> None:
        self.backbone.reset(camera_id)
        if camera_id is None:
            self._contadores.clear()
        else:
            self._contadores.pop(camera_id, None)

    def resumen(self) -> str:
        c = self.cfg
        gpu = torch.cuda.get_device_name(0) if self.es_cuda else "CPU"
        modo = "TensorRT" if self.trt is not None else (
            "torch.compile" if c.compile else (
                "cuda-graphs" if c.cuda_graphs else "eager"))
        return (f"device={self.device} ({gpu})  precision={c.precision}  "
                f"modo={modo}  channels_last={c.channels_last}  "
                f"input={c.input_size}px  max_batch={c.max_batch}  "
                f"anclas={self.anchors.shape[0]}")

    def vram_mb(self) -> float:
        if not self.es_cuda:
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)


# --------------------------------------------------------------------------- #
# Autotest: ¿el motor da lo mismo que el predict() original?
# --------------------------------------------------------------------------- #
def _autotest() -> int:
    print("=" * 70)
    print("AUTOTEST · motor optimizado vs ObjectsHead.predict() original")
    print("=" * 70)
    torch.manual_seed(0)
    np.random.seed(0)

    umbral = 0.10
    cfg = EngineConfig(
        precision="fp32", cuda_graphs=False, channels_last=False,
        compile=False, warmup=2, score_thresh=umbral, max_batch=2,
        coords_originales=False, entrada_bgr=False,
    )
    eng = ObjectsEngine(cfg)

    # Con la init por defecto todos los scores dan ~0.01 y no habría nada que
    # comparar. Desparramamos el bias para que salgan detecciones de verdad:
    # el objetivo es validar decode + top-k + NMS, no la calidad del modelo.
    with torch.no_grad():
        nn.init.normal_(eng.head.cls_logits.bias, mean=-3.0, std=1.2)
    print(f"[autotest] {eng.resumen()}\n")

    s = cfg.input_size
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (s, s, 3), dtype=np.uint8) for _ in range(2)]

    res = eng.infer_batch(frames, camera_ids=["a", "b"])

    # Referencia: el camino viejo, backbone + head.predict()
    with torch.no_grad():
        x, _ = eng.pre(frames)
        feats = eng.backbone.extractor(x)
        pyr = eng.backbone.fpn(feats)
        ref = eng.head.predict(pyr, image_sizes=[(s, s)] * 2,
                               camera_ids=["a", "b"], frame_idxs=[0, 0])

    ok = True
    for i in range(2):
        a = sorted([(d.class_id, round(d.score, 4)) + tuple(round(v, 2) for v in d.bbox_xyxy)
                    for d in res[i].detections])
        b = sorted([(d.class_id, round(d.score, 4)) + tuple(round(v, 2) for v in d.bbox_xyxy)
                    for d in ref[i]])
        igual = a == b
        ok &= igual
        marca = "OK " if igual else "DIF"
        print(f"  [{marca}] imagen {i}: motor={len(a)} dets · original={len(b)} dets")
        if not igual:
            solo_motor = [t for t in a if t not in b][:3]
            solo_ref = [t for t in b if t not in a][:3]
            if solo_motor:
                print(f"        solo en el motor:   {solo_motor}")
            if solo_ref:
                print(f"        solo en el original:{solo_ref}")

    # Chequeo 2: escalado a coordenadas del frame original
    cfg2 = EngineConfig(precision="fp32", cuda_graphs=False, channels_last=False,
                        warmup=1, score_thresh=umbral, coords_originales=True,
                        entrada_bgr=False)
    eng2 = ObjectsEngine(cfg2)
    with torch.no_grad():
        nn.init.normal_(eng2.head.cls_logits.bias, mean=-3.0, std=1.2)
    grande = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    r = eng2.infer(grande, camera_id="cam-x")
    dentro = all(0 <= d.bbox_xyxy[0] <= 1920 and 0 <= d.bbox_xyxy[1] <= 1080
                 for d in r.detections)
    print(f"  [{'OK ' if dentro else 'DIF'}] cajas 1080p dentro del frame "
          f"({len(r.detections)} dets, {r.latencia_ms:.0f} ms)")
    ok &= dentro

    # Chequeo 3: compatibilidad con el gate del VLM
    try:
        sys.path.insert(0, os.path.join(_RAIZ, "06_fusion_decision"))
        from vlm_gate import VLMGate
        d = VLMGate().decide(r, head_uncertainty=r.incertidumbre)
        print(f"  [OK ] VLMGate come el FrameResult -> run={d.run} motivo={d.reason!r}")
    except Exception as e:
        print(f"  [DIF] VLMGate falló: {e}")
        ok = False

    print("\n" + ("TODO OK ✅" if ok else "HAY DIFERENCIAS ⚠️"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI: reemplazo directo de probar_objetos.py
# --------------------------------------------------------------------------- #
_COLORES = {
    "humo": (160, 160, 160), "llama": (0, 120, 255), "persona": (0, 200, 0),
    "pistola": (0, 0, 255), "cuchillo": (60, 20, 220), "celular": (255, 200, 0),
    "paquete": (30, 105, 160),
}



def _modo_imagenes(args) -> int:
    """Corre el modelo sobre una carpeta de imágenes y guarda las cajas
    dibujadas. Si al lado hay labels/ en formato YOLO, dibuja también la
    verdad de referencia en blanco fino: así se ven de un vistazo los falsos
    negativos (caja blanca sin caja de color) y los falsos positivos (caja de
    color sin blanca). Un número de mAP no muestra eso."""
    import cv2

    carpeta = Path(args.fuente)
    if not carpeta.is_dir():
        print(f"No es una carpeta: {carpeta}")
        return 2
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imgs = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in exts)
    if not imgs:
        print(f"No hay imágenes en {carpeta}")
        return 2

    # labels/<split>/ paralelo a images/<split>/
    dir_lbl = None
    partes = list(carpeta.parts)
    if "images" in partes:
        i = len(partes) - 1 - partes[::-1].index("images")
        partes[i] = "labels"
        cand = Path(*partes)
        if cand.is_dir():
            dir_lbl = cand

    paso = max(1, len(imgs) // args.n) if args.n else 1
    muestra = imgs[::paso][:args.n] if args.n else imgs

    cfg = EngineConfig(pesos=args.pesos, precision=args.precision,
                       cuda_graphs=False, score_thresh=args.umbral,
                       max_batch=1, entrada_bgr=True)
    eng = ObjectsEngine(cfg)
    print(f"[motor] {eng.resumen()}\n")

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    n_det = Counter()
    n_gt = Counter()
    sin_nada = 0
    for ruta in muestra:
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        h, w = img.shape[:2]
        r = eng.infer(img, camera_id="lote")

        # verdad de referencia primero, para que quede debajo
        if dir_lbl:
            txt = dir_lbl / (ruta.stem + ".txt")
            if txt.exists():
                for linea in txt.read_text(encoding="utf-8",
                                           errors="ignore").splitlines():
                    pr = linea.split()
                    if len(pr) != 5:
                        continue
                    c = int(pr[0]); cx, cy, bw, bh = (float(v) for v in pr[1:])
                    n_gt[c] += 1
                    x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        for d in r.detections:
            n_det[d.class_id] += 1
            x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
            col = _COLORES.get(d.label, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"{d.label} {d.score:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if not r.detections:
            sin_nada += 1
        cv2.imwrite(str(salida / ruta.name), img)

    print(f"  {len(muestra)} imágenes · {sin_nada} sin ninguna detección")
    print(f"\n  {'clase':<10} {'detectadas':>11} {'en la verdad':>13}")
    print("  " + "-" * 36)
    for i, nom in enumerate(DEFAULT_CLASSES):
        marca = ""
        if n_gt[i] and not n_det[i]:
            marca = "  <- no detecta NADA de esta clase"
        elif n_det[i] and not n_gt[i]:
            marca = "  <- inventa (no hay ninguna acá)"
        print(f"  {i} {nom:<8} {n_det[i]:>11} {n_gt[i]:>13}{marca}")
    print(f"\n  Imágenes anotadas en: {salida}")
    print("  Blanco fino = verdad de referencia · color = lo que detectó el modelo.")
    print("  Blanca sin color = se le escapó. Color sin blanca = falso positivo.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Motor CUDA de la cabeza de objetos (producción)")
    ap.add_argument("fuente", nargs="?", default="0",
                    help="índice de webcam (0, 1, ...), ruta a video o RTSP")
    ap.add_argument("--autotest", action="store_true",
                    help="valida el motor contra el predict() original y sale")
    ap.add_argument("--ver", action="store_true", help="ventana con las cajas")
    ap.add_argument("--pesos", default=None, help="state_dict de la cabeza (.pt)")
    ap.add_argument("--umbral", type=float, default=0.30)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--precision", default="fp16",
                    choices=["fp16", "bf16", "fp32", "fp16-full"])
    ap.add_argument("--cuda-graphs", dest="graphs", action="store_true", default=True)
    ap.add_argument("--sin-cuda-graphs", dest="graphs", action="store_false")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile (primer frame tarda ~1 min, después vuela)")
    ap.add_argument("--trt", default=None, help="engine .plan de TensorRT")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = sin límite")
    ap.add_argument("--imagenes", action="store_true",
                    help="tratar `fuente` como carpeta de imágenes y guardar "
                         "las cajas dibujadas (compara contra labels/ si están)")
    ap.add_argument("--n", type=int, default=40,
                    help="cuántas imágenes muestrear (0 = todas)")
    ap.add_argument("--salida", default="revision",
                    help="carpeta donde guardar las imágenes anotadas")
    args = ap.parse_args()

    if args.autotest:
        return _autotest()

    if args.imagenes:
        return _modo_imagenes(args)

    import cv2

    cfg = EngineConfig(
        pesos=args.pesos, input_size=args.input_size, precision=args.precision,
        cuda_graphs=args.graphs, compile=args.compile, trt_engine=args.trt,
        score_thresh=args.umbral, max_batch=1,
    )
    eng = ObjectsEngine(cfg)
    print(f"[motor] {eng.resumen()}")

    fuente: Union[int, str] = int(args.fuente) if args.fuente.isdigit() else args.fuente
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"No pude abrir la fuente: {args.fuente!r}")
        return 2

    n = 0
    lat: List[float] = []
    t_ini = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = eng.infer(frame, camera_id="cam-0")
        lat.append(r.latencia_ms)
        n += 1

        for d in r.detections:
            x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
            gate = "  [needs_vlm]" if d.needs_vlm else ""
            print(f"  frame {r.frame_idx:5d}  {d.label:8s} score={d.score:.3f}  "
                  f"caja=({x1},{y1},{x2},{y2}){gate}")
            if args.ver:
                col = _COLORES.get(d.label, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{d.label} {d.score:.2f}",
                            (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col, 2)

        if n % 30 == 0:
            fps = n / (time.perf_counter() - t_ini)
            med = sum(lat[-30:]) / min(30, len(lat))
            print(f"[frame {r.frame_idx}] fps={fps:.1f}  latencia={med:.1f} ms  "
                  f"novedad={r.novelty:.2f}σ  dets={len(r.detections)}  "
                  f"vram={eng.vram_mb():.0f} MB")

        if args.ver:
            try:
                cv2.imshow("HORUS · objetos (CUDA)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                print("[motor] este OpenCV no tiene GUI (servidor headless): "
                      "sigo sin ventana. Instalá opencv-python en un equipo con "
                      "escritorio si querés ver las cajas.")
                args.ver = False
        if args.max_frames and n >= args.max_frames:
            break

    cap.release()
    if args.ver:
        # En un servidor headless cv2 no tiene GUI y esto revienta.
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    total = time.perf_counter() - t_ini
    if n:
        lat_s = sorted(lat)
        p95 = lat_s[min(len(lat_s) - 1, int(0.95 * len(lat_s)))]
        print(f"\n[fin] {n} frames en {total:.1f}s -> {n / total:.1f} fps  "
              f"(latencia media {sum(lat)/n:.1f} ms · p95 {p95:.1f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
