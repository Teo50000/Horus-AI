# -*- coding: utf-8 -*-
"""
entrenar_objetos_cuda.py · HORUS — entrenamiento optimizado de la cabeza de objetos.

Reemplaza a entrenar_objetos.py. Mismo dataset, mismo formato YOLO, mismos
checkpoints; lo que cambia es cuánto tarda y cuánto te dice.

Qué se optimizó
---------------
1. Normalización en GPU. El DataLoader ahora devuelve uint8 (442 KB por foto
   de 384px) en vez de float32 (1.7 MB): 4x menos tráfico por PCIe y mucho
   menos trabajo del CPU, que es lo que suele frenar el entrenamiento.
2. BF16 en vez de FP16. En Ada/Ampere bf16 tiene el mismo throughput y no
   necesita GradScaler ni se va a NaN. Con --precision fp16 volvés al scaler.
3. channels_last + TF32 + cuDNN benchmark: convoluciones por tensor cores.
4. AdamW fusionado (un solo kernel para todo el update).
5. torch.compile opcional sobre backbone y cabeza.
6. Cacheo de features del backbone. El backbone está CONGELADO: sus salidas
   no cambian nunca. Con --cachear-features se calculan una vez, se guardan
   en disco en fp16 y las épocas siguientes saltean ResNet-50 entero
   (~10x más rápido por época). Ver la nota de tamaño más abajo.
7. Acumulación de gradiente (--acumular) para batch efectivo grande sin VRAM.
8. EMA de los pesos: la copia promediada suele validar mejor que la cruda.
9. Warmup + cosine, clipping, early stopping, checkpoint reanudable de verdad
   (optimizador, scheduler, scaler y época incluidos).
10. mAP@0.5 y mAP@0.5:0.95 por época, no solo la pérdida. La pérdida de
    validación de una red de detección dice poco; el mAP dice si sirve.

Uso
---
    python entrenar_objetos_cuda.py --dataset datasets/mezcla_v1
    python entrenar_objetos_cuda.py --dataset datasets/mezcla_v1 \\
        --batch 16 --precision bf16 --cachear-features --compile
    python entrenar_objetos_cuda.py --reanudar checkpoints/head_last.pt

Autotest (dataset sintético, no necesita fotos ni GPU):
    python entrenar_objetos_cuda.py --autotest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(os.path.dirname(_AQUI))
for _p in (os.path.join(_RAIZ, "03_backbone"), _AQUI):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared_backbone import BackboneConfig, SharedBackbone  # noqa: E402
from objects_head import ObjectsHead, ObjectsHeadConfig  # noqa: E402

TAM = 384
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

_FPN_LEVELS = ("p3", "p4", "p5")
_ANCHOR_SIZES = ((32,), (64,), (128,))


# --------------------------------------------------------------------------- #
# Dataset: devuelve uint8, la normalización se hace en GPU
# --------------------------------------------------------------------------- #
class DatasetYoloRapido(Dataset):
    EXT_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, raiz: Path, split: str, aumentar: bool = False,
                 tam: int = TAM) -> None:
        self.dir_img = raiz / "images" / split
        self.dir_lbl = raiz / "labels" / split
        if not self.dir_img.is_dir():
            sys.exit(f"No existe {self.dir_img} — revisá datasets/LEEME.txt")
        self.archivos = sorted(p for p in self.dir_img.iterdir()
                               if p.suffix.lower() in self.EXT_IMG)
        if not self.archivos:
            sys.exit(f"No hay imágenes en {self.dir_img}")
        self.aumentar = aumentar
        self.tam = tam

    def __len__(self) -> int:
        return len(self.archivos)

    def _labels(self, nombre: str) -> Tuple[np.ndarray, np.ndarray]:
        txt = self.dir_lbl / (nombre + ".txt")
        cajas, clases = [], []
        if txt.exists():
            for linea in txt.read_text(encoding="utf-8",
                                       errors="ignore").splitlines():
                partes = linea.split()
                if len(partes) != 5:
                    continue
                try:
                    clases.append(int(partes[0]))
                    cajas.append([float(v) for v in partes[1:]])
                except ValueError:
                    continue
        return (np.array(cajas, dtype=np.float32).reshape(-1, 4),
                np.array(clases, dtype=np.int64))

    @staticmethod
    def rel_a_xyxy(rel: np.ndarray, tam: int) -> np.ndarray:
        if not len(rel):
            return np.zeros((0, 4), dtype=np.float32)
        cx, cy = rel[:, 0] * tam, rel[:, 1] * tam
        w, h = rel[:, 2] * tam, rel[:, 3] * tam
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        return xyxy.clip(0, tam).astype(np.float32)

    def __getitem__(self, i: int):
        import cv2
        ruta = self.archivos[i]
        img = cv2.imread(str(ruta))
        if img is None:
            raise RuntimeError(f"No pude leer {ruta}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.tam, self.tam):
            img = cv2.resize(img, (self.tam, self.tam),
                             interpolation=cv2.INTER_LINEAR)

        rel, clases = self._labels(ruta.stem)

        if self.aumentar and random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            if len(rel):
                rel = rel.copy()
                rel[:, 0] = 1.0 - rel[:, 0]

        # uint8 CHW: 4x menos bytes al GPU que float32
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return x, {"boxes": torch.from_numpy(self.rel_a_xyxy(rel, self.tam)),
                   "labels": torch.from_numpy(clases)}


class DatasetSintetico(Dataset):
    """Para el autotest: rectángulos de colores, sin necesidad de fotos."""

    def __init__(self, n: int = 24, tam: int = 128, clases: int = 7) -> None:
        self.n, self.tam, self.clases = n, tam, clases

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.default_rng(i)
        img = rng.integers(30, 90, (self.tam, self.tam, 3), dtype=np.uint8)
        k = int(rng.integers(1, 4))
        cajas, labels = [], []
        for _ in range(k):
            w = int(rng.integers(20, self.tam // 2))
            h = int(rng.integers(20, self.tam // 2))
            x1 = int(rng.integers(0, self.tam - w))
            y1 = int(rng.integers(0, self.tam - h))
            c = int(rng.integers(0, self.clases))
            img[y1:y1 + h, x1:x1 + w] = (40 * c) % 255
            cajas.append([x1, y1, x1 + w, y1 + h])
            labels.append(c)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return x, {"boxes": torch.tensor(cajas, dtype=torch.float32),
                   "labels": torch.tensor(labels, dtype=torch.int64)}


def revisar_dataset(ds: DatasetYoloRapido, split: str, clases: Sequence[str],
                    fatal: bool = True) -> Dict[str, int]:
    """Chequeo previo. Una corrida de 50 épocas con los índices de clase mal
    remapeados (ver datasets/LEEME.txt) son horas tiradas: mejor 5 segundos acá.
    """
    n_cajas = 0
    por_clase = [0] * len(clases)
    sin_label, vacias, fuera_rango, fuera_01, degeneradas = 0, 0, [], 0, 0

    for p in ds.archivos:
        txt = ds.dir_lbl / (p.stem + ".txt")
        if not txt.exists():
            sin_label += 1
            continue
        rel, cls = ds._labels(p.stem)
        if not len(cls):
            vacias += 1
            continue
        for c in cls:
            if 0 <= c < len(clases):
                por_clase[c] += 1
            else:
                fuera_rango.append(int(c))
        n_cajas += len(cls)
        if len(rel):
            if rel.min() < -1e-6 or rel.max() > 1 + 1e-6:
                fuera_01 += 1
            degeneradas += int(((rel[:, 2] <= 0) | (rel[:, 3] <= 0)).sum())

    print(f"\n[dataset:{split}] {len(ds.archivos)} fotos · {n_cajas} cajas")
    if sin_label:
        print(f"[dataset:{split}]   {sin_label} sin .txt  (cuentan como "
              f"'acá no hay nada' — está bien si es a propósito)")
    if vacias:
        print(f"[dataset:{split}]   {vacias} con .txt vacío (idem)")

    ancho = max(len(c) for c in clases)
    for i, c in enumerate(clases):
        n = por_clase[i]
        pct = 100.0 * n / max(1, n_cajas)
        barra = "#" * int(pct / 2)
        aviso = "  <- SIN EJEMPLOS" if n == 0 else ""
        print(f"[dataset:{split}]   {i} {c:<{ancho}}  {n:>7}  {pct:5.1f}% "
              f"{barra}{aviso}")

    problemas = []
    if fuera_rango:
        unicos = sorted(set(fuera_rango))
        problemas.append(
            f"hay índices de clase fuera de 0..{len(clases)-1}: {unicos[:10]}. "
            "Casi seguro te falta remapear los índices de alguno de los "
            "datasets fuente (ver datasets/LEEME.txt).")
    if fuera_01:
        problemas.append(
            f"{fuera_01} fotos tienen coordenadas fuera de 0..1. El formato "
            "YOLO es relativo al tamaño de la imagen, no en píxeles.")
    if degeneradas:
        problemas.append(f"{degeneradas} cajas con ancho o alto 0.")

    if problemas:
        print()
        for p in problemas:
            print(f"[dataset:{split}] ❌ {p}")
        if fatal and (fuera_rango or fuera_01):
            sys.exit(f"\nCorregí el dataset '{split}' antes de entrenar. "
                     "Entrenar así es tiempo de GPU tirado.")

    vacias = [f"{i} {n}" for i, n in enumerate(clases) if por_clase[i] == 0]
    if vacias and fatal and split == "train":
        print(f"\n[dataset:{split}] ❌ Estas clases NO tienen un solo ejemplo de "
              f"entrenamiento:\n     {', '.join(vacias)}")
        print("     Entrenar así produce un modelo CIEGO a esas clases: no va a")
        print("     fallar, simplemente nunca las va a detectar. Para un sistema")
        print("     de prevención, esa es la peor forma de fallar.")
        print("\n     Conseguí el dato faltante (ver datasets/DATASETS.md), o si")
        print("     es a propósito — por ejemplo para probar el pipeline —")
        print("     agregá:  --permitir-clases-vacias")
        sys.exit(1)

    vivas = [n for n in por_clase if n > 0]
    if len(vivas) > 1 and max(vivas) > 8 * min(vivas):
        print(f"[dataset:{split}] ⚠ desbalance fuerte "
              f"({max(vivas)} vs {min(vivas)} cajas). La red se va a sesgar a "
              "la clase mayoritaria; ver la nota de BALANCE en datasets/LEEME.txt.")

    return {"fotos": len(ds.archivos), "cajas": n_cajas}


def agrupar(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def agrupar_features(batch):
    feats = {k: torch.stack([b[0][k] for b in batch]) for k in batch[0][0]}
    return feats, [b[1] for b in batch]


# --------------------------------------------------------------------------- #
# Normalización en GPU
# --------------------------------------------------------------------------- #
class NormalizadorGPU:
    def __init__(self, device: torch.device, channels_last: bool = True) -> None:
        self.mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1) * 255.0
        self.std = torch.tensor(_STD, device=device).view(1, 3, 1, 1) * 255.0
        self.channels_last = channels_last

    def __call__(self, x_uint8: Tensor) -> Tensor:
        x = x_uint8.float()
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------- #
# Cache de features del backbone congelado
# --------------------------------------------------------------------------- #
class CacheFeatures(Dataset):
    """El backbone está congelado -> sus salidas son constantes por imagen.
    Se calculan una vez y se guardan en memmaps fp16. Cada muestra guarda dos
    variantes: original y espejada, para no perder el flip de aumentación.

    Ojo con el tamaño: a 384px son ~1.55 MB por variante, o sea ~3.1 MB por
    foto. 5.000 fotos = ~15 GB en disco. El script te lo dice antes de empezar
    y te deja cancelar.
    """

    def __init__(self, base: DatasetYoloRapido, backbone: SharedBackbone,
                 dir_cache: Path, device: torch.device, batch: int = 16,
                 precision: str = "bf16", rehacer: bool = False) -> None:
        self.base = base
        self.dir = dir_cache
        self.meta_path = dir_cache / "meta.json"
        self.niveles = _FPN_LEVELS

        if rehacer and dir_cache.exists():
            shutil.rmtree(dir_cache)
        dir_cache.mkdir(parents=True, exist_ok=True)

        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text())
            if self.meta.get("n") == len(base):
                print(f"[cache] reuso {dir_cache} ({len(base)} fotos)")
                self._abrir()
                return
            print("[cache] la cache no coincide con el dataset: la rehago")

        self._construir(backbone, device, batch, precision)

    # -- construcción ---------------------------------------------------- #
    @torch.no_grad()
    def _construir(self, backbone, device, batch, precision) -> None:
        base = self.base
        aumentar_orig = base.aumentar
        base.aumentar = False           # la cache guarda las dos variantes

        dl = DataLoader(base, batch_size=batch, shuffle=False, num_workers=4,
                        collate_fn=agrupar, pin_memory=(device.type == "cuda"))
        norm = NormalizadorGPU(device)

        muestra_x, _ = next(iter(dl))
        feats0 = self._forward(backbone, norm(muestra_x[:1].to(device)), precision, device)
        formas = {k: tuple(v.shape[1:]) for k, v in feats0.items()}
        por_foto = sum(int(np.prod(s)) for s in formas.values()) * 2 * 2  # fp16, x2 variantes
        total_gb = por_foto * len(base) / 1e9
        print(f"[cache] {len(base)} fotos · {por_foto/1e6:.1f} MB c/u "
              f"-> {total_gb:.1f} GB en {self.dir}")
        if total_gb > 40:
            print("[cache] ⚠ son más de 40 GB. Si no te entra, corré sin "
                  "--cachear-features (Ctrl-C ahora).")
            time.sleep(5)

        mm = {}
        for k, s in formas.items():
            mm[k] = np.lib.format.open_memmap(
                self.dir / f"{k}.npy", mode="w+", dtype=np.float16,
                shape=(len(base) * 2, *s))

        t0, i = time.perf_counter(), 0
        for x, _ in dl:
            x = x.to(device, non_blocking=True)
            for var in (0, 1):
                xv = torch.flip(x, dims=[3]) if var else x
                f = self._forward(backbone, norm(xv), precision, device)
                for k, v in f.items():
                    v = v.float().half().cpu().numpy()
                    for j in range(v.shape[0]):
                        mm[k][(i + j) * 2 + var] = v[j]
            i += x.shape[0]
            if i % (batch * 10) == 0 or i >= len(base):
                vel = i / max(1e-6, time.perf_counter() - t0)
                eta = (len(base) - i) / max(1e-6, vel)
                print(f"[cache]   {i}/{len(base)}  {vel:.0f} fotos/s  ETA {eta:.0f}s")
        for v in mm.values():
            v.flush()

        self.meta = {"n": len(base), "formas": {k: list(v) for k, v in formas.items()},
                     "tam": base.tam}
        self.meta_path.write_text(json.dumps(self.meta))
        base.aumentar = aumentar_orig
        self._abrir()
        print(f"[cache] construida en {(time.perf_counter()-t0)/60:.1f} min")

    @staticmethod
    @torch.no_grad()
    def _forward(backbone, x, precision, device) -> Dict[str, Tensor]:
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        usar = device.type == "cuda" and dt is not None
        with torch.autocast("cuda", dtype=dt or torch.float16, enabled=usar):
            pyr = backbone.fpn(backbone.extractor(x))
        return {k: pyr[k] for k in _FPN_LEVELS}

    def _abrir(self) -> None:
        self.mm = {k: np.load(self.dir / f"{k}.npy", mmap_mode="r")
                   for k in self.niveles}

    # -- acceso ----------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        var = 1 if (self.base.aumentar and random.random() < 0.5) else 0
        feats = {k: torch.from_numpy(np.array(self.mm[k][i * 2 + var]))
                 for k in self.niveles}
        rel, clases = self.base._labels(self.base.archivos[i].stem)
        if var and len(rel):
            rel = rel.copy()
            rel[:, 0] = 1.0 - rel[:, 0]
        return feats, {"boxes": torch.from_numpy(
            DatasetYoloRapido.rel_a_xyxy(rel, self.base.tam)),
            "labels": torch.from_numpy(clases)}


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
class EMA:
    def __init__(self, modelo: nn.Module, decay: float = 0.9995) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in modelo.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, modelo: nn.Module) -> None:
        for k, v in modelo.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)

    def state_dict(self) -> Dict[str, Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def aplicar_a(self, modelo: nn.Module) -> Dict[str, Tensor]:
        backup = {k: v.detach().clone() for k, v in modelo.state_dict().items()
                  if k in self.shadow}
        modelo.load_state_dict(self.shadow, strict=False)
        return backup


# --------------------------------------------------------------------------- #
# mAP
# --------------------------------------------------------------------------- #
def _iou_matriz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def calcular_map(preds: List[Dict], gts: List[Dict], num_clases: int,
                 iou_thrs: Sequence[float] = (0.5,)) -> Dict[str, float]:
    """AP estilo VOC-2010 (área bajo la curva completa) promediada por clase."""
    salida: Dict[str, float] = {}
    aps_por_thr: List[float] = []

    for thr in iou_thrs:
        aps: List[float] = []
        por_clase: Dict[int, float] = {}
        for c in range(num_clases):
            scores, tp, n_gt = [], [], 0
            for p, g in zip(preds, gts):
                mp = p["labels"] == c
                mg = g["labels"] == c
                pb, ps = p["boxes"][mp], p["scores"][mp]
                gb = g["boxes"][mg]
                n_gt += len(gb)
                if not len(pb):
                    continue
                orden = np.argsort(-ps)
                pb, ps = pb[orden], ps[orden]
                usados = np.zeros(len(gb), dtype=bool)
                ious = _iou_matriz(pb, gb)
                for i in range(len(pb)):
                    scores.append(ps[i])
                    if not len(gb):
                        tp.append(0.0)
                        continue
                    j = int(np.argmax(ious[i]))
                    if ious[i, j] >= thr and not usados[j]:
                        usados[j] = True
                        tp.append(1.0)
                    else:
                        tp.append(0.0)
            if n_gt == 0:
                continue
            if not scores:
                aps.append(0.0)
                por_clase[c] = 0.0
                continue
            orden = np.argsort(-np.array(scores))
            tp_arr = np.array(tp)[orden]
            ctp = np.cumsum(tp_arr)
            cfp = np.cumsum(1 - tp_arr)
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            # envolvente de precisión decreciente
            prec = np.maximum.accumulate(prec[::-1])[::-1]
            rec = np.concatenate([[0.0], rec])
            prec = np.concatenate([[prec[0] if len(prec) else 0.0], prec])
            ap_c = float(np.sum(np.diff(rec) * prec[1:]))
            aps.append(ap_c)
            por_clase[c] = ap_c
        ap_thr = float(np.mean(aps)) if aps else 0.0
        aps_por_thr.append(ap_thr)
        salida[f"mAP@{thr:.2f}"] = ap_thr
        if thr == iou_thrs[0]:
            salida["_por_clase"] = dict(por_clase)

    salida["mAP"] = float(np.mean(aps_por_thr)) if aps_por_thr else 0.0
    return salida


# --------------------------------------------------------------------------- #
# Entrenador
# --------------------------------------------------------------------------- #
@dataclass
class Ajustes:
    dataset: str = "datasets/mezcla_v1"
    epocas: int = 50
    batch: int = 8
    acumular: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epocas: float = 1.0
    workers: int = 4
    precision: str = "bf16"
    channels_last: bool = True
    compile: bool = False
    cachear_features: bool = False
    rehacer_cache: bool = False
    ema: float = 0.9995
    paciencia: int = 12
    seed: int = 0
    clip: float = 10.0
    tam: int = TAM
    eval_cada: int = 1
    umbral_eval: float = 0.05
    reanudar: Optional[str] = None
    revisar: bool = True
    permitir_vacias: bool = False


def _sembrar(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _flags_backend() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def entrenar(a: Ajustes, dir_base: Path, autotest: bool = False) -> float:
    _sembrar(a.seed)
    _flags_backend()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    es_cuda = device.type == "cuda"
    if not es_cuda:
        a.precision = "fp32"
        a.channels_last = False
        print("[setup] sin CUDA -> fp32 (va a ser MUY lento; esto es para probar "
              "que el script corre, no para entrenar de verdad)")
    else:
        print(f"[setup] GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
    print(f"[setup] precision={a.precision}  channels_last={a.channels_last}  "
          f"compile={a.compile}  cache={a.cachear_features}")

    # --- datos ------------------------------------------------------------
    if autotest:
        ds_tr = DatasetSintetico(24, a.tam)
        ds_va = DatasetSintetico(8, a.tam)
    else:
        raiz = Path(a.dataset)
        if not raiz.is_absolute():
            raiz = dir_base / a.dataset
        ds_tr = DatasetYoloRapido(raiz, "train", aumentar=True, tam=a.tam)
        ds_va = DatasetYoloRapido(raiz, "val", aumentar=False, tam=a.tam)
    print(f"[setup] train: {len(ds_tr)} · val: {len(ds_va)}")

    if not autotest and a.revisar:
        clases = ObjectsHeadConfig(fpn_levels=_FPN_LEVELS,
                                   anchor_sizes=_ANCHOR_SIZES).classes
        revisar_dataset(ds_tr, "train", clases, fatal=not a.permitir_vacias)
        revisar_dataset(ds_va, "val", clases, fatal=False)

    # --- modelo -----------------------------------------------------------
    backbone = SharedBackbone(
        BackboneConfig(pretrained=not autotest, input_size=a.tam,
                       freeze_encoder=True)
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    if a.channels_last:
        backbone = backbone.to(memory_format=torch.channels_last)

    cfg = ObjectsHeadConfig(fpn_levels=_FPN_LEVELS, anchor_sizes=_ANCHOR_SIZES)
    head = ObjectsHead(cfg).to(device)
    if a.channels_last:
        head = head.to(memory_format=torch.channels_last)

    # --- cache de features -------------------------------------------------
    usar_cache = a.cachear_features and not autotest
    if usar_cache:
        dir_cache = dir_base / "cache" / f"{Path(a.dataset).name}_{a.tam}"
        ds_tr_f = CacheFeatures(ds_tr, backbone, dir_cache / "train", device,
                                precision=a.precision, rehacer=a.rehacer_cache)
        ds_va_f = CacheFeatures(ds_va, backbone, dir_cache / "val", device,
                                precision=a.precision, rehacer=a.rehacer_cache)
        collate, ds_tr_use, ds_va_use = agrupar_features, ds_tr_f, ds_va_f
    else:
        collate, ds_tr_use, ds_va_use = agrupar, ds_tr, ds_va

    kw = dict(num_workers=a.workers, collate_fn=collate,
              pin_memory=es_cuda, persistent_workers=a.workers > 0)
    if a.workers > 0:
        kw["prefetch_factor"] = 4
    dl_tr = DataLoader(ds_tr_use, batch_size=a.batch, shuffle=True,
                       drop_last=len(ds_tr_use) > a.batch, **kw)
    dl_va = DataLoader(ds_va_use, batch_size=a.batch, shuffle=False, **kw)

    if a.compile:
        print("[setup] torch.compile: la primera época va a tardar más")
        # Compilamos .forward (no el módulo): así compute_loss lo usa y el
        # state_dict no queda con el prefijo _orig_mod. de OptimizedModule.
        head.forward = torch.compile(head.forward)
        if not usar_cache:
            backbone.fpn.forward = torch.compile(backbone.fpn.forward)
            backbone.extractor.forward = torch.compile(backbone.extractor.forward)

    norm = NormalizadorGPU(device, a.channels_last)

    # --- optimizador ------------------------------------------------------
    fused_ok = es_cuda
    try:
        opt = torch.optim.AdamW(head.parameters(), lr=a.lr,
                                weight_decay=a.weight_decay, fused=fused_ok)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(head.parameters(), lr=a.lr,
                                weight_decay=a.weight_decay)
        fused_ok = False
    print(f"[setup] AdamW fused={fused_ok}")

    pasos_epoca = max(1, math.ceil(len(dl_tr) / a.acumular))
    total_pasos = pasos_epoca * a.epocas
    warmup_pasos = int(pasos_epoca * a.warmup_epocas)

    def lr_lambda(paso: int) -> float:
        if paso < warmup_pasos:
            return (paso + 1) / max(1, warmup_pasos)
        p = (paso - warmup_pasos) / max(1, total_pasos - warmup_pasos)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    usar_amp = es_cuda and a.precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if a.precision == "bf16" else torch.float16
    # bf16 no necesita GradScaler: tiene el mismo rango que fp32.
    scaler = torch.amp.GradScaler("cuda", enabled=(usar_amp and a.precision == "fp16"))

    ema = EMA(head, a.ema) if a.ema > 0 else None

    dir_ckpt = dir_base / "checkpoints"
    dir_ckpt.mkdir(exist_ok=True, parents=True)

    # El backbone está congelado, así que basta guardarlo UNA vez. Sin esto, la
    # inferencia construiría un ResNet-50 distinto al que entrenó la cabeza y
    # las detecciones serían ruido. objects_engine.py lo busca solo, al lado
    # del checkpoint, por el nombre que se anota en "backbone_file".
    ruta_bb = dir_ckpt / "backbone.pt"
    torch.save(backbone.state_dict(), ruta_bb)
    print(f"[setup] backbone congelado guardado en {ruta_bb.name} "
          f"({ruta_bb.stat().st_size/1e6:.0f} MB) — va junto al checkpoint")

    # --- reanudar de verdad: pesos + optimizador + scheduler + scaler ------
    epoca_ini = 1
    if a.reanudar:
        ck = torch.load(a.reanudar, map_location=device)
        pesos = ck.get("head_raw") or ck.get("head") or ck
        head.load_state_dict(pesos)
        if isinstance(ck, dict):
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
            if "sched" in ck:
                sched.load_state_dict(ck["sched"])
            if "scaler" in ck and scaler.is_enabled():
                try:
                    scaler.load_state_dict(ck["scaler"])
                except Exception:
                    pass
            if ema and "head" in ck:
                ema.shadow = {k: v.to(device).float()
                              for k, v in ck["head"].items()
                              if k in ema.shadow}
            epoca_ini = int(ck.get("epoca", 0)) + 1
        print(f"[setup] reanudado desde {a.reanudar} -> arranco en la época "
              f"{epoca_ini}")

    def autocast():
        return torch.autocast("cuda", dtype=amp_dtype, enabled=usar_amp)

    def piramide(x_uint8) -> Dict[str, Tensor]:
        if usar_cache:
            return {k: v.to(device, non_blocking=True).float()
                    for k, v in x_uint8.items()}
        x = norm(x_uint8.to(device, non_blocking=True))
        with torch.no_grad():
            return backbone.fpn(backbone.extractor(x))

    # --- bucle -------------------------------------------------------------
    mejor, sin_mejora, paso = -1.0, 0, 0
    hist: List[Dict] = []
    print()

    for epoca in range(epoca_ini, a.epocas + 1):
        head.train()
        t0 = time.perf_counter()
        acum = {"loss_cls": 0.0, "loss_box": 0.0}
        n_batches = 0
        opt.zero_grad(set_to_none=True)

        for it, (x, targets) in enumerate(dl_tr):
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                       for t in targets]
            with autocast():
                pyr = piramide(x)
                b = next(iter(pyr.values())).shape[0]
                losses = head.compute_loss(pyr, [(a.tam, a.tam)] * b, targets)
                loss = (losses["loss_cls"] + losses["loss_box"]) / a.acumular

            scaler.scale(loss).backward()

            if (it + 1) % a.acumular == 0 or (it + 1) == len(dl_tr):
                if a.clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), a.clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                paso += 1
                if ema:
                    ema.update(head)

            for k in acum:
                acum[k] += float(losses[k].detach())
            n_batches += 1

        dt = time.perf_counter() - t0
        ips = len(ds_tr_use) / max(1e-6, dt)

        # --- validación ----------------------------------------------------
        metricas = {}
        val_loss = 0.0
        if epoca % a.eval_cada == 0 or epoca == a.epocas:
            backup = ema.aplicar_a(head) if ema else None
            head.eval()
            preds, gts = [], []
            thr_orig = head.cfg.score_thresh
            head.cfg.score_thresh = a.umbral_eval
            with torch.no_grad():
                for x, targets in dl_va:
                    tg = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    with autocast():
                        pyr = piramide(x)
                        b = next(iter(pyr.values())).shape[0]
                        losses = head.compute_loss(pyr, [(a.tam, a.tam)] * b, tg)
                    val_loss += float(losses["loss_cls"] + losses["loss_box"])

                    # predict() va en fp32 y FUERA del autocast a propósito:
                    # sus pesos son fp32, y el NMS y el decode de torchvision
                    # no toleran bien bf16/fp16 mezclado. Corre una vez por
                    # época, así que el costo extra es irrelevante.
                    pyr32 = {k: v.float() for k, v in pyr.items()}
                    dets = head.predict(pyr32, [(a.tam, a.tam)] * b)
                    for i in range(b):
                        preds.append({
                            "boxes": np.array([d.bbox_xyxy for d in dets[i]],
                                              dtype=np.float32).reshape(-1, 4),
                            "scores": np.array([d.score for d in dets[i]],
                                               dtype=np.float32),
                            "labels": np.array([d.class_id for d in dets[i]],
                                               dtype=np.int64),
                        })
                        gts.append({
                            "boxes": tg[i]["boxes"].cpu().numpy().reshape(-1, 4),
                            "labels": tg[i]["labels"].cpu().numpy(),
                        })
            head.cfg.score_thresh = thr_orig
            val_loss /= max(1, len(dl_va))
            metricas = calcular_map(preds, gts, cfg.num_classes,
                                    iou_thrs=(0.5, 0.75))
            if backup:
                head.load_state_dict(backup, strict=False)

        m50 = metricas.get("mAP@0.50", 0.0)
        vram = torch.cuda.max_memory_allocated() / 1e9 if es_cuda else 0.0
        print(f"época {epoca:3d}/{a.epocas}  "
              f"cls={acum['loss_cls']/max(1,n_batches):.4f} "
              f"box={acum['loss_box']/max(1,n_batches):.4f}  "
              f"val={val_loss:.4f}  mAP50={m50:.4f} "
              f"mAP75={metricas.get('mAP@0.75', 0.0):.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  "
              f"{dt:.0f}s ({ips:.1f} img/s)  vram={vram:.1f}GB")
        por_clase = metricas.get("_por_clase") or {}
        if por_clase:
            partes = [f"{cfg.classes[i][:4]}={por_clase.get(i, 0.0):.2f}"
                      for i in range(cfg.num_classes)]
            print("         AP50 por clase: " + "  ".join(partes))
        hist.append({"epoca": epoca, "val": val_loss,
                     **{k: v for k, v in metricas.items() if k != "_por_clase"},
                     "ap50_por_clase": {cfg.classes[i]: por_clase.get(i, 0.0)
                                        for i in range(cfg.num_classes)}})

        ck = {
            "head": (ema.state_dict() if ema else head.state_dict()),
            "head_raw": head.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "epoca": epoca,
            "clases": list(cfg.classes),
            "fpn_levels": list(cfg.fpn_levels),
            "anchor_sizes": [list(s) for s in cfg.anchor_sizes],
            "tam": a.tam,
            "metricas": metricas,
            # Con qué backbone se entrenó esta cabeza. El motor lo usa para
            # reconstruir exactamente el mismo camino en inferencia.
            "backbone_file": ruta_bb.name,
            "backbone_pretrained": not autotest,
        }
        torch.save(ck, dir_ckpt / "head_last.pt")
        # Se reescribe en CADA época, no al final: si la corrida se corta con
        # Ctrl-C o se cae, el historial de lo ya hecho tiene que sobrevivir.
        (dir_ckpt / "historial.json").write_text(json.dumps(hist, indent=2))
        if m50 > mejor:
            mejor, sin_mejora = m50, 0
            torch.save(ck, dir_ckpt / "head_best.pt")
            print(f"         ★ nuevo mejor mAP50 ({m50:.4f}) -> head_best.pt")
        else:
            sin_mejora += 1
            if a.paciencia and sin_mejora >= a.paciencia:
                print(f"\n[early stop] {a.paciencia} épocas sin mejorar el mAP50")
                break

    print(f"\n[fin] mejor mAP@0.5 = {mejor:.4f}")
    print("Probalo con:")
    print("  python objects_engine.py 0 --ver --pesos checkpoints/head_best.pt")
    print("Exportalo con:")
    print("  python exportar_objetos.py --pesos checkpoints/head_best.pt --trt --fp16")
    return mejor


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenar la cabeza de objetos (CUDA)")
    ap.add_argument("--dataset", default="datasets/mezcla_v1")
    ap.add_argument("--epocas", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--acumular", type=int, default=1,
                    help="pasos de acumulación: batch efectivo = batch*acumular")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--precision", default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--sin-channels-last", dest="cl", action="store_false", default=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--cachear-features", action="store_true",
                    help="precalcula las features del backbone congelado (mucho "
                         "más rápido por época, ocupa disco)")
    ap.add_argument("--rehacer-cache", action="store_true")
    ap.add_argument("--ema", type=float, default=0.9995, help="0 desactiva")
    ap.add_argument("--paciencia", type=int, default=12, help="0 desactiva")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tam", type=int, default=TAM)
    ap.add_argument("--reanudar", default=None)
    ap.add_argument("--sin-revisar", dest="revisar", action="store_false",
                    default=True,
                    help="saltear el chequeo previo del dataset")
    ap.add_argument("--permitir-clases-vacias", action="store_true",
                    help="entrenar aunque haya clases sin un solo ejemplo "
                         "(útil para probar el pipeline; NO para el modelo real)")
    ap.add_argument("--autotest", action="store_true",
                    help="corre 2 épocas con datos sintéticos y sale")
    args = ap.parse_args()

    a = Ajustes(dataset=args.dataset, epocas=args.epocas, batch=args.batch,
                acumular=args.acumular, lr=args.lr, workers=args.workers,
                precision=args.precision, channels_last=args.cl,
                compile=args.compile, cachear_features=args.cachear_features,
                rehacer_cache=args.rehacer_cache, ema=args.ema,
                paciencia=args.paciencia, seed=args.seed, tam=args.tam,
                reanudar=args.reanudar, revisar=args.revisar,
                permitir_vacias=args.permitir_clases_vacias)

    if args.autotest:
        a.epocas, a.batch, a.workers, a.tam = 2, 4, 0, 128
        a.paciencia, a.ema, a.compile, a.reanudar = 0, 0.99, False, None
        a.permitir_vacias = True
        print("=" * 70)
        print("AUTOTEST · 2 épocas con dataset sintético")
        print("=" * 70)
        entrenar(a, Path(_AQUI), autotest=True)
        print("\nEl script entrena de punta a punta ✅")
        return 0

    entrenar(a, Path(_AQUI))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
