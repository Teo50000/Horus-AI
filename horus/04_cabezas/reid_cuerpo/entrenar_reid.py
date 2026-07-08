from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_RAIZ, "03_backbone"))

from shared_backbone import SharedBackbone
from reid_head import ReIDBodyHead, ReIDHeadConfig

ALTO, ANCHO = 256, 128
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class DatasetReID(Dataset):
    EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, raiz: Path, split: str, aumentar: bool,
                 mapa_ids: Dict[str, int] | None = None):
        self.dir = raiz / split
        if not self.dir.is_dir():
            sys.exit(f"No existe {self.dir} — revisá datasets/LEEME.txt")
        self.aumentar = aumentar
        self.items: List[Tuple[Path, str]] = []
        self._indexar()

        if not self.items:
            sys.exit(f"No hay imágenes en {self.dir}")

        ids = sorted({pid for _, pid in self.items})
        self.mapa_ids = mapa_ids or {pid: i for i, pid in enumerate(ids)}
        self.labels = [self.mapa_ids[pid] for _, pid in self.items]

    def _indexar(self) -> None:
        subdirs = [d for d in self.dir.iterdir() if d.is_dir()]
        if subdirs:
            for d in subdirs:
                for p in d.iterdir():
                    if p.suffix.lower() in self.EXT:
                        self.items.append((p, d.name))
        else:
            for p in self.dir.iterdir():
                if p.suffix.lower() in self.EXT:
                    pid = p.stem.split("_")[0]
                    if pid.lstrip("-").isdigit() and int(pid) < 0:
                        continue
                    self.items.append((p, pid))

    @property
    def num_ids(self) -> int:
        return len(self.mapa_ids)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ruta, pid = self.items[i]
        img = cv2.imread(str(ruta))
        if img is None:
            raise RuntimeError(f"No pude leer {ruta}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (ANCHO, ALTO), interpolation=cv2.INTER_LINEAR)

        if self.aumentar:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
            img = _pad_random_crop(img)

        x = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()

        if self.aumentar:
            x = _random_erasing(x)

        return x, self.labels[i]

def _pad_random_crop(img: np.ndarray, pad: int = 10) -> np.ndarray:
    h, w = img.shape[:2]
    lienzo = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    y = random.randint(0, 2 * pad)
    x = random.randint(0, 2 * pad)
    return lienzo[y:y + h, x:x + w]

def _random_erasing(x: torch.Tensor, prob: float = 0.5,
                    sl: float = 0.02, sh: float = 0.33, r1: float = 0.3) -> torch.Tensor:
    if random.random() > prob:
        return x
    c, h, w = x.shape
    area = h * w
    for _ in range(100):
        te = random.uniform(sl, sh) * area
        ar = random.uniform(r1, 1 / r1)
        eh, ew = int(round((te * ar) ** 0.5)), int(round((te / ar) ** 0.5))
        if ew < w and eh < h:
            y0, x0 = random.randint(0, h - eh), random.randint(0, w - ew)
            x[:, y0:y0 + eh, x0:x0 + ew] = torch.empty((c, eh, ew)).normal_()
            return x
    return x

class SamplerPK(Sampler):
    def __init__(self, labels: List[int], P: int, K: int):
        self.P, self.K = P, K
        self.batch = P * K
        self.por_id: Dict[int, List[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            self.por_id[lab].append(idx)
        self.ids = list(self.por_id.keys())
        if len(self.ids) < P:
            sys.exit(f"El dataset tiene {len(self.ids)} identidades pero --P={P}. "
                     f"Bajá --P.")
        self.len = (len(labels) // self.batch) * self.batch

    def __len__(self) -> int:
        return self.len

    def __iter__(self):
        pool = {i: random.sample(v, len(v)) for i, v in self.por_id.items()}
        disponibles = list(self.ids)
        salida: List[int] = []
        while len(disponibles) >= self.P and len(salida) < self.len:
            elegidos = random.sample(disponibles, self.P)
            for pid in elegidos:
                muestras = pool[pid]
                if len(muestras) < self.K:
                    muestras = random.choices(self.por_id[pid], k=self.K)
                else:
                    muestras, pool[pid] = muestras[:self.K], muestras[self.K:]
                salida.extend(muestras)
                if len(pool[pid]) < self.K:
                    disponibles.remove(pid)
        return iter(salida[:self.len])

def factor_lr(epoca: int, warmup: int, total: int) -> float:
    if epoca <= warmup:
        return epoca / max(1, warmup)
    prog = (epoca - warmup) / max(1, total - warmup)
    return 0.5 * (1 + np.cos(np.pi * prog))

@torch.no_grad()
def extraer_piramide(backbone: SharedBackbone, x: torch.Tensor):
    return backbone.fpn(backbone.extractor(x))

def main() -> None:
    ap = argparse.ArgumentParser(description="Entrenar la cabeza ReID de cuerpo (CUDA)")
    ap.add_argument("--dataset", default="datasets/personas_v1")
    ap.add_argument("--epocas", type=int, default=120)
    ap.add_argument("--P", type=int, default=16, help="identidades por batch")
    ap.add_argument("--K", type=int, default=4, help="instancias por identidad")
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.3,
                    help="margen del triplet; <=0 usa soft-margin")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile (PyTorch 2.x, primer batch más lento)")
    ap.add_argument("--bf16", action="store_true",
                    help="usar bfloat16 en vez de fp16 (Ampere+, más estable)")
    ap.add_argument("--reanudar", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    en_cuda = device == "cuda"
    print(f"[setup] device = {device}"
          + ("" if en_cuda else "  (OJO: en CPU esto va a ser MUY lento)"))
    if en_cuda:
        print(f"[setup] GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    fmt = torch.channels_last if en_cuda else torch.contiguous_format
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    raiz = Path(args.dataset) if Path(args.dataset).is_absolute() \
        else Path(__file__).parent / args.dataset

    ds_train = DatasetReID(raiz, "train", aumentar=True)
    print(f"[setup] train: {len(ds_train)} imágenes · {ds_train.num_ids} identidades")
    sampler = SamplerPK(ds_train.labels, args.P, args.K)
    dl_train = DataLoader(
        ds_train, batch_sampler=_como_batches(sampler, args.P * args.K),
        num_workers=args.workers, pin_memory=en_cuda,
        persistent_workers=en_cuda and args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )

    backbone = SharedBackbone().to(device).eval().to(memory_format=fmt)
    for p in backbone.parameters():
        p.requires_grad = False

    cfg = ReIDHeadConfig(
        in_channels=backbone.cfg.fpn_dim,
        roi_level="p4", roi_stride=backbone.strides["p4"],
        embed_dim=backbone.cfg.embed_dim,
        num_ids=ds_train.num_ids,
        triplet_margin=args.margin,
    )
    head = ReIDBodyHead(cfg).to(device).to(memory_format=fmt)
    if args.reanudar:
        head.load_state_dict(torch.load(args.reanudar, map_location=device))
        print(f"[setup] reanudando desde {args.reanudar}")

    forward_head = head
    if args.compile and hasattr(torch, "compile"):
        forward_head = torch.compile(head)
        print("[setup] torch.compile ACTIVO (el primer batch tarda en compilar)")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=5e-4,
                            fused=en_cuda)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: factor_lr(e + 1, args.warmup, args.epocas))
    scaler = torch.amp.GradScaler("cuda", enabled=en_cuda and not args.bf16)

    dir_ckpt = Path(__file__).parent / "checkpoints"
    dir_ckpt.mkdir(exist_ok=True)
    _guardar_meta(dir_ckpt, cfg, ds_train.mapa_ids)
    mejor = float("inf")

    for epoca in range(1, args.epocas + 1):
        head.train()
        t0 = time.perf_counter()
        acum = {"loss": 0.0, "loss_id": 0.0, "loss_triplet": 0.0, "acc": 0.0}
        nb = 0

        for imgs, labels in dl_train:
            imgs = imgs.to(device, non_blocking=True).to(memory_format=fmt)
            labels = labels.to(device, non_blocking=True)

            piramide = extraer_piramide(backbone, imgs)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=en_cuda):
                out = forward_head(piramide, None)
                loss_id = torch.nn.functional.cross_entropy(
                    out["logits"], labels, label_smoothing=cfg.label_smoothing)
                loss_tri = _triplet(out["feat"], labels, cfg.triplet_margin)
                loss = cfg.id_weight * loss_id + cfg.triplet_weight * loss_tri

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
                opt.step()

            with torch.no_grad():
                acc = (out["logits"].argmax(1) == labels).float().mean()
            acum["loss"] += float(loss)
            acum["loss_id"] += float(loss_id)
            acum["loss_triplet"] += float(loss_tri)
            acum["acc"] += float(acc)
            nb += 1

        sched.step()
        dt = time.perf_counter() - t0
        prom = {k: v / max(1, nb) for k, v in acum.items()}
        print(f"época {epoca:3d}/{args.epocas}  loss={prom['loss']:.4f}  "
              f"id={prom['loss_id']:.4f}  tri={prom['loss_triplet']:.4f}  "
              f"acc={prom['acc']:.3f}  lr={sched.get_last_lr()[0]:.2e}  {dt:.0f}s")

        torch.save(head.state_dict(), dir_ckpt / "reid_last.pt")
        if prom["loss"] < mejor:
            mejor = prom["loss"]
            torch.save(head.state_dict(), dir_ckpt / "reid_best.pt")
            print(f"         ★ nuevo mejor ({mejor:.4f}) -> reid_best.pt")

    print(f"\n[fin] mejor loss de entrenamiento: {mejor:.4f}")
    print("Probalo con:  python probar_reid.py --pesos checkpoints/reid_best.pt")

def _triplet(feat, labels, margin):
    from reid_head import batch_hard_triplet
    return batch_hard_triplet(feat, labels, margin)

def _como_batches(sampler: SamplerPK, batch: int):
    class _BS:
        def __iter__(self_inner):
            buf = []
            for idx in sampler:
                buf.append(idx)
                if len(buf) == batch:
                    yield buf
                    buf = []
        def __len__(self_inner):
            return len(sampler) // batch
    return _BS()

def _guardar_meta(dir_ckpt: Path, cfg: ReIDHeadConfig, mapa_ids: Dict[str, int]):
    import json
    from dataclasses import asdict
    meta = {"cfg": asdict(cfg),
            "mapa_ids": mapa_ids,
            "input_hw": [ALTO, ANCHO]}
    (dir_ckpt / "reid_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
