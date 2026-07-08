# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_RAIZ, "03_backbone"))

from shared_backbone import SharedBackbone
from objects_head import ObjectsHead, ObjectsHeadConfig

TAM = 384

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DatasetYolo(Dataset):
    EXT_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, raiz: Path, split: str, aumentar: bool = False):
        self.dir_img = raiz / "images" / split
        self.dir_lbl = raiz / "labels" / split
        if not self.dir_img.is_dir():
            sys.exit(f"No existe {self.dir_img} — revisá datasets/LEEME.txt")
        self.archivos = sorted(
            p for p in self.dir_img.iterdir() if p.suffix.lower() in self.EXT_IMG
        )
        if not self.archivos:
            sys.exit(f"No hay imágenes en {self.dir_img}")
        self.aumentar = aumentar

    def __len__(self) -> int:
        return len(self.archivos)

    def _leer_labels(self, nombre: str):
        txt = self.dir_lbl / (nombre + ".txt")
        cajas, clases = [], []
        if txt.exists():
            for linea in txt.read_text().splitlines():
                partes = linea.split()
                if len(partes) != 5:
                    continue
                clases.append(int(partes[0]))
                cajas.append([float(v) for v in partes[1:]])
        return np.array(cajas, dtype=np.float32).reshape(-1, 4), \
            np.array(clases, dtype=np.int64)

    def __getitem__(self, i: int):
        ruta = self.archivos[i]
        img = cv2.imread(str(ruta))
        if img is None:
            raise RuntimeError(f"No pude leer {ruta}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (TAM, TAM), interpolation=cv2.INTER_LINEAR)

        cajas_rel, clases = self._leer_labels(ruta.stem)

        if self.aumentar and random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            if len(cajas_rel):
                cajas_rel[:, 0] = 1.0 - cajas_rel[:, 0]

        x = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(x).permute(2, 0, 1)

        if len(cajas_rel):
            cx, cy = cajas_rel[:, 0] * TAM, cajas_rel[:, 1] * TAM
            w, h = cajas_rel[:, 2] * TAM, cajas_rel[:, 3] * TAM
            xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            xyxy = xyxy.clip(0, TAM)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)

        target = {
            "boxes": torch.from_numpy(xyxy.astype(np.float32)),
            "labels": torch.from_numpy(clases),
        }
        return x, target


def agrupar(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets


@torch.no_grad()
def extraer_piramide(backbone: SharedBackbone, x: torch.Tensor):
    feats = backbone.extractor(x)
    return backbone.fpn(feats)


def perdida_batch(head, backbone, imgs, targets, device):
    imgs = imgs.to(device, non_blocking=True)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    piramide = extraer_piramide(backbone, imgs)
    sizes = [(TAM, TAM)] * imgs.shape[0]
    return head.compute_loss(piramide, sizes, targets)


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrenar la cabeza de objetos")
    ap.add_argument("--dataset", default="datasets/mezcla_v1",
                    help="carpeta del dataset en formato YOLO")
    ap.add_argument("--epocas", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8,
                    help="bajalo a 4 si te quedás sin memoria de GPU")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2,
                    help="procesos que cargan fotos en paralelo")
    ap.add_argument("--reanudar", default=None,
                    help="checkpoint .pt para continuar un entrenamiento")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device = {device}"
          + ("" if device == "cuda" else "  (OJO: en CPU va a ser LENTO)"))
    if device == "cuda":
        print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    raiz = Path(__file__).parent / args.dataset if not Path(args.dataset).is_absolute() \
        else Path(args.dataset)

    ds_train = DatasetYolo(raiz, "train", aumentar=True)
    ds_val = DatasetYolo(raiz, "val", aumentar=False)
    print(f"[setup] train: {len(ds_train)} fotos · val: {len(ds_val)} fotos")
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=agrupar,
                          pin_memory=(device == "cuda"))
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=agrupar,
                        pin_memory=(device == "cuda"))

    backbone = SharedBackbone().to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    cfg = ObjectsHeadConfig(
        fpn_levels=("p3", "p4", "p5"),
        anchor_sizes=((32,), (64,), (128,)),
    )
    head = ObjectsHead(cfg).to(device)
    if args.reanudar:
        head.load_state_dict(torch.load(args.reanudar, map_location=device))
        print(f"[setup] reanudando desde {args.reanudar}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epocas)

    usar_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)

    dir_ckpt = Path(__file__).parent / "checkpoints"
    dir_ckpt.mkdir(exist_ok=True)
    mejor_val = float("inf")

    for epoca in range(1, args.epocas + 1):
        head.train()
        t0, acum = time.perf_counter(), {"loss_cls": 0.0, "loss_box": 0.0}
        for imgs, targets in dl_train:
            with torch.amp.autocast("cuda", enabled=usar_amp):
                losses = perdida_batch(head, backbone, imgs, targets, device)
                loss = losses["loss_cls"] + losses["loss_box"]

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=10.0)
            scaler.step(opt)
            scaler.update()

            for k in acum:
                acum[k] += float(losses[k].detach())
        sched.step()
        n = len(dl_train)

        head.eval()
        val_total = 0.0
        with torch.no_grad():
            for imgs, targets in dl_val:
                with torch.amp.autocast("cuda", enabled=usar_amp):
                    losses = perdida_batch(head, backbone, imgs, targets, device)
                val_total += float(losses["loss_cls"] + losses["loss_box"])
        val_total /= max(1, len(dl_val))

        dt = time.perf_counter() - t0
        print(f"época {epoca:3d}/{args.epocas}  "
              f"cls={acum['loss_cls'] / n:.4f}  box={acum['loss_box'] / n:.4f}  "
              f"val={val_total:.4f}  lr={sched.get_last_lr()[0]:.2e}  {dt:.0f}s")

        torch.save(head.state_dict(), dir_ckpt / "head_last.pt")
        if val_total < mejor_val:
            mejor_val = val_total
            torch.save(head.state_dict(), dir_ckpt / "head_best.pt")
            print(f"         ★ nuevo mejor val ({val_total:.4f}) -> head_best.pt")

    print(f"\n[fin] mejor pérdida de validación: {mejor_val:.4f}")
    print("Probalo con:  python probar_objetos.py 0 --ver "
          "--pesos checkpoints/head_best.pt")


if __name__ == "__main__":
    main()
