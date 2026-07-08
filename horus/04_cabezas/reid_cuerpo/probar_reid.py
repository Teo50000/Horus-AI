from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_RAIZ, "03_backbone"))

from shared_backbone import SharedBackbone
from reid_head import ReIDBodyHead, ReIDHeadConfig

ALTO, ANCHO = 256, 128
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def cargar_tensor(ruta: Path) -> torch.Tensor:
    img = cv2.imread(str(ruta))
    if img is None:
        sys.exit(f"No pude leer {ruta}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (ANCHO, ALTO), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(x).permute(2, 0, 1)

def reunir_rutas(fuentes: List[str]) -> List[Path]:
    rutas: List[Path] = []
    for f in fuentes:
        p = Path(f)
        if p.is_dir():
            rutas += sorted(q for q in p.iterdir() if q.suffix.lower() in EXT)
        elif p.suffix.lower() in EXT:
            rutas.append(p)
    if not rutas:
        sys.exit("No encontré imágenes válidas en la fuente indicada.")
    return rutas

def main() -> None:
    ap = argparse.ArgumentParser(description="Probar la cabeza ReID de cuerpo")
    ap.add_argument("fuentes", nargs="+", help="imágenes o carpeta de recortes")
    ap.add_argument("--pesos", default=None, help="state_dict entrenado (.pt)")
    ap.add_argument("--meta", default=None,
                    help="reid_meta.json (default: junto a los pesos)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device = {device}")

    cfg = ReIDHeadConfig()
    if args.pesos:
        meta = Path(args.meta) if args.meta \
            else Path(args.pesos).parent / "reid_meta.json"
        if meta.exists():
            d = json.loads(meta.read_text())["cfg"]
            cfg = ReIDHeadConfig(**{k: v for k, v in d.items()
                                    if k in ReIDHeadConfig.__dataclass_fields__})

    backbone = SharedBackbone().to(device).eval()
    head = ReIDBodyHead(cfg).to(device).eval()
    if args.pesos:
        head.load_state_dict(torch.load(args.pesos, map_location=device))
        print(f"[setup] pesos cargados de {args.pesos}")
    else:
        print("[setup] SIN pesos entrenados (embeddings sin entrenar, solo demo)")

    rutas = reunir_rutas(args.fuentes)
    batch = torch.stack([cargar_tensor(r) for r in rutas]).to(device)

    with torch.no_grad():
        piramide = backbone.fpn(backbone.extractor(batch))
        emb = head.embed(piramide, boxes=None)

    sim = (emb @ emb.t()).cpu().numpy()

    nombres = [r.name for r in rutas]
    ancho = max(len(n) for n in nombres)
    print("\nSimilitud coseno (1.0 = idéntico):")
    print(" " * (ancho + 2) + "  ".join(f"{i:>5d}" for i in range(len(rutas))))
    for i, n in enumerate(nombres):
        fila = "  ".join(f"{sim[i, j]:5.2f}" for j in range(len(rutas)))
        print(f"{n:<{ancho}}  {fila}")

    if len(rutas) == 2:
        s = float(sim[0, 1])
        veredicto = "MISMA persona" if s > 0.5 else "personas DISTINTAS"
        print(f"\n-> similitud = {s:.3f}  ({veredicto}, umbral típico 0.5)")

if __name__ == "__main__":
    main()
