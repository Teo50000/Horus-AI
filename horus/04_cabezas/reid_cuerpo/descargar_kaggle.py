from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from preparar_reid_dataset import EXT, Item, escribir, indexar

DEFAULTS: Dict[str, str] = {
    "market": "pengcw1/market-1501:Market-1501-v15.09.15/bounding_box_train",
}

PREFERIDOS = ("bounding_box_train", "train", "mask_train_v2", "images")


def parse_spec(spec: str) -> Tuple[str, str, Optional[str]]:
    if "=" not in spec:
        nombre = spec.strip()
        if nombre not in DEFAULTS:
            sys.exit(f"No conozco '{nombre}'. Usa  nombre=owner/dataset[:subcarpeta]")
        spec = f"{nombre}={DEFAULTS[nombre]}"
    nombre, resto = spec.split("=", 1)
    slug, sub = (resto.split(":", 1) + [None])[:2]
    return nombre.strip(), slug.strip(), (sub.strip() if sub else None)


def descargar(slug: str) -> Path:
    try:
        import kagglehub
    except ImportError:
        sys.exit("Falta kagglehub:  pip install kagglehub\n"
                 "Y configura tu token de Kaggle (~/.kaggle/kaggle.json).")
    print(f"[kaggle] bajando {slug} ...")
    return Path(kagglehub.dataset_download(slug))


def resolver_dir(root: Path, sub: Optional[str]) -> Path:
    if sub:
        cand = root / sub
        if cand.is_dir():
            return cand
        hoja = sub.replace("\\", "/").split("/")[-1]
        for d in root.rglob(hoja):
            if d.is_dir():
                return d
        sys.exit(f"No encontre la subcarpeta '{sub}' dentro de {root}")

    for pref in PREFERIDOS:
        for d in sorted(root.rglob(pref)):
            if d.is_dir():
                return d

    mejor, mejor_n = root, -1
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in EXT)
        if n > mejor_n:
            mejor, mejor_n = d, n
    if mejor_n <= 0:
        sys.exit(f"No encontre imagenes dentro de {root}")
    return mejor


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Descargar datasets de Kaggle y prepararlos (con tags) para reid_cuerpo")
    ap.add_argument("--kaggle", nargs="+", required=True, metavar="nombre=owner/dataset[:sub]",
                    help="fuentes de Kaggle; ej: market  o  msmt=usuario/msmt17:MSMT17/train")
    ap.add_argument("--salida", required=True, help="carpeta del dataset resultante")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--symlink", action="store_true",
                    help="usar symlinks en vez de copiar (por defecto copia)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed)

    items_por_fuente: Dict[str, List[Item]] = {}
    for spec in args.kaggle:
        nombre, slug, sub = parse_spec(spec)
        root = descargar(slug)
        dir_img = resolver_dir(root, sub)
        items = indexar(dir_img)
        n_ids = len({pid for _, pid in items})
        print(f"[fuente] {nombre:<10} {dir_img}  ->  {n_ids} identidades - {len(items)} imgs")
        items_por_fuente[nombre] = items

    escribir(items_por_fuente, Path(args.salida),
             copiar=not args.symlink, val_frac=args.val_frac)


if __name__ == "__main__":
    main()
