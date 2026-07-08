from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

Item = Tuple[Path, str]


def indexar(dir_fuente: Path) -> List[Item]:
    if not dir_fuente.is_dir():
        sys.exit(f"No existe la carpeta: {dir_fuente}")

    subdirs = [d for d in dir_fuente.iterdir() if d.is_dir()]
    items: List[Item] = []

    if subdirs:
        for d in subdirs:
            for p in _imgs(d):
                items.append((p, d.name))
    else:
        for p in _imgs(dir_fuente):
            pid = p.stem.split("_")[0]
            if pid.lstrip("-").isdigit() and int(pid) <= 0:
                continue
            items.append((p, pid))

    if not items:
        sys.exit(f"No encontre imagenes en {dir_fuente}")
    return items


def _imgs(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in EXT)


def escribir(items_por_fuente: Dict[str, List[Item]], salida: Path,
             copiar: bool, val_frac: float) -> None:
    train_dir = salida / "train"
    val_dir = salida / "val"
    for d in (train_dir, val_dir):
        d.mkdir(parents=True, exist_ok=True)

    por_id: Dict[str, List[Path]] = defaultdict(list)
    for prefijo, items in items_por_fuente.items():
        for ruta, pid in items:
            por_id[f"{prefijo}_{pid}"].append(ruta)

    ids = sorted(por_id.keys())
    random.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    ids_val = set(ids[:n_val])

    tags_train = sorted(g for g in por_id if g not in ids_val)
    tags_val = sorted(g for g in por_id if g in ids_val)
    label_train = {g: i for i, g in enumerate(tags_train)}
    label_val = {g: i for i, g in enumerate(tags_val)}

    filas: List[Dict[str, object]] = []
    n_train_img = n_val_img = 0
    for gid in ids:
        es_val = gid in ids_val
        split = "val" if es_val else "train"
        label = label_val[gid] if es_val else label_train[gid]
        destino = (val_dir if es_val else train_dir) / gid
        destino.mkdir(exist_ok=True)
        for k, ruta in enumerate(por_id[gid]):
            dst = destino / f"{k:04d}{ruta.suffix.lower()}"
            _colocar(ruta, dst, copiar)
            filas.append({
                "ruta": str(dst.relative_to(salida)).replace("\\", "/"),
                "tag": gid,
                "label": label,
                "split": split,
            })
            if es_val:
                n_val_img += 1
            else:
                n_train_img += 1

    _escribir_etiquetas(salida, filas, label_train, label_val)

    n_train_id = len(tags_train)
    print(f"\n[hecho] salida: {salida}")
    print(f"  train: {n_train_id} identidades - {n_train_img} imagenes")
    if val_frac > 0:
        print(f"  val:   {len(tags_val)} identidades - {n_val_img} imagenes")
    print(f"  tags:  etiquetas.json + etiquetas.csv ({len(filas)} imagenes etiquetadas)")
    print(f"\nEntrenar con:\n  python entrenar_reid.py --dataset {salida} "
          f"--epocas 120 --P 16 --K 4 --compile")


def _escribir_etiquetas(salida: Path, filas: List[Dict[str, object]],
                        label_train: Dict[str, int],
                        label_val: Dict[str, int]) -> None:
    manifiesto = {
        "num_ids_train": len(label_train),
        "num_ids_val": len(label_val),
        "mapa_train": label_train,
        "mapa_val": label_val,
        "imagenes": filas,
    }
    (salida / "etiquetas.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")

    with (salida / "etiquetas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ruta", "tag", "label", "split"])
        w.writeheader()
        w.writerows(filas)


def _colocar(src: Path, dst: Path, copiar: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copiar:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preparar/combinar datasets ReID para reid_cuerpo",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", nargs="+", required=True, metavar="prefijo=ruta",
                    help="una o mas fuentes ya descargadas en formato  nombre=ruta")
    ap.add_argument("--salida", required=True, help="carpeta del dataset resultante")
    ap.add_argument("--copiar", action="store_true",
                    help="copiar archivos en vez de crear symlinks")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="fraccion de identidades para val (0.0 = sin val)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    salida = Path(args.salida)

    fuentes: Dict[str, Path] = {}
    for e in args.entrada:
        if "=" not in e:
            sys.exit(f"Formato invalido: {e!r} - usa  prefijo=ruta")
        pref, ruta = e.split("=", 1)
        fuentes[pref] = Path(ruta)

    items_por_fuente: Dict[str, List[Item]] = {}
    for pref, ruta in fuentes.items():
        items = indexar(ruta)
        n_ids = len({pid for _, pid in items})
        print(f"[fuente] {pref:<10} {ruta}  ->  {n_ids} identidades - {len(items)} imgs")
        items_por_fuente[pref] = items

    escribir(items_por_fuente, salida, args.copiar, args.val_frac)


if __name__ == "__main__":
    main()
