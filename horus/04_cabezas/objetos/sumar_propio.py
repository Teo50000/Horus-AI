# -*- coding: utf-8 -*-
"""
sumar_propio.py · HORUS — mete los frames de grabar_propio.py en el dataset.

Corre DESPUÉS de `bajar_datasets.py --armar` (que borra y rehace la mezcla).

Qué hace
--------
1. Busca `horus_propio.json` debajo de ORIGEN (en Kaggle el zip a veces queda
   una o dos carpetas más abajo de lo esperado).
2. Copia cada frame a `<dataset>/images/train/propio-<escena>__<nombre>.jpg`
   con un .txt vacío al lado: no hay cajas, a propósito.
3. Escribe `<dataset>/ignorar.json` con las clases que esa escena NO garantiza.
   Un .txt vacío solo, sin esto, diría "acá no hay NADA": le enseñaría que el
   celular de tu escritorio no es un celular. Con esto dice "acá no hay
   pistola ni cuchillo; del resto no opino".

El entrenador (entrenar_objetos_cuda.py) lee ignorar.json solo.

    python sumar_propio.py /kaggle/input/horus-propio --dataset datasets/mezcla_v1

Se puede correr dos veces: primero borra lo que había sumado antes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

CLASES = ("humo", "llama", "persona", "pistola", "cuchillo", "celular", "paquete")
MANIFIESTO = "horus_propio.json"
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def encontrar_manifiesto(origen: Path) -> Path:
    if (origen / MANIFIESTO).exists():
        return origen / MANIFIESTO
    hallados = sorted(origen.rglob(MANIFIESTO))
    if not hallados:
        sys.exit(f"No hay {MANIFIESTO} debajo de {origen}. ¿Es la carpeta que "
                 "armó grabar_propio.py (datasets/propio)?")
    if len(hallados) > 1:
        print(f"[propio] ⚠ hay {len(hallados)} manifiestos; uso {hallados[0]}")
    return hallados[0]


def sumar(origen: Path, dataset: Path, repetir: int = 1,
          split: str = "train") -> Dict[str, int]:
    man = encontrar_manifiesto(origen)
    raiz = man.parent
    datos = json.loads(man.read_text(encoding="utf-8"))
    escenas = datos.get("escenas") or {}
    if not escenas:
        sys.exit(f"{man} no declara ninguna escena.")

    dir_img = dataset / "images" / split
    dir_lbl = dataset / "labels" / split
    if not dir_img.is_dir():
        sys.exit(f"No existe {dir_img}. Corré primero bajar_datasets.py --armar.")
    dir_lbl.mkdir(parents=True, exist_ok=True)

    ruta_ign = dataset / "ignorar.json"
    ignorar: Dict[str, List[str]] = {}
    if ruta_ign.exists():
        ignorar = json.loads(ruta_ign.read_text(encoding="utf-8"))

    cuenta: Dict[str, int] = {}
    for escena, info in escenas.items():
        ausentes = list(info.get("ausentes") or [])
        malas = [c for c in ausentes if c not in CLASES]
        if malas or not ausentes:
            sys.exit(f"escena '{escena}': ausentes inválidos {ausentes}")
        pref = f"propio-{escena}__"

        # Idempotente: lo sumado antes con este prefijo se va.
        for d in (dir_img, dir_lbl):
            for p in d.glob(pref + "*"):
                p.unlink()

        fotos = sorted(p for p in (raiz / escena / "images").glob("*")
                       if p.suffix.lower() in EXT)
        if not fotos:
            print(f"[propio] ⚠ escena '{escena}' sin fotos, la salteo")
            continue
        n = 0
        for foto in fotos:
            for k in range(max(1, repetir)):
                sufijo = f"_r{k}" if k else ""
                dst = dir_img / f"{pref}{foto.stem}{sufijo}{foto.suffix.lower()}"
                shutil.copy2(foto, dst)
                (dir_lbl / (dst.stem + ".txt")).write_text("", encoding="utf-8")
                n += 1
        ignorar[pref] = [c for c in CLASES if c not in ausentes]
        cuenta[escena] = n
        print(f"[propio] {escena:<8} {len(fotos):>5} fotos ×{max(1, repetir)} "
              f"= {n:>5} en {split}   seguro NO hay: {', '.join(ausentes)}")

    ruta_ign.write_text(json.dumps(ignorar, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[propio] reglas en {ruta_ign}:")
    for pref, cls in ignorar.items():
        print(f"           {pref:<20} no anota {cls}")
    return cuenta


def main() -> int:
    ap = argparse.ArgumentParser(description="Sumar frames propios al dataset")
    ap.add_argument("origen", help="carpeta de grabar_propio.py (o una que la contenga)")
    ap.add_argument("--dataset", default="datasets/mezcla_v1")
    ap.add_argument("--repetir", type=int, default=2,
                    help="copias de cada frame: ~600 frames contra ~13.000 "
                         "fotos pesan poco; el flip del entrenamiento las varía")
    a = ap.parse_args()
    cuenta = sumar(Path(a.origen), Path(a.dataset), a.repetir)
    if not cuenta:
        sys.exit("No se sumó nada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
