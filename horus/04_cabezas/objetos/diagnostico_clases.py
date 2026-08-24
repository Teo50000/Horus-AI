# -*- coding: utf-8 -*-
"""
diagnostico_clases.py · HORUS — por qué una clase rinde mal.

No mira el modelo: mira el DATASET. La causa más común de que una clase
concreta no levante no es el entrenamiento, es que esa clase aparece sin
etiquetar en imágenes que vienen de otras fuentes. Cada vez que el modelo la
detecta ahí, se lo penaliza como falso positivo — se le está enseñando
activamente a NO detectarla.

Uso:
    python diagnostico_clases.py                 # todas las clases
    python diagnostico_clases.py --clase 2       # solo persona
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
CLASES = ("humo", "llama", "persona", "pistola", "cuchillo", "celular", "paquete")
EXT_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def leer(txt: Path):
    filas = []
    if txt.exists():
        for l in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = l.split()
            if len(p) == 5:
                try:
                    filas.append(int(p[0]))
                except ValueError:
                    pass
    return filas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mezcla_v1")
    ap.add_argument("--clase", type=int, default=None)
    args = ap.parse_args()

    raiz = _AQUI / args.dataset
    d = raiz / "labels" / "train"
    if not d.is_dir():
        sys.exit(f"No existe {d}")

    # El nombre de archivo conserva la fuente: "<fuente>__000123.txt"
    por_fuente = defaultdict(lambda: {"imgs": 0, "cajas": Counter()})
    for txt in d.glob("*.txt"):
        fuente = txt.name.split("__")[0]
        e = por_fuente[fuente]
        e["imgs"] += 1
        for c in leer(txt):
            e["cajas"][c] += 1

    print("=" * 78)
    print("DE DÓNDE SALE CADA CLASE")
    print("=" * 78)
    ancho = max(len(f) for f in por_fuente)
    print(f"  {'fuente':<{ancho}} {'imgs':>6}   cajas por clase")
    print("  " + "-" * 72)
    total = Counter()
    for f in sorted(por_fuente):
        e = por_fuente[f]
        det = "  ".join(f"{CLASES[c][:4]}={n}" for c, n in sorted(e["cajas"].items()))
        print(f"  {f:<{ancho}} {e['imgs']:>6}   {det or '(solo negativos)'}")
        total.update(e["cajas"])

    objetivo = [args.clase] if args.clase is not None else list(range(len(CLASES)))
    print("\n" + "=" * 78)
    print("RIESGO DE ETIQUETAS FALTANTES")
    print("=" * 78)
    print("  Una clase está en riesgo si aparece en la vida real dentro de")
    print("  imágenes de OTRAS fuentes donde nadie la etiquetó.\n")

    for c in objetivo:
        con = [f for f, e in por_fuente.items() if e["cajas"].get(c)]
        sin = [(f, e["imgs"]) for f, e in por_fuente.items()
               if not e["cajas"].get(c) and e["imgs"] > 50]
        n_sin = sum(n for _, n in sin)
        n_con = sum(por_fuente[f]["imgs"] for f in con)
        print(f"  clase {c} · {CLASES[c]}  ({total[c]} cajas)")
        print(f"    etiquetada en : {', '.join(con) or '(ninguna)'}  "
              f"-> {n_con} imgs")
        print(f"    NO etiquetada en: {n_sin} imgs de "
              f"{', '.join(f for f, _ in sin) or '(ninguna)'}")
        if n_con:
            ratio = n_sin / n_con
            print(f"    ratio sin/con = {ratio:.1f}x", end="")
            if ratio >= 1.0:
                print("   <- RIESGO ALTO: hay más imágenes donde la clase")
                print("       puede aparecer sin etiqueta que donde sí está.")
            else:
                print()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
