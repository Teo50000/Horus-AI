# -*- coding: utf-8 -*-
"""
entrenar_yolo.py · YOLO11 para lo que COCO no sabe: pistola, humo, llama, paquete.

26/09. El detector preentrenado (motor_yolo.py) ya se encarga de persona,
cuchillo y celular. Para las otras cuatro clases seguía corriendo la cabeza
propia arriba de un ResNet-50, que es lo más lento de todo el servicio. Este
script entrena un YOLO11 chico SOLO con esas cuatro, partiendo de los pesos de
COCO (que ya saben ver bordes, texturas, objetos), con el mismo dataset que
armaba la cabeza propia.

Por qué cuatro clases y no siete
--------------------------------
El dataset mezcla fuentes que no anotan lo mismo: D-Fire y Pyro-SDIS tienen
personas SIN etiquetar. La cabeza propia lo resolvía ignorando la clase persona
en esas fotos (v4); YOLO no tiene esa opción, y entrenar persona así le
enseñaría que "la gente en una foto de incendio es fondo". Con cuatro clases el
problema desaparece: para este modelo una persona es fondo SIEMPRE, y la
persona la sigue viendo el YOLO de COCO, que fue entrenado con cientos de miles.

    python entrenar_yolo.py --mezcla datasets/mezcla_v1 --propio auto
    python entrenar_yolo.py --prueba          # un minuto en CPU, para ver que anda

Sale `horus_yolo4.pt` y `horus_yolo4_<tam>.onnx`. Van a `modelos/` y el servicio
los usa solos (ver `motor_yolo.pesos_horus_por_defecto`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CLASES_7 = ("humo", "llama", "persona", "pistola", "cuchillo", "celular", "paquete")
CLASES_YOLO = ("pistola", "humo", "llama", "paquete")
REMAP = {CLASES_7.index(c): i for i, c in enumerate(CLASES_YOLO)}
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _carpetas(mezcla: Path, split: str) -> Tuple[Path, Path]:
    """Las dos disposiciones que conviven: images/<split> y <split>/images."""
    for img, lbl in ((mezcla / "images" / split, mezcla / "labels" / split),
                     (mezcla / split / "images", mezcla / split / "labels")):
        if img.is_dir():
            return img, lbl
    raise SystemExit(f"no encuentro las imágenes de '{split}' en {mezcla}")


def _enlazar(origen: Path, destino: Path) -> None:
    """Symlink si se puede (Kaggle, Linux); copia si no (Windows sin permisos)."""
    if destino.exists() or destino.is_symlink():
        destino.unlink()
    try:
        destino.symlink_to(origen.resolve())
    except OSError:
        shutil.copy2(origen, destino)


def _propio(raiz: Optional[Path]) -> List[Tuple[Path, str]]:
    """Frames de la cámara propia que son negativos SEGUROS para las 4 clases.

    Solo las escenas que declaran ausentes a las cuatro (la "vacia" las declara
    a todas). Una escena donde alguna podría estar no se usa: YOLO no tiene
    forma de decir "de esto no opino" y una caja faltante le enseña que el
    objeto es fondo.
    """
    if raiz is None:
        return []
    man = raiz / "horus_propio.json"
    if not man.exists():
        hallados = sorted(raiz.rglob("horus_propio.json"))
        if not hallados:
            print(f"[propio] no hay horus_propio.json en {raiz}")
            return []
        man = hallados[0]
    datos = json.loads(man.read_text(encoding="utf-8"))
    out = []
    for escena, info in (datos.get("escenas") or {}).items():
        ausentes = set(info.get("ausentes") or [])
        # 26/09: la vacia se revisó a mano (ninguna de las 7 clases); los
        # manifiestos viejos solo declaraban persona/pistola/cuchillo.
        if escena == "vacia":
            ausentes = set(CLASES_7)
        if not set(CLASES_YOLO) <= ausentes:
            print(f"[propio] escena '{escena}': no declara ausentes a las 4 clases, la salteo")
            continue
        fotos = sorted(p for p in (man.parent / escena / "images").glob("*")
                       if p.suffix.lower() in EXT)
        out += [(f, escena) for f in fotos]
        print(f"[propio] escena '{escena}': {len(fotos)} frames como negativos")
    return out


def armar(mezcla: Path, destino: Path, propio: Optional[Path] = None,
          repetir_propio: int = 4, frac_negativos: float = 0.25,
          limite: Optional[int] = None, semilla: int = 0) -> Path:
    """Arma el dataset de 4 clases y devuelve la ruta al data.yaml."""
    rnd = random.Random(semilla)
    if destino.exists():
        shutil.rmtree(destino)
    cuenta: Dict[str, Counter] = {}
    for split in ("train", "val"):
        dimg, dlbl = _carpetas(mezcla, split)
        oimg, olbl = destino / "images" / split, destino / "labels" / split
        oimg.mkdir(parents=True, exist_ok=True)
        olbl.mkdir(parents=True, exist_ok=True)
        positivas, negativas = [], []
        fotos = sorted(p for p in dimg.iterdir() if p.suffix.lower() in EXT)
        if limite:
            fotos = fotos[:limite]
        for foto in fotos:
            lbl = dlbl / (foto.stem + ".txt")
            filas = []
            if lbl.exists():
                for linea in lbl.read_text().splitlines():
                    partes = linea.split()
                    if len(partes) == 5 and int(partes[0]) in REMAP:
                        filas.append(f"{REMAP[int(partes[0])]} " + " ".join(partes[1:]))
            (positivas if filas else negativas).append((foto, filas))
        # Fondo: hace falta (es lo que enseña a NO disparar), pero si son la
        # mitad del dataset el modelo aprende que casi nunca hay nada.
        max_neg = int(len(positivas) * frac_negativos / max(1e-6, 1 - frac_negativos))
        rnd.shuffle(negativas)
        elegidas = positivas + negativas[:max_neg]
        c = Counter()
        for foto, filas in elegidas:
            _enlazar(foto, oimg / foto.name)
            (olbl / (foto.stem + ".txt")).write_text("\n".join(filas))
            for f in filas:
                c[CLASES_YOLO[int(f.split()[0])]] += 1
            c["_fotos"] += 1
            c["_fondo"] += not filas
        if split == "train":
            for foto, escena in _propio(propio):
                for k in range(max(1, repetir_propio)):
                    nombre = f"propio-{escena}__{foto.stem}_r{k}{foto.suffix.lower()}"
                    _enlazar(foto, oimg / nombre)
                    (olbl / (Path(nombre).stem + ".txt")).write_text("")
                    c["_fotos"] += 1
                    c["_propio"] += 1
        cuenta[split] = c

    yaml = destino / "data.yaml"
    yaml.write_text(
        f"path: {destino.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASES_YOLO)),
        encoding="utf-8")
    for split, c in cuenta.items():
        print(f"[dataset] {split}: {c['_fotos']} fotos ({c['_fondo']} de fondo, "
              f"{c['_propio']} de tu cámara) · cajas: "
              + ", ".join(f"{n}={c[n]}" for n in CLASES_YOLO))
    vacias = [n for n in CLASES_YOLO if cuenta["train"][n] == 0]
    if vacias:
        raise SystemExit(f"sin una sola caja de {vacias} en train: no entreno a ciegas")
    return yaml


# --------------------------------------------------------------------------- #
# Entrenar y exportar
# --------------------------------------------------------------------------- #
def entrenar(data_yaml: Path, modelo: str = "yolo11n.pt", tam: int = 416,
             epocas: int = 80, batch: int = -1, device: str = "",
             paciencia: int = 20, proyecto: str = "runs_horus",
             nombre: str = "yolo4", workers: int = 4) -> Path:
    from ultralytics import YOLO
    m = YOLO(modelo)
    m.train(
        data=str(data_yaml), imgsz=tam, epochs=epocas, batch=batch,
        device=device or None, patience=paciencia, project=proyecto, name=nombre,
        exist_ok=True, seed=0, workers=workers, cos_lr=True, close_mosaic=10,
        # La realidad es peor que las fotos del dataset: luz de más o de menos,
        # colores corridos, cuadros movidos. Aumentación fuerte de brillo y
        # algo de rotación; el desenfoque lo agrega Albumentations si está.
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.5, degrees=5.0, mixup=0.1,
        plots=False, verbose=True)
    best = Path(proyecto) / nombre / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"no se generó {best}")
    return best


def exportar(best: Path, salida: Path, tam: int = 416) -> Dict[str, Path]:
    from ultralytics import YOLO
    salida.mkdir(parents=True, exist_ok=True)
    pt = salida / "horus_yolo4.pt"
    shutil.copy2(best, pt)
    onnx = Path(YOLO(str(best)).export(format="onnx", imgsz=tam, simplify=True))
    destino_onnx = salida / f"horus_yolo4_{tam}.onnx"
    shutil.move(str(onnx), destino_onnx)
    print(f"[exportar] {pt.name} ({pt.stat().st_size / 1e6:.1f} MB) · "
          f"{destino_onnx.name} ({destino_onnx.stat().st_size / 1e6:.1f} MB)")
    return {"pt": pt, "onnx": destino_onnx}


def validar(pt: Path, data_yaml: Path, tam: int = 416) -> Dict[str, float]:
    from ultralytics import YOLO
    r = YOLO(str(pt)).val(data=str(data_yaml), imgsz=tam, verbose=False, plots=False)
    aps = {CLASES_YOLO[int(i)]: float(r.box.ap50[k]) for k, i in enumerate(r.box.ap_class_index)}
    print(f"[val] mAP50={r.box.map50:.3f} · " + " ".join(f"{n}={aps.get(n, 0):.2f}" for n in CLASES_YOLO))
    return aps


# --------------------------------------------------------------------------- #
def _prueba_sintetica(dest: Path) -> Path:
    """Un mini dataset de mentira en el formato de mezcla_v1: 7 clases."""
    import numpy as np
    import cv2
    rnd = np.random.default_rng(0)
    for split, n in (("train", 24), ("val", 8)):
        (dest / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((240, 320, 3), 90, np.uint8)
            filas = []
            for c in (i % 7, (i + 3) % 7):
                x, y = rnd.integers(20, 250), rnd.integers(20, 170)
                cv2.rectangle(img, (int(x), int(y)), (int(x) + 50, int(y) + 50),
                              tuple(int(v) for v in rnd.integers(0, 255, 3)), -1)
                filas.append(f"{c} {(x + 25) / 320:.4f} {(y + 25) / 240:.4f} "
                             f"{50 / 320:.4f} {50 / 240:.4f}")
            cv2.imwrite(str(dest / "images" / split / f"f{i}.jpg"), img)
            (dest / "labels" / split / f"f{i}.txt").write_text("\n".join(filas))
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mezcla", default="datasets/mezcla_v1")
    ap.add_argument("--propio", default="auto",
                    help='carpeta de grabar_propio.py, "auto" (la busca en /kaggle/input '
                         'y en datasets/propio) o "no"')
    ap.add_argument("--destino", default="datasets/yolo4")
    ap.add_argument("--modelo", default="yolo11n.pt")
    ap.add_argument("--tam", type=int, default=416)
    ap.add_argument("--epocas", type=int, default=80)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default="")
    ap.add_argument("--salida", default="modelos_yolo")
    ap.add_argument("--solo-armar", dest="solo_armar", action="store_true")
    ap.add_argument("--prueba", action="store_true",
                    help="dataset sintético chico y 1 época: verifica el camino entero")
    a = ap.parse_args()

    if a.prueba:
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        mezcla = _prueba_sintetica(tmp / "mezcla")
        yaml = armar(mezcla, tmp / "yolo4", None)
        best = entrenar(yaml, a.modelo, tam=160, epocas=1, batch=8, device="cpu",
                        proyecto=str(tmp / "runs"), workers=0)
        rutas = exportar(best, tmp / "salida", tam=160)
        from motor_yolo import ConfigYolo, MotorYolo
        m = MotorYolo(ConfigYolo(pesos=str(rutas["onnx"]), tam=160, verboso=True))
        ok = set(m.mapa.values()) == set(CLASES_YOLO)
        print("PRUEBA", "OK" if ok else "FALLA", "· clases que ve el motor:", sorted(m.mapa.values()))
        return 0 if ok else 1

    propio: Optional[Path] = None
    if a.propio == "auto":
        for base in (Path("/kaggle/input"), Path("datasets/propio")):
            if base.exists():
                hall = sorted(base.rglob("horus_propio.json"))
                if hall:
                    propio = hall[0].parent
                    break
    elif a.propio != "no":
        propio = Path(a.propio)
    yaml = armar(Path(a.mezcla), Path(a.destino), propio)
    if a.solo_armar:
        return 0
    best = entrenar(yaml, a.modelo, tam=a.tam, epocas=a.epocas, batch=a.batch,
                    device=a.device)
    rutas = exportar(best, Path(a.salida), tam=a.tam)
    validar(rutas["pt"], yaml, a.tam)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
