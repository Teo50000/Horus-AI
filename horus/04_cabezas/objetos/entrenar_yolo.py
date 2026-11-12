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
    python entrenar_yolo.py --mezcla datasets/mezcla_v1 --propio auto \
        --extra datasets/armas_ext --comparar ruta/a/horus_yolo4_v3.pt
    python entrenar_yolo.py --prueba          # un minuto en CPU, para ver que anda

--extra (06/11): una carpeta de armas_ext.py con datasets públicos de armas.
Su train entra ENTERO (los negativos de ahí son los difíciles: manos con
celular, billetera, tarjeta, cuchillo); su val y su prueba se miden aparte, así
el val de mezcla_v1 sigue siendo comparable con las versiones anteriores.

07/11 (v5): --extra acepta varias carpetas (armas_ext y fuego_ext, la de llamas
chicas de casa). Cada una aporta su val_<nombre> y su prueba_<nombre>. La medida
por foto ahora es para las cuatro clases en una sola pasada: lo que más importa
de la llama es en cuántas fotos SIN fuego nace un track (una llama sola da
ALERTA a los 1,5 s). Con --modelo <pesos.pt de la v4> se afina la v4 en vez de
arrancar de COCO: la mitad de épocas para el mismo resultado.

    python entrenar_yolo.py --mezcla datasets/mezcla_v1 --propio auto \
        --extra datasets/armas_ext datasets/fuego_ext \
        --modelo v4/horus_yolo4.pt --comparar v4/horus_yolo4.pt --epocas 40 \
        --lr0 0.003 --optimizador SGD --warmup 1 --paciencia 0 --val-con-extras

Lección de la primera v5 (07/11, run 366713820): afinando con lo de siempre
(optimizer auto = MuSGD a 0,01, que además IGNORA --lr0, y warmup de 3 épocas
con bias a 0,1) el val se cayó de 0,67 a 0,51 en la época 3 y recién volvía a
0,67 en la 21. La paciencia de 20 cortó ahí y best.pt quedó en la época 1:
casi la v4. Por eso: SGD a 0,003, una época de warmup sin el bias alto, sin
corte temprano (--paciencia 0), y best.pt se elige con el val de las tres
fuentes juntas (--val-con-extras), no solo con el de mezcla_v1 (que no tiene
una sola vela). val_mezcla se sigue midiendo aparte, igual que siempre.

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
from typing import Dict, List, Optional, Sequence, Tuple, Union

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


def _extra(extra: Optional[Path], parte: str) -> List[Tuple[Path, List[str]]]:
    """(foto, filas de 4 clases) de una parte de la carpeta de armas_ext.py."""
    if extra is None:
        return []
    dimg, dlbl = extra / "images" / parte, extra / "labels" / parte
    if not dimg.is_dir():
        return []
    out = []
    for foto in sorted(p for p in dimg.iterdir() if p.suffix.lower() in EXT):
        lbl = dlbl / (foto.stem + ".txt")
        filas = []
        if lbl.exists():
            for linea in lbl.read_text().splitlines():
                partes = linea.split()
                if len(partes) == 5 and int(partes[0]) in REMAP:
                    filas.append(f"{REMAP[int(partes[0])]} " + " ".join(partes[1:]))
        out.append((foto, filas))
    return out


def ruta_val_mezcla(destino: Path) -> Path:
    """data.yaml del val PURO de mezcla_v1 cuando el de entrenamiento suma los extras."""
    return destino.parent / f"{destino.name}_val_mezcla" / "data.yaml"


def nombre_extra(extra: Path) -> str:
    """datasets/armas_ext -> 'armas'; datasets/fuego_ext -> 'fuego'."""
    n = Path(extra).name
    return n[:-4] if n.endswith("_ext") else n


def armar_eval(extra: Optional[Path], destino: Path, parte: str,
               etiqueta: Optional[str] = None) -> Optional[Path]:
    """Un dataset solo-val con la parte `parte` del extra; devuelve su data.yaml."""
    fotos = _extra(extra, parte)
    if not fotos:
        return None
    etiqueta = etiqueta or nombre_extra(extra)
    d = destino.parent / f"{destino.name}_{parte}_{etiqueta}"
    if d.exists():
        shutil.rmtree(d)
    (d / "images" / "val").mkdir(parents=True)
    (d / "labels" / "val").mkdir(parents=True)
    for foto, filas in fotos:
        _enlazar(foto, d / "images" / "val" / foto.name)
        (d / "labels" / "val" / (foto.stem + ".txt")).write_text("\n".join(filas))
    yaml = d / "data.yaml"
    yaml.write_text(
        f"path: {d.resolve()}\ntrain: images/val\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASES_YOLO)),
        encoding="utf-8")
    con = sum(1 for _, f in fotos if f)
    print(f"[dataset] {parte} de {etiqueta}: {len(fotos)} fotos ({con} con caja, "
          f"{len(fotos) - con} sin ninguna)")
    return yaml


def armar(mezcla: Path, destino: Path, propio: Optional[Path] = None,
          repetir_propio: int = 4, frac_negativos: float = 0.25,
          limite: Optional[int] = None, semilla: int = 0,
          extra: Union[None, Path, Sequence[Path]] = None,
          val_extras: bool = False) -> Path:
    """Arma el dataset de 4 clases y devuelve la ruta al data.yaml.

    val_extras: el val del entrenamiento (con el que se elige best.pt) suma el
    val de cada extra; el val puro de mezcla_v1 queda en ruta_val_mezcla().
    """
    rnd = random.Random(semilla)
    extras = [extra] if isinstance(extra, (str, Path)) else list(extra or [])
    if destino.exists():
        shutil.rmtree(destino)
    puro = ruta_val_mezcla(destino).parent
    if puro.exists():
        shutil.rmtree(puro)
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
            # 06/11: datasets públicos de armas. Entran enteros, sin el tope de
            # negativos: los de ahí son "mano con algo que no es un arma", que
            # es justo lo que le faltaba a la v3 (la media del 06/11).
            # 07/11: lo mismo con fuego_ext (llamas chicas de casa y sus
            # negativos difíciles: lámparas, luces).
            for ex in extras:
                ne = nombre_extra(ex)
                for foto, filas in _extra(ex, "train"):
                    _enlazar(foto, oimg / foto.name)
                    (olbl / (foto.stem + ".txt")).write_text("\n".join(filas))
                    for f in filas:
                        c[CLASES_YOLO[int(f.split()[0])]] += 1
                    c["_fotos"] += 1
                    c["_fondo"] += not filas
                    c["_extra"] += 1
                    c[f"_extra_{ne}"] += 1
        if split == "val" and val_extras and extras:
            # primero, una copia del val puro de mezcla_v1 (la vara de siempre)
            (puro / "images" / "val").mkdir(parents=True, exist_ok=True)
            (puro / "labels" / "val").mkdir(parents=True, exist_ok=True)
            for foto in sorted(oimg.iterdir()):
                _enlazar(foto, puro / "images" / "val" / foto.name)
                shutil.copy2(olbl / (foto.stem + ".txt"), puro / "labels" / "val" / (foto.stem + ".txt"))
            (puro / "data.yaml").write_text(
                f"path: {puro.resolve()}\ntrain: images/val\nval: images/val\n"
                "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASES_YOLO)),
                encoding="utf-8")
            for ex in extras:
                ne = nombre_extra(ex)
                for foto, filas in _extra(ex, "val"):
                    _enlazar(foto, oimg / foto.name)
                    (olbl / (foto.stem + ".txt")).write_text("\n".join(filas))
                    for f in filas:
                        c[CLASES_YOLO[int(f.split()[0])]] += 1
                    c["_fotos"] += 1
                    c["_fondo"] += not filas
                    c["_extra"] += 1
                    c[f"_extra_{ne}"] += 1
        cuenta[split] = c

    yaml = destino / "data.yaml"
    yaml.write_text(
        f"path: {destino.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASES_YOLO)),
        encoding="utf-8")
    for split, c in cuenta.items():
        de_extras = ", ".join(f"{k[7:]} {v}" for k, v in sorted(c.items()) if k.startswith("_extra_"))
        print(f"[dataset] {split}: {c['_fotos']} fotos ({c['_fondo']} de fondo, "
              f"{c['_propio']} de tu cámara, {c['_extra']} de extras: {de_extras or '-'}) · cajas: "
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
             nombre: str = "yolo4", workers: int = 4,
             lr0: Optional[float] = None, optimizador: Optional[str] = None,
             warmup: Optional[float] = None) -> Path:
    from ultralytics import YOLO
    m = YOLO(modelo)
    extra_args: dict = {}
    if lr0:
        # con optimizer="auto" ultralytics ignora lr0 (pasó el 07/11): hay que nombrarlo
        extra_args.update(lr0=lr0, optimizer=optimizador or "SGD")
    elif optimizador:
        extra_args["optimizer"] = optimizador
    if warmup is not None:
        # sin el bias a 0,1 del warmup de siempre: a un modelo ya entrenado lo desarma
        extra_args.update(warmup_epochs=warmup, warmup_bias_lr=lr0 or 0.01)
    r = m.train(
        data=str(data_yaml), imgsz=tam, epochs=epocas, batch=batch,
        device=device or None, patience=paciencia, project=proyecto, name=nombre,
        exist_ok=True, seed=0, workers=workers, cos_lr=True, close_mosaic=10,
        # La realidad es peor que las fotos del dataset: luz de más o de menos,
        # colores corridos, cuadros movidos. Aumentación fuerte de brillo y
        # algo de rotación; el desenfoque lo agrega Albumentations si está.
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.5, degrees=5.0, mixup=0.1,
        plots=False, verbose=True, **extra_args)
    # 05/11: el ultralytics de ahora guarda un `project` relativo adentro de
    # runs/<tarea>/ (quedó en runs/detect/runs_horus/yolo4) y acá se buscaba
    # runs_horus/yolo4: 1,4 h de T4 entrenadas y nada exportado. La carpeta
    # real la sabe el entrenador; las otras quedan como respaldo.
    candidatas = []
    for base in (getattr(getattr(m, "trainer", None), "save_dir", None),
                 getattr(r, "save_dir", None)):
        if base:
            candidatas.append(Path(base) / "weights" / "best.pt")
    candidatas += [Path(proyecto) / nombre / "weights" / "best.pt",
                   Path("runs") / "detect" / proyecto / nombre / "weights" / "best.pt"]
    best = next((p for p in candidatas if p.exists()), None)
    if best is None:
        raise SystemExit("no se generó best.pt; lo busqué en: "
                         + ", ".join(str(p) for p in candidatas))
    print(f"[entrenar] pesos: {best}")
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


def guardar_ultimo(best: Path, salida: Path) -> Optional[Path]:
    """Copia last.pt (la última época) junto a horus_yolo4.pt, para medirlo también:
    best.pt es la mejor época según el val del entrenamiento, que no es todo."""
    last = Path(best).parent / "last.pt"
    if not last.exists():
        return None
    salida.mkdir(parents=True, exist_ok=True)
    dst = salida / "horus_yolo4_ultimo.pt"
    shutil.copy2(last, dst)
    try:
        from ultralytics.utils.torch_utils import strip_optimizer
        strip_optimizer(str(dst))
    except Exception as e:
        print(f"[ultimo] quedó con el optimizador adentro: {e!r}")
    print(f"[ultimo] {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def validar(pt: Path, data_yaml: Path, tam: int = 416, nombre: str = "val") -> Dict[str, float]:
    from ultralytics import YOLO
    r = YOLO(str(pt)).val(data=str(data_yaml), imgsz=tam, verbose=False, plots=False,
                          project=str(Path(data_yaml).parent / "_val"), name=Path(pt).stem,
                          exist_ok=True)
    aps = {CLASES_YOLO[int(i)]: float(r.box.ap50[k]) for k, i in enumerate(r.box.ap_class_index)}
    print(f"[{nombre}] {Path(pt).name}: mAP50={r.box.map50:.3f} · "
          + " ".join(f"{n}={aps[n]:.2f}" for n in CLASES_YOLO if n in aps), flush=True)
    return aps


def por_imagen(pt: Path, data_yaml: Path, tam: int = 416,
               clases: Sequence[str] = CLASES_YOLO,
               umbrales=(0.35, 0.50), nombre: str = "val") -> Dict[str, dict]:
    """Lo que ve el servicio: ¿nace un track de cada clase en la foto?

    El AP no dice cuántas fotos SIN arma (o sin fuego) disparan al umbral con
    el que nace un track (0,35). Esto sí: por foto, la confianza máxima de cada
    clase, contra si la foto tiene o no una caja de esa clase. Una sola pasada
    del modelo para todas las clases. Devuelve {clase: {...}}.
    """
    from ultralytics import YOLO
    m = YOLO(str(pt))
    d = Path(data_yaml).parent
    fotos = sorted(p for p in (d / "images" / "val").iterdir() if p.suffix.lower() in EXT)
    filas = []             # (fuente, {clase: conf máx}, {clases anotadas})
    for i in range(0, len(fotos), 64):
        lote = fotos[i:i + 64]
        for foto, r in zip(lote, m.predict([str(p) for p in lote], imgsz=tam, conf=0.05,
                                           verbose=False, stream=False)):
            cl = r.boxes.cls.tolist() if r.boxes is not None else []
            cf = r.boxes.conf.tolist() if r.boxes is not None else []
            mx = {c: max([s for k, s in zip(cl, cf) if int(k) == CLASES_YOLO.index(c)], default=0.0)
                  for c in clases}
            lbl = d / "labels" / "val" / (foto.stem + ".txt")
            anotadas = set()
            if lbl.exists():
                for l in lbl.read_text().splitlines():
                    t = l.split()
                    if t and t[0].isdigit() and int(t[0]) < len(CLASES_YOLO):
                        anotadas.add(CLASES_YOLO[int(t[0])])
            fuente = foto.name.split("__")[0] if "__" in foto.name else "todo"
            filas.append((fuente, mx, anotadas))

    def resumir(c, s_):
        o = {"fotos_con": len(c), "fotos_sin": len(s_)}
        for u in umbrales:
            o[f"recall_{u:.2f}"] = round(sum(x >= u for x in c) / max(1, len(c)), 4)
            o[f"falsas_{u:.2f}"] = round(sum(x >= u for x in s_) / max(1, len(s_)), 4)
        return o

    def linea(o):
        ve = lambda u: f"ve {o[f'recall_{u:.2f}']:.1%} / " if o["fotos_con"] else ""
        return (f"{o['fotos_con']} con, {o['fotos_sin']} sin · "
                + " · ".join(f"@{u:.2f} {ve(u)}dispara en "
                             f"{o[f'falsas_{u:.2f}']:.1%} de las sin" for u in umbrales))

    res: Dict[str, dict] = {}
    fuentes = sorted({f for f, _, _ in filas})
    for clase in clases:
        con = [(f, mx[clase]) for f, mx, an in filas if clase in an]
        sin = [(f, mx[clase]) for f, mx, an in filas if clase not in an]
        out = resumir([x for _, x in con], [x for _, x in sin])
        print(f"[{nombre}] {Path(pt).name} por foto ({clase}): " + linea(out), flush=True)
        if len(fuentes) > 1:
            out["por_fuente"] = {}
            for fu in fuentes:
                o = resumir([x for f, x in con if f == fu], [x for f, x in sin if f == fu])
                out["por_fuente"][fu] = o
                print(f"    {fu:<12} " + linea(o), flush=True)
        res[clase] = out
    return res


def medir_todo(pesos: Dict[str, Path], yamls: Dict[str, Path], tam: int,
               salida: Path) -> dict:
    """AP50 y la medida por foto (las 4 clases), para cada modelo en cada dataset;
    a metricas.json: {modelo: {dataset: {"ap50": {...}, "por_foto": {clase: {...}}}}}."""
    res: Dict[str, dict] = {}
    for nm, pt in pesos.items():
        if pt is None or not Path(pt).exists():
            continue
        for ny, y in yamls.items():
            if y is None:
                continue
            try:
                res.setdefault(nm, {})[ny] = {"ap50": validar(pt, y, tam, ny),
                                              "por_foto": por_imagen(pt, y, tam, nombre=ny)}
            except Exception as e:     # una medida rota no puede tirar el entrenamiento
                print(f"[medir] {nm} en {ny}: {e!r}", flush=True)
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "metricas.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


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
    ap.add_argument("--proyecto", default="runs_horus",
                    help="carpeta de las corridas; en Kaggle conviene una de "
                         "/kaggle/working para que los pesos queden en Output")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="carpetas de armas_ext.py (armas_ext, fuego_ext, ...)")
    ap.add_argument("--comparar", default="",
                    help="pesos .pt de la versión anterior, para medirla igual")
    ap.add_argument("--lr0", type=float, default=None,
                    help="tasa inicial (con SGD salvo --optimizador); sin esto, la de ultralytics")
    ap.add_argument("--optimizador", default=None, help="SGD, AdamW, MuSGD...; sin esto, auto")
    ap.add_argument("--warmup", type=float, default=None,
                    help="épocas de warmup, sin el bias alto (para afinar un modelo ya entrenado)")
    ap.add_argument("--paciencia", type=int, default=20,
                    help="épocas sin mejorar antes de cortar; 0 = no cortar nunca")
    ap.add_argument("--val-con-extras", dest="val_con_extras", action="store_true",
                    help="best.pt se elige con el val de mezcla_v1 + el de cada extra")
    ap.add_argument("--solo-armar", dest="solo_armar", action="store_true")
    ap.add_argument("--prueba", action="store_true",
                    help="dataset sintético chico y 1 época: verifica el camino entero")
    a = ap.parse_args()

    if a.prueba:
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        mezcla = _prueba_sintetica(tmp / "mezcla")
        extras = []
        for nom, pos in (("armas_ext", "3"), ("fuego_ext", "1")):      # pistola / llama (7 clases)
            ext = tmp / nom
            extras.append(ext)
            for parte in ("train", "val", "prueba"):
                (ext / "images" / parte).mkdir(parents=True)
                (ext / "labels" / parte).mkdir(parents=True)
                for f in sorted((mezcla / "images" / ("val" if parte != "train" else "train")).iterdir())[:6]:
                    shutil.copy2(f, ext / "images" / parte / f"{nom[:2]}__{parte}_{f.name}")
                    (ext / "labels" / parte / f"{nom[:2]}__{parte}_{f.stem}.txt").write_text(
                        f"{pos} 0.5 0.5 0.2 0.2" if f.stem.endswith(("0", "2", "4")) else "5 0.5 0.5 0.2 0.2")
        yaml = armar(mezcla, tmp / "yolo4", None, extra=extras, val_extras=True)
        yamls = {"val_mezcla": ruta_val_mezcla(tmp / "yolo4")}
        n_val = len(list((tmp / "yolo4" / "images" / "val").iterdir()))
        n_puro = len(list((ruta_val_mezcla(tmp / "yolo4").parent / "images" / "val").iterdir()))
        assert n_puro > 0 and n_val == n_puro + 12, (n_val, n_puro)    # 6 de val por extra
        for ext in extras:
            yamls[f"val_{nombre_extra(ext)}"] = armar_eval(ext, tmp / "yolo4", "val")
            yamls[f"prueba_{nombre_extra(ext)}"] = armar_eval(ext, tmp / "yolo4", "prueba")
        best = entrenar(yaml, a.modelo, tam=160, epocas=1, batch=8, device="cpu",
                        proyecto=str(tmp / "runs"), workers=0)
        rutas = exportar(best, tmp / "salida", tam=160)
        # afinar desde los pesos recién hechos (como la v5 desde la v4)
        best2 = entrenar(yaml, str(rutas["pt"]), tam=160, epocas=1, batch=8, device="cpu",
                         proyecto=str(tmp / "runs2"), workers=0, lr0=0.005, optimizador="SGD",
                         warmup=1, paciencia=0)
        assert best2.exists(), best2
        ultimo = guardar_ultimo(best2, tmp / "salida")
        assert ultimo is not None and ultimo.exists(), ultimo
        res = medir_todo({"nuevo": rutas["pt"], "ultimo": ultimo, "anterior": rutas["pt"]},
                         yamls, 160, tmp / "salida")
        esperados = {"val_mezcla", "val_armas", "prueba_armas", "val_fuego", "prueba_fuego"}
        assert set(res.get("nuevo", {})) == esperados, res.get("nuevo", {}).keys()
        assert set(res.get("ultimo", {})) == esperados, res.get("ultimo", {}).keys()
        pf = res["nuevo"]["val_fuego"]["por_foto"]
        assert set(pf) == set(CLASES_YOLO) and pf["llama"]["fotos_con"] == 3, pf["llama"]
        assert res["nuevo"]["val_armas"]["por_foto"]["pistola"]["fotos_con"] == 3
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
    extras = [Path(e) for e in a.extra if e]
    for ex in extras:
        if not (ex / "images" / "train").is_dir():
            raise SystemExit(f"--extra {ex}: no tiene images/train (¿corrió armas_ext.py?)")
    yaml = armar(Path(a.mezcla), Path(a.destino), propio, extra=extras,
                 val_extras=a.val_con_extras)
    yamls = {"val_mezcla": ruta_val_mezcla(Path(a.destino)) if a.val_con_extras and extras else yaml}
    for ex in extras:
        yamls[f"val_{nombre_extra(ex)}"] = armar_eval(ex, Path(a.destino), "val")
        yamls[f"prueba_{nombre_extra(ex)}"] = armar_eval(ex, Path(a.destino), "prueba")
    if a.solo_armar:
        return 0
    best = entrenar(yaml, a.modelo, tam=a.tam, epocas=a.epocas, batch=a.batch,
                    device=a.device, proyecto=a.proyecto, lr0=a.lr0, paciencia=a.paciencia,
                    optimizador=a.optimizador, warmup=a.warmup)
    rutas = exportar(best, Path(a.salida), tam=a.tam)
    pesos = {"nuevo": rutas["pt"]}
    ultimo = guardar_ultimo(best, Path(a.salida))
    if ultimo:
        pesos["ultimo"] = ultimo
    pesos["anterior"] = Path(a.comparar) if a.comparar else None
    medir_todo(pesos, yamls, a.tam, Path(a.salida))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())