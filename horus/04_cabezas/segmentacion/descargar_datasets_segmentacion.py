#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
descargar_datasets_segmentacion.py
==================================

Arma el dataset `mezcla_seg_v1` para la cabeza de segmentacion de Horus.

    0 = fondo      1 = agua      2 = humo      3 = fuego      255 = ignorar

Por que 255 existe
------------------
Ninguna fuente publica anota las cuatro clases a la vez. Un dataset de fuego
marca las llamas y deja TODO lo demas como fondo -- incluida la columna de
humo que sale de esas llamas. Si se entrena con eso tal cual, el humo real
recibe supervision "esto es fondo" y la clase 2 nunca despega.

Este script resuelve eso en dos capas:

1. `anotadas`: cada muestra declara que clases sabe etiquetar su fuente.
   Se guarda en el manifiesto. `entrenar_segmentacion_cuda.py` calcula la
   perdida con un softmax restringido a ese subconjunto, asi que predecir
   "agua" sobre el fondo de una foto de incendio no se penaliza.
2. `halo`: alrededor de cada region positiva se marca un anillo de N pixeles
   como 255. Ahi es donde vive el humo sin etiquetar pegado a la llama, la
   espuma del borde del agua y el error de trazado del anotador.

Uso tipico
----------
    python descargar_datasets_segmentacion.py --listar
    python descargar_datasets_segmentacion.py --kaggle --hf --gdown
    python descargar_datasets_segmentacion.py --todo --roboflow-key XXXX
    python descargar_datasets_segmentacion.py --verificar

Se puede cortar y volver a correr: cada paso deja un centinela `.ok` y no se
repite. `--purgar-crudos` borra los zips y los arboles extraidos una vez que
la conversion termino (recupera la mayor parte del espacio).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

try:
    import numpy as np
    from PIL import Image, ImageFilter
except ImportError:  # pragma: no cover
    sys.exit("Faltan dependencias base:  pip install numpy pillow")

Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------------
# Constantes del esquema de etiquetas
# --------------------------------------------------------------------------

CLASES = ["fondo", "agua", "humo", "fuego"]
N_CLASES = len(CLASES)
IGNORAR = 255

# Paleta para que las mascaras se puedan mirar con cualquier visor.
PALETA = [0, 0, 0] * 256
PALETA[0:3] = [0, 0, 0]         # fondo   negro
PALETA[3:6] = [30, 110, 220]    # agua    azul
PALETA[6:9] = [170, 170, 170]   # humo    gris
PALETA[9:12] = [235, 90, 20]    # fuego   naranja
PALETA[765:768] = [255, 0, 255]  # 255 ignorar  magenta

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PISTAS_MASCARA = ("mask", "masks", "groundtruth", "ground_truth", "gt",
                  "label", "labels", "annotation", "annotations", "seg",
                  "segmentation", "target", "mascara", "mascaras")
SUFIJOS_MASCARA = ("_mask", "-mask", "_gt", "-gt", "_label", "-label",
                   "_seg", "-seg", "_m", "_anno", "_annotation")


# --------------------------------------------------------------------------
# Registro de fuentes
# --------------------------------------------------------------------------

@dataclass
class Fuente:
    nombre: str
    resumen: str
    clases: tuple[int, ...]          # que clases sabe anotar (0 siempre incluido)
    metodo: str                      # http | kaggle | hf | gdown | git | roboflow | manual
    recursos: tuple = ()             # depende del metodo
    conversor: str = "binaria"       # nombre de la funcion de conversion
    args_conv: dict = field(default_factory=dict)
    vista: str = "suelo"             # suelo | aereo | sintetico
    gb: float = 0.0
    tope: int | None = None          # maximo de muestras a quedarse
    prob_negativos: float = 1.0      # con que probabilidad se guarda una
                                     # muestra sin ninguna clase positiva
    niega: tuple[int, ...] = ()      # clases que el fondo de esta fuente puede
                                     # negar con confianza. Medido: negar FUEGO
                                     # en escenas de agua ayuda (+0.014 IoU);
                                     # negar HUMO las hunde (-0.057), porque
                                     # nubes y niebla son humo visualmente y el
                                     # modelo solo puede obedecer prediciendo
                                     # menos humo en todos lados.
    halo: int = 6                    # anillo de ignorar alrededor de lo positivo
    licencia: str = ""
    url: str = ""
    nota: str = ""
    grupo: str = "base"              # base | credencial | aereo | opcional

    @property
    def necesita_credencial(self) -> bool:
        return self.metodo in {"kaggle", "roboflow"}


FUENTES: list[Fuente] = [
    # ---------------------------------------------------------------- agua
    Fuente(
        nombre="cocostuff",
        resumen="COCO-Stuff 164k: rio/mar/agua + el mejor pozo de negativos que existe",
        clases=(0, 1),
        metodo="http",
        recursos=(
            ("http://images.cocodataset.org/zips/train2017.zip", "train2017.zip"),
            ("http://images.cocodataset.org/zips/val2017.zip", "val2017.zip"),
            ("http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/"
             "stuffthingmaps_trainval2017.zip", "stuffthingmaps.zip"),
        ),
        conversor="cocostuff",
        gb=19.5,
        # COCO-Stuff son 123k imagenes y solo ~11k tienen agua. Guardar las
        # 112k restantes enteras inflaria la mezcla y el disco sin aportar
        # nada nuevo: con ~8k negativos duros (nubes, vapor, asfalto mojado,
        # atardeceres) ya alcanza para que el modelo no vea agua ni fuego en
        # todos lados. El filtro es por hash del nombre, asi que es estable
        # entre corridas.
        prob_negativos=0.07,
        # Una escena de COCO con agua no tiene fuego. `--negativos` decide
        # cuanto de `niega` se aplica: por defecto solo el fuego, porque tres
        # corridas mostraron que negar humo aca hunde la clase humo.
        niega=(2, 3),
        halo=0,          # anotacion densa y prolija: no hace falta halo
        licencia="CC BY 4.0 (anotaciones) / imagenes COCO",
        url="https://github.com/nightrome/cocostuff",
        nota="Aporta ~11k imagenes con agua y el fondo duro (nubes, nieve, "
             "vapor) que evita que el modelo vea agua en todos lados.",
    ),
    Fuente(
        nombre="ade20k",
        resumen="ADE20K SceneParse150: agua/mar/rio/lago/pileta a nivel del suelo",
        clases=(0, 1),
        metodo="http",
        recursos=(
            ("http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
             "ADEChallengeData2016.zip"),
        ),
        conversor="ade20k",
        niega=(2, 3),
        gb=1.0,
        halo=0,
        licencia="BSD-3 / uso academico",
        url="https://groups.csail.mit.edu/vision/datasets/ADE20K/",
    ),
    Fuente(
        nombre="atlantis",
        resumen="ATLANTIS: 5.195 fotos de cuerpos de agua, 56 etiquetas finas",
        clases=(0, 1),
        metodo="git",
        recursos=("https://github.com/smhassanerfani/atlantis.git",),
        conversor="atlantis",
        niega=(2, 3),
        gb=2.5,
        halo=2,
        licencia="ver repo",
        url="https://github.com/smhassanerfani/atlantis",
        nota="Si el repo no trae las imagenes, bajalas a mano segun el README "
             "y dejalas en descargas_crudas/atlantis/ ; el conversor las toma igual.",
    ),
    Fuente(
        nombre="riwa",
        resumen="RIWA: segmentacion de rios a nivel del suelo, camara de mano",
        clases=(0, 1),
        metodo="kaggle",
        recursos=("franzwagner/river-water-segmentation-dataset",),
        niega=(2, 3),
        gb=1.2,
        halo=4,
        licencia="ver Kaggle",
        url="https://www.kaggle.com/datasets/franzwagner/river-water-segmentation-dataset",
        grupo="credencial",
    ),
    Fuente(
        nombre="flood_area",
        resumen="Flood Area Segmentation: inundacion urbana vista desde el piso",
        clases=(0, 1),
        metodo="kaggle",
        recursos=("faizalkarim/flood-area-segmentation",),
        niega=(2, 3),
        gb=0.4,
        halo=6,
        licencia="ver Kaggle",
        url="https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation",
        grupo="credencial",
        nota="La fuente mas parecida a lo que ve una camara fija inundada.",
    ),
    Fuente(
        nombre="flood_semantic",
        resumen="Flood Semantic Segmentation Dataset",
        clases=(0, 1),
        metodo="kaggle",
        recursos=("lihuayang111265/flood-semantic-segmentation-dataset",),
        niega=(2, 3),
        gb=0.6,
        halo=6,
        licencia="ver Kaggle",
        url="https://www.kaggle.com/datasets/lihuayang111265/flood-semantic-segmentation-dataset",
        grupo="credencial",
    ),
    Fuente(
        nombre="flood_masks",
        resumen="Flood images + mask segmentation",
        clases=(0, 1),
        metodo="kaggle",
        recursos=("saiharshitjami/flood-images-mask-segmentation",),
        niega=(2, 3),
        gb=0.5,
        halo=6,
        licencia="ver Kaggle",
        url="https://www.kaggle.com/datasets/saiharshitjami/flood-images-mask-segmentation",
        grupo="credencial",
    ),
    Fuente(
        nombre="floodnet",
        resumen="FloodNet: inundacion post-huracan desde dron (10 clases)",
        clases=(0, 1),
        metodo="kaggle",
        recursos=("aletbm/aerial-imagery-dataset-floodnet-challenge",),
        conversor="floodnet",
        niega=(2, 3),
        vista="aereo",
        gb=6.0,
        halo=0,
        licencia="CC BY-NC-SA",
        url="https://github.com/BinaLab/FloodNet-Supervised_v1.0",
        grupo="aereo",
        nota="Vista cenital: util para robustez de textura, no para geometria "
             "de camara fija. Entra solo con --incluir-aereo.",
    ),

    # ---------------------------------------------------------------- humo
    Fuente(
        nombre="multinatsmoke",
        resumen="MultiNatSmoke (WACV 2026): humo real diverso, totalmente etiquetado",
        clases=(0, 2),
        metodo="hf",
        recursos=(("hongjinzhao0615/MultiNatSmoke", "MultiNatSmokeDataset.zip"),),
        gb=3.0,
        halo=10,
        licencia="mezcla CC BY / CC BY-NC-SA / CC0 -- ver repo",
        url="https://huggingface.co/datasets/hongjinzhao0615/MultiNatSmoke",
        nota="La fuente de humo mas util de la lista: humo real, escenas variadas.",
    ),
    Fuente(
        nombre="smoke100k",
        resumen="Smoke100k: 100k imagenes con humo sintetico compuesto y su mascara",
        clases=(0, 2),
        metodo="gdown",
        recursos=("1a3C010h7zzNPomMpddb74X3GffZx-R9C",),
        conversor="smoke100k",
        vista="sintetico",
        gb=12.0,
        tope=12000,
        halo=8,
        licencia="uso academico",
        url="https://bigmms.github.io/cheng_gcce19_smoke100k/",
        nota="Sintetico y enorme. Topeado a 12k para que no se coma la mezcla; "
             "sirve para bordes translucidos, no para color realista.",
    ),
    Fuente(
        nombre="smoke5k",
        resumen="SMOKE5K: 4k sinteticas + 1.4k reales con mascara",
        clases=(0, 2),
        metodo="manual",
        recursos=(),
        gb=1.5,
        halo=8,
        licencia="uso academico",
        url="https://github.com/redlessme/Transmission-BVM",
        grupo="opcional",
        nota="El link vive en Google Drive y cambia seguido. Bajalo a mano y "
             "descomprimilo en descargas_crudas/smoke5k/ ; el conversor generico "
             "lo levanta solo (empareja imagen y mascara por nombre).",
    ),

    # --------------------------------------------------------------- fuego
    Fuente(
        nombre="bowfire",
        resumen="BoWFire: fuego real con mascara pixel a pixel (chico pero limpio)",
        clases=(0, 3),
        metodo="http",
        recursos=(("https://bitbucket.org/gbdi/bowfire-dataset/downloads/BoWFireDataset.zip",
                   "BoWFireDataset.zip"),),
        args_conv={"excluir": ("train",)},
        gb=0.1,
        halo=8,
        licencia="uso academico",
        url="https://bitbucket.org/gbdi/bowfire-dataset/",
        nota="Solo el split de test trae mascaras; train son recortes de "
             "clasificacion y se descartan.",
    ),
    Fuente(
        nombre="fire_seg_diversis",
        resumen="Fire Segmentation Image Dataset",
        clases=(0, 3),
        metodo="kaggle",
        recursos=("diversisai/fire-segmentation-image-dataset",),
        gb=1.5,
        halo=8,
        licencia="ver Kaggle",
        url="https://www.kaggle.com/datasets/diversisai/fire-segmentation-image-dataset",
        grupo="credencial",
    ),
    Fuente(
        nombre="fire_seg_killa",
        resumen="Fire Segmentation Dataset",
        clases=(0, 3),
        metodo="kaggle",
        recursos=("killa92/fire-segmentation-dataset",),
        gb=0.8,
        halo=8,
        licencia="ver Kaggle",
        url="https://www.kaggle.com/datasets/killa92/fire-segmentation-dataset",
        grupo="credencial",
    ),
    Fuente(
        nombre="corsican",
        resumen="Corsican Fire Database: ~500 fotos de fuego con mascara",
        clases=(0, 3),
        metodo="manual",
        recursos=(),
        gb=1.0,
        halo=6,
        licencia="registro obligatorio",
        url="https://cfdb.univ-corse.fr/",
        grupo="opcional",
        nota="Pide cuenta. Es el mejor set de fuego real que hay: si vas a "
             "registrarte en uno solo, que sea este. Descomprimilo en "
             "descargas_crudas/corsican/.",
    ),

    # ------------------------------------------------------- humo + fuego
    Fuente(
        nombre="rf_fuego_humo",
        resumen="Roboflow: fire-and-smoke-segmentation (poligonos -> mascara)",
        clases=(0, 2, 3),
        metodo="roboflow",
        recursos=(("roboflow-universe-projects", "fire-and-smoke-segmentation"),),
        conversor="coco_poligonos",
        gb=1.0,
        halo=6,
        licencia="ver Roboflow Universe",
        url="https://universe.roboflow.com/roboflow-universe-projects/fire-and-smoke-segmentation",
        grupo="credencial",
        nota="Unica fuente que anota humo Y fuego juntos: es la que le ensena "
             "al modelo a separarlos en la misma escena.",
    ),
    Fuente(
        nombre="rf_fuego_humo_2",
        resumen="Roboflow: fire-smoke (segunda opinion, otra distribucion)",
        clases=(0, 2, 3),
        metodo="roboflow",
        recursos=(("project-sdorc", "fire-smoke-az0ku"),),
        conversor="coco_poligonos",
        gb=0.8,
        halo=6,
        licencia="ver Roboflow Universe",
        url="https://universe.roboflow.com/project-sdorc/fire-smoke-az0ku",
        grupo="credencial",
    ),
]

POR_NOMBRE = {f.nombre: f for f in FUENTES}


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def log(msg: str, nivel: str = "") -> None:
    marca = {"": "  ", "ok": "OK", "warn": "!!", "err": "XX", "paso": "->"}[nivel]
    print(f"[{marca}] {msg}", flush=True)


def humano(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def centinela(p: Path) -> Path:
    return p.parent / (p.name + ".ok")


def ya_esta(p: Path) -> bool:
    return centinela(p).exists()


def marcar(p: Path, info: str = "") -> None:
    centinela(p).write_text(info or time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def libre_gb(p: Path) -> float:
    try:
        return shutil.disk_usage(p).free / 1024 ** 3
    except OSError:
        return float("inf")


# --------------------------------------------------------------------------
# Descarga
# --------------------------------------------------------------------------

def bajar_http(url: str, destino: Path, reintentos: int = 5) -> Path:
    """Descarga con reanudacion por Range y reintento exponencial."""
    import urllib.error
    import urllib.request

    destino.parent.mkdir(parents=True, exist_ok=True)
    if ya_esta(destino):
        log(f"{destino.name} ya bajado", "ok")
        return destino

    parcial = destino.with_suffix(destino.suffix + ".parcial")
    for intento in range(1, reintentos + 1):
        desde = parcial.stat().st_size if parcial.exists() else 0
        pedido = urllib.request.Request(url, headers={"User-Agent": "horus/1.0"})
        if desde:
            pedido.add_header("Range", f"bytes={desde}-")
        try:
            with urllib.request.urlopen(pedido, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0)) + desde
                modo = "ab" if desde and r.status == 206 else "wb"
                if modo == "wb":
                    desde = 0
                t0, ultimo = time.time(), 0.0
                with open(parcial, modo) as fh:
                    while True:
                        trozo = r.read(1 << 20)
                        if not trozo:
                            break
                        fh.write(trozo)
                        desde += len(trozo)
                        ahora = time.time()
                        if ahora - ultimo > 2:
                            ultimo = ahora
                            vel = desde / max(ahora - t0, 1e-6)
                            pct = f"{100 * desde / total:5.1f}%" if total else "  ?  "
                            print(f"\r      {destino.name}  {pct}  "
                                  f"{humano(desde)}  {humano(vel)}/s   ", end="", flush=True)
            print()
            parcial.replace(destino)
            marcar(destino, f"{url}\n{destino.stat().st_size} bytes")
            log(f"{destino.name} listo ({humano(destino.stat().st_size)})", "ok")
            return destino
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print()
            espera = min(2 ** intento, 60)
            log(f"{destino.name}: {e} -- reintento {intento}/{reintentos} en {espera}s", "warn")
            time.sleep(espera)
    raise RuntimeError(f"no se pudo bajar {url}")


def bajar_kaggle(slug: str, destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    if ya_esta(destino):
        log(f"kaggle:{slug} ya bajado", "ok")
        return destino
    exe = shutil.which("kaggle")
    if exe:
        cmd = [exe, "datasets", "download", "-d", slug, "-p", str(destino), "--unzip"]
    else:
        cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
               "-d", slug, "-p", str(destino), "--unzip"]
    log(f"kaggle datasets download -d {slug}", "paso")
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"kaggle fallo para {slug}:\n{r.stdout}\n{r.stderr}")
    marcar(destino, slug)
    return destino


def bajar_hf(repo: str, archivo: str | None, destino: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise RuntimeError("falta huggingface_hub:  pip install huggingface_hub")
    destino.mkdir(parents=True, exist_ok=True)
    if ya_esta(destino):
        log(f"hf:{repo} ya bajado", "ok")
        return destino
    log(f"huggingface: {repo}", "paso")
    if archivo:
        hf_hub_download(repo_id=repo, filename=archivo, repo_type="dataset",
                        local_dir=str(destino))
    else:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(destino))
    marcar(destino, repo)
    return destino


def bajar_gdown(ident: str, destino: Path) -> Path:
    try:
        import gdown
    except ImportError:
        raise RuntimeError("falta gdown:  pip install gdown")
    destino.mkdir(parents=True, exist_ok=True)
    if ya_esta(destino):
        log(f"gdrive:{ident} ya bajado", "ok")
        return destino
    log(f"gdown carpeta {ident}", "paso")
    gdown.download_folder(id=ident, output=str(destino), quiet=False,
                          use_cookies=False, remaining_ok=True)
    marcar(destino, ident)
    return destino


def bajar_git(url: str, destino: Path) -> Path:
    if ya_esta(destino):
        log(f"git:{url} ya clonado", "ok")
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        shutil.rmtree(destino, ignore_errors=True)
    log(f"git clone --depth 1 {url}", "paso")
    r = subprocess.run(["git", "clone", "--depth", "1", url, str(destino)],
                       text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git clone fallo:\n{r.stderr}")
    marcar(destino, url)
    return destino


def bajar_roboflow(ws: str, proyecto: str, destino: Path, api_key: str) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError:
        raise RuntimeError("falta roboflow:  pip install roboflow")
    destino.mkdir(parents=True, exist_ok=True)
    if ya_esta(destino):
        log(f"roboflow:{ws}/{proyecto} ya bajado", "ok")
        return destino
    log(f"roboflow {ws}/{proyecto}", "paso")
    rf = Roboflow(api_key=api_key)
    prj = rf.workspace(ws).project(proyecto)
    versiones = prj.versions()
    if not versiones:
        raise RuntimeError(f"{ws}/{proyecto} no tiene versiones publicadas")
    versiones[0].download("coco-segmentation", location=str(destino), overwrite=True)
    marcar(destino, f"{ws}/{proyecto}")
    return destino


def extraer(archivo: Path, destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    if ya_esta(destino):
        return destino
    log(f"extrayendo {archivo.name}", "paso")
    if zipfile.is_zipfile(archivo):
        with zipfile.ZipFile(archivo) as z:
            z.extractall(destino)
    elif tarfile.is_tarfile(archivo):
        with tarfile.open(archivo) as t:
            t.extractall(destino)
    else:
        raise RuntimeError(f"formato desconocido: {archivo}")
    marcar(destino, archivo.name)
    return destino


def extraer_zips_internos(raiz: Path, profundidad: int = 2) -> None:
    """Muchos datasets de Kaggle/HF traen un zip adentro del zip."""
    for _ in range(profundidad):
        zips = [z for z in raiz.rglob("*.zip") if not ya_esta(z.parent / z.stem)]
        if not zips:
            return
        for z in zips:
            try:
                extraer(z, z.parent / z.stem)
            except Exception as e:
                log(f"no se pudo abrir {z.name}: {e}", "warn")


# --------------------------------------------------------------------------
# Emparejado generico imagen <-> mascara
# --------------------------------------------------------------------------

def _normalizar(nombre: str) -> str:
    n = nombre.lower()
    for s in SUFIJOS_MASCARA:
        if n.endswith(s):
            n = n[: -len(s)]
            break
    return n


def _es_mascara(p: Path, pistas: Sequence[str]) -> bool:
    """Mascara si vive en una carpeta con nombre de mascara o si el nombre
    del archivo termina en uno de los sufijos tipicos (_mask, _gt, ...)."""
    if {x.lower() for x in p.parts[:-1]} & set(pistas):
        return True
    return _normalizar(p.stem) != p.stem.lower()


def emparejar(raiz: Path, pistas: Sequence[str] = PISTAS_MASCARA,
              excluir: Sequence[str] = ()) -> list[tuple[Path, Path]]:
    """Empareja imagen y mascara por nombre, sin asumir el layout del dataset.

    Es lo que hace que este script sobreviva a que un dataset de Kaggle
    reorganice sus carpetas: no hay rutas cableadas, solo nombres.
    """
    excl = tuple(x.lower() for x in excluir)
    imgs: dict[str, Path] = {}
    mascaras: dict[str, Path] = {}
    for p in raiz.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXT_IMG:
            continue
        rel = str(p.relative_to(raiz)).lower().replace("\\", "/")
        if excl and any(x in rel for x in excl):
            continue
        clave = _normalizar(p.stem)
        if _es_mascara(p, pistas):
            mascaras.setdefault(clave, p)
        else:
            imgs.setdefault(clave, p)
    # Un dataset con carpetas image/ y mask/ y nombres identicos cae en imgs
    # dos veces; el chequeo de carpeta ya lo separo arriba.
    pares = [(imgs[k], mascaras[k]) for k in imgs.keys() & mascaras.keys()]
    pares.sort()
    return pares


# --------------------------------------------------------------------------
# Conversores  ->  ceden (ruta_imagen, mascara_uint8, clases_anotadas)
# --------------------------------------------------------------------------

# La mascara viaja como funcion, no como array: un conversor puede ceder
# cientos de miles de muestras y materializarlas todas seria decenas de GB
# de RAM. Se construye recien dentro del worker que la va a guardar.
Muestra = tuple[Path, Callable[[], "np.ndarray"], tuple[int, ...]]


def leer_indices(p: Path) -> np.ndarray:
    """Valores CRUDOS de la mascara, sin pasar por la paleta.

    `Image.open(x).convert("L")` sobre un PNG indexado (modo P) devuelve la
    LUMINANCIA del color de la paleta, no el indice de clase: la clase 2 con
    paleta gris sale 170. COCO-Stuff, ADE20K y nuestras propias mascaras son
    todas modo P, asi que convertir a "L" corrompe cada etiqueta en silencio.
    """
    im = Image.open(p)
    if im.mode == "P":
        return np.asarray(im, dtype=np.uint8)
    return np.asarray(im.convert("L"), dtype=np.uint8)


def leer_binaria(p: Path) -> np.ndarray:
    """Mascara blanco/negro -> booleano. Aca SI conviene la luminancia:
    una fuente puede guardar el positivo como RGB blanco, como 255 en L o
    como indice 1 de una paleta de dos colores; los tres dan 255 al convertir.
    """
    return np.asarray(Image.open(p).convert("L"), dtype=np.uint8)


def _remapeo(ruta: Path, agua: set[int], ignorar: set[int]) -> Callable[[], np.ndarray]:
    """Closure que lee la mascara indexada y la remapea al esquema de Horus."""
    def hacer() -> np.ndarray:
        m = leer_indices(ruta)
        out = np.zeros(m.shape, np.uint8)
        out[np.isin(m, list(agua))] = 1
        out[np.isin(m, list(ignorar))] = IGNORAR
        return out
    return hacer


def conv_binaria(raiz: Path, f: Fuente, **kw) -> Iterator[Muestra]:
    """Mascara blanco/negro -> la clase positiva de la fuente."""
    positiva = [c for c in f.clases if c != 0]
    if len(positiva) != 1:
        raise RuntimeError(f"{f.nombre}: conv_binaria necesita exactamente una clase positiva")
    clase = positiva[0]
    pares = emparejar(raiz, excluir=kw.get("excluir", ()))
    if not pares:
        log(f"{f.nombre}: no se emparejo ninguna imagen con su mascara", "warn")
    def construir(ruta: Path):
        def hacer():
            m = leer_binaria(ruta)
            out = np.zeros(m.shape, np.uint8)
            out[m > 127] = clase
            return out
        return hacer

    for img, msk in pares:
        yield img, construir(msk), f.clases


def conv_cocostuff(raiz: Path, f: Fuente, **kw) -> Iterator[Muestra]:
    """PNG con valor = labelId-1 de labels.txt; 255 = sin etiquetar."""
    AGUA = {147, 154, 177}          # river, sea, water-other
    IGN = {119, 178, 255}           # fog (parece humo), waterdrops (lente), void
    # Los zips anidan distinto segun de donde salieron: se busca por
    # CONTENIDO (la carpeta que realmente tiene los .png / los .jpg) en vez
    # de cablear rutas, que es lo que se rompe cuando cambia el empaquetado.
    for split in ("train2017", "val2017"):
        d_m = next((d for d in raiz.rglob(split)
                    if d.is_dir() and "stuffthingmap" in str(d).lower()
                    and next(d.glob("*.png"), None)), None)
        d_i = next((d for d in raiz.rglob(split)
                    if d.is_dir() and next(d.glob("*.jpg"), None)), None)
        if not d_m or not d_i:
            log(f"{f.nombre}: no encontre {split} "
                f"({'faltan mascaras' if not d_m else 'faltan imagenes'})", "warn")
            continue
        for msk in sorted(d_m.glob("*.png")):
            img = d_i / (msk.stem + ".jpg")
            if not img.exists():
                continue
            yield img, _remapeo(msk, AGUA, IGN), f.clases


def conv_ade20k(raiz: Path, f: Fuente, **kw) -> Iterator[Muestra]:
    """PNG con valor = indice 1..150; 0 = sin etiquetar."""
    AGUA = {22, 27, 61, 110, 114, 129}   # water, sea, river, pileta, cascada, lago
    IGN = {0, 105}                       # sin etiquetar, fuente ornamental
    base = next((d for d in raiz.rglob("ADEChallengeData2016") if d.is_dir()), raiz)
    for split in ("training", "validation"):
        d_i = base / "images" / split
        d_m = base / "annotations" / split
        if not d_i.is_dir() or not d_m.is_dir():
            continue
        for msk in sorted(d_m.glob("*.png")):
            img = d_i / (msk.stem + ".jpg")
            if not img.exists():
                continue
            yield img, _remapeo(msk, AGUA, IGN), f.clases


def conv_atlantis(raiz: Path, f: Fuente, **kw) -> Iterator[Muestra]:
    """ATLANTIS: 56 etiquetas; las 18 naturales + 17 artificiales son agua."""
    # El repo numera 1..56 con los waterbodies primero; el resto son
    # etiquetas generales (cielo, persona, edificio...). Se colapsa todo
    # lo que sea cuerpo de agua a la clase 1 y lo demas a fondo.
    AGUA = set(range(1, 36))
    pares = emparejar(raiz)
    if not pares:
        log(f"{f.nombre}: no se encontraron imagenes -- bajalas a mano a "
            f"descargas_crudas/atlantis/ (ver {f.url})", "warn")
    for img, msk in pares:
        yield img, _remapeo(msk, AGUA, {255}), f.clases


def conv_floodnet(raiz: Path, f: Fuente, **kw) -> Iterator[Muestra]:
    """FloodNet: 0 fondo,1 edificio-inundado,3 calle-inundada,5 agua,8 pileta."""
    AGUA = {3, 5, 8}
    IGN = {1}      # estructura inundada: mitad agua, mitad edificio
    for img, msk in emparejar(raiz):
        yield img, _remapeo(msk, AGUA, IGN), f.clases


def conv_smoke100k(raiz: Path, f: Fuente, **kw) -> Iterator[Muestra]:
    """smoke_image/ + smoke_mask/ ; smoke_free_image/ es el mismo stem y sobra."""
    def construir(ruta: Path):
        def hacer():
            m = leer_binaria(ruta)
            out = np.zeros(m.shape, np.uint8)
            out[m > 127] = 2
            return out
        return hacer

    for img, msk in emparejar(raiz, excluir=("smoke_free",)):
        yield img, construir(msk), f.clases


def conv_coco_poligonos(raiz: Path, f: Fuente, **kw) -> Iterator[Muestra]:
    """Export COCO-segmentation de Roboflow -> mascara rasterizada."""
    try:
        from pycocotools import mask as cocomask  # noqa: F401
        from pycocotools.coco import COCO
    except ImportError:
        raise RuntimeError("falta pycocotools:  pip install pycocotools")
    from PIL import ImageDraw

    jsons = sorted(raiz.rglob("_annotations.coco.json"))
    if not jsons:
        jsons = sorted(raiz.rglob("*.json"))
    for j in jsons:
        try:
            coco = COCO(str(j))
        except Exception:
            continue
        # nombre de categoria -> clase de Horus
        mapa: dict[int, int] = {}
        for cid, cat in coco.cats.items():
            n = cat["name"].lower()
            if "smoke" in n or "humo" in n:
                mapa[cid] = 2
            elif "fire" in n or "flame" in n or "fuego" in n or "llama" in n:
                mapa[cid] = 3
            elif "water" in n or "flood" in n or "agua" in n:
                mapa[cid] = 1
        if not mapa:
            continue
        for iid, meta in coco.imgs.items():
            img = j.parent / meta["file_name"]
            if not img.exists():
                continue
            anns = coco.loadAnns(coco.getAnnIds(imgIds=iid))
            # fuego encima de humo: se dibuja el humo primero
            anns = sorted(anns, key=lambda a: mapa.get(a["category_id"], 0))

            def construir(anns=anns, w=meta["width"], h=meta["height"]):
                def hacer():
                    lienzo = Image.new("L", (w, h), 0)
                    dib = ImageDraw.Draw(lienzo)
                    for a in anns:
                        c = mapa.get(a["category_id"])
                        segs = a.get("segmentation")
                        if c is None or not segs or isinstance(segs, dict):
                            continue          # RLE: fuera de alcance
                        for poly in segs:
                            if len(poly) >= 6:
                                dib.polygon([(poly[i], poly[i + 1])
                                             for i in range(0, len(poly) - 1, 2)], fill=c)
                    return np.array(lienzo)
                return hacer

            yield img, construir(), f.clases


CONVERSORES: dict[str, Callable[..., Iterator[Muestra]]] = {
    "binaria": conv_binaria,
    "cocostuff": conv_cocostuff,
    "ade20k": conv_ade20k,
    "atlantis": conv_atlantis,
    "floodnet": conv_floodnet,
    "smoke100k": conv_smoke100k,
    "coco_poligonos": conv_coco_poligonos,
}


# --------------------------------------------------------------------------
# Halo de ignorar
# --------------------------------------------------------------------------

def aplicar_halo(m: np.ndarray, px: int) -> np.ndarray:
    """Marca como 255 un anillo de `px` alrededor de toda region positiva.

    Ahi es donde vive lo que la fuente no anoto: el humo pegado a la llama,
    la espuma del borde del agua, el error de trazado del anotador.
    """
    if px <= 0:
        return m
    pos = (m > 0) & (m != IGNORAR)
    if not pos.any():
        return m
    img = Image.fromarray((pos * 255).astype(np.uint8))
    k = 5
    for _ in range(max(1, round(px / (k // 2)))):
        img = img.filter(ImageFilter.MaxFilter(k))
    dilatada = np.array(img) > 127
    anillo = dilatada & ~pos & (m != IGNORAR)
    m = m.copy()
    m[anillo] = IGNORAR
    return m


# --------------------------------------------------------------------------
# Escritura del dataset final
# --------------------------------------------------------------------------

def es_val(stem: str, frac: float) -> bool:
    h = int(hashlib.md5(stem.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 1000) < frac * 1000


def guardar(muestra: Muestra, f: Fuente, salida: Path, idx: int,
            lado_max: int, val_frac: float, calidad: int) -> dict | None:
    img_p, hacer_mascara, anotadas = muestra
    try:
        m = hacer_mascara()
    except Exception:
        return None

    # Muestra sin ninguna clase positiva: es un negativo. Se descartan por
    # hash del nombre (estable entre corridas) segun `prob_negativos`, antes
    # de gastar el reencode del JPEG, que es la parte cara.
    if f.prob_negativos < 1.0 and not ((m > 0) & (m != IGNORAR)).any():
        h = int(hashlib.md5(img_p.stem.encode("utf-8")).hexdigest()[:8], 16)
        if (h % 10000) >= f.prob_negativos * 10000:
            return None

    try:
        img = Image.open(img_p).convert("RGB")
    except Exception:
        return None
    if img.size != (m.shape[1], m.shape[0]):
        m = np.array(Image.fromarray(m).resize(img.size, Image.NEAREST))

    lado = max(img.size)
    if lado > lado_max:
        esc = lado_max / lado
        nuevo = (max(1, round(img.width * esc)), max(1, round(img.height * esc)))
        img = img.resize(nuevo, Image.BILINEAR)
        m = np.array(Image.fromarray(m).resize(nuevo, Image.NEAREST))

    m = aplicar_halo(m, f.halo)

    # `sin_otras` amplia lo que la muestra declara saber: no cambia un solo
    # pixel de la mascara, cambia que el entrenamiento SI penalice predecir
    # humo o fuego sobre este fondo.
    if f.niega:
        anotadas = tuple(sorted(set(anotadas) | set(f.niega)))

    nombre = f"{f.nombre}_{idx:07d}"
    split = "val" if es_val(nombre, val_frac) else "train"
    d_i = salida / "imagenes" / split
    d_m = salida / "mascaras" / split
    d_i.mkdir(parents=True, exist_ok=True)
    d_m.mkdir(parents=True, exist_ok=True)

    ruta_i = d_i / f"{nombre}.jpg"
    ruta_m = d_m / f"{nombre}.png"
    img.save(ruta_i, "JPEG", quality=calidad, optimize=True)
    pm = Image.fromarray(m, mode="P")
    pm.putpalette(PALETA)
    pm.save(ruta_m, "PNG", optimize=True)

    cuentas = np.bincount(m.reshape(-1), minlength=256)
    return {
        "img": f"imagenes/{split}/{nombre}.jpg",
        "msk": f"mascaras/{split}/{nombre}.png",
        "fuente": f.nombre,
        "vista": f.vista,
        "split": split,
        "anotadas": list(anotadas),
        "w": img.width,
        "h": img.height,
        "px": {str(c): int(cuentas[c]) for c in range(N_CLASES) if cuentas[c]},
        "px_ign": int(cuentas[IGNORAR]),
    }


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------

def obtener(f: Fuente, crudos: Path, args) -> Path | None:
    d = crudos / f.nombre
    try:
        if f.metodo == "http":
            d.mkdir(parents=True, exist_ok=True)
            for url, nombre in f.recursos:
                arch = bajar_http(url, d / nombre)
                extraer(arch, d / Path(nombre).stem)
        elif f.metodo == "kaggle":
            bajar_kaggle(f.recursos[0], d)
        elif f.metodo == "hf":
            repo, archivo = f.recursos[0]
            bajar_hf(repo, archivo, d)
        elif f.metodo == "gdown":
            bajar_gdown(f.recursos[0], d)
        elif f.metodo == "git":
            bajar_git(f.recursos[0], d)
        elif f.metodo == "roboflow":
            ws, prj = f.recursos[0]
            bajar_roboflow(ws, prj, d, args.roboflow_key)
        elif f.metodo == "manual":
            if not d.exists() or not any(d.rglob("*")):
                log(f"{f.nombre}: pendiente de descarga manual -> {f.url}", "warn")
                log(f"   descomprimilo en {d} y volve a correr el script")
                return None
        extraer_zips_internos(d)
        return d
    except Exception as e:
        log(f"{f.nombre}: {e}", "err")
        return None


def procesar(f: Fuente, crudo: Path, salida: Path, args) -> list[dict]:
    conv = CONVERSORES[f.conversor]
    tope = f.tope if args.tope_fuente is None else min(f.tope or 10 ** 9, args.tope_fuente)

    # Las muestras son (ruta, closure, clases): livianas. Se puede juntar
    # todas y recien despues muestrear, sin cargar una sola mascara.
    muestras = list(conv(crudo, f, **f.args_conv))
    if not muestras:
        log(f"{f.nombre}: el conversor no produjo ninguna muestra", "warn")
        return []
    if tope and len(muestras) > tope:
        # Muestreo aleatorio con semilla fija en vez de cortar por orden
        # alfabetico, que sesgaria el subconjunto a una sola escena.
        random.Random(1234).shuffle(muestras)
        muestras = muestras[:tope]
        log(f"{f.nombre}: topeado a {tope} muestras", "warn")

    log(f"{f.nombre}: convirtiendo {len(muestras)} muestras", "paso")
    filas: list[dict] = []
    hechas = 0
    TANDA = 2000                      # acota la cola de futuros en memoria
    with ThreadPoolExecutor(max_workers=args.hilos) as ex:
        for i0 in range(0, len(muestras), TANDA):
            tanda = muestras[i0:i0 + TANDA]
            futs = [ex.submit(guardar, m, f, salida, i0 + j, args.lado_max,
                              args.val_frac, args.calidad)
                    for j, m in enumerate(tanda)]
            for fut in as_completed(futs):
                fila = fut.result()
                if fila:
                    filas.append(fila)
                hechas += 1
                if hechas % 500 == 0:
                    print(f"\r      {f.nombre}: {hechas}/{len(muestras)}  "
                          f"guardadas {len(filas)}", end="", flush=True)
    print()
    descartadas = len(muestras) - len(filas)
    if descartadas:
        log(f"{f.nombre}: {descartadas} descartadas "
            f"(negativos filtrados o archivos ilegibles)")
    return filas


def escribir_reporte(filas: list[dict], salida: Path) -> str:
    por_fuente: dict[str, int] = {}
    por_clase = np.zeros(N_CLASES, dtype=np.int64)
    imgs_con_clase = np.zeros(N_CLASES, dtype=np.int64)
    splits: dict[str, int] = {}
    for r in filas:
        por_fuente[r["fuente"]] = por_fuente.get(r["fuente"], 0) + 1
        splits[r["split"]] = splits.get(r["split"], 0) + 1
        for c, n in r["px"].items():
            por_clase[int(c)] += n
            imgs_con_clase[int(c)] += 1
    total_px = max(int(por_clase.sum()), 1)

    L = ["# mezcla_seg_v1 -- reporte de armado", "",
         f"Muestras: **{len(filas)}**  (train {splits.get('train', 0)} / "
         f"val {splits.get('val', 0)})", "",
         "## Pixeles por clase", "", "| clase | pixeles | % | imagenes que la contienen |",
         "|---|---:|---:|---:|"]
    for c in range(N_CLASES):
        L.append(f"| {c} {CLASES[c]} | {int(por_clase[c]):,} | "
                 f"{100 * por_clase[c] / total_px:.2f}% | {int(imgs_con_clase[c]):,} |")
    L += ["", "## Muestras por fuente", "", "| fuente | muestras |", "|---|---:|"]
    for k, v in sorted(por_fuente.items(), key=lambda x: -x[1]):
        L.append(f"| {k} | {v:,} |")

    L += ["", "## Avisos", ""]
    avisos = []
    for c in range(1, N_CLASES):
        if imgs_con_clase[c] == 0:
            avisos.append(f"- **{CLASES[c]} no tiene ni una imagen.** El "
                          f"entrenamiento va a abortar: falta una fuente de esa clase.")
        elif imgs_con_clase[c] < 200:
            avisos.append(f"- {CLASES[c]}: solo {int(imgs_con_clase[c])} imagenes. "
                          f"Es poco; sumale otra fuente.")
    vivos = [por_clase[c] for c in range(1, N_CLASES) if por_clase[c] > 0]
    if len(vivos) > 1 and max(vivos) / max(min(vivos), 1) > 20:
        avisos.append(f"- Desbalance de pixeles >20x entre clases positivas. "
                      f"`entrenar_segmentacion_cuda.py` lo compensa con pesos, "
                      f"pero conviene topear la fuente dominante con --tope-fuente.")
    L += avisos or ["- Ninguno."]
    txt = "\n".join(L) + "\n"
    (salida / "reporte.md").write_text(txt, encoding="utf-8")
    return txt


def verificar(salida: Path) -> int:
    """Relee el dataset armado y confirma que nada quedo roto."""
    man = salida / "manifiesto.jsonl"
    if not man.exists():
        log("no hay manifiesto: corre el armado primero", "err")
        return 1
    filas = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
    problemas = 0
    vistas = set()
    for r in filas:
        pi, pm = salida / r["img"], salida / r["msk"]
        if not pi.exists() or not pm.exists():
            log(f"falta archivo de {r['img']}", "err")
            problemas += 1
            continue
        if r["img"] in vistas:
            log(f"duplicado {r['img']}", "err")
            problemas += 1
        vistas.add(r["img"])
    log(f"revisando {len(filas)} muestras (muestreo de pixeles sobre 300)", "paso")
    for r in random.Random(0).sample(filas, min(300, len(filas))):
        m = leer_indices(salida / r["msk"])
        vals = set(np.unique(m).tolist()) - {IGNORAR}
        if not vals <= set(range(N_CLASES)):
            log(f"{r['msk']} tiene valores fuera de 0..{N_CLASES - 1}: {sorted(vals)}", "err")
            problemas += 1
        fuera = vals - set(r["anotadas"])
        if fuera:
            log(f"{r['msk']} usa clases {sorted(fuera)} que su fuente no declara", "err")
            problemas += 1
        img = Image.open(salida / r["img"])
        if img.size != (m.shape[1], m.shape[0]):
            log(f"{r['img']} y su mascara no coinciden en tamano", "err")
            problemas += 1
    if problemas:
        log(f"{problemas} problemas encontrados", "err")
    else:
        log("dataset consistente", "ok")
    return 1 if problemas else 0


DEPS = ["kaggle", "huggingface_hub", "gdown", "roboflow", "pycocotools"]


def instalar_deps() -> int:
    """Instala las dependencias opcionales con el MISMO interprete que corre
    este script. `pip install` a secas puede ir a parar a otro Python."""
    log(f"instalando con {sys.executable}", "paso")
    r = subprocess.run([sys.executable, "-m", "pip", "install", *DEPS])
    if r.returncode:
        log("fallo la instalacion; proba a mano:", "err")
        log(f"  {sys.executable} -m pip install {' '.join(DEPS)}")
    else:
        log("dependencias listas", "ok")
    return r.returncode


def _hay(modulo: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(modulo) is not None


def _kaggle_ok() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def estado(crudos: Path, salida: Path, args) -> int:
    """Que fuente esta lista para bajarse, cual ya se bajo y cual ya se convirtio.

    Se corre ANTES de comprometer horas de descarga: es mucho mejor enterarse
    aca de que falta la key de Roboflow que a los 40 GB.
    """
    convertidas: set[str] = set()
    man = salida / "manifiesto.jsonl"
    if man.exists():
        convertidas = {json.loads(l)["fuente"]
                       for l in man.read_text(encoding="utf-8").splitlines() if l.strip()}

    faltan: list[str] = []
    filas = []
    for f in FUENTES:
        d = crudos / f.nombre
        bajada = ya_esta(d) or any(
            ya_esta(d / Path(n).stem) for _, n in (f.recursos if f.metodo == "http" else ()))
        if f.metodo == "manual":
            bajada = d.exists() and any(d.rglob("*"))

        if f.metodo == "kaggle":
            listo, motivo = _kaggle_ok(), "falta ~/.kaggle/kaggle.json"
        elif f.metodo == "roboflow":
            listo = bool(args.roboflow_key) and _hay("roboflow") and _hay("pycocotools")
            motivo = ("falta --roboflow-key" if not args.roboflow_key
                      else "falta pip install roboflow pycocotools")
        elif f.metodo == "hf":
            listo, motivo = _hay("huggingface_hub"), "falta pip install huggingface_hub"
        elif f.metodo == "gdown":
            listo, motivo = _hay("gdown"), "falta pip install gdown"
        elif f.metodo == "git":
            listo, motivo = shutil.which("git") is not None, "falta git en el PATH"
        elif f.metodo == "manual":
            listo, motivo = bajada, f"bajalo a mano: {f.url}"
        else:
            listo, motivo = True, ""

        if not listo and motivo not in faltan:
            faltan.append(motivo)
        marca = ("convertida" if f.nombre in convertidas else
                 "bajada" if bajada else
                 "lista" if listo else "NO")
        filas.append((f, marca, "" if listo or bajada else motivo))

    print()
    print(f"{'fuente':<20} {'clases':<16} {'metodo':<10} {'GB~':>6}  {'estado':<11} nota")
    print("-" * 100)
    pendiente = 0.0
    for f, marca, nota in filas:
        cl = "+".join(CLASES[c] for c in f.clases if c != 0)
        print(f"{f.nombre:<20} {cl:<16} {f.metodo:<10} {f.gb:>6.1f}  {marca:<11} {nota}")
        if marca in ("lista", "bajada"):
            pendiente += f.gb
    print("-" * 100)
    print(f"Convertidas: {len(convertidas)}   "
          f"Por procesar: ~{pendiente:.0f} GB   "
          f"Libre en disco: {libre_gb(crudos):.0f} GB")

    if faltan:
        print("\nPara sumar las que faltan:")
        for m in faltan:
            print(f"  - {m}")
    for c in range(1, N_CLASES):
        if not any(c in f.clases for f, m, _ in filas if m in ("lista", "bajada", "convertida")):
            log(f"NINGUNA fuente de {CLASES[c]} esta lista: el entrenamiento "
                f"va a abortar", "err")
    return 0


def reanotar(salida: Path, modo: str = "fuego") -> int:
    """Reescribe `anotadas` en el manifiesto segun el registro actual.

    No toca ni una imagen ni una mascara: `anotadas` no describe los pixeles,
    describe que sabe cada fuente. Cambiarla es de segundos y no obliga a
    rehacer horas de conversion.
    """
    man = salida / "manifiesto.jsonl"
    if not man.exists():
        log(f"no existe {man}", "err")
        return 1
    filas = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
    # `solo-agua` deja afuera a los datasets de escenas generales: sus nubes,
    # niebla y vapor son visualmente humo, y declararlos "seguro no es humo" le
    # pide al modelo una distincion que a 384px no esta ahi.
    permitido = {"ninguno": set(), "fuego": {3}, "humo+fuego": {2, 3}}[modo]
    cambios: dict[str, tuple] = {}
    for r in filas:
        f = POR_NOMBRE.get(r["fuente"])
        if f is None:
            continue
        nuevas = sorted(set(f.clases) | (set(f.niega) & permitido))
        if nuevas != r["anotadas"]:
            cambios.setdefault(r["fuente"], (tuple(r["anotadas"]), tuple(nuevas)))
            r["anotadas"] = nuevas
    if not cambios:
        log("nada que cambiar: el manifiesto ya coincide con el registro", "ok")
        return 0
    with open(man, "w", encoding="utf-8") as fh:
        for r in filas:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for nom, (viejo, nuevo) in sorted(cambios.items()):
        v = "+".join(CLASES[c] for c in viejo)
        n = "+".join(CLASES[c] for c in nuevo)
        log(f"{nom}: {v}  ->  {n}", "ok")
    log(f"{len(cambios)} fuentes reanotadas sobre {len(filas)} muestras. "
        f"Volve a entrenar desde cero (no reanudes: cambio la supervision).", "warn")
    return 0


def tabla_fuentes() -> str:
    L = ["", f"{'fuente':<20} {'clases':<16} {'metodo':<10} {'vista':<10} "
             f"{'GB~':>6}  resumen", "-" * 118]
    for f in FUENTES:
        cl = "+".join(CLASES[c] for c in f.clases if c != 0)
        L.append(f"{f.nombre:<20} {cl:<16} {f.metodo:<10} {f.vista:<10} "
                 f"{f.gb:>6.1f}  {f.resumen}")
    L.append("-" * 118)
    L.append(f"{'TOTAL':<20} {'':<16} {'':<10} {'':<10} {sum(f.gb for f in FUENTES):>6.1f}")
    return "\n".join(L)


def seleccionar(args) -> list[Fuente]:
    sel = list(FUENTES)
    if args.solo:
        pedidas = [x.strip() for x in args.solo.split(",") if x.strip()]
        desconocidas = [p for p in pedidas if p not in POR_NOMBRE]
        if desconocidas:
            sys.exit(f"fuentes desconocidas: {desconocidas}\n{tabla_fuentes()}")
        return [POR_NOMBRE[p] for p in pedidas]
    if args.saltar:
        fuera = {x.strip() for x in args.saltar.split(",")}
        sel = [f for f in sel if f.nombre not in fuera]
    if not args.incluir_aereo:
        sel = [f for f in sel if f.vista != "aereo"]
    if not args.kaggle:
        sel = [f for f in sel if f.metodo != "kaggle"]
    if not args.hf:
        sel = [f for f in sel if f.metodo != "hf"]
    if not args.gdown:
        sel = [f for f in sel if f.metodo != "gdown"]
    if not args.roboflow_key:
        sel = [f for f in sel if f.metodo != "roboflow"]
    return sel


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Arma mezcla_seg_v1 (fondo/agua/humo/fuego) para Horus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=tabla_fuentes())
    ap.add_argument("--raiz", type=Path, default=Path(__file__).resolve().parent,
                    help="carpeta de 04_cabezas/segmentacion (default: la de este script)")
    ap.add_argument("--salida", default="datasets/mezcla_seg_v1")
    ap.add_argument("--crudos", default="descargas_crudas")
    ap.add_argument("--listar", action="store_true", help="muestra las fuentes y sale")
    ap.add_argument("--estado", action="store_true",
                    help="que fuente esta lista, cual ya se bajo y cual se convirtio")
    ap.add_argument("--instalar-deps", action="store_true",
                    help="instala kaggle/huggingface_hub/gdown/roboflow/pycocotools")
    ap.add_argument("--verificar", action="store_true", help="revisa el dataset ya armado")
    ap.add_argument("--reanotar", action="store_true",
                    help="reescribe `anotadas` en el manifiesto segun el registro "
                         "actual, sin rehacer imagenes")
    ap.add_argument("--negativos", choices=["fuego", "humo+fuego", "ninguno"],
                    default="fuego",
                    help="con --reanotar: que puede negar el fondo de las fuentes de "
                         "agua. 'fuego' (medido como el mejor), 'humo+fuego' (hunde "
                         "el humo), 'ninguno' (esquema original)")
    ap.add_argument("--solo", help="lista de fuentes separadas por coma")
    ap.add_argument("--saltar", help="lista de fuentes a excluir")
    ap.add_argument("--todo", action="store_true", help="atajo: --kaggle --hf --gdown --incluir-aereo")
    ap.add_argument("--kaggle", action="store_true", help="usa ~/.kaggle/kaggle.json")
    ap.add_argument("--hf", action="store_true", help="usa huggingface_hub")
    ap.add_argument("--gdown", action="store_true", help="usa gdown para Google Drive")
    ap.add_argument("--roboflow-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    ap.add_argument("--incluir-aereo", action="store_true",
                    help="suma fuentes de dron (otra geometria que una camara fija)")
    ap.add_argument("--lado-max", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--calidad", type=int, default=92)
    ap.add_argument("--tope-fuente", type=int, default=None,
                    help="maximo global de muestras por fuente")
    ap.add_argument("--hilos", type=int, default=max(4, (os.cpu_count() or 8)))
    ap.add_argument("--solo-descarga", action="store_true")
    ap.add_argument("--solo-convertir", action="store_true")
    ap.add_argument("--purgar-crudos", action="store_true",
                    help="borra zips y extraidos al terminar la conversion")
    args = ap.parse_args()

    if args.todo:
        args.kaggle = args.hf = args.gdown = args.incluir_aereo = True

    if args.instalar_deps:
        return instalar_deps()

    if args.listar:
        print(tabla_fuentes())
        return 0

    raiz = args.raiz.resolve()
    salida = raiz / args.salida
    crudos = raiz / args.crudos
    salida.mkdir(parents=True, exist_ok=True)
    crudos.mkdir(parents=True, exist_ok=True)

    if args.estado:
        return estado(crudos, salida, args)

    if args.reanotar:
        return reanotar(salida, args.negativos)

    if args.verificar:
        return verificar(salida)

    sel = seleccionar(args)
    if not sel:
        sys.exit("no quedo ninguna fuente seleccionada")

    necesario = sum(f.gb for f in sel) * 1.6   # crudo + convertido
    libre = libre_gb(raiz)
    log(f"{len(sel)} fuentes  ~{necesario:.0f} GB necesarios  {libre:.0f} GB libres",
        "warn" if libre < necesario else "ok")
    if libre < necesario:
        log("puede no entrar; usa --purgar-crudos o corre por tandas con --solo", "warn")

    filas: list[dict] = []
    man = salida / "manifiesto.jsonl"
    hechas: set[str] = set()
    if man.exists():
        filas = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
        hechas = {r["fuente"] for r in filas}
        log(f"manifiesto previo: {len(filas)} muestras de {len(hechas)} fuentes", "ok")

    for f in sel:
        print()
        log(f"=== {f.nombre} :: {f.resumen}", "paso")
        if f.nombre in hechas and not args.solo:
            log("ya convertida (borra sus lineas del manifiesto para rehacerla)", "ok")
            continue
        crudo = None if args.solo_convertir else obtener(f, crudos, args)
        if args.solo_convertir:
            crudo = crudos / f.nombre
        if crudo is None or not crudo.exists():
            continue
        if args.solo_descarga:
            continue
        try:
            nuevas = procesar(f, crudo, salida, args)
        except Exception as e:
            log(f"{f.nombre}: conversion fallo: {e}", "err")
            continue
        filas = [r for r in filas if r["fuente"] != f.nombre] + nuevas
        with open(man, "w", encoding="utf-8") as fh:
            for r in filas:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"{f.nombre}: {len(nuevas)} muestras", "ok")
        if args.purgar_crudos:
            shutil.rmtree(crudo, ignore_errors=True)
            centinela(crudo).unlink(missing_ok=True)
            log(f"{f.nombre}: crudos borrados", "ok")

    if args.solo_descarga:
        log("descarga terminada (--solo-descarga)", "ok")
        return 0

    (salida / "clases.json").write_text(json.dumps({
        "clases": CLASES,
        "ignorar": IGNORAR,
        "nota": ("`anotadas` lista las clases que la fuente de cada muestra sabe "
                 "etiquetar. El entrenamiento aplica softmax restringido a ese "
                 "subconjunto: asi el fondo de una foto de incendio no ensena "
                 "'aca no hay agua' cuando la fuente nunca miro el agua."),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(escribir_reporte(filas, salida))
    log(f"dataset en {salida}", "ok")
    log("revisalo con:  python descargar_datasets_segmentacion.py --verificar", "ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
