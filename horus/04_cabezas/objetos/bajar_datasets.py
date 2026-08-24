# -*- coding: utf-8 -*-
"""
bajar_datasets.py · HORUS — baja las fuentes públicas y arma datasets/mezcla_v1.

Hace tres cosas, y se pueden correr por separado:

  1. DESCARGAR   cada fuente a datasets/_crudo/<fuente>/
  2. NORMALIZAR  cada fuente a YOLO con TUS índices (0..6) en datasets/_normalizado/
  3. ARMAR       deduplicar + balancear + partir train/val -> datasets/mezcla_v1/

Uso típico (todo de una):
    python bajar_datasets.py --todo

Por partes (recomendado la primera vez, son decenas de GB):
    python bajar_datasets.py --listar
    python bajar_datasets.py --descargar d-fire pyro-sdis
    python bajar_datasets.py --armar

Las fuentes que necesitan credencial se saltean solas y te dicen qué falta:
    Roboflow  -> https://app.roboflow.com/settings/api  (export ROBOFLOW_API_KEY=...)
    Kaggle    -> https://www.kaggle.com/settings  (~/.kaggle/kaggle.json)

Dependencias (se instalan solo las de las fuentes que uses):
    pip install datasets huggingface_hub    # pyro-sdis
    pip install roboflow                    # fuentes de Roboflow
    pip install fiftyone                    # Open Images
    pip install kaggle                      # mirror de D-Fire

Todo lo que baja va a datasets/_crudo/ y datasets/_normalizado/, que están en
.gitignore. El resultado final es datasets/mezcla_v1/.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_AQUI = Path(__file__).resolve().parent
DIR_DATASETS = _AQUI / "datasets"
DIR_CRUDO = DIR_DATASETS / "_crudo"
DIR_NORM = DIR_DATASETS / "_normalizado"
DIR_SALIDA = DIR_DATASETS / "mezcla_v1"

CLASES = ("humo", "llama", "persona", "pistola", "cuchillo", "celular", "paquete")
EXT_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Cuántas cajas por clase queremos como máximo en la mezcla final. El objetivo
# es balance: 15.000 personas contra 2.000 cuchillos sesga la red.
TOPE_POR_CLASE = 5000
TOPE_NEGATIVOS = 4000
FRACCION_VAL = 0.15


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(msg, flush=True)


def _importable(mod: str) -> bool:
    """¿Está instalado el módulo? SIN importarlo.

    Importante que no lo importe: el paquete `kaggle` intenta autenticarse al
    importarse y escupe todo el instructivo de login (o directamente revienta
    si no hay credencial). find_spec solo mira si el módulo existe en disco.
    De paso, no cargar fiftyone hace que el chequeo sea instantáneo."""
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _clave_roboflow() -> str:
    """Lee la key limpiando comillas y espacios: copiar/pegar suele arrastrarlos
    y Roboflow devuelve un 401 críptico en vez de decir 'te sobra una comilla'."""
    return os.environ.get("ROBOFLOW_API_KEY", "").strip().strip("\'\"").strip()


def _pista_clave(k: str) -> str:
    """Muestra la key sin exponerla entera."""
    if not k:
        return "(vacía)"
    if len(k) <= 8:
        return f"{k!r} — solo {len(k)} caracteres, es muy corta"
    return f"{k[:4]}...{k[-3:]} ({len(k)} caracteres)"


def _clave_sospechosa(k: str) -> bool:
    obvias = {"tu_clave_aca", "tu_clave", "your_api_key", "xxx", "clave"}
    return k.lower() in obvias or len(k) < 12 or " " in k


def _imagenes(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in EXT_IMG)


def _leer_yolo(txt: Path) -> List[Tuple[int, float, float, float, float]]:
    filas = []
    if not txt.exists():
        return filas
    for linea in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = linea.split()
        if len(p) != 5:
            continue
        try:
            filas.append((int(p[0]), *(float(v) for v in p[1:])))
        except ValueError:
            continue
    return filas


def _escribir_yolo(txt: Path, filas: Sequence[Tuple[int, float, float, float, float]]) -> None:
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text("".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}\n"
                           for c, a, b, w, h in filas), encoding="utf-8")


def _sano(fila) -> bool:
    """Descarta cajas degeneradas o fuera de 0..1 (el chequeo del entrenador
    aborta con esas, así que las filtramos acá)."""
    _, cx, cy, w, h = fila
    if w <= 0 or h <= 0:
        return False
    return all(-0.001 <= v <= 1.001 for v in (cx, cy, w, h))


def _clip(fila):
    c, cx, cy, w, h = fila
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    return (c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def dhash(ruta: Path, tam: int = 8) -> Optional[int]:
    """Hash perceptual de diferencias. Sirve para detectar la misma foto
    reempaquetada en dos datasets distintos (pasa MUCHO: varios proyectos de
    Roboflow son re-subidas de D-Fire)."""
    import cv2
    img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (tam + 1, tam), interpolation=cv2.INTER_AREA)
    diff = img[:, 1:] > img[:, :-1]
    return int(sum(1 << i for i, v in enumerate(diff.flatten()) if v))


# --------------------------------------------------------------------------- #
# Definición de fuentes
# --------------------------------------------------------------------------- #
@dataclass
class Fuente:
    clave: str
    nombre: str
    licencia: str
    comercial: bool
    url: str
    requiere: str                      # "" | "roboflow" | "kaggle" | "manual"
    descargar: Callable[[Path], bool]
    normalizar: Callable[[Path, Path], int]
    nota: str = ""
    clases: Tuple[int, ...] = ()       # qué clases nuestras aporta
    gb: float = 1.0                    # tamaño aproximado en disco
    deps: Tuple[str, ...] = ()         # módulos pip que necesita


# ---------- descargadores -------------------------------------------------- #
def _dl_dfire(dest: Path) -> bool:
    """D-Fire: los links oficiales son de OneDrive (no scriptables). Usamos el
    mirror de Kaggle, que es el mismo dato en formato YOLO."""
    if any(dest.rglob("*.txt")):
        log("  ya está descargado")
        return True
    if not _importable("kaggle"):
        log("  ⚠ falta el CLI de kaggle: pip install kaggle")
        log("    o bajalo a mano de https://github.com/gaiasd/DFireDataset")
        log(f"    y descomprimilo en {dest}")
        return False
    dest.mkdir(parents=True, exist_ok=True)

    slug = "sayedgamal99/smoke-fire-detection-yolo"
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
           "-d", slug, "-p", str(dest), "--unzip"]
    log(f"  bajando {slug} (~4 GB, puede tardar)")
    log(f"  $ {' '.join(cmd[1:])}")
    # SIN capture_output: así ves la barra de progreso del CLI en vivo. Si se
    # captura, una descarga de 4 GB parece un cuelgue y un error queda mudo.
    r = subprocess.run(cmd)

    n_img = len(_imagenes(dest))
    n_lbl = len(list(dest.rglob("*.txt")))
    if r.returncode != 0 or n_img == 0:
        log("")
        log(f"  ⚠ NO se descargó nada (código {r.returncode}, "
            f"{n_img} imágenes en disco)")
        log("    Cosas para chequear, en orden:")
        log("      1. ¿Aceptaste las reglas del dataset? Entrá una vez a")
        log(f"         https://www.kaggle.com/datasets/{slug}")
        log("         y hacé clic en Download; algunos datasets lo exigen.")
        log("      2. Probá el comando a mano para ver el error completo:")
        log(f"         {' '.join(cmd[1:])}")
        log("      3. Si Kaggle no coopera, bajalo del origen:")
        log("         https://github.com/gaiasd/DFireDataset")
        log(f"         y descomprimilo en {dest}")
        return False

    log(f"  OK · {n_img} imágenes y {n_lbl} etiquetas en {dest.name}")
    return True


def _dl_pyro(dest: Path) -> bool:
    """Pyro-SDIS desde HuggingFace. El repo trae las imágenes y las etiquetas
    YOLO como columnas del dataset, hay que materializarlas a disco."""
    if (dest / "listo.flag").exists():
        log("  ya está descargado")
        return True
    if not _importable("datasets"):
        log("  ⚠ falta: pip install datasets")
        return False
    from datasets import load_dataset

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "images").mkdir(exist_ok=True)
    (dest / "labels").mkdir(exist_ok=True)
    ds = load_dataset("pyronear/pyro-sdis")
    total = 0
    for split in ds:
        for ej in ds[split]:
            nombre = Path(str(ej.get("image_name") or f"{total}.jpg")).stem
            try:
                ej["image"].convert("RGB").save(dest / "images" / f"{nombre}.jpg",
                                                quality=92)
            except Exception:
                continue
            (dest / "labels" / f"{nombre}.txt").write_text(
                str(ej.get("annotations") or ""), encoding="utf-8")
            total += 1
            if total % 2000 == 0:
                log(f"    {total} imágenes...")
    (dest / "listo.flag").write_text(str(total))
    log(f"  {total} imágenes materializadas")
    return True


def _dl_roboflow(ws: str, proy: str, ver: int):
    def f(dest: Path) -> bool:
        if any(dest.rglob("*.txt")):
            log("  ya está descargado")
            return True
        key = _clave_roboflow()
        if not key:
            log("  ⚠ falta ROBOFLOW_API_KEY (https://app.roboflow.com/settings/api)")
            return False
        if not _importable("roboflow"):
            log("  ⚠ falta: pip install roboflow")
            return False
        from roboflow import Roboflow
        dest.mkdir(parents=True, exist_ok=True)
        try:
            p = Roboflow(api_key=key).workspace(ws).project(proy)
            p.version(ver).download("yolov8", location=str(dest))
            return True
        except Exception as e:
            msg = str(e)
            log(f"  ⚠ falló: {msg[:160]}")
            if "does not exist" in msg or "401" in msg or "revoked" in msg:
                log("")
                log("    La key llegó a Roboflow pero la rechazó. Chequeá, en orden:")
                log(f"      1. ¿Qué guardaste?  ->  {_pista_clave(key)}")
                log("      2. Tiene que ser la 'Private API Key' del workspace,")
                log("         NO la 'Publishable Key'. Está en")
                log("         https://app.roboflow.com/settings/api")
                log("      3. Si usaste setx, cerrá y reabrí PowerShell.")
            log(f"    O bajalo a mano de https://universe.roboflow.com/{ws}/{proy}")
            return False
    return f


def _dl_openimages(mapa: Dict[str, int], cupo: int):
    """Baja Open Images UNA CLASE POR VEZ (para controlar el cupo), pero en
    cada imagen CONSERVA las cajas de TODAS nuestras clases.

    Esto último es la parte que importa. Si al bajar 'Knife' se tiran las cajas
    de 'Person' que hay en esas mismas fotos, el dataset termina con miles de
    imágenes llenas de gente que el .txt declara vacías. Cada vez que el modelo
    detecta bien a esa persona, la pérdida lo castiga como falso positivo: se
    le está enseñando a NO ver personas. Es la razón por la que 'persona' puede
    rendir peor que una clase con diez veces menos datos.

    Además filtra dos banderas que arruinan el entrenamiento:
      IsGroupOf=1   -> la caja rodea un GRUPO, no un objeto. Convertirla en
                       'persona' enseña a dibujar una caja gigante sobre
                       cualquier multitud.
      IsDepiction=1 -> estatua, dibujo, póster. Falsos positivos sobre la
                       publicidad de una marquesina.
    """
    nombres = list(mapa.keys())          # el ORDEN define el índice exportado

    def f(dest: Path) -> bool:
        if not _importable("fiftyone"):
            log("  ⚠ falta: pip install fiftyone")
            return False
        import fiftyone as fo
        import fiftyone.zoo as foz
        from fiftyone import ViewField as F

        dest.mkdir(parents=True, exist_ok=True)

        # Restos del formato viejo (una sola bajada mezclada en dest/images).
        if (dest / "images").is_dir():
            log("  encontré un export viejo. Lo borro y rehago por clase.")
            for viejo_n in ("images", "labels", "listo.flag", "dataset.yaml",
                            "data.yaml"):
                q = dest / viejo_n
                if q.is_dir():
                    shutil.rmtree(q, ignore_errors=True)
                elif q.exists():
                    q.unlink()

        ok_alguna = False
        for nombre_oi, clase in mapa.items():
            sub = dest / f"clase{clase}_{nombre_oi.replace(' ', '_')}"
            if (sub / "listo.flag").exists():
                log(f"    {nombre_oi}: ya está")
                ok_alguna = True
                continue
            log(f"    {nombre_oi} (hasta {cupo} imgs, conservando todas "
                f"nuestras clases)...")
            try:
                ds = foz.load_zoo_dataset(
                    "open-images-v7", split="train", label_types=["detections"],
                    classes=[nombre_oi], max_samples=cupo, shuffle=True, seed=51,
                    dataset_name=f"oi_{nombre_oi.replace(' ', '_')}_{cupo}",
                )
                campo = ("ground_truth" if "ground_truth" in ds.get_field_schema()
                         else "detections")
                # fiftyone baja las imágenes que CONTIENEN la clase pedida, pero
                # carga TODAS sus etiquetas. Antes filtrábamos a una sola clase;
                # ahora nos quedamos con las 6 nuestras que estén presentes.
                vista = ds.filter_labels(
                    campo,
                    F("label").is_in(nombres)
                    & (F("IsGroupOf") != True)      # noqa: E712
                    & (F("IsDepiction") != True),   # noqa: E712
                )
                sub.mkdir(parents=True, exist_ok=True)
                # classes=nombres (la lista COMPLETA) -> el índice del .txt es
                # el mismo en todas las subcarpetas.
                vista.export(export_dir=str(sub),
                             dataset_type=fo.types.YOLOv5Dataset,
                             label_field=campo, split="train", classes=nombres)
                (sub / "listo.flag").write_text(str(clase))
                ok_alguna = True
            except Exception as e:
                log(f"    ⚠ {nombre_oi} falló: {str(e)[:150]}")
        return ok_alguna
    return f


def _hacer_norm_openimages(mapa: Dict[str, int]):
    """El índice del .txt sigue el orden de `mapa`, igual en toda subcarpeta."""
    remapeo = {i: clase for i, clase in enumerate(mapa.values())}

    def f(crudo: Path, salida: Path) -> int:
        total = 0
        for sub in sorted(crudo.iterdir()):
            if not sub.is_dir() or not sub.name.startswith("clase"):
                continue
            total += _norm_yolo(remapeo)(sub, salida / sub.name)
        # juntar las subcarpetas en un solo images/labels
        (salida / "images").mkdir(parents=True, exist_ok=True)
        (salida / "labels").mkdir(parents=True, exist_ok=True)
        for sub in sorted(salida.iterdir()):
            if not sub.is_dir() or sub.name in ("images", "labels"):
                continue
            for img in _imagenes(sub / "images"):
                shutil.move(str(img), salida / "images" / img.name)
                t = sub / "labels" / (img.stem + ".txt")
                if t.exists():
                    shutil.move(str(t), salida / "labels" / t.name)
            shutil.rmtree(sub, ignore_errors=True)
        return total
    return f


def _dl_manual(instrucciones: str):
    def f(dest: Path) -> bool:
        if any(dest.rglob("*")):
            log("  ya hay archivos, asumo que lo bajaste")
            return True
        dest.mkdir(parents=True, exist_ok=True)
        log("  ⚠ esta fuente es de descarga MANUAL:")
        for l in instrucciones.strip().splitlines():
            log(f"    {l.strip()}")
        log(f"    Descomprimila en: {dest}")
        return False
    return f


# ---------- normalizadores ------------------------------------------------- #
def _norm_yolo(remapeo: Dict[int, int], subdir_img: str = "",
               subdir_lbl: str = "", conservar_negativos: bool = False):
    """Copia pares imagen/label remapeando índices. Las clases que no estén en
    `remapeo` se descartan (ej: purse/bill/card de Sohas).

    conservar_negativos decide qué hacer con una foto cuyas cajas se
    descartaron TODAS. Es una distinción que importa:

      False (default, seguro): se tira la foto. Correcto cuando lo descartado
        era un objeto real que igual está en la imagen — p.ej. una caja
        IsGroupOf de Open Images: el objeto sigue ahí, solo que sin etiqueta
        usable. Dejarla como "negativo" le enseñaría a NO detectar eso.

      True: se conserva con label vacío, como negativo. Correcto cuando la
        clase descartada garantiza que NO hay nada nuestro en la foto — p.ej.
        la clase 'other' de FireAndSmoke (atardeceres, faroles, balizas): son
        justo los falsos positivos que queremos que aprenda a ignorar.
    """
    def f(crudo: Path, salida: Path) -> int:
        salida_i = salida / "images"
        salida_l = salida / "labels"
        salida_i.mkdir(parents=True, exist_ok=True)
        salida_l.mkdir(parents=True, exist_ok=True)

        base_i = crudo / subdir_img if subdir_img else crudo
        n = 0
        for img in _imagenes(base_i):
            # buscar el .txt: al lado, o en el labels/ paralelo
            cand = [img.with_suffix(".txt")]
            partes = list(img.parts)
            if "images" in partes:
                i = len(partes) - 1 - partes[::-1].index("images")
                partes[i] = "labels"
                cand.append(Path(*partes).with_suffix(".txt"))
            if subdir_lbl:
                cand.append(crudo / subdir_lbl / (img.stem + ".txt"))
            txt = next((c for c in cand if c.exists()), None)

            filas_out = []
            if txt is not None:
                for fila in _leer_yolo(txt):
                    if fila[0] not in remapeo:
                        continue
                    fila = _clip((remapeo[fila[0]], *fila[1:]))
                    if _sano(fila):
                        filas_out.append(fila)
                if not filas_out and _leer_yolo(txt) and not conservar_negativos:
                    continue          # tenía cajas pero ninguna nos sirve
            # sin txt o txt vacío => negativo, se conserva

            destino = salida_i / f"{crudo.name}__{n:06d}{img.suffix.lower()}"
            try:
                shutil.copy2(img, destino)
            except OSError:
                continue
            _escribir_yolo(salida_l / (destino.stem + ".txt"), filas_out)
            n += 1
        return n
    return f


def _norm_yolo_nombres(mapa: Dict[str, int], conservar_negativos: bool = False):
    """Igual que _norm_yolo pero lee los nombres de clase del data.yaml del
    dataset y mapea POR NOMBRE. Los exports de Roboflow cambian el orden de
    clases entre versiones; mapear por índice es una bomba de tiempo."""
    def f(crudo: Path, salida: Path) -> int:
        yml = next((p for p in crudo.rglob("*.yaml")
                    if p.name in ("data.yaml", "dataset.yaml")), None)
        if yml is None:
            log("  ⚠ no encontré data.yaml, no puedo mapear por nombre")
            return 0
        txt = yml.read_text(encoding="utf-8", errors="ignore")
        nombres: List[str] = []
        for linea in txt.splitlines():
            if linea.strip().startswith("names:"):
                resto = linea.split(":", 1)[1].strip()
                if resto.startswith("["):
                    nombres = [x.strip().strip("'\"")
                               for x in resto.strip("[]").split(",") if x.strip()]
                break
        if not nombres:                       # formato de lista multilínea
            dentro = False
            for linea in txt.splitlines():
                if linea.strip().startswith("names:"):
                    dentro = True
                    continue
                if dentro:
                    if linea.strip().startswith("-"):
                        nombres.append(linea.split("-", 1)[1].strip().strip("'\""))
                    elif linea.strip() and not linea.startswith(" "):
                        break
        if not nombres:
            log("  ⚠ no pude leer 'names' del data.yaml")
            return 0
        log(f"  clases del dataset: {nombres}")
        remapeo = {i: mapa[n.lower()] for i, n in enumerate(nombres)
                   if n.lower() in mapa}
        if not remapeo:
            log(f"  ⚠ ninguna de {list(mapa)} está en {nombres}. Se descarta.")
            return 0
        log("  remapeo por nombre: "
            + ", ".join(f"{nombres[i]}->{c}" for i, c in remapeo.items()))
        return _norm_yolo(remapeo, conservar_negativos=conservar_negativos)(
            crudo, salida)
    return f


def _norm_voc(remapeo_nombre: Dict[str, int]):
    """Pascal VOC (.xml) -> YOLO. Lo usan Monash Guns y Sohas."""
    def f(crudo: Path, salida: Path) -> int:
        import xml.etree.ElementTree as ET
        salida_i = salida / "images"
        salida_l = salida / "labels"
        salida_i.mkdir(parents=True, exist_ok=True)
        salida_l.mkdir(parents=True, exist_ok=True)

        # Índice {stem: ruta} construido UNA vez. Monash Guns separa
        # Annotations/ de JPEGImages/, y buscar cada imagen con rglob por
        # separado sería O(n^2): 5.500 recorridos del árbol completo.
        indice = {}
        for img in _imagenes(crudo):
            indice.setdefault(img.stem, img)

        n = 0
        for xml in sorted(crudo.rglob("*.xml")):
            try:
                raiz = ET.parse(xml).getroot()
            except ET.ParseError:
                continue
            img = indice.get(xml.stem)
            if img is None:
                continue

            tam = raiz.find("size")
            W = float(tam.find("width").text) if tam is not None else 0
            H = float(tam.find("height").text) if tam is not None else 0
            if W <= 0 or H <= 0:
                import cv2
                m = cv2.imread(str(img))
                if m is None:
                    continue
                H, W = float(m.shape[0]), float(m.shape[1])

            filas = []
            for obj in raiz.findall("object"):
                nom = (obj.findtext("name") or "").strip().lower()
                if nom not in remapeo_nombre:
                    continue
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                try:
                    x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
                    x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
                except (TypeError, ValueError):
                    continue
                fila = _clip((remapeo_nombre[nom], (x1 + x2) / 2 / W,
                              (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
                if _sano(fila):
                    filas.append(fila)
            if not filas:
                continue
            destino = salida_i / f"{crudo.name}__{n:06d}{img.suffix.lower()}"
            try:
                shutil.copy2(img, destino)
            except OSError:
                continue
            _escribir_yolo(salida_l / (destino.stem + ".txt"), filas)
            n += 1
        return n
    return f


# ---------- catálogo ------------------------------------------------------- #
# Clases de Open Images -> clases nuestras. El ORDEN importa: define el índice
# con el que se exportan los .txt, y por eso descarga y normalización comparten
# este mismo diccionario.
_MAPA_OI: Dict[str, int] = {
    "Person": 2, "Handgun": 3, "Knife": 4,
    "Kitchen knife": 4, "Mobile phone": 5, "Box": 6,
}


def catalogo() -> Dict[str, Fuente]:
    f: Dict[str, Fuente] = {}

    f["d-fire"] = Fuente(
        "d-fire", "D-Fire (humo + llama)", "CC0-1.0", True,
        "https://github.com/gaiasd/DFireDataset", "kaggle",
        _dl_dfire, _norm_yolo({0: 0, 1: 1}),
        "Base de humo/llama. Ya usa 0=smoke 1=fire: coincide con nuestras clases.",
        clases=(0, 1), gb=4.0, deps=("kaggle",))

    f["pyro-sdis"] = Fuente(
        "pyro-sdis", "Pyro-SDIS (humo, cámara fija)", "Apache-2.0", True,
        "https://huggingface.co/datasets/pyronear/pyro-sdis", "",
        _dl_pyro, _norm_yolo({0: 0, 1: 0}),
        "Humo tenue y lejano desde torres fijas. Lo que a D-Fire le falta. "
        "OJO: sus etiquetas usan el índice 1 para smoke aunque la doc diga 0; "
        "mapeamos los dos al 0 nuestro porque es un dataset de una sola clase.",
        clases=(0,), gb=3.3, deps=("datasets",))

    f["monash-guns"] = Fuente(
        "monash-guns", "Monash Guns (pistola en CCTV)", "MIT (orig.) / CC BY 4.0 (mirror)",
        True, "https://universe.roboflow.com/arms/the-monash-guns-dataset/dataset/2",
        "roboflow", _dl_roboflow("arms", "the-monash-guns-dataset", 2),
        _norm_yolo_nombres({"pistol": 3, "pistols": 3, "gun": 3, "guns": 3,
                            "handgun": 3, "weapon": 3, "firearm": 3}),
        "Escenificado sobre cámaras CCTV reales: es la mejor fuente de pistola "
        "que existe con licencia usable. El Drive del repo original está caído "
        "(404 a agosto 2026), así que vamos por el mirror de Roboflow.",
        clases=(3,), gb=1.2, deps=("roboflow",))

    f["paquetes-puerta"] = Fuente(
        "paquetes-puerta", "package-at-front-door", "MIT", True,
        "https://universe.roboflow.com/package-detection/package-at-front-door",
        "roboflow", _dl_roboflow("package-detection", "package-at-front-door", 2),
        _norm_yolo({0: 6}),
        "Perspectiva de cámara de puerta: la correcta para vigilancia.",
        clases=(6,), gb=0.3, deps=("roboflow",))

    f["cajas"] = Fuente(
        "cajas", "boxes-iqjbg", "CC BY 4.0", True,
        "https://universe.roboflow.com/joel-tnzu2/boxes-iqjbg", "roboflow",
        _dl_roboflow("joel-tnzu2", "boxes-iqjbg", 1),
        _norm_yolo_nombres({"box": 6, "boxes": 6, "parcel": 6, "package": 6,
                            "cardboard": 6, "carton": 6}),
        "Volumen limpio de paquete/caja.",
        clases=(6,), gb=0.6, deps=("roboflow",))

    f["negativos-fuego"] = Fuente(
        "negativos-fuego", "FireAndSmoke (clase 'other')", "CC BY 4.0", True,
        "https://universe.roboflow.com/catargiuconstantin/firesmokedataset",
        "roboflow", _dl_roboflow("catargiuconstantin", "firesmokedataset", 2),
        _norm_yolo({0: 1, 1: 0}),
        "OJO: verificá el orden de clases en su data.yaml antes de confiar "
        "en el remapeo. Su clase 'other' (atardeceres, faroles) se descarta "
        "y queda como negativo, que es justo lo que queremos.",
        clases=(0, 1), gb=2.5, deps=("roboflow",))

    f["openimages"] = Fuente(
        "openimages", "Open Images V7 (persona/cuchillo/celular/paquete)",
        "CC BY 4.0 (anot.) + CC BY 2.0 (imgs)", True,
        "https://storage.googleapis.com/openimages/web/index.html", "",
        _dl_openimages(_MAPA_OI, 2500),
        _hacer_norm_openimages(_MAPA_OI),
        "Baja cada clase por separado con su propio cupo, si no 'Person' se "
        "come todo. Filtra IsGroupOf (cajas sobre multitudes enteras) e "
        "IsDepiction (estatuas, pósters), que si se cuelan generan falsos "
        "positivos sobre publicidad. OJO con Handgun: son fotos de Flickr, "
        "objetos grandes y centrados. Sirve para que la clase exista y para "
        "pre-entrenar features, pero para detectar un arma de 25 px en una "
        "cámara real hace falta Monash Guns o metraje propio.",
        clases=(2, 3, 4, 5, 6), gb=3.6, deps=("fiftyone",))

    f["sohas"] = Fuente(
        "sohas", "OD-WeaponDetection / Sohas", "CC BY-SA 4.0 (ver aviso)", False,
        "https://github.com/ari-dasci/OD-WeaponDetection", "manual",
        _dl_manual("""
            Bajá 'Sohas weapon Detection' de:
              https://github.com/ari-dasci/OD-WeaponDetection
            o el mirror con descarga directa (1,73 GB):
              https://datasetninja.com/od-weapon-detection-sohas-detection
        """),
        _norm_voc({"pistol": 3, "knife": 4, "smartphone": 5}),
        "CC BY-SA: share-alike, y son fotos web relicenciadas por la UGR "
        "sobre imágenes de terceros. Único público con CELULAR anotado junto "
        "a pistola. Revisá el riesgo legal antes de usarlo en producto.",
        clases=(3, 4, 5), gb=1.8, deps=())

    return f


# --------------------------------------------------------------------------- #
# Armado: dedup + balance + split
# --------------------------------------------------------------------------- #
def armar(dir_norm: Path, salida: Path, tope: int, tope_neg: int,
          frac_val: float, seed: int = 0) -> None:
    random.seed(seed)
    log("\n" + "=" * 70)
    log("ARMANDO mezcla_v1")
    log("=" * 70)

    pares: List[Tuple[Path, Path]] = []
    for sub in sorted(dir_norm.iterdir()) if dir_norm.is_dir() else []:
        for img in _imagenes(sub / "images"):
            txt = sub / "labels" / (img.stem + ".txt")
            pares.append((img, txt))
    if not pares:
        sys.exit(f"No hay nada normalizado en {dir_norm}. "
                 "Corré primero --descargar y --normalizar.")
    log(f"  {len(pares)} imágenes normalizadas en total")

    # --- dedup sobre el pool COMPLETO ------------------------------------
    log("  deduplicando (dHash sobre todo el pool)...")
    vistos: Dict[int, Path] = {}
    unicos: List[Tuple[Path, Path]] = []
    dups = 0
    for i, (img, txt) in enumerate(pares):
        h = dhash(img)
        if h is None:
            continue
        if h in vistos:
            dups += 1
            continue
        vistos[h] = img
        unicos.append((img, txt))
        if (i + 1) % 5000 == 0:
            log(f"    {i+1}/{len(pares)}  ({dups} duplicados)")
    log(f"  {dups} duplicados descartados -> quedan {len(unicos)}")

    # --- balance ----------------------------------------------------------
    log(f"  balanceando (tope {tope} cajas/clase, {tope_neg} negativos)...")
    con_cajas, negativos = [], []
    for img, txt in unicos:
        (con_cajas if _leer_yolo(txt) else negativos).append((img, txt))
    random.shuffle(con_cajas)
    random.shuffle(negativos)

    # Prioriza imágenes de las clases más escasas: ordena por la rareza de su
    # clase menos representada, así las clases chicas entran completas.
    global_cnt = Counter()
    for _, txt in con_cajas:
        for f in _leer_yolo(txt):
            global_cnt[f[0]] += 1
    def rareza(par):
        cs = {f[0] for f in _leer_yolo(par[1])}
        return min((global_cnt[c] for c in cs), default=10**9)
    con_cajas.sort(key=rareza)

    cnt = Counter()
    elegidas: List[Tuple[Path, Path]] = []
    for img, txt in con_cajas:
        filas = _leer_yolo(txt)
        cs = {f[0] for f in filas}
        if all(cnt[c] >= tope for c in cs):
            continue
        elegidas.append((img, txt))
        for f in filas:
            cnt[f[0]] += 1
    elegidas += negativos[:tope_neg]
    random.shuffle(elegidas)
    log(f"  {len(elegidas)} imágenes elegidas "
        f"({len(elegidas) - min(len(negativos), tope_neg)} con cajas + "
        f"{min(len(negativos), tope_neg)} negativas)")

    # --- split + escritura -------------------------------------------------
    n_val = max(1, int(len(elegidas) * frac_val))
    grupos = {"val": elegidas[:n_val], "train": elegidas[n_val:]}

    if salida.exists():
        shutil.rmtree(salida)
    for split, items in grupos.items():
        (salida / "images" / split).mkdir(parents=True, exist_ok=True)
        (salida / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, txt in items:
            shutil.copy2(img, salida / "images" / split / img.name)
            _escribir_yolo(salida / "labels" / split / (img.stem + ".txt"),
                           _leer_yolo(txt))

    # --- reporte -----------------------------------------------------------
    log("\n  RESULTADO")
    log(f"  {'clase':<10} {'train':>8} {'val':>8}")
    log("  " + "-" * 28)
    resumen = {}
    for split in ("train", "val"):
        c = Counter()
        for t in (salida / "labels" / split).glob("*.txt"):
            for f in _leer_yolo(t):
                c[f[0]] += 1
        resumen[split] = c
    for i, nom in enumerate(CLASES):
        log(f"  {i} {nom:<8} {resumen['train'][i]:>8} {resumen['val'][i]:>8}")
    log("  " + "-" * 28)
    log(f"  {'imágenes':<10} {len(grupos['train']):>8} {len(grupos['val']):>8}")

    (salida / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASES)}\nnames: {list(CLASES)}\n", encoding="utf-8")

    sin_val = [n for i, n in enumerate(CLASES)
               if resumen["train"][i] > 0 and resumen["val"][i] == 0]
    if sin_val:
        log(f"\n  ⚠ sin ejemplos de VALIDACIÓN: {', '.join(sin_val)}")
        log("    su mAP va a dar 0 siempre porque no hay contra qué medir. "
            "Subí --val o conseguí más dato de esas clases.")

    faltan = [n for i, n in enumerate(CLASES) if resumen["train"][i] == 0]
    if faltan:
        log(f"\n  ⚠ sin ejemplos de entrenamiento: {', '.join(faltan)}")
        log("    esas clases no se van a aprender. Conseguí dato o sacálas "
            "de DEFAULT_CLASSES en objects_head.py.")
    log(f"\n  Listo: {salida}")
    log("  Entrenar:  python entrenar_objetos_cuda.py --dataset datasets/mezcla_v1")



# --------------------------------------------------------------------------- #
# Preflight: decir QUÉ va a funcionar antes de esperar una hora
# --------------------------------------------------------------------------- #
def _gb_libres(d: Path) -> float:
    try:
        return shutil.disk_usage(d).free / 1e9
    except OSError:
        return -1.0


def _estado_fuente(f: "Fuente") -> Tuple[bool, str]:
    # Lo primero: ¿ya está en disco? Una fuente descargada no necesita
    # credenciales ni dependencias, y contarla como "faltante" hace que el
    # chequeo mienta sobre el estado real del proyecto.
    n = len(_imagenes(DIR_CRUDO / f.clave))
    if n:
        return True, f"ya descargado · {n} imgs"

    faltan = [m for m in f.deps if not _importable(m)]
    if f.requiere == "kaggle":
        if "kaggle" in faltan:
            return False, "pip install kaggle"
        # El CLI nuevo de Kaggle acepta varias formas de credencial.
        d = Path.home() / ".kaggle"
        if not (any((d / n).exists() for n in ("kaggle.json", "access_token"))
                or os.environ.get("KAGGLE_API_TOKEN")
                or os.environ.get("KAGGLE_KEY")):
            return False, "sin credencial -> corré:  python -m kaggle auth login"
    if f.requiere == "roboflow":
        k = _clave_roboflow()
        if not k:
            return False, "falta ROBOFLOW_API_KEY (app.roboflow.com/settings/api)"
        if _clave_sospechosa(k):
            return False, f"ROBOFLOW_API_KEY sospechosa: {_pista_clave(k)}"
    if f.requiere == "manual":
        return False, "descarga manual (ver --listar)"
    if faltan:
        return False, "pip install " + " ".join(faltan)
    return True, "listo"


def verificar_entorno(cat: Dict[str, "Fuente"], solo_comerciales: bool) -> None:
    log("=" * 74)
    log("CHEQUEO PREVIO")
    log("=" * 74)

    log(f"  python      : {sys.version.split()[0]}")
    for mod in ("cv2", "numpy"):
        log(f"  {mod:<12}: {'OK' if _importable(mod) else 'FALTA -> pip install opencv-python numpy'}")

    libres = _gb_libres(DIR_DATASETS.parent)
    log(f"  disco libre : {libres:.0f} GB" if libres >= 0 else "  disco libre : desconocido")

    log("")
    log(f"  {'fuente':<18} {'clases':<10} {'GB':>5}  estado")
    log("  " + "-" * 68)
    total_gb, por_bajar, ya = 0.0, [], []
    for k, f in cat.items():
        if solo_comerciales and not f.comercial:
            continue
        ok, msg = _estado_fuente(f)
        cls = ",".join(str(c) for c in f.clases) or "-"
        ya_esta = msg.startswith("ya descargado")
        marca = "==" if ya_esta else ("OK" if ok else "--")
        gb = "  -  " if ya_esta else f"{f.gb:>5.1f}"
        log(f"  {marca} {k:<15} {cls:<10} {gb}  {msg}")
        if ya_esta:
            ya.append(f)
        elif ok:
            total_gb += f.gb
            por_bajar.append(f)

    log("  " + "-" * 68)
    if ya:
        log(f"  ==  {len(ya)} fuentes ya en disco, no se vuelven a bajar")
    log(f"  OK  {len(por_bajar)} fuentes por descargar · ~{total_gb:.0f} GB "
        f"(más ~{total_gb:.0f} GB de la copia normalizada)")
    listas = ya + por_bajar

    if libres >= 0 and libres < total_gb * 2.2:
        log(f"  ⚠ con {libres:.0f} GB libres NO alcanza. Hacen falta ~{total_gb*2.2:.0f} GB.")
        log("    Podés bajar de a una fuente y borrar datasets/_crudo/<fuente> "
            "después de normalizarla.")

    cubiertas = set()
    for f in listas:
        cubiertas.update(f.clases)
    # Lo ya normalizado cuenta aunque su fuente cruda se haya borrado.
    for sub in (DIR_NORM.iterdir() if DIR_NORM.is_dir() else []):
        f = cat.get(sub.name)
        if f and _imagenes(sub / "images"):
            cubiertas.update(f.clases)

    huerfanas = [f"{i} {n}" for i, n in enumerate(CLASES) if i not in cubiertas]
    if huerfanas:
        log(f"\n  ⚠ clases SIN ninguna fuente disponible: {', '.join(huerfanas)}")
        log("    No se van a aprender. Resolvé las credenciales de arriba, o")
        log("    grabá dato propio (ver datasets/DATASETS.md).")
    else:
        log("\n  Las 7 clases tienen fuente (descargada o descargable).")

    salida = DIR_SALIDA / "labels" / "train"
    if salida.is_dir():
        c = Counter()
        for t in salida.glob("*.txt"):
            for fila in _leer_yolo(t):
                c[fila[0]] += 1
        if c:
            log("")
            log("  Lo que YA tenés armado en mezcla_v1 (cajas de train):")
            for i, nom in enumerate(CLASES):
                estado = "" if c[i] else "   <- VACÍA"
                log(f"    {i} {nom:<9} {c[i]:>7}{estado}")
    log("")


def instalar_deps(cat: Dict[str, "Fuente"]) -> None:
    faltan = sorted({m for f in cat.values() for m in f.deps if not _importable(m)})
    faltan += [m for m in ("cv2",) if not _importable(m)]
    if not faltan:
        log("  nada que instalar, ya está todo")
        return
    pip_names = {"cv2": "opencv-python"}
    paquetes = [pip_names.get(m, m) for m in faltan]
    log(f"  instalando: {' '.join(paquetes)}")
    r = subprocess.run([sys.executable, "-m", "pip", "install", *paquetes])
    log("  listo" if r.returncode == 0 else "  ⚠ pip falló, instalalos a mano")



# --------------------------------------------------------------------------- #
# Diagnóstico: por qué una fuente pierde imágenes al normalizar
# --------------------------------------------------------------------------- #
def diagnosticar(cat: Dict[str, "Fuente"], clave: str, n: int = 8) -> None:
    f = cat.get(clave)
    if f is None:
        sys.exit(f"fuente desconocida: {clave}. Ver --listar")
    crudo = DIR_CRUDO / clave
    imgs = _imagenes(crudo)
    if not imgs:
        sys.exit(f"No hay imágenes en {crudo}")

    norm = DIR_NORM / clave
    n_norm = len(_imagenes(norm / "images"))
    log("=" * 74)
    log(f"DIAGNÓSTICO · {clave}")
    log("=" * 74)
    log(f"  imágenes en _crudo      : {len(imgs)}")
    log(f"  imágenes en _normalizado: {n_norm}")
    if n_norm and n_norm < len(imgs):
        log(f"  -> se están perdiendo {len(imgs) - n_norm} "
            f"({100*(1-n_norm/len(imgs)):.0f}%)")
    log("")

    # Muestreo determinista a lo largo de toda la lista, no solo el principio.
    paso = max(1, len(imgs) // n)
    muestra = imgs[::paso][:n]

    veredictos = Counter()
    for img in muestra:
        cand = [img.with_suffix(".txt")]
        partes = list(img.parts)
        if "images" in partes:
            i = len(partes) - 1 - partes[::-1].index("images")
            partes[i] = "labels"
            cand.append(Path(*partes).with_suffix(".txt"))
        txt = next((c for c in cand if c.exists()), None)

        log(f"  {img.name}")
        if txt is None:
            log("    label   : NO EXISTE  -> se conserva como negativo")
            veredictos["negativo (sin label)"] += 1
            continue

        crudo_txt = txt.read_text(encoding="utf-8", errors="ignore")
        muestra_txt = crudo_txt[:120].replace("\n", "\\n")
        log(f"    label   : {txt.name}  ({len(crudo_txt)} bytes)")
        log(f"    crudo   : {muestra_txt!r}")

        filas = _leer_yolo(txt)
        log(f"    parseadas: {len(filas)} filas"
            + (f"  primera={filas[0]}" if filas else ""))

        if not crudo_txt.strip():
            log("    veredicto: label VACÍO -> se conserva como negativo")
            veredictos["negativo (label vacío)"] += 1
            continue
        if not filas:
            log("    veredicto: el label tiene texto pero NO parsea como YOLO")
            log("               -> se conserva como negativo (sospechoso)")
            veredictos["no parsea"] += 1
            continue

        clases_vistas = sorted({fl[0] for fl in filas})
        log(f"    clases   : {clases_vistas}")
        malas = [fl for fl in filas if not _sano(_clip(fl))]
        if malas:
            log(f"    ⚠ {len(malas)} cajas degeneradas o fuera de 0..1, "
                f"ej: {malas[0]}")
            if len(malas) == len(filas):
                log("    veredicto: TODAS las cajas son inválidas -> IMAGEN "
                    "DESCARTADA (acá se están perdiendo)")
                veredictos["DESCARTADA (cajas inválidas)"] += 1
                log("")
                continue
        log(f"    veredicto: {len(filas) - len(malas)} cajas utilizables")
        veredictos["con cajas"] += 1
        log("")

    log("  " + "-" * 68)
    log("  RESUMEN DE LA MUESTRA")
    for k, v in veredictos.most_common():
        log(f"    {k:<28} {v}/{len(muestra)}")
    if veredictos.get("no parsea"):
        log("")
        log("  El label tiene contenido pero no es formato YOLO. Casi seguro el")
        log("  descargador guardó el campo crudo del dataset (una lista, un dict)")
        log("  en vez del texto de las cajas. Pegame el 'crudo' de arriba.")
    log("")


# --------------------------------------------------------------------------- #
def main() -> int:
    cat = catalogo()
    ap = argparse.ArgumentParser(description="Bajar y armar el dataset de objetos")
    ap.add_argument("--listar", action="store_true", help="mostrar las fuentes y salir")
    ap.add_argument("--verificar", action="store_true",
                    help="chequear deps, credenciales, disco y clases cubiertas")
    ap.add_argument("--instalar-deps", action="store_true",
                    help="pip install de todo lo que falte")
    ap.add_argument("--diagnosticar", metavar="FUENTE", default=None,
                    help="mostrar por qué una fuente pierde imágenes al normalizar")
    ap.add_argument("--login-kaggle", action="store_true",
                    help="abrir el login de Kaggle (evita tener que acordarse "
                         "de 'python -m kaggle auth login')")
    ap.add_argument("--descargar", nargs="*", metavar="FUENTE",
                    help="descargar estas fuentes (sin argumentos = todas)")
    ap.add_argument("--normalizar", nargs="*", metavar="FUENTE",
                    help="normalizar a YOLO con nuestros índices")
    ap.add_argument("--armar", action="store_true",
                    help="deduplicar + balancear + partir -> mezcla_v1")
    ap.add_argument("--todo", action="store_true", help="las tres etapas")
    ap.add_argument("--solo-comerciales", action="store_true",
                    help="saltear las fuentes con licencia problemática")
    ap.add_argument("--tope", type=int, default=TOPE_POR_CLASE)
    ap.add_argument("--tope-negativos", type=int, default=TOPE_NEGATIVOS)
    ap.add_argument("--val", type=float, default=FRACCION_VAL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.instalar_deps:
        instalar_deps(cat)
        if not any([args.descargar is not None, args.normalizar is not None,
                    args.armar, args.todo, args.verificar]):
            return 0

    if args.login_kaggle:
        if not _importable("kaggle"):
            log("Falta el paquete: pip install kaggle")
            return 1
        log("Abriendo el login de Kaggle en el navegador...")
        return subprocess.run(
            [sys.executable, "-m", "kaggle", "auth", "login"]).returncode

    if args.diagnosticar:
        diagnosticar(cat, args.diagnosticar)
        return 0

    if args.verificar:
        verificar_entorno(cat, args.solo_comerciales)
        return 0

    if args.listar or not any([args.descargar is not None, args.normalizar is not None,
                               args.armar, args.todo]):
        print(f"\n{'clave':<18} {'licencia':<28} {'com.':<5} requiere")
        print("-" * 78)
        for k, f in cat.items():
            print(f"{k:<18} {f.licencia:<28} {'sí' if f.comercial else 'NO':<5} "
                  f"{f.requiere or '-'}")
            print(f"{'':<18} {f.url}")
            if f.nota:
                for l in f.nota.split(". "):
                    if l.strip():
                        print(f"{'':<18} · {l.strip().rstrip('.')}.")
            print()
        print("Ejemplos:")
        print("  python bajar_datasets.py --verificar          <- empezá por acá")
        print("  python bajar_datasets.py --login-kaggle       <- para D-Fire")
        print("  python bajar_datasets.py --instalar-deps")
        print("  python bajar_datasets.py --todo --solo-comerciales")
        print("  python bajar_datasets.py --descargar d-fire pyro-sdis")
        print("  python bajar_datasets.py --armar")
        return 0

    def seleccion(arg) -> List[str]:
        claves = list(cat) if not arg else arg
        if args.solo_comerciales:
            claves = [k for k in claves if cat[k].comercial]
        malas = [k for k in claves if k not in cat]
        if malas:
            sys.exit(f"fuente desconocida: {malas}. Ver --listar")
        return claves

    DIR_DATASETS.mkdir(parents=True, exist_ok=True)
    DIR_CRUDO.mkdir(parents=True, exist_ok=True)
    DIR_NORM.mkdir(parents=True, exist_ok=True)

    if args.todo:
        verificar_entorno(cat, args.solo_comerciales)

    if args.todo or args.descargar is not None:
        claves = seleccion(args.descargar if args.descargar else None)
        log("\n" + "=" * 70)
        log("DESCARGA")
        log("=" * 70)
        resultados = {}
        for k in claves:
            f = cat[k]
            log(f"\n[{k}] {f.nombre}  ({f.licencia})")
            d = DIR_CRUDO / k
            try:
                f.descargar(d)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log(f"  ⚠ error: {str(e)[:200]}")
            # No confiamos en el valor de retorno: miramos el disco.
            resultados[k] = len(_imagenes(d))

        log("\n" + "-" * 62)
        log("  RESUMEN DE LA DESCARGA")
        for k, n in resultados.items():
            log(f"    {'OK ' if n else 'NADA'}  {k:<18} {n:>7} imágenes")
        log("-" * 62)
        vacias = [k for k, n in resultados.items() if not n]
        if vacias:
            log(f"  Sin datos: {', '.join(vacias)} — mirá los mensajes de arriba.")

    if args.todo or args.normalizar is not None:
        claves = seleccion(args.normalizar if args.normalizar else None)
        log("\n" + "=" * 70)
        log("NORMALIZACIÓN (remapeo de índices a 0..6)")
        log("=" * 70)
        for k in claves:
            f = cat[k]
            crudo = DIR_CRUDO / k
            if not crudo.is_dir() or not any(crudo.rglob("*")):
                log(f"[{k}] sin datos crudos, se saltea")
                continue
            salida = DIR_NORM / k
            if salida.exists():
                shutil.rmtree(salida)
            entrada = len(_imagenes(crudo))
            try:
                n = f.normalizar(crudo, salida)
                log(f"[{k}] {n} imágenes normalizadas"
                    + (f" (de {entrada})" if entrada and n != entrada else ""))
                # Si se cae más de la mitad, casi siempre es un remapeo mal
                # puesto: las cajas quedan fuera del mapa y la imagen entera se
                # descarta. Es exactamente el error que costó descubrir con
                # pyro-sdis (usaba clase 1, no 0).
                if entrada and n < entrada * 0.5:
                    log(f"[{k}] ⚠ se perdió el {100*(1-n/entrada):.0f}% de las "
                        "imágenes. Suele ser el remapeo de clases mal puesto.")
                    log(f"[{k}]   revisalo con:  python bajar_datasets.py "
                        f"--diagnosticar {k}")
            except Exception as e:
                log(f"[{k}] ⚠ error: {str(e)[:200]}")

    if args.todo or args.armar:
        armar(DIR_NORM, DIR_SALIDA, args.tope, args.tope_negativos,
              args.val, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
