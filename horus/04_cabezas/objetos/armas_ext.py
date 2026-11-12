# -*- coding: utf-8 -*-
"""
armas_ext.py · Datasets públicos de armas para el YOLO de 4 clases (06/11).

Por qué
-------
El 06/11 a las 12:23 la v3 marcó "pistola" (0,65, sostenida 9 s) en una media que
Teo tenía en la mano. La pistola de la v3 sale solo de Open Images: 490 cajas de
fotos de Flickr, armas grandes y centradas, y ni un negativo del tipo "mano
sosteniendo algo que NO es un arma". Este script trae datasets públicos con las
dos cosas: pistolas en contextos variados (cámaras, videos, calle) y manos con
celulares, billeteras, tarjetas, billetes y cuchillos.

Qué arma
--------
Una carpeta con el formato de mezcla_v1 (7 clases: humo llama persona pistola
cuchillo celular paquete), en tres partes:

  images|labels/train    entra ENTERA al YOLO4 (positivos y negativos, sin tope)
  images|labels/val      val aparte de las fuentes nuevas; no se mezcla con el de
                         mezcla_v1, así la métrica vieja sigue siendo comparable
  images|labels/prueba   fuentes que el modelo no ve nunca: dicen si generaliza

Reglas por fuente (lista "fuentes" del JSON de --config)
-------------------------------------------------------
  alias     prefijo de los archivos (corto, sin espacios)
  ruta      carpeta bajo /kaggle/input, o absoluta
  mapa      clase del dataset -> clase nuestra. Lo que no está ni en mapa, ni en
            excluir, ni en ignorar deja la foto AFUERA (y se avisa): mejor perder
            una foto que enseñarle algo mal.
  excluir   si la foto tiene una de estas clases no entra. Los rifles van acá: la
            clase es "pistola" y no queremos enseñarle ni que un rifle es pistola
            ni que es fondo.
  ignorar   la caja se tira pero la foto entra ("person" en un dataset que no
            anota a todas las personas: al YOLO4 la persona no le importa).
  vacias    "negativo" si el dataset trae fotos sin cajas A PROPÓSITO (carpeta
            no_weapon); "descartar" si no (suelen ser fotos que nadie anotó).
  rol       "train" (90/10 en train/val) o "prueba".
  video     true si son cuadros de video: el split va por video y no por cuadro,
            para que el val no tenga el cuadro de al lado de uno de train.
  tope_pos, tope_neg   máximo de fotos con pistola / sin pistola de esa fuente.
  nombres   opcional: lista o dict de nombres YOLO si el data.yaml no sirve.
  bloque    (07/11) con video: agrupa por número de cuadro de a `bloque`
            ("frame_0391" -> grupo frame#13 con bloque 30). Para datasets donde
            todos los cuadros se llaman frame_NNNN: sin esto serían UN grupo y
            caerían todos del mismo lado del split.
  siempre   (07/11) clases que se ignoran pero cuya foto entra SIN tope: los
            negativos difíciles anotados a propósito (lámparas, luces, cosas
            naranjas que no son fuego).
  lado_min  (07/11) lado relativo mínimo (raíz de ancho x alto) de una caja de
            pistola, llama o humo; si hay una más chica, la foto no entra. Una
            llama de 3 px a 416 es un punto brillante: enseñarle que eso es
            fuego es enseñarle a ver fuego en los LEDs del cuarto.

07/11 · fuego_ext.json usa este mismo script para llamas chicas de casa (velas,
encendedores, fósforos, fuegos de interior). Una caja de llama o humo demasiado
chica deja la foto afuera, igual que una pistola: tirarla dejaría un fuego sin
anotar. Con las fuentes de armas el resultado es idéntico al de la v4 (mismo val
y misma prueba de armas).

Duplicados: dHash de 64 bits (y el del espejo). Lo de val y prueba que esté a
distancia <= 5 de algo de train, o de algo del val de mezcla_v1, se saca: los
datasets de armas de Kaggle son remezclas unos de otros y sin esto la prueba
"de afuera" mediría memoria.

    python armas_ext.py --config armas_ext.json --salida datasets/armas_ext \\
        --mezcla datasets/mezcla_v1
    python armas_ext.py --prueba     # datos sintéticos, verifica el camino entero
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CLASES_7 = ("humo", "llama", "persona", "pistola", "cuchillo", "celular", "paquete")
IDX = {c: i for i, c in enumerate(CLASES_7)}
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PISTOLA = IDX["pistola"]
# Clases que no se pueden tirar si la caja es mínima: la foto se va entera.
MINIMA_EXCLUYE = {PISTOLA, IDX["llama"], IDX["humo"]}
COLOR = {PISTOLA: (255, 40, 40), IDX["llama"]: (255, 150, 0), IDX["humo"]: (210, 210, 210)}


def norm(n) -> str:
    return re.sub(r"[\s_\-]+", " ", str(n).strip().lower())


# --------------------------------------------------------------------------- #
# Lectura de una fuente
# --------------------------------------------------------------------------- #
# /kaggle/input es un montaje de red: cada exists()/stat()/open() es un pedido.
# Por eso: un solo os.walk por fuente, las etiquetas se buscan en la lista ya
# hecha (sin stat) y se leen en paralelo. El relevamiento del 06/11 tardaba más
# de 10 min por dataset haciendo exists() por foto; así, 8 s.
def _listar(raiz: Path) -> List[str]:
    todos = []
    for r, _, fs in os.walk(raiz, followlinks=True):
        todos += [os.path.join(r, x) for x in fs]
    return todos


def _leer(p: str) -> Optional[str]:
    try:
        with open(p, errors="ignore") as fh:
            return fh.read()
    except Exception:
        return None


def _en_paralelo(fn, xs, hilos: int = 32):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(hilos) as ex:
        return list(ex.map(fn, xs))


def _nombres_yolo(todos: List[str], forzados=None) -> Dict[int, str]:
    if forzados:
        if isinstance(forzados, dict):
            return {int(k): norm(v) for k, v in forzados.items()}
        return {i: norm(v) for i, v in enumerate(forzados)}
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        for y in sorted(a for a in todos if a.endswith((".yaml", ".yml"))):
            try:
                d = yaml.safe_load(_leer(y) or "")
            except Exception:
                continue
            n = d.get("names") if isinstance(d, dict) else None
            if isinstance(n, dict) and n:
                return {int(k): norm(v) for k, v in n.items()}
            if isinstance(n, list) and n:
                return {i: norm(v) for i, v in enumerate(n)}
    for c in sorted(a for a in todos if a.endswith(("classes.txt", ".names"))):
        ls = [l.strip() for l in (_leer(c) or "").splitlines() if l.strip()]
        if ls:
            return {i: norm(v) for i, v in enumerate(ls)}
    return {}


def _etiqueta_yolo(img: str, txts: set) -> Optional[str]:
    b = os.path.splitext(img)[0]
    if b + ".txt" in txts:
        return b + ".txt"
    partes = img.split(os.sep)
    if "images" in partes:
        i = len(partes) - 1 - partes[::-1].index("images")
        q = partes.copy()
        q[i] = "labels"
        c = os.path.splitext(os.sep.join(q))[0] + ".txt"
        if c in txts:
            return c
    return None


def _cajas_yolo(texto: str, nombres: Dict[int, str]) -> List[Tuple[str, float, float, float, float]]:
    out = []
    for linea in texto.splitlines():
        t = linea.split()
        if len(t) < 5:
            continue
        try:
            c = int(float(t[0]))
            v = [float(x) for x in t[1:]]
        except ValueError:
            continue
        if len(v) == 4:
            x, y, w, h = v
        elif len(v) >= 6 and len(v) % 2 == 0:          # polígono de segmentación
            xs, ys = v[0::2], v[1::2]
            x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        else:
            continue
        out.append((nombres.get(c, f"#{c}"), x, y, w, h))
    return out


def _cajas_voc(texto: str) -> List[Tuple[str, float, float, float, float]]:
    try:
        t = ET.fromstring(texto)
    except Exception:
        return []
    sw = float(t.findtext("size/width") or 0)
    sh = float(t.findtext("size/height") or 0)
    if not sw or not sh:
        return []
    out = []
    for o in t.findall("object"):
        b = o.find("bndbox")
        if b is None:
            continue
        x1, y1, x2, y2 = (float(b.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
        out.append((norm(o.findtext("name") or "?"), (x1 + x2) / 2 / sw, (y1 + y2) / 2 / sh,
                    (x2 - x1) / sw, (y2 - y1) / sh))
    return out


def recorrer(raiz: Path, f: Optional[dict] = None) -> List[Tuple[Path, Optional[list]]]:
    """(imagen, cajas) de una fuente. cajas=None: la foto no tiene etiqueta.

    f["formato"]: "auto" (YOLO, si no VOC, si no COCO), "yolo", "voc" o "coco".
    f["filtro"]: regex sobre la ruta relativa; solo entran las fotos que la cumplen.
    """
    f = f or {}
    todos = _listar(raiz)
    filtro = re.compile(f["filtro"]) if f.get("filtro") else None
    rel = lambda a: os.path.relpath(a, raiz).replace(os.sep, "/")
    imgs = sorted(a for a in todos if os.path.splitext(a)[1].lower() in EXT
                  and (filtro is None or filtro.search(rel(a))))
    set_imgs = set(imgs)
    formato = f.get("formato", "auto")
    out: Dict[str, list] = {}
    if formato in ("auto", "yolo"):
        txts = {a for a in todos if a.endswith(".txt")}
        nombres = _nombres_yolo(todos, f.get("nombres"))
        pares = [(im, _etiqueta_yolo(im, txts)) for im in imgs]
        pares = [(im, lb) for im, lb in pares if lb]
        for (im, _), tx in zip(pares, _en_paralelo(_leer, [lb for _, lb in pares])):
            if tx is not None:
                out[im] = _cajas_yolo(tx, nombres)
    if formato in ("auto", "voc"):
        xmls = {Path(a).stem: a for a in todos if a.lower().endswith(".xml")}
        faltan = [im for im in imgs if im not in out and Path(im).stem in xmls]
        for im, tx in zip(faltan, _en_paralelo(_leer, [xmls[Path(im).stem] for im in faltan])):
            if tx is not None:
                out[im] = _cajas_voc(tx)
    if formato in ("auto", "coco"):
        por_nombre: Dict[str, List[str]] = collections.defaultdict(list)
        for im in imgs:
            por_nombre[os.path.basename(im)].append(im)
        for j in sorted(a for a in todos if a.lower().endswith(".json")
                        and (filtro is None or filtro.search(rel(a)))):
            try:
                d = json.loads(_leer(j) or "")
            except Exception:
                continue
            if not (isinstance(d, dict) and {"images", "annotations", "categories"} <= set(d)):
                continue
            cats = {c["id"]: norm(c["name"]) for c in d["categories"]}
            anns = collections.defaultdict(list)
            for a in d["annotations"]:
                anns[a.get("image_id")].append(a)
            base_j = os.path.dirname(j)
            for info in d["images"]:
                fn = str(info.get("file_name", ""))
                cand = os.path.normpath(os.path.join(base_j, fn))
                if cand not in set_imgs:
                    # por nombre, pero solo cerca del json (su carpeta o la de al lado):
                    # dos splits pueden tener fotos con el mismo nombre
                    arriba = os.path.dirname(base_j) + os.sep
                    lst = [x for x in por_nombre.get(os.path.basename(fn), []) if x.startswith(arriba)]
                    mismos = [x for x in lst if os.path.dirname(x) == base_j]
                    cand = mismos[0] if mismos else (lst[0] if len(lst) == 1 else None)
                if cand is None or cand in out:
                    continue
                W, H = float(info.get("width") or 0), float(info.get("height") or 0)
                if not W or not H:
                    continue
                bx = []
                for a in anns.get(info.get("id"), []):
                    if not a.get("bbox"):
                        continue
                    x, y, w, h = (float(v) for v in a["bbox"][:4])
                    bx.append((cats.get(a.get("category_id"), f"#{a.get('category_id')}"),
                               (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
                out[cand] = bx
    return [(Path(im), out.get(im)) for im in imgs]


# --------------------------------------------------------------------------- #
# Hash perceptual (dHash 64 bits) e índice para buscar parecidos
# --------------------------------------------------------------------------- #
def _hash_pil(I) -> Tuple[int, int]:
    px = list(I.convert("L").resize((9, 8)).tobytes())
    h = hf = 0
    for fila in range(8):
        f = px[fila * 9:(fila + 1) * 9]
        for col in range(8):
            h = (h << 1) | (f[col] > f[col + 1])
        r = f[::-1]
        for col in range(8):
            hf = (hf << 1) | (r[col] > r[col + 1])
    return h, hf


def dhash(p: Path) -> Optional[Tuple[int, int]]:
    from PIL import Image
    try:
        with Image.open(p) as I:
            I.draft("L", (64, 64))
            return _hash_pil(I)
    except Exception:
        return None


def _copiar(args) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Lee la foto UNA vez: la copia achicada (lado mayor <= `lado`, EXIF aplicado,
    JPEG) al disco local y devuelve su dHash. Leer de /kaggle/input en cada época
    es lo lento; además una foto de 4000 px se decodifica 80 veces para usarla a 416.
    """
    k, origen, destino, lado = args
    from PIL import Image, ImageOps
    try:
        with Image.open(origen) as I:
            I.draft("RGB", (lado, lado))
            I = ImageOps.exif_transpose(I).convert("RGB")
            w, h = I.size
            if max(w, h) > lado:
                f = lado / max(w, h)
                I = I.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BILINEAR)
            I.save(destino, "JPEG", quality=90)
            return k, _hash_pil(I)
    except Exception:
        return k, None


class Indice:
    """Distancia de Hamming <= tol contra todo lo agregado (4 tablas de 16 bits)."""

    def __init__(self, tol: int = 5):
        self.tol = tol
        self.t = [collections.defaultdict(list) for _ in range(4)]

    def agregar(self, h: int) -> None:
        for k in range(4):
            self.t[k][(h >> (16 * k)) & 0xFFFF].append(h)

    def parecido(self, h: int) -> bool:
        for k in range(4):
            for o in self.t[k].get((h >> (16 * k)) & 0xFFFF, ()):
                if bin(o ^ h).count("1") <= self.tol:
                    return True
        return False


# --------------------------------------------------------------------------- #
def _grupo_video(p: Path, bloque: Optional[int] = None) -> str:
    """'video3_frame_0012' -> 'video3_frame': los cuadros de un video, juntos.

    Solo si lo que queda todavía identifica un video (tiene un número o dice
    frame/video/clip); 'nw123' sigue siendo su propio grupo, si no todos los
    negativos de una carpeta caerían del mismo lado.

    Con `bloque`: 'frame_0391' -> 'frame#13' (391 // 30). Las copias aumentadas
    de Roboflow (".rf.<hash>") de un mismo cuadro caen siempre juntas.
    """
    s = p.stem.split(".rf.")[0]
    s = re.sub(r"_(jpg|jpeg|png)$", "", s, flags=re.I)
    if bloque:
        m = re.search(r"(\d+)$", s)
        if m:
            return f"{p.parent.name}/{s[:m.start()].rstrip('_-. ')}#{int(m.group(1)) // bloque}"
        return f"{p.parent.name}/{s}"
    pref = re.sub(r"[\W_]*\d+$", "", s)
    if pref != s and (re.search(r"\d", pref) or re.search(r"frame|video|vid|clip|cam", pref, re.I)):
        s = pref
    return str(p.parent.name) + "/" + s


def _al_val(clave: str, frac: float) -> bool:
    import hashlib
    return int(hashlib.md5(clave.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def clasificar(cajas: Optional[list], f: dict, avisos: collections.Counter):
    """-> ("pos"|"neg"|None, filas 7 clases, motivo)"""
    if cajas is None:
        return (("neg", [], "") if f.get("vacias") == "negativo" else (None, [], "sin_etiqueta"))
    mapa = {norm(k): v for k, v in f.get("mapa", {}).items()}
    excluir = {norm(x) for x in f.get("excluir", [])}
    ignorar = {norm(x) for x in f.get("ignorar", [])}
    filas = []
    for n, x, y, w, h in cajas:
        n = norm(n)
        if n in excluir:
            return None, [], f"excluye:{n}"
        if n in ignorar or ("*" in ignorar and n not in mapa):
            continue
        if n not in mapa:
            avisos[n] += 1
            return None, [], f"desconocida:{n}"
        x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
        x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
        if x2 - x1 < 0.003 or y2 - y1 < 0.003:
            # Una pistola de menos de un píxel a 416 no se puede aprender, pero
            # tampoco se puede tirar la caja y dejar la foto como negativo.
            # Lo mismo una llama o un humo (07/11).
            if IDX[mapa[n]] in MINIMA_EXCLUYE:
                return None, [], f"{mapa[n]}_minima"
            continue
        if (f.get("lado_min") and IDX[mapa[n]] in MINIMA_EXCLUYE
                and ((x2 - x1) * (y2 - y1)) ** 0.5 < f["lado_min"]):
            return None, [], f"{mapa[n]}_chica"
        filas.append(f"{IDX[mapa[n]]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                     f"{x2 - x1:.6f} {y2 - y1:.6f}")
    if not filas and not cajas:
        return (("neg", [], "") if f.get("vacias") == "negativo" else (None, [], "vacia"))
    if any(int(r.split()[0]) == PISTOLA for r in filas):
        return "pos", filas, ""
    return "neg", filas, ""


def _hoja(destino: Path, fotos: list) -> None:
    """12 fotos con sus cajas, JPEG en base64 de menos de 100 KB: el visor de
    Kaggle muestra hasta ahí. Cada foto es (img, filas) o (img, filas, crudas):
    las crudas son cajas ignoradas, en amarillo con el nombre original.
    Pistola en rojo, llama naranja, humo gris claro, el resto azul."""
    import base64
    import io
    from PIL import Image, ImageDraw
    hoja = Image.new("RGB", (4 * 240, 3 * 180), (20, 20, 20))
    for k, item in enumerate(fotos[:12]):
        img, filas = item[0], item[1]
        crudas = item[2] if len(item) > 2 else []
        try:
            with Image.open(img) as I:
                I.draft("RGB", (480, 360))
                I = I.convert("RGB").resize((240, 180))
        except Exception:
            continue
        d = ImageDraw.Draw(I)
        cajas = [(CLASES_7[int(c)], COLOR.get(int(c), (60, 140, 255)), float(cx), float(cy), float(w), float(h))
                 for c, cx, cy, w, h in (x.split() for x in filas)]
        cajas += [(str(n)[:12], (255, 255, 0), x, y, w, h) for n, x, y, w, h in crudas]
        for nom, col, cx, cy, w, h in cajas:
            d.rectangle([(cx - w / 2) * 240, (cy - h / 2) * 180, (cx + w / 2) * 240, (cy + h / 2) * 180],
                        outline=col, width=2)
            d.text(((cx - w / 2) * 240 + 2, max(0, (cy - h / 2) * 180 - 10)), nom, fill=col)
        hoja.paste(I, ((k % 4) * 240, (k // 4) * 180))
    destino.parent.mkdir(parents=True, exist_ok=True)
    for q in (50, 40, 30):
        b = io.BytesIO()
        hoja.save(b, "JPEG", quality=q)
        if len(b.getvalue()) * 4 / 3 < 100_000:
            break
    destino.write_text(base64.b64encode(b.getvalue()).decode())


def armar(config: dict, salida: Path, mezcla: Optional[Path], base: Path,
          frac_val: float = 0.10, semilla: int = 0, tol: int = 5,
          lado: int = 1024, procesos: Optional[int] = None, revisar: bool = False) -> dict:
    """revisar=True: solo clasifica y cuenta, y deja una hoja de muestras por fuente
    (pistola en rojo, el resto en azul) en salida/revision/. No copia nada."""
    from multiprocessing import Pool
    rnd = random.Random(semilla)
    if salida.exists():
        shutil.rmtree(salida)
    for parte in ("train", "val", "prueba"):
        (salida / "images" / parte).mkdir(parents=True, exist_ok=True)
        (salida / "labels" / parte).mkdir(parents=True, exist_ok=True)
    tmp = salida / "_copias"
    tmp.mkdir()

    # 1) el val de mezcla_v1: nada nuevo puede parecerse a eso
    idx_mezcla_val = Indice(tol)
    if mezcla is not None and not revisar:
        for d in (mezcla / "images" / "val", mezcla / "val" / "images"):
            if d.is_dir():
                for p in d.iterdir():
                    if p.suffix.lower() in EXT:
                        h = dhash(p)
                        if h:
                            idx_mezcla_val.agregar(h[0])
                print(f"[ext] val de mezcla_v1 indexado: {d}", flush=True)
                break

    resumen = {}
    candidatas = []        # (fuente, img, filas)
    fuentes = config["fuentes"]
    # las de train primero: lo de prueba se compara contra todo lo de train
    fuentes = sorted(fuentes, key=lambda f: f.get("rol", "train") != "train")
    for f in fuentes:
        raiz = Path(f["ruta"])
        if not raiz.is_absolute():
            raiz = base / raiz
        r = collections.Counter()
        avisos = collections.Counter()
        if not raiz.exists():
            print(f"[ext] ⚠ {f['alias']}: no existe {raiz}; la salteo", flush=True)
            resumen[f["alias"]] = {"error": f"no existe {raiz}"}
            continue
        pos, neg = [], []
        siempre = {norm(x) for x in f.get("siempre", [])}
        ign_set = {norm(x) for x in f.get("ignorar", [])}
        mapa_set = {norm(x) for x in f.get("mapa", {})}
        fijas = []                  # (img, filas, tipo): negativos difíciles, sin tope
        con_ignoradas, vacias = [], []   # solo para las hojas de --revisar
        for img, cajas in recorrer(raiz, f):
            r["fotos"] += 1
            if revisar and cajas is not None:
                if not cajas:
                    vacias.append((img, []))
                else:
                    ign = [c for c in cajas if norm(c[0]) in ign_set
                           or ("*" in ign_set and norm(c[0]) not in mapa_set)]
                    if ign:
                        con_ignoradas.append((img, None, ign))
            tipo, filas, motivo = clasificar(cajas, f, avisos)
            if tipo is None:
                r[motivo.split(":")[0]] += 1
                if motivo.startswith("excluye"):
                    r[motivo] += 1
                continue
            if siempre and cajas and any(norm(c[0]) in siempre for c in cajas):
                fijas.append((img, filas, tipo))
                continue
            (pos if tipo == "pos" else neg).append((img, filas))
        rnd.shuffle(pos)
        rnd.shuffle(neg)
        pos = pos[: f.get("tope_pos", len(pos))]
        neg = neg[: f.get("tope_neg", len(neg))]
        if fijas:
            r["fijas_sin_tope"] = len(fijas)
            pos += [(i, fl) for i, fl, t in fijas if t == "pos"]
            neg += [(i, fl) for i, fl, t in fijas if t == "neg"]
        candidatas += [(f, img, filas) for img, filas in pos + neg]
        r["pos_elegidas"], r["neg_elegidas"] = len(pos), len(neg)
        if avisos:
            r["_clases_desconocidas"] = dict(avisos.most_common(12))
        cl = collections.Counter(CLASES_7[int(x.split()[0])] for _, fs in pos + neg for x in fs)
        r["_cajas_elegidas"] = dict(cl)
        lados = sorted((float(x.split()[3]) * float(x.split()[4])) ** 0.5
                       for _, fs in pos for x in fs if int(x.split()[0]) == PISTOLA)
        if lados:
            r["_pistola_lado_mediano"] = round(lados[len(lados) // 2], 3)
        for nc in ("llama", "humo"):
            ls = sorted((float(x.split()[3]) * float(x.split()[4])) ** 0.5
                        for _, fs in pos + neg for x in fs if int(x.split()[0]) == IDX[nc])
            if ls:
                r[f"_{nc}_lado_mediano"] = round(ls[len(ls) // 2], 3)
                r[f"_{nc}_chicas_<0.05"] = round(sum(v < 0.05 for v in ls) / len(ls), 3)
        con = sum(1 for _, fs in pos + neg if fs)
        r["fotos_con_cajas"], r["fotos_sin_cajas"] = con, len(pos) + len(neg) - con
        resumen[f["alias"]] = r
        print(f"[ext] {f['alias']}: " + json.dumps(dict(r), ensure_ascii=False), flush=True)
        if revisar:
            if pos:
                muestra = rnd.sample(pos, min(8, len(pos))) + rnd.sample(neg, min(4, len(neg)))
            else:
                # sin pistolas (fuego): 9 con cajas y 3 sin, si hay
                con_c = [x for x in neg if x[1]]
                sin_c = [x for x in neg if not x[1]]
                k_sin = min(3, len(sin_c))
                muestra = rnd.sample(con_c, min(12 - k_sin, len(con_c))) + rnd.sample(sin_c, k_sin)
            _hoja(salida / "revision" / f"hoja_{f['alias']}.txt", muestra)
            if con_ignoradas:
                _hoja(salida / "revision" / f"hoja_{f['alias']}_ignoradas.txt",
                      [(i, [], ig) for i, _, ig in rnd.sample(con_ignoradas, min(12, len(con_ignoradas)))])
                r["revision_con_ignoradas"] = len(con_ignoradas)
            if vacias:
                _hoja(salida / "revision" / f"hoja_{f['alias']}_vacias.txt",
                      rnd.sample(vacias, min(12, len(vacias))))
                r["revision_etiqueta_vacia"] = len(vacias)

    if revisar:
        for alias in list(resumen):
            resumen[alias] = dict(resumen[alias])
        (salida / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False))
        return resumen

    # copias locales + hash, en paralelo
    trabajos = [(k, str(img), str(tmp / f"{k:07d}.jpg"), lado)
                for k, (_, img, _) in enumerate(candidatas)]
    hashes: Dict[int, Optional[Tuple[int, int]]] = {}
    with Pool(procesos or os.cpu_count() or 2) as pool:
        for i, (k, h) in enumerate(pool.imap_unordered(_copiar, trabajos, chunksize=16)):
            hashes[k] = h
            if (i + 1) % 2000 == 0:
                print(f"[ext] copiadas {i + 1}/{len(trabajos)}", flush=True)

    elegidas = []          # (fuente, rol_final, copia, filas, hash)
    vistos_exactos = set()
    for k, (f, img, filas) in enumerate(candidatas):
        r = resumen[f["alias"]]
        h = hashes.get(k)
        copia = tmp / f"{k:07d}.jpg"
        if h is None:
            r["ilegible"] += 1
            continue
        if h[0] in vistos_exactos:
            r["duplicada"] += 1
            continue
        if idx_mezcla_val.parecido(h[0]) or idx_mezcla_val.parecido(h[1]):
            r["igual_a_val_mezcla"] += 1
            continue
        vistos_exactos.add(h[0])
        rol = f.get("rol", "train")
        if rol == "train":
            clave = _grupo_video(img, f.get("bloque")) if f.get("video") else str(img)
            rol = "val" if _al_val(f["alias"] + "|" + clave, frac_val) else "train"
        elegidas.append((f["alias"], rol, copia, filas, h))
    for alias in list(resumen):
        resumen[alias] = dict(resumen[alias])

    # 2) val y prueba no pueden parecerse a nada de train
    idx_train = Indice(tol)
    for _, rol, _, _, h in elegidas:
        if rol == "train":
            idx_train.agregar(h[0])
            idx_train.agregar(h[1])
    final = collections.defaultdict(collections.Counter)
    n = 0
    for alias, rol, copia, filas, h in elegidas:
        if rol != "train" and idx_train.parecido(h[0]):
            final[alias][f"{rol}_parecida_a_train"] += 1
            continue
        nombre = f"{alias}__{n:06d}.jpg"
        n += 1
        destino = salida / "images" / rol / nombre
        os.replace(copia, destino)
        (salida / "labels" / rol / (destino.stem + ".txt")).write_text("\n".join(filas))
        final[alias][f"{rol}_fotos"] += 1
        final[alias][f"{rol}_pistolas"] += sum(int(x.split()[0]) == PISTOLA for x in filas)
        final[alias][f"{rol}_sin_pistola"] += not any(int(x.split()[0]) == PISTOLA for x in filas)
    shutil.rmtree(tmp, ignore_errors=True)
    for alias, c in final.items():
        resumen.setdefault(alias, {}).update({"final": dict(c)})
    tot = collections.Counter()
    for c in final.values():
        tot.update(c)
    resumen["_total"] = dict(tot)
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False))
    print("[ext] total: " + json.dumps(dict(tot), ensure_ascii=False), flush=True)
    return resumen


# --------------------------------------------------------------------------- #
def _prueba(tmp: Path) -> int:
    """Tres fuentes sintéticas (YOLO, VOC, video) y una de prueba con un duplicado."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)

    def foto(p: Path, semilla: int):
        r = np.random.default_rng(semilla)
        a = (r.random((120, 160, 3)) * 255).astype("uint8")
        Image.fromarray(a).save(p)

    a = tmp / "in" / "a"           # YOLO con data.yaml, clases pistol/phone/rifle
    for s in ("train", "valid"):
        (a / s / "images").mkdir(parents=True)
        (a / s / "labels").mkdir(parents=True)
    (a / "data.yaml").write_text("names: ['Pistol', 'phone', 'rifle', 'person']\n")
    k = 0
    for s in ("train", "valid"):
        for i in range(30):
            foto(a / s / "images" / f"a{k}.jpg", k)
            c = [0, 1, 2, 3][i % 4]
            (a / s / "labels" / f"a{k}.txt").write_text(f"{c} 0.5 0.5 0.2 0.3\n")
            k += 1
    b = tmp / "in" / "b"           # VOC
    (b / "Annotations").mkdir(parents=True)
    (b / "JPEGImages").mkdir(parents=True)
    for i in range(20):
        foto(b / "JPEGImages" / f"b{i}.jpg", 1000 + i)
        (b / "Annotations" / f"b{i}.xml").write_text(
            f"<annotation><size><width>160</width><height>120</height></size><object>"
            f"<name>{'handgun' if i % 2 else 'wallet'}</name><bndbox><xmin>10</xmin><ymin>10</ymin>"
            f"<xmax>60</xmax><ymax>50</ymax></bndbox></object></annotation>")
    c = tmp / "in" / "c"           # cuadros de video + negativos sin etiqueta a propósito
    (c / "images").mkdir(parents=True)
    (c / "labels").mkdir(parents=True)
    (c / "no_weapon").mkdir(parents=True)
    (c / "classes.txt").write_text("gun\n")
    for v in range(6):
        for fr in range(5):
            foto(c / "images" / f"video{v}_frame_{fr:04d}.jpg", 2000 + v * 10 + fr)
            (c / "labels" / f"video{v}_frame_{fr:04d}.txt").write_text("0 0.4 0.4 0.1 0.1\n")
    for i in range(8):
        foto(c / "no_weapon" / f"nw{i}.jpg", 3000 + i)
    d = tmp / "in" / "d"           # prueba: 5 nuevas + 1 copia exacta de una de 'b'
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir(parents=True)
    (d / "classes.txt").write_text("pistol\n")
    for i in range(5):
        foto(d / "images" / f"d{i}.jpg", 4000 + i)
        (d / "labels" / f"d{i}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    shutil.copy2(b / "JPEGImages" / "b1.jpg", d / "images" / "copia.jpg")
    (d / "labels" / "copia.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    e = tmp / "in" / "e"           # fuego: 10 llamas, 5 solo "light", 5 llamas mínimas
    (e / "images").mkdir(parents=True)
    (e / "labels").mkdir(parents=True)
    (e / "data.yaml").write_text("names: ['Fire', 'light']\n")
    for i in range(22):            # + 2 llamas chicas (lado 0,012 < lado_min 0,02)
        foto(e / "images" / f"frame_{i:04d}_jpg.rf.{i:032x}.jpg", 5000 + i)
        fila = ("0 0.5 0.5 0.2 0.3" if i < 10 else "1 0.3 0.3 0.1 0.1" if i < 15
                else "0 0.5 0.5 0.001 0.002" if i < 20 else "0 0.5 0.5 0.012 0.012")
        (e / "labels" / f"frame_{i:04d}_jpg.rf.{i:032x}.txt").write_text(fila + "\n")

    cfg = {"fuentes": [
        {"alias": "a", "ruta": "a", "mapa": {"pistol": "pistola", "phone": "celular"},
         "excluir": ["rifle"], "ignorar": ["person"]},
        {"alias": "b", "ruta": "b", "mapa": {"handgun": "pistola"}},     # wallet desconocida
        {"alias": "c", "ruta": "c", "mapa": {"gun": "pistola"}, "vacias": "negativo",
         "video": True},
        {"alias": "d", "ruta": "d", "mapa": {"pistol": "pistola"}, "rol": "prueba"},
        {"alias": "e", "ruta": "e", "mapa": {"fire": "llama"}, "ignorar": ["light"],
         "siempre": ["light"], "tope_neg": 3, "video": True, "bloque": 30, "lado_min": 0.02},
    ]}
    res = armar(cfg, tmp / "salida", None, tmp / "in", frac_val=0.3)
    ok = True
    def chk(cond, msg):
        nonlocal ok
        print(("  ok  " if cond else "  MAL ") + msg)
        ok &= bool(cond)
    chk(res["a"].get("excluye") == 14, "a: 14 fotos con rifle excluidas")
    chk(res["a"]["pos_elegidas"] == 16 and res["a"]["neg_elegidas"] == 30,
        "a: 16 pistolas, 30 negativos (16 celular + 14 persona ignorada->vacía)")
    chk(res["b"].get("desconocida") == 10 and res["b"].get("_clases_desconocidas") == {"wallet": 10},
        "b: wallet no declarada -> 10 fotos afuera con aviso")
    chk(res["c"]["neg_elegidas"] == 8, "c: los 8 no_weapon entran como negativos")
    # split por video: todos los cuadros de un video en el mismo lado
    lados = collections.defaultdict(set)
    for v in range(6):
        for fr in range(5):
            img = tmp / "in" / "c" / "images" / f"video{v}_frame_{fr:04d}.jpg"
            lados[v].add(_al_val("c|" + _grupo_video(img), 0.3))
    chk(all(len(x) == 1 for x in lados.values()), "c: cada video queda entero en train o en val")
    chk(all(p.suffix == ".jpg" and not p.is_symlink() for p in (tmp / "salida" / "images").rglob("*.*")),
        "las fotos son copias locales en JPEG")
    chk(not (tmp / "salida" / "_copias").exists(), "no quedan copias temporales")
    chk(res["d"]["final"].get("prueba_fotos") == 5 and
        (res["d"].get("duplicada", 0) + res["d"]["final"].get("prueba_parecida_a_train", 0)) == 1,
        "d: la copia de una foto de train no entra a prueba")
    neg_lbl = [p for p in (tmp / "salida" / "labels" / "train").glob("a__*")
               if p.read_text().strip().startswith("5 ")]
    chk(len(neg_lbl) > 0, "a: los celulares quedan anotados como clase 5 (celular)")
    chk(res["e"].get("llama_minima") == 5, "e: las 5 llamas mínimas dejan la foto afuera")
    chk(res["e"].get("llama_chica") == 2, "e: las 2 llamas de lado < lado_min también")
    chk(res["e"].get("fijas_sin_tope") == 5 and res["e"]["neg_elegidas"] == 8,
        "e: tope 3 de llamas + las 5 de 'light' (sin tope)")
    e_lbl = sorted((tmp / "salida" / "labels").rglob("e__*.txt"))
    chk(sum(1 for p in e_lbl if p.read_text().strip().startswith("1 ")) == 3
        and sum(1 for p in e_lbl if not p.read_text().strip()) == 5,
        "e: 3 fotos con llama (clase 1) y 5 vacías (la luz no se anota)")
    g = {_grupo_video(Path(f"x/images/frame_{i:04d}_jpg.rf.ab.jpg"), 30) for i in range(60)}
    chk(g == {"images/frame#0", "images/frame#1"}, f"bloque 30: 60 cuadros en 2 grupos ({sorted(g)})")
    chk(_grupo_video(Path("x/images/frame_0391_jpg.rf.ab.jpg")) == "images/frame",
        "sin bloque, el grupo de video de siempre")
    print("PRUEBA", "OK" if ok else "FALLA")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="armas_ext.json")
    ap.add_argument("--salida", default="datasets/armas_ext")
    ap.add_argument("--mezcla", default="datasets/mezcla_v1",
                    help="para sacar lo que se parezca a su val; '' para no usarla")
    ap.add_argument("--base", default="/kaggle/input")
    ap.add_argument("--frac-val", dest="frac_val", type=float, default=0.10)
    ap.add_argument("--prueba", action="store_true")
    ap.add_argument("--revisar", action="store_true",
                    help="solo cuenta y deja hojas de muestras en <salida>/revision/")
    a = ap.parse_args()
    if a.prueba:
        import tempfile
        return _prueba(Path(tempfile.mkdtemp()))
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    mezcla = Path(a.mezcla) if a.mezcla else None
    armar(cfg, Path(a.salida), mezcla if mezcla and mezcla.exists() else None,
          Path(a.base), frac_val=a.frac_val, revisar=a.revisar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
