# -*- coding: utf-8 -*-
"""
entrenar_segmentacion_cuda.py · HORUS — entrenamiento de la cabeza de segmentacion.

Clases: 0 fondo · 1 agua · 2 humo · 3 fuego.  Mismo backbone congelado, mismos
nombres de checkpoint y mismos flags que `objetos/entrenar_objetos_cuda.py`.

Que se optimizo
---------------
1. **Solo `extractor` + `fpn`, nunca `SharedBackbone.forward()`.** El forward
   completo tambien calcula el embedding, el GRU temporal y la novedad, y esa
   ultima hace `float(...norm())`: un sync GPU->CPU por batch, ademas de
   acumular estado por camara que en entrenamiento no sirve para nada. Se
   llama `backbone.fpn(backbone.extractor(x))`, igual que hace el camino de
   objetos.
2. **Backbone congelado en `no_grad` + `eval`.** Sin activaciones intermedias
   guardadas, la VRAM del batch la dicta la cabeza sola.
3. **Normalizacion en GPU.** El DataLoader devuelve uint8 (442 KB por foto de
   384) en vez de float32 (1,7 MB): 4x menos trafico por PCIe.
4. **BF16 en vez de FP16.** Mismo throughput en Ada/Ampere, sin GradScaler ni
   NaN. Con `--precision fp16` volves al scaler.
5. channels_last + TF32 + cuDNN benchmark, AdamW fusionado, `--compile`.
6. Acumulacion de gradiente, EMA, warmup+cosine, clipping, early stopping,
   reanudacion de verdad (optimizador, scheduler, scaler y epoca).
7. **mIoU por epoca, no la perdida.** En segmentacion la perdida de validacion
   dice poco: con 90% de fondo, un modelo que predice fondo siempre tiene
   buena perdida y mIoU de riesgo cero.
8. **El dataset se revisa antes de la epoca 1** y se aborta si algo no cierra,
   en vez de entrenar toda la noche contra etiquetas rotas.

Lo que NO se hizo, a proposito
------------------------------
**No se cachean features.** En objetos convenia (~3 MB por foto). Aca p3 a 384
son 48x48x256 y hay tres niveles: ~1 MB por foto en fp16, pero por *cada*
recorte aleatorio. Cachear mataria el aumento de datos, que es lo que sostiene
las clases con pocas imagenes.

Uso
---
    python entrenar_segmentacion_cuda.py --revisar
    python entrenar_segmentacion_cuda.py --autotest
    python entrenar_segmentacion_cuda.py --epocas 60 --batch 16
    python entrenar_segmentacion_cuda.py --reanudar checkpoints/head_last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import copy
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError:
    sys.exit("Falta PyTorch. Con CUDA:\n"
             "  py -3.13 -m pip install torch torchvision "
             "--index-url https://download.pytorch.org/whl/cu126")

from PIL import Image

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(os.path.dirname(_AQUI))
for _p in (os.path.join(_RAIZ, "03_backbone"), _AQUI):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared_backbone import BackboneConfig, SharedBackbone  # noqa: E402
from segmentation_head import (  # noqa: E402
    DEFAULT_CLASSES, IGNORE_INDEX, SegmentationHead, SegmentationHeadConfig,
)

Image.MAX_IMAGE_PIXELS = None

TAM = 384                                   # = BackboneConfig.input_size
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_FPN_LEVELS = ("p3", "p4", "p5")

CLASES = list(DEFAULT_CLASSES)
C = len(CLASES)

# Remapeo old->new cuando se entrena con un subconjunto de clases (--clases).
# 255 = la clase se ignora. Se llena en main(); por defecto es la identidad.
REMAP = np.arange(256, dtype=np.uint8)


def fijar_clases(pedidas: list[str]) -> None:
    """Entrena con un subconjunto de clases sin rehacer el dataset.

    El dataset en disco siempre tiene las 4. Lo que cambia es que las clases
    que no se piden pasan a 255 (ignorar): sus pixeles no supervisan nada y el
    modelo no tiene un logit para ellas. Sirve para recortar alcance sin perder
    el trabajo de conversion, y es reversible con solo cambiar el flag.
    """
    global CLASES, C, REMAP
    faltan = [c for c in pedidas if c not in DEFAULT_CLASSES]
    if faltan:
        sys.exit(f"clases desconocidas: {faltan}. Validas: {list(DEFAULT_CLASSES)}")
    if pedidas[0] != DEFAULT_CLASSES[0]:
        sys.exit(f"la primera clase tiene que ser '{DEFAULT_CLASSES[0]}': es la que "
                 f"comparten todas las fuentes y ata las escalas entre datasets")
    REMAP = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for nuevo, nombre in enumerate(pedidas):
        REMAP[DEFAULT_CLASSES.index(nombre)] = nuevo
    REMAP[IGNORE_INDEX] = IGNORE_INDEX
    CLASES, C = list(pedidas), len(pedidas)


def abrir_mascara(p) -> Image.Image:
    """Mascara como modo L con los INDICES crudos.

    Nuestras mascaras son PNG con paleta. `convert("L")` sobre un modo P
    devuelve la luminancia del color, no el indice: la clase 2 sale 170 y la
    255 sale 105. Entrenar con eso no falla, entrena contra basura.
    """
    im = Image.open(p)
    if im.mode == "P":
        return Image.fromarray(np.asarray(im, dtype=np.uint8), mode="L")
    return im if im.mode == "L" else im.convert("L")


# --------------------------------------------------------------------------- #
# Dataset: devuelve uint8, la normalizacion se hace en GPU
# --------------------------------------------------------------------------- #
@dataclass
class Muestra:
    img: Path
    msk: Path
    anotadas: Tuple[int, ...]
    fuente: str


class DatasetSeg(Dataset):
    """Lee `mezcla_seg_v1` (ver descargar_datasets_segmentacion.py).

    Geometria: el camino caliente pasa por `FramePreprocessor`, que reescala el
    frame entero a 384x384 **deformando el aspecto** (un 1280x720 se comprime
    1,78x mas en horizontal que en vertical). Entrenar con recortes cuadrados
    le ensenaria al modelo una geometria que en produccion no ve nunca.
      * train: recorte aleatorio con area y aspecto variables, reescalado a
        384x384 -- cubre la deformacion de produccion y ademas aumenta.
      * val:   el frame entero a 384x384, exactamente como FramePreprocessor.
    """

    def __init__(self, raiz: Path, split: str, tam: int = TAM,
                 aumentar: bool = False, con_fuente: bool = False):
        man = raiz / "manifiesto.jsonl"
        if not man.exists():
            sys.exit(f"No existe {man}\nArmalo con:  "
                     f"python descargar_datasets_segmentacion.py --todo")
        self.items: List[Muestra] = []
        for linea in man.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            r = json.loads(linea)
            if r["split"] != split:
                continue
            self.items.append(Muestra(raiz / r["img"], raiz / r["msk"],
                                      tuple(r["anotadas"]), r["fuente"]))
        if not self.items:
            sys.exit(f"El split '{split}' quedo vacio en {raiz}")
        self.tam = tam
        self.aumentar = aumentar
        self.con_fuente = con_fuente
        self.fuentes = sorted({i.fuente for i in self.items})
        self._idx_f = {n: i for i, n in enumerate(self.fuentes)}

    def __len__(self) -> int:
        return len(self.items)

    def _caja(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """Recorte estilo Inception: area 40-100%, aspecto 1/2 a 2."""
        for _ in range(10):
            area = w * h * random.uniform(0.4, 1.0)
            rel = math.exp(random.uniform(math.log(0.5), math.log(2.0)))
            cw, ch = int(round(math.sqrt(area * rel))), int(round(math.sqrt(area / rel)))
            if cw <= w and ch <= h:
                x = random.randint(0, w - cw)
                y = random.randint(0, h - ch)
                return x, y, x + cw, y + ch
        return 0, 0, w, h

    def __getitem__(self, i: int):
        it = self.items[i]
        img = Image.open(it.img).convert("RGB")
        msk = abrir_mascara(it.msk)
        t = self.tam
        if self.aumentar:
            caja = self._caja(img.width, img.height)
            img = img.resize((t, t), Image.BILINEAR, box=caja)
            msk = msk.resize((t, t), Image.NEAREST, box=caja)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.resize((t, t), Image.BILINEAR)
            msk = msk.resize((t, t), Image.NEAREST)

        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)
        y = torch.from_numpy(REMAP[np.asarray(msk, dtype=np.uint8)].copy()).long()
        anot = torch.zeros(C, dtype=torch.bool)
        for c in it.anotadas:
            if REMAP[c] != IGNORE_INDEX:
                anot[int(REMAP[c])] = True
        if self.con_fuente:
            return x, y, anot, self._idx_f[it.fuente]
        return x, y, anot


def crear_sampler(ds: DatasetSeg) -> WeightedRandomSampler:
    """Sube la probabilidad de las fuentes chicas.

    Sin esto COCO-Stuff, que aporta la mayoria de las muestras, hace que el
    fuego aparezca una vez cada varias epocas efectivas.
    """
    cuenta = Counter(i.fuente for i in ds.items)
    peso = {k: (1.0 / v) ** 0.5 for k, v in cuenta.items()}
    w = torch.tensor([peso[i.fuente] for i in ds.items], dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(ds), replacement=True)


# --------------------------------------------------------------------------- #
# Metrica
# --------------------------------------------------------------------------- #
class Confusion:
    """Matriz de confusion acumulada en GPU con bincount.

    Nada de `.item()` por batch: cada uno seria un sync GPU->CPU, el mismo
    problema que tenia `predict()` en el camino original de objetos.
    """

    def __init__(self, dev):
        self.cm = torch.zeros(C, C, dtype=torch.long, device=dev)

    def sumar(self, pred: Tensor, y: Tensor) -> None:
        k = y != IGNORE_INDEX
        self.cm += torch.bincount((y[k] * C + pred[k]).flatten(),
                                  minlength=C * C).reshape(C, C)

    def iou(self) -> np.ndarray:
        cm = self.cm.double()
        inter = cm.diag()
        union = cm.sum(0) + cm.sum(1) - inter
        return (inter / union.clamp(min=1)).cpu().numpy()


class IoUPorImagen:
    """IoU promediado POR IMAGEN, no sobre el pozo de pixeles.

    El IoU global es una metrica de masa: la mitad de los pixeles de humo de
    validacion vive en 90 imagenes de 1.124, asi que ese numero lo deciden 90
    fotos y salta cada vez que el modelo cambia de opinion en dos de ellas.
    Medido sobre v3, el humo se mueve 0,044 por epoca contra 0,012 del fuego:
    3,7x mas volatil, y no porque el modelo sea 3,7x mas inestable.

    Promediar por imagen le da el mismo peso a una foto con humo enorme que a
    una con una columna fina. Es mas estable y mas parecido a "en que fraccion
    de las camaras acierto". Se reportan los dos: si saltan juntos cambio el
    modelo, si salta solo el global fue un puñado de fotos grandes.
    """

    def __init__(self, dev):
        self.suma = torch.zeros(C, dtype=torch.double, device=dev)
        self.cuenta = torch.zeros(C, dtype=torch.double, device=dev)

    def sumar(self, pred: Tensor, y: Tensor) -> None:
        val = y != IGNORE_INDEX
        for c in range(C):
            pc, yc = (pred == c) & val, (y == c) & val
            inter = (pc & yc).sum(dim=(1, 2)).double()
            union = (pc | yc).sum(dim=(1, 2)).double()
            hay = union > 0
            self.suma[c] += torch.where(hay, inter / union.clamp(min=1),
                                        torch.zeros_like(union)).sum()
            self.cuenta[c] += hay.sum()

    def iou(self) -> np.ndarray:
        return (self.suma / self.cuenta.clamp(min=1)).cpu().numpy()


# --------------------------------------------------------------------------- #
# Revision del dataset (se corre SIEMPRE antes de la epoca 1)
# --------------------------------------------------------------------------- #
def revisar_dataset(raiz: Path, muestras: int = 400) -> Tuple[bool, np.ndarray]:
    man = raiz / "manifiesto.jsonl"
    if not man.exists():
        print(f"[XX] no existe {man}")
        return False, np.zeros(C)
    filas = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    if not filas:
        print("[XX] el manifiesto esta vacio")
        return False, np.zeros(C)

    fatal: List[str] = []
    aviso: List[str] = []
    splits = Counter(r["split"] for r in filas)
    if splits.get("val", 0) < 50:
        aviso.append(f"solo {splits.get('val', 0)} muestras de validacion")

    # La frecuencia se mide sobre las imagenes donde la clase APARECE, no sobre
    # las que la ANOTAN.
    #
    # Medirla sobre lo anotado tiene un efecto perverso: al declarar que 30.000
    # fotos de agua tampoco tienen fuego, el denominador de fuego crece 5,6x sin
    # que aparezca un solo pixel nuevo, la formula lee "el fuego se volvio 5,6x
    # mas raro" y le dispara el peso. Eso paso de verdad: fuego 2,11 -> 2,76 y
    # fondo 0,34 -> 0,19, y hundio el IoU de agua un 28% aunque las etiquetas de
    # agua no habian cambiado ni un byte.
    #
    # Esta definicion (Eigen & Fergus) es invariante a ampliar `anotadas`.
    px = np.zeros(C)
    px_pres = np.zeros(C)
    imgs = np.zeros(C, dtype=np.int64)
    for r in filas:
        tot = sum(int(v) for v in r["px"].values()) + r.get("px_ign", 0)
        for c, n in r["px"].items():
            nc = int(REMAP[int(c)])
            if nc == IGNORE_INDEX:
                continue
            px_pres[nc] += tot
            px[nc] += int(n)
            imgs[nc] += 1

    for c in range(1, C):
        if imgs[c] == 0:
            fatal.append(f"la clase {c} ({CLASES[c]}) no aparece en NINGUNA muestra")
        elif imgs[c] < 100:
            aviso.append(f"clase {c} ({CLASES[c]}): solo {imgs[c]} imagenes")

    for r in filas:
        if not set(r["anotadas"]) <= set(range(len(DEFAULT_CLASSES))) or 0 not in r["anotadas"]:
            fatal.append(f"{r['img']}: 'anotadas'={r['anotadas']} invalido "
                         f"(subconjunto de 0..{C - 1} y tiene que incluir el 0)")
            break

    for r in random.Random(7).sample(filas, min(muestras, len(filas))):
        pi, pm = raiz / r["img"], raiz / r["msk"]
        if not pi.exists() or not pm.exists():
            fatal.append(f"falta en disco: {r['img']} o su mascara")
            break
        try:
            m = np.asarray(abrir_mascara(pm))
            im = Image.open(pi)
        except Exception as e:
            fatal.append(f"{r['img']}: no se pudo abrir ({e})")
            break
        if im.size != (m.shape[1], m.shape[0]):
            fatal.append(f"{r['img']}: imagen {im.size} y mascara "
                         f"{(m.shape[1], m.shape[0])} no coinciden")
            break
        vals = set(np.unique(m).tolist()) - {IGNORE_INDEX}
        if not vals <= set(range(len(DEFAULT_CLASSES))):
            fatal.append(f"{r['msk']}: valores fuera de 0..{len(DEFAULT_CLASSES) - 1}: "
                         f"{sorted(vals)}")
            break
        if vals - set(r["anotadas"]):
            fatal.append(f"{r['msk']}: usa clases {sorted(vals - set(r['anotadas']))} "
                         f"que su fuente no declara en 'anotadas'")
            break

    frec = px / np.maximum(px_pres, 1)
    vivas = frec[1:][frec[1:] > 0]
    if len(vivas) > 1 and vivas.max() / vivas.min() > 8:
        aviso.append(f"desbalance de {vivas.max() / vivas.min():.0f}x entre clases "
                     f"positivas; se compensa con pesos, pero conviene topear la "
                     f"fuente dominante con --tope-fuente al armar el dataset")

    print(f"\n[datos] {len(filas)} muestras  "
          f"(train {splits.get('train', 0)} / val {splits.get('val', 0)})")
    for c in range(C):
        print(f"        {c} {CLASES[c]:<6} {100 * frec[c]:6.3f}% de pixeles  "
              f"en {imgs[c]:>7,} imagenes")
    for a in aviso:
        print(f"[!!] {a}")
    for f in fatal:
        print(f"[XX] {f}")
    if fatal:
        print("[XX] dataset rechazado: corregilo antes de entrenar")
        return False, frec
    print("[OK] dataset aceptado")
    return True, frec


def pesos_de_clase(frec: np.ndarray) -> Tensor:
    """Median frequency balancing con raiz: compensa sin volverse inestable."""
    f = np.where(frec > 0, frec, np.nan)
    med = np.nanmedian(f)
    w = np.sqrt(med / np.where(np.isnan(f), med, f))
    w = np.clip(w, 0.2, 8.0)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def jitter_color(x: Tensor) -> Tensor:
    """Brillo/contraste/saturacion por muestra, vectorizado en GPU, sobre 0..1.

    Antes de normalizar: sobre el tensor ya normalizado, "multiplicar por 1,2"
    no es subir el brillo, es escalar una distancia a la media de ImageNet.
    Las camaras de vigilancia cambian exposicion y balance de blancos todo el
    tiempo; sin esto el modelo se ata al color exacto del dataset.
    """
    b = x.shape[0]
    def f(lo, hi):
        return torch.empty(b, 1, 1, 1, device=x.device).uniform_(lo, hi)
    x = x * f(0.75, 1.25)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * f(0.75, 1.25) + m
    g = x.mean(dim=1, keepdim=True)
    return ((x - g) * f(0.7, 1.3) + g).clamp_(0.0, 1.0)


def main() -> float:
    ap = argparse.ArgumentParser(
        description="Entrena la cabeza de segmentacion de Horus (fondo/agua/humo/fuego).")
    ap.add_argument("--dataset", default="datasets/mezcla_seg_v1")
    ap.add_argument("--tam", type=int, default=TAM)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--acumular", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 8)))
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--channels-last", action="store_true", default=True)
    ap.add_argument("--ema", type=float, default=0.999, help="0 desactiva")
    ap.add_argument("--peso-dice", type=float, default=0.5)
    ap.add_argument("--clases", default=",".join(DEFAULT_CLASSES),
                    help="subconjunto de clases a entrenar, ej \"fondo,agua,fuego\". "
                         "No hace falta rehacer el dataset: las que no estan pasan "
                         "a ignorar. 'fondo' tiene que ir siempre primero.")
    ap.add_argument("--afinar", choices=["no", "privado", "compartido"], default="no",
                    help="descongelar la ultima etapa del backbone. 'privado' "
                         "(recomendado) le da a esta cabeza una copia propia de "
                         "layer4+FPN y deja el tronco compartido intacto: +39%% de "
                         "computo en el camino caliente en vez de +100%%. "
                         "'compartido' modifica el backbone que usan las 4 cabezas.")
    ap.add_argument("--lr-backbone", type=float, default=None,
                    help="lr de la parte afinada (default: --lr / 10)")
    ap.add_argument("--pesos", help="fija los pesos por clase a mano, ej "
                                    "\"0.34,0.60,0.96,2.11\"; util para comparar corridas")
    ap.add_argument("--paciencia", type=int, default=12)
    ap.add_argument("--reanudar")
    ap.add_argument("--revisar", action="store_true", help="valida el dataset y sale")
    ap.add_argument("--autotest", action="store_true", help="prueba el camino sin dataset")
    ap.add_argument("--diagnostico", metavar="CKPT",
                    help="carga un checkpoint, corre validacion y desglosa POR FUENTE "
                         "donde se equivoca; sirve para separar falsos positivos de "
                         "detecciones perdidas")
    ap.add_argument("--semilla", type=int, default=1234)
    a = ap.parse_args()

    fijar_clases([x.strip() for x in a.clases.split(",") if x.strip()])
    if C != len(DEFAULT_CLASSES):
        print(f"[setup] entrenando {C} clases: {CLASES}  "
              f"(fuera: {[c for c in DEFAULT_CLASSES if c not in CLASES]})")

    random.seed(a.semilla); np.random.seed(a.semilla)
    torch.manual_seed(a.semilla); torch.cuda.manual_seed_all(a.semilla)

    dir_base = Path(_AQUI)
    raiz_ds = Path(a.dataset)
    if not raiz_ds.is_absolute():
        raiz_ds = dir_base / raiz_ds

    # Se chequea ANTES de bajar pesos de ImageNet y construir el modelo: si el
    # checkpoint no esta, enterarse despues de dos minutos de setup es peor.
    if a.reanudar:
        a.reanudar = Path(a.reanudar)
        if not a.reanudar.is_absolute():
            a.reanudar = dir_base / a.reanudar
        if not a.reanudar.exists():
            sys.exit(
                f"No existe {a.reanudar}\n\n"
                f"--reanudar continua un entrenamiento que YA corrio. Si es la\n"
                f"primera vez, sacaselo:\n\n"
                f"  python entrenar_segmentacion_cuda.py --epocas {a.epocas} "
                f"--batch {a.batch} --workers {a.workers}\n")

    if a.revisar:
        return 0.0 if revisar_dataset(raiz_ds)[0] else -1.0

    # --- placa ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        p = torch.cuda.get_device_properties(0)
        print(f"[gpu] {p.name}  {p.total_memory / 1e9:.0f} GB  sm_{p.major}{p.minor}  "
              f"torch {torch.__version__}  cuda {torch.version.cuda}")
        if a.precision == "bf16" and not torch.cuda.is_bf16_supported():
            print("[!!] la placa no soporta bf16: paso a fp16")
            a.precision = "fp16"
    else:
        print("[!!] sin CUDA: entrena en CPU (lento, solo para probar)")
        if os.environ.get("HORUS_AMP_CPU") != "1":
            a.precision = "fp32"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[a.precision]
    # HORUS_AMP_CPU=1 fuerza autocast en CPU. No sirve para entrenar (es mas
    # lento), sirve para que `prueba_segmentacion.py` recorra el mismo camino
    # de dtypes que la GPU: sin esto, un tensor bf16 entrando a una capa fp32
    # solo revienta en la placa del usuario.
    usar_amp = a.precision != "fp32" and (
        device.type == "cuda" or os.environ.get("HORUS_AMP_CPU") == "1")

    # --- modelo -----------------------------------------------------------
    backbone = SharedBackbone(
        BackboneConfig(pretrained=not a.autotest, input_size=a.tam,
                       freeze_encoder=True)
    ).to(device).eval()
    for p_ in backbone.parameters():
        p_.requires_grad_(False)

    cfg = SegmentationHeadConfig(classes=tuple(CLASES), fpn_levels=_FPN_LEVELS,
                                 in_channels=backbone.cfg.fpn_dim,
                                 dice_weight=a.peso_dice)
    head = SegmentationHead(cfg).to(device)
    if a.channels_last:
        backbone = backbone.to(memory_format=torch.channels_last)
        head = head.to(memory_format=torch.channels_last)
    print(f"[setup] cabeza entrenable: "
          f"{sum(q.numel() for q in head.parameters() if q.requires_grad) / 1e6:.2f} M "
          f"parametros; backbone congelado "
          f"({sum(q.numel() for q in backbone.parameters()) / 1e6:.1f} M)")

    # --- afinado de la ultima etapa ---------------------------------------
    # Medido a 384x384: el extractor completo son 24,0 GFLOPs, layer4 sola 4,8 y
    # la FPN 4,6. Duplicar el backbone entero para que segmentacion tenga el
    # suyo cuesta +100% en el camino caliente; darle copia propia de layer4+FPN
    # cuesta +39%, y el tronco (layer1-3, el 80% del computo) se sigue
    # calculando UNA vez para las cuatro cabezas. Por eso 'privado' es el
    # recomendado: es el unico que no rompe la promesa del diagrama.
    afinables: list[nn.Parameter] = []
    layer4_priv = fpn_priv = None
    if a.afinar == "privado":
        # `extractor.layer4` es un contenedor del grafo de
        # create_feature_extractor: guarda los pesos pero no tiene forward (el
        # grafo cablea las conexiones residuales por afuera). No se puede copiar
        # y llamar. Se arma un layer4 limpio de torchvision y se le cargan esos
        # pesos: las claves del state_dict coinciden exactamente.
        from torchvision.models import resnet50 as _r50
        layer4_priv = _r50(weights=None).layer4
        layer4_priv.load_state_dict(backbone.extractor.layer4.state_dict())
        layer4_priv = layer4_priv.to(device)
        fpn_priv = copy.deepcopy(backbone.fpn).to(device)
        for m_ in (layer4_priv, fpn_priv):
            for q in m_.parameters():
                q.requires_grad_(True)
            afinables += list(m_.parameters())
        # BN congelada: con batch 16 las estadisticas de un batch son ruidosas y
        # reestimarlas desestabiliza justo las clases raras. Se entrenan los
        # pesos, no las corridas de BatchNorm.
        layer4_priv.eval()
        print(f"[setup] afinado PRIVADO: layer4+FPN propios, "
            f"{sum(q.numel() for q in afinables) / 1e6:.1f} M parametros mas. "
            f"El tronco compartido queda intacto.")
    elif a.afinar == "compartido":
        for m_ in (backbone.extractor.layer4, backbone.fpn):
            for q in m_.parameters():
                q.requires_grad_(True)
            afinables += list(m_.parameters())
        print("[!!] afinado COMPARTIDO: estas modificando el backbone que usan las")
        print("[!!] 4 cabezas. La cabeza de objetos va a necesitar reentrenarse")
        print("[!!] contra este backbone nuevo, o el camino caliente necesita dos.")
    if a.afinar != "no" and a.channels_last and layer4_priv is not None:
        layer4_priv = layer4_priv.to(memory_format=torch.channels_last)
        fpn_priv = fpn_priv.to(memory_format=torch.channels_last)

    media = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    desvio = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    def a_gpu(x_uint8: Tensor) -> Tensor:
        x = x_uint8.to(device, non_blocking=True)
        if a.channels_last:
            x = x.to(memory_format=torch.channels_last)
        return x.float().div_(255.0)

    def piramide(x01: Tensor) -> Dict[str, Tensor]:
        """Solo extractor+fpn.

        `SharedBackbone.forward()` ademas calcula embedding, GRU temporal y
        novedad, y esta ultima hace `float(...norm())`: un sync GPU->CPU por
        batch y estado por camara que en entrenamiento no sirve. Es el mismo
        atajo que toma el camino de objetos.
        """
        if a.afinar == "privado":
            with torch.no_grad():
                f = backbone.extractor((x01 - media) / desvio)
            # p3 y p4 salen del tronco congelado; p5 de la layer4 propia.
            # La FPN propia es necesaria porque el top-down de p5 baja a p4 y p3.
            return fpn_priv({"p3": f["p3"], "p4": f["p4"],
                             "p5": layer4_priv(f["p4"])})
        if a.afinar == "compartido":
            return backbone.fpn(backbone.extractor((x01 - media) / desvio))
        with torch.no_grad():
            return backbone.fpn(backbone.extractor((x01 - media) / desvio))

    # --- autotest ---------------------------------------------------------
    if a.autotest:
        print("[!!] autotest: backbone SIN pesos de ImageNet — comprueba que el "
              "camino corre, no que aprenda")
        opt = torch.optim.AdamW(head.parameters(), lr=a.lr)
        # Dos "fuentes" sinteticas con etiquetas parciales distintas, armadas
        # para cualquier numero de clases: la 0 anota {fondo, 1}, la 1 anota
        # {fondo, ultima}. Sirve igual con --clases recortadas.
        if C < 3:
            sys.exit("--autotest necesita al menos 3 clases")
        anot = torch.zeros(2, C, dtype=torch.bool, device=device)
        anot[:, 0] = True
        anot[0, 1] = True
        anot[1, C - 1] = True
        for i in range(4):
            x = torch.randint(0, 255, (2, 3, a.tam, a.tam), dtype=torch.uint8)
            y = torch.stack([
                torch.randint(0, 2, (a.tam, a.tam)),                    # 0 / 1
                torch.randint(0, 2, (a.tam, a.tam)) * (C - 1),          # 0 / ultima
            ]).to(device)
            y[0, :10, :10] = IGNORE_INDEX
            y[1, :10, :10] = 1              # clase que la fuente 1 NO anota
            # La cabeza va DENTRO del autocast: si queda afuera, recibe features
            # bf16 con pesos fp32 y revienta con "Input type (CUDABFloat16Type)
            # and weight type (torch.cuda.FloatTensor) should be the same".
            with torch.autocast(device.type, dtype=amp_dtype, enabled=usar_amp):
                feats = piramide(a_gpu(x))
                perdidas = head.compute_loss(feats, y, anot)
            perdida = perdidas["loss_ce"] + perdidas["loss_dice"]
            perdida.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            print(f"        paso {i}: total {float(perdida.detach()):.4f}  "
                  f"ce {float(perdidas['loss_ce'].detach()):.4f}  "
                  f"dice {float(perdidas['loss_dice'].detach()):.4f}  "
                  f"px descartados {int(perdidas['px_descartados'])}")
            assert torch.isfinite(perdida), "perdida no finita"
            assert int(perdidas["px_descartados"]) == 100, "el guard no disparo"
        print(f"[OK] niveles: { {k: tuple(v.shape) for k, v in feats.items()} }")
        print("[OK] autotest OK: el camino completo corre")
        return 0.0

    # --- diagnostico ------------------------------------------------------
    if a.diagnostico:
        ruta = Path(a.diagnostico)
        if not ruta.is_absolute():
            ruta = dir_base / ruta
        if not ruta.exists():
            sys.exit(f"No existe {ruta}")
        ck = torch.load(ruta, map_location=device, weights_only=False)
        head.load_state_dict(ck.get("head") or ck["head_raw"])
        head.eval()
        print(f"[diag] {ruta.name}  epoca {ck.get('epoca')}  "
              f"mIoU-riesgo {ck.get('metricas', {}).get('miou_riesgo', float('nan')):.4f}")

        ds = DatasetSeg(raiz_ds, "val", a.tam, aumentar=False, con_fuente=True)
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False,
                        num_workers=a.workers, pin_memory=(device.type == "cuda"))
        nf = len(ds.fuentes)
        cmf = torch.zeros(nf * C * C, dtype=torch.long, device=device)
        # Area predicha y real por imagen y por clase: con eso se calcula la
        # metrica que de verdad decide si esto sirve en produccion.
        areas_p, areas_r, anots = [], [], []
        with torch.inference_mode():
            for i, (x, y, _anot, fte) in enumerate(dl):
                x01 = a_gpu(x)
                yg = y.to(device, non_blocking=True)
                fg = fte.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=usar_amp):
                    logits = head(piramide(x01), yg.shape[-2:])
                # argmax LIBRE: es lo que hace el modelo en produccion
                pred = logits.float().argmax(1)
                k = yg != IGNORE_INDEX
                idx = fg[:, None, None] * (C * C) + yg * C + pred
                cmf += torch.bincount(idx[k], minlength=nf * C * C)
                n_px = float(yg.shape[-1] * yg.shape[-2])
                areas_p.append(torch.stack([(pred == c).sum((1, 2)) for c in range(C)], 1)
                               .float().div_(n_px).cpu())
                areas_r.append(torch.stack([(yg == c).sum((1, 2)) for c in range(C)], 1)
                               .float().div_(n_px).cpu())
                anots.append(_anot.clone())
                if i % 20 == 0:
                    print(f"\r  {i}/{len(dl)}", end="", flush=True)
        print()
        cmf = cmf.reshape(nf, C, C).cpu().numpy().astype("float64")

        cm = cmf.sum(0)
        print("\n=== Matriz de confusion global (fila = verdad, % de esa fila) ===")
        cab = "verdad / predice"
        print(f"{cab:<18}" + "".join(f"{c:>10}" for c in CLASES))
        for i in range(C):
            tot = max(cm[i].sum(), 1)
            print(f"{CLASES[i]:<18}" + "".join(f"{100 * cm[i, j] / tot:9.2f}%" for j in range(C)))

        print("\n=== De los pixeles de FONDO de cada fuente, que predice ===")
        print("(un % alto fuera de 'fondo' = falso positivo: el modelo ve esa clase")
        print(" donde la fuente dice que no hay nada)\n")
        print(f"{'fuente':<20}{'px fondo':>12}" + "".join(f"{c:>10}" for c in CLASES))
        peor = []
        for f_i, nom in enumerate(ds.fuentes):
            fila = cmf[f_i, 0]
            tot = fila.sum()
            if tot < 1:
                continue
            print(f"{nom:<20}{int(tot):>12,}" +
                  "".join(f"{100 * fila[j] / tot:9.2f}%" for j in range(C)))
            peor.append((fila[2] / tot, nom, "humo"))
            peor.append((fila[3] / tot, nom, "fuego"))

        print("\n=== De los pixeles VERDADEROS de cada clase, que predice ===")
        print(f"{'clase':<20}{'px':>12}" + "".join(f"{c:>10}" for c in CLASES))
        for i in range(1, C):
            fila = cm[i]
            tot = max(fila.sum(), 1)
            print(f"{CLASES[i]:<20}{int(fila.sum()):>12,}" +
                  "".join(f"{100 * fila[j] / tot:9.2f}%" for j in range(C)))

        # ---------------------------------------------------------------
        # Lo que decide si el modelo sirve: no la mascara, la ALERTA.
        # Horus no necesita el contorno exacto del agua. Necesita contestar
        # "¿hay agua en esta camara y cuanta?". Eso es una decision por imagen
        # con un umbral de area, y un IoU de 0,6 puede dar una deteccion casi
        # perfecta a nivel alerta. Se mide sobre las imagenes cuya fuente SI
        # anota la clase (las unicas donde la verdad se conoce).
        # ---------------------------------------------------------------
        AP = torch.cat(areas_p); AR = torch.cat(areas_r); AN = torch.cat(anots)
        print("\n=== Deteccion a nivel ALERTA (por imagen, no por pixel) ===")
        print("sobre las imagenes cuya fuente anota la clase\n")
        print(f"{'clase':<8}{'umbral':>8}{'imgs':>8}{'precision':>11}{'recall':>9}"
              f"{'F1':>8}   falsa alarma donde la fuente no la anota")
        for c in range(1, C):
            sabe = AN[:, c]
            if sabe.sum() < 10:
                continue
            for t in (0.002, 0.005, 0.01, 0.02, 0.05):
                pp = (AP[sabe, c] >= t)
                rr = (AR[sabe, c] >= t)
                tp = (pp & rr).sum().item(); fp = (pp & ~rr).sum().item()
                fn = (~pp & rr).sum().item()
                prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                ciego = ~sabe
                fa = (AP[ciego, c] >= t).float().mean().item() if ciego.sum() else float("nan")
                marca = " <-" if abs(t - 0.005) < 1e-9 else "   "
                print(f"{CLASES[c]:<8}{t:>8.3f}{int(rr.sum()):>8}{100*prec:>10.1f}%"
                      f"{100*rec:>8.1f}%{100*f1:>7.1f}%{100*fa:>12.1f}%{marca}")
            print()
        print("  <- es el umbral que usa SegmentationHead.predict() por defecto")
        print("  'falsa alarma' es cota SUPERIOR: parte puede ser humo/agua real sin etiquetar")

        print("\n=== Veredicto ===")
        for c in (2, 3):
            recall = cm[c, c] / max(cm[c].sum(), 1)
            fp = cm[:, c].sum() - cm[c, c]
            tp = cm[c, c]
            print(f"{CLASES[c]}: encuentra el {100 * recall:.1f}% de lo que SI es "
                  f"{CLASES[c]}, y por cada pixel acertado inventa "
                  f"{fp / max(tp, 1):.1f}")
        peor.sort(reverse=True)
        print("\nPeores falsos positivos sobre fondo:")
        for v, nom, cl in peor[:5]:
            print(f"  {nom:<20} {100 * v:6.2f}% de su fondo predicho como {cl}")
        return 0.0

    # --- datos ------------------------------------------------------------
    ok, frec = revisar_dataset(raiz_ds)
    if not ok:
        return -1.0
    if a.pesos:
        v = [float(x) for x in a.pesos.replace(" ", "").split(",")]
        if len(v) != C:
            sys.exit(f"--pesos necesita {C} numeros separados por coma")
        pesos = torch.tensor(v, dtype=torch.float32, device=device)
        print("[setup] pesos fijados a mano")
    else:
        pesos = pesos_de_clase(frec).to(device)
    print(f"[setup] pesos por clase: "
          f"{dict(zip(CLASES, np.round(pesos.cpu().numpy(), 2).tolist()))}")

    ds_tr = DatasetSeg(raiz_ds, "train", a.tam, aumentar=True)
    ds_va = DatasetSeg(raiz_ds, "val", a.tam, aumentar=False)
    comun = dict(num_workers=a.workers, pin_memory=(device.type == "cuda"),
                 persistent_workers=a.workers > 0,
                 prefetch_factor=4 if a.workers > 0 else None)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, sampler=crear_sampler(ds_tr),
                       drop_last=True, **comun)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, **comun)
    print(f"[datos] train {len(ds_tr):,} / val {len(ds_va):,}  batch {a.batch}"
          f"{f' x{a.acumular} acum' if a.acumular > 1 else ''}  tam {a.tam}")

    # --- optimizacion -----------------------------------------------------
    grupos = [{"params": list(head.parameters()), "lr": a.lr}]
    if afinables:
        # lr mas chico en la parte preentrenada: viene de ImageNet y no queremos
        # borrarla, solo adaptarla.
        grupos.append({"params": afinables, "lr": a.lr_backbone or a.lr / 10})
        print(f"[setup] lr cabeza {a.lr:.1e} / lr afinado "
              f"{(a.lr_backbone or a.lr / 10):.1e}")
    opt = torch.optim.AdamW(grupos, lr=a.lr, weight_decay=a.wd,
                            fused=(device.type == "cuda"))
    pasos_ep = max(1, len(dl_tr) // a.acumular)
    total = pasos_ep * a.epocas

    def factor_lr(paso: int) -> float:
        if paso < a.warmup:
            return (paso + 1) / a.warmup
        t = (paso - a.warmup) / max(1, total - a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor_lr)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(a.precision == "fp16" and device.type == "cuda"))

    ema = None
    if a.ema > 0:
        ema = torch.optim.swa_utils.AveragedModel(
            head, avg_fn=lambda p, n, _: a.ema * p + (1 - a.ema) * n)

    head_base = head                      # referencia sin compilar: es la que se guarda
    if a.compile:
        try:
            head = torch.compile(head)
            backbone.fpn.forward = torch.compile(backbone.fpn.forward)
            backbone.extractor.forward = torch.compile(backbone.extractor.forward)
            print("[setup] torch.compile activo (la primera epoca compila y va lenta)")
        except Exception as e:
            print(f"[!!] torch.compile no arranco: {e}")

    dir_ckpt = dir_base / "checkpoints"
    dir_ckpt.mkdir(exist_ok=True, parents=True)

    # El backbone esta congelado, asi que basta guardarlo UNA vez. Sin esto, la
    # inferencia construiria un ResNet-50 distinto al que entreno la cabeza y la
    # segmentacion seria ruido. Va al lado del checkpoint, con el nombre que se
    # anota en "backbone_file". Misma regla que en la cabeza de objetos.
    ruta_bb = dir_ckpt / "backbone.pt"
    torch.save(backbone.state_dict(), ruta_bb)
    print(f"[setup] backbone congelado guardado en {ruta_bb.name} "
          f"({ruta_bb.stat().st_size / 1e6:.0f} MB) — va junto al checkpoint")

    epoca_ini, mejor, sin_mejora, paso = 1, -1.0, 0, 0
    hist: List[Dict] = []
    if a.reanudar:
        ck = torch.load(a.reanudar, map_location=device, weights_only=False)
        head_base.load_state_dict(ck.get("head_raw") or ck["head"])
        if ck.get("afinado") and fpn_priv is not None:
            layer4_priv.load_state_dict(ck["afinado"]["layer4"])
            fpn_priv.load_state_dict(ck["afinado"]["fpn"])
        for k, obj in (("opt", opt), ("sched", sched), ("scaler", scaler)):
            if k in ck:
                obj.load_state_dict(ck[k])
        epoca_ini = int(ck.get("epoca", 0)) + 1
        mejor = float(ck.get("mejor_miou", -1.0))
        paso = int(ck.get("paso", 0))
        print(f"[setup] reanudado desde {a.reanudar} (epoca {epoca_ini}, "
              f"mejor mIoU {mejor:.4f})")

    print(f"[setup] {a.precision} | channels_last | TF32 | backbone en no_grad\n")

    for epoca in range(epoca_ini, a.epocas + 1):
        head.train()
        if fpn_priv is not None:
            fpn_priv.train()
            layer4_priv.eval()          # BN congelada, ver arriba
        t0 = time.perf_counter()
        suma = torch.zeros((), device=device)
        n = 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y, anot) in enumerate(dl_tr):
            x01 = jitter_color(a_gpu(x))
            yg = y.to(device, non_blocking=True)
            ag = anot.to(device, non_blocking=True)
            # backbone y cabeza dentro del MISMO autocast: partirlo deja features
            # bf16 entrando a capas fp32.  El backward va afuera, como debe ser.
            with torch.autocast(device.type, dtype=amp_dtype, enabled=usar_amp):
                feats = piramide(x01)
                perdidas = head.compute_loss(feats, yg, ag, class_weights=pesos)
            perdida = perdidas["loss_ce"] + perdidas["loss_dice"]
            scaler.scale(perdida / a.acumular).backward()
            if (i + 1) % a.acumular == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    list(head.parameters()) + afinables, 5.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step(); paso += 1
                if ema is not None:
                    ema.update_parameters(head)
            suma += perdida.detach(); n += 1     # sin .item(): no sincroniza
            if i % 50 == 0:
                print(f"\r  ep {epoca:>3} {i:>5}/{len(dl_tr)}  "
                      f"perdida {float(perdida.detach()):.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}   ", end="", flush=True)
        print()

        # --- validacion ---------------------------------------------------
        modelo_val = ema.module if ema is not None else head
        modelo_val.eval()
        if fpn_priv is not None:
            fpn_priv.eval()
        cm_r, cm_l = Confusion(device), Confusion(device)
        por_img = IoUPorImagen(device)
        with torch.inference_mode():
            for x, y, anot in dl_va:
                x01 = a_gpu(x)
                yg = y.to(device, non_blocking=True)
                ag = anot.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=usar_amp):
                    logits = modelo_val(piramide(x01), yg.shape[-2:])
                logits = logits.float()
                libre = logits.argmax(1)
                cm_l.sumar(libre, yg)                # las 4 clases compitiendo
                por_img.sumar(libre, yg)
                cm_r.sumar(logits.masked_fill(
                    ~ag[:, :, None, None],
                    torch.finfo(torch.float32).min).argmax(1), yg)
        iou_r, iou_l, iou_img = cm_r.iou(), cm_l.iou(), por_img.iou()
        miou = float(np.mean(iou_r[1:]))            # sin fondo: es el que importa

        seg = time.perf_counter() - t0
        print(f"[ep {epoca:>3}] {seg:6.0f}s  perdida {float(suma) / max(n, 1):.4f}  "
              f"mIoU-riesgo {miou:.4f}  " +
              "  ".join(f"{CLASES[c]} {iou_r[c]:.3f}" for c in range(C)))
        print(f"          sin restringir: " +
              "  ".join(f"{CLASES[c]} {iou_l[c]:.3f}" for c in range(C)))
        print(f"          por imagen:     " +
              "  ".join(f"{CLASES[c]} {iou_img[c]:.3f}" for c in range(C)))

        metricas = {"miou_riesgo": miou, "iou": iou_r.tolist(),
                    "iou_libre": iou_l.tolist(), "iou_por_imagen": iou_img.tolist(),
                    "seg": seg}
        hist.append({"epoca": epoca, **metricas})
        ck = {
            "head": (ema.module.state_dict() if ema else head_base.state_dict()),
            "head_raw": head_base.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "epoca": epoca,
            "paso": paso,
            "clases": list(cfg.classes),
            "fpn_levels": list(cfg.fpn_levels),
            "tam": a.tam,
            "metricas": metricas,
            "mejor_miou": max(mejor, miou),
            # Con que backbone se entreno esta cabeza. El motor lo usa para
            # reconstruir exactamente el mismo camino en inferencia.
            "afinar": a.afinar,
            "afinado": ({"layer4": layer4_priv.state_dict(),
                         "fpn": fpn_priv.state_dict()} if fpn_priv is not None
                        else None),
            "backbone_file": ruta_bb.name,
            "backbone_pretrained": not a.autotest,
        }
        torch.save(ck, dir_ckpt / "head_last.pt")
        if miou > mejor:
            mejor, sin_mejora = miou, 0
            torch.save(ck, dir_ckpt / "head_best.pt")
            print(f"          * nuevo mejor mIoU-riesgo ({miou:.4f}) -> head_best.pt")
        else:
            sin_mejora += 1
            if a.paciencia and sin_mejora >= a.paciencia:
                print(f"\n[early stop] {a.paciencia} epocas sin mejorar el mIoU")
                break

    (dir_ckpt / "historial.json").write_text(json.dumps(hist, indent=2))
    print(f"\n[fin] mejor mIoU-riesgo = {mejor:.4f}")
    print("head_best.pt y backbone.pt VAN JUNTOS: la cabeza esta entrenada contra")
    print("esos features exactos. Misma regla que en la cabeza de objetos.")
    return mejor


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r >= 0 else 1)
