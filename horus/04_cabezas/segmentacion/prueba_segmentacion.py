# -*- coding: utf-8 -*-
"""
prueba_segmentacion.py · HORUS — prueba de humo del camino de segmentacion.

Fabrica datasets falsos con la forma de los reales, los convierte, arma el
manifiesto, valida, y corre un entrenamiento de punta a punta contra el
`SharedBackbone` de verdad. No baja nada de internet.

    python prueba_segmentacion.py

Correrlo despues de tocar cualquiera de los tres scripts. Que cubre:

  * emparejado imagen/mascara sin rutas cableadas
  * lectura de PNG indexado (el bug clasico: convert("L") sobre modo P
    devuelve la luminancia del color, no el indice de clase)
  * remapeo de indices de COCO-Stuff y ADE20K
  * rasterizado de poligonos COCO con el orden humo -> fuego
  * halo de ignorar y filtro de negativos por hash
  * mascaras diferidas (no se materializa un array por muestra)
  * la cabeza contra la piramide real p3/p4/p5 del backbone compartido
  * el guard que rechaza un manifiesto que miente sobre `anotadas`
  * perdida con softmax restringido, mIoU, checkpoints, reanudar, EMA
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

AQUI = Path(__file__).resolve().parent
TMP = Path(tempfile.gettempdir()) / "horus_prueba_seg"
FALLOS: list[str] = []


def cargar(nombre: str, archivo: str):
    spec = importlib.util.spec_from_file_location(nombre, AQUI / archivo)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = m
    spec.loader.exec_module(m)
    return m


def check(cond, msg):
    print(f"  [{'OK' if cond else 'XX'}] {msg}")
    if not cond:
        FALLOS.append(msg)


def titulo(t):
    print(f"\n=== {t} " + "=" * max(0, 58 - len(t)))


dl = cargar("dl", "descargar_datasets_segmentacion.py")
shutil.rmtree(TMP, ignore_errors=True)
CRUDO = TMP / "crudo"
ARGS = Namespace(lado_max=256, val_frac=0.25, calidad=90, hilos=4, tope_fuente=None)


# ------------------------------------------------------------------ fabricas
def fab_binaria(nombre, n, forma):
    d = CRUDO / nombre
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "masks").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        w, h = random.choice([(320, 240), (256, 256), (400, 300)])
        img = Image.new("RGB", (w, h), (30, 30, 30))
        m = Image.new("L", (w, h), 0)
        x0, y0 = random.randint(0, w // 2), random.randint(0, h // 2)
        caja = [x0, y0, x0 + w // 3, y0 + h // 3]
        for dib, col in ((ImageDraw.Draw(img), (200, 60, 20)), (ImageDraw.Draw(m), 255)):
            (dib.ellipse if forma == "elipse" else dib.rectangle)(caja, fill=col)
        img.save(d / "images" / f"{nombre}_{i:03d}.jpg")
        m.save(d / "masks" / f"{nombre}_{i:03d}.png")


def fab_ade():
    d = CRUDO / "ade20k" / "ADEChallengeData2016"
    for split in ("training", "validation"):
        (d / "images" / split).mkdir(parents=True, exist_ok=True)
        (d / "annotations" / split).mkdir(parents=True, exist_ok=True)
        for i in range(8):
            w, h = 300, 220
            Image.new("RGB", (w, h), (40, 60, 90)).save(
                d / "images" / split / f"ADE_{split}_{i:03d}.jpg")
            a = np.full((h, w), 13, np.uint8)
            a[h // 2:, :] = 22       # water
            a[:20, :] = 0            # sin etiquetar -> ignorar
            a[30:40, :] = 105        # fuente ornamental -> ignorar
            Image.fromarray(a).save(d / "annotations" / split / f"ADE_{split}_{i:03d}.png")


def fab_cocostuff():
    d = CRUDO / "cocostuff"
    for split in ("train2017", "val2017"):
        (d / split / split).mkdir(parents=True, exist_ok=True)
        (d / "stuffthingmaps" / split).mkdir(parents=True, exist_ok=True)
        for i in range(5):
            Image.new("RGB", (80, 60), (20, 40, 60)).save(d / split / split / f"{i:012d}.jpg")
            a = np.full((60, 80), 90, np.uint8)
            a[40:, :] = 177          # water-other
            a[:5, :] = 119           # fog -> ignorar
            a[6:8, :] = 255          # void -> ignorar
            Image.fromarray(a).convert("P").save(
                d / "stuffthingmaps" / split / f"{i:012d}.png")


def fab_roboflow():
    d = CRUDO / "rf" / "train"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        Image.new("RGB", (100, 100), (0, 0, 0)).save(d / f"f{i}.jpg")
    (d / "_annotations.coco.json").write_text(json.dumps({
        "images": [{"id": i, "file_name": f"f{i}.jpg", "width": 100, "height": 100}
                   for i in range(3)],
        "categories": [{"id": 1, "name": "smoke"}, {"id": 2, "name": "Fire"}],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 1, "iscrowd": 0, "area": 1,
             "bbox": [10, 10, 50, 50], "segmentation": [[10, 10, 60, 10, 60, 60, 10, 60]]},
            {"id": 2, "image_id": 0, "category_id": 2, "iscrowd": 0, "area": 1,
             "bbox": [20, 20, 20, 20], "segmentation": [[20, 20, 40, 20, 40, 40, 20, 40]]}],
    }))


def fab_negativos():
    d = CRUDO / "neg"
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "masks").mkdir(parents=True, exist_ok=True)
    for i in range(80):
        Image.new("RGB", (64, 64), (10, 10, 10)).save(d / "images" / f"x{i:03d}.jpg")
        m = np.zeros((64, 64), np.uint8)
        if i >= 60:
            m[10:30, 10:30] = 255
        Image.fromarray(m).save(d / "masks" / f"x{i:03d}.png")


random.seed(0)
fab_binaria("fuego_falso", 14, "elipse")
fab_binaria("humo_falso", 12, "rect")
fab_ade(); fab_cocostuff(); fab_roboflow(); fab_negativos()


# --------------------------------------------------------------- conversores
titulo("conversores")
SALIDA = TMP / "datasets" / "mezcla_seg_v1"
SALIDA.mkdir(parents=True)
FUENTES = [
    dl.Fuente(nombre="fuego_falso", resumen="", clases=(0, 3), metodo="manual", halo=6),
    dl.Fuente(nombre="humo_falso", resumen="", clases=(0, 2), metodo="manual", halo=8),
    dl.Fuente(nombre="ade20k", resumen="", clases=(0, 1), metodo="manual",
              conversor="ade20k", halo=0),
    dl.Fuente(nombre="cocostuff", resumen="", clases=(0, 1), metodo="manual",
              conversor="cocostuff", halo=0),
    dl.Fuente(nombre="rf", resumen="", clases=(0, 2, 3), metodo="manual",
              conversor="coco_poligonos", halo=0),
]
filas = []
for f in FUENTES:
    nuevas = dl.procesar(f, CRUDO / f.nombre, SALIDA, ARGS)
    check(bool(nuevas), f"{f.nombre}: {len(nuevas)} muestras")
    filas += nuevas
with open(SALIDA / "manifiesto.jsonl", "w", encoding="utf-8") as fh:
    for r in filas:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
(SALIDA / "clases.json").write_text("{}", encoding="utf-8")

por_f: dict[str, list] = {}
for r in filas:
    por_f.setdefault(r["fuente"], []).append(r)

m = dl.leer_indices(SALIDA / por_f["cocostuff"][0]["msk"])
check(set(np.unique(m).tolist()) == {0, 1, 255} and bool((m[40:, :] == 1).all()),
      "cocostuff: indices remapeados leyendo un PNG con paleta")

r_fuego = next(r for r in por_f["rf"] if r["px"].get("3"))
m = dl.leer_indices(SALIDA / r_fuego["msk"])
check(m[30, 30] == 3 and m[15, 15] == 2, "roboflow: poligonos con fuego encima de humo")
check(all(r["px_ign"] > 0 for r in por_f["fuego_falso"]), "halo aplicado en fuego_falso")
check(all(r["px_ign"] > 0 for r in por_f["ade20k"]), "ade20k conserva su propio ignorar")

m = np.zeros((60, 60), np.uint8); m[20:40, 20:40] = 3
check(bool((dl.aplicar_halo(m, 0) == m).all()), "halo 0 no toca nada")
h = dl.aplicar_halo(m, 6)
check(int((h == dl.IGNORAR).sum()) > 0 and bool((h[20:40, 20:40] == 3).all()),
      "halo marca el anillo sin comerse la region positiva")

prim = next(dl.conv_binaria(CRUDO / "fuego_falso",
                            dl.Fuente(nombre="x", resumen="", clases=(0, 3),
                                      metodo="manual")))
check(callable(prim[1]) and prim[1]().shape[0] > 0, "las mascaras son diferidas")

titulo("filtro de negativos")
for prob, esperado in ((1.0, 80), (0.0, 20)):
    sal = TMP / f"neg_{prob}"; sal.mkdir(parents=True)
    fn = dl.Fuente(nombre="neg", resumen="", clases=(0, 3), metodo="manual",
                   halo=0, prob_negativos=prob)
    fs = dl.procesar(fn, CRUDO / "neg", sal, ARGS)
    pos = sum(1 for r in fs if "3" in r["px"])
    check(len(fs) == esperado and pos == 20,
          f"prob_negativos={prob}: {len(fs)} guardadas, {pos} positivas")

titulo("reanotar (negativos confiables)")
# `sin_otras` es lo que convierte el fondo de una foto de agua en supervision
# negativa para humo y fuego. Se prueba que --reanotar lo aplique sin tocar
# una sola mascara.
antes = {r["fuente"]: tuple(r["anotadas"]) for r in filas}
hash_msk = {}
for r in filas[:40]:
    hash_msk[r["msk"]] = dl.leer_indices(SALIDA / r["msk"]).tobytes()

def anot_por_fuente():
    return {r["fuente"]: tuple(r["anotadas"])
            for r in (json.loads(l) for l in
                      open(SALIDA / "manifiesto.jsonl", encoding="utf-8"))}

check(dl.reanotar(SALIDA, "fuego") == 0, "reanotar --negativos fuego")
n1 = anot_por_fuente()
check(n1.get("cocostuff") == (0, 1, 3) and antes.get("cocostuff") == (0, 1),
      f"cocostuff: {antes.get('cocostuff')} -> {n1.get('cocostuff')} "
      f"(niega fuego, NO humo: negar humo hunde la clase)")
check(n1.get("fuego_falso") == antes.get("fuego_falso"),
      "una fuente de fuego no se amplia: su fondo casi siempre tiene humo")
check(all(dl.leer_indices(SALIDA / k).tobytes() == v for k, v in hash_msk.items()),
      "ninguna mascara cambio: solo cambio lo que la fuente declara saber")
check(dl.reanotar(SALIDA, "fuego") == 0, "reanotar es idempotente")

dl.reanotar(SALIDA, "humo+fuego")
check(anot_por_fuente().get("cocostuff") == (0, 1, 2, 3), "modo humo+fuego amplia a las 4")
dl.reanotar(SALIDA, "ninguno")
check(anot_por_fuente().get("cocostuff") == (0, 1), "modo ninguno vuelve al original")
dl.reanotar(SALIDA, "fuego")
filas = [json.loads(l) for l in open(SALIDA / "manifiesto.jsonl", encoding="utf-8")]

titulo("verificacion del dataset")
print(dl.escribir_reporte(filas, SALIDA))
check(dl.verificar(SALIDA) == 0, "verificar() no encuentra problemas")


# -------------------------------------------------------- cabeza y backbone
titulo("cabeza sobre el backbone compartido")
import torch  # noqa: E402

tr = cargar("tr", "entrenar_segmentacion_cuda.py")
from segmentation_head import SegmentationHead, SegmentationHeadConfig  # noqa: E402
from shared_backbone import BackboneConfig, SharedBackbone  # noqa: E402

# Sin pesos de ImageNet: la prueba mide que el camino corre, no que aprenda.
bb = SharedBackbone(BackboneConfig(pretrained=False, input_size=128,
                                   freeze_encoder=True)).eval()
for q in bb.parameters():
    q.requires_grad_(False)
with torch.no_grad():
    feats = bb.fpn(bb.extractor(torch.randn(2, 3, 128, 128)))
check(set(feats) == {"p3", "p4", "p5"}, f"piramide real: "
      f"{ {k: tuple(v.shape) for k, v in sorted(feats.items())} }")

cab = SegmentationHead(SegmentationHeadConfig(in_channels=bb.cfg.fpn_dim))
logits = cab(feats, (128, 128))
check(tuple(logits.shape) == (2, 4, 128, 128), f"logits {tuple(logits.shape)}")

y = torch.zeros(2, 128, 128, dtype=torch.long); y[0, 64:, :] = 1; y[1, :30, :30] = 3
anot = torch.tensor([[True, True, False, False], [True, False, True, True]])
p = cab.compute_loss(feats, y, anot)
check(bool(torch.isfinite(p["loss_ce"])) and int(p["px_descartados"]) == 0,
      f"perdida restringida finita (ce {float(p['loss_ce'].detach()):.3f})")

mala = cab.compute_loss(feats, torch.full((2, 128, 128), 3, dtype=torch.long),
                        torch.tensor([[True, True, False, False]] * 2))
check(bool(torch.isfinite(mala["loss_ce"])) and
      int(mala["px_descartados"]) == 2 * 128 * 128,
      "clase fuera de 'anotadas': se descarta en vez de mandar la perdida a infinito")


# ------------------------------------------------------------- entrenamiento
# Autocast forzado en CPU: sin esto, la prueba nunca recorre el camino de
# dtypes mixtos y un tensor bf16 entrando a una capa fp32 solo aparece en la
# placa del usuario, despues de bajar 50 GB de datos.
os.environ["HORUS_AMP_CPU"] = "1"

titulo("entrenamiento de punta a punta (autocast bf16 forzado)")
ok, frec = tr.revisar_dataset(SALIDA)
check(ok, "revisar_dataset acepta el dataset")
pesos = tr.pesos_de_clase(frec)
check(bool(np.isfinite(pesos.numpy()).all() and (pesos.numpy() > 0).all()),
      f"pesos por clase: {dict(zip(tr.CLASES, np.round(pesos.numpy(), 2).tolist()))}")

man = SALIDA / "manifiesto.jsonl"
bkp = man.read_text(encoding="utf-8")
lineas = bkp.splitlines()
roto = json.loads(next(l for l in lineas if '"fuego_falso"' in l))
roto["anotadas"] = [0, 1]                    # miente: dice agua, tiene fuego
man.write_text(json.dumps(roto) + "\n" + "\n".join(lineas), encoding="utf-8")
check(tr.revisar_dataset(SALIDA)[0] is False,
      "revisar_dataset rechaza un manifiesto que miente")
man.write_text(bkp, encoding="utf-8")

_SB = tr.SharedBackbone
tr.SharedBackbone = lambda cfg: _SB(BackboneConfig(
    pretrained=False, input_size=cfg.input_size, freeze_encoder=True))

CK = SALIDA.parent / "checkpoints"
base = ["tr", "--dataset", str(SALIDA), "--tam", "128", "--batch", "2",
        "--workers", "0", "--warmup", "2", "--paciencia", "9"]
try:
    tr._AQUI = str(SALIDA.parent)            # checkpoints al lado del dataset falso
    sys.argv = base + ["--epocas", "2", "--ema", "0"]
    check(tr.main() >= 0, "2 epocas sin EMA")
    sys.argv = base + ["--epocas", "4", "--ema", "0.9",
                       "--reanudar", str(CK / "head_last.pt")]
    check(tr.main() >= 0, "reanudar 2 epocas mas con EMA")
finally:
    tr.SharedBackbone = _SB

ck = torch.load(CK / "head_best.pt", map_location="cpu", weights_only=False)
check(ck["backbone_file"] == "backbone.pt" and (CK / "backbone.pt").exists(),
      "el checkpoint apunta al backbone y el backbone existe")
check(not [k for k in ck["head_raw"] if k.startswith(("_orig_mod.", "module."))],
      "state_dict sin prefijos de compile/DDP")
check(ck["clases"] == tr.CLASES and ck["fpn_levels"] == list(tr._FPN_LEVELS)
      and ck["tam"] == 128,
      f"checkpoint completo (epoca {ck['epoca']}, "
      f"mIoU {ck['metricas']['miou_riesgo']:.4f})")

# La cabeza guardada tiene que cargar en una cabeza limpia.
limpia = SegmentationHead(SegmentationHeadConfig(in_channels=256))
limpia.load_state_dict(ck["head_raw"])
check(True, "head_best.pt carga en una SegmentationHead limpia")


# El IoU global es una metrica de masa y la domina un puñado de fotos grandes.
# --clases: recortar alcance sin rehacer el dataset
titulo("--clases (subconjunto sin rehacer el dataset)")
tr.SharedBackbone = lambda cfg: _SB(BackboneConfig(
    pretrained=False, input_size=cfg.input_size, freeze_encoder=True))
try:
    tr._AQUI = str(SALIDA.parent)
    sys.argv = base + ["--epocas", "1", "--ema", "0", "--clases", "fondo,agua,fuego"]
    check(tr.main() >= 0, "entrena 3 clases sobre el dataset de 4")
finally:
    tr.SharedBackbone = _SB
ck3 = torch.load(CK / "head_last.pt", map_location="cpu", weights_only=False)
check(ck3["clases"] == ["fondo", "agua", "fuego"] and len(ck3["metricas"]["iou"]) == 3,
      f"el checkpoint guarda 3 clases: {ck3['clases']}")
check(ck3["head_raw"]["salida.3.bias"].shape[0] == 3,
      "la capa de salida tiene 3 canales, no 4 con uno muerto")
tr.fijar_clases(list(tr.DEFAULT_CLASSES))          # restaurar para el resto
check(tr.C == 4, "fijar_clases vuelve a las 4")

titulo("IoU por imagen")
_d = torch.device("cpu")
y = torch.zeros(4, 10, 10, dtype=torch.long); y[:, :5] = 2
pr = torch.zeros(4, 10, 10, dtype=torch.long); pr[:, :4] = 2
g, i = tr.Confusion(_d), tr.IoUPorImagen(_d); g.sumar(pr, y); i.sumar(pr, y)
check(abs(g.iou()[2] - i.iou()[2]) < 1e-9,
      f"con imagenes identicas coinciden ({g.iou()[2]:.3f})")
y = torch.zeros(4, 10, 10, dtype=torch.long); pr = torch.zeros(4, 10, 10, dtype=torch.long)
y[0] = 2; pr[0] = 2                      # una foto enorme, perfecta
y[1:, 0, 0] = 2                          # tres con un pixel de humo, no detectado
g, i = tr.Confusion(_d), tr.IoUPorImagen(_d); g.sumar(pr, y); i.sumar(pr, y)
check(g.iou()[2] > 0.95 and i.iou()[2] < 0.30,
      f"con una foto dominante: global {g.iou()[2]:.3f} vs por imagen {i.iou()[2]:.3f}")


# --afinar privado: la promesa es que el tronco compartido no se toca.
import hashlib  # noqa: E402
md5_antes = hashlib.md5((CK / "backbone.pt").read_bytes()).hexdigest()
l4_antes = torch.load(CK / "head_best.pt", map_location="cpu",
                      weights_only=False).get("afinado")
tr.SharedBackbone = lambda cfg: _SB(BackboneConfig(
    pretrained=False, input_size=cfg.input_size, freeze_encoder=True))
try:
    tr._AQUI = str(SALIDA.parent)
    sys.argv = base + ["--epocas", "2", "--ema", "0", "--afinar", "privado",
                       "--lr", "1e-3"]
    check(tr.main() >= 0, "--afinar privado entrena")
finally:
    tr.SharedBackbone = _SB

ck_af = torch.load(CK / "head_last.pt", map_location="cpu", weights_only=False)
check(ck_af.get("afinar") == "privado" and ck_af.get("afinado") is not None
      and {"layer4", "fpn"} <= set(ck_af["afinado"]),
      "el checkpoint guarda layer4 y FPN propios")
check(l4_antes is None, "una corrida sin afinar no guarda parte afinada")

from torchvision.models import resnet50 as _r50r  # noqa: E402
l4_limpia = _r50r(weights=None).layer4.state_dict()
l4_priv = ck_af["afinado"]["layer4"]
check(set(l4_priv) == set(l4_limpia)
      and all(l4_priv[k].shape == l4_limpia[k].shape for k in l4_limpia),
      f"layer4 propia calza exacto con un layer4 de torchvision ({len(l4_priv)} tensores)")

# lo que de verdad importa: que los gradientes hayan llegado hasta ahi
bb_sd = torch.load(CK / "backbone.pt", map_location="cpu", weights_only=False)
cambiados = sum(1 for k, v in l4_priv.items()
                if v.dtype.is_floating_point
                and not torch.equal(v, bb_sd[f"extractor.layer4.{k}"]))
check(cambiados > 0,
      f"{cambiados} tensores de layer4 se movieron: el gradiente llega al backbone")
check(hashlib.md5((CK / "backbone.pt").read_bytes()).hexdigest() == md5_antes,
      "el backbone COMPARTIDO quedo byte a byte igual: las otras 3 cabezas no se enteran")


# El diagnostico es la herramienta para decidir si un mIoU bajo son falsos
# positivos o detecciones perdidas. Se comprueba que corra y desglose por fuente.
tr.SharedBackbone = lambda cfg: _SB(BackboneConfig(
    pretrained=False, input_size=cfg.input_size, freeze_encoder=True))
try:
    tr._AQUI = str(SALIDA.parent)
    sys.argv = base + ["--diagnostico", str(CK / "head_best.pt")]
    check(tr.main() >= 0, "--diagnostico corre sobre un checkpoint")
finally:
    tr.SharedBackbone = _SB


# ------------------------------------------------------------- convergencia
# El test que de verdad importa: tres fuentes que anotan UNA clase cada una,
# y el modelo tiene que aprender las tres a la vez. Si el softmax restringido
# no funcionara, cada fuente le ensenaria a las otras dos que su clase no
# existe y el mIoU se quedaria clavado cerca de cero.
titulo("convergencia con etiquetas parciales")
CONV = TMP / "conv"
SAL2 = CONV / "datasets" / "m"
SAL2.mkdir(parents=True)
random.seed(1)
filas2 = []
for nombre, clase, color, banda in (("agua", 1, (20, 60, 180), "abajo"),
                                    ("humo", 2, (170, 170, 170), "arriba"),
                                    ("fuego", 3, (230, 90, 20), "arriba")):
    d = CONV / "crudo" / nombre
    (d / "images").mkdir(parents=True); (d / "masks").mkdir(parents=True)
    for i in range(90):
        w, h = 160, 120
        a = np.full((h, w, 3), (40, 60, 90), np.uint8)
        mk = np.zeros((h, w), np.uint8)
        c = random.randint(40, 90)
        if banda == "abajo":
            a[c:, :] = color; mk[c:, :] = 255
        else:
            a[:c, :] = color; mk[:c, :] = 255
        Image.fromarray(a).save(d / "images" / f"{nombre}{i:03d}.jpg")
        Image.fromarray(mk).save(d / "masks" / f"{nombre}{i:03d}.png")
    filas2 += dl.procesar(
        dl.Fuente(nombre=nombre, resumen="", clases=(0, clase),
                  metodo="manual", halo=0),
        d, SAL2, Namespace(lado_max=160, val_frac=0.25, calidad=92,
                           hilos=4, tope_fuente=None))
with open(SAL2 / "manifiesto.jsonl", "w", encoding="utf-8") as fh:
    for r in filas2:
        fh.write(json.dumps(r) + "\n")

tr.SharedBackbone = lambda cfg: _SB(BackboneConfig(
    pretrained=False, input_size=cfg.input_size, freeze_encoder=True))
try:
    tr._AQUI = str(CONV)
    sys.argv = ["tr", "--dataset", str(SAL2), "--tam", "96", "--batch", "8",
                "--workers", "0", "--warmup", "5", "--epocas", "6",
                "--lr", "5e-3", "--ema", "0", "--paciencia", "30"]
    miou = tr.main()
finally:
    tr.SharedBackbone = _SB
check(miou > 0.45,
      f"mIoU-riesgo {miou:.3f} en 6 epocas con backbone random: el softmax "
      f"restringido aprende las 3 clases desde fuentes que anotan una sola")
ck2 = torch.load(CONV / "checkpoints" / "head_best.pt", map_location="cpu",
                 weights_only=False)
iou = ck2["metricas"]["iou"]
check(all(v > 0.2 for v in iou[1:]),
      f"ninguna clase quedo en cero: {[round(v, 3) for v in iou]}")


titulo("resultado")
if FALLOS:
    print(f"  {len(FALLOS)} FALLARON:")
    for f in FALLOS:
        print(f"    - {f}")
    sys.exit(1)
print("  todo verde")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0)
