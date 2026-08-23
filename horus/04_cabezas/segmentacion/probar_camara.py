# -*- coding: utf-8 -*-
"""
probar_camara.py · HORUS — la cabeza de segmentacion sobre una camara en vivo.

Abre la webcam, corre backbone + cabeza en cada frame y pinta encima las
regiones detectadas con el area y el estado de alerta. Es el mismo camino que
usa produccion: `FramePreprocessor` reescala el frame entero al tamano de
entrada deformando el aspecto, asi que aca se hace igual — si lo hicieras con
recorte central verias otra cosa que la que ve Horus.

    python probar_camara.py                       # camara 0
    python probar_camara.py --camara 1
    python probar_camara.py --foto prueba.jpg     # sin camara
    python probar_camara.py --guardar demo.mp4

Teclas:  q / ESC salir   ·   m alterna la mascara   ·   g guarda un PNG
         + / -  mueve el umbral de la clase elegida con --clase-umbral

Sobre el fosforo
----------------
Una llama de fosforo a medio metro ocupa ~0,3-0,8% del cuadro. El umbral de
fuego es 0,002 (0,2%), calibrado justo para eso: con el 0,005 que habia antes
se perdia. Aun asi es una prueba exigente, porque el modelo se entreno con
incendios y fogatas, no con llamas chiquitas en interiores. Si no lo agarra,
probá con un encendedor sostenido, una vela, o acercando el fosforo a la
camara: no es que el modelo este roto, es cambio de dominio.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("Falta OpenCV:  python -m pip install opencv-python")

try:
    import torch
except ImportError:
    sys.exit("Falta PyTorch. Ver LEEME_SEGMENTACION.txt")

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(os.path.dirname(_AQUI))
for _p in (os.path.join(_RAIZ, "03_backbone"), _AQUI):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared_backbone import BackboneConfig, SharedBackbone  # noqa: E402
from segmentation_head import (  # noqa: E402
    SegmentationHead, SegmentationHeadConfig,
)

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

# BGR, igual que la paleta de las mascaras del dataset
COLOR = {"fondo": (0, 0, 0), "agua": (220, 110, 30), "humo": (170, 170, 170),
         "fuego": (20, 90, 235)}


def cargar(pesos: Path, device):
    """Arma el camino de inferencia desde el checkpoint.

    `head_best.pt` y `backbone.pt` viajan juntos: el nombre del segundo esta
    anotado en el primero. Si falta, se avisa fuerte en vez de segmentar mal
    en silencio.
    """
    ck = torch.load(pesos, map_location=device, weights_only=False)
    clases = tuple(ck.get("clases", ("fondo", "agua", "humo", "fuego")))
    tam = int(ck.get("tam", 384))

    ruta_bb = pesos.parent / ck.get("backbone_file", "backbone.pt")
    if not ruta_bb.exists():
        sys.exit(f"Falta {ruta_bb}.\nLa cabeza esta entrenada contra ESE backbone "
                 f"exacto; con otro predice cualquier cosa.")

    backbone = SharedBackbone(
        BackboneConfig(pretrained=False, input_size=tam, freeze_encoder=True))
    backbone.load_state_dict(torch.load(ruta_bb, map_location=device,
                                        weights_only=False))
    backbone = backbone.to(device).eval()

    head = SegmentationHead(SegmentationHeadConfig(
        classes=clases, in_channels=backbone.cfg.fpn_dim))
    head.load_state_dict(ck.get("head") or ck["head_raw"])
    head = head.to(device).eval()

    # Parte afinada, si la corrida uso --afinar privado
    afin = ck.get("afinado")
    l4 = fpn = None
    if afin:
        from torchvision.models import resnet50
        l4 = resnet50(weights=None).layer4
        l4.load_state_dict(afin["layer4"])
        l4 = l4.to(device).eval()
        import copy
        fpn = copy.deepcopy(backbone.fpn)
        fpn.load_state_dict(afin["fpn"])
        fpn = fpn.to(device).eval()

    print(f"[ok] {pesos.name}  epoca {ck.get('epoca')}  tam {tam}  clases {list(clases)}")
    if afin:
        print("[ok] con layer4+FPN afinados (--afinar privado)")
    print("[ok] umbrales de alerta: " +
          "  ".join(f"{clases[k]} {100 * head.umbral(k):.2f}%"
                    for k in range(1, len(clases))))
    return backbone, head, l4, fpn, clases, tam


def piramide(backbone, l4, fpn, x):
    feats = backbone.extractor(x)
    if l4 is not None:
        return fpn({"p3": feats["p3"], "p4": feats["p4"], "p5": l4(feats["p4"])})
    return backbone.fpn(feats)


def inferir(frame, backbone, head, l4, fpn, tam, device, media, desvio,
            amp, usar_amp, idx=0):
    """MISMO preproceso que FramePreprocessor: el frame ENTERO a tam x tam,
    deformando el aspecto. Con recorte central verias otra cosa que produccion."""
    chico = cv2.resize(frame, (tam, tam), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(chico, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0)
    x = x.to(memory_format=torch.channels_last).float().div_(255.0)
    x = (x - media) / desvio
    with torch.inference_mode():
        with torch.autocast(device.type, dtype=amp, enabled=usar_amp):
            feats = piramide(backbone, l4, fpn, x)
            res = head.predict(feats, [(tam, tam)], camera_ids=["vivo"],
                               frame_idxs=[idx])[0]
    return res, res.mask.cpu().numpy().astype(np.uint8)


def dibujar(frame, mask, res, clases, fps, ver_mascara, head):
    h, w = frame.shape[:2]
    salida = frame.copy()

    if ver_mascara:
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        capa = np.zeros_like(frame)
        for k in range(1, len(clases)):
            capa[m == k] = COLOR.get(clases[k], (255, 0, 255))
        hay = m > 0
        salida[hay] = cv2.addWeighted(frame, 0.45, capa, 0.55, 0)[hay]
        # contorno, que ayuda a ver donde cree que termina la region
        for k in range(1, len(clases)):
            cont, _ = cv2.findContours((m == k).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(salida, cont, -1, COLOR.get(clases[k], (255, 0, 255)), 2)

    # panel
    alto = 26 * len(clases) + 16
    cv2.rectangle(salida, (0, 0), (330, alto), (0, 0, 0), -1)
    salida = cv2.addWeighted(frame if not ver_mascara else salida, 0.15,
                             salida, 0.85, 0)
    cv2.putText(salida, f"{fps:5.1f} FPS", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y = 48
    alerta = False
    for k in range(1, len(clases)):
        nom = clases[k]
        a = res.area.get(nom, 0.0)
        u = head.umbral(k)
        activa = a >= u
        alerta = alerta or activa
        col = COLOR.get(nom, (255, 255, 255)) if activa else (110, 110, 110)
        txt = (f"{nom:<6} {100 * a:5.2f}%  (umbral {100 * u:.2f}%)"
               + ("  ALERTA" if activa else ""))
        cv2.putText(salida, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    col, 2 if activa else 1, cv2.LINE_AA)
        y += 26
    if alerta:
        cv2.rectangle(salida, (0, 0), (salida.shape[1] - 1, salida.shape[0] - 1),
                      (0, 0, 255), 4)
    if res.needs_vlm:
        cv2.putText(salida, "dudoso -> VLM", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return salida


def main() -> int:
    ap = argparse.ArgumentParser(description="Segmentacion de Horus en vivo.")
    ap.add_argument("--pesos", default="checkpoints/head_best.pt")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--foto", help="corre sobre una imagen y guarda el resultado, "
                                   "sin necesitar pantalla")
    ap.add_argument("--salida", help="con --foto: donde guardar (default: "
                                     "<nombre>_horus.png)")
    ap.add_argument("--guardar", help="graba el resultado a un .mp4")
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--alto", type=int, default=720)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--clase-umbral", default="fuego",
                    help="que umbral mueven las teclas + y -")
    a = ap.parse_args()

    ruta = Path(a.pesos)
    if not ruta.is_absolute():
        ruta = Path(_AQUI) / ruta
    if not ruta.exists():
        sys.exit(f"No existe {ruta}\nEntrena primero:  "
                 f"python entrenar_segmentacion_cuda.py --epocas 30 --batch 16")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[ok] {torch.cuda.get_device_name(0)}")
    else:
        print("[!!] sin CUDA: va a ir a pocos FPS")
        a.precision = "fp32"
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16,
           "fp32": torch.float32}[a.precision]
    usar_amp = device.type == "cuda" and a.precision != "fp32"

    backbone, head, l4, fpn, clases, tam = cargar(ruta, device)
    media = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    desvio = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    if a.foto:
        frame = cv2.imread(a.foto)
        if frame is None:
            sys.exit(f"no pude abrir {a.foto}")
        res, mask = inferir(frame, backbone, head, l4, fpn, tam, device,
                            media, desvio, amp, usar_amp)
        vista = dibujar(frame, mask, res, clases, 0.0, True, head)
        destino = a.salida or str(Path(a.foto).with_suffix("")) + "_horus.png"
        cv2.imwrite(destino, vista)
        print(f"\n{'clase':<8}{'area':>9}{'umbral':>9}   estado")
        for k in range(1, len(clases)):
            ar = res.area.get(clases[k], 0.0)
            u = head.umbral(k)
            print(f"{clases[k]:<8}{100 * ar:>8.3f}%{100 * u:>8.2f}%   "
                  f"{'ALERTA' if ar >= u else '-'}")
        print(f"\n[ok] {destino}")
        return 0

    if False:
        cap = None
    else:
        cap = cv2.VideoCapture(a.camara, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not cap.isOpened():
            sys.exit(f"no pude abrir la camara {a.camara}. Proba --camara 1, "
                     f"o cerra la app que la este usando.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.ancho)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.alto)

    grabador = None
    ver_mascara = True
    fps = 0.0
    n = 0
    idx_umbral = list(clases).index(a.clase_umbral) if a.clase_umbral in clases else 3
    umbrales = list(head.cfg.area_thresh) if not isinstance(
        head.cfg.area_thresh, (int, float)) else [head.cfg.area_thresh] * len(clases)

    print("\nq/ESC salir · m mascara · g guarda PNG · +/- umbral de "
          f"'{clases[idx_umbral]}'\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[!!] se corto el video")
                break

            t0 = time.perf_counter()
            res, mask = inferir(frame, backbone, head, l4, fpn, tam, device,
                                media, desvio, amp, usar_amp, n)
            dt = time.perf_counter() - t0
            fps = 1.0 / dt if n == 0 else 0.9 * fps + 0.1 / dt
            n += 1

            vista = dibujar(frame, mask, res, clases, fps, ver_mascara, head)
            if a.guardar:
                if grabador is None:
                    grabador = cv2.VideoWriter(
                        a.guardar, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                        (vista.shape[1], vista.shape[0]))
                grabador.write(vista)

            cv2.imshow("HORUS - segmentacion", vista)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("m"):
                ver_mascara = not ver_mascara
            if k == ord("g"):
                nom = f"captura_{n:05d}.png"
                cv2.imwrite(nom, vista)
                print(f"[ok] {nom}")
            if k in (ord("+"), ord("=")):
                umbrales[idx_umbral] = min(0.5, umbrales[idx_umbral] * 1.5 + 1e-4)
                head.cfg.area_thresh = tuple(umbrales)
                print(f"  umbral {clases[idx_umbral]} = "
                      f"{100 * umbrales[idx_umbral]:.3f}%")
            if k in (ord("-"), ord("_")):
                umbrales[idx_umbral] = max(1e-5, umbrales[idx_umbral] / 1.5)
                head.cfg.area_thresh = tuple(umbrales)
                print(f"  umbral {clases[idx_umbral]} = "
                      f"{100 * umbrales[idx_umbral]:.3f}%")
    finally:
        if cap is not None:
            cap.release()
        if grabador is not None:
            grabador.release()
            print(f"[ok] video en {a.guardar}")
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
