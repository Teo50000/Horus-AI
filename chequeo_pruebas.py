# -*- coding: utf-8 -*-
"""
chequeo_pruebas.py · ¿se puede probar Horus en esta máquina, y qué falta?

No corre nada pesado y no baja nada. Mira tres cosas y dice qué habilita o
bloquea cada una:

  1. el entorno  (python, torch, cv2, fastapi, ...)
  2. los pesos   (los .pt de cada cabeza)
  3. la config   (topologia.json, zonas.json — las escribís vos, describen tu sitio)

Lo importante no es el OK o el FALTA, es la columna de la derecha: qué se
puede probar hoy y qué no.

    python chequeo_pruebas.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
HORUS = RAIZ / "horus"

VERDE, ROJO, AMAR, GRIS, FIN = "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[0m"
if os.name == "nt" and not os.environ.get("WT_SESSION"):
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        VERDE = ROJO = AMAR = GRIS = FIN = ""


def marca(ok, opcional=False):
    if ok:
        return f"{VERDE}OK   {FIN}"
    return f"{AMAR}falta{FIN}" if opcional else f"{ROJO}FALTA{FIN}"


def titulo(t):
    print(f"\n{t}")
    print("-" * 78)


# --------------------------------------------------------------------------- #
def entorno():
    titulo("1 · ENTORNO")
    print(f"  python {sys.version.split()[0]}  ({sys.executable})")
    faltan = []
    modulos = [
        ("torch", False, "todo: es el motor de las cuatro cabezas"),
        ("numpy", False, "todo"),
        ("cv2", False, "leer imágenes y video (pip install opencv-python)"),
        ("fastapi", False, "el backend y su suite"),
        ("sqlmodel", False, "el backend y su suite"),
        ("uvicorn", False, "levantar el backend"),
        # Este es el que no arrastra nadie y rompe callado: sin soporte de
        # WebSocket, uvicorn no hace el upgrade, el pedido del panel cae en una
        # ruta GET que contesta 200, y el panel se queda sin recibir UNA sola
        # alerta sin mostrar un error. Medido el 18/09.
        ("websockets", False, "que el PANEL RECIBA ALERTAS en vivo"),
        ("mediapipe", True, "caídas EN VIVO; la suite de caídas no lo necesita"),
        ("torchvision", False, "la cabeza de objetos"),
    ]
    for mod, opcional, para in modulos:
        hay = importlib.util.find_spec(mod) is not None
        print(f"  {marca(hay, opcional)}  {mod:<14} {GRIS}{para}{FIN}")
        if not hay and not opcional:
            faltan.append(mod)

    if importlib.util.find_spec("torch") is not None:
        import torch
        cuda = torch.cuda.is_available()
        print(f"\n  torch {torch.__version__} · CUDA: "
              + (f"{VERDE}sí{FIN} ({torch.cuda.get_device_name(0)})" if cuda
                 else f"{AMAR}no{FIN} — todo corre en CPU, lento pero sirve para fotos"))
    return faltan


# --------------------------------------------------------------------------- #
def pesos():
    titulo("2 · PESOS — qué cabeza puede correr")
    filas = [
        ("objetos", HORUS/"04_cabezas/objetos/modelos/head_best_solo.pt",
         "personas, arma, cuchillo, celular, paquete, fuego y humo"),
        ("segmentación", HORUS/"04_cabezas/segmentacion/checkpoints/head_v4_produccion.pt",
         "fuego y humo por máscara — F1 99,1 % en fuego"),
        ("  └ su backbone", HORUS/"04_cabezas/segmentacion/checkpoints/backbone.pt",
         "va al lado del anterior"),
        ("agresión", HORUS/"fight/checkpoints/modelo_fight.pt",
         "pelea entre dos personas"),
        ("caídas", HORUS/"fall/checkpoints/modelo_stgcn.pt",
         "ST-GCN sobre los 33 puntos de MediaPipe"),
        ("ReID", HORUS/"04_cabezas/reid_cuerpo/modelos/reid.pt",
         "seguir a la misma persona entre cámaras"),
    ]
    listos = []
    for nombre, ruta, para in filas:
        hay = ruta.exists()
        mb = f"{ruta.stat().st_size/1e6:6.1f} MB" if hay else "        "
        print(f"  {marca(hay, nombre.strip() in ('ReID',))}  {nombre:<16} {mb}  {GRIS}{para}{FIN}")
        if hay:
            listos.append(nombre.strip())

    viejo = HORUS/"04_cabezas/objetos/modelos/objetos_v2.pt"
    if viejo.exists() and not filas[0][1].exists():
        print(f"\n  {AMAR}aviso{FIN}  está objetos_v2.pt pero NO sirve: necesita un "
              f"backbone.pt que se perdió.\n         El que sirve es head_best_solo.pt, "
              f"que es autocontenido.")
    return listos


# --------------------------------------------------------------------------- #
def config():
    titulo("3 · CONFIG DE TU SITIO — qué reglas quedan dormidas sin esto")
    filas = [
        (HORUS/"06_fusion_decision/topologia.json",
         "intrusion (merodeo anda igual, con 25 s genéricos en vez de por zona)",
         "zonas, horarios y cámaras vecinas"),
        (HORUS/"02_preproceso_roi/zonas.json",
         "nada (es opcional)", "qué píxeles ni se miran — privacidad y GPU"),
    ]
    for ruta, duerme, que in filas:
        hay = ruta.exists()
        ejemplo = ruta.with_suffix(".example.json")
        print(f"  {marca(hay, True)}  {ruta.name:<18} {GRIS}{que}{FIN}")
        if not hay:
            print(f"         sin esto duerme: {duerme}")
            if ejemplo.exists():
                print(f"         plantilla: {GRIS}{ejemplo.relative_to(RAIZ)}{FIN}")


# --------------------------------------------------------------------------- #
def veredicto(faltan_mod, listos):
    titulo("QUÉ PODÉS PROBAR AHORA")
    puede = []
    no = []

    (puede if not faltan_mod else no).append(
        ("las 8 suites de modelos + la del backend", "2_pruebas.bat"))
    (puede if not faltan_mod else no).append(
        ("levantar el backend y abrir /docs", "3_backend.bat"))
    (puede if "objetos" in listos else no).append(
        ("detección sobre tus fotos", "4_revisar_fotos.bat <carpeta>"))
    (puede if "segmentación" in listos else no).append(
        ("fuego y humo sobre tus fotos", "horus/04_cabezas/segmentacion/probar_camara.py"))

    for que, como in puede:
        print(f"  {VERDE}sí {FIN}  {que:<45} {GRIS}{como}{FIN}")
    for que, como in no:
        print(f"  {ROJO}no {FIN}  {que:<45} {GRIS}{como}{FIN}")

    if "websockets" in faltan_mod:
        print(f"\n  {ROJO}Ojo con `websockets`{FIN}: sin ese paquete el backend arranca igual,")
        print(f"  el panel abre, se ven las cámaras… y no llega NI UNA alerta, sin")
        print(f"  ningún error a la vista. Se ve idéntico a una noche tranquila.")
        print(f"  {GRIS}pip install websockets{FIN}")

    if faltan_mod:
        print(f"\n  Instalá lo que falta:  {GRIS}pip install {' '.join(faltan_mod)}{FIN}")
    if "objetos" not in listos:
        print(f"\n  Para la cabeza de objetos falta bajar head_best_solo.pt (93 MB) de")
        print(f"  la salida del notebook de rescate en Kaggle, y ponerlo en")
        print(f"  {GRIS}horus/04_cabezas/objetos/modelos/{FIN}")


def main() -> int:
    print("=" * 78)
    print("HORUS · ¿está listo para probar?")
    print("=" * 78)
    faltan_mod = entorno()
    listos = pesos()
    config()
    veredicto(faltan_mod, listos)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
