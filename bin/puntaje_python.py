# -*- coding: utf-8 -*-
"""Cuantas cosas le faltan a ESTE python para correr Horus. Sale con ese
numero como codigo de salida (0 = tiene todo). Lo usa elegir_python.bat.

Usa find_spec y no import: importar torch tarda segundos, y esto se corre una
vez por cada python instalado en cada arranque.

Por que existe: la PC de desarrollo tiene 3.11, 3.12 (de la Store), 3.13 y
3.14, y `python` a secas era el 3.14, vacio. El lanzador decia [NO] caidas con
torch y mediapipe instalados en otro python de la misma maquina.
"""
import importlib.util
import sys

# lo que necesitan los modelos (servicio.py) y el backend (FASTAPI)
NECESARIOS = ("torch", "mediapipe", "cv2", "numpy",
              "fastapi", "uvicorn", "sqlmodel", "websockets")

faltan = sum(importlib.util.find_spec(m) is None for m in NECESARIOS)
# 3.13 o mas nuevo: mediapipe puede no tener version (ver instalar_todo.py).
# Si empata con uno de 3.10-3.12, que gane el otro.
if sys.version_info >= (3, 13) or sys.version_info < (3, 10):
    faltan += 1
sys.exit(faltan)
