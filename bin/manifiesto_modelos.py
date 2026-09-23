#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manifiesto_modelos.py - la lista de pesos, con su huella.

Los modelos NO van a git y no pueden ir: `.gitignore` los excluye a
proposito porque backbone.pt son 103 MB y GitHub rechaza de plano cualquier
archivo de mas de 100 MB. O sea que clonar el repo en otra maquina te deja
todo el codigo y CERO modelos.

Este archivo es lo que si puede viajar por git: nombre, tamano y sha256 de
cada peso. Pesa un par de kilobytes y sirve para contestar, en la otra
maquina, la unica pregunta que importa: lo que llego, ?es lo que tenia que
llegar?

Esa pregunta no es retorica. Las tres formas de que un peso este "presente"
y no sirva:

  - OneDrive con "Archivos a pedido": el archivo figura en la carpeta con su
    nombre y su tamano, y el contenido todavia esta en la nube. Se ve igual
    que uno descargado.
  - una copia por USB o por red que se corto a la mitad.
  - un archivo viejo con el mismo nombre. head_best_solo.pt existio en varias
    versiones; el que anda es uno solo.

Las tres dan "el archivo esta ahi" y ninguna carga.

    python bin\\manifiesto_modelos.py            verifica contra el manifiesto
    python bin\\manifiesto_modelos.py --generar   lo rehace desde este disco
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFIESTO = os.path.join(RAIZ, "modelos.sha256.json")

# Los que el lanzador mira para decidir que cabezas prende. Si agregas una
# cabeza, va aca: lo que no esta en esta lista no lo revisa nadie.
PESOS = [
    ("objetos",      "horus/04_cabezas/objetos/modelos/head_best_solo.pt",
     "personas, armas, fuego, humo, paquetes"),
    ("segmentacion", "horus/04_cabezas/segmentacion/checkpoints/head_v4_produccion.pt",
     "fuego y humo por pixel, la que mejor anda"),
    ("backbone",     "horus/04_cabezas/segmentacion/checkpoints/backbone.pt",
     "el tronco compartido; sin esto no carga ninguna cabeza"),
    ("agresion",     "horus/fight/checkpoints/modelo_fight.pt",
     "peleas"),
    ("caidas",       "horus/fall/checkpoints/modelo_demo_todo.pt",
     "desmayos y caidas (el ST-GCN)"),
]


def sha256(ruta: str) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for trozo in iter(lambda: fh.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def generar() -> int:
    salida = {"_ojo": ("Generado por bin/manifiesto_modelos.py. Los pesos no "
                       "estan en git (son mas de 100 MB); esto es para "
                       "verificar que la copia que llego a otra maquina es la "
                       "correcta."),
              "pesos": {}}
    faltan = []
    for nombre, rel, qué in PESOS:
        ruta = os.path.join(RAIZ, rel)
        if not os.path.exists(ruta):
            faltan.append(rel)
            print("  FALTA  %-14s %s" % (nombre, rel))
            continue
        tam = os.path.getsize(ruta)
        print("  ...    %-14s %6.1f MB  calculando huella" % (nombre, tam / 1048576))
        salida["pesos"][nombre] = {"ruta": rel, "bytes": tam,
                                   "sha256": sha256(ruta), "para": qué}
        print("  OK     %-14s %6.1f MB  %s" % (nombre, tam / 1048576,
                                               salida["pesos"][nombre]["sha256"][:16]))
    with open(MANIFIESTO, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(salida, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print()
    print("Escrito: %s" % MANIFIESTO)
    if faltan:
        print("OJO: %d peso(s) no estaban en este disco y quedaron fuera del "
              "manifiesto." % len(faltan))
        return 1
    return 0


def verificar(rapido: bool = False) -> int:
    if not os.path.exists(MANIFIESTO):
        print("No hay manifiesto. Generalo en la maquina que SI tiene los")
        print("modelos:  python bin/manifiesto_modelos.py --generar")
        return 2
    with open(MANIFIESTO, encoding="utf-8") as fh:
        datos = json.load(fh)

    problemas = 0
    for nombre, info in datos["pesos"].items():
        ruta = os.path.join(RAIZ, info["ruta"])
        if not os.path.exists(ruta):
            print("  FALTA     %-14s %s" % (nombre, info["ruta"]))
            problemas += 1
            continue
        tam = os.path.getsize(ruta)
        if tam != info["bytes"]:
            print("  CORTADO   %-14s tiene %.1f MB y deberia tener %.1f MB"
                  % (nombre, tam / 1048576, info["bytes"] / 1048576))
            print("            (una copia a medias, o OneDrive que todavia no "
                  "lo bajo)")
            problemas += 1
            continue
        if rapido:
            print("  OK(tam)   %-14s %6.1f MB" % (nombre, tam / 1048576))
            continue
        h = sha256(ruta)
        if h != info["sha256"]:
            print("  DISTINTO  %-14s mismo tamano, otro contenido" % nombre)
            print("            esperaba %s..." % info["sha256"][:16])
            print("            y tiene   %s..." % h[:16])
            print("            (suele ser una version vieja con el mismo nombre)")
            problemas += 1
        else:
            print("  OK        %-14s %6.1f MB  %s" % (nombre, tam / 1048576,
                                                      info["para"]))
    print()
    if problemas:
        print("%d problema(s). Esos modelos NO van a cargar." % problemas)
        return 1
    print("Los %d pesos estan completos y son los correctos."
          % len(datos["pesos"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--generar", action="store_true",
                    help="rehacer el manifiesto desde este disco")
    ap.add_argument("--rapido", action="store_true",
                    help="solo tamanos, sin calcular huellas (es mas rapido "
                         "pero no detecta un archivo viejo con el mismo peso)")
    a = ap.parse_args()

    print("HORUS - pesos de los modelos")
    print("=" * 66)
    return generar() if a.generar else verificar(a.rapido)


if __name__ == "__main__":
    raise SystemExit(main())
