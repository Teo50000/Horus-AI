#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probar_manifiesto.py - las tres formas de que un peso "este" y no sirva.

Este verificador existe para una sola situacion: llevar los modelos a otra
maquina y saber, ANTES de perder una tarde, si lo que llego es lo que tenia
que llegar. Si el verificador se equivoca justo en eso, es peor que no
tenerlo — dice que esta todo bien y el sistema no carga.

Los tres casos, y por que ninguno se ve a simple vista:

  1. no esta          lo mas facil, pero hay que nombrarlo bien
  2. cortado          una copia a medias, o OneDrive con "archivos a pedido":
                      el nombre esta en la carpeta y el contenido en la nube
  3. otro contenido   mismo nombre, mismo tamano, version vieja. head_best_solo.pt
                      existio en varias; la que anda es una sola

Corre sobre archivos de mentira en una carpeta temporal: no toca los pesos
de verdad ni tarda en calcular huellas de 100 MB.

    python bin/probar_manifiesto.py
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AQUI)

import manifiesto_modelos as mm                              # noqa: E402


@contextlib.contextmanager
def _repo_de_mentira():
    """Dos pesos chiquitos, con la misma estructura que los de verdad."""
    d = tempfile.mkdtemp(prefix="horus_pesos_")
    viejos = (mm.RAIZ, mm.MANIFIESTO, mm.PESOS)
    mm.RAIZ = d
    mm.MANIFIESTO = os.path.join(d, "modelos.sha256.json")
    mm.PESOS = [("objetos", "pesos/objetos.pt", "las cajas"),
                ("backbone", "pesos/backbone.pt", "el tronco")]
    os.makedirs(os.path.join(d, "pesos"))
    for nombre, contenido in (("objetos.pt", b"A" * 4096),
                              ("backbone.pt", b"B" * 8192)):
        with open(os.path.join(d, "pesos", nombre), "wb") as fh:
            fh.write(contenido)
    try:
        yield d
    finally:
        mm.RAIZ, mm.MANIFIESTO, mm.PESOS = viejos
        shutil.rmtree(d, ignore_errors=True)


def _callado(fn, *a, **k):
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        cod = fn(*a, **k)
    return cod, salida.getvalue()


def main() -> int:
    fallas, corridos = [], []

    def caso(nombre, ok, detalle=""):
        corridos.append(nombre)
        print(("  OK    " if ok else "  FALLA ") + "%-26s %s" % (nombre, detalle))
        if not ok:
            fallas.append(nombre)

    print("=" * 74)
    print("HORUS · verificador de pesos — sobre archivos de mentira")
    print("=" * 74)

    with _repo_de_mentira() as d:
        cod, _ = _callado(mm.generar)
        caso("genera el manifiesto", cod == 0 and os.path.exists(mm.MANIFIESTO))

        cod, out = _callado(mm.verificar)
        caso("todo bien da verde", cod == 0 and "completos" in out)

        # 1 · falta
        os.remove(os.path.join(d, "pesos", "objetos.pt"))
        cod, out = _callado(mm.verificar)
        caso("detecta el que falta", cod == 1 and "FALTA" in out)

        # 2 · cortado (OneDrive a medias, copia interrumpida)
        with open(os.path.join(d, "pesos", "objetos.pt"), "wb") as fh:
            fh.write(b"A" * 100)
        cod, out = _callado(mm.verificar)
        caso("detecta el cortado", cod == 1 and "CORTADO" in out,
             "100 bytes donde iban 4096")

        # 3 · mismo tamano, otro contenido: el mas dificil y el mas comun
        with open(os.path.join(d, "pesos", "objetos.pt"), "wb") as fh:
            fh.write(b"Z" * 4096)
        cod, out = _callado(mm.verificar)
        caso("detecta la version vieja", cod == 1 and "DISTINTO" in out,
             "mismo peso, otra huella")

        # ...y que el modo --rapido NO pretenda detectarlo. Un chequeo que
        # promete lo que no puede es la peor variante de todas.
        cod, out = _callado(mm.verificar, True)
        caso("el modo rapido no miente", cod == 0 and "OK(tam)" in out,
             "solo mira tamanos, y lo dice")

        # vuelve a estar bien
        with open(os.path.join(d, "pesos", "objetos.pt"), "wb") as fh:
            fh.write(b"A" * 4096)
        cod, _ = _callado(mm.verificar)
        caso("se recupera al restaurarlo", cod == 0)

    print("-" * 74)
    if fallas:
        print("%d de %d FALLA(S): %s" % (len(fallas), len(corridos),
                                         ", ".join(fallas)))
        return 1
    print("%d/%d pruebas OK" % (len(corridos), len(corridos)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
