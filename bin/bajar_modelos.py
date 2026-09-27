#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bajar_modelos.py - baja los pesos del release de GitHub y los deja en su lugar.

Clonar el repo NO trae los modelos (manifiesto_modelos.py explica por que).
Viven en un release de GitHub:

    https://github.com/Teo50000/Horus-AI/releases/tag/modelos-2026-09

Esto baja cada uno, lo pone en la carpeta donde lo busca HORUS.bat y lo
verifica contra modelos.sha256.json ANTES de darlo por bueno. Un archivo que
no coincide con el manifiesto no se instala.

  - Si ya tenes uno y esta bien, lo saltea (volver a correrlo no cuesta nada).
  - Si tenes uno distinto con el mismo nombre (una version vieja), no lo pisa:
    lo renombra a .anterior y deja el correcto en su lugar.
  - Si se corta a la mitad, lo que quedo es un .bajando que nadie carga; la
    proxima vez empieza de nuevo.

    python bin\\bajar_modelos.py                     los 5
    python bin\\bajar_modelos.py objetos agresion    solo esos
    python bin\\bajar_modelos.py --tag otro-release  de otro release

Cuando entrenes y quieras repartir pesos nuevos, son tres pasos y van juntos:
  1. python bin\\manifiesto_modelos.py --generar
  2. release nuevo en GitHub con los 5 archivos (con el mismo nombre)
  3. cambiar TAG aca abajo, y commitear el manifiesto y este archivo juntos.
Si el manifiesto dice una cosa y el release tiene otra, esto lo avisa
(DISTINTO) y no instala nada: mejor un [NO] claro que un modelo equivocado.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifiesto_modelos as mm  # noqa: E402

REPO = "Teo50000/Horus-AI"
TAG = "modelos-2026-09"

# Para las pruebas: HORUS_PESOS_URL=file:///carpeta/con/los/pesos
URL_BASE = os.environ.get("HORUS_PESOS_URL",
                          "https://github.com/%s/releases/download/{tag}" % REPO)
INTENTOS = 3


def mb(n: int) -> str:
    return "%.1f MB" % (n / 1048576)


def ya_esta_bien(ruta: str, info: dict) -> bool:
    return (os.path.exists(ruta)
            and os.path.getsize(ruta) == info["bytes"]
            and mm.sha256(ruta) == info["sha256"])


def bajar(url: str, destino: str, esperado: int) -> str:
    """Baja url a destino calculando el sha256 en el camino. Devuelve la huella."""
    pedido = urllib.request.Request(url, headers={"User-Agent": "horus-bajar-modelos"})
    h = hashlib.sha256()
    hecho = 0
    ultimo = 0.0
    with urllib.request.urlopen(pedido, timeout=60) as r, open(destino, "wb") as fh:
        while True:
            trozo = r.read(1 << 20)
            if not trozo:
                break
            fh.write(trozo)
            h.update(trozo)
            hecho += len(trozo)
            ahora = time.time()
            if ahora - ultimo > 0.5:
                ultimo = ahora
                pct = 100.0 * hecho / esperado if esperado else 0
                print("\r            %s de %s  (%3.0f%%)   "
                      % (mb(hecho), mb(esperado), pct), end="", flush=True)
    print("\r" + " " * 60 + "\r", end="", flush=True)
    return h.hexdigest()


def instalar(nombre: str, info: dict, tag: str) -> bool:
    ruta = os.path.join(mm.RAIZ, info["ruta"])
    archivo = os.path.basename(info["ruta"])
    url = "%s/%s" % (URL_BASE.format(tag=tag).rstrip("/"), archivo)

    if ya_esta_bien(ruta, info):
        print("  YA ESTABA %-14s %9s  %s" % (nombre, mb(info["bytes"]), info["para"]))
        return True

    print("  BAJANDO   %-14s %9s  %s" % (nombre, mb(info["bytes"]), archivo))
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    tmp = ruta + ".bajando"

    for intento in range(1, INTENTOS + 1):
        try:
            huella = bajar(url, tmp, info["bytes"])
            break
        except urllib.error.HTTPError as e:
            _borrar(tmp)
            if e.code == 404:
                print("  NO ESTA   %-14s el release '%s' no tiene %s" % (nombre, tag, archivo))
                print("            %s" % url)
                return False
            error = "HTTP %d" % e.code
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            _borrar(tmp)
            error = str(getattr(e, "reason", e))
        if intento < INTENTOS:
            print("            fallo (%s), reintento %d de %d..." % (error, intento + 1, INTENTOS))
            time.sleep(2 * intento)
    else:
        print("  FALLO     %-14s %s" % (nombre, error))
        print("            (sin internet, o GitHub no contesta; volve a correrlo)")
        return False

    tam = os.path.getsize(tmp)
    if tam != info["bytes"] or huella != info["sha256"]:
        _borrar(tmp)
        print("  DISTINTO  %-14s lo que hay en el release no es lo que dice el manifiesto"
              % nombre)
        print("            esperaba %s (%s), llego %s (%s)"
              % (info["sha256"][:16], mb(info["bytes"]), huella[:16], mb(tam)))
        print("            No lo instale. El release y modelos.sha256.json tienen que")
        print("            subirse juntos (ver el comentario arriba de todo en este archivo).")
        return False

    if os.path.exists(ruta):
        viejo = ruta + ".anterior"
        _borrar(viejo)
        os.replace(ruta, viejo)
        print("            el que tenias era otro: quedo como %s" % os.path.basename(viejo))
    os.replace(tmp, ruta)
    print("  OK        %-14s %9s  verificado" % (nombre, mb(info["bytes"])))
    return True


def _borrar(ruta: str) -> None:
    try:
        os.remove(ruta)
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cuales", nargs="*",
                    help="solo estos (objetos, segmentacion, backbone, agresion, caidas)")
    ap.add_argument("--tag", default=TAG, help="el release de GitHub (por defecto %s)" % TAG)
    a = ap.parse_args()

    print("HORUS - bajar los pesos de los modelos")
    print("=" * 66)
    if not os.path.exists(mm.MANIFIESTO):
        print("No encuentro modelos.sha256.json. Sin el manifiesto no puedo saber")
        print("si lo que baja es lo correcto. Actualiza el repo (git pull).")
        return 2
    with open(mm.MANIFIESTO, encoding="utf-8") as fh:
        pesos = json.load(fh)["pesos"]

    desconocidos = [c for c in a.cuales if c not in pesos]
    if desconocidos:
        print("No conozco: %s. Los que hay: %s"
              % (", ".join(desconocidos), ", ".join(pesos)))
        return 2
    elegidos = {k: v for k, v in pesos.items() if not a.cuales or k in a.cuales}

    print("Release: https://github.com/%s/releases/tag/%s" % (REPO, a.tag))
    print()
    malos = [n for n, info in elegidos.items() if not instalar(n, info, a.tag)]
    print()

    if malos:
        print("%d de %d no quedaron: %s. Esas cabezas NO van a cargar."
              % (len(malos), len(elegidos), ", ".join(malos)))
        return 1
    print("Los %d pesos estan en su lugar y son los correctos." % len(elegidos)
          if len(elegidos) > 1 else "Quedo en su lugar y es el correcto.")

    task = os.path.join(mm.RAIZ, "horus", "fall", "pose_landmarker.task")
    if "caidas" in elegidos and not os.path.exists(task):
        print()
        print("Para caidas falta ademas pose_landmarker.task (es de Google, no esta")
        print("en el release):  python bin\\instalar_caidas.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
