# -*- coding: utf-8 -*-
"""Lee de la entrada y escribe en pantalla Y en un archivo, linea por linea.

Reemplaza a `Tee-Object` de PowerShell, que bufferea: el archivo quedaba
minutos atrasado —a veces vacio hasta que el proceso terminaba— que es
exactamente lo que no queres de un log cuando estas mirando por que algo no
anda. Aca cada linea se escribe y se vacia al toque, en los dos lados.

    python -u servicio.py ... 2>&1 | python -u bin\\tee.py logs\\modelos.txt
"""

import io
import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("uso: tee.py <archivo>", file=sys.stderr)
        return 2

    destino = sys.argv[1]
    carpeta = os.path.dirname(os.path.abspath(destino))
    if carpeta and not os.path.isdir(carpeta):
        os.makedirs(carpeta, exist_ok=True)

    entrada = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8",
                               errors="replace", newline="")
    salida = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", newline="")

    with io.open(destino, "w", encoding="utf-8", errors="replace",
                 newline="") as fh:
        for linea in entrada:
            salida.write(linea)
            salida.flush()
            fh.write(linea)
            fh.flush()          # esto es todo el punto del archivo
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
