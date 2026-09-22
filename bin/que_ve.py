# -*- coding: utf-8 -*-
"""que_ve.py · ¿qué está viendo Horus ahora mismo?

Contesta la pregunta que el video no contesta: cuando una alerta NO salta,
¿es que la cabeza no ve nada, que lo ve por debajo del umbral, o que la regla
pide algo más?

Le pregunta al servicio (puerto 8010) cada segundo y muestra, por cámara, el
mejor score de cada clase y el área que la segmentación marca como fuego o
humo — sin filtrar por umbral, que es justamente la gracia.

    python bin/que_ve.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("HORUS_SERVICIO_URL", "http://127.0.0.1:8010") + "/estado"

# Sin proxy: el servicio corre en esta misma máquina.
_ABRIR = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Los umbrales con los que la fusión decide. Sirven para poder decir "lo ve
# pero no llega", que es la mitad del valor de esta pantalla.
UMBRAL = {"fuego": 0.35, "humo": 0.35, "arma": 0.25, "cuchillo": 0.25,
          "persona": 0.35, "paquete": 0.35, "celular": 0.35}


def barra(v: float, tope: float = 1.0, ancho: int = 22) -> str:
    n = max(0, min(ancho, int(round(v / tope * ancho))))
    return "#" * n + "." * (ancho - n)


def main() -> int:
    print("=" * 74)
    print("HORUS · qué está viendo, alerte o no")
    print("=" * 74)
    print("  Ctrl+C para salir.")
    print()

    while True:
        try:
            with _ABRIR.open(URL, timeout=3) as r:
                e = json.loads(r.read().decode("utf-8"))
        except Exception as ex:                          # noqa: BLE001
            print(f"\r  El servicio no contesta en {URL} ({type(ex).__name__}). "
                  f"¿Está corriendo? HORUS_herramientas.bat -> M", end="", flush=True)
            time.sleep(2)
            continue

        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 74)
        print(f"HORUS · qué está viendo  ·  {e.get('modelos')}  ·  "
              f"{e.get('ticks', 0)} ticks  ·  {e.get('eventos', 0)} eventos")
        print("=" * 74)

        viendo = e.get("viendo") or {}
        if not viendo:
            print()
            print("  Ninguna cámara está entregando frames.")
            print("  Si el panel dice MODELOS SIN CAMARAS, falta el backend")
            print("  o falta dar de alta una cámara.")

        for cam, datos in sorted(viendo.items()):
            print()
            print(f"  [{cam}]")
            objs = datos.get("objetos") or {}
            if objs:
                for clase, score in list(objs.items())[:8]:
                    u = UMBRAL.get(clase)
                    marca = ""
                    if u is not None:
                        marca = "  ALERTA" if score >= u else f"  (no llega a {u:.2f})"
                    print(f"     {clase:<20} {score:5.3f}  {barra(score)}{marca}")
            else:
                print("     objetos: no ve nada")

            seg = datos.get("segmentacion_pct_area") or {}
            interesantes = {k: v for k, v in seg.items()
                            if k not in ("fondo",) and v > 0.0}
            if seg:
                if interesantes:
                    for clase, pct in sorted(interesantes.items(), key=lambda kv: -kv[1]):
                        print(f"     seg:{clase:<16} {pct:5.2f}% del cuadro")
                else:
                    print("     segmentacion: 0% de fuego, humo y agua")
            else:
                print("     segmentacion: apagada (se prende con --segmentacion)")

        dormidas = e.get("reglas_dormidas") or {}
        if dormidas:
            print()
            print("  REGLAS QUE NO PUEDEN CORRER (les falta una cabeza o la topología):")
            for tipo, veces in sorted(dormidas.items(), key=lambda kv: -kv[1]):
                print(f"     {tipo:<22} {veces} veces")
            print("  Ojo: estas nunca van a alertar. No es que no pase nada.")

        print()
        print("  " + "-" * 70)
        print("  Actualiza cada segundo. Ctrl+C para salir.")
        time.sleep(1.0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(0)
