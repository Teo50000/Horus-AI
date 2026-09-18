# -*- coding: utf-8 -*-
"""
crear_topologia.py · arranca un topologia.json con TU tamaño de frame.

Hace la parte tediosa —la estructura del JSON, el tamaño real del frame— y
te deja la parte que solo podés decidir vos: qué pedazo del piso está
prohibido y en qué horario.

Genera UNA cámara con UNA zona de tránsito que cubre todo el cuadro. Con eso:

  merodeo    pasa a usar el tiempo de permanencia de la zona en vez de los
             25 s genéricos del default
  intrusion  SIGUE DORMIDA, y está bien que así sea: intrusión significa
             "esta persona está donde no debe, a una hora que no debe", y
             eso nadie lo puede adivinar por vos. Una zona restringida
             inventada daría alertas inventadas.

    python crear_topologia.py <carpeta con fotos de la camara>
    python crear_topologia.py --tam 1080 1920
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def tam_desde_carpeta(carpeta: Path):
    fotos = sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXT)
    if not fotos:
        sys.exit(f"no encontré imágenes en {carpeta}")
    import cv2
    tam, vistas = None, 0
    for f in fotos[:20]:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        vistas += 1
        if tam is None:
            tam = [h, w]
        elif tam != [h, w]:
            print(f"  ⚠ {f.name} mide {h}x{w} y otras miden {tam[0]}x{tam[1]}.")
            print("    Los polígonos van en píxeles del frame ORIGINAL, así que")
            print("    una cámara por tamaño. Uso el primero.")
            break
    if tam is None:
        sys.exit("ninguna imagen se pudo leer")
    print(f"  {vistas} foto(s) leídas · frame {tam[0]}x{tam[1]} (alto x ancho)")
    return tam


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpeta", nargs="?", help="carpeta con fotos de la cámara")
    ap.add_argument("--tam", nargs=2, type=int, metavar=("ALTO", "ANCHO"))
    ap.add_argument("--camara", default="cam-1")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--salida", default="topologia.json")
    a = ap.parse_args()

    print("=" * 70)
    print("HORUS · arrancar un topologia.json")
    print("=" * 70)

    if a.tam:
        tam = list(a.tam)
        print(f"  frame {tam[0]}x{tam[1]} (dado a mano)")
    elif a.carpeta:
        tam = tam_desde_carpeta(Path(a.carpeta))
    else:
        ap.error("pasá una carpeta con fotos, o --tam ALTO ANCHO")

    h, w = tam
    topo = {
        "_ojo": "GENERADO por crear_topologia.py. La zona 'todo-el-cuadro' es "
                "un punto de partida, no tu sitio. Para que INTRUSION despierte "
                "tenés que agregar una zona 'restringida' con su horario "
                "permitido, dibujada sobre el piso real. Mirá "
                "topologia.example.json, que tiene todos los campos explicados.",
        "simetrica": True,
        "camaras": {
            a.camara: {
                "nombre": a.camara,
                "fps": a.fps,
                "tam_frame": [h, w],
                "solapa": [],
                "vecinos": [],
                "zonas": [
                    {
                        "nombre": "todo-el-cuadro",
                        "tipo": "transito",
                        "prioridad": 0,
                        "merodeo_s": 25.0,
                        "puntos": [[0, 0], [w, 0], [w, h], [0, h]],
                    }
                ],
            }
        },
    }

    salida = AQUI / a.salida
    if salida.exists():
        print(f"\n  ⚠ ya existe {salida.name}. No lo piso.")
        print(f"    Si querés regenerarlo, borralo o usá --salida otro.json")
        return 1
    salida.write_text(json.dumps(topo, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\n  escrito: {salida}")

    sys.path.insert(0, str(AQUI))
    from topologia import Topologia
    t = Topologia.cargar(str(salida))
    print("\n" + t.resumen())
    problemas = t.verificar()
    print("  verificación:", "OK" if not problemas else problemas)

    print("\n" + "-" * 70)
    print("  merodeo   ahora usa los 25 s de la zona")
    print("  intrusion SIGUE DORMIDA hasta que agregues una zona 'restringida'")
    print("            con su horario. Eso lo tenés que dibujar vos sobre el")
    print("            piso real: es la diferencia entre una alerta y un ruido.")
    print("-" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
