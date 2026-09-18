# -*- coding: utf-8 -*-
"""
diagnostico_camaras.py · ¿por qué no aparece la cámara en la lista?

El panel llena esa lista con `GET /video/cameras/available`, que prueba los
índices 0 a 4 con `cv2.VideoCapture(i)` a secas. En Windows eso usa MSMF, y
MSMF no abre muchas webcams —o tarda diez segundos en cada índice, y para
cuando contesta el navegador ya cortó. El síntoma es exactamente ese: "la
cámara anda en la app Cámara de Windows y en Discord, pero acá no aparece".

Este script prueba CADA índice con CADA backend, y además pide un frame de
verdad: `isOpened()` puede decir que sí y después no entregar una sola imagen,
que es lo que pasa cuando Windows tiene el permiso de cámara cortado o cuando
otro programa ya la tiene agarrada.

    python diagnostico_camaras.py
"""

from __future__ import annotations

import os
import sys
import time

INDICES = range(6)


def main() -> int:
    print("=" * 74)
    print("HORUS · qué cámaras ve esta máquina")
    print("=" * 74)
    print()

    try:
        import cv2
    except ImportError:
        print("  No está OpenCV. Instalalo con:")
        print("      pip install opencv-python")
        return 2

    print(f"  OpenCV {cv2.__version__} · {sys.platform}")
    print()

    backends = []
    if os.name == "nt":
        for nombre in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
            if hasattr(cv2, nombre):
                backends.append((nombre, getattr(cv2, nombre)))
    else:
        for nombre in ("CAP_V4L2", "CAP_ANY"):
            if hasattr(cv2, nombre):
                backends.append((nombre, getattr(cv2, nombre)))

    print(f"  {'idx':<5} {'backend':<10} {'abre':<6} {'frame':<7} {'tamaño':<12} {'tardó':<8}")
    print("  " + "-" * 62)

    sirven = []
    abren_sin_frame = []

    for i in INDICES:
        for nombre, flag in backends:
            t0 = time.perf_counter()
            cap = None
            try:
                cap = cv2.VideoCapture(i, flag)
                abierto = cap.isOpened()
                ok, frame = (False, None)
                if abierto:
                    # isOpened() no alcanza: hay que pedir una imagen.
                    for _ in range(3):
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            break
                        time.sleep(0.15)
                tam = f"{frame.shape[1]}x{frame.shape[0]}" if (ok and frame is not None) else "-"
            except Exception as e:                       # noqa: BLE001
                abierto, ok, tam = False, False, f"ERROR {type(e).__name__}"
            finally:
                if cap is not None:
                    # SIEMPRE, abra o no. Un capture sin liberar deja el
                    # dispositivo tomado y el siguiente intento falla solo.
                    cap.release()
            ms = (time.perf_counter() - t0) * 1000

            if abierto or ms > 900:
                print(f"  {i:<5} {nombre:<10} {'sí' if abierto else 'no':<6} "
                      f"{'sí' if ok else 'NO':<7} {tam:<12} {ms:7.0f} ms")
            if abierto and ok:
                sirven.append((i, nombre, tam))
                break            # con este backend alcanza, no probamos los otros
            if abierto and not ok:
                abren_sin_frame.append((i, nombre))

    print()
    print("=" * 74)
    if sirven:
        print("  CÁMARAS QUE ANDAN:")
        for i, nombre, tam in sirven:
            print(f"     índice {i} · {nombre} · {tam}")
        print()
        usa_dshow = any(n == "CAP_DSHOW" for _, n, _ in sirven)
        if usa_dshow and os.name == "nt":
            print("  Abren con DirectShow. Si no te aparecían en el panel, era eso:")
            print("  el backend las probaba con MSMF, que en Windows no las abre.")
            print("  Ya está corregido en video_rutas.py — reiniciá HORUS.bat.")
    elif abren_sin_frame:
        print("  Se abren pero NO entregan imagen.")
        print()
        print("  Casi siempre es una de estas dos:")
        print("    1. Windows tiene cortado el permiso. Andá a")
        print("       Configuración → Privacidad y seguridad → Cámara")
        print("       y prendé 'Permitir que las aplicaciones de escritorio")
        print("       accedan a la cámara'. Esa opción de abajo es la que")
        print("       importa para python, no la de arriba.")
        print("    2. Ya la tiene otro programa. En Windows una webcam la")
        print("       abre UNO solo: Zoom, Teams, Discord, la app Cámara, o")
        print("       la ventana 'Horus modelos' de HORUS.bat. Cerralos y")
        print("       probá de nuevo.")
    else:
        print("  No se encontró ninguna cámara en los índices 0 a 5.")
        print()
        print("  Para descartar que sea Horus: abrí la app 'Cámara' de Windows.")
        print("    - Si ahí tampoco se ve, el problema es del sistema o del cable.")
        print("    - Si ahí SÍ se ve, cerrá esa app (mientras esté abierta")
        print("      ninguna otra la puede usar) y corré esto de nuevo.")
    print("=" * 74)
    return 0 if sirven else 1


if __name__ == "__main__":
    raise SystemExit(main())
