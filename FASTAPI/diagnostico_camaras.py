# -*- coding: utf-8 -*-
"""
diagnostico_camaras.py · ¿por qué no aparece la cámara en la lista?

Prueba cada índice con cada backend, PIDE UNA IMAGEN de verdad (isOpened()
miente: puede decir que sí y no entregar un solo frame), mide cuánto tarda
cada intento, y traduce el código de error de Media Foundation, que es donde
está la respuesta de verdad.

Dos cosas que el log crudo de OpenCV hace mal y acá se corrigen:

  - "backend is generally available but can't be used to capture by index"
    suena a que falta el backend. No: OpenCV imprime esa línea ante CUALQUIER
    fallo de apertura, aunque el backend soporte índices perfectamente
    (opencv/opencv#26519). Lo único que significa es "no pude abrir".

  - El error -1072875772 no dice nada así. En hexa es 0xC00D3704, que tiene
    nombre y significado.

    python diagnostico_camaras.py
"""

from __future__ import annotations

import os
import sys
import time

INDICES = range(6)

# Los HRESULT de Media Foundation que aparecen con webcams, traducidos.
MF_ERRORES = {
    0xC00D3704: ("MF_E_HW_MFT_FAILED_START_STREAMING",
                 "el driver acepta abrir la cámara pero se niega a largar "
                 "video. Casi siempre: la tiene otro programa, o Windows "
                 "tiene cortado el permiso de cámara para apps de escritorio"),
    0xC00D3E85: ("MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED",
                 "la cámara se desconectó o el driver se cayó mientras estaba abierta"),
    0xC00D3EA2: ("MF_E_VIDEO_RECORDING_DEVICE_LOCKED",
                 "la cámara está tomada por otro proceso"),
    0x80070005: ("E_ACCESSDENIED",
                 "Windows denegó el acceso: es el permiso de privacidad"),
}


def traducir(codigo: int) -> str:
    h = codigo & 0xFFFFFFFF
    if h in MF_ERRORES:
        nombre, que = MF_ERRORES[h]
        return f"0x{h:08X} {nombre}\n       {que}"
    return f"0x{h:08X} (sin traducción conocida)"


def main() -> int:
    # Que OpenCV no escupa sus warnings encima de la tabla.
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "1")

    print("=" * 74)
    print("HORUS · qué cámaras ve esta máquina")
    print("=" * 74)
    print()

    try:
        import cv2
    except ImportError:
        print("  No está OpenCV:  pip install opencv-python")
        return 2

    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass

    ver = cv2.__version__
    print(f"  OpenCV {ver} · {sys.platform}")
    mayor = int(ver.split(".")[0]) if ver.split(".")[0].isdigit() else 0
    if mayor >= 5:
        print()
        print("  OJO: estás en OpenCV 5.x. La rama 4.x sigue mantenida y salió")
        print("  DESPUÉS de la 5.0; en Windows su captura de video está mucho")
        print("  más rodada. Si acá abajo no anda ninguna, probá:")
        print("      pip install \"opencv-python<5\"")
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

    print(f"  {'idx':<5} {'backend':<10} {'abre':<6} {'imagen':<8} {'tamaño':<12} {'tardó':<9}")
    print("  " + "-" * 64)

    sirven, abren_sin_imagen, lentos = [], [], []

    for i in INDICES:
        visto = False
        for nombre, flag in backends:
            t0 = time.perf_counter()
            cap = None
            abierto = ok = False
            tam = "-"
            try:
                cap = cv2.VideoCapture(i, flag)
                abierto = cap.isOpened()
                if abierto:
                    for _ in range(3):
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            tam = f"{frame.shape[1]}x{frame.shape[0]}"
                            break
                        time.sleep(0.1)
            except Exception as e:                       # noqa: BLE001
                tam = f"ERROR {type(e).__name__}"
            finally:
                if cap is not None:
                    cap.release()                        # SIEMPRE, abra o no
            ms = (time.perf_counter() - t0) * 1000

            if abierto or ms > 900:
                visto = True
                print(f"  {i:<5} {nombre:<10} {'sí' if abierto else 'no':<6} "
                      f"{'sí' if ok else 'NO':<8} {tam:<12} {ms:8.0f} ms")
            if ms > 2000:
                lentos.append((i, nombre, ms))
            if abierto and ok:
                sirven.append((i, nombre, tam, ms))
                break
            if abierto and not ok:
                abren_sin_imagen.append((i, nombre))
        if not visto and i > 1 and not sirven and not abren_sin_imagen:
            break                                        # nada por acá, no seguimos

    print()
    print("=" * 74)
    if sirven:
        print("  CÁMARAS QUE ANDAN")
        for i, nombre, tam, ms in sirven:
            print(f"     índice {i} · {nombre} · {tam} · abrió en {ms:.0f} ms")
        print()
        print("  Ponelas en el panel: menú de cámaras -> agregar.")
    elif abren_sin_imagen:
        i, nombre = abren_sin_imagen[0]
        print(f"  HAY una cámara en el índice {i}: {nombre} la abre.")
        print("  Pero NO entrega imagen. El código de error de Windows dice:")
        print()
        print("     " + traducir(0xC00D3704))
        print()
        print("  En orden, lo que hay que descartar:")
        print()
        print("   1. ¿Anda fuera de Horus? Abrí la app 'Cámara' de Windows.")
        print("      - Si ahí TAMPOCO se ve: el problema no es Horus. Es el")
        print("        driver, el permiso o el cable. Seguí con el punto 2.")
        print("      - Si ahí SÍ se ve: cerrá esa app (mientras esté abierta,")
        print("        ninguna otra puede usar la cámara) y volvé a correr esto.")
        print()
        print("   2. El permiso de Windows. Configuración -> Privacidad y")
        print("      seguridad -> Cámara. Hay DOS interruptores:")
        print("        - 'Acceso a la cámara'                      (arriba)")
        print("        - 'Permitir que las aplicaciones de escritorio")
        print("           accedan a la cámara'                     (ABAJO DE TODO)")
        print("      Python entra por el de abajo. Ese es el que suele estar")
        print("      apagado y el que nadie mira.")
        print()
        print("   3. Qué programa la tiene. Cerrá Zoom, Teams, Discord, la app")
        print("      Cámara, OBS, y cualquier ventana 'Horus modelos' o")
        print("      'Horus backend' que haya quedado abierta de antes.")
        if mayor >= 5:
            print()
            print("   4. Si ninguna de las tres, es OpenCV 5:")
            print("         pip install \"opencv-python<5\"")
            print("      y volvé a correr esto.")
    else:
        print("  No se encontró ninguna cámara.")
        print()
        print("  Abrí la app 'Cámara' de Windows para descartar que sea Horus.")

    if lentos:
        print()
        print("  ---------------------------------------------------------------")
        peor = max(m for _, _, m in lentos)
        print(f"  Aparte: hay intentos que tardan hasta {peor/1000:.1f} s.")
        print("  Eso solo ya vaciaba la lista del panel: la busqueda recorria")
        print("  cinco indices en serie y el navegador cortaba antes de que")
        print("  contestara. Ya esta arreglado (se busca una vez al arrancar el")
        print("  backend y queda cacheado), pero si ves la lista tardar, es esto.")
    print("=" * 74)
    return 0 if sirven else 1


if __name__ == "__main__":
    raise SystemExit(main())
