# -*- coding: utf-8 -*-
"""instalar_caidas.py · deja la cabeza de caídas lista para correr.

La cabeza necesita tres cosas y solo una estaba:

    modelo_demo_todo.pt    el ST-GCN entrenado         ya lo tenés
    mediapipe              el paquete de python        se instala acá
    pose_landmarker.task   el modelo de pose de Google se baja acá

Lo del `.task` merece una explicación, porque es lo que trabó todo:

MediaPipe 0.10.x SACÓ la API vieja (`mp.solutions.pose`), que traía el modelo
adentro del paquete. Lo único que queda es la API de Tasks, y esa pide un
archivo de modelo que hay que bajar aparte. O sea que `pip install mediapipe`
ya no alcanza, y el mensaje de error no lo dice.

Qué variante bajar: no quedó registrado con cuál se extrajeron los keypoints
de entrenamiento —el archivo se llamaba `pose_landmarker.task` a secas y se
perdió. Las tres devuelven los mismos 33 puntos en el mismo espacio
normalizado, así que el ST-GCN entiende cualquiera; cambia la precisión y el
costo. Se baja `full`, que es el del medio y el que usa por defecto la API
vieja (`model_complexity=1`), o sea lo más parecido a lo que probablemente
había. Queda anotado en `pose_landmarker.variante.txt`: si las caídas se
comportan raro, probar otra variante es lo primero a mirar.

    python bin/instalar_caidas.py
    python bin/instalar_caidas.py --variante heavy
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)
_FALL = os.path.join(_RAIZ, "horus", "fall")
_TASK = os.path.join(_FALL, "pose_landmarker.task")
_VARIANTE = os.path.join(_FALL, "pose_landmarker.variante.txt")

# Los tamaños son aproximados y sirven para detectar una descarga cortada o
# una página de error guardada como si fuera el modelo.
VARIANTES = {
    "lite":  (5_000_000,  9_000_000),
    "full":  (7_000_000, 12_000_000),
    "heavy": (25_000_000, 35_000_000),
}


def url_de(variante: str) -> str:
    return (f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_{variante}/float16/latest/"
            f"pose_landmarker_{variante}.task")


def paso(n: int, texto: str) -> None:
    print()
    print(f"  {n}. {texto}")


def instalar_mediapipe() -> bool:
    try:
        import mediapipe                                 # noqa: F401
        print(f"     ya está (mediapipe {mediapipe.__version__})")
        return True
    except ImportError:
        pass
    print("     instalando... (baja ~70 MB, tarda un rato)")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "mediapipe"],
                       capture_output=False)
    if r.returncode != 0:
        print()
        print("     No se pudo instalar mediapipe.")
        print("     Si dice algo de 'Microsoft Store', tu python es el de la")
        print("     Store y a veces se pelea con pip. Instalá el de")
        print("     python.org y probá de nuevo.")
        return False
    try:
        import mediapipe                                 # noqa: F401
        return True
    except ImportError as e:
        print(f"     instalado pero no importa: {e}")
        return False


def bajar_task(variante: str) -> bool:
    minimo, maximo = VARIANTES[variante]

    if os.path.exists(_TASK):
        tam = os.path.getsize(_TASK)
        if any(lo <= tam <= hi for lo, hi in VARIANTES.values()):
            print(f"     ya está ({tam/1e6:.1f} MB)")
            return True
        print(f"     el que hay mide {tam} bytes: no es un modelo. Lo rehago.")

    url = url_de(variante)
    print(f"     bajando {variante} de storage.googleapis.com ...")
    tmp = _TASK + ".parcial"
    try:
        # Sin proxy: si hay un HTTP_PROXY de una VPN o un antivirus, este
        # pedido se va a un proxy que probablemente no lo deje pasar.
        abrir = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with abrir.open(url, timeout=90) as r, open(tmp, "wb") as fh:
            while True:
                trozo = r.read(1 << 16)
                if not trozo:
                    break
                fh.write(trozo)
    except Exception as e:                               # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"     no se pudo bajar: {type(e).__name__}: {str(e)[:140]}")
        print()
        print(f"     Bajalo a mano de:")
        print(f"       {url}")
        print(f"     y guardalo como:")
        print(f"       {_TASK}")
        return False

    tam = os.path.getsize(tmp)
    if not (minimo <= tam <= maximo):
        os.remove(tmp)
        print(f"     lo que llegó mide {tam} bytes y {variante} debería medir "
              f"entre {minimo/1e6:.0f} y {maximo/1e6:.0f} MB.")
        print("     Probablemente bajó una página de error, no el modelo.")
        return False

    if os.path.exists(_TASK):
        os.remove(_TASK)
    os.rename(tmp, _TASK)
    with open(_VARIANTE, "w", encoding="utf-8") as fh:
        fh.write(f"{variante}\n{url}\n{tam} bytes\n")
    print(f"     listo ({tam/1e6:.1f} MB, variante {variante})")
    return True


def verificar() -> bool:
    """Que el archivo sea un modelo de verdad y que la cabeza entera cargue.

    No alcanza con que exista: un HTML de error guardado con ese nombre pesa
    poco pero podría entrar en el rango, y un modelo cortado también. La única
    prueba que vale es construir el landmarker y pedirle una inferencia.
    """
    sys.path.insert(0, _FALL)
    sys.path.insert(0, os.path.join(_FALL, "src"))
    try:
        import numpy as np
        from detector_caidas import BackendPoseMediaPipe
    except Exception as e:                               # noqa: BLE001
        print(f"     no pude importar la cabeza: {type(e).__name__}: {e}")
        return False

    try:
        pose = BackendPoseMediaPipe(_TASK)
    except Exception as e:                               # noqa: BLE001
        print(f"     el modelo no carga: {type(e).__name__}: {str(e)[:160]}")
        return False

    try:
        # Un recorte gris. No hay nadie, así que no va a devolver puntos — lo
        # que se prueba es que la inferencia corra sin explotar.
        kp = pose(np.full((256, 128, 3), 127, dtype=np.uint8))
        print(f"     el landmarker corre (sobre un recorte vacío devolvió "
              f"{'nada' if kp is None else str(getattr(kp, 'shape', '?'))})")
    except Exception as e:                               # noqa: BLE001
        print(f"     carga pero no infiere: {type(e).__name__}: {str(e)[:160]}")
        return False

    # Y el ST-GCN, que es la otra mitad.
    try:
        from detector_caidas import ConfigCaidas, DetectorCaidas
        cfg = ConfigCaidas()
        if not os.path.exists(cfg.checkpoint):
            print(f"     falta el checkpoint {cfg.checkpoint}")
            return False
        DetectorCaidas(cfg, verboso=False)
        print(f"     el ST-GCN carga de {os.path.basename(cfg.checkpoint)}")
    except Exception as e:                               # noqa: BLE001
        print(f"     el ST-GCN no carga: {type(e).__name__}: {str(e)[:160]}")
        return False

    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variante", default="full", choices=sorted(VARIANTES))
    a = ap.parse_args()

    print("=" * 70)
    print("HORUS · dejar lista la cabeza de caídas")
    print("=" * 70)

    paso(1, "mediapipe")
    if not instalar_mediapipe():
        return 1

    paso(2, "pose_landmarker.task (el modelo de pose de Google)")
    if not bajar_task(a.variante):
        return 1

    paso(3, "probar que todo cargue de verdad")
    if not verificar():
        return 1

    print()
    print("=" * 70)
    print("  LISTO. La cabeza de caídas ya puede correr.")
    print()
    print("  Cerrá y volvé a abrir HORUS.bat: va a decir")
    print("     [SI] caidas         desmayos y caidas")
    print("  y la regla `caida` deja de figurar dormida.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
