#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preparar_pc.py - que le falta a ESTA maquina para correr Horus.

Pensado para la segunda PC: clonas el repo, corres esto, y te dice pieza por
pieza que falta y con que comando se arregla. No instala nada solo, a
proposito: en una maquina prestada conviene ver la lista antes.

Lo primero que hay que entender, porque es lo que mas tiempo hace perder:

    CLONAR EL REPO NO TE TRAE LOS MODELOS.

`.gitignore` excluye `*.pt` a proposito y no es un descuido: backbone.pt son
103 MB y GitHub rechaza de plano cualquier archivo de mas de 100 MB. Asi que
un clon limpio trae todo el codigo, todas las pruebas, y cero pesos. El
lanzador va a decir [NO] en cada cabeza y va a tener razon.

Los pesos viajan aparte (OneDrive, un pendrive, Releases de GitHub) y
`bin/manifiesto_modelos.py` confirma que lo que llego es lo que tenia que
llegar.

Lo segundo, que importa justo cuando la maquina es mas potente:

    `pip install torch` en Windows instala la version de CPU.

O sea que podes tener una placa de 24 GB y el sistema corriendo entero por
procesador, sin un solo error, solo mas lento. Es exactamente la confusion
que este proyecto trata de no tener en ningun lado: nada roto, nada avisando,
y el sistema no haciendo lo que crees.

    python bin\\preparar_pc.py
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VERDE, ROJO, AMARILLO = "OK  ", "FALTA", "OJO "
faltantes: list = []


def bloque(t: str) -> None:
    print()
    print(t)
    print("-" * max(len(t), 30))


def decir(estado: str, que: str, detalle: str = "") -> None:
    print("  [%s] %-26s %s" % (estado, que, detalle))


def anotar(titulo: str, comando: str) -> None:
    faltantes.append((titulo, comando))


def _hay(mod: str):
    try:
        return importlib.import_module(mod)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
def revisar_python() -> None:
    bloque("1. Python")
    v = sys.version_info
    ok = v >= (3, 9)
    decir(VERDE if ok else ROJO, "version",
          "%d.%d.%d" % (v.major, v.minor, v.micro))
    if not ok:
        anotar("Python 3.9 o mas nuevo", "instalalo de python.org")
    decir(VERDE, "ejecutable", sys.executable)


def revisar_pesos() -> None:
    bloque("2. Los pesos de los modelos")
    print("  (clonar el repo NO los trae: son mas de 100 MB y GitHub no los")
    print("   acepta. Vienen por OneDrive, pendrive o Releases.)")
    print()
    sys.path.insert(0, os.path.join(RAIZ, "bin"))
    try:
        import manifiesto_modelos as mm
    except ImportError as e:
        decir(ROJO, "manifiesto", "no pude importarlo: %s" % e)
        return
    if not os.path.exists(mm.MANIFIESTO):
        decir(AMARILLO, "manifiesto", "no esta; reviso solo si existen")
        for nombre, rel, _q in mm.PESOS:
            hay = os.path.exists(os.path.join(RAIZ, rel))
            decir(VERDE if hay else ROJO, nombre, rel)
            if not hay:
                anotar("el peso de %s" % nombre, "copiar %s" % rel)
        return
    codigo = mm.verificar()
    if codigo != 0:
        anotar("pesos incompletos o equivocados",
               "python bin\\manifiesto_modelos.py   (para ver cuales)")


def revisar_torch() -> None:
    bloque("3. Torch y la placa de video")
    torch = _hay("torch")
    if torch is None:
        decir(ROJO, "torch", "no esta instalado")
        anotar("torch", "ver el comando de abajo (OJO con la version de CPU)")
        _consejo_cuda()
        return
    decir(VERDE, "torch", torch.__version__)

    try:
        hay_cuda = torch.cuda.is_available()
    except Exception as e:                                   # noqa: BLE001
        hay_cuda = False
        decir(AMARILLO, "cuda", "consulta fallida: %s" % e)

    compilado = getattr(getattr(torch, "version", None), "cuda", None)
    if hay_cuda:
        try:
            nombre = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            decir(VERDE, "placa", "%s · %.1f GB · CUDA %s"
                  % (nombre, mem, compilado))
        except Exception:
            decir(VERDE, "placa", "disponible (CUDA %s)" % compilado)
        return

    if compilado is None:
        decir(ROJO, "placa", "este torch es la build de CPU")
        print()
        print("       Esto es lo que mas tiempo hace perder: el sistema va a")
        print("       arrancar, no va a tirar un solo error, y va a correr")
        print("       TODO por procesador. Si viniste a esta PC por la placa,")
        print("       asi no la estas usando.")
        anotar("torch con CUDA", "ver el comando de abajo")
        _consejo_cuda()
    else:
        decir(AMARILLO, "placa", "torch trae CUDA %s pero no ve ninguna placa"
              % compilado)
        print("       Suele ser el driver de NVIDIA viejo o que la placa no")
        print("       es NVIDIA. Proba `nvidia-smi` en una consola.")


def _consejo_cuda() -> None:
    print()
    print("       `pip install torch` a secas baja la build de CPU.")
    print("       Para la de GPU hay que pedirla por el indice de PyTorch:")
    print()
    print("           pip uninstall -y torch torchvision")
    print("           pip install torch torchvision --index-url \\")
    print("               https://download.pytorch.org/whl/cu124")
    print()
    print("       El cu124 depende de tu driver. El selector oficial da el")
    print("       comando exacto: pytorch.org/get-started/locally")


def revisar_backend() -> None:
    bloque("4. El backend")
    criticos = {
        "fastapi": "el servidor",
        "uvicorn": "lo que lo corre",
        "websockets": "SIN ESTO el panel no recibe una sola alerta",
        "sqlmodel": "la base",
        "dotenv": "lee el .env del mail; sin esto no arranca",
        "cv2": "el video de las camaras",
    }
    falta = []
    for mod, para in criticos.items():
        if _hay(mod) is None:
            decir(ROJO, mod, para)
            falta.append(mod)
        else:
            decir(VERDE, mod, para)
    if falta:
        anotar("dependencias del backend",
               "pip install -r FASTAPI\\requirements.txt")

    base = os.path.join(RAIZ, "FASTAPI", "horus.db")
    if os.path.exists(base):
        decir(VERDE, "horus.db", "%.1f KB" % (os.path.getsize(base) / 1024))
    else:
        decir(AMARILLO, "horus.db", "no esta (tampoco va a git)")
        print("       No es un error: se crea sola al arrancar. Pero arranca")
        print("       VACIA, o sea sin camaras y sin contactos de emergencia.")
        print("       Se cargan desde el panel, o copias la de la otra maquina.")


def revisar_panel() -> None:
    bloque("5. El panel")
    node = shutil.which("node")
    if node:
        try:
            v = subprocess.run([node, "--version"], capture_output=True,
                               text=True, timeout=20).stdout.strip()
        except Exception:
            v = "?"
        decir(VERDE, "node", v)
    else:
        decir(ROJO, "node", "no esta en el PATH")
        anotar("node", "instalalo de nodejs.org")

    mods = os.path.join(RAIZ, "HorusAI", "node_modules")
    if os.path.isdir(mods):
        decir(VERDE, "node_modules", "instalado")
    else:
        decir(ROJO, "node_modules", "falta")
        anotar("dependencias del panel", "cd HorusAI  &&  npm install")


def revisar_caidas() -> None:
    bloque("6. La cabeza de caidas")
    if _hay("mediapipe") is None:
        decir(ROJO, "mediapipe", "no esta")
    else:
        import mediapipe
        decir(VERDE, "mediapipe", mediapipe.__version__)
    task = os.path.join(RAIZ, "horus", "fall", "pose_landmarker.task")
    if os.path.exists(task):
        decir(VERDE, "pose_landmarker.task", "%.1f MB"
              % (os.path.getsize(task) / 1048576))
    else:
        decir(ROJO, "pose_landmarker.task", "no esta (no viaja por git)")
    if _hay("mediapipe") is None or not os.path.exists(task):
        anotar("la cabeza de caidas", "python bin\\instalar_caidas.py")


def revisar_config() -> None:
    bloque("7. Configuracion")
    topo = os.path.join(RAIZ, "horus", "06_fusion_decision", "topologia.json")
    if os.path.exists(topo):
        decir(VERDE, "topologia.json", "si viaja por git")
    else:
        decir(AMARILLO, "topologia.json", "sin zonas, intrusion queda dormida")

    env = os.path.join(RAIZ, "FASTAPI", ".env")
    if os.path.exists(env):
        decir(VERDE, ".env", "las credenciales del mail estan")
    else:
        decir(AMARILLO, ".env", "no esta (no va a git, y esta bien que no vaya)")
        print("       El mail va a estar apagado en esta maquina hasta que lo")
        print("       configures: panel -> Ajustes -> Aviso por mail.")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 66)
    print("HORUS - que le falta a esta maquina")
    print("=" * 66)
    print("  repo: %s" % RAIZ)

    revisar_python()
    revisar_pesos()
    revisar_torch()
    revisar_backend()
    revisar_panel()
    revisar_caidas()
    revisar_config()

    bloque("Resumen")
    if not faltantes:
        print("  No falta nada. Arranca con HORUS.bat y fijate que diga [SI]")
        print("  en las cabezas que esperas.")
        return 0
    print("  Falta %d cosa(s):" % len(faltantes))
    print()
    for i, (que, como) in enumerate(faltantes, 1):
        print("   %d. %s" % (i, que))
        print("      %s" % como)
    print()
    print("  Volve a correr esto despues de cada una.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
