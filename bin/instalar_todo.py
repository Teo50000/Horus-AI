#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
instalar_todo.py - deja una PC recien clonada lista para correr Horus.

Lo corre INSTALAR.bat (doble clic). Hace, en orden, todo lo que antes habia
que saber de memoria:

  1. Python y pip
  2. torch: con CUDA si hay placa NVIDIA
  3. las dependencias de los modelos y del backend (los requirements.txt)
  4. los pesos de los modelos (bajar_modelos.py, del release de GitHub)
  5. la cabeza de caidas (instalar_caidas.py: mediapipe + el .task de Google)
  6. el panel (npm install)
  7. el chequeo final (preparar_pc.py): la lista en verde y rojo

El paso 2 es el que justifica que esto exista. `pip install torch` en Windows
baja la build de CPU: el sistema arranca, no tira un solo error, y corre todo
por procesador aunque la placa este ahi al lado. Aca se le pregunta al driver
(nvidia-smi) que version de CUDA soporta y se pide torch al indice de PyTorch
que corresponde. Si hay un torch de CPU instalado y hay placa, se cambia.

Se puede correr cuantas veces quieras: lo que ya esta bien, lo saltea. Si un
paso falla, sigue con los demas y al final dice que quedo pendiente.

    python bin\\instalar_todo.py
    python bin\\instalar_todo.py --cpu          torch sin CUDA aunque haya placa
    python bin\\instalar_todo.py --simular      dice que haria, sin instalar nada
    python bin\\instalar_todo.py --solo torch   un paso solo (python, torch,
                                                 requirements, pesos, caidas,
                                                 panel, chequeo)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# Indices de PyTorch, del mas nuevo al mas viejo. Se prueban solo los que el
# driver soporta (CUDA del driver >= CUDA del indice), empezando por el mas
# alto. Si PyTorch deja de publicar alguno, pip falla rapido con ese y se pasa
# al siguiente: la lista puede tener de mas sin romper nada.
INDICES_CUDA = [
    ((13, 0), "cu130"),
    ((12, 9), "cu129"),
    ((12, 8), "cu128"),   # las RTX 50xx necesitan de aca para arriba
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
]
URL_INDICE = "https://download.pytorch.org/whl/%s"

PASOS = ["python", "torch", "requirements", "pesos", "caidas", "panel", "chequeo"]

SIMULAR = False
resultados: list = []          # (paso, ok, nota)


# --------------------------------------------------------------------------- #
def titulo(n: int, texto: str) -> None:
    print()
    print("=" * 70)
    print("  %d. %s" % (n, texto))
    print("=" * 70)


def correr(cmd: list, cwd: str | None = None) -> int:
    """Corre un comando mostrando su salida. En --simular solo lo muestra."""
    visible = " ".join('"%s"' % c if " " in c else c for c in cmd)
    print("  > %s" % visible)
    if SIMULAR:
        return 0
    try:
        return subprocess.run(cmd, cwd=cwd).returncode
    except OSError as e:
        print("  no pude correrlo: %s" % e)
        return 1


def pip(*args: str) -> int:
    return correr([PY, "-m", "pip", "install", "--disable-pip-version-check", *args])


def anotar(paso: str, ok: bool, nota: str = "") -> bool:
    resultados.append((paso, ok, nota))
    return ok


# --------------------------------------------------------------------------- #
def paso_python() -> bool:
    titulo(1, "Python y pip")
    v = sys.version_info
    print("  Python %d.%d.%d  (%s)" % (v.major, v.minor, v.micro, PY))
    if v < (3, 10):
        print()
        print("  Es muy viejo: torch ya no publica versiones para Python < 3.10.")
        print("  Instala el 3.12 de https://www.python.org/downloads/ (marcando")
        print("  'Add python.exe to PATH') y volve a abrir INSTALAR.bat.")
        return anotar("python", False, "Python %d.%d es muy viejo: instalar 3.12" % (v.major, v.minor))
    if v >= (3, 13):
        print()
        print("  OJO: con Python %d.%d mediapipe (la cabeza de caidas) puede no" % (v.major, v.minor))
        print("  tener version todavia. Si el paso 5 falla, la salida es instalar")
        print("  Python 3.12 y correr esto de nuevo. Lo demas anda igual.")
    if "WindowsApps" in PY:
        print()
        print("  OJO: este es el Python de la Microsoft Store. A veces se pelea con")
        print("  pip. Si algo falla con permisos, instala el de python.org.")
    print()
    if pip("--upgrade", "pip") != 0:
        print("  No pude actualizar pip. Sigo con el que hay.")
    return anotar("python", True)


# --------------------------------------------------------------------------- #
def cuda_del_driver() -> tuple:
    """(version CUDA que soporta el driver, nombre de la placa) o (None, None)."""
    exe = shutil.which("nvidia-smi")
    if not exe and os.name == "nt":
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                            "System32", "nvidia-smi.exe")
        exe = cand if os.path.exists(cand) else None
    if not exe:
        return None, None
    try:
        tabla = subprocess.run([exe], capture_output=True, text=True,
                               timeout=30).stdout
        nombre = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return leer_nvidia_smi(tabla), (nombre.strip().splitlines() or ["placa NVIDIA"])[0]


def leer_nvidia_smi(tabla: str) -> tuple | None:
    """La CUDA que soporta el driver sale en el encabezado: 'CUDA Version: 12.8'."""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", tabla)
    return (int(m.group(1)), int(m.group(2))) if m else None


def indices_para(driver: tuple) -> list:
    return [tag for minimo, tag in INDICES_CUDA if driver >= minimo]


def estado_torch() -> dict | None:
    """Lo que hay instalado, preguntado en un python nuevo (no en este)."""
    codigo = ("import json, torch; print(json.dumps({'version': torch.__version__, "
              "'cuda': torch.version.cuda, 'anda': torch.cuda.is_available()}))")
    r = subprocess.run([PY, "-c", codigo], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def paso_torch(forzar_cpu: bool, driver_forzado: tuple | None, indice_forzado: str | None) -> bool:
    titulo(2, "torch (el que decide si se usa la placa de video)")
    if driver_forzado:
        driver, placa = driver_forzado, "(driver indicado a mano)"
    else:
        driver, placa = cuda_del_driver()

    if driver:
        print("  Placa: %s  -  el driver soporta hasta CUDA %d.%d" % (placa, driver[0], driver[1]))
    else:
        print("  No encuentro placa NVIDIA (nvidia-smi no esta o no contesta).")
    quiere_cuda = bool(driver) and not forzar_cpu
    if driver and forzar_cpu:
        print("  --cpu: instalo la de CPU igual.")

    hay = estado_torch()
    if hay:
        print("  Instalado: torch %s  (%s)" % (
            hay["version"],
            "CUDA %s, %s" % (hay["cuda"], "ve la placa" if hay["anda"] else "NO ve la placa")
            if hay["cuda"] else "build de CPU"))
        if hay["anda"] or not quiere_cuda:
            print("  Esta bien, no toco nada.")
            return anotar("torch", True, "ya estaba" + ("" if hay["anda"] else " (CPU)"))
        print()
        if hay["cuda"] is None:
            print("  Hay placa pero este torch es el de CPU: lo cambio por el de CUDA.")
        else:
            print("  Este torch trae CUDA %s pero no ve la placa (suele ser que el" % hay["cuda"])
            print("  driver es mas viejo que esa CUDA). Lo cambio por uno que le sirva.")
        correr([PY, "-m", "pip", "uninstall", "-y", "torch", "torchvision"])

    if not quiere_cuda:
        print()
        print("  Instalo la build de CPU. Anda, pero todo mas lento.")
        if pip("torch", "torchvision") != 0:
            return anotar("torch", False, "no se pudo instalar torch")
        return anotar("torch", True, "CPU (no hay placa NVIDIA)" if not driver else "CPU (--cpu)")

    candidatos = [indice_forzado] if indice_forzado else indices_para(driver)
    if not candidatos:
        print("  El driver es demasiado viejo para cualquier torch con CUDA.")
        print("  Actualizalo desde https://www.nvidia.com/Download/index.aspx y")
        print("  volve a correr esto. Mientras tanto instalo la de CPU.")
        pip("torch", "torchvision")
        return anotar("torch", False, "driver NVIDIA muy viejo: quedo la build de CPU")

    for tag in candidatos:
        print()
        print("  Pruebo con %s (baja unos 2,5 GB la primera vez, paciencia)..." % tag)
        if pip("torch", "torchvision", "--index-url", URL_INDICE % tag) != 0:
            print("  No salio con %s. Pruebo el siguiente." % tag)
            continue
        if SIMULAR:
            return anotar("torch", True, "CUDA (%s)" % tag)
        hay = estado_torch()
        if hay and hay["anda"]:
            print("  Listo: torch %s ve la placa." % hay["version"])
            return anotar("torch", True, "CUDA %s (%s)" % (hay["cuda"], tag))
        print("  Se instalo pero no ve la placa. Pruebo el siguiente.")
        correr([PY, "-m", "pip", "uninstall", "-y", "torch", "torchvision"])

    print()
    print("  Ningun torch con CUDA vio la placa. Dejo la build de CPU para que al")
    print("  menos arranque. Lo mas comun es el driver: actualizalo y volve a")
    print("  correr INSTALAR.bat.")
    pip("torch", "torchvision")
    return anotar("torch", False, "no hubo forma de que vea la placa: quedo la de CPU")


# --------------------------------------------------------------------------- #
def paso_requirements() -> bool:
    titulo(3, "Dependencias de los modelos y del backend")
    print("  torch ya esta, asi que pip lo da por cumplido y no lo pisa con el de CPU.")
    print()
    malos = []
    for req in ("horus/requirements.txt", "FASTAPI/requirements.txt"):
        if pip("-r", os.path.join(RAIZ, *req.split("/"))) != 0:
            malos.append(req)
    if malos:
        return anotar("requirements", False, "fallo: " + ", ".join(malos))
    return anotar("requirements", True)


def paso_pesos() -> bool:
    titulo(4, "Los pesos de los modelos (del release de GitHub, unos 270 MB)")
    rc = correr([PY, os.path.join(RAIZ, "bin", "bajar_modelos.py")])
    return anotar("pesos", rc == 0, "" if rc == 0 else "faltan pesos: volve a correrlo")


def paso_caidas() -> bool:
    titulo(5, "La cabeza de caidas (mediapipe + el modelo de pose de Google)")
    rc = correr([PY, os.path.join(RAIZ, "bin", "instalar_caidas.py")])
    return anotar("caidas", rc == 0, "" if rc == 0 else "ver el mensaje del paso 5")


def version_node() -> tuple | None:
    node = shutil.which("node")
    if not node:
        return None
    try:
        v = subprocess.run([node, "--version"], capture_output=True, text=True,
                           timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"v?(\d+)\.(\d+)", v)
    return (int(m.group(1)), int(m.group(2))) if m else None


def paso_panel() -> bool:
    titulo(6, "El panel (Node.js + npm install)")
    v = version_node()
    npm = shutil.which("npm")
    if not v or not npm:
        print("  No encuentro Node.js.")
        print()
        print("  Baja el LTS de https://nodejs.org, instalalo con las opciones que")
        print("  trae, cerra esta ventana y volve a abrir INSTALAR.bat.")
        return anotar("panel", False, "falta Node.js (nodejs.org, version LTS)")
    print("  Node %d.%d" % v)
    # El panel usa Vite 7, que pide Node 20.19+ o 22.12+.
    if v < (20, 19) or (v[0] == 21) or (v[0] == 22 and v < (22, 12)):
        print("  Es viejo para Vite 7 (pide 20.19 o 22.12 en adelante). Instala el")
        print("  LTS de https://nodejs.org y volve a correr esto.")
        return anotar("panel", False, "Node %d.%d es viejo: instalar el LTS" % v)
    rc = correr([npm, "install"], cwd=os.path.join(RAIZ, "HorusAI"))
    return anotar("panel", rc == 0, "" if rc == 0 else "fallo npm install")


def paso_chequeo() -> bool:
    titulo(7, "Chequeo final: que quedo y que no")
    correr([PY, os.path.join(RAIZ, "bin", "preparar_pc.py")])
    return True


# --------------------------------------------------------------------------- #
def main() -> int:
    global SIMULAR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cpu", action="store_true", help="torch sin CUDA aunque haya placa")
    ap.add_argument("--simular", action="store_true", help="dice que haria, sin instalar nada")
    ap.add_argument("--solo", choices=PASOS, action="append", help="correr solo ese paso")
    ap.add_argument("--driver", help=argparse.SUPPRESS)   # "12.8": para probar sin placa
    ap.add_argument("--cuda", help="forzar un indice de PyTorch (cu126, cu128, ...)")
    a = ap.parse_args()
    SIMULAR = a.simular

    driver = None
    if a.driver:
        mayor, menor = a.driver.split(".")
        driver = (int(mayor), int(menor))

    print("HORUS - instalar todo lo que hace falta en esta PC")
    if SIMULAR:
        print("(SIMULANDO: muestro los comandos, no instalo nada)")

    pasos = {
        "python": paso_python,
        "torch": lambda: paso_torch(a.cpu, driver, a.cuda),
        "requirements": paso_requirements,
        "pesos": paso_pesos,
        "caidas": paso_caidas,
        "panel": paso_panel,
        "chequeo": paso_chequeo,
    }
    elegidos = a.solo or PASOS
    for nombre in PASOS:
        if nombre not in elegidos:
            continue
        ok = pasos[nombre]()
        if nombre == "python" and not ok:
            break                      # sin un Python que sirva no tiene sentido seguir

    print()
    print("=" * 70)
    print("  RESUMEN")
    print("=" * 70)
    for paso, ok, nota in resultados:
        if paso == "chequeo":
            continue
        print("  %-5s %-13s %s" % ("OK" if ok else "FALTA", paso, nota))
    pendientes = [r for r in resultados if not r[1]]
    print()
    if pendientes:
        print("  Quedaron %d cosa(s) pendientes (arriba dice cual y como)." % len(pendientes))
        print("  Arreglalas y volve a abrir INSTALAR.bat: lo que ya esta lo saltea.")
        return 1
    print("  Todo listo. Ahora: doble clic en HORUS.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
