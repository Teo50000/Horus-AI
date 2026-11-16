# -*- mode: python ; coding: utf-8 -*-
"""Build the packaged model service used by the Tauri installer."""

import os
import importlib.util

from PyInstaller.utils.hooks import collect_all


ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
SOURCE_DIRS = [
    os.path.join(ROOT, "horus", "00_servicio"),
    os.path.join(ROOT, "horus", "03_backbone"),
    os.path.join(ROOT, "horus", "04_cabezas", "objetos"),
    os.path.join(ROOT, "horus", "04_cabezas", "segmentacion"),
    os.path.join(ROOT, "horus", "05_tracking"),
    os.path.join(ROOT, "horus", "05_tracking", "local"),
    os.path.join(ROOT, "horus", "05_tracking", "global_reid"),
    os.path.join(ROOT, "horus", "06_fusion_decision"),
    os.path.join(ROOT, "horus", "07_alerta"),
    os.path.join(ROOT, "horus", "fall"),
    os.path.join(ROOT, "horus", "fight"),
    os.path.join(ROOT, "horus", "fight", "src"),
]

datas = []
binaries = []
hiddenimports = []
for package in ("cv2", "numpy", "torch", "torchvision", "ultralytics",
                "mediapipe", "onnxruntime"):
    if importlib.util.find_spec(package) is None:
        continue
    collected = collect_all(package)
    datas += collected[0]
    binaries += collected[1]
    hiddenimports += collected[2]
# Módulos que el servicio importa adentro de funciones o después de tocar
# sys.path: el análisis estático de PyInstaller no siempre los ve, y sin ellos
# el .exe se cae al arrancar con "No module named ..." (pasó con
# detector_agresion al usar --agresion).
for modulo in ("modelo_fight", "detector_agresion", "segmentation_engine",
               "segmentation_head", "detector_caidas", "motor_yolo",
               "guarda_armas", "objects_engine", "objects_head"):
    hiddenimports.append(modulo)


a = Analysis(
    [os.path.join(ROOT, "horus", "00_servicio", "servicio.py")],
    pathex=SOURCE_DIRS,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="servicio_modelos",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
