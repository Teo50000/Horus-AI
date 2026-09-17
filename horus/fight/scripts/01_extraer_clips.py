import cv2
import numpy as np
import os
import glob

RAIZ = "../data/raw/RWF-2000"
SALIDA = "../data/processed/frames"
N_FRAMES = 32
LADO = 128  # un poco mas grande que el input real (112) para poder hacer
            # random-crop como augmentation durante el entrenamiento

CLASES = {"Fight": "fight", "NonFight": "nofight"}


def muestrear_frames(ruta_video, n_frames=N_FRAMES, lado=LADO):
    """Lee el video una sola vez, secuencial (buscar con CAP_PROP_POS_FRAMES
    es poco confiable en .avi), y me quedo con n_frames repartidos uniforme
    a lo largo de todo el clip. Si el clip tiene menos frames que n_frames,
    los indices quedan repetidos solos -- no hace falta rellenar aparte.
    """
    cap = cv2.VideoCapture(ruta_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, n_frames).round().astype(int)
    objetivo = set(indices.tolist())

    capturados = {}
    pos = 0
    while pos <= indices.max():
        ok, frame = cap.read()
        if not ok:
            break
        if pos in objetivo:
            frame = cv2.resize(frame, (lado, lado))
            capturados[pos] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pos += 1
    cap.release()

    if not capturados:
        return None

    salida = []
    ultimo_valido = None
    for i in indices:
        if i in capturados:
            ultimo_valido = capturados[i]
        if ultimo_valido is None:
            return None  # no se pudo leer ni siquiera el primer frame pedido
        salida.append(ultimo_valido)

    return np.stack(salida)  # (n_frames, lado, lado, 3) uint8, RGB


def procesar_split(split):
    total_ok, total_fail = 0, 0
    for carpeta_clase, label in CLASES.items():
        carpeta_entrada = os.path.join(RAIZ, split, carpeta_clase)
        carpeta_salida = os.path.join(SALIDA, split, carpeta_clase)
        os.makedirs(carpeta_salida, exist_ok=True)

        videos = sorted(glob.glob(os.path.join(carpeta_entrada, "*.avi")))
        print(f"{split}/{carpeta_clase}: {len(videos)} videos")

        for ruta in videos:
            frames = muestrear_frames(ruta)
            if frames is None:
                print(f"  {os.path.basename(ruta)}: no se pudo leer, salteando")
                total_fail += 1
                continue

            nombre = os.path.splitext(os.path.basename(ruta))[0] + ".npz"
            np.savez(
                os.path.join(carpeta_salida, nombre),
                frames=frames,
                label=label,
                archivo_origen=os.path.basename(ruta),
            )
            total_ok += 1

    return total_ok, total_fail


if __name__ == "__main__":
    for split in ("train", "val"):
        ok, fail = procesar_split(split)
        print(f"{split}: {ok} ok, {fail} fallidos\n")
