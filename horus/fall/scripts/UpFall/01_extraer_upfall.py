import cv2
import numpy as np
import os
import glob
from datetime import datetime
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

RAIZ = r"C:\Users\48792583\Downloads\UpFallDataset"
CSV = "up_fall_omnifall.csv"
SALIDA = "../../data/keypoints/upfall"
MIN_FRAMES = 8

# 1 fall (la transición), 2 fallen (en el suelo) y el resto es adl
LABELS_EMERGENCIA = {1, 2}

MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pose_landmarker.task")

base_options = python.BaseOptions(model_asset_path=MODELO_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
)
landmarker = vision.PoseLandmarker.create_from_options(options)
yolo_model = YOLO("yolov8n.pt")


def timestamp_de(ruta):
    """'2018-07-04T12_06_21.055272.png' -> datetime"""
    base = os.path.splitext(os.path.basename(ruta))[0]
    return datetime.strptime(base, "%Y-%m-%dT%H_%M_%S.%f")


def listar_frames(carpeta):
    """Devuelve (rutas ordenadas, segundos desde el primer frame)."""
    archivos = glob.glob(os.path.join(carpeta, "*.png"))
    if not archivos:
        return [], []
    archivos.sort(key=timestamp_de)
    t0 = timestamp_de(archivos[0])
    segundos = [(timestamp_de(a) - t0).total_seconds() for a in archivos]
    return archivos, segundos


def detectar_y_recortar(frame, margen=0.2):
    res = yolo_model(frame, classes=[0], verbose=False)
    if len(res[0].boxes) == 0:
        return None
    b = res[0].boxes
    x1, y1, x2, y2 = b.xyxy[b.conf.argmax().item()].cpu().numpy()
    h, w = frame.shape[:2]
    aw, ah = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - aw * margen)); y1 = max(0, int(y1 - ah * margen))
    x2 = min(w, int(x2 + aw * margen)); y2 = min(h, int(y2 + ah * margen))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def keypoints_de_carpeta(carpeta):
    """Extrae keypoints de TODOS los frames, una sola vez por trial."""
    archivos, segundos = listar_frames(carpeta)
    if not archivos:
        return np.array([]), []

    frames = []
    for ruta in archivos:
        img = cv2.imread(ruta)
        if img is None:
            frames.append(np.zeros((33, 4)))
            continue
        crop = detectar_y_recortar(img)
        if crop is None:
            frames.append(np.zeros((33, 4)))
            continue
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        r = landmarker.detect(mp_img)
        frames.append(np.array([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks[0]])
                      if r.pose_landmarks else np.zeros((33, 4)))
    return np.array(frames), segundos


def procesar_trial(nombre_carpeta, segmentos):
    carpeta = os.path.join(RAIZ, nombre_carpeta)
    kp_todos, segundos = keypoints_de_carpeta(carpeta)

    if len(kp_todos) == 0:
        print(f"  {nombre_carpeta}: sin frames, salteando")
        return 0

    segundos = np.array(segundos)
    guardados = 0

    for i, seg in enumerate(segmentos.itertuples()):
        # busco qué frames caen dentro del rango de tiempo del segmento
        mask = (segundos >= seg.start) & (segundos < seg.end)
        clip = kp_todos[mask]
        if len(clip) < MIN_FRAMES:
            continue

        label_bin = "fall" if seg.label in LABELS_EMERGENCIA else "adl"
        nombre = f"{nombre_carpeta}_seg{i}_{label_bin}.npz"

        np.savez(
            os.path.join(SALIDA, nombre),
            keypoints=clip,
            label=label_bin,
            label_omnifall=seg.label,
            sujeto=f"upfall_s{seg.subject}",  # para no chocar con los otros datasets
            actividad=nombre_carpeta,
        )
        guardados += 1

    return guardados


if __name__ == "__main__":
    os.makedirs(SALIDA, exist_ok=True)

    df = pd.read_csv(CSV)
    # 'Subject17/Activity1/Trial3/Subject17Activity1Trial3Camera2' última parte
    df["carpeta"] = df["path"].str.split("/").str[-1]

    disponibles = {d for d in os.listdir(RAIZ) if os.path.isdir(os.path.join(RAIZ, d))}
    df = df[df["carpeta"].isin(disponibles)]

    carpetas = sorted(df["carpeta"].unique())
    print(f"Trials a procesar: {len(carpetas)}\n")

    total = 0
    for k, carpeta in enumerate(carpetas, 1):
        segmentos = df[df["carpeta"] == carpeta]
        print(f"[{k}/{len(carpetas)}] {carpeta} ({len(segmentos)} segmentos)...")
        try:
            total += procesar_trial(carpeta, segmentos)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nListo: {total} segmentos guardados en {SALIDA}")