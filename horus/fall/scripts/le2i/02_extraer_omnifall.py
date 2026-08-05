import cv2
import numpy as np
import os
import re
import glob
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

RAIZ = r"D:\download\FallDataset"
SALIDA = "../../data/keypoints/le2i_omnifall"
MIN_FRAMES = 8

# clases de Omnifall que cuentan como emergencia: 1=fall (la caída), 2=fallen (en el suelo)
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


def numero_de(nombre_archivo):
    nums = re.findall(r"\d+", os.path.basename(nombre_archivo))
    return int(nums[-1]) if nums else None


def indexar_videos(escenario):
    patron = os.path.join(RAIZ, escenario, "**", "*.avi")
    indice = {}
    for ruta in glob.glob(patron, recursive=True):
        n = numero_de(ruta)
        if n is not None:
            indice[n] = ruta
    return indice


def detectar_y_recortar_persona(frame, margen=0.2):
    resultados = yolo_model(frame, classes=[0], verbose=False)
    if len(resultados[0].boxes) == 0:
        return None

    boxes = resultados[0].boxes
    mejor_idx = boxes.conf.argmax().item()
    x1, y1, x2, y2 = boxes.xyxy[mejor_idx].cpu().numpy()

    h, w = frame.shape[:2]
    ancho, alto = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - ancho * margen))
    y1 = max(0, int(y1 - alto * margen))
    x2 = min(w, int(x2 + ancho * margen))
    y2 = min(h, int(y2 + alto * margen))
    return frame[y1:y2, x1:x2]


def extraer_keypoints_video(ruta_video):
    """Extrae los keypoints de TODOS los frames del video, una sola vez."""
    cap = cv2.VideoCapture(ruta_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        crop = detectar_y_recortar_persona(frame)
        if crop is None or crop.size == 0:
            frames.append(np.zeros((33, 4)))
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = landmarker.detect(mp_image)

        if resultado.pose_landmarks:
            lm = resultado.pose_landmarks[0]
            frames.append(np.array([[p.x, p.y, p.z, p.visibility] for p in lm]))
        else:
            frames.append(np.zeros((33, 4)))

    cap.release()
    return np.array(frames), fps


def procesar_video(escenario, num_video, ruta_video, segmentos):
    """segmentos: filas del csv correspondientes a este video."""
    kp_todos, fps = extraer_keypoints_video(ruta_video)

    if len(kp_todos) == 0 or fps <= 0:
        print(f"  {escenario}/video_{num_video}: sin frames o FPS inválido, salteando")
        return 0

    guardados = 0
    for i, seg in enumerate(segmentos.itertuples()):
        # los tiempos vienen en segundos -> los paso a índices de frame
        f_ini = int(round(seg.start * fps))
        f_fin = int(round(seg.end * fps))
        f_ini = max(0, f_ini)
        f_fin = min(len(kp_todos), f_fin)

        clip = kp_todos[f_ini:f_fin]
        if len(clip) < MIN_FRAMES:
            continue

        label_bin = "fall" if seg.label in LABELS_EMERGENCIA else "adl"
        nombre = f"{escenario}_video_{num_video}_seg{i}_{label_bin}.npz"

        np.savez(
            os.path.join(SALIDA, nombre),
            keypoints=clip,
            label=label_bin,
            label_omnifall=seg.label,
            sujeto=f"le2i_s{seg.subject}",   # prefijo para no chocar con los de CAUCAFall
            escenario=escenario,
            cam=seg.cam,
        )
        guardados += 1

    return guardados


if __name__ == "__main__":
    os.makedirs(SALIDA, exist_ok=True)

    df = pd.read_csv("le2i_omnifall.csv")
    df["escenario"] = df["path"].str.split("/").str[0]
    df["num_video"] = df["path"].str.extract(r"video_(\d+)").astype(int)

    indices = {esc: indexar_videos(esc) for esc in df["escenario"].unique()}
    total_segmentos = 0

    for (escenario, num_video), segmentos in df.groupby(["escenario", "num_video"]):
        ruta_video = indices[escenario].get(num_video)
        if ruta_video is None:
            print(f"Sin archivo para {escenario}/video_{num_video}, salteando")
            continue

        print(f"Procesando {escenario}/video_{num_video} ({len(segmentos)} segmentos)...")
        try:
            n = procesar_video(escenario, num_video, ruta_video, segmentos)
            total_segmentos += n
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nListo: {total_segmentos} segmentos guardados en {SALIDA}")