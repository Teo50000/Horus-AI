import cv2
import numpy as np
import os
import glob
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

VENTANA_MINIMA = 8  # frames mínimos para guardar un clip adl


def leer_anotacion(ruta_txt):
    """Devuelve (inicio_caida, fin_caida, bboxes) donde bboxes es {frame_num: (x1,y1,x2,y2)}"""
    with open(ruta_txt) as f:
        lineas = [l.strip() for l in f.readlines() if l.strip()]

    try:
        inicio_caida = int(lineas[0])
        fin_caida = int(lineas[1])
        resto = lineas[2:]
    except (ValueError, IndexError):
        # este archivo no tiene el header de inicio/fin -> asumimos sin caída
        inicio_caida = 0
        fin_caida = 0
        resto = lineas

    bboxes = {}
    for linea in resto:
        partes = linea.split(",")
        if len(partes) < 6:
            continue
        frame_num = int(partes[0])
        x1, y1, x2, y2 = int(partes[2]), int(partes[3]), int(partes[4]), int(partes[5])
        bboxes[frame_num] = (x1, y1, x2, y2)

    return inicio_caida, fin_caida, bboxes


def extraer_keypoints_video(ruta_video, bboxes, margen=0.2):
    cap = cv2.VideoCapture(ruta_video)
    frames_keypoints = []
    frame_idx = 1  # la anotación empieza a contar en 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        bbox = bboxes.get(frame_idx)
        if bbox is None:
            frames_keypoints.append(np.zeros((33, 4)))
            frame_idx += 1
            continue

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        ancho_box = x2 - x1
        alto_box = y2 - y1

        x1 = max(0, int(x1 - ancho_box * margen))
        y1 = max(0, int(y1 - alto_box * margen))
        x2 = min(w, int(x2 + ancho_box * margen))
        y2 = min(h, int(y2 + alto_box * margen))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            frames_keypoints.append(np.zeros((33, 4)))
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = landmarker.detect(mp_image)

        if resultado.pose_landmarks:
            lm = resultado.pose_landmarks[0]
            kp = np.array([[p.x, p.y, p.z, p.visibility] for p in lm])
        else:
            kp = np.zeros((33, 4))

        frames_keypoints.append(kp)
        frame_idx += 1

    cap.release()
    return np.array(frames_keypoints)


def procesar_video(ruta_video, ruta_txt, nombre_base, salida_dir):
    os.makedirs(salida_dir, exist_ok=True)
    inicio, fin, bboxes = leer_anotacion(ruta_txt)
    kp_todos = extraer_keypoints_video(ruta_video, bboxes)

    if len(kp_todos) == 0:
        print(f"  {nombre_base}: sin frames, salteando")
        return

    if inicio == 0 and fin == 0:
        np.savez(os.path.join(salida_dir, f"{nombre_base}_adl_full.npz"),
                 keypoints=kp_todos, label="adl", archivo_origen=nombre_base)
        print(f"  {nombre_base}: sin caída, guardado como adl_full ({len(kp_todos)} frames)")
        return

    idx_inicio = inicio - 1
    idx_fin = fin

    clip_fall = kp_todos[idx_inicio:idx_fin]
    if len(clip_fall) > 0:
        np.savez(os.path.join(salida_dir, f"{nombre_base}_fall.npz"),
                 keypoints=clip_fall, label="fall", archivo_origen=nombre_base)

    clip_antes = kp_todos[:idx_inicio]
    if len(clip_antes) >= VENTANA_MINIMA:
        np.savez(os.path.join(salida_dir, f"{nombre_base}_adl_antes.npz"),
                 keypoints=clip_antes, label="adl", archivo_origen=nombre_base)

    clip_despues = kp_todos[idx_fin:]
    if len(clip_despues) >= VENTANA_MINIMA:
        np.savez(os.path.join(salida_dir, f"{nombre_base}_adl_despues.npz"),
                 keypoints=clip_despues, label="adl", archivo_origen=nombre_base)

    print(f"  {nombre_base}: fall={len(clip_fall)} frames, "
          f"adl_antes={len(clip_antes)}, adl_despues={len(clip_despues)}")


def procesar_carpeta(carpeta_videos, carpeta_anotaciones, salida_dir, prefijo=""):
    os.makedirs(salida_dir, exist_ok=True)
    videos = glob.glob(os.path.join(carpeta_videos, "*.avi"))

    for ruta_video in videos:
        nombre_video = os.path.splitext(os.path.basename(ruta_video))[0]
        nombre_base = f"{prefijo}_{nombre_video}" if prefijo else nombre_video
        ruta_txt = os.path.join(carpeta_anotaciones, f"{nombre_video}.txt")

        if not os.path.exists(ruta_txt):
            print(f"Sin anotación para {nombre_base}, salteando")
            continue

        print(f"Procesando {nombre_base}...")
        try:
            procesar_video(ruta_video, ruta_txt, nombre_base, salida_dir)
        except Exception as e:
            print(f"  ERROR en {nombre_base}: {e}")
            continue


if __name__ == "__main__":
    RAIZ = r"D:\download\FallDataset"
    carpetas = ["Home_01", "Home_02", "Coffee_room_01", "Coffee_room_02", "Office", "Lecture_room"]

    for carpeta in carpetas:
        ruta_base = os.path.join(RAIZ, carpeta, carpeta)
        procesar_carpeta(
            carpeta_videos=os.path.join(ruta_base, "Videos"),
            carpeta_anotaciones=os.path.join(ruta_base, "Annotation_files"),
            salida_dir="../../data/keypoints/le2i",
            prefijo=carpeta,
        )