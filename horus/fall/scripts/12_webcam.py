import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.append("../src")
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from modelo_stgcn import STGCN
from grafo_mediapipe import construir_matriz_adyacencia
from preproceso import normalizar_frame, visibilidad_confiable
from detector_caidas import DetectorCaidas, ConfigDetector

_AQUI = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(_AQUI, "..", "checkpoints", "modelo_demo_todo.pt")
MODELO_POSE_PATH = os.path.join(_AQUI, "..", "pose_landmarker.task")
YOLO_PATH = os.path.join(_AQUI, "yolov8n.pt")

ANCHO_PROC = 640      # redimensiono antes de YOLO para no perder fps


def detectar_pose(frame, yolo, landmarker, device_yolo):
    """YOLO (mejor persona) -> recorte -> MediaPipe. Devuelve kp crudo (33,4) o None."""
    h, w = frame.shape[:2]
    # conf explicito: el default de ultralytics (~0.25) deja pasar detecciones
    # espurias en cuarto vacio (sombras, objetos) que despues el pipeline
    # confunde con una pose de caida.
    res = yolo(frame, classes=[0], conf=0.5, verbose=False, device=device_yolo)
    if not len(res[0].boxes):
        return None
    b = res[0].boxes
    x1, y1, x2, y2 = b.xyxy[b.conf.argmax().item()].cpu().numpy()
    aw, ah = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - aw * .2)); y1 = max(0, int(y1 - ah * .2))
    x2 = min(w, int(x2 + aw * .2)); y2 = min(h, int(y2 + ah * .2))
    crop = frame[y1:y2, x1:x2]
    if not crop.size:
        return None
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    r = landmarker.detect(img)
    if not r.pose_landmarks:
        return None
    return np.array([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks[0]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", type=int, default=0, help="indice de la webcam (0 = default)")
    ap.add_argument("--grabar", metavar="RUTA.npz",
                    help="vuelca (t, kp crudo por frame) para reproducir la sesion offline con 17_evaluar_evento.py")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    if not os.path.exists(CHECKPOINT):
        raise SystemExit(f"No encontré el checkpoint en {CHECKPOINT} — entrená primero (14_entrenar_final.py)")
    A = construir_matriz_adyacencia()
    modelo = STGCN(A, canales_entrada=4).to(device)
    modelo.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    modelo.eval()
    print("Modelo cargado del checkpoint")

    bo = python.BaseOptions(model_asset_path=MODELO_POSE_PATH)
    op = vision.PoseLandmarkerOptions(base_options=bo, running_mode=vision.RunningMode.IMAGE,
                                      num_poses=1, min_pose_detection_confidence=.3,
                                      min_pose_presence_confidence=.3)
    landmarker = vision.PoseLandmarker.create_from_options(op)
    yolo = YOLO(YOLO_PATH)
    device_yolo = 0 if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la cámara {args.camara}")
    print("\nCámara abierta. Apretá 'q' sobre la ventana para salir.\n")

    cfg = ConfigDetector()
    detector = DetectorCaidas(modelo, device, cfg)
    alarmas = []
    grabacion_t, grabacion_kp = [], []

    tiempos = {"pose": [], "detector": [], "total": []}
    t_inicio = time.perf_counter()

    while True:
        t_frame = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            print("Se cortó la señal de la cámara")
            break

        h0, w0 = frame.shape[:2]
        if w0 > ANCHO_PROC:
            escala = ANCHO_PROC / w0
            frame = cv2.resize(frame, (ANCHO_PROC, int(h0 * escala)))
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        kp = detectar_pose(frame, yolo, landmarker, device_yolo)
        tiempos["pose"].append(time.perf_counter() - t0)

        if args.grabar:
            grabacion_t.append(t_frame - t_inicio)
            grabacion_kp.append(kp if kp is not None else np.zeros((33, 4)))

        # descarto detecciones con baja confianza en cadera/hombro: suelen ser
        # recortes cortados (persona saliendo de cuadro, o semi-oculta acostada)
        # que dan una pose geometricamente distorsionada aunque mediapipe igual
        # devuelva landmarks.
        kp_norm = normalizar_frame(kp) if (kp is not None and visibilidad_confiable(kp)) else None

        t0 = time.perf_counter()
        estado = detector.actualizar(kp_norm, t_frame)
        tiempos["detector"].append(time.perf_counter() - t0)

        if estado.alarma_nueva:
            t_seg = t_frame - t_inicio
            alarmas.append(t_seg)
            print(f"  ALARMA a los {t_seg:.1f}s")

        if estado.n_buffer < cfg.ventana:
            texto, color = f"cargando buffer {estado.n_buffer}/{cfg.ventana}", (180, 180, 180)
        elif estado.alarma_activa:
            texto, color = "CAIDA DETECTADA", (0, 0, 255)
        else:
            texto, color = f"P={estado.prob:.2f}", (0, 200, 0)

        fps_inst = 1 / max(tiempos["total"][-1], 1e-6) if tiempos["total"] else 0
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, texto, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        cv2.putText(frame, f"{fps_inst:.0f} fps", (w - 90, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 200, 200), 1)

        cv2.imshow("Horus - deteccion de caidas", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        tiempos["total"].append(time.perf_counter() - t_frame)

    cap.release()
    cv2.destroyAllWindows()

    if args.grabar:
        np.savez(args.grabar, t=np.array(grabacion_t), kp=np.array(grabacion_kp),
                 fps=len(grabacion_t) / max(grabacion_t[-1], 1e-6) if grabacion_t else 0.0)
        print(f"\nSesión grabada en {args.grabar}: {len(grabacion_t)} frames")

    print(f"\nAlarmas disparadas: {[f'{a:.1f}s' for a in alarmas]}")
    print("\n--- velocidad del pipeline ---")
    for k in ["pose", "detector", "total"]:
        if tiempos[k]:
            print(f"{k:>9}: {np.mean(tiempos[k]) * 1000:6.1f} ms/frame")
    if tiempos["total"]:
        print(f"\nfps promedio: {1 / np.mean(tiempos['total']):.1f}")
