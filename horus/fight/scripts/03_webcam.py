import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch

sys.path.append("../src")
from ultralytics import YOLO
from modelo_fight import crear_modelo, MEDIA_KINETICS, STD_KINETICS

CHECKPOINT = "../checkpoints/modelo_fight.pt"

CAMARA = 0
ANCHO_PROC = 640
LADO = 128            # los frames se guardan a 128 y se recortan a 112, igual que en el entrenamiento
LADO_FINAL = 112
VENTANA = 32          # frames por clip que espera el modelo

# RWF-2000 son clips de 5s. Muestreo el buffer POR TIEMPO (no por frame) para que
# el clip de 32 frames siempre cubra ~5s, sin importar a cuántos fps vaya el webcam
# -- si no, con un webcam a 30fps el clip serían ~1s y el modelo vería el movimiento
# "acelerado" respecto a lo que entrenó.
DURACION_CLIP_SEG = 5.0
INTERVALO_BUFFER = DURACION_CLIP_SEG / VENTANA

PASO_CLASIF_SEG = 0.75    # cada cuánto corro el clasificador (no cada frame)
MIN_PERSONAS = 2          # el gate: sin 2+ personas no tiene sentido buscar agresión
UMBRAL = 0.5
PERSISTENCIA_SEG = 2.0    # cuánto tiene que sostenerse la prob de fight para disparar la alarma
TOLERANCIA_GATE_SEG = 1.5  # cuánto tolero el gate cerrado antes de tirar el buffer

_media = np.array(MEDIA_KINETICS, dtype=np.float32)
_std = np.array(STD_KINETICS, dtype=np.float32)


def preprocesar_frame(frame_bgr):
    """BGR -> RGB, resize a 128x128. Devuelve uint8 (128,128,3)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (LADO, LADO))


def armar_clip(buffer):
    """deque de (128,128,3) uint8 -> tensor (1,3,32,112,112) normalizado,
    mismo preprocesamiento que RWFDataset en modo val (center crop, sin flip)."""
    frames = np.stack(buffer).astype(np.float32) / 255.0  # (T,128,128,3)
    off = (LADO - LADO_FINAL) // 2
    frames = frames[:, off:off + LADO_FINAL, off:off + LADO_FINAL, :]
    frames = (frames - _media) / _std
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,T,H,W)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    if not os.path.exists(CHECKPOINT):
        raise SystemExit(f"No encontré el checkpoint en {CHECKPOINT} — entrená primero con 02_entrenar.py")
    modelo = crear_modelo(preentrenado=False).to(device)
    modelo.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    modelo.eval()
    print("Modelo de agresión cargado")

    # en el sistema integrado este YOLO es el mismo que ya corre para caídas,
    # acá lo instancio aparte solo para probar el módulo suelto
    yolo = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAMARA)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la cámara {CAMARA}")
    print("\nCámara abierta. Apretá 'q' sobre la ventana para salir.\n")

    buffer = deque(maxlen=VENTANA)          # últimos 32 frames muestreados (128x128 RGB)
    ultimo_agregado_ts = 0.0
    ultima_clasif_ts = 0.0
    ultimo_gate_abierto_ts = None
    prob_actual = 0.0
    inicio_racha = None
    alarma_activa = False
    alarmas = []

    tiempos = {"yolo": [], "modelo": [], "total": []}
    t_inicio = time.perf_counter()
    n_frame = 0

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

        # --- YOLO: solo para contar personas (el gate) ---
        t0 = time.perf_counter()
        res = yolo(frame, classes=[0], conf=0.5, verbose=False)
        tiempos["yolo"].append(time.perf_counter() - t0)
        n_personas = len(res[0].boxes)
        gate_abierto = n_personas >= MIN_PERSONAS

        if gate_abierto:
            ultimo_gate_abierto_ts = t_frame
            if t_frame - ultimo_agregado_ts >= INTERVALO_BUFFER:
                buffer.append(preprocesar_frame(frame))
                ultimo_agregado_ts = t_frame
        elif ultimo_gate_abierto_ts is None or (t_frame - ultimo_gate_abierto_ts) > TOLERANCIA_GATE_SEG:
            # gate cerrado hace rato: lo que hay en el buffer ya no representa
            # la escena actual, lo tiro y reseteo la racha
            buffer.clear()
            inicio_racha = None
            alarma_activa = False
            prob_actual = 0.0

        # --- clasificación: gate abierto, buffer lleno, y cada PASO_CLASIF_SEG ---
        t0 = time.perf_counter()
        if gate_abierto and len(buffer) == VENTANA and t_frame - ultima_clasif_ts >= PASO_CLASIF_SEG:
            ultima_clasif_ts = t_frame
            clip = armar_clip(buffer).to(device)
            with torch.no_grad():
                prob_actual = torch.softmax(modelo(clip), 1)[0, 1].item()

            t_seg = time.perf_counter() - t_inicio
            if prob_actual >= UMBRAL:
                if inicio_racha is None:
                    inicio_racha = t_seg
                elif not alarma_activa and (t_seg - inicio_racha) >= PERSISTENCIA_SEG:
                    alarma_activa = True
                    alarmas.append(t_seg)
                    print(f"  ALARMA (agresión) a los {t_seg:.1f}s")
            else:
                inicio_racha = None
                alarma_activa = False
        tiempos["modelo"].append(time.perf_counter() - t0)

        # --- dibujo del estado ---
        if not gate_abierto:
            texto, color = f"gate cerrado ({n_personas} pers.)", (150, 150, 150)
        elif len(buffer) < VENTANA:
            texto, color = f"cargando buffer {len(buffer)}/{VENTANA}", (180, 180, 180)
        elif alarma_activa:
            texto, color = "AGRESION DETECTADA", (0, 0, 255)
        else:
            texto, color = f"P={prob_actual:.2f}  ({n_personas} pers.)", (0, 200, 0)

        fps_inst = 1 / max(tiempos["total"][-1], 1e-6) if tiempos["total"] else 0
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, texto, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        cv2.putText(frame, f"{fps_inst:.0f} fps", (w - 90, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 200, 200), 1)

        cv2.imshow("Horus - deteccion de agresiones", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        tiempos["total"].append(time.perf_counter() - t_frame)
        n_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nAlarmas disparadas: {[f'{a:.1f}s' for a in alarmas]}")
    print("\n--- velocidad del pipeline ---")
    for k in ["yolo", "modelo", "total"]:
        if tiempos[k]:
            print(f"{k:>7}: {np.mean(tiempos[k]) * 1000:6.1f} ms/frame")
    if tiempos["total"]:
        print(f"\nfps promedio: {1 / np.mean(tiempos['total']):.1f}")
