import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests
import torch

# todo relativo al archivo, no al directorio de trabajo: asi el script corre
# desde donde sea, no solo haciendo cd a 07_alerta/
_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_AQUI, "..", "fall", "src"))
sys.path.append(os.path.join(_AQUI, "..", "fight", "src"))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from modelo_stgcn import STGCN
from grafo_mediapipe import construir_matriz_adyacencia
from modelo_fight import crear_modelo as crear_modelo_fight, MEDIA_KINETICS, STD_KINETICS

# Un solo loop de camara + un solo YOLO por frame, alimenta los dos pipelines
# (caidas y agresion) en paralelo -- comparten la deteccion de personas pero
# cada uno tiene su propio buffer, su propio modelo y su propia alarma.

# --- backend / alertas ---
URL_ALERTA = "http://localhost:8000/camaras/prediccion"
CAMARA_ID = "cam1"  # en un despliegue con varias camaras, cada proceso corre con su propio id

# --- camara ---
CAMARA = 0
ANCHO_PROC = 640

# =================== caidas (ST-GCN sobre pose) ===================
CHECKPOINT_FALL = os.path.join(_AQUI, "..", "fall", "checkpoints", "modelo_demo_todo.pt")
MODELO_POSE_PATH = os.path.join(_AQUI, "..", "fall", "pose_landmarker.task")

VENTANA_FALL, PASO_FALL = 32, 4
UMBRAL_FALL = 0.5
PERSISTENCIA_FALL = 1.0
TOLERANCIA_SIN_DETECCION_SEG = 0.6
UMBRAL_VISIBILIDAD = 0.5

FACTOR_PICO_VELOCIDAD = 3.0
VENTANA_POST_PICO_SEG = 4.0
ALPHA_BASELINE = 0.05
VENTANA_VERTICAL_PREVIA_SEG = 2.0
CAIDA_VERTICAL_MINIMA = 0.35

IDX_CI, IDX_CD, IDX_HI, IDX_HD = 23, 24, 11, 12


def normalizar_frame_pose(kp):
    """Normaliza UN frame de pose: (33,4) -> (33,2). Causal por definicion."""
    cad = (kp[IDX_CI, :2] + kp[IDX_CD, :2]) / 2
    hom = (kp[IDX_HI, :2] + kp[IDX_HD, :2]) / 2
    esc = np.linalg.norm(hom - cad)
    esc = esc if esc > 1e-6 else 1e-6
    return (kp[:, :2] - cad) / esc


def visibilidad_confiable(kp):
    vis = kp[[IDX_CI, IDX_CD, IDX_HI, IDX_HD], 3]
    return vis.mean() >= UMBRAL_VISIBILIDAD


# =================== agresion (MC3-18 sobre clip de video) ===================
CHECKPOINT_FIGHT = os.path.join(_AQUI, "..", "fight", "checkpoints", "modelo_fight.pt")

LADO, LADO_FINAL = 128, 112
VENTANA_FIGHT = 32
DURACION_CLIP_SEG = 5.0  # RWF-2000 son clips de 5s; muestreo el buffer por tiempo
INTERVALO_BUFFER_FIGHT = DURACION_CLIP_SEG / VENTANA_FIGHT

PASO_CLASIF_SEG_FIGHT = 0.75
MIN_PERSONAS = 2
UMBRAL_FIGHT = 0.5
PERSISTENCIA_FIGHT = 2.0
TOLERANCIA_GATE_SEG = 1.5
# si paso mas de esto sin poder agregar frames (gate que parpadea), el clip
# quedaria pegoteado de momentos no contiguos -- nada parecido a los 5s
# continuos con los que entreno el modelo. Mejor arrancar el buffer de nuevo.
GAP_MAX_BUFFER_SEG = 1.0

_media_fight = np.array(MEDIA_KINETICS, dtype=np.float32)
_std_fight = np.array(STD_KINETICS, dtype=np.float32)


def preprocesar_frame_fight(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (LADO, LADO))


def armar_clip_fight(buffer):
    frames = np.stack(buffer).astype(np.float32) / 255.0
    off = (LADO - LADO_FINAL) // 2
    frames = frames[:, off:off + LADO_FINAL, off:off + LADO_FINAL, :]
    frames = (frames - _media_fight) / _std_fight
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)


def enviar_alerta(tipo, probabilidad):
    """POST al backend con el tipo de evento, la probabilidad y la camara que
    disparo. Va en un hilo aparte: si el backend esta caido, requests se queda
    esperando el timeout y no quiero que el loop de video se frene justo en el
    momento de la alarma."""
    payload = {
        "camara_id": CAMARA_ID,
        "tipo": tipo,
        "probabilidad": round(float(probabilidad), 3),
        "timestamp": time.time(),
    }

    def _postear():
        try:
            requests.post(URL_ALERTA, json=payload, timeout=2)
        except requests.RequestException as e:
            print(f"  (no se pudo avisar al backend: {e})")

    threading.Thread(target=_postear, daemon=True).start()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    if not os.path.exists(CHECKPOINT_FALL):
        raise SystemExit(f"No encontré el checkpoint de caídas en {CHECKPOINT_FALL}")
    if not os.path.exists(CHECKPOINT_FIGHT):
        raise SystemExit(f"No encontré el checkpoint de agresión en {CHECKPOINT_FIGHT}")

    A = construir_matriz_adyacencia()
    modelo_fall = STGCN(A, canales_entrada=4).to(device)
    modelo_fall.load_state_dict(torch.load(CHECKPOINT_FALL, map_location=device))
    modelo_fall.eval()

    modelo_fight = crear_modelo_fight(preentrenado=False).to(device)
    modelo_fight.load_state_dict(torch.load(CHECKPOINT_FIGHT, map_location=device))
    modelo_fight.eval()
    print("Modelos de caída y agresión cargados")

    bo = python.BaseOptions(model_asset_path=MODELO_POSE_PATH)
    op = vision.PoseLandmarkerOptions(base_options=bo, running_mode=vision.RunningMode.IMAGE,
                                      num_poses=1, min_pose_detection_confidence=.3,
                                      min_pose_presence_confidence=.3)
    landmarker = vision.PoseLandmarker.create_from_options(op)
    yolo = YOLO("yolov8n.pt")  # un solo YOLO, alimenta caídas (mejor persona) y agresión (conteo)

    cap = cv2.VideoCapture(CAMARA)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la cámara {CAMARA}")
    print(f"\nCámara abierta (id={CAMARA_ID}). Apretá 'q' sobre la ventana para salir.\n")

    # --- estado caidas ---
    buffer_fall = deque(maxlen=VENTANA_FALL)
    ultimo_norm = None
    anterior_norm = None
    ultima_deteccion_ts = None
    prob_fall = 0.0
    inicio_racha_fall = None
    alarma_fall = False
    alarmas_fall = []
    baseline_vel = None
    ultimo_pico_ts = None
    historial_vertical = deque()

    # --- estado agresion ---
    buffer_fight = deque(maxlen=VENTANA_FIGHT)
    ultimo_agregado_ts = 0.0
    ultima_clasif_ts_fight = 0.0
    ultimo_gate_abierto_ts = None
    prob_fight = 0.0
    inicio_racha_fight = None
    alarma_fight = False
    alarmas_fight = []

    # ventana rodante: este script queda corriendo indefinidamente, con listas
    # sin tope las metricas se comen la RAM de a poco
    tiempos = {k: deque(maxlen=2000) for k in
               ("yolo", "pose", "modelo_fall", "modelo_fight", "total")}
    n_frame = 0
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

        # ---------- deteccion de personas: UNA sola vez, la usan los dos pipelines ----------
        t0 = time.perf_counter()
        res = yolo(frame, classes=[0], conf=0.5, verbose=False)
        tiempos["yolo"].append(time.perf_counter() - t0)
        boxes = res[0].boxes
        n_personas = len(boxes)

        # ==================== pipeline de caidas ====================
        crop = None
        if n_personas:
            x1, y1, x2, y2 = boxes.xyxy[boxes.conf.argmax().item()].cpu().numpy()
            aw, ah = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - aw * .2)); y1 = max(0, int(y1 - ah * .2))
            x2 = min(w, int(x2 + aw * .2)); y2 = min(h, int(y2 + ah * .2))
            c = frame[y1:y2, x1:x2]
            crop = c if c.size else None

        t0 = time.perf_counter()
        kp_norm = None
        if crop is not None:
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            r = landmarker.detect(img)
            if r.pose_landmarks:
                kp = np.array([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks[0]])
                if visibilidad_confiable(kp):
                    kp_norm = normalizar_frame_pose(kp)
        tiempos["pose"].append(time.perf_counter() - t0)

        if kp_norm is not None:
            ultima_deteccion_ts = t_frame
        sin_deteccion_seg = (t_frame - ultima_deteccion_ts) if ultima_deteccion_ts is not None else None

        if sin_deteccion_seg is None or sin_deteccion_seg > TOLERANCIA_SIN_DETECCION_SEG:
            buffer_fall.clear()
            ultimo_norm = None
            anterior_norm = None
            inicio_racha_fall = None
            alarma_fall = False
            prob_fall = 0.0
            historial_vertical.clear()
            baseline_vel = None
            ultimo_pico_ts = None
        else:
            deteccion_real_este_frame = kp_norm is not None
            if kp_norm is None:
                kp_norm = ultimo_norm

            if kp_norm is not None:
                ultimo_norm = kp_norm
                vel = kp_norm - anterior_norm if anterior_norm is not None else np.zeros_like(kp_norm)
                anterior_norm = kp_norm
                buffer_fall.append(np.concatenate([kp_norm, vel], axis=1))

                if deteccion_real_este_frame:
                    v_escalar = np.linalg.norm(vel[[IDX_CI, IDX_CD, IDX_HI, IDX_HD]], axis=1).mean()
                    verticalidad = -((kp_norm[IDX_HI, 1] + kp_norm[IDX_HD, 1]) / 2)

                    if baseline_vel is None:
                        baseline_vel = v_escalar
                    elif baseline_vel > 1e-6 and v_escalar > baseline_vel * FACTOR_PICO_VELOCIDAD:
                        cobertura_seg = t_frame - historial_vertical[0][0] if historial_vertical else 0.0
                        if cobertura_seg < VENTANA_VERTICAL_PREVIA_SEG * 0.8:
                            ultimo_pico_ts = t_frame
                        else:
                            vertical_previa_max = max(v for _, v in historial_vertical)
                            if (vertical_previa_max - verticalidad) >= CAIDA_VERTICAL_MINIMA:
                                ultimo_pico_ts = t_frame
                    else:
                        baseline_vel = baseline_vel * (1 - ALPHA_BASELINE) + v_escalar * ALPHA_BASELINE

                    historial_vertical.append((t_frame, verticalidad))
                    while historial_vertical and t_frame - historial_vertical[0][0] > VENTANA_VERTICAL_PREVIA_SEG:
                        historial_vertical.popleft()

        t0 = time.perf_counter()
        if len(buffer_fall) == VENTANA_FALL and n_frame % PASO_FALL == 0:
            clip_fall = torch.tensor(np.array(buffer_fall), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                prob_fall = torch.softmax(modelo_fall(clip_fall), 1)[0, 1].item()

            t_seg = time.perf_counter() - t_inicio
            if prob_fall >= UMBRAL_FALL:
                if inicio_racha_fall is None:
                    inicio_racha_fall = t_seg
                elif not alarma_fall and (t_seg - inicio_racha_fall) >= PERSISTENCIA_FALL:
                    hubo_movimiento_brusco = (
                        ultimo_pico_ts is not None
                        and (t_frame - ultimo_pico_ts) <= VENTANA_POST_PICO_SEG
                    )
                    if hubo_movimiento_brusco:
                        alarma_fall = True
                        alarmas_fall.append(t_seg)
                        print(f"  ALARMA (caída) a los {t_seg:.1f}s")
                        enviar_alerta("caida", prob_fall)
            else:
                inicio_racha_fall = None
                alarma_fall = False
        tiempos["modelo_fall"].append(time.perf_counter() - t0)

        # ==================== pipeline de agresion ====================
        gate_abierto = n_personas >= MIN_PERSONAS

        if gate_abierto:
            ultimo_gate_abierto_ts = t_frame
            if t_frame - ultimo_agregado_ts >= INTERVALO_BUFFER_FIGHT:
                if buffer_fight and (t_frame - ultimo_agregado_ts) > GAP_MAX_BUFFER_SEG:
                    buffer_fight.clear()  # hubo un hueco, el clip seria discontinuo
                buffer_fight.append(preprocesar_frame_fight(frame))
                ultimo_agregado_ts = t_frame
        elif ultimo_gate_abierto_ts is None or (t_frame - ultimo_gate_abierto_ts) > TOLERANCIA_GATE_SEG:
            buffer_fight.clear()
            inicio_racha_fight = None
            alarma_fight = False
            prob_fight = 0.0

        t0 = time.perf_counter()
        if (gate_abierto and len(buffer_fight) == VENTANA_FIGHT
                and t_frame - ultima_clasif_ts_fight >= PASO_CLASIF_SEG_FIGHT):
            ultima_clasif_ts_fight = t_frame
            clip_fight = armar_clip_fight(buffer_fight).to(device)
            with torch.no_grad():
                prob_fight = torch.softmax(modelo_fight(clip_fight), 1)[0, 1].item()

            t_seg = time.perf_counter() - t_inicio
            if prob_fight >= UMBRAL_FIGHT:
                if inicio_racha_fight is None:
                    inicio_racha_fight = t_seg
                elif not alarma_fight and (t_seg - inicio_racha_fight) >= PERSISTENCIA_FIGHT:
                    alarma_fight = True
                    alarmas_fight.append(t_seg)
                    print(f"  ALARMA (agresión) a los {t_seg:.1f}s")
                    enviar_alerta("agresion", prob_fight)
            else:
                inicio_racha_fight = None
                alarma_fight = False
        tiempos["modelo_fight"].append(time.perf_counter() - t0)

        # ---------- dibujo ----------
        if len(buffer_fall) < VENTANA_FALL:
            texto_fall, color_fall = f"caida: cargando {len(buffer_fall)}/{VENTANA_FALL}", (180, 180, 180)
        elif alarma_fall:
            texto_fall, color_fall = "CAIDA DETECTADA", (0, 0, 255)
        else:
            texto_fall, color_fall = f"caida P={prob_fall:.2f}", (0, 200, 0)

        if not gate_abierto:
            texto_fight, color_fight = f"agresion: gate cerrado ({n_personas} pers.)", (150, 150, 150)
        elif len(buffer_fight) < VENTANA_FIGHT:
            texto_fight, color_fight = f"agresion: cargando {len(buffer_fight)}/{VENTANA_FIGHT}", (180, 180, 180)
        elif alarma_fight:
            texto_fight, color_fight = "AGRESION DETECTADA", (0, 0, 255)
        else:
            texto_fight, color_fight = f"agresion P={prob_fight:.2f}", (0, 200, 0)

        fps_inst = 1 / max(tiempos["total"][-1], 1e-6) if tiempos["total"] else 0
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(frame, texto_fall, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, color_fall, 2)
        cv2.putText(frame, texto_fight, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, .55, color_fight, 2)
        cv2.putText(frame, f"{fps_inst:.0f} fps  [{CAMARA_ID}]", (w - 150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 200, 200), 1)

        cv2.imshow("Horus - caidas + agresion", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        tiempos["total"].append(time.perf_counter() - t_frame)
        n_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nAlarmas de caída: {[f'{a:.1f}s' for a in alarmas_fall]}")
    print(f"Alarmas de agresión: {[f'{a:.1f}s' for a in alarmas_fight]}")
    print("\n--- velocidad del pipeline (ultimos 2000 frames) ---")
    for k in ["yolo", "pose", "modelo_fall", "modelo_fight", "total"]:
        if tiempos[k]:
            print(f"{k:>12}: {np.mean(tiempos[k]) * 1000:6.1f} ms/frame")
    if tiempos["total"]:
        print(f"\nfps promedio: {1 / np.mean(tiempos['total']):.1f}")
