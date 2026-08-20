import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import deque
import cv2, os, time
import sys
sys.path.append("../src")

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from modelo_stgcn import STGCN
from grafo_mediapipe import construir_matriz_adyacencia

CARPETA = "../data/processed"
CHECKPOINT = "../checkpoints/modelo_demo_todo.pt"

CAMARA = 0            # 0 = webcam por defecto; probá 1 o 2 si tenés varias
ANCHO_PROC = 640      # redimensiono antes de YOLO para no perder fps
VENTANA, PASO, EPOCAS = 32, 4, 40

UMBRAL = 0.5          # punto de operación elegido en el barrido
PERSISTENCIA = 1.0    # segundos que la probabilidad debe sostenerse para disparar
TOLERANCIA_SIN_DETECCION_SEG = 0.6  # cuánto tiempo real tolero sin detección antes de resetear
UMBRAL_VISIBILIDAD = 0.5  # visibility mínima (mediapipe) de cadera/hombro para confiar en el frame

# --- regla de persistencia: exigir un pico de movimiento real antes de la alarma ---
# Una pose estática en el piso es ambigua (fallen vs. lying calmo). Lo que sí
# distingue una caída real es que HUBO un movimiento brusco justo antes. Sin esto,
# alguien que se acuesta y se queda quieto mucho tiempo le da al clasificador
# muchas oportunidades de fallar una sola vez y sostenerlo el segundo necesario.
FACTOR_PICO_VELOCIDAD = 3.0   # cuánto por encima del "ruido normal" cuenta como pico
VENTANA_POST_PICO_SEG = 4.0   # cuánto tiempo después de un pico dejo disparar la alarma
ALPHA_BASELINE = 0.05         # suavizado del nivel de "ruido normal" de velocidad

IDX_CI, IDX_CD, IDX_HI, IDX_HD = 23, 24, 11, 12
MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pose_landmarker.task")


def normalizar_frame(kp):
    """Normaliza UN frame: (33,4) -> (33,2). Solo usa ese frame, es causal por definición."""
    cad = (kp[IDX_CI, :2] + kp[IDX_CD, :2]) / 2
    hom = (kp[IDX_HI, :2] + kp[IDX_HD, :2]) / 2
    esc = np.linalg.norm(hom - cad)
    esc = esc if esc > 1e-6 else 1e-6
    return (kp[:, :2] - cad) / esc


def visibilidad_confiable(kp):
    """Promedio de 'visibility' de cadera/hombro. Un recorte cortado en el borde
    de cuadro (persona saliendo, o acostada y semi-ocluida) suele devolver
    landmarks igual, pero con visibility baja en estos puntos clave."""
    vis = kp[[IDX_CI, IDX_CD, IDX_HI, IDX_HD], 3]
    return vis.mean() >= UMBRAL_VISIBILIDAD


class DatasetLista(Dataset):
    def __init__(self, items):
        self.d, self.m = items, {"adl": 0, "fall": 1}

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        kp = self.d[i]["keypoints"]
        n = kp.shape[0]
        if n >= VENTANA:
            s = np.random.randint(0, n - VENTANA + 1)
            c = kp[s:s + VENTANA]
        else:
            c = np.concatenate([kp, np.repeat(kp[-1:], VENTANA - n, axis=0)], axis=0)
        return torch.tensor(c, dtype=torch.float32), self.m[self.d[i]["label"]]


def obtener_modelo(A, device):
    modelo = STGCN(A, canales_entrada=4).to(device)
    if os.path.exists(CHECKPOINT):
        modelo.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        print("Modelo cargado del checkpoint")
        return modelo

    # entreno con TODO el dataset: la persona frente a la cámara no está en él,
    # así que no hay riesgo de fuga
    tr = list(np.load(f"{CARPETA}/todos.npy", allow_pickle=True))
    print(f"Entrenando con {len(tr)} secuencias...")
    loader = DataLoader(DatasetLista(tr), batch_size=8, shuffle=True)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    na = sum(1 for d in tr if d["label"] == "adl")
    nf = sum(1 for d in tr if d["label"] == "fall")
    p = torch.tensor([1 / na, 1 / nf], dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=(p / p.sum() * 2).to(device))

    for ep in range(EPOCAS):
        modelo.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(modelo(x), y).backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  época {ep+1}/{EPOCAS}")

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(modelo.state_dict(), CHECKPOINT)
    print(f"Modelo guardado en {CHECKPOINT}")
    return modelo


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    A = construir_matriz_adyacencia()
    modelo = obtener_modelo(A, device)
    modelo.eval()

    bo = python.BaseOptions(model_asset_path=MODELO_PATH)
    op = vision.PoseLandmarkerOptions(base_options=bo, running_mode=vision.RunningMode.IMAGE,
                                      num_poses=1, min_pose_detection_confidence=.3,
                                      min_pose_presence_confidence=.3)
    landmarker = vision.PoseLandmarker.create_from_options(op)
    yolo = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAMARA)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la cámara {CAMARA}")

    print("\nCámara abierta. Apretá 'q' sobre la ventana para salir.\n")

    #  estado del sistema en vivo
    buffer = deque(maxlen=VENTANA)   # últimos 32 frames procesados
    ultimo_norm = None               # último frame válido, para rellenar huecos sin mirar al futuro
    anterior_norm = None             # frame previo, para la velocidad
    ultima_deteccion_ts = None       # reloj de la última detección confiable (None = todavía ninguna)
    prob_actual = 0.0
    inicio_racha = None
    alarma_activa = False
    alarmas = []

    baseline_vel = None               # nivel "normal" de velocidad, se adapta solo en calma
    ultimo_pico_ts = None             # reloj del último movimiento brusco real detectado

    tiempos = {"yolo": [], "pose": [], "modelo": [], "total": []}
    n_frame = 0
    t_inicio = time.perf_counter()

    while True:
        t_frame = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            print("Se cortó la señal de la cámara")
            break

        # redimensiono para que YOLO no se coma todos los fps
        h0, w0 = frame.shape[:2]
        if w0 > ANCHO_PROC:
            escala = ANCHO_PROC / w0
            frame = cv2.resize(frame, (ANCHO_PROC, int(h0 * escala)))
        h, w = frame.shape[:2]

        #detección de persona
        t0 = time.perf_counter()
        res = yolo(frame, classes=[0], verbose=False)
        tiempos["yolo"].append(time.perf_counter() - t0)

        crop = None
        if len(res[0].boxes):
            b = res[0].boxes
            x1, y1, x2, y2 = b.xyxy[b.conf.argmax().item()].cpu().numpy()
            aw, ah = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - aw * .2)); y1 = max(0, int(y1 - ah * .2))
            x2 = min(w, int(x2 + aw * .2)); y2 = min(h, int(y2 + ah * .2))
            c = frame[y1:y2, x1:x2]
            crop = c if c.size else None

        #pose
        t0 = time.perf_counter()
        kp_norm = None
        if crop is not None:
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            r = landmarker.detect(img)
            if r.pose_landmarks:
                kp = np.array([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks[0]])
                # descarto detecciones con baja confianza en cadera/hombro: suelen
                # ser recortes cortados (persona saliendo de cuadro, o semi-oculta
                # acostada) que dan una pose geométricamente distorsionada aunque
                # mediapipe igual devuelva landmarks.
                if visibilidad_confiable(kp):
                    kp_norm = normalizar_frame(kp)
        tiempos["pose"].append(time.perf_counter() - t0)

        if kp_norm is not None:
            ultima_deteccion_ts = t_frame

        sin_deteccion_seg = (t_frame - ultima_deteccion_ts) if ultima_deteccion_ts is not None else None

        if sin_deteccion_seg is None or sin_deteccion_seg > TOLERANCIA_SIN_DETECCION_SEG:
            # nunca hubo detección, o hace rato que no hay ninguna confiable:
            # dejo de arrastrar la última pose y reseteo todo el estado en vez
            # de seguir clasificando con datos viejos/congelados.
            buffer.clear()
            ultimo_norm = None
            anterior_norm = None
            inicio_racha = None
            alarma_activa = False
            prob_actual = 0.0
        else:
            deteccion_real_este_frame = kp_norm is not None

            # hueco corto (parpadeo de detección): repito el último válido,
            # nunca miro hacia adelante
            if kp_norm is None:
                kp_norm = ultimo_norm

            if kp_norm is not None:
                ultimo_norm = kp_norm
                vel = kp_norm - anterior_norm if anterior_norm is not None else np.zeros_like(kp_norm)
                anterior_norm = kp_norm
                buffer.append(np.concatenate([kp_norm, vel], axis=1))

                # detecto picos de movimiento solo con detecciones reales: un
                # frame repetido (fallback) tiene vel=0 y ensuciaría el baseline.
                if deteccion_real_este_frame:
                    v_escalar = np.linalg.norm(vel[[IDX_CI, IDX_CD, IDX_HI, IDX_HD]], axis=1).mean()
                    if baseline_vel is None:
                        baseline_vel = v_escalar
                    elif baseline_vel > 1e-6 and v_escalar > baseline_vel * FACTOR_PICO_VELOCIDAD:
                        ultimo_pico_ts = t_frame
                    else:
                        baseline_vel = baseline_vel * (1 - ALPHA_BASELINE) + v_escalar * ALPHA_BASELINE

        #clasificación: solo con el buffer lleno y cada PASO frames
        t0 = time.perf_counter()
        if len(buffer) == VENTANA and n_frame % PASO == 0:
            clip = torch.tensor(np.array(buffer), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                prob_actual = torch.softmax(modelo(clip), 1)[0, 1].item()

            # el tiempo lo da el reloj, no el contador de frames:
            # con webcam los fps son variables
            t_seg = time.perf_counter() - t_inicio

            if prob_actual >= UMBRAL:
                if inicio_racha is None:
                    inicio_racha = t_seg
                elif not alarma_activa and (t_seg - inicio_racha) >= PERSISTENCIA:
                    hubo_movimiento_brusco = (
                        ultimo_pico_ts is not None
                        and (t_frame - ultimo_pico_ts) <= VENTANA_POST_PICO_SEG
                    )
                    if hubo_movimiento_brusco:
                        alarma_activa = True
                        alarmas.append(t_seg)
                        print(f"  ALARMA a los {t_seg:.1f}s")
                    # si no hubo un movimiento brusco reciente, no disparo:
                    # es alguien quieto en el piso desde hace rato, no alguien
                    # que se acaba de caer.
            else:
                inicio_racha = None
                alarma_activa = False
        tiempos["modelo"].append(time.perf_counter() - t0)

        #dibujo el estado
        if len(buffer) < VENTANA:
            texto, color = f"cargando buffer {len(buffer)}/{VENTANA}", (180, 180, 180)
        elif alarma_activa:
            texto, color = "CAIDA DETECTADA", (0, 0, 255)
        else:
            texto, color = f"P={prob_actual:.2f}", (0, 200, 0)

        fps_inst = 1 / max(tiempos["total"][-1], 1e-6) if tiempos["total"] else 0
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, texto, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        cv2.putText(frame, f"{fps_inst:.0f} fps", (w - 90, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 200, 200), 1)

        cv2.imshow("Horus - deteccion de caidas", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        tiempos["total"].append(time.perf_counter() - t_frame)
        n_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nAlarmas disparadas: {[f'{a:.1f}s' for a in alarmas]}")
    print("\n--- velocidad del pipeline ---")
    for k in ["yolo", "pose", "modelo", "total"]:
        if tiempos[k]:
            print(f"{k:>7}: {np.mean(tiempos[k]) * 1000:6.1f} ms/frame")
    if tiempos["total"]:
        print(f"\nfps promedio: {1 / np.mean(tiempos['total']):.1f}")