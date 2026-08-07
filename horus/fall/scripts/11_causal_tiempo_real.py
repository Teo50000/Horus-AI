import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import deque
import cv2, re, glob, os, time
import pandas as pd
import sys
sys.path.append("../src")

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from modelo_stgcn import STGCN
from grafo_mediapipe import construir_matriz_adyacencia

CARPETA = "../data/processed"
RAIZ_VIDEOS = r"D:\download\FallDataset"
CSV_OMNIFALL = "le2i/le2i_omnifall.csv"
CHECKPOINT = "../checkpoints/modelo_demo_s0.pt"
SUJETO = 0
ESCENARIO, NUM_VIDEO = "Home_02", 31
VENTANA, PASO, EPOCAS = 32, 4, 40

UMBRAL = 0.5          # punto de operación elegido del barrido
PERSISTENCIA = 1.0    # segundos que la probabilidad debe sostenerse

IDX_CI, IDX_CD, IDX_HI, IDX_HD = 23, 24, 11, 12
MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pose_landmarker.task")


def normalizar_frame(kp):
    #Normaliza un frame: (33,4) -> (33,2). Solo usa ese frame, es causal por definición.
    cad = (kp[IDX_CI, :2] + kp[IDX_CD, :2]) / 2
    hom = (kp[IDX_HI, :2] + kp[IDX_HD, :2]) / 2
    esc = np.linalg.norm(hom - cad)
    esc = esc if esc > 1e-6 else 1e-6
    return (kp[:, :2] - cad) / esc


# ---------- entrenamiento (igual que antes, solo para tener el modelo) ----------
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

    items = list(np.load(f"{CARPETA}/todos.npy", allow_pickle=True))
    tr = [d for d in items if d["grupo"] != f"le2i_s{SUJETO}"]
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
    torch.save(modelo.state_dict(), CHECKPOINT)
    print(f"Modelo guardado en {CHECKPOINT}")
    return modelo


def buscar_video(esc, num):
    for r in glob.glob(os.path.join(RAIZ_VIDEOS, esc, "**", "*.avi"), recursive=True):
        n = re.findall(r"\d+", os.path.basename(r))
        if n and int(n[-1]) == num:
            return r
    return None


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

    ruta = buscar_video(ESCENARIO, NUM_VIDEO)
    cap = cv2.VideoCapture(ruta)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("demo_causal.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (w, h))

    # ---------- estado del sistema "en vivo" ----------
    buffer = deque(maxlen=VENTANA)   # últimos 32 frames ya procesados
    ultimo_norm = None               # último frame válido, para rellenar huecos SIN mirar al futuro
    anterior_norm = None             # frame previo, para calcular la velocidad
    prob_actual = 0.0
    inicio_racha = None              # cuándo empezó la racha por encima del umbral
    alarma_activa = False
    alarmas = []

    tiempos = {"yolo": [], "pose": [], "modelo": [], "total": []}
    n_frame = 0

    while True:
        t_frame = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        # --- 1. detección de persona ---
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

        # --- 2. pose ---
        t0 = time.perf_counter()
        kp_norm = None
        if crop is not None:
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            r = landmarker.detect(img)
            if r.pose_landmarks:
                kp = np.array([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks[0]])
                kp_norm = normalizar_frame(kp)
        tiempos["pose"].append(time.perf_counter() - t0)

        # sin detección: repito el último válido (nunca miro frames futuros)
        if kp_norm is None:
            kp_norm = ultimo_norm
        if kp_norm is None:
            n_frame += 1
            out.write(frame)
            continue
        ultimo_norm = kp_norm

        vel = kp_norm - anterior_norm if anterior_norm is not None else np.zeros_like(kp_norm)
        anterior_norm = kp_norm
        buffer.append(np.concatenate([kp_norm, vel], axis=1))

        # --- 3. clasificación, solo con el buffer lleno y cada PASO frames ---
        t0 = time.perf_counter()
        if len(buffer) == VENTANA and n_frame % PASO == 0:
            clip = torch.tensor(np.array(buffer), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                prob_actual = torch.softmax(modelo(clip), 1)[0, 1].item()

            t_seg = n_frame / fps_video
            if prob_actual >= UMBRAL:
                if inicio_racha is None:
                    inicio_racha = t_seg
                elif not alarma_activa and (t_seg - inicio_racha) >= PERSISTENCIA:
                    alarma_activa = True
                    alarmas.append(t_seg)
                    print(f"  ALARMA a los {t_seg:.2f}s")
            else:
                inicio_racha = None
                alarma_activa = False
        tiempos["modelo"].append(time.perf_counter() - t0)

        # --- 4. dibujo el estado sobre el frame ---
        color = (0, 0, 255) if alarma_activa else (0, 200, 0)
        texto = "CAIDA DETECTADA" if alarma_activa else f"P={prob_actual:.2f}"
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, texto, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        out.write(frame)

        tiempos["total"].append(time.perf_counter() - t_frame)
        n_frame += 1

    cap.release()
    out.release()

    print(f"\nAlarmas disparadas: {[f'{a:.2f}s' for a in alarmas]}")

    df = pd.read_csv(CSV_OMNIFALL)
    df["escenario"] = df["path"].str.split("/").str[0]
    df["num"] = df["path"].str.extract(r"video_(\d+)").astype(int)
    segs = df[(df["escenario"] == ESCENARIO) & (df["num"] == NUM_VIDEO)]
    caidas = segs[segs["label"].isin([1, 2])]
    if len(caidas):
        print(f"Caída real empieza a los {caidas['start'].min():.2f}s")

    print("\n--- velocidad del pipeline ---")
    for k in ["yolo", "pose", "modelo", "total"]:
        ms = np.mean(tiempos[k]) * 1000
        print(f"{k:>7}: {ms:6.1f} ms/frame")
    fps_real = 1 / np.mean(tiempos["total"])
    print(f"\nfps alcanzable: {fps_real:.1f} (el video es de {fps_video:.1f} fps)")
    print("Tiempo real: " + ("SI" if fps_real >= fps_video else "NO"))
    print("Guardado demo_causal.mp4")