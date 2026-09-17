import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import re
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
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

VENTANA = 32
PASO = 4    # cada cuántos frames vuelvo a clasificar
EPOCAS = 40

# --- video a analizar ---
ESCENARIO = "Home_02"
NUM_VIDEO = 31

IDX_CADERA_IZQ, IDX_CADERA_DER = 23, 24
IDX_HOMBRO_IZQ, IDX_HOMBRO_DER = 11, 12

MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pose_landmarker.task")
base_options = python.BaseOptions(model_asset_path=MODELO_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.IMAGE,
    num_poses=1, min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
landmarker = vision.PoseLandmarker.create_from_options(options)
yolo_model = YOLO("yolov8n.pt")


#mismas transformaciones que en normalizar
def interpolar_frames_faltantes(kp):
    n = kp.shape[0]
    valido = kp.reshape(n, -1).sum(axis=1) != 0
    if valido.sum() == 0:
        return kp
    idx_val = np.where(valido)[0]
    out = kp.copy()
    for i in range(n):
        if not valido[i]:
            antes = idx_val[idx_val < i]
            despues = idx_val[idx_val > i]
            if len(antes) and len(despues):
                a, d = antes[-1], despues[0]
                peso = (i - a) / (d - a)
                out[i] = kp[a] * (1 - peso) + kp[d] * peso
            elif len(antes):
                out[i] = kp[antes[-1]]
            elif len(despues):
                out[i] = kp[despues[0]]
    return out


def normalizar_esqueleto(kp):
    cadera = (kp[:, IDX_CADERA_IZQ, :2] + kp[:, IDX_CADERA_DER, :2]) / 2
    hombro = (kp[:, IDX_HOMBRO_IZQ, :2] + kp[:, IDX_HOMBRO_DER, :2]) / 2
    escala = np.linalg.norm(hombro - cadera, axis=1, keepdims=True)
    escala = np.where(escala < 1e-6, 1e-6, escala)
    out = (kp[:, :, :2] - cadera[:, None, :]) / escala[:, None, :]
    return out


def agregar_velocidades(kp_norm):
    vel = np.zeros_like(kp_norm)
    vel[1:] = kp_norm[1:] - kp_norm[:-1]
    return np.concatenate([kp_norm, vel], axis=2)


#entrenamiento
class DatasetLista(Dataset):
    def __init__(self, items):
        self.datos = items
        self.label_map = {"adl": 0, "fall": 1}

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        item = self.datos[idx]
        kp = item["keypoints"]
        n = kp.shape[0]
        if n >= VENTANA:
            inicio = np.random.randint(0, n - VENTANA + 1)
            clip = kp[inicio:inicio + VENTANA]
        else:
            clip = np.concatenate([kp, np.repeat(kp[-1:], VENTANA - n, axis=0)], axis=0)
        return torch.tensor(clip, dtype=torch.float32), self.label_map[item["label"]]


def entrenar_excluyendo(sujeto_excluido, A, device):
    items = list(np.load(f"{CARPETA}/todos.npy", allow_pickle=True))
    train = [d for d in items if d["grupo"] != sujeto_excluido]
    print(f"Entrenando con {len(train)} secuencias (excluyendo {sujeto_excluido})")

    loader = DataLoader(DatasetLista(train), batch_size=8, shuffle=True)
    modelo = STGCN(A, canales_entrada=4).to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    n_adl = sum(1 for d in train if d["label"] == "adl")
    n_fall = sum(1 for d in train if d["label"] == "fall")
    pesos = torch.tensor([1.0 / n_adl, 1.0 / n_fall], dtype=torch.float32)
    pesos = pesos / pesos.sum() * 2
    criterio = nn.CrossEntropyLoss(weight=pesos.to(device))

    for ep in range(EPOCAS):
        modelo.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterio(modelo(x), y).backward()
            optimizer.step()
        if (ep + 1) % 10 == 0:
            print(f"  época {ep+1}/{EPOCAS}")
    return modelo


#extracción del video
def buscar_video(escenario, num):
    for ruta in glob.glob(os.path.join(RAIZ_VIDEOS, escenario, "**", "*.avi"), recursive=True):
        nums = re.findall(r"\d+", os.path.basename(ruta))
        if nums and int(nums[-1]) == num:
            return ruta
    return None


def extraer_keypoints(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = yolo_model(frame, classes=[0], verbose=False)
        if len(res[0].boxes) == 0:
            frames.append(np.zeros((33, 4)))
            continue
        b = res[0].boxes
        x1, y1, x2, y2 = b.xyxy[b.conf.argmax().item()].cpu().numpy()
        h, w = frame.shape[:2]
        aw, ah = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - aw * .2)); y1 = max(0, int(y1 - ah * .2))
        x2 = min(w, int(x2 + aw * .2)); y2 = min(h, int(y2 + ah * .2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            frames.append(np.zeros((33, 4)))
            continue
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        r = landmarker.detect(mp_img)
        if r.pose_landmarks:
            frames.append(np.array([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks[0]]))
        else:
            frames.append(np.zeros((33, 4)))
    cap.release()
    return np.array(frames), fps


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    df = pd.read_csv(CSV_OMNIFALL)
    df["escenario"] = df["path"].str.split("/").str[0]
    df["num_video"] = df["path"].str.extract(r"video_(\d+)").astype(int)
    segs = df[(df["escenario"] == ESCENARIO) & (df["num_video"] == NUM_VIDEO)]

    if segs.empty:
        raise SystemExit(f"No hay anotaciones para {ESCENARIO}/video_{NUM_VIDEO}")

    sujeto = f"le2i_s{segs.iloc[0]['subject']}"
    print(f"Video {ESCENARIO}/video_{NUM_VIDEO}, sujeto {sujeto}")

    A = construir_matriz_adyacencia()
    modelo = entrenar_excluyendo(sujeto, A, device)

    ruta = buscar_video(ESCENARIO, NUM_VIDEO)
    print(f"Extrayendo keypoints de {ruta}...")
    kp, fps = extraer_keypoints(ruta)
    kp = agregar_velocidades(normalizar_esqueleto(interpolar_frames_faltantes(kp)))
    print(f"{len(kp)} frames a {fps:.1f} fps")

    #ventana deslizante
    modelo.eval()
    centros, probs = [], []
    with torch.no_grad():
        for ini in range(0, len(kp) - VENTANA + 1, PASO):
            clip = torch.tensor(kp[ini:ini + VENTANA], dtype=torch.float32).unsqueeze(0).to(device)
            p = torch.softmax(modelo(clip), dim=1)[0, 1].item()
            centros.append((ini + VENTANA / 2) / fps)
            probs.append(p)

    # --- gráfico ---
    plt.figure(figsize=(12, 4))
    for s in segs.itertuples():
        if s.label in (1, 2):
            color = "red" if s.label == 1 else "orange"
            plt.axvspan(s.start, s.end, alpha=.25, color=color)
    plt.plot(centros, probs, linewidth=1.5)
    plt.axhline(.5, linestyle="--", color="gray", linewidth=1)
    plt.xlabel("segundos")
    plt.ylabel("P(caída)")
    plt.title(f"{ESCENARIO}/video_{NUM_VIDEO} — rojo=fall, naranja=fallen")
    plt.ylim(-.05, 1.05)
    plt.tight_layout()
    plt.savefig("ventana_deslizante.png", dpi=120)
    print("Guardado ventana_deslizante.png")