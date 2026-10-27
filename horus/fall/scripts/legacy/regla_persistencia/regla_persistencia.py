#lo que hago con este script es no disparar la alarma en cuanto la probabilidad pasa 0.5, sino exigir que se mantenga arriba durante N segundos seguidos.
#todo esto sacado gracias al grafico que consegui con las ventanas
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2, re, glob, os
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
CACHE = "../data/processed/probs_le2i_s0.npz"

SUJETO = 0            # sujeto a evaluar (queda fuera del entrenamiento)
VENTANA, PASO, EPOCAS = 32, 4, 40
SEG_DESCARTE = 1.5    # ignoro el arranque, donde las velocidades son 0

IDX_CI, IDX_CD, IDX_HI, IDX_HD = 23, 24, 11, 12

MODELO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pose_landmarker.task")


def interpolar(kp):
    n = kp.shape[0]
    valido = kp.reshape(n, -1).sum(axis=1) != 0
    if valido.sum() == 0:
        return kp
    idx = np.where(valido)[0]
    out = kp.copy()
    for i in range(n):
        if not valido[i]:
            a_, d_ = idx[idx < i], idx[idx > i]
            if len(a_) and len(d_):
                a, d = a_[-1], d_[0]
                w = (i - a) / (d - a)
                out[i] = kp[a] * (1 - w) + kp[d] * w
            elif len(a_):
                out[i] = kp[a_[-1]]
            elif len(d_):
                out[i] = kp[d_[0]]
    return out


def normalizar(kp):
    cad = (kp[:, IDX_CI, :2] + kp[:, IDX_CD, :2]) / 2
    hom = (kp[:, IDX_HI, :2] + kp[:, IDX_HD, :2]) / 2
    esc = np.linalg.norm(hom - cad, axis=1, keepdims=True)
    esc = np.where(esc < 1e-6, 1e-6, esc)
    return (kp[:, :, :2] - cad[:, None, :]) / esc[:, None, :]


def con_velocidades(k):
    v = np.zeros_like(k)
    v[1:] = k[1:] - k[:-1]
    return np.concatenate([k, v], axis=2)


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


def entrenar(sujeto_fuera, A, device):
    items = list(np.load(f"{CARPETA}/todos.npy", allow_pickle=True))
    tr = [d for d in items if d["grupo"] != sujeto_fuera]
    print(f"Entrenando con {len(tr)} secuencias (sin {sujeto_fuera})")
    loader = DataLoader(DatasetLista(tr), batch_size=8, shuffle=True)
    modelo = STGCN(A, canales_entrada=4).to(device)
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
    return modelo


def buscar_video(esc, num):
    for r in glob.glob(os.path.join(RAIZ_VIDEOS, esc, "**", "*.avi"), recursive=True):
        n = re.findall(r"\d+", os.path.basename(r))
        if n and int(n[-1]) == num:
            return r
    return None


def probs_de_video(ruta, modelo, device, landmarker, yolo):
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        res = yolo(f, classes=[0], verbose=False)
        if len(res[0].boxes) == 0:
            frames.append(np.zeros((33, 4))); continue
        b = res[0].boxes
        x1, y1, x2, y2 = b.xyxy[b.conf.argmax().item()].cpu().numpy()
        h, w = f.shape[:2]
        aw, ah = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - aw * .2)); y1 = max(0, int(y1 - ah * .2))
        x2 = min(w, int(x2 + aw * .2)); y2 = min(h, int(y2 + ah * .2))
        c = f[y1:y2, x1:x2]
        if c.size == 0:
            frames.append(np.zeros((33, 4))); continue
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
        r = landmarker.detect(img)
        frames.append(np.array([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks[0]])
                      if r.pose_landmarks else np.zeros((33, 4)))
    cap.release()

    kp = con_velocidades(normalizar(interpolar(np.array(frames))))
    modelo.eval()
    t, pr = [], []
    with torch.no_grad():
        for i in range(0, len(kp) - VENTANA + 1, PASO):
            c = torch.tensor(kp[i:i + VENTANA], dtype=torch.float32).unsqueeze(0).to(device)
            pr.append(torch.softmax(modelo(c), 1)[0, 1].item())
            t.append((i + VENTANA / 2) / fps)
    return np.array(t), np.array(pr), len(frames) / fps


def episodios_alarma(t, p, umbral, duracion):
    #Devuelve los instantes donde una racha supera 'duracion' segundos por encima del umbral.
    alarmas, inicio = [], None
    for i in range(len(t)):
        if p[i] >= umbral:
            if inicio is None:
                inicio = t[i]
            elif t[i] - inicio >= duracion:
                alarmas.append(t[i])
                inicio = None  # reinicio: cuento el episodio como uno solo
        else:
            inicio = None
    return alarmas


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(CSV_OMNIFALL)
    df["escenario"] = df["path"].str.split("/").str[0]
    df["num"] = df["path"].str.extract(r"video_(\d+)").astype(int)
    mios = df[df["subject"] == SUJETO]
    videos = mios[["escenario", "num"]].drop_duplicates().values.tolist()
    print(f"Videos del sujeto {SUJETO}: {len(videos)}")

    if os.path.exists(CACHE):
        print("Usando cache de probabilidades")
        z = np.load(CACHE, allow_pickle=True)
        cache = z["cache"].item()
    else:
        A = construir_matriz_adyacencia()
        modelo = entrenar(f"le2i_s{SUJETO}", A, device)

        bo = python.BaseOptions(model_asset_path=MODELO_PATH)
        op = vision.PoseLandmarkerOptions(base_options=bo, running_mode=vision.RunningMode.IMAGE,
                                          num_poses=1, min_pose_detection_confidence=.3,
                                          min_pose_presence_confidence=.3)
        landmarker = vision.PoseLandmarker.create_from_options(op)
        yolo = YOLO("yolov8n.pt")

        cache = {}
        for k, (esc, num) in enumerate(videos, 1):
            ruta = buscar_video(esc, num)
            if ruta is None:
                continue
            print(f"[{k}/{len(videos)}] {esc}/video_{num}")
            t, p, dur = probs_de_video(ruta, modelo, device, landmarker, yolo)
            cache[f"{esc}/video_{num}"] = {"t": t, "p": p, "dur": dur}
        np.savez(CACHE, cache=cache)
        print(f"Cache guardado en {CACHE}")

    #barrido de parámetros
    print(f"\n{'umbral':>7} {'dur(s)':>7} {'detect':>8} {'latencia':>9} {'falsas/min':>11}")
    for umbral in [0.5, 0.7, 0.9]:
        for duracion in [0.0, 0.5, 1.0, 2.0]:
            detectadas = total_eventos = 0
            latencias, falsas, minutos_normales = [], 0, 0.0

            for clave, d in cache.items():
                esc, vid = clave.split("/")
                num = int(vid.replace("video_", ""))
                segs = df[(df["escenario"] == esc) & (df["num"] == num)]
                caidas = segs[segs["label"].isin([1, 2])]
                t_ini = caidas["start"].min() if len(caidas) else None

                mask = d["t"] >= SEG_DESCARTE
                alarmas = episodios_alarma(d["t"][mask], d["p"][mask], umbral, duracion)

                if t_ini is not None:
                    total_eventos += 1
                    validas = [a for a in alarmas if a >= t_ini]
                    if validas:
                        detectadas += 1
                        latencias.append(validas[0] - t_ini)
                    falsas += len([a for a in alarmas if a < t_ini])
                    minutos_normales += t_ini / 60
                else:
                    falsas += len(alarmas)
                    minutos_normales += d["dur"] / 60

            lat = f"{np.mean(latencias):.2f}s" if latencias else "-"
            fpm = falsas / minutos_normales if minutos_normales else 0
            print(f"{umbral:>7.1f} {duracion:>7.1f} {detectadas:>4}/{total_eventos:<3} "
                  f"{lat:>9} {fpm:>11.2f}")