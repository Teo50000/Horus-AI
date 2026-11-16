import numpy as np
import torch
from torch.utils.data import Dataset

# Unica implementacion del Dataset para el ST-GCN. Recibe la lista de items de
# todos.npy (o un subconjunto): dicts con keypoints (n,33,4)=x,y,vx,vy y label.

VENTANA = 32

# pares de puntos que se intercambian al espejar (izq <-> der en MediaPipe)
PARES_ESPEJO = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

LABEL_MAP = {"adl": 0, "fall": 1}


def espejar(clip):
    clip = clip.copy()
    clip[:, :, 0] = -clip[:, :, 0]  # invierto la x (ya esta centrada en la cadera)
    clip[:, :, 2] = -clip[:, :, 2]  # y tambien la velocidad en x, si no queda incoherente
    for i, j in PARES_ESPEJO:
        clip[:, [i, j]] = clip[:, [j, i]]
    return clip


def simular_camara(kp, ventana, fps_datos=25.0, fps_rango=(8.0, 15.0), perdida_max=0.2):
    """El clip que armaria la APP con esta secuencia, en vez de 32 frames
    limpios seguidos.

    En la app la pose llega a ~10 fps con jitter, los huecos de deteccion se
    tapan repitiendo la ultima pose, y el buffer se remuestrea a 25 fps
    (compuertas_caida.remuestrear_clip). El modelo entrenaba solo con video
    limpio a 25 fps: nunca veia las mesetas ni el movimiento interpolado que
    le llegan de verdad. Devuelve None si la secuencia es muy corta."""
    from compuertas_caida import duracion_ventana, remuestrear_clip
    n = kp.shape[0]
    t_nat = np.arange(n) / fps_datos
    dur = duracion_ventana(ventana + 1, 25.0)
    if t_nat[-1] < dur + 0.2:
        return None
    fps_cam = np.random.uniform(*fps_rango)
    t_fin = np.random.uniform(dur + 0.2, t_nat[-1])
    t_cam = np.arange(t_fin, t_fin - dur - 0.3, -1.0 / fps_cam)[::-1]
    t_cam = np.clip(t_cam + np.random.uniform(-0.2, 0.2, len(t_cam)) / fps_cam, 0, t_nat[-1])
    t_cam = np.sort(t_cam)
    t_cam[-1] = t_fin
    idx = np.clip(np.round(t_cam * fps_datos).astype(int), 0, n - 1)
    poses = kp[idx, :, :2].copy()
    perdidas = np.random.rand(len(idx)) < np.random.uniform(0.0, perdida_max)
    for j in range(1, len(poses)):
        if perdidas[j]:
            poses[j] = poses[j - 1]          # como la app: se repite la ultima pose
    return remuestrear_clip(t_cam, poses, t_fin, ventana, 25.0).astype(np.float32)


class DatasetLista(Dataset):
    def __init__(self, items, entrenamiento=False, ventana=VENTANA, simular_app=0.0):
        self.datos = items
        self.entrenamiento = entrenamiento  # solo aumento en train, nunca en val/test
        self.ventana = ventana
        # probabilidad de reemplazar la ventana limpia por `simular_camara`
        self.simular_app = simular_app

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        item = self.datos[idx]
        kp = item["keypoints"]
        label = LABEL_MAP[item["label"]]

        n = kp.shape[0]
        clip = None
        if self.entrenamiento and self.simular_app > 0 and np.random.rand() < self.simular_app:
            clip = simular_camara(kp, self.ventana)   # None si es muy corta
        if clip is None:
            if n >= self.ventana:
                if self.entrenamiento:
                    inicio = np.random.randint(0, n - self.ventana + 1)  # ventana aleatoria
                else:
                    inicio = (n - self.ventana) // 2  # centro fijo, para que val/test sean reproducibles
                clip = kp[inicio:inicio + self.ventana]
            else:
                relleno = np.repeat(kp[-1:], self.ventana - n, axis=0)
                clip = np.concatenate([kp, relleno], axis=0)

        if self.entrenamiento and np.random.rand() < 0.5:
            clip = espejar(clip)

        return torch.tensor(clip, dtype=torch.float32), label
