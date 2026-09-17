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


class DatasetLista(Dataset):
    def __init__(self, items, entrenamiento=False, ventana=VENTANA):
        self.datos = items
        self.entrenamiento = entrenamiento  # solo aumento en train, nunca en val/test
        self.ventana = ventana

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        item = self.datos[idx]
        kp = item["keypoints"]
        label = LABEL_MAP[item["label"]]

        n = kp.shape[0]
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
