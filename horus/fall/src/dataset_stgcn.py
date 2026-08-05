import numpy as np
import torch
from torch.utils.data import Dataset

VENTANA = 32

# pares de puntos que se intercambian al espejar (izq <-> der en MediaPipe)
PARES_ESPEJO = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


class FallDatasetSTGCN(Dataset):
    def __init__(self, archivo_npy, entrenamiento=False):
        self.datos = np.load(archivo_npy, allow_pickle=True)
        self.label_map = {"adl": 0, "fall": 1}
        self.entrenamiento = entrenamiento  # solo aumento en train, nunca en val/test

    def __len__(self):
        return len(self.datos)

    def espejar(self, clip):
        clip = clip.copy()
        clip[:, :, 0] = -clip[:, :, 0]  # invierto la x (ya está centrada en la cadera)
        for i, j in PARES_ESPEJO:
            clip[:, [i, j]] = clip[:, [j, i]]  # intercambio los puntos izq/der
        return clip

    def __getitem__(self, idx):
        item = self.datos[idx]
        kp = item["keypoints"]  # (n_frames, 33, 2)
        label = self.label_map[item["label"]]

        n = kp.shape[0]
        if n >= VENTANA:
            if self.entrenamiento:
                inicio = np.random.randint(0, n - VENTANA + 1)  # ventana aleatoria
            else:
                inicio = (n - VENTANA) // 2  # centro fijo, para que val/test sean reproducibles
            clip = kp[inicio:inicio + VENTANA]
        else:
            faltan = VENTANA - n
            relleno = np.repeat(kp[-1:], faltan, axis=0)
            clip = np.concatenate([kp, relleno], axis=0)

        if self.entrenamiento and np.random.rand() < 0.5:
            clip = self.espejar(clip)

        return torch.tensor(clip, dtype=torch.float32), label