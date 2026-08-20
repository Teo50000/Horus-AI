import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("../src")

from modelo_stgcn import STGCN
from grafo_mediapipe import construir_matriz_adyacencia

# Entrena el modelo "de despliegue" (el que carga 12_webcam.py) usando TODO
# todos.npy, sin dejar ningún grupo afuera. La evaluación con grupo afuera ya
# la hace 08_validacion_cruzada.py / 13_cv_3clases.py; este script es sólo
# para producir el checkpoint que se usa en inferencia real.

CARPETA = "../data/processed"
CHECKPOINT_SALIDA = "../checkpoints/modelo_demo_todo.pt"
VENTANA = 32
EPOCAS = 40

PARES_ESPEJO = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


class DatasetLista(Dataset):
    def __init__(self, items, entrenamiento=False):
        self.datos = items
        self.label_map = {"adl": 0, "fall": 1}
        self.entrenamiento = entrenamiento

    def __len__(self):
        return len(self.datos)

    def espejar(self, clip):
        clip = clip.copy()
        clip[:, :, 0] = -clip[:, :, 0]
        clip[:, :, 2] = -clip[:, :, 2]
        for i, j in PARES_ESPEJO:
            clip[:, [i, j]] = clip[:, [j, i]]
        return clip

    def __getitem__(self, idx):
        item = self.datos[idx]
        kp = item["keypoints"]
        label = self.label_map[item["label"]]

        n = kp.shape[0]
        if n >= VENTANA:
            inicio = np.random.randint(0, n - VENTANA + 1) if self.entrenamiento else (n - VENTANA) // 2
            clip = kp[inicio:inicio + VENTANA]
        else:
            relleno = np.repeat(kp[-1:], VENTANA - n, axis=0)
            clip = np.concatenate([kp, relleno], axis=0)

        if self.entrenamiento and np.random.rand() < 0.5:
            clip = self.espejar(clip)

        return torch.tensor(clip, dtype=torch.float32), label


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    items = list(np.load(f"{CARPETA}/todos.npy", allow_pickle=True))
    print(f"Entrenando con {len(items)} secuencias (dataset completo, sin grupo afuera)")

    A = construir_matriz_adyacencia()
    loader = DataLoader(DatasetLista(items, entrenamiento=True), batch_size=8, shuffle=True)

    modelo = STGCN(A, canales_entrada=4).to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    n_adl = sum(1 for d in items if d["label"] == "adl")
    n_fall = sum(1 for d in items if d["label"] == "fall")
    pesos = torch.tensor([1.0 / max(n_adl, 1), 1.0 / max(n_fall, 1)], dtype=torch.float32)
    pesos = pesos / pesos.sum() * 2
    criterio = nn.CrossEntropyLoss(weight=pesos.to(device))

    for epoca in range(EPOCAS):
        modelo.train()
        loss_total = 0
        matriz = np.zeros((2, 2), dtype=int)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = modelo(x)
            loss = criterio(pred, y)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            for real, predicho in zip(y.cpu().numpy(), pred.argmax(1).cpu().numpy()):
                matriz[real, predicho] += 1

        rec_adl = matriz[0, 0] / max(matriz[0].sum(), 1)
        rec_fall = matriz[1, 1] / max(matriz[1].sum(), 1)
        print(f"Época {epoca+1}/{EPOCAS} - loss: {loss_total:.4f} - "
              f"rec_adl(train): {rec_adl:.2%} - rec_fall(train): {rec_fall:.2%}")

    import os
    os.makedirs("../checkpoints", exist_ok=True)
    torch.save(modelo.state_dict(), CHECKPOINT_SALIDA)
    print(f"\nModelo guardado en {CHECKPOINT_SALIDA}")
