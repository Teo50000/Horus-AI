import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

VENTANA = 32
CARPETA = "../data/processed"

# --- Dataset ---
class FallDataset(Dataset):
    def __init__(self, archivo_npy):
        self.datos = np.load(archivo_npy, allow_pickle=True)
        self.label_map = {"adl": 0, "fall": 1}

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        item = self.datos[idx]
        kp = item["keypoints"]  # (n_frames, 33, 2)
        label = self.label_map[item["label"]]

        n = kp.shape[0]
        if n >= VENTANA:
            # recorte del medio (ahí suele estar la acción principal)
            inicio = (n - VENTANA) // 2
            clip = kp[inicio:inicio + VENTANA]
        else:
            # rellenar repitiendo el último frame
            faltan = VENTANA - n
            relleno = np.repeat(kp[-1:], faltan, axis=0)
            clip = np.concatenate([kp, relleno], axis=0)

        clip = clip.reshape(VENTANA, -1)  # aplanar: (32, 33*2) = (32, 66)
        return torch.tensor(clip, dtype=torch.float32), label


# --- Modelo: LSTM chico ---
class ClasificadorLSTM(nn.Module):
    def __init__(self, entrada=66, oculto=64, clases=2):
        super().__init__()
        self.lstm = nn.LSTM(entrada, oculto, batch_first=True)
        self.fc = nn.Linear(oculto, clases)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n: último estado oculto
        out = self.fc(h_n[-1])
        return out


# --- Entrenamiento ---
def entrenar():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    train_ds = FallDataset(f"{CARPETA}/train.npy")
    val_ds = FallDataset(f"{CARPETA}/val.npy")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    modelo = ClasificadorLSTM().to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = nn.CrossEntropyLoss()

    epocas = 30
    for epoca in range(epocas):
        modelo.train()
        loss_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = modelo(x)
            loss = criterio(pred, y)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        # Validación
        modelo.eval()
        correctos, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = modelo(x)
                correctos += (pred.argmax(1) == y).sum().item()
                total += y.size(0)

        acc_val = correctos / total if total > 0 else 0
        print(f"Época {epoca+1}/{epocas} - loss: {loss_total:.4f} - val_acc: {acc_val:.2%}")

    torch.save(modelo.state_dict(), "../checkpoints/modelo_baseline.pt")
    print("Modelo guardado en ../checkpoints/modelo_baseline.pt")


if __name__ == "__main__":
    import os
    os.makedirs("../checkpoints", exist_ok=True)
    entrenar()