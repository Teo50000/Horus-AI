import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

VENTANA = 32
CARPETA = "../data/processed"


# --- Mismo Dataset que en 04_entrenar_baseline.py ---
class FallDataset(Dataset):
    def __init__(self, archivo_npy):
        self.datos = np.load(archivo_npy, allow_pickle=True)
        self.label_map = {"adl": 0, "fall": 1}

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        item = self.datos[idx]
        kp = item["keypoints"]
        label = self.label_map[item["label"]]

        n = kp.shape[0]
        if n >= VENTANA:
            inicio = (n - VENTANA) // 2
            clip = kp[inicio:inicio + VENTANA]
        else:
            faltan = VENTANA - n
            relleno = np.repeat(kp[-1:], faltan, axis=0)
            clip = np.concatenate([kp, relleno], axis=0)

        clip = clip.reshape(VENTANA, -1)
        return torch.tensor(clip, dtype=torch.float32), label


# --- Mismo modelo que en 04_entrenar_baseline.py ---
class ClasificadorLSTM(nn.Module):
    def __init__(self, entrada=66, oculto=64, clases=2):
        super().__init__()
        self.lstm = nn.LSTM(entrada, oculto, batch_first=True)
        self.fc = nn.Linear(oculto, clases)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


def evaluar():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    test_ds = FallDataset(f"{CARPETA}/test.npy")
    test_loader = DataLoader(test_ds, batch_size=8)

    modelo = ClasificadorLSTM().to(device)
    modelo.load_state_dict(torch.load("../checkpoints/modelo_baseline.pt", map_location=device))
    modelo.eval()

    correctos, total = 0, 0
    matriz_confusion = np.zeros((2, 2), dtype=int)  # filas=real, columnas=predicho

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = modelo(x)
            pred_clase = pred.argmax(1)

            correctos += (pred_clase == y).sum().item()
            total += y.size(0)

            for real, predicho in zip(y.cpu().numpy(), pred_clase.cpu().numpy()):
                matriz_confusion[real, predicho] += 1

    print(f"\nAccuracy en test: {correctos/total:.2%} ({correctos}/{total})")
    print("\nMatriz de confusión (filas=real, columnas=predicho):")
    print("             pred_adl   pred_fall")
    print(f"real_adl      {matriz_confusion[0,0]:>7}    {matriz_confusion[0,1]:>7}")
    print(f"real_fall     {matriz_confusion[1,0]:>7}    {matriz_confusion[1,1]:>7}")


if __name__ == "__main__":
    evaluar()