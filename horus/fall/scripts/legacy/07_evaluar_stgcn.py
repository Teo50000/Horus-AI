import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("../src")

from dataset_stgcn import FallDatasetSTGCN
from modelo_stgcn import STGCN
from grafo_mediapipe import construir_matriz_adyacencia

CARPETA = "../data/processed"

def evaluar():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A = construir_matriz_adyacencia()
    test_ds = FallDatasetSTGCN(f"{CARPETA}/test.npy")
    test_loader = DataLoader(test_ds, batch_size=8)

    modelo = STGCN(A).to(device)
    modelo.load_state_dict(torch.load("../checkpoints/modelo_stgcn.pt", map_location=device))
    modelo.eval()

    correctos, total = 0, 0
    matriz_confusion = np.zeros((2, 2), dtype=int)

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