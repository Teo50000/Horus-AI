import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import numpy as np
sys.path.append("../src")

from dataset_stgcn import FallDatasetSTGCN
from modelo_stgcn import STGCN
from grafo_mediapipe import construir_matriz_adyacencia

CARPETA = "../data/processed"

def entrenar():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    A = construir_matriz_adyacencia()

    train_ds = FallDatasetSTGCN(f"{CARPETA}/train.npy", entrenamiento=True)
    val_ds = FallDatasetSTGCN(f"{CARPETA}/val.npy")  # sin aumento

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    modelo = STGCN(A).to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = nn.CrossEntropyLoss()

    epocas = 40
    mejor_val = 0.0  # ahora guardo por balanced accuracy, no por accuracy pelada

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

        # --- validación: armo la matriz de confusión para ver qué predice en cada clase ---
        modelo.eval()
        matriz_val = np.zeros((2, 2), dtype=int)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = modelo(x).argmax(1)
                for real, predicho in zip(y.cpu().numpy(), pred.cpu().numpy()):
                    matriz_val[real, predicho] += 1

        total = matriz_val.sum()
        acc_val = matriz_val.trace() / total if total > 0 else 0

        # recall por clase: qué proporción de cada clase real detecta bien
        rec_adl = matriz_val[0, 0] / max(matriz_val[0].sum(), 1)
        rec_fall = matriz_val[1, 1] / max(matriz_val[1].sum(), 1)
        bal_acc = (rec_adl + rec_fall) / 2  # métrica que no se deja engañar por el desbalance

        # guardo solo si mejoró la balanced accuracy
        if bal_acc > mejor_val:
            mejor_val = bal_acc
            torch.save(modelo.state_dict(), "../checkpoints/modelo_stgcn.pt")
            print(f"  ↑ nuevo mejor modelo guardado (bal_acc: {bal_acc:.2%})")

        print(f"Época {epoca+1}/{epocas} - loss: {loss_total:.4f} - "
              f"acc: {acc_val:.2%} - bal_acc: {bal_acc:.2%} - "
              f"rec_adl: {rec_adl:.2%} - rec_fall: {rec_fall:.2%}")

    print(f"Mejor bal_acc: {mejor_val:.2%} - modelo guardado en ../checkpoints/modelo_stgcn.pt")
    
if __name__ == "__main__":
    import os
    os.makedirs("../checkpoints", exist_ok=True)
    entrenar()