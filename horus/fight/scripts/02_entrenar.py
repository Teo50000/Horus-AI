import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("../src")
from modelo_fight import crear_modelo
from dataset_fight import RWFDataset

CARPETA_TRAIN = "../data/processed/frames/train"
CARPETA_VAL = "../data/processed/frames/val"
CHECKPOINT_SALIDA = "../checkpoints/modelo_fight.pt"
EPOCAS = 12
BATCH = 8
LR = 1e-4


def evaluar(modelo, loader, device):
    modelo.eval()
    matriz = np.zeros((2, 2), dtype=int)  # filas=real, columnas=predicho
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = modelo(x).argmax(1)
            for real, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                matriz[real, p] += 1
    return matriz


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    train_loader = DataLoader(RWFDataset(CARPETA_TRAIN, entrenamiento=True),
                               batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader = DataLoader(RWFDataset(CARPETA_VAL, entrenamiento=False),
                             batch_size=BATCH, num_workers=2)
    print(f"train: {len(train_loader.dataset)} clips, val: {len(val_loader.dataset)} clips")

    modelo = crear_modelo().to(device)

    # si ya habia un entrenamiento anterior (se corto a mitad de camino, por
    # ejemplo) retomo desde ese checkpoint en vez de arrancar de nuevo desde
    # los pesos de Kinetics -- no tiene sentido tirar el progreso ya hecho.
    mejor_bal_acc = 0.0
    os.makedirs(os.path.dirname(CHECKPOINT_SALIDA), exist_ok=True)
    if os.path.exists(CHECKPOINT_SALIDA):
        print(f"Retomando desde checkpoint existente: {CHECKPOINT_SALIDA}")
        modelo.load_state_dict(torch.load(CHECKPOINT_SALIDA, map_location=device))

    optimizer = torch.optim.Adam(modelo.parameters(), lr=LR)
    criterio = nn.CrossEntropyLoss()

    if os.path.exists(CHECKPOINT_SALIDA):
        matriz = evaluar(modelo, val_loader, device)
        rec_nofight = matriz[0, 0] / max(matriz[0].sum(), 1)
        rec_fight = matriz[1, 1] / max(matriz[1].sum(), 1)
        mejor_bal_acc = (rec_nofight + rec_fight) / 2
        print(f"bal_acc del checkpoint cargado: {mejor_bal_acc:.2%} "
              f"(solo piso el archivo si supero esto)")

    for epoca in range(EPOCAS):
        modelo.train()
        loss_total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterio(modelo(x), y)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        matriz = evaluar(modelo, val_loader, device)
        rec_nofight = matriz[0, 0] / max(matriz[0].sum(), 1)
        rec_fight = matriz[1, 1] / max(matriz[1].sum(), 1)
        bal_acc = (rec_nofight + rec_fight) / 2

        print(f"Epoca {epoca+1}/{EPOCAS} - loss: {loss_total:.4f} - "
              f"bal_acc: {bal_acc:.2%} - rec_nofight: {rec_nofight:.2%} - rec_fight: {rec_fight:.2%}")

        # guardo el mejor checkpoint por balanced accuracy en val, no el ultimo --
        # con pocas epocas y dataset chico el ultimo no siempre es el mejor
        if bal_acc > mejor_bal_acc:
            mejor_bal_acc = bal_acc
            torch.save(modelo.state_dict(), CHECKPOINT_SALIDA)

    print(f"\nMejor bal_acc en val: {mejor_bal_acc:.2%} - checkpoint en {CHECKPOINT_SALIDA}")
