import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_stgcn import DatasetLista
from grafo_mediapipe import construir_matriz_adyacencia
from modelo_stgcn import STGCN

# UNICA implementacion del entrenamiento del ST-GCN. La usan 18_entrenar.py (el
# checkpoint de despliegue y los folds de validacion) y 17_evaluar_evento.py
# (los modelos LOSO del harness), asi no vuelven a divergir en tres copias.
#
# La receta es la de 14_entrenar_final.py a proposito: Adam 1e-3, batch 8,
# 40 epocas, entropia cruzada ponderada por frecuencia de clase. Lo que se
# agrega es semilla, registro por epoca y seleccion de la mejor epoca cuando
# hay un grupo dejado afuera -- nada que cambie el modelo resultante.

RECETA = dict(epocas=40, batch=8, lr=1e-3, ventana=32, canales=4)


def fijar_semilla(semilla):
    """Todo el azar que afecta al entrenamiento: pesos iniciales, orden de los
    batches, ventana aleatoria y espejado del aumento de datos. Sin esto, dos
    corridas del mismo codigo dan modelos distintos y no se puede saber si un
    cambio mejoro o fue suerte."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _init_worker(worker_id, semilla):
    # cada proceso del DataLoader arranca con su propia semilla derivada; si no,
    # todos sortean la misma ventana aleatoria
    s = semilla + worker_id
    np.random.seed(s)
    random.seed(s)


def pesos_clase(items, device):
    """Las caidas son minoria (317 de 1088). Sin ponderar, al modelo le conviene
    decir 'adl' siempre."""
    n_adl = sum(1 for d in items if d["label"] == "adl")
    n_fall = sum(1 for d in items if d["label"] == "fall")
    p = torch.tensor([1.0 / max(n_adl, 1), 1.0 / max(n_fall, 1)], dtype=torch.float32)
    return (p / p.sum() * 2).to(device)


def evaluar(modelo, loader, device):
    """Matriz de confusion 2x2 sobre la ventana central de cada clip."""
    modelo.eval()
    matriz = np.zeros((2, 2), dtype=int)   # filas = real, columnas = predicho
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = modelo(x).argmax(1)
            for real, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                matriz[real, p] += 1
    return matriz


def metricas_de(matriz):
    rec_adl = matriz[0, 0] / max(matriz[0].sum(), 1)
    rec_fall = matriz[1, 1] / max(matriz[1].sum(), 1)
    return dict(bal_acc=(rec_adl + rec_fall) / 2, rec_adl=rec_adl, rec_fall=rec_fall)


def entrenar(items_train, device, semilla, items_val=None, receta=None,
             ruta_csv=None, verbose=False):
    """Entrena y devuelve (modelo, historia, mejor).

    items_val=None  -> no hay senal externa, devuelve la ULTIMA epoca (es el caso
                       del checkpoint de despliegue: no se sacrifica ningun grupo).
    items_val dado  -> mide cada epoca sobre ese grupo y devuelve tambien la MEJOR
                       por balanced accuracy, que es como se decide cuantas epocas
                       conviene entrenar.
    """
    r = dict(RECETA, **(receta or {}))
    fijar_semilla(semilla)

    generador = torch.Generator()
    generador.manual_seed(semilla)
    train_loader = DataLoader(
        DatasetLista(items_train, entrenamiento=True, ventana=r["ventana"]),
        batch_size=r["batch"], shuffle=True, generator=generador,
        worker_init_fn=lambda wid: _init_worker(wid, semilla),
    )
    val_loader = None
    if items_val:
        val_loader = DataLoader(DatasetLista(items_val, ventana=r["ventana"]), batch_size=r["batch"])

    A = construir_matriz_adyacencia()
    modelo = STGCN(A, canales_entrada=r["canales"]).to(device)
    opt = torch.optim.Adam(modelo.parameters(), lr=r["lr"])
    criterio = nn.CrossEntropyLoss(weight=pesos_clase(items_train, device))

    historia = []
    mejor = None
    mejor_pesos = None

    for epoca in range(1, r["epocas"] + 1):
        modelo.train()
        loss_total = 0.0
        matriz_train = np.zeros((2, 2), dtype=int)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            salida = modelo(x)
            loss = criterio(salida, y)
            loss.backward()
            opt.step()
            loss_total += loss.item()
            for real, p in zip(y.cpu().numpy(), salida.argmax(1).detach().cpu().numpy()):
                matriz_train[real, p] += 1

        fila = dict(epoca=epoca, loss=round(loss_total, 4))
        fila.update({f"train_{k}": round(v, 4) for k, v in metricas_de(matriz_train).items()})

        if val_loader is not None:
            m = metricas_de(evaluar(modelo, val_loader, device))
            fila.update({f"val_{k}": round(v, 4) for k, v in m.items()})
            if mejor is None or m["bal_acc"] > mejor["val_bal_acc"]:
                mejor = dict(epoca=epoca, val_bal_acc=m["bal_acc"],
                             val_rec_adl=m["rec_adl"], val_rec_fall=m["rec_fall"])
                mejor_pesos = {k: v.detach().clone() for k, v in modelo.state_dict().items()}

        historia.append(fila)
        if verbose:
            print("  " + "  ".join(f"{k}={v}" for k, v in fila.items()), flush=True)

    if ruta_csv:
        os.makedirs(os.path.dirname(os.path.abspath(ruta_csv)), exist_ok=True)
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(historia[0].keys()))
            w.writeheader()
            w.writerows(historia)

    return modelo, historia, mejor, mejor_pesos
