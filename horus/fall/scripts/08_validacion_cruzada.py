import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("../src")

from modelo_stgcn import STGCN
from grafo_mediapipe import construir_matriz_adyacencia
from collections import defaultdict

NOMBRES = {0: "walk", 1: "fall", 2: "fallen", 3: "sit_down", 4: "sitting",
           5: "lie_down", 6: "lying", 7: "stand_up", 8: "standing", 9: "other"}
CARPETA = "../data/processed"
VENTANA = 32
EPOCAS = 40

PARES_ESPEJO = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


class DatasetLista(Dataset):
#Igual que FallDatasetSTGCN pero recibe una lista en vez de un archivo.
    def __init__(self, items, entrenamiento=False):
        self.datos = items
        self.label_map = {"adl": 0, "fall": 1}
        self.entrenamiento = entrenamiento

    def __len__(self):
        return len(self.datos)

    def espejar(self, clip):
        clip = clip.copy()
        clip[:, :, 0] = -clip[:, :, 0]  # invierto la x
        clip[:, :, 2] = -clip[:, :, 2]  # y también la velocidad en x, si no queda incoherente
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


def cargar_todo():
    #es un solo archivo con todo el dataset.
    return list(np.load(f"{CARPETA}/todos.npy", allow_pickle=True))


def entrenar_fold(items_train, items_test, A, device, semilla):
    torch.manual_seed(semilla)  # misma inicialización en cada fold, para comparar peras con peras
    np.random.seed(semilla)

    train_loader = DataLoader(DatasetLista(items_train, entrenamiento=True), batch_size=8, shuffle=True)
    test_loader = DataLoader(DatasetLista(items_test), batch_size=8)

    modelo = STGCN(A, canales_entrada=4).to(device)    
    optimizer = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    # peso inverso a la frecuencia de cada clase en este fold,
    # para que las caídas (minoría) pesen más y el modelo no se vaya a todo adl por la diferencia de cantidad en el dataset
    n_adl = sum(1 for d in items_train if d["label"] == "adl")
    n_fall = sum(1 for d in items_train if d["label"] == "fall")
    pesos = torch.tensor([1.0 / max(n_adl, 1), 1.0 / max(n_fall, 1)], dtype=torch.float32)
    pesos = pesos / pesos.sum() * 2  # normalizo para que promedien 1
    criterio = nn.CrossEntropyLoss(weight=pesos.to(device))

    for _ in range(EPOCAS):
        modelo.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterio(modelo(x), y)
            loss.backward()
            optimizer.step()

    # evalúo el modelo final (sin elegir mejor época)
    modelo.eval()
    matriz = np.zeros((2, 2), dtype=int)
    predicciones = []  # en el mismo orden que items_test
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = modelo(x).argmax(1)
            predicciones.extend(pred.cpu().numpy().tolist())
            for real, predicho in zip(y.cpu().numpy(), pred.cpu().numpy()):
                matriz[real, predicho] += 1
    return matriz, predicciones


def bal_acc_de(matriz):
    rec_adl = matriz[0, 0] / max(matriz[0].sum(), 1)
    rec_fall = matriz[1, 1] / max(matriz[1].sum(), 1)
    return (rec_adl + rec_fall) / 2, rec_adl, rec_fall


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando:", device)

    items = cargar_todo()
    A = construir_matriz_adyacencia()
    grupos = sorted({d["grupo"] for d in items})
    print(f"{len(items)} secuencias, {len(grupos)} grupos: {grupos}\n")

    matriz_total = np.zeros((2, 2), dtype=int)
    por_subclase = defaultdict(lambda: [0, 0])  # clave -> [aciertos, total]

    for i, grupo in enumerate(grupos):
        items_test = [d for d in items if d["grupo"] == grupo]
        items_train = [d for d in items if d["grupo"] != grupo]

        matriz, preds = entrenar_fold(items_train, items_test, A, device, semilla=i)
        matriz_total += matriz

        # acumulo acierto por subclase de Omnifall (walk, fall, fallen, sit_down,etc)
        for item, p in zip(items_test, preds):
            sub = item["subclase"]
            nombre = NOMBRES.get(sub, str(sub)) if isinstance(sub, int) else sub
            clave = f"{nombre} ({item['label']})"
            correcto = (p == 1) == (item["label"] == "fall")
            por_subclase[clave][0] += int(correcto)
            por_subclase[clave][1] += 1

        bal, rec_adl, rec_fall = bal_acc_de(matriz)
        print(f"Fold '{grupo}': {len(items_test)} muestras - "
              f"bal_acc: {bal:.2%} - rec_adl: {rec_adl:.2%} - rec_fall: {rec_fall:.2%}")

    print("\n resultado global (todas las predicciones juntas)")
    bal, rec_adl, rec_fall = bal_acc_de(matriz_total)
    total = matriz_total.sum()
    print(f"Accuracy: {matriz_total.trace()/total:.2%} ({matriz_total.trace()}/{total})")
    print(f"Balanced accuracy: {bal:.2%}")
    print(f"Recall adl: {rec_adl:.2%} | Recall fall: {rec_fall:.2%}")
    print("\nMatriz de confusión acumulada:")
    print("             pred_adl   pred_fall")
    print(f"real_adl      {matriz_total[0,0]:>7}    {matriz_total[0,1]:>7}")
    print(f"real_fall     {matriz_total[1,0]:>7}    {matriz_total[1,1]:>7}")

    print("\n acieto por subclase")
    for clave in sorted(por_subclase, key=lambda k: -por_subclase[k][1]):
        ok, tot = por_subclase[clave]
        print(f"{clave:<24} {ok:>4}/{tot:<4} = {ok/tot:.1%}")                                                   