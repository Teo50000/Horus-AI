"""Entrena el ST-GCN de caidas. Reemplaza a 14_entrenar_final.py agregando
semilla fija, registro por epoca y checkpoints con metadatos. La receta de
entrenamiento es la misma (Adam 1e-3, batch 8, 40 epocas, CE ponderada), asi
que el modelo que sale es equivalente -- lo que cambia es que ahora es
reproducible y queda documentado.

Dos modos, que responden a dos preguntas distintas:

  --dejar-afuera le2i_s0     Entrena sin ese grupo y lo mide cada epoca. Sirve
                             para DECIDIR la receta (cuantas epocas, que
                             hiperparametros) sin contaminar el numero que se
                             reporta. Guarda la mejor epoca y la ultima.

  (sin argumentos)           Entrena con TODO todos.npy y guarda la ultima epoca
                             en modelo_demo_todo.pt -- el checkpoint que carga
                             12_webcam.py. No hay grupo afuera, asi que no hay
                             senal para elegir "la mejor": su garantia es que la
                             receta se valido aparte, con el modo de arriba.

La evaluacion honesta a nivel evento la hace 17_evaluar_evento.py --loso.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.append("../src")
from entrenamiento import RECETA, entrenar, evaluar, metricas_de  # noqa: E402
from dataset_stgcn import DatasetLista  # noqa: E402
import checkpoint  # noqa: E402

TODOS = "../data/processed/todos.npy"
CHECKPOINTS = "../checkpoints"
RESULTADOS = "../resultados"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dejar-afuera", metavar="GRUPO",
                    help="entrena sin ese grupo y lo usa para medir cada epoca")
    ap.add_argument("--epocas", type=int, default=RECETA["epocas"])
    ap.add_argument("--batch", type=int, default=RECETA["batch"])
    ap.add_argument("--lr", type=float, default=RECETA["lr"])
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", help="ruta del checkpoint (default segun el modo)")
    ap.add_argument("--tag", help="nombre para el csv de la corrida")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    todos = list(np.load(TODOS, allow_pickle=True))
    grupos = sorted({d["grupo"] for d in todos})
    receta = dict(epocas=args.epocas, batch=args.batch, lr=args.lr)

    if args.dejar_afuera:
        if args.dejar_afuera not in grupos:
            raise SystemExit(f"'{args.dejar_afuera}' no es un grupo. Hay: {grupos}")
        items_train = [d for d in todos if d["grupo"] != args.dejar_afuera]
        items_val = [d for d in todos if d["grupo"] == args.dejar_afuera]
        tag = args.tag or f"sin_{args.dejar_afuera}"
        salida = args.salida or os.path.join(CHECKPOINTS, f"validacion_{tag}.pt")
    else:
        items_train, items_val = todos, None
        tag = args.tag or "despliegue"
        salida = args.salida or os.path.join(CHECKPOINTS, "modelo_demo_todo.pt")

    print(f"Usando: {device} | semilla: {args.semilla} | receta: {receta}")
    print(f"train: {len(items_train)} secuencias"
          + (f" | val ({args.dejar_afuera}): {len(items_val)}" if items_val else " | sin grupo afuera"))

    ruta_csv = os.path.join(RESULTADOS, f"entrenamiento_{tag}.csv")
    modelo, historia, mejor, mejor_pesos = entrenar(
        items_train, device, args.semilla, items_val=items_val,
        receta=receta, ruta_csv=ruta_csv, verbose=bool(items_val),
    )

    metricas = dict(historia[-1])
    if mejor:
        print(f"\nmejor epoca: {mejor['epoca']} (val_bal_acc {mejor['val_bal_acc']:.2%}) | "
              f"ultima epoca: {historia[-1].get('val_bal_acc', float('nan')):.2%}")
        metricas["mejor_epoca"] = mejor["epoca"]
        metricas["mejor_val_bal_acc"] = mejor["val_bal_acc"]

        ruta_mejor = salida.replace(".pt", "_mejor.pt")
        estado = modelo.state_dict()
        modelo.load_state_dict(mejor_pesos)
        checkpoint.guardar(ruta_mejor, modelo, config=dict(receta, semilla=args.semilla,
                           dejado_afuera=args.dejar_afuera, epoca=mejor["epoca"],
                           n_train=len(items_train), seleccion="mejor val_bal_acc"),
                           metricas=mejor)
        modelo.load_state_dict(estado)
        print(f"  mejor epoca  -> {ruta_mejor}")

    checkpoint.guardar(salida, modelo, config=dict(receta, semilla=args.semilla,
                       dejado_afuera=args.dejar_afuera, epoca=args.epocas,
                       n_train=len(items_train), seleccion="ultima epoca"),
                       metricas=metricas)
    print(f"  ultima epoca -> {salida}")
    print(f"  curva por epoca -> {ruta_csv}")
