import torch
import torch.nn as nn

import checkpoint
from grafo_mediapipe import construir_matriz_adyacencia
from modelo_stgcn import STGCN

# Entrenar la misma receta con distinta semilla mueve el recall de eventos
# varios puntos: una sola corrida es una tirada, no una medicion. En vez de
# buscar la semilla con suerte, promedio varias -- los errores independientes
# de cada una se cancelan y el resultado deja de depender del azar.
#
# Es casi gratis en este pipeline: el ST-GCN son 115k parametros y ~1.5 ms
# amortizados por frame, contra 25 ms de YOLO y 38 ms de MediaPipe. Cuatro
# modelos siguen siendo ruido frente a la percepcion.


class Ensamble(nn.Module):
    """Promedia las PROBABILIDADES de varios modelos y devuelve log(promedio).

    El logaritmo no es decorativo: el detector hace softmax() sobre la salida,
    y softmax(log(p)) == p cuando p ya suma 1. Asi el promedio llega intacto
    y DetectorCaidas no necesita saber que adentro hay mas de un modelo.
    """

    def __init__(self, modelos):
        super().__init__()
        if not modelos:
            raise ValueError("un ensamble necesita al menos un modelo")
        self.modelos = nn.ModuleList(modelos)

    def forward(self, x):
        p = torch.stack([torch.softmax(m(x), 1) for m in self.modelos]).mean(0)
        return torch.log(p.clamp_min(1e-12))


def cargar_checkpoint(ruta, device, canales_entrada=4):
    """Un .pt puede traer un modelo o varios (`state_dicts`, ver
    checkpoint.guardar). Devuelve (modelo, meta) listo para eval: un STGCN o
    un Ensamble, que se usan igual. Es lo que carga la app y lo que mide el
    harness, asi que lo que se evalua es lo que corre."""
    estado, meta = checkpoint.cargar(ruta, device=device)
    A = construir_matriz_adyacencia()
    modelos = []
    for sd in meta.get("state_dicts") or [estado]:
        m = STGCN(A, canales_entrada=canales_entrada).to(device)
        m.load_state_dict(sd)
        modelos.append(m.eval())
    if len(modelos) == 1:
        return modelos[0], meta
    return Ensamble(modelos).to(device).eval(), meta


def cargar_ensamble(rutas, device, canales_entrada=4):
    A = construir_matriz_adyacencia()
    modelos = []
    for r in rutas:
        m = STGCN(A, canales_entrada=canales_entrada).to(device)
        checkpoint.cargar(r, modelo=m, device=device)
        modelos.append(m.eval())
    return Ensamble(modelos).to(device).eval()


if __name__ == "__main__":
    # chequeo que el truco del logaritmo devuelva exactamente el promedio
    import sys
    rutas = sys.argv[1:]
    if not rutas:
        raise SystemExit("uso: python ensamble.py checkpoint1.pt checkpoint2.pt ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ens = cargar_ensamble(rutas, device)
    x = torch.randn(2, 32, 33, 4, device=device)
    with torch.no_grad():
        individuales = torch.stack([torch.softmax(m(x), 1) for m in ens.modelos])
        esperado = individuales.mean(0)[:, 1]
        obtenido = torch.softmax(ens(x), 1)[:, 1]
    print(f"{len(rutas)} modelos | P(fall) de cada uno: "
          + ", ".join(f"{p:.4f}" for p in individuales[:, 0, 1].tolist()))
    print(f"promedio esperado: {esperado[0]:.6f}  |  via softmax(log p): {obtenido[0]:.6f}")
    print(f"error maximo: {(esperado - obtenido).abs().max():.2e}")
    assert torch.allclose(esperado, obtenido, atol=1e-6), "el softmax no recupera el promedio"
    print("OK")
