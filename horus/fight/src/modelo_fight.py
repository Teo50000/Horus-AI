import torch.nn as nn
from torchvision.models.video import mc3_18, MC3_18_Weights

# MC3-18: 3D CNN liviana (mezcla conv 3D solo al principio, después 2D+temporal),
# pre-entrenada en Kinetics-400 (dataset gigante sobre comportamiento de personas).
MEDIA_KINETICS = [0.43216, 0.394666, 0.37645]
STD_KINETICS = [0.22803, 0.22145, 0.216989]


def crear_modelo(preentrenado=True):
    pesos = MC3_18_Weights.DEFAULT if preentrenado else None
    modelo = mc3_18(weights=pesos)
    modelo.fc = nn.Linear(modelo.fc.in_features, 2)  # 0=nofight, 1=fight
    return modelo
