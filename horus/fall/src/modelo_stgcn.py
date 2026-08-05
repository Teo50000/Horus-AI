import torch
import torch.nn as nn
import numpy as np


class ConvGrafoEspacial(nn.Module):
    """Aplica la 'mezcla' entre puntos conectados según la matriz de adyacencia."""
    def __init__(self, canales_entrada, canales_salida, matriz_adyacencia):
        super().__init__()
        # Normalizamos la matriz de adyacencia (técnica estándar en GCN, evita que
        # los puntos con muchas conexiones dominen el cálculo)
        A = matriz_adyacencia
        grados = A.sum(axis=1)
        grados_inv_sqrt = np.power(grados, -0.5, where=grados > 0)
        D_inv_sqrt = np.diag(grados_inv_sqrt)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        self.register_buffer("A", torch.tensor(A_norm, dtype=torch.float32))
        self.conv = nn.Conv2d(canales_entrada, canales_salida, kernel_size=1)

    def forward(self, x):
        # x: (batch, canales, frames, puntos)
        # Multiplicamos cada frame por la matriz de adyacencia normalizada:
        # esto "mezcla" la información de cada punto con la de sus vecinos conectados
        x = torch.einsum("bcfp,pq->bcfq", x, self.A)
        x = self.conv(x)
        return x


class BloqueSTGCN(nn.Module):
    """Un bloque = convolución espacial (grafo) + convolución temporal."""
    def __init__(self, canales_entrada, canales_salida, matriz_adyacencia, stride_temporal=1):
        super().__init__()
        self.conv_espacial = ConvGrafoEspacial(canales_entrada, canales_salida, matriz_adyacencia)
        self.bn1 = nn.BatchNorm2d(canales_salida)

        self.conv_temporal = nn.Conv2d(
            canales_salida, canales_salida,
            kernel_size=(9, 1), stride=(stride_temporal, 1), padding=(4, 0)
        )
        self.bn2 = nn.BatchNorm2d(canales_salida)
        self.relu = nn.ReLU()

        # Conexión residual (ayuda a que el entrenamiento sea más estable)
        if canales_entrada == canales_salida and stride_temporal == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(canales_entrada, canales_salida, kernel_size=1, stride=(stride_temporal, 1)),
                nn.BatchNorm2d(canales_salida)
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.bn1(self.conv_espacial(x)))
        x = self.bn2(self.conv_temporal(x))
        x = self.relu(x + res)
        return x


class STGCN(nn.Module):
    def __init__(self, matriz_adyacencia, canales_entrada=2, clases=2):
        super().__init__()
        A = matriz_adyacencia

        self.bloques = nn.ModuleList([
            BloqueSTGCN(canales_entrada, 32, A),
            BloqueSTGCN(32, 32, A),
            BloqueSTGCN(32, 64, A, stride_temporal=2),
            BloqueSTGCN(64, 64, A),
        ])

        self.pool_global = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, clases)

    def forward(self, x):
        # x entra como: (batch, frames, puntos, canales) -> reordenar a (batch, canales, frames, puntos)
        x = x.permute(0, 3, 1, 2)

        for bloque in self.bloques:
            x = bloque(x)

        x = self.pool_global(x)          # (batch, canales, 1, 1)
        x = x.view(x.size(0), -1)        # (batch, canales)
        return self.fc(x)