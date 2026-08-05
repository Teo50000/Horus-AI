import numpy as np

CONEXIONES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

def construir_matriz_adyacencia(n_puntos=33):
    A = np.zeros((n_puntos, n_puntos))
    for i, j in CONEXIONES:
        A[i, j] = 1
        A[j, i] = 1
    A += np.eye(n_puntos)
    return A