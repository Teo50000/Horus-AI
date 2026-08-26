# -*- coding: utf-8 -*-
"""
comun.py · HORUS — utilidades compartidas por el tracking local y el global.

Solo numpy. A propósito: el tracking corre DESPUÉS del único sync GPU->CPU
del motor de inferencia, sobre listas de detecciones que ya están en el host.
Meter torch acá volvería a atar esta capa a la placa sin ganar nada — son
matrices de 100x100 en el peor caso.

Contiene:
  - iou_matriz()       IoU vectorizado entre dos conjuntos de cajas
  - contencion_matriz() qué fracción de una caja está dentro de otra
  - solape_maximo()    IoU o contención, el que sea mayor
  - asignar()          asignación óptima (húngaro) con umbral de corte
  - conversiones de caja xyxy <-> cxcyah (la parametrización del Kalman)
  - punto_pie()        el punto que representa "dónde está parado" un objeto

Sobre el húngaro
----------------
Se usa `scipy.optimize.linear_sum_assignment` si scipy está instalado; si no,
hay una implementación propia (Jonker-Volgenant / caminos aumentantes, la
misma familia que usa scipy) que da el MISMO resultado. Está verificada
contra scipy sobre matrices aleatorias en `probar_tracking.py --autotest`.

No es un detalle cosmético: el greedy "el par de mayor IoU primero" que usan
muchos trackers caseros se equivoca justo cuando importa — dos personas que
se cruzan. El óptimo global cuesta lo mismo a estas escalas.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:                                     # scipy es opcional
    from scipy.optimize import linear_sum_assignment as _lsa
    HAY_SCIPY = True
except Exception:                        # pragma: no cover
    _lsa = None
    HAY_SCIPY = False


# Costo con el que se rellenan los pares prohibidos. Finito a propósito: con
# np.inf el algoritmo se rompe. Los pares que salgan con este costo se
# descartan después, al filtrar por umbral.
COSTO_PROHIBIDO = 1e6


# --------------------------------------------------------------------------- #
# Cajas
# --------------------------------------------------------------------------- #
def a_array(cajas: Sequence[Sequence[float]]) -> np.ndarray:
    """Lista de cajas -> array (N, 4) float32. Devuelve (0, 4) si está vacía."""
    if cajas is None or len(cajas) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(cajas, dtype=np.float32).reshape(-1, 4)


def iou_matriz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU de cada caja de `a` (N,4) contra cada una de `b` (M,4) -> (N, M)."""
    a = a_array(a)
    b = a_array(b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def contencion_matriz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Qué fracción de cada caja de `a` cae adentro de cada una de `b`. (N, M)

    Hace falta porque **el IoU no ve las cajas anidadas**. El torso de una
    persona dentro de su cuerpo entero puede dar IoU 0,35 — por debajo de
    cualquier umbral de NMS razonable — y sin embargo está contenido al 100%.
    Filtrar duplicados solo por IoU deja pasar justo el caso que más molesta:
    varias cajas apiladas sobre la misma persona.
    """
    a = a_array(a)
    b = a_array(b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    area_a = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    return (inter / np.maximum(area_a[:, None], 1e-9)).astype(np.float32)


def solape_maximo(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """El más exigente de tres criterios entre cada par: IoU, cuánto de `a`
    está dentro de `b`, y cuánto de `b` está dentro de `a`. (N, M)

    Un duplicado puede ser parecido (IoU alto), estar adentro, o contener al
    otro. Los tres casos son el mismo objeto contado dos veces.
    """
    a = a_array(a)
    b = a_array(b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.maximum(iou_matriz(a, b),
                      np.maximum(contencion_matriz(a, b),
                                 contencion_matriz(b, a).T))


def expandir(cajas: np.ndarray, factor: float) -> np.ndarray:
    """Agranda cada caja un `factor` de su propio ancho y alto por lado.

    Es el "buffered IoU": dos cajas que no se tocan pero están cerca pasan a
    tener solape medible, y el costo de asociación deja de ser un acantilado
    en 0. Proporcional a la caja y no en píxeles fijos, por la misma razón de
    siempre — un objeto de 20 px y uno de 400 px no comparten escala.
    """
    cajas = a_array(cajas)
    if factor <= 0 or cajas.shape[0] == 0:
        return cajas
    w = (cajas[:, 2] - cajas[:, 0]) * factor
    h = (cajas[:, 3] - cajas[:, 1]) * factor
    fuera = cajas.copy()
    fuera[:, 0] -= w
    fuera[:, 2] += w
    fuera[:, 1] -= h
    fuera[:, 3] += h
    return fuera


def xyxy_a_cxcyah(caja: Sequence[float]) -> np.ndarray:
    """(x1,y1,x2,y2) -> (cx, cy, aspecto=w/h, alto). Estado del Kalman."""
    x1, y1, x2, y2 = (float(v) for v in caja)
    w = max(x2 - x1, 1e-3)
    h = max(y2 - y1, 1e-3)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=np.float64)


def cxcyah_a_xyxy(estado: Sequence[float]) -> Tuple[float, float, float, float]:
    """Inverso de xyxy_a_cxcyah, tolerante a un alto/aspecto degenerado."""
    cx, cy, a, h = (float(v) for v in estado[:4])
    h = max(h, 1e-3)
    w = max(a * h, 1e-3)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def punto_pie(caja: Sequence[float]) -> Tuple[float, float]:
    """Centro-abajo de la caja: dónde toca el piso el objeto.

    Para decidir si alguien está DENTRO de una zona del plano, el centro de la
    caja miente — una persona parada en el borde de la zona tiene el centro
    del torso afuera. El punto que importa son los pies.
    """
    x1, y1, x2, y2 = (float(v) for v in caja)
    return ((x1 + x2) / 2.0, y2)


def centro(caja: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in caja)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def diagonal(caja: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in caja)
    return float(np.hypot(x2 - x1, y2 - y1))


# --------------------------------------------------------------------------- #
# Asignación óptima
# --------------------------------------------------------------------------- #
def asignar(costo: np.ndarray,
            costo_max: float = np.inf) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Empareja filas con columnas minimizando el costo total.

    Devuelve (pares, filas_sueltas, columnas_sueltas). Un par cuyo costo supere
    `costo_max` se deshace y sus dos partes vuelven a "sueltas": el óptimo
    global puede asignar un par malísimo con tal de cerrar la cuenta, y ese par
    es exactamente un cambio de identidad.
    """
    costo = np.asarray(costo, dtype=np.float64)
    if costo.size == 0:
        return [], list(range(costo.shape[0])), list(range(costo.shape[1]))

    if HAY_SCIPY:
        fil, col = _lsa(costo)
    else:                                # pragma: no cover - camino sin scipy
        fil, col = _hungaro(costo)

    pares: List[Tuple[int, int]] = []
    usadas_f, usadas_c = set(), set()
    for i, j in zip(fil, col):
        if costo[i, j] <= costo_max:
            pares.append((int(i), int(j)))
            usadas_f.add(int(i))
            usadas_c.add(int(j))

    sueltas_f = [i for i in range(costo.shape[0]) if i not in usadas_f]
    sueltas_c = [j for j in range(costo.shape[1]) if j not in usadas_c]
    return pares, sueltas_f, sueltas_c


def _hungaro(costo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Húngaro por caminos aumentantes con potenciales. O(n^2 * m).

    Implementación propia para no obligar a instalar scipy en la caja de
    producción. `probar_tracking.py --autotest` la compara contra scipy.
    """
    c = np.asarray(costo, dtype=np.float64)
    traspuesta = False
    if c.shape[0] > c.shape[1]:
        c = c.T
        traspuesta = True
    n, m = c.shape

    u = np.zeros(n + 1)                       # potenciales de fila
    v = np.zeros(m + 1)                       # potenciales de columna
    p = np.zeros(m + 1, dtype=np.int64)       # p[j] = fila asignada a la columna j
    camino = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        usada = np.zeros(m + 1, dtype=bool)

        while True:
            usada[j0] = True
            i0 = int(p[j0])
            libres = ~usada[1:]
            actual = c[i0 - 1] - u[i0] - v[1:]
            mejora = libres & (actual < minv[1:])
            minv[1:][mejora] = actual[mejora]
            camino[1:][mejora] = j0

            cand = np.where(libres)[0]
            k = cand[int(np.argmin(minv[1:][cand]))]
            j1 = int(k) + 1
            delta = float(minv[j1])

            # Reajuste de potenciales: las columnas ya visitadas conservan su
            # costo reducido, las libres bajan delta.
            u[p[usada]] += delta
            v[usada] -= delta
            minv[~usada] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:                              # deshacer el camino aumentante
            j1 = int(camino[j0])
            p[j0] = p[j1]
            j0 = j1

    filas, cols = [], []
    for j in range(1, m + 1):
        if p[j] != 0:
            filas.append(int(p[j]) - 1)
            cols.append(j - 1)

    f = np.asarray(filas, dtype=np.int64)
    g = np.asarray(cols, dtype=np.int64)
    if traspuesta:
        f, g = g, f
    orden = np.argsort(f)
    return f[orden], g[orden]
