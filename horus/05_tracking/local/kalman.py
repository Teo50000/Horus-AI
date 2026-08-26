# -*- coding: utf-8 -*-
"""
kalman.py · HORUS — filtro de Kalman de velocidad constante para el tracking.

Estado de 8 dimensiones: (cx, cy, aspecto, alto, vcx, vcy, vaspecto, valto).

Por qué esta parametrización y no (x1,y1,x2,y2): las cuatro esquinas están
correlacionadas entre sí (si la caja se mueve, las cuatro se mueven juntas) y
el filtro las trata como independientes. Centro + forma separa lo que se mueve
de lo que no, que es justo lo que hace falta para predecir dónde va a estar la
persona en el próximo frame.

Por qué importa acá: el reporte de campo dice que la persona "se pierde con
movimiento rápido". Sin predicción, un tracker compara la caja del frame t-1
contra la del frame t; si la persona se movió más que su propio ancho, el IoU
da 0 y el track muere. Con predicción se compara contra dónde DEBERÍA estar,
y el IoU vuelve a ser alto.

El ruido es proporcional al alto de la caja (no constante): un objeto lejano
ocupa 20 px y uno cercano 400, y un error de 10 px significa cosas muy
distintas en cada caso. Es la variante que usa ByteTrack y es la razón de que
funcione en escenas con mucha profundidad — que es toda cámara de vigilancia.

Sin dependencias fuera de numpy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class FiltroKalman:
    """Velocidad constante en (cx, cy, aspecto, alto).

    Uso:
        kf = FiltroKalman()
        media, cov = kf.iniciar(medicion)          # medicion = (cx,cy,a,h)
        media, cov = kf.predecir(media, cov)       # una vez por frame
        media, cov = kf.corregir(media, cov, med)  # solo si hubo asociación
    """

    def __init__(self,
                 peso_posicion: float = 1.0 / 20.0,
                 peso_velocidad: float = 1.0 / 160.0) -> None:
        ndim = 4
        self.ndim = ndim
        self.peso_pos = float(peso_posicion)
        self.peso_vel = float(peso_velocidad)

        # x_{t} = F x_{t-1}. Con dt = 1 frame: posición += velocidad.
        self._F = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._F[i, ndim + i] = 1.0
        # z = H x. Solo se observa la posición/forma, nunca la velocidad.
        self._H = np.eye(ndim, 2 * ndim, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def iniciar(self, medicion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Track nuevo: posición = la medición, velocidad = 0 pero muy incierta."""
        med = np.asarray(medicion, dtype=np.float64).reshape(4)
        media = np.concatenate([med, np.zeros(4)])
        h = med[3]

        std = np.array([
            2 * self.peso_pos * h,      # cx
            2 * self.peso_pos * h,      # cy
            1e-2,                       # aspecto: casi seguro
            2 * self.peso_pos * h,      # alto
            10 * self.peso_vel * h,     # vcx: no sabemos nada
            10 * self.peso_vel * h,     # vcy
            1e-5,                       # v del aspecto: no cambia
            10 * self.peso_vel * h,     # v del alto
        ])
        return media, np.diag(np.square(std))

    def predecir(self, media: np.ndarray,
                 cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Un frame hacia adelante. Se llama SIEMPRE, haya o no detección."""
        h = max(float(media[3]), 1e-3)
        std = np.array([
            self.peso_pos * h, self.peso_pos * h, 1e-2, self.peso_pos * h,
            self.peso_vel * h, self.peso_vel * h, 1e-5, self.peso_vel * h,
        ])
        Q = np.diag(np.square(std))

        media = self._F @ media
        cov = self._F @ cov @ self._F.T + Q
        return media, cov

    def proyectar(self, media: np.ndarray,
                  cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estado -> espacio de medición, con el ruido del detector encima."""
        h = max(float(media[3]), 1e-3)
        std = np.array([
            self.peso_pos * h, self.peso_pos * h, 1e-1, self.peso_pos * h,
        ])
        R = np.diag(np.square(std))
        return self._H @ media, self._H @ cov @ self._H.T + R

    def corregir(self, media: np.ndarray, cov: np.ndarray,
                 medicion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Incorpora la detección asociada."""
        med = np.asarray(medicion, dtype=np.float64).reshape(4)
        media_p, cov_p = self.proyectar(media, cov)

        # K = P H^T S^-1, resuelto como sistema lineal en vez de invirtiendo S.
        # Es más estable numéricamente y S es 4x4: cuesta nada.
        try:
            K = np.linalg.solve(cov_p.T, (cov @ self._H.T).T).T
        except np.linalg.LinAlgError:        # pragma: no cover
            K = (cov @ self._H.T) @ np.linalg.pinv(cov_p)

        innovacion = med - media_p
        nueva_media = media + K @ innovacion
        nueva_cov = cov - K @ cov_p @ K.T
        # Simetrizar: el redondeo acumulado la desbalancea y termina dando
        # covarianzas no definidas positivas después de miles de frames.
        nueva_cov = 0.5 * (nueva_cov + nueva_cov.T)
        return nueva_media, nueva_cov

    # ------------------------------------------------------------------ #
    def distancia_gating(self, media: np.ndarray, cov: np.ndarray,
                         mediciones: np.ndarray) -> np.ndarray:
        """Mahalanobis al cuadrado del estado contra varias mediciones (N,4).

        Sirve para prohibir asociaciones geométricamente imposibles antes de
        mirar la apariencia. Con 4 grados de libertad, el 95% cae por debajo
        de 9.4877 (chi-cuadrado).
        """
        media_p, cov_p = self.proyectar(media, cov)
        med = np.asarray(mediciones, dtype=np.float64).reshape(-1, 4)
        d = med - media_p
        try:
            L = np.linalg.cholesky(cov_p)
            z = np.linalg.solve(L, d.T)
            return np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:        # pragma: no cover
            inv = np.linalg.pinv(cov_p)
            return np.einsum("ij,jk,ik->i", d, inv, d)


# 95% para 4 grados de libertad. El umbral de gating que usa el tracker.
CHI2_95_4GL: float = 9.4877
