"""Las piezas de la deteccion de caidas que NO son el modelo.

Las usan las dos puertas de entrada al ST-GCN:

  - fall/src/maquina_caida.py   una persona, la mide el harness offline
                                (17_evaluar_evento.py) y la usan los scripts
                                de webcam
  - fall/detector_caidas.py     la cabeza de la app: muchas personas por
                                camara, clasifica en lote y le deja la
                                politica del evento a ReglaCaida

Si cada una tuviera su copia, lo que mide el harness dejaria de ser lo que
corre en la app -- que es exactamente lo que paso antes: la cabeza de la app
heredo los bugs de 12_webcam.py y nadie los podia ver porque la evaluacion
miraba otro codigo.

Solo numpy, a proposito: la cabeza de la app tiene que poder importarse sin
torch ni mediapipe (la suite de fusion corre asi).
"""
from collections import deque

import numpy as np

from preproceso import IDX_CI, IDX_CD, IDX_HI, IDX_HD

TORSO = [IDX_CI, IDX_CD, IDX_HI, IDX_HD]


def verticalidad_de(kp_norm):
    """~1 con el torso vertical (de pie), ~0 horizontal (acostado). Pose ya
    normalizada: cadera en el origen, escala = largo del torso."""
    return -float((kp_norm[IDX_HI, 1] + kp_norm[IDX_HD, 1]) / 2.0)


def duracion_ventana(ventana, fps_canonica):
    """32 muestras a 25 fps son 31 intervalos = 1.24 s. Es lo que vio el
    modelo entrenando: 32 frames consecutivos de video a 25 fps."""
    return (ventana - 1) / fps_canonica


def remuestrear_clip(ts, poses, t_fin, ventana, fps_canonica):
    """Lleva un buffer de poses tomadas a cualquier ritmo al clip que espera
    el ST-GCN: `ventana` muestras equiespaciadas a `fps_canonica`, terminando
    en `t_fin`, con las velocidades recalculadas sobre la secuencia nueva.

    Sin esto, a 10 fps la ventana del modelo cubria 3.2 s en vez de 1.24 s y
    las velocidades eran 2.5x mas grandes que en entrenamiento. Medido en
    Le2i: -12 puntos de recall y +0.7 s de latencia (resultados/README.md).

    Interpolacion lineal. Los huecos de deteccion entran al buffer como la
    ultima pose repetida, asi que interpolar entre dos valores iguales da una
    meseta: el hueco no se rellena inventando movimiento.

    ts: (n,) creciente   poses: (n, 33, 2)   -> (ventana, 33, 4) = x, y, vx, vy
    """
    ts = np.asarray(ts, dtype=float)
    poses = np.asarray(poses, dtype=float)
    objetivo = np.linspace(t_fin - duracion_ventana(ventana, fps_canonica), t_fin, ventana)

    if len(ts) == 1:
        xy = np.repeat(poses[:1], ventana, axis=0)
    else:
        j = np.clip(np.searchsorted(ts, objetivo, side="right") - 1, 0, len(ts) - 2)
        dt = ts[j + 1] - ts[j]
        w = np.where(dt > 1e-9, (objetivo - ts[j]) / np.where(dt > 1e-9, dt, 1.0), 0.0)
        w = np.clip(w, 0.0, 1.0)[:, None, None]
        xy = poses[j] * (1 - w) + poses[j + 1] * w

    vel = np.zeros_like(xy)
    vel[1:] = xy[1:] - xy[:-1]
    return np.concatenate([xy, vel], axis=2)


def cobertura(ts_buffer, ventana, fps_canonica):
    """Fraccion de la ventana temporal que el buffer ya cubre, 0..1. Con 1 se
    puede clasificar."""
    if len(ts_buffer) < 2:
        return 0.0
    return min((ts_buffer[-1] - ts_buffer[0]) / duracion_ventana(ventana, fps_canonica), 1.0)


class PuertaCaida:
    """El filtro que separa "se cayo" de "esta acostado".

    El clasificador solo no puede: 32 frames de alguien quieto en el piso se
    ven igual se haya caido hace dos segundos o se haya acostado hace una hora.
    Lo que distingue la caida es la FIRMA: hubo un movimiento brusco del torso
    y, ademas, el torso paso de vertical a horizontal.

    Tres arreglos respecto de la version de 12_webcam.py (y de la cabeza de la
    app, que era una copia), todos medidos en el harness:

    1. El nivel de "movimiento normal" (baseline) se sembraba con la velocidad
       del primer frame, que es 0 por construccion; recibia un paso de EMA y se
       congelaba, porque la rama "es pico" no lo actualizaba. Contra ese piso
       TODO frame era pico: el filtro de velocidad no filtraba nada. Ahora se
       siembra con la mediana de `warmup_seg` de detecciones reales, la EMA es
       en tiempo continuo (invariante a fps) y se actualiza siempre, y hay piso.
    2. Pico y caida de verticalidad se exigian en el MISMO frame. No pasan
       juntos: la velocidad es maxima a mitad de la caida y el torso termina de
       bajar despues. Exigirlos juntos perdia 16 caidas de 130.
    3. La velocidad esta en unidades de torso por SEGUNDO, no por frame: el
       mismo movimiento da el mismo numero a 25 fps que a 10.
    """

    def __init__(self, factor_pico=2.5, tau_baseline_seg=0.8, warmup_seg=0.4,
                 piso_baseline=0.05, ventana_vertical_previa_seg=2.0,
                 caida_vertical_minima=0.35, ventana_pico_a_caida_seg=1.0,
                 bypass_historial_corto=False):
        self.factor_pico = factor_pico
        self.tau_baseline_seg = tau_baseline_seg
        self.warmup_seg = warmup_seg
        self.piso_baseline = piso_baseline
        self.ventana_vertical_previa_seg = ventana_vertical_previa_seg
        self.caida_vertical_minima = caida_vertical_minima
        self.ventana_pico_a_caida_seg = ventana_pico_a_caida_seg
        self.bypass_historial_corto = bypass_historial_corto
        self.reiniciar()

    def reiniciar(self):
        # Al perder a la persona se tira todo. Si quedaran de antes, un
        # baseline viejo y alto bloquea una caida real al volver, y un pico
        # viejo valida una alarma con un movimiento que no tiene nada que ver.
        self.baseline = None
        self._warmup = []
        self._ultima_real_ts = None
        self._ultimo_pico_vel_ts = None
        self.ultimo_pico_ts = None          # ultima firma COMPLETA (pico + torso cayo)
        self._historial = deque()           # (t, verticalidad)
        self.ultima_verticalidad = None

    def actualizar(self, kp_norm, vel, t):
        """Llamar SOLO con detecciones reales. Un frame repetido para tapar un
        hueco tiene velocidad 0 y arrastraria el baseline hacia abajo."""
        verticalidad = verticalidad_de(kp_norm)
        self.ultima_verticalidad = verticalidad

        dt = (t - self._ultima_real_ts) if self._ultima_real_ts is not None else None
        self._ultima_real_ts = t
        if dt is not None and dt > 1e-6:
            v = float(np.linalg.norm(vel[TORSO], axis=1).mean()) / dt

            if self.baseline is None:
                self._warmup.append((t, v))
                if t - self._warmup[0][0] >= self.warmup_seg:
                    self.baseline = max(float(np.median([x for _, x in self._warmup])),
                                        self.piso_baseline)
            else:
                if v > self.baseline * self.factor_pico:
                    self._ultimo_pico_vel_ts = t

                if (self._ultimo_pico_vel_ts is not None
                        and (t - self._ultimo_pico_vel_ts) <= self.ventana_pico_a_caida_seg):
                    cubierto = t - self._historial[0][0] if self._historial else 0.0
                    if cubierto < self.ventana_vertical_previa_seg * 0.8:
                        # recien la vemos: no puedo probar que ya estaba
                        # acostada antes. Medido: dejarlo pasar solo sumaba
                        # gente levantandose al arrancar el video.
                        if self.bypass_historial_corto:
                            self.ultimo_pico_ts = t
                    else:
                        previa = max(x for _, x in self._historial)
                        if (previa - verticalidad) >= self.caida_vertical_minima:
                            self.ultimo_pico_ts = t

                alpha = 1.0 - np.exp(-dt / self.tau_baseline_seg)
                v_ema = min(v, self.baseline * self.factor_pico)
                self.baseline = max((1 - alpha) * self.baseline + alpha * v_ema, self.piso_baseline)

        self._historial.append((t, verticalidad))
        while self._historial and t - self._historial[0][0] > self.ventana_vertical_previa_seg:
            self._historial.popleft()

    def firma_reciente(self, t, ventana_seg):
        return self.ultimo_pico_ts is not None and (t - self.ultimo_pico_ts) <= ventana_seg
