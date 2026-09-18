from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from preproceso import IDX_CI, IDX_CD, IDX_HI, IDX_HD

# Maquina de estados de la alarma de caida. Es LA implementacion: la usan
# 12_webcam.py, 07_alerta/webcam_integrado.py y el harness offline
# (17_evaluar_evento.py), asi lo que se mide offline es lo que corre en vivo.
#
# Causal: solo mira el pasado. Cada frame entra por actualizar(kp_norm, t) donde
# kp_norm es la pose normalizada de ese frame (o None si no hubo deteccion
# confiable) y t es el reloj en segundos (perf_counter en vivo, timestamp del
# stream offline). Toda la logica temporal usa ese unico t.


@dataclass
class ConfigDetector:
    ventana: int = 32               # frames por clip que espera el ST-GCN
    paso: int = 4                   # clasifico cada `paso` llamadas, no cada frame
    umbral: float = 0.5             # punto de operacion elegido en el barrido
    persistencia: float = 1.0       # segundos que la prob debe sostenerse para disparar
    # cuanto tolero sin deteccion antes de resetear el buffer. MediaPipe pierde a la
    # persona con frecuencia cuando esta en el piso (37% de las caidas de Le2i s0
    # tienen un hueco > 0.6s dentro del evento); con 0.6s el reset mataba la racha
    # en medio de la caida. 2.0s: +10.7 puntos de recall a 10 fps con las mismas
    # falsas/hora (ver resultados/README.md, corridas C3_*).
    tolerancia_sin_deteccion_seg: float = 2.0

    # --- regla de persistencia: exigir un pico de movimiento real antes de la alarma ---
    # Una pose estatica en el piso es ambigua (fallen vs. lying calmo). Lo que si
    # distingue una caida real es que HUBO un movimiento brusco justo antes. Sin esto,
    # alguien que se acuesta y se queda quieto mucho tiempo le da al clasificador
    # muchas oportunidades de fallar una sola vez y sostenerlo el segundo necesario.
    factor_pico_velocidad: float = 2.5   # cuanto por encima del "ruido normal" cuenta como pico (barrido: 2.5 > 3.0 en recall sin falsas extra)
    ventana_post_pico_seg: float = 4.0   # cuanto tiempo despues de un pico dejo disparar la alarma
    # El "ruido normal" es una EMA de la velocidad (en unidades de torso POR SEGUNDO,
    # asi el ratio no depende de los fps de la camara). tau = constante de tiempo;
    # 0.8s equivale al alpha=0.05 por frame que habia a 25 fps.
    tau_baseline_seg: float = 0.8
    # antes de detectar picos junto este tiempo de detecciones reales y siembro el
    # baseline con su mediana. Sin esto el baseline arrancaba del primer frame
    # (vel=0 por construccion) y quedaba congelado en ~0 para siempre: todo era pico.
    warmup_baseline_seg: float = 0.4
    piso_baseline: float = 0.05          # unidades de torso/seg; evita que una quietud perfecta lo lleve a 0

    # --- un pico de velocidad NO alcanza: tiene que ser una transicion real de
    # pie -> acostado. Si la persona ya esta en el piso y se mueve (se acomoda,
    # se da vuelta), el baseline de velocidad ya cayo casi a cero por la quietud
    # previa, y ese movimiento normal se ve como un "pico" gigante en terminos
    # relativos aunque no haya pasado nada. Para filtrar eso, exijo que la
    # verticalidad del torso (cadera->hombro) haya caido de verdad: si ya venia
    # baja (acostado), un pico de velocidad no arma la alarma.
    ventana_vertical_previa_seg: float = 2.0  # cuanto miro hacia atras para saber si "antes" estaba de pie
    caida_vertical_minima: float = 0.35       # cuanta verticalidad tiene que haber bajado para contar como caida real
    # el pico de velocidad y la caida de verticalidad NO pasan en el mismo frame:
    # la velocidad es maxima a mitad de la caida, el torso termina de bajar despues.
    # Acepto la firma "pico + verticalidad cayo" si la caida se confirma hasta este
    # tiempo despues del pico.
    ventana_pico_a_caida_seg: float = 1.0
    # si recien volvimos a ver a la persona (historial de verticalidad corto) no
    # puedo probar que "antes estaba de pie". True = acepto el pico igual (prefiero
    # un falso positivo a perder una caida que paso justo al reaparecer). Medido en
    # Le2i: casi todo lo que deja pasar es gente levantandose rapido al arrancar el
    # video, y una persona que aparece de pie y cae enseguida SI tiene historial
    # suficiente. Ver resultados/README.md.
    bypass_historial_corto: bool = False

    # False = alarma solo con modelo + persistencia, sin exigir pico/verticalidad.
    # Sirve para medir cuanto aportan las heuristicas; en produccion va True.
    exigir_pico: bool = True


@dataclass
class Estado:
    prob: float                 # ultima P(fall) calculada (0.0 si el buffer no esta lleno o hubo reset)
    alarma_nueva: bool          # True solo en la llamada en que la alarma pasa de apagada a activa
    alarma_activa: bool
    n_buffer: int               # cuantos frames hay en el buffer (hasta `ventana`)
    pico_reciente: bool         # hubo un pico de velocidad valido dentro de ventana_post_pico_seg
    verticalidad: float | None  # ultima verticalidad del torso medida en una deteccion real
    clasifico: bool = False     # True si en esta llamada corrio el modelo (prob es fresca)


class DetectorCaidas:
    def __init__(self, modelo, device, cfg: ConfigDetector | None = None):
        self.modelo = modelo
        self.device = device
        self.cfg = cfg or ConfigDetector()
        self.reiniciar()

    def reiniciar(self):
        c = self.cfg
        self.buffer = deque(maxlen=c.ventana)   # ultimos `ventana` frames procesados (x,y,vx,vy)
        self.ultimo_norm = None                 # ultimo frame valido, para rellenar huecos sin mirar al futuro
        self.anterior_norm = None               # frame previo, para la velocidad
        self.ultima_deteccion_ts = None         # reloj de la ultima deteccion confiable (None = todavia ninguna)
        self.prob_actual = 0.0
        self.inicio_racha = None
        self.alarma_activa = False
        self.baseline_vel = None                # nivel "normal" de velocidad (torso/seg), EMA
        self.warmup_vels = []                   # (t, v) hasta sembrar el baseline
        self.ultima_real_ts = None              # reloj de la ultima deteccion REAL (para dt de la velocidad)
        self.ultimo_pico_vel_ts = None          # reloj del ultimo pico de velocidad (sin mirar verticalidad)
        self.ultimo_pico_ts = None              # reloj de la ultima firma de caida completa (pico + torso cayo)
        self.historial_vertical = deque()       # (timestamp, verticalidad) recientes, solo detecciones reales
        self.ultima_verticalidad = None
        self.n_llamadas = 0

    def _resetear_estado(self):
        # nunca hubo deteccion, o hace rato que no hay ninguna confiable: dejo de
        # arrastrar la ultima pose y reseteo todo en vez de seguir clasificando
        # con datos viejos/congelados. baseline y pico tambien son historia vieja:
        # si quedan de antes de irse de cuadro, pueden bloquear una caida real al
        # volver (baseline stale muy alto) o validar una alarma con un pico que no
        # tiene nada que ver con lo que paso al reaparecer.
        self.buffer.clear()
        self.ultimo_norm = None
        self.anterior_norm = None
        self.inicio_racha = None
        self.alarma_activa = False
        self.prob_actual = 0.0
        self.historial_vertical.clear()
        self.baseline_vel = None
        self.warmup_vels = []
        self.ultima_real_ts = None
        self.ultimo_pico_vel_ts = None
        self.ultimo_pico_ts = None

    def _actualizar_pico(self, kp_norm, vel, t):
        c = self.cfg
        # ~1 con el torso vertical (de pie), ~0 con el torso horizontal (acostado)
        verticalidad = -((kp_norm[IDX_HI, 1] + kp_norm[IDX_HD, 1]) / 2)
        self.ultima_verticalidad = verticalidad

        # velocidad del torso en unidades de torso POR SEGUNDO, medida entre
        # detecciones reales (si hubo un hueco, dt es el hueco entero)
        dt = (t - self.ultima_real_ts) if self.ultima_real_ts is not None else None
        self.ultima_real_ts = t
        if dt is not None and dt > 1e-6:
            v = np.linalg.norm(vel[[IDX_CI, IDX_CD, IDX_HI, IDX_HD]], axis=1).mean() / dt

            if self.baseline_vel is None:
                # warmup: siembro con la mediana de las primeras detecciones para no
                # arrancar de un valor espurio. Mientras tanto no detecto picos.
                self.warmup_vels.append((t, v))
                if t - self.warmup_vels[0][0] >= c.warmup_baseline_seg:
                    self.baseline_vel = max(float(np.median([x for _, x in self.warmup_vels])), c.piso_baseline)
            else:
                if v > self.baseline_vel * c.factor_pico_velocidad:
                    self.ultimo_pico_vel_ts = t

                # firma de caida = hubo un pico de velocidad hace poco Y la verticalidad
                # veia "de pie" hace poco y ahora cayo (transicion real). Si ya estaba
                # acostado, la verticalidad previa tambien era baja y un movimiento no
                # cuenta como caida. Los dos criterios se evaluan en frames distintos:
                # el pico llega a mitad de la caida, el torso termina de bajar despues.
                #
                # excepcion: si recien volvimos a ver a la persona (poco historial
                # todavia, ej. reaparecio en cuadro o venia de un hueco largo), no
                # puedo probar que "ya estaba acostada de antes" porque no la vi.
                # En ese caso prefiero un falso positivo ocasional a perderme una
                # caida real que paso justo al reaparecer.
                if (self.ultimo_pico_vel_ts is not None
                        and (t - self.ultimo_pico_vel_ts) <= c.ventana_pico_a_caida_seg):
                    cobertura_seg = t - self.historial_vertical[0][0] if self.historial_vertical else 0.0
                    if cobertura_seg < c.ventana_vertical_previa_seg * 0.8:
                        if c.bypass_historial_corto:
                            self.ultimo_pico_ts = t
                    else:
                        vertical_previa_max = max(x for _, x in self.historial_vertical)
                        if (vertical_previa_max - verticalidad) >= c.caida_vertical_minima:
                            self.ultimo_pico_ts = t

                # EMA en tiempo continuo, se actualiza SIEMPRE. El pico entra recortado
                # a factor*baseline para que un solo golpe no infle el nivel de ruido
                # (y deje pasar el siguiente); el piso evita que la quietud lo lleve
                # a ~0 y todo se vuelva pico -- exactamente el bug que habia.
                alpha = 1.0 - np.exp(-dt / c.tau_baseline_seg)
                v_para_ema = min(v, self.baseline_vel * c.factor_pico_velocidad)
                self.baseline_vel = max((1 - alpha) * self.baseline_vel + alpha * v_para_ema, c.piso_baseline)

        self.historial_vertical.append((t, verticalidad))
        while self.historial_vertical and t - self.historial_vertical[0][0] > c.ventana_vertical_previa_seg:
            self.historial_vertical.popleft()

    def _pico_reciente(self, t):
        if not self.cfg.exigir_pico:
            return True
        return (self.ultimo_pico_ts is not None
                and (t - self.ultimo_pico_ts) <= self.cfg.ventana_post_pico_seg)

    def actualizar(self, kp_norm, t) -> Estado:
        c = self.cfg

        if kp_norm is not None:
            self.ultima_deteccion_ts = t
        sin_deteccion_seg = (t - self.ultima_deteccion_ts) if self.ultima_deteccion_ts is not None else None

        if sin_deteccion_seg is None or sin_deteccion_seg > c.tolerancia_sin_deteccion_seg:
            self._resetear_estado()
        else:
            deteccion_real_este_frame = kp_norm is not None
            # hueco corto (parpadeo de deteccion): repito el ultimo valido,
            # nunca miro hacia adelante
            if kp_norm is None:
                kp_norm = self.ultimo_norm

            if kp_norm is not None:
                self.ultimo_norm = kp_norm
                vel = kp_norm - self.anterior_norm if self.anterior_norm is not None else np.zeros_like(kp_norm)
                self.anterior_norm = kp_norm
                self.buffer.append(np.concatenate([kp_norm, vel], axis=1))

                # detecto picos de movimiento solo con detecciones reales: un
                # frame repetido (fallback) tiene vel=0 y ensuciaria el baseline.
                if deteccion_real_este_frame:
                    self._actualizar_pico(kp_norm, vel, t)

        # clasificacion: solo con el buffer lleno y cada `paso` llamadas
        alarma_nueva = False
        clasifico = False
        if len(self.buffer) == c.ventana and self.n_llamadas % c.paso == 0:
            clasifico = True
            clip = torch.tensor(np.array(self.buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                self.prob_actual = torch.softmax(self.modelo(clip), 1)[0, 1].item()

            if self.prob_actual >= c.umbral:
                if self.inicio_racha is None:
                    self.inicio_racha = t
                elif not self.alarma_activa and (t - self.inicio_racha) >= c.persistencia:
                    # si no hubo un movimiento brusco reciente, no disparo: es
                    # alguien quieto en el piso desde hace rato, no alguien que
                    # se acaba de caer.
                    if self._pico_reciente(t):
                        self.alarma_activa = True
                        alarma_nueva = True
            else:
                self.inicio_racha = None
                self.alarma_activa = False

        self.n_llamadas += 1
        return Estado(
            prob=self.prob_actual,
            alarma_nueva=alarma_nueva,
            alarma_activa=self.alarma_activa,
            n_buffer=len(self.buffer),
            pico_reciente=self._pico_reciente(t),
            verticalidad=self.ultima_verticalidad,
            clasifico=clasifico,
        )
