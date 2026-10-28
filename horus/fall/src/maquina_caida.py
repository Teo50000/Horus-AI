from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from compuertas_caida import PuertaCaida, cobertura, duracion_ventana, margen_buffer, remuestrear_clip

# Maquina de estados de la alarma de caida para UNA persona. La usan los
# scripts de webcam (12_webcam.py, 07_alerta/webcam_integrado.py) y el harness
# offline (17_evaluar_evento.py), asi lo que se mide offline es lo que corre.
#
# La cabeza de la app (fall/detector_caidas.py) es la version multi-persona:
# comparte el remuestreo y la compuerta (src/compuertas_caida.py) pero
# clasifica en lote y le deja la persistencia a ReglaCaida.
#
# Causal: solo mira el pasado. Cada frame entra por actualizar(kp_norm, t) donde
# kp_norm es la pose normalizada de ese frame (o None si no hubo deteccion
# confiable) y t es el reloj en segundos (perf_counter en vivo, timestamp del
# stream offline). Toda la logica temporal usa ese unico t.


@dataclass
class ConfigMaquina:
    ventana: int = 32               # muestras por clip que espera el ST-GCN
    # El modelo entreno con ventanas de 32 frames consecutivos a ~25 fps (1.24s de
    # movimiento) y velocidades = diferencia entre frames a esa tasa. Si le doy un
    # frame de camara por muestra, a 10 fps la ventana cubre 3.2s y las velocidades
    # son 2.5x mas grandes: fuera de distribucion en las dos features. Por eso el
    # buffer guarda (t, pose) y se REMUESTREA a esta tasa antes de clasificar.
    fps_canonica: float = 25.0
    paso_clasif_seg: float = 0.16   # cada cuanto clasifico (por reloj, no cada N frames)
    # punto de operacion elegido con 19_barrer_alarma.py sobre LOSO Le2i, comparando
    # a IGUAL tasa de falsas contra la linea base. OJO: el barrido se hizo sobre los
    # mismos videos que se reportan, no es CV anidada. Y es una propiedad DEL
    # CHECKPOINT: otra semilla cambia la escala de probabilidades (21_calibrar_umbral.py
    # lo escribe en el .pt y checkpoint.aplicar_config lo pisa aca).
    umbral: float = 0.7
    persistencia: float = 1.0       # segundos que la prob debe sostenerse para disparar
    # cuanto tolero sin deteccion antes de resetear el buffer. MediaPipe pierde a la
    # persona con frecuencia cuando esta en el piso (37% de las caidas de Le2i s0
    # tienen un hueco > 0.6s dentro del evento); con 0.6s el reset mataba la racha
    # en medio de la caida. 2.0s: +10.7 puntos de recall a 10 fps con las mismas
    # falsas/hora (ver resultados/README.md, corridas C3_*).
    tolerancia_sin_deteccion_seg: float = 2.0

    # --- la compuerta: ver PuertaCaida en compuertas_caida.py ---
    factor_pico_velocidad: float = 2.5   # barrido: 2.5 > 3.0 en recall sin falsas extra
    ventana_post_pico_seg: float = 4.0   # cuanto despues de la firma dejo disparar
    tau_baseline_seg: float = 0.8
    warmup_baseline_seg: float = 0.4
    piso_baseline: float = 0.05
    ventana_vertical_previa_seg: float = 2.0
    caida_vertical_minima: float = 0.35
    ventana_pico_a_caida_seg: float = 1.0
    bypass_historial_corto: bool = False

    # False = alarma solo con modelo + persistencia, sin exigir la firma.
    # Sirve para medir cuanto aporta la compuerta; en produccion va True.
    exigir_pico: bool = True


@dataclass
class Estado:
    prob: float                 # ultima P(fall) calculada (0.0 si el buffer no esta lleno o hubo reset)
    alarma_nueva: bool          # True solo en la llamada en que la alarma pasa de apagada a activa
    alarma_activa: bool
    n_buffer: int               # muestras acumuladas en el buffer
    pico_reciente: bool         # hubo una firma de caida dentro de ventana_post_pico_seg
    verticalidad: float | None  # ultima verticalidad del torso medida en una deteccion real
    clasifico: bool = False     # True si en esta llamada corrio el modelo (prob es fresca)
    listo: bool = False         # el buffer ya cubre la ventana temporal completa
    cobertura: float = 0.0      # fraccion de la ventana temporal cubierta, 0..1


def crear_puerta(c):
    return PuertaCaida(
        factor_pico=c.factor_pico_velocidad, tau_baseline_seg=c.tau_baseline_seg,
        warmup_seg=c.warmup_baseline_seg, piso_baseline=c.piso_baseline,
        ventana_vertical_previa_seg=c.ventana_vertical_previa_seg,
        caida_vertical_minima=c.caida_vertical_minima,
        ventana_pico_a_caida_seg=c.ventana_pico_a_caida_seg,
        bypass_historial_corto=c.bypass_historial_corto)


class MaquinaCaida:
    def __init__(self, modelo, device, cfg: ConfigMaquina | None = None):
        self.modelo = modelo
        self.device = device
        self.cfg = cfg or ConfigMaquina()
        self.puerta = crear_puerta(self.cfg)
        self.reiniciar()

    @property
    def ventana_seg(self):
        return duracion_ventana(self.cfg.ventana, self.cfg.fps_canonica)

    def reiniciar(self):
        self.buffer = deque()                   # (t, pose (33,2)) crudos, se remuestrean al clasificar
        self.ultimo_norm = None                 # ultimo frame valido, para tapar huecos sin mirar al futuro
        self.ultima_clasif_ts = None
        self.ultima_deteccion_ts = None         # None = todavia ninguna deteccion confiable
        self.prob_actual = 0.0
        self.inicio_racha = None
        self.alarma_activa = False
        self.puerta.reiniciar()
        self.n_llamadas = 0

    def _resetear_estado(self):
        # hace rato que no hay deteccion confiable: dejo de arrastrar la ultima
        # pose y reseteo todo en vez de seguir clasificando con datos congelados
        self.buffer.clear()
        self.ultimo_norm = None
        self.ultima_clasif_ts = None
        self.inicio_racha = None
        self.alarma_activa = False
        self.prob_actual = 0.0
        self.puerta.reiniciar()

    def _pico_reciente(self, t):
        if not self.cfg.exigir_pico:
            return True
        return self.puerta.firma_reciente(t, self.cfg.ventana_post_pico_seg)

    def _armar_clip(self, t):
        c = self.cfg
        return remuestrear_clip([x[0] for x in self.buffer], [x[1] for x in self.buffer],
                                t, c.ventana, c.fps_canonica)

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
                # la velocidad para la compuerta se mide contra la ultima muestra
                # del buffer, sobre el stream crudo a proposito: interpolar
                # difumina un golpe de un frame
                anterior = self.buffer[-1][1] if self.buffer else None
                vel = kp_norm - anterior if anterior is not None else np.zeros_like(kp_norm)

                self.ultimo_norm = kp_norm
                self.buffer.append((t, kp_norm))
                # me guardo un poco mas que la ventana para tener el punto de apoyo
                # anterior al borde izquierdo al interpolar
                while len(self.buffer) > 2 and self.buffer[1][0] < t - margen_buffer(c.ventana, c.fps_canonica):
                    self.buffer.popleft()

                if deteccion_real_este_frame:
                    self.puerta.actualizar(kp_norm, vel, t)

        cob = cobertura([x[0] for x in self.buffer], c.ventana, c.fps_canonica)
        listo = cob >= 1.0

        alarma_nueva = False
        clasifico = False
        toca_clasificar = (self.ultima_clasif_ts is None
                           or (t - self.ultima_clasif_ts) >= c.paso_clasif_seg)
        if listo and toca_clasificar:
            clasifico = True
            self.ultima_clasif_ts = t
            clip = torch.tensor(self._armar_clip(t), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                self.prob_actual = torch.softmax(self.modelo(clip), 1)[0, 1].item()

            if self.prob_actual >= c.umbral:
                if self.inicio_racha is None:
                    self.inicio_racha = t
                elif not self.alarma_activa and (t - self.inicio_racha) >= c.persistencia:
                    # sin firma de caida reciente no disparo: es alguien quieto en
                    # el piso desde hace rato, no alguien que se acaba de caer
                    if self._pico_reciente(t):
                        self.alarma_activa = True
                        alarma_nueva = True
            else:
                self.inicio_racha = None
                self.alarma_activa = False

        self.n_llamadas += 1
        return Estado(
            listo=listo,
            cobertura=cob,
            prob=self.prob_actual,
            alarma_nueva=alarma_nueva,
            alarma_activa=self.alarma_activa,
            n_buffer=len(self.buffer),
            pico_reciente=self._pico_reciente(t),
            verticalidad=self.puerta.ultima_verticalidad,
            clasifico=clasifico,
        )
