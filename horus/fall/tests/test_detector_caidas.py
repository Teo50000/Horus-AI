"""Golden test: DetectorCaidas tiene que producir EXACTAMENTE lo mismo que el
loop viejo de 12_webcam.py (copiado abajo, congelado). No testea que la logica
sea "correcta" — testea que el refactor no la cambio. Sin esto, la linea base
del harness offline no se puede comparar con lo que corria en vivo.

Correr:  python tests/test_detector_caidas.py   (desde horus/fall/)
"""
import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from detector_caidas import DetectorCaidas, ConfigDetector  # noqa: E402
from preproceso import IDX_CI, IDX_CD, IDX_HI, IDX_HD        # noqa: E402

FPS = 25


# ---------------------------------------------------------------------------
# Referencia: el loop de 12_webcam.py (lineas 211-309 al momento del refactor),
# transcripto literalmente con las constantes como parametros y un solo reloj.
# NO TOCAR: si cambia la logica, cambia en DetectorCaidas y se actualiza la
# referencia a conciencia, no al reves.
# ---------------------------------------------------------------------------
def referencia_loop_viejo(stream, modelo, device, cfg: ConfigDetector):
    VENTANA, PASO = cfg.ventana, cfg.paso
    UMBRAL, PERSISTENCIA = cfg.umbral, cfg.persistencia
    TOLERANCIA_SIN_DETECCION_SEG = cfg.tolerancia_sin_deteccion_seg
    FACTOR_PICO_VELOCIDAD = cfg.factor_pico_velocidad
    VENTANA_POST_PICO_SEG = cfg.ventana_post_pico_seg
    ALPHA_BASELINE = cfg.alpha_baseline
    VENTANA_VERTICAL_PREVIA_SEG = cfg.ventana_vertical_previa_seg
    CAIDA_VERTICAL_MINIMA = cfg.caida_vertical_minima

    buffer = deque(maxlen=VENTANA)
    ultimo_norm = None
    anterior_norm = None
    ultima_deteccion_ts = None
    prob_actual = 0.0
    inicio_racha = None
    alarma_activa = False
    alarmas = []
    baseline_vel = None
    ultimo_pico_ts = None
    historial_vertical = deque()
    n_frame = 0

    probs, activas = [], []
    for kp_norm, t_frame in stream:
        if kp_norm is not None:
            ultima_deteccion_ts = t_frame
        sin_deteccion_seg = (t_frame - ultima_deteccion_ts) if ultima_deteccion_ts is not None else None

        if sin_deteccion_seg is None or sin_deteccion_seg > TOLERANCIA_SIN_DETECCION_SEG:
            buffer.clear()
            ultimo_norm = None
            anterior_norm = None
            inicio_racha = None
            alarma_activa = False
            prob_actual = 0.0
            historial_vertical.clear()
            baseline_vel = None
            ultimo_pico_ts = None
        else:
            deteccion_real_este_frame = kp_norm is not None
            if kp_norm is None:
                kp_norm = ultimo_norm

            if kp_norm is not None:
                ultimo_norm = kp_norm
                vel = kp_norm - anterior_norm if anterior_norm is not None else np.zeros_like(kp_norm)
                anterior_norm = kp_norm
                buffer.append(np.concatenate([kp_norm, vel], axis=1))

                if deteccion_real_este_frame:
                    v_escalar = np.linalg.norm(vel[[IDX_CI, IDX_CD, IDX_HI, IDX_HD]], axis=1).mean()
                    verticalidad = -((kp_norm[IDX_HI, 1] + kp_norm[IDX_HD, 1]) / 2)

                    if baseline_vel is None:
                        baseline_vel = v_escalar
                    elif baseline_vel > 1e-6 and v_escalar > baseline_vel * FACTOR_PICO_VELOCIDAD:
                        cobertura_seg = t_frame - historial_vertical[0][0] if historial_vertical else 0.0
                        if cobertura_seg < VENTANA_VERTICAL_PREVIA_SEG * 0.8:
                            ultimo_pico_ts = t_frame
                        else:
                            vertical_previa_max = max(v for _, v in historial_vertical)
                            if (vertical_previa_max - verticalidad) >= CAIDA_VERTICAL_MINIMA:
                                ultimo_pico_ts = t_frame
                    else:
                        baseline_vel = baseline_vel * (1 - ALPHA_BASELINE) + v_escalar * ALPHA_BASELINE

                    historial_vertical.append((t_frame, verticalidad))
                    while historial_vertical and t_frame - historial_vertical[0][0] > VENTANA_VERTICAL_PREVIA_SEG:
                        historial_vertical.popleft()

        if len(buffer) == VENTANA and n_frame % PASO == 0:
            clip = torch.tensor(np.array(buffer), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                prob_actual = torch.softmax(modelo(clip), 1)[0, 1].item()

            t_seg = t_frame
            if prob_actual >= UMBRAL:
                if inicio_racha is None:
                    inicio_racha = t_seg
                elif not alarma_activa and (t_seg - inicio_racha) >= PERSISTENCIA:
                    hubo_movimiento_brusco = (
                        ultimo_pico_ts is not None
                        and (t_frame - ultimo_pico_ts) <= VENTANA_POST_PICO_SEG
                    )
                    if hubo_movimiento_brusco:
                        alarma_activa = True
                        alarmas.append(t_seg)
            else:
                inicio_racha = None
                alarma_activa = False

        probs.append(prob_actual)
        activas.append(alarma_activa)
        n_frame += 1

    return probs, activas, alarmas


# ---------------------------------------------------------------------------
# Modelo falso determinista: P(fall) alta cuando los hombros estan bajos
# (cerca de la cadera) en promedio a lo largo de la ventana.
# ---------------------------------------------------------------------------
class ModeloFalso(nn.Module):
    def forward(self, clip):  # clip: (1, T, 33, 4)
        y_hombros = clip[0, :, [IDX_HI, IDX_HD], 1].mean()
        logit_fall = 6.0 * (y_hombros + 0.5)
        return torch.stack([torch.zeros_like(logit_fall), logit_fall]).unsqueeze(0)


# ---------------------------------------------------------------------------
# Poses sinteticas normalizadas (33,2): cadera en el origen, escala = 1.
# De pie: hombros en y=-1 (verticalidad ~1). Acostado: hombros en x=-1, y~0.
# ---------------------------------------------------------------------------
def pose(estado, rng):
    kp = np.zeros((33, 2))
    kp[IDX_CI] = (-0.15, 0.0); kp[IDX_CD] = (0.15, 0.0)
    if estado == "de_pie":
        kp[IDX_HI] = (-0.2, -1.0); kp[IDX_HD] = (0.2, -1.0)
        kp[0] = (0.0, -1.3)           # nariz
        kp[27] = (-0.15, 1.2); kp[28] = (0.15, 1.2)  # tobillos
    elif estado == "acostado":
        kp[IDX_HI] = (-1.0, -0.05); kp[IDX_HD] = (-1.0, 0.05)
        kp[0] = (-1.3, 0.0)
        kp[27] = (1.2, -0.05); kp[28] = (1.2, 0.05)
    else:
        raise ValueError(estado)
    # jitter chico y determinista: sin el, baseline_vel queda exactamente 0
    # y la regla del pico (ratio) no se ejercita de forma realista
    return kp + rng.normal(0, 0.004, kp.shape)


def tramo(estado, seg, rng):
    return [pose(estado, rng) for _ in range(int(seg * FPS))]


def transicion(desde, hasta, seg, rng):
    a, b = pose(desde, rng), pose(hasta, rng)
    n = max(int(seg * FPS), 1)
    return [a + (b - a) * (i + 1) / n for i in range(n)]


def con_tiempos(frames, t0=0.0):
    return [(f, t0 + i / FPS) for i, f in enumerate(frames)]


def escenario_caida():
    """De pie 3s -> caida 0.2s -> quieto 3s -> se levanta 1.5s -> de pie 1s
    -> sin deteccion 1s (> tolerancia, resetea) -> de pie 2s."""
    rng = np.random.default_rng(0)
    frames = (tramo("de_pie", 3, rng) + transicion("de_pie", "acostado", 0.2, rng)
              + tramo("acostado", 3, rng) + transicion("acostado", "de_pie", 1.5, rng)
              + tramo("de_pie", 1, rng))
    frames = [(f, i / FPS) for i, f in enumerate(frames)]
    t = frames[-1][1]
    frames += [(None, t + (i + 1) / FPS) for i in range(int(1.0 * FPS))]
    t = frames[-1][1]
    frames += [(f, t + (i + 1) / FPS) for i, f in enumerate(tramo("de_pie", 2, rng))]
    return frames


def escenario_ya_acostado():
    """Acostado desde el arranque 6s -> se acomoda 0.3s -> quieto 3s.
    P(fall) es alta todo el tiempo pero no hubo transicion de pie->piso."""
    rng = np.random.default_rng(1)
    a = pose("acostado", rng)
    b = a.copy(); b[:, 0] += 0.25
    n = int(0.3 * FPS)
    acomodo = [a + (b - a) * (i + 1) / n for i in range(n)]
    frames = tramo("acostado", 6, rng) + acomodo + [b + rng.normal(0, 0.004, b.shape) for _ in range(3 * FPS)]
    return con_tiempos(frames)


def escenario_parpadeo():
    """De pie 3s -> 1s alternando deteccion/None cada frame (huecos de 1 frame,
    bajo la tolerancia: repite el ultimo valido) -> caida 0.2s -> quieto 3s."""
    rng = np.random.default_rng(2)
    frames = tramo("de_pie", 3, rng)
    for i, f in enumerate(tramo("de_pie", 1, rng)):
        frames.append(None if i % 2 else f)
    frames += transicion("de_pie", "acostado", 0.2, rng) + tramo("acostado", 3, rng)
    return con_tiempos(frames)


def correr_nuevo(stream, modelo, device, cfg):
    det = DetectorCaidas(modelo, device, cfg)
    probs, activas, alarmas = [], [], []
    for kp, t in stream:
        e = det.actualizar(kp, t)
        probs.append(e.prob)
        activas.append(e.alarma_activa)
        if e.alarma_nueva:
            alarmas.append(t)
    return probs, activas, alarmas


def comparar(nombre, stream, alarmas_esperadas):
    modelo = ModeloFalso().eval()
    cfg = ConfigDetector()
    p_ref, a_ref, al_ref = referencia_loop_viejo(stream, modelo, "cpu", cfg)
    p_new, a_new, al_new = correr_nuevo(stream, modelo, "cpu", cfg)

    assert p_ref == p_new, f"[{nombre}] las probabilidades difieren"
    assert a_ref == a_new, f"[{nombre}] alarma_activa por frame difiere"
    assert al_ref == al_new, f"[{nombre}] instantes de alarma difieren: ref={al_ref} new={al_new}"
    assert len(al_new) == alarmas_esperadas, \
        f"[{nombre}] el escenario deberia producir {alarmas_esperadas} alarma(s), dio {al_new}"
    print(f"  {nombre}: OK  ({len(stream)} frames, alarmas={[round(a, 2) for a in al_new]})")


def test_caida_normal():
    comparar("caida normal + reset por hueco", escenario_caida(), alarmas_esperadas=1)


def test_ya_acostado():
    # BUG CONOCIDO de la logica original (documentado, no corregido en esta fase):
    # dispara a los ~2.4s aunque nunca hubo transicion de pie -> piso. Causa: el
    # baseline de velocidad se siembra con el primer frame (vel=0), recibe UN paso
    # de EMA (~5% de una muestra de jitter) y despues se congela, porque la rama
    # "es pico" no lo actualiza y con ese piso todo frame es pico. Durante los
    # primeros 1.6s de historial el gate de verticalidad no se chequea (excepcion
    # de "recien reaparecio"), asi que el pico espurio valida la alarma.
    # Se corrige en la fase 4 con el harness midiendo el antes/despues; ahi este
    # test pasa a esperar 0 alarmas y la referencia congelada se retira.
    comparar("ya acostado, sin transicion (bug conocido)", escenario_ya_acostado(), alarmas_esperadas=1)


def test_parpadeo_de_deteccion():
    comparar("parpadeo de deteccion + caida", escenario_parpadeo(), alarmas_esperadas=1)


if __name__ == "__main__":
    print("Golden test DetectorCaidas vs loop viejo de 12_webcam.py")
    test_caida_normal()
    test_ya_acostado()
    test_parpadeo_de_deteccion()
    print("TODO OK")
