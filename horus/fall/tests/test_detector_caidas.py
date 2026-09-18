"""Tests de comportamiento de DetectorCaidas sobre streams sinteticos.

Historia: la primera version de este archivo era un "golden test" que exigia
igualdad exacta con el loop viejo de 12_webcam.py, y sirvio para validar el
refactor (fase 1). Al arreglar el baseline de velocidad (que estaba congelado
en ~0 y hacia que todo fuera pico) la referencia vieja dejo de ser la verdad,
asi que ahora los escenarios declaran lo que DEBE pasar. La regresion real del
sistema completo la da 17_evaluar_evento.py sobre los streams de Le2i.

Correr:  python tests/test_detector_caidas.py   (desde horus/fall/)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from detector_caidas import DetectorCaidas, ConfigDetector  # noqa: E402
from preproceso import IDX_CI, IDX_CD, IDX_HI, IDX_HD        # noqa: E402

FPS = 25


class ModeloFalso(nn.Module):
    """P(fall) alta cuando los hombros estan bajos (cerca de la cadera) en
    promedio a lo largo de la ventana. Determinista."""
    def forward(self, clip):  # clip: (1, T, 33, 4)
        y_hombros = clip[0, :, [IDX_HI, IDX_HD], 1].mean()
        logit_fall = 6.0 * (y_hombros + 0.5)
        return torch.stack([torch.zeros_like(logit_fall), logit_fall]).unsqueeze(0)


# Poses sinteticas normalizadas (33,2): cadera en el origen, escala = 1.
# De pie: hombros en y=-1 (verticalidad ~1). Acostado: hombros en x=-1, y~0.
def pose(estado, rng):
    kp = np.zeros((33, 2))
    kp[IDX_CI] = (-0.15, 0.0); kp[IDX_CD] = (0.15, 0.0)
    if estado == "de_pie":
        kp[IDX_HI] = (-0.2, -1.0); kp[IDX_HD] = (0.2, -1.0)
        kp[0] = (0.0, -1.3)
        kp[27] = (-0.15, 1.2); kp[28] = (0.15, 1.2)
    elif estado == "acostado":
        kp[IDX_HI] = (-1.0, -0.05); kp[IDX_HD] = (-1.0, 0.05)
        kp[0] = (-1.3, 0.0)
        kp[27] = (1.2, -0.05); kp[28] = (1.2, 0.05)
    else:
        raise ValueError(estado)
    # jitter chico y determinista, como el que mete MediaPipe en una persona quieta
    return kp + rng.normal(0, 0.004, kp.shape)


def tramo(estado, seg, rng):
    return [pose(estado, rng) for _ in range(int(seg * FPS))]


def transicion(desde, hasta, seg, rng):
    a, b = pose(desde, rng), pose(hasta, rng)
    n = max(int(seg * FPS), 1)
    return [a + (b - a) * (i + 1) / n + rng.normal(0, 0.004, a.shape) for i in range(n)]


def con_tiempos(frames):
    return [(f, i / FPS) for i, f in enumerate(frames)]


def escenario_caida():
    """De pie 3s -> caida 0.2s -> quieto 3s -> se levanta 1.5s -> de pie 1s
    -> sin deteccion 1s (> tolerancia, resetea) -> de pie 2s.  => 1 alarma."""
    rng = np.random.default_rng(0)
    frames = (tramo("de_pie", 3, rng) + transicion("de_pie", "acostado", 0.2, rng)
              + tramo("acostado", 3, rng) + transicion("acostado", "de_pie", 1.5, rng)
              + tramo("de_pie", 1, rng))
    frames += [None] * int(1.0 * FPS)
    frames += tramo("de_pie", 2, rng)
    return con_tiempos(frames)


def escenario_ya_acostado():
    """Acostado desde el arranque 6s -> se acomoda 0.3s -> quieto 3s.
    P(fall) alta todo el tiempo, pero nunca hubo transicion de pie -> piso.
    => 0 alarmas. (Con el baseline congelado esto disparaba a los 2.4s.)"""
    rng = np.random.default_rng(1)
    a = pose("acostado", rng)
    b = a.copy(); b[:, 0] += 0.25
    n = int(0.3 * FPS)
    acomodo = [a + (b - a) * (i + 1) / n for i in range(n)]
    frames = tramo("acostado", 6, rng) + acomodo + [b + rng.normal(0, 0.004, b.shape) for _ in range(3 * FPS)]
    return con_tiempos(frames)


def escenario_parpadeo():
    """De pie 3s -> 1s alternando deteccion/None cada frame (huecos de 1 frame,
    bajo la tolerancia: repite el ultimo valido) -> caida 0.2s -> quieto 3s.  => 1 alarma."""
    rng = np.random.default_rng(2)
    frames = tramo("de_pie", 3, rng)
    for i, f in enumerate(tramo("de_pie", 1, rng)):
        frames.append(None if i % 2 else f)
    frames += transicion("de_pie", "acostado", 0.2, rng) + tramo("acostado", 3, rng)
    return con_tiempos(frames)


def escenario_acostarse_despacio():
    """De pie 3s -> se acuesta en 3.5s (controlado, como irse a dormir) -> quieto 4s.
    Termina en el piso con P(fall) alta, pero sin movimiento brusco.  => 0 alarmas.
    Es lo que el gate de pico existe para filtrar."""
    rng = np.random.default_rng(3)
    frames = tramo("de_pie", 3, rng) + transicion("de_pie", "acostado", 3.5, rng) + tramo("acostado", 4, rng)
    return con_tiempos(frames)


def escenario_sale_y_vuelve_de_pie():
    """De pie 3s -> sale de cuadro 1.5s -> vuelve de pie 3s.  => 0 alarmas."""
    rng = np.random.default_rng(4)
    frames = tramo("de_pie", 3, rng) + [None] * int(1.5 * FPS) + tramo("de_pie", 3, rng)
    return con_tiempos(frames)


def correr(stream, cfg=None):
    det = DetectorCaidas(ModeloFalso().eval(), "cpu", cfg or ConfigDetector())
    alarmas, prob_max = [], 0.0
    for kp, t in stream:
        e = det.actualizar(kp, t)
        prob_max = max(prob_max, e.prob)
        if e.alarma_nueva:
            alarmas.append(round(t, 2))
    return alarmas, prob_max


def chequear(nombre, stream, esperadas, prob_alta_esperada=None):
    alarmas, prob_max = correr(stream)
    assert len(alarmas) == esperadas, f"[{nombre}] esperaba {esperadas} alarma(s), dio {alarmas}"
    if prob_alta_esperada is not None:
        assert (prob_max >= 0.5) == prob_alta_esperada, \
            f"[{nombre}] P(fall) max={prob_max:.2f}, esperaba {'alta' if prob_alta_esperada else 'baja'}"
    print(f"  {nombre}: OK  ({len(stream)} frames, alarmas={alarmas}, P max={prob_max:.2f})")


def test_caida_normal():
    chequear("caida normal + reset por hueco", escenario_caida(), esperadas=1)


def test_ya_acostado_no_dispara():
    chequear("ya acostado, sin transicion", escenario_ya_acostado(), esperadas=0, prob_alta_esperada=True)


def test_parpadeo_de_deteccion():
    chequear("parpadeo de deteccion + caida", escenario_parpadeo(), esperadas=1)


def test_acostarse_despacio_no_dispara():
    chequear("acostarse despacio", escenario_acostarse_despacio(), esperadas=0, prob_alta_esperada=True)


def test_sale_y_vuelve_de_pie():
    chequear("sale de cuadro y vuelve de pie", escenario_sale_y_vuelve_de_pie(), esperadas=0, prob_alta_esperada=False)


def test_sin_pico_dispara_en_acostado():
    """Con exigir_pico=False el detector es solo modelo + persistencia: el
    escenario 'ya acostado' SI tiene que disparar. Verifica que el gate es lo
    que lo frena, no otra cosa."""
    alarmas, _ = correr(escenario_ya_acostado(), ConfigDetector(exigir_pico=False))
    assert len(alarmas) == 1, f"[sin pico] esperaba 1 alarma, dio {alarmas}"
    print(f"  sin gate de pico, ya acostado dispara: OK (alarmas={alarmas})")


if __name__ == "__main__":
    print("Tests de comportamiento DetectorCaidas")
    test_caida_normal()
    test_ya_acostado_no_dispara()
    test_parpadeo_de_deteccion()
    test_acostarse_despacio_no_dispara()
    test_sale_y_vuelve_de_pie()
    test_sin_pico_dispara_en_acostado()
    print("TODO OK")
