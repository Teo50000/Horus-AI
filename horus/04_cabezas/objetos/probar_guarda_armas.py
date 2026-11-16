# -*- coding: utf-8 -*-
"""Autotest de guarda_armas.py — sin GPU, sin cámara, sin pesos.

    python probar_guarda_armas.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guarda_armas import ConfigGuardaArmas, GuardaArmas  # noqa: E402


class D:
    def __init__(self, caja, score, label):
        self.bbox_xyxy = tuple(float(v) for v in caja)
        self.score = score
        self.label = label


def escena():
    """Un aula: techo claro arriba con un tubo fluorescente quemado, una
    persona en el medio, piso gris."""
    img = np.full((720, 1280, 3), 120, np.uint8)
    img[:250] = (205, 205, 200)                    # techo claro
    img[80:110, 700:980] = 255                     # tubo de luz quemado
    img[640:644, 60:380] = 250                     # reflejo fino en una mesa
    img[636:648, 60:380] = np.maximum(img[636:648, 60:380], 200)
    img[200:700, 450:750] = (60, 70, 90)           # persona (ropa oscura)
    img[480:500, 690:770] = (150, 150, 155)        # cuchillo en la mano: hoja
    img[485:495, 692:740] = (240, 240, 240)        # con un filo brillante fino
    img[482:498, 760:790] = (40, 25, 20)           # mango
    return img


def correr() -> int:
    img = escena()
    persona = D((450, 200, 750, 700), 0.90, "persona")
    casos = [
        ("tubo de luz como cuchillo",  D((700, 80, 980, 110), 0.62, "cuchillo"), False),
        ("tubo de luz como pistola",   D((690, 70, 990, 120), 0.75, "pistola"), False),
        ("reflejo en la mesa",         D((60, 634, 380, 650), 0.58, "cuchillo"), False),
        ("cuchillo en la mano",        D((685, 476, 795, 502), 0.66, "cuchillo"), True),
        ("cuchillo sin nadie cerca",   D((60, 560, 160, 590), 0.70, "cuchillo"), False),
        ("pistola sin portador, dudosa", D((60, 560, 120, 610), 0.55, "pistola"), False),
        ("pistola sin portador, segura", D((60, 560, 120, 610), 0.85, "pistola"), True),
        ("cuchillo gigante",           D((100, 250, 700, 600), 0.60, "cuchillo"), False),
    ]
    fallas = 0
    for nombre, d, pasa in casos:
        g = GuardaArmas(ConfigGuardaArmas(debug_dir=None))
        out = g.filtrar("cam-0", img, [persona, d], ts=0.0)
        paso = d in out
        ok = paso == pasa
        fallas += not ok
        print(f"  {'OK   ' if ok else 'FALLA'}  {nombre:32s} "
              f"{'pasa' if paso else 'descartada'}  {g.resumen()['descartadas']}")

    # Punto fijo: algo gris (no saturado, la fotometría no lo agarra) que
    # aparece siempre en el mismo lugar junto a la persona termina marcado.
    g = GuardaArmas(ConfigGuardaArmas(debug_dir=None))
    fijo = D((600, 300, 640, 330), 0.70, "cuchillo")
    pasaron = []
    for t in range(120):                       # 36 s a 3,3 ticks/s
        out = g.filtrar("cam-0", img, [persona, fijo], ts=t * 0.3)
        pasaron.append(fijo in out)
    corte = pasaron.index(False) if False in pasaron else None
    # quieto 3 s NO alcanza (alguien apuntando); 20 s sí
    ok = corte is not None and corte * 0.3 >= 15.0 and not pasaron[-1]
    fallas += not ok
    print(f"  {'OK   ' if ok else 'FALLA'}  {'punto fijo aprendido':32s} "
          f"se descarta desde los {corte * 0.3 if corte else '-'} s quieto")

    # Y un arma que SE MUEVE en el mismo lugar no se marca como fija.
    g = GuardaArmas(ConfigGuardaArmas(debug_dir=None))
    pasaron = []
    for t in range(30):
        dx = (t % 6) * 25
        movil = D((560 + dx, 300 + dx // 2, 600 + dx, 330 + dx // 2), 0.70, "cuchillo")
        pasaron.append(movil in g.filtrar("cam-0", img, [persona, movil], ts=t * 0.3))
    ok = all(pasaron)
    fallas += not ok
    print(f"  {'OK   ' if ok else 'FALLA'}  {'arma que se mueve no es fija':32s} pasó {sum(pasaron)}/30")

    # Las demás clases pasan intactas y en orden.
    g = GuardaArmas()
    otros = [D((1, 1, 50, 50), 0.3, "humo"), persona, D((5, 5, 9, 9), 0.9, "celular")]
    ok = g.filtrar("cam-0", img, otros) == otros
    fallas += not ok
    print(f"  {'OK   ' if ok else 'FALLA'}  {'otras clases intactas':32s}")

    print("-" * 70)
    print(f"{'todo OK' if not fallas else f'{fallas} FALLA(S)'}")
    return 1 if fallas else 0


if __name__ == "__main__":
    raise SystemExit(correr())
