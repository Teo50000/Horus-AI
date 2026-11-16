# -*- coding: utf-8 -*-
"""
guarda_armas.py · HORUS — filtro de falsos positivos de pistola y cuchillo.

16/11 — por qué existe
----------------------
"Confunde las luces (focos del techo, reflejos, brillos) con cuchillos. Se ve
con segundos de testing." Medido en el aula: apenas se abre el panel salta
"Arma" y en `/estado` aparece `cuchillo 0.515` sin ningún cuchillo en cuadro.

La causa es de los dos detectores:

  - YOLO11n de COCO tiene AP ~0,2 en `knife`. Un tubo fluorescente o el borde
    brillante de un plafón es exactamente lo que COCO llama cuchillo: algo
    alargado, metálico y con un brillo especular.
  - La cabeza propia (y el YOLO de Horus) vieron pistolas en fotos de catálogo
    sobre fondo blanco. Un manchón quemado de luz con borde oscuro se le
    parece más que una pistola real en CCTV.

Subir el umbral solo no alcanza: los brillos llegan a 0,5-0,8. Lo que separa
una luz de un arma no es la confianza, es la FOTOMETRÍA y el CONTEXTO, y eso
es lo que mira esta guarda, sin tocar los pesos:

1. **Brillo especular.** Una fuente de luz o su reflejo es un área quemada:
   muchos píxeles saturados, poco color, y mucho más clara que su entorno.
   Un cuchillo en una mano tiene, como mucho, un filo con un brillo fino: la
   caja incluye mango y mano, y el % saturado queda bajo.
2. **Portador.** Un arma que importa la tiene alguien. Sin ninguna persona
   cerca, un cuchillo no pasa; una pistola solo si el detector está muy
   seguro (puede asomar el brazo y no el cuerpo).
3. **Geometría.** Un "cuchillo" que ocupa un octavo del cuadro no es un
   cuchillo visto por una cámara de vigilancia.
4. **Puntos fijos aprendidos.** Las luces no se mueven. Si en el mismo lugar
   del cuadro aparece una y otra vez un "arma" que no se desplaza, ese lugar
   se marca como fijo y lo que salga ahí se descarta (con olvido lento, para
   que un arma real que entra después no quede ciega para siempre). Es lo
   que agarra los reflejos que no llegan a estar saturados.

Lo descartado NO desaparece del diagnóstico: queda en `self.descartes` con el
motivo, y con HORUS_DEBUG_ARMAS=<carpeta> se guarda el recorte en disco para
poder revisar qué se tiró y por qué.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

CLASES_ARMA: Tuple[str, ...] = ("pistola", "cuchillo")


@dataclass
class ConfigGuardaArmas:
    clases: Tuple[str, ...] = CLASES_ARMA

    # --- 1. brillo especular -------------------------------------------
    saturado: int = 235            # nivel (0-255) que cuenta como quemado
    frac_saturada_max: float = 0.18
    brillo_medio_max: float = 195.0
    # Más clara que su entorno Y con núcleo quemado = fuente de luz.
    contraste_entorno_max: float = 45.0
    frac_saturada_con_contraste: float = 0.06
    # Poco color: las luces y sus reflejos son blancos/grises.
    croma_luz_max: float = 18.0

    # --- 2. portador -----------------------------------------------------
    exigir_portador: Dict[str, float] = field(default_factory=lambda: {
        # clase -> score con el que se acepta SIN persona cerca
        "cuchillo": 1.01,          # nunca sin portador
        "pistola": 0.70,
    })
    margen_portador: float = 0.20  # cuánto se agranda la caja de la persona

    # --- 3. geometría ----------------------------------------------------
    area_max: Dict[str, float] = field(default_factory=lambda: {
        "cuchillo": 0.10, "pistola": 0.15})
    lado_min_px: float = 6.0

    # Franja de luz: lo quemado recorre casi todo el largo de la caja (un
    # tubo, el reflejo de un tubo en una mesa). En un cuchillo el brillo es
    # solo el filo: el mango y la mano son oscuros.
    franja_largo_min: float = 0.75
    franja_frac_min: float = 0.06

    # --- 4. puntos fijos ---------------------------------------------------
    # Cuenta SEGUNDOS de presencia quieta cuadro a cuadro, no detecciones:
    # alguien apuntando quieto 3 s no puede quedar silenciado. Una luz que
    # está ahí toda la noche, sí.
    celdas: Tuple[int, int] = (32, 18)  # grilla (x, y) del cuadro
    fijo_umbral_s: float = 20.0    # segundos quieto en el mismo lugar
    fijo_olvido_s: float = 600.0   # vida media de lo acumulado
    fijo_iou_quieto: float = 0.60  # contra el cuadro anterior = no se movió
    fijo_peso_luz: float = 4.0     # lo que ya cayó por fotometría acumula 4x

    debug_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("HORUS_DEBUG_ARMAS") or None)


def _label(d: Any) -> str:
    return getattr(d, "label", None) or getattr(d, "clase", "?")


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class _MapaFijos:
    """Segundos de presencia quieta por celda del cuadro, con olvido."""

    def __init__(self, cfg: ConfigGuardaArmas) -> None:
        self.cfg = cfg
        nx, ny = cfg.celdas
        self.seg = np.zeros((ny, nx), dtype=np.float32)
        self.ts = time.time()
        self._prev: List[Tuple[Tuple[float, ...], float]] = []
        self._actual: List[Tuple[Tuple[float, ...], float]] = []

    def _decaer(self, ts: float) -> None:
        dt = ts - self.ts
        if dt > 0:
            self.seg *= float(0.5 ** (dt / self.cfg.fijo_olvido_s))
        self.ts = ts

    def _celdas(self, caja, ancho: int, alto: int):
        nx, ny = self.cfg.celdas
        x1, y1, x2, y2 = caja
        cx0 = int(np.clip(x1 / ancho * nx, 0, nx - 1))
        cx1 = int(np.clip(x2 / ancho * nx, 0, nx - 1))
        cy0 = int(np.clip(y1 / alto * ny, 0, ny - 1))
        cy1 = int(np.clip(y2 / alto * ny, 0, ny - 1))
        return cx0, cx1, cy0, cy1

    def nuevo_cuadro(self, ts: float) -> None:
        self._decaer(ts)
        if self._actual:
            self._prev = self._actual
        self._actual = []

    def es_fijo(self, caja, ancho: int, alto: int) -> bool:
        cx0, cx1, cy0, cy1 = self._celdas(caja, ancho, alto)
        zona = self.seg[cy0:cy1 + 1, cx0:cx1 + 1]
        return bool(zona.size and zona.mean() >= self.cfg.fijo_umbral_s)

    def votar(self, caja, ancho: int, alto: int, ts: float,
              peso: float = 1.0) -> None:
        """Suma el tiempo transcurrido SOLO si en el cuadro anterior había
        una detección de arma casi idéntica (no se movió)."""
        caja = tuple(float(v) for v in caja)
        self._actual.append((caja, ts))
        mejor, dt = 0.0, 0.0
        for c, t in self._prev:
            if 0.0 < ts - t <= 1.5:
                i = _iou(caja, c)
                if i > mejor:
                    mejor, dt = i, ts - t
        if mejor >= self.cfg.fijo_iou_quieto:
            cx0, cx1, cy0, cy1 = self._celdas(caja, ancho, alto)
            self.seg[cy0:cy1 + 1, cx0:cx1 + 1] += peso * min(dt, 1.0)


class GuardaArmas:
    """Uso:
        guarda = GuardaArmas()
        dets = guarda.filtrar("cam-1", frame_bgr, dets)
    Solo toca las clases de arma; el resto pasa intacto y en el mismo orden.
    """

    def __init__(self, cfg: Optional[ConfigGuardaArmas] = None) -> None:
        self.cfg = cfg or ConfigGuardaArmas()
        self._fijos: Dict[str, _MapaFijos] = {}
        self.descartes: Dict[str, int] = {}
        self.aceptadas = 0
        self._n_debug = 0
        if self.cfg.debug_dir:
            os.makedirs(self.cfg.debug_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def filtrar(self, cam: str, frame: Optional[np.ndarray],
                dets: Sequence[Any], ts: Optional[float] = None) -> List[Any]:
        dets = list(dets or ())
        if not dets or not any(_label(d) in self.cfg.clases for d in dets):
            return dets
        ts = time.time() if ts is None else float(ts)
        img = None if frame is None else np.asarray(frame)
        alto, ancho = (img.shape[:2] if img is not None else (720, 1280))
        fijos = self._fijos.setdefault(cam, _MapaFijos(self.cfg))
        fijos.nuevo_cuadro(ts)
        personas = [d.bbox_xyxy for d in dets if _label(d) == "persona"
                    and float(getattr(d, "score", 0)) >= 0.30]

        salida: List[Any] = []
        for d in dets:
            lbl = _label(d)
            if lbl not in self.cfg.clases:
                salida.append(d)
                continue
            motivo = self._motivo(lbl, d, img, ancho, alto, personas, fijos, ts)
            if motivo is None:
                self.aceptadas += 1
                salida.append(d)
            else:
                self.descartes[motivo] = self.descartes.get(motivo, 0) + 1
            # Toda detección de arma vota: una luz que el filtro fotométrico
            # ya atrapa también termina marcando su lugar, y así se cubren
            # los cuadros en que el brillo baja (cámara que ajusta exposición).
            fijos.votar(d.bbox_xyxy, ancho, alto, ts,
                        peso=(self.cfg.fijo_peso_luz
                              if motivo and motivo.startswith("luz") else 1.0))
            self._volcar(cam, img, d, motivo)
        return salida

    # ------------------------------------------------------------------ #
    def _motivo(self, lbl, d, img, ancho, alto, personas, fijos, ts) -> Optional[str]:
        c = self.cfg
        x1, y1, x2, y2 = (float(v) for v in d.bbox_xyxy)
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        score = float(getattr(d, "score", 0.0))

        # 3. geometría
        if w < c.lado_min_px or h < c.lado_min_px:
            return "muy_chica"
        if (w * h) / float(max(ancho * alto, 1)) > c.area_max.get(lbl, 1.0):
            return "muy_grande"

        # 1. fotometría
        if img is not None:
            m = self._fotometria(img, x1, y1, x2, y2)
            if m is not None:
                frac_sat, media, contraste, croma, largo = m
                if frac_sat >= c.frac_saturada_max:
                    return "luz_saturada"
                if frac_sat >= c.franja_frac_min and largo >= c.franja_largo_min:
                    return "luz_franja"
                if media >= c.brillo_medio_max and croma <= c.croma_luz_max:
                    return "luz_brillo"
                if (contraste >= c.contraste_entorno_max
                        and frac_sat >= c.frac_saturada_con_contraste):
                    return "luz_reflejo"

        # 4. lugar fijo de la escena
        if fijos.es_fijo((x1, y1, x2, y2), ancho, alto):
            return "punto_fijo"

        # 2. portador
        tope = c.exigir_portador.get(lbl)
        if tope is not None and score < tope:
            if not any(self._cerca(p, (x1, y1, x2, y2)) for p in personas):
                return "sin_portador"
        return None

    def _cerca(self, persona, arma) -> bool:
        px1, py1, px2, py2 = persona
        mw = (px2 - px1) * self.cfg.margen_portador
        mh = (py2 - py1) * self.cfg.margen_portador * 0.5
        cx, cy = (arma[0] + arma[2]) / 2.0, (arma[1] + arma[3]) / 2.0
        return (px1 - mw) <= cx <= (px2 + mw) and (py1 - mh) <= cy <= (py2 + mh)

    def _fotometria(self, img: np.ndarray, x1: float, y1: float, x2: float,
                    y2: float) -> Optional[Tuple[float, ...]]:
        """(fracción saturada, brillo medio, brillo - entorno, croma media,
        fracción del lado largo que recorre lo saturado)."""
        H, W = img.shape[:2]
        a, b = int(max(0, x1)), int(max(0, y1))
        c, d = int(min(W, x2)), int(min(H, y2))
        if c - a < 2 or d - b < 2:
            return None
        rec = img[b:d, a:c].astype(np.float32)
        if rec.ndim == 2:
            gris = rec
            croma = 0.0
        else:
            gris = rec.mean(axis=2)
            croma = float((rec.max(axis=2) - rec.min(axis=2)).mean())
        # el canal más alto: una luz amarilla satura R y G, no la media
        pico = rec.max(axis=2) if rec.ndim == 3 else rec
        quemado = pico >= self.cfg.saturado
        frac_sat = float(quemado.mean())
        media = float(gris.mean())
        # lo quemado proyectado sobre el lado largo de la caja
        eje = 0 if quemado.shape[1] >= quemado.shape[0] else 1
        perfil = quemado.any(axis=eje)
        largo = float(perfil.mean()) if perfil.size else 0.0

        # entorno: un anillo de medio lado alrededor de la caja
        mw, mh = (c - a) // 2 + 2, (d - b) // 2 + 2
        A, B = max(0, a - mw), max(0, b - mh)
        C, D = min(W, c + mw), min(H, d + mh)
        ext = img[B:D, A:C].astype(np.float32)
        ext_g = ext.mean(axis=2) if ext.ndim == 3 else ext
        n_ext = ext_g.size - gris.size
        if n_ext > 16:
            entorno = (float(ext_g.sum()) - float(gris.sum())) / n_ext
        else:
            entorno = media
        return frac_sat, media, media - entorno, croma, largo

    # ------------------------------------------------------------------ #
    def _volcar(self, cam, img, d, motivo) -> None:
        if not self.cfg.debug_dir or img is None or self._n_debug > 4000:
            return
        try:
            import cv2
            x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
            H, W = img.shape[:2]
            m = 12
            rec = img[max(0, y1 - m):min(H, y2 + m), max(0, x1 - m):min(W, x2 + m)]
            if rec.size == 0:
                return
            self._n_debug += 1
            nombre = (f"{int(time.time() * 1000)}_{cam}_{_label(d)}_"
                      f"{float(d.score):.2f}_{motivo or 'OK'}.jpg")
            cv2.imwrite(os.path.join(self.cfg.debug_dir, nombre), rec)
        except Exception:                                # noqa: BLE001
            pass

    def resumen(self) -> Dict[str, Any]:
        return {"aceptadas": self.aceptadas, "descartadas": dict(self.descartes)}
