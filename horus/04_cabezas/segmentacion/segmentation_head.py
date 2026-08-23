# -*- coding: utf-8 -*-
"""
segmentation_head.py · HORUS — cabeza de segmentacion sobre el backbone compartido.

Clases:  0 fondo · 1 agua · 2 humo · 3 fuego

Es la cabeza de "superficies/regiones" del diagrama (ej: agua). Consume la
piramide que ya calculo `SharedBackbone` -- la imagen se procesa UNA vez para
las cuatro cabezas -- y devuelve un mapa de clases del tamano del frame.

Por que Semantic-FPN y no algo mas grande
-----------------------------------------
El backbone entrega `p3/p4/p5` (strides 8/16/32). Las tres ramas suben a
stride 8, se suman, y un 1x1 clasifica. Es la receta de Panoptic-FPN: barata,
sin capas nuevas raras que TensorRT no sepa fusionar, y suficiente para
regiones grandes y difusas como agua, humo y fuego. Un ASPP o un decoder tipo
U-Net aportarian en bordes finos, que aca no es lo que decide la alerta.

La salida sale a stride 8 y se sube bilineal al tamano de entrada. No hay
nivel a stride 4 en el backbone compartido y no conviene agregarlo: cambiar
`_RETURN_NODES` en `shared_backbone.py` le mueve el piso a la cabeza de
objetos, que ya esta validada.

Decisiones que conviene recordar
--------------------------------
* **GroupNorm, no BatchNorm.** El batch efectivo por GPU es chico y las
  estadisticas de BN meten ruido justo en las clases raras (fuego), que es
  donde menos se lo banca.
* **El sesgo de salida arranca en el prior.** El fondo es ~90% de los pixeles;
  sin esto se gastan las primeras epocas descubriendo eso.
* **El argmax se queda afuera del engine TensorRT.** Misma razon por la que el
  NMS se quedo en PyTorch en objetos: el umbral de area y la resolucion de
  salida se tocan seguido y no queres reconstruir el `.plan` cada vez.
* **La perdida acepta `anotadas`** (softmax restringido). Ver `compute_loss`.

Prueba rapida:
    python segmentation_head.py
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEFAULT_CLASSES: Tuple[str, ...] = ("fondo", "agua", "humo", "fuego")

# Las que disparan alerta por si solas. El fondo obviamente no.
CRITICAL_CLASSES: Tuple[str, ...] = ("agua", "humo", "fuego")

IGNORE_INDEX: int = 255


@dataclass
class SegmentationHeadConfig:
    classes: Tuple[str, ...] = DEFAULT_CLASSES

    in_channels: int = 256
    hidden: int = 128
    fpn_levels: Tuple[str, ...] = ("p3", "p4", "p5")

    # Fraccion del frame que tiene que cubrir una clase para reportarse.
    # Una camara de vigilancia ve reflejos y luces naranjas todo el tiempo:
    # sin un piso de area, cada atardecer es un incendio.
    # UNO POR CLASE, y no por gusto: medido con --diagnostico sobre 4.097
    # imagenes de validacion, el F1 a nivel alerta cambia mucho con el umbral y
    # el optimo NO es el mismo para las tres.
    #
    #            0.002    0.005    0.010    0.020    0.050
    #   agua     95.3%    95.4%    95.4%    94.8%    94.0%   -> plano, 0.005
    #   humo     89.0%    88.1%    91.9%    95.6%    94.4%   -> pico en 0.020
    #   fuego    99.1%    97.5%    89.6%    65.9%    12.6%   -> pico en 0.002
    #
    # El fuego vive en regiones chicas (1,6% del cuadro de mediana): un umbral
    # de 0,02 se come un tercio de los incendios reales. El humo es lo
    # contrario: subir el piso filtra la neblina y gana 7,5 puntos de F1. Un
    # solo numero para las tres deja ~9 puntos sobre la mesa.
    area_thresh: Tuple[float, ...] = (0.0, 0.005, 0.020, 0.002)
    prob_thresh: float = 0.50

    # Banda de duda que se manda al VLM en vez de alertar o descartar.
    vlm_gate_band: Tuple[float, float] = (0.005, 0.02)

    # Perdida
    ignore_index: int = IGNORE_INDEX
    dice_weight: float = 0.5
    prior_fondo: float = 0.90           # sesgo inicial de la capa de salida

    @property
    def num_classes(self) -> int:
        return len(self.classes)


@dataclass
class SegResult:
    """Lo que ve la capa de fusion. Duck typing, igual que `Detection`."""
    mask: Tensor                         # (H, W) uint8 con 0..num_classes-1
    area: Dict[str, float]               # fraccion del frame por clase
    score: Dict[str, float]              # prob media dentro de la region
    camera_id: str = "unknown"
    frame_idx: int = -1
    ts: float = field(default_factory=time.time)
    needs_vlm: bool = False

    @property
    def clases_activas(self) -> List[str]:
        return [k for k, v in self.area.items() if v > 0]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("mask")                    # la mascara no va al log
        return d


class _RamaFPN(nn.Module):
    """Sube un nivel de la piramide hasta el stride del nivel mas fino.

    `pasos` es cuantas veces hay que duplicar la resolucion: 0 para p3, 1 para
    p4, 2 para p5. Cada paso es conv+GN+ReLU y despues el upsample, para que
    la interpolacion trabaje sobre features ya suavizadas y no al reves.
    """

    def __init__(self, in_ch: int, hidden: int, pasos: int):
        super().__init__()
        capas: List[nn.Module] = []
        for i in range(max(1, pasos)):
            capas += [
                nn.Conv2d(in_ch if i == 0 else hidden, hidden, 3, padding=1, bias=False),
                nn.GroupNorm(32, hidden),
                nn.ReLU(inplace=True),
            ]
            if pasos:
                capas.append(nn.Upsample(scale_factor=2, mode="bilinear",
                                         align_corners=False))
        self.rama = nn.Sequential(*capas)

    def forward(self, x: Tensor) -> Tensor:
        return self.rama(x)


class SegmentationHead(nn.Module):
    def __init__(self, cfg: Optional[SegmentationHeadConfig] = None,
                 in_channels: Optional[int] = None,
                 num_classes: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or SegmentationHeadConfig()
        if in_channels is not None:
            self.cfg.in_channels = in_channels
        if num_classes is not None and num_classes != self.cfg.num_classes:
            self.cfg.classes = tuple(f"clase_{i}" for i in range(num_classes))
        c = self.cfg

        self.ramas = nn.ModuleDict({
            nivel: _RamaFPN(c.in_channels, c.hidden, pasos)
            for pasos, nivel in enumerate(c.fpn_levels)
        })
        self.salida = nn.Sequential(
            nn.Conv2d(c.hidden, c.hidden, 3, padding=1, bias=False),
            nn.GroupNorm(32, c.hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(c.hidden, c.num_classes, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Sesgo inicial en el prior de fondo: log(p/(1-p)) contra el resto
        # repartido en partes iguales.
        p = c.prior_fondo
        resto = (1.0 - p) / max(1, c.num_classes - 1)
        import math
        sesgo = [math.log(p)] + [math.log(resto)] * (c.num_classes - 1)
        with torch.no_grad():
            self.salida[-1].bias.copy_(torch.tensor(sesgo))

    # ------------------------------------------------------------------ #
    def forward(self, fpn_feats: Dict[str, Tensor],
                out_size: Optional[Tuple[int, int]] = None) -> Tensor:
        """(B, num_classes, H, W). `out_size` en pixeles del tensor de entrada."""
        base: Optional[Tensor] = None
        for nivel in self.cfg.fpn_levels:
            y = self.ramas[nivel](fpn_feats[nivel])
            if base is None:
                base = y
            else:
                if y.shape[-2:] != base.shape[-2:]:
                    y = F.interpolate(y, base.shape[-2:], mode="bilinear",
                                      align_corners=False)
                base = base + y
        logits = self.salida(base)
        if out_size is not None and tuple(logits.shape[-2:]) != tuple(out_size):
            logits = F.interpolate(logits, out_size, mode="bilinear",
                                   align_corners=False)
        return logits

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        fpn_feats: Dict[str, Tensor],
        image_sizes: Sequence[Tuple[int, int]],
        camera_ids: Optional[Sequence[str]] = None,
        frame_idxs: Optional[Sequence[int]] = None,
    ) -> List[SegResult]:
        c = self.cfg
        h, w = image_sizes[0]
        logits = self.forward(fpn_feats, (h, w))
        probs = logits.softmax(dim=1)
        conf, pred = probs.max(dim=1)

        # Debajo del umbral de probabilidad es fondo: preferimos perder una
        # region dudosa antes que alertar por ruido.
        pred = torch.where(conf >= c.prob_thresh, pred, torch.zeros_like(pred))

        # Un solo sync GPU->CPU para todo el batch, no uno por clase.
        n = c.num_classes
        idx = torch.arange(pred.shape[0], device=pred.device)[:, None] * n
        cuentas = torch.bincount((idx + pred.flatten(1)).flatten(),
                                 minlength=pred.shape[0] * n).view(-1, n)
        suma_conf = torch.zeros_like(cuentas, dtype=torch.float32)
        suma_conf.scatter_add_(1, pred.flatten(1), conf.flatten(1).float())
        cuentas_cpu = cuentas.cpu()
        conf_cpu = suma_conf.cpu()

        total = float(h * w)
        salida: List[SegResult] = []
        for i in range(pred.shape[0]):
            area, score = {}, {}
            for k in range(1, n):
                px = int(cuentas_cpu[i, k])
                frac = px / total
                if frac >= self.umbral(k):
                    area[c.classes[k]] = frac
                    score[c.classes[k]] = float(conf_cpu[i, k]) / max(px, 1)
            salida.append(SegResult(
                mask=pred[i].to(torch.uint8),
                area=area,
                score=score,
                camera_id=camera_ids[i] if camera_ids else "unknown",
                frame_idx=frame_idxs[i] if frame_idxs else -1,
                needs_vlm=self._gate_to_vlm(area),
            ))
        return salida

    def umbral(self, clase: int) -> float:
        """Umbral de area de esa clase. Acepta un escalar por compatibilidad."""
        u = self.cfg.area_thresh
        if isinstance(u, (int, float)):
            return float(u)
        return float(u[clase]) if clase < len(u) else float(u[-1])

    def _gate_to_vlm(self, area: Dict[str, float]) -> bool:
        lo, hi = self.cfg.vlm_gate_band
        return any(k in CRITICAL_CLASSES and lo <= v <= hi for k, v in area.items())

    # ------------------------------------------------------------------ #
    def compute_loss(
        self,
        fpn_feats: Dict[str, Tensor],
        targets: Tensor,
        anotadas: Optional[Tensor] = None,
        class_weights: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """CE + Dice, restringidas a las clases que cada muestra sabe anotar.

        `targets`  (B, H, W) long, con `ignore_index` en lo que no se supervisa.
        `anotadas` (B, num_classes) bool. None = todas las clases anotadas.

        Por que el softmax restringido: el dataset mezcla fuentes que anotan
        clases distintas. Una foto de incendio marca la llama y deja el humo de
        esa misma llama como fondo. Con CE plena, la clase humo recibe
        supervision negativa masiva y nunca despega. Enmascarar los logits de
        las clases que la fuente no miro equivale a decir "de lo que esta
        fuente sabe distinguir, la respuesta es esta". La clase fondo, que
        todas comparten, es la que ata las escalas entre datasets.
        """
        c = self.cfg
        n = c.num_classes
        logits = self.forward(fpn_feats, targets.shape[-2:]).float()
        if anotadas is None:
            anotadas = torch.ones(logits.shape[0], n, dtype=torch.bool,
                                  device=logits.device)
        mask = anotadas[:, :, None, None]
        l = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        # Un pixel cuya clase objetivo no esta en `anotadas` es una
        # contradiccion de la fuente. Si se deja pasar, su log-prob es -inf y
        # toda la perdida del batch se va a infinito: un dato mal declarado
        # entre un millon apagaria el entrenamiento entero.
        es_ign = targets == c.ignore_index
        y0 = torch.where(es_ign, torch.zeros_like(targets), targets)
        permitido = anotadas.gather(1, y0.reshape(y0.shape[0], -1)).reshape(y0.shape)
        descartados = (~permitido & ~es_ign).sum()
        y = torch.where(es_ign | permitido, targets,
                        torch.full_like(targets, c.ignore_index))

        w = (class_weights if class_weights is not None
             else torch.ones(n, device=logits.device)).to(logits.dtype)

        # Reduccion a mano: con reduction="mean", un batch sin ningun pixel
        # valido devuelve NaN (0/0) y contamina los pesos para siempre.
        por_px = F.cross_entropy(l, y, weight=w, ignore_index=c.ignore_index,
                                 reduction="none")
        val = y != c.ignore_index
        peso_px = torch.where(val, w[y.clamp(max=n - 1)], torch.zeros_like(por_px))
        ce = por_px.sum() / peso_px.sum().clamp(min=1e-6)

        p = torch.softmax(l, dim=1) * val[:, None]
        oh = F.one_hot(torch.where(val, y, torch.zeros_like(y)), n)
        oh = oh.permute(0, 3, 1, 2).to(p.dtype) * val[:, None]
        oh = oh * mask
        inter = (p * oh).sum(dim=(2, 3))
        union = p.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
        dice_c = 1.0 - (2 * inter + 1.0) / (union + 1.0)
        dice = (dice_c * mask.squeeze(-1).squeeze(-1)).sum() / mask.sum().clamp(min=1)

        return {
            "loss_ce": ce,
            "loss_dice": dice * c.dice_weight,
            "px_descartados": descartados,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = SegmentationHeadConfig()
    head = SegmentationHead(cfg)

    # p3/p4/p5 tal como los entrega SharedBackbone a 384 de entrada
    feats = {"p3": torch.randn(2, 256, 48, 48),
             "p4": torch.randn(2, 256, 24, 24),
             "p5": torch.randn(2, 256, 12, 12)}

    logits = head(feats, (384, 384))
    print(f"logits: {tuple(logits.shape)}")
    assert logits.shape == (2, cfg.num_classes, 384, 384)

    head.eval()
    res = head.predict(feats, [(384, 384)] * 2, camera_ids=["cam-3", "cam-4"],
                       frame_idxs=[1042, 1043])
    for r in res:
        print(r.to_dict())
    assert all(not r.area for r in res), (
        "sin entrenar y con el sesgo en el prior, todo tiene que dar fondo")

    # Se fuerza agua en la mitad de abajo para ejercitar el camino con region.
    with torch.no_grad():
        head.salida[-1].bias.copy_(torch.tensor([0.0, 6.0, 0.0, 0.0]))
    res = head.predict(feats, [(384, 384)] * 2)
    print("con agua forzada:", res[0].to_dict())
    assert res[0].area.get("agua", 0) > 0.9 and res[0].clases_activas == ["agua"]

    head.train()
    y = torch.zeros(2, 384, 384, dtype=torch.long)
    y[0, 200:, :] = 1                    # agua abajo
    y[1, :100, 100:200] = 3              # fuego arriba
    y[1, :10, :10] = IGNORE_INDEX
    anot = torch.tensor([[True, True, False, False],
                         [True, False, True, True]])
    perdidas = head.compute_loss(feats, y, anot)
    print({k: float(v.detach()) for k, v in perdidas.items()})
    assert torch.isfinite(perdidas["loss_ce"]), "perdida no finita"
    assert int(perdidas["px_descartados"]) == 0

    # Fuente que declara agua pero la mascara trae fuego: se descarta, no explota.
    mala = head.compute_loss(feats, torch.full((2, 384, 384), 3, dtype=torch.long),
                             torch.tensor([[True, True, False, False]] * 2))
    print("etiquetas contradictorias ->", {k: float(v.detach()) for k, v in mala.items()})
    assert torch.isfinite(mala["loss_ce"])
    assert int(mala["px_descartados"]) == 2 * 384 * 384
    # umbral por clase: el de fuego tiene que ser mas bajo que el de humo
    h2 = SegmentationHead(SegmentationHeadConfig())
    assert h2.umbral(3) < h2.umbral(1) < h2.umbral(2), "umbrales por clase al reves"
    assert SegmentationHead(SegmentationHeadConfig(area_thresh=0.01)).umbral(2) == 0.01, \
        "no acepta un escalar por compatibilidad"
    print(f"umbrales: " + "  ".join(f"{DEFAULT_CLASSES[k]} {h2.umbral(k):.3f}"
                                    for k in range(1, 4)))
    print("OK")
