"""Gate del VLM — decide cuándo despertar al VLM (set abierto).

Vive en la capa de fusión porque la decisión combina señales que el
backbone no conoce: la incertidumbre de las cabezas y (a futuro) la
persistencia temporal del tracking. El backbone solo aporta datos
pasivos: el score de novedad ya viene calculado en SharedFeatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class GateConfig:
    novelty_threshold: float = 3.0      # sigmas sobre lo "normal" de la escena
    uncertainty_threshold: float = 0.45  # duda de las cabezas (0-1)


@dataclass
class GateDecision:
    run: bool
    reason: str  # "novedad", "duda", "novedad+duda" o ""

    def __bool__(self) -> bool:
        return self.run


class VLMGate:
    """Decide si vale la pena despertar al VLM para un frame dado.

    Uso:
        gate = VLMGate()
        decision = gate.decide(features, head_uncertainty=0.6)
        if decision:  # o decision.run
            ...mandar crop al VLM...
    """

    def __init__(self, config: Optional[GateConfig] = None) -> None:
        self.cfg = config or GateConfig()

    def decide(self, features: Any, head_uncertainty: float = 0.0) -> GateDecision:
        # features es SharedFeatures; solo se usa .novelty (duck typing para
        # no acoplar este módulo al paquete del backbone).
        # Durante el warmup del backbone, novelty llega en 0.0, así que el
        # gate no dispara por novedad hasta que la escena "normal" se aprendió.
        by_novelty = features.novelty >= self.cfg.novelty_threshold
        by_doubt = head_uncertainty >= self.cfg.uncertainty_threshold

        if by_novelty and by_doubt:
            return GateDecision(True, "novedad+duda")
        if by_novelty:
            return GateDecision(True, "novedad")
        if by_doubt:
            return GateDecision(True, "duda")
        return GateDecision(False, "")

    def should_run(self, features: Any, head_uncertainty: float = 0.0) -> bool:
        return self.decide(features, head_uncertainty).run
