# -*- coding: utf-8 -*-
"""
motor_fusion.py · HORUS — de hallazgos a eventos.

Las reglas miran un frame. Este archivo mira el tiempo, y hace las cuatro
cosas que faltan para que una detección se convierta en algo que valga
despertar a alguien:

1. **Persistencia.** Un hallazgo suelto no abre nada. Hacen falta N
   confirmaciones dentro de una ventana. Un frame raro no es un evento.

2. **Identidad en el tiempo.** El mismo incendio en veinte frames seguidos es
   UN evento que dura, no veinte alertas. Y si el hallazgo trae `global_id`,
   la persona que cruza de cámara sigue siendo el mismo evento — es para esto
   que existe el tracking global.

3. **Correlación.** Arma + intrusión sobre la misma persona no son dos
   avisos: son un robo. Esta es la parte que le da sentido al nombre de la
   capa; el resto es contabilidad.

4. **Higiene de alertas.** Cierre por silencio, enfriamiento antes de poder
   reabrir, y un tope de eventos abiertos. Un sistema que alerta de más se
   apaga solo, porque el operador deja de mirarlo.

El gate del VLM (`vlm_gate.py`) se consulta acá y no en el backbone, por la
razón que dice ese archivo: la decisión combina señales que el backbone no
conoce. La verificación del VLM **no bloquea**: el evento sale igual y el VLM
lo enriquece después (`08_vlm_verificacion`), tal como está en el diagrama.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

_AQUI = os.path.dirname(os.path.abspath(__file__))
if _AQUI not in sys.path:
    sys.path.insert(0, _AQUI)

from contratos import Evento, Hallazgo, ObservacionCamara, Severidad  # noqa: E402
from reglas import Contexto, Regla, reglas_por_defecto  # noqa: E402
from topologia import Topologia  # noqa: E402
from vlm_gate import GateConfig, VLMGate  # noqa: E402


# --------------------------------------------------------------------------- #
@dataclass
class ConfigFusion:
    # --- persistencia -----------------------------------------------------
    confirmaciones_min: int = 2        # hallazgos antes de abrir el evento
    ventana_confirmacion_s: float = 3.0
    # Lo crítico no espera: un incendio confirmado por las dos cabezas abre
    # con la primera. La latencia de una confirmación de más se paga en
    # segundos y el diagrama promete ~2 s de cámara a alerta.
    confirmaciones_criticas: int = 1

    # --- ciclo de vida ----------------------------------------------------
    silencio_s: float = 5.0            # sin hallazgos -> se cierra
    cooldown_s: float = 30.0           # antes de poder reabrir lo mismo
    max_abiertos: int = 64

    # --- salida -----------------------------------------------------------
    severidad_min: Severidad = Severidad.AVISO
    reemitir_s: float = 20.0           # re-avisar de un evento que sigue vivo

    # --- correlación ------------------------------------------------------
    correlacion_s: float = 15.0
    suprimir_componentes: bool = True  # si sale "robo", callar arma+intrusión

    verboso: bool = False


@dataclass
class _Pendiente:
    hallazgos: Deque[Hallazgo] = field(default_factory=lambda: deque(maxlen=32))
    primera_ts: float = 0.0

    def podar(self, ts: float, ventana: float) -> None:
        while self.hallazgos and ts - self.hallazgos[0].ts > ventana:
            self.hallazgos.popleft()


# --------------------------------------------------------------------------- #
class MotorFusion:
    """El orquestador de la capa.

    Uso:
        motor = MotorFusion(topologia=topo)
        eventos = motor.procesar([obs_cam1, obs_cam2, ...])   # una vez por frame
        for e in eventos:
            alerta.emitir(e)          # 07_alerta
    """

    def __init__(self,
                 topologia: Optional[Topologia] = None,
                 cfg: Optional[ConfigFusion] = None,
                 reglas: Optional[Sequence[Regla]] = None,
                 gate: Optional[VLMGate] = None,
                 fps: float = 10.0) -> None:
        self.cfg = cfg or ConfigFusion()
        self.topo = topologia
        self.reglas: List[Regla] = list(reglas) if reglas is not None else reglas_por_defecto()
        self.gate = gate or VLMGate(GateConfig())
        self.ctx = Contexto(topo=topologia, fps=fps)

        self._pendientes: Dict[Tuple, _Pendiente] = {}
        self._abiertos: Dict[Tuple, Evento] = {}
        self._cooldown: Dict[Tuple, float] = {}
        self._ultimo_aviso: Dict[Tuple, float] = {}
        self._n = 0
        self.historial: Deque[Evento] = deque(maxlen=500)
        self.stats: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def procesar(self, observaciones: Iterable[ObservacionCamara],
                 ts: Optional[float] = None) -> List[Evento]:
        """Un tick del pipeline. Devuelve lo que hay que avisar AHORA:
        eventos recién abiertos, eventos que escalaron de severidad, eventos
        vivos que toca re-avisar, y eventos que se cerraron."""
        obs_list = list(observaciones)
        ts = float(ts if ts is not None else
                   (max((o.ts for o in obs_list), default=time.time())))

        hallazgos: List[Hallazgo] = []
        for obs in obs_list:
            gate = self.gate.decide(obs, obs.incertidumbre)
            for regla in self.reglas:
                for h in regla.evaluar(obs, self.ctx):
                    if not h.ts:
                        h.ts = obs.ts
                    if gate.run:
                        h.necesita_vlm = True
                        h.evidencia.setdefault("vlm_motivo", gate.reason)
                    hallazgos.append(h)
                    self.stats[h.tipo] = self.stats.get(h.tipo, 0) + 1

        hallazgos = self._correlacionar(hallazgos, ts)

        salida: List[Evento] = []
        for h in hallazgos:
            e = self._integrar(h, ts)
            if e is not None:
                salida.append(e)

        salida.extend(self._cerrar_vencidos(ts))
        salida.extend(self._reemitir(ts, {id(e) for e in salida}))
        return [e for e in salida if e.severidad >= self.cfg.severidad_min]

    # ------------------------------------------------------------------ #
    def _correlacionar(self, hallazgos: List[Hallazgo],
                       ts: float) -> List[Hallazgo]:
        """Arma + intrusión sobre la misma persona = robo.

        Se agrupa por `global_id` cuando lo hay y por cámara cuando no. El
        `global_id` es lo que permite que el arma vista en una cámara y la
        intrusión detectada en otra se junten en un solo evento.
        """
        if not hallazgos:
            return hallazgos
        c = self.cfg

        grupos: Dict[Any, List[Hallazgo]] = {}
        for h in hallazgos:
            grupos.setdefault(h.global_id if h.global_id is not None
                              else ("cam", h.camera_id), []).append(h)

        extra: List[Hallazgo] = []
        suprimir: set = set()
        for clave, grupo in grupos.items():
            tipos = {h.tipo for h in grupo}
            if not ({"arma"} & tipos and {"intrusion"} & tipos):
                continue
            arma = max((h for h in grupo if h.tipo == "arma"),
                       key=lambda h: h.confianza)
            intr = max((h for h in grupo if h.tipo == "intrusion"),
                       key=lambda h: h.confianza)
            extra.append(Hallazgo(
                tipo="robo", severidad=Severidad.CRITICO,
                confianza=min(0.97, 0.5 * (arma.confianza + intr.confianza) + 0.15),
                camera_id=intr.camera_id, zona=intr.zona, ts=ts,
                motivo=(f"{arma.evidencia.get('clase', 'arma')} + persona en "
                        f"{intr.zona or 'zona restringida'} fuera de horario"),
                track_ids=sorted(set(arma.track_ids) | set(intr.track_ids)),
                global_id=arma.global_id if arma.global_id is not None else intr.global_id,
                necesita_vlm=True,
                evidencia={"arma": arma.evidencia, "intrusion": intr.evidencia,
                           "correlacion": "arma+intrusion"}))
            if c.suprimir_componentes:
                suprimir.update((id(arma), id(intr)))

        if not extra:
            return hallazgos
        return [h for h in hallazgos if id(h) not in suprimir] + extra

    # ------------------------------------------------------------------ #
    def _integrar(self, h: Hallazgo, ts: float) -> Optional[Evento]:
        c = self.cfg
        clave = h.clave()

        # ¿Ya está abierto? Se sostiene y, si escaló, se re-avisa.
        ev = self._abiertos.get(clave)
        if ev is not None:
            subio = h.severidad > ev.severidad
            ev.ts_ultimo = max(ev.ts_ultimo, h.ts or ts)
            ev.confianza = max(ev.confianza, h.confianza)
            ev.severidad = max(ev.severidad, h.severidad)
            ev.confirmaciones += 1
            ev.estado = "sostenido"
            ev.necesita_vlm = ev.necesita_vlm or h.necesita_vlm
            ev.evidencia.update(h.evidencia)
            if h.camera_id not in ev.camaras:
                ev.camaras.append(h.camera_id)
            for t in h.track_ids:
                if t not in ev.track_ids:
                    ev.track_ids.append(t)
            if h.global_id is not None:
                ev.global_id = h.global_id
            if subio:
                ev.motivo = h.motivo
                self._ultimo_aviso[clave] = ts
                return ev
            return None

        # Enfriamiento: lo mismo, en el mismo lugar, recién cerrado.
        hasta = self._cooldown.get(clave)
        if hasta is not None:
            if ts < hasta:
                return None
            del self._cooldown[clave]

        # Acumular confirmaciones.
        pend = self._pendientes.setdefault(clave, _Pendiente(primera_ts=h.ts or ts))
        pend.hallazgos.append(h)
        pend.podar(ts, c.ventana_confirmacion_s)

        necesarias = (c.confirmaciones_criticas if h.severidad >= Severidad.CRITICO
                      else c.confirmaciones_min)
        if len(pend.hallazgos) < necesarias:
            return None

        del self._pendientes[clave]
        if len(self._abiertos) >= c.max_abiertos:
            self._cerrar_mas_viejo(ts)
        return self._abrir(clave, h, pend, ts)

    def _abrir(self, clave: Tuple, h: Hallazgo,
               pend: _Pendiente, ts: float) -> Evento:
        self._n += 1
        primera = min((x.ts for x in pend.hallazgos), default=h.ts or ts)
        ev = Evento(
            evento_id=f"E{self._n:06d}",
            tipo=h.tipo, severidad=h.severidad, camera_id=h.camera_id,
            ts_inicio=primera, ts_ultimo=h.ts or ts,
            confianza=max(x.confianza for x in pend.hallazgos),
            motivo=h.motivo, zona=h.zona, estado="abierto",
            global_id=h.global_id, track_ids=list(h.track_ids),
            camaras=[h.camera_id], confirmaciones=len(pend.hallazgos),
            necesita_vlm=h.necesita_vlm,
            vlm_motivo=str(h.evidencia.get("vlm_motivo", "")),
            evidencia=dict(h.evidencia))
        self._abiertos[clave] = ev
        self._ultimo_aviso[clave] = ts
        self.historial.append(ev)
        if self.cfg.verboso:
            print(ev.linea())
        return ev

    # ------------------------------------------------------------------ #
    def _cerrar_vencidos(self, ts: float) -> List[Evento]:
        c = self.cfg
        cerrados: List[Evento] = []
        for clave, ev in list(self._abiertos.items()):
            if ts - ev.ts_ultimo < c.silencio_s:
                continue
            ev.estado = "cerrado"
            del self._abiertos[clave]
            self._cooldown[clave] = ts + c.cooldown_s
            self._ultimo_aviso.pop(clave, None)
            cerrados.append(ev)

        for clave, p in list(self._pendientes.items()):
            p.podar(ts, c.ventana_confirmacion_s)
            if not p.hallazgos:
                del self._pendientes[clave]
        return cerrados

    def _cerrar_mas_viejo(self, ts: float) -> None:
        if not self._abiertos:
            return
        clave = min(self._abiertos, key=lambda k: self._abiertos[k].ts_ultimo)
        ev = self._abiertos.pop(clave)
        ev.estado = "cerrado"
        self._cooldown[clave] = ts + self.cfg.cooldown_s

    def _reemitir(self, ts: float, ya: set) -> List[Evento]:
        """Un evento que sigue vivo se vuelve a avisar cada `reemitir_s`. Un
        incendio de tres minutos no puede desaparecer del tablero porque ya se
        avisó una vez."""
        fuera = []
        for clave, ev in self._abiertos.items():
            if id(ev) in ya:
                continue
            if ts - self._ultimo_aviso.get(clave, 0.0) >= self.cfg.reemitir_s:
                self._ultimo_aviso[clave] = ts
                fuera.append(ev)
        return fuera

    # ------------------------------------------------------------------ #
    @property
    def abiertos(self) -> List[Evento]:
        return sorted(self._abiertos.values(),
                      key=lambda e: (-int(e.severidad), e.ts_inicio))

    def reglas_dormidas(self) -> Dict[str, int]:
        """Reglas que nunca pudieron correr, y cuántas veces. Si algo no
        alerta nunca, esto dice si es porque no pasó nada o porque falta una
        cabeza."""
        return {r.tipo: r.veces_dormida for r in self.reglas if r.veces_dormida}

    def verificar(self) -> List[str]:
        avisos: List[str] = []
        if self.topo is None:
            avisos.append("sin topología: intrusión y merodeo por zona no "
                          "pueden evaluarse")
        else:
            avisos.extend(self.topo.verificar())
        return avisos

    def reset(self) -> None:
        self._pendientes.clear()
        self._abiertos.clear()
        self._cooldown.clear()
        self._ultimo_aviso.clear()
        self.ctx.limpiar()
        self.stats.clear()

    def resumen(self) -> str:
        dormidas = self.reglas_dormidas()
        partes = [f"[fusión] {len(self._abiertos)} abiertos, "
                  f"{len(self._pendientes)} pendientes, "
                  f"{self._n} eventos en total"]
        if self.stats:
            partes.append("  hallazgos: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.stats.items())))
        if dormidas:
            partes.append("  reglas dormidas (falta la cabeza): " + ", ".join(
                f"{k}×{v}" for k, v in sorted(dormidas.items())))
        for ev in self.abiertos[:5]:
            partes.append("  " + ev.linea())
        return "\n".join(partes)
