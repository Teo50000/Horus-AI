# -*- coding: utf-8 -*-
"""
reglas.py · HORUS — las reglas que convierten tracks en hallazgos.

Una regla mira UN frame de UNA cámara y devuelve candidatos a evento
(`Hallazgo`). No decide nada definitivo: el que exige persistencia, deduplica
y abre el evento es `motor_fusion.py`. Esa separación es a propósito — así una
regla se puede leer, discutir y probar sola.

Principios que siguen todas
---------------------------
- **Solo tracks confirmados.** Una detección suelta no es evidencia de nada;
  ese filtro ya lo hizo el tracking y repetirlo acá sería peor.
- **Persistencia antes que umbral.** Contra falsas alarmas conviene pedir que
  algo se sostenga, no que puntúe más alto. Subir el umbral se lleva puestos
  los positivos débiles — está medido en `calibracion-umbrales-objetos`, es
  exactamente lo que hacía que no se detectara ningún paquete.
- **Una regla que necesita una cabeza ausente queda DORMIDA**, no devuelve
  vacío. "No hubo caídas" y "no hay cabeza de caídas" no pueden parecer lo
  mismo en un sistema cuyo objetivo es no perderse nada.

Cabezas que faltan
------------------
`ReglaCaida` y `ReglaAccionExterna` están escritas completas y probadas contra
observaciones sintéticas, pero hoy duermen porque no hay cabeza de pose ni
clasificador de acción. El día que lleguen, lo único que hay que hacer es
llenar `AccionObs` con el adaptador de `contratos.py` y las reglas se
despiertan solas. Para agregar una acción nueva (pelea, vandalismo,
aglomeración) alcanza con una línea en `ACCIONES_EXTERNAS`.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
if _AQUI not in sys.path:
    sys.path.insert(0, _AQUI)

from contratos import (  # noqa: E402
    CABEZA_ACCION, CABEZA_OBJETOS, CABEZA_POSE, CABEZA_SEGMENTACION,
    AccionObs, Hallazgo, ObservacionCamara, Severidad,
)
from topologia import Topologia, Zona  # noqa: E402


# --------------------------------------------------------------------------- #
# Contexto compartido
# --------------------------------------------------------------------------- #
@dataclass
class Contexto:
    """Memoria entre frames. Cada regla tiene su propio cajón (`mem`), así
    dos reglas no se pisan el estado."""
    topo: Optional[Topologia] = None
    fps: float = 10.0
    _estado: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def mem(self, regla: Any) -> Dict[str, Any]:
        return self._estado.setdefault(getattr(regla, "tipo", str(regla)), {})

    def zona(self, track: Any) -> Optional[Zona]:
        return self.topo.zona_de_track(track) if self.topo else None

    def limpiar(self) -> None:
        self._estado.clear()


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class Regla:
    tipo: str = "generico"
    requiere: FrozenSet[str] = frozenset({CABEZA_OBJETOS})
    requiere_alguna: FrozenSet[str] = frozenset()   # basta con UNA de estas
    necesita_topologia: bool = False                # sin topologia.json, DUERME

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)
        self.veces_dormida = 0

    # -------------------------------------------------------------- #
    def puede_correr(self, obs: ObservacionCamara) -> bool:
        if not self.requiere.issubset(obs.cabezas):
            return False
        if self.requiere_alguna and not (self.requiere_alguna & obs.cabezas):
            return False
        return True

    def evaluar(self, obs: ObservacionCamara, ctx: Contexto) -> List[Hallazgo]:
        # La topología cuenta igual que una cabeza: una regla que la necesita y
        # no la tiene está CIEGA, no tranquila. Antes `intrusion` devolvía lista
        # vacía en silencio cuando faltaba topologia.json, o sea que el panel
        # mostraba lo mismo que si de verdad no hubiera entrado nadie. Es
        # exactamente la confusión que `reglas_dormidas()` existe para evitar.
        if not self.puede_correr(obs) or (self.necesita_topologia and ctx.topo is None):
            self.veces_dormida += 1
            return []
        return self._evaluar(obs, ctx)

    def _evaluar(self, obs: ObservacionCamara,
                 ctx: Contexto) -> List[Hallazgo]:      # pragma: no cover
        raise NotImplementedError

    def faltantes(self, obs: ObservacionCamara,
                  ctx: Optional[Contexto] = None) -> List[str]:
        falta = sorted(self.requiere - obs.cabezas)
        if self.requiere_alguna and not (self.requiere_alguna & obs.cabezas):
            falta.append("|".join(sorted(self.requiere_alguna)))
        if self.necesita_topologia and (ctx is None or ctx.topo is None):
            falta.append("topologia.json")
        return falta


# --------------------------------------------------------------------------- #
# Incendio
# --------------------------------------------------------------------------- #
class ReglaIncendio(Regla):
    """Humo y fuego, cruzando las dos cabezas que los ven.

    La corroboración cruzada es el punto. La cabeza de segmentación tiene F1
    99,1% en fuego y 95,6% en humo a nivel alerta (medido sobre 4.097
    imágenes); la de objetos aporta la caja, que es lo que se recorta para el
    VLM y para el clip. Cuando las dos coinciden no hace falta nada más.

    Cuando NO coinciden, la asimetría importa:

      - Fuego de segmentación solo ya es evidencia fuerte (es la clase más
        confiable del sistema entero).
      - Humo solo, de cualquiera de las dos, NO lo es: neblina, vapor y nubes
        son humo a 384 px, y está documentado que el modelo no puede
        separarlos. Va como aviso y se lo manda al VLM.

    Y el crecimiento: un incendio real crece. Un reflejo de atardecer se
    queda del mismo tamaño. Comparar el área contra la de hace unos segundos
    separa las dos cosas mejor que cualquier umbral instantáneo.
    """

    tipo = "incendio"
    requiere = frozenset()
    requiere_alguna = frozenset({CABEZA_OBJETOS, CABEZA_SEGMENTACION})

    persistencia_s: float = 1.5
    critico_s: float = 6.0
    ventana_crecimiento_s: float = 5.0
    crecimiento_min: float = 1.30          # +30% de área = está creciendo

    def _evaluar(self, obs: ObservacionCamara, ctx: Contexto) -> List[Hallazgo]:
        llamas = [t for t in obs.confirmados("llama")
                  if t.visto_s >= self.persistencia_s]
        humos = [t for t in obs.confirmados("humo")
                 if t.visto_s >= self.persistencia_s]

        seg = obs.seg
        area_fuego = seg.de("fuego") if seg else 0.0
        area_humo = seg.de("humo") if seg else 0.0
        # `SegResult.area` solo trae las clases que ya pasaron su umbral de
        # área por clase, así que estar presente ES haber pasado el filtro.
        seg_fuego = area_fuego > 0.0
        seg_humo = area_humo > 0.0

        if not (llamas or humos or seg_fuego or seg_humo):
            self._olvidar(ctx, obs.camera_id)
            return []

        crece, factor = self._crecimiento(ctx, obs, area_fuego + area_humo)

        obj_fuego = bool(llamas)
        hay_fuego = obj_fuego or seg_fuego
        hay_humo = bool(humos) or seg_humo

        if hay_fuego and (obj_fuego and seg_fuego):
            conf, sev, motivo = 0.95, Severidad.CRITICO, "fuego confirmado por las dos cabezas"
        elif seg_fuego and hay_humo:
            conf, sev, motivo = 0.90, Severidad.CRITICO, "fuego y humo en segmentación"
        elif seg_fuego:
            conf, sev, motivo = 0.85, Severidad.ALERTA, "fuego en segmentación"
        elif obj_fuego:
            conf, sev, motivo = 0.70, Severidad.ALERTA, "llama detectada por objetos"
        elif seg_humo and humos:
            conf, sev, motivo = 0.70, Severidad.ALERTA, "humo en las dos cabezas"
        else:
            conf, sev, motivo = 0.50, Severidad.AVISO, "humo sin confirmar (puede ser vapor o neblina)"

        if crece:
            conf = min(0.98, conf + 0.10)
            motivo += f", el área creció {factor:.1f}x"
            if sev < Severidad.ALERTA:
                sev = Severidad.ALERTA

        cajas = llamas + humos
        antiguedad = max((t.visto_s for t in cajas), default=0.0)
        if antiguedad >= self.critico_s and conf >= 0.70:
            sev = Severidad.CRITICO
            motivo += f", sostenido {antiguedad:.0f} s"

        return [Hallazgo(
            tipo=self.tipo, severidad=sev, confianza=conf,
            camera_id=obs.camera_id, motivo=motivo, ts=obs.ts,
            zona=self._zona(ctx, cajas),
            track_ids=[t.track_id for t in cajas],
            necesita_vlm=(conf < 0.85) or any(t.needs_vlm for t in cajas),
            evidencia={
                "objetos_llama": [round(t.score, 3) for t in llamas],
                "objetos_humo": [round(t.score, 3) for t in humos],
                "seg_area_fuego": round(area_fuego, 5),
                "seg_area_humo": round(area_humo, 5),
                "crecimiento": round(factor, 2),
                "persistencia_s": round(antiguedad, 1),
            })]

    # -------------------------------------------------------------- #
    def _crecimiento(self, ctx: Contexto, obs: ObservacionCamara,
                     area: float) -> Tuple[bool, float]:
        if area <= 0.0:
            return False, 1.0
        mem = ctx.mem(self).setdefault("area", {})
        hist = mem.setdefault(obs.camera_id, [])
        hist.append((obs.ts, area))
        corte = obs.ts - self.ventana_crecimiento_s
        while len(hist) > 2 and hist[0][0] < corte:
            hist.pop(0)
        if len(hist) < 3 or (hist[-1][0] - hist[0][0]) < self.ventana_crecimiento_s * 0.5:
            return False, 1.0
        base = max(hist[0][1], 1e-6)
        factor = area / base
        return factor >= self.crecimiento_min, factor

    def _olvidar(self, ctx: Contexto, cam: str) -> None:
        ctx.mem(self).get("area", {}).pop(cam, None)

    def _zona(self, ctx: Contexto, tracks: Sequence[Any]) -> Optional[str]:
        for t in tracks:
            z = ctx.zona(t)
            if z is not None:
                return z.nombre
        return None


# --------------------------------------------------------------------------- #
# Arma
# --------------------------------------------------------------------------- #
class ReglaArma(Regla):
    """Pistola o cuchillo a la vista, y quién la lleva.

    Esta regla NUNCA da CRITICO por sí sola, y siempre pide VLM. No es
    timidez: el dataset lo dice explícito. Las 565 cajas de pistola salen de
    fotos de Flickr (arma grande, centrada, nítida) y está medido que entrenar
    solo con fotos web da 0% mAP en objetos chicos — en cámara real un arma
    ocupa 16-47 px. La cabeza marca el candidato, el VLM verifica. Nadie en el
    mercado despliega detección de armas en lazo cerrado.

    Lo que sí aporta la fusión: **a quién** se la está viendo. Si el arma cae
    adentro de la caja de una persona trackeada, el hallazgo se cuelga del
    `global_id` de esa persona y no de la cámara — así el evento la sigue si
    cambia de cámara, y el correlador puede cruzarlo con intrusión.
    """

    tipo = "arma"
    requiere = frozenset({CABEZA_OBJETOS})

    persistencia_s: float = 0.4
    contencion_min: float = 0.40           # del arma dentro de la persona

    def _evaluar(self, obs: ObservacionCamara, ctx: Contexto) -> List[Hallazgo]:
        armas = [t for t in obs.confirmados("pistola", "cuchillo")
                 if t.visto_s >= self.persistencia_s]
        if not armas:
            return []
        personas = obs.confirmados("persona")

        fuera: List[Hallazgo] = []
        for a in armas:
            portador = _portador(a, personas, self.contencion_min)
            conf = min(0.85, 0.45 + 0.5 * a.score)
            if portador is not None:
                conf = min(0.90, conf + 0.10)
                motivo = f"{a.clase} en poder de una persona"
            else:
                motivo = f"{a.clase} a la vista, sin portador identificado"

            z = ctx.zona(portador or a)
            fuera.append(Hallazgo(
                tipo=self.tipo,
                severidad=Severidad.ALERTA,       # el correlador puede subirla
                confianza=conf, camera_id=obs.camera_id, motivo=motivo,
                zona=z.nombre if z else None, ts=obs.ts,
                track_ids=[a.track_id] + ([portador.track_id] if portador else []),
                global_id=getattr(portador, "global_id", None) if portador else None,
                necesita_vlm=True,                # siempre, por diseño
                evidencia={
                    "clase": a.clase, "score": round(a.score, 3),
                    "persistencia_s": round(a.visto_s, 1),
                    "portador_track": getattr(portador, "track_id", None),
                    "nota": ("clase no desplegable en lazo cerrado: requiere "
                             "verificación del VLM (ver dataset-objetos-mezcla-v1)"),
                }))
        return fuera


# --------------------------------------------------------------------------- #
# Intrusión
# --------------------------------------------------------------------------- #
class ReglaIntrusion(Regla):
    """Persona dentro de una zona restringida fuera del horario permitido.

    Decide con el punto de los pies, no con el centro de la caja: una persona
    parada justo en el borde de la zona tiene el torso afuera, y usar el
    centro deja pasar exactamente el caso que más importa.
    """

    tipo = "intrusion"
    requiere = frozenset({CABEZA_OBJETOS})
    necesita_topologia = True      # sin zonas no hay "restringida" que violar

    persistencia_s: float = 1.5            # una sombra cruzando no cuenta

    def _evaluar(self, obs: ObservacionCamara, ctx: Contexto) -> List[Hallazgo]:
        if ctx.topo is None:
            return []
        fuera: List[Hallazgo] = []
        for t in obs.confirmados("persona"):
            z = ctx.zona(t)
            if z is None or not z.restringida or z.permitida(obs.ts):
                continue
            if t.visto_s < self.persistencia_s:
                continue
            conf = min(0.95, 0.60 + 0.20 * min(t.visto_s / 5.0, 1.0) + 0.15 * t.score)
            fuera.append(Hallazgo(
                tipo=self.tipo, severidad=Severidad.ALERTA, confianza=conf,
                camera_id=obs.camera_id, zona=z.nombre, ts=obs.ts,
                motivo=f"persona en {z.nombre} fuera de horario",
                track_ids=[t.track_id], global_id=t.global_id,
                necesita_vlm=False,
                evidencia={"permanencia_s": round(t.visto_s, 1),
                           "score": round(t.score, 3),
                           "horario": (None if z.horario_permitido is None else
                                       f"{z.horario_permitido.desde}-"
                                       f"{z.horario_permitido.hasta}")}))
        return fuera


# --------------------------------------------------------------------------- #
# Merodeo
# --------------------------------------------------------------------------- #
class ReglaMerodeo(Regla):
    """Alguien que se queda. No es lo mismo que alguien que pasa.

    La diferencia se mide con `radio_permanencia` del tracker (qué tan grande
    es la región donde estuvo), no con la velocidad instantánea ni con el
    recorrido acumulado: alguien que camina en círculos frente a una puerta
    tiene recorrido alto y radio chico, y es justo el caso que hay que
    detectar.
    """

    tipo = "merodeo"
    requiere = frozenset({CABEZA_OBJETOS})

    radio_max_diag: float = 1.5            # veces la diagonal de su propia caja
    merodeo_s_defecto: float = 25.0

    def _evaluar(self, obs: ObservacionCamara, ctx: Contexto) -> List[Hallazgo]:
        fuera: List[Hallazgo] = []
        for t in obs.confirmados("persona"):
            z = ctx.zona(t)
            limite = z.merodeo_s if z is not None else self.merodeo_s_defecto
            if t.visto_s < limite:
                continue
            diag = _diag(t.bbox_xyxy)
            if t.radio_permanencia > self.radio_max_diag * diag:
                continue                   # se movió: está pasando, no merodeando
            conf = min(0.90, 0.50 + 0.40 * min(t.visto_s / (limite * 3.0), 1.0))
            fuera.append(Hallazgo(
                tipo=self.tipo, severidad=Severidad.AVISO, confianza=conf,
                camera_id=obs.camera_id, zona=z.nombre if z else None, ts=obs.ts,
                motivo=(f"persona {t.visto_s:.0f} s en "
                        f"{z.nombre if z else 'el cuadro'} sin irse"),
                track_ids=[t.track_id], global_id=t.global_id,
                evidencia={"permanencia_s": round(t.visto_s, 1),
                           "radio_px": round(t.radio_permanencia, 1),
                           "recorrido_px": round(t.recorrido, 1),
                           "umbral_s": limite}))
        return fuera


# --------------------------------------------------------------------------- #
# Paquete abandonado
# --------------------------------------------------------------------------- #
class ReglaPaqueteAbandonado(Regla):
    """Un bulto que se queda quieto y del que nadie se hace cargo.

    Dos condiciones, y las dos hacen falta: el paquete tiene que estar quieto
    hace rato Y no puede haber tenido una persona cerca en los últimos
    segundos. Solo con la primera, cualquier caja apoyada en un depósito es
    una alerta permanente.

    Con `paquete` en 0.40 el modelo detecta 18 de 20 cajas reales, pero son
    fotos de Flickr — en cámara puntúa más bajo. Por eso lo que aguanta acá es
    la persistencia (5 s de track confirmado), no la confianza.
    """

    tipo = "paquete_abandonado"
    requiere = frozenset({CABEZA_OBJETOS})

    abandono_s: float = 20.0               # quieto sin dueño
    sin_dueno_s: float = 10.0              # hace cuánto que no hay nadie cerca
    radio_dueno_diag: float = 3.0          # veces la diagonal del paquete

    def _evaluar(self, obs: ObservacionCamara, ctx: Contexto) -> List[Hallazgo]:
        paquetes = [t for t in obs.confirmados("paquete") if t.quieto]
        if not paquetes:
            return []
        personas = obs.confirmados("persona")
        mem = ctx.mem(self).setdefault("visto_dueno", {})

        fuera: List[Hallazgo] = []
        for p in paquetes:
            clave = (obs.camera_id, p.track_id)
            radio = self.radio_dueno_diag * _diag(p.bbox_xyxy)
            cerca = [q for q in personas if _dist_centros(p.bbox_xyxy, q.bbox_xyxy) <= radio]
            if cerca:
                mem[clave] = obs.ts
                continue

            ultimo = mem.get(clave)
            if ultimo is None:
                # Nunca se vio a nadie con él: puede ser mobiliario. Se le
                # exige el doble de tiempo antes de molestar a nadie.
                if p.quieto_s < self.abandono_s * 2:
                    continue
                sin_dueno = p.quieto_s
                motivo = f"bulto quieto {p.quieto_s:.0f} s, nunca hubo nadie cerca"
                conf = 0.45
            else:
                sin_dueno = obs.ts - ultimo
                if sin_dueno < self.sin_dueno_s or p.quieto_s < self.abandono_s:
                    continue
                motivo = (f"paquete quieto {p.quieto_s:.0f} s, "
                          f"sin nadie cerca hace {sin_dueno:.0f} s")
                conf = min(0.85, 0.55 + 0.30 * min(sin_dueno / 60.0, 1.0))

            z = ctx.zona(p)
            sev = Severidad.AVISO
            if z is not None and z.tipo in ("restringida", "acceso"):
                sev = Severidad.ALERTA

            fuera.append(Hallazgo(
                tipo=self.tipo, severidad=sev, confianza=conf,
                camera_id=obs.camera_id, zona=z.nombre if z else None,
                motivo=motivo, ts=obs.ts, track_ids=[p.track_id],
                necesita_vlm=True,         # ¿es un paquete o es una silla?
                evidencia={"quieto_s": round(p.quieto_s, 1),
                           "sin_dueno_s": round(sin_dueno, 1),
                           "score": round(p.score, 3)}))
        return fuera


# --------------------------------------------------------------------------- #
# Caída — dormida hasta que exista la cabeza
# --------------------------------------------------------------------------- #
class ReglaCaida(Regla):
    """Persona en el piso.

    **Hoy duerme**: no hay cabeza de pose ni clasificador de acción. La lógica
    está escrita y probada contra observaciones sintéticas, y acepta las dos
    formas en que puede llegar el modelo del compañero:

      a) **Clasificador de acción** — manda `AccionObs(accion="caida",
         score=...)` y acá solo se confirma persistencia.
      b) **Estimador de pose** — manda 17 keypoints y la geometría se calcula
         acá: ángulo del torso contra la vertical y altura de la cabeza contra
         la de la cadera.

    En los dos casos hace falta lo mismo después: **quietud**. Sentarse en el
    piso y caerse se ven casi igual en un frame; lo que los separa es que
    después de una caída la persona no se levanta. Por eso la regla no alerta
    con la postura sola, y por eso la severidad sube con los segundos.
    """

    tipo = "caida"
    requiere = frozenset()
    requiere_alguna = frozenset({CABEZA_POSE, CABEZA_ACCION})

    umbral_accion: float = 0.50
    angulo_torso_min: float = 55.0         # grados desde la vertical
    quieto_s: float = 2.0                  # sin levantarse
    critico_s: float = 15.0
    aspecto_tumbado: float = 1.00          # w/h de una caja de persona acostada
    permitir_heuristica_bbox: bool = False  # sin pose ni acción; ver abajo

    def puede_correr(self, obs: ObservacionCamara) -> bool:
        """`CABEZA_ACCION` sola NO alcanza para ver caídas.

        18/09, medido con la configuración de Teo —cabeza de objetos y de
        agresión prendidas, la de caídas apagada por falta de mediapipe:

            caida       despierta      <-- mentira

        El pipeline declara `CABEZA_ACCION` por tener la cabeza de agresión
        INSTALADA, que está bien: un rato sin dos personas en cuadro no es un
        rato sin cabeza. Pero esa cabeza es un MC3-18 entrenado en RWF-2000 y
        lo único que emite es `pelea`. No sabe lo que es una caída y no la va a
        emitir nunca.

        O sea que la regla se declaraba capaz de ver caídas porque había
        instalada una cabeza que no puede verlas. Es exactamente la confusión
        que dice el comentario de `ObservacionCamara.cabezas`: "no hay caídas"
        y "no hay cabeza de caídas" volvían a ser indistinguibles, y encima en
        la regla escrita para que no lo fueran.

        Ahora corre si:
          - hay estimador de pose (`CABEZA_POSE`) — el detector de caídas lo
            declara siempre, esté o no clasificando en este frame; o
          - llega una acción `caida` de verdad en este frame, que es como se
            ve un clasificador de acción que SÍ sabe de caídas.

        Con la cabeza de agresión sola, ninguna de las dos: duerme, y lo dice.
        """
        if CABEZA_POSE in obs.cabezas:
            return True
        return any(getattr(a, "accion", None) == self.tipo
                   for a in (obs.acciones or ()))

    def _evaluar(self, obs: ObservacionCamara, ctx: Contexto) -> List[Hallazgo]:
        personas = {t.track_id: t for t in obs.confirmados("persona")}
        mem = ctx.mem(self).setdefault("desde", {})
        fuera: List[Hallazgo] = []

        for a in obs.acciones:
            caido, conf, motivo = self._postura(a)
            if not caido:
                continue

            persona = self._persona_de(a, personas)
            clave = (obs.camera_id, a.track_id if a.track_id is not None
                     else (getattr(persona, "track_id", None) or _redondear(a.bbox_xyxy)))
            desde = mem.setdefault(clave, obs.ts)
            tumbado_s = obs.ts - desde

            quieto_ok = persona is None or persona.quieto or persona.quieto_s >= self.quieto_s
            if tumbado_s < self.quieto_s or not quieto_ok:
                continue                   # todavía puede ser alguien sentándose

            sev = Severidad.ALERTA
            if tumbado_s >= self.critico_s:
                sev = Severidad.CRITICO
                motivo += f", sin levantarse hace {tumbado_s:.0f} s"

            z = ctx.zona(persona) if persona is not None else None
            fuera.append(Hallazgo(
                tipo=self.tipo, severidad=sev,
                confianza=min(0.95, conf + 0.10 * min(tumbado_s / 10.0, 1.0)),
                camera_id=obs.camera_id, zona=z.nombre if z else None,
                motivo=motivo, ts=obs.ts,
                track_ids=[t for t in (getattr(persona, "track_id", None),
                                       a.track_id) if t is not None],
                global_id=getattr(persona, "global_id", None),
                necesita_vlm=conf < 0.75,
                evidencia={"fuente": a.fuente, "score": round(a.score, 3),
                           "tumbado_s": round(tumbado_s, 1),
                           "quieto_s": round(getattr(persona, "quieto_s", 0.0), 1)}))

        # Olvidar a los que ya se levantaron, para que la próxima caída
        # empiece a contar de cero.
        vivos = {(obs.camera_id, a.track_id) for a in obs.acciones}
        for k in [k for k in mem if k[0] == obs.camera_id and k not in vivos
                  and not isinstance(k[1], tuple)]:
            mem.pop(k, None)
        return fuera

    # -------------------------------------------------------------- #
    def _postura(self, a: AccionObs) -> Tuple[bool, float, str]:
        """(¿está tumbado?, confianza, motivo). Los dos caminos posibles."""
        if a.accion == "caida" and a.score >= self.umbral_accion:
            return True, min(0.90, a.score), "postura de caída (clasificador)"

        if a.tiene_keypoints:
            ang = _angulo_torso(a)
            if ang is not None and ang >= self.angulo_torso_min:
                conf = 0.55 + 0.30 * min((ang - self.angulo_torso_min) / 35.0, 1.0)
                return True, conf, f"torso a {ang:.0f}° de la vertical"

            cabeza = a.kp("nariz")
            cad_i, cad_d = a.kp("cadera_izq"), a.kp("cadera_der")
            if cabeza and (cad_i or cad_d):
                cy = np.mean([p[1] for p in (cad_i, cad_d) if p])
                if cabeza[1] >= cy:        # y crece hacia abajo
                    return True, 0.60, "cabeza a la altura de la cadera o por debajo"
            return False, 0.0, ""

        if self.permitir_heuristica_bbox:
            # Último recurso: caja más ancha que alta. Sirve para probar el
            # camino completo sin la cabeza, pero confunde a cualquiera que
            # se agache y NO debería usarse en producción.
            x1, y1, x2, y2 = a.bbox_xyxy
            h = max(y2 - y1, 1e-3)
            if (x2 - x1) / h >= self.aspecto_tumbado:
                return True, 0.40, "caja más ancha que alta (heurística, sin pose)"
        return False, 0.0, ""

    def _persona_de(self, a: AccionObs, personas: Dict[int, Any]) -> Optional[Any]:
        if a.track_id is not None and a.track_id in personas:
            return personas[a.track_id]
        mejor, mejor_iou = None, 0.0
        for t in personas.values():
            v = _iou(a.bbox_xyxy, t.bbox_xyxy)
            if v > mejor_iou:
                mejor, mejor_iou = t, v
        return mejor if mejor_iou >= 0.30 else None


# --------------------------------------------------------------------------- #
# Acciones externas — el enchufe para los modelos que faltan
# --------------------------------------------------------------------------- #
@dataclass
class DefAccion:
    """Cómo tratar una etiqueta que produce un modelo externo."""
    accion: str
    tipo_evento: str
    severidad: Severidad = Severidad.ALERTA
    umbral: float = 0.55
    persistencia_s: float = 1.0
    necesita_vlm: bool = True
    motivo: str = ""

    # Acciones que son de la ESCENA y no de una persona. Un clasificador de
    # clips (agresión, aglomeración) mira el cuadro entero y no puede decir
    # "esta persona pelea": la pelea es el conjunto. Con esto la regla sabe
    # que no tiene que esperar un `track_id` y que la identidad hay que
    # reconstruirla mirando quién está adentro de la región.
    por_camara: bool = False

    # Una pelea con una sola persona en cuadro no es una pelea. El
    # clasificador ya tiene su propia compuerta, pero la regla no le cree de
    # palabra: es la misma política que "arma nunca es crítico sola".
    min_personas: int = 0

    # Cuánto tiene que estar una persona adentro de la región para contarla
    # como participante.
    contencion_min: float = 0.50


# Agregar un modelo de acción nuevo = agregar una línea acá. No hay que tocar
# el motor ni escribir una regla.
ACCIONES_EXTERNAS: Tuple[DefAccion, ...] = (
    DefAccion("robo", "robo", Severidad.CRITICO, umbral=0.60,
              persistencia_s=0.8, motivo="acción de robo detectada"),
    # pelea: la produce `fight/detector_agresion.py` (MC3-18 sobre el clip).
    # bal. acc medida 88,75%, y lo que peor separa es el juego brusco — por eso
    # nunca sale crítica sola: escala solo por correlación (ver motor_fusion).
    DefAccion("pelea", "pelea", Severidad.ALERTA, umbral=0.60,
              persistencia_s=1.5, motivo="forcejeo entre personas",
              por_camara=True, min_personas=2),
    DefAccion("vandalismo", "vandalismo", Severidad.ALERTA, umbral=0.60,
              persistencia_s=1.5, motivo="daño a la propiedad"),
    DefAccion("aglomeracion", "aglomeracion", Severidad.AVISO, umbral=0.55,
              persistencia_s=3.0, motivo="concentración de personas",
              por_camara=True, min_personas=4),
)


class ReglaAccionExterna(Regla):
    """Convierte cualquier etiqueta de `ACCIONES_EXTERNAS` en hallazgo.

    Existe para que el modelo que empuje el compañero entre sin cirugía: se
    llena `AccionObs` con `acciones_desde_clasificador()` y listo. `caida` se
    excluye porque tiene su propia regla, que además mira quietud.
    """

    tipo = "accion_externa"
    requiere = frozenset()
    requiere_alguna = frozenset({CABEZA_ACCION, CABEZA_POSE})

    def __init__(self, definiciones: Sequence[DefAccion] = ACCIONES_EXTERNAS,
                 **kw: Any) -> None:
        super().__init__(**kw)
        self.defs = {d.accion: d for d in definiciones if d.accion != "caida"}

    def _evaluar(self, obs: ObservacionCamara, ctx: Contexto) -> List[Hallazgo]:
        if not self.defs:
            return []
        mem = ctx.mem(self).setdefault("desde", {})
        personas = [t for t in obs.confirmados("persona")]
        por_id = {t.track_id: t for t in personas}
        fuera: List[Hallazgo] = []

        vistas = set()
        for a in obs.acciones:
            d = self.defs.get(a.accion)
            if d is None or a.score < d.umbral:
                continue

            participantes = self._participantes(a, d, personas, por_id)
            if len(participantes) < d.min_personas:
                continue

            clave = self._clave(obs.camera_id, a, d)
            vistas.add(clave)
            desde = mem.setdefault(clave, obs.ts)
            if obs.ts - desde < d.persistencia_s:
                continue

            ancla = por_id.get(a.track_id) if a.track_id is not None else None
            if ancla is None and participantes:
                ancla = participantes[0]
            z = ctx.zona(ancla) if ancla is not None else None
            gids = [g for g in (getattr(p, "global_id", None)
                                for p in participantes) if g is not None]
            fuera.append(Hallazgo(
                tipo=d.tipo_evento, severidad=d.severidad,
                confianza=min(0.95, a.score), camera_id=obs.camera_id,
                zona=z.nombre if z else None,
                motivo=d.motivo or f"acción {a.accion}", ts=obs.ts,
                track_ids=sorted({p.track_id for p in participantes}),
                global_id=getattr(ancla, "global_id", None),
                necesita_vlm=d.necesita_vlm,
                evidencia={"accion": a.accion, "score": round(a.score, 3),
                           "fuente": a.fuente,
                           "persistencia_s": round(obs.ts - desde, 1),
                           "participantes": len(participantes),
                           "global_ids": sorted(gids)}))

        for k in [k for k in mem if k[0] == obs.camera_id and k not in vistas]:
            mem.pop(k, None)
        return fuera

    # ------------------------------------------------------------------ #
    @staticmethod
    def _clave(cam: str, a: AccionObs, d: DefAccion) -> Tuple:
        """La clave con la que se acumula la persistencia.

        Redondear la caja —lo que hacía la primera versión cuando no había
        `track_id`— es un bug silencioso: la caja de una pelea se mueve, cada
        frame genera una clave nueva, `desde` se reinicia y la persistencia
        **nunca** se cumple. Medido: con la caja quieta sale el evento, con la
        caja desplazándose 3 px por frame no sale ninguno, para siempre y sin
        una sola excepción.

        Sin `track_id` no hay identidad que separar, así que la honesta es la
        cámara: dos instancias anónimas de la misma acción en una cámara no
        son distinguibles y fundirlas es lo correcto.
        """
        if a.track_id is not None and not d.por_camara:
            return (cam, a.accion, a.track_id)
        return (cam, a.accion)

    @staticmethod
    def _participantes(a: AccionObs, d: DefAccion,
                       personas: Sequence[Any],
                       por_id: Dict[int, Any]) -> List[Any]:
        """Quiénes están adentro de la región de la acción.

        Un clasificador de clips dice "acá hay una pelea", no quién pelea. Sin
        esta reconstrucción el hallazgo sale con `track_ids=[]` y
        `global_id=None`, y entonces **no se puede correlacionar con nada**:
        ni con el arma que lleva uno de los dos, ni con la caída del otro. Es
        justo la correlación lo que convierte una pelea en una agresión.

        Se mide contención (la persona adentro de la región), no IoU: la
        región es la unión de varias personas, así que cada una por separado
        tiene IoU bajo contra ella.
        """
        if a.track_id is not None and not d.por_camara:
            p = por_id.get(a.track_id)
            return [p] if p is not None else []
        caja = tuple(float(v) for v in a.bbox_xyxy)
        if caja[2] <= caja[0] or caja[3] <= caja[1]:
            return list(personas)
        return [p for p in personas
                if _contencion(p.bbox_xyxy, caja) >= d.contencion_min]


# --------------------------------------------------------------------------- #
# Conjunto por defecto
# --------------------------------------------------------------------------- #
def reglas_por_defecto() -> List[Regla]:
    return [
        ReglaIncendio(),
        ReglaArma(),
        ReglaIntrusion(),
        ReglaMerodeo(),
        ReglaPaqueteAbandonado(),
        ReglaCaida(),
        ReglaAccionExterna(),
    ]


# --------------------------------------------------------------------------- #
# Geometría
# --------------------------------------------------------------------------- #
def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _contencion(chico: Sequence[float], grande: Sequence[float]) -> float:
    """Qué fracción de `chico` está adentro de `grande`."""
    ax1, ay1, ax2, ay2 = (float(v) for v in chico)
    bx1, by1, bx2, by2 = (float(v) for v in grande)
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = max((ax2 - ax1) * (ay2 - ay1), 1e-6)
    return (ix * iy) / area


def _diag(caja: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in caja)
    return float(math.hypot(x2 - x1, y2 - y1))


def _centro(caja: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in caja)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _dist_centros(a: Sequence[float], b: Sequence[float]) -> float:
    ca, cb = _centro(a), _centro(b)
    return float(math.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def _redondear(caja: Sequence[float]) -> Tuple[int, ...]:
    """Clave estable para una caja sin track. Redondeada a 32 px para que un
    temblor de la detección no cree una clave nueva en cada frame."""
    return tuple(int(v) // 32 for v in caja)


def _portador(arma: Any, personas: Sequence[Any], minimo: float) -> Optional[Any]:
    mejor, mejor_c = None, 0.0
    for p in personas:
        c = _contencion(arma.bbox_xyxy, p.bbox_xyxy)
        if c > mejor_c:
            mejor, mejor_c = p, c
    return mejor if mejor_c >= minimo else None


def _angulo_torso(a: AccionObs) -> Optional[float]:
    """Ángulo del torso contra la vertical, en grados. 0 = de pie."""
    hi, hd = a.kp("hombro_izq"), a.kp("hombro_der")
    ci, cd = a.kp("cadera_izq"), a.kp("cadera_der")
    hombros = [p for p in (hi, hd) if p]
    caderas = [p for p in (ci, cd) if p]
    if not hombros or not caderas:
        return None
    hx = float(np.mean([p[0] for p in hombros]))
    hy = float(np.mean([p[1] for p in hombros]))
    cx = float(np.mean([p[0] for p in caderas]))
    cy = float(np.mean([p[1] for p in caderas]))
    dx, dy = cx - hx, cy - hy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return float(math.degrees(math.atan2(abs(dx), abs(dy))))
