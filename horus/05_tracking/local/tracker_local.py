# -*- coding: utf-8 -*-
"""
tracker_local.py · HORUS — tracking por cámara, ventana temporal por clase.

Qué resuelve
------------
El motor de objetos contesta "qué hay en ESTE frame". La fusión necesita otra
cosa: "qué hay en esta escena y desde hace cuánto". Ese salto lo da el
tracking, y de paso arregla los dos problemas que reportó la prueba de campo:

1. **La persona se pierde con movimiento rápido.** Sin predicción, comparar la
   caja de t-1 contra la de t da IoU 0 en cuanto alguien se mueve más que su
   propio ancho. Acá cada track predice con Kalman dónde debería estar, y la
   asociación se hace contra esa predicción.

2. **Paquete y celular parpadean.** Un fantasma aparece en un frame y no
   vuelve; un objeto real acumula historia. La confirmación por clase mata el
   parpadeo sin tocar el modelo ni subir umbrales (subir el umbral filtra por
   confianza y se lleva puestos los positivos débiles — está medido en
   `calibracion-umbrales-objetos`).

Asociación en dos etapas (ByteTrack)
------------------------------------
La idea que hace la diferencia: las detecciones de score BAJO casi nunca son
objetos nuevos, pero muy seguido son un objeto conocido momentáneamente
ocluido o borroso. Entonces:

  - Etapa 1: los tracks se pelean por las detecciones de score alto.
  - Etapa 2: los tracks que quedaron sin pareja miran las de score bajo.
  - Una detección de score bajo puede SOSTENER un track, pero **nunca puede
    crear uno**. Ahí está la asimetría que recupera personas sin inventar
    fantasmas.

Para que la etapa 2 tenga con qué trabajar, el motor tiene que entregar
detecciones por debajo del umbral de campo. Ver `config_motor_para_tracking()`
más abajo: el motor baja el piso a 0.10 y **el tracker se queda con la tabla
de umbrales por clase**. No es opcional; con el motor filtrando en 0.25 la
etapa 2 no recibe nada y el tracker se degrada a un SORT común.

Qué NO hace
-----------
No cruza cámaras (eso es `global_reid/`), no decide eventos (eso es
`06_fusion_decision/`) y no toca la GPU: corre sobre las detecciones que ya
están en el host, después del único sync del motor. Solo numpy.

Uso
---
    from tracker_local import TrackerMultiCamara, ConfigTracker

    tk = TrackerMultiCamara(ConfigTracker(fps=10.0))

    for res in eng.infer_batch(frames, camera_ids=cams):
        tracks = tk.actualizar_desde_resultado(res)
        for t in tracks:
            if t.confirmado:
                print(t.track_id, t.clase, t.bbox_xyxy, t.visto_s, t.quieto)
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from itertools import count
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
_PADRE = os.path.dirname(_AQUI)
for _p in (_AQUI, _PADRE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from comun import (  # noqa: E402
    a_array, asignar, centro, contencion_matriz, cxcyah_a_xyxy, diagonal,
    expandir, iou_matriz, punto_pie, xyxy_a_cxcyah,
)
from kalman import CHI2_95_4GL, FiltroKalman  # noqa: E402


# --------------------------------------------------------------------------- #
# Parámetros por clase
# --------------------------------------------------------------------------- #
@dataclass
class ParametrosClase:
    """La 'ventana por clase' del diagrama, en segundos (no en frames).

    En segundos a propósito: el pipeline corre a 8-12 FPS según la carga, y
    una ventana medida en frames cambia de significado cuando cambia el FPS.
    """

    ventana_s: float = 1.5        # cuánto sobrevive un track sin verse
    confirmar_s: float = 0.30     # cuánto tarda en confirmarse
    tentativo_s: float = 0.20     # tolerancia de un track todavía sin confirmar
    iou_alto: float = 0.25        # umbral de asociación, etapa 1
    iou_bajo: float = 0.45        # etapa 2 (más estricto: las cajas son ruidosas)
    score_alto: float = 0.40      # de acá para arriba, etapa 1 y puede nacer
    score_bajo: float = 0.15      # de acá para arriba, etapa 2 (solo sostiene)
    kalman: bool = True           # False para lo que no se traslada
    gating: bool = True           # prohibir asociaciones geométricamente absurdas

    # Etapa 3, recuperación. `buffer` agranda las cajas antes de medir IoU,
    # para que un objeto que se movió más que su propio ancho siga teniendo
    # solape con su predicción. 0 desactiva la etapa.
    buffer_recuperacion: float = 0.8
    iou_recuperacion: float = 0.20

    # Duplicados dentro de la clase. El NMS del motor es por IoU, y el IoU NO
    # ve las cajas anidadas: el torso de una persona dentro de su cuerpo
    # entero da IoU ~0.50 y sobrevive. Por eso hacen falta dos criterios, y
    # separados: son cosas distintas y quieren umbrales distintos.
    #
    #   dedup_iou         dos cajas que son casi la misma caja.
    #   dedup_contencion  una caja adentro de la otra. Va ALTO a propósito:
    #                     una persona parcialmente tapada por otra puede
    #                     quedar 60-70% dentro de su caja, y esa es una
    #                     persona de verdad que no se puede perder. Un torso
    #                     o unas piernas quedan al 95-100%.
    dedup_iou: float = 0.65
    dedup_contencion: float = 0.85
    nacer_iou_max: float = 0.60         # no nace un track encima de otro
    nacer_contencion_max: float = 0.85
    # Para las clases que se apoyan en el piso (personas): una caja contenida
    # que pisa MÁS ABAJO que la que la contiene no es una parte de ella, es
    # otro objeto parado más cerca de la cámara. Ver `_pisa_mas_abajo`.
    respetar_pie: bool = True


# Los números salen de tres lugares: los umbrales medidos en
# `calibracion-umbrales-objetos`, lo observado en la prueba de campo, y la
# física de cada clase. No son valores de catálogo.
#
#   persona   iou_alto bajo (0.25) porque se mueve rápido y la predicción no
#             es perfecta; confirmación corta porque perder una persona en un
#             robo cuesta más que dibujar una caja de más.
#   pistola   confirmación cortísima: es la clase donde la latencia mata. Y
#   cuchillo  el dataset no las tiene desplegables todavía, así que igual las
#             verifica el VLM aguas abajo — acá conviene ser permisivo.
#   humo      SIN Kalman: el humo no se traslada, se expande. Un modelo de
#             velocidad constante pelea contra eso y empeora la predicción.
#             iou_alto 0.10 porque la caja de humo cambia de forma sola.
#   llama     igual pero menos amorfa.
#   celular   confirmación larga (0.8 s): es una de las dos clases que
#             parpadea en cámara. Nace solo con score >= 0.45 (el medido).
#   paquete   ventana larguísima (5 s): un paquete se queda quieto y hay que
#             sobrevivir a que alguien pase por delante y lo tape. Sin Kalman,
#             no se mueve. Confirmación de 1 s contra el parpadeo.
PARAMETROS_POR_CLASE: Dict[str, ParametrosClase] = {
    "persona": ParametrosClase(
        ventana_s=1.5, confirmar_s=0.30, iou_alto=0.25, iou_bajo=0.45,
        score_alto=0.25, score_bajo=0.10, kalman=True,
        buffer_recuperacion=1.0, iou_recuperacion=0.20,
        dedup_iou=0.60, dedup_contencion=0.85,
        nacer_iou_max=0.55, nacer_contencion_max=0.85),
    "pistola": ParametrosClase(
        ventana_s=1.0, confirmar_s=0.20, iou_alto=0.20, iou_bajo=0.40,
        score_alto=0.35, score_bajo=0.15, kalman=True,
        buffer_recuperacion=1.0, iou_recuperacion=0.20,
        dedup_iou=0.60, dedup_contencion=0.85,
        nacer_iou_max=0.55, nacer_contencion_max=0.85),
    "cuchillo": ParametrosClase(
        ventana_s=1.0, confirmar_s=0.20, iou_alto=0.20, iou_bajo=0.40,
        score_alto=0.35, score_bajo=0.15, kalman=True,
        buffer_recuperacion=1.0, iou_recuperacion=0.20,
        dedup_iou=0.60, dedup_contencion=0.85,
        nacer_iou_max=0.55, nacer_contencion_max=0.85),
    "humo": ParametrosClase(
        ventana_s=3.0, confirmar_s=0.60, iou_alto=0.10, iou_bajo=0.30,
        score_alto=0.35, score_bajo=0.15, kalman=False, gating=False,
        buffer_recuperacion=0.3, iou_recuperacion=0.10,
        dedup_iou=0.70, dedup_contencion=0.90,
        nacer_iou_max=0.65, nacer_contencion_max=0.90, respetar_pie=False),
    "llama": ParametrosClase(
        ventana_s=2.0, confirmar_s=0.40, iou_alto=0.15, iou_bajo=0.35,
        score_alto=0.35, score_bajo=0.15, kalman=False, gating=False,
        buffer_recuperacion=0.3, iou_recuperacion=0.12,
        dedup_iou=0.65, dedup_contencion=0.88,
        nacer_iou_max=0.60, nacer_contencion_max=0.88, respetar_pie=False),
    "celular": ParametrosClase(
        ventana_s=0.8, confirmar_s=0.80, iou_alto=0.30, iou_bajo=0.50,
        score_alto=0.45, score_bajo=0.25, kalman=True,
        buffer_recuperacion=0.4, iou_recuperacion=0.25,
        dedup_iou=0.65, dedup_contencion=0.85,
        nacer_iou_max=0.60, nacer_contencion_max=0.85),
    "paquete": ParametrosClase(
        ventana_s=5.0, confirmar_s=1.00, iou_alto=0.35, iou_bajo=0.55,
        score_alto=0.40, score_bajo=0.25, kalman=False,
        buffer_recuperacion=0.2, iou_recuperacion=0.25,
        dedup_iou=0.65, dedup_contencion=0.88,
        nacer_iou_max=0.60, nacer_contencion_max=0.88),
}

# Piso que tiene que usar el motor de inferencia para que la etapa 2 exista.
# Por debajo de esto no hay nada útil, solo ruido de anclas.
PISO_MOTOR: float = 0.10


def config_motor_para_tracking() -> Dict[str, Any]:
    """Overrides para `EngineConfig` cuando la salida va a un tracker.

        from objects_engine import EngineConfig
        cfg = EngineConfig(**config_motor_para_tracking(), pesos=...)

    Dos cambios:

      score_thresh = 0.10   el tracker necesita ver lo que el umbral de campo
                            descarta, para la etapa 2. Los umbrales por clase
                            no desaparecen: se aplican acá, como `score_alto`
                            en `PARAMETROS_POR_CLASE`.
      devolver_embedding    lo consume global_reid/.

    18/09 — por qué esto era más largo y por qué rompía todo:

    Devolvía además `umbral_por_clase=None` y `confirmar_frames=0`, dos
    parámetros que `EngineConfig` NO tiene. `EngineConfig(**esto)` explotaba
    con `TypeError: got an unexpected keyword argument 'umbral_por_clase'`, y
    como esta función es la única puerta por la que el pipeline construye el
    motor de objetos, el servicio moría al arrancar. Cada vez. Desde siempre.

    Las diez suites daban verde porque ninguna pasaba por acá: el pipeline
    acepta un `motor_objetos` ya armado y las pruebas le pasan uno de mentira,
    que es justo lo que hace falta para probar la fusión sin GPU. La única
    forma de tocar esta línea era arrancar el servicio de verdad, con pesos de
    verdad, en una máquina de verdad.

    Los dos parámetros pedían apagar cosas que el motor no hace: no tiene
    tabla de umbrales por clase (eso vive en el tracker) ni confirmación
    K-de-N (eso lo hace el tracker, mejor, asociando por identidad). O sea que
    pedían apagar algo que ya estaba apagado, y de paso mataban el arranque.

    `caso_la_config_del_motor_existe()` en probar_tracking.py verifica ahora
    que cada clave de acá sea un campo real de EngineConfig.
    """
    return {
        "score_thresh": PISO_MOTOR,
        "devolver_embedding": True,      # lo consume global_reid/
    }


@dataclass
class ConfigTracker:
    fps: float = 10.0
    por_clase: Dict[str, ParametrosClase] = field(
        default_factory=lambda: {k: ParametrosClase(**vars(v))
                                 for k, v in PARAMETROS_POR_CLASE.items()})
    defecto: ParametrosClase = field(default_factory=ParametrosClase)

    # Quietud / permanencia — lo que consumen merodeo y paquete abandonado.
    historia_s: float = 30.0      # cuánto centro guardar por track
    radio_quieto: float = 0.15    # fracción de la diagonal de la caja
    quieto_s: float = 1.0         # cuánto tiene que estar quieto para contar

    max_tracks_por_clase: int = 64   # techo duro; protege contra una tormenta
    emitir_no_confirmados: bool = True

    # Duplicados ENTRE clases. El NMS del motor corre por clase, así que la
    # misma región puede salir a la vez como persona y como paquete: son dos
    # pozos de tracks distintos y terminan siendo dos cajas sobre la misma
    # cosa. Con este solape se queda la de mejor score relativo a SU propio
    # umbral (comparar scores crudos entre clases no significa nada).
    # 1.0 desactiva el filtro.
    dedup_entre_clases: float = 0.80

    def params(self, clase: str) -> ParametrosClase:
        return self.por_clase.get(clase, self.defecto)

    def frames(self, segundos: float, minimo: int = 1) -> int:
        return max(minimo, int(round(segundos * max(self.fps, 1e-3))))


# --------------------------------------------------------------------------- #
# Salida
# --------------------------------------------------------------------------- #
@dataclass
class TrackLocal:
    """Lo que ve la capa de fusión. Mismo espíritu que `Detection` y
    `SegResult`: un dataclass plano, sin tensores, serializable."""

    track_id: int
    clase: str
    class_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    camera_id: str
    frame_idx: int
    ts: float

    estado: str = "tentativo"       # "tentativo" | "confirmado" | "perdido"
    hits: int = 0                   # veces que se asoció a una detección
    edad: int = 0                   # frames desde que nació
    frames_sin_ver: int = 0

    ts_nacimiento: float = 0.0
    visto_s: float = 0.0            # tiempo desde el nacimiento
    velocidad: Tuple[float, float] = (0.0, 0.0)   # px/frame del centro
    recorrido: float = 0.0          # px acumulados
    desplazamiento: float = 0.0     # px entre el primer centro y el actual
    radio_permanencia: float = 0.0  # radio de la región donde estuvo
    quieto: bool = False
    quieto_s: float = 0.0

    needs_vlm: bool = False
    embedding: Optional[np.ndarray] = None   # lo llena global_reid/
    global_id: Optional[int] = None          # lo llena global_reid/

    @property
    def confirmado(self) -> bool:
        return self.estado == "confirmado"

    @property
    def visible(self) -> bool:
        return self.frames_sin_ver == 0

    def punto_pie(self) -> Tuple[float, float]:
        return punto_pie(self.bbox_xyxy)

    def centro(self) -> Tuple[float, float]:
        return centro(self.bbox_xyxy)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding")               # el vector no va al log
        d["bbox_xyxy"] = [round(float(v), 1) for v in self.bbox_xyxy]
        return d


# --------------------------------------------------------------------------- #
# Track interno
# --------------------------------------------------------------------------- #
_CONTADOR = count(1)


class _Track:
    __slots__ = ("id", "clase", "class_id", "caja", "score", "p", "kf",
                 "media", "cov", "hits", "edad", "sin_ver", "estado",
                 "ts_nac", "frame_nac", "centros", "recorrido", "quieto_desde",
                 "needs_vlm", "ultimo_ts", "ultimo_frame", "global_id",
                 "embedding")

    def __init__(self, clase: str, class_id: int, caja: Sequence[float],
                 score: float, p: ParametrosClase, kf: Optional[FiltroKalman],
                 ts: float, frame_idx: int, max_centros: int) -> None:
        self.id = next(_CONTADOR)
        self.clase = clase
        self.class_id = class_id
        self.caja = np.asarray(caja, dtype=np.float64).reshape(4)
        self.score = float(score)
        self.p = p
        self.kf = kf
        self.media = self.cov = None
        if kf is not None:
            self.media, self.cov = kf.iniciar(xyxy_a_cxcyah(self.caja))

        self.hits = 1
        self.edad = 1
        self.sin_ver = 0
        self.estado = "tentativo"
        self.ts_nac = ts
        self.frame_nac = frame_idx
        self.ultimo_ts = ts
        self.ultimo_frame = frame_idx

        self.centros: Deque[Tuple[float, float]] = deque(maxlen=max_centros)
        self.centros.append(centro(self.caja))
        self.recorrido = 0.0
        self.quieto_desde: Optional[float] = None
        self.needs_vlm = False
        self.global_id: Optional[int] = None
        self.embedding: Optional[np.ndarray] = None

    # -------------------------------------------------------------- #
    def predecir(self) -> np.ndarray:
        """Dónde debería estar este track en el frame que viene."""
        self.edad += 1
        if self.kf is None or self.media is None:
            return self.caja
        self.media, self.cov = self.kf.predecir(self.media, self.cov)
        return np.asarray(cxcyah_a_xyxy(self.media), dtype=np.float64)

    def corregir(self, caja: Sequence[float], score: float,
                 ts: float, frame_idx: int, radio_quieto: float) -> None:
        caja = np.asarray(caja, dtype=np.float64).reshape(4)
        if self.kf is not None and self.media is not None:
            self.media, self.cov = self.kf.corregir(
                self.media, self.cov, xyxy_a_cxcyah(caja))
            self.caja = np.asarray(cxcyah_a_xyxy(self.media), dtype=np.float64)
        else:
            self.caja = caja

        self.score = float(score)
        self.hits += 1
        self.sin_ver = 0
        self.ultimo_ts = ts
        self.ultimo_frame = frame_idx

        c = centro(self.caja)
        if self.centros:
            prev = self.centros[-1]
            self.recorrido += float(np.hypot(c[0] - prev[0], c[1] - prev[1]))
        self.centros.append(c)
        self._actualizar_quietud(ts, radio_quieto)

    def perdido(self) -> np.ndarray:
        """Frame sin asociación: se propaga la predicción, no se corrige."""
        self.sin_ver += 1
        if self.kf is not None and self.media is not None:
            self.caja = np.asarray(cxcyah_a_xyxy(self.media), dtype=np.float64)
        return self.caja

    # -------------------------------------------------------------- #
    def _actualizar_quietud(self, ts: float, radio_frac: float) -> None:
        """Quieto = el centro no se salió de un radio proporcional a su
        propia caja. Proporcional y no en píxeles fijos: un objeto lejano
        de 20 px y uno cercano de 400 px no pueden compartir el umbral."""
        radio = max(radio_frac * diagonal(self.caja), 2.0)
        n = min(len(self.centros), 16)
        recientes = np.asarray(list(self.centros)[-n:], dtype=np.float64)
        base = recientes.mean(axis=0)
        disp = float(np.max(np.hypot(recientes[:, 0] - base[0],
                                     recientes[:, 1] - base[1]))) if n else 0.0
        if disp <= radio:
            if self.quieto_desde is None:
                self.quieto_desde = ts
        else:
            self.quieto_desde = None

    def metricas(self) -> Tuple[float, float, float]:
        """(desplazamiento neto, radio de permanencia, reservado)."""
        pts = np.asarray(self.centros, dtype=np.float64)
        if pts.shape[0] == 0:
            return 0.0, 0.0, 0.0
        neto = float(np.hypot(*(pts[-1] - pts[0]))) if pts.shape[0] > 1 else 0.0
        base = pts.mean(axis=0)
        radio = float(np.max(np.hypot(pts[:, 0] - base[0], pts[:, 1] - base[1])))
        return neto, radio, 0.0

    def velocidad(self) -> Tuple[float, float]:
        """px/frame del centro. Del Kalman si lo hay; si no, diferencia
        finita entre los dos ultimos centros."""
        if self.media is not None:
            return float(self.media[4]), float(self.media[5])
        if len(self.centros) > 1:
            a, b = self.centros[-2], self.centros[-1]
            return float(b[0] - a[0]), float(b[1] - a[1])
        return 0.0, 0.0

    def salida(self, camera_id: str, frame_idx: int, ts: float,
               quieto_min_s: float) -> TrackLocal:
        neto, radio, _ = self.metricas()
        q_s = 0.0 if self.quieto_desde is None else max(0.0, ts - self.quieto_desde)
        vx, vy = self.velocidad()
        return TrackLocal(
            track_id=self.id,
            clase=self.clase,
            class_id=self.class_id,
            bbox_xyxy=tuple(float(v) for v in self.caja),
            score=self.score,
            camera_id=camera_id,
            frame_idx=frame_idx,
            ts=ts,
            estado=self.estado,
            hits=self.hits,
            edad=self.edad,
            frames_sin_ver=self.sin_ver,
            ts_nacimiento=self.ts_nac,
            visto_s=max(0.0, ts - self.ts_nac),
            velocidad=(vx, vy),
            recorrido=self.recorrido,
            desplazamiento=neto,
            radio_permanencia=radio,
            quieto=q_s >= quieto_min_s,
            quieto_s=q_s,
            needs_vlm=self.needs_vlm,
            embedding=self.embedding,
            global_id=self.global_id,
        )


# --------------------------------------------------------------------------- #
# Tracker de una cámara
# --------------------------------------------------------------------------- #
class TrackerLocal:
    """Un tracker por cámara. Los tracks de clases distintas nunca se
    asocian entre sí: una persona no puede convertirse en un paquete, y
    permitirlo solo agrega modos de falla."""

    def __init__(self, camera_id: str = "cam-0",
                 cfg: Optional[ConfigTracker] = None) -> None:
        self.camera_id = camera_id
        self.cfg = cfg or ConfigTracker()
        self._kf = FiltroKalman()
        self._tracks: Dict[str, List[_Track]] = {}
        self._frame = 0
        self._descartados_por_techo = 0
        self.suprimidas_duplicadas = 0     # por solape con otra deteccion
        self.suprimidas_entre_clases = 0   # misma region, dos clases
        self.nacimientos_bloqueados = 0    # querian nacer encima de un track

    # -------------------------------------------------------------- #
    def actualizar(self, detecciones: Iterable[Any],
                   frame_idx: Optional[int] = None,
                   ts: Optional[float] = None) -> List[TrackLocal]:
        """Un frame. `detecciones` puede ser una lista de `Detection` del
        motor de objetos o cualquier objeto con .bbox_xyxy/.score/.label
        (duck typing, para no atar esta capa al paquete de las cabezas)."""
        ts = time.time() if ts is None else float(ts)
        if frame_idx is None:
            frame_idx = self._frame
        self._frame = frame_idx + 1

        por_clase: Dict[str, List[Any]] = {}
        for d in detecciones or ():
            por_clase.setdefault(_label(d), []).append(d)

        por_clase = self._dedup_entre_clases(por_clase)

        # Las clases sin detecciones este frame también se procesan: sus tracks
        # tienen que envejecer, no quedar congelados esperando.
        for clase in list(self._tracks.keys()):
            por_clase.setdefault(clase, [])

        salida: List[TrackLocal] = []
        for clase, dets in por_clase.items():
            salida.extend(self._actualizar_clase(clase, dets, frame_idx, ts))
        return salida

    def actualizar_desde_resultado(self, res: Any,
                                   usar_crudas: bool = True) -> List[TrackLocal]:
        """Toma un `FrameResult` de objects_engine.

        `usar_crudas=True` usa `detecciones_crudas` (lo previo a la
        confirmación K-de-N del motor) porque el tracker hace esa
        confirmación mejor. Si el motor ya viene configurado con
        `config_motor_para_tracking()`, ambas listas son la misma.
        """
        dets = None
        if usar_crudas:
            dets = getattr(res, "detecciones_crudas", None)
        if dets is None:
            dets = getattr(res, "detections", res)

        cam = getattr(res, "camera_id", self.camera_id)
        if cam != self.camera_id:
            raise ValueError(
                f"este TrackerLocal es de {self.camera_id!r} y llegó un frame "
                f"de {cam!r}. Usá TrackerMultiCamara.")

        return self.actualizar(dets,
                               frame_idx=getattr(res, "frame_idx", None),
                               ts=None)

    # -------------------------------------------------------------- #
    def _actualizar_clase(self, clase: str, dets: List[Any],
                          frame_idx: int, ts: float) -> List[TrackLocal]:
        c = self.cfg
        p = c.params(clase)
        tracks = self._tracks.setdefault(clase, [])

        # 0. Sacar los duplicados de esta clase ANTES de asociar.
        #
        #    El motor ya corrió NMS, pero el NMS mide IoU y el IoU no ve las
        #    cajas anidadas: el torso de una persona adentro de su cuerpo
        #    entero da IoU ~0.50 y sobrevive. Con el piso del motor en 0.10
        #    (que es lo que la etapa 2 necesita) sobreviven varias por
        #    persona, y como la asignación es uno a uno, la que gana se lleva
        #    el track y TODAS las demás nacen como tracks nuevos. De ahí salen
        #    las cinco cajas apiladas sobre la misma persona.
        dets = self._deduplicar(dets, p)

        # 1. Predecir. Se hace SIEMPRE, incluso sin detecciones: es lo que
        #    permite que un track sobreviva a una oclusión.
        cajas_pred = np.asarray([t.predecir() for t in tracks],
                                dtype=np.float64).reshape(-1, 4)

        # 2. Partir las detecciones por confianza.
        altas, bajas = [], []
        for d in dets:
            s = _score(d)
            if s >= p.score_alto:
                altas.append(d)
            elif s >= p.score_bajo:
                bajas.append(d)

        libres = list(range(len(tracks)))

        # 3. Etapa 1 — tracks contra detecciones confiables, IoU estricto.
        #    Estricto a propósito: es la etapa que tiene que resolver bien una
        #    multitud, donde dos personas cercanas compiten por la misma caja.
        libres, sueltas_altas = self._emparejar(
            tracks, libres, cajas_pred, altas, p, p.iou_alto, ts, frame_idx)

        # 4. Etapa 2 — los que quedaron, contra las dudosas. Solo tracks que
        #    ya se habían confirmado alguna vez: darle detecciones débiles a
        #    un track tentativo es la receta para que un fantasma se
        #    autoconfirme con más ruido.
        if bajas:
            elegibles = [i for i in libres
                         if tracks[i].estado != "tentativo"]
            siguen_libres, _ = self._emparejar(
                tracks, elegibles, cajas_pred, bajas, p, p.iou_bajo, ts, frame_idx)
            emparejados = set(elegibles) - set(siguen_libres)
            libres = [i for i in libres if i not in emparejados]

        # 5. Etapa 3 — recuperación con IoU expandido.
        #
        #    Sin esto, un objeto que se desplaza más que su propio ancho entre
        #    frames tiene IoU exactamente 0 contra su predicción y el track
        #    muere. Y el peor caso es el arranque: un track recién nacido
        #    todavía tiene velocidad 0, así que predice que el objeto se queda
        #    quieto. Una persona corriendo nunca llega a confirmarse — que es
        #    literalmente lo que reportó la prueba de campo.
        #
        #    Agrandar las cajas antes de medir IoU convierte ese acantilado en
        #    una pendiente. Se hace en una etapa aparte y DESPUÉS de las
        #    estrictas, para no arruinar la resolución de multitudes: acá solo
        #    llegan los tracks que ya nadie reclamó.
        #
        #    Solo contra detecciones de score ALTO: recuperar algo lejano con
        #    una detección débil es como se fabrica un cambio de identidad.
        #    Y sin gating de Kalman — la covarianza de un track nuevo dice que
        #    ese salto es imposible, y justamente por eso hay que ignorarla.
        if p.buffer_recuperacion > 0 and libres and sueltas_altas:
            libres, sueltas_altas = self._emparejar(
                tracks, libres, cajas_pred, sueltas_altas, p,
                p.iou_recuperacion, ts, frame_idx,
                buffer=p.buffer_recuperacion, usar_gating=False)

        # 6. Tracks sin pareja: envejecen.
        for i in libres:
            tracks[i].perdido()
            if tracks[i].estado == "confirmado":
                tracks[i].estado = "perdido"

        # 7. Detecciones altas sin dueño: tracks nuevos.
        #
        #    Con un portero: una detección que se superpone con un track que
        #    ya existe no puede fundar uno nuevo. La dedup de arriba saca los
        #    duplicados que llegan JUNTOS en el mismo frame; esto saca los que
        #    aparecen después, cuando el húngaro ya le dio la caja buena al
        #    track y sobra una parcial que igual lo pisa.
        sueltas_altas = self._filtrar_nacimientos(tracks, sueltas_altas, p)

        max_c = c.frames(p.ventana_s)
        for d in sueltas_altas:
            if len(tracks) >= c.max_tracks_por_clase:
                self._descartados_por_techo += 1
                break
            tracks.append(_Track(
                clase=clase, class_id=_class_id(d), caja=_caja(d),
                score=_score(d), p=p,
                kf=self._kf if p.kalman else None,
                ts=ts, frame_idx=frame_idx,
                max_centros=c.frames(c.historia_s)))

        # 8. Promociones y bajas.
        min_hits = max(2, c.frames(p.confirmar_s))
        tol_tentativo = c.frames(p.tentativo_s)
        vivos: List[_Track] = []
        for t in tracks:
            if t.estado == "tentativo":
                if t.hits >= min_hits:
                    t.estado = "confirmado"
                elif t.sin_ver > tol_tentativo:
                    continue          # el fantasma muere acá, y muere barato
            elif t.estado == "perdido":
                if t.sin_ver > max_c:
                    continue
                if t.sin_ver == 0:
                    t.estado = "confirmado"
            t.needs_vlm = _needs_vlm(t)
            vivos.append(t)
        self._tracks[clase] = vivos

        emitir = c.emitir_no_confirmados
        return [t.salida(self.camera_id, frame_idx, ts, c.quieto_s)
                for t in vivos if emitir or t.estado != "tentativo"]

    # -------------------------------------------------------------- #
    @staticmethod
    def _pisa_mas_abajo(candidatas: np.ndarray, referencia: np.ndarray,
                        margen: float = 0.02) -> np.ndarray:
        """¿El borde inferior de cada candidata está por debajo del de la
        referencia? (N,) booleanos.

        Es el desempate que un solo frame sí puede dar. Una caja contenida
        dentro de otra puede ser dos cosas muy distintas:

          - el torso o las piernas de esa misma persona — un duplicado;
          - otra persona parada MÁS CERCA de la cámara — nadie que se pueda
            perder.

        Geométricamente son idénticas, pero en una cámara de vigilancia todo
        el mundo se apoya en el mismo piso: el que está más cerca pisa más
        abajo en la imagen. Una parte de una persona nunca puede tener los
        pies por debajo de los pies de esa persona.

        El margen (2% del alto de la referencia) es para que el temblor de la
        caja no cuente como estar adelante.

        OJO: esto solo perdona a la caja que está ADENTRO de la otra. A la que
        se traga a la otra no, porque una caja que envuelve a la buena y
        además sobresale un poco por abajo no es alguien más cerca: es la
        misma detección inflada.
        """
        ref = np.asarray(referencia, dtype=np.float64).reshape(4)
        alto = max(ref[3] - ref[1], 1.0)
        return np.asarray(candidatas, dtype=np.float64).reshape(-1, 4)[:, 3] \
            > ref[3] + max(2.0, margen * alto)

    def _deduplicar(self, dets: List[Any], p: ParametrosClase) -> List[Any]:
        """Supresión por solape dentro de una misma clase.

        Es un NMS más, pero con el criterio correcto: se queda la de mejor
        score y mata a las que se le superponen por IoU **o por contención**.
        Las dos matrices se calculan una sola vez y el bucle después solo
        indexa — con detecciones chicas el costo es el overhead de numpy, no
        la aritmética, así que hacer una llamada por iteración sale caro.
        """
        n = len(dets)
        if n < 2 or (p.dedup_iou >= 1.0 and p.dedup_contencion >= 1.0):
            return dets

        cajas = a_array([_caja(d) for d in dets]).astype(np.float64)
        iou = iou_matriz(cajas, cajas)
        # cont[i, j] = fracción de la caja i que está adentro de la j.
        cont = contencion_matriz(cajas, cajas)
        y2 = cajas[:, 3]
        altos = np.maximum(cajas[:, 3] - cajas[:, 1], 1.0)

        orden = np.argsort(-np.asarray([_score(d) for d in dets]))
        muerto = np.zeros(n, dtype=bool)

        for pos in range(n):
            i = int(orden[pos])
            if muerto[i]:
                continue
            resto = orden[pos + 1:]
            resto = resto[~muerto[resto]]
            if resto.size == 0:
                break
            dentro = cont[resto, i]          # la peor, adentro de la mejor
            afuera = cont[i, resto]          # la mejor, adentro de la peor
            if p.respetar_pie:
                # La que pisa más abajo está adelante: es otro objeto, no una
                # parte de este. Se la perdona aunque esté 100% adentro. Solo
                # se perdona `dentro`: si la candidata se TRAGA a la buena,
                # sobresalir un poco por abajo es una caja inflada, no alguien
                # parado más cerca.
                dentro = np.where(y2[resto] > y2[i] + max(2.0, 0.02 * altos[i]),
                                  0.0, dentro)
            mata = ((iou[i, resto] >= p.dedup_iou)
                    | (np.maximum(dentro, afuera) >= p.dedup_contencion))
            muerto[resto[mata]] = True

        vivos = [k for k in range(n) if not muerto[k]]
        self.suprimidas_duplicadas += n - len(vivos)
        return [dets[k] for k in vivos]

    def _dedup_entre_clases(self, por_clase: Dict[str, List[Any]]
                            ) -> Dict[str, List[Any]]:
        """La misma región detectada como dos clases distintas es una sola.

        El NMS del motor corre por clase, así que el torso de una persona
        puede salir a la vez como `persona` y como `paquete`. Son dos pozos de
        tracks separados y terminan siendo dos cajas sobre la misma cosa.

        Se compara el score **relativo al umbral de su propia clase**: 0.50 en
        paquete (umbral 0.40) vale más que 0.50 en persona (umbral 0.25),
        porque el segundo está mucho más cerca de su piso. Comparar los scores
        crudos entre clases no significa nada.

        **Acá se mide IoU y nunca contención**, al revés que en la dedup de
        una sola clase. Es la diferencia entre las dos: un celular, una
        pistola o un paquete en la mano de alguien están al 100% ADENTRO de
        la caja de la persona, y son exactamente lo que el sistema existe
        para ver. Filtrar por contención entre clases los borraría a todos.
        Con IoU, un celular contra una persona da ~0.02 y no se toca; solo se
        colapsan dos cajas que son la misma caja con dos etiquetas.
        """
        umbral = self.cfg.dedup_entre_clases
        if umbral >= 1.0 or len(por_clase) < 2:
            return por_clase
        plano = [(clase, d) for clase, ds in por_clase.items() for d in ds]
        n = len(plano)
        if n < 2:
            return por_clase

        cajas = a_array([_caja(d) for _, d in plano])
        iou = iou_matriz(cajas, cajas)
        clase_de = np.asarray([cl for cl, _ in plano])
        rel = np.asarray([_score(d) / max(self.cfg.params(cl).score_alto, 1e-3)
                          for cl, d in plano])

        orden = np.argsort(-rel)
        muerto = np.zeros(n, dtype=bool)
        for pos in range(n):
            i = int(orden[pos])
            if muerto[i]:
                continue
            resto = orden[pos + 1:]
            resto = resto[~muerto[resto]]
            if resto.size == 0:
                break
            # Solo se matan entre clases DISTINTAS: dentro de una clase ya se
            # ocupa `_deduplicar`, con sus umbrales propios.
            mata = (clase_de[resto] != clase_de[i]) & (iou[i, resto] >= umbral)
            muerto[resto[mata]] = True

        self.suprimidas_entre_clases += int(muerto.sum())
        salida: Dict[str, List[Any]] = {clase: [] for clase in por_clase}
        for k in range(n):
            if not muerto[k]:
                clase, d = plano[k]
                salida[clase].append(d)
        return salida

    def _filtrar_nacimientos(self, tracks: List[_Track], dets: List[Any],
                             p: ParametrosClase) -> List[Any]:
        """Saca las detecciones que quieren nacer encima de algo que ya está.

        Incremental: cada detección que sí nace se suma a la lista contra la
        que se comparan las siguientes. Sin eso, en el primer frame no hay
        tracks todavía y las cinco cajas de la misma persona nacerían todas
        juntas — que es justo como empieza el problema.
        """
        n = len(dets)
        if n == 0 or p.nacer_iou_max >= 1.0:
            return dets

        cajas = a_array([_caja(d) for d in dets]).astype(np.float64)
        orden = np.argsort(-np.asarray([_score(d) for d in dets]))

        # Contra los tracks que ya existen: una sola pasada matricial.
        bloqueado = np.zeros(n, dtype=bool)
        if tracks:
            ref = np.asarray([t.caja for t in tracks], dtype=np.float64)
            dentro = contencion_matriz(cajas, ref)
            afuera = contencion_matriz(ref, cajas).T
            if p.respetar_pie:
                margen = np.maximum(2.0, 0.02 * np.maximum(ref[:, 3] - ref[:, 1], 1.0))
                adelante = cajas[:, 3][:, None] > (ref[:, 3] + margen)[None, :]
                dentro = np.where(adelante, 0.0, dentro)
            bloqueado = ((iou_matriz(cajas, ref).max(axis=1) >= p.nacer_iou_max)
                         | (np.maximum(dentro, afuera).max(axis=1)
                            >= p.nacer_contencion_max))

        # Y entre las candidatas mismas, de la mejor a la peor.
        iou_dd = iou_matriz(cajas, cajas)
        cont_dd = contencion_matriz(cajas, cajas)
        y2 = cajas[:, 3]
        altos = np.maximum(cajas[:, 3] - cajas[:, 1], 1.0)
        aceptadas: List[int] = []
        for k in orden:
            k = int(k)
            if bloqueado[k]:
                continue
            if aceptadas:
                a = np.asarray(aceptadas)
                dentro = cont_dd[k, a]
                afuera = cont_dd[a, k]
                if p.respetar_pie:
                    dentro = np.where(y2[k] > y2[a] + np.maximum(2.0, 0.02 * altos[a]),
                                      0.0, dentro)
                if (float(iou_dd[k, a].max()) >= p.nacer_iou_max
                        or float(np.maximum(dentro, afuera).max())
                        >= p.nacer_contencion_max):
                    continue
            aceptadas.append(k)

        self.nacimientos_bloqueados += n - len(aceptadas)
        return [dets[k] for k in sorted(aceptadas)]

    # -------------------------------------------------------------- #
    def _emparejar(self, tracks: List[_Track], indices: List[int],
                   cajas_pred: np.ndarray, dets: List[Any],
                   p: ParametrosClase, iou_min: float,
                   ts: float, frame_idx: int,
                   buffer: float = 0.0,
                   usar_gating: bool = True) -> Tuple[List[int], List[Any]]:
        """Asocia `indices` (tracks) con `dets`. Devuelve (tracks que quedaron
        libres, detecciones que quedaron sueltas)."""
        if not indices or not dets:
            return list(indices), list(dets)

        cajas_det = a_array([_caja(d) for d in dets])
        pred = cajas_pred[indices]
        iou = iou_matriz(expandir(pred, buffer), expandir(cajas_det, buffer))
        costo = 1.0 - iou.astype(np.float64)

        # Gating: si el Kalman dice que esa detección está a más de 4 sigmas,
        # no importa cuánto IoU tenga — es otra cosa.
        if usar_gating and p.gating and p.kalman:
            med = np.asarray([xyxy_a_cxcyah(b) for b in cajas_det],
                             dtype=np.float64)
            for fila, idx in enumerate(indices):
                t = tracks[idx]
                if t.media is None:
                    continue
                d2 = t.kf.distancia_gating(t.media, t.cov, med)
                costo[fila, d2 > CHI2_95_4GL] = 1e6

        pares, filas_libres, cols_libres = asignar(costo, costo_max=1.0 - iou_min)

        for fila, col in pares:
            tracks[indices[fila]].corregir(
                _caja(dets[col]), _score(dets[col]), ts, frame_idx,
                self.cfg.radio_quieto)

        libres = [indices[f] for f in filas_libres]
        sueltas = [dets[c] for c in cols_libres]
        return libres, sueltas

    # -------------------------------------------------------------- #
    @property
    def tracks(self) -> List[TrackLocal]:
        ts = time.time()
        return [t.salida(self.camera_id, self._frame - 1, ts, self.cfg.quieto_s)
                for lista in self._tracks.values() for t in lista]

    @property
    def confirmados(self) -> List[TrackLocal]:
        return [t for t in self.tracks if t.confirmado]

    def reset(self) -> None:
        self._tracks.clear()
        self._frame = 0

    def resumen(self) -> str:
        tot = sum(len(v) for v in self._tracks.values())
        conf = sum(1 for v in self._tracks.values() for t in v
                   if t.estado == "confirmado")
        detalle = ", ".join(f"{k}:{len(v)}" for k, v in sorted(self._tracks.items())
                            if v) or "vacío"
        extra = ""
        if self.suprimidas_duplicadas or self.suprimidas_entre_clases \
                or self.nacimientos_bloqueados:
            extra = (f"  duplicadas {self.suprimidas_duplicadas} "
                     f"(+{self.suprimidas_entre_clases} entre clases), "
                     f"nacimientos bloqueados {self.nacimientos_bloqueados}")
        return (f"[{self.camera_id}] frame {self._frame}  tracks {tot} "
                f"({conf} confirmados)  {detalle}{extra}")


def _needs_vlm(t: _Track) -> bool:
    """Un track confirmado de clase crítica que nunca superó la banda alta
    de confianza es exactamente el caso que el VLM tiene que mirar."""
    return (t.clase in ("pistola", "cuchillo", "humo", "llama")
            and t.estado == "confirmado"
            and t.score < 0.60)


# --------------------------------------------------------------------------- #
# Varias cámaras
# --------------------------------------------------------------------------- #
class TrackerMultiCamara:
    """Un `TrackerLocal` por cámara, creado al vuelo. Es lo que se usa en
    producción: `infer_batch()` devuelve N `FrameResult` y esto los reparte."""

    def __init__(self, cfg: Optional[ConfigTracker] = None) -> None:
        self.cfg = cfg or ConfigTracker()
        self._por_camara: Dict[str, TrackerLocal] = {}

    def tracker(self, camera_id: str) -> TrackerLocal:
        tk = self._por_camara.get(camera_id)
        if tk is None:
            tk = TrackerLocal(camera_id, self.cfg)
            self._por_camara[camera_id] = tk
        return tk

    def actualizar_desde_resultado(self, res: Any,
                                   usar_crudas: bool = True) -> List[TrackLocal]:
        cam = getattr(res, "camera_id", "cam-0")
        return self.tracker(cam).actualizar_desde_resultado(res, usar_crudas)

    def actualizar_lote(self, resultados: Sequence[Any],
                        usar_crudas: bool = True) -> Dict[str, List[TrackLocal]]:
        """El lote entero de `infer_batch()` -> tracks por cámara."""
        return {getattr(r, "camera_id", f"cam-{i}"):
                self.actualizar_desde_resultado(r, usar_crudas)
                for i, r in enumerate(resultados)}

    def camaras(self) -> List[str]:
        return sorted(self._por_camara)

    def reset(self, camera_id: Optional[str] = None) -> None:
        if camera_id is None:
            self._por_camara.clear()
        else:
            self._por_camara.pop(camera_id, None)

    def resumen(self) -> str:
        return "\n".join(tk.resumen() for tk in self._por_camara.values())


# --------------------------------------------------------------------------- #
# Duck typing sobre las detecciones
# --------------------------------------------------------------------------- #
def _caja(d: Any) -> Sequence[float]:
    if hasattr(d, "bbox_xyxy"):
        return d.bbox_xyxy
    if isinstance(d, dict):
        return d["bbox_xyxy"]
    return d[:4]                                        # (x1,y1,x2,y2,...)


def _score(d: Any) -> float:
    if hasattr(d, "score"):
        return float(d.score)
    if isinstance(d, dict):
        return float(d.get("score", 1.0))
    return float(d[4]) if len(d) > 4 else 1.0


def _label(d: Any) -> str:
    if hasattr(d, "label"):
        return str(d.label)
    if isinstance(d, dict):
        return str(d.get("label", "objeto"))
    return str(d[5]) if len(d) > 5 else "objeto"


def _class_id(d: Any) -> int:
    if hasattr(d, "class_id"):
        return int(d.class_id)
    if isinstance(d, dict):
        return int(d.get("class_id", -1))
    return -1
