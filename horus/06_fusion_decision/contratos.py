# -*- coding: utf-8 -*-
"""
contratos.py · HORUS — qué entra y qué sale de la capa de fusión.

Este archivo es la frontera. Todo lo que las cabezas producen entra por acá
convertido a dataclasses planos, sin tensores y sin torch. Tres razones:

1. La fusión se puede probar sin GPU, sin cámaras y sin modelos: se arman
   observaciones a mano y se mira qué eventos salen (`probar_fusion.py`).
2. Cambiar una cabeza no obliga a tocar las reglas. Cambia el adaptador.
3. Todo es serializable, así que el mismo objeto que decide el evento es el
   que se guarda en el log y se manda a `07_alerta`.

Cabezas que todavía no existen
------------------------------
Pose (caídas) y el clasificador de acciones no están entrenados. Este archivo
igual define su contrato, y `ObservacionCamara.cabezas` declara cuáles están
realmente vivas — las reglas que dependen de una cabeza ausente quedan
DORMIDAS en vez de romper o, peor, de decidir con datos vacíos.

El contrato de acciones (`AccionObs`) acepta las dos formas en que puede
llegar un modelo de caída o de robo, porque desde acá no se sabe cuál va a
ser:

  a) **Estimador de pose** — devuelve 17 keypoints por persona y la regla
     calcula la geometría (torso horizontal, cadera baja).
  b) **Clasificador de acción** — devuelve directamente "caida" con un score,
     y la regla solo confirma persistencia.

Un modelo nuevo se engancha llenando `AccionObs` y agregando una línea en la
tabla `ACCIONES_EXTERNAS` de `reglas.py`. No hay que tocar el motor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Severidad
# --------------------------------------------------------------------------- #
class Severidad(IntEnum):
    """Qué tan fuerte es el evento. Ordenable a propósito: `07_alerta` decide
    el canal (log / push / sirena) comparando contra un piso."""
    INFO = 0        # se registra, no molesta a nadie
    AVISO = 1       # aparece en el tablero
    ALERTA = 2      # push al operador
    CRITICO = 3     # push + sirena + clip, y no espera confirmación del VLM

    @property
    def etiqueta(self) -> str:
        return {0: "info", 1: "aviso", 2: "alerta", 3: "crítico"}[int(self)]


# Nombres canónicos de las cabezas. Se usan en `ObservacionCamara.cabezas` y
# en `Regla.requiere`.
CABEZA_OBJETOS = "objetos"
CABEZA_SEGMENTACION = "segmentacion"
CABEZA_POSE = "pose"
CABEZA_ACCION = "accion"
CABEZA_REID = "reid"


# --------------------------------------------------------------------------- #
# Entradas
# --------------------------------------------------------------------------- #
@dataclass
class SegObs:
    """Lo que la cabeza de segmentación le dice a la fusión.

    No incluye la máscara: la fusión decide por área y confianza, no por
    contorno. Es la lección de `camino-cuda-segmentacion` — el mIoU medía el
    contorno y la decisión del producto no lo necesita.
    """
    area: Dict[str, float] = field(default_factory=dict)     # fracción del cuadro
    score: Dict[str, float] = field(default_factory=dict)    # prob media
    camera_id: str = "cam-0"
    frame_idx: int = -1
    ts: float = 0.0
    needs_vlm: bool = False

    def de(self, clase: str) -> float:
        return float(self.area.get(clase, 0.0))

    def conf(self, clase: str) -> float:
        return float(self.score.get(clase, 0.0))


# Orden COCO-17, que es el que devuelven YOLO-pose, MMPose y casi todo lo
# demás. Si la cabeza propia usa otro, se remapea en el adaptador y no en las
# reglas.
KEYPOINTS_COCO = (
    "nariz", "ojo_izq", "ojo_der", "oreja_izq", "oreja_der",
    "hombro_izq", "hombro_der", "codo_izq", "codo_der",
    "muneca_izq", "muneca_der", "cadera_izq", "cadera_der",
    "rodilla_izq", "rodilla_der", "tobillo_izq", "tobillo_der",
)
IDX_KP = {n: i for i, n in enumerate(KEYPOINTS_COCO)}


@dataclass
class AccionObs:
    """Una observación de acción/postura sobre una persona.

    Contrato para las cabezas que todavía no llegaron. Se llena con lo que se
    tenga: keypoints, una etiqueta de acción, o las dos.

        AccionObs(accion="caida", score=0.81, bbox_xyxy=(...), track_id=17)
        AccionObs(accion="", keypoints=kp17x3, bbox_xyxy=(...), fuente="pose")
    """
    bbox_xyxy: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    accion: str = ""                      # "caida", "robo", "pelea", ...
    score: float = 0.0                    # confianza de esa etiqueta
    keypoints: Optional[np.ndarray] = None    # (17, 3) -> x, y, confianza
    track_id: Optional[int] = None        # si la cabeza ya viene trackeada
    camera_id: str = "cam-0"
    frame_idx: int = -1
    ts: float = 0.0
    fuente: str = "accion"                # "pose" | "accion" | nombre del modelo

    @property
    def tiene_keypoints(self) -> bool:
        return (self.keypoints is not None
                and getattr(self.keypoints, "shape", (0,))[0] >= 17)

    def kp(self, nombre: str, conf_min: float = 0.30) -> Optional[Tuple[float, float]]:
        """Un keypoint por nombre, o None si no está o es poco confiable."""
        if not self.tiene_keypoints:
            return None
        i = IDX_KP.get(nombre)
        if i is None:
            return None
        x, y, c = (float(v) for v in self.keypoints[i][:3])
        return (x, y) if c >= conf_min else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("keypoints")
        return d


@dataclass
class ObservacionCamara:
    """Todo lo que se sabe de UNA cámara en UN frame. Es la unidad que come
    el motor de fusión."""

    camera_id: str
    frame_idx: int = -1
    ts: float = field(default_factory=time.time)

    tracks: List[Any] = field(default_factory=list)        # TrackLocal de 05_tracking
    seg: Optional[SegObs] = None
    acciones: List[AccionObs] = field(default_factory=list)

    novelty: float = 0.0                  # del backbone, para VLMGate
    incertidumbre: float = 0.0            # de las cabezas, para VLMGate
    tam_frame: Tuple[int, int] = (0, 0)   # (alto, ancho) en px del frame original

    # Qué cabezas produjeron ESTE frame. Una regla que necesita una cabeza que
    # no está en este conjunto no se evalúa. Sin esto, "no hay caídas" y "no
    # hay cabeza de caídas" serían indistinguibles, que es la peor confusión
    # posible en un sistema cuyo objetivo es no perderse nada.
    cabezas: FrozenSet[str] = frozenset({CABEZA_OBJETOS})

    def confirmados(self, *clases: str) -> List[Any]:
        """Tracks confirmados, opcionalmente filtrados por clase."""
        out = [t for t in self.tracks if getattr(t, "estado", "") == "confirmado"]
        if clases:
            out = [t for t in out if getattr(t, "clase", "") in clases]
        return out

    def hay(self, cabeza: str) -> bool:
        return cabeza in self.cabezas


# --------------------------------------------------------------------------- #
# Salidas
# --------------------------------------------------------------------------- #
@dataclass
class Hallazgo:
    """Lo que devuelve una regla en un frame: un candidato a evento.

    Un hallazgo NO es un evento. El motor exige que se sostenga en el tiempo
    antes de abrir uno — un frame no alcanza para despertar a nadie.
    """
    tipo: str
    severidad: Severidad
    confianza: float
    camera_id: str
    motivo: str = ""
    zona: Optional[str] = None
    track_ids: List[int] = field(default_factory=list)
    global_id: Optional[int] = None
    evidencia: Dict[str, Any] = field(default_factory=dict)
    necesita_vlm: bool = False
    ts: float = 0.0

    def clave(self) -> Tuple:
        """Identidad del hallazgo a lo largo del tiempo. Dos hallazgos con la
        misma clave en frames distintos son el MISMO evento en curso.

        Si hay `global_id` se usa ese y no el track local: la persona que
        cruza de cámara tiene que seguir siendo el mismo evento, no uno nuevo.
        """
        if self.global_id is not None:
            return (self.tipo, "gid", self.global_id)
        return (self.tipo, self.camera_id, self.zona or "")


@dataclass
class Evento:
    """Lo que la fusión le entrega a `07_alerta`. Serializable entero."""

    evento_id: str
    tipo: str
    severidad: Severidad
    camera_id: str
    ts_inicio: float
    ts_ultimo: float
    confianza: float = 0.0
    motivo: str = ""
    zona: Optional[str] = None
    estado: str = "abierto"               # "abierto" | "sostenido" | "cerrado"
    global_id: Optional[int] = None
    track_ids: List[int] = field(default_factory=list)
    camaras: List[str] = field(default_factory=list)
    confirmaciones: int = 0
    necesita_vlm: bool = False
    vlm_motivo: str = ""
    evidencia: Dict[str, Any] = field(default_factory=dict)

    @property
    def duracion_s(self) -> float:
        return max(0.0, self.ts_ultimo - self.ts_inicio)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severidad"] = self.severidad.etiqueta
        d["duracion_s"] = round(self.duracion_s, 2)
        return d

    def linea(self) -> str:
        """Una línea legible, del estilo del diagrama: 'robo · ID 17'."""
        quien = f" · ID {self.global_id}" if self.global_id is not None else ""
        return (f"[{self.severidad.etiqueta.upper():>7}] {self.tipo}{quien} · "
                f"{self.camera_id}{' · ' + self.zona if self.zona else ''} · "
                f"{self.confianza:.2f} · {self.motivo}")


# --------------------------------------------------------------------------- #
# Adaptadores — de las cabezas a los contratos
# --------------------------------------------------------------------------- #
def seg_desde_resultado(sr: Any) -> SegObs:
    """`SegResult` de la cabeza de segmentación -> `SegObs`.

    Duck typing y sin tocar `.mask`: leer la máscara acá traería un tensor de
    torch a una capa que no lo necesita ni lo quiere.
    """
    return SegObs(
        area=dict(getattr(sr, "area", {}) or {}),
        score=dict(getattr(sr, "score", {}) or {}),
        camera_id=str(getattr(sr, "camera_id", "cam-0")),
        frame_idx=int(getattr(sr, "frame_idx", -1)),
        ts=float(getattr(sr, "ts", 0.0)),
        needs_vlm=bool(getattr(sr, "needs_vlm", False)),
    )


def acciones_desde_pose(salida: Any,
                        camera_id: str = "cam-0",
                        frame_idx: int = -1,
                        ts: Optional[float] = None,
                        orden_keypoints: Sequence[str] = KEYPOINTS_COCO,
                        fuente: str = "pose") -> List[AccionObs]:
    """Adaptador para la cabeza de pose cuando llegue.

    Acepta las formas en que suelen venir estas salidas:

      - array (N, 17, 3)                          -> keypoints, sin cajas
      - lista de dicts con 'keypoints' y 'bbox'   -> YOLO-pose, MMPose
      - lista de objetos con .keypoints / .bbox_xyxy

    Si el modelo del compañero usa otro orden de keypoints, se pasa
    `orden_keypoints` y se remapea acá. Las reglas siempre ven COCO-17.
    """
    ts = time.time() if ts is None else float(ts)
    remap = None
    if tuple(orden_keypoints) != tuple(KEYPOINTS_COCO):
        idx = {n: i for i, n in enumerate(orden_keypoints)}
        remap = [idx.get(n, -1) for n in KEYPOINTS_COCO]

    def _kp(arr) -> Optional[np.ndarray]:
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] < 1:
            return None
        if a.shape[1] == 2:               # sin confianza: se asume 1.0
            a = np.concatenate([a, np.ones((a.shape[0], 1), np.float32)], axis=1)
        if remap is not None:
            out = np.zeros((17, 3), dtype=np.float32)
            for destino, origen in enumerate(remap):
                if 0 <= origen < a.shape[0]:
                    out[destino] = a[origen][:3]
            return out
        return a[:, :3]

    def _caja(kp: Optional[np.ndarray], dado) -> Tuple[float, float, float, float]:
        if dado is not None:
            return tuple(float(v) for v in dado)
        if kp is None:
            return (0.0, 0.0, 0.0, 0.0)
        vis = kp[kp[:, 2] >= 0.30]
        if vis.shape[0] == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (float(vis[:, 0].min()), float(vis[:, 1].min()),
                float(vis[:, 0].max()), float(vis[:, 1].max()))

    salidas: List[AccionObs] = []
    arr = np.asarray(salida) if isinstance(salida, np.ndarray) else None
    if arr is not None and arr.ndim == 3:
        items = [{"keypoints": arr[i]} for i in range(arr.shape[0])]
    elif arr is not None and arr.ndim == 2:
        items = [{"keypoints": arr}]
    else:
        items = list(salida or ())

    for it in items:
        if isinstance(it, dict):
            kp = _kp(it.get("keypoints", it.get("kpts")))
            caja = it.get("bbox_xyxy", it.get("bbox"))
            score = float(it.get("score", it.get("conf", 1.0)))
            tid = it.get("track_id")
            accion = str(it.get("accion", it.get("label", "")) or "")
        else:
            kp = _kp(getattr(it, "keypoints", None))
            caja = getattr(it, "bbox_xyxy", getattr(it, "bbox", None))
            score = float(getattr(it, "score", 1.0))
            tid = getattr(it, "track_id", None)
            accion = str(getattr(it, "accion", getattr(it, "label", "")) or "")

        salidas.append(AccionObs(
            bbox_xyxy=_caja(kp, caja), accion=accion, score=score,
            keypoints=kp, track_id=None if tid is None else int(tid),
            camera_id=camera_id, frame_idx=frame_idx, ts=ts, fuente=fuente))
    return salidas


def acciones_desde_clasificador(salida: Any,
                                camera_id: str = "cam-0",
                                frame_idx: int = -1,
                                ts: Optional[float] = None,
                                mapa: Optional[Dict[str, str]] = None,
                                fuente: str = "accion") -> List[AccionObs]:
    """Adaptador para un clasificador de acción (caída, robo, pelea...).

    `mapa` traduce las etiquetas del modelo a las canónicas de Horus:

        acciones_desde_clasificador(salida, mapa={"fall": "caida",
                                                  "theft": "robo"})

    Cada item puede ser un dict {'label'/'accion', 'score'/'conf', 'bbox'} o
    un objeto con esos atributos.
    """
    ts = time.time() if ts is None else float(ts)
    mapa = mapa or {}
    fuera: List[AccionObs] = []
    for it in salida or ():
        if isinstance(it, dict):
            lab = str(it.get("accion", it.get("label", "")) or "")
            score = float(it.get("score", it.get("conf", 0.0)))
            caja = it.get("bbox_xyxy", it.get("bbox")) or (0, 0, 0, 0)
            tid = it.get("track_id")
        else:
            lab = str(getattr(it, "accion", getattr(it, "label", "")) or "")
            score = float(getattr(it, "score", 0.0))
            caja = getattr(it, "bbox_xyxy", getattr(it, "bbox", None)) or (0, 0, 0, 0)
            tid = getattr(it, "track_id", None)
        fuera.append(AccionObs(
            bbox_xyxy=tuple(float(v) for v in caja),
            accion=mapa.get(lab, lab), score=score,
            track_id=None if tid is None else int(tid),
            camera_id=camera_id, frame_idx=frame_idx, ts=ts, fuente=fuente))
    return fuera


def observacion(camera_id: str,
                resultado_objetos: Any = None,
                tracks: Optional[Sequence[Any]] = None,
                seg: Any = None,
                acciones: Optional[Sequence[AccionObs]] = None,
                cabezas: Optional[Sequence[str]] = None) -> ObservacionCamara:
    """Arma la `ObservacionCamara` de un frame juntando lo que haya.

        obs = observacion("cam-1",
                          resultado_objetos=res,        # FrameResult
                          tracks=tracks_de_esa_camara,  # TrackLocal
                          seg=seg_result)               # SegResult o None

    Las cabezas presentes se deducen de lo que se pasó, salvo que se declaren
    explícitamente.
    """
    r = resultado_objetos
    presentes = set()
    if r is not None or tracks:
        presentes.add(CABEZA_OBJETOS)
    if seg is not None:
        presentes.add(CABEZA_SEGMENTACION)
    if acciones:
        presentes.update(a.fuente if a.fuente in (CABEZA_POSE, CABEZA_ACCION)
                         else CABEZA_ACCION for a in acciones)
    if tracks and any(getattr(t, "global_id", None) is not None for t in tracks):
        presentes.add(CABEZA_REID)
    if cabezas is not None:
        presentes = set(cabezas)

    alto, ancho = 0, 0
    if r is not None:
        alto, ancho = tuple(getattr(r, "original_size", (0, 0)))

    return ObservacionCamara(
        camera_id=camera_id,
        frame_idx=int(getattr(r, "frame_idx", -1)) if r is not None else -1,
        ts=float(getattr(r, "ts", 0.0)) or time.time(),
        tracks=list(tracks or ()),
        seg=seg_desde_resultado(seg) if seg is not None else None,
        acciones=list(acciones or ()),
        novelty=float(getattr(r, "novelty", 0.0)) if r is not None else 0.0,
        incertidumbre=float(getattr(r, "incertidumbre", 0.0)) if r is not None else 0.0,
        tam_frame=(int(alto), int(ancho)),
        cabezas=frozenset(presentes),
    )
