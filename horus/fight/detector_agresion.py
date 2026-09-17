# -*- coding: utf-8 -*-
"""
detector_agresion.py · HORUS — la cabeza de agresión, enchufada a la fusión.

Envuelve el MC3-18 de `fight/src/modelo_fight.py` (fine-tuneado sobre RWF-2000,
bal. acc medida 88,75 %) y lo convierte en `AccionObs`, que es lo único que la
capa de fusión entiende. La decisión del evento NO se toma acá: la toma
`ReglaAccionExterna` con la definición `pelea`, y la escalada a crítico la toma
la correlación de `motor_fusion.py`.

Tres diferencias con `scripts/03_webcam.py`, ninguna cosmética
-------------------------------------------------------------
**1 · No corre su propio YOLO.** El script abre `yolov8n.pt` para contar
personas. Horus ya tiene detector y un tracker que las sigue con identidad;
un segundo detector cuesta una pasada de GPU por frame y, peor, hace que dos
modelos discrepen sobre dónde está la gente — el evento diría "pelea" sobre
cajas que no son las de ningún track y la regla no podría decir quién peleó.
La compuerta se calcula sobre los `TrackLocal` que ya están.

**2 · El buffer se llena desde UNA persona, y recién se clasifica con DOS.**
El script tira el buffer cuando la compuerta se cierra, así que al aparecer el
segundo cuerpo hay que esperar 5 s enteros de clip antes de poder opinar — y
los 5 s que importan son justo los del principio. Acá el clip se sigue
llenando con una sola persona en cuadro (alguien se acerca, que es como
empieza una pelea) y la compuerta solo decide si se *clasifica*.

**3 · No decide, no late y no alarma.** El script sostenía su propia
persistencia y disparaba la alarma. Eso ya existe aguas abajo y está probado:
`persistencia_s = 1.5` en la definición de `pelea`, el ciclo de vida en el
motor, y la correlación que la sube a crítico solo si hay un arma o alguien
en el piso. Duplicarlo acá daría dos relojes que se contradicen.

Qué sale
--------
Una `AccionObs` por cámara mirada, **siempre**, aunque no haya nada que
reportar (`accion=""`). Si solo se emitiera al ver una pelea, un frame
tranquilo dejaría a la regla sin su cabeza y figuraría DORMIDA: "no hubo
peleas" y "no hay modelo de peleas" volverían a ser lo mismo, que es la peor
confusión posible en un sistema cuyo objetivo es no perderse nada.

La caja de la observación es la **unión de los participantes**, no el cuadro
entero: es lo que le permite a la fusión reconstruir quién peleó y, con eso,
correlacionar con el arma que lleva uno o la caída del otro. Al modelo, en
cambio, se le pasa el **frame entero**, porque es lo que vio en entrenamiento
(RWF-2000 son clips completos); recortar a la unión cambiaría la distribución
de entrada.

Uso
---
    from detector_agresion import DetectorAgresion, ConfigAgresion

    det = DetectorAgresion(ConfigAgresion(checkpoint="checkpoints/modelo_fight.pt"))
    acciones = det.procesar("cam-1", frame_bgr, tracks, ts=time.time())
    obs = observacion("cam-1", tracks=tracks, acciones=acciones,
                      cabezas={CABEZA_OBJETOS, CABEZA_ACCION})

Verificación: `python probar_detector_agresion.py` — sin torch y sin pesos.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
_FUSION = os.path.normpath(os.path.join(_AQUI, "..", "06_fusion_decision"))
for _p in (_AQUI, os.path.join(_AQUI, "src"), _FUSION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contratos import AccionObs, CABEZA_ACCION  # noqa: E402

# Normalización de Kinetics-400 — la misma con la que se entrenó. Se replica
# acá en vez de importarla de `modelo_fight` para que este módulo se pueda
# cargar sin torch instalado; el autotest lo verifica contra el original.
MEDIA_KINETICS = (0.43216, 0.394666, 0.37645)
STD_KINETICS = (0.22803, 0.22145, 0.216989)


# --------------------------------------------------------------------------- #
@dataclass
class ConfigAgresion:
    """Los números que no se pueden tocar sin cambiar el modelo están marcados."""

    checkpoint: str = os.path.join(_AQUI, "checkpoints", "modelo_fight.pt")

    # --- fijos por entrenamiento -----------------------------------------
    ventana: int = 32               # frames por clip que espera el MC3-18
    duracion_clip_s: float = 5.0    # RWF-2000 son clips de 5 s
    lado: int = 128                 # se guarda a 128...
    lado_final: int = 112           # ...y se recorta al centro a 112

    # --- política de corrida ---------------------------------------------
    paso_clasif_s: float = 0.75     # cada cuánto se clasifica, no cada frame
    min_personas: int = 2           # la compuerta: sin dos, no hay pelea
    min_alto_px: int = 60           # una persona de 20 px es 2 px tras el resize
    umbral: float = 0.50            # de acá para arriba se etiqueta "pelea"
    tolerancia_gate_s: float = 1.5  # compuerta cerrada más que esto -> tirar clip
    max_lote: int = 8               # cámaras por forward
    dispositivo: str = "auto"
    etiqueta: str = "pelea"
    clase_persona: str = "persona"

    def intervalo_buffer(self) -> float:
        """Cada cuánto entra un frame al clip.

        Se muestrea **por tiempo y no por frame**: si se guardara un frame de
        cada N, el clip de 32 frames duraría 1 s con una webcam a 30 FPS y 4 s
        con el pipeline a 8 FPS, y el modelo vería el mismo movimiento
        acelerado o frenado según la carga de la máquina. Entrenó sobre 5 s.
        """
        return self.duracion_clip_s / max(1, self.ventana)


@dataclass
class _EstadoCam:
    buffer: Deque[np.ndarray] = field(default_factory=deque)
    ultimo_guardado: float = -1e9
    ultima_clasif: float = -1e9
    gate_abierto_ts: float = -1e9
    prob: float = 0.0
    etiquetado: bool = False


class ErrorCheckpointAgresion(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
class DetectorAgresion:
    """Clip por cámara -> MC3-18 -> `AccionObs`.

    `clasificador` se puede inyectar: recibe un array `(N, T, H, W, 3)` de
    float32 ya normalizado y devuelve N probabilidades de pelea. Sirve para
    probar todo el cableado sin torch, sin pesos y sin GPU — la misma
    propiedad que hace que la capa de fusión se pueda probar sin GPU, y que
    agregar esta cabeza no podía romper.
    """

    def __init__(self, cfg: Optional[ConfigAgresion] = None,
                 clasificador: Optional[Callable[[np.ndarray], Sequence[float]]] = None) -> None:
        self.cfg = cfg or ConfigAgresion()
        self._cams: Dict[str, _EstadoCam] = {}
        self._clasificador = clasificador
        self._modelo = None
        self._torch = None
        self._media = np.asarray(MEDIA_KINETICS, dtype=np.float32)
        self._std = np.asarray(STD_KINETICS, dtype=np.float32)
        self.stats = {"frames": 0, "clasificaciones": 0, "clips_tirados": 0,
                      "etiquetas": 0, "gate_cerrado": 0}
        if clasificador is None:
            self._cargar()

    # ------------------------------------------------------------------ #
    def _cargar(self) -> None:
        """Carga el checkpoint y **verifica que sea el modelo correcto**.

        `modelo_fight.pt` es un `state_dict` pelado: sin época, sin métrica y
        sin huella de los pesos. Un checkpoint que carga con `strict=False` y
        devuelve probabilidades no es un checkpoint que anda — es exactamente
        la falla que dejó a la cabeza de objetos alucinando durante un mes sin
        tirar una sola excepción. Acá se exige coincidencia exacta y se corta
        con un mensaje si no la hay.
        """
        # El archivo primero: que falte el checkpoint es lo más común y no
        # tiene nada que ver con torch. Preguntar por torch antes devolvería
        # "instalá torch" a alguien cuyo problema es que no copió los pesos.
        ruta = self.cfg.checkpoint
        if not os.path.exists(ruta):
            raise ErrorCheckpointAgresion(
                f"No existe el checkpoint {ruta}. Está en la rama MateProModel "
                f"(horus/fight/checkpoints/modelo_fight.pt, 46 MB). Sin él la "
                f"cabeza no arranca: correr 'igual pero sin agresión' le haría "
                f"creer al operador que el sistema la está mirando.")

        try:
            import torch  # noqa: WPS433
            from modelo_fight import crear_modelo
        except ImportError as e:  # pragma: no cover - depende del entorno
            raise ErrorCheckpointAgresion(
                f"falta una dependencia para correr agresión ({e}). "
                f"Instalá torch/torchvision, o inyectá `clasificador=` para probar "
                f"el cableado sin el modelo.") from e

        modelo = crear_modelo(preentrenado=False)
        estado = torch.load(ruta, map_location="cpu")
        if isinstance(estado, dict) and "state_dict" in estado:
            estado = estado["state_dict"]

        faltan, sobran = modelo.load_state_dict(estado, strict=False)
        if faltan or sobran:
            raise ErrorCheckpointAgresion(
                f"{ruta} no es el state_dict de este MC3-18: faltan "
                f"{len(faltan)} tensores y sobran {len(sobran)}. Cargarlo a "
                f"medias clasificaría ruido con cara de modelo.\n"
                f"  faltan: {list(faltan)[:4]}\n  sobran: {list(sobran)[:4]}")

        dev = self.cfg.dispositivo
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        modelo.eval().to(dev)
        self._torch, self._modelo, self._dev = torch, modelo, dev

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def procesar(self, camera_id: str, frame_bgr: np.ndarray,
                 tracks: Sequence[Any],
                 ts: Optional[float] = None) -> List[AccionObs]:
        """Una cámara. Devuelve siempre una lista de UNA `AccionObs`."""
        return self.procesar_lote([(camera_id, frame_bgr, tracks)], ts)[camera_id]

    def procesar_lote(self, items: Sequence[Tuple[str, np.ndarray, Sequence[Any]]],
                      ts: Optional[float] = None) -> Dict[str, List[AccionObs]]:
        """Varias cámaras en un solo forward.

        Lo que se puede juntar por frame se junta: es la misma regla que bajó
        8x la consulta de la galería del ReID. A batch 1 el MC3-18 no llega ni
        al 20 % del pico de la placa.
        """
        ts = time.time() if ts is None else float(ts)
        pendientes: List[Tuple[str, np.ndarray]] = []
        cajas: Dict[str, Tuple[float, float, float, float]] = {}
        participantes: Dict[str, int] = {}

        for camera_id, frame, tracks in items:
            st = self._cams.setdefault(camera_id, _EstadoCam())
            self.stats["frames"] += 1

            personas = self._personas(tracks)
            participantes[camera_id] = len(personas)
            cajas[camera_id] = self._union(personas)

            if personas:
                st.gate_abierto_ts = ts
                if ts - st.ultimo_guardado >= self.cfg.intervalo_buffer():
                    st.buffer.append(self._preprocesar(frame))
                    st.ultimo_guardado = ts
                    while len(st.buffer) > self.cfg.ventana:
                        st.buffer.popleft()
            elif ts - st.gate_abierto_ts > self.cfg.tolerancia_gate_s:
                # Nadie en cuadro hace rato: el clip que quedó es de otra
                # escena. Mezclarlo con lo que venga después arma un clip
                # mitad de hace un minuto y mitad de ahora.
                if st.buffer:
                    self.stats["clips_tirados"] += 1
                st.buffer.clear()
                st.prob, st.etiquetado = 0.0, False

            listo = (len(st.buffer) >= self.cfg.ventana
                     and len(personas) >= self.cfg.min_personas
                     and ts - st.ultima_clasif >= self.cfg.paso_clasif_s)
            if len(personas) < self.cfg.min_personas:
                self.stats["gate_cerrado"] += 1
                st.prob, st.etiquetado = 0.0, False
            if listo:
                pendientes.append((camera_id, self._armar_clip(st.buffer)))
                st.ultima_clasif = ts

        if pendientes:
            probs = self._clasificar(np.stack([c for _, c in pendientes]))
            for (camera_id, _), p in zip(pendientes, probs):
                st = self._cams[camera_id]
                st.prob = float(p)
                st.etiquetado = st.prob >= self.cfg.umbral
                self.stats["clasificaciones"] += 1
                if st.etiquetado:
                    self.stats["etiquetas"] += 1

        salida: Dict[str, List[AccionObs]] = {}
        for camera_id, _, _ in items:
            st = self._cams[camera_id]
            salida[camera_id] = [AccionObs(
                bbox_xyxy=cajas[camera_id],
                accion=self.cfg.etiqueta if st.etiquetado else "",
                score=st.prob if st.etiquetado else 0.0,
                track_id=None,                     # la pelea es de varios
                camera_id=camera_id, ts=ts, fuente=CABEZA_ACCION)]
        return salida

    def olvidar(self, camera_id: str) -> None:
        self._cams.pop(camera_id, None)

    def resumen(self) -> Dict[str, Any]:
        return dict(self.stats, camaras=len(self._cams),
                    listas=sum(1 for s in self._cams.values()
                               if len(s.buffer) >= self.cfg.ventana))

    # ------------------------------------------------------------------ #
    # Interno
    # ------------------------------------------------------------------ #
    def _personas(self, tracks: Sequence[Any]) -> List[Any]:
        """Personas confirmadas y lo bastante grandes para sobrevivir al resize.

        Un cuerpo de 20 px en 1080p mide 2 px después de llevar el cuadro a
        128: no aporta señal y sí infla la cuenta de la compuerta, que es
        justamente lo que decide si se gasta una pasada de GPU."""
        out = []
        for t in tracks:
            if getattr(t, "estado", "") != "confirmado":
                continue
            if getattr(t, "clase", "") != self.cfg.clase_persona:
                continue
            caja = getattr(t, "bbox_xyxy", None)
            if caja is None:
                continue
            if float(caja[3]) - float(caja[1]) < self.cfg.min_alto_px:
                continue
            out.append(t)
        return out

    @staticmethod
    def _union(personas: Sequence[Any]) -> Tuple[float, float, float, float]:
        if not personas:
            return (0.0, 0.0, 0.0, 0.0)
        cajas = [tuple(float(v) for v in p.bbox_xyxy) for p in personas]
        return (min(c[0] for c in cajas), min(c[1] for c in cajas),
                max(c[2] for c in cajas), max(c[3] for c in cajas))

    def _preprocesar(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR -> RGB, resize a `lado`. Devuelve uint8 (lado, lado, 3).

        El cuadro entero, no un recorte: es lo que el modelo vio en
        entrenamiento.
        """
        arr = np.asarray(frame_bgr)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"se esperaba un frame HxWx3, llegó {arr.shape}")
        rgb = arr[:, :, ::-1]
        return _resize_uint8(rgb, self.cfg.lado)

    def _armar_clip(self, buffer: Deque[np.ndarray]) -> np.ndarray:
        """(T, lado, lado, 3) uint8 -> (T, 112, 112, 3) float32 normalizado.

        Center crop y sin flip: el mismo preprocesamiento que el dataset en
        modo validación.
        """
        frames = np.stack(list(buffer)[-self.cfg.ventana:]).astype(np.float32) / 255.0
        off = (self.cfg.lado - self.cfg.lado_final) // 2
        f = self.cfg.lado_final
        frames = frames[:, off:off + f, off:off + f, :]
        return (frames - self._media) / self._std

    def _clasificar(self, clips: np.ndarray) -> List[float]:
        if self._clasificador is not None:
            return [float(p) for p in self._clasificador(clips)]
        torch = self._torch
        salidas: List[float] = []
        with torch.no_grad():
            for i in range(0, len(clips), self.cfg.max_lote):
                lote = clips[i:i + self.cfg.max_lote]
                x = torch.from_numpy(np.ascontiguousarray(lote))
                x = x.permute(0, 4, 1, 2, 3).to(self._dev)   # (N,3,T,H,W)
                logits = self._modelo(x)
                p = torch.softmax(logits.float(), dim=1)[:, 1]
                salidas.extend(p.cpu().tolist())
        return salidas


# --------------------------------------------------------------------------- #
def _resize_uint8(img: np.ndarray, lado: int) -> np.ndarray:
    """Resize por vecino más cercano, sin depender de OpenCV.

    Si cv2 está, se usa (bilineal, que es con lo que se preparó el dataset).
    Si no, se cae a numpy para que el módulo se pueda importar y probar en
    cualquier lado. La diferencia importa para la inferencia real, no para el
    cableado — por eso el autotest no la mide y la corrida en vivo sí usa cv2.
    """
    try:
        import cv2  # noqa: WPS433
        return cv2.resize(np.ascontiguousarray(img), (lado, lado),
                          interpolation=cv2.INTER_LINEAR)
    except ImportError:
        alto, ancho = img.shape[:2]
        fy = (np.arange(lado) * alto / lado).astype(np.int32)
        fx = (np.arange(lado) * ancho / lado).astype(np.int32)
        return np.ascontiguousarray(img[fy][:, fx])
