# -*- coding: utf-8 -*-
"""
detector_caidas.py · HORUS — el modelo de caídas, enchufado a la fusión.

Qué hay del otro lado
---------------------
El modelo de `fall/` es una cadena de tres piezas:

    recorte de persona -> MediaPipe Pose (33 puntos) -> ST-GCN (32 frames) -> P(caída)

y encima, en `scripts/12_webcam.py`, una capa de reglas escritas a mano que
decide cuándo esa probabilidad se convierte en alarma: visibilidad mínima,
persistencia, pico de velocidad y caída de verticalidad.

Este archivo lo convierte en una cabeza de Horus. Tres diferencias con el
script de webcam, y ninguna es cosmética:

1. **No corre su propio YOLO.** El script detectaba personas con `yolov8n.pt`
   y se quedaba con la caja de mayor confianza. Horus ya tiene un detector de
   personas y un tracker que las sigue con identidad. Correr un segundo
   detector cuesta GPU y, peor, hace que dos modelos discrepen sobre dónde
   está la gente: el evento diría "caída" sobre una caja que no es la de
   ningún track, y `ReglaCaida._persona_de()` tendría que adivinar por IoU.
   Acá el recorte sale del `TrackLocal`.

2. **Es multi-persona.** `num_poses=1` sobre el frame entero significa una
   sola persona por cámara — inaceptable en vigilancia, y justo la que se cae
   suele no ser la de mayor confianza. Acá hay un buffer **por `track_id`**,
   y MediaPipe corre sobre el recorte de cada uno. Como cada
   recorte tiene una sola persona, `num_poses=1` pasa de ser una limitación a
   ser lo correcto.

3. **No decide el evento.** El script disparaba la alarma solo. Acá se emite
   una `AccionObs` y la decisión la toma `ReglaCaida`, que ya sabe pedir
   quietud sostenida y subir la severidad con los segundos — y lo hace mejor,
   porque usa el `quieto_s` del tracker (Kalman sobre la identidad) en vez de
   reinventarlo mirando keypoints.

Lo que cambió en octubre, y por qué
-----------------------------------
Esta cabeza era una copia de las reglas de `12_webcam.py`, y heredó cuatro
bugs que solo se vieron al medir el sistema como corre de verdad: video
continuo, causal, sujetos que el modelo nunca vio (`scripts/17_evaluar_evento.py`
sobre 190 videos de Le2i, `resultados/README.md`). A 10 fps, la tasa a la que
corre Horus, el recall de eventos pasó de 48,5 % a 62,3 % con las mismas
falsas alarmas por hora.

1. **El ST-GCN veía las caídas en cámara lenta.** Entrenó con 32 frames
   seguidos de video a 25 fps (1,24 s de movimiento). El buffer guardaba un
   frame del pipeline por muestra, y a 10 fps eso son 3,2 s con velocidades
   2,5x más grandes: fuera de distribución en las dos features. Ahora el
   buffer guarda (ts, pose) y se **remuestrea a 25 fps** antes de clasificar.
   Y se reclasifica por reloj, no cada N frames: antes, latencia y falsas
   alarmas dependían de los fps de la cámara.
2. **El filtro de pico no filtraba.** El nivel de "movimiento normal" se
   sembraba con la velocidad del primer frame —cero por construcción— y
   quedaba congelado ahí: contra cero, todo era pico.
3. **Pico y caída del torso se pedían en el mismo frame**, y no pasan juntos:
   la velocidad es máxima a mitad de la caída y el torso termina de bajar
   después. Se perdían 16 caídas de 130.
4. **El reset mataba caídas.** MediaPipe pierde a la persona justo cuando está
   en el piso: en el 37 % de las caídas hay un hueco de más de 0,6 s dentro
   del evento, y con esa tolerancia el estado se tiraba en medio de la caída.

Las cuatro viven en `src/compuertas_caida.py`, que comparte con la máquina que
mide el harness: lo que se evalúa offline es lo que corre acá.

El reparto, entonces
--------------------
**Acá queda todo lo que necesita los 33 puntos crudos**, porque la fusión
nunca los ve: la puerta de visibilidad, el pico de velocidad y la caída de
verticalidad. Esas tres son evidencia sobre *si hubo una transición de pie a
piso*, y son lo único que separa "se cayó" de "está acostado tranquilo hace
media hora" — el clasificador solo no puede, porque las dos cosas se ven
igual en 32 frames de alguien quieto.

**En la fusión queda la política del evento**: persistencia, quietud,
severidad, correlación con otras reglas, ciclo de vida. No se duplica acá.

Por qué se emite también cuando NO hay caída
--------------------------------------------
`ObservacionCamara.cabezas` declara qué cabezas produjeron el frame, y una
regla que no encuentra su cabeza queda DORMIDA. Si este detector solo emitiera
cuando ve una caída, un frame tranquilo no declararía la cabeza y `ReglaCaida`
figuraría dormida — o sea, "no hubo caídas" y "no hay modelo de caídas" se
verían igual, que es exactamente la confusión que la capa de fusión existe
para evitar. Por eso se emite una `AccionObs` por persona mirada, con
`accion=""` cuando no hay nada que reportar.

Coordenadas: hay dos, a propósito
---------------------------------
- Al ST-GCN va lo mismo que vio en entrenamiento: MediaPipe normalizado al
  **recorte** (x/ancho_recorte, y/alto_recorte), centrado en la cadera y
  escalado por el torso. El recorte se agranda 20% como en el entrenamiento;
  cambiar ese margen cambia la distorsión de aspecto y el modelo ve otra cosa.
- A la regla van keypoints COCO-17 en **píxeles del frame**, que es el espacio
  en el que `ReglaCaida` compara la cabeza contra la cadera y en el que vive
  `bbox_xyxy`.

Mezclarlas es un error silencioso: los dos son arrays de floats y ninguno
tira excepción.

Uso
---
    from detector_caidas import ConfigCaidas, DetectorCaidas

    det = DetectorCaidas(ConfigCaidas(checkpoint="checkpoints/modelo_demo_todo.pt"))
    acciones = det.procesar("cam-1", frame, tracks, ts=time.time())

o, directamente, cableado en el pipeline:

    pipe = PipelineHorus(pesos=..., detector_caidas=det)

Para probar sin torch ni mediapipe se inyectan los dos backends:

    DetectorCaidas(cfg, backend_pose=falso_pose, clasificador=falso_clf)
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_AQUI)                      # .../horus
for _p in (os.path.join(_AQUI, "src"),
           os.path.join(_RAIZ, "06_fusion_decision")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contratos import CABEZA_ACCION, CABEZA_POSE, AccionObs  # noqa: E402
# solo numpy: esta cabeza se tiene que poder importar sin torch ni mediapipe
from compuertas_caida import (  # noqa: E402
    PuertaCaida, cobertura, duracion_ventana, remuestrear_clip,
)
from preproceso import normalizar_frame  # noqa: E402,F401  (la misma que vio el entrenamiento)


# --------------------------------------------------------------------------- #
# Índices de MediaPipe Pose (33 puntos)
# --------------------------------------------------------------------------- #
IDX_CI, IDX_CD = 23, 24          # cadera izquierda / derecha
IDX_HI, IDX_HD = 11, 12          # hombro izquierdo / derecho
IDX_TORSO = (IDX_CI, IDX_CD, IDX_HI, IDX_HD)

N_PUNTOS_MP = 33

# MediaPipe -> COCO-17, en el orden de `contratos.KEYPOINTS_COCO`.
# Es el remapeo que `acciones_desde_pose(orden_keypoints=...)` documenta; acá
# se hace a mano porque además hay que pasar de recorte a píxeles del frame.
MP_A_COCO: Tuple[int, ...] = (
    0,      # nariz
    2, 5,   # ojo izq/der
    7, 8,   # oreja izq/der
    11, 12,  # hombro izq/der
    13, 14,  # codo izq/der
    15, 16,  # muñeca izq/der
    23, 24,  # cadera izq/der
    25, 26,  # rodilla izq/der
    27, 28,  # tobillo izq/der
)


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
@dataclass
class ConfigCaidas:
    """Los valores por defecto son los medidos en `fall/`; los que agrega
    Horus están marcados."""

    # --- modelo ---
    checkpoint: str = os.path.join(_AQUI, "checkpoints", "modelo_demo_todo.pt")
    modelo_pose: str = os.path.join(_AQUI, "pose_landmarker.task")
    canales: int = 4                 # posición + velocidad, como el checkpoint
    device: Optional[str] = None     # None = cuda si hay

    # --- ventana temporal ---
    ventana: int = 32                # muestras que come el ST-GCN
    fps_canonica: float = 25.0       # el buffer se remuestrea a esto: los fps del entrenamiento
    paso_clasif_s: float = 0.16      # cada cuánto se reclasifica un track (por reloj)

    # --- puertas que necesitan los 33 puntos (ver compuertas_caida.PuertaCaida) ---
    umbral_visibilidad: float = 0.50
    tolerancia_sin_ver_s: float = 2.0      # era 0,6: tiraba el estado en medio de la caída
    factor_pico_velocidad: float = 2.5     # barrido: 2,5 > 3,0 en recall sin falsas extra
    ventana_post_pico_s: float = 4.0
    tau_baseline_s: float = 0.8            # EMA del movimiento normal, en segundos
    warmup_baseline_s: float = 0.4         # se siembra con la mediana de esto, no con un frame
    piso_baseline: float = 0.05            # torsos/seg: la quietud no lo lleva a cero
    ventana_vertical_previa_s: float = 2.0
    caida_vertical_minima: float = 0.35
    ventana_pico_a_caida_s: float = 1.0    # el torso termina de bajar después del pico
    bypass_historial_corto: bool = False   # ver `calentamiento_s`: ese caso lo cubre la geometría
    margen_recorte: float = 0.20     # igual que en entrenamiento; ver cabecera

    # --- lo que agrega Horus ---
    solo_confirmados: bool = True    # el tracker ya filtró el parpadeo
    min_alto_px: int = 80            # una persona de 40 px da una pose inventada
    max_personas: int = 4            # tope de recortes por cámara por frame
    emitir_en_calentamiento: bool = True   # ver `_emitir`
    # Alguien que entra a cuadro YA tirado no tiene transición que probar: el
    # filtro de pico lo bloquea siempre. Durante sus primeros `calentamiento_s`
    # (desde que aparece o desde un reinicio), si el clasificador ve postura de
    # caída y lo único que falta es esa prueba, se le manda la pose a
    # `ReglaCaida` para que decida por geometría. Tiene que ser MAYOR que
    # `ReglaCaida.quieto_s` (2 s), o la regla nunca llega a juntar la quietud.
    # 3 s es lo que la ventana vieja daba de hecho a 10 fps (32 frames).
    calentamiento_s: float = 3.0
    # Por debajo no se etiqueta "caida". Si el checkpoint trae un umbral
    # calibrado (`scripts/21_calibrar_umbral.py`), se usa ese: cada
    # entrenamiento deja las probabilidades en otra escala. OJO: no puede
    # quedar por debajo de `ReglaCaida.umbral_accion` (0,50), o la regla
    # descarta en silencio lo que esta cabeza etiquetó.
    umbral_reporte: float = 0.50

    def __post_init__(self) -> None:
        if self.ventana < 4:
            raise ValueError("ventana muy corta para el ST-GCN")
        if self.paso_clasif_s <= 0 or self.fps_canonica <= 0:
            raise ValueError("paso_clasif_s y fps_canonica tienen que ser > 0")

    @property
    def ventana_s(self) -> float:
        return duracion_ventana(self.ventana, self.fps_canonica)

    def nueva_puerta(self) -> PuertaCaida:
        return PuertaCaida(
            factor_pico=self.factor_pico_velocidad, tau_baseline_seg=self.tau_baseline_s,
            warmup_seg=self.warmup_baseline_s, piso_baseline=self.piso_baseline,
            ventana_vertical_previa_seg=self.ventana_vertical_previa_s,
            caida_vertical_minima=self.caida_vertical_minima,
            ventana_pico_a_caida_seg=self.ventana_pico_a_caida_s,
            bypass_historial_corto=self.bypass_historial_corto)


# --------------------------------------------------------------------------- #
# Estado por track
# --------------------------------------------------------------------------- #
class _EstadoTrack:
    """Todo lo que hay que recordar de UNA persona. Uno por (cámara, track).

    El script de webcam tenía estas mismas variables sueltas en el `main`, que
    es lo mismo que decir que soportaba una sola persona.
    """

    __slots__ = ("buffer", "ultimo_norm", "puerta", "ultima_vista_ts",
                 "prob", "ts_ultima_clasificacion", "caida_confirmada",
                 "ultimo_toque_ts", "inicio_ts", "_cfg")

    def __init__(self, cfg: "ConfigCaidas", ts: float) -> None:
        self._cfg = cfg
        # (ts, pose normalizada (33,2)). Sin maxlen: lo que se guarda lo decide
        # el reloj, no una cuenta de frames -- ver `agregar`.
        self.buffer: deque = deque()
        self.ultimo_norm: Optional[np.ndarray] = None
        self.puerta: PuertaCaida = cfg.nueva_puerta()
        self.ultima_vista_ts: Optional[float] = None
        self.prob: float = 0.0
        self.ts_ultima_clasificacion: Optional[float] = None
        # Engancha en True cuando el clasificador confirma una caída con una
        # firma reciente que la respalde, y se apaga cuando la persona se
        # levanta. Ver `DetectorCaidas._resolver_latch`.
        self.caida_confirmada: bool = False
        self.ultimo_toque_ts: Optional[float] = None
        self.inicio_ts: float = ts          # desde cuándo se la mira (o desde el último reinicio)

    def agregar(self, ts: float, kp_norm: np.ndarray) -> np.ndarray:
        """Mete una muestra y devuelve su velocidad contra la anterior.

        La velocidad para la puerta se mide sobre el stream crudo, a
        propósito: interpolarla difumina un golpe de un frame. Se guarda un
        poco más que la ventana para tener el punto de apoyo del borde
        izquierdo al remuestrear."""
        anterior = self.buffer[-1][1] if self.buffer else None
        vel = kp_norm - anterior if anterior is not None else np.zeros_like(kp_norm)
        self.buffer.append((ts, kp_norm))
        limite = ts - self._cfg.ventana_s
        while len(self.buffer) > 2 and self.buffer[1][0] < limite:
            self.buffer.popleft()
        return vel

    @property
    def listo(self) -> bool:
        """El buffer ya cubre la ventana que vio el modelo (1,24 s), sin
        importar a cuántos fps llegaron las muestras."""
        c = self._cfg
        return cobertura([x[0] for x in self.buffer], c.ventana, c.fps_canonica) >= 1.0

    def clip(self, ts: float) -> np.ndarray:
        c = self._cfg
        return remuestrear_clip([x[0] for x in self.buffer], [x[1] for x in self.buffer],
                                ts, c.ventana, c.fps_canonica)

    def reiniciar(self, ts: float) -> None:
        """Se perdió a la persona demasiado tiempo: no se sigue clasificando
        con pose congelada.

        El baseline y la firma también se tiran (`PuertaCaida.reiniciar`). Si
        quedaran de antes de que la persona saliera de cuadro pueden hacer las
        dos cosas malas: un baseline viejo y alto bloquea una caída real al
        volver, y una firma vieja valida una alarma con un movimiento que no
        tiene nada que ver.
        """
        self.buffer.clear()
        self.ultimo_norm = None
        self.puerta.reiniciar()
        self.prob = 0.0
        self.ts_ultima_clasificacion = None
        self.caida_confirmada = False
        self.inicio_ts = ts


# --------------------------------------------------------------------------- #
# Backends — se importan tarde, a propósito
# --------------------------------------------------------------------------- #
# `detector_caidas` tiene que poder importarse en una máquina sin torch y sin
# mediapipe: `probar_detector_caidas.py` corre así, y la suite de fusión
# entera vive de esa propiedad. Si estos imports estuvieran arriba, agregar
# esta cabeza haría que la fusión ya no se pudiera probar sin GPU.
class BackendPoseMediaPipe:
    """Recorte BGR -> (33, 4) con x, y, z, visibility.

    x e y vienen normalizados al recorte, que es lo que vio el entrenamiento.
    """

    def __init__(self, modelo_pose: str, min_conf: float = 0.30) -> None:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.exists(modelo_pose):
            raise FileNotFoundError(
                f"no encuentro {modelo_pose}.\n"
                f"\n"
                f"Es el modelo de pose de MediaPipe, y hay que bajarlo aparte.\n"
                f"Sorprende porque `pip install mediapipe` no alcanza: la 0.10\n"
                f"SACÓ la API vieja (`mp.solutions.pose`), que traía el modelo\n"
                f"adentro del paquete. La de Tasks, que es la única que queda,\n"
                f"pide este archivo.\n"
                f"\n"
                f"    HORUS_herramientas.bat -> opción 5\n"
                f"\n"
                f"o a mano:\n"
                f"    python bin/instalar_caidas.py")

        self._cv2 = cv2
        self._mp = mp
        opciones = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=modelo_pose),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,                       # un recorte = una persona
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf)
        self._lm = vision.PoseLandmarker.create_from_options(opciones)

    def __call__(self, recorte_bgr: np.ndarray) -> Optional[np.ndarray]:
        img = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=self._cv2.cvtColor(recorte_bgr, self._cv2.COLOR_BGR2RGB))
        r = self._lm.detect(img)
        if not r.pose_landmarks:
            return None
        return np.array([[p.x, p.y, p.z, p.visibility]
                         for p in r.pose_landmarks[0]], dtype=np.float32)


class ClasificadorSTGCN:
    """Clips (B, ventana, 33, canales) -> P(caída) por clip.

    Es batch nativo. Con 4 personas en cuadro, 4 forwards de batch 1 cuestan
    casi lo mismo que uno de batch 4 en overhead de lanzamiento, y ese overhead
    es la mayor parte cuando la red es esta de chica. Misma lección que el
    rebuild de la galería en `tracker_global`: lo que se puede juntar por
    frame, se junta.
    """

    def __init__(self, checkpoint: str, canales: int = 4,
                 device: Optional[str] = None) -> None:
        import torch
        from grafo_mediapipe import construir_matriz_adyacencia
        from modelo_stgcn import STGCN

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        A = construir_matriz_adyacencia(N_PUNTOS_MP)
        self.modelo = STGCN(A, canales_entrada=canales).to(self.device)

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"no encuentro el checkpoint {checkpoint}. Los de fall/ son "
                f"modelo_demo_todo.pt (entrenado con todo, el de despliegue) y "
                f"modelo_demo_s0.pt (deja le2i_s0 afuera, para evaluar).")
        # src/checkpoint.py: lee el formato con metadatos (config, git, métricas,
        # umbral calibrado) y también el state_dict pelado de siempre, y lo hace
        # con weights_only=True -- un .pt es un pickle y no se ejecuta a ciegas.
        import checkpoint as _ckpt
        estado, self.meta = _ckpt.cargar(checkpoint, device=self.device)
        umbral = (self.meta.get("config") or {}).get("umbral")
        self.umbral_calibrado: Optional[float] = None if umbral is None else float(umbral)
        # Un checkpoint de 2 canales en un modelo de 4 no tira error de forma
        # obvia si se carga con strict=False: entra a medias y clasifica
        # ruido. Se compara antes.
        clave = "bloques.0.conv_espacial.conv.weight"
        if clave in estado:
            esperados = int(estado[clave].shape[1])
            if esperados != canales:
                raise RuntimeError(
                    f"{os.path.basename(checkpoint)} fue entrenado con "
                    f"canales_entrada={esperados} y se está pidiendo "
                    f"{canales}. Con 2 son solo posiciones; con 4 es posición "
                    f"+ velocidad. No son intercambiables.")
        self.modelo.load_state_dict(estado)
        self.modelo.eval()

    def __call__(self, clips: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.tensor(np.ascontiguousarray(clips),
                         dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return torch.softmax(self.modelo(x), 1)[:, 1].cpu().numpy()


# --------------------------------------------------------------------------- #
# Utilidades geométricas
# --------------------------------------------------------------------------- #
# `normalizar_frame` viene de src/preproceso.py, la misma función que usaron
# los datos de entrenamiento. Antes había una copia acá: si las dos se
# desviaran, el modelo vería otra distribución y no hay umbral que lo arregle.


def visibilidad_torso(kp: np.ndarray) -> float:
    """Promedio de `visibility` en cadera y hombros.

    Un recorte cortado por el borde del cuadro, o una persona semi-tapada,
    devuelve landmarks igual — MediaPipe siempre devuelve 33 — pero con
    visibility baja justo en estos cuatro. Es la señal de que la geometría que
    sigue está inventada.
    """
    if kp.shape[1] < 4:
        return 1.0
    return float(np.mean(kp[list(IDX_TORSO), 3]))


def recorte_de_caja(frame: np.ndarray,
                    caja: Sequence[float],
                    margen: float) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
    """Recorta la caja agrandada `margen` por lado, sin salirse del frame."""
    alto, ancho = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in caja)
    aw, ah = x2 - x1, y2 - y1
    if aw <= 0 or ah <= 0:
        return None
    x1 = max(0, int(x1 - aw * margen))
    y1 = max(0, int(y1 - ah * margen))
    x2 = min(ancho, int(x2 + aw * margen))
    y2 = min(alto, int(y2 + ah * margen))
    if x2 <= x1 or y2 <= y1:
        return None
    rec = frame[y1:y2, x1:x2]
    if rec.size == 0:
        return None
    return rec, (float(x1), float(y1), float(x2), float(y2))


def kp_a_coco_px(kp: np.ndarray,
                 caja_recorte: Tuple[float, float, float, float]) -> np.ndarray:
    """(33, 4) del recorte -> (17, 3) COCO en píxeles del frame.

    Los dos pasos van juntos a propósito: remapear sin desnormalizar deja
    keypoints en [0,1] mezclados con una `bbox_xyxy` en píxeles, y la regla
    compara las dos cosas.
    """
    x1, y1, x2, y2 = caja_recorte
    ancho, alto = x2 - x1, y2 - y1
    out = np.zeros((17, 3), dtype=np.float32)
    for destino, origen in enumerate(MP_A_COCO):
        px = x1 + float(kp[origen, 0]) * ancho
        py = y1 + float(kp[origen, 1]) * alto
        conf = float(kp[origen, 3]) if kp.shape[1] >= 4 else 1.0
        out[destino] = (px, py, conf)
    return out


# --------------------------------------------------------------------------- #
# El detector
# --------------------------------------------------------------------------- #
class DetectorCaidas:
    """Una instancia por caja. Guarda estado de todas las cámaras y personas."""

    def __init__(self,
                 cfg: Optional[ConfigCaidas] = None,
                 backend_pose: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
                 clasificador: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 verboso: bool = True) -> None:
        self.cfg = cfg or ConfigCaidas()
        self._pose = backend_pose
        self._clf = clasificador
        self._verboso = verboso
        self._estados: Dict[Tuple[str, int], _EstadoTrack] = {}
        self.stats: Dict[str, int] = {
            "frames": 0, "personas": 0, "pose_ok": 0, "pose_falla": 0,
            "descarte_visibilidad": 0, "descarte_chica": 0,
            "clasificaciones": 0, "reportes": 0, "calentamiento": 0,
            "bloqueadas_sin_pico": 0, "a_geometria": 0,
        }

    def cargar(self) -> "DetectorCaidas":
        """Carga MediaPipe y el ST-GCN ya, en vez de en el primer frame.

        El servicio lo llama al arrancar: si falta el .task, mediapipe o el
        checkpoint, tiene que cortar ahí y no a mitad de la noche con la
        cámara ya andando y el operador creyendo que se miran las caídas.
        """
        self._asegurar_backends()
        return self

    # ------------------------------------------------------------------ #
    def _asegurar_backends(self) -> None:
        if self._pose is None:
            self._pose = BackendPoseMediaPipe(self.cfg.modelo_pose)
        if self._clf is None:
            self._clf = ClasificadorSTGCN(self.cfg.checkpoint,
                                          canales=self.cfg.canales,
                                          device=self.cfg.device)
            calibrado = getattr(self._clf, "umbral_calibrado", None)
            origen = "default"
            if calibrado is not None:
                # Un umbral calibrado por debajo del de ReglaCaida no da más
                # recall: da etiquetas que la regla tira sin decir nada. Se
                # sube al piso de la regla y se avisa siempre, verboso o no.
                from reglas import ReglaCaida                  # noqa: E402
                piso = ReglaCaida.umbral_accion
                origen = "calibrado en el checkpoint"
                if calibrado < piso:
                    print(f"[caidas] AVISO: el umbral calibrado "
                          f"{calibrado:.2f} queda debajo del de ReglaCaida "
                          f"({piso:.2f}); uso {piso:.2f}. Recalibrá contra "
                          f"una tasa de falsas más baja o bajá la regla.",
                          file=sys.stderr)
                    calibrado = piso
                    origen = "calibrado, subido al piso de ReglaCaida"
                self.cfg.umbral_reporte = calibrado
            if self._verboso:
                print(f"[caidas] ST-GCN en {self._clf.device} · "
                      f"{os.path.basename(self.cfg.checkpoint)} · "
                      f"umbral {self.cfg.umbral_reporte:.2f} ({origen})")

    # ------------------------------------------------------------------ #
    def _elegir(self, tracks: Sequence[Any]) -> List[Any]:
        """Qué personas se miran este frame.

        Tres filtros, y los tres cuestan plata si no están:

        - Solo `persona`, y confirmadas: el tracker ya se comió el parpadeo.
        - Alto mínimo: una persona de 40 px devuelve 33 landmarks igual, pero
          son inventados. Misma lección que el filtro de calidad del ReID —
          es preferible no opinar que opinar mal, porque una pose basura no se
          queda quieta: entra al buffer y contamina 32 frames.
        - Tope por cámara: MediaPipe corre una vez por persona y es el costo
          dominante. Se priorizan las cajas más grandes, que son las que dan
          pose confiable; si hay una multitud, esto no es lo que la va a
          resolver.
        """
        cfg = self.cfg
        elegidos = []
        for t in tracks or ():
            if getattr(t, "clase", "") != "persona":
                continue
            if cfg.solo_confirmados and getattr(t, "estado", "") != "confirmado":
                continue
            x1, y1, x2, y2 = (float(v) for v in t.bbox_xyxy)
            if (y2 - y1) < cfg.min_alto_px:
                self.stats["descarte_chica"] += 1
                continue
            elegidos.append((( x2 - x1) * (y2 - y1), t))
        elegidos.sort(key=lambda p: -p[0])
        return [t for _, t in elegidos[:cfg.max_personas]]

    # ------------------------------------------------------------------ #
    # La puerta (pico de velocidad + caída del torso) vive en
    # `compuertas_caida.PuertaCaida`, compartida con la máquina que mide el
    # harness. Ver la cabecera de ese archivo para los tres bugs que tenía la
    # versión de acá.

    # ------------------------------------------------------------------ #
    def procesar(self,
                 camera_id: str,
                 frame: np.ndarray,
                 tracks: Sequence[Any],
                 ts: Optional[float] = None) -> List[AccionObs]:
        """Un frame de UNA cámara -> una `AccionObs` por persona mirada.

        Se emite también cuando no hay nada: ver la cabecera. Lo que cambia es
        la etiqueta, no la presencia.
        """
        cfg = self.cfg
        ts = time.time() if ts is None else float(ts)
        if self._pose is None or self._clf is None:
            self._asegurar_backends()

        self.stats["frames"] += 1
        elegidos = self._elegir(tracks)
        mirados: List[Tuple[Any, _EstadoTrack, Optional[np.ndarray]]] = []
        pendientes: List[_EstadoTrack] = []

        for t in elegidos:
            self.stats["personas"] += 1
            clave = (camera_id, int(t.track_id))
            est = self._estados.get(clave)
            if est is None:
                est = self._estados[clave] = _EstadoTrack(cfg, ts)
            est.ultimo_toque_ts = ts

            kp = kp_coco = None
            corte = recorte_de_caja(frame, t.bbox_xyxy, cfg.margen_recorte)
            if corte is not None:
                recorte, caja_rec = corte
                kp = self._pose(recorte)
                if kp is None:
                    self.stats["pose_falla"] += 1
                elif visibilidad_torso(kp) < cfg.umbral_visibilidad:
                    self.stats["descarte_visibilidad"] += 1
                    kp = None
                else:
                    self.stats["pose_ok"] += 1
                    kp_coco = kp_a_coco_px(kp, caja_rec)

            # El hueco se mide contra la pose ANTERIOR, nunca contra esta.
            #
            # El script de webcam actualizaba el reloj primero y le salía
            # bien, porque miraba UN frame entero: durante el hueco seguía
            # entrando al bucle sin detección y se reseteaba solo. Acá el
            # estado es por persona, y una persona que desaparece del listado
            # de tracks no se toca hasta que vuelve. Con el orden de allá,
            # el frame en que reaparece da `sin_ver = 0` — el reset no ocurre
            # nunca y el buffer queda mitad de hace un minuto y mitad de
            # ahora. Clasifica, y clasifica cualquier cosa.
            previa = est.ultima_vista_ts
            if kp is not None:
                est.ultima_vista_ts = ts
            if previa is not None and (ts - previa) > cfg.tolerancia_sin_ver_s:
                # Hace rato que no hay pose confiable: se tira todo en vez de
                # seguir clasificando con datos congelados. "Hace rato" son 2 s
                # y no 0,6: MediaPipe pierde a la persona justo cuando está en
                # el piso, y con 0,6 el estado se tiraba EN MEDIO de la caída.
                est.reiniciar(ts)

            if est.ultima_vista_ts is not None:
                real = kp is not None
                kp_norm = normalizar_frame(kp) if real else est.ultimo_norm
                if kp_norm is not None:
                    est.ultimo_norm = kp_norm
                    vel = est.agregar(ts, kp_norm)
                    if real:
                        # Un frame repetido tiene vel=0 y ensuciaría el
                        # baseline hacia abajo, que después convierte
                        # cualquier movimiento normal en "pico".
                        est.puerta.actualizar(kp_norm, vel, ts)

            toca = (est.ts_ultima_clasificacion is None
                    or (ts - est.ts_ultima_clasificacion) >= cfg.paso_clasif_s)
            if toca and est.listo:
                pendientes.append(est)
            mirados.append((t, est, kp_coco))

        # --- clasificación, todas las personas del frame en un solo forward ---
        if pendientes:
            # cada clip se remuestrea a 25 fps terminando en ESTE ts, así el
            # modelo ve 1,24 s de movimiento a la escala con la que entrenó,
            # venga el pipeline a 8 o a 25 fps
            clips = np.stack([e.clip(ts) for e in pendientes]).astype(np.float32)
            probs = np.asarray(self._clf(clips), dtype=np.float32).reshape(-1)
            self.stats["clasificaciones"] += len(pendientes)
            for est, p in zip(pendientes, probs):
                est.prob = float(p)
                est.ts_ultima_clasificacion = ts
                self._resolver_latch(est, ts)

        fuera = [self._emitir(camera_id, t, est, kp_coco, ts)
                 for t, est, kp_coco in mirados]

        self._recolectar(ts)
        return fuera

    # ------------------------------------------------------------------ #
    def _resolver_latch(self, est: _EstadoTrack, ts: float) -> None:
        """¿Esta persona está caída, ahora mismo?

        El enganche importa y es fácil de perder al portar el script de
        webcam. Allá `alarma_activa` quedaba prendida hasta que la
        probabilidad bajaba; acá, si se exigiera un pico *reciente* en cada
        frame, la etiqueta se apagaría a los `ventana_post_pico_s` (4 s) de la
        caída — y con ella el evento, que cierra a los 5 s de silencio. Se
        perdería justo lo que `ReglaCaida` promete: la severidad que sube con
        los segundos y llega a CRÍTICO a los 15 s **sin levantarse**.

        Entonces: la firma hace falta para **empezar**, no para seguir. Se
        apaga cuando el clasificador dice que ya no está caída — o sea, cuando
        se levantó — o cuando el track se reinicia.
        """
        cfg = self.cfg
        if est.prob < cfg.umbral_reporte:
            est.caida_confirmada = False
            return
        if est.caida_confirmada:
            return
        if est.puerta.firma_reciente(ts, cfg.ventana_post_pico_s):
            est.caida_confirmada = True
        else:
            self.stats["bloqueadas_sin_pico"] += 1

    # ------------------------------------------------------------------ #
    def _emitir(self, camera_id: str, t: Any, est: _EstadoTrack,
                kp_coco: Optional[np.ndarray], ts: float) -> AccionObs:
        cfg = self.cfg
        caja = tuple(float(v) for v in t.bbox_xyxy)
        frame_idx = int(getattr(t, "frame_idx", -1))
        base = dict(bbox_xyxy=caja, track_id=int(t.track_id),
                    camera_id=camera_id, frame_idx=frame_idx, ts=ts)

        if est.listo:
            if est.caida_confirmada:
                self.stats["reportes"] += 1
                return AccionObs(accion="caida", score=float(est.prob),
                                 keypoints=None, fuente=CABEZA_ACCION, **base)

            # El clasificador ve postura de caída pero la puerta no la deja
            # pasar porque no hay firma de transición. En un track recién
            # aparecido eso NO prueba nada: no lo vimos de pie, así que no
            # puede haber transición que medir — es alguien que entró a cuadro
            # ya tirado. Ese caso se le pasa a la geometría de `ReglaCaida`,
            # que le pide 2 s de quietud y lo manda al VLM. Fuera de esa
            # ventana, o si el modelo dice que NO está caído, gana el modelo.
            #
            # Antes esto lo cubría, sin querer, el bug de fps: a 10 fps los 32
            # frames de buffer tardaban 3,2 s en llenarse y durante ese rato se
            # mandaba la pose. A 25 fps eran 1,3 s y el caso quedaba sin cubrir.
            if (cfg.emitir_en_calentamiento and kp_coco is not None
                    and est.prob >= cfg.umbral_reporte
                    and (ts - est.inicio_ts) < cfg.calentamiento_s):
                self.stats["a_geometria"] += 1
                return AccionObs(accion="", score=0.0, keypoints=kp_coco,
                                 fuente=CABEZA_POSE, **base)

            return AccionObs(accion="", score=float(est.prob), keypoints=None,
                             fuente=CABEZA_ACCION, **base)

        # Ventana a medio llenar: el ST-GCN necesita 1,24 s de movimiento y
        # todavía no puede opinar. Mientras tanto se mandan los keypoints, para
        # que `ReglaCaida` pueda usar su camino geométrico. Es más débil que el
        # clasificador, y por eso NO se manda una vez que el clasificador puede
        # hablar: una heurística no pisa a un modelo entrenado. Pero durante el
        # calentamiento es la única evidencia que hay, y alguien que entra a
        # cuadro ya tirado en el piso es exactamente el caso que no se puede
        # perder. El costo está acotado: la regla igual le exige 2 s de quietud
        # y manda a verificar al VLM.
        if cfg.emitir_en_calentamiento and kp_coco is not None:
            self.stats["calentamiento"] += 1
            return AccionObs(accion="", score=0.0, keypoints=kp_coco,
                             fuente=CABEZA_POSE, **base)

        return AccionObs(accion="", score=0.0, keypoints=None,
                         fuente=CABEZA_ACCION, **base)

    # ------------------------------------------------------------------ #
    def _recolectar(self, ts: float, vejez_s: float = 30.0) -> None:
        """Los `track_id` no se reciclan: sin esto, el diccionario de estados
        crece para siempre en un servicio que corre semanas."""
        viejos = [k for k, e in self._estados.items()
                  if e.ultimo_toque_ts is not None
                  and ts - e.ultimo_toque_ts > vejez_s]
        for k in viejos:
            self._estados.pop(k, None)

    # ------------------------------------------------------------------ #
    def olvidar_camara(self, camera_id: str) -> None:
        for k in [k for k in self._estados if k[0] == camera_id]:
            self._estados.pop(k, None)

    def reset(self) -> None:
        self._estados.clear()
        for k in self.stats:
            self.stats[k] = 0

    def resumen(self) -> str:
        s = self.stats
        return (f"[caidas] {s['frames']} frames · {s['personas']} personas · "
                f"pose {s['pose_ok']}/{s['pose_ok'] + s['pose_falla'] + s['descarte_visibilidad']} "
                f"(vis {s['descarte_visibilidad']}, chicas {s['descarte_chica']}) · "
                f"{s['clasificaciones']} clasificaciones · {s['reportes']} reportes · "
                f"{s['bloqueadas_sin_pico']} bloqueadas sin pico · "
                f"{s['calentamiento']} en calentamiento · "
                f"{s['a_geometria']} a geometria · "
                f"{len(self._estados)} tracks vivos")
