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
   suele no ser la de mayor confianza. Acá hay un buffer de 32 frames **por
   `track_id`**, y MediaPipe corre sobre el recorte de cada uno. Como cada
   recorte tiene una sola persona, `num_poses=1` pasa de ser una limitación a
   ser lo correcto.

3. **No decide el evento.** El script disparaba la alarma solo. Acá se emite
   una `AccionObs` y la decisión la toma `ReglaCaida`, que ya sabe pedir
   quietud sostenida y subir la severidad con los segundos — y lo hace mejor,
   porque usa el `quieto_s` del tracker (Kalman sobre la identidad) en vez de
   reinventarlo mirando keypoints.

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
    ventana: int = 32                # frames que come el ST-GCN
    paso: int = 4                    # cada cuántos frames se reclasifica un track

    # --- puertas que necesitan los 33 puntos ---
    umbral_visibilidad: float = 0.50
    tolerancia_sin_ver_s: float = 0.60
    factor_pico_velocidad: float = 3.0
    ventana_post_pico_s: float = 4.0
    alpha_baseline: float = 0.05
    ventana_vertical_previa_s: float = 2.0
    caida_vertical_minima: float = 0.35
    margen_recorte: float = 0.20     # igual que en entrenamiento; ver cabecera

    # --- lo que agrega Horus ---
    solo_confirmados: bool = True    # el tracker ya filtró el parpadeo
    min_alto_px: int = 80            # una persona de 40 px da una pose inventada
    max_personas: int = 4            # tope de recortes por cámara por frame
    emitir_en_calentamiento: bool = True   # ver `_emitir_pose`
    umbral_reporte: float = 0.50     # por debajo no se etiqueta "caida"

    def __post_init__(self) -> None:
        if self.ventana < 4:
            raise ValueError("ventana muy corta para el ST-GCN")
        if self.paso < 1:
            raise ValueError("paso >= 1")


# --------------------------------------------------------------------------- #
# Estado por track
# --------------------------------------------------------------------------- #
class _EstadoTrack:
    """Todo lo que hay que recordar de UNA persona. Uno por (cámara, track).

    El script de webcam tenía estas mismas variables sueltas en el `main`, que
    es lo mismo que decir que soportaba una sola persona.
    """

    __slots__ = ("buffer", "ultimo_norm", "anterior_norm", "baseline_vel",
                 "ultimo_pico_ts", "historial_vertical", "ultima_vista_ts",
                 "prob", "n", "ts_ultima_clasificacion", "caida_confirmada",
                 "ultimo_toque_ts")

    def __init__(self, ventana: int) -> None:
        self.buffer: deque = deque(maxlen=ventana)
        self.ultimo_norm: Optional[np.ndarray] = None
        self.anterior_norm: Optional[np.ndarray] = None
        self.baseline_vel: Optional[float] = None
        self.ultimo_pico_ts: Optional[float] = None
        self.historial_vertical: deque = deque()
        self.ultima_vista_ts: Optional[float] = None
        self.prob: float = 0.0
        self.n: int = 0
        self.ts_ultima_clasificacion: float = 0.0
        # Engancha en True cuando el clasificador confirma una caída con un
        # pico reciente que la respalde, y se apaga cuando la persona se
        # levanta. Ver `DetectorCaidas._resolver_latch`.
        self.caida_confirmada: bool = False
        self.ultimo_toque_ts: Optional[float] = None

    def reiniciar(self) -> None:
        """Se perdió a la persona demasiado tiempo: no se sigue clasificando
        con pose congelada.

        El baseline y el pico también se tiran. Si quedaran de antes de que la
        persona saliera de cuadro pueden hacer las dos cosas malas: un baseline
        viejo y alto bloquea una caída real al volver, y un pico viejo valida
        una alarma con un movimiento que no tiene nada que ver.
        """
        self.buffer.clear()
        self.ultimo_norm = None
        self.anterior_norm = None
        self.baseline_vel = None
        self.ultimo_pico_ts = None
        self.historial_vertical.clear()
        self.prob = 0.0
        self.caida_confirmada = False


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
        estado = torch.load(checkpoint, map_location=self.device)
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
def normalizar_frame(kp: np.ndarray) -> np.ndarray:
    """(33, >=2) -> (33, 2), centrado en la cadera y escalado por el torso.

    Es causal por definición: mira un frame y nada más. Copiada tal cual de
    `scripts/12_webcam.py` — si esto se desvía del entrenamiento, el modelo ve
    otra distribución y no hay umbral que lo arregle.
    """
    cad = (kp[IDX_CI, :2] + kp[IDX_CD, :2]) / 2.0
    hom = (kp[IDX_HI, :2] + kp[IDX_HD, :2]) / 2.0
    esc = float(np.linalg.norm(hom - cad))
    esc = esc if esc > 1e-6 else 1e-6
    return (kp[:, :2] - cad) / esc


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
            "bloqueadas_sin_pico": 0,
        }

    # ------------------------------------------------------------------ #
    def _asegurar_backends(self) -> None:
        if self._pose is None:
            self._pose = BackendPoseMediaPipe(self.cfg.modelo_pose)
        if self._clf is None:
            self._clf = ClasificadorSTGCN(self.cfg.checkpoint,
                                          canales=self.cfg.canales,
                                          device=self.cfg.device)
            if self._verboso:
                print(f"[caidas] ST-GCN en {self._clf.device} · "
                      f"{os.path.basename(self.cfg.checkpoint)}")

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
    def _actualizar_puertas(self, est: _EstadoTrack, kp_norm: np.ndarray,
                            vel: np.ndarray, ts: float) -> None:
        """Pico de velocidad + caída de verticalidad.

        Esto es lo único que separa "se cayó" de "está acostado". El
        clasificador no puede: 32 frames de alguien quieto en el piso se ven
        igual se haya caído hace dos segundos o se haya acostado hace una
        hora. Lo que distingue la caída es que **hubo un movimiento brusco
        justo antes**, y que **antes de eso estaba parado**.

        Las dos condiciones hacen falta. Solo el pico no alcanza: si la
        persona ya está en el piso y se acomoda, el baseline de velocidad ya
        cayó casi a cero por la quietud previa, y ese movimiento normal se ve
        como un pico gigante en términos relativos.
        """
        cfg = self.cfg
        v_escalar = float(np.linalg.norm(vel[list(IDX_TORSO)], axis=1).mean())
        # ~1 con el torso vertical (de pie), ~0 con el torso horizontal.
        verticalidad = -float((kp_norm[IDX_HI, 1] + kp_norm[IDX_HD, 1]) / 2.0)

        if est.baseline_vel is None:
            est.baseline_vel = v_escalar
        elif (est.baseline_vel > 1e-6
              and v_escalar > est.baseline_vel * cfg.factor_pico_velocidad):
            cobertura = (ts - est.historial_vertical[0][0]
                         if est.historial_vertical else 0.0)
            if cobertura < cfg.ventana_vertical_previa_s * 0.8:
                # Recién la vemos: no puedo probar que "ya estaba acostada de
                # antes", porque no la vi antes. Prefiero un falso positivo
                # ocasional a perderme una caída que pasó justo al aparecer.
                est.ultimo_pico_ts = ts
            else:
                vertical_previa = max(v for _, v in est.historial_vertical)
                if (vertical_previa - verticalidad) >= cfg.caida_vertical_minima:
                    est.ultimo_pico_ts = ts
        else:
            est.baseline_vel = (est.baseline_vel * (1.0 - cfg.alpha_baseline)
                                + v_escalar * cfg.alpha_baseline)

        est.historial_vertical.append((ts, verticalidad))
        while (est.historial_vertical
               and ts - est.historial_vertical[0][0] > cfg.ventana_vertical_previa_s):
            est.historial_vertical.popleft()

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
                est = self._estados[clave] = _EstadoTrack(cfg.ventana)
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
            # nunca y el buffer queda con 32 frames que son mitad de hace un
            # minuto y mitad de ahora. Clasifica, y clasifica cualquier cosa.
            previa = est.ultima_vista_ts
            if kp is not None:
                est.ultima_vista_ts = ts
            if previa is not None and (ts - previa) > cfg.tolerancia_sin_ver_s:
                # Hace rato que no hay pose confiable: se tira todo en vez de
                # seguir clasificando con datos congelados. El baseline y el
                # pico se van con el resto: si sobrevivieran, un pico viejo
                # podría validar una alarma con un movimiento de hace un
                # minuto, y un baseline viejo y alto podría bloquear una
                # caída real al reaparecer.
                est.reiniciar()

            if est.ultima_vista_ts is not None:
                real = kp is not None
                kp_norm = normalizar_frame(kp) if real else est.ultimo_norm
                if kp_norm is not None:
                    est.ultimo_norm = kp_norm
                    vel = (kp_norm - est.anterior_norm
                           if est.anterior_norm is not None
                           else np.zeros_like(kp_norm))
                    est.anterior_norm = kp_norm
                    est.buffer.append(np.concatenate([kp_norm, vel], axis=1))
                    if real:
                        # Un frame repetido tiene vel=0 y ensuciaría el
                        # baseline hacia abajo, que después convierte
                        # cualquier movimiento normal en "pico".
                        self._actualizar_puertas(est, kp_norm, vel, ts)

            est.n += 1
            if len(est.buffer) == cfg.ventana and est.n % cfg.paso == 0:
                pendientes.append(est)
            mirados.append((t, est, kp_coco))

        # --- clasificación, todas las personas del frame en un solo forward ---
        if pendientes:
            clips = np.stack([np.asarray(e.buffer, dtype=np.float32)
                              for e in pendientes])
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

        Entonces: el pico hace falta para **empezar**, no para seguir. Se
        apaga cuando el clasificador dice que ya no está caída — o sea, cuando
        se levantó — o cuando el track se reinicia.
        """
        cfg = self.cfg
        if est.prob < cfg.umbral_reporte:
            est.caida_confirmada = False
            return
        if est.caida_confirmada:
            return
        hubo_pico = (est.ultimo_pico_ts is not None
                     and (ts - est.ultimo_pico_ts) <= cfg.ventana_post_pico_s)
        if hubo_pico:
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

        if len(est.buffer) == cfg.ventana:
            if est.caida_confirmada:
                self.stats["reportes"] += 1
            return AccionObs(accion="caida" if est.caida_confirmada else "",
                             score=float(est.prob), keypoints=None,
                             fuente=CABEZA_ACCION, **base)

        # Buffer a medio llenar: el ST-GCN necesita 32 frames (~3 s a 10 FPS) y
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
                f"{len(self._estados)} tracks vivos")
