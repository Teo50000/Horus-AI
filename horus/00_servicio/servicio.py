# -*- coding: utf-8 -*-
"""
servicio.py · HORUS — el proceso que corre 24/7 con la GPU.

Es lo que faltaba para unir las tres partes: el backend maneja las cámaras y
el historial, el front las muestra, y esto de acá mira los videos y decide.

    ┌─ FastAPI (backend) ──────────┐        ┌─ servicio.py (esta caja) ───┐
    │  GET  /camaras/config        │◄───────┤  consulta las cámaras       │
    │  POST /alertas               │◄───────┤  manda las alertas          │
    │  WebSocket -> front          │        │  modelos + tracking + fusión│
    └──────────────────────────────┘        └─────────────────────────────┘

Cómo hace que "al conectar una cámara ya esté el modelo andando"
---------------------------------------------------------------
Los modelos se cargan UNA vez, cuando arranca el servicio, antes de que
exista ninguna cámara. Enganchar una cámara después no vuelve a cargar nada:
solo se le abre una fuente de video y se suma al lote de inferencia que ya
está corriendo. Del alta en el backend a la primera detección pasan los
segundos del sondeo, no los ~20 s que tarda un backbone en subir a la placa.

Cuatro decisiones que importan
------------------------------
1. **Un hilo de captura por cámara, y guarda UN solo frame.** Nunca una cola.
   Si la inferencia va a 10 FPS y la cámara entrega 25, encolar acumula un
   retraso que crece toda la noche: a las tres horas estarías alertando sobre
   algo que pasó hace veinte minutos. Guardar solo el último frame significa
   descartar los intermedios, que es exactamente lo correcto en vigilancia.

2. **Un solo hilo de inferencia con todas las cámaras en un lote.** El
   backbone es el 53% del costo por frame y a batch 1 la GPU no llega ni al
   20% de su pico. Juntar las cámaras en un forward es lo que hace que la
   cuenta de `dimensionamiento-camaras-por-gpu` cierre.

3. **El servicio arranca aunque el backend esté caído.** Carga los modelos,
   levanta las cámaras del archivo de respaldo si lo hay, y sigue
   reintentando. Un sistema de alertas que no arranca porque otro servicio no
   está es un sistema que no sirve.

4. **Las alertas que no se pudieron entregar van a disco** y se reenvían
   cuando el backend vuelve. Eso ya lo hace `07_alerta/alerta.py`.

Correr
------
    python servicio.py --pesos ../04_cabezas/objetos/modelos/head_best_solo.pt \
                       --backend http://127.0.0.1:8000 \
                       --topologia ../06_fusion_decision/topologia.json

Sin GPU ni cámaras, para probar el cableado entero:

    python servicio.py --simular
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))

# Sin proxy, a propósito. El backend vive en esta misma máquina o en la LAN, y
# un HTTP_PROXY en el entorno —de una VPN, de un antivirus, del equipo de
# sistemas— mandaría el pedido a un proxy que no sabe resolver 127.0.0.1. La
# falla sería muda: el servicio creería que el backend no está, no vería
# ninguna cámara, y se quedaría cargado esperando para siempre. Es la misma
# trampa que ya está documentada en horus/07_alerta/alerta.py, que sí la
# esquivaba.
_SIN_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_RAIZ = os.path.dirname(_AQUI)
for _p in (_AQUI,
           os.path.join(_RAIZ, "05_tracking"),
           os.path.join(_RAIZ, "05_tracking", "local"),
           os.path.join(_RAIZ, "05_tracking", "global_reid"),
           os.path.join(_RAIZ, "06_fusion_decision"),
           os.path.join(_RAIZ, "07_alerta"),
           os.path.join(_RAIZ, "04_cabezas", "objetos"),
           os.path.join(_RAIZ, "fall")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alerta import ConfigAlerta, EmisorAlertas, TransporteHTTP  # noqa: E402
from evidencia import Grabadora  # noqa: E402


# --------------------------------------------------------------------------- #
@dataclass
class ConfigServicio:
    # --- backend ----------------------------------------------------------
    backend: str = "http://127.0.0.1:8000"
    token: Optional[str] = None
    ruta_camaras: str = "/camaras/config"
    ruta_alertas: str = "/alertas"
    sondeo_s: float = 5.0                  # cada cuánto se re-consultan las cámaras

    # --- modelos ----------------------------------------------------------
    pesos: Optional[str] = None
    pesos_backbone: Optional[str] = None
    topologia: Optional[str] = None
    device: Optional[str] = None
    input_size: Optional[int] = None

    # --- cabeza de caídas (fall/) ----------------------------------------
    # Apagada por defecto: cuesta una pasada de MediaPipe por persona por
    # frame, que es el costo dominante de esta cabeza, y no todas las
    # instalaciones la quieren. Si se pide y no se puede cargar, el servicio
    # NO arranca — ver `Servicio.iniciar()`.
    caidas: bool = False
    caidas_checkpoint: Optional[str] = None
    caidas_max_personas: int = 4

    # --- cabeza de segmentación (04_cabezas/segmentacion) ----------------
    # 18/09: el servicio no la prendía NUNCA. El pipeline la acepta desde
    # siempre (`motor_segmentacion`) y nadie se la pasaba, así que el sistema
    # corría sin la única cabeza que de verdad anda bien —F1 99,1 % medido en
    # fuego y humo— y sin decirlo. Es además el único camino completo a una
    # alerta real que no depende del backbone de objetos.
    segmentacion: bool = False
    segmentacion_checkpoint: Optional[str] = None

    # --- cabeza de agresión (fight/) -------------------------------------
    # Apagada por defecto igual que caídas: cuesta una pasada de MC3-18 cada
    # 0,75 s por cámara con dos personas en cuadro.
    agresion: bool = False
    agresion_checkpoint: Optional[str] = None

    # --- video ------------------------------------------------------------
    fps: float = 10.0                      # ritmo de análisis, no de la cámara
    max_batch: int = 4
    reconexion_s: float = 3.0
    ancho_captura: int = 1280              # 0 = lo que dé la cámara
    alto_captura: int = 720
    # Una RTSP puede anunciar 4K aunque el modelo y el panel trabajen en
    # 720p. Limitar el frame evita que una sola fuente agote CPU y memoria al
    # inferir, dibujar y guardar evidencia.
    max_ancho_captura: int = 1280
    max_alto_captura: int = 720

    # --- salida -----------------------------------------------------------
    spool: str = "alertas_pendientes.jsonl"
    severidad_min: int = 1
    sitio: str = ""
    nodo: str = "caja-01"

    # --- streaming propio -------------------------------------------------
    puerto_stream: int = 8010              # 0 = no levantar el servidor MJPEG
    dibujar: bool = True
    # 16/11 — el video del panel ya no va al ritmo de los modelos. Ver
    # `_difundir`: se manda cada cuadro NUEVO de la cámara (hasta fps_video)
    # con las últimas cajas encima, al ancho que pide la celda del panel.
    fps_video: float = 25.0
    calidad_video: int = 75
    ancho_video_max: int = 1280
    # Las detecciones sueltas (gris) y los tracks tentativos sirven para
    # calibrar, no para vigilar: en pantalla son ruido. --crudas las prende.
    dibujar_crudas: bool = False

    # --- detector preentrenado (04_cabezas/objetos/motor_yolo.py) --------
    # 26/09: YOLO11 de COCO para persona, cuchillo y celular. La cabeza propia
    # queda solo para pistola, humo, llama y paquete. "auto" = se usa si están
    # los pesos en modelos/ y ultralytics instalado; si no, se sigue como antes
    # y se dice fuerte. "no" = solo la cabeza propia. Otra cosa = ruta.
    yolo: str = "auto"
    yolo_tam: int = 416
    yolo_bolsos: bool = False

    # --- evidencia (07_alerta/evidencia.py) ------------------------------
    # 26/09: cada alerta que abre o sube de severidad lleva la captura del
    # momento y un clip de los `clip_s` segundos anteriores.
    evidencia: bool = True
    clip_s: float = 8.0

    # --- respaldo y pruebas ----------------------------------------------
    camaras_json: Optional[str] = None     # lista local si el backend no está
    simular: bool = False
    verboso: bool = True

    def url(self, ruta: str) -> str:
        return self.backend.rstrip("/") + ruta


def _log(cfg: ConfigServicio, *a: Any) -> None:
    if cfg.verboso:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _hay_pesos_yolo() -> bool:
    try:
        from motor_yolo import pesos_por_defecto
        return pesos_por_defecto() is not None
    except Exception:                                    # noqa: BLE001
        return False


def _ascii(texto: str) -> str:
    """cv2.putText solo sabe ASCII: "CRÍTICO · cam-1" salía "CR??TICO ?? cam-1".
    Antes se veía solo en el panel; desde el 26/09 va en la captura que llega
    por mail, así que se escribe sin tildes."""
    import unicodedata
    t = str(texto).replace("·", "|").replace("—", "-").replace("–", "-")
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")


# --------------------------------------------------------------------------- #
# Fuente de video
# --------------------------------------------------------------------------- #
class FuenteCamara:
    """Un hilo que mantiene abierta UNA cámara y guarda su último frame.

    Guarda uno solo, no una cola: ver la decisión 1 del encabezado.
    """

    def __init__(self, cam_id: str, url: str, cfg: ConfigServicio,
                 config_id: Optional[int] = None) -> None:
        self.cam_id = cam_id
        self.url = url
        self.cfg = cfg
        self.config_id = config_id

        self._frame: Optional[np.ndarray] = None
        self._ts = 0.0
        self._seq = 0                      # sube con cada cuadro nuevo
        self._lock = threading.Lock()
        # Despierta al video del panel apenas hay un cuadro nuevo (sin
        # sondear): comparte el lock de arriba.
        self._nuevo = threading.Condition(self._lock)
        self._parar = threading.Event()
        self._hilo: Optional[threading.Thread] = None

        self.estado = "conectando"         # conectando | ok | caida
        self.frames_leidos = 0
        self.reconexiones = 0
        self.ultimo_error = ""

    # ------------------------------------------------------------------ #
    def iniciar(self) -> None:
        self._hilo = threading.Thread(target=self._correr,
                                      name=f"cam-{self.cam_id}", daemon=True)
        self._hilo.start()

    def parar(self) -> None:
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=3.0)

    @property
    def frame(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            return (None if self._frame is None else self._frame.copy()), self._ts

    def ultimo(self) -> Tuple[Optional[np.ndarray], float, int]:
        """(cuadro, ts, número) SIN copiar. Cada lectura de la cámara crea un
        array nuevo y nadie escribe sobre uno ya publicado, así que el que
        lo recibe lo puede leer tranquilo. Lo usa el video del panel, que
        pide un cuadro 25 veces por segundo y no tiene por qué copiarlo."""
        with self._lock:
            return self._frame, self._ts, self._seq

    def esperar(self, visto: int, timeout: float = 1.0
                ) -> Tuple[Optional[np.ndarray], float, int]:
        """Como `ultimo()`, pero espera hasta que haya un cuadro distinto de
        `visto` (o se cumpla el timeout)."""
        with self._nuevo:
            self._nuevo.wait_for(lambda: self._seq != visto or self._parar.is_set(),
                                 timeout=timeout)
            return self._frame, self._ts, self._seq

    # ------------------------------------------------------------------ #
    def _abrir(self):
        if self.cfg.simular:
            return _CapturaSimulada(self.cam_id)
        import cv2
        fuente: Any = int(self.url) if str(self.url).isdigit() else self.url

        # En Windows OpenCV usa MSMF por defecto y hay muchas webcams que MSMF
        # no abre (o tarda diez segundos), mientras DirectShow las abre al
        # toque. El síntoma es "la cámara anda en la app de Windows y en
        # Discord pero acá no": esos programas no pasan por MSMF. Para RTSP no
        # aplica, eso va por FFMPEG.
        cap = None
        if isinstance(fuente, int) and os.name == "nt":
            cap = cv2.VideoCapture(fuente, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = None
        if cap is None:
            if isinstance(fuente, str) and fuente.lower().startswith(
                    ("rtsp://", "rtsps://")):
                parametros = []
                for nombre, valor in (
                        ("CAP_PROP_OPEN_TIMEOUT_MSEC", 5000),
                        ("CAP_PROP_READ_TIMEOUT_MSEC", 5000)):
                    propiedad = getattr(cv2, nombre, None)
                    if propiedad is not None:
                        parametros.extend((propiedad, valor))
                try:
                    cap = (cv2.VideoCapture(fuente, cv2.CAP_FFMPEG, parametros)
                           if parametros else cv2.VideoCapture(fuente))
                except (TypeError, cv2.error):
                    # Algunas versiones de OpenCV no aceptan parámetros al
                    # abrir una fuente; se conserva el fallback compatible.
                    cap = cv2.VideoCapture(fuente)
            else:
                cap = cv2.VideoCapture(fuente)

        if cap.isOpened():
            # Que OpenCV no acumule frames viejos adentro suyo. Sin esto, el
            # búfer interno de RTSP mete su propio retraso además del nuestro.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if self.cfg.ancho_captura:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.ancho_captura)
            if self.cfg.alto_captura:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.alto_captura)
        return cap

    def _limitar_frame(self, frame: np.ndarray) -> np.ndarray:
        """Mantiene el aspecto y evita procesar RTSP 2K/4K innecesariamente."""
        max_ancho = self.cfg.max_ancho_captura
        max_alto = self.cfg.max_alto_captura
        if not max_ancho or not max_alto:
            return frame
        alto, ancho = frame.shape[:2]
        escala = min(max_ancho / ancho, max_alto / alto, 1.0)
        if escala >= 1.0:
            return frame
        import cv2
        return cv2.resize(
            frame, (max(1, int(ancho * escala)), max(1, int(alto * escala))),
            interpolation=cv2.INTER_AREA)

    def _correr(self) -> None:
        cap = None
        while not self._parar.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                    self.reconexiones += 1
                cap = self._abrir()
                if cap is None or not cap.isOpened():
                    self.estado = "caida"
                    self.ultimo_error = f"no abre {self.url!r}"
                    _log(self.cfg, f"  [{self.cam_id}] no abre, reintento en "
                                   f"{self.cfg.reconexion_s:.0f} s")
                    self._parar.wait(self.cfg.reconexion_s)
                    continue
                self.estado = "ok"
                _log(self.cfg, f"  [{self.cam_id}] conectada")

            ok, frame = cap.read()
            if not ok or frame is None:
                # Una RTSP se cae sola cada tanto. No es excepcional, es el
                # caso normal: se reconecta y sigue.
                self.estado = "caida"
                cap.release()
                cap = None
                continue

            frame = self._limitar_frame(frame)
            with self._lock:
                self._frame = frame
                self._ts = time.time()
                self._seq += 1
                self._nuevo.notify_all()
            self.frames_leidos += 1
            self.estado = "ok"

        if cap is not None:
            cap.release()

    def resumen(self) -> Dict[str, Any]:
        return {"camara": self.cam_id, "config_id": self.config_id,
                "url": self.url, "estado": self.estado,
                "frames": self.frames_leidos, "reconexiones": self.reconexiones,
                "error": self.ultimo_error}


class _CapturaSimulada:
    """Video sintético: una figura que se mueve. Para probar sin cámara."""

    def __init__(self, cam_id: str) -> None:
        self.n = 0
        self.cam_id = cam_id
        self._abierta = True

    def isOpened(self) -> bool:
        return self._abierta

    def read(self):
        self.n += 1
        img = np.full((480, 640, 3), 30, dtype=np.uint8)
        x = int((self.n * 6) % 560)
        img[200:400, x:x + 60] = (180, 180, 190)
        return True, img

    def set(self, *a):
        return True

    def release(self) -> None:
        self._abierta = False


# --------------------------------------------------------------------------- #
# Motor simulado — permite probar TODO el cableado sin GPU ni pesos
# --------------------------------------------------------------------------- #
class _Det:
    __slots__ = ("bbox_xyxy", "score", "label", "class_id", "camera_id",
                 "frame_idx", "ts", "needs_vlm")

    def __init__(self, caja, score, label, class_id, cam, idx, ts):
        self.bbox_xyxy = tuple(float(v) for v in caja)
        self.score, self.label, self.class_id = float(score), label, int(class_id)
        self.camera_id, self.frame_idx, self.ts = cam, idx, ts
        self.needs_vlm = False


class _Resultado:
    __slots__ = ("detections", "detecciones_crudas", "camera_id", "frame_idx",
                 "novelty", "incertidumbre", "original_size", "input_size",
                 "latencia_ms", "embedding", "ts")

    def __init__(self, dets, cam, idx, tam):
        self.detections = self.detecciones_crudas = dets
        self.camera_id, self.frame_idx = cam, idx
        self.novelty = self.incertidumbre = 0.0
        self.original_size = self.input_size = tam
        self.latencia_ms = 0.0
        self.embedding = None
        self.ts = time.time()


class MotorSimulado:
    """Hace de cabeza de objetos sin torch.

    Emite una persona caminando y, pasados unos segundos, una llama. Alcanza
    para verificar que el cableado entero funciona — tracking, fusión, alerta,
    backend, WebSocket — antes de que la GPU entre en la ecuación.
    """

    def __init__(self, fps: float = 10.0) -> None:
        self.fps = fps
        self._n: Dict[str, int] = {}

    def resumen(self) -> str:
        return "MOTOR SIMULADO (sin GPU, sin pesos) — solo para probar el cableado"

    def infer_batch(self, frames, camera_ids=None, frame_idxs=None):
        cams = list(camera_ids or [f"cam-{i}" for i in range(len(frames))])
        salida = []
        for frame, cam in zip(frames, cams):
            n = self._n.get(cam, 0)
            self._n[cam] = n + 1
            alto, ancho = frame.shape[:2]
            t = n / max(self.fps, 1e-6)

            dets = []
            x = 40 + (n * 6) % (ancho - 140)
            dets.append(_Det((x, 200, x + 60, 400), 0.62, "persona", 2, cam, n, t))
            if t > 4.0:                        # a los 4 s aparece fuego
                dets.append(_Det((ancho - 180, 120, ancho - 90, 240),
                                 0.71, "llama", 1, cam, n, t))
            salida.append(_Resultado(dets, cam, n, (alto, ancho)))
        return salida


# --------------------------------------------------------------------------- #
# Transporte que le agrega el id de configuración a cada alerta
# --------------------------------------------------------------------------- #
class TransporteBackend:
    """`TransporteHTTP` + el `camara_config_id` que el backend necesita.

    Horus nombra las cámaras `cam-3`; el backend las guarda por id numérico.
    La traducción se hace acá, en el borde, para no meterle conocimiento del
    backend ni a la fusión ni a `alerta.py`.
    """

    nombre = "backend"

    def __init__(self, cfg_alerta: ConfigAlerta,
                 mapa: Dict[str, Optional[int]],
                 grabadora: Optional[Grabadora] = None) -> None:
        # 18/09, MEDIDO: acá decía `TransporteHTTP(cfg_alerta)`, pasándole el
        # objeto de configuración entero donde va la URL. El primer parámetro
        # de TransporteHTTP es `url: str`, así que self.url quedaba siendo un
        # ConfigAlerta y urllib moría con "unknown url type: configalerta(...".
        # El emisor atrapa el error, manda la alerta al spool y sigue: o sea
        # que TODAS las alertas del servicio iban a un archivo en disco y
        # ninguna llegaba al backend, sin una sola línea de error visible.
        self._http = TransporteHTTP(cfg_alerta.url, cfg_alerta.token)
        self.mapa = mapa
        self.grabadora = grabadora
        # evento_id -> severidad con la que ya se mandó evidencia.
        self._con_evidencia: Dict[str, int] = {}

    def enviar(self, payload: Dict[str, Any]) -> None:
        cam = payload.get("sitio", {}).get("camara", "")
        cid = self.mapa.get(cam)
        if cid is not None:
            payload.setdefault("sitio", {})["camara_config_id"] = cid
        self._adjuntar(payload, cam)
        self._http.enviar(payload)

    def _adjuntar(self, payload: Dict[str, Any], cam: str) -> None:
        """La captura y el clip, solo cuando el evento ABRE o SUBE de severidad.

        Son los mismos dos momentos en que el backend lo muestra en el panel y
        manda el mail. Los re-avisos cada 20 s no llevan nada: el backend les
        hereda la evidencia del mensaje anterior, y mandar un clip nuevo cada
        20 s sería llenar el disco con el mismo incendio.

        Nunca levanta: una alerta sin imagen es mejor que una alerta que no
        sale. Si el POST falla y se reintenta, el payload ya trae los
        adjuntos y no se vuelven a armar.
        """
        if self.grabadora is None or "adjuntos" in payload:
            return
        evento = str(payload.get("id", ""))
        sev = int(payload.get("severidad_num", 0) or 0)
        if sev <= self._con_evidencia.get(evento, -1):
            return
        try:
            adj = self.grabadora.adjuntos(cam)
        except Exception as exc:                         # noqa: BLE001
            print(f"[evidencia] no pude armar la evidencia de {evento}: {exc}",
                  file=sys.stderr, flush=True)
            return
        if not adj:
            return
        payload["adjuntos"] = adj
        self._con_evidencia[evento] = sev
        if len(self._con_evidencia) > 1000:              # no crecer para siempre
            for k in list(self._con_evidencia)[:500]:
                self._con_evidencia.pop(k, None)


# --------------------------------------------------------------------------- #
# Servicio
# --------------------------------------------------------------------------- #
class Servicio:
    def __init__(self, cfg: ConfigServicio) -> None:
        self.cfg = cfg
        self.fuentes: Dict[str, FuenteCamara] = {}
        self.config_ids: Dict[str, Optional[int]] = {}
        self.pipe: Any = None
        self.emisor: Any = None

        self._parar = threading.Event()
        self._hilos: List[threading.Thread] = []
        self._anotados: Dict[str, bytes] = {}
        self._lock_anotados = threading.Lock()
        # Video del panel (ver `_difundir`). Lo que dibujar por cámara, el
        # último JPEG por (cámara, ancho) y cuántos lo están mirando.
        self._prim: Dict[str, Dict[str, Any]] = {}
        self._video: Dict[Tuple[str, int], Tuple[int, bytes]] = {}
        self._cond_video = threading.Condition()
        self._espectadores: Dict[Tuple[str, int], int] = {}
        self._difusores: Dict[str, threading.Thread] = {}
        self._srv_http: Any = None
        # Los últimos segundos de cada cámara, para adjuntarlos a la alerta.
        self.grabadora: Optional[Grabadora] = (
            Grabadora(segundos=cfg.clip_s) if cfg.evidencia else None)

        self.detector: Optional[str] = None
        self.ticks = 0
        self.eventos = 0
        self.arranque = 0.0
        self.listo_en_s = 0.0
        self.fase = "arrancando"
        self._ultimo_resumen = 0.0

    # ------------------------------------------------------------------ #
    def iniciar(self) -> None:
        c = self.cfg
        self.arranque = time.time()

        # El servidor HTTP arranca ANTES de los modelos, a propósito.
        #
        # 18/09, medido: cargar el motor de objetos tarda 17 s en CPU. Hasta
        # hoy el servidor se levantaba DESPUES, asi que durante esos 17 s el
        # puerto 8010 no contestaba nada y el panel mostraba "MODELOS
        # APAGADOS" — indistinguible de que el servicio no estuviera. Uno abre
        # el panel, lee APAGADOS, y da por hecho que algo se rompio.
        #
        # "Todavia no cargo" y "no hay nadie" son cosas distintas y tienen que
        # verse distintas. Es el mismo criterio que reglas_dormidas() en la
        # fusion.
        self.fase = "cargando"
        if c.puerto_stream:
            self._levantar_http()

        _log(c, "cargando modelos…")

        from pipeline import PipelineHorus

        motor = MotorSimulado(c.fps) if c.simular else None
        topo = c.topologia if (c.topologia and os.path.exists(c.topologia)) else None
        if c.topologia and topo is None:
            _log(c, f"aviso: no encuentro {c.topologia}; sigo sin zonas")

        motor_seg = None
        if c.segmentacion and not c.simular:
            # Mismo trato que las otras cabezas: si se pide y no carga, el
            # servicio NO arranca. Correr sin la cabeza pero creyendo que está
            # es la falla que este sistema no se puede permitir.
            try:
                _seg = os.path.normpath(os.path.join(
                    _AQUI, "..", "04_cabezas", "segmentacion"))
                if _seg not in sys.path:
                    sys.path.insert(0, _seg)
                from segmentation_engine import (ConfigSegmentacion,
                                                 MotorSegmentacion)
                cfg_s = ConfigSegmentacion()
                if c.segmentacion_checkpoint:
                    cfg_s.pesos = c.segmentacion_checkpoint
                if c.device:
                    cfg_s.device = c.device
                motor_seg = MotorSegmentacion(cfg_s)
            except Exception as e:                       # noqa: BLE001
                raise RuntimeError(
                    f"se pidió la cabeza de segmentación y no se pudo cargar: "
                    f"{type(e).__name__}: {e}\nHacen falta "
                    f"horus/04_cabezas/segmentacion/checkpoints/"
                    f"head_v4_produccion.pt y backbone.pt.") from e

        det_caidas = None
        if c.caidas:
            # Se construye acá, y si falla se corta. Arrancar "igual pero sin
            # caídas" sería lo peor posible: el operador cree que el sistema
            # las está mirando. Misma razón por la que una regla sin cabeza se
            # declara dormida en vez de devolver "no pasó nada".
            try:
                # Mismo armado de path que `PipelineHorus._crear_caidas`. Acá
                # faltaba, y `--caidas` (que HORUS.bat pasa solo si están los
                # pesos) cortaba el arranque con "No module named
                # 'detector_caidas'".
                _fall = os.path.normpath(os.path.join(_AQUI, "..", "fall"))
                for _p in (_fall, os.path.join(_fall, "src")):
                    if _p not in sys.path:
                        sys.path.insert(0, _p)
                from detector_caidas import ConfigCaidas, DetectorCaidas
                cfg_c = ConfigCaidas(max_personas=c.caidas_max_personas)
                if c.caidas_checkpoint:
                    cfg_c.checkpoint = c.caidas_checkpoint
                # .cargar(): los backends son perezosos; sin esto un .task o
                # un mediapipe faltante no caía en este try sino en el primer
                # frame, con el servicio ya "andando".
                det_caidas = DetectorCaidas(cfg_c, verboso=c.verboso).cargar()
            except Exception as e:                       # noqa: BLE001
                raise RuntimeError(
                    f"se pidió la cabeza de caídas y no se pudo cargar: "
                    f"{type(e).__name__}: {e}\nHace falta `pip install "
                    f"mediapipe torch` y que estén horus/fall/"
                    f"pose_landmarker.task y el checkpoint. Sin eso el "
                    f"servicio no arranca a propósito: correr sin la cabeza "
                    f"pero creyendo que está es la falla que este sistema no "
                    f"se puede permitir.") from e

        cfg_tracker = None
        pesos_propios = None if c.simular else c.pesos
        if not c.simular and c.yolo != "no":
            yolo = self._cargar_yolo()
            if yolo is not None:
                from motor_yolo import MotorMixto
                from tracker_local import ConfigTracker
                # Si ya está el YOLO reentrenado con pistola/humo/llama/paquete
                # (entrenar_yolo.py), la cabeza propia no se carga: era lo más
                # lento del servicio (un ResNet-50 entero por cuadro).
                propio = self._cargar_yolo_horus()
                if propio is None and c.pesos:
                    propio = PipelineHorus._crear_motor(
                        c.pesos, c.pesos_backbone, max(c.max_batch, 1), c.device,
                        c.verboso)
                motor = MotorMixto(yolo, propio) if propio is not None else yolo
                pesos_propios = None
                cfg_tracker = ConfigTracker(fps=c.fps)
                # Otra escala de scores: YOLO le da 0,6-0,9 a una persona de
                # verdad y la silla con ropa le llega a 0,3-0,4 (medido en
                # los 293 cuadros vacíos: 33 cuadros >= 0,25, 8 >= 0,40).
                p = cfg_tracker.params("persona")
                p.score_alto, p.score_bajo = 0.40, 0.20
                if propio is None:
                    self.detector = "YOLO11"
                elif hasattr(propio, "mapa"):
                    self.detector = "YOLO11 + YOLO11 de Horus (pistola, humo, llama, paquete)"
                else:
                    self.detector = "YOLO11 + cabeza propia (pistola, humo, llama, paquete)"
        detector_agresion = None
        if c.agresion and not c.simular:
            # Mismo armado de path que `PipelineHorus._crear_agresion`: sin
            # esto, corriendo desde el código (HORUS.bat / Horus.exe en modo
            # proyecto) `--agresion` cortaba el arranque con "No module named
            # 'detector_agresion'".
            _fight = os.path.normpath(os.path.join(_AQUI, "..", "fight"))
            if _fight not in sys.path:
                sys.path.insert(0, _fight)
            from detector_agresion import ConfigAgresion
            detector_agresion = ConfigAgresion()
            # Sin --agresion-checkpoint queda el de por defecto: pasar None
            # pisaba la ruta y el chequeo del archivo reventaba con TypeError.
            if c.agresion_checkpoint:
                detector_agresion.checkpoint = c.agresion_checkpoint
        self.pipe = PipelineHorus(
            pesos=pesos_propios,
            pesos_backbone=c.pesos_backbone,
            topologia=topo, fps=c.fps, max_batch=max(c.max_batch, 1),
            cfg_tracker=cfg_tracker,
            motor_objetos=motor, motor_segmentacion=motor_seg,
            device=c.device, detector_agresion=detector_agresion,
            detector_caidas=det_caidas, verboso=c.verboso)

        cfg_al = ConfigAlerta(
            url=c.url(c.ruta_alertas), token=c.token, spool=c.spool,
            severidad_min=c.severidad_min, sitio=c.sitio, nodo=c.nodo)
        self.emisor = EmisorAlertas(
            cfg_al, transporte=TransporteBackend(cfg_al, self.config_ids,
                                                 self.grabadora))
        # (antes acá se hacía `self.pipe.emisor = self.emisor`, que no servía
        # de nada: el pipeline no lee ese atributo. Quien emite es el bucle.)

        # Qué está mirando y qué no. Un operador tiene que poder leer esto de
        # un vistazo: la diferencia entre "no pasó nada" y "nadie lo estaba
        # mirando" empieza acá.
        for nombre, prendida, porque in (
                (f"objetos     {' [' + self.detector + ']' if self.detector else ''}",
                 self.pipe.objetos is not None, "--pesos"),
                ("segmentación", self.pipe.segmentacion is not None, "--segmentacion"),
                ("agresión    ", self.pipe.agresion is not None, "--agresion"),
                ("caídas      ", self.pipe.caidas is not None, "--caidas")):
            _log(c, f"  cabeza {nombre} {'SI' if prendida else 'no'}"
                    + ("" if prendida else f"  (se prende con {porque})"))

        self.listo_en_s = time.time() - self.arranque
        self.fase = "listo"
        _log(c, f"modelos listos en {self.listo_en_s:.1f} s — "
                f"de acá en más, una cámara nueva empieza a analizarse en el "
                f"siguiente frame")

        for destino, nombre in ((self._bucle_descubrir, "descubrir"),
                                (self._bucle_inferir, "inferir")):
            h = threading.Thread(target=destino, name=nombre, daemon=True)
            h.start()
            self._hilos.append(h)

        # (el servidor HTTP ya se levantó al principio de iniciar(), antes de
        # cargar los modelos: ver el comentario de allá arriba.)

    # ------------------------------------------------------------------ #
    def _cargar_yolo(self):
        """El detector preentrenado. None si no está y se pidió "auto".

        Con "auto" que falte no es un error: se sigue con la cabeza propia,
        como antes, pero se dice fuerte, porque con ella el sistema ve gente
        donde no hay. Pedido con una ruta y sin poder cargar: no arranca.
        """
        c = self.cfg
        try:
            from motor_yolo import ConfigYolo, MotorYolo
            dev = c.device or "cpu"
            if dev != "cpu":
                try:
                    import torch
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:                        # noqa: BLE001
                    dev = "cpu"
            return MotorYolo(ConfigYolo(
                pesos=None if c.yolo == "auto" else c.yolo, tam=c.yolo_tam,
                bolsos=c.yolo_bolsos, device=dev, verboso=c.verboso))
        except Exception as e:                           # noqa: BLE001
            if c.yolo != "auto":
                raise RuntimeError(f"se pidió YOLO ({c.yolo}) y no carga: {e}") from e
            _log(c, "=" * 70)
            _log(c, f"  AVISO: sin detector preentrenado ({type(e).__name__}: {e}).")
            _log(c, "  Sigo con la cabeza propia, que ve personas donde no hay.")
            _log(c, "  Para arreglarlo:  pip install ultralytics onnxruntime")
            _log(c, "=" * 70)
            return None

    def _cargar_yolo_horus(self):
        """El YOLO de 4 clases de `entrenar_yolo.py`, si está en modelos/."""
        c = self.cfg
        try:
            from motor_yolo import ConfigYolo, MotorYolo, pesos_horus_por_defecto
            ruta = pesos_horus_por_defecto()
            if ruta is None:
                return None
            return MotorYolo(ConfigYolo(pesos=ruta, tam=c.yolo_tam,
                                        device="cpu" if not c.device else c.device,
                                        verboso=c.verboso))
        except Exception as e:                           # noqa: BLE001
            _log(c, f"aviso: el YOLO de Horus no carga ({type(e).__name__}: {e}); "
                    "sigo con la cabeza propia para pistola/humo/llama/paquete")
            return None

    # ------------------------------------------------------------------ #
    def _bucle_descubrir(self) -> None:
        """Cada `sondeo_s` pregunta qué cámaras hay y ajusta las fuentes."""
        while not self._parar.is_set():
            try:
                self._sincronizar(self._pedir_camaras())
            except Exception as exc:
                _log(self.cfg, f"aviso al sincronizar cámaras: {exc}")
            self._parar.wait(self.cfg.sondeo_s)

    def _pedir_camaras(self) -> List[Dict[str, Any]]:
        c = self.cfg
        try:
            req = urllib.request.Request(c.url(c.ruta_camaras))
            if c.token:
                req.add_header("Authorization", f"Bearer {c.token}")
            with _SIN_PROXY.open(req, timeout=4.0) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            # El backend puede no estar todavía. No es motivo para no analizar:
            # si hay lista local, se usa esa.
            if c.camaras_json and os.path.exists(c.camaras_json):
                with open(c.camaras_json, encoding="utf-8") as fh:
                    return json.load(fh)
            if not self.fuentes:
                _log(c, f"backend no responde ({exc}); espero")
            return []

    @staticmethod
    def fuente_de(fila: Dict[str, Any]) -> Optional[str]:
        """La fuente de video de una fila de `CamaraConfig`.

        El backend guarda `rtsp_url` (cámara IP) **o** `usb_index` (webcam).
        Ojo con el índice: la webcam por defecto es `usb_index = 0`, así que
        hay que preguntar por `is not None` y no por verdad — un `or` se lo
        come y esa cámara nunca se engancha.

        Se aceptan además `rstp_url` y `url` porque aparecen en versiones
        viejas de la base y en los archivos de respaldo.
        """
        for clave in ("rtsp_url", "rstp_url", "url"):
            valor = fila.get(clave)
            if valor not in (None, ""):
                return str(valor)
        indice = fila.get("usb_index")
        if indice is not None:
            return str(indice)
        return None

    def _sincronizar(self, filas: List[Dict[str, Any]]) -> None:
        quiero: Dict[str, Tuple[str, Optional[int]]] = {}
        for f in filas:
            if f.get("activa") is False:
                continue
            cid = f.get("id")
            url = self.fuente_de(f)
            if url is None:
                _log(self.cfg, f"  cámara {cid} sin rtsp_url ni usb_index: la salteo")
                continue
            quiero[f"cam-{cid}" if cid is not None else url] = (url, cid)

        for cam in list(self.fuentes):
            if cam not in quiero:
                _log(self.cfg, f"  [{cam}] baja")
                self.fuentes.pop(cam).parar()
                self.config_ids.pop(cam, None)
                if self.grabadora is not None:
                    self.grabadora.olvidar(cam)

        for cam, (url, cid) in quiero.items():
            actual = self.fuentes.get(cam)
            if actual is not None and actual.url == url:
                continue
            if actual is not None:
                actual.parar()
            t0 = time.perf_counter()
            fuente = FuenteCamara(cam, url, self.cfg, cid)
            fuente.iniciar()
            self.fuentes[cam] = fuente
            self.config_ids[cam] = cid
            _log(self.cfg, f"  [{cam}] alta ({url}) — enganchada en "
                           f"{(time.perf_counter() - t0) * 1000:.0f} ms, "
                           f"los modelos ya estaban cargados")

    # ------------------------------------------------------------------ #
    def _bucle_inferir(self) -> None:
        """El camino caliente: junta el último frame de cada cámara y decide."""
        periodo = 1.0 / max(self.cfg.fps, 0.1)
        while not self._parar.is_set():
            t0 = time.perf_counter()
            lote: Dict[str, np.ndarray] = {}
            ts_lote: Dict[str, float] = {}
            for cam, fuente in list(self.fuentes.items()):
                frame, ts = fuente.frame
                if frame is not None and time.time() - ts < 5.0:
                    lote[cam] = frame
                    ts_lote[cam] = ts
                if len(lote) >= self.cfg.max_batch:
                    break

            if lote:
                try:
                    eventos = self.pipe.procesar(lote)
                    self.eventos += len(eventos)
                    # Dibujar ANTES de emitir: la captura que viaja con la
                    # alerta tiene que ser el cuadro que la disparó, con las
                    # cajas encima, no el del tick anterior.
                    self._grabar(lote, ts_lote)
                    for ev in eventos:
                        _log(self.cfg, "  " + ev.linea())
                        # 18/09, MEDIDO: acá el bucle imprimía el evento y
                        # nada más. `self.pipe.emisor = self.emisor` de más
                        # arriba no hace nada: pipeline.py no lee ese atributo
                        # en ningún lado. O sea que el servicio detectaba un
                        # incendio, escribía "[ALERTA] incendio" en su consola
                        # con toda la ceremonia, y NO avisaba a nadie: ni un
                        # POST al backend, ni una fila en la base, ni un frame
                        # por el websocket, ni un mail.
                        #
                        # Verificado corriendo el servicio contra el backend
                        # real: 6 eventos en la consola, 0 filas en /alertas.
                        # Es la peor falla posible en este sistema, y encima
                        # la que mejor disimula: la consola se llena de
                        # alertas y da la sensación de que está funcionando.
                        if self.emisor is not None:
                            self.emisor.emitir(ev)
                except Exception as exc:
                    _log(self.cfg, f"error en el pipeline: {exc}")
                self.ticks += 1

            # Cada 30 s, una linea con lo que esta viendo cada regla.
            #
            # Sin esto, mirar la consola y no ver alertas no distingue "no
            # paso nada" de "la regla nunca pudo correr". Es exactamente lo
            # que reglas_dormidas() existe para responder, y hasta hoy solo
            # se imprimia al CERRAR el servicio — o sea, cuando ya no servia.
            if time.time() - self._ultimo_resumen >= 30.0:
                self._ultimo_resumen = time.time()
                self._log_diagnostico()

            resto = periodo - (time.perf_counter() - t0)
            if resto > 0:
                self._parar.wait(resto)

    def _log_diagnostico(self) -> None:
        reglas = self._diagnostico_reglas()
        if not reglas:
            return
        dormidas = [r for r in reglas if not r["puede_correr"]]
        vieron = [r for r in reglas if r["hallazgos"]]
        cams = sum(1 for f in self.fuentes.values() if f.estado == "ok")

        _log(self.cfg, f"— {self.ticks} ticks · {cams}/{len(self.fuentes)} "
                       f"cámara(s) dando video · {self.eventos} evento(s)")
        if vieron:
            _log(self.cfg, "   vio: " + ", ".join(
                f"{r['tipo']}×{r['hallazgos']}" for r in sorted(
                    vieron, key=lambda x: -x["hallazgos"])))
        else:
            _log(self.cfg, "   no vio nada todavía")
        if dormidas:
            _log(self.cfg, "   NO PUEDEN CORRER: " + " · ".join(
                f"{r['tipo']} (falta {', '.join(r['falta'])})" for r in dormidas))

    # Qué color lleva cada clase de la segmentación en el video anotado.
    # BGR, que es como los quiere cv2.
    _COLOR_SEG = {
        "fuego": (0, 80, 255),      # naranja fuerte
        "llama": (0, 80, 255),
        "humo":  (190, 190, 190),   # gris
        "agua":  (255, 170, 0),     # celeste
    }

    def _grabar(self, lote: Dict[str, np.ndarray],
                ts_lote: Optional[Dict[str, float]] = None) -> None:
        """El video anotado para el panel y, de paso, para la evidencia."""
        try:
            if self.grabadora is not None:
                for cam, frame in lote.items():
                    self.grabadora.guardar_crudo(cam, frame)
            if self.cfg.dibujar and (self.cfg.puerto_stream or self.grabadora is not None):
                self._anotar(lote, ts_lote or {})
            elif self.grabadora is not None:
                for cam, frame in lote.items():
                    self.grabadora.agregar_cuadro(cam, frame)
        except Exception as exc:                         # noqa: BLE001
            # Dibujar o grabar nunca puede frenar el análisis.
            _log(self.cfg, f"aviso al dibujar: {exc}")

    # 26/09 — "se superponen muchos cuadrados cuando aparezco, como si
    # estuviera subdividido". Parte era del tracker (ver `_fusionar_tracks` y
    # `_medir_fps` en tracker_local.py) y parte era de acá: la capa 2 dibujaba
    # TODA detección cruda con score >= 0,10 que no tuviera un track con la
    # esquina de arriba a menos de 20 px. El torso, las piernas o una caja
    # corrida de la misma persona no comparten esquina con su track, así que
    # se dibujaban todas, en gris, encima de ella. Incluidas las que la dedup
    # del tracker ya había descartado por duplicadas.
    _PISO_CRUDAS = 0.20        # por debajo, ruido: no ayuda a calibrar nada

    @staticmethod
    def _solape(a, b) -> Tuple[float, float, float]:
        """(IoU, fracción de a dentro de b, fracción de b dentro de a)."""
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        aa = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
        ab = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
        return inter / (aa + ab - inter), inter / aa, inter / ab

    def _crudas_para_dibujar(self, tracker, tracks, dets) -> List[Any]:
        """Las detecciones sueltas que vale la pena mostrar: las que el modelo
        vio y NO terminaron siendo (parte de) un track."""
        por_clase: Dict[str, List[Any]] = {}
        for d in dets or ():
            try:
                if float(d.score) < self._PISO_CRUDAS:
                    continue
            except Exception:                            # noqa: BLE001
                continue
            lbl = getattr(d, "label", None) or getattr(d, "clase", "?")
            por_clase.setdefault(lbl, []).append(d)
        salida: List[Any] = []
        for lbl, ds in por_clase.items():
            # La misma dedup que usa el tracker: lo que él tiró por duplicado
            # tampoco se dibuja.
            try:
                antes = tracker.suprimidas_duplicadas    # dibujar no ensucia sus contadores
                ds = tracker._deduplicar(ds, tracker.cfg.params(lbl))
                tracker.suprimidas_duplicadas = antes
            except Exception:                            # noqa: BLE001
                pass
            cajas = [t.bbox_xyxy for t in tracks if t.clase == lbl]
            for d in ds:
                encima = False
                for c in cajas:
                    iou, dentro, envuelve = self._solape(d.bbox_xyxy, c)
                    if iou >= 0.20 or dentro >= 0.50 or envuelve >= 0.50:
                        encima = True
                        break
                if not encima:
                    salida.append(d)
        return salida

    def _tracks_para_dibujar(self, tracks) -> List[Any]:
        """Un track perdido (naranja) que quedó encima de uno visible de su
        misma clase no se dibuja: es el rastro de la misma cosa, que el
        tracker todavía no soltó, y en pantalla parece una segunda persona.
        Sigue existiendo para el tracker y las reglas; solo no se pinta."""
        visibles = [t for t in tracks if t.frames_sin_ver == 0]
        salida = []
        for t in tracks:
            if t.frames_sin_ver > 0 and any(
                    v.clase == t.clase and max(self._solape(t.bbox_xyxy, v.bbox_xyxy)) >= 0.50
                    for v in visibles):
                continue
            salida.append(t)
        return salida

    # Color por clase de los tracks confirmados y visibles (BGR). Las armas en
    # rojo y el fuego en naranja: es lo primero que tiene que encontrar el ojo.
    _COLOR_CLASE = {
        "persona":  (90, 220, 90),
        "pistola":  (60, 60, 255),
        "cuchillo": (60, 60, 255),
        "llama":    (0, 140, 255),
        "humo":     (200, 200, 200),
        "celular":  (255, 200, 0),
        "paquete":  (255, 170, 60),
    }

    def _primitivas(self, cam: str, alto: int, ancho: int,
                    ts_frame: float) -> Dict[str, Any]:
        """Lo que hay que dibujar sobre esta cámara, en coordenadas del cuadro
        analizado. Se arma UNA vez por tick de los modelos y después se dibuja
        sobre todos los cuadros que la cámara entregue hasta el próximo.

        18/09 — por qué se dibujan tres capas y no solo los tracks: "no
        detecta el fuego" y "lo detecta pero no llega al umbral" se veían
        igual. Las tres siguen, pero 16/11: las detecciones sueltas y los
        tracks tentativos solo con --crudas. En el panel de un operador son
        cajas grises que aparecen y desaparecen encima de la gente, y se leen
        como "el sistema no sabe dónde está la persona".
        """
        pipe = self.pipe
        cajas: List[Dict[str, Any]] = []
        tracker = pipe.tracking.tracker(cam)
        fps = float(getattr(tracker, "fps_efectivo", self.cfg.fps) or self.cfg.fps)
        tracks = tracker.tracks

        for tr in self._tracks_para_dibujar(tracks):
            tentativo = tr.estado == "tentativo"
            if tentativo and not self.cfg.dibujar_crudas:
                continue
            visible = tr.frames_sin_ver == 0
            if tentativo:
                color, grosor = (0, 200, 255), 1
                texto = f"{tr.clase} {tr.score:.2f}?"
            elif visible:
                color, grosor = self._COLOR_CLASE.get(tr.clase, (90, 220, 90)), 2
                texto = f"#{tr.track_id} {tr.clase} {int(round(tr.score * 100))}%"
            else:
                # Perdido: el tracker lo sostiene por si reaparece. Fino y sin
                # etiqueta, para que no se lea como una segunda persona.
                color, grosor, texto = (60, 160, 255), 1, ""
            vx, vy = getattr(tr, "velocidad", (0.0, 0.0)) or (0.0, 0.0)
            cajas.append({
                "id": tr.track_id,
                "caja": tuple(float(v) for v in tr.bbox_xyxy),
                "color": color, "grosor": grosor, "texto": _ascii(texto),
                # px/s del centro. Solo para lo que se está viendo: lo perdido
                # ya viene extrapolado por el Kalman.
                "v": (float(vx) * fps, float(vy) * fps) if visible else (0.0, 0.0),
            })

        if self.cfg.dibujar_crudas:
            for d in self._crudas_para_dibujar(
                    tracker, tracks,
                    getattr(pipe, "ultimas_detecciones", {}).get(cam, [])):
                try:
                    caja = tuple(float(v) for v in d.bbox_xyxy)
                except Exception:                        # noqa: BLE001
                    continue
                lbl = getattr(d, "label", None) or getattr(d, "clase", "?")
                cajas.append({"id": None, "caja": caja, "color": (150, 150, 150),
                              "grosor": 1, "texto": _ascii(f"{lbl} {d.score:.2f}"),
                              "v": (0.0, 0.0)})

        # La máscara de segmentación, como mapa de clases chico + paleta.
        seg_dib = None
        lineas: List[Tuple[str, Tuple[int, int, int]]] = []
        clases_seg = list(getattr(getattr(pipe, "segmentacion", None),
                                  "clases", ()) or ())
        seg = getattr(pipe, "ultimo_seg", {}).get(cam)
        if seg is not None and clases_seg and getattr(seg, "mask", None) is not None:
            try:
                m = seg.mask
                m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                m = m.astype(np.uint8)
                lut = np.zeros((256, 3), np.uint8)
                partes = []
                total = float(m.size) or 1.0
                for idx, nombre in enumerate(clases_seg):
                    if nombre == "fondo" or idx > 255:
                        continue
                    n = int((m == idx).sum())
                    if n == 0:
                        continue
                    lut[idx] = self._COLOR_SEG.get(nombre, (0, 255, 255))
                    # El % se muestra SIEMPRE, aunque no llegue al umbral.
                    partes.append(f"{nombre} {100.0 * n / total:.2f}%")
                if partes:
                    seg_dib = (m, lut)
                    lineas.append((_ascii("seg: " + "  ".join(partes)), (0, 200, 255)))
            except Exception:                            # noqa: BLE001
                seg_dib = None

        for ev in pipe.fusion.abiertos:
            if getattr(ev, "camera_id", cam) != cam and cam not in (
                    getattr(ev, "camaras", None) or ()):
                continue
            lineas.append((_ascii(ev.linea())[:90], (80, 80, 255)))
            if len(lineas) >= 3:
                break

        return {"ts": ts_frame, "tam": (alto, ancho), "cajas": cajas,
                "seg": seg_dib, "lineas": lineas, "_cache": {}}

    @staticmethod
    def _dibujar(img: np.ndarray, prim: Dict[str, Any],
                 cajas: Optional[List[Dict[str, Any]]] = None,
                 con_lineas: bool = True) -> None:
        """Dibuja `prim` sobre `img` (en el lugar). `img` puede estar
        achicada: las coordenadas se escalan contra el tamaño analizado.

        `con_lineas`: los textos de arriba (eventos abiertos, % de
        segmentación). Van en la evidencia del mail, donde explican la
        captura; en el video en vivo no (16/11): el panel ya muestra la alerta,
        y la franja tapaba la esquina de la escena."""
        import cv2
        h, w = img.shape[:2]
        alto0, ancho0 = prim["tam"]
        ex, ey = w / float(max(ancho0, 1)), h / float(max(alto0, 1))

        # 1. segmentación (cacheada por tamaño: se repite 25 veces por tick)
        if prim.get("seg") is not None:
            cache = prim["_cache"]
            clave = (w, h)
            if clave not in cache:
                m, lut = prim["seg"]
                mr = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                pix = mr > 0
                cache.clear()
                cache[clave] = (pix, lut[mr][pix].astype(np.float32))
            pix, color = cache[clave]
            if pix.any():
                img[pix] = (img[pix].astype(np.float32) * 0.65 + color * 0.35).astype(np.uint8)

        # 2. cajas con etiqueta legible (fondo del color de la caja)
        # 16/11: la letra se escala con el ancho de SALIDA. Con un tope fijo
        # de 0,38 una celda de 480 px mostraba etiquetas de 5 px de alto.
        fs = float(min(0.7, max(0.42, w / 900.0)))
        for c in (prim["cajas"] if cajas is None else cajas):
            x1, y1, x2, y2 = c["caja"]
            p1 = (int(x1 * ex), int(y1 * ey))
            p2 = (int(x2 * ex), int(y2 * ey))
            cv2.rectangle(img, p1, p2, c["color"], c["grosor"], cv2.LINE_AA)
            if c["texto"]:
                (tw, th), base = cv2.getTextSize(c["texto"], cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                ty = p1[1] - 4 if p1[1] - th - 6 > 0 else p1[1] + th + 6
                # que la etiqueta no se salga por la derecha del cuadro
                tx = max(0, min(p1[0], w - tw - 7))
                cv2.rectangle(img, (tx, ty - th - 3), (tx + tw + 6, ty + base - 1),
                              c["color"], -1)
                cv2.putText(img, c["texto"], (tx + 3, ty - 1), cv2.FONT_HERSHEY_SIMPLEX,
                            fs, (15, 15, 15), 1, cv2.LINE_AA)

        # 3. textos de arriba sobre una franja oscura
        y = 0
        for texto, color in (prim.get("lineas", ()) if con_lineas else ()):
            (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            sub = img[y:y + th + 10, 0:min(w, tw + 16)]
            sub[:] = (sub.astype(np.float32) * 0.4).astype(np.uint8)
            cv2.putText(img, texto, (8, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, color, 1, cv2.LINE_AA)
            y += th + 10

    def _anotar(self, lote: Dict[str, np.ndarray],
                ts_lote: Dict[str, float]) -> None:
        """Una vez por tick: arma lo que hay que dibujar (para el video del
        panel, que lo pinta sobre cada cuadro nuevo) y el cuadro anotado de la
        evidencia (la captura y el clip del mail)."""
        try:
            import cv2
        except ImportError:
            return
        for cam, frame in lote.items():
            alto, ancho = frame.shape[:2]
            prim = self._primitivas(cam, alto, ancho, ts_lote.get(cam, time.time()))
            self._prim[cam] = prim                       # asignación atómica
            if self.grabadora is None and not self.cfg.puerto_stream:
                continue
            img = frame.copy()
            self._dibujar(img, prim)
            ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                datos = jpg.tobytes()
                with self._lock_anotados:
                    self._anotados[cam] = datos
                if self.grabadora is not None:
                    self.grabadora.agregar(cam, datos)

    # ------------------------------------------------------------------ #
    # Video del panel
    # ------------------------------------------------------------------ #
    _ANCHOS = (320, 480, 640, 960, 1280, 1920)

    def _ancho_valido(self, pedido: int) -> int:
        """El ancho pedido redondeado a un escalón (para que dos celdas casi
        iguales compartan la misma codificación)."""
        tope = int(self.cfg.ancho_video_max or 1280)
        if pedido <= 0:
            return tope
        for a in self._ANCHOS:
            if a >= pedido:
                return min(a, tope)
        return tope

    def _mirar(self, clave: Tuple[str, int], delta: int) -> None:
        with self._cond_video:
            n = self._espectadores.get(clave, 0) + delta
            if n > 0:
                self._espectadores[clave] = n
            else:
                self._espectadores.pop(clave, None)
                self._video.pop(clave, None)
        if delta > 0:
            self._asegurar_difusor(clave[0])

    def _asegurar_difusor(self, cam: str) -> None:
        with self._cond_video:
            h = self._difusores.get(cam)
            if h is not None and h.is_alive():
                return
            h = threading.Thread(target=self._difundir, args=(cam,),
                                 name=f"video-{cam}", daemon=True)
            self._difusores[cam] = h
        h.start()

    @staticmethod
    def _suavizar(prim: Optional[Dict[str, Any]], ts_cuadro: float,
                  estado: Dict[int, np.ndarray]) -> List[Dict[str, Any]]:
        """Las cajas del último tick, llevadas al instante de ESTE cuadro.

        Los modelos ven ~3-4 cuadros por segundo y el video muestra 25. Sin
        esto las cajas saltarían 3 veces por segundo y quedarían atrás de una
        persona que camina. Dos pasos:
          1. extrapolar con la velocidad del Kalman el tiempo que pasó desde
             el cuadro analizado (con freno y tope: nunca más de media caja);
          2. suavizar contra lo dibujado en el cuadro anterior, para que el
             salto de un tick al siguiente sea un deslizamiento corto.
        """
        if prim is None:
            estado.clear()
            return []
        dt = min(max(ts_cuadro - float(prim["ts"]), 0.0), 0.6)
        salida: List[Dict[str, Any]] = []
        vivos = set()
        for c in prim["cajas"]:
            x1, y1, x2, y2 = c["caja"]
            vx, vy = c["v"]
            bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            dx = float(np.clip(vx * dt * 0.7, -0.5 * bw, 0.5 * bw))
            dy = float(np.clip(vy * dt * 0.7, -0.5 * bh, 0.5 * bh))
            obj = np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy], np.float64)
            k = c["id"]
            if k is not None:
                vivos.add(k)
                prev = estado.get(k)
                if prev is not None:
                    # Solo se suaviza si es "la misma caja": un id que saltó
                    # lejos (reasignación) se dibuja donde está, sin barrido.
                    sx = max(0.0, min(prev[2], obj[2]) - max(prev[0], obj[0]))
                    sy = max(0.0, min(prev[3], obj[3]) - max(prev[1], obj[1]))
                    if sx * sy > 0.1 * bw * bh:
                        obj = prev + 0.5 * (obj - prev)
                estado[k] = obj
            salida.append({**c, "caja": tuple(obj)})
        for k in list(estado):
            if k not in vivos:
                del estado[k]
        return salida

    def _difundir(self, cam: str) -> None:
        """Un hilo por cámara MIRADA: cada cuadro nuevo de la cámara (hasta
        `fps_video`) sale al panel con las últimas cajas encima.

        16/11 — "el video se arma cuadro por cuadro, muy poco fluido". El
        stream mandaba el cuadro ANOTADO, que solo existe cuando terminan los
        modelos: en esta PC, ~3-4 por segundo. Ahora el video y el análisis
        van separados: la cámara entrega 25-30 cuadros, el video los muestra
        todos, y los modelos siguen a su ritmo. Lo que se dibuja encima se
        lleva al instante de cada cuadro (`_suavizar`).

        Barato a propósito: se codifica al ancho que pide la celda (una celda
        de 480 px no necesita un JPEG de 1280), una sola vez por ancho aunque
        miren varios, y si nadie mira el hilo se apaga solo.
        """
        import cv2
        periodo = 1.0 / max(self.cfg.fps_video, 1.0)
        calidad = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.calidad_video)]
        visto = -1
        sin_nadie: Optional[float] = None
        suave: Dict[int, np.ndarray] = {}
        try:
            while not self._parar.is_set():
                fuente = self.fuentes.get(cam)
                with self._cond_video:
                    anchos = sorted({a for (c, a), n in self._espectadores.items()
                                     if c == cam and n > 0})
                if fuente is None:
                    return
                if not anchos:
                    sin_nadie = sin_nadie or time.time()
                    if time.time() - sin_nadie > 10.0:
                        return
                    self._parar.wait(0.2)
                    continue
                sin_nadie = None

                frame, ts, seq = fuente.esperar(visto, timeout=1.0)
                if frame is None or seq == visto:
                    continue
                t0 = time.perf_counter()
                visto = seq
                prim = self._prim.get(cam) if self.cfg.dibujar else None
                cajas = self._suavizar(prim, ts, suave)
                alto, ancho = frame.shape[:2]
                for a in anchos:
                    if a < ancho:
                        img = cv2.resize(frame, (a, max(2, int(round(alto * a / ancho)))),
                                         interpolation=cv2.INTER_AREA)
                    else:
                        img = frame.copy()
                    if prim is not None:
                        self._dibujar(img, prim, cajas,
                                      con_lineas=self.cfg.dibujar_crudas)
                    ok, jpg = cv2.imencode(".jpg", img, calidad)
                    if not ok:
                        continue
                    with self._cond_video:
                        self._video[(cam, a)] = (seq, jpg.tobytes())
                        self._cond_video.notify_all()
                resto = periodo - (time.perf_counter() - t0)
                if resto > 0:
                    self._parar.wait(resto)
        except Exception as exc:                         # noqa: BLE001
            _log(self.cfg, f"  [{cam}] video del panel: {type(exc).__name__}: {exc}")
        finally:
            with self._cond_video:
                if self._difusores.get(cam) is threading.current_thread():
                    self._difusores.pop(cam, None)
                self._cond_video.notify_all()

    # ------------------------------------------------------------------ #
    def _levantar_http(self) -> None:
        """Servidor propio: el video con las cajas dibujadas y el estado.

        Vive acá y no en el backend porque el servicio es el único que tiene
        la cámara abierta. Muchas cámaras IP no aceptan dos conexiones al
        mismo stream, así que si el backend también la abriera, una de las dos
        se quedaría sin video.
        """
        import http.server

        servicio = self

        class Manejador(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                ruta, _, consulta = self.path.partition("?")
                if ruta.startswith("/camaras/") and ruta.endswith("/stream"):
                    cam = ruta[len("/camaras/"):-len("/stream")]
                    ancho = 0
                    for par in consulta.split("&"):
                        k, _, v = par.partition("=")
                        if k == "ancho" and v.isdigit():
                            ancho = int(v)
                    return self._stream(cam, ancho)
                if ruta in ("/estado", "/"):
                    return self._json(servicio.estado())
                self.send_response(404); self.end_headers()

            def _json(self, datos):
                cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.end_headers()
                self.wfile.write(cuerpo)

            def _stream(self, cam, ancho=0):
                """MJPEG del difusor. 16/11: antes mandaba el último cuadro
                ANOTADO cada 1/fps segundos: como el anotado solo cambia
                cuando terminan los modelos (~3-4 veces por segundo en CPU),
                el panel recibía 10 fps de cuadros repetidos. Ahora espera a
                que haya un cuadro nuevo de verdad y lo manda al toque."""
                import socket
                if cam not in servicio.fuentes:
                    self.send_response(404); self.end_headers()
                    return
                ancho = servicio._ancho_valido(ancho)
                clave = (cam, ancho)
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    self.connection.setsockopt(socket.IPPROTO_TCP,
                                               socket.TCP_NODELAY, 1)
                except OSError:
                    pass
                servicio._mirar(clave, +1)
                visto = -1
                try:
                    while not servicio._parar.is_set() and cam in servicio.fuentes:
                        with servicio._cond_video:
                            servicio._cond_video.wait_for(
                                lambda: servicio._video.get(clave, (visto,))[0] != visto
                                or servicio._parar.is_set(), timeout=2.0)
                            seq, jpg = servicio._video.get(clave, (visto, b""))
                        if not jpg:
                            continue
                        visto = seq
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                            + jpg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError,
                        ConnectionAbortedError, OSError):
                    pass
                finally:
                    servicio._mirar(clave, -1)

            def log_message(self, *a):
                pass

        self._srv_http = http.server.ThreadingHTTPServer(
            ("0.0.0.0", self.cfg.puerto_stream), Manejador)
        h = threading.Thread(target=self._srv_http.serve_forever, daemon=True)
        h.start()
        self._hilos.append(h)
        _log(self.cfg, f"video anotado en "
                       f"http://127.0.0.1:{self.cfg.puerto_stream}/camaras/<id>/stream")

    # ------------------------------------------------------------------ #
    def viendo(self) -> Dict[str, Any]:
        """Qué está viendo cada cabeza AHORA, alerte o no.

        18/09 — "no detecta el fuego, detecta personas y paquetes". Sin esto no
        hay forma de contestar esa frase. Una alerta que no salta puede ser
        cuatro cosas muy distintas:

          1. la cabeza no ve nada;
          2. la cabeza lo ve, pero por debajo del umbral;
          3. lo ve y pasa el umbral, pero la regla pide algo más (persistencia,
             dos personas, una zona) y eso no se cumple;
          4. la regla no puede ni correr porque le falta una cabeza — y eso ya
             lo dice `reglas_dormidas`.

        Desde afuera las cuatro se ven igual: nada. Acá se separan.

        Devuelve, por cámara, el mejor score de cada clase de objetos y el
        porcentaje de área de cada clase de segmentación, sin filtrar por
        umbral: la gracia es justamente ver lo que el umbral descarta.
        """
        if self.pipe is None:
            return {}

        out: Dict[str, Any] = {}
        detec = getattr(self.pipe, "ultimas_detecciones", {}) or {}
        segs = getattr(self.pipe, "ultimo_seg", {}) or {}
        clases_seg = list(getattr(getattr(self.pipe, "segmentacion", None),
                                  "clases", ()) or ())

        for cam in set(detec) | set(segs):
            mejor: Dict[str, float] = {}
            for d in detec.get(cam, []):
                clase = getattr(d, "clase", None) or getattr(d, "label", "?")
                score = float(getattr(d, "score", 0.0))
                if score > mejor.get(clase, 0.0):
                    mejor[clase] = round(score, 3)

            area: Dict[str, float] = {}
            s = segs.get(cam)
            fr = getattr(s, "fracciones", None) or getattr(s, "areas", None)
            if isinstance(fr, dict):
                area = {k: round(float(v) * 100, 3) for k, v in fr.items()}
            elif fr is not None:
                try:
                    area = {clases_seg[i] if i < len(clases_seg) else str(i):
                            round(float(v) * 100, 3) for i, v in enumerate(fr)}
                except Exception:                        # noqa: BLE001
                    area = {}

            out[cam] = {
                "objetos": dict(sorted(mejor.items(), key=lambda kv: -kv[1])),
                "segmentacion_pct_area": area,
            }
        return out

    def estado(self) -> Dict[str, Any]:
        st = dict(self.emisor.stats) if self.emisor else {}
        if self.pipe is None:
            # Todavía cargando. Se contesta igual —y esa es la gracia: el panel
            # tiene que poder distinguir "esperá que estoy subiendo los modelos
            # a la placa" de "no hay ningún servicio corriendo".
            return {
                "ok": True,
                "fase": getattr(self, "fase", "cargando"),
                "modelos": "cargando",
                "arriba_s": round(time.time() - self.arranque, 1) if self.arranque else 0,
                "listo_en_s": 0,
                "ticks": 0, "eventos": 0, "caidas": None,
                "camaras": [],
                "alertas": {"enviadas": 0, "fallidas": 0, "en_cola": 0, "en_disco": 0},
            }
        return {
            "ok": True,
            "fase": getattr(self, "fase", "listo"),
            "modelos": "simulados" if self.cfg.simular else "cargados",
            "detector": self.detector or ("simulado" if self.cfg.simular else "cabeza propia"),
            "listo_en_s": round(self.listo_en_s, 2),
            "arriba_s": round(time.time() - self.arranque, 1) if self.arranque else 0,
            "ticks": self.ticks,
            "eventos": self.eventos,
            # Que se vea desde afuera si la cabeza de caídas está mirando.
            # "0 caídas" con la cabeza apagada no significa lo mismo que con
            # la cabeza prendida, y esta es la única forma de distinguirlo sin
            # leer los logs de arranque.
            "caidas": (self.pipe.caidas.stats
                       if getattr(self.pipe, "caidas", None) is not None
                       else None),
            "camaras": [f.resumen() for f in self.fuentes.values()],
            # Cuántos segundos de video hay guardados por cámara para adjuntar
            # a la próxima alerta. Cero con la cámara andando = algo anda mal.
            "evidencia": (self.grabadora.resumen()
                          if self.grabadora is not None else None),
            # Por qué NO está alertando cada regla.
            #
            # 18/09: toda esta información existía desde siempre —
            # `reglas_dormidas()`, `veces_dormida`, `faltantes()`, `stats`—
            # y no salía de la consola del servicio. Si el sistema no avisaba
            # de un incendio, el operador no tenía forma de distinguir "no
            # hubo fuego" de "la regla de incendio nunca pudo correr porque le
            # falta una cabeza". Justo la distinción por la que existe
            # `reglas_dormidas()`.
            "reglas": self._diagnostico_reglas(),
            "cabezas": {
                "objetos": self.pipe.objetos is not None,
                "segmentacion": self.pipe.segmentacion is not None,
                "agresion": self.pipe.agresion is not None,
                "caidas": self.pipe.caidas is not None,
                "topologia": self.pipe.topo is not None,
            },
            "viendo": self.viendo(),
            "reglas_dormidas": (self.pipe.fusion.reglas_dormidas()
                                if getattr(self.pipe, "fusion", None) else {}),
            # 18/09: esto leía `emisor.enviados`, `emisor.fallidos` y
            # `emisor.pendientes_en_spool()`, y NINGUNO de los tres existe.
            # Los dos primeros iban con getattr(..., 0), así que el resumen
            # decía "alertas enviadas 0" para siempre, hubiera mandado una o
            # trescientas. El tercero no tenía default, así que cerrar el
            # servicio con Ctrl+C terminaba en un AttributeError en vez de en
            # el resumen. Los nombres de verdad están en EmisorAlertas.stats.
            "alertas": {
                "enviadas": st.get("entregados", 0),
                "fallidas": st.get("descartados", 0),
                "en_cola": self.emisor.pendientes() if self.emisor else 0,
                "en_disco": st.get("spool", 0),
            },
        }

    def _diagnostico_reglas(self) -> List[Dict[str, Any]]:
        """Una línea por regla: si puede correr, qué le falta, y cuánto vio.

        `hallazgos` cuenta las veces que la regla encontró algo (no eventos
        emitidos: un incendio sostenido son muchos hallazgos y un evento).
        `dormida` son las veces que ni siquiera pudo evaluarse.
        """
        motor = getattr(self.pipe, "fusion", None)
        if motor is None:
            return []

        # Las cabezas que hay, para poder decir qué le falta a cada regla sin
        # esperar a que pase un frame.
        vivas = set()
        if self.pipe.objetos is not None:
            vivas.add("objetos")
        if self.pipe.segmentacion is not None:
            vivas.add("segmentacion")
        if self.pipe.agresion is not None:
            vivas.add("accion")
        if self.pipe.caidas is not None:
            vivas.add("pose")

        out = []
        for r in motor.reglas:
            falta = sorted(set(getattr(r, "requiere", frozenset())) - vivas)
            alguna = set(getattr(r, "requiere_alguna", frozenset()))
            if alguna and not (alguna & vivas):
                falta.append("|".join(sorted(alguna)))
            if getattr(r, "necesita_topologia", False) and motor.topo is None:
                falta.append("topologia.json")
            out.append({
                "tipo": r.tipo,
                "puede_correr": not falta,
                "falta": falta,
                "dormida": int(getattr(r, "veces_dormida", 0)),
                "hallazgos": int(motor.stats.get(r.tipo, 0)),
            })
        return out

    def parar(self) -> None:
        self._parar.set()
        for f in self.fuentes.values():
            f.parar()
        if self._srv_http is not None:
            self._srv_http.shutdown()
        if self.emisor is not None:
            self.emisor.cerrar(timeout_s=10.0)

    def resumen(self) -> str:
        e = self.estado()
        cams = ", ".join(f"{c['camara']}:{c['estado']}" for c in e["camaras"]) or "ninguna"
        a = e["alertas"]
        return (f"[servicio] {e['ticks']} ticks · {e['eventos']} eventos · "
                f"cámaras: {cams} · alertas: {a['enviadas']} entregadas, "
                f"{a['en_cola']} en cola, {a['en_disco']} en disco, "
                f"{a['fallidas']} descartadas")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--token")
    ap.add_argument("--pesos")
    ap.add_argument("--pesos-backbone", dest="pesos_backbone")
    ap.add_argument("--topologia")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-batch", dest="max_batch", type=int, default=4)
    ap.add_argument("--sondeo", type=float, default=5.0)
    ap.add_argument("--puerto-stream", dest="puerto_stream", type=int, default=8010)
    ap.add_argument("--camaras-json", dest="camaras_json",
                    help="lista local, por si el backend no está")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--segmentacion", action="store_true",
                    # El %% es a proposito: argparse le hace %-formato al
                    # texto de ayuda, y un % suelto rompe --help entero.
                    help="prender la cabeza de fuego y humo (la que mejor "
                         "anda: F1 99,1 %%)")
    ap.add_argument("--segmentacion-checkpoint", dest="segmentacion_checkpoint",
                    help="por defecto 04_cabezas/segmentacion/checkpoints/"
                         "head_v4_produccion.pt")
    ap.add_argument("--agresion", action="store_true",
                    help="prender la cabeza de peleas (fight/)")
    ap.add_argument("--agresion-checkpoint", dest="agresion_checkpoint",
                    help="ruta al checkpoint de peleas")
    ap.add_argument("--caidas", action="store_true",
                    help="prender la cabeza de caídas (horus/fall). Cuesta una "
                         "pasada de MediaPipe por persona por frame")
    ap.add_argument("--caidas-checkpoint", dest="caidas_checkpoint",
                    help="por defecto fall/checkpoints/modelo_demo_todo.pt")
    ap.add_argument("--caidas-max-personas", dest="caidas_max_personas",
                    type=int, default=4,
                    help="tope de personas por cámara y frame (default 4)")
    ap.add_argument("--simular", action="store_true",
                    help="sin GPU ni cámaras: verifica el cableado entero")
    ap.add_argument("--segundos", type=float, default=0.0,
                    help="cortar solo después de N segundos (para probar)")
    ap.add_argument("--yolo", default="auto",
                    help='"auto" (default), "no", o la ruta a un .onnx/.pt de YOLO')
    ap.add_argument("--yolo-tam", dest="yolo_tam", type=int, default=416)
    ap.add_argument("--yolo-bolsos", dest="yolo_bolsos", action="store_true",
                    help="mochila/cartera/valija cuentan como paquete (apagado: "
                         "en una casa hay bolsos por todos lados)")
    ap.add_argument("--sin-evidencia", dest="sin_evidencia", action="store_true",
                    help="no adjuntar captura ni clip a las alertas")
    ap.add_argument("--clip", type=float, default=8.0,
                    help="segundos de video ANTERIORES a la alerta que se adjuntan")
    ap.add_argument("--fps-video", dest="fps_video", type=float, default=25.0,
                    help="tope de cuadros por segundo del video del panel "
                         "(independiente del ritmo de los modelos)")
    ap.add_argument("--crudas", action="store_true",
                    help="dibujar también las detecciones que no llegaron a "
                         "track y los tracks tentativos (para calibrar)")
    args = ap.parse_args()

    cfg = ConfigServicio(
        backend=args.backend, token=args.token, pesos=args.pesos,
        pesos_backbone=args.pesos_backbone, topologia=args.topologia,
        fps=args.fps, max_batch=args.max_batch, sondeo_s=args.sondeo,
        puerto_stream=args.puerto_stream, camaras_json=args.camaras_json,
        device="cpu" if args.cpu else None, simular=args.simular,
        segmentacion=args.segmentacion,
        segmentacion_checkpoint=args.segmentacion_checkpoint,
        agresion=args.agresion,
        agresion_checkpoint=args.agresion_checkpoint,
        caidas=args.caidas, caidas_checkpoint=args.caidas_checkpoint,
        caidas_max_personas=args.caidas_max_personas,
        evidencia=not args.sin_evidencia, clip_s=args.clip,
        yolo=args.yolo, yolo_tam=args.yolo_tam, yolo_bolsos=args.yolo_bolsos,
        fps_video=args.fps_video, dibujar_crudas=args.crudas)

    # 16/11: en la instalación (servicio_modelos.exe) los pesos NO están al
    # lado del código sino en %LOCALAPPDATA%\com.tauri-app.horusai\modelos,
    # con la misma estructura que el proyecto. motor_yolo buscaba solo en su
    # propia carpeta (adentro del .exe), no encontraba yolo11n.pt y el
    # servicio caía en silencio a la cabeza propia para TODO: personas con
    # cajas malas y cuchillos en las luces. Se le dice dónde mirar: la misma
    # carpeta de --pesos, que es donde la app deja los demás modelos.
    if cfg.pesos and not os.environ.get("HORUS_MODELOS_DIR"):
        os.environ["HORUS_MODELOS_DIR"] = os.path.dirname(os.path.abspath(cfg.pesos))
    hay_yolo = cfg.yolo != "no" and (cfg.yolo != "auto" or _hay_pesos_yolo())
    if not cfg.simular and not cfg.pesos and not cfg.segmentacion and not hay_yolo:
        print("hace falta al menos una cabeza: --pesos (objetos) o "
              "--segmentacion (fuego y humo).")
        print("Para probar el cableado sin modelos ni cámaras: --simular")
        return 2

    print("=" * 78)
    print("HORUS · servicio" + ("  [SIMULADO]" if cfg.simular else ""))
    print("=" * 78)

    srv = Servicio(cfg)
    srv.iniciar()
    limite = time.time() + args.segundos if args.segundos else None
    try:
        while limite is None or time.time() < limite:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print()
    finally:
        srv.parar()
        print()
        print(srv.resumen())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
