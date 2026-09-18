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
    python servicio.py --pesos ../04_cabezas/objetos/modelos/objetos_v2.pt \
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
_RAIZ = os.path.dirname(_AQUI)
for _p in (_AQUI,
           os.path.join(_RAIZ, "05_tracking"),
           os.path.join(_RAIZ, "05_tracking", "local"),
           os.path.join(_RAIZ, "05_tracking", "global_reid"),
           os.path.join(_RAIZ, "06_fusion_decision"),
           os.path.join(_RAIZ, "07_alerta"),
           os.path.join(_RAIZ, "04_cabezas", "objetos")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alerta import ConfigAlerta, EmisorAlertas, TransporteHTTP  # noqa: E402


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

    # --- video ------------------------------------------------------------
    fps: float = 10.0                      # ritmo de análisis, no de la cámara
    max_batch: int = 4
    reconexion_s: float = 3.0
    ancho_captura: int = 0                 # 0 = lo que dé la cámara
    alto_captura: int = 0

    # --- salida -----------------------------------------------------------
    spool: str = "alertas_pendientes.jsonl"
    severidad_min: int = 1
    sitio: str = ""
    nodo: str = "caja-01"

    # --- streaming propio -------------------------------------------------
    puerto_stream: int = 8010              # 0 = no levantar el servidor MJPEG
    dibujar: bool = True

    # --- respaldo y pruebas ----------------------------------------------
    camaras_json: Optional[str] = None     # lista local si el backend no está
    simular: bool = False
    verboso: bool = True

    def url(self, ruta: str) -> str:
        return self.backend.rstrip("/") + ruta


def _log(cfg: ConfigServicio, *a: Any) -> None:
    if cfg.verboso:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


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
        self._lock = threading.Lock()
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

    def _correr(self) -> None:
        cap = None
        while not self._parar.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                    self.reconexiones += 1
                cap = self._abrir()
                if not cap.isOpened():
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

            with self._lock:
                self._frame = frame
                self._ts = time.time()
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
                 mapa: Dict[str, Optional[int]]) -> None:
        self._http = TransporteHTTP(cfg_alerta)
        self.mapa = mapa

    def enviar(self, payload: Dict[str, Any]) -> None:
        cam = payload.get("sitio", {}).get("camara", "")
        cid = self.mapa.get(cam)
        if cid is not None:
            payload.setdefault("sitio", {})["camara_config_id"] = cid
        self._http.enviar(payload)


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
        self._srv_http: Any = None

        self.ticks = 0
        self.eventos = 0
        self.arranque = 0.0
        self.listo_en_s = 0.0

    # ------------------------------------------------------------------ #
    def iniciar(self) -> None:
        c = self.cfg
        self.arranque = time.time()
        _log(c, "cargando modelos…")

        from pipeline import PipelineHorus

        motor = MotorSimulado(c.fps) if c.simular else None
        topo = c.topologia if (c.topologia and os.path.exists(c.topologia)) else None
        if c.topologia and topo is None:
            _log(c, f"aviso: no encuentro {c.topologia}; sigo sin zonas")

        det_caidas = None
        if c.caidas:
            # Se construye acá, y si falla se corta. Arrancar "igual pero sin
            # caídas" sería lo peor posible: el operador cree que el sistema
            # las está mirando. Misma razón por la que una regla sin cabeza se
            # declara dormida en vez de devolver "no pasó nada".
            try:
                from detector_caidas import ConfigCaidas, DetectorCaidas
                cfg_c = ConfigCaidas(max_personas=c.caidas_max_personas)
                if c.caidas_checkpoint:
                    cfg_c.checkpoint = c.caidas_checkpoint
                det_caidas = DetectorCaidas(cfg_c, verboso=c.verboso)
            except Exception as e:                       # noqa: BLE001
                raise RuntimeError(
                    f"se pidió la cabeza de caídas y no se pudo cargar: "
                    f"{type(e).__name__}: {e}\nHace falta `pip install "
                    f"mediapipe torch` y que estén horus/fall/"
                    f"pose_landmarker.task y el checkpoint. Sin eso el "
                    f"servicio no arranca a propósito: correr sin la cabeza "
                    f"pero creyendo que está es la falla que este sistema no "
                    f"se puede permitir.") from e

        self.pipe = PipelineHorus(
            pesos=None if c.simular else c.pesos,
            pesos_backbone=c.pesos_backbone,
            topologia=topo, fps=c.fps, max_batch=max(c.max_batch, 1),
            motor_objetos=motor, device=c.device,
            detector_caidas=det_caidas, verboso=c.verboso)

        cfg_al = ConfigAlerta(
            url=c.url(c.ruta_alertas), token=c.token, spool=c.spool,
            severidad_min=c.severidad_min, sitio=c.sitio, nodo=c.nodo)
        self.emisor = EmisorAlertas(
            cfg_al, transporte=TransporteBackend(cfg_al, self.config_ids))
        self.pipe.emisor = self.emisor

        self.listo_en_s = time.time() - self.arranque
        _log(c, f"modelos listos en {self.listo_en_s:.1f} s — "
                f"de acá en más, una cámara nueva empieza a analizarse en el "
                f"siguiente frame")

        for destino, nombre in ((self._bucle_descubrir, "descubrir"),
                                (self._bucle_inferir, "inferir")):
            h = threading.Thread(target=destino, name=nombre, daemon=True)
            h.start()
            self._hilos.append(h)

        if c.puerto_stream:
            self._levantar_http()

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
            with urllib.request.urlopen(req, timeout=4.0) as r:
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
            for cam, fuente in list(self.fuentes.items()):
                frame, ts = fuente.frame
                if frame is not None and time.time() - ts < 5.0:
                    lote[cam] = frame
                if len(lote) >= self.cfg.max_batch:
                    break

            if lote:
                try:
                    eventos = self.pipe.procesar(lote)
                    self.eventos += len(eventos)
                    for ev in eventos:
                        _log(self.cfg, "  " + ev.linea())
                    if self.cfg.dibujar and self.cfg.puerto_stream:
                        self._anotar(lote)
                except Exception as exc:
                    _log(self.cfg, f"error en el pipeline: {exc}")
                self.ticks += 1

            resto = periodo - (time.perf_counter() - t0)
            if resto > 0:
                self._parar.wait(resto)

    def _anotar(self, lote: Dict[str, np.ndarray]) -> None:
        """Dibuja los tracks y guarda el JPEG para el stream."""
        try:
            import cv2
        except ImportError:
            return
        for cam, frame in lote.items():
            img = frame.copy()
            for t in self.pipe.tracking.tracker(cam).tracks:
                if t.estado == "tentativo":
                    continue
                x1, y1, x2, y2 = (int(v) for v in t.bbox_xyxy)
                color = (60, 220, 60) if t.frames_sin_ver == 0 else (60, 160, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"#{t.track_id} {t.clase} {t.score:.2f}",
                            (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1, cv2.LINE_AA)
            y = 20
            for ev in self.pipe.fusion.abiertos[:4]:
                cv2.putText(img, ev.linea()[:90], (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1,
                            cv2.LINE_AA)
                y += 20
            ok, jpg = cv2.imencode(".jpg", img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                with self._lock_anotados:
                    self._anotados[cam] = jpg.tobytes()

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
                if self.path.startswith("/camaras/") and self.path.endswith("/stream"):
                    cam = self.path[len("/camaras/"):-len("/stream")]
                    return self._stream(cam)
                if self.path in ("/estado", "/"):
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

            def _stream(self, cam):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not servicio._parar.is_set():
                        with servicio._lock_anotados:
                            jpg = servicio._anotados.get(cam)
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                        time.sleep(1.0 / max(servicio.cfg.fps, 1.0))
                except (BrokenPipeError, ConnectionResetError):
                    pass

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
    def estado(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "modelos": "simulados" if self.cfg.simular else "cargados",
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
            "alertas": {
                "enviadas": getattr(self.emisor, "enviados", 0),
                "fallidas": getattr(self.emisor, "fallidos", 0),
                "en_disco": self.emisor.pendientes_en_spool() if self.emisor else 0,
            },
        }

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
        return (f"[servicio] {e['ticks']} ticks · {e['eventos']} eventos · "
                f"cámaras: {cams} · alertas enviadas {e['alertas']['enviadas']}, "
                f"en disco {e['alertas']['en_disco']}")


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
    args = ap.parse_args()

    cfg = ConfigServicio(
        backend=args.backend, token=args.token, pesos=args.pesos,
        pesos_backbone=args.pesos_backbone, topologia=args.topologia,
        fps=args.fps, max_batch=args.max_batch, sondeo_s=args.sondeo,
        puerto_stream=args.puerto_stream, camaras_json=args.camaras_json,
        device="cpu" if args.cpu else None, simular=args.simular,
        caidas=args.caidas, caidas_checkpoint=args.caidas_checkpoint,
        caidas_max_personas=args.caidas_max_personas)

    if not cfg.simular and not cfg.pesos:
        print("falta --pesos (o usá --simular para probar el cableado)")
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
