# -*- coding: utf-8 -*-
"""
evidencia.py · la captura y el clip que viajan con cada alerta.

26/09. "Falta que se envíe un video/screenshot." Una alerta que dice
"incendio en Cocina" y nada más obliga a quien la recibe a creerle al sistema
o a ir a mirar. Con la imagen del momento, y los segundos de antes, decide en
dos segundos si es de verdad.

Cómo funciona
-------------
`Grabadora` guarda, por cámara, los últimos `segundos` de cuadros como JPEG
(los mismos que se ven en el panel, con las cajas dibujadas). Cuando sale una
alerta, `adjuntos(cam)` devuelve:

  - la CAPTURA: el último cuadro;
  - el CLIP: los segundos ANTERIORES a la alerta, que son los que explican por
    qué saltó. No se espera a grabar los de después: la alerta sale ya.

Lo que se reutiliza de los compañeros
-------------------------------------
  - Mateo guardaba los cuadros de cada cámara en su propia carpeta
    (`imageExtractor.py`) y en `webcam_integrado.py` armaba el clip de la pelea
    con un buffer POR TIEMPO (5 s, "RWF-2000 son clips de 5 s"), no por
    cantidad de cuadros. Acá es lo mismo: a 2 cuadros por segundo o a 10, el
    clip cubre los mismos segundos.
  - El backend (rama main) ya tenía `snapshot_url` y `clip_url` en la fila
    `Camara` "para otro sprint". Es adonde va a parar esto.

El formato del clip
-------------------
El panel es un WebView (Chromium) y lo tiene que poder reproducir: H.264 en
.mp4 o VP8 en .webm. `mp4v`, lo que OpenCV escribe "siempre", NO se reproduce
en un navegador. Así que al primer uso se prueba qué códec anda de verdad en
esta máquina (escribir y releer), y si ninguno anda se arma un GIF, que se ve
en todos lados —incluido el mail en el celular— aunque pese más.

Solo numpy, OpenCV y (para el GIF) Pillow, que ya vienen con el resto.
"""

from __future__ import annotations

import base64
import os
import tempfile
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# El códec elegido se prueba una sola vez por proceso.
_CODEC: Optional[Tuple[str, str, str]] = None
_CODEC_PROBADO = False
# Los que anduvieron en la prueba pero fallaron con un clip de verdad.
_FALLIDOS: set = set()
_CODEC_LOCK = threading.Lock()

CANDIDATOS = (
    # (fourcc, extensión, tipo MIME)
    ("avc1", ".mp4", "video/mp4"),
    ("H264", ".mp4", "video/mp4"),
    ("VP80", ".webm", "video/webm"),
    ("VP90", ".webm", "video/webm"),
)


def _probar_codec(fourcc: str, ext: str) -> bool:
    import cv2
    fd, ruta = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        # 26/09: la prueba era con 64x48 y en la PC de Teo H.264 (OpenH264)
        # pasaba la prueba y después fallaba con los cuadros de verdad
        # ("Unable to create encoder" en cada alerta). Se prueba con el tamaño
        # que se va a usar.
        w = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*fourcc), 5.0, (640, 360))
        if not w.isOpened():
            return False
        for i in range(6):
            img = np.full((360, 640, 3), 40 * i, dtype=np.uint8)
            w.write(img)
        w.release()
        if os.path.getsize(ruta) < 200:
            return False
        cap = cv2.VideoCapture(ruta)
        ok, _ = cap.read()
        cap.release()
        return bool(ok)
    except Exception:                                    # noqa: BLE001
        return False
    finally:
        try:
            os.remove(ruta)
        except OSError:
            pass


def codec_video() -> Optional[Tuple[str, str, str]]:
    """(fourcc, extensión, mime) del primer códec reproducible que anda acá,
    o None si no hay ninguno (entonces el clip sale como GIF)."""
    global _CODEC, _CODEC_PROBADO
    with _CODEC_LOCK:
        if _CODEC_PROBADO:
            return _CODEC
        _CODEC_PROBADO = True
        forzado = os.environ.get("HORUS_CLIP_FORMATO", "").lower()
        if forzado == "gif":
            _CODEC = None
            return None
        try:
            import cv2  # noqa: F401
        except ImportError:
            _CODEC = None
            return None
        # Probar un códec que no está deja un par de líneas de error de
        # OpenCV/FFmpeg en la consola. Es esperable (se prueban en orden hasta
        # que uno anda), así que se silencia mientras dura la prueba.
        import cv2
        nivel = None
        try:
            nivel = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except Exception:                                # noqa: BLE001
            pass
        try:
            for fourcc, ext, mime in CANDIDATOS:
                if fourcc in _FALLIDOS:
                    continue
                if _probar_codec(fourcc, ext):
                    _CODEC = (fourcc, ext, mime)
                    break
        finally:
            if nivel is not None:
                try:
                    cv2.utils.logging.setLogLevel(nivel)
                except Exception:                        # noqa: BLE001
                    pass
        return _CODEC


def _decodificar(jpgs: List[bytes], lado_max: int) -> List[np.ndarray]:
    import cv2
    cuadros = []
    tam = None
    for j in jpgs:
        img = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if tam is None:
            h, w = img.shape[:2]
            esc = min(1.0, lado_max / float(max(h, w)))
            # Par: H.264 no acepta dimensiones impares.
            tam = (max(2, int(w * esc) // 2 * 2), max(2, int(h * esc) // 2 * 2))
        if (img.shape[1], img.shape[0]) != tam:
            img = cv2.resize(img, tam, interpolation=cv2.INTER_AREA)
        cuadros.append(img)
    return cuadros


def armar_clip(jpgs: List[bytes], fps: float,
               lado_max: int = 640) -> Optional[Tuple[bytes, str]]:
    """Los JPEG en un video reproducible (o GIF). (bytes, mime) o None."""
    if len(jpgs) < 2:
        return None
    fps = float(min(max(fps, 1.0), 15.0))
    try:
        cuadros = _decodificar(jpgs, lado_max)
    except Exception:                                    # noqa: BLE001
        return None
    if len(cuadros) < 2:
        return None

    codec = codec_video()
    if codec is not None:
        import cv2
        fourcc, ext, mime = codec
        fd, ruta = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        try:
            h, w = cuadros[0].shape[:2]
            wr = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if wr.isOpened():
                for c in cuadros:
                    wr.write(c)
                wr.release()
                with open(ruta, "rb") as fh:
                    datos = fh.read()
                if len(datos) > 200:
                    return datos, mime
        except Exception:                                # noqa: BLE001
            pass
        finally:
            try:
                os.remove(ruta)
            except OSError:
                pass
        # Anduvo en la prueba y no con este clip: no se vuelve a intentar
        # (cada intento deja tres líneas de error en la consola). La próxima
        # alerta prueba el códec siguiente; esta sale como GIF.
        _descartar(fourcc)

    return _gif(cuadros, fps)


def _descartar(fourcc: str) -> None:
    global _CODEC, _CODEC_PROBADO
    with _CODEC_LOCK:
        _FALLIDOS.add(fourcc)
        _CODEC, _CODEC_PROBADO = None, False


def _gif(cuadros: List[np.ndarray], fps: float) -> Optional[Tuple[bytes, str]]:
    """El plan B. Se ve en todos lados (panel, Gmail en el celular), pero con
    256 colores y más peso: se achica a 480 px de lado."""
    try:
        from PIL import Image
    except ImportError:
        return None
    import io
    try:
        imgs = []
        for c in cuadros:
            im = Image.fromarray(c[:, :, ::-1])          # BGR -> RGB
            if max(im.size) > 480:
                im.thumbnail((480, 480))
            imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
        buf = io.BytesIO()
        imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0, optimize=True)
        return buf.getvalue(), "image/gif"
    except Exception:                                    # noqa: BLE001
        return None


class Grabadora:
    """Los últimos `segundos` de cada cámara, en JPEG, listos para adjuntar.

    Thread-safe: la escribe el bucle de inferencia y la lee el hilo del
    emisor de alertas.
    """

    def __init__(self, segundos: float = 8.0, max_cuadros: int = 160,
                 calidad: int = 75) -> None:
        self.segundos = float(segundos)
        self.max_cuadros = int(max_cuadros)
        self.calidad = int(calidad)
        self._buf: Dict[str, Deque[Tuple[float, bytes]]] = {}
        # El último cuadro SIN dibujar de cada cámara. Se guarda el array (se
        # codifica solo cuando sale una alerta, que es raro) y viaja como
        # `cruda`: es el que sirve para reentrenar si alguien marca la alerta
        # como falsa. Uno con cajas y textos pintados no sirve de dato.
        self._crudo: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    # -------------------------------------------------------------- #
    def agregar(self, cam: str, jpg: bytes, ts: Optional[float] = None) -> None:
        if not jpg:
            return
        ts = time.time() if ts is None else float(ts)
        with self._lock:
            d = self._buf.setdefault(cam, deque(maxlen=self.max_cuadros))
            d.append((ts, bytes(jpg)))
            # Por TIEMPO, no por cantidad (ver el encabezado).
            while d and ts - d[0][0] > self.segundos:
                d.popleft()

    def agregar_cuadro(self, cam: str, frame: np.ndarray,
                       ts: Optional[float] = None) -> None:
        """Para cuando no hay video anotado: se codifica el cuadro crudo."""
        try:
            import cv2
            ok, jpg = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.calidad])
            if ok:
                self.agregar(cam, jpg.tobytes(), ts)
        except Exception:                                # noqa: BLE001
            pass

    def guardar_crudo(self, cam: str, frame: np.ndarray) -> None:
        with self._lock:
            self._crudo[cam] = frame

    def olvidar(self, cam: str) -> None:
        with self._lock:
            self._buf.pop(cam, None)
            self._crudo.pop(cam, None)

    # -------------------------------------------------------------- #
    def _copia(self, cam: str) -> List[Tuple[float, bytes]]:
        with self._lock:
            return list(self._buf.get(cam, ()))

    def captura(self, cam: str) -> Optional[bytes]:
        cuadros = self._copia(cam)
        return cuadros[-1][1] if cuadros else None

    def clip(self, cam: str) -> Optional[Tuple[bytes, str]]:
        cuadros = self._copia(cam)
        if len(cuadros) < 2:
            return None
        dur = max(cuadros[-1][0] - cuadros[0][0], 1e-3)
        fps = (len(cuadros) - 1) / dur
        return armar_clip([j for _, j in cuadros], fps)

    def adjuntos(self, cam: str, con_clip: bool = True) -> Dict[str, dict]:
        """Lo que va en `payload["adjuntos"]`. Vacío si no hay nada todavía."""
        out: Dict[str, dict] = {}
        cap = self.captura(cam)
        if cap:
            out["captura"] = {"tipo": "image/jpeg",
                              "b64": base64.b64encode(cap).decode("ascii")}
        with self._lock:
            crudo = self._crudo.get(cam)
        if crudo is not None:
            try:
                import cv2
                ok, jpg = cv2.imencode(".jpg", crudo, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok:
                    out["cruda"] = {"tipo": "image/jpeg",
                                    "b64": base64.b64encode(jpg.tobytes()).decode("ascii")}
            except Exception:                            # noqa: BLE001
                pass
        if con_clip:
            c = self.clip(cam)
            if c:
                datos, mime = c
                cuadros = self._copia(cam)
                out["clip"] = {
                    "tipo": mime,
                    "b64": base64.b64encode(datos).decode("ascii"),
                    "segundos": round(cuadros[-1][0] - cuadros[0][0], 1) if cuadros else 0,
                    "cuadros": len(cuadros),
                }
        return out

    def resumen(self) -> Dict[str, dict]:
        with self._lock:
            return {cam: {"cuadros": len(d),
                          "segundos": round(d[-1][0] - d[0][0], 1) if len(d) > 1 else 0}
                    for cam, d in self._buf.items()}
