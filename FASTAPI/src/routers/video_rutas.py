from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
# import numpy as np
import cv2
import json
import os
import threading
import urllib.request
import time
from typing import Any, Dict, Optional
from sqlmodel import Session
from src.database import engine
import src.models.video_model as video_model  
#from fastapi.templating import Jinja2Templates

from src.models.camara_model import CamaraConfig
from src.models.video_model import VideoCamera, gen
from fastapi import APIRouter
video_router = APIRouter()


# --------------------------------------------------------------------------- #
def _backends():
    """Los backends de captura a probar, en el orden que conviene.

    18/09: acá estaba el motivo de "la cámara no aparece en la lista". En
    Windows `cv2.VideoCapture(i)` usa MSMF, y MSMF no abre muchas webcams —o
    tarda diez segundos por índice, y para cuando contesta el navegador ya
    cortó el fetch. Las mismas cámaras abren al toque con DirectShow, que es
    lo que usan la app Cámara de Windows y Discord. El servicio
    (horus/00_servicio) ya hacía esto bien; el backend no.
    """
    import os
    if os.name == "nt":
        orden = ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY")
    else:
        orden = ("CAP_V4L2", "CAP_ANY")
    return [getattr(cv2, n) for n in orden if hasattr(cv2, n)]


SERVICIO_URL = os.environ.get("HORUS_SERVICIO_URL", "http://127.0.0.1:8010")


def servicio_tiene(config_id: int) -> Optional[str]:
    """Si el servicio de modelos ya tiene esta cámara abierta, su cam_id.

    18/09, el problema que trajo Teo: "me detecta la cámara pero el modelo no
    corre cuando la cam está prendida". En Windows una webcam la abre UN
    proceso a la vez. El panel le pedía el video al backend, el backend abría
    la cámara, y entonces el servicio no podía: los modelos corriendo sobre
    nada, y en pantalla todo normal.

    Turnarse no sirve. La cámara la abre el servicio, que es el que la
    necesita para analizar, y el video sale de ahí — ya viene con las cajas
    dibujadas, que además es lo que uno quiere ver. Este chequeo es para que
    el backend no se la saque ni aunque alguien pegue el endpoint viejo a
    mano.
    """
    try:
        with urllib.request.urlopen(f"{SERVICIO_URL}/estado", timeout=0.6) as r:
            estado = json.loads(r.read().decode("utf-8"))
    except Exception:                                    # noqa: BLE001
        return None                                      # no está: seguimos nosotros
    for c in estado.get("camaras", []):
        if c.get("config_id") == config_id:
            return c.get("camara")
    return None


def _abrir(fuente):
    """Abre una cámara probando los backends en orden. Devuelve el capture o None.

    Para índices USB prueba backend por backend; para una URL RTSP va derecho
    al default, que es FFMPEG.
    """
    if not isinstance(fuente, int):
        cap = cv2.VideoCapture(fuente)
        return cap if cap.isOpened() else (cap.release() or None)

    for flag in _backends():
        cap = cv2.VideoCapture(fuente, flag)
        if cap.isOpened():
            return cap
        # SIEMPRE, abra o no: un capture sin liberar deja el dispositivo
        # tomado y hace fallar el intento siguiente por su cuenta.
        cap.release()
    return None


@video_router.get('/video_feed/{camara_config_id}', tags=["Streaming video"])
def video_feed(camara_config_id: int):
    with Session(engine) as session:
        config = session.get(CamaraConfig, camara_config_id)
        if not config:
            return JSONResponse(status_code=404, content={"error": "Cámara no encontrada"})
        
        # determinar la fuente
        if config.rtsp_url:
                source = config.rtsp_url
        elif config.usb_index is not None:
                source = config.usb_index
        else:
                return JSONResponse(content={"error": "Debe proveer rtsp_url o usb_index"}, status_code=400)
            
        # ¿La tiene el servicio de modelos? Entonces el video sale de ahí.
        cam = servicio_tiene(camara_config_id)
        if cam:
            return RedirectResponse(
                url=f"{SERVICIO_URL}/camaras/{cam}/stream", status_code=307)

        # validar conexión
        test_cap = _abrir(source)
        if test_cap is None:
            return JSONResponse(status_code=409, content={
                "error": "No se pudo abrir la cámara",
                "fuente": str(source),
                "posibles_causas": [
                    "Ya la tiene otro programa: en Windows una webcam la abre "
                    "UNO solo a la vez (Zoom, Teams, Discord, la app Cámara, o "
                    "la ventana 'Horus modelos' de HORUS.bat).",
                    "Windows tiene cortado el permiso de cámara para "
                    "aplicaciones de escritorio.",
                    "La cámara se desconectó.",
                ],
                "para_ver_que_pasa": "python FASTAPI/diagnostico_camaras.py",
            })
        test_cap.release()
        return StreamingResponse(
            gen(VideoCamera(source), camara_config_id),
            media_type="multipart/x-mixed-replace;boundary=frame"
        )
    
# --------------------------------------------------------------------------- #
# Búsqueda de cámaras
#
# 18/09, medido en la máquina de Teo: abrir el índice 0 con MSMF tarda 9,3
# SEGUNDOS antes de fallar. La versión anterior de este endpoint recorría los
# índices 0..4 en serie dentro del request, o sea entre 20 y 45 segundos de
# espera. El navegador cortaba el fetch mucho antes, el modal recibía un error
# que no miraba nadie, y la lista quedaba vacía. O sea: aunque la cámara
# hubiera andado perfecto, igual no aparecía.
#
# Ahora la búsqueda pasa fuera del request: una al arrancar el backend, y
# después cada tanto o cuando se pide de prepo. El endpoint contesta al toque
# con lo último que se sabe, y dice si todavía está buscando.
# --------------------------------------------------------------------------- #
_CACHE: Dict[str, Any] = {"lista": [], "motivo": None, "ts": 0.0,
                          "buscando": False, "tardo_s": 0.0}
_CACHE_LOCK = threading.Lock()
FRESCO_S = 120.0


def _probar(indice: int) -> Optional[dict]:
    """Abre, pide una imagen, cierra. None si no sirve.

    `isOpened()` no alcanza: un dispositivo puede abrir y después no entregar
    un solo frame —permiso de Windows cortado, o la webcam tomada por otro
    programa. Ofrecer en la lista una cámara así es peor que no ofrecerla: el
    usuario la agrega y el recuadro queda negro para siempre sin decir por qué.
    """
    cap = _abrir(indice)
    if cap is None:
        return None
    try:
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                return {"usb_index": indice, "nombre": f"Cámara {indice}",
                        "resolucion": f"{frame.shape[1]}x{frame.shape[0]}"}
        return {"usb_index": indice, "_sin_imagen": True}
    finally:
        cap.release()


def _buscar_camaras() -> None:
    t0 = time.time()
    encontradas, sin_imagen, vacios = [], [], 0
    for i in range(5):
        r = _probar(i)
        if r is None:
            vacios += 1
            # Si los dos primeros índices no existen, no hay nada más atrás:
            # seguir probando solo suma segundos de espera.
            if vacios >= 2 and not encontradas and not sin_imagen:
                break
            continue
        vacios = 0
        (sin_imagen if r.get("_sin_imagen") else encontradas).append(r)

    if encontradas:
        motivo = None
    elif sin_imagen:
        idx = [c["usb_index"] for c in sin_imagen]
        motivo = (f"la camara del indice {idx[0]} se abre pero no entrega "
                  f"imagen: la tiene otro programa, o Windows tiene cortado "
                  f"el permiso de camara para apps de escritorio "
                  f"(Configuracion > Privacidad > Camara, el interruptor de "
                  f"abajo de todo)")
    else:
        motivo = "no se encontro ninguna camara conectada"

    with _CACHE_LOCK:
        _CACHE.update(lista=encontradas, motivo=motivo, ts=time.time(),
                      buscando=False, tardo_s=round(time.time() - t0, 1))


def refrescar_camaras(forzar: bool = False) -> None:
    """Dispara la búsqueda en un hilo, si no hay una en curso."""
    with _CACHE_LOCK:
        if _CACHE["buscando"]:
            return
        if not forzar and (time.time() - _CACHE["ts"]) < FRESCO_S:
            return
        _CACHE["buscando"] = True
    threading.Thread(target=_buscar_camaras, name="horus-buscar-camaras",
                     daemon=True).start()


@video_router.get('/cameras/available', tags=["Streaming video"])
def get_available_cameras(refrescar: bool = False):
    """Las cámaras que esta máquina puede abrir de verdad. Contesta al toque."""
    refrescar_camaras(forzar=refrescar)
    with _CACHE_LOCK:
        lista = list(_CACHE["lista"])
        motivo = _CACHE["motivo"]
        buscando = _CACHE["buscando"]
        nunca = _CACHE["ts"] == 0.0
        tardo = _CACHE["tardo_s"]

    # La respuesta sigue siendo una lista: el panel viejo no se entera del
    # cambio. El contexto va en cabeceras.
    cab = {}
    if buscando and nunca:
        cab["X-Horus-Buscando"] = "1"
        cab["X-Horus-Motivo"] = ("buscando camaras... en Windows cada intento "
                                 "puede tardar varios segundos")
    elif motivo:
        cab["X-Horus-Motivo"] = motivo
    if tardo:
        cab["X-Horus-Tardo"] = str(tardo)
    return JSONResponse(content=lista, headers=cab)


@video_router.post('/stop_feed/{camara_config_id}', tags=["Streaming video"])
def stop_stream(camara_config_id: int):
    was_running = video_model.is_stream_running(camara_config_id)
    video_model.stop_stream(camara_config_id)
    return {
        "status": "Streaming detenido",
        "camara_config_id": camara_config_id,
        "was_running": was_running
    }
    
@video_router.get('/preview/{usb_index}', tags=["Streaming video"])
def video_preview(usb_index: int):
    source = usb_index
    test_cap = _abrir(source)
    if not test_cap.isOpened():
        test_cap.release()
        return JSONResponse(content={"error": "No se pudo conectar"}, status_code=400)
    test_cap.release()
    # usamos usb_index como id temporal para el stream
    return StreamingResponse(
        gen(VideoCamera(source), usb_index),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

@video_router.post('/stop_preview/{usb_index}', tags=["Streaming video"])
def stop_preview(usb_index: int):
    was_running = video_model.is_stream_running(usb_index)
    video_model.stop_stream(usb_index)
    return {
        "status": "Preview detenido",
        "usb_index": usb_index,
        "was_running": was_running
    }