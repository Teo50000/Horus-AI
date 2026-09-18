from fastapi.responses import JSONResponse, StreamingResponse
# import numpy as np
import cv2
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
    
@video_router.get('/cameras/available', tags=["Streaming video"])
def get_available_cameras():
    """Las cámaras que esta máquina puede abrir DE VERDAD.

    Dos cosas cambiaron el 18/09, las dos por el mismo síntoma —"no me aparece
    la cámara en la lista":

    1. Se prueban los backends en orden (DirectShow primero en Windows). Ver
       la nota en `_backends()`.
    2. No alcanza con `isOpened()`: se pide un frame. Un dispositivo puede
       abrir y después no entregar una sola imagen, que es justo lo que pasa
       con el permiso de cámara cortado o con la webcam tomada por otro
       programa. Ofrecer en la lista una cámara que no da imagen es peor que
       no ofrecerla: el usuario la agrega y el recuadro queda negro para
       siempre sin decir por qué.
    """
    disponibles = []
    abren_sin_imagen = []

    for i in range(5):
        cap = _abrir(i)
        if cap is None:
            continue
        try:
            ok, frame = False, None
            for _ in range(3):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
            if ok and frame is not None:
                disponibles.append({
                    "usb_index": i,
                    "nombre": f"Cámara {i}",
                    "resolucion": f"{frame.shape[1]}x{frame.shape[0]}",
                })
            else:
                abren_sin_imagen.append(i)
        finally:
            cap.release()

    # La lista sigue siendo una lista: el panel viejo no se entera del cambio.
    # El "por qué está vacía" viaja en una cabecera, para que el panel nuevo
    # pueda decirlo en vez de mostrar un cuadro en blanco.
    cabeceras = {}
    if not disponibles:
        if abren_sin_imagen:
            cabeceras["X-Horus-Motivo"] = (
                f"indices {abren_sin_imagen} abren pero no dan imagen: la "
                f"camara la tiene otro programa, o Windows tiene cortado el "
                f"permiso de camara para apps de escritorio")
        else:
            cabeceras["X-Horus-Motivo"] = (
                "no se encontro ninguna camara en los indices 0 a 4")
    return JSONResponse(content=disponibles, headers=cabeceras)

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