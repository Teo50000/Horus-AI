from fastapi.responses import JSONResponse, StreamingResponse
# import numpy as np
import time
import cv2
from fastapi.responses import StreamingResponse
from sqlmodel import Session
from src.database import engine
#from fastapi.templating import Jinja2Templates

from src.models.camara_model import CamaraConfig
from src.models.video_model import VideoCamera, gen
from fastapi import APIRouter
video_router = APIRouter()


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
        test_cap = cv2.VideoCapture(source)
        if not test_cap.isOpened():
            test_cap.release()
            return JSONResponse(content={"error": "No se pudo conectar a la cámara"}, status_code=400)
        test_cap.release()
        return StreamingResponse(gen(VideoCamera(source)), media_type="multipart/x-mixed-replace;boundary=frame")
    
@video_router.get('/cameras/available', tags=["Streaming video"])
def get_available_cameras():
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append({"usb_index": i, "nombre": f"Cámara {i}"})
            cap.release()
    return JSONResponse(content=available)