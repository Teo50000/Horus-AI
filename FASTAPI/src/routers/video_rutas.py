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


@video_router.get('/video_feed/{camara_config_id}')
def video_feed(camara_config_id: int):
    with Session(engine) as session:
        config = session.get(CamaraConfig, camara_config_id)
        if not config:
            return JSONResponse(status_code=404, content={"error": "Cámara no encontrada"})
        
        # determinar la fuente
        source = config.rtsp_url if config.rtsp_url else config.usb_index
        return StreamingResponse(gen(VideoCamera(source)), media_type="multipart/x-mixed-replace;boundary=frame")