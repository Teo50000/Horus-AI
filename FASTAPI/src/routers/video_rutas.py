from fastapi.responses import StreamingResponse
# import numpy as np
import time
import cv2
from fastapi.responses import StreamingResponse
#from fastapi.templating import Jinja2Templates

from src.models.video_model import VideoCamera, gen
from fastapi import APIRouter
video_router = APIRouter()


@video_router.get('/video_feed')
def video_feed():
    return StreamingResponse(gen(VideoCamera()), media_type="multipart/x-mixed-replace;boundary=frame")