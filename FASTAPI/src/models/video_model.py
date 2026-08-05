# import numpy as np
import time
import cv2

#from fastapi.templating import Jinja2Templates
#templates = Jinja2Templates(directory="templates")

class VideoCamera(object):
    def __init__(self, source):
        #self.video = cv2.VideoCapture(1)
        #self.video = cv2.VideoCapture("rtsp://localhost:8554/live")
        #si lo queremos hacer con direccion IP, habria que cambiarlo a self.video = cv2.VideoCapture('rstp://direccion_ip:puerto/video_feed')
        self.video = cv2.VideoCapture(source)
        self.video.set(3, 1920)  # float `width`
        self.video.set(4, 1080)  # float `height`
        # self.video = cv2.VideoCapture('Class_Det.mp4')
        # self.video = cv2.VideoCapture(args["input"])

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success or image is None:  # ← línea nueva
            return None  
        print(image.shape)
        image=cv2.resize(image,(640,360))
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def check_cameras():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Cámara encontrada en índice: {i}")
            cap.release()
        else:
            print(f"Índice {i}: no hay cámara")

#@app.get('/')
#def index(request: Request):
#    return templates.TemplateResponse("index.html", context={"request": request})

def gen(camera):
        try:
            while True:
                frame = camera.get_frame()
                if frame is None:
                    continue
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as e:
            print(f"Error en el stream: {e}")
        finally:
            # Esto garantiza que la cámara se libere pase lo que pase
            camera.video.release()
            print("Stream finalizado con éxito - Cámara liberada físicamente.")
