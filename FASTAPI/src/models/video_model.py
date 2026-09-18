import time
import cv2

#from fastapi.templating import Jinja2Templates
#templates = Jinja2Templates(directory="templates")

class VideoCamera(object):
    def __init__(self, source):
        #self.video = cv2.VideoCapture(1)
        #self.video = cv2.VideoCapture("rtsp://localhost:8554/live")
        #si lo queremos hacer con direccion IP, habria que cambiarlo a self.video = cv2.VideoCapture('rstp://direccion_ip:puerto/video_feed')
        # 18/09: era `cv2.VideoCapture(source)` a secas. En Windows eso usa
        # MSMF, que no abre muchas webcams que DirectShow abre al toque — el
        # mismo motivo por el que no aparecían en la lista. El orden de
        # backends vive en un solo lugar, en video_rutas.
        from src.routers.video_rutas import _abrir
        self.video = _abrir(source) or cv2.VideoCapture(source)
        self.video.set(3, 1920)  # float `width`
        self.video.set(4, 1080)  # float `height`
        # self.video = cv2.VideoCapture('Class_Det.mp4')
        # self.video = cv2.VideoCapture(args["input"])

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success or image is None:
            return None
        # 18/09: acá había un `print(image.shape)`.
        #
        # Son 30 prints por segundo por cámara a la consola de Windows, que es
        # lentísima escribiendo. Y lo importante: si alguien hace clic adentro
        # de esa ventana, Windows entra en modo selección (QuickEdit) y BLOQUEA
        # la escritura a stdout. El proceso se congela en el primer print y no
        # sigue hasta que apretás Esc. Desde afuera se ve exactamente como
        # "el backend se apagó después de un rato".
        image = cv2.resize(image, (640, 360))
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

# Qué stream está corriendo para cada cámara.
#
# 18/09: esto era `dict[int, bool]`, y tenía una carrera fea. El <img> del
# panel se vuelve a conectar cada vez que React lo re-renderiza (cambia el
# `?t=`), así que arranca un `gen()` nuevo para la MISMA cámara mientras el
# viejo todavía está vivo. El viejo, al terminar, hacía `stop_stream` y `pop`
# sobre esa clave — y apagaba el stream NUEVO. El video se cortaba solo a los
# pocos segundos y no había forma de saber por qué.
#
# Ahora cada stream lleva su propio número: solo se apaga a sí mismo.
stream_running: dict[int, int] = {}
_ultimo_token: dict[int, int] = {}


def start_stream(camara_config_id: int) -> int:
    token = _ultimo_token.get(camara_config_id, 0) + 1
    _ultimo_token[camara_config_id] = token
    stream_running[camara_config_id] = token
    return token


def stop_stream(camara_config_id: int) -> None:
    """Apaga el stream de esta cámara, sea cual sea."""
    stream_running.pop(camara_config_id, None)


def is_stream_running(camara_config_id: int) -> bool:
    return camara_config_id in stream_running


def gen(camera: VideoCamera, camara_config_id: int):
    mio = start_stream(camara_config_id)
    try:
        while stream_running.get(camara_config_id) == mio:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)   # no quemar CPU cuando no hay frame
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except (GeneratorExit, ConnectionResetError, BrokenPipeError):
        # El navegador cerró la pestaña o cambió de cámara. Es lo normal.
        pass
    except Exception as e:                               # noqa: BLE001
        print(f"[video] error en el stream de la camara {camara_config_id}: {e}")
    finally:
        # Solo apago si el que está anotado sigo siendo yo: si arrancó otro
        # stream para esta cámara, es suyo y no lo toco.
        if stream_running.get(camara_config_id) == mio:
            stream_running.pop(camara_config_id, None)
        # La cámara se libera pase lo que pase: si no, queda tomada y ni el
        # servicio de modelos ni el próximo stream la pueden abrir.
        try:
            camera.video.release()
        except Exception:                                # noqa: BLE001
            pass
