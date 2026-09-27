from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
# import numpy as np
import cv2
import json
import os
import threading
import urllib.request
import time
import urllib.error
from typing import Any, Dict, Optional, Tuple
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

# Sin proxy, a propósito. El servicio corre en esta misma máquina, y un
# HTTP_PROXY en el entorno —de una VPN, de un antivirus, del equipo de
# sistemas— mandaría el pedido a un proxy que no sabe resolver 127.0.0.1. La
# consecuencia sería silenciosa: el backend creería que el servicio no está,
# abriría la cámara él, y le sacaría el video a los modelos. Es la misma
# trampa que ya está documentada en horus/07_alerta/alerta.py.
_SIN_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def estado_servicio(timeout: float = 3.0) -> Tuple[str, Optional[dict]]:
    """("arriba", estado) · ("ocupado", None) · ("apagado", None).

    26/09 — "cuando pineas la cámara se apaga y se prende y después se cae
    todo el backend". La regla de acá abajo decía: si el servicio no me
    contesta en 0,6 s, abro la cámara yo. Pero hay tres motivos para no
    contestar y solo uno quiere decir "no hay servicio":

      - APAGADO: nadie escucha en el puerto. La conexión se rechaza al
        instante. Recién ahí el backend puede abrir la cámara.
      - OCUPADO: el servicio está, pero tarda. En la PC de casa corre tres
        modelos en CPU y el GIL se lo disputan; 0,6 s es poco. Abrir la cámara
        acá es sacársela a los modelos.
      - CARGANDO: el servicio contesta, pero todavía no enganchó la cámara (los
        modelos tardan ~16 s en subir). Mismo error: el backend se la llevaba
        justo antes de que el servicio la pidiera.

    Medido en logs\modelos.txt: al pinear, "[cam-1] conectada" cada 11 s con
    errores de MSMF en el medio. Eran el servicio y el backend peleándose la
    webcam: la luz se apaga y se prende. Y OpenCV con MSMF, cuando le sacan el
    dispositivo de abajo, puede matar el proceso entero sin dejar un
    traceback — que es exactamente lo que no quedó en logs\backend.txt.
    """
    # El timeout es largo A PROPÓSITO. En Windows, conectarse a un puerto de
    # 127.0.0.1 donde no escucha nadie no falla al instante: el sistema
    # reintenta el SYN y recién al segundo, más o menos, devuelve "conexión
    # rechazada". Con un timeout corto eso se vería como "ocupado" y el
    # backend no abriría nunca la cámara aunque el servicio estuviera apagado.
    # Un servicio vivo pero lento, en cambio, ACEPTA la conexión al toque (lo
    # hace el sistema operativo) y tarda en contestar: eso sí es timeout.
    import socket
    try:
        with _SIN_PROXY.open(f"{SERVICIO_URL}/estado", timeout=timeout) as r:
            return "arriba", json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        motivo = getattr(e, "reason", None)
        if isinstance(motivo, ConnectionRefusedError):
            return "apagado", None
        if isinstance(motivo, (socket.timeout, TimeoutError)):
            return "ocupado", None
        # Nombre que no resuelve, red caída, etc.: no hay servicio al que
        # sacarle nada.
        return "apagado", None
    except (socket.timeout, TimeoutError):
        return "ocupado", None
    except ConnectionRefusedError:
        return "apagado", None
    except Exception:                                    # noqa: BLE001
        # Contestó algo que no es JSON: hay ALGO escuchando. Mejor no abrir.
        return "ocupado", None


def servicio_tiene(config_id: int) -> Optional[str]:
    """Si el servicio de modelos ya tiene esta cámara abierta, su cam_id.

    18/09, el problema que trajo Teo: "me detecta la cámara pero el modelo no
    corre cuando la cam está prendida". En Windows una webcam la abre UN
    proceso a la vez. La cámara la abre el servicio, y el video sale de ahí.
    """
    fase, estado = estado_servicio()
    if fase != "arriba":
        return None
    for c in (estado or {}).get("camaras", []):
        if c.get("config_id") == config_id:
            return c.get("camara")
    return None


def _indices_del_servicio(estado: Optional[dict]) -> Dict[int, str]:
    """{usb_index: cam_id} de las webcams que el servicio tiene abiertas."""
    out: Dict[int, str] = {}
    for c in (estado or {}).get("camaras", []):
        url = str(c.get("url", ""))
        if url.isdigit():
            out[int(url)] = c.get("camara")
    return out


def _no_la_abro(fase: str, estado: Optional[dict]) -> JSONResponse:
    """La respuesta cuando la cámara es del servicio pero todavía no la sirve."""
    if fase == "ocupado":
        detalle = ("el servicio de modelos está corriendo pero tardó en "
                   "contestar; la cámara es de él y no se la saco")
    elif (estado or {}).get("fase") == "cargando":
        detalle = ("el servicio de modelos está cargando; en unos segundos "
                   "engancha la cámara y el video sale de ahí")
    else:
        detalle = ("el servicio de modelos está corriendo y todavía no "
                   "enganchó esta cámara (lo hace cada 5 s)")
    return JSONResponse(status_code=503, headers={"Retry-After": "3"}, content={
        "error": "La cámara la va a abrir el servicio de modelos",
        "detalle": detalle,
    })


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
            
        # ¿Hay servicio de modelos? Entonces la cámara es SUYA, la tenga
        # abierta todavía o no. Ver `estado_servicio`.
        fase, estado = estado_servicio()
        if fase != "apagado":
            for c in (estado or {}).get("camaras", []):
                if c.get("config_id") == camara_config_id:
                    return RedirectResponse(
                        url=f"{SERVICIO_URL}/camaras/{c.get('camara')}/stream",
                        status_code=307)
            return _no_la_abro(fase, estado)

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
    # Las webcams que ya tiene el servicio NO se abren para probarlas: se sabe
    # que andan (están dando video) y abrirlas acá se las saca a los modelos.
    fase, estado = estado_servicio()
    if fase == "ocupado":
        # El servicio está pero no contestó: no sé cuáles tiene. Mejor no
        # tocar ninguna que abrirle la suya. Queda la lista anterior.
        with _CACHE_LOCK:
            _CACHE.update(motivo="el servicio de modelos está ocupado; "
                                 "reintentá en unos segundos",
                          buscando=False, ts=time.time() - FRESCO_S + 10)
        return
    del_servicio = _indices_del_servicio(estado) if fase == "arriba" else {}
    for i in range(5):
        if i in del_servicio:
            vacios = 0
            encontradas.append({"usb_index": i, "nombre": f"Cámara {i}",
                                "en_uso": "servicio de modelos"})
            continue
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
    # Si esa webcam ya la tiene el servicio (es una cámara ya dada de alta),
    # la vista previa sale de él: abrirla acá es la misma pelea de siempre.
    fase, estado = estado_servicio()
    if fase == "ocupado":
        return _no_la_abro(fase, estado)
    cam = _indices_del_servicio(estado).get(usb_index) if fase == "arriba" else None
    if cam:
        return RedirectResponse(url=f"{SERVICIO_URL}/camaras/{cam}/stream",
                                status_code=307)
    source = usb_index
    test_cap = _abrir(source)
    # 26/09: `_abrir` devuelve None cuando no abre, y esto hacía
    # `None.isOpened()` -> 500 en vez de decir que no se pudo conectar.
    if test_cap is None:
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