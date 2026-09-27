from fastapi import FastAPI, Body, Path, Query
import uvicorn
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import datetime
import os
from contextlib import asynccontextmanager
from sqlmodel import SQLModel
from fastapi.middleware.cors import CORSMiddleware
from src.routers.camara_rutas import camara_router
from src.database import crear_tablas
from src.models import camara_model # IMPORTANTE para que SQLModel reconozca la tabla antes de crearla
from src.routers.video_rutas import video_router
from src.routers.alerta_rutas import alerta_router
from src.routers.config_rutas import config_router
from src.models import alerta_model # IMPORTANTE: registra la tabla antes de crearla

# --------------------------------------------------------------------------- #
# 26/09 — "cuando pineas la cámara se apaga y se prende, y después se cae todo
# el backend". En logs\backend.txt no quedaba NADA: ni un traceback. Eso deja
# dos candidatos, y los dos se cubren acá.
#
# 1. Un crash nativo. OpenCV con MSMF puede tirar abajo el proceso entero
#    (access violation) cuando la webcam está tomada por otro proceso —el
#    servicio de modelos—, y Python no llega a escribir nada. `faulthandler`
#    escribe el stack aunque el proceso muera así, y va a parar al log.
# 2. La consola congelada. Si alguien hace clic adentro de la ventana, Windows
#    entra en "modo selección" (QuickEdit) y BLOQUEA a quien escriba en ella.
#    Con un log por cada pedido, el backend se frena en el siguiente print y
#    desde afuera es idéntico a que se cayó. Se apaga el QuickEdit de esta
#    consola, y además se callan del log los pedidos que el panel hace cada
#    pocos segundos, que eran casi todo el texto.
# --------------------------------------------------------------------------- #
import faulthandler
import logging
import sys as _sys

try:
    faulthandler.enable(file=_sys.stderr, all_threads=True)
except Exception:                                        # noqa: BLE001
    pass


def _apagar_quickedit() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.CreateFileW("CONIN$", 0xC0000000, 3, None, 3, 0, None)
        if h in (0, -1):
            return
        modo = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(modo)):
            ENABLE_QUICK_EDIT_MODE, ENABLE_EXTENDED_FLAGS = 0x40, 0x80
            k32.SetConsoleMode(h, (modo.value & ~ENABLE_QUICK_EDIT_MODE)
                               | ENABLE_EXTENDED_FLAGS)
        k32.CloseHandle(h)
    except Exception:                                    # noqa: BLE001
        pass


class _SinSondeos(logging.Filter):
    """Saca del log de accesos los pedidos de rutina que devolvieron 200.

    Un error en esas mismas rutas SÍ se ve: solo se calla el 200 de siempre.
    """

    def filter(self, rec: logging.LogRecord) -> bool:
        try:
            return not self._es_rutina(rec)
        except Exception:                                # noqa: BLE001
            return True

    def _es_rutina(self, rec: logging.LogRecord) -> bool:
        args = rec.args if isinstance(rec.args, tuple) else ()
        # uvicorn.access: (cliente, método, ruta, versión, status)
        if len(args) >= 5:
            metodo, ruta, status = str(args[1]), str(args[2]), args[4]
            ruta = ruta.split("?", 1)[0]
            return (metodo == "GET" and status == 200 and
                    ruta in ("/camaras/config", "/estado", "/camaras/emergencia",
                             "/config/mail"))
        return False


_apagar_quickedit()
logging.getLogger("uvicorn.access").addFilter(_SinSondeos())


def hay_websockets() -> Optional[str]:
    """Devuelve el nombre de la implementación de WebSocket, o None.

    18/09, medido: sin `websockets` ni `wsproto` instalados, uvicorn no sabe
    hacer el upgrade y deja pasar el pedido como HTTP normal. El GET cae en
    `/alertas/{evento_id}` con evento_id="ws", que contesta 200 y
    `{"encontrado": false}`. El navegador dice "Unexpected response code: 200"
    en la consola —donde nadie mira— y el panel queda mostrando cámaras en
    silencio, indistinguible de una noche tranquila.

    Para un sistema que existe para avisar, quedarse sordo sin decirlo es la
    peor falla posible. Por eso se avisa fuerte al arrancar.
    """
    for mod in ("websockets", "wsproto"):
        try:
            __import__(mod)
            return mod
        except ImportError:
            continue
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    crear_tablas()
    impl = hay_websockets()
    if impl is None:
        print("=" * 70)
        print("  ATENCION: no hay soporte de WebSocket instalado.")
        print("  El panel NO va a recibir ni una alerta en vivo, y no va a")
        print("  mostrar ningun error: se va a ver igual que si no pasara nada.")
        print()
        print("      pip install websockets")
        print()
        print("  (o pip install -r FASTAPI/requirements.txt)")
        print("=" * 70)
    else:
        print(f"[backend] websockets: {impl} · el panel puede recibir alertas")

    # El mail, con el mismo criterio que el websocket de arriba.
    #
    # 22/09, medido en la base de Teo: 41 alertas de severidad 2 guardadas, 5
    # eventos distintos, CERO mails. No había archivo .env, así que todos los
    # envíos morían en el login, la excepción la comía un try y quedaba un
    # print en una ventana que nadie mira. El panel mostraba la alerta igual.
    # Un sistema de avisos que no avisa y no lo dice es peor que uno apagado.
    from src.services.email_service import RUTA_ENV, estado_mail
    listo_mail, motivo_mail = estado_mail()
    if listo_mail:
        print(f"[backend] mail: {motivo_mail}")
    else:
        print("=" * 70)
        print("  ATENCION: el aviso por MAIL esta APAGADO.")
        print(f"  {motivo_mail}")
        print()
        print("  Las alertas se van a guardar y se van a ver en el panel,")
        print("  pero NO le va a llegar un mail a nadie. Cada alerta queda")
        print('  marcada con mail_estado="apagado" para que se note.')
        print()
        print("  Se configura DESDE EL PANEL: Ajustes -> Aviso por mail.")
        print("  Pide la cuenta y una 'contrasena de aplicacion' de Google")
        print("  (16 letras, NO la clave del mail), prueba el login contra el")
        print("  servidor y recien ahi la guarda. No hace falta reiniciar.")
        print()
        print(f"  Si preferis el archivo a mano: {RUTA_ENV}")
        print("      EMAIL_SENDER=el-gmail-de-horus@gmail.com")
        print("      EMAIL_PASSWORD=la-clave-de-aplicacion-de-16-letras")
        print("=" * 70)

    # Buscar las cámaras ACÁ y no cuando el panel las pide.
    #
    # En Windows abrir un índice que no anda puede tardar 9 segundos, y son
    # cinco índices: el fetch del modal se moría esperando y la lista salía
    # vacía aunque hubiera cámara. Se busca una vez al arrancar, en un hilo, y
    # para cuando alguien abre el modal ya está la respuesta.
    from src.routers.video_rutas import refrescar_camaras
    refrescar_camaras(forzar=True)

    yield

app = FastAPI(lifespan=lifespan)

app.title =  "Mi primer API con FastAPI"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        #"http://127.0.0.1:5500",
        #"http://localhost:5500",
        "http://localhost:1420",
        # 18/09: faltaba la forma por IP. Para el navegador "localhost" y
        # "127.0.0.1" son dos origenes distintos, asi que abrir el panel en
        # http://127.0.0.1:1420 hacia que TODOS los fetch fueran bloqueados por
        # CORS. En la consola se ve como si el backend estuviera caido.
        "http://127.0.0.1:1420",
        # `npm run dev` suelto (sin Tauri) usa el puerto 5173 de Vite.
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://tauri.localhost",      # Tauri producción
        "tauri://localhost"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags = ["Home"])
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def home():
    return "Hello world!!"


@app.get("/estado", tags=["Home"])
def estado():
    """Lo que el panel necesita saber para no mentir.

    El panel ya avisa cuando se cae el websocket y cuando los modelos están
    apagados. Faltaba el tercer canal: el mail. Un tablero que dice "En vivo"
    con el mail apagado está diciendo media verdad.
    """
    from src.services.email_service import estado_mail
    listo, motivo = estado_mail()
    impl = hay_websockets()
    return {
        "websocket": {"ok": impl is not None, "implementacion": impl},
        "mail": {"ok": listo, "motivo": motivo},
    }


# La captura y el clip de cada alerta (ver src/services/evidencia.py). Se
# sirven como archivos: el panel los muestra con <img>/<video> y el mail los
# lleva adjuntos.
from fastapi.staticfiles import StaticFiles
from src.services.evidencia import PREFIJO_URL, asegurar_carpeta
app.mount(PREFIJO_URL, StaticFiles(directory=asegurar_carpeta()), name="evidencia")

app.include_router(prefix='/camaras', router=camara_router)
app.include_router(prefix='/video', router=video_router)
app.include_router(prefix='/alertas', router=alerta_router)
app.include_router(prefix='/config', router=config_router)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )