from fastapi import FastAPI, Body, Path, Query
import uvicorn
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import datetime
from contextlib import asynccontextmanager
from sqlmodel import SQLModel
from fastapi.middleware.cors import CORSMiddleware
from src.routers.camara_rutas import camara_router
from src.database import crear_tablas
from src.models import camara_model # IMPORTANTE para que SQLModel reconozca la tabla antes de crearla
from src.routers.video_rutas import video_router
from src.routers.alerta_rutas import alerta_router
from src.models import alerta_model # IMPORTANTE: registra la tabla antes de crearla

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


app.include_router(prefix='/camaras', router=camara_router)
app.include_router(prefix='/video', router=video_router)
app.include_router(prefix='/alertas', router=alerta_router)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )