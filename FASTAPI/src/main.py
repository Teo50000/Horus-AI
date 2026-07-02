from fastapi import FastAPI, Body, Path, Query
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    crear_tablas()
    yield

app = FastAPI(lifespan=lifespan)

app.title =  "Mi primer API con FastAPI"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        #"http://127.0.0.1:5500",
        #"http://localhost:5500",
        "http://localhost:1420"
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

