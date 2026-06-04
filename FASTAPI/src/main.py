from fastapi import FastAPI, Body, Path, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import datetime
from contextlib import asynccontextmanager

from sqlmodel import SQLModel
from src.routers.camara_rutas import camara_router
from src.database import crear_tablas
from src.models import camara_model # IMPORTANTE para que SQLModel reconozca la tabla antes de crearla

@asynccontextmanager
async def lifespan(app: FastAPI):
    crear_tablas()
    yield

app = FastAPI(lifespan=lifespan)

app.title =  "Mi primer API con FastAPI"

@app.get("/", tags = ["Home"])
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def home():
    return "Hello world!!"


app.include_router(prefix='/camaras', router=camara_router)

