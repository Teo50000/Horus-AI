from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

class Camara(BaseModel):
    camera_id: Optional[int] = None # Optional indica que este campo no es obligatorio, si no se proporciona se asignará el valor None
    event_type: str
    confidence: float
    timestamp: str

camara = [
    {
        "camera_id": 1,
        "event_type": "fire",
        "confidence": 0.94,
        "timestamp": "2026-05-07T10:32:00"
    },
    {
        "camera_id": 2,
        "event_type": "normal",
        "confidence": 0.99,
        "timestamp": "2026-05-07T10:35:00"
    }
]

app.title =  "Mi primer API con FastAPI"

@app.get("/", tags = ["Home"])
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def home():
    return "Hello world!!"

@app.get("/camaras", tags = ["Camaras"])
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def get_camaras() -> List[Camara]:
    #puede devolver distintos tipos de datos como diccionarios:
    #return {"Hello": "World!!"}
    #return HTMLResponse("<h1>Hola mundo!!</h1>")
    return camara

#parametros por ruta:
@app.get("/camaras/{id}", tags = ["Camaras"])
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def get_camara(id: int) -> Camara:
    #puede devolver distintos tipos de datos como diccionarios:
    #return {"Hello": "World!!"}
    #return HTMLResponse("<h1>Hola mundo!!</h1>")
    #return id
    for cam in camara:
        if cam["camera_id"] == id:
            return cam
    return[]

#parametros por query:
@app.get("/camaras/", tags = ["Camaras"])
def get_modelos_por_cat(categoria: str, año: str) -> Camara:
    for cam in camara:
        if cam["event_type"] == categoria:
            return cam
    return[]

#metodo post
@app.post("/camaras", tags=["Camaras"])
def añadir_camara(nueva_camara: Camara) -> List[Camara]:
    camara.append(nueva_camara.model_dump())# model_dump convierte el objeto Camara en un diccionario para que pueda ser añadido a la lista camara
    return camara

# metodo put
@app.put("/camaras/{id}", tags=["Camaras"])
def actualizar_camara(id: int, cam: Camara) -> List[Camara]:
    for i in camara:
        if i["camera_id"] == id:
            i["event_type"] = cam.event_type
            i["confidence"] = cam.confidence
            i["timestamp"] = cam.timestamp
    return camara

#metodo delete
@app.delete("/camaras/{id}", tags=["Camaras"])
def eliminar_camara(id: int) -> List[Camara]:
    for cam in camara:
        if cam["camera_id"] == id:
            camara.remove(cam)
    return camara
