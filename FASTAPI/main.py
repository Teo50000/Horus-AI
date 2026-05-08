from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse

app = FastAPI()

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
def get_camaras():
    #puede devolver distintos tipos de datos como diccionarios:
    #return {"Hello": "World!!"}
    #return HTMLResponse("<h1>Hola mundo!!</h1>")
    return camara

#parametros por ruta:
@app.get("/camaras/{id}", tags = ["Camaras"])
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def get_camara(id: int):
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
def get_modelos_por_cat(categoria: str, año: str):
    for cam in camara:
        if cam["event_type"] == categoria:
            return cam
    return[]

#metodo post
@app.post("/camaras", tags=["Camaras"])
def añadir_camara(camera_id: int = Body(),
                    event_type: str = Body(), 
                    confidence: float = Body(), 
                    timestamp: str = Body()):
    camara.append({
        "camera_id": camera_id,
        "event_type": event_type,
        "confidence": confidence,
        "timestamp": timestamp
    })
