from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

camara = [
    {
    "id": 1,
    "prediccion": "Robo",
    "modelo": "Alpha 7 III",
    "confianza": 0.95
    }
]

app.title =  "Mi primer API con FastAPI"

@app.get("/", tags = ["Home"])
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def home():
    return "Hello world!!"

@app.get("/camaras", tags = ["Home"])
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def get_camaras():
    #puede devolver distintos tipos de datos como diccionarios:
    #return {"Hello": "World!!"}
    #return HTMLResponse("<h1>Hola mundo!!</h1>")
    return camara

@app.get("/camaras/{id}", tags = ["Home"])
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def get_camara(id: int):
    #puede devolver distintos tipos de datos como diccionarios:
    #return {"Hello": "World!!"}
    #return HTMLResponse("<h1>Hola mundo!!</h1>")
    return id

