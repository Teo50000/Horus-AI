from typing import List

from fastapi import Query
from fastapi.responses import JSONResponse
from src.routers.models.camara_model import Camara

camaras: List[Camara] = [
    Camara(camera_id=1, event_type="fire", confidence=0.94, timestamp="2026-05-07T10:32:00"),
    Camara(camera_id=2, event_type="normal", confidence=0.99, timestamp="2026-05-07T10:35:00")
]

@app.get("/camaras", tags = ["Camaras"], status_code=200, response_description="Nos debe devolver una respuesta exitosa")
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def get_camaras() -> List[Camara]:
    #puede devolver distintos tipos de datos como diccionarios:
    #return {"Hello": "World!!"}
    #return HTMLResponse("<h1>Hola mundo!!</h1>")
    content = [movie.model_dump() for movie in camara] # model_dump convierte cada objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content=content, status_code = 200)

#parametros por ruta:
@app.get("/camaras/{id}", tags = ["Camaras"])
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def get_camara(id: int = Path(gt=0)) -> Camara | dict:
    #puede devolver distintos tipos de datos como diccionarios:
    #return {"Hello": "World!!"}
    #return HTMLResponse("<h1>Hola mundo!!</h1>")
    #return id
    for cam in camara:
        if cam.camera_id == id:
            return JSONResponse(content=cam.model_dump(), status_code = 200) # model_dump convierte el objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content={}, status_code = 404) # si no se encuentra la cámara con el id proporcionado, se devuelve un diccionario vacío con un código de estado 404 (No encontrado)

#parametros por query:
@app.get("/camaras/", tags = ["Camaras"])
def get_modelos_por_cat(categoria: str = Query(min_length=5, max_length=20)) -> Camara | dict:
    for cam in camara:
        if cam.event_type == categoria:
            return JSONResponse(content=cam.model_dump()) # model_dump convierte el objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content={})

#metodo post
@app.post("/camaras", tags=["Camaras"])
def añadir_camara(nueva_camara: Camara) -> List[Camara]:
    camara.append(nueva_camara)# model_dump convierte el objeto Camara en un diccionario para que pueda ser añadido a la lista camara
    content = [movie.model_dump() for movie in camara] # model_dump convierte cada objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content=content)

# metodo put
@app.put("/camaras/{id}", tags=["Camaras"])
def actualizar_camara(id: int, cam: Camara) -> List[Camara]:
    for i in camara:
        if i.camera_id == id:
            i.event_type = cam.event_type
            i.confidence = cam.confidence
            i.timestamp = cam.timestamp
    content = [movie.model_dump() for movie in camara] # model_dump convierte cada objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content=content)

#metodo delete
@app.delete("/camaras/{id}", tags=["Camaras"])
def eliminar_camara(id: int) -> List[Camara]:
    for cam in camara:
        if cam.camera_id == id:
            camara.remove(cam)
    content = [movie.model_dump() for movie in camara] # model_dump convierte cada objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content=content)
