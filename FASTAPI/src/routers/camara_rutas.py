from typing import List

from fastapi import Query, Path, APIRouter
from fastapi.responses import JSONResponse
from sqlmodel import Session
from src.database import engine
from src.models.camara_model import Camara, CamaraConfig
from src.services.websockets import manager
from fastapi import WebSocket, WebSocketDisconnect
camara_router = APIRouter()
import cv2

camaras: List[Camara] = [
    Camara(camera_id=1, event_type="fire", confidence=0.94, timestamp="2026-05-07T10:32:00"),
    Camara(camera_id=2, event_type="normal", confidence=0.99, timestamp="2026-05-07T10:35:00")
]

@camara_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # solo mantiene la conexión abierta
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@camara_router.get("", tags = ["Camaras"], status_code=200, response_description="Nos debe devolver una respuesta exitosa")
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def get_camaras() -> List[Camara]:
    #puede devolver distintos tipos de datos como diccionarios:
    #return {"Hello": "World!!"}
    #return HTMLResponse("<h1>Hola mundo!!</h1>")
    content = [cam.model_dump() for cam in camaras] # model_dump convierte cada objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content=content, status_code = 200)

#parametros por ruta:
@camara_router.get("/{id}", tags = ["Camaras"])
#funcion encargada de devolver un mensaje al acceder a la ruta raíz del servidor
def get_camara(id: int = Path(gt=0)) -> Camara | dict:
    #puede devolver distintos tipos de datos como diccionarios:
    #return {"Hello": "World!!"}
    #return HTMLResponse("<h1>Hola mundo!!</h1>")
    #return id
    for cam in camaras:
        if cam.camera_id == id:
            return JSONResponse(content=cam.model_dump(), status_code = 200) # model_dump convierte el objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content={}, status_code = 404) # si no se encuentra la cámara con el id proporcionado, se devuelve un diccionario vacío con un código de estado 404 (No encontrado)

#parametros por query:
@camara_router.get("/by_category", tags = ["Camaras"])
def get_modelos_por_cat(categoria: str = Query(min_length=5, max_length=20)) -> Camara | dict:
    for cam in camaras:
        if cam.event_type == categoria:
            return JSONResponse(content=cam.model_dump()) # model_dump convierte el objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content={})

#metodo post
@camara_router.post("", tags=["Camaras"])
def añadir_camara(nueva_camara: Camara) -> List[Camara]:
    camaras.append(nueva_camara)# model_dump convierte el objeto Camara en un diccionario para que pueda ser añadido a la lista camaras
    content = [cam.model_dump() for cam in camaras] # model_dump convierte cada objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content=content)

#metodo post pero guardandolo en la base de datos
@camara_router.post("/prediccion", tags=["Camaras"])
async def recibir_prediccion(camara: Camara):
    with Session(engine) as session:
        session.add(camara)
        session.commit()
        session.refresh(camara)
        await manager.broadcast(camara.model_dump_json())
        return camara
    
@camara_router.post("/config", tags=["Camaras"])
def agregar_camara_config(config: CamaraConfig):
    test_cap = cv2.VideoCapture(config.rtsp_url)
    if not test_cap.isOpened():
        test_cap.release()
        return JSONResponse(content={"error": "No se pudo conectar a la cámara"}, status_code=400)
    test_cap.release()
    
    with Session(engine) as session:
        session.add(config)
        session.commit()
        session.refresh(config)
        return config

# metodo put
@camara_router.put("/{id}", tags=["Camaras"])
def actualizar_camara(id: int, cam: Camara) -> List[Camara]:
    for i in camaras:
        if i.camera_id == id:
            i.event_type = cam.event_type
            i.confidence = cam.confidence
            i.timestamp = cam.timestamp
    content = [c.model_dump() for c in camaras] # model_dump convierte cada objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content=content)

#metodo delete
@camara_router.delete("/{id}", tags=["Camaras"])
def eliminar_camara(id: int) -> List[Camara]:
    for cam in camaras:
        if cam.camera_id == id:
            camaras.remove(cam)
    content = [c.model_dump() for c in camaras] # model_dump convierte cada objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content=content)
