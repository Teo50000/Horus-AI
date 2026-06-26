from typing import List

from fastapi import Query, Path, APIRouter
from fastapi.responses import JSONResponse
from sqlmodel import Session
from src.database import engine
from src.models.camara_model import Camara, CamaraConfig, NumeroEmergencia
from src.services.websockets import manager
from fastapi import WebSocket, WebSocketDisconnect
camara_router = APIRouter()
import cv2

camaras: List[Camara] = [
    Camara(camera_id=1, event_type="fire", confidence=0.94, timestamp="2026-05-07T10:32:00"),
    Camara(camera_id=2, event_type="normal", confidence=0.99, timestamp="2026-05-07T10:35:00")
]
telefono: List[NumeroEmergencia] = []
config_camaras: List[CamaraConfig] = []

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


#metodo get para la configuracion de camaras
@camara_router.get("/config", tags = ["Camaras"])
def get_camara_config(id: int) -> CamaraConfig | dict:
    for cam in config_camaras:
        if cam.id == id:
            return JSONResponse(content=cam.model_dump(), status_code = 200) # model_dump convierte el objeto Camara en un diccionario para que pueda ser devuelto como respuesta
    return JSONResponse(content={}, status_code = 404)

@camara_router.get("/emergencia", tags = ["Telefonos"])
def get_tel_emergencia(id: int) -> NumeroEmergencia | dict:
    for tel in telefono:
        if tel.id == id:
            return JSONResponse(content=tel.model_dump(), status_code = 200)
    return JSONResponse(content={}, status_code = 404)

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
        await manager.broadcast(camara.model_dump_json()) #aca va lo de websocket para enviar alerta al frontend
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
    

@camara_router.post("/emergencia", tags=["Telefonos"])
def agregar_telefono(nuevo_tel: NumeroEmergencia) -> List[NumeroEmergencia]:

    with Session(engine) as session:
        session.add(nuevo_tel)
        session.commit()
        session.refresh(nuevo_tel)
        return nuevo_tel


# metodo put para el config de camaras
@camara_router.put("/config/{id}", tags=["Camaras"])
def actualizar_camara(id: int, cam: CamaraConfig) -> List[CamaraConfig]:
    with Session(engine) as session:
        # 1. Buscar el registro en la base de datos
        db_item = session.get(CamaraConfig, id)
        if not db_item:
            return JSONResponse(status_code=404, detail="Item no encontrado")
        cam_data = cam.model_dump(exclude_unset=True)
        
        # 3. Actualizar los campos del modelo con los nuevos valores
        db_item.sqlmodel_update(cam_data)
        
        # 4. Guardar los cambios
        session.add(db_item)
        session.commit()
        session.refresh(db_item)
        
        return db_item

@camara_router.put("/emergencia/{id}", tags=["Telefonos"])
def actualizar_telefono(id: int, tel: NumeroEmergencia) -> List[NumeroEmergencia]:
    with Session(engine) as session:
    # 1. buscarlo en la DB
        tel_db = session.get(NumeroEmergencia, id)
    
    # 2. si no existe, error
        if not tel_db:
            return JSONResponse(content={"error": "No encontrado"}, status_code=404)
        
        tel_data = tel.model_dump(exclude_unset=True)
    
    # 3. modificar los campos
        tel_db.telefono = tel.telefono
        tel_db.nombre = tel.nombre
        
        tel_db.sqlmodel_update(tel_data)
        # 4. guardar
        session.add(tel_db)
        session.commit()
        session.refresh(tel_db)
        return tel_db

#metodo put 
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
