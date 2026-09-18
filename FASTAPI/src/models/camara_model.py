from typing import Optional
from pydantic import BaseModel, field_validator
from sqlmodel import SQLModel, Field
from dotenv import load_dotenv


class Camara(SQLModel, table=True):
    camera_id: Optional[int] = Field(default=None, primary_key=True)
    event_type: str
    confidence: float
    timestamp: str
    camara_config_id: Optional[int] = Field(default=None, foreign_key="camaraconfig.id")
    camara_config_nombre: Optional[str] = Field(default=None, foreign_key="camaraconfig.nombre")
    description: Optional[str] = None
    snapshot_url: Optional[str] = None # = "/snapshots/camera_3_2026-06-18_14-32.jpg" , para agregar dsp en otro sprint
    clip_url: Optional[str] = None # = "/clips/camera_3_2026-06-18_14-32.mp4" , para agregar dsp en otro sprint

    
    @field_validator("confidence")
    def validate_confidence(cls, value):
        # 2026-09-15: el piso estaba en 0.90 y rechazaba la mayoria de las
        # alertas reales de Horus. La capa de fusion trabaja al reves a
        # proposito -- "persistencia antes que umbral", ver
        # calibracion-umbrales-objetos: subir el umbral filtra por confianza
        # y se lleva puestos los positivos debiles. Una pelea confirmada sale
        # con 0.82 y un arma con 0.50, y el arma con 0.50 es justo la que el
        # VLM tiene que mirar. Con el piso en 0.90 esas filas explotaban y la
        # alerta contestaba 500.
        if not 0.0 <= value <= 1.0:
            raise ValueError("La confianza tiene que estar entre 0 y 1")
        return value
    #@field_validator("event_type")
    #@classmethod
    #def validate_event_type(cls, value):
    #        if value not in ["fire", "desmayo", "robos"]:
    #            raise ValueError("El tipo de evento debe ser 'fire', 'desmayo' o 'robos'")
    #        return value

class CamaraConfig(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    rtsp_url: Optional[str] = None
    usb_index: Optional[int] = None
    nombre: Optional[str]

class NumeroEmergencia(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    telefono: Optional[str] = None
    nombre: Optional[str] = None  # ej: "Policía", "Bomberos", "Jefe"