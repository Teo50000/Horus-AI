from typing import Optional
from pydantic import BaseModel, field_validator
from sqlmodel import SQLModel, Field


class Camara(SQLModel, table=True):
    camera_id: Optional[int] = Field(default=None, primary_key=True)
    event_type: str
    confidence: float
    timestamp: str
    camara_config_id: int = Field(foreign_key="camaraconfig.id")
    description: str
    snapshot_url: Optional[str] = None # = "/snapshots/camera_3_2026-06-18_14-32.jpg" , para agregar dsp en otro sprint
    clip_url: Optional[str] = None # = "/clips/camera_3_2026-06-18_14-32.mp4" , para agregar dsp en otro sprint

    
    @field_validator("confidence")
    def validate_confidence(cls, value):
        if value < 0.9 or value> 1:
            raise ValueError("La confianza debe ser mayor al 90% pero menor al 100%")
        return value
    #@field_validator("event_type")
    #@classmethod
    #def validate_event_type(cls, value):
    #        if value not in ["fire", "desmayo", "robos"]:
    #            raise ValueError("El tipo de evento debe ser 'fire', 'desmayo' o 'robos'")
    #        return value

class CamaraConfig(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    rstp_url: str
    user_id: int #FK hacia el usuario que la eligió/agregó
    nombre: Optional[str] = None