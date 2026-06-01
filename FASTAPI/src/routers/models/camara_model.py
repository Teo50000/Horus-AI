from typing import Optional

from pydantic import BaseModel, field_validator


class Camara(BaseModel):
    camera_id: Optional[int] = None # Optional indica que este campo no es obligatorio, si no se proporciona se asignará el valor None
    event_type: str
    confidence: float
    timestamp: str

    
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
