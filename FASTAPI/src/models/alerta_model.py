"""
alerta_model.py · la alerta de Horus del lado del backend.

⚠ Reescrito el 2026-09-15 desde `contrato-alertas-backend.md`. El original se
perdió y no estaba en ninguna rama; quedó solo el `.pyc`.

Guarda el mensaje **entero**. La fila de `Camara` que ya existía se sigue
escribiendo para que el panel viejo no se entere de nada, pero es una
proyección con pérdida: no tiene severidad, ni estado, ni qué modelo aportó
qué, ni el `global_id` que sigue a la persona entre cámaras. Cuando el front
quiera mostrar "lo vieron dos modelos", eso vive acá.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field as PField
from sqlmodel import Field, SQLModel, UniqueConstraint


class Alerta(SQLModel, table=True):
    """Un mensaje de Horus. NO un evento: un evento son varias filas.

    La clave natural es `(evento_id, secuencia)` y está declarada UNIQUE a
    propósito: el emisor reintenta tres veces y después manda al spool, así
    que el mismo mensaje puede llegar dos veces si solo se perdió la
    respuesta. Con la restricción, el segundo intento choca y se contesta 200
    sin duplicar; sin ella, un backend que tarda en responder llena el
    historial de copias.
    """

    __table_args__ = (UniqueConstraint("evento_id", "secuencia",
                                       name="uq_alerta_evento_secuencia"),)

    id: Optional[int] = Field(default=None, primary_key=True)

    evento_id: str = Field(index=True)
    secuencia: int
    version: int = 1

    tipo: str = Field(index=True)
    severidad: str
    severidad_num: int = Field(index=True)
    estado: str = "abierto"
    confianza: float = 0.0
    motivo: str = ""

    camara: str = Field(index=True)
    camaras: str = "[]"                 # JSON: todas las que participaron
    zona: Optional[str] = None
    instalacion: Optional[str] = None
    nodo: Optional[str] = None

    ts_inicio: str = ""                 # ISO 8601 con offset
    ts_ultimo: str = ""
    ts_emitido: str = ""
    epoch_inicio: float = 0.0
    duracion_s: float = 0.0

    global_id: Optional[int] = Field(default=None, index=True)
    tracks: str = "[]"                  # JSON

    aportes: str = "[]"                 # JSON: qué puso cada modelo
    modelos: str = "[]"                 # JSON: atajo, los modelos distintos

    necesita_vlm: bool = False
    verificacion_estado: str = "no_requiere"
    verificacion_motivo: Optional[str] = None

    confirmaciones: int = 0
    evidencia: str = "{}"               # JSON crudo, para depurar

    camara_config_id: Optional[int] = Field(default=None, index=True)
    camara_evento_id: Optional[int] = None   # la fila de compatibilidad
    recibido_en: str = ""


# --------------------------------------------------------------------------- #
# Lo que entra por el POST
# --------------------------------------------------------------------------- #
class _Sitio(BaseModel):
    camara: str
    camaras: List[str] = PField(default_factory=list)
    zona: Optional[str] = None
    instalacion: Optional[str] = None
    nodo: Optional[str] = None


class _Tiempo(BaseModel):
    inicio: str
    ultimo: str
    emitido: str
    duracion_s: float = 0.0
    epoch_inicio: float = 0.0


class _Sujeto(BaseModel):
    global_id: Optional[int] = None
    tracks: List[int] = PField(default_factory=list)


class _Aporte(BaseModel):
    modelo: str
    aporta: str
    score: float = 0.0
    detalle: Optional[Dict[str, Any]] = None


class _Verificacion(BaseModel):
    necesita_vlm: bool = False
    motivo: Optional[str] = None
    estado: str = "no_requiere"


class AlertaEntrante(BaseModel):
    """El mensaje del contrato.

    `extra="allow"` a propósito: si Horus agrega un campo, el backend tiene
    que seguir aceptando el mensaje en vez de devolver 422 y hacer que el
    emisor lo reintente tres veces y lo mande al spool. La versión
    incompatible se anuncia subiendo `version`, no agregando campos.
    """

    model_config = ConfigDict(extra="allow")

    version: int = 1
    id: str
    secuencia: int = PField(ge=1)
    tipo: str
    severidad: str
    severidad_num: int = PField(ge=0, le=3)
    estado: str = "abierto"
    confianza: float = PField(default=0.0, ge=0.0, le=1.0)
    motivo: str = ""

    sitio: _Sitio
    tiempo: _Tiempo
    sujeto: _Sujeto = PField(default_factory=_Sujeto)
    aportes: List[_Aporte] = PField(default_factory=list)
    modelos: List[str] = PField(default_factory=list)
    verificacion: _Verificacion = PField(default_factory=_Verificacion)
    confirmaciones: int = 0
    evidencia: Dict[str, Any] = PField(default_factory=dict)
    origen: Dict[str, Any] = PField(default_factory=dict)


class RespuestaAlerta(BaseModel):
    recibido: bool
    accion: str                 # "nueva" | "actualizada" | "duplicada" | "tarde"
    evento_id: str
    secuencia: int
    emitido_al_panel: bool = False
    detalle: str = ""
