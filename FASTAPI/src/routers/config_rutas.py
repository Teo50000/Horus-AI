"""
config_rutas.py · lo que se configura desde el panel y no desde un archivo.

22/09. El aviso por mail estaba entero del lado del código —`email_service`,
la tabla de contactos, la pantalla de Ajustes para cargarlos— y aun así no
salía un solo mail, porque faltaba la única pieza que ningún código puede
generar solo: la credencial de la cuenta que manda.

Esa pieza vivía en un archivo `.env` que había que escribir a mano, fuera de
la aplicación. Para el que usa el sistema eso es indistinguible de que esté
roto: la pantalla de contactos está, los contactos están cargados, y no llega
nada. Acá se termina eso — se configura desde adentro, como todo lo demás.

Tres reglas que no se negocian
------------------------------
**1 · La clave NUNCA vuelve.** `GET` contesta si el mail puede salir y desde
qué dirección; la contraseña no sale de esta máquina para ningún lado que no
sea el servidor SMTP. No hay endpoint que la devuelva, ni entera ni cortada.

**2 · Se prueba antes de guardar.** Si el servidor rechaza la clave no se
escribe nada. Un `.env` con una clave inválida es peor que no tener `.env`:
apaga el cartel de "ALERTAS SIN MAIL" del panel y deja el sistema igual de
mudo, pero ya sin nadie mirándolo.

**3 · Se recarga en caliente.** Las credenciales se leían al importar el
módulo. Sin recargar, guardar la clave y seguir viendo "ALERTAS SIN MAIL"
serían compatibles, y el que la guardó se iría convencido de que no anduvo.

Sobre la seguridad de esto: el backend escucha en 127.0.0.1, así que este
endpoint no se alcanza desde la red — solo desde esta máquina. Es lo que
corresponde a una app de escritorio y conviene no pretender más de lo que es.
Si algún día el backend se expone a la red, ESTO es lo primero que hay que
poner detrás de autenticación.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from src.database import get_session
from src.models.camara_model import NumeroEmergencia
from src.services import email_service as es

config_router = APIRouter(tags=["Configuración"])


class EstadoMail(BaseModel):
    ok: bool
    remitente: Optional[str] = None
    motivo: str
    destinos: List[str] = Field(default_factory=list)
    # Para que el panel pueda explicar la diferencia entre "no puedo mandar"
    # y "no hay a quién". Son dos formas distintas de no avisarle a nadie.
    hay_destinos: bool = False


class CredencialesMail(BaseModel):
    remitente: str
    clave: str


class ResultadoGuardado(BaseModel):
    ok: bool
    motivo: str
    remitente: Optional[str] = None


def _destinos(session: Session) -> List[str]:
    return [d for d in (c.direccion_mail()
                        for c in session.exec(select(NumeroEmergencia)).all()) if d]


@config_router.get("/mail", response_model=EstadoMail)
def estado_del_mail(session: Session = Depends(get_session)) -> EstadoMail:
    """Si pasa algo ahora, ¿le llega a alguien? Sin la clave, obviamente."""
    ok, motivo = es.estado_mail()
    destinos = _destinos(session)
    return EstadoMail(ok=ok, remitente=es.EMAIL_SENDER if ok else None,
                      motivo=motivo, destinos=destinos,
                      hay_destinos=bool(destinos))


@config_router.post("/mail", response_model=ResultadoGuardado)
def guardar_mail(cred: CredencialesMail) -> ResultadoGuardado:
    """Probar la clave contra el servidor y, solo si anda, guardarla.

    Ni la clave ni ningún pedazo de ella aparece en la respuesta, en los logs
    ni en los mensajes de error. Si algo falla, lo que vuelve es el motivo.
    """
    try:
        ok, motivo = es.guardar_credenciales(cred.remitente, cred.clave)
    except Exception as e:                                   # noqa: BLE001
        # A propósito sin el detalle crudo: una excepción de smtplib puede
        # traer el diálogo del servidor, y ahí adentro va el usuario.
        print(f"[config] falló al guardar las credenciales de mail: "
              f"{type(e).__name__}")
        return ResultadoGuardado(ok=False, motivo="error inesperado al probar "
                                                  "la clave; mirá la consola "
                                                  "del backend")
    if ok:
        print(f"[config] mail configurado desde el panel · remitente "
              f"{cred.remitente}")
    return ResultadoGuardado(ok=ok, motivo=motivo,
                             remitente=cred.remitente if ok else None)
