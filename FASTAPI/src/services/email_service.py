"""
email_service.py · el aviso que sale del sistema.

22/09. Hasta hoy este archivo tenía un solo problema y era grave: cuando el
mail NO salía, no se enteraba nadie. `EMAIL_SENDER` y `EMAIL_PASSWORD` salen
de un `.env` que no existía, así que todos los intentos morían en el login,
la excepción la comía un `try` de arriba y quedaba un `print` en una ventana
que nadie mira. El panel mostraba la alerta, la fila se guardaba, la API
contestaba 200. Un sistema que no avisó se veía igual que uno que avisó.

Es el mismo criterio que `reglas_dormidas()` en la fusión y que el cartel de
websocket al arrancar: un canal apagado tiene que DECIR que está apagado.

Acá eso son tres cosas:

1. `estado_mail()` contesta si se puede mandar y, si no, por qué — para el
   cartel de arranque y para el panel.
2. Sin credenciales se levanta `MailApagado` ANTES de tocar la red. No es un
   fallo de envío: es que nunca se intentó, y son dos cosas distintas.
3. El que llama guarda el resultado en la fila de la alerta. Ver
   `alerta_rutas._avisar_por_mail`.

La contraseña no se imprime, no se devuelve y no se loguea en ningún caso.
Para Gmail va una "contraseña de aplicación" de 16 letras, no la del mail.
"""

import os
import smtplib
from email.mime.text import MIMEText
from typing import Tuple

from dotenv import load_dotenv

# Explícito y no "el .env que haya en el directorio actual": el backend se
# arranca desde FASTAPI\ con el .bat, pero también a mano desde la raíz del
# repo, y entonces no encontraba nada sin decir por qué.
AQUI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA_ENV = os.path.join(AQUI, ".env")
load_dotenv(RUTA_ENV)
load_dotenv()                      # por si ya lo tenían en otro lado

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_TIMEOUT = float(os.getenv("SMTP_TIMEOUT", "10"))


class MailApagado(RuntimeError):
    """No hay credenciales: el mail no se intentó siquiera.

    Distinto de un fallo de envío. Un fallo se reintenta; esto se configura.
    """


def estado_mail() -> Tuple[bool, str]:
    """¿Puede salir un mail ahora mismo? Y si no, por qué.

    Nunca devuelve la contraseña. El remitente sí, que es una dirección y es
    justamente lo que hay que poder verificar de un vistazo.
    """
    faltan = [n for n, v in (("EMAIL_SENDER", EMAIL_SENDER),
                             ("EMAIL_PASSWORD", EMAIL_PASSWORD)) if not v]
    if not faltan:
        return True, "listo · remitente %s por %s:%d" % (
            EMAIL_SENDER, SMTP_HOST, SMTP_PORT)
    hay_archivo = os.path.exists(RUTA_ENV)
    return False, "falta %s%s" % (
        " y ".join(faltan),
        "" if hay_archivo else " (no existe el archivo .env en FASTAPI\\)")


def enviar_alerta_email(event_type: str, nombre_camara: str, confidence: float,
                        timestamp: str, email_receiver: str) -> None:
    listo, motivo = estado_mail()
    if not listo:
        # Antes esto llegaba hasta smtplib y explotaba con un error de login
        # que no decía nada del problema real.
        raise MailApagado(motivo)
    if not email_receiver or "@" not in str(email_receiver):
        raise ValueError("destinatario sin dirección de mail: %r" % email_receiver)

    msg = MIMEText(
        f"Se detectó un evento en {nombre_camara}.\n\n"
        f"Tipo: {event_type}\n"
        f"Confianza: {confidence * 100:.0f}%\n"
        f"Hora: {timestamp}"
    )
    msg['Subject'] = f" Alerta Horus AI — {event_type}"
    msg['From'] = EMAIL_SENDER      # el gmail que creen para Horus
    msg['To'] = email_receiver

    # 2026-09-15: timeout explicito. Sin el, un SMTP inalcanzable (sin red,
    # puerto 465 filtrado, gmail lento) deja la conexion colgada para siempre
    # y con ella la tarea de fondo que manda el aviso. La alerta ya se guardo
    # y ya salio al panel, pero el worker queda tomado y el siguiente aviso
    # no sale nunca.
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
