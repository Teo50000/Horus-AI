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


def recargar() -> None:
    """Volver a leer el .env sin reiniciar el backend.

    22/09. Las credenciales se leen una vez, al importar el módulo. Con la
    configuración metida en el panel eso alcanzaba para que guardar la clave
    y seguir viendo "ALERTAS SIN MAIL" fueran compatibles: el archivo estaba
    bien y el proceso tenía los valores viejos en memoria. El usuario habría
    quedado convencido de que no le funcionó.
    """
    global EMAIL_SENDER, EMAIL_PASSWORD, SMTP_HOST, SMTP_PORT, SMTP_TIMEOUT
    load_dotenv(RUTA_ENV, override=True)
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
    SMTP_TIMEOUT = float(os.getenv("SMTP_TIMEOUT", "10"))


def probar_credenciales(remitente: str, clave: str) -> Tuple[bool, str]:
    """Autenticar contra el servidor SIN mandar nada y SIN guardar nada.

    Se usa antes de escribir el .env. Guardar una clave que el servidor
    rechaza dejaría un archivo que parece configurado, apagaría el cartel del
    panel, y el sistema seguiría sin avisar — con la diferencia de que ahora
    nadie lo estaría mirando.

    Nunca devuelve ni registra la clave, solo el veredicto.
    """
    import socket
    if not remitente or "@" not in remitente:
        return False, "la dirección del remitente no parece un mail"
    if not clave:
        return False, "falta la contraseña de aplicación"
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as s:
            s.login(remitente, clave)
        return True, "el servidor aceptó la clave"
    except smtplib.SMTPAuthenticationError:
        return False, ("el servidor rechazó la clave. Acordate de que Gmail "
                       "no acepta la contraseña normal de la cuenta: tiene "
                       "que ser una contraseña de aplicación de 16 letras")
    except (socket.timeout, TimeoutError):
        return False, (f"{SMTP_HOST}:{SMTP_PORT} no contestó a tiempo. Suele "
                       "ser el firewall o la red bloqueando ese puerto")
    except OSError as e:
        return False, f"no pude conectarme a {SMTP_HOST}:{SMTP_PORT}: {e}"
    except Exception as e:                                   # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def guardar_credenciales(remitente: str, clave: str) -> Tuple[bool, str]:
    """Probar y, solo si anda, escribir el .env y recargar.

    Los espacios se sacan acá: Google muestra la clave en cuatro grupos de
    cuatro y copiarla tal cual es el error más común.
    """
    remitente = (remitente or "").strip()
    clave = (clave or "").replace(" ", "").replace("\t", "").strip()

    ok, motivo = probar_credenciales(remitente, clave)
    if not ok:
        return False, motivo

    contenido = (
        "# Generado desde el panel de Horus (Ajustes -> Aviso por mail).\n"
        "# No se sube al repositorio: esta en .gitignore.\n"
        "# Si alguna vez se filtra, se revoca desde la cuenta de Google y se\n"
        "# genera otra, sin tocar la contrasena de la cuenta.\n"
        f"EMAIL_SENDER={remitente}\n"
        f"EMAIL_PASSWORD={clave}\n")
    tmp = RUTA_ENV + ".parcial"
    try:
        os.makedirs(os.path.dirname(RUTA_ENV), exist_ok=True)
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(contenido)
        os.replace(tmp, RUTA_ENV)
        try:
            os.chmod(RUTA_ENV, 0o600)
        except OSError:
            pass
    except OSError as e:
        return False, f"no pude escribir {RUTA_ENV}: {e}"

    recargar()
    return True, "listo"


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
