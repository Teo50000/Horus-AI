import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv


load_dotenv()

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


def enviar_alerta_email(event_type: str, nombre_camara: str, confidence: float, timestamp: str, email_receiver: str):
    msg = MIMEText(
        f"Se detectó un evento en {nombre_camara}.\n\n"
        f"Tipo: {event_type}\n"
        f"Confianza: {confidence * 100:.0f}%\n"
        f"Hora: {timestamp}"
    )
    msg['Subject'] = f" Alerta Horus AI — {event_type}"
    msg['From'] = EMAIL_SENDER      # el gmail que creen para Horus
    msg['To'] = email_receiver           # el número que recibe la alerta

    # 2026-09-15: timeout explicito. Sin el, un SMTP inalcanzable (sin red,
    # puerto 465 filtrado, gmail lento) deja la conexion colgada para siempre
    # y con ella la tarea de fondo que manda el aviso. La alerta ya se guardo
    # y ya salio al panel, pero el worker queda tomado y el siguiente aviso
    # no sale nunca.
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=10) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)