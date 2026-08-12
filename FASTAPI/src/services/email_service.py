import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv


load_dotenv()

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")


def enviar_alerta_email(event_type: str, nombre_camara: str, confidence: float, timestamp: str, email_receiver=str):
    msg = MIMEText(
        f"Se detectó un evento en {nombre_camara}.\n\n"
        f"Tipo: {event_type}\n"
        f"Confianza: {confidence * 100:.0f}%\n"
        f"Hora: {timestamp}"
    )
    msg['Subject'] = f" Alerta Horus AI — {event_type}"
    msg['From'] = EMAIL_SENDER      # el gmail que creen para Horus
    msg['To'] = email_receiver           # el número que recibe la alerta

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)