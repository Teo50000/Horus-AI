from dotenv import load_dotenv
load_dotenv()  # carga el .env antes que cualquier otra cosa

from src.services.email_service import enviar_alerta_email

try:
    enviar_alerta_email(
        event_type="Incendio",
        nombre_camara="Camara entrada",
        confidence=0.97,
        timestamp="2026-07-04T16:10:00",
        email_receiver="laritamuina@gmail.com"
    )
    print("✅ Email enviado correctamente")
except Exception as e:
    print(f"❌ Error: {e}")