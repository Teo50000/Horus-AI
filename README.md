# Horus-AI

## Para activar el entorno virtual creado debo ejecutar:
    < .\venv\Scripts\Activate.ps1 >
    para desactivarlo: <deactivate>
    
## Para clonar Branch especifico:
    git clone -b main --single-branch https://github.com/Teo50000/Horus-AI.git
    C:\Users\48861938\Documents\GitHub\Horus-AI

## Dar permisos para mi ventana actual:
 < Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass >

## Arrancar video con direccion IP
PS C:\Users\48861938\Downloads\ffmpeg\ffmpeg-8.1.1-essentials_build\bin>
.\ffmpeg.exe -re -stream_loop -1 -i C:\Users\48861938\Downloads\video_test_2.mp4 -c:v copy -c:a aac -f rtsp rtsp://localhost:8554/live
Ejecutar aplicacion mdmtx

## Estructura del Backend

FastAPI no es “la IA”; es el recepcionista y archivador. Valida mensajes, los convierte en filas SQLite, evita duplicados, avisa al panel y manda el correo en segundo plano. Me falta revisar las rutas restantes y el WebSocket para explicarte también cómo el panel recibe los cambios en vivo.

## ¿Qué es el backend?

El backend es el **portero, secretario y archivador** de Horus.

No es el cerebro que mira las imágenes. Ese trabajo lo hace el servicio de modelos en `00_servicio`.

El backend:

1. Recibe peticiones.
2. Guarda datos en SQLite.
3. Recibe alertas del servicio de IA.
4. Envía novedades al panel.
5. Guarda fotos y vídeos.
6. Envía correos.
7. Entrega información cuando el panel pregunta.

Su código principal está en `main.py`.

---

## Cómo arranca

Cuando ejecutás `HORUS.bat`, se levantan varias cosas:

```text
Backend       http://127.0.0.1:8000
Servicio IA   http://127.0.0.1:8010
Panel         http://localhost:1420
```

El backend es el puerto `8000`.

Al iniciar:

1. Crea las tablas de la base de datos.
2. Revisa si la base vieja necesita nuevas columnas.
3. Comprueba si puede usar WebSockets.
4. Comprueba si el correo está configurado.
5. Prepara las carpetas de evidencias.
6. Registra sus rutas HTTP.

También tiene documentación automática:

```text
http://127.0.0.1:8000/docs
```

Ahí se pueden probar los endpoints desde el navegador.

---

# Las piezas principales

## 1. `main.py`: la puerta del edificio

En `main.py` se crea la aplicación FastAPI:

```python
app = FastAPI(lifespan=lifespan)
```

Pensalo así:

```text
FastAPI = edificio
rutas   = puertas del edificio
```

Después se conectan las distintas puertas:

```python
app.include_router(prefix="/camaras", router=camara_router)
app.include_router(prefix="/video", router=video_router)
app.include_router(prefix="/alertas", router=alerta_router)
app.include_router(prefix="/config", router=config_router)
```

Eso significa:

- Todo lo de cámaras empieza por `/camaras`.
- Todo lo de vídeo empieza por `/video`.
- Todo lo de alertas empieza por `/alertas`.
- Todo lo de configuración empieza por `/config`.

---

## 2. `database.py`: el archivador

Está en `database.py`.

Usa:

```text
SQLite + SQLModel
```

SQLite es una base de datos guardada en un archivo:

```text
horus.db
```

No hace falta instalar un servidor de base de datos como PostgreSQL para que funcione.

### ¿Qué guarda?

Entre otras cosas:

- Cámaras configuradas.
- Alertas.
- Contactos de emergencia.
- Estado de revisión de cada alerta.
- Rutas de imágenes y clips.

Al arrancar se ejecuta:

```python
SQLModel.metadata.create_all(motor)
```

Eso crea las tablas si no existen.

Además, `migrar()` intenta actualizar bases antiguas. Por ejemplo, si una versión vieja no tenía la columna `mail_estado`, la agrega.

Antes de reconstruir una tabla problemática, hace una copia de seguridad. Es importante porque el historial de alertas no debería desaparecer solo porque cambió el código.

---

# Las dos tablas de cámaras

Esta parte es confusa porque hay dos modelos parecidos.

## `CamaraConfig`: la cámara real

Está en `camara_model.py`.

Representa una cámara que Horus debe abrir:

```json
{
  "id": 3,
  "nombre": "Entrada",
  "rtsp_url": "rtsp://usuario:clave@192.168.1.50:554/stream",
  "usb_index": null
}
```

Puede ser:

### Cámara IP

```json
{
  "nombre": "Entrada",
  "rtsp_url": "rtsp://usuario:clave@192.168.1.50:554/stream"
}
```

### Cámara USB

```json
{
  "nombre": "Webcam",
  "usb_index": 0
}
```

Esta tabla responde a:

> ¿Qué cámaras existen y dónde están sus vídeos?

---

## `Camara`: registro antiguo de una alerta

También está en `camara_model.py`.

No representa necesariamente una cámara física. Es una fila histórica que dice:

```text
La cámara Entrada detectó fuego a las 15:20.
```

Guarda cosas como:

- Tipo de evento.
- Confianza.
- Momento.
- Nombre de cámara.
- Imagen de evidencia.
- Clip de evidencia.

La tabla existe para mantener compatible el panel antiguo.

Una forma fácil de recordarlo:

```text
CamaraConfig = la cámara que vigilo
Camara       = un aviso que esa cámara produjo
```

---

# Las rutas HTTP

Una ruta es una dirección a la que alguien puede llamar.

Por ejemplo:

```text
GET /estado
```

significa:

> “Backend, decime cómo estás”.

## Estado del sistema

```text
GET /estado
```

Devuelve si están disponibles:

- WebSocket.
- Correo.

Ejemplo conceptual:

```json
{
  "websocket": {
    "ok": true,
    "implementacion": "websockets"
  },
  "mail": {
    "ok": false,
    "motivo": "faltan credenciales"
  }
}
```
## Cámaras

Las rutas están en `camara_rutas.py`.

### Ver cámaras configuradas

```text
GET /camaras/config
```

### Añadir una cámara

```text
POST /camaras/config
```

Ejemplo:

```json
{
  "nombre": "Entrada",
  "rtsp_url": "rtsp://usuario:clave@192.168.1.50:554/stream"
}
```

### Modificar una cámara

```text
PUT /camaras/config/3
```

### Ver una cámara concreta

```text
GET /camaras/config/3
```

También existen rutas antiguas como:

```text
GET /camaras
GET /camaras/prediccion
POST /camaras/prediccion
```

Las rutas de predicción sirven para recibir eventos en el formato antiguo.

---

# Cómo llega una alerta

El servicio de IA manda un `POST`:

```text
POST /alertas
```

El mensaje contiene información parecida a esta:

```json
{
  "version": 1,
  "id": "evento-123",
  "secuencia": 1,
  "tipo": "incendio",
  "severidad": "critica",
  "severidad_num": 3,
  "confianza": 0.87,
  "motivo": "Se detectó humo y llama",
  "sitio": {
    "camara": "cam-3",
    "camara_config_id": 3,
    "zona": "cocina"
  },
  "tiempo": {
    "inicio": "2026-10-12T15:20:00",
    "ultimo": "2026-10-12T15:20:02",
    "emitido": "2026-10-12T15:20:03",
    "duracion_s": 2,
    "epoch_inicio": 0
  }
}
```

El modelo de ese mensaje está en `alerta_model.py`.

FastAPI/Pydantic revisa automáticamente:

- Que exista el tipo de evento.
- Que la confianza esté entre `0` y `1`.
- Que la severidad esté entre `0` y `3`.
- Que exista la cámara.
- Que existan los tiempos.

Si el mensaje está mal formado, responde con error `422`.

---

# ¿Por qué existe `evento_id` y `secuencia`?

Una alerta no es necesariamente un único mensaje.

Por ejemplo, un incendio puede producir:

```text
evento-123, secuencia 1: se abrió el evento
evento-123, secuencia 2: sigue ocurriendo
evento-123, secuencia 3: aumentó la severidad
evento-123, secuencia 4: terminó
```

El backend usa:

```text
(evento_id, secuencia)
```

como combinación única.

Esto evita duplicar alertas si el servicio reintenta enviar el mismo mensaje.

Si llega dos veces:

```text
evento-123 + secuencia 1
```

el backend contesta algo como:

```json
{
  "recibido": true,
  "accion": "duplicada"
}
```

No crea otra fila.

---

# ¿Qué pasa dentro de `/alertas`?

En `alerta_rutas.py`, aproximadamente ocurre esto:

```text
1. Recibir el mensaje.
2. Validarlo.
3. Buscar si ya llegó antes.
4. Encontrar la cámara real.
5. Guardar fotos y vídeos adjuntos.
6. Guardar la alerta en SQLite.
7. Avisar al panel si corresponde.
8. Mandar mail si la severidad es suficiente.
9. Responder al servicio de IA.
```

## Alertas repetidas

El backend guarda todos los mensajes en la base, pero no llena el panel con diez tarjetas iguales.

Por defecto, el historial muestra una sola fila por evento:

```text
GET /alertas
```

Para ver todos los mensajes de un evento:

```text
GET /alertas/evento-123
```

---

# WebSocket: cómo el panel recibe alertas en vivo

HTTP funciona así:

```text
Panel pregunta → Backend responde
```

Eso sirve para consultar el historial.

Pero para alertas en tiempo real sería incómodo que el panel preguntara cada segundo. Por eso existe WebSocket:

```text
Backend → Panel
```

La gestión está en `websockets.py`.

El panel abre:

```text
WebSocket /alertas/ws
```

Cuando llega una alerta nueva:

```text
Backend guarda la alerta
Backend envía un mensaje WebSocket
Panel recibe el mensaje
Panel muestra la alerta
```

Si el panel está cerrado, la alerta igualmente se guarda en SQLite. Cuando se abre de nuevo, la recupera mediante:

```text
GET /alertas
```

Por eso WebSocket es para “en vivo”, pero la base de datos es el historial permanente.

---

# El vídeo de las cámaras

Las rutas están en `video_rutas.py`.

El panel pide:

```text
GET /video/video_feed/3
```

El backend revisa:

```text
¿Existe la cámara 3?
¿Tiene rtsp_url o usb_index?
¿El servicio de modelos está encendido?
```

## Si el servicio de modelos está encendido

El backend no abre la cámara otra vez.

Le dice al navegador:

```text
El servicio de modelos ya tiene esa cámara.
Pedile el vídeo a él.
```

Redirige hacia:

```text
http://127.0.0.1:8010/camaras/cam-3/stream
```

Esto evita que dos procesos intenten abrir simultáneamente la misma webcam o stream.

## Si el servicio de modelos está apagado

El backend intenta abrir la cámara directamente como modo de emergencia.

La idea es:

```text
Servicio IA encendido → él controla la cámara
Servicio IA apagado   → backend puede mostrar vídeo básico
```

Pero en ese segundo caso no hay detección de IA.

---

# Evidencias: fotos y clips

El servicio de modelos adjunta información en base64:

```text
captura = imagen con cajas dibujadas
cruda   = imagen original
clip    = vídeo de los segundos anteriores
```

El backend recibe esos datos y los procesa en `evidencia.py`.

Los guarda en:

```text
FASTAPI/evidencia/capturas
FASTAPI/evidencia/crudas
FASTAPI/evidencia/clips
```

Después guarda rutas como:

```text
/evidencia/capturas/camera_3_2026-10-12_15-20-00.jpg
```

El backend publica esa carpeta mediante:

```python
app.mount("/evidencia", StaticFiles(...))
```

Así el panel puede mostrar directamente la imagen.

También hay límites para no llenar el disco:

- Máximo de archivos.
- Máximo de tamaño total.
- Eliminación de los archivos más antiguos.

---

# Correo electrónico

El correo no se manda dentro del camino principal de la alerta.

El backend hace esto:

```text
Guardar alerta
Responder correctamente
Intentar mandar el correo en segundo plano
```

Eso es importante. Si el servidor SMTP está caído, la alerta no debería perderse.

Los estados posibles incluyen:

```text
enviado
apagado
fallo
sin_destinos
no_corresponde
parcial
```

La configuración está en:

```text
/config/mail
```

### Consultar estado del mail

```text
GET /config/mail
```

### Guardar credenciales

```text
POST /config/mail
```

El backend prueba primero la contraseña y solo la guarda si funciona. La contraseña no se devuelve al panel.

---

# Revisar si una alerta fue correcta o falsa

Una persona puede revisar una alerta:

```text
POST /alertas/evento-123/revision
```

Con:

```json
{
  "veredicto": "correcta",
  "nota": "Se confirmó humo real"
}
```

O:

```json
{
  "veredicto": "falsa",
  "nota": "Era vapor de la cocina"
}
```

El backend guarda esa revisión y copia la evidencia a:

```text
FASTAPI/revision/correctas
FASTAPI/revision/falsas
```

Esto sirve para entrenar mejor los modelos en el futuro.

---

# El backend no decide si hay fuego

Esta separación es fundamental:

```text
Servicio de modelos:
    "Creo que vi fuego"

Backend:
    "Guardo ese mensaje y aviso a quien corresponda"
```

El backend no analiza píxeles ni ejecuta YOLO. Recibe el resultado del servicio de IA.

El flujo completo es:

```text
Cámara
  ↓
OpenCV en servicio.py
  ↓
Modelos de IA
  ↓
Tracking y reglas
  ↓
POST http://127.0.0.1:8000/alertas
  ↓
FastAPI
  ↓
SQLite + evidencia + WebSocket + mail
```

## Resumen muy corto

```text
main.py              crea el servidor
database.py          maneja SQLite
camara_rutas.py      administra cámaras
video_rutas.py       entrega vídeo
alerta_rutas.py      recibe y procesa alertas
alerta_model.py      define el formato de las alertas
camara_model.py      define cámaras y contactos
evidencia.py         guarda fotos y clips
websockets.py        avisa al panel en tiempo real
config_rutas.py      configura el correo
```

El backend es, en esencia, el lugar donde Horus **recibe, ordena, guarda y comunica** lo que descubrió el servicio de inteligencia artificial.
Portabilidad: Si otro desarrollador clona el repositorio, no necesita configurar contraseñas ni puertos de base de datos. Solo ejecuta el código y la BD se crea automáticamente.
Simplicidad para "Edge Computing": Dado que este sistema usa OpenCV para leer cámaras (posiblemente ejecutándose en una computadora en el sitio o en un dispositivo de borde), una base de datos ligera como SQLite tiene mucho sentido para un registro local instantáneo de los eventos detectados.


