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


### Sprint 1: Aprender fundamentos de FastAPI y Estructuración
En esta etapa, el objetivo fue levantar el servidor y entender cómo FastAPI maneja las peticiones web.

*   **Modularización:** En lugar de tener todo amontonado en el main.py, usamos `APIRouter`. Configuramos un prefijo (`/camaras`) en main.py, lo que nos permitió exportar todas las rutas a camara_rutas.py manteniendo el archivo principal limpio.
*   **Usos de Métodos HTTP:** Implementamos un CRUD completo para gestionar las detecciones. Usamos `@camara_router.get` para consultar las cámaras, `@camara_router.post` para recibir nuevas detecciones, y `@camara_router.put`/`@camara_router.delete` para actualizar o descartar registros.
*   **Tipos de Respuesta:** Elegimos explícitamente usar `JSONResponse` envolviendo nuestros datos. Al combinarse con `.model_dump()`, aseguramos que siempre respondamos en un formato JSON estándar que cualquier frontend o servicio entenderá.
*   **Códigos de Estado:** Controlamos activamente la comunicación HTTP. Establecimos `status_code=200` para respuestas exitosas y `status_code=404` cuando se intenta acceder a una cámara que no existe.

### Sprint 1.2: Crear el primer endpoint y simular la conexión de IA
El objetivo aquí era preparar el terreno: simular cómo la Inteligencia Artificial se va a comunicar con nuestra API.

*   **El Mock de Datos (Simulación):** Creamos la lista `camaras` con eventos quemados (ej. `event_type="fire"`, `confidence=0.94`). Esto representa la base de datos temporal donde la IA registrará lo que ve.
*   **Parámetros por Ruta (Path Params):** Construimos el endpoint `/{id}` para buscar detecciones específicas de una cámara.
*   **Parámetros por Query (Query Params):** Introdujimos el filtro `/?categoria=fire`, esencial para buscar eventos como "incendios" o "robos" en todo el parque de cámaras.
*   **Validaciones de Parámetros:** Utilizamos las clases `Path` y `Query` de FastAPI para proteger las entradas. Por ejemplo, forzamos que el `id` de la cámara debe ser estrictamente mayor a 0 (`Path(gt=0)`) y que el filtro de la categoría tenga entre 5 y 20 caracteres (`Query(min_length=5, max_length=20)`).

### Sprint 1.3: Integrar Pydantic para proteger la información
Este fue el sprint más crítico para evitar entrada de "datos basura" desde el modelo de IA.

*   **Pydantic y Modelos de Datos:** Creamos el archivo camara_model.py definiendo la clase `Camara` que hereda de `BaseModel`. Tipamos fuertemente los datos: el ID es numérico (`int`), el evento texto (`str`) y la certeza decimal (`float`).
*   **Validaciones de Datos Complejas:** Implementamos decoradores de Pydantic (`@field_validator`). No solo validamos el "tipo" de dato, sino la "lógica de negocio".
*   **Errores Personalizados:** Dentro de ese validador, añadimos una regla estricta: *Si la IA detecta un evento pero su nivel de certeza (confidence) es menor al 90%, lo rechazamos*. Esto lo logramos lanzando un `ValueError("La confianza debe ser mayor al 90%...")`.

### Sprint 2: Conexion SQLModel y SQLite
El objetivo de este sprint fue abandonar los datos en memoria (listas temporales) e implementar una persistencia de datos real, de manera que la información sobre las detecciones sobreviva al reinicio del servidor.

*   **¿Qué hicimos?:** Integramos una base de datos relacional local utilizando **SQLite**, interactuando con ella a través de la librería **SQLModel** (que une el poder de validación de Pydantic con el ORM de SQLAlchemy).
*   **¿Cómo lo hicimos?:**
    *   **Motor y Sesiones (`database.py`):** Creamos el archivo de configuración de la base de datos. Instanciamos un `engine` que apunta a un archivo local `horus.db`. Luego, creamos la función `crear_tablas()` para inicializar el esquema en la base de datos y diseñamos la dependencia `get_session()` para inyectar una sesión de base de datos segura y aislada en cada endpoint.
    *   **Modelos de Tabla (`camara_model.py`):** Actualizamos el modelo añadiendo `SQLModel, table=True` a la clase `Camara`. Ahora, además de validar los datos de entrada, la clase actúa como definición de la tabla SQL, estableciendo atributos como `primary_key=True` en `camera_id`.
*   **¿Para qué lo hicimos?:** Para asegurar la **persistencia** del historial de detecciones e imágenes generadas por el sistema. Además, al usar SQLModel y FastAPI juntos, garantizamos que las rutas (nuestros CRUDs) puedan realizar consultas reales (`select()`, `add()`, `commit()`) a la DB validando automáticamente la integridad de los datos como su `confidence` > 90%.
### Ventajas de SQLite
Ideal para la fase de prototipado (Sprint 2): SQLite guarda todo en un solo archivo local (horus.db). Esto significa que no necesitas levantar y configurar un servidor de base de datos pesado (como PostgreSQL o MySQL) a través de Docker o instalándolo en tu PC.
Portabilidad: Si otro desarrollador clona el repositorio, no necesita configurar contraseñas ni puertos de base de datos. Solo ejecuta el código y la BD se crea automáticamente.
Simplicidad para "Edge Computing": Dado que este sistema usa OpenCV para leer cámaras (posiblemente ejecutándose en una computadora en el sitio o en un dispositivo de borde), una base de datos ligera como SQLite tiene mucho sentido para un registro local instantáneo de los eventos detectados.


