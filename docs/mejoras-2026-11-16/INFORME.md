# Horus — mejoras de modelos, video y panel (16/11/2026)

Rama: **`mejoras-ia-video-ui`** (sale de `Models`). Nada se mergeó a `main`.

Este documento explica qué estaba mal, cómo se diagnosticó, qué se cambió y
qué quedó pendiente. Las capturas de antes y después están en `capturas/`.

---

## Resumen

| Problema | Causa raíz | Resultado medido |
|---|---|---|
| Luces y brillos detectados como cuchillos/pistolas | La app instalada no cargaba YOLO y usaba la cabeza propia para todo; además, armas confirmadas en 0,2 s con score 0,35 y sin ningún filtro fotométrico | 0 de 78 luces/brillos reales del aula pasan como arma; autotest de la guarda 11/11 |
| Recuadro de persona mal encuadrado | El tracker publicaba el estado del Kalman (corrido hacia donde la persona *estaba*, con el ancho de cuando nació el track) en vez de la caja medida | La caja que se dibuja es la que vio el detector en ese cuadro |
| Video poco fluido (cuadro por cuadro) | El stream mandaba el cuadro anotado, que solo cambia cuando terminan los modelos | De ~4 cuadros únicos/s a **24,6 cuadros únicos/s** |
| "Incendios" falsos (otro error encontrado) | La segmentación disparaba con un solo cuadro y contaba como humo la cara y las manos de la gente | Humo de un cuadro: 0 eventos; humo sostenido: avisa a los 3,1 s |
| Front: tamaños, scrollbar, carteles, ayuda, menú | Ver sección 4 | Ver capturas |

Pruebas: tracking **13/13**, fusión **31/31** (3 escenarios nuevos), guarda de
armas **11/11** (nueva), manifiesto **7/7**, `cargo check` y `vite build` sin
errores.

---

## 1. Detección de armas: luces y brillos como cuchillos

### Diagnóstico

1. **La app instalada no usaba YOLO.** El `/estado` del servicio instalado
   decía `"detector": "cabeza propia"` y veía `cuchillo 0.515` en la webcam
   sin ningún cuchillo. `servicio_modelos.exe` busca los pesos de YOLO en
   `motor_yolo.MODELOS`, que en el ejecutable empaquetado es una carpeta
   interna y vacía. Los pesos que baja la app van a
   `%LOCALAPPDATA%\com.tauri-app.horusai\modelos\...`, y además
   `yolo11n.pt` y `horus_yolo4.pt` ni siquiera estaban en el manifiesto de
   descarga. Resultado: la cabeza propia (AP50 0,25 en personas) hacía
   todo, en silencio.
2. **Desde el código fuente también salía "Arma".** Con YOLO cargado, a los
   3 s saltó "pistola en poder de una persona" (0,75 y 0,86). La pistola y el
   cuchillo nacían como track con score 0,35, se confirmaban en 0,20 s y la
   regla pedía 0,4 s. El comentario del código decía "acá conviene ser
   permisivo, lo verifica el VLM aguas abajo", pero el VLM no está
   desplegado: lo permisivo iba directo a la alerta.
3. **Nada miraba la imagen de la caja.** Subir el umbral solo no alcanza: los
   brillos llegan a 0,5-0,8 de confianza.

### Solución

**Nuevo filtro `horus/04_cabezas/objetos/guarda_armas.py`**, enganchado en el
pipeline entre el detector y el tracker (`pipeline.py`, paso 1.a). Solo toca
pistola y cuchillo; el resto pasa intacto. Descarta una detección de arma si:

| Criterio | Qué detecta | Motivo en el log |
|---|---|---|
| ≥ 18 % de píxeles quemados (≥ 235) | Tubo o foco de luz | `luz_saturada` |
| Lo quemado recorre ≥ 75 % del largo de la caja | Franja de luz, reflejo en una mesa | `luz_franja` |
| Brillo medio ≥ 195 y casi sin color | Plafón, pared iluminada | `luz_brillo` |
| Mucho más clara que su entorno y con núcleo quemado | Destello, reflejo | `luz_reflejo` |
| No hay ninguna persona cerca | Cuchillo "suelto" (pistola solo con score ≥ 0,70) | `sin_portador` |
| Ocupa más del 10 % (cuchillo) / 15 % (pistola) del cuadro | Cajas absurdas de la cabeza propia | `muy_grande` |
| Mismo lugar, quieto cuadro a cuadro durante ≥ 20 s | Luces y reflejos que no se mueven | `punto_fijo` |

El "punto fijo" cuenta **segundos de presencia quieta**, no detecciones, y
con olvido lento (vida media 10 min): alguien apuntando quieto 3 s **no**
queda silenciado; una luz que está toda la noche, sí. Las detecciones que ya
cayeron por fotometría acumulan 4 veces más rápido.

Con `HORUS_DEBUG_ARMAS=<carpeta>` el servicio guarda el recorte de cada
detección de arma con su decisión, para revisar qué se tiró y por qué.

**Umbrales más estrictos** (`tracker_local.py`, `reglas.py`):

| | Antes | Ahora |
|---|---|---|
| Score para nacer (pistola / cuchillo) | 0,35 / 0,35 | 0,50 / 0,55 |
| Tiempo para confirmar el track | 0,20 s | 0,50 s |
| Persistencia que pide la regla "arma" | 0,4 s | 1,0 s |

**App instalada** (`motor_yolo.py`, `servicio.py`, `lib.rs`,
`modelos.sha256.json`, `bin/manifiesto_modelos.py`):

- `motor_yolo.carpetas_modelos()` busca los pesos en `HORUS_MODELOS_DIR`, en
  la carpeta del código y en la de `%LOCALAPPDATA%`. `servicio.py` define
  `HORUS_MODELOS_DIR` con la carpeta de `--pesos`.
- El manifiesto incluye `yolo11n.pt` y `horus_yolo4.pt` como **opcionales**.
  La app de Tauri entiende el campo `opcional`: si uno no se puede bajar,
  avisa y sigue con la cabeza propia en vez de quedarse en "No pude preparar
  los modelos".

### Verificación

- `probar_guarda_armas.py` (nuevo): 11/11. Incluye tubo de luz, reflejo,
  cuchillo real en la mano (pasa), sin portador, caja gigante, punto fijo a
  los 20,7 s y un arma que se mueve (pasa 30/30 cuadros).
- Sobre un cuadro real del aula: las 13 regiones brillantes (paneles de luz,
  ventanas), cada una con 3 márgenes de caja y como pistola y como cuchillo
  con score 0,62: **0 de 78 pasan**.
- Fusión: escenario nuevo `arma_reflejo` (un "cuchillo" de 0,6 s) → 0 eventos.

---

## 2. Recuadro de persona mal encuadrado

### Diagnóstico

`_Track.corregir()` en `tracker_local.py` guardaba como caja del track el
**estado del Kalman** (`cxcyah_a_xyxy(self.media)`), no la detección. Ese
estado mezcla predicción y medición, y en el **aspecto** (ancho/alto) el
filtro confiaba 10 veces más en la predicción (ruido de proceso 1e-2 contra
1e-1 de medición, el default de ByteTrack pensado para 30 FPS). Con el
servicio a 3-4 ticks/s en CPU, la caja quedaba atrasada respecto de la
persona y con el ancho que tenía cuando nació el track. En la app instalada
se sumaba que las cajas venían de la cabeza propia.

### Solución

- `tracker_local.py`: cuando hay medición, la caja que se publica es la
  medida. El Kalman sigue prediciendo para asociar y para cubrir oclusiones.
- `kalman.py`: ruido del aspecto equilibrado (proceso 3e-2, medición 4e-2)
  para que el filtro siga a una persona que se agacha o gira.
- En el video fluido, las cajas se llevan al instante de cada cuadro (ver 3).

---

## 3. Fluidez del video y eficiencia

### Diagnóstico

Medido con un cliente MJPEG: el stream entregaba 10 cuadros/s pero **solo ~4
eran nuevos**. El servidor mandaba el último JPEG anotado cada 1/fps
segundos, y ese JPEG solo cambia cuando termina un tick de los modelos. La
cámara entregaba 25-30 cuadros que nadie mostraba.

Otros problemas de eficiencia encontrados:

- Se codificaba un JPEG de 1280 px aunque la celda del panel mide ~450 px.
- El panel pedía el video al backend, que para cada stream consultaba el
  `/estado` completo del servicio (timeout de 3 s) y recién después
  redirigía.
- Cuando el servicio se reiniciaba, el `<img>` MJPEG quedaba congelado en el
  último cuadro sin avisar (el navegador no dispara `onerror` en un stream
  que ya empezó).

### Solución (`horus/00_servicio/servicio.py`)

- **Video separado del análisis.** Un hilo "difusor" por cámara mirada
  (`_difundir`) toma cada cuadro nuevo de la cámara (hasta `--fps-video`,
  25 por defecto), le dibuja las últimas cajas y lo empuja al panel. Los
  modelos siguen a su ritmo.
- **Sin sondeo.** `FuenteCamara.esperar()` despierta al difusor apenas llega
  un cuadro (condición compartida con el lock), y el stream espera a que
  haya un JPEG nuevo antes de mandar (`Content-Length` + `TCP_NODELAY`).
- **Cajas llevadas al instante del cuadro** (`_suavizar`): se extrapolan con
  la velocidad del Kalman el tiempo transcurrido desde el cuadro analizado
  (con freno y tope de media caja) y se suavizan contra el cuadro anterior.
  Las cajas se deslizan en vez de saltar 3 veces por segundo.
- **Tamaño justo.** `/camaras/<id>/stream?ancho=480` codifica al ancho de la
  celda (escalones 320-1920), una sola vez por ancho aunque miren varios.
- **Nadie mirando = cero costo.** El difusor se apaga a los 10 s sin
  espectadores.
- **Dibujo en primitivas.** `_primitivas()` arma una vez por tick lo que hay
  que dibujar; `_dibujar()` lo pinta sobre cualquier tamaño (la máscara de
  segmentación se cachea por tamaño). Etiquetas con fondo de color y letra
  proporcional al ancho de salida; no se salen del cuadro.
- **Menos ruido en vivo.** Detecciones sueltas grises y tracks tentativos
  solo con `--crudas` (son para calibrar). La franja de eventos arriba del
  video queda solo en la evidencia del mail.

### Solución en el panel

- La celda pide el video **directo al servicio** con el ancho de la celda.
  Si el servicio no está, cae al video crudo del backend como antes.
- `useServicioVideo` detecta reinicios del servicio (por `arriba_s`) y sube
  una "generación" que va en la URL: el video se reconecta solo.

### Verificación

| Medición | Antes | Después |
|---|---|---|
| Cuadros únicos por segundo en el panel | ~4 | **24,6** |
| Reconexión tras reiniciar el servicio | Congelado sin aviso | Skeleton y video de vuelta en ~3 s |
| Ticks de inferencia con el panel mirando | — | 9,6/s (tope 10) |

---

## 4. Front

### Diagnóstico

- **10 tamaños de letra distintos** (10, 11, 12, 13, 15, 16, 18, 20, 26 y
  28 px) sin criterio entre paneles.
- Los tres paneles tenían el mismo ancho (22vw) pero **paddings distintos**;
  Ajustes tenía doble padding y se veía más angosto.
- Botones de ícono en `1.5vw` y checkbox en `1vw`: diminutos en notebooks.
- Scrollbar con estilo solo en 2 de 3 listas.
- Los carteles "En vivo / Analizando N / Alertas sin mail" fijos arriba a la
  derecha, **encima de la cámara** de esa esquina.
- "Ayuda" era la palabra en naranja, sin forma de botón, y no hacía nada.
- Al abrir el menú la grilla **saltaba** de tamaño y el video con
  `object-fit: cover` se recortaba distinto (se perdía media escena).
- Una celda con error (un 503 mientras el servicio engancha la cámara)
  quedaba en "Sin señal" **para siempre**.
- La grilla de cámaras pineadas se perdía al reiniciar.
- `PreviewModal` tenía `localhost:8000` escrito a mano e ignoraba el prop
  `urlStop` (cortaba el endpoint equivocado en la vista previa de una
  cámara nueva).

### Solución

| Archivo | Cambio |
|---|---|
| `App.css` | Tokens de diseño (escala tipográfica `--fs-xs…--fs-xl`, colores, medidas), scrollbar global sin flechitas, skeleton loader, foco visible por teclado |
| `Dashboard.jsx/.css` | Menú en un contenedor que **anima su ancho** (0,28 s): la grilla se achica en continuo. El panel queda montado hasta que termina de cerrarse. |
| `CamaraGrid.jsx/.css` | La grilla mantiene **16:9 exacto** (container queries) y se centra; celdas vacías clickeables ("+ Agregar cámara") que abren el menú |
| `CeldaCamara.jsx/.css` | Video directo del servicio al ancho justo; nombre sobre degradé; insignia IA abajo; aviso "Sin señal" si la cámara cayó |
| `VideoStream.jsx/.css` | Skeleton con brillo hasta el primer cuadro; reconexión automática con espera creciente; `object-fit: contain` (no recorta la escena) |
| `Sidebar/EstadoSistema.jsx` (nuevo) | Los tres estados en el sidebar: discretos en calma, naranja/rojo y latiendo si algo falla; detalle al pasar el mouse o con el teclado |
| `Sidebar.jsx/.css` | Botón "?" con ícono; etiquetas accesibles en los botones |
| `AyudaModal` (nuevo) | Guía corta: cómo ver cámaras, leyenda de colores de las cajas, estados, alertas. Cierra con Esc |
| `Skeleton/SkeletonLista.jsx` (nuevo) | Filas fantasma para Cámaras, Historial y Ajustes mientras cargan |
| Paneles (Cámaras, Historial, Ajustes) | Misma caja, título de panel, ancho `clamp(300px, 26vw, 440px)` (un poco más que antes) |
| 20 CSS de componentes | Tamaños de letra pasados a la escala |
| `useGrid.js` | La grilla se recuerda en este equipo (`localStorage`, con validación) |
| `PreviewModal.jsx` | URLs desde `config.js` y `urlStop` respetado |

### Verificación

Capturas a 1440×860 con Playwright (`capturas/despues_*`). Medido cuadro a
cuadro al abrir el menú: el panel pasa de 0 a 374 px en ~250 ms y la grilla
de 1305 a 930 px **manteniendo proporción 1,778 en todos los cuadros**.

---

## 5. Otro error de modelos encontrado: incendios falsos

El historial mostró 7 avisos de "Incendio" en ~15 minutos de prueba en el
aula, sin fuego. La evidencia (`capturas/antes_humo_falso_en_cara.jpg`)
muestra la segmentación marcando **humo sobre la cara y las manos** de una
persona (piel con desenfoque de movimiento) y agua en una pared blanca.

Dos causas:

1. La regla de incendio pedía 1,5 s a los tracks de objetos pero **nada a la
   máscara de segmentación**: un solo cuadro con 2 % de "humo" alcanzaba.
2. El humo que cae encima de una persona contaba igual que en la escena.

Solución:

- `reglas.py`: la segmentación tiene que sostenerse (humo 3 s, fuego 1 s,
  tolerando un cuadro perdido). Si la otra cabeza ya lo confirma con un
  track sostenido, cuenta en el acto: no se atrasa un crítico real.
- `pipeline.py` (`_seg_sin_personas`): el área de humo se recalcula sin los
  píxeles dentro de personas confirmadas. El fuego no se toca: una persona
  en llamas es lo más grave que puede pasar.

Escenarios nuevos en `probar_fusion.py`: `humo_parpadeo` (0 eventos),
`humo_sostenido` (avisa a los 3,1 s), `arma_reflejo` (0 eventos).

---

## 6. Archivos tocados

**Modelos y servicio**

- `horus/04_cabezas/objetos/guarda_armas.py` (nuevo)
- `horus/04_cabezas/objetos/probar_guarda_armas.py` (nuevo)
- `horus/04_cabezas/objetos/motor_yolo.py`
- `horus/05_tracking/local/tracker_local.py`
- `horus/05_tracking/local/kalman.py`
- `horus/05_tracking/probar_tracking.py` (la pistola de prueba pasó de 0,48 a 0,62: el caso mide solapes, no umbrales)
- `horus/06_fusion_decision/pipeline.py`
- `horus/06_fusion_decision/reglas.py`
- `horus/06_fusion_decision/probar_fusion.py`
- `horus/00_servicio/servicio.py`

**Instalación**

- `HorusAI/src-tauri/src/lib.rs`
- `modelos.sha256.json`
- `bin/manifiesto_modelos.py`

**Panel**: ver tabla de la sección 4.

Los cambios que ya tenías sin commitear en `HorusAI/src-tauri/Cargo.toml` y
`horus/00_servicio/servicio_modelos.spec` no los toqué.

---

## 7. Pendiente y no verificado

- **Subir `yolo11n.pt` y `horus_yolo4.pt` al bucket**
  (`horus-app-release.s3.amazonaws.com/modelos/`) y regenerar el instalador
  (`servicio_modelos.exe` con PyInstaller y la app de Tauri). Hasta entonces
  la app instalada sigue con la cabeza propia (ahora con la guarda de armas
  y los umbrales nuevos, que ya la mejoran). No subí nada ni generé el
  instalador.
- **Recall de armas reales.** No hay un dataset de armas en esta PC, así que
  no medí si la guarda descarta armas de verdad. Antes de producción, correr
  `benchmark_objetos.py` sobre el set de prueba Gunmen con y sin la guarda.
  Riesgo conocido: un arma cromada bajo luz muy fuerte puede caer como
  `luz_saturada`.
- **Verificación final en vivo.** Después de los últimos reinicios la webcam
  entregó negro puro a ~1 cuadro/s (cámara tapada o aula a oscuras). La
  fluidez y las cajas se verificaron antes de eso; conviene mirarlo de nuevo
  con la cámara destapada.
- Una cámara pineada que después se borra queda en la grilla recordada
  (muestra "Esperando video…"); se saca con la ✕.

## 8. Segunda ronda (misma fecha): cámara negra, "error de CORS" e íconos

### "No se me conecta la cámara" (imagen negra)

La webcam andaba (probada directo: 24 fps con imagen normal). El problema
era una pelea entre procesos:

1. Si el servicio de modelos se reiniciaba, el panel caía enseguida al video
   del backend, y el backend abría la webcam.
2. El servicio volvía y abría la cámara mientras el backend la tenía.
   Windows le daba cuadros en **cero absoluto a 1 fps**, con `read()` OK.
3. Como la lectura "andaba", el servicio no reconectaba nunca.

Arreglos, en tres capas:

- `servicio.py` (`FuenteCamara`): cero absoluto sostenido 4 s (una escena
  oscura de verdad nunca es cero, el sensor tiene ruido) → la cámara figura
  caída con el motivo, se suelta y se reabre. Probado con una cámara falsa:
  reconecta y vuelve la imagen.
- `useServicioVideo.js` + `CeldaCamara.jsx`: el panel solo usa el video del
  backend si el servicio falta **de verdad** (nunca contestó, o no contesta
  hace 15 s). Durante un reinicio sigue pidiéndole al servicio y reintenta.
  Si pasa del backend al servicio, le avisa al backend que suelte la cámara.
- `video_rutas.py`: si el servicio contestó hace menos de 20 s, el backend
  asume que se está reiniciando y no abre la webcam (contesta 503 y el panel
  reintenta). Cubre también la vista previa y cualquier otro cliente.

### "Access to fetch … blocked by CORS policy" al renombrar una cámara

No era CORS. `PUT /camaras/config/{id}` guardaba el nombre en la base, pero
la función declaraba que devolvía una *lista* y devolvía un objeto: FastAPI
fallaba al validar y contestaba 500. Starlette arma ese 500 por fuera del
middleware de CORS, sin cabeceras, y el navegador lo muestra como CORS.

- `camara_rutas.py`: corregido el tipo de retorno en `PUT /camaras/config/{id}`
  y en `PUT /camaras/emergencia/{id}` (editar un contacto tenía el mismo bug).
- `main.py`: un middleware dentro del de CORS convierte cualquier error no
  atrapado en un 500 con cabeceras y con el motivo real.

Verificado: el `PUT` con `Origin: http://localhost:1420` responde 200 con
`access-control-allow-origin`.

### Íconos del menú

Eran el 100 % del ancho del sidebar (hasta 52 px, ícono al 70 %). Ahora
tienen tamaño fijo y coherente con el resto: botón de 40 px, ícono de 22 px
(la misma medida que el "?"), logo de 34 px y sidebar de 92 px fijo, así los
estados no se cortan ("En ..."). Tokens `--sidebar-boton` y `--sidebar-icono`
en `App.css`.

### Vista previa

`PreviewModal` pide el video a 960 px y el backend pasa ese ancho al
redirigir al servicio: etiquetas legibles y menos datos.

## 9. Tercera ronda: contraseña con ojito, skeleton e instalador

### Ojito en la contraseña

El ojito que se veía antes era el botón **nativo de Edge**: la app de
escritorio corre sobre WebView2. En Chrome no aparece y en Edge se va
apenas el campo pierde el foco. Ahora hay un componente propio,
`components/InputContrasena`, que anda igual en todos lados, se usa con el
teclado y oculta el botón nativo para que no haya dos. Se usa en la
contraseña de la cámara IP y en la del mail.

### "Esperando video…" en toda la celda

`.skeleton` (App.css) se carga después de `VideoStream.css` y su
`position: relative` le ganaba al `absolute` del skeleton, que quedaba como
una franja arriba. Con doble clase ocupa la celda entera.

### Instalador `horus-setup.exe` (v0.3.0)

El instalador anterior tenía un `backend.exe` del 5/11, sin ninguno de los
arreglos del backend. Se rehicieron los dos binarios y el paquete:

```bash
cd FASTAPI
python -m PyInstaller backend.spec --noconfirm --workpath <fuera-del-repo> --distpath <dist>
```

```bash
cd horus/00_servicio
python -m PyInstaller servicio_modelos.spec --noconfirm --workpath <fuera-del-repo> --distpath <dist>
```

Copiarlos a `HorusAI/src-tauri/binaries/` como
`backend-x86_64-pc-windows-msvc.exe` y
`servicio_modelos-x86_64-pc-windows-msvc.exe`, y después:

```bash
cd HorusAI
npx tauri build
```

El `--workpath` va fuera del repo porque `FASTAPI/build/` tiene archivos
versionados que PyInstaller pisaría.

- `servicio_modelos.spec`: se agregaron como `hiddenimports` los módulos que
  se cargan de forma dinámica (`detector_agresion`, `segmentation_engine`,
  `detector_caidas`, `guarda_armas`, …). Sin `detector_agresion` el `.exe`
  se caía al arrancar con `--agresion`.
- La versión pasa de 0.2.0 a 0.3.0 (`tauri.conf.json`, `Cargo.toml`).
- Verificado antes de empaquetar, con los `.exe` corriendo como en la app
  instalada (mismos argumentos y carpeta `%LOCALAPPDATA%`): el servicio
  carga **YOLO11 + YOLO de Horus** (antes `cabeza propia`), la cabeza de
  agresión carga, y una cámara dada de alta en `backend.exe` se ve a
  24,4 fps.
- Resultado: `instalador/horus-setup.exe` (≈ 423 MB), ignorado por Git. No
  se probó instalarlo en esta PC.

## 10. Cómo probarlo

```bash
git checkout mejoras-ia-video-ui
```

```bash
HORUS.bat
```

Pruebas sin cámara ni GPU:

```bash
python horus/04_cabezas/objetos/probar_guarda_armas.py
```

```bash
python horus/05_tracking/probar_tracking.py
```

```bash
python horus/06_fusion_decision/probar_fusion.py
```
