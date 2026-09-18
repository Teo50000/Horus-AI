@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
title HORUS

echo ==============================================================
echo   H O R U S
echo ==============================================================
echo.

rem ===============================================================
rem  1. Python
rem ===============================================================
python --version >nul 2>&1
if errorlevel 1 (
  echo  No encuentro Python. Instalalo de python.org y marca
  echo  "Add python.exe to PATH" en la primera pantalla del instalador.
  echo.
  pause
  exit /b 1
)

rem ===============================================================
rem  2. Dependencias del backend
rem
rem     `websockets` es el que rompe callado: sin el, uvicorn no hace
rem     el upgrade del WebSocket, el panel abre, se ven las camaras...
rem     y no llega NI UNA alerta, sin ningun error a la vista.
rem ===============================================================
python -c "import fastapi, uvicorn, sqlmodel, websockets" >nul 2>&1
if errorlevel 1 (
  echo  Falta alguna dependencia del backend. Instalando una sola vez...
  echo.
  python -m pip install -q -r FASTAPI\requirements.txt
  echo.
)

rem ===============================================================
rem  3. Que modelos hay
rem
rem     Cada cabeza se prende solo si sus pesos estan, y lo que falta
rem     se dice en voz alta. Un sistema de vigilancia no puede correr
rem     sin una cabeza y dejarte creer que la esta mirando.
rem
rem     Las rutas van RELATIVAS a horus\00_servicio, que es desde donde
rem     arranca el servicio: asi no hay comillas anidadas que romper.
rem ===============================================================
set FLAGS=
set HAY_VISION=0

set M_OBJ=horus\04_cabezas\objetos\modelos\head_best_solo.pt
set M_SEG=horus\04_cabezas\segmentacion\checkpoints\head_v4_produccion.pt
set M_FIGHT=horus\fight\checkpoints\modelo_fight.pt
set M_FALL=horus\fall\checkpoints\modelo_stgcn.pt
set M_TOPO=horus\06_fusion_decision\topologia.json

echo  Cabezas:
if exist "%M_OBJ%" (
  echo    [SI] objetos        personas, armas, fuego, humo, paquetes
  set FLAGS=!FLAGS! --pesos ..\04_cabezas\objetos\modelos\head_best_solo.pt
  set HAY_VISION=1
) else (
  echo    [NO] objetos        falta %M_OBJ%
)

if exist "%M_SEG%" (
  echo    [SI] segmentacion   fuego y humo, la que mejor anda
  set FLAGS=!FLAGS! --segmentacion
  set HAY_VISION=1
) else (
  echo    [NO] segmentacion   falta %M_SEG%
)

if exist "%M_FIGHT%" (
  echo    [SI] agresion       peleas
  set FLAGS=!FLAGS! --agresion
) else (
  echo    [NO] agresion       falta %M_FIGHT%
)

if exist "%M_FALL%" (
  python -c "import mediapipe" >nul 2>&1
  if errorlevel 1 (
    echo    [NO] caidas         falta mediapipe: HORUS_herramientas.bat, opcion 5
  ) else (
    echo    [SI] caidas         desmayos y caidas
    set FLAGS=!FLAGS! --caidas --caidas-checkpoint ..\fall\checkpoints\modelo_stgcn.pt
  )
) else (
  echo    [NO] caidas         falta %M_FALL%
)

if exist "%M_TOPO%" (
  echo    [SI] topologia      zonas y horarios
  set FLAGS=!FLAGS! --topologia ..\06_fusion_decision\topologia.json
) else (
  echo    [NO] topologia      sin zonas: intrusion y merodeo quedan DORMIDAS
  echo                        se arma con HORUS_herramientas.bat, opcion 6
)
echo.

if "!HAY_VISION!"=="0" (
  echo  Ningun modelo de vision: el servicio no tendria nada que mirar,
  echo  asi que arranco solo el backend y el panel.
  echo.
  set FLAGS=
)

rem ===============================================================
rem  4. El panel
rem ===============================================================
if not exist "HorusAI\node_modules" (
  echo  Primera vez: instalando las dependencias del panel.
  echo  Tarda unos minutos. Una sola vez.
  echo.
  pushd HorusAI
  call npm install
  popd
  echo.
)

rem ===============================================================
rem  5. Arrancar
rem ===============================================================
echo ==============================================================
echo  Backend  http://127.0.0.1:8000     ^(/docs para probar a mano^)
echo  Panel    http://localhost:1420
echo.
echo  Se abren varias ventanas. Ctrl+C en cada una para cortar.
echo ==============================================================
echo.

rem Cada ventana deja su salida en logs\. Sin esto, cuando algo se cae de
rem madrugada no queda ni rastro de por que: la ventana se cierra o el texto
rem se va del buffer de la consola y no hay a donde volver a mirar.
rem
rem `python -u` es importante: sin el, python bufferea la salida y el archivo
rem queda vacio justo cuando mas lo necesitas, que es cuando el proceso murio
rem sin llegar a vaciar el buffer.
if not exist "logs" mkdir "logs"

rem `cmd /k call "ruta"` y no `cmd /k ""ruta""`: la segunda forma anida
rem comillas y cmd las come mal. Si los argumentos van adentro de esas
rem comillas, la ventana ni llega a abrirse — y como no se abre, tampoco
rem escribe en el log, asi que no queda ni rastro de que fallo.
rem Los flags viajan por variable de entorno, que `start` hereda solo.
rem ¿Ya hay un backend corriendo de antes?
rem
rem Si lo hay, arrancar otro no sirve: el segundo no puede tomar el puerto
rem 8000, muere con "address already in use", y queda una ventana con un
rem error que parece grave y no lo es. Peor: el backend viejo puede ser de
rem una version anterior del codigo, que es exactamente lo que estuvo
rem pasando hoy.
curl -s -o nul --max-time 2 http://127.0.0.1:8000/ >nul 2>&1
if not errorlevel 1 (
  echo  Ya hay un backend escuchando en el 8000: uso ese.
  echo  Si acabas de cambiar codigo del backend, cerra su ventana
  echo  ^(la que dice "Horus backend"^) y volve a correr esto.
  echo.
  goto backend_listo
)

start "Horus backend" cmd /k call "%~dp0bin\arrancar_backend.bat"

echo  Esperando al backend...
set /a ESPERA=0
:esperar_backend
curl -s -o nul --max-time 1 http://127.0.0.1:8000/ >nul 2>&1
if not errorlevel 1 goto backend_listo
set /a ESPERA+=1
if !ESPERA! GEQ 40 (
  echo  Tardo mas de 40 segundos. Mira su ventana para ver que paso.
  goto backend_listo
)
timeout /t 1 /nobreak >nul
goto esperar_backend
:backend_listo

if not "!FLAGS!"=="" (
  echo  Levantando los modelos. La primera carga tarda ^(~20 s con GPU^).
  set "HORUS_FLAGS=!FLAGS!"
  start "Horus modelos" cmd /k call "%~dp0bin\arrancar_modelos.bat"
)

start "Horus panel" cmd /k call "%~dp0bin\arrancar_panel.bat"

echo  Esperando al panel...
set /a ESPERA=0
:esperar_panel
curl -s -o nul --max-time 1 http://localhost:1420/ >nul 2>&1
if not errorlevel 1 goto panel_listo
set /a ESPERA+=1
if !ESPERA! GEQ 90 (
  echo  Tardo mas de 90 segundos. Abrilo a mano: http://localhost:1420
  goto fin
)
timeout /t 1 /nobreak >nul
goto esperar_panel
:panel_listo
start "" http://localhost:1420

:fin
echo.
echo ==============================================================
echo  Listo. Esta ventana ya no hace falta.
echo.
echo  Arriba a la derecha del panel hay un cartel verde "En vivo".
echo  Si esta rojo, el panel no esta recibiendo alertas.
echo.
rem ¿Arrancaron los modelos de verdad?
rem
rem Hasta hoy, si la ventana de modelos moria al segundo, HORUS.bat seguia
rem como si nada y el unico sintoma era un cartel en el panel. Ahora se espera
rem y, si no contesta, se dice aca mismo con el motivo.
rem
rem El bucle va FUERA de un bloque `if (...)`: una etiqueta adentro de
rem parentesis no funciona en batch: el salto se lleva puesto el bloque.
if "!FLAGS!"=="" goto sin_modelos

echo  Esperando a que carguen los modelos ^(15 a 40 s en CPU^)...
set /a ESPERA=0

:esperar_modelos
curl -s -o nul --max-time 1 http://127.0.0.1:8010/estado >nul 2>&1
if not errorlevel 1 goto modelos_listos
set /a ESPERA+=1
if !ESPERA! GEQ 60 goto modelos_no_arrancaron
timeout /t 1 /nobreak >nul
goto esperar_modelos

:modelos_no_arrancaron
echo.
echo ==============================================================
echo  LOS MODELOS NO ARRANCARON.
echo.
echo  Lo ultimo que dijo su ventana:
echo ==============================================================
if exist "logs\modelos.txt" (
  powershell -NoProfile -Command "Get-Content 'logs\modelos.txt' -Tail 15"
) else (
  echo  No se escribio logs\modelos.txt: la ventana ni llego a abrirse.
)
echo ==============================================================
echo.
goto fin

:modelos_listos
echo  Modelos arriba.

:sin_modelos

echo  Si algo se cae, no hace falta que copies nada: queda escrito en
echo     logs\backend.txt   y   logs\modelos.txt
echo  y los ves con HORUS_herramientas.bat, opcion L.
echo.
echo  OJO con hacer clic adentro de las ventanas negras: Windows entra en
echo  modo seleccion y CONGELA el programa hasta que apretes Esc. Se ve
echo  igual que si se hubiera caido.
echo.
echo  Para mandar una alerta de prueba y verla llegar:
echo     HORUS_herramientas.bat  ^(opcion 2^)
echo ==============================================================
echo.
pause
