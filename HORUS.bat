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

start "Horus backend" cmd /k ""%~dp0bin\arrancar_backend.bat""

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
  start "Horus modelos" cmd /k ""%~dp0bin\arrancar_modelos.bat"!FLAGS!"
)

start "Horus panel" cmd /k "cd /d "%~dp0HorusAI" && npm run dev"

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
