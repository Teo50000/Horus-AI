@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
title Horus - herramientas

:menu
cls
echo ==============================================================
echo   HORUS - herramientas
echo ==============================================================
echo.
echo   Para usar el sistema no hace falta nada de esto:
echo   con HORUS.bat alcanza. Esto es lo de vez en cuando.
echo.
echo   1  Correr las 11 suites de pruebas
echo   2  Mandar una alerta de prueba al panel
echo   3  Ver que esta instalado y que falta
echo   4  Pasar el detector por una carpeta de fotos tuyas
echo   5  Dejar lista la cabeza de CAIDAS
echo   6  Dibujar la zona restringida ^(prende INTRUSION^)
echo   7  Adelgazar head_best_solo.pt  ^(93 MB -^> 33 MB^)
echo   8  Subir los commits a GitHub
echo   9  Solo el backend, sin panel ni modelos
echo   M  Arrancar SOLO los modelos, aca mismo
echo   Q  Que esta viendo Horus ahora mismo ^(alerte o no^)
echo   L  Ver por que se cayo algo ^(los ultimos logs^)
echo   V  Ver que camaras encuentra tu PC
echo   C  Compactar el repositorio ^(.git ocupa 2,9 GB^)
echo.
echo   0  Salir
echo.
set /p OPCION=  Numero:

if "%OPCION%"=="1" goto pruebas
if "%OPCION%"=="2" goto alerta
if "%OPCION%"=="3" goto chequeo
if "%OPCION%"=="4" goto fotos
if "%OPCION%"=="5" goto mediapipe
if "%OPCION%"=="6" goto topologia
if "%OPCION%"=="7" goto adelgazar
if "%OPCION%"=="8" goto subir
if "%OPCION%"=="9" goto backend
if /i "%OPCION%"=="M" goto modelos
if /i "%OPCION%"=="Q" goto queve
if /i "%OPCION%"=="L" goto logs
if /i "%OPCION%"=="V" goto camaras
if /i "%OPCION%"=="C" goto compactar
if "%OPCION%"=="0" exit /b
goto menu

rem ===============================================================
:pruebas
cls
echo ==============================================================
echo  HORUS - las 11 suites
echo ==============================================================
echo.
set FALLAS=0
set SALIDA=%TEMP%\horus_suite.txt

call :suite "horus\03_backbone"             probar_huella_backbone.py
call :suite "horus\04_cabezas\segmentacion" probar_segmentation_engine.py
call :suite "horus\04_cabezas\objetos"      probar_tope_y_guarda.py
call :suite "horus\05_tracking"             probar_tracking.py
call :suite "horus\06_fusion_decision"      probar_fusion.py
call :suite "horus\07_alerta"               probar_alerta.py
call :suite "horus\fight"                   probar_detector_agresion.py
call :suite "horus\fall"                    probar_detector_caidas.py
call :suite "horus\00_servicio"            probar_servicio.py
call :suite "FASTAPI"                       probar_alertas_backend.py

rem La del panel es javascript, asi que va con node y aparte.
pushd HorusAI
where node >nul 2>&1
if errorlevel 1 (
  echo   SALTEADA probar_fuente_video.mjs   ^(no encuentro node^)
) else (
  node pruebas\probar_fuente_video.mjs > "%SALIDA%" 2>&1
  if errorlevel 1 (
    echo   FALLA  probar_fuente_video.mjs
    type "%SALIDA%"
    set /a FALLAS+=1
  ) else (
    for /f "usebackq delims=" %%L in (`findstr /R /V "^$" "%SALIDA%"`) do set RES2=%%L
    echo   OK     probar_fuente_video.mjs          !RES2!
  )
)
popd

echo ==============================================================
if !FALLAS!==0 (
  echo   TODO VERDE
) else (
  echo   !FALLAS! suite^(s^) con fallas
)
echo ==============================================================
echo.
pause
goto menu

:suite
pushd "%~1"
python "%~2" > "%SALIDA%" 2>&1
set RC=!ERRORLEVEL!
set RESUMEN=
for /f "usebackq delims=" %%L in (`findstr /R /V "^$" "%SALIDA%"`) do set RESUMEN=%%L
if !RC!==0 (
  echo   OK     %~2   !RESUMEN!
) else (
  echo   FALLA  %~2   ^(codigo !RC!^)
  echo   ------------------------------------------------------------
  type "%SALIDA%"
  echo   ------------------------------------------------------------
  set /a FALLAS+=1
)
popd
exit /b

rem ===============================================================
:alerta
cls
echo  Tene HORUS.bat corriendo y el panel abierto para verla llegar.
echo.
echo  Escenarios:  incendio   agresion_arma   caida_pose   merodeo
echo.
set ESC=
set /p ESC=  Cual (Enter = incendio):
if "%ESC%"=="" set ESC=incendio
pushd FASTAPI
python enviar_alerta_prueba.py %ESC%
popd
echo.
pause
goto menu

rem ===============================================================
:chequeo
cls
python chequeo_pruebas.py
echo.
pause
goto menu

rem ===============================================================
:fotos
cls
echo  Pasa el detector de objetos por 40 fotos tuyas y te las deja
echo  con las cajas dibujadas, para que veas con tus ojos que detecta
echo  y que se pierde.
echo.
echo  Ejemplo:  C:\Users\terra\Pictures\camaras
echo.
set CARPETA=
set /p CARPETA=  Carpeta con fotos:
if "%CARPETA%"=="" goto menu
if not exist "horus\04_cabezas\objetos\modelos\head_best_solo.pt" (
  echo.
  echo  FALTA horus\04_cabezas\objetos\modelos\head_best_solo.pt
  echo.
  pause
  goto menu
)
pushd horus\04_cabezas\objetos
echo.
echo  En CPU tarda un rato. Paciencia.
python objects_engine.py "%CARPETA%" --imagenes --n 40 --precision fp32 --pesos modelos\head_best_solo.pt --salida revision
popd
echo.
echo  Listo. Las fotos con cajas quedaron en:
echo     %~dp0horus\04_cabezas\objetos\revision\
echo.
start "" "%~dp0horus\04_cabezas\objetos\revision"
pause
goto menu

rem ===============================================================
:mediapipe
cls
echo ==============================================================
echo  Dejar lista la cabeza de caidas
echo ==============================================================
echo.
echo  Le faltan dos cosas, no una:
echo.
echo    mediapipe             el paquete de python
echo    pose_landmarker.task  el modelo de pose de Google
echo.
echo  Lo segundo sorprende: MediaPipe 0.10 SACO la API vieja, que traia
echo  el modelo adentro del paquete. Con la nueva hay que bajarlo
echo  aparte, y el error no lo dice. Por eso `pip install mediapipe`
echo  solo no alcanzaba.
echo.
echo  Esto hace las dos, y despues prueba que carguen de verdad.
echo.
python bin\instalar_caidas.py
echo.
pause
goto menu

rem ===============================================================
:topologia
cls
echo  ZONAS
echo.
echo  Mientras topologia.json no tenga una zona 'restringida', la regla
echo  de INTRUSION se declara DORMIDA: no dice "no entro nadie", dice
echo  "no puedo mirar esto". Merodeo no depende de esto, anda igual.
echo.
echo  Una zona restringida es "aca no se puede estar a esta hora", y eso
echo  no lo puede adivinar el programa: lo tenes que marcar vos sobre tu
echo  propio piso. Por eso este paso existe.
echo.
if not exist "horus\06_fusion_decision\topologia.json" (
  echo  Primero el archivo base, con el tamano de tu cuadro.
  echo.
  pushd horus\06_fusion_decision
  python crear_topologia.py
  popd
  echo.
)
echo  Ahora la zona. Se abre tu camara, haces clic en las esquinas del
echo  area prohibida y elegis el horario en que SI se puede estar.
echo.
echo  OJO: en Windows la camara es de un proceso por vez. Cerra el panel
echo  y los modelos antes, o la camara no va a abrir.
echo.
set /p DIBUJAR=  Dibujarla ahora? [S/n]: 
if /i "%DIBUJAR%"=="n" goto menu
echo.
python bin\dibujar_zona.py
if errorlevel 2 echo.
if errorlevel 2 echo  Quedo a medias: la zona esta escrita pero no alerta. Mira arriba.
echo.
pause
goto menu

rem ===============================================================
:adelgazar
cls
echo  Saca el estado de AdamW y los pesos crudos, que solo sirven para
echo  RETOMAR un entrenamiento. Los 20 tensores irreconstruibles quedan
echo  adentro, asi que el .pt sigue siendo autocontenido.
echo     93 MB  ->  33 MB
echo.
if not exist "horus\04_cabezas\objetos\modelos\head_best_solo.pt" (
  echo  FALTA modelos\head_best_solo.pt
  echo.
  pause
  goto menu
)
pushd horus\04_cabezas\objetos
if not exist "checkpoints" mkdir checkpoints
copy /y "modelos\head_best_solo.pt" "checkpoints\head_best_solo_completo.pt" >nul
python exportar_objetos.py --adelgazar "modelos\head_best_solo.pt"
if exist "modelos\head_best_solo_deploy.pt" (
  move /y "modelos\head_best_solo_deploy.pt" "modelos\head_best_solo.pt" >nul
  echo.
  echo  Listo. El completo quedo en checkpoints\head_best_solo_completo.pt
)
popd
echo.
pause
goto menu

rem ===============================================================
:subir
cls
if exist ".git\index.lock" del /f /q ".git\index.lock"
echo  Commits que todavia no estan en GitHub:
echo.
git log --oneline origin/Models..HEAD
echo.
git push origin Models
echo.
git status --short --branch
echo.
pause
goto menu

rem ===============================================================
:queve
cls
echo  Tene los modelos corriendo ^(opcion M, o HORUS.bat^).
echo.
python bin\que_ve.py
echo.
pause
goto menu

rem ===============================================================
:modelos
cls
echo ==============================================================
echo  El servicio de modelos, EN ESTA VENTANA
echo ==============================================================
echo.
echo  Sin `start`, sin ventana aparte, sin nada en el medio: si algo
echo  falla, el error aparece aca abajo y no se va a ningun lado.
echo.
echo  Tarda entre 15 y 40 segundos en cargar. Cuando diga
echo  "modelos listos", el panel tiene que pasar a "Analizando N".
echo.
echo  Ctrl+C para cortar.
echo ==============================================================
echo.

set FLAGS=
if exist "horus\04_cabezas\objetos\modelos\head_best_solo.pt" (
  set FLAGS=!FLAGS! --pesos ..\04_cabezas\objetos\modelos\head_best_solo.pt
)
if exist "horus\04_cabezas\segmentacion\checkpoints\head_v4_produccion.pt" (
  set FLAGS=!FLAGS! --segmentacion
)
if exist "horus\fight\checkpoints\modelo_fight.pt" (
  set FLAGS=!FLAGS! --agresion
)
if exist "horus\fall\checkpoints\modelo_stgcn.pt" (
  python -c "import mediapipe" >nul 2>&1
  if errorlevel 1 (
    echo  [caidas apagada: falta mediapipe, opcion 5]
  ) else (
    rem Sin --caidas-checkpoint: el default de ConfigCaidas es
    rem modelo_demo_todo.pt, que es EL de despliegue ("entrenado con
    rem todo"). Aca decia modelo_stgcn.pt, que es otro archivo y no
    rem figura como opcion de despliegue en ningun lado.
    set FLAGS=!FLAGS! --caidas
  )
)
if exist "horus\06_fusion_decision\topologia.json" (
  set FLAGS=!FLAGS! --topologia ..\06_fusion_decision\topologia.json
)
if "!FLAGS!"=="" (
  echo  No encontre ningun peso. Mira la opcion 3.
  echo.
  pause
  goto menu
)

echo  python servicio.py!FLAGS!
echo.
rem Tambien al log: si arrancas los modelos por aca, la opcion L tiene que
rem mostrar ESTO y no lo de la corrida anterior. Un log viejo que parece
rem nuevo es peor que no tener log.
pushd horus\00_servicio
python -u servicio.py!FLAGS! 2>&1 | python -u "%~dp0bin\tee.py" "%~dp0logs\modelos.txt"
popd
echo.
echo ==============================================================
echo  El servicio termino. Si fue por un error, esta justo arriba.
echo ==============================================================
echo.
pause
goto menu

rem ===============================================================
:logs
cls
echo ==============================================================
echo  Lo ultimo que dijeron el backend y los modelos
echo ==============================================================
echo.
if not exist "logs\backend.txt" if not exist "logs\modelos.txt" (
  echo  Todavia no hay logs. Se escriben cuando arrancas con HORUS.bat.
  echo.
  pause
  goto menu
)
rem De cuando es cada archivo. Sin esto, mirar un log de hace tres horas
rem creyendo que es de recien manda a buscar un problema que ya no existe —
rem justo lo que paso hoy.
for %%F in ("logs\backend.txt") do if exist "%%F" echo  backend.txt: %%~tF
for %%F in ("logs\modelos.txt") do if exist "%%F" echo  modelos.txt: %%~tF
echo  ahora:       %date% %time:~0,5%
echo.
echo  --- BACKEND  ^(ultimas 25 lineas de logs\backend.txt^) ---------
echo.
if exist "logs\backend.txt" (
  powershell -NoProfile -Command "Get-Content 'logsackend.txt' -Tail 25"
) else (
  echo  ^(no hay^)
)
echo.
echo  --- MODELOS  ^(ultimas 25 lineas de logs\modelos.txt^) ---------
echo.
if exist "logs\modelos.txt" (
  powershell -NoProfile -Command "Get-Content 'logs\modelos.txt' -Tail 25"
) else (
  echo  ^(no hay^)
)
echo.
echo ==============================================================
echo  Los archivos completos estan en la carpeta logs\
echo ==============================================================
echo.
pause
goto menu

rem ===============================================================
:camaras
cls
echo  Prueba cada indice con cada backend y pide una imagen de verdad.
echo  Si alguna otra cosa tiene la camara agarrada (Zoom, Discord, la app
echo  Camara de Windows, o la ventana "Horus modelos"), cerrala primero.
echo.
pushd FASTAPI
python diagnostico_camaras.py
popd
echo.
pause
goto menu

rem ===============================================================
:compactar
cls
echo ==============================================================
echo  Compactar .git
echo ==============================================================
echo.
echo  De los 2,9 GB que ocupa .git, 2,37 GB son objetos sueltos sin
echo  empaquetar. Esto los comprime.
echo.
echo  NO toca el historial, ni las ramas, ni los stashes, ni nada de
echo  lo que tenes commiteado. Es solo como estan guardados los
echo  archivos adentro de .git.
echo.
echo  Tarda varios minutos y usa bastante CPU. No cierres la ventana.
echo.
set SEGUIR=
set /p SEGUIR=  Dale (s/n):
if /i not "%SEGUIR%"=="s" goto menu
echo.
echo  Antes:
git count-objects -vH ^| findstr /R "^size"
echo.
echo  Trabajando...
git gc
echo.
echo  Despues:
git count-objects -vH ^| findstr /R "^size"
echo.
echo  Si queres achicarlo MUCHO mas hay que reescribir el historial,
echo  porque adentro quedaron backend.exe de 70 MB de commits viejos.
echo  Eso rompe el clon de cualquiera que ya tenga el repo, asi que
echo  es una decision de a dos, no un boton.
echo.
pause
goto menu

rem ===============================================================
:backend
cls
echo  Backend solo, en http://127.0.0.1:8000
echo  Probalo a mano en http://127.0.0.1:8000/docs
echo.
echo  Ctrl+C para cortar.
echo.
pushd FASTAPI
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
popd
pause
goto menu
