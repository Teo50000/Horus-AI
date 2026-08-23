@echo off
REM ===================================================================
REM  HORUS - preparar el dataset de la cabeza de segmentacion
REM  Clases: 0 fondo  1 agua  2 humo  3 fuego
REM  Doble clic y seguir las instrucciones.
REM ===================================================================
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   HORUS - preparacion del dataset de segmentacion
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No encuentro python en el PATH.
    echo         Instalalo de https://www.python.org/downloads/
    echo         y marca "Add Python to PATH" en el instalador.
    echo.
    pause
    exit /b 1
)

python -c "import torch" >nul 2>nul
if errorlevel 1 (
    echo [AVISO] Este python no tiene torch. Vas a poder armar el dataset,
    echo         pero no entrenar. Para instalarlo:
    echo           python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    echo.
)

echo [1/5] Instalando dependencias que falten...
echo.
python descargar_datasets_segmentacion.py --instalar-deps
if errorlevel 1 (
    echo.
    echo [ERROR] Fallo la instalacion. Proba a mano:
    echo         python -m pip install kaggle huggingface_hub gdown roboflow pycocotools
    echo.
    pause
    exit /b 1
)

echo.
echo [2/5] Revisando que fuentes estan listas...
echo.
python descargar_datasets_segmentacion.py --estado

echo.
echo ============================================================
echo   LEE LA TABLA DE ARRIBA
echo ============================================================
echo   Las marcadas NO no se van a bajar. Para sumarlas:
echo.
echo     Kaggle    -^> https://www.kaggle.com/settings
echo                  bajar kaggle.json a %%USERPROFILE%%\.kaggle\
echo     Roboflow  -^> https://app.roboflow.com/settings/api
echo                  setx ROBOFLOW_API_KEY "tu_clave"
echo                  (cerra y abri esta ventana despues)
echo.
echo   OJO: si "humo" o "fuego" no tienen ninguna fuente lista, el
echo   entrenamiento va a abortar. Roboflow es la unica fuente que
echo   anota humo Y fuego en la misma escena: vale la pena la key.
echo.
echo   Podes seguir igual: se baja lo que este listo.
echo ============================================================
echo.
pause

echo.
echo [3/5] Cuanto queres bajar?
echo.
echo    1 = Rapido      ~5 GB    una fuente por clase. Para probar el
echo                             camino completo hoy mismo.
echo    2 = Sin logins  ~23 GB   todo lo que no pide credenciales.
echo    3 = Todo        ~53 GB   las 17 fuentes. Es el que da el mejor
echo                             modelo. Pico de disco ~85 GB salvo que
echo                             uses la opcion de purgar (abajo).
echo.
set "OPCION="
set /p OPCION=Opcion [1/2/3]:
if "%OPCION%"=="" set "OPCION=1"

set "PURGAR="
if "%OPCION%"=="1" goto :sin_purga

echo.
echo    Borrar los zips y los arboles extraidos apenas se convierte cada
echo    fuente? Baja el pico de disco de ~85 GB a ~35 GB. Si despues
echo    queres rehacer una fuente, la vuelve a bajar.
echo.
set "RESP="
set /p RESP=Purgar crudos? [S/n]:
if /i not "%RESP%"=="n" set "PURGAR=--purgar-crudos"
:sin_purga

set "ARGS=--todo %PURGAR%"
if "%OPCION%"=="1" set "ARGS=--solo ade20k,bowfire,multinatsmoke,rf_fuego_humo --hf"
if "%OPCION%"=="2" set "ARGS=%PURGAR%"

echo.
echo [4/5] Descargando y convirtiendo... esto tarda.
echo       Se puede cortar con Ctrl-C y volver a correr: cada paso deja
echo       un .ok al lado y no se repite. Las descargas HTTP reanudan.
echo.
python descargar_datasets_segmentacion.py %ARGS%
if errorlevel 1 (
    echo.
    echo [ERROR] Algo fallo. Revisa los mensajes de arriba.
    pause
    exit /b 1
)

echo.
echo [5/5] Verificando el dataset armado...
echo.
python descargar_datasets_segmentacion.py --verificar
if errorlevel 1 (
    echo.
    echo [ERROR] El dataset armado tiene archivos rotos o inconsistentes.
    echo         Mira los mensajes [XX] de arriba.
    echo.
    pause
    exit /b 1
)

python -c "import torch" >nul 2>nul
if errorlevel 1 goto :sin_torch

python entrenar_segmentacion_cuda.py --revisar
if errorlevel 1 (
    echo.
    echo [ERROR] El dataset no paso la revision del entrenamiento. Mira los
    echo         mensajes [XX] de arriba y datasets\mezcla_seg_v1\reporte.md
    echo.
    pause
    exit /b 1
)
:sin_torch

echo.
echo ============================================================
echo   LISTO. El dataset quedo en datasets\mezcla_seg_v1
echo ============================================================
echo.
echo   Mira primero  datasets\mezcla_seg_v1\reporte.md
echo   Si alguna clase tiene menos de ~2000 imagenes, sumale otra
echo   fuente antes de entrenar.
echo.
echo   Para entrenar:
echo     python entrenar_segmentacion_cuda.py --epocas 30 --batch 16 --workers 6
echo.
pause
