@echo off
REM ===================================================================
REM  HORUS - preparar el dataset de la cabeza de objetos
REM  Doble clic y seguir las instrucciones.
REM ===================================================================
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   HORUS - preparacion del dataset de objetos
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

echo [1/4] Instalando dependencias que falten...
echo       (la primera vez tarda: fiftyone y datasets son pesados)
echo.
python bajar_datasets.py --instalar-deps
if errorlevel 1 (
    echo.
    echo [ERROR] Fallo la instalacion. Proba correr a mano:
    echo         python -m pip install kaggle datasets gdown roboflow fiftyone opencv-python
    echo.
    pause
    exit /b 1
)

echo.
echo [2/4] Chequeando que se puede bajar...
echo.
python bajar_datasets.py --verificar --solo-comerciales

echo.
echo ============================================================
echo   LEE LA TABLA DE ARRIBA
echo ============================================================
echo   Las fuentes marcadas -- no se van a bajar.
echo.
echo   Para las credenciales que falten:
echo     ROBOFLOW_API_KEY  -^> https://app.roboflow.com/settings/api
echo                          setx ROBOFLOW_API_KEY "tu_clave"
echo                          (cerra y abri esta ventana despues)
echo     Kaggle            -^> https://www.kaggle.com/settings
echo                          bajar kaggle.json a %%USERPROFILE%%\.kaggle\
echo.
echo   Podes seguir igual: se baja lo que este listo.
echo ============================================================
echo.
pause

echo.
echo [3/4] Descargando y normalizando... (esto tarda, son varios GB)
echo.
python bajar_datasets.py --todo --solo-comerciales
if errorlevel 1 (
    echo.
    echo [ERROR] Algo fallo. Revisa los mensajes de arriba.
    pause
    exit /b 1
)

echo.
echo [4/4] Listo. El dataset quedo en datasets\mezcla_v1
echo.
echo   Para entrenar:
echo     python entrenar_objetos_cuda.py --dataset datasets/mezcla_v1 --batch 16
echo.
pause
