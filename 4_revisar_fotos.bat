@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Uso:  4_revisar_fotos.bat ^<carpeta con fotos^>
  echo.
  echo Ejemplo:  4_revisar_fotos.bat C:\Users\terra\Pictures\camaras
  echo.
  echo Deja las fotos con las cajas dibujadas en horus\04_cabezas\objetos\revision\
  pause
  exit /b
)
set PESOS=%~dp0horus\04_cabezas\objetos\modelos\head_best_solo.pt
if not exist "%PESOS%" (
  echo FALTA el modelo:  %PESOS%
  echo Bajalo de la salida del notebook de rescate en Kaggle.
  pause
  exit /b
)
cd horus\04_cabezas\objetos
echo Revisando 40 fotos de "%~1" en CPU. Tarda un rato.
python objects_engine.py "%~1" --imagenes --n 40 --precision fp32 --pesos "%PESOS%" --salida revision
echo.
echo Listo. Mira las imagenes en:  %~dp0horus\04_cabezas\objetos\revision\
pause
