@echo off
setlocal
cd /d "%~dp0\horus\06_fusion_decision"
if "%~1"=="" (
  echo Uso:  7_crear_topologia.bat ^<carpeta con fotos de la camara^>
  echo   o:  7_crear_topologia.bat --tam 1080 1920
  echo.
  echo Lee el tamano real de tus fotos y arranca un topologia.json.
  echo Deja intrusion dormida a proposito: esa zona la tenes que dibujar vos.
  echo.
  pause
  exit /b
)
python crear_topologia.py %*
echo.
pause
