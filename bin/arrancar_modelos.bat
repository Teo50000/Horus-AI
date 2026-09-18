@echo off
rem Lanzador del servicio de modelos. Ver la nota en arrancar_backend.bat.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set RAIZ=%~dp0..
if not exist "%RAIZ%\logs" mkdir "%RAIZ%\logs"
cd /d "%RAIZ%\horus\00_servicio"

where powershell >nul 2>&1
if errorlevel 1 (
  echo La salida va a logs\modelos.txt
  python -u servicio.py %* > "%RAIZ%\logs\modelos.txt" 2>&1
) else (
  python -u servicio.py %* 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%RAIZ%\logs\modelos.txt' -Encoding utf8"
)

echo.
echo ==============================================================
echo  EL SERVICIO DE MODELOS TERMINO.
echo.
echo  Si el panel dice "MODELOS APAGADOS", es por esto. El motivo
echo  esta arriba y en:
echo     logs\modelos.txt
echo ==============================================================
echo.
pause
