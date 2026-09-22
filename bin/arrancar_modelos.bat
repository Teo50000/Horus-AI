@echo off
rem Lanzador del servicio de modelos. Ver la nota en arrancar_backend.bat.
rem
rem Los flags llegan por HORUS_FLAGS y no como argumentos: meterlos en la
rem linea del `start ... cmd /k "..."` anida comillas, cmd las come mal, y la
rem ventana ni se abre. `start` hereda el entorno, asi que esto no falla.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set RAIZ=%~dp0..
if not exist "%RAIZ%\logs" mkdir "%RAIZ%\logs"
cd /d "%RAIZ%\horus\00_servicio"

rem `bin\tee.py` y no `Tee-Object`: PowerShell bufferea y el archivo
rem quedaba minutos atrasado, a veces vacio hasta que el proceso moria.
rem Un log que no se refresca no sirve para mirar por que algo no anda.
python -u servicio.py %HORUS_FLAGS% 2>&1 | python -u "%RAIZ%\bin\tee.py" "%RAIZ%\logs\modelos.txt"

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
