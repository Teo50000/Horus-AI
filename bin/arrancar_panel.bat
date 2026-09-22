@echo off
rem Lanzador del panel. Ver la nota en arrancar_backend.bat.
chcp 65001 >nul
set RAIZ=%~dp0..
if not exist "%RAIZ%\logs" mkdir "%RAIZ%\logs"
cd /d "%RAIZ%\HorusAI"

rem `bin\tee.py` y no `Tee-Object`: PowerShell bufferea y el archivo
rem quedaba minutos atrasado, a veces vacio hasta que el proceso moria.
rem Un log que no se refresca no sirve para mirar por que algo no anda.
call npm run dev 2>&1 | python -u "%RAIZ%\bin\tee.py" "%RAIZ%\logs\panel.txt"

echo.
echo ==============================================================
echo  EL PANEL TERMINO. El motivo esta arriba y en logs\panel.txt
echo ==============================================================
echo.
pause
