@echo off
rem Lanzador del backend. Existe para que la salida quede ADEMAS en un archivo.
rem
rem Sin esto, cuando algo se cae no queda rastro: la ventana se cierra, o el
rem texto se va del buffer de la consola y no hay a donde volver a mirar.
rem bin\tee.py lo muestra en pantalla Y lo escribe, vaciando cada linea.
rem
rem `python -u` es clave: sin el, python bufferea la salida y el archivo queda
rem vacio justo cuando mas lo necesitas — cuando el proceso murio sin llegar a
rem vaciar el buffer.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set RAIZ=%~dp0..
if not exist "%RAIZ%\logs" mkdir "%RAIZ%\logs"
cd /d "%RAIZ%\FASTAPI"

rem `bin\tee.py` y no `Tee-Object`: PowerShell bufferea y el archivo
rem quedaba minutos atrasado, a veces vacio hasta que el proceso moria.
rem Un log que no se refresca no sirve para mirar por que algo no anda.
python -u -m uvicorn src.main:app --host 127.0.0.1 --port 8000 2>&1 | python -u "%RAIZ%\bin\tee.py" "%RAIZ%\logs\backend.txt"

echo.
echo ==============================================================
echo  EL BACKEND TERMINO.
echo.
echo  Si no lo cortaste vos con Ctrl+C, arriba esta el motivo, y
echo  tambien quedo en:
echo     logs\backend.txt
echo ==============================================================
echo.
pause
