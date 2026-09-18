@echo off
rem Lanzador del backend. Existe para que la salida quede ADEMAS en un archivo.
rem
rem Sin esto, cuando algo se cae no queda rastro: la ventana se cierra, o el
rem texto se va del buffer de la consola y no hay a donde volver a mirar.
rem Tee-Object lo muestra en pantalla Y lo escribe. El -Encoding utf8
rem es necesario: por defecto escribe UTF-16, que deja un byte nulo
rem entre cada letra y hace el archivo ilegible en cualquier editor.
rem
rem `python -u` es clave: sin el, python bufferea la salida y el archivo queda
rem vacio justo cuando mas lo necesitas — cuando el proceso murio sin llegar a
rem vaciar el buffer.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set RAIZ=%~dp0..
if not exist "%RAIZ%\logs" mkdir "%RAIZ%\logs"
cd /d "%RAIZ%\FASTAPI"

where powershell >nul 2>&1
if errorlevel 1 (
  rem Sin powershell no hay tee: al menos que quede el archivo.
  echo La salida va a logs\backend.txt
  python -u -m uvicorn src.main:app --host 127.0.0.1 --port 8000 > "%RAIZ%\logs\backend.txt" 2>&1
) else (
  python -u -m uvicorn src.main:app --host 127.0.0.1 --port 8000 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%RAIZ%\logs\backend.txt' -Encoding utf8"
)

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
