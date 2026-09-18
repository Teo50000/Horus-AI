@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ==============================================================
echo  HORUS - backend + panel
echo ==============================================================
echo.
echo  Backend  http://127.0.0.1:8000        (/docs para probar a mano)
echo  Panel    http://localhost:1420
echo.
echo  Se abren DOS ventanas. Cerralas con Ctrl+C cada una.
echo ==============================================================
echo.
rem Sin soporte de WebSocket el backend arranca igual, el panel abre, se ven
rem las camaras... y no llega NI UNA alerta, sin ningun error a la vista.
python -c "import websockets" 2>nul
if errorlevel 1 (
  echo.
  echo  Falta el paquete "websockets". Sin el, el panel NO recibe alertas
  echo  y no te avisa: se ve igual que una noche tranquila. Lo instalo.
  echo.
  python -m pip install -r FASTAPI\requirements.txt
  echo.
)

if not exist "HorusAI\node_modules" (
  echo Falta HorusAI\node_modules. Instalando una vez...
  pushd HorusAI
  call npm install
  popd
)
start "Horus backend" cmd /k "chcp 65001 >nul && set PYTHONIOENCODING=utf-8 && cd /d "%~dp0FASTAPI" && python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 3 >nul
start "Horus panel" cmd /k "cd /d "%~dp0HorusAI" && npm run dev"
echo.
echo Listo. Cuando el panel diga "Local: http://localhost:1420", abrilo.
echo.
echo Para ver una alerta de verdad en el panel, con las dos ventanas
echo arriba corre en una tercera:
echo.
echo     9_alerta_de_prueba.bat
echo.
pause
