@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0\FASTAPI"
echo ==============================================================
echo  Backend de Horus en http://127.0.0.1:8000
echo.
echo  Abri en el navegador:  http://127.0.0.1:8000/docs
echo  Ahi podes probar los 18 endpoints a mano, incluido POST /alertas
echo.
echo  Ctrl+C para cortar.
echo ==============================================================
echo.
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
pause
