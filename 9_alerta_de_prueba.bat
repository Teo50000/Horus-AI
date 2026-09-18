@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0FASTAPI"
echo Mandando una alerta de prueba al backend...
echo Tene el panel abierto en http://localhost:1420 para verla llegar.
echo.
python enviar_alerta_prueba.py %*
echo.
pause
