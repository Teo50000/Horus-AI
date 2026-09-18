@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ==============================================================
echo  Instalar mediapipe
echo.
echo  Hace falta SOLO para correr caidas EN VIVO sobre video: es
echo  quien saca los 33 puntos del esqueleto que come el ST-GCN.
echo  La suite de caidas NO lo necesita (inyecta poses de mentira),
echo  por eso te da 18/18 sin esto.
echo ==============================================================
echo.
python -m pip install mediapipe
echo.
echo --- verificacion ---
python -c "import mediapipe; print('mediapipe', mediapipe.__version__)"
echo.
echo Si fallo por permisos: tu python es el de la Microsoft Store, que
echo instala en una carpeta protegida. Probá:
echo    python -m pip install --user mediapipe
echo.
pause
