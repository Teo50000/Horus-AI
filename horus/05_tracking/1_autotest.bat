@echo off
chcp 65001 >nul
REM ====================================================================
REM  PASO 1 - Verificar el tracking SIN GPU, SIN pesos y SIN camara.
REM
REM  11 pruebas, unos 5 segundos. Verifica el hungaro contra scipy, que
REM  una persona rapida no cambie de ID, que dos que se cruzan no se
REM  intercambien, que un fantasma que parpadea nunca se confirme, la
REM  oclusion, el ReID entre camaras, el filtro de calidad, la latencia
REM  del matching y la carga de CPU.
REM
REM  Si esto pasa, el tracking esta sano. Lo que falle despues en la
REM  camara es del modelo o de los umbrales, no del tracker.
REM ====================================================================
cd /d "%~dp0"

python probar_tracking.py --autotest

echo.
pause
