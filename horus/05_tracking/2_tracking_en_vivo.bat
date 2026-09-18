@echo off
chcp 65001 >nul
REM ====================================================================
REM  PASO 2 - Tracking sobre la camara real.
REM
REM  Igual que 2_camara_en_vivo.bat de la cabeza de objetos, pero con el
REM  tracker encima. Cada caja lleva su #ID y los segundos que lleva
REM  viva; los objetos quietos dicen QUIETO.
REM
REM  QUE MIRAR:
REM   - Una persona caminando rapido tiene que conservar el MISMO #ID.
REM   - Paquete y celular fantasma no deberian llegar a dibujarse: el
REM     tracker solo muestra tracks confirmados (celular pide 0.8 s,
REM     paquete 1.0 s de presencia continua).
REM   - Si un objeto real tarda mucho en aparecer, bajale confirmar_s a
REM     esa clase en local/tracker_local.py (PARAMETROS_POR_CLASE).
REM   - q o ESC para salir, r para reiniciar los tracks.
REM
REM  OJO: aca NO se pasa --umbral-clase. No es un olvido. El tracker se
REM  queda con la tabla de umbrales por clase y el motor entrega todo
REM  desde 0.10 para arriba, porque la segunda etapa de asociacion
REM  necesita las detecciones debiles para sostener un track. Eso lo
REM  hace config_motor_para_tracking() solo.
REM ====================================================================
cd /d "%~dp0"

set PESOS=..\04_cabezas\objetos\modelos\head_best_solo.pt
set BACKBONE=..\04_cabezas\objetos\checkpoints_v2\backbone.pt

if not exist "%PESOS%" (
  echo  ERROR: no encuentro %PESOS%
  pause
  exit /b 1
)

REM  --pesos-backbone es opcional: sin el, el motor baja los pesos de
REM  ImageNet, que son los mismos con los que se entreno la cabeza. Se
REM  pasa igual para no depender de internet ni ver el aviso.
if exist "%BACKBONE%" (
  python probar_tracking.py 0 --ver --pesos "%PESOS%" --pesos-backbone "%BACKBONE%" --input-size 768 --fps 10
) else (
  python probar_tracking.py 0 --ver --pesos "%PESOS%" --input-size 768 --fps 10
)

echo.
pause
