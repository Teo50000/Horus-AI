@echo off
chcp 65001 >nul
REM ====================================================================
REM  PASO 2 - El camino caliente entero sobre la camara.
REM
REM    camara -> objetos -> tracking local -> tracking global -> fusion
REM
REM  Dibuja los tracks y, arriba, los eventos abiertos. Los eventos que
REM  se van abriendo tambien salen por consola.
REM
REM  SIN topologia.json solo pueden dispararse incendio, arma, merodeo
REM  (sin zona) y paquete abandonado. Intrusion y merodeo por zona
REM  necesitan las zonas: una persona no esta "en la camara 2", esta en
REM  el deposito.
REM
REM  Para armarla: copiar topologia.example.json como topologia.json y
REM  dibujar los poligonos en pixeles del frame original. Verificar con
REM  el paso 3.
REM ====================================================================
cd /d "%~dp0"

set PESOS=..\04_cabezas\objetos\modelos\head_best_solo.pt
set BACKBONE=..\04_cabezas\objetos\checkpoints_v2\backbone.pt

if not exist "%PESOS%" (
  echo  ERROR: no encuentro %PESOS%
  pause
  exit /b 1
)

set EXTRA=
if exist "topologia.json" (
  set EXTRA=--topologia topologia.json
) else (
  echo  AVISO: no hay topologia.json. Intrusion y merodeo por zona no
  echo         van a evaluarse. Copia topologia.example.json y editalo.
  echo.
)
if exist "%BACKBONE%" set EXTRA=%EXTRA% --pesos-backbone "%BACKBONE%"

python pipeline.py 0 --ver --pesos "%PESOS%" --camara cam-0 --fps 10 %EXTRA%

echo.
pause
