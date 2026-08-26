@echo off
chcp 65001 >nul
REM ====================================================================
REM  PASO 1 - Verificar la capa de fusion SIN GPU y SIN modelos.
REM
REM  20 escenarios sinteticos, unos 2 segundos. La fusion no mira
REM  pixeles, mira tracks: se le fabrican los tracks a mano y se
REM  verifica que decida bien. Un escenario que en la vida real tarda
REM  40 segundos aca tarda milisegundos, y es repetible.
REM
REM  Incluye los casos que NO tienen que disparar, que son los que mas
REM  importan: persona en horario laboral, persona de paso, paquete con
REM  dueno al lado, alguien que se agacha, y escena vacia un minuto.
REM
REM  Para ver cada evento generado:  python probar_fusion.py --detalle
REM  Para uno solo:                  python probar_fusion.py --caso robo
REM  Lista de casos:                 python probar_fusion.py --listar
REM ====================================================================
cd /d "%~dp0"

python probar_fusion.py

echo.
pause
