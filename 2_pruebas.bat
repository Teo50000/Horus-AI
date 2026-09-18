@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo ==============================================================
echo  HORUS - todas las suites
echo ==============================================================
echo.
set FALLAS=0

call :suite "horus\03_backbone"             probar_huella_backbone.py
call :suite "horus\04_cabezas\segmentacion" probar_segmentation_engine.py
call :suite "horus\04_cabezas\objetos"      probar_tope_y_guarda.py
call :suite "horus\05_tracking"             probar_tracking.py
call :suite "horus\06_fusion_decision"      probar_fusion.py
call :suite "horus\07_alerta"               probar_alerta.py
call :suite "horus\fight"                   probar_detector_agresion.py
call :suite "horus\fall"                    probar_detector_caidas.py
call :suite "FASTAPI"                       probar_alertas_backend.py

echo.
echo ==============================================================
if %FALLAS%==0 (echo  TODO VERDE) else (echo  %FALLAS% suite^(s^) con fallas - mirá el detalle arriba)
echo ==============================================================
echo.
pause
exit /b

:suite
pushd %~1
echo --- %~2
python %~2 2>&1 | findstr /C:"pruebas OK" /C:"escenarios OK" /C:"FALLA" /C:"Error" /C:"Traceback" /C:"/18" /C:"/12"
if errorlevel 1 set /a FALLAS+=1
popd
echo.
exit /b
