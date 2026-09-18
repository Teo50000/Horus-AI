@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo ==============================================================
echo  HORUS - las 9 suites
echo ==============================================================
echo.
set FALLAS=0
set SALIDA=%TEMP%\horus_suite.txt

call :suite "horus\03_backbone"             probar_huella_backbone.py
call :suite "horus\04_cabezas\segmentacion" probar_segmentation_engine.py
call :suite "horus\04_cabezas\objetos"      probar_tope_y_guarda.py
call :suite "horus\05_tracking"             probar_tracking.py
call :suite "horus\06_fusion_decision"      probar_fusion.py
call :suite "horus\07_alerta"               probar_alerta.py
call :suite "horus\fight"                   probar_detector_agresion.py
call :suite "horus\fall"                    probar_detector_caidas.py
call :suite "FASTAPI"                       probar_alertas_backend.py

echo ==============================================================
if !FALLAS!==0 (
  echo   TODO VERDE
) else (
  echo   !FALLAS! suite^(s^) con fallas
)
echo ==============================================================
echo.
pause
exit /b

:suite
pushd "%~1"
python "%~2" > "%SALIDA%" 2>&1
set RC=!ERRORLEVEL!
set RESUMEN=
for /f "usebackq delims=" %%L in (`findstr /R /V "^$" "%SALIDA%"`) do set RESUMEN=%%L
if !RC!==0 (
  echo   OK     %~2   !RESUMEN!
) else (
  echo   FALLA  %~2   ^(codigo !RC!^)
  echo   ------------------------------------------------------------
  type "%SALIDA%"
  echo   ------------------------------------------------------------
  set /a FALLAS+=1
)
popd
exit /b
