@echo off
REM ===================================================================
REM  destrabar_y_verificar.bat
REM
REM  1) Borra .git\index.lock, que traba todo git desde el 27/08.
REM  2) Rearma horus\fight\checkpoints\modelo_fight.pt desde sus partes.
REM  3) Corre las cuatro suites que se pueden correr en este checkout.
REM
REM  No commitea nada: los comandos quedan impresos al final.
REM ===================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ===================================================================
echo  HORUS - destrabar y verificar
echo ===================================================================
echo.

REM ---------------------------------------------------------------- 1
echo [1/3] .git\index.lock
if exist ".git\index.lock" (
    del /f /q ".git\index.lock"
    if exist ".git\index.lock" (
        echo        ERROR: no se pudo borrar. Cerra GitHub Desktop y reintenta.
    ) else (
        echo        OK - borrado, git vuelve a funcionar.
    )
) else (
    echo        no estaba: git ya estaba libre.
)
echo.

REM ---------------------------------------------------------------- 2
echo [2/3] checkpoint de agresion
set "CK=horus\fight\checkpoints"
if exist "%CK%\modelo_fight.pt" (
    echo        ya estaba armado.
) else (
    if not exist "%CK%\modelo_fight.pt.000" (
        echo        ERROR: no encuentro las partes en %CK%
        goto :suites
    )
    pushd "%CK%"
    copy /b modelo_fight.pt.000+modelo_fight.pt.001+modelo_fight.pt.002 modelo_fight.pt >nul
    for %%A in (modelo_fight.pt) do set "TAM=%%~zA"
    if "!TAM!"=="46046004" (
        echo        OK - !TAM! bytes.
        del modelo_fight.pt.000 modelo_fight.pt.001 modelo_fight.pt.002
        echo        partes borradas.
    ) else (
        echo        ERROR: quedo en !TAM! bytes, tendria que ser 46046004.
        echo        No borro las partes.
    )
    popd
)
echo.

REM ---------------------------------------------------------------- 3
:suites
echo [3/3] suites
where python >nul 2>&1
if errorlevel 1 (
    echo        No encuentro python en el PATH. Salteo las suites.
    goto :fin
)
echo.
call :correr "horus\06_fusion_decision" "probar_fusion.py"          "fusion"
call :correr "horus\fight"              "probar_detector_agresion.py" "agresion"
call :correr "horus\07_alerta"          "probar_alerta.py"          "emisor"
call :correr "horus\05_tracking"        "probar_tracking.py --autotest" "tracking"
echo.
echo   (FASTAPI\probar_alertas_backend.py no corre en este checkout:
echo    FASTAPI\src\ solo tiene los dos archivos nuevos. El resto vive en
echo    la rama main. Ver FASTAPI\alertas-backend-2026-09-15.patch)

:fin
echo.
echo ===================================================================
echo  Para commitear, cuando estes conforme:
echo.
echo     git status
echo     git add -A
echo     git commit -m "agresion integrada + fusion refinada + emisor y endpoint"
echo ===================================================================
echo.
pause
exit /b

REM ---------------------------------------------------------------- sub
:correr
pushd %~1
echo   --- %~3 ---------------------------------------------------------
python %~2
if errorlevel 1 (echo   ^>^> %~3: FALLO) else (echo   ^>^> %~3: OK)
popd
echo.
exit /b
