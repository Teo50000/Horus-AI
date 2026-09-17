@echo off
REM Commit + push de todo lo del 15/09 a la rama Models.
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "LOG=%~dp0push_log.txt"
> "%LOG%" echo HORUS - commit y push  %DATE% %TIME%

REM --- git ------------------------------------------------------------
set "GIT=git"
where git >nul 2>&1
if errorlevel 1 (
    for /d %%D in ("%LOCALAPPDATA%\GitHubDesktop\app-*") do (
        if exist "%%D\resources\app\git\cmd\git.exe" set "GIT=%%D\resources\app\git\cmd\git.exe"
    )
)
>>"%LOG%" echo git: !GIT!

REM --- limpiar lo descartable ------------------------------------------
>>"%LOG%" echo.
>>"%LOG%" echo == limpieza ==
for %%F in (verificar_log.bat resultado_suites.txt buscar_backbone.bat busqueda_backbone.txt buscar_backbone_D.bat busqueda_D.txt) do (
    if exist "%%F" ( del /f /q "%%F" & >>"%LOG%" echo    borrado %%F )
)
if exist ".git\index.lock" del /f /q ".git\index.lock"

REM --- estado -----------------------------------------------------------
>>"%LOG%" echo.
>>"%LOG%" echo == rama ==
"!GIT!" rev-parse --abbrev-ref HEAD >>"%LOG%" 2>&1
>>"%LOG%" echo.
>>"%LOG%" echo == lo que entra ==
"!GIT!" add -A >>"%LOG%" 2>&1
"!GIT!" status --short >>"%LOG%" 2>&1

REM --- commit -----------------------------------------------------------
>>"%LOG%" echo.
>>"%LOG%" echo == commit ==
"!GIT!" commit -F "%~dp0mensaje_commit.txt" >>"%LOG%" 2>&1
if errorlevel 1 >>"%LOG%" echo    (nada que commitear, o fallo)

REM --- push -------------------------------------------------------------
>>"%LOG%" echo.
>>"%LOG%" echo == push ==
"!GIT!" push origin HEAD >>"%LOG%" 2>&1
if errorlevel 1 (
    >>"%LOG%" echo    PUSH FALLO
) else (
    >>"%LOG%" echo    PUSH OK
)

>>"%LOG%" echo.
>>"%LOG%" echo == commit final ==
"!GIT!" log --oneline -1 >>"%LOG%" 2>&1
"!GIT!" status --short --branch >>"%LOG%" 2>&1
>>"%LOG%" echo ---- fin ----

echo Listo. Resultado en push_log.txt
timeout /t 3 >nul
