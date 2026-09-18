@echo off
rem Lanzador del panel. Ver la nota en arrancar_backend.bat.
chcp 65001 >nul
set RAIZ=%~dp0..
if not exist "%RAIZ%\logs" mkdir "%RAIZ%\logs"
cd /d "%RAIZ%\HorusAI"

where powershell >nul 2>&1
if errorlevel 1 (
  call npm run dev > "%RAIZ%\logs\panel.txt" 2>&1
) else (
  call npm run dev 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%RAIZ%\logs\panel.txt' -Encoding utf8"
)

echo.
echo ==============================================================
echo  EL PANEL TERMINO. El motivo esta arriba y en logs\panel.txt
echo ==============================================================
echo.
pause
