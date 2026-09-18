@echo off
cd /d "%~dp0"
echo El commit ya esta hecho. Esto solo lo sube a GitHub.
echo.
if exist ".git\index.lock" del /f /q ".git\index.lock"
git log --oneline -1
echo.
git push origin Models
echo.
git status --short --branch
echo.
pause
