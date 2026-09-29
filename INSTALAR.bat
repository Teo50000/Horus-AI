@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
title HORUS - instalar

rem ===============================================================
rem  Una sola vez, despues de clonar el repo. Instala todo lo que
rem  necesita Horus en esta PC y al final muestra que quedo y que no.
rem  Correrlo de nuevo no rompe nada: lo que ya esta, lo saltea.
rem
rem  Lo hace bin\instalar_todo.py; aca solo se chequea que haya un
rem  Python para correrlo.
rem ===============================================================

python --version >nul 2>&1
if errorlevel 1 (
  echo.
  echo  No encuentro Python.
  echo.
  echo  Bajalo de https://www.python.org/downloads/  ^(el 3.12 va bien^)
  echo  y en la PRIMERA pantalla del instalador marca
  echo  "Add python.exe to PATH". Despues cerra esta ventana y volve
  echo  a abrir INSTALAR.bat.
  echo.
  pause
  exit /b 1
)

echo ==============================================================
echo   HORUS - instalacion
echo ==============================================================
echo.
echo  Va a instalar: torch ^(con la placa de video si hay NVIDIA^),
echo  las dependencias de python, los pesos de los modelos, la
echo  cabeza de caidas y el panel. La primera vez tarda: torch con
echo  CUDA solo ya son unos 2,5 GB.
echo.
echo  Para el panel hace falta Node.js ^(https://nodejs.org, el LTS^).
echo.
pause

python bin\instalar_todo.py %*
echo.
pause
