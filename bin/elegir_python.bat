@echo off
rem ===============================================================
rem  Deja en %HORUS_PY% el python con el que corre Horus. Se usa con
rem      call "%~dp0bin\elegir_python.bat"
rem  y despues %HORUS_PY% en vez de `python`.
rem
rem  `python` a secas es el primero del PATH, y nada garantiza que
rem  sea el que tiene torch y mediapipe: en la PC de desarrollo era
rem  un 3.14 vacio, y el lanzador decia [NO] caidas con todo
rem  instalado en el 3.12 de al lado. Aca se prueba cada python que
rem  conoce el lanzador `py` y gana el que tiene mas de lo que hace
rem  falta (bin\puntaje_python.py). A igualdad, el orden de abajo:
rem  3.12 primero porque es el que pide el LEEME.
rem
rem  HORUS_PYTHON fuerza uno, si alguna vez hace falta:
rem      set HORUS_PYTHON="C:\ruta\python.exe"
rem
rem  Si ya hay %HORUS_PY% (lo heredo de HORUS.bat) no se vuelve a elegir:
rem  modelos, backend y panel tienen que correr con el mismo.
rem ===============================================================
if defined HORUS_PY goto :eof
if defined HORUS_PYTHON (
  set "HORUS_PY=%HORUS_PYTHON%"
  goto :eof
)
set "HORUS_PY="
set MEJOR_PY=99
for %%V in (3.12 3.11 3.10 3.13 3.14) do call :probar "py -%%V"
call :probar python
if not defined HORUS_PY set "HORUS_PY=python"
set MEJOR_PY=
set R_PY=
goto :eof

:probar
rem 9009 = el comando no existe; py devuelve 103 si no tiene esa
rem version. Todo lo que pase de 50 es "este no", no un puntaje.
%~1 "%~dp0puntaje_python.py" >nul 2>&1
set R_PY=%errorlevel%
if %R_PY% GEQ 50 goto :eof
if %R_PY% LSS %MEJOR_PY% (
  set "HORUS_PY=%~1"
  set MEJOR_PY=%R_PY%
)
goto :eof
