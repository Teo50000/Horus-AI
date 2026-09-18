@echo off
cd /d "%~dp0\horus\04_cabezas\objetos"
echo ==============================================================
echo  Adelgazar head_best_solo.pt para que entre en git
echo.
echo  Saca el estado de AdamW y los pesos crudos, que solo sirven
echo  para RETOMAR un entrenamiento. Deja los 20 tensores
echo  irreconstruibles adentro, asi que el .pt sigue siendo
echo  autocontenido.
echo.
echo     93 MB  ->  33 MB
echo.
echo  El original queda en checkpoints\ por si algun dia queres
echo  retomar el entrenamiento desde esa epoca.
echo ==============================================================
echo.
if not exist "modelos\head_best_solo.pt" (
  echo FALTA modelos\head_best_solo.pt
  pause
  exit /b
)
if not exist "checkpoints" mkdir checkpoints
copy /y "modelos\head_best_solo.pt" "checkpoints\head_best_solo_completo.pt" >nul
python exportar_objetos.py --adelgazar "modelos\head_best_solo.pt"
if exist "modelos\head_best_solo_deploy.pt" (
  move /y "modelos\head_best_solo_deploy.pt" "modelos\head_best_solo.pt" >nul
  echo.
  echo Listo. modelos\head_best_solo.pt ahora pesa:
  for %%F in ("modelos\head_best_solo.pt") do echo    %%~zF bytes
  echo El completo quedo en checkpoints\head_best_solo_completo.pt
)
echo.
pause
