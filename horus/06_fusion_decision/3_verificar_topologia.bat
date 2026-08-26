@echo off
chcp 65001 >nul
REM ====================================================================
REM  PASO 3 - Revisar topologia.json antes de confiar en el.
REM
REM  Una topologia mal escrita no rompe nada: degrada en silencio, que
REM  es peor. Esto avisa de vecinos inexistentes, camaras sin zonas,
REM  poligonos de menos de 3 puntos y zonas restringidas sin horario
REM  (que van a alertar las 24 horas, correcto solo para una boveda).
REM ====================================================================
cd /d "%~dp0"

if not exist "topologia.json" (
  echo  No hay topologia.json todavia. Copiando el ejemplo...
  copy /y topologia.example.json topologia.json >nul
  echo  Listo: edita topologia.json con las zonas reales y volve a correr esto.
  echo.
)

python -c "from topologia import Topologia; t=Topologia.cargar('topologia.json'); print(t.resumen()); print(); avisos=t.verificar(); print('Sin avisos.' if not avisos else 'Avisos:'); [print('  -', a) for a in avisos]"

echo.
pause
