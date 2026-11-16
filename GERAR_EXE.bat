@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Gerando o Mobilia Studio (.exe)... isso leva uns 5-10 minutos.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "packaging\build_windows.ps1"
if errorlevel 1 (
  echo.
  echo *** Deu erro. Tire um print desta janela e mande para o Claude. ***
) else (
  echo.
  echo Pronto! Abrindo a pasta com o programa...
  start "" "dist"
)
echo.
pause
