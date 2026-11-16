@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Gerando e instalando o 7xAi... isso leva uns 5-10 minutos.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "packaging\build_windows.ps1"
if errorlevel 1 (
  echo.
  echo *** Deu erro. Tire um print desta janela e mande para o Claude. ***
) else (
  echo.
  echo Pronto! O 7xAi foi instalado e o ícone está na sua Área de Trabalho.
)
echo.
pause
