# Gera o .exe (PyInstaller) e o instalador (Inno Setup) no Windows.
# Uso:  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m pip install -r requirements\ui.txt
python -m PyInstaller --noconfirm --distpath dist --workpath build packaging\mobilia_studio.spec
& dist\MobiliaStudio\MobiliaStudio.exe --selftest
if ($LASTEXITCODE -ne 0) { throw "Autoteste do executável falhou" }
$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
& $iscc packaging\installer.iss
Write-Host "Pronto: dist\MobiliaStudio-Setup-*.exe"
