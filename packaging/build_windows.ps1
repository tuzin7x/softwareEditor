# Gera o .exe (PyInstaller) e, se o Inno Setup estiver instalado, o instalador.
# Uso (na pasta do projeto):  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# 1) Python 3.13 (o mesmo usado no CI). "py -3.13" vem com o instalador oficial do python.org.
$py = $null
try { & py -3.13 -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @("py", "-3.13") } } catch {}
if (-not $py) {
    try {
        $v = (& python -c "import sys; print('%d.%d' % sys.version_info[:2])") 2>$null
        if ($v -eq "3.13") { $py = @("python") }
    } catch {}
}
if (-not $py) { throw "Python 3.13 não encontrado. Instale em https://www.python.org/downloads/ (marque 'Add python.exe to PATH')." }
$exe = $py[0]; $pyArgs = @($py | Select-Object -Skip 1)
Write-Host "Usando Python:" (& $exe @pyArgs -c "import sys; print(sys.version)")

# 2) Ambiente isolado só para o build (não mexe no Python do sistema).
if (-not (Test-Path ".venv-build\Scripts\python.exe")) { & $exe @pyArgs -m venv .venv-build }
$vpy = ".venv-build\Scripts\python.exe"
& $vpy -m pip install --upgrade pip
& $vpy -m pip install -r requirements\ui.txt
if ($LASTEXITCODE -ne 0) { throw "Falha ao instalar as dependências da interface" }

# 3) Executável (pasta dist\MobiliaStudio).
& $vpy -m PyInstaller --noconfirm --distpath dist --workpath build packaging\mobilia_studio.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller falhou" }

# 4) Autoteste. O .exe é um app de janela, então é preciso esperar o processo terminar.
$env:QT_QPA_PLATFORM = "offscreen"
$p = Start-Process -FilePath "dist\MobiliaStudio\MobiliaStudio.exe" -ArgumentList "--selftest" -Wait -PassThru -NoNewWindow
Remove-Item Env:QT_QPA_PLATFORM
if ($p.ExitCode -ne 0) { throw "Autoteste do executável falhou (código $($p.ExitCode))" }
Write-Host "Autoteste OK: dist\MobiliaStudio\MobiliaStudio.exe"

# 5) Instalador (opcional).
$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (Test-Path $iscc) {
    & $iscc packaging\installer.iss
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup falhou" }
    Write-Host "Pronto: instalador em dist\MobiliaStudio-Setup-*.exe"
} else {
    Write-Host "Inno Setup 6 não encontrado: instalador não gerado (o programa já roda por dist\MobiliaStudio\MobiliaStudio.exe)."
    Write-Host "Para gerar o instalador, instale o Inno Setup 6 (https://jrsoftware.org/isdl.php) e rode este script de novo."
}
