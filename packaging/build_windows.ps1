# Gera o 7xAi.exe (PyInstaller), testa, INSTALA para o usuário atual e cria o ícone na Área de Trabalho.
# Se o Inno Setup 6 estiver instalado, gera também o instalador (opcional, para levar a outros PCs).
# Uso (na pasta do projeto):  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
# Compatível com o Windows PowerShell 5.1 (o que vem no Windows). Arquivo em UTF-8 com BOM e CRLF.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$AppName = "7xAi"
$ExeName = "7xAi.exe"

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
$version = (& $vpy -c "import mobilia; print(mobilia.__version__)").Trim()

# 3) Executável (pasta dist\7xAi, programa dist\7xAi\7xAi.exe).
& $vpy -m PyInstaller --noconfirm --distpath dist --workpath build packaging\7xai.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller falhou" }
$distDir = (Resolve-Path "dist\7xAi").Path
# pasta da época do nome antigo (dist\MobiliaStudio): some para não confundir
if (Test-Path "dist\MobiliaStudio") { Remove-Item -LiteralPath "dist\MobiliaStudio" -Recurse -Force -ErrorAction SilentlyContinue }

# 4) Autoteste. O .exe é um app de janela, então é preciso esperar o processo terminar.
$env:QT_QPA_PLATFORM = "offscreen"
$p = Start-Process -FilePath (Join-Path $distDir $ExeName) -ArgumentList "--selftest" -Wait -PassThru -NoNewWindow
Remove-Item Env:QT_QPA_PLATFORM
if ($p.ExitCode -ne 0) { throw "Autoteste do executável falhou (código $($p.ExitCode))" }
Write-Host "Autoteste OK: dist\7xAi\$ExeName"

# 5) Instala para o usuário atual (sem administrador) em %LOCALAPPDATA%\Programs\7xAi.
#    Os dados (motor de IA, modelos, trabalhos) continuam em %LOCALAPPDATA%\MobiliaStudio: nada é baixado de novo.
$installDir = Join-Path $env:LOCALAPPDATA "Programs\7xAi"
$open = @(Get-Process -Name "7xAi" -ErrorAction SilentlyContinue)
if ($open.Count -gt 0) {
    throw "O 7xAi está aberto. Feche o 7xAi e rode o GERAR_EXE.bat de novo."
}
if (-not (Test-Path $installDir)) { New-Item -ItemType Directory -Path $installDir -Force | Out-Null }
& robocopy $distDir $installDir /MIR /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
$rc = $LASTEXITCODE
# robocopy: códigos de 0 a 7 são sucesso; 8 ou mais é falha (em geral, arquivo em uso).
if ($rc -ge 8) {
    throw "Não foi possível copiar o 7xAi para $installDir (robocopy código $rc). Se o 7xAi estiver aberto, feche-o e rode o GERAR_EXE.bat de novo."
}
$global:LASTEXITCODE = 0
$installedExe = Join-Path $installDir $ExeName
if (-not (Test-Path $installedExe)) { throw "A instalação não encontrou $installedExe" }
Write-Host "7xAi instalado em: $installDir"

# 6) Ícone "7xAi" na Área de Trabalho e no Menu Iniciar (funciona com Área de Trabalho no OneDrive).
$shell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$programs = [Environment]::GetFolderPath("Programs")

function Remove-OldShortcuts([string]$Folder, [bool]$Recurse) {
    # atalhos antigos que abriam o MobiliaStudio.exe (nome anterior do programa)
    if (-not $Folder -or -not (Test-Path -LiteralPath $Folder)) { return }
    $links = @(Get-ChildItem -LiteralPath $Folder -Filter "*.lnk" -Recurse:$Recurse -ErrorAction SilentlyContinue)
    foreach ($l in $links) {
        $target = ""
        try { $target = [string]$shell.CreateShortcut($l.FullName).TargetPath } catch { $target = "" }
        if ($target -and $target.EndsWith("\MobiliaStudio.exe", [System.StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $l.FullName -Force -ErrorAction SilentlyContinue
            Write-Host "Atalho antigo removido: $($l.FullName)"
        }
    }
}

function New-AppShortcut([string]$LinkPath) {
    $s = $shell.CreateShortcut($LinkPath)
    $s.TargetPath = $installedExe
    $s.WorkingDirectory = $installDir
    $s.IconLocation = "$installedExe,0"
    $s.Description = "7xAi - decoração virtual em vídeo"
    $s.Save()
}

Remove-OldShortcuts $desktop $false
Remove-OldShortcuts $programs $true
$desktopLink = Join-Path $desktop "7xAi.lnk"
New-AppShortcut $desktopLink
Write-Host "Ícone criado na Área de Trabalho: $desktopLink"
if ($programs) {
    New-AppShortcut (Join-Path $programs "7xAi.lnk")
    Write-Host "Atalho criado no Menu Iniciar."
}

# 7) Instalador (opcional, para instalar em outros PCs).
$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (Test-Path $iscc) {
    & $iscc "/DAppVersion=$version" packaging\installer.iss
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup falhou" }
    Write-Host "Instalador para outros PCs: dist\7xAi-Setup-$version.exe"
} else {
    Write-Host "Inno Setup 6 não encontrado: o instalador para outros PCs não foi gerado (não é preciso neste PC)."
    Write-Host "Para gerá-lo, instale o Inno Setup 6 (https://jrsoftware.org/isdl.php) e rode este script de novo."
}
exit 0
