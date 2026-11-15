# Diagnóstico de hardware do Mobília Studio (Windows 10/11).
# Uso: clique com o botão direito > "Executar com o PowerShell"
#   ou, num terminal:  powershell -ExecutionPolicy Bypass -File .\diagnostico_hardware.ps1
# Não instala nada e não envia dados: só lê as informações e grava um relatório
# "diagnostico_mobilia.txt" na Área de Trabalho, para você conferir ou compartilhar.

$ErrorActionPreference = "SilentlyContinue"
$out = New-Object System.Collections.Generic.List[string]
function Linha($t) { $out.Add($t); Write-Host $t }

Linha "=== Mobília Studio — diagnóstico de hardware ==="
Linha ("Data: " + (Get-Date -Format "yyyy-MM-dd HH:mm"))
$os = Get-CimInstance Win32_OperatingSystem
Linha ("Windows: " + $os.Caption + " build " + $os.BuildNumber + " (" + $os.OSArchitecture + ")")
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
Linha ("CPU: " + $cpu.Name.Trim() + " | núcleos: " + $cpu.NumberOfCores + " | threads: " + $cpu.NumberOfLogicalProcessors)
$ram = [math]::Round(($os.TotalVisibleMemorySize / 1MB), 1)
Linha ("RAM: $ram GB")

Linha ""
Linha "--- Placas de vídeo (Windows) ---"
Get-CimInstance Win32_VideoController | ForEach-Object {
    Linha ("* " + $_.Name + " | driver " + $_.DriverVersion)
}

Linha ""
Linha "--- NVIDIA (nvidia-smi) ---"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    $cand = "$env:SystemRoot\System32\nvidia-smi.exe"
    if (Test-Path $cand) { $smi = $cand }
}
if ($smi) {
    $q = & $smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap,pcie.link.gen.current,power.limit --format=csv,noheader 2>$null
    if ($q) {
        foreach ($l in $q) { Linha ("GPU: " + $l) }
        $cuda = (& $smi 2>$null | Select-String "CUDA Version").ToString()
        if ($cuda) { Linha ("Driver: " + $cuda.Trim()) }
    } else {
        Linha "nvidia-smi encontrado, mas não respondeu (driver com problema?)."
    }
} else {
    Linha "nvidia-smi NÃO encontrado: sem GPU NVIDIA ou driver não instalado."
}

Linha ""
Linha "--- Disco ---"
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object {
    $free = [math]::Round($_.FreeSpace / 1GB, 1); $size = [math]::Round($_.Size / 1GB, 1)
    Linha ("Unidade " + $_.DeviceID + " livre: $free GB de $size GB")
}

Linha ""
Linha "--- Recomendação (estimativa) ---"
$vram = 0
if ($smi) {
    $mem = & $smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($mem) { $vram = [int]$mem }
}
if ($vram -ge 11000) { Linha "Perfil ULTRA: render até 2560 px, 192 amostras, OptiX." }
elseif ($vram -ge 7500) { Linha "Perfil ALTA: render 1920 px, 128 amostras, OptiX." }
elseif ($vram -ge 5500) { Linha "Perfil MÉDIA: 1920 px, 96 amostras, modelos carregados um a um." }
elseif ($vram -gt 0) { Linha "Perfil ECONÔMICA: 1600 px, modelo de geometria menor." }
else { Linha "Sem GPU NVIDIA: modo CPU (funciona, mas é muito mais lento)." }
Linha "Espaço necessário: ~12 GB (runtime Python+CUDA ~6 GB, modelos ~3 GB, assets 3D ~0,5 GB, trabalhos)."

$dest = Join-Path ([Environment]::GetFolderPath("Desktop")) "diagnostico_mobilia.txt"
$out | Out-File -FilePath $dest -Encoding utf8
Write-Host ""
Write-Host "Relatório salvo em: $dest"
