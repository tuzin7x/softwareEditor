"""Detecção de hardware (GPU NVIDIA, VRAM, driver) e escolha automática de perfil.

O software roda em qualquer máquina, mas foi otimizado para GPUs NVIDIA:
- análise (MoGe-2 + OneFormer) em PyTorch CUDA, um modelo por vez (VRAM baixa);
- render 3D em Blender Cycles com OptiX (RTX) ou CUDA, em processo separado,
  depois que os modelos de IA foram descarregados da GPU.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass
class GpuInfo:
    name: str
    vram_total_mb: int
    vram_free_mb: int | None = None
    driver: str | None = None
    compute_capability: str | None = None


@dataclass
class HardwareReport:
    os: str
    cpu: str
    cpu_threads: int
    ram_gb: float | None
    gpus: list[GpuInfo] = field(default_factory=list)
    torch_cuda: bool = False
    torch_version: str | None = None
    nvenc: bool | None = None

    @property
    def best_gpu(self) -> GpuInfo | None:
        return max(self.gpus, key=lambda g: g.vram_total_mb) if self.gpus else None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@dataclass
class Profile:
    """Perfil de execução escolhido conforme a GPU."""

    name: str
    torch_device: str            # "cuda" | "cpu"
    use_fp16: bool
    geometry_model: str          # id HF do MoGe
    geometry_tokens: int | None  # resolução interna do MoGe (None = padrão)
    render_device: str           # OPTIX | CUDA | CPU
    render_samples: int          # amostras do quadro final (hold); as outras camadas derivam dele
    max_long_edge: int
    description: str
    layer_samples: dict | None = None   # {"final", "marco", "dinamico"} (ver render/layers.py)

    def __post_init__(self) -> None:
        from .render.layers import clamp_layers, samples_by_layer

        self.layer_samples = clamp_layers(self.layer_samples or samples_by_layer(self.render_samples),
                                          self.render_samples)


def _ram_gb() -> float | None:
    try:
        if hasattr(os, "sysconf"):
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError):
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX(); st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 1e9, 1)
        except Exception:
            return None
    return None


def _gpus_from_nvml() -> list[GpuInfo]:
    try:
        import pynvml  # pacote nvidia-ml-py
    except ImportError:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    out: list[GpuInfo] = []
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                cc = f"{major}.{minor}"
            except Exception:
                cc = None
            out.append(GpuInfo(name, int(mem.total // 2**20), int(mem.free // 2**20), driver, cc))
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out


def _gpus_from_smi() -> list[GpuInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        res = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return []
    out = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                out.append(GpuInfo(parts[0], int(float(parts[1])), int(float(parts[2])), parts[3],
                                   parts[4] if len(parts) > 4 else None))
            except ValueError:
                continue
    return out


def detect_hardware(check_torch: bool = True) -> HardwareReport:
    rep = HardwareReport(
        os=f"{platform.system()} {platform.release()} ({platform.machine()})",
        cpu=platform.processor() or platform.machine(),
        cpu_threads=os.cpu_count() or 1,
        ram_gb=_ram_gb(),
    )
    rep.gpus = _gpus_from_nvml() or _gpus_from_smi()
    if check_torch:
        try:
            import torch

            rep.torch_version = torch.__version__
            rep.torch_cuda = bool(torch.cuda.is_available())
            if rep.torch_cuda and not rep.gpus:
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    rep.gpus.append(GpuInfo(p.name, int(p.total_memory // 2**20), None, None,
                                            f"{p.major}.{p.minor}"))
        except Exception:
            rep.torch_cuda = False
    return rep


def choose_profile(rep: HardwareReport) -> Profile:
    """Escolhe o perfil de execução a partir da VRAM disponível.

    A análise usa um modelo por vez e libera a VRAM antes do render, então o
    pico é o do maior modelo (MoGe-2 ViT-L fp16 ≈ 2–3 GB em 1–2 MP) ou o do
    Cycles (cena + texturas 2K ≈ 1–3 GB). 6 GB já executam o perfil completo.
    """
    gpu = rep.best_gpu
    if gpu is None or not rep.torch_cuda:
        return Profile("cpu", "cpu", False, "Ruicheng/moge-2-vitb-normal", 1500,
                       "CPU", 32, 1280,
                       "Sem GPU NVIDIA utilizável: modo CPU (lento, qualidade de prévia).")
    vram = gpu.vram_total_mb
    try:
        cc = float(gpu.compute_capability or 0)
    except ValueError:
        cc = 0.0
    # OptiX requer RT cores (Turing 7.5+). Comparação numérica: "12.0" > "7.5".
    rtx = "RTX" in gpu.name.upper() or cc >= 7.5
    render_dev = "OPTIX" if rtx else "CUDA"
    if vram >= 11000:
        return Profile("ultra", "cuda", True, "Ruicheng/moge-2-vitl-normal", None,
                       render_dev, 192, 2560,
                       f"{gpu.name} ({vram // 1024} GB): qualidade máxima, render até 2560 px.")
    if vram >= 7500:
        return Profile("alta", "cuda", True, "Ruicheng/moge-2-vitl-normal", 2500,
                       render_dev, 128, 1920,
                       f"{gpu.name} ({vram // 1024} GB): alta qualidade, 1920 px.")
    if vram >= 5500:
        return Profile("media", "cuda", True, "Ruicheng/moge-2-vitl-normal", 1800,
                       render_dev, 96, 1920,
                       f"{gpu.name} ({vram // 1024} GB): VRAM moderada, modelos carregados um a um.")
    return Profile("economica", "cuda", True, "Ruicheng/moge-2-vitb-normal", 1500,
                   render_dev, 64, 1600,
                   f"{gpu.name} ({vram // 1024} GB): pouca VRAM, modelo de geometria menor.")


def nvenc_available(ffmpeg_exe: str, codec_args: list[str] | None = None) -> bool:
    """Testa se o FFmpeg consegue abrir o encoder NVENC (requer GPU NVIDIA + driver).

    ``codec_args`` são as opções do encode de verdade (preset, AQ temporal, perfil...): uma GPU
    que abre o NVENC "puro" mas não tem um desses recursos passaria no teste e só falharia no
    meio do trabalho, sem volta para o x264."""
    args = list(codec_args) if codec_args else ["-c:v", "h264_nvenc"]
    try:
        res = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=c=black:s=256x256:d=0.1", "-vf", "format=yuv420p", *args, "-pix_fmt", "yuv420p",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return res.returncode == 0
    except Exception:
        return False


# ----------------------------------------------------------------------------- perfis da V3
# PLANO_V3 §5.3. O modo B sem SDXL (OneFormer + SAM 2.1 + big-lama, um modelo por vez, pico
# ~2–3 GB) roda em qualquer GPU NVIDIA e, mais devagar, na CPU: é o perfil padrão. O SDXL (refino
# da sala vazia no modo B e, depois, o modo A novo) pede ~6,7–7,2 GB: com menos VRAM livre usa
# ``enable_model_cpu_offload`` (exige RAM ≥ 16 GB) e abaixo de ~5 GB fica desligado. Números de
# VRAM e de tempo da GPU são ESTIMATIVAS (não medidos nesta máquina sem GPU).

PERFIL_B_SEM_SDXL = "B sem SDXL"
PERFIL_B_COM_SDXL = "B com SDXL"
PERFIL_A = "A"
SDXL_VRAM_FULL_MB = 7000      # VRAM livre para o SDXL inteiro na GPU
SDXL_VRAM_MIN_MB = 5000       # abaixo disto o SDXL não roda (nem com offload)
SDXL_RAM_OFFLOAD_GB = 16.0    # offload para a RAM precisa de folga no sistema


@dataclass
class PerfilV3:
    nome: str                 # "B sem SDXL" | "B com SDXL" | "A"
    modo: str                 # "A" (foto vazia) | "B" (foto mobiliada)
    dispositivo: str          # "cuda" | "cpu"
    sdxl: bool                # o SDXL entra neste trabalho
    sdxl_offload: bool        # SDXL com ``enable_model_cpu_offload``
    motor_a: str | None       # modo A: "v2" (motor 3D atual) | "sdxl" (PT-V3-9); None no modo B
    motivo: str
    avisos: list[str] = field(default_factory=list)


def vram_free_mb() -> int | None:
    """VRAM livre agora (``torch.cuda.mem_get_info``), ou None sem CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free // 2**20)
    except Exception:
        return None


def sdxl_capacity(rep: HardwareReport) -> tuple[bool, bool, str]:
    """(cabe, com_offload, motivo) do SDXL nesta máquina, pela VRAM livre (ou total) e pela RAM."""
    gpu = rep.best_gpu
    if gpu is None or not rep.torch_cuda:
        return False, False, "sem GPU NVIDIA utilizável"
    free = gpu.vram_free_mb if gpu.vram_free_mb is not None else gpu.vram_total_mb
    free = min(free, gpu.vram_total_mb)
    if free >= SDXL_VRAM_FULL_MB:
        return True, False, f"{free} MB de VRAM livre"
    if gpu.vram_total_mb < SDXL_VRAM_MIN_MB or free < SDXL_VRAM_MIN_MB:
        return False, False, f"VRAM insuficiente para o SDXL ({free} MB livres; mínimo {SDXL_VRAM_MIN_MB})"
    if rep.ram_gb is None or rep.ram_gb < SDXL_RAM_OFFLOAD_GB:
        return False, False, (f"{free} MB de VRAM livre pedem offload para a RAM, mas a RAM do sistema "
                              f"({rep.ram_gb or '?'} GB) é menor que {SDXL_RAM_OFFLOAD_GB:.0f} GB")
    return True, True, f"{free} MB de VRAM livre: SDXL com offload para a RAM"


def choose_profile_v3(rep: HardwareReport, modo: str = "B", sdxl_instalado: bool | None = None,
                      sdxl_ligado: bool = True) -> PerfilV3:
    """Perfil da V3 para a foto (``modo`` "A" = vazia, "B" = mobiliada) nesta máquina.

    ``sdxl_instalado`` None consulta os pesos (hoje sempre falso: o SDXL ainda não é baixado).
    ``sdxl_ligado`` é a opção do usuário. Sem SDXL o modo B usa só o LaMa e o modo A, o motor v2."""
    if modo not in ("A", "B"):
        raise ValueError(f"modo inválido: {modo!r}")
    if sdxl_instalado is None:
        from .analysis.weights import sdxl_installed

        sdxl_instalado = sdxl_installed()
    cuda = rep.best_gpu is not None and rep.torch_cuda
    dev = "cuda" if cuda else "cpu"
    cabe, offload, por_que = sdxl_capacity(rep)
    avisos: list[str] = []
    if not cuda:
        avisos.append("Sem GPU NVIDIA: o processamento roda na CPU e fica bem mais lento.")
    usa = cabe and sdxl_instalado and sdxl_ligado
    if not sdxl_ligado:
        por_que = "SDXL desligado nas opções"
    elif not sdxl_instalado:
        por_que = "pesos do SDXL não instalados"
    if modo == "A":
        return PerfilV3(PERFIL_A, "A", dev, usa, usa and offload, "sdxl" if usa else "v2",
                        "Foto vazia: modo A " + ("com SDXL" if usa else f"no motor 3D atual ({por_que})"), avisos)
    if usa:
        return PerfilV3(PERFIL_B_COM_SDXL, "B", dev, True, offload, None,
                        f"Foto mobiliada: sala vazia por LaMa + refino SDXL ({por_que})", avisos)
    return PerfilV3(PERFIL_B_SEM_SDXL, "B", dev, False, False, None,
                    f"Foto mobiliada: sala vazia só com LaMa ({por_que})", avisos)


def sysmem_fallback_suspected(step_s: float, expected_s: float, factor: float = 4.0) -> bool:
    """Sonda do *sysmem fallback* do driver NVIDIA no Windows: sem OOM, a GPU passa a usar a RAM e
    fica de 5 a 20× mais lenta. Um passo da sonda ``factor`` vezes mais lento que o esperado para a
    GPU indica isso (vai para o ``report.json``; o chamador desliga o SDXL ou liga o offload)."""
    return expected_s > 0 and step_s > factor * expected_s


if __name__ == "__main__":
    r = detect_hardware()
    print(r.to_json())
    print(choose_profile(r))
    print(choose_profile_v3(r))
