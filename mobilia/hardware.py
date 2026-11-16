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


def nvenc_available(ffmpeg_exe: str) -> bool:
    """Testa se o FFmpeg consegue abrir o encoder NVENC (requer GPU NVIDIA + driver)."""
    try:
        res = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=c=black:s=256x256:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return res.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    r = detect_hardware()
    print(r.to_json())
    print(choose_profile(r))
