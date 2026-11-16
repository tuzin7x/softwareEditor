"""Configuração global: diretórios de dados, cache de modelos e parâmetros padrão."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    """Diretório de dados do usuário (modelos, assets, trabalhos).

    Windows: %LOCALAPPDATA%\\MobiliaStudio ; Linux/macOS: ~/.local/share/mobilia-studio.
    Pode ser sobrescrito com a variável de ambiente MOBILIA_HOME.
    """
    env = os.environ.get("MOBILIA_HOME")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "MobiliaStudio"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "mobilia-studio"


DATA_DIR = _default_data_dir()
ASSETS_DIR = DATA_DIR / "assets"
JOBS_DIR = DATA_DIR / "jobs"

# Pesos dos modelos (Hugging Face). Todos com licença que permite uso comercial (MIT).
GEOMETRY_MODEL_ID = "Ruicheng/moge-2-vitl-normal"      # MoGe-2 ViT-L (MIT)
GEOMETRY_MODEL_ID_SMALL = "Ruicheng/moge-2-vitb-normal"  # para GPUs com pouca VRAM / CPU
SEGMENTATION_MODEL_ID = "shi-labs/oneformer_ade20k_swin_large"  # OneFormer ADE20K (MIT)


@dataclass
class RenderSettings:
    """Parâmetros do render/vídeo. Os perfis de hardware ajustam estes valores."""

    fps: int = 24                      # cinematográfico (desfoque de movimento ligado)
    max_long_edge: int = 1920          # resolução máxima do vídeo (lado maior)
    samples: int = 128                 # amostras Cycles por pixel (com denoise)
    preview_samples: int = 16
    use_motion_blur: bool = True
    shutter: float = 0.5               # fração do intervalo entre quadros (180°)
    denoise: bool = True
    seed: int = 7                      # semente fixa: ruído estável entre quadros
    device: str = "AUTO"               # AUTO | OPTIX | CUDA | CPU
    texture_res: str = "2k"            # resolução das texturas dos assets


@dataclass
class VideoSettings:
    codec: str = "libx264"             # libx264 | h264_nvenc | libx265 | hevc_nvenc
    crf: int = 14                      # qualidade (menor = melhor)
    preset: str = "medium"
    pix_fmt: str = "yuv420p"           # compatibilidade máxima (Instagram/WhatsApp)
    hold_start_s: float = 0.8          # foto vazia antes da primeira peça surgir
    hold_end_s: float = 2.5            # ambiente pronto no final


@dataclass
class AppConfig:
    render: RenderSettings = field(default_factory=RenderSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    data_dir: Path = DATA_DIR

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"
