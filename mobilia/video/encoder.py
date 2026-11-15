"""Codificação MP4 com FFmpeg (quadros RGB enviados por pipe, sem arquivos temporários).

Observação honesta: H.264/H.265 em yuv420p são formatos COM PERDA — a garantia
bit a bit vale para os quadros compostos (mestres) e é verificada antes do encode.
O MP4 final é codificado em alta qualidade (CRF baixo) e com a conversão RGB→YUV
explícita em BT.709 (o padrão do swscale seria BT.601, o que deslocaria as cores),
e o relatório mede o PSNR do vídeo decodificado contra os quadros mestres.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def ffmpeg_exe() -> str:
    env = os.environ.get("MOBILIA_FFMPEG")
    if env and Path(env).exists():
        return env
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise RuntimeError("FFmpeg não encontrado.")


@dataclass
class EncodeSettings:
    codec: str = "libx264"            # libx264 | h264_nvenc | libx265 | hevc_nvenc
    crf: int = 14
    preset: str = "slow"
    pix_fmt: str = "yuv420p"
    keyint_s: float = 4.0


def _codec_args(s: EncodeSettings, fps: int) -> list[str]:
    g = str(max(1, int(round(s.keyint_s * fps))))
    if s.codec == "libx264":
        return ["-c:v", "libx264", "-preset", s.preset, "-crf", str(s.crf), "-profile:v", "high",
                "-g", g, "-x264-params", "aq-mode=3:aq-strength=0.8:deblock=-1,-1"]
    if s.codec == "libx265":
        return ["-c:v", "libx265", "-preset", s.preset, "-crf", str(s.crf), "-tag:v", "hvc1",
                "-x265-params", f"keyint={g}:aq-mode=3:log-level=error"]
    if s.codec == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", str(s.crf + 2),
                "-b:v", "0", "-profile:v", "high", "-spatial-aq", "1", "-temporal-aq", "1", "-g", g]
    if s.codec == "hevc_nvenc":
        return ["-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", str(s.crf + 2),
                "-b:v", "0", "-spatial-aq", "1", "-temporal-aq", "1", "-g", g, "-tag:v", "hvc1"]
    raise ValueError(f"codec não suportado: {s.codec}")


class VideoWriter:
    def __init__(self, path: str | Path, width: int, height: int, fps: int, settings: EncodeSettings | None = None,
                 log_path: str | Path | None = None):
        if width % 2 or height % 2:
            raise ValueError("dimensões precisam ser pares para yuv420p")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.s = settings or EncodeSettings()
        self.size = (width, height)
        self.frames = 0
        tmp = self.path.with_name(self.path.stem + ".partial" + self.path.suffix)
        self._tmp = tmp
        cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-vf", f"scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,"
                      f"format={self.s.pix_fmt}",
               *_codec_args(self.s, fps),
               "-pix_fmt", self.s.pix_fmt, "-color_primaries", "bt709", "-color_trc", "bt709",
               "-colorspace", "bt709", "-color_range", "tv", "-movflags", "+faststart", "-an", str(tmp)]
        self._log = open(log_path, "wb") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def write(self, rgb8: np.ndarray) -> None:
        if rgb8.shape[1] != self.size[0] or rgb8.shape[0] != self.size[1]:
            raise ValueError("tamanho de quadro inconsistente")
        self.proc.stdin.write(np.ascontiguousarray(rgb8, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> Path:
        self.proc.stdin.close()
        rc = self.proc.wait()
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        if rc != 0:
            raise RuntimeError(f"FFmpeg falhou (código {rc}). Veja o log de encode.")
        os.replace(self._tmp, self.path)
        return self.path


def decode_frames(path: str | Path, width: int, height: int, frames: list[int] | None = None):
    """Decodifica o MP4 para RGB (BT.709) — usado para medir a fidelidade do encode."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int,format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    n = width * height * 3
    idx = 0
    want = set(frames) if frames is not None else None
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        idx += 1
        if want is None or idx in want:
            yield idx, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    p.stdout.close()
    p.wait()


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if mask is not None:
        d = d[mask]
    mse = float(d.mean()) if d.size else 0.0
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
