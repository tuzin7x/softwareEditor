"""Codificação MP4 com FFmpeg (quadros RGB enviados por pipe, sem arquivos temporários).

Observação honesta: H.264/H.265 em yuv420p são formatos COM PERDA — a garantia
bit a bit vale para os quadros compostos (mestres) e é verificada antes do encode.
O MP4 final é codificado em alta qualidade (CRF baixo) e com a conversão RGB→YUV
explícita em BT.709 (o padrão do swscale seria BT.601, o que deslocaria as cores),
e o relatório mede o PSNR do vídeo decodificado contra os quadros mestres.

Codificador (PT5): NVENC quando a GPU NVIDIA consegue codificar (a CPU fica livre para a
composição, que roda em paralelo ao render), senão libx264 com preset ``medium`` (~1,7× mais rápido
que ``slow`` na mesma qualidade visual: −0,3 dB de PSNR em CRF 14). A verificação decodifica só
alguns quadros, cada um com um *seek* na entrada (``seek_frames``), sem ler o vídeo inteiro.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
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
    preset: str = "medium"
    pix_fmt: str = "yuv420p"
    keyint_s: float = 4.0


# Força o codificador ("libx264", "h264_nvenc"...); vazio ou "auto" = escolha automática.
ENV_CODEC = "MOBILIA_CODIFICADOR"
_NVENC: dict[str, bool] = {}
_NVENC_LOCK = threading.Lock()


def nvenc_available() -> bool:
    """O FFmpeg consegue abrir o NVENC nesta máquina? (GPU NVIDIA + driver; testado uma vez por
    processo com um encode minúsculo de verdade, porque o encoder pode estar compilado no FFmpeg
    sem haver GPU que o rode). O teste usa as MESMAS opções do encode do vídeo (``_codec_args``):
    se a GPU não tiver um recurso pedido (p. ex. AQ temporal), o vídeo sai em x264 em vez de o
    FFmpeg morrer no primeiro quadro, já com o render em andamento."""
    with _NVENC_LOCK:
        if "ok" not in _NVENC:
            from ..hardware import nvenc_available as probe

            try:
                _NVENC["ok"] = bool(probe(ffmpeg_exe(), _codec_args(EncodeSettings(codec="h264_nvenc"), 24)))
            except Exception:
                _NVENC["ok"] = False
        return _NVENC["ok"]


def probe_nvenc_background() -> None:
    """Faz o teste do NVENC (~1 s: FFmpeg + contexto CUDA) numa thread, já no início do trabalho,
    para ele não ficar no caminho crítico logo antes do render (``choose_codec`` espera o
    resultado, se ainda não saiu). Não faz nada quando ``MOBILIA_CODIFICADOR`` força o codec."""
    forced = os.environ.get(ENV_CODEC, "").strip().lower()
    if (forced and forced != "auto") or "ok" in _NVENC:
        return
    threading.Thread(target=nvenc_available, name="teste-nvenc", daemon=True).start()


def choose_codec(gpu: bool) -> str:
    """``h264_nvenc`` quando há GPU NVIDIA que codifica, senão ``libx264`` (``MOBILIA_CODIFICADOR``
    força um deles)."""
    forced = os.environ.get(ENV_CODEC, "").strip().lower()
    if forced and forced != "auto":
        return forced
    return "h264_nvenc" if gpu and nvenc_available() else "libx264"


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

    def abort(self) -> None:
        """Interrompe o FFmpeg e apaga o arquivo parcial (trabalho com erro). Pode ser chamado de
        outra thread enquanto uma escrita está presa no pipe: matar o processo a destrava."""
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        self.proc.wait()
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        try:
            self._tmp.unlink()
        except OSError:
            pass

    def close(self) -> Path:
        self.proc.stdin.close()
        rc = self.proc.wait()
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        if rc != 0:
            raise RuntimeError(f"FFmpeg falhou (código {rc}). Veja o log de encode.")
        os.replace(self._tmp, self.path)
        return self.path


_TO_RGB = "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int,format=rgb24"


def decode_frames(path: str | Path, width: int, height: int, frames: list[int] | None = None):
    """Decodifica o MP4 inteiro para RGB (BT.709), em ordem (quadro 1 = primeiro)."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-vf", _TO_RGB, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
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


def seek_frames(path: str | Path, width: int, height: int, frames: list[int], fps: float):
    """Decodifica só os quadros pedidos (1 = primeiro), sem ler o vídeo inteiro.

    Cada quadro é um FFmpeg com *seek* na entrada (``-ss`` antes de ``-i``): ele pula para o
    quadro-chave anterior e decodifica só até o instante pedido (seek exato, o padrão do FFmpeg).
    O instante fica 1/4 de quadro antes do quadro k: o primeiro quadro com tempo ≥ esse instante é
    o k, sem depender de arredondamento da base de tempo. Os quadros rodam em paralelo."""
    n = width * height * 3
    procs = []
    for k in sorted({int(f) for f in frames if int(f) >= 1}):
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
        if k > 1:
            cmd += ["-ss", f"{(k - 1.25) / float(fps):.6f}"]
        cmd += ["-i", str(path), "-frames:v", "1", "-vf", _TO_RGB, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        procs.append((k, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))))
    for k, p in procs:
        buf, _ = p.communicate()
        if len(buf) >= n:
            yield k, np.frombuffer(buf[:n], np.uint8).reshape(height, width, 3)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if mask is not None:
        d = d[mask]
    mse = float(d.mean()) if d.size else 0.0
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


# ============================================================================================================
# MKV sem perdas para auditoria (PT-V3-7): RGB direto, sem conversão para YUV
# ============================================================================================================
LOSSLESS_CODECS = ("libx264rgb", "ffv1")


class LosslessWriter:
    """MKV sem perdas dos quadros mestres (auditoria: decodificado, cada quadro volta byte a byte).

    ``libx264rgb`` com ``-qp 0`` (rápido, padrão) ou ``ffv1`` (arquivo, mais lento). Os quadros vão em RGB sem
    filtro de cor, por isso a ida e volta é exata. Mesmo contrato do :class:`VideoWriter` (``write``/``close``/
    ``abort``; o arquivo só aparece no ``close`` sem erro)."""

    def __init__(self, path: str | Path, width: int, height: int, fps: int, codec: str = "libx264rgb",
                 log_path: str | Path | None = None):
        if codec not in LOSSLESS_CODECS:
            raise ValueError(f"codec sem perdas não suportado: {codec}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        self.frames = 0
        self._tmp = self.path.with_name(self.path.stem + ".partial" + self.path.suffix)
        if codec == "libx264rgb":
            cargs = ["-c:v", "libx264rgb", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "rgb24"]
        else:
            cargs = ["-c:v", "ffv1", "-level", "3", "-slices", "16", "-g", "1", "-pix_fmt", "rgb24"]
        cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               *cargs, "-an", "-f", "matroska", str(self._tmp)]
        self._log = open(log_path, "wb") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    write = VideoWriter.write
    abort = VideoWriter.abort
    close = VideoWriter.close


def decode_lossless(path: str | Path, width: int, height: int):
    """Quadros RGB do MKV sem perdas, em ordem (1 = primeiro), sem conversão de cor."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    n = width * height * 3
    idx = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            idx += 1
            yield idx, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        p.stdout.close()
        p.wait()
