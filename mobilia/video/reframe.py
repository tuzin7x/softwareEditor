"""Recorte do vídeo "móveis entrando" (modo B) para 16:9 (1920×1080) ou 9:16 (1080×1920).

O MP4 do modo B tem a proporção da foto (o último quadro é a foto, verificado byte a byte nos quadros mestres).
Para redes sociais o mesmo vídeo é recortado, sem barras, em volta de onde os móveis entram: o centro do recorte
é o centro da área que muda entre a sala vazia e a foto (``sala_vazia.png`` da pasta do trabalho), limitado à
foto. O arquivo original fica; o recortado sai ao lado com o sufixo do formato.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

FORMATS = {"16x9": (1920, 1080), "9x16": (1080, 1920)}


def crop_box(width: int, height: int, fmt: str, center: tuple[float, float] | None = None
             ) -> tuple[int, int, int, int]:
    """(x, y, largura, altura) do maior recorte com a proporção ``fmt`` dentro de ``width``×``height``,
    centrado em ``center`` (x, y; padrão: o centro), lados pares."""
    W, H = FORMATS[fmt]
    ar = W / H
    cw, ch = (width, width / ar) if width / height <= ar else (height * ar, height)
    cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
    cx, cy = center if center is not None else (width / 2, height / 2)
    x = int(np.clip(round(cx - cw / 2), 0, width - cw)) // 2 * 2
    y = int(np.clip(round(cy - ch / 2), 0, height - ch)) // 2 * 2
    return x, y, cw, ch


def change_center(work_dir: Path) -> tuple[float, float] | None:
    """Centro (x, y) da área onde a sala vazia difere da foto (o que entra no vídeo)."""
    from PIL import Image, ImageOps

    plate_p = work_dir / "sala_vazia.png"
    if not plate_p.is_file():
        return None
    try:
        rep = json.loads((work_dir / "report.json").read_text(encoding="utf-8"))
        photo_p = rep.get("foto", {}).get("arquivo")
        plate = np.asarray(Image.open(plate_p).convert("RGB")).astype(np.int16)
        photo = np.asarray(ImageOps.exif_transpose(Image.open(photo_p)).convert("RGB").resize(
            (plate.shape[1], plate.shape[0]))).astype(np.int16)
    except Exception:  # noqa: BLE001 - sem centro: recorte centrado
        return None
    d = np.abs(plate - photo).max(axis=2) > 24
    if d.mean() < 0.002:
        return None
    ys, xs = np.nonzero(d)
    return float(xs.mean()), float(ys.mean())


def _probe_size(path: str) -> tuple[int, int]:
    import imageio_ffmpeg

    r = imageio_ffmpeg.read_frames(path)
    meta = next(r)
    r.close()
    return tuple(int(v) for v in meta["size"])


def reframe_mp4(path: str, fmt: str, report_path: str | None = None, out: str | None = None) -> str:
    """Recorta ``path`` para ``fmt`` ("16x9" | "9x16") e devolve o caminho do novo MP4."""
    from .encoder import ffmpeg_exe

    if fmt not in FORMATS:
        raise ValueError(f"Formato de vídeo inválido: {fmt!r} (use 16x9 ou 9x16).")
    src = Path(path)
    dst = Path(out) if out else src.with_name(f"{src.stem}_{fmt}.mp4")
    w, h = _probe_size(str(src))
    center = change_center(Path(report_path).parent) if report_path else None
    x, y, cw, ch = crop_box(w, h, fmt, center)
    W, H = FORMATS[fmt]
    vf = f"crop={cw}:{ch}:{x}:{y},scale={W}:{H}:flags=lanczos,setsar=1"
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset",
           "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-movflags", "+faststart", "-an", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(dst)


__all__ = ["reframe_mp4", "crop_box", "change_center", "FORMATS"]
