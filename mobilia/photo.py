"""Leitura da foto original e conversões de cor (sRGB <-> linear).

A foto é carregada uma única vez e nunca é modificada: todo o pipeline trabalha
sobre uma cópia em memória e o compositor devolve os bytes originais nos pixels
que não foram cobertos por objetos/sombras.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass
class Photo:
    rgb8: np.ndarray            # (H, W, 3) uint8 — placa de fundo imutável
    path: Path
    icc_profile: bytes | None

    @property
    def height(self) -> int:
        return int(self.rgb8.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb8.shape[1])

    def linear(self) -> np.ndarray:
        """Foto em float32 linear (Rec.709/sRGB primárias)."""
        return srgb_to_linear(self.rgb8.astype(np.float32) / 255.0)


def load_photo(path: str | Path, max_long_edge: int | None = None) -> Photo:
    """Carrega a foto respeitando a orientação EXIF.

    Se ``max_long_edge`` for dado e a foto for maior, ela é reduzida UMA vez
    (Lanczos). A versão reduzida passa a ser a placa imutável do vídeo — vídeos
    H.264 acima de ~4K não são práticos; isso é uma reamostragem, não uma edição
    de conteúdo, e é registrado no relatório do trabalho.
    """
    path = Path(path)
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    icc = im.info.get("icc_profile")
    if im.mode != "RGB":
        im = im.convert("RGB")
    if max_long_edge and max(im.size) > max_long_edge:
        scale = max_long_edge / max(im.size)
        new = (int(round(im.width * scale)), int(round(im.height * scale)))
        im = im.resize(new, Image.LANCZOS)
    # H.264/H.265 em yuv420p exigem dimensões pares: recorta 1 px se necessário
    # (nunca reescala). O recorte é documentado no relatório.
    w, h = im.size
    if w % 2 or h % 2:
        im = im.crop((0, 0, w - (w % 2), h - (h % 2)))
    arr = np.asarray(im, dtype=np.uint8).copy()
    arr.setflags(write=False)
    return Photo(rgb8=arr, path=path, icc_profile=icc)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def to_uint8(x01: np.ndarray) -> np.ndarray:
    return (np.clip(x01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
