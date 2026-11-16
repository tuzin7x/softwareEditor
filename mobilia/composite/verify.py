"""Verificação independente da regra da placa (PT5).

Fora da **máscara de influência** (os pixels a menos de um raio de algum pixel de móvel), o quadro
composto tem de ser byte a byte a foto: ``np.array_equal(out8[~mascara], placa8[~mascara])``.

A máscara é calculada aqui, a partir só do alfa das camadas 3D, sem nada da lógica interna do
compositor (recorte, pesos de sombra, cópia dos bytes): um erro no recorte, na colagem ou na
conta da sombra que encoste em pixels fora dela faz a verificação falhar.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# lado do bloco da máscara: ela é montada numa grade de blocos (barata) e é um SUPERCONJUNTO da
# vizinhança euclidiana exata (nunca acusa um pixel que a composição podia mudar)
TILE_PX = 4
# folga além do raio de influência: antisserrilhado/desfoque de nitidez do alfa (≤ 4 px) e margem
MARGIN_PX = 8


def influence_mask(alpha: np.ndarray | None, origin: tuple[int, int], shape: tuple[int, int],
                   radius_px: float, tile: int = TILE_PX) -> np.ndarray:
    """Máscara (H, W) que contém todo pixel a menos de ``radius_px`` de algum pixel com alfa > 0.

    ``alpha`` (h×w) está na posição ``origin`` = (x, y) do quadro ``shape`` = (H, W). Um bloco da
    grade entra quando a distância euclidiana entre blocos até algum bloco com CG é ≤ raio/bloco
    + 1,5 (pixels a distância d ficam a menos de d/bloco + √2 blocos): superconjunto garantido."""
    H, W = int(shape[0]), int(shape[1])
    nty, ntx = -(-H // tile), -(-W // tile)
    tiles = np.zeros((nty, ntx), bool)
    if alpha is not None:
        a = np.asarray(alpha) > 0
        if a.any():
            ox, oy = int(origin[0]), int(origin[1])
            h, w = a.shape
            py, px = oy % tile, ox % tile                     # alinha o recorte à grade do quadro
            th, tw = -(-(py + h) // tile), -(-(px + w) // tile)
            buf = np.zeros((th * tile, tw * tile), bool)
            buf[py:py + h, px:px + w] = a
            blk = buf.reshape(th, tile, tw, tile).any(axis=(1, 3))
            ty, tx = oy // tile, ox // tile
            tiles[ty:ty + th, tx:tx + tw] |= blk[:nty - ty, :ntx - tx]
            tiles = ndimage.distance_transform_edt(~tiles) <= float(radius_px) / tile + 1.5
    return tiles[np.arange(H) // tile][:, np.arange(W) // tile]


def outside_mask_equal(out8: np.ndarray, plate8: np.ndarray, mask: np.ndarray) -> tuple[bool, int, int]:
    """(``np.array_equal(out8[~mask], plate8[~mask])``, pixels divergentes, maior diferença).

    Mesma resposta do ``array_equal`` sobre o complemento da máscara, comparando o quadro inteiro
    canal a canal (sem copiar os pixels de fora da máscara)."""
    if out8.shape != plate8.shape:
        return False, int(mask.size), 255
    neq = out8 != plate8
    changed = neq[..., 0]
    for i in range(1, neq.shape[-1]):
        changed = changed | neq[..., i]
    bad = changed & ~mask
    if not bad.any():
        return True, 0, 0
    d = np.abs(out8[bad].astype(np.int16) - plate8[bad].astype(np.int16))
    return False, int(np.count_nonzero(bad)), int(d.max())
