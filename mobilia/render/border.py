"""Região de render (border) em pixels inteiros (Python puro: usado pelo worker e pelos testes).

O Cycles converte o border normalizado (0..1, origem embaixo à esquerda) em pixels por
TRUNCAMENTO (``int(border.xmin * largura)`` em float32). O retângulo anotado no ``frames.json``
era arredondado "para fora" (piso/teto) e podia ficar 1 px maior que o renderizado: no EXR de
quadro inteiro essa faixa chega com todos os canais 0 (daí ``mask_unrendered``), e num EXR
recortado o tamanho do recorte não bateria com o retângulo anotado.

Aqui o retângulo em pixels vem primeiro (arredondado para fora, como antes) e o border
entregue ao Blender aponta para 1/4 de pixel dentro de cada aresta: truncar (ou arredondar)
esse valor devolve exatamente o mesmo pixel, com folga muito maior que o erro do float32.
Assim o retângulo anotado É o renderizado, e o EXR recortado (``use_crop_to_border``) tem
exatamente ``x1 − x0`` × ``y1 − y0`` pixels.
"""
from __future__ import annotations

import math

import numpy as np

# fração de pixel somada a cada aresta inteira (robusta a truncamento e a arredondamento)
EDGE_OFFSET = 0.25
# tolerância (px) ao arredondar para fora: 20/61 · 61 = 20,000001 não vira 21
SNAP = 1e-6
# compressão do EXR por quadro: RLE gravou ~30% mais rápido que ZIP (sala de teste a 640 px, CPU),
# com arquivo ~1,5× maior; sem compressão o arquivo cresce muito mais e o antivírus do Windows
# varre cada byte gravado
EXR_CODEC = "RLE"
EXR_CODECS = ("NONE", "RLE", "ZIPS", "ZIP", "PIZ")


def pixel_border(b, width: int, height: int):
    """Converte o border normalizado do Blender ``(min_x, min_y, max_x, max_y)`` em
    ``(border_px, border_blender)``.

    ``border_px`` = ``[x0, y0, x1, y1]`` com origem no topo-esquerdo e fim exclusivo (o mesmo
    formato do ``frames.json``); ``border_blender`` = valores para ``sc.render.border_*`` que o
    Cycles converte exatamente nesse retângulo. ``None`` se o retângulo ficar vazio.
    """
    W, H = int(width), int(height)

    def lo(v: float, n: int) -> int:        # para fora, sem ganhar 1 px por erro de float
        return max(0, min(n, int(math.floor(v * n + SNAP))))

    def hi(v: float, n: int) -> int:
        return max(0, min(n, int(math.ceil(v * n - SNAP))))

    x0, x1 = lo(b[0], W), hi(b[2], W)
    y0, y1 = lo(1.0 - b[3], H), hi(1.0 - b[1], H)       # topo (origem em cima)
    if x1 <= x0 or y1 <= y0:
        return None
    # no Blender a origem é embaixo: linhas [H − y1, H − y0)
    by0, by1 = H - y1, H - y0

    def edge(p: int, n: int) -> float:
        return 1.0 if p >= n else (p + EDGE_OFFSET) / n

    return [x0, y0, x1, y1], (edge(x0, W), edge(by0, H), edge(x1, W), edge(by1, H))


def blender_disprect(border_blender, width: int, height: int) -> list[int]:
    """O retângulo que o Blender renderiza para um border (emula o truncamento em float32),
    convertido para ``[x0, y0, x1, y1]`` com origem no topo. Usado nos testes."""
    W, H = int(width), int(height)
    f = [np.float32(v) for v in border_blender]
    xmin, ymin = int(f[0] * np.float32(W)), int(f[1] * np.float32(H))
    xmax, ymax = int(f[2] * np.float32(W)), int(f[3] * np.float32(H))
    return [xmin, H - ymax, xmax, H - ymin]


def exr_codec(name: str | None) -> str:
    """Nome do codec do EXR (``image_settings.exr_codec``); desconhecido → ``EXR_CODEC``."""
    n = str(name or EXR_CODEC).upper()
    return n if n in EXR_CODECS else EXR_CODEC
