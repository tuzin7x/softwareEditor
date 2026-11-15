"""Composição exata das camadas 3D sobre a foto original.

Fórmula (espaço linear, alfa pré-multiplicado do Cycles):
    saída = foto · S' · (1 − A) + C
onde C/A = móveis (Combined RGBA) e S' = fator de sombra (passe Shadow Catcher)
restrito à vizinhança dos móveis e suavizado na borda dessa região.

GARANTIA: todo pixel com A == 0 e S' == 1 recebe os BYTES ORIGINAIS da foto
(cópia direta, sem reconversão de cor). O compositor verifica isso em cada quadro
e registra quantos pixels foram alterados (móveis + sombras de contato).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..photo import linear_to_srgb, srgb_to_linear


@dataclass
class CompositeSettings:
    shadow_eps: float = 0.006          # |S−1| abaixo disso é ruído de render -> 1
    influence_radius_frac: float = 0.10   # raio da vizinhança (fração da diagonal) em volta dos móveis
    feather_px_frac: float = 0.012     # suavização da borda da região de sombra
    alpha_eps: float = 1.0 / 1024.0    # alfa abaixo disso é considerado 0
    grain_strength: float = 1.0        # 1 = igual ao ruído estimado da foto
    match_sharpness: bool = True


@dataclass
class FrameStats:
    frame: int
    changed_pixels: int = 0
    object_pixels: int = 0
    shadow_pixels: int = 0
    preserved_exact: bool = True
    max_diff_outside: int = 0


@dataclass
class PlateInfo:
    """Estatísticas da foto usadas para casar o CG com a câmera (calculadas uma vez)."""

    noise_sigma: float = 0.0           # desvio do ruído do sensor em sRGB 0..1
    blur_sigma_px: float = 0.0         # nitidez equivalente (desfoque gaussiano a aplicar no CG)
    grain: np.ndarray | None = field(default=None, repr=False)  # padrão de grão fixo (H, W, 3)


def estimate_plate(photo8: np.ndarray, seed: int = 7) -> PlateInfo:
    """Estima ruído do sensor (regiões planas) e cria um padrão de grão FIXO.

    O fundo é uma foto estática: o grão do CG também precisa ser estático, senão os
    móveis "chiam" sobre uma foto parada. Usamos o mesmo padrão em todos os quadros.
    """
    g = photo8.astype(np.float32).mean(-1) / 255.0
    # ruído: resíduo de alta frequência em regiões de baixo gradiente
    lap = ndimage.laplace(ndimage.gaussian_filter(g, 0.6)) - ndimage.laplace(ndimage.gaussian_filter(g, 1.6))
    grad = np.hypot(ndimage.sobel(ndimage.gaussian_filter(g, 2.0), 0), ndimage.sobel(ndimage.gaussian_filter(g, 2.0), 1))
    flat = grad < np.percentile(grad, 30)
    resid = g - ndimage.gaussian_filter(g, 1.2)
    sigma = float(1.4826 * np.median(np.abs(resid[flat] - np.median(resid[flat])))) if flat.any() else 0.0
    sigma = float(np.clip(sigma, 0.0, 0.03))
    # nitidez: energia relativa de alta frequência nas bordas (fotos JPEG costumam ser ~0,5–0,8 px)
    edges = grad > np.percentile(grad, 97)
    hf = np.abs(lap[edges]).mean() if edges.any() else 0.0
    blur = float(np.clip(0.9 - 40.0 * hf, 0.0, 0.9))
    rng = np.random.default_rng(seed)
    grain = rng.normal(0.0, 1.0, size=photo8.shape).astype(np.float32)
    grain = ndimage.gaussian_filter(grain, sigma=(0.5, 0.5, 0)) * 1.6   # grão levemente correlacionado
    return PlateInfo(noise_sigma=sigma, blur_sigma_px=blur, grain=grain)


class Compositor:
    def __init__(self, photo8: np.ndarray, settings: CompositeSettings | None = None, seed: int = 7):
        if photo8.dtype != np.uint8 or photo8.ndim != 3:
            raise ValueError("a placa deve ser uint8 (H, W, 3)")
        self.photo8 = photo8
        self.photo_lin = srgb_to_linear(photo8.astype(np.float32) / 255.0)
        self.s = settings or CompositeSettings()
        self.H, self.W = photo8.shape[:2]
        self.diag = float(np.hypot(self.W, self.H))
        self.plate = estimate_plate(photo8, seed)

    # ------------------------------------------------------------------
    def composite(self, frame: int, rgba: np.ndarray | None, shadow: np.ndarray | None,
                  border: list[int] | None) -> tuple[np.ndarray, FrameStats]:
        """Compõe um quadro. ``rgba``/``shadow`` são passes do Cycles (float, H×W×4 / H×W×3)."""
        st = FrameStats(frame)
        if rgba is None:
            return self.photo8.copy(), st
        A = rgba[..., 3].astype(np.float32)
        C = rgba[..., :3].astype(np.float32)
        S = shadow.mean(-1).astype(np.float32) if shadow is not None else np.ones_like(A)
        valid = np.zeros_like(A, dtype=bool)
        if border is not None:
            x0, y0, x1, y1 = border
            valid[max(0, y0):min(self.H, y1), max(0, x0):min(self.W, x1)] = True
        else:
            valid[:] = True
        A = np.where(valid & (A > self.s.alpha_eps), np.clip(A, 0, 1), 0.0).astype(np.float32)
        C = np.where(A[..., None] > 0, np.maximum(C, 0), 0.0).astype(np.float32)
        S = np.where(valid, S, 1.0)

        # --- fator de sombra: só na vizinhança dos móveis, com decaimento suave
        obj = A > 0
        feather = max(2.0, self.s.feather_px_frac * self.diag)
        if obj.any():
            r = max(8.0, self.s.influence_radius_frac * self.diag)
            dist_to_obj = ndimage.distance_transform_edt(~obj)
            t = np.clip((r - dist_to_obj) / (0.65 * r), 0.0, 1.0)     # 1 perto do móvel -> 0 no raio r
            w = t * t * (3 - 2 * t)
        else:
            w = np.zeros_like(A)
        if border is not None:
            edge = ndimage.distance_transform_edt(valid)
            e = np.clip(edge / feather, 0.0, 1.0)
            w = w * e * e * (3 - 2 * e)
        S = ndimage.gaussian_filter(S, 0.7)                      # remove ruído residual de pixel isolado
        # luz rebatida pelos móveis pode clarear o entorno; em cantos escuros a razão do
        # shadow catcher explode (denominador pequeno) — limitamos o ganho a +15%
        S = np.minimum(S, 1.15)
        dev = S - 1.0
        # encolhimento suave: desvios menores que eps (ruído de render) viram exatamente 0
        dev = np.sign(dev) * np.maximum(np.abs(dev) - self.s.shadow_eps, 0.0)
        w = np.where(valid, w, 0.0)
        S_eff = np.where(w * np.abs(dev) > 0, 1.0 + dev * w, 1.0).astype(np.float32)

        # --- casamento com a câmera: nitidez e grão só na camada CG
        if self.s.match_sharpness and self.plate.blur_sigma_px > 0.05 and obj.any():
            sig = self.plate.blur_sigma_px
            C = np.stack([ndimage.gaussian_filter(C[..., i], sig) for i in range(3)], -1)
            A_b = ndimage.gaussian_filter(A, sig)
            # mantém a máscara de influência coerente com o alfa desfocado
            A = np.where(A_b > self.s.alpha_eps, A_b, 0.0).astype(np.float32)
            C = np.where(A[..., None] > 0, C, 0.0)

        out_lin = self.photo_lin * S_eff[..., None] * (1.0 - A[..., None]) + C
        out = linear_to_srgb(out_lin)
        if self.s.grain_strength > 0 and self.plate.noise_sigma > 0:
            g = self.plate.grain * (self.plate.noise_sigma * self.s.grain_strength)
            out = out + g * A[..., None]          # grão proporcional à cobertura do móvel
        out8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        # --- preservação estrita: fora de (móvel ∪ sombra) copia os bytes originais
        touched = (A > 0) | (S_eff != 1.0)
        out8[~touched] = self.photo8[~touched]
        diff = np.abs(out8.astype(np.int16) - self.photo8.astype(np.int16)).max(-1)
        st.object_pixels = int((A > 0).sum())
        st.shadow_pixels = int(((S_eff != 1.0) & ~(A > 0)).sum())
        st.changed_pixels = int((diff > 0).sum())
        st.max_diff_outside = int(diff[~touched].max()) if (~touched).any() else 0
        st.preserved_exact = st.max_diff_outside == 0
        return out8, st


class LayerAccumulator:
    """Mantém as camadas 3D de tela cheia quando o render é incremental.

    Quadros "full" substituem tudo; quadros "incremental" só re-renderizam a região das
    peças em movimento, que é colada sobre o estado anterior com uma transição suave
    (as duas renders mostram a mesma cena estática fora das peças em movimento).
    """

    def __init__(self, height: int, width: int, feather_px: int = 12):
        self.H, self.W = height, width
        self.feather = max(1, int(feather_px))
        self.reset()

    def reset(self) -> None:
        self.rgba = np.zeros((self.H, self.W, 4), np.float32)
        self.shadow = np.ones((self.H, self.W, 3), np.float32)
        self.empty = True

    def _weights(self, border: list[int]) -> tuple[tuple[slice, slice], np.ndarray]:
        x0, y0, x1, y1 = [int(v) for v in border]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.W, x1), min(self.H, y1)
        h, w = max(0, y1 - y0), max(0, x1 - x0)
        f = float(self.feather)
        # rampa só nas bordas internas da região (na borda da imagem não há o que misturar)
        yy = np.arange(h, dtype=np.float32) + 0.5
        xx = np.arange(w, dtype=np.float32) + 0.5
        dy = np.full(h, np.inf, np.float32)
        dx = np.full(w, np.inf, np.float32)
        if y0 > 0:
            dy = np.minimum(dy, yy)
        if y1 < self.H:
            dy = np.minimum(dy, h - yy)
        if x0 > 0:
            dx = np.minimum(dx, xx)
        if x1 < self.W:
            dx = np.minimum(dx, w - xx)
        wy = np.clip(dy / f, 0.0, 1.0)
        wx = np.clip(dx / f, 0.0, 1.0)
        return (slice(y0, y1), slice(x0, x1)), np.outer(wy, wx).astype(np.float32)

    def update(self, rgba: np.ndarray, shadow: np.ndarray | None, border: list[int] | None, mode: str) -> None:
        if shadow is None:
            shadow = np.ones(rgba.shape[:2] + (3,), np.float32)
        if border is None:
            border = [0, 0, self.W, self.H]
        sl, wgt = self._weights(border)
        if mode != "incremental" or self.empty:
            self.reset()
            self.rgba[sl] = rgba[sl]
            self.shadow[sl] = shadow[sl]
            self.empty = False
            return
        w4 = wgt[..., None]
        self.rgba[sl] = self.rgba[sl] * (1 - w4) + rgba[sl] * w4
        self.shadow[sl] = self.shadow[sl] * (1 - w4) + shadow[sl] * w4


def load_render_passes(exr_path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Lê Combined (RGBA) e Shadow Catcher (RGB) de um EXR multi-part do Blender 5."""
    import OpenEXR

    f = OpenEXR.File(str(exr_path), separate_channels=True)
    chs = {}
    for part in f.parts:
        for name, c in part.channels.items():
            chs[name] = np.asarray(c.pixels, dtype=np.float32)
    comb = [k for k in chs if k.endswith("Combined.R")][0][:-2]
    rgba = np.stack([chs[comb + ".R"], chs[comb + ".G"], chs[comb + ".B"], chs[comb + ".A"]], -1)
    sk = [k for k in chs if "Shadow Catcher" in k and k.endswith(".R")]
    shadow = None
    if sk:
        b = sk[0][:-2]
        shadow = np.stack([chs[b + ".R"], chs[b + ".G"], chs[b + ".B"]], -1)
    return rgba, shadow
