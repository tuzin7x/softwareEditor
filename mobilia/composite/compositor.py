"""Composição exata das camadas 3D sobre a foto original.

Fórmula (espaço linear, alfa pré-multiplicado do Cycles):
    saída = foto · S' · (1 − A) + C
onde C/A = móveis (Combined RGBA) e S' = fator de sombra (passe Shadow Catcher)
restrito à vizinhança dos móveis e suavizado na borda dessa região.

GARANTIA: todo pixel com A == 0 e S' == 1 recebe os BYTES ORIGINAIS da foto
(cópia direta, sem reconversão de cor). O compositor registra quantos pixels foram alterados
(móveis + sombras de contato); a conferência de cada quadro contra a foto é independente desta
conta (``verify.py``: ``array_equal`` fora da máscara de influência).

RECORTE (PT5): a conta só é feita no retângulo dos pixels com móvel ampliado pelo raio de
influência e pelos raios dos dois desfoques gaussianos (``Compositor.crop_pad``). Fora dele o
resultado de quadro inteiro já seria a foto (sombra com peso 0 e alfa 0), e dentro dele cada
valor intermediário é o mesmo do quadro inteiro: o resultado é idêntico byte a byte (testado).

AJUSTE DO CG (PT8, ``grade.py``): mapa de ganho residual, curva de tom e joelho das altas luzes
multiplicam só C onde A > 0, antes do casamento de nitidez. A placa e a sombra não mudam.

SOMBRA DE CONTATO (PT10, ``contact.py``): a oclusão analítica ``c`` das silhuetas de baixo das
peças de piso, levada à imagem pela homografia do piso, entra como ``S' = S · (1 − k·c)`` antes do
peso da região de influência: fora dela nada muda (mesma garantia da placa).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..photo import linear_to_srgb, srgb_to_linear
from .grade import CGGrade


# desfoque do fator de sombra contra ruído de pixel isolado (raio do núcleo: 3 px)
SHADOW_SIGMA = 0.7


def _channel_mean(x: np.ndarray) -> np.ndarray:
    """``x.mean(-1).astype(float32)`` para 3 canais, bit a bit igual e ~5× mais rápido (a redução
    no eixo curto do NumPy soma em ordem e divide por 3; em float32 isso é exato)."""
    if x.dtype == np.float32 and x.shape[-1] == 3:
        return (x[..., 0] + x[..., 1] + x[..., 2]) / np.float32(3)
    return x.mean(-1).astype(np.float32)


def gauss_radius(sigma: float, truncate: float = 4.0) -> int:
    """Raio do núcleo do ``ndimage.gaussian_filter`` (a mesma conta do SciPy)."""
    return int(truncate * float(sigma) + 0.5) if sigma > 0 else 0


@dataclass
class CompositeSettings:
    shadow_eps: float = 0.006          # |S−1| abaixo disso é ruído de render -> 1
    influence_radius_frac: float = 0.10   # raio da vizinhança (fração da diagonal) em volta dos móveis
    feather_px_frac: float = 0.012     # suavização da borda da região de sombra
    alpha_eps: float = 1.0 / 1024.0    # alfa abaixo disso é considerado 0
    grain_strength: float = 1.0        # 1 = igual ao ruído estimado da foto
    match_sharpness: bool = True
    contact_strength: float = 0.65     # k da sombra de contato (S' = S · (1 − k·c)); 0 desliga


@dataclass
class FrameStats:
    frame: int
    changed_pixels: int = 0
    object_pixels: int = 0
    shadow_pixels: int = 0
    preserved_exact: bool = True
    max_diff_outside: int = 0
    contact_pixels: int = 0           # pixels em que a sombra de contato (PT10) entrou
    box: list[int] | None = None      # retângulo composto [x0, y0, x1, y1]; fora dele, a foto


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
    @property
    def influence_px(self) -> float:
        """Raio (px) da vizinhança dos móveis em que a sombra pode mudar a foto."""
        return max(8.0, self.s.influence_radius_frac * self.diag)

    @property
    def sharpness_sigma(self) -> float:
        """Desfoque aplicado ao CG para casar a nitidez da foto (0 = nenhum)."""
        sig = self.plate.blur_sigma_px
        return float(sig) if self.s.match_sharpness and sig > 0.05 else 0.0

    @property
    def crop_pad(self) -> int:
        """Margem do recorte em volta da caixa dos pixels com móvel.

        Um pixel com peso de sombra > 0 está a menos de ``influence_px`` de um móvel; o
        gaussiano da sombra (3 px) e o da nitidez só leem vizinhos dentro do seu raio. Com
        essa margem nenhum valor que importa sente a borda do recorte (+2 px de folga)."""
        return int(math.ceil(self.influence_px)) + gauss_radius(SHADOW_SIGMA) \
            + gauss_radius(self.sharpness_sigma) + 2

    def composition_box(self, alpha: np.ndarray, origin: tuple[int, int] = (0, 0),
                        border: list[int] | None = None) -> list[int] | None:
        """Retângulo [x0, y0, x1, y1] (quadro inteiro) fora do qual a composição é a foto.

        ``alpha`` é o alfa das camadas posicionado em ``origin`` (x, y). Devolve None quando não há
        nenhum pixel de móvel (o quadro é a foto). Com ``border``, o recorte também cobre o border
        mais 1 px: a rampa da borda (distância ao pixel fora do border mais próximo) sai exata."""
        ox, oy = int(origin[0]), int(origin[1])
        h, w = alpha.shape[:2]
        m = alpha > self.s.alpha_eps
        bx = None
        if border is not None:
            bx = [max(0, int(border[0])), max(0, int(border[1])),
                  min(self.W, int(border[2])), min(self.H, int(border[3]))]
            v = np.zeros_like(m)
            v[max(0, bx[1] - oy):max(0, min(h, bx[3] - oy)), max(0, bx[0] - ox):max(0, min(w, bx[2] - ox))] = True
            m &= v
        rows = np.flatnonzero(m.any(1))
        if not rows.size:
            return None
        cols = np.flatnonzero(m.any(0))
        p = self.crop_pad
        box = [max(0, ox + int(cols[0]) - p), max(0, oy + int(rows[0]) - p),
               min(self.W, ox + int(cols[-1]) + 1 + p), min(self.H, oy + int(rows[-1]) + 1 + p)]
        if bx is not None:
            if bx == [0, 0, self.W, self.H]:
                # sem pixel fora do border a transformada de distância depende da origem da matriz
                return [0, 0, self.W, self.H]
            box = [min(box[0], max(0, bx[0] - 1)), min(box[1], max(0, bx[1] - 1)),
                   max(box[2], min(self.W, bx[2] + 1)), max(box[3], min(self.H, bx[3] + 1))]
        return box

    def composite(self, frame: int, rgba: np.ndarray | None, shadow: np.ndarray | None,
                  border: list[int] | None, origin: tuple[int, int] | None = None,
                  grade: CGGrade | None = None, contact=None) -> tuple[np.ndarray, FrameStats]:
        """Compõe um quadro. ``rgba``/``shadow`` são passes do Cycles (float, h×w×4 / h×w×3).

        Sem ``origin`` as camadas são do quadro inteiro. Com ``origin`` = (x, y) elas cobrem só um
        retângulo do quadro (p. ex. o border do render) e fora dele valem RGBA 0 e sombra 1. A
        conta é feita só em ``composition_box``; o resto do quadro é cópia da foto. ``grade``: o
        ajuste do CG da calibração (só a camada CG muda). ``contact``: a sombra de contato do quadro
        (``contact.ContactFrame``, com ``occlusion(box)``), só dentro da região de influência."""
        st = FrameStats(frame)
        if rgba is None:
            return self.photo8.copy(), st
        ox, oy = (0, 0) if origin is None else (int(origin[0]), int(origin[1]))
        h, w = rgba.shape[:2]
        box = self.composition_box(rgba[..., 3], (ox, oy), border)
        if box is None:                      # nenhum pixel de móvel: o quadro inteiro é a foto
            return self.photo8.copy(), st
        x0, y0, x1, y1 = box
        if ox <= x0 and oy <= y0 and x1 <= ox + w and y1 <= oy + h:
            sl = (slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))
            rg = rgba[sl]
            sh = shadow[sl] if shadow is not None else None
        else:                                # o recorte passa das camadas: fundo neutro em volta
            ix0, iy0, ix1, iy1 = max(x0, ox), max(y0, oy), min(x1, ox + w), min(y1, oy + h)
            rg = np.zeros((y1 - y0, x1 - x0, rgba.shape[2]), rgba.dtype)
            rg[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = rgba[iy0 - oy:iy1 - oy, ix0 - ox:ix1 - ox]
            sh = None
            if shadow is not None:
                sh = np.ones((y1 - y0, x1 - x0, shadow.shape[2]), shadow.dtype)
                sh[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = shadow[iy0 - oy:iy1 - oy, ix0 - ox:ix1 - ox]
        sub = self._composite_region(st, rg, sh, border, box, grade, contact)
        out8 = self.photo8.copy()
        out8[y0:y1, x0:x1] = sub
        st.box = list(box)
        return out8, st

    def _composite_region(self, st: FrameStats, rgba: np.ndarray, shadow: np.ndarray | None,
                          border: list[int] | None, box: list[int], grade: CGGrade | None = None,
                          contact=None) -> np.ndarray:
        """A conta da composição no retângulo ``box`` do quadro (camadas já recortadas nele).

        Mesmas operações, na mesma ordem, do quadro inteiro; ``diag`` (raio de influência e
        rampa) continua sendo o do quadro inteiro. O ajuste do CG (``grade``) é por pixel e lê o
        mapa de ganho na posição absoluta do quadro: recortar não muda nada."""
        X0, Y0, X1, Y1 = box
        sl = (slice(Y0, Y1), slice(X0, X1))
        photo8, photo_lin = self.photo8[sl], self.photo_lin[sl]
        grain = self.plate.grain[sl] if self.plate.grain is not None else None
        A = rgba[..., 3].astype(np.float32, copy=False)
        C = rgba[..., :3].astype(np.float32, copy=False)
        S = _channel_mean(shadow) if shadow is not None else np.ones_like(A)
        hr, wr = A.shape
        valid = None                                   # None = tudo válido (sem border)
        if border is not None:
            valid = np.zeros_like(A, dtype=bool)
            x0, y0, x1, y1 = border
            ya, yb = max(0, y0) - Y0, min(self.H, y1) - Y0
            xa, xb = max(0, x0) - X0, min(self.W, x1) - X0
            valid[max(0, ya):max(0, min(hr, yb)), max(0, xa):max(0, min(wr, xb))] = True
        keep = A > self.s.alpha_eps if valid is None else valid & (A > self.s.alpha_eps)
        A = np.where(keep, np.clip(A, 0, 1), 0.0).astype(np.float32, copy=False)
        C = np.where(A[..., None] > 0, np.maximum(C, 0), 0.0).astype(np.float32, copy=False)
        if grade is not None:
            # luz/cor do CG casadas com a foto (mapa de ganho, curva de tom, joelho): só onde A > 0
            C = grade.apply(C, A, box)
        if valid is not None:
            S = np.where(valid, S, 1.0)

        # --- fator de sombra: só na vizinhança dos móveis, com decaimento suave
        obj = A > 0
        feather = max(2.0, self.s.feather_px_frac * self.diag)
        if obj.any():
            r = self.influence_px
            dist_to_obj = ndimage.distance_transform_edt(~obj)
            t = np.clip((r - dist_to_obj) / (0.65 * r), 0.0, 1.0)     # 1 perto do móvel -> 0 no raio r
            w = t * t * (3 - 2 * t)
        else:
            w = np.zeros_like(A)
        if valid is not None:
            edge = ndimage.distance_transform_edt(valid)
            e = np.clip(edge / feather, 0.0, 1.0)
            w = w * e * e * (3 - 2 * e)
        S = ndimage.gaussian_filter(S, SHADOW_SIGMA)             # remove ruído residual de pixel isolado
        # luz rebatida pelos móveis pode clarear o entorno; em cantos escuros a razão do
        # shadow catcher explode (denominador pequeno) — limitamos o ganho a +15%
        S = np.minimum(S, 1.15)
        k = float(self.s.contact_strength)
        if contact is not None and k > 0 and obj.any():
            # sombra de contato analítica (PT10): entra antes do peso da influência, que a zera
            # fora da vizinhança dos móveis
            c = contact.occlusion(box, w > 0)             # fora da influência o peso a zera
            if c is not None:
                cm = c > 1e-4
                if valid is not None:
                    cm &= valid
                st.contact_pixels = int(np.count_nonzero(cm & (w > 0)))
                if cm.any():
                    S = np.where(cm, S * (1.0 - k * np.clip(c, 0.0, 1.0)), S).astype(np.float32)
        dev = S - 1.0
        # encolhimento suave: desvios menores que eps (ruído de render) viram exatamente 0
        dev = np.sign(dev) * np.maximum(np.abs(dev) - self.s.shadow_eps, 0.0)
        if valid is not None:
            w = np.where(valid, w, 0.0)
        S_eff = np.where(w * np.abs(dev) > 0, 1.0 + dev * w, 1.0).astype(np.float32)

        # --- casamento com a câmera: nitidez e grão só na camada CG
        if self.sharpness_sigma > 0 and obj.any():
            sig = self.sharpness_sigma
            C = np.stack([ndimage.gaussian_filter(C[..., i], sig) for i in range(3)], -1)
            A_b = ndimage.gaussian_filter(A, sig)
            # mantém a máscara de influência coerente com o alfa desfocado
            A = np.where(A_b > self.s.alpha_eps, A_b, 0.0).astype(np.float32)
            C = np.where(A[..., None] > 0, C, 0.0)

        out_lin = photo_lin * S_eff[..., None] * (1.0 - A[..., None]) + C
        out = linear_to_srgb(out_lin)
        if self.s.grain_strength > 0 and self.plate.noise_sigma > 0:
            g = grain * (self.plate.noise_sigma * self.s.grain_strength)
            out = out + g * A[..., None]          # grão proporcional à cobertura do móvel
        out8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        # --- preservação estrita: fora de (móvel ∪ sombra) copia os bytes originais
        cg, shaded = A > 0, S_eff != 1.0
        touched = cg | shaded
        np.copyto(out8, photo8, where=~touched[..., None])
        neq = out8 != photo8
        changed = neq[..., 0] | neq[..., 1] | neq[..., 2]
        st.object_pixels = int(np.count_nonzero(cg))
        st.shadow_pixels = int(np.count_nonzero(shaded & ~cg))
        st.changed_pixels = int(np.count_nonzero(changed))
        bad = changed & ~touched
        st.max_diff_outside = (int(np.abs(out8[bad].astype(np.int16) - photo8[bad].astype(np.int16)).max())
                               if bad.any() else 0)
        st.preserved_exact = st.max_diff_outside == 0
        return out8


def border_weights(border: list[int], H: int, W: int, feather: float) -> tuple[tuple[slice, slice], np.ndarray]:
    """Fatia do ``border`` (cortado na imagem) e pesos 0..1 que sobem em ``feather`` px a partir
    das bordas INTERNAS do retângulo (na borda da imagem não há o que misturar)."""
    x0, y0, x1, y1 = [int(v) for v in border]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    h, w = max(0, y1 - y0), max(0, x1 - x0)
    f = float(feather)
    yy = np.arange(h, dtype=np.float32) + 0.5
    xx = np.arange(w, dtype=np.float32) + 0.5
    dy = np.full(h, np.inf, np.float32)
    dx = np.full(w, np.inf, np.float32)
    if y0 > 0:
        dy = np.minimum(dy, yy)
    if y1 < H:
        dy = np.minimum(dy, h - yy)
    if x0 > 0:
        dx = np.minimum(dx, xx)
    if x1 < W:
        dx = np.minimum(dx, w - xx)
    wy = np.clip(dy / f, 0.0, 1.0)
    wx = np.clip(dx / f, 0.0, 1.0)
    return (slice(y0, y1), slice(x0, x1)), np.outer(wy, wx).astype(np.float32)


def feather_shadow(shadow: np.ndarray, border: list[int], H: int, W: int, feather: int) -> np.ndarray:
    """Sombra do recorte ``border`` com a rampa (smoothstep) até o neutro nas bordas internas:
    fora do border a sombra é 1, e sem rampa o fim do border vira um degrau na parede/piso."""
    if feather <= 0:
        return shadow
    _, ws = border_weights(border, H, W, feather)
    ws = (ws * ws * (3 - 2 * ws))[..., None]          # smoothstep
    return 1.0 + (shadow - 1.0) * ws


class LayerAccumulator:
    """Mantém as camadas 3D de tela cheia quando o render é incremental.

    Quadros "full" substituem tudo; quadros "incremental" só re-renderizam a região das
    peças em movimento, que é colada sobre o estado anterior com uma transição suave
    (as duas renders mostram a mesma cena estática fora das peças em movimento).
    """

    def __init__(self, height: int, width: int, feather_px: int = 12, shadow_feather_px: int = 0):
        self.H, self.W = height, width
        self.feather = max(1, int(feather_px))
        # rampa da SOMBRA nas bordas internas do retângulo renderizado: fora dele a sombra
        # é neutra (1); sem rampa, o fim do border vira um degrau visível na parede/piso
        self.shadow_feather = max(0, int(shadow_feather_px))
        self.reset()

    def reset(self) -> None:
        self.rgba = np.zeros((self.H, self.W, 4), np.float32)
        self.shadow = np.ones((self.H, self.W, 3), np.float32)
        self.empty = True

    def _weights(self, border: list[int], feather: float | None = None) -> tuple[tuple[slice, slice], np.ndarray]:
        return border_weights(border, self.H, self.W, float(self.feather if feather is None else feather))

    def update(self, rgba: np.ndarray, shadow: np.ndarray | None, border: list[int] | None, mode: str) -> None:
        if shadow is None:
            shadow = np.ones(rgba.shape[:2] + (3,), np.float32)
        if border is None:
            border = [0, 0, self.W, self.H]
        sl, wgt = self._weights(border)
        if mode != "incremental" or self.empty:
            self.reset()
            self.rgba[sl] = rgba[sl]
            self.shadow[sl] = feather_shadow(shadow[sl], border, self.H, self.W, self.shadow_feather)
            self.empty = False
            return
        w4 = wgt[..., None]
        self.rgba[sl] = self.rgba[sl] * (1 - w4) + rgba[sl] * w4
        self.shadow[sl] = self.shadow[sl] * (1 - w4) + shadow[sl] * w4


def read_render_passes(exr_path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Lê Combined (RGBA) e Shadow Catcher (RGB) de um EXR multi-part do Blender 5, no tamanho
    em que foi gravado (quadro inteiro ou só o recorte do border).

    Aceita o EXR do ``write_still`` ("ViewLayer.Combined.R", com os passes "Noisy ...") e o do
    nó File Output do worker ("Combined.R", só os dois passes usados)."""
    import OpenEXR

    f = OpenEXR.File(str(exr_path), separate_channels=True)
    chs = {}
    for part in f.parts:
        for name, c in part.channels.items():
            if "Noisy" in name:            # passes do denoise: nunca usados (só converte o que entra)
                continue
            chs[name] = c.pixels
    comb = [k for k in chs if k.endswith("Combined.R")][0][:-2]
    rgba = _stack_f32([chs[comb + ".R"], chs[comb + ".G"], chs[comb + ".B"], chs[comb + ".A"]])
    sk = [k for k in chs if k.endswith("Shadow Catcher.R")]
    shadow = None
    if sk:
        b = sk[0][:-2]
        shadow = _stack_f32([chs[b + ".R"], chs[b + ".G"], chs[b + ".B"]])
    return rgba, shadow


def _stack_f32(chans: list[np.ndarray]) -> np.ndarray:
    """Canais (half/float) empilhados em float32 numa passada só (half→float32 é exato)."""
    out = np.empty(chans[0].shape + (len(chans),), np.float32)
    for i, c in enumerate(chans):
        out[..., i] = c
    return out


def paste_crop(rgba: np.ndarray, shadow: np.ndarray | None, border: list[int] | None,
               shape: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray | None]:
    """Cola os passes de um EXR recortado (``use_crop_to_border``) num quadro inteiro ``shape``.

    Fora do recorte fica exatamente o que o EXR de quadro inteiro traria depois de
    ``mask_unrendered``: RGBA = 0 e sombra neutra (1). O recorte tem de ter o tamanho do
    ``border`` (o worker alinha o border aos pixels que o Cycles renderiza, ver
    ``render/border.py``); qualquer outro tamanho é erro, nunca um palpite de posição."""
    h, w = rgba.shape[:2]
    if shape is None or (h, w) == tuple(shape):
        return rgba, shadow
    H, W = int(shape[0]), int(shape[1])
    if border is None:
        raise ValueError(f"EXR {w}×{h} menor que o quadro {W}×{H} e sem border para posicioná-lo")
    x0, y0, x1, y1 = [int(v) for v in border]
    if (y1 - y0, x1 - x0) != (h, w) or x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        raise ValueError(f"EXR recortado {w}×{h} não bate com o border {list(border)} do quadro {W}×{H}")
    full = np.zeros((H, W, 4), np.float32)
    full[y0:y1, x0:x1] = rgba
    sh = None
    if shadow is not None:
        sh = np.ones((H, W, 3), np.float32)
        sh[y0:y1, x0:x1] = shadow
    return full, sh


def load_render_passes(exr_path: str, border: list[int] | None = None,
                       shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Passes do quadro no tamanho do quadro inteiro ``shape`` (altura, largura).

    EXR de quadro inteiro: lido como está. EXR recortado: colado em ``border`` (ver
    ``paste_crop``). Nos dois casos o resultado é o mesmo, byte a byte. Sem ``shape``, devolve o
    tamanho gravado (compatível com o uso antigo)."""
    rgba, shadow = read_render_passes(exr_path)
    # máscara antes de colar: fora do recorte a sombra já entra neutra, igual ao quadro inteiro
    return paste_crop(rgba, mask_unrendered(rgba, shadow), border, shape)


def layers_from_passes(rgba: np.ndarray, shadow: np.ndarray | None, border: list[int] | None,
                       shape: tuple[int, int], shadow_feather: int = 0
                       ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Camadas de um quadro "full" só no retângulo do ``border``: (rgba, sombra, origem (x, y)).

    É exatamente o que ``LayerAccumulator.update(..., "full")`` deixa dentro do border (fora dele:
    RGBA 0 e sombra 1), sem montar buffers do quadro inteiro. ``rgba``/``shadow`` podem ser do
    quadro inteiro ou o recorte do EXR (tamanho do border, ver ``paste_crop``). A máscara dos
    pixels não renderizados (``mask_unrendered``) é aplicada aqui."""
    H, W = int(shape[0]), int(shape[1])
    h, w = rgba.shape[:2]
    if border is None:
        border = [0, 0, W, H]
    if (h, w) == (H, W):
        x0, y0 = max(0, int(border[0])), max(0, int(border[1]))
        x1, y1 = max(x0, min(W, int(border[2]))), max(y0, min(H, int(border[3])))
        rgba = rgba[y0:y1, x0:x1]
        shadow = shadow[y0:y1, x0:x1] if shadow is not None else None
        origin = (x0, y0)
    else:
        x0, y0, x1, y1 = [int(v) for v in border]
        if (y1 - y0, x1 - x0) != (h, w) or x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            raise ValueError(f"EXR recortado {w}×{h} não bate com o border {list(border)} do quadro {W}×{H}")
        origin = (x0, y0)
    shadow = mask_unrendered(rgba, shadow)
    if shadow is None:
        shadow = np.ones(rgba.shape[:2] + (3,), np.float32)
    return rgba, feather_shadow(shadow, border, H, W, int(shadow_feather)), origin


def load_layers(exr_path: str, border: list[int] | None, shape: tuple[int, int], shadow_feather: int = 0
                ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Lê um EXR (recortado ou de quadro inteiro) e devolve as camadas no retângulo do border
    (ver ``layers_from_passes``), prontas para ``Compositor.composite(..., origin=...)``."""
    rgba, shadow = read_render_passes(exr_path)
    return layers_from_passes(rgba, shadow, border, shape, shadow_feather)


def mask_unrendered(rgba: np.ndarray, shadow: np.ndarray | None) -> np.ndarray | None:
    """Pixels fora da região de render (border) chegam com TODOS os canais = 0.

    Um shadow catcher renderizado nunca dá razão exatamente 0 com CG vazio (há sempre luz
    ambiente), então esses pixels são "não renderizados": sombra neutra (1), nunca preta.
    Sem isso, um arredondamento de 1 px entre o retângulo anotado e o que o Cycles
    renderizou vira uma linha escura sobre a foto.
    """
    if shadow is None:
        return None
    blank = (shadow[..., 0] == 0) & (shadow[..., 1] == 0) & (shadow[..., 2] == 0)
    for i in range(rgba.shape[-1]):
        blank &= rgba[..., i] == 0
    if blank.any():
        shadow = shadow.copy()
        shadow[blank] = 1.0
    return shadow
