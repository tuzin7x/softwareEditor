"""Arte abstrata procedural para os quadros de parede (só numpy, sem modelos nem bancos de imagem).

A pintura é GERADA localmente a partir da paleta da sala e de uma semente: nada é baixado e não
há risco de licença (a obra é do próprio programa). Determinística: mesma entrada -> mesmos bytes.

Estilos (``ART_STYLES``):
- ``formas``  manchas orgânicas sobrepostas (modernismo de meados do século): contorno de pincel
              levemente irregular, tinta com veladura e marcas de cerda;
- ``campos``  campos de cor empilhados com bordas esfumadas (pintura de "color field");
- ``arcos``   arcos concêntricos e um disco (abstrato geométrico boho), traço à mão.

``generate_art`` devolve a imagem sRGB (uint8, altura x largura x 3) e um mapa de relevo da tinta
(float32 0..1, empastamento + marcas de pincel) usado como *bump* no material da tela. A trama da
tela em si fica no shader (escala física fixa), não na imagem.

Este módulo não importa ``bpy``: roda no layout, nos testes e dentro do Blender.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

ART_STYLES: tuple[str, ...] = ("formas", "campos", "arcos")

# Resolução da imagem: ~900 px por metro de pintura (um quadro de 0,8 m ocupa poucas centenas de
# pixels mesmo numa foto de 4K), lado maior limitado para gerar em ~0,1–0,3 s.
ART_PX_PER_M = 900.0
ART_MAX_SIDE = 1536
ART_MIN_SIDE = 192

Color = Sequence[float]


def art_resolution(width_m: float, height_m: float, px_per_m: float = ART_PX_PER_M,
                   max_side: int = ART_MAX_SIDE, min_side: int = ART_MIN_SIDE) -> tuple[int, int]:
    """(largura, altura) em pixels da imagem da pintura, na proporção exata da área pintada."""
    w, h = float(width_m), float(height_m)
    if w <= 0 or h <= 0:
        raise ValueError(f"Área pintada inválida: {w} x {h} m")
    s = px_per_m
    if max(w, h) * s > max_side:
        s = max_side / max(w, h)
    if min(w, h) * s < min_side:
        s = min(min_side / min(w, h), max_side / max(w, h))
    return max(8, int(round(w * s))), max(8, int(round(h * s)))


# --------------------------------------------------------------------------- utilidades


def _rgb(c: Color | str) -> np.ndarray:
    if isinstance(c, str):
        s = c.strip().lstrip("#")
        return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.float64) / 255.0
    v = np.asarray([float(x) for x in list(c)[:3]], np.float64)
    if v.max() > 1.0:
        v = v / 255.0
    return np.clip(v, 0.0, 1.0)


def _smooth(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _value_noise(h: int, w: int, cell_y: float, cell_x: float, rng: np.random.Generator) -> np.ndarray:
    """Ruído de valor (0..1) com células de ``cell_y`` x ``cell_x`` pixels, interpolação suave."""
    gy = int(math.ceil(h / cell_y)) + 2
    gx = int(math.ceil(w / cell_x)) + 2
    g = rng.random((gy, gx))
    y = np.arange(h) / cell_y
    x = np.arange(w) / cell_x
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    ty = y - y0
    tx = x - x0
    ty = (ty * ty * (3 - 2 * ty))[:, None]
    tx = (tx * tx * (3 - 2 * tx))[None, :]
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def _fbm(h: int, w: int, cell: float, rng: np.random.Generator, octaves: int = 3,
         aspect: float = 1.0) -> np.ndarray:
    """Soma de oitavas de ruído de valor, normalizada para 0..1. ``aspect`` > 1 alonga em x."""
    acc = np.zeros((h, w))
    amp, tot = 1.0, 0.0
    for _ in range(octaves):
        acc += amp * _value_noise(h, w, max(1.0, cell), max(1.0, cell * aspect), rng)
        tot += amp
        amp *= 0.5
        cell /= 2.0
    return acc / tot


def _brush(h: int, w: int, rng: np.random.Generator, scale: float, angle: float) -> np.ndarray:
    """Marcas de cerda: ruído muito alongado na direção ``angle`` (rad), média 0, ~±1."""
    if abs(math.sin(angle)) > 0.7:          # pinceladas verticais: gera transposto
        n = _fbm(w, h, max(1.5, 0.004 * scale), rng, octaves=2, aspect=18.0).T
    else:
        n = _fbm(h, w, max(1.5, 0.004 * scale), rng, octaves=2, aspect=18.0)
    return (n - 0.5) * 2.0


class _Canvas:
    """Acumula camadas de tinta (cor sRGB linear-ish 0..1) e o relevo."""

    def __init__(self, h: int, w: int, ground: np.ndarray, rng: np.random.Generator):
        self.h, self.w, self.rng = h, w, rng
        self.scale = float(max(h, w))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        self.x = (xx + 0.5) / self.scale            # coordenadas normalizadas pelo lado maior
        self.y = (yy + 0.5) / self.scale
        mott = _fbm(h, w, 0.18 * self.scale, rng, octaves=3) - 0.5
        self.rgb = np.clip(ground[None, None, :] * (1.0 + 0.05 * mott[..., None]), 0, 1)
        self.relief = 0.15 + 0.1 * _fbm(h, w, 0.05 * self.scale, rng, octaves=2)

    def paint(self, alpha: np.ndarray, color: np.ndarray, opacity: float = 0.92, streak: float = 0.06,
              angle: float = 0.0, thick: float = 0.35) -> None:
        """Pinta ``color`` com cobertura ``alpha`` (0..1): veladura + marcas de pincel + empastamento."""
        br = _brush(self.h, self.w, self.rng, self.scale, angle)
        k = 1.0 + streak * br
        col = np.clip(color[None, None, :] * k[..., None], 0, 1)
        a = np.clip(alpha * opacity * (1.0 - 0.12 * np.clip(-br, 0, 1)), 0, 1)[..., None]
        self.rgb = self.rgb * (1 - a) + col * a
        self.relief = np.maximum(self.relief, (0.3 + thick * (0.5 + 0.5 * br)) * alpha)


def _edge_alpha(d: np.ndarray, soft: float, rough: np.ndarray, amount: float) -> np.ndarray:
    """Cobertura de uma forma pela distância normalizada ``d`` (1 = contorno) com borda de pincel."""
    return _smooth(1.0 + soft, 1.0 - soft, d + amount * (rough - 0.5))


# --------------------------------------------------------------------------- estilos


def _formas(cv: _Canvas, colors: list[np.ndarray]) -> None:
    rng = cv.rng
    ax, ay = cv.w / cv.scale, cv.h / cv.scale
    n = int(rng.integers(4, 7))
    rough = _fbm(cv.h, cv.w, 0.012 * cv.scale, rng, octaves=2)
    order = rng.permutation(len(colors))
    # composição em diagonal: centros ao longo de uma curva suave, com folga até as bordas
    t = np.sort(rng.uniform(0.0, 1.0, n))
    for i in range(n):
        cxp = ax * (0.12 + 0.76 * t[i]) + rng.normal(0, 0.06 * ax)
        cyp = ay * (0.25 + 0.5 * rng.random()) + 0.18 * ay * math.sin(2.8 * t[i] + rng.random())
        r = (0.16 + 0.14 * rng.random()) * min(ax, ay) * (1.25 if i == 0 else 1.0)
        ex = 1.0 + 0.45 * rng.random()
        th = rng.uniform(0, math.pi)
        c, s = math.cos(th), math.sin(th)
        dx, dy = cv.x - cxp, cv.y - cyp
        u = (c * dx + s * dy) / (r * ex)
        v = (-s * dx + c * dy) / r
        phi = np.arctan2(v, u)
        wob = sum(rng.uniform(0.0, 0.05) * np.cos(k * phi + rng.uniform(0, 6.3)) for k in (2, 3, 5))
        d = np.hypot(u, v) * (1.0 + wob)
        alpha = _edge_alpha(d, 0.012, rough, 0.04)
        col = colors[order[i % len(colors)]]
        cv.paint(alpha, col, opacity=float(rng.uniform(0.78, 0.95)), streak=0.07, angle=th)


def _campos(cv: _Canvas, colors: list[np.ndarray]) -> None:
    rng = cv.rng
    ax, ay = cv.w / cv.scale, cv.h / cv.scale
    n = int(rng.integers(2, 4))
    m = 0.08 * min(ax, ay)                       # margem até a borda da tela
    gap = 0.035 * ay
    hs = rng.uniform(0.7, 1.3, n)
    hs = (ay - 2 * m - gap * (n - 1)) * hs / hs.sum()
    y0 = m
    rough = _fbm(cv.h, cv.w, 0.02 * cv.scale, rng, octaves=3)
    order = rng.permutation(len(colors))
    for i in range(n):
        y1 = y0 + hs[i]
        cy, hy = (y0 + y1) / 2, (y1 - y0) / 2
        hx = ax / 2 - m * rng.uniform(0.8, 1.3)
        # distância "retangular" suave (bordas esfumadas, cantos arredondados)
        du = np.abs(cv.x - ax / 2) / hx
        dv = np.abs(cv.y - cy) / hy
        d = (du ** 8 + dv ** 8) ** (1 / 8)
        alpha = _edge_alpha(d, 0.035, rough, 0.08)
        col = colors[order[i % len(colors)]]
        # leve degradê interno (luz da pintura): mais denso no centro do campo
        k = 1.0 - 0.06 * np.clip(dv, 0, 1)
        cv.paint(alpha * k, col, opacity=float(rng.uniform(0.85, 0.96)), streak=0.08, angle=0.0, thick=0.25)
        y0 = y1 + gap


def _arcos(cv: _Canvas, colors: list[np.ndarray]) -> None:
    rng = cv.rng
    ax, ay = cv.w / cv.scale, cv.h / cv.scale
    rough = _fbm(cv.h, cv.w, 0.01 * cv.scale, rng, octaves=2)
    base_y = ay * rng.uniform(0.72, 0.84)
    cxp = ax * rng.uniform(0.38, 0.62)
    R = min(0.33 * ax, 0.55 * ay)
    n = int(rng.integers(3, 5))
    band = R / (n + 0.6)
    order = rng.permutation(len(colors))
    dist = np.hypot(cv.x - cxp, cv.y - base_y)
    above = _smooth(base_y + 0.004, base_y - 0.004, cv.y + 0.006 * (rough - 0.5))
    for i in range(n):
        ro, ri = R - i * band, R - (i + 0.82) * band
        if ri <= 0:
            ri = 0.0
        mid, half = (ro + ri) / 2, (ro - ri) / 2
        d = np.abs(dist - mid) / max(half, 1e-6)
        alpha = _edge_alpha(d, 0.04, rough, 0.08) * above
        cv.paint(alpha, colors[order[i % len(colors)]], opacity=0.93, streak=0.06,
                 angle=float(rng.uniform(0, math.pi)))
    # disco ("sol") no alto, deslocado do eixo dos arcos
    sx = ax * (0.2 if cxp > ax / 2 else 0.8) + rng.normal(0, 0.03 * ax)
    sy = ay * rng.uniform(0.18, 0.3)
    sr = 0.1 * min(ax, ay) * rng.uniform(0.8, 1.2)
    d = np.hypot(cv.x - sx, cv.y - sy) / sr
    cv.paint(_edge_alpha(d, 0.03, rough, 0.06), colors[order[-1]], opacity=0.9, streak=0.05, angle=0.5)


_STYLE_FUNCS = {"formas": _formas, "campos": _campos, "arcos": _arcos}


# --------------------------------------------------------------------------- API


def generate_art(width_px: int, height_px: int, colors: Sequence[Color | str], ground: Color | str,
                 style: str = "formas", seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Gera a pintura: (imagem sRGB uint8 HxWx3, relevo float32 HxW em 0..1).

    ``colors`` = cores de tinta (sRGB 0..1, 0..255 ou "#RRGGBB"), ``ground`` = fundo da tela.
    Mesmos argumentos -> mesmos bytes (``numpy.random.default_rng(seed)``).
    """
    if style not in _STYLE_FUNCS:
        raise ValueError(f"Estilo de arte desconhecido: {style!r} (válidos: {', '.join(ART_STYLES)})")
    w, h = int(width_px), int(height_px)
    if w < 8 or h < 8:
        raise ValueError(f"Resolução de arte inválida: {w} x {h}")
    cols = [_rgb(c) for c in colors] or [np.array([0.45, 0.4, 0.35])]
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    cv = _Canvas(h, w, _rgb(ground), rng)
    _STYLE_FUNCS[style](cv, cols)
    img = np.clip(np.round(cv.rgb * 255.0), 0, 255).astype(np.uint8)
    relief = np.clip(cv.relief, 0, 1).astype(np.float32)
    return img, relief


def crop_to_aspect(img: np.ndarray, aspect: float) -> np.ndarray:
    """Recorta (centrado) uma imagem para a proporção largura/altura ``aspect`` — nunca estica."""
    h, w = img.shape[:2]
    if w / h > aspect:
        nw = max(1, int(round(h * aspect)))
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    nh = max(1, int(round(w / aspect)))
    y0 = (h - nh) // 2
    return img[y0:y0 + nh]
