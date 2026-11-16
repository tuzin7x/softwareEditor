"""Análise de estilo e paleta a partir da foto (leve, determinística, sem modelos extras).

Mede na própria foto:
- piso: material (madeira, carpete, piso frio, cimento) e tom (claro, médio, escuro), usando
  cor em CIELAB e textura numa vista RETIFICADA de cima do piso (a geometria da sala permite
  "desentortar" a perspectiva: tábuas viram linhas paralelas, carpete vira ruído isotrópico);
- paredes: cor dominante e temperatura;
- exposição geral (normaliza a luminosidade pela região mais clara da foto).

Com isso escolhe um estilo coerente (moderno por padrão, escandinavo, clássico ou industrial),
uma paleta que conversa com piso e paredes (tapete, têxteis, madeira e metal dos procedurais)
e ordena os assets do catálogo por estilo e contraste de tom (ex.: sofá escuro em piso claro,
tapete claro em piso escuro).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from ..analysis.room import FREE
from . import catalog
from .catalog import AssetSpec

STYLES = ("moderno", "escandinavo", "classico", "industrial")

STYLE_LABELS = {"moderno": "moderno", "escandinavo": "escandinavo", "classico": "clássico",
                "industrial": "industrial"}
FLOOR_LABELS = {"madeira": "piso de madeira", "carpete": "carpete", "piso_frio": "piso frio (cerâmica/porcelanato)",
                "cimento": "cimento queimado"}

# Afinidade entre o estilo escolhido e os estilos declarados no catálogo (1 = mesmo estilo).
STYLE_COMPAT: dict[str, dict[str, float]] = {
    "moderno": {"moderno": 1.0, "escandinavo": 0.65, "industrial": 0.5, "boho": 0.4, "classico": 0.35,
                "rustico": 0.3},
    "escandinavo": {"escandinavo": 1.0, "moderno": 0.7, "boho": 0.6, "rustico": 0.5, "industrial": 0.3,
                    "classico": 0.25},
    "classico": {"classico": 1.0, "rustico": 0.5, "moderno": 0.4, "escandinavo": 0.3, "boho": 0.3,
                 "industrial": 0.2},
    "industrial": {"industrial": 1.0, "moderno": 0.7, "rustico": 0.5, "escandinavo": 0.3, "boho": 0.3,
                   "classico": 0.2},
}

# Cores de destaque por estilo (sRGB). A escolhida é a que melhor conversa com as paredes.
ACCENTS: dict[str, list[tuple[str, str]]] = {
    "moderno": [("terracota", "#B5603E"), ("azul petróleo", "#2F5D66"), ("oliva", "#7B7D45"),
                ("mostarda", "#C49A3A"), ("grafite", "#3D3F42")],
    "escandinavo": [("sálvia", "#93A684"), ("azul acinzentado", "#8199AB"), ("rosa antigo", "#C49C92"),
                    ("aveia", "#CDBDA2")],
    "classico": [("azul-marinho", "#26385A"), ("bordô", "#7A2E3B"), ("verde-inglês", "#2F4B3A"),
                 ("dourado", "#B08F45")],
    "industrial": [("ferrugem", "#994A2C"), ("conhaque", "#9A5B2E"), ("verde militar", "#5B5F3B"),
                   ("grafite", "#34363A")],
}
RUG_PATTERNS = {"moderno": "geometrico", "escandinavo": "listras", "classico": "persa", "industrial": "borda"}
WOODS = {"nogueira": "#5E4332", "carvalho": "#A47A50", "freijó": "#8C6A48", "carvalho claro": "#CDAE84",
         "laca branca": "#ECE9E3"}
METALS = {"preto": "#1F1F21", "latão": "#B08D57", "ferro grafite": "#3A3B3E", "branco": "#EDEBE6"}


@dataclass
class StyleProfile:
    style: str
    floor_material: str
    floor_tone: str
    floor_rgb: list[float]
    wall_rgb: list[float]
    wall_tone: str
    warmth: float
    brightness: float
    palette: dict
    scores: dict
    features: dict = field(default_factory=dict)

    def describe(self) -> str:
        temp = "quentes" if self.warmth > 0.25 else ("frios" if self.warmth < -0.15 else "neutros")
        piso_tom = {"claro": "claro", "medio": "de tom médio", "escuro": "escuro"}[self.floor_tone]
        parede = {"claro": "claras", "medio": "de tom médio", "escuro": "escuras"}[self.wall_tone]
        return (f"Estilo {STYLE_LABELS[self.style]}: {FLOOR_LABELS[self.floor_material]} {piso_tom}, "
                f"paredes {parede} em tons {temp}; cor de destaque {self.palette['accent_name']}.")

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- cor

def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gam(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb01) -> np.ndarray:
    """sRGB (0..1, (..., 3)) -> CIELAB (D65)."""
    xyz = _lin(rgb01) @ _M_XYZ.T / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def lab_to_rgb(lab) -> np.ndarray:
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WHITE
    return np.clip(_gam(xyz @ np.linalg.inv(_M_XYZ).T), 0, 1)


def _lch(lab) -> tuple[float, float, float]:
    L, a, b = (float(v) for v in lab)
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def _from_lch(L: float, C: float, h_deg: float) -> np.ndarray:
    h = math.radians(h_deg)
    return lab_to_rgb(np.array([L, C * math.cos(h), C * math.sin(h)]))


def _r(c) -> list[float]:
    return [round(float(v), 4) for v in np.asarray(c).ravel()[:3]]


# ----------------------------------------------------------------------------- máscaras

def _analysis_size(photo: np.ndarray, long_side: int = 640) -> tuple[int, int]:
    h, w = photo.shape[:2]
    s = min(1.0, long_side / max(h, w))
    return max(16, int(round(w * s))), max(16, int(round(h * s)))


def _geometric_masks(ctx, aw: int, ah: int) -> tuple[np.ndarray, np.ndarray]:
    """Piso e paredes estimados só pela geometria (quando não há segmentação)."""
    u = (np.arange(aw) + 0.5) * ctx.W / aw
    v = (np.arange(ah) + 0.5) * ctx.H / ah
    uu, vv = np.meshgrid(u, v)
    d = ctx.rays_room(np.column_stack([uu.ravel(), vv.ravel()]))
    o = ctx.cam_pos
    t_floor = np.where(d[:, 2] < -1e-6, -o[2] / np.where(d[:, 2] < -1e-6, d[:, 2], -1), np.inf)
    hit = o[:2] + d[:, :2] * np.where(np.isfinite(t_floor), t_floor, 0)[:, None]
    floor = np.isfinite(t_floor) & (ctx.states(hit) == FREE)
    t_wall = np.full(len(d), np.inf)
    for w in ctx.walls:
        n = w.normal
        den = d[:, :2] @ n
        ok = np.abs(den) > 1e-6
        t = np.where(ok, ((w.p0 - o[:2]) @ n) / np.where(ok, den, 1), np.inf)
        p = o + d * np.where(np.isfinite(t), t, 0)[:, None]
        s = (p[:, :2] - w.p0) @ w.direction
        good = ok & (t > 0) & (s > 0.05) & (s < w.length - 0.05) & (p[:, 2] > 0.15) & (p[:, 2] < ctx.ceiling - 0.15)
        for op in w.openings:
            good &= ~((s > op.s0 - 0.05) & (s < op.s1 + 0.05) & (p[:, 2] > op.z0 - 0.05) & (p[:, 2] < op.z1 + 0.05))
        t_wall = np.where(good & (t < t_wall), t, t_wall)
    wall = np.isfinite(t_wall) & (t_wall < t_floor)
    floor &= ~wall
    return floor.reshape(ah, aw), wall.reshape(ah, aw)


def _masks(photo: np.ndarray, seg, ctx, aw: int, ah: int) -> tuple[np.ndarray, np.ndarray, str]:
    floor = wall = None
    src = "geometria"
    if seg is not None:
        try:
            f = cv2.resize(seg.mask("floor").astype(np.uint8), (aw, ah), interpolation=cv2.INTER_NEAREST) > 0
            w = cv2.resize(seg.mask("wall").astype(np.uint8), (aw, ah), interpolation=cv2.INTER_NEAREST) > 0
            if f.mean() > 0.02:
                floor = f
            if w.mean() > 0.02:
                wall = w
            src = "segmentação"
        except (KeyError, AttributeError):
            pass
    if (floor is None or wall is None) and ctx is not None:
        gf, gw = _geometric_masks(ctx, aw, ah)
        floor = gf if floor is None else floor
        wall = gw if wall is None else wall
    if floor is None:      # sem nada: faixa inferior da imagem como piso
        floor = np.zeros((ah, aw), bool)
        floor[int(ah * 0.7):] = True
    if wall is None:
        wall = np.zeros((ah, aw), bool)
        wall[int(ah * 0.25):int(ah * 0.55)] = True
    # descarta bordas (rodapés, sombras de contato)
    floor = cv2.erode(floor.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    wall = cv2.erode(wall.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return floor, wall, src


# ----------------------------------------------------------------------------- textura do piso

def _rectified_floor(photo: np.ndarray, ctx, seg_floor_full: np.ndarray | None,
                     res: float = 0.012, side: float = 3.6) -> tuple[np.ndarray, np.ndarray] | None:
    """Vista de cima (1 cm/pixel) do trecho de piso livre mais próximo da câmera."""
    cells = ctx.cell_centers()[ctx.free]
    if len(cells) < 40:
        return None
    dist = np.linalg.norm(cells - ctx.cam_xy, axis=1)
    near = cells[dist <= np.percentile(dist, 55)]
    c = near.mean(0)
    half = side / 2
    xs = np.arange(c[0] - half, c[0] + half, res)
    ys = np.arange(c[1] - half, c[1] + half, res)
    gx, gy = np.meshgrid(xs, ys[::-1])
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    uv, z = ctx.project(pts)
    ph, pw = photo.shape[:2]
    sx, sy = pw / ctx.W, ph / ctx.H
    mapx = (uv[:, 0] * sx - 0.5).astype(np.float32).reshape(gx.shape)
    mapy = (uv[:, 1] * sy - 0.5).astype(np.float32).reshape(gx.shape)
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # antisserrilhamento: a textura é medida na mesma escala física (res) qualquer que seja a
    # resolução da foto — sem isso, fotos maiores parecem "mais granuladas" (carpete) que menores.
    zc = z[(z > 0.3)]
    if len(zc):
        gsd = float(np.median(zc)) / (ctx.fx * sx)          # metros por pixel da foto no piso
        sigma = 0.5 * res / max(gsd, 1e-6)
        if sigma > 0.6:
            gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    rect = cv2.remap(gray, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = (z.reshape(gx.shape) > 0.3) & (rect >= 0)
    valid &= (ctx.states(pts[:, :2]) == FREE).reshape(gx.shape)
    if seg_floor_full is not None:
        fm = cv2.remap(seg_floor_full.astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid &= fm > 0
    # afasta ~15 cm das bordas (rodapés e quinas criam linhas retas que imitam tábuas)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    if valid.sum() < 2000:
        return None
    return rect, valid


def _texture_features(gray: np.ndarray, mask: np.ndarray) -> dict:
    med = float(np.median(gray[mask]))
    g = np.where(mask, gray, med).astype(np.float32)
    # remove iluminação (sombras, manchas de sol) e normaliza contraste local
    base = cv2.GaussianBlur(g, (0, 0), 10.0)
    g = (g - base) / (base + 0.05)
    spec = np.abs(g) > 4.0 * float(np.std(g[mask]))      # reflexos/manchas pontuais
    mask = mask & ~(cv2.dilate(spec.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    g = np.where(mask, g, 0.0).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    fine = float(np.std(hp[mask]))
    gs = cv2.GaussianBlur(g, (0, 0), 1.2)
    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    mag = np.hypot(gx, gy)[inner]
    mag = np.minimum(mag, np.percentile(mag, 95))          # poucas arestas fortes não dominam
    ang = (np.degrees(np.arctan2(gy, gx))[inner]) % 180.0
    hist, _ = np.histogram(ang, bins=36, range=(0, 180), weights=mag)
    hist = np.convolve(np.r_[hist[-1], hist, hist[0]], [0.25, 0.5, 0.25], mode="valid")
    k = int(np.argmax(hist))
    aniso = float(hist[k] / (hist.mean() + 1e-9))
    k2 = (k + 18) % 36
    second = float(max(hist[(k2 + j) % 36] for j in (-1, 0, 1)) / (hist[k] + 1e-9))
    # coerência do tensor de estrutura (veios e frestas de tábuas são fortemente orientados)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 4.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 4.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 4.0)
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-9)
    coherence = float(np.median(coh[inner]))
    return dict(fine=round(fine, 4), aniso=round(aniso, 3), second_peak=round(second, 3),
                coherence=round(coherence, 3), main_angle=float(k * 5))


def _classify_floor(tex: dict | None, L: float, C: float, hue: float) -> str:
    """Classifica o material do piso pela textura na vista retificada (escala física fixa de
    1,2 cm/pixel, com antisserrilhamento) e pela cor. Calibrado em fotos reais em 3 resoluções e
    2 qualidades de JPEG: carpete (anisotropia ~1,8, coerência ~0,40, energia fina ~0,008) e
    assoalho de madeira (anisotropia ~3,6, coerência ~0,61, picos de orientação únicos)."""
    woodish_hue = 30.0 <= hue <= 100.0
    if tex is not None:
        fine, aniso, coh, second = tex["fine"], tex["aniso"], tex["coherence"], tex["second_peak"]
        if coh >= 0.5 and aniso >= 2.5 and second < 0.6 and (C >= 5 or woodish_hue):
            return "madeira"                         # tábuas: uma direção dominante
        if aniso >= 1.5 and second >= 0.6 and coh >= 0.3:
            return "piso_frio"                       # rejunte em grade (duas direções)
        if aniso < 2.5:
            if fine >= 0.006 and (C >= 4 or L < 70):
                return "carpete"                     # granulado isotrópico
            return "cimento" if (C < 8 and L < 70) else "piso_frio"
        if C >= 10 and woodish_hue:
            return "madeira"
        return "piso_frio"
    if C >= 12 and woodish_hue:
        return "madeira"
    return "piso_frio" if L > 60 else "cimento"


def _tone(L: float) -> str:
    return "claro" if L >= 50 else ("medio" if L >= 36 else "escuro")


# ----------------------------------------------------------------------------- análise principal

def analyze_style(photo_rgb8: np.ndarray, seg=None, ctx=None) -> StyleProfile:
    """Analisa a foto e devolve estilo + paleta. Determinístico (não usa aleatoriedade)."""
    photo = np.ascontiguousarray(photo_rgb8[..., :3])
    aw, ah = _analysis_size(photo)
    small = cv2.resize(photo, (aw, ah), interpolation=cv2.INTER_AREA)
    lab = rgb_to_lab(small.astype(np.float64) / 255.0)
    floor_m, wall_m, mask_src = _masks(photo, seg, ctx, aw, ah)
    L_all = lab[..., 0]
    L_ref = float(np.clip(np.percentile(L_all, 97), 55.0, 100.0))
    k = 96.0 / L_ref                                   # normalização de exposição

    def stats(mask, pct_L):
        if mask.sum() < 30:
            mask = np.ones_like(mask)
        v = lab[mask]
        Lv = min(100.0, float(np.percentile(v[:, 0], pct_L)) * k)
        a, b = float(np.median(v[:, 1])), float(np.median(v[:, 2]))
        return np.array([Lv, a, b])

    f_lab = stats(floor_m, 50)
    w_lab = stats(wall_m, 70)
    fL, fC, fh = _lch(f_lab)
    wL, wC, wh = _lch(w_lab)

    tex = None
    if ctx is not None:
        seg_floor_full = None
        if seg is not None:
            try:
                sf = seg.mask("floor")
                if sf.mean() > 0.02:      # mesma resolução da FOTO (o remapeamento usa pixels da foto)
                    seg_floor_full = cv2.resize(sf.astype(np.uint8), (photo.shape[1], photo.shape[0]),
                                                interpolation=cv2.INTER_NEAREST) > 0
            except (KeyError, AttributeError):
                seg_floor_full = None
        rr = _rectified_floor(photo, ctx, seg_floor_full)
        if rr is not None:
            tex = _texture_features(*rr)
    material = _classify_floor(tex, fL, fC, fh)
    floor_tone = _tone(fL)
    wall_tone = "claro" if wL >= 70 else ("medio" if wL >= 45 else "escuro")
    warmth = float(np.clip((0.55 * w_lab[2] + 0.25 * f_lab[2] + 0.35 * w_lab[1]) / 18.0 - 0.15, -1, 1))
    brightness = float(np.clip(np.median(L_all) * k / 100.0, 0, 1))

    scores = {s: 0.0 for s in STYLES}
    scores["moderno"] = 1.0
    if material == "madeira":
        if floor_tone == "claro":
            scores["escandinavo"] += 1.6
            scores["moderno"] += 0.3
        elif floor_tone == "escuro":
            scores["classico"] += 1.2
            scores["moderno"] += 0.5
        else:
            scores["moderno"] += 0.5
            scores["escandinavo"] += 0.5
            scores["classico"] += 0.5
    elif material == "cimento":
        scores["industrial"] += 1.8
    elif material == "piso_frio":
        scores["moderno"] += 0.5
        if floor_tone == "escuro":
            scores["industrial"] += 0.8
    elif material == "carpete":
        scores["classico"] += 0.4
        scores["moderno"] += 0.3
    if wL >= 70 and wC < 8:                              # paredes brancas
        scores["escandinavo"] += 0.7
        scores["moderno"] += 0.3
    elif wC < 6 and 40 <= wL < 65:                       # paredes cinza
        scores["industrial"] += 0.6
    if wC >= 18 and 30 <= wh <= 100:
        scores["classico"] += 0.6
    if brightness < 0.4:
        scores["industrial"] += 0.2
        scores["classico"] += 0.2
    style = max(STYLES, key=lambda s: (round(scores[s], 6), s == "moderno"))

    palette = _palette(style, material, floor_tone, f_lab, w_lab, warmth)
    feats = dict(mask_source=mask_src, exposure_ref_L=round(L_ref, 2), floor_lab=_r(f_lab),
                 wall_lab=_r(w_lab), texture=tex)
    return StyleProfile(style=style, floor_material=material, floor_tone=floor_tone,
                        floor_rgb=_r(lab_to_rgb(f_lab)), wall_rgb=_r(lab_to_rgb(w_lab)), wall_tone=wall_tone,
                        warmth=round(warmth, 3), brightness=round(brightness, 3), palette=palette,
                        scores={k2: round(v, 3) for k2, v in scores.items()}, features=feats)


def _palette(style: str, material: str, floor_tone: str, f_lab: np.ndarray, w_lab: np.ndarray,
             warmth: float) -> dict:
    """Paleta harmônica: destaque escolhido pela relação de matiz com as paredes."""
    wL, wC, wh = _lch(w_lab)
    fL, fC, fh = _lch(f_lab)
    best, best_score = None, -1e9
    for i, (name, hx) in enumerate(ACCENTS[style]):
        aL, aC, ah = _lch(rgb_to_lab(hex_to_rgb(hx)))
        warm_acc = 20 <= ah <= 100
        score = -0.05 * i                                     # ordem = preferência do estilo
        if wC >= 8:                                           # parede com cor: harmonia de matiz
            dh = abs((ah - wh + 180) % 360 - 180)
            score += 0.6 if dh < 35 else (0.8 if dh > 150 else 0.0)
        score += 0.35 * (1 if warm_acc else -1) * warmth        # paredes quentes -> destaque quente
        score -= 0.4 * max(0.0, 12 - abs(aL - fL)) / 12          # destaque precisa destacar do piso
        if score > best_score:
            best, best_score = (name, hx), score
    accent = hex_to_rgb(best[1])
    aL, aC, ah = _lch(rgb_to_lab(accent))
    # segunda cor: neutra quente/fria derivada das paredes
    acc2 = _from_lch(min(88.0, max(70.0, wL + 5)), min(10.0, wC * 0.6 + 3), wh if wC > 3 else 70.0)
    # tapete: contraste com o piso (claro sobre escuro; médio/escuro sobre claro)
    if floor_tone == "escuro":
        rug_base = _from_lch(84.0, 6.0, 75.0 if warmth >= 0 else 240.0)
        rug_pat = _from_lch(58.0, min(22.0, aC * 0.5), ah)
    elif floor_tone == "medio":
        rug_base = _from_lch(80.0, 7.0, 75.0 if warmth >= 0 else 240.0)
        rug_pat = _from_lch(45.0, min(24.0, aC * 0.55), ah)
    else:
        rug_base = _from_lch(max(34.0, min(52.0, fL - 22.0)), min(14.0, aC * 0.35), ah)
        rug_pat = _from_lch(min(86.0, fL + 18.0), 6.0, 75.0 if warmth >= 0 else 230.0)
    # madeira dos procedurais: contrasta com o piso
    if material == "madeira" and floor_tone == "claro":
        wood_name = "nogueira" if style != "escandinavo" else "laca branca"
    elif floor_tone == "escuro":
        wood_name = "carvalho claro"
    else:
        wood_name = {"moderno": "nogueira", "escandinavo": "carvalho claro", "classico": "freijó",
                     "industrial": "nogueira"}[style]
    metal_name = {"moderno": "preto", "escandinavo": "branco" if floor_tone != "claro" else "preto",
                  "classico": "latão", "industrial": "ferro grafite"}[style]
    linen = _from_lch(93.0, 3.0, 80.0 if warmth >= 0 else 250.0)
    shade = _from_lch(91.0, 6.0, 80.0)
    textiles = [accent, acc2, _from_lch(max(25.0, aL - 12), aC * 0.7, ah)]
    return dict(
        accent=_r(accent), accent_name=best[0], accent2=_r(acc2),
        textiles=[_r(t) for t in textiles],
        rug=dict(base=_r(rug_base), pattern=_r(rug_pat), pattern_kind=RUG_PATTERNS[style]),
        wood=_r(hex_to_rgb(WOODS[wood_name])), wood_name=wood_name,
        metal=_r(hex_to_rgb(METALS[metal_name])), metal_name=metal_name,
        linen=_r(linen), lamp_shade=_r(shade),
        floor=dict(material=material, tone=floor_tone, rgb=_r(lab_to_rgb(f_lab))),
        wall=dict(rgb=_r(lab_to_rgb(w_lab))),
    )


# ----------------------------------------------------------------------------- escolha de assets

UPHOLSTERY = {"sofa", "armchair", "ottoman", "pillows", "throw"}
_TONE_UPH = {
    "claro": {"escuro": 1.0, "madeira": 0.8, "colorido": 0.7, "neutro": 0.6, "claro": 0.2},
    "medio": {"escuro": 0.75, "claro": 0.75, "neutro": 0.7, "colorido": 0.7, "madeira": 0.5},
    "escuro": {"claro": 1.0, "neutro": 0.8, "colorido": 0.7, "madeira": 0.4, "escuro": 0.2},
}


def asset_preference(spec: AssetSpec, profile: StyleProfile) -> float:
    """Nota de adequação de um asset ao estilo/tons da foto (maior = melhor)."""
    compat = STYLE_COMPAT[profile.style]
    style_score = max(compat.get(s, 0.1) for s in spec.styles) if spec.styles else 0.3
    if spec.role in UPHOLSTERY:
        tone_score = _TONE_UPH[profile.floor_tone].get(spec.tone, 0.5)
    elif profile.floor_material == "madeira":
        tone_score = {"madeira": 0.35, "claro": 0.8, "escuro": 0.8, "neutro": 0.7}.get(spec.tone, 0.6)
    else:
        tone_score = {"madeira": 0.9, "claro": 0.7, "escuro": 0.7, "neutro": 0.6}.get(spec.tone, 0.6)
    return round(2.0 * style_score + 1.0 * tone_score, 4)


def rank_assets(role: str, profile: StyleProfile) -> list[AssetSpec]:
    """Assets do papel ordenados por adequação (desempate estável pelo id)."""
    cands = catalog.by_role(role)
    return sorted(cands, key=lambda a: (-asset_preference(a, profile), a.asset_id))


def proc_params(role: str, profile: StyleProfile, size) -> dict:
    """Parâmetros dos assets procedurais (cores sRGB 0..1 e medidas), coerentes com a paleta.

    Contrato com o gerador procedural (``PlacedItem.params``; ``size`` = tamanho final, igual a
    ``PlacedItem.size``):
    - proc:rug          color, pattern_color, pattern (geometrico|listras|persa|borda), thickness
    - proc:floor_lamp   shade_color, metal_color (sempre apagada)
    - proc:table_lamp   base_color, shade_color, metal_color (sempre apagado)
    - proc:bed          frame_color, linen_color, accent_color, mattress_top (altura do colchão,
                        onde a manta é apoiada), headboard_height, headboard_depth (cabeceira em +Y)
    - proc:desk         top_color, leg_color
    - proc:throw_blanket color
    - proc:curtain      color
    - proc:book_stack   colors (capas de baixo para cima), count
    """
    out = _proc_colors(role, profile, size)
    out["size"] = [round(float(v), 4) for v in size]      # o gerador dimensiona pelo params["size"]
    return out


def _proc_colors(role: str, profile: StyleProfile, size) -> dict:
    P = profile.palette
    if role == "rug":
        return dict(color=P["rug"]["base"], pattern_color=P["rug"]["pattern"], pattern=P["rug"]["pattern_kind"],
                    thickness=0.012)
    if role == "floor_lamp":       # luminárias sempre apagadas: a luz da foto original não muda
        return dict(shade_color=P["lamp_shade"], metal_color=P["metal"])
    if role == "table_lamp":
        return dict(base_color=P["accent2"], shade_color=P["lamp_shade"], metal_color=P["metal"])
    if role == "bed":
        from .surfaces import BED_HEADBOARD_DEPTH, BED_MATTRESS_TOP
        return dict(frame_color=P["wood"], linen_color=P["linen"], accent_color=P["accent"],
                    mattress_top=BED_MATTRESS_TOP, headboard_height=float(size[2]),
                    headboard_depth=BED_HEADBOARD_DEPTH)
    if role == "desk":
        return dict(top_color=P["wood"], leg_color=P["metal"])
    if role == "throw":
        return dict(color=P["accent"])
    if role == "curtain":
        return dict(color=P["linen"])
    if role == "books":            # pilha: neutro profundo embaixo, neutro claro no meio, destaque em cima
        hue = _lch(rgb_to_lab(np.asarray(P["accent2"], np.float64)))[2]
        return dict(colors=[_r(_from_lch(27.0, 5.0, hue)), P["accent2"], P["accent"]], count=3)
    return {}
