"""'Vídeo antes e depois': a foto original vira as versões redesenhadas, com movimento de câmera lento.

Roteiro (30 fps, 10–20 s, H.264 yuv420p pronto para WhatsApp/Instagram):

1. **Antes** — a foto original, com um *push-in* lento (a câmera começa em zoom 1,0).
2. **Revelação** — uma linha vertical clara (com puxador) atravessa a imagem da direita para a esquerda e revela a
   primeira versão; para no meio por um instante (antes | depois lado a lado) e termina a passada.
3. **Versões** — cada versão seguinte entra por *crossfade*, sempre com o movimento Ken Burns (zoom entre 1,0 e
   1,12 e uma panorâmica suave, com aceleração e frenagem suaves).

Como todas as versões são a MESMA sala vista da mesma câmera, a câmera é uma só para o vídeo inteiro (função do
tempo): no *crossfade* e na revelação as duas imagens recebem exatamente o mesmo enquadramento, então paredes,
janelas e portas ficam paradas e só os móveis e materiais se transformam, sem "imagem dupla".

Formatos: ``16x9`` (1920×1080) e ``9x16`` (1080×1920). O quadro é sempre preenchido pela foto, sem barras: a fonte
é recortada para a proporção de saída (no vertical, em volta do centro da imagem puxado para onde o redesenho mais
mudou a foto, e a panorâmica percorre a sala; numa foto 4:3 no 16x9, a panorâmica vertical percorre a altura que o
recorte deixou de fora).

Rótulos opcionais na identidade 7xAi (marfim e latão, fonte Inter): "ANTES / Foto original" embaixo à esquerda,
"DEPOIS / <estilo>" embaixo à direita (acompanham a imagem a que pertencem, então a revelação também "vira" o
rótulo) e a marca 7xAi pequena no canto superior direito.

Desempenho: cada quadro é um ``cv2.warpAffine`` (bilinear, ~1:1 porque as imagens são ampliadas uma vez para a
escala de saída) por imagem visível, mais misturas em regiões pequenas; os quadros vão por *pipe* ao FFmpeg
(``mobilia.video.encoder``: NVENC quando há placa NVIDIA, senão libx264).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

FORMATS = {"16x9": (1920, 1080), "9x16": (1080, 1920)}
_FORMAT_ALIASES = {"16:9": "16x9", "paisagem": "16x9", "horizontal": "16x9", "9:16": "9x16", "retrato": "9x16",
                   "vertical": "9x16", "stories": "9x16", "reels": "9x16"}

IVORY = (244, 239, 231)      # #F4EFE7 (marca)
BRASS = (205, 174, 118)      # #CDAE76
BRASS_HI = (234, 213, 168)   # #EAD5A8
GRAPHITE = (15, 15, 17)      # #0F0F11

MIN_S, MAX_S = 10.0, 20.0
MAX_DESIGNS = 6
FONTS_DIR = Path(__file__).resolve().parents[1] / "ui" / "fonts"


def norm_format(fmt: str) -> str:
    f = (fmt or "16x9").strip().lower()
    f = _FORMAT_ALIASES.get(f, f)
    if f not in FORMATS:
        raise ValueError(f"Formato de vídeo inválido: {fmt!r} (use 16x9 ou 9x16).")
    return f


# ------------------------------------------------------------------------------------------- curvas

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def _hermite(t: float, ts: list[float], vs: list[float]) -> float:
    """Curva cúbica suave pelos pontos (Catmull-Rom com velocidade zero nas pontas = começa e termina devagar)."""
    if t <= ts[0]:
        return vs[0]
    if t >= ts[-1]:
        return vs[-1]
    k = int(np.searchsorted(ts, t, side="right") - 1)
    n = len(ts)

    def tangent(i):
        if i == 0 or i == n - 1:
            return 0.0
        return (vs[i + 1] - vs[i - 1]) / (ts[i + 1] - ts[i - 1])

    h = ts[k + 1] - ts[k]
    s = (t - ts[k]) / h
    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * vs[k] + h10 * h * tangent(k) + h01 * vs[k + 1] + h11 * h * tangent(k + 1)


# ------------------------------------------------------------------------------------------ roteiro

@dataclass
class Timeline:
    """Tempos (s) do roteiro. ``shots[i]`` = (início, fim) do intervalo em que a versão i está 100% visível
    (sem contar as transições); ``fades[i]`` = (início, fim) do crossfade da versão i-1 para a i (i >= 1)."""
    fps: int
    total: float
    intro: tuple[float, float]
    wipe: tuple[float, float]
    wipe_hold: tuple[float, float]
    shots: list[tuple[float, float]]
    fades: list[tuple[float, float] | None]

    @property
    def frames(self) -> int:
        return int(round(self.total * self.fps))


def plan_timeline(n: int, duration: float | None = None, fps: int = 30) -> Timeline:
    """Roteiro para ``n`` versões. Sem ``duration``: 10–20 s conforme ``n`` (~3,5 s por versão)."""
    if n < 1:
        raise ValueError("Precisa de pelo menos uma versão (depois).")
    intro, wipe_a, wipe_hold, wipe_b, fade = 1.8, 1.0, 0.7, 1.0, 1.0
    wipe = wipe_a + wipe_hold + wipe_b
    if duration is None:
        total = float(np.clip(intro + wipe + n * 2.6 + (n - 1) * fade, MIN_S, MAX_S))
    else:
        total = float(np.clip(duration, 6.0, 60.0))
    hold = (total - intro - wipe - (n - 1) * fade) / n
    if hold < 1.2:
        raise ValueError(f"{n} versões não cabem em {total:.0f} s; use menos versões ou um vídeo mais longo.")
    t = intro + wipe
    shots, fades = [], [None]
    shots.append((t, t + hold))
    t += hold
    for _ in range(1, n):
        fades.append((t, t + fade))
        t += fade
        shots.append((t, t + hold))
        t += hold
    w0 = intro
    return Timeline(fps=fps, total=total, intro=(0.0, intro), wipe=(w0, w0 + wipe),
                    wipe_hold=(w0 + wipe_a, w0 + wipe_a + wipe_hold), shots=shots, fades=fades)


# ------------------------------------------------------------------------------------------- imagens

def _load_rgb(src) -> np.ndarray:
    if isinstance(src, np.ndarray):
        a = src
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, 2)
        return np.ascontiguousarray(a[..., :3], dtype=np.uint8)
    from PIL import Image, ImageOps

    p = Path(src)
    if not p.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {src}")
    with Image.open(p) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()


def _cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Recorta no centro para a proporção w:h e redimensiona para w×h."""
    H, W = img.shape[:2]
    if abs(W / H - w / h) > 1e-3:
        if W / H > w / h:
            cw = int(round(H * w / h))
            x0 = (W - cw) // 2
            img = img[:, x0:x0 + cw]
        else:
            ch = int(round(W * h / w))
            y0 = (H - ch) // 2
            img = img[y0:y0 + ch]
    if img.shape[1] != w or img.shape[0] != h:
        down = img.shape[1] > w
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if down else cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(img)


def unique_labels(names: list[str]) -> list[str]:
    """Rótulos repetidos (duas versões do mesmo estilo) ganham "· versão 2", "· versão 3"..."""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        k = n.strip().lower()
        if not k:
            out.append(n)
            continue
        seen[k] = seen.get(k, 0) + 1
        out.append(n if seen[k] == 1 else f"{n} · versão {seen[k]}")
    return out


def _change_center(before: np.ndarray, afters: list[np.ndarray]) -> tuple[float, float]:
    """Centro (0..1) de onde o redesenho mais mudou a foto (os móveis novos), para o recorte inteligente."""
    s = 256 / max(before.shape[:2])
    size = (max(8, int(before.shape[1] * s)), max(8, int(before.shape[0] * s)))
    b = cv2.resize(before, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    acc = np.zeros(b.shape[:2], np.float32)
    for a in afters:
        acc += np.abs(cv2.resize(a, size, interpolation=cv2.INTER_AREA).astype(np.float32) - b).mean(2)
    acc = cv2.GaussianBlur(acc, (0, 0), 6)
    acc = np.maximum(acc - np.percentile(acc, 50), 0)
    if acc.sum() <= 1e-6:
        return 0.5, 0.5
    ys, xs = np.mgrid[0:acc.shape[0], 0:acc.shape[1]]
    return float((acc * xs).sum() / acc.sum() / acc.shape[1]), float((acc * ys).sum() / acc.sum() / acc.shape[0])


# -------------------------------------------------------------------------------------------- câmera

@dataclass
class Camera:
    """Enquadramento no tempo, em coordenadas da imagem-fonte (já ampliada para a escala de saída)."""
    src_w: int
    src_h: int
    out_w: int
    out_h: int
    cx: float
    cy: float
    keys_t: list[float]
    keys_z: list[float]
    keys_px: list[float]
    keys_py: list[float]
    max_pan: tuple[float, float]       # amplitude máxima da panorâmica (px na fonte), x e y
    zoom_max: float = 1.12

    def rect(self, t: float) -> tuple[float, float, float, float]:
        """(x0, y0, largura, altura) do recorte no instante t."""
        z = min(self.zoom_max, max(1.0, _hermite(t, self.keys_t, self.keys_z)))   # a curva pode passar um fio
        cw, ch = self.out_w / z, self.out_h / z
        # a base (z = 1) já é o maior recorte da proporção de saída; com zoom, sobra margem para a panorâmica
        sx = min(self.max_pan[0], self.cx - cw / 2, self.src_w - self.cx - cw / 2)
        sy = min(self.max_pan[1], self.cy - ch / 2, self.src_h - self.cy - ch / 2)
        x = self.cx + _hermite(t, self.keys_t, self.keys_px) * max(0.0, sx)
        y = self.cy + _hermite(t, self.keys_t, self.keys_py) * max(0.0, sy)
        x0 = float(np.clip(x - cw / 2, 0, self.src_w - cw))
        y0 = float(np.clip(y - ch / 2, 0, self.src_h - ch))
        return x0, y0, cw, ch

    def matrix(self, t: float) -> np.ndarray:
        x0, y0, cw, ch = self.rect(t)
        kx, ky = self.out_w / cw, self.out_h / ch
        return np.array([[kx, 0, -x0 * kx], [0, ky, -y0 * ky]], np.float64)


def _camera(tl: Timeline, src_w: int, src_h: int, out_w: int, out_h: int, center: tuple[float, float],
            fmt: str, zoom: float) -> Camera:
    """Câmera única do vídeo: push-in do zoom 1,0 até ``zoom`` durante o antes/revelação/1ª versão, depois respira
    entre ``zoom`` e ~(1+zoom)/2 a cada versão; a panorâmica atravessa a sala devagar (no 9x16, de um lado ao
    outro: sensação de passeio)."""
    # base: o maior recorte com a proporção de saída que cabe na fonte (a fonte foi ampliada para essa escala)
    vertical = fmt == "9x16"
    cx = src_w * (0.5 + (0.75 if vertical else 0.5) * (center[0] - 0.5))
    cy = src_h * (0.5 + 0.35 * (center[1] - 0.5))
    pan_x = 0.10 * src_w if vertical else 0.035 * src_w
    pan_y = 0.03 * src_h
    slack_y = max(0.0, src_h - out_h) / 2          # fonte mais alta que a saída (4:3 no 16x9): folga vertical
    if slack_y > pan_y:
        # a panorâmica vertical mostra parte do teto e do piso que o recorte deixou de fora (sem barras)
        pan_y = min(slack_y, max(pan_y, 0.55 * slack_y))
    if vertical:   # deixa margem dos dois lados para a panorâmica simétrica
        cx = float(np.clip(cx, out_w / 2 + pan_x, src_w - out_w / 2 - pan_x)) if src_w > out_w + 2 * pan_x \
            else src_w / 2
    cx = float(np.clip(cx, out_w / 2, src_w - out_w / 2))
    cy = float(np.clip(cy, out_h / 2, src_h - out_h / 2))
    if slack_y > 0.03 * src_h:                      # centra o recorte para a panorâmica ter os dois lados
        cy = float(np.clip(cy, out_h / 2 + 0.5 * pan_y, src_h - out_h / 2 - 0.5 * pan_y))
    mid = (1.0 + zoom) / 2
    ts, zs = [0.0], [1.0]
    n = len(tl.shots)
    ends = [tl.shots[0][1] + 0.5 * (tl.fades[1][1] - tl.fades[1][0]) if n > 1 else tl.total]
    for i in range(1, n):
        ends.append(tl.shots[i][1] + (0.5 * (tl.fades[i + 1][1] - tl.fades[i + 1][0]) if i + 1 < n else 0.0))
    for i, e in enumerate(ends):
        ts.append(min(e, tl.total))
        zs.append(zoom if i % 2 == 0 else mid)
    # panorâmica: varre de um lado ao outro ao longo do vídeo (monótona = sem vaivém), centrada no ponto
    # "inteligente"; a vertical oscila de leve. No 16x9 a margem lateral nasce do zoom (em z = 1 não há).
    m = len(ts)
    pxs = [float(v) for v in np.linspace(-0.85, 0.85, m)]
    pys = [0.0] + [(-0.4 if i % 2 == 0 else 0.25) for i in range(m - 1)]
    if slack_y > 0.03 * src_h:      # 4:3 no 16x9: desce devagar do teto para o piso ao longo do vídeo
        pys = [float(v) for v in np.linspace(-0.8, 0.8, m)]
    return Camera(src_w, src_h, out_w, out_h, cx, cy, ts, zs, pxs, pys, (pan_x, pan_y), zoom)


# ------------------------------------------------------------------------------------- rótulos/marca

def _font(name: str, size: int):
    from PIL import ImageFont

    for cand in (FONTS_DIR / name, FONTS_DIR / "Inter-Medium.otf"):
        try:
            return ImageFont.truetype(str(cand), size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_block(kicker: str, title: str, scale: float, align: str = "left") -> np.ndarray:
    """Sprite RGBA: chamada pequena em latão (versalete espaçado) + título em marfim, com sombra suave."""
    from PIL import Image, ImageDraw, ImageFilter

    fk = _font("Inter-SemiBold.otf", max(12, int(round(24 * scale))))
    ft = _font("InterDisplay-Medium.otf", max(18, int(round(58 * scale))))
    track = 0.22 * fk.size
    kicker = kicker.upper()

    def kw():
        return sum(fk.getlength(c) for c in kicker) + track * max(0, len(kicker) - 1)

    tw = ft.getlength(title) if title else 0.0
    w = int(math.ceil(max(kw(), tw))) + int(60 * scale)
    gap = int(round(10 * scale))
    kh = fk.size
    h = int(kh * 1.3 + (gap + ft.size * 1.25 if title else 0)) + int(60 * scale)
    pad = int(30 * scale)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow = Image.new("L", (w, h), 0)
    ds = ImageDraw.Draw(shadow)

    def put(draw, fill, dx=0, dy=0):
        x = pad if align == "left" else w - pad - kw()
        for c in kicker:
            draw.text((x + dx, pad + dy), c, font=fk, fill=fill)
            x += fk.getlength(c) + track
        if title:
            tx = pad if align == "left" else w - pad - tw
            draw.text((tx + dx, pad + kh * 1.3 + gap - ft.size * 0.12 + dy), title, font=ft, fill=fill)

    put(ds, 200, 0, int(2 * scale))
    shadow = shadow.filter(ImageFilter.GaussianBlur(5 * scale))
    # tinta: chamada em latão, título em marfim (desenha de novo, cada parte com a sua cor)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    x = pad if align == "left" else w - pad - kw()
    for c in kicker:
        dr.text((x, pad), c, font=fk, fill=BRASS_HI + (255,))
        x += fk.getlength(c) + track
    if title:
        tx = pad if align == "left" else w - pad - tw
        dr.text((tx, pad + kh * 1.3 + gap - ft.size * 0.12), title, font=ft, fill=IVORY + (255,))
    ink = np.asarray(img).astype(np.float32) / 255.0
    sh = np.asarray(shadow).astype(np.float32)[..., None] / 255.0 * 0.55
    a = ink[..., 3:4]
    out_a = a + sh * (1 - a)
    rgb = np.where(out_a > 1e-6, ink[..., :3] * a / np.maximum(out_a, 1e-6), 0.0)
    return np.concatenate([rgb * 255.0, out_a * 255.0], 2).astype(np.uint8)


def _with_shadow(ink: np.ndarray, height: int) -> np.ndarray:
    """Acrescenta margem e sombra suave a um sprite RGBA (alfa reto)."""
    pad = max(4, height // 3)
    a = np.pad(ink.astype(np.float32) / 255.0, ((pad, pad), (pad, pad), (0, 0)))
    sh = cv2.GaussianBlur(a[..., 3], (0, 0), max(1.0, height / 14)) * 0.5
    sh = np.roll(sh, max(1, height // 25), axis=0)[..., None]
    al = a[..., 3:4]
    out_a = al + sh * (1 - al)
    rgb = np.where(out_a > 1e-6, a[..., :3] * al / np.maximum(out_a, 1e-6), 0.0)
    return np.concatenate([rgb * 255.0, out_a * 255.0], 2).astype(np.uint8)


def _wordmark(height: int) -> np.ndarray:
    """Marca 7xAi em RGBA com sombra. Fonte: ``res/marca_7xai.png``, exportada das mesmas formas vetoriais da
    janela e do ícone (``mobilia.ui.brand``; o motor não depende do Qt). Sem o PNG: Qt, e por fim texto Inter."""
    png = Path(__file__).resolve().parent / "res" / "marca_7xai.png"
    try:
        from PIL import Image

        with Image.open(png) as im:
            im = im.convert("RGBA")
            # o PNG tem 2 px de margem em volta das maiúsculas de 128 px
            k = height / (im.height - 4)
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
            return _with_shadow(np.asarray(im), height)
    except OSError:
        pass
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPainter

        from ..ui import brand

        st = dict(brand.HEADER)
        w = int(math.ceil(brand.wordmark_width(height, st["weight"], st["cut"], st["track"]))) + 4
        img = QImage(w, height + 4, QImage.Format_RGBA8888)
        img.fill(Qt.transparent)
        p = QPainter(img)
        brand.paint_wordmark(p, 2, 2, height, **st)
        p.end()
        a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
        return _with_shadow(a[:, :img.width()].copy(), height)
    except Exception:  # noqa: BLE001 - sem Qt: texto
        return _with_shadow(_wordmark_text(height), height)


def _wordmark_text(height: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    f = _font("Inter-SemiBold.otf", int(height * 1.35))
    parts = [("7", IVORY), ("x", BRASS), ("Ai", IVORY)]
    w = int(sum(f.getlength(s) for s, _ in parts)) + 8
    img = Image.new("RGBA", (w, int(height * 1.8)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = 4
    for s, c in parts:
        d.text((x, 0), s, font=f, fill=c + (255,))
        x += f.getlength(s)
    return np.asarray(img).copy()


def _blend_sprite(frame: np.ndarray, sprite: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    """Alfa-mistura ``sprite`` (RGBA) em ``frame`` (RGB uint8, no lugar) com o canto superior esquerdo em (x, y)."""
    if alpha <= 0.0:
        return
    H, W = frame.shape[:2]
    h, w = sprite.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = s[..., 3:4] * (alpha / 255.0)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi + (s[..., :3] - roi) * a + 0.5).astype(np.uint8)


def _divider_sprites(out_h: int, scale: float) -> tuple[np.ndarray, np.ndarray, int]:
    """(faixa RGBA da linha com brilho/sombra, puxador RGBA, meia-largura da faixa)."""
    from PIL import Image, ImageDraw, ImageFilter

    half = int(round(18 * scale))
    xs = np.arange(-half, half + 1, dtype=np.float32)
    lw = max(1.5, 1.6 * scale)                              # meia-largura da linha clara
    core = np.clip(lw + 0.5 - np.abs(xs), 0, 1)              # linha com borda suave (antisserrilhado)
    glow = np.exp(-(xs / (6 * scale)) ** 2) * 0.35           # halo claro bem leve
    shade = np.exp(-(xs / (12 * scale)) ** 2) * 0.28         # sombra que destaca a linha em paredes claras
    a_light = np.clip(core + glow * (1 - core), 0, 1)
    a = np.clip(a_light + shade * (1 - a_light), 0, 1)
    rgb = np.where(a[:, None] > 1e-6, (np.array(IVORY, np.float32)[None] * a_light[:, None]) / np.maximum(a, 1e-6)[:, None], 0)
    strip = np.concatenate([rgb, a[:, None] * 255.0], 1)[None].repeat(out_h, 0).astype(np.uint8)
    # puxador: círculo marfim com duas setas em grafite, desenhado em 4x e reduzido
    r = int(round(30 * scale))
    S = 4
    size = (2 * r + int(24 * scale)) | 1
    big = Image.new("RGBA", (size * S, size * S), (0, 0, 0, 0))
    sh = Image.new("L", (size * S, size * S), 0)
    c = size * S / 2
    ImageDraw.Draw(sh).ellipse([c - r * S, c - r * S + 3 * S * scale, c + r * S, c + r * S + 3 * S * scale], fill=150)
    sh = sh.filter(ImageFilter.GaussianBlur(6 * S * scale))
    d = ImageDraw.Draw(big)
    d.ellipse([c - r * S, c - r * S, c + r * S, c + r * S], fill=IVORY + (255,))
    k = r * S * 0.30
    for sgn in (-1, 1):   # setas ‹ ›
        tip = c + sgn * r * S * 0.62
        base = tip - sgn * k
        d.polygon([(tip, c), (base, c - k), (base, c + k)], fill=GRAPHITE + (255,))
    big = Image.alpha_composite(Image.merge("RGBA", (*Image.new("RGB", big.size, (0, 0, 0)).split(), sh)), big)
    knob = np.asarray(big.resize((size, size), Image.LANCZOS)).copy()
    return strip, knob, half


# --------------------------------------------------------------------------------------------- vídeo

@dataclass
class ShowcaseResult:
    output: str
    size: tuple[int, int]
    frames: int
    fps: int
    duration_s: float
    labels: list[str]
    codec: str
    timings: dict = field(default_factory=dict)

    def to_event(self) -> dict:
        return {"event": "done", "output": self.output, "size": list(self.size), "frames": self.frames,
                "fps": self.fps, "duration_s": round(self.duration_s, 2), "labels": self.labels,
                "codec": self.codec, "s": self.timings.get("total_s")}


def infer_label(path) -> str:
    """Nome do estilo para o rótulo "Depois", a partir do design.json da pasta ou do nome do arquivo."""
    if not isinstance(path, (str, Path)):
        return ""
    try:
        from ..design import presets

        cat = presets.catalog()
    except Exception:  # noqa: BLE001
        return ""
    p = Path(path)
    rep = p.parent / "design.json"
    if rep.is_file():
        try:
            d = json.loads(rep.read_text(encoding="utf-8"))
            if d.get("funcao") == "evento" and d.get("evento") in cat["eventos"]:
                return cat["eventos"][d["evento"]]
            if d.get("estilo") in cat["estilos"]:
                return cat["estilos"][d["estilo"]]
        except (OSError, ValueError):
            pass
    stem = p.stem.lower()
    for key in sorted(cat["estilos"], key=len, reverse=True):
        if f"_{key}_" in f"_{stem}_":
            return cat["estilos"][key]
    for key in sorted(cat["eventos"], key=len, reverse=True):
        if f"_{key}_" in f"_{stem}_":
            return cat["eventos"][key]
    return ""


class Showcase:
    """Gera os quadros do vídeo (``frame(i)``) e o MP4 (``render``)."""

    def __init__(self, before, afters: list, fmt: str = "16x9", fps: int = 30, duration: float | None = None,
                 labels: list[str] | None = None, show_labels: bool = True, brand: bool = True, zoom: float = 1.12):
        self.fmt = norm_format(fmt)
        self.W, self.H = FORMATS[self.fmt]
        self.fps = int(fps)
        afters = list(afters)
        if not afters:
            raise ValueError("Escolha pelo menos uma versão (depois) para o vídeo.")
        names = list(labels) if labels is not None else [infer_label(a) for a in afters]
        names = unique_labels((names + [""] * len(afters))[:len(afters)])
        self.dropped = max(0, len(afters) - MAX_DESIGNS)
        afters, names = afters[:MAX_DESIGNS], names[:MAX_DESIGNS]
        self.labels = names
        self.tl = plan_timeline(len(afters), duration, self.fps)
        imgs = [_load_rgb(a) for a in afters]
        h0, w0 = imgs[0].shape[:2]
        imgs = [_cover(a, w0, h0) for a in imgs]
        bef = _cover(_load_rgb(before), w0, h0)
        center = _change_center(bef, imgs)
        # o quadro é SEMPRE preenchido pela foto (nada de laterais escuras/desfocadas): numa fonte mais "alta" que
        # a saída (4:3 no 16x9) o recorte perde altura, e a câmera percorre essa folga devagar (ver ``_camera``)
        # escala da fonte: o maior recorte com a proporção de saída cabe exatamente no quadro de saída (≈1:1)
        k = max(self.W / w0, self.H / h0)
        sw, sh = int(math.ceil(w0 * k)), int(math.ceil(h0 * k))
        self.src = [self._prep(x, sw, sh, k) for x in [bef] + imgs]
        self.cam = _camera(self.tl, sw, sh, self.W, self.H, center, self.fmt, zoom)
        self.show_labels = show_labels
        self.scale = min(self.W, self.H) / 1080.0
        self._decor(names, brand)

    @staticmethod
    def _prep(img: np.ndarray, sw: int, sh: int, k: float) -> np.ndarray:
        if k > 1.0:
            out = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LANCZOS4)
            if k > 1.25:   # ampliação grande (versão de 768–1024 px em 1080p): nitidez leve, sem halo
                blur = cv2.GaussianBlur(out, (0, 0), 0.8 * k)
                out = cv2.addWeighted(out, 1.35, blur, -0.35, 0)
            return out
        return cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

    def _decor(self, names: list[str], brand: bool) -> None:
        s = self.scale
        vertical = self.fmt == "9x16"
        self.lab_before = _text_block("Antes", "Foto original", s, "left")
        self.lab_after = [_text_block("Depois", n, s, "right") for n in names]
        mx = int(round((40 if vertical else 56) * s))
        bottom = int(round(self.H * (0.20 if vertical else 0.035)))   # 9x16: acima da faixa de botões do Reels
        self.pos_before = (mx - int(30 * s), self.H - bottom - self.lab_before.shape[0])
        self.pos_after = [(self.W - mx + int(30 * s) - a.shape[1], self.H - bottom - a.shape[0])
                          for a in self.lab_after]
        # escurecimento suave atrás dos rótulos (base da imagem), para ler em qualquer foto
        gh = int(self.H * (0.42 if vertical else 0.34))
        y1 = self.H - bottom + int(20 * s)
        y0 = max(0, y1 - gh)
        prof = smoothstep(np.linspace(0, 1, y1 - y0, dtype=np.float32))
        if y1 < self.H:
            prof = np.concatenate([prof, np.ones(self.H - y1, np.float32) * prof[-1]])[: self.H - y0]
        self.grad = (y0, (1.0 - 0.42 * prof)[:, None, None].astype(np.float32))
        self.mark = None
        if brand:
            mh = int(round(34 * s))
            self.mark = _wordmark(mh)
            top = int(round(self.H * (0.09 if vertical else 0.035)))   # 9x16: abaixo do cabeçalho do Reels
            self.pos_mark = (self.W - int(round(48 * s)) - self.mark.shape[1] + mh // 3, top - mh // 3)
        self.strip, self.knob, self.strip_half = _divider_sprites(self.H, s)

    # ------------------------------------------------------------------------------------------------
    def _layer(self, idx: int, M: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(self.src[idx], M, (self.W, self.H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def _label(self, frame: np.ndarray, idx: int, alpha: float) -> None:
        if not self.show_labels or alpha <= 0:
            return
        if idx == 0:
            _blend_sprite(frame, self.lab_before, *self.pos_before, alpha)
        else:
            _blend_sprite(frame, self.lab_after[idx - 1], *self.pos_after[idx - 1], alpha)

    def _shade(self, frame: np.ndarray) -> None:
        if self.show_labels:
            y0, g = self.grad
            frame[y0:] = (frame[y0:].astype(np.float32) * g + 0.5).astype(np.uint8)

    def state(self, t: float) -> dict:
        """Quais camadas estão visíveis no instante t (para o quadro e para os testes)."""
        tl = self.tl
        if t < tl.wipe[0]:
            return {"kind": "before"}
        if t < tl.wipe[1]:
            a0, a1 = tl.wipe[0], tl.wipe_hold[0]
            b0, b1 = tl.wipe_hold[1], tl.wipe[1]
            if t < a1:
                pos = 1.0 - 0.5 * smootherstep((t - a0) / (a1 - a0))
            elif t < b0:
                pos = 0.5
            else:
                pos = 0.5 - 0.5 * smootherstep((t - b0) / (b1 - b0))
            return {"kind": "wipe", "pos": float(pos)}
        for i in range(1, len(tl.shots)):
            f0, f1 = tl.fades[i]
            if f0 <= t < f1:
                return {"kind": "fade", "from": i, "to": i + 1, "mix": float(smoothstep((t - f0) / (f1 - f0)))}
        idx = 1
        for i, (s0, _) in enumerate(tl.shots):
            if t >= s0:
                idx = i + 1
        return {"kind": "design", "index": idx}

    def frame(self, i: int) -> np.ndarray:
        t = i / self.fps
        M = self.cam.matrix(t)
        st = self.state(t)
        lab_in = float(smoothstep((t - 0.25) / 0.6))       # rótulos entram depois do primeiro instante
        if st["kind"] == "before":
            out = self._layer(0, M)
            self._shade_mix(out, lab_in)
            self._label(out, 0, lab_in)
        elif st["kind"] == "wipe":
            b = self._layer(0, M)
            a = self._layer(1, M)
            x = int(round(st["pos"] * self.W))
            out = a
            out[:, :x] = b[:, :x]
            self._shade(out)
            # rótulos por cima da composição (nunca cortados pelo divisor): cada um some/entra com a distância
            # do divisor até ele — o "Antes" se apaga antes de a linha chegar nele, o "Depois" só aparece depois que ela passou
            fw = 90.0 * self.scale
            if self.show_labels:
                bx1 = self.pos_before[0] + self.lab_before.shape[1]
                ax0 = self.pos_after[0][0]
                self._label(out, 0, float(smoothstep((x - bx1) / fw)))
                self._label(out, 1, float(smoothstep((ax0 - x) / fw)))
            if 0 < x < self.W:
                _blend_sprite(out, self.strip, x - self.strip_half, 0, 1.0)
                kh, kw = self.knob.shape[:2]
                _blend_sprite(out, self.knob, x - kw // 2, self.H // 2 - kh // 2, 1.0)
        elif st["kind"] == "fade":
            a = self._layer(st["from"], M)
            b = self._layer(st["to"], M)
            out = cv2.addWeighted(a, 1.0 - st["mix"], b, st["mix"], 0)
            self._shade(out)
            # rótulos em sequência (sai um, entra o outro), nunca sobrepostos no mesmo lugar
            self._label(out, st["from"], 1.0 - float(smoothstep(st["mix"] / 0.45)))
            self._label(out, st["to"], float(smoothstep((st["mix"] - 0.55) / 0.45)))
        else:
            out = self._layer(st["index"], M)
            self._shade(out)
            self._label(out, st["index"], 1.0)
        if self.mark is not None:
            _blend_sprite(out, self.mark, *self.pos_mark, 0.92 * lab_in if st["kind"] == "before" else 0.92)
        return out

    def _shade_mix(self, frame: np.ndarray, alpha: float) -> None:
        if self.show_labels and alpha > 0:
            y0, g = self.grad
            g = 1.0 - (1.0 - g) * alpha
            frame[y0:] = (frame[y0:].astype(np.float32) * g + 0.5).astype(np.uint8)

    @property
    def n_frames(self) -> int:
        return self.tl.frames

    def render(self, out_path, progress=None, codec: str | None = None, crf: int = 18) -> ShowcaseResult:
        from .encoder import EncodeSettings, VideoWriter, choose_codec

        t0 = time.perf_counter()
        codec = codec or choose_codec(True)
        vw = VideoWriter(out_path, self.W, self.H, self.fps,
                         EncodeSettings(codec=codec, crf=crf, preset="medium", keyint_s=2.0))
        n = self.n_frames
        gen = 0.0
        try:
            for i in range(n):
                g0 = time.perf_counter()
                f = self.frame(i)
                gen += time.perf_counter() - g0
                vw.write(f)
                if progress and (i % 15 == 0 or i == n - 1):
                    progress((i + 1) / n)
            path = vw.close()
        except BaseException:
            vw.abort()
            raise
        total = time.perf_counter() - t0
        return ShowcaseResult(output=str(path), size=(self.W, self.H), frames=n, fps=self.fps,
                              duration_s=n / self.fps, labels=self.labels, codec=codec,
                              timings={"quadros_s": round(gen, 2), "total_s": round(total, 2)})


def render_showcase(before, afters: list, out_path, fmt: str = "16x9", fps: int = 30, duration: float | None = None,
                    labels: list[str] | None = None, show_labels: bool = True, brand: bool = True,
                    progress=None, codec: str | None = None) -> ShowcaseResult:
    """Atalho: monta o :class:`Showcase` e grava o MP4. ``progress(fração)``."""
    t0 = time.perf_counter()
    sc = Showcase(before, afters, fmt, fps, duration, labels, show_labels, brand)
    prep = time.perf_counter() - t0
    res = sc.render(out_path, progress=progress, codec=codec)
    res.timings["preparo_s"] = round(prep, 2)
    res.timings["total_s"] = round(time.perf_counter() - t0, 2)
    res.timings["versoes_omitidas"] = sc.dropped
    return res


__all__ = ["FORMATS", "Showcase", "ShowcaseResult", "Timeline", "plan_timeline", "render_showcase", "infer_label",
           "norm_format"]
