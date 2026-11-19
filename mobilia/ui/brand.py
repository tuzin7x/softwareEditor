"""Marca 7xAi na identidade 7x Intelligence: o emblema hexagonal "7x" e o logotipo "7xAi".

Fonte única da marca: a janela desenha o emblema e o logotipo com estas formas, e o ícone do executável
(``packaging/assets/icon.ico``), os SVGs da marca e a marca d'água do vídeo (``mobilia/video/res``) são gerados a
partir delas (``python scripts/make_icon.py``).

Emblema ("Monogram" do site 7x Intelligence, arte do próprio dono): hexágono facetado em grade de 48, contorno em
gradiente violeta (#C084FC -> #7C3AED -> #4C1D95), miolo #120B22, marcas de mira nos quatro lados, o "7" claro e o
"x" violeta. Tamanhos ópticos do ícone (cada um desenhado para a sua grade, não só reduzido):
  * L  (>= 40 px): emblema completo (marcas de mira e faceta interna);
  * S  (< 40 px):  sem as marcas de mira e com traços mais grossos, para o "7x" continuar legível em 16-32 px.

Logotipo "7xAi" (unidades de desenho, y para baixo, linha de base em y = CAP): letras desenhadas em vetor (sem
depender de fonte), CAP = 100 e XH = 70, hastes retas; "7x" em gradiente violeta (violet-200 -> purple-500, como o
"7x" do site) e "Ai" no branco do tema.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QPolygonF,
                           QRadialGradient)

CAP = 100.0
XH = 70.0

# Cores da marca (as do emblema do site 7x Intelligence; o tema usa os tokens oklch equivalentes).
INK = "#ECEFF2"           # foreground do site, oklch(0.95 0.005 260)
MONO_HI = "#C084FC"       # gradiente do contorno (purple-400 -> violet-600 -> violet-900)
MONO_MID = "#7C3AED"
MONO_LO = "#4C1D95"
MONO_FILL = "#120B22"     # miolo do hexágono
MONO_TICK = "#C4B5FD"     # marcas de mira (violet-300)
MONO_SEVEN = "#F5F3FF"    # o "7" (violet-50)
MONO_X = "#C084FC"        # o "x"
GLOW = "#8B5CF6"          # brilho em volta do emblema (drop-shadow 0 0 8px #8b5cf6 no site)
WORD_FROM = "#DDD6FE"     # "7x" do logotipo: from-violet-200 ...
WORD_TO = "#A855F7"       # ... to-purple-500


# ------------------------------------------------------------------------------------------ letras

# ------------------------------------------------------------------------------------------ letras

def _poly(points) -> QPainterPath:
    p = QPainterPath()
    p.addPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
    p.closeSubpath()
    return p


def _merge(*paths: QPainterPath) -> QPainterPath:
    out = QPainterPath()
    for p in paths:
        out.addPath(p)
    return out


def _h(w: float, s: float) -> float:
    """Largura horizontal de um traço inclinado (dx/dy = s) com espessura perpendicular w."""
    return w * math.sqrt(1.0 + s * s)


def glyph_7(w: float, x0: float, width: float = 60.0, s: float = 0.30) -> QPainterPath:
    hd = _h(w * 0.96, s)
    top_r = x0 + width
    foot = top_r - s * CAP
    return _poly([(x0, 0), (top_r, 0), (foot, CAP), (foot - hd, CAP), (top_r - s * w - hd, w), (x0, w)])


def glyph_x(w: float, x0: float, width: float = 60.0, cut: float = 0.0) -> QPainterPath:
    top = CAP - XH
    s = 0.5
    for _ in range(8):                        # s tal que s * XH + h(s) = width
        s = (width - _h(w * 0.92, s)) / XH
    h = _h(w * 0.92, s)
    a = _poly([(x0, top), (x0 + h, top), (x0 + width, CAP), (x0 + width - h, CAP)])
    b = _poly([(x0 + width - h, top), (x0 + width, top), (x0 + h, CAP), (x0, CAP)])
    if cut > 0:                               # fresta: "/" passa por baixo do "\"
        n = math.hypot(XH, width - h)
        ox, oy = XH / n * cut, -(width - h) / n * cut
        wide = _poly([(x0 - ox, top - oy), (x0 + h + ox, top + oy), (x0 + width + ox, CAP + oy),
                      (x0 + width - h - ox, CAP - oy)])
        return a.united(b.subtracted(wide))
    return a.united(b)


def glyph_A(w: float, x0: float, width: float = 72.0, bar_y: float = 64.0) -> QPainterPath:
    t = w * 1.05                               # topo plano
    cx = x0 + width / 2
    xl0, xr0 = cx - t / 2, cx + t / 2
    s = (xl0 - x0) / CAP
    h = _h(w * 0.96, s)
    left = _poly([(xl0, 0), (xl0 + h, 0), (x0 + h, CAP), (x0, CAP)])
    right = _poly([(xr0 - h, 0), (xr0, 0), (x0 + width, CAP), (x0 + width - h, CAP)])
    bar = QPainterPath()
    bar.addRect(QRectF(x0, bar_y, width, w * 0.82))
    hull = _poly([(xl0, 0), (xr0, 0), (x0 + width, CAP), (x0, CAP)])
    return left.united(right).united(bar.intersected(hull))


def glyph_i(w: float, x0: float) -> QPainterPath:
    top = CAP - XH
    stem = QPainterPath()
    stem.addRect(QRectF(x0, top, w, XH))
    dot = QPainterPath()
    dot.addRect(QRectF(x0, 0, w, w))          # ponto quadrado alinhado ao topo das maiúsculas
    return _merge(stem, dot)


def wordmark_parts(weight: float = 18.0, cut: float = 0.0, track: float = 0.0
                   ) -> tuple[QPainterPath, QPainterPath, QRectF]:
    """(tinta "A i", acento "7x", caixa) em unidades de desenho. ``weight``: espessura das hastes.

    Como no site 7x Intelligence, o "7x" leva o gradiente violeta da marca e o resto fica claro."""
    w = weight
    p7 = glyph_7(w, 0.0)
    x = 63.0 + track
    px = glyph_x(w, x, width=60.0 + (w - 18) * 0.6, cut=cut)
    x = px.boundingRect().right() + 4.0 + track
    pA = glyph_A(w, x, width=72.0 + (w - 18) * 0.8)
    x = pA.boundingRect().right() + 9.0 + (w - 18) * 0.2 + track
    pi = glyph_i(w, x)
    ink = _merge(pA, pi)
    acc = _merge(p7, px)
    return ink, acc, ink.boundingRect().united(acc.boundingRect())



# Estilo do logotipo no cabeçalho (peso encorpado, com a junta no x).
HEADER = dict(weight=20.0, cut=4.0, track=3.0)


# ---------------------------------------------------------------------------------------- emblema

HEX = [(24, 2), (43, 13), (43, 35), (24, 46), (5, 35), (5, 13)]
HEX_IN = [(24, 8), (38, 16), (38, 32), (24, 40), (10, 32), (10, 16)]
TICKS = [((24, 3), (24, 8)), ((24, 40), (24, 45)), ((5.5, 24), (10, 24)), ((38, 24), (42.5, 24))]
SEVEN = [(14, 16), (25, 16), (18, 33)]
EX = [((26, 23), (34, 32)), ((34, 23), (26, 32))]
# tamanho óptico pequeno: o mesmo desenho, sem mira nem faceta, "7x" maior e mais grosso
SEVEN_S = [(12.5, 14.5), (26, 14.5), (17.5, 34.5)]
EX_S = [((25.5, 22), (35.5, 34)), ((35.5, 22), (25.5, 34))]


def mono_optical(size: float) -> str:
    return "L" if size >= 40 else "S"


def _mono_spec(optical: str) -> dict:
    if optical == "S":
        return dict(edge=3.4, seven=SEVEN_S, seven_w=5.2, ex=EX_S, ex_w=4.8, ticks=False, inner=False)
    return dict(edge=2.5, seven=SEVEN, seven_w=3.6, ex=EX, ex_w=3.2, ticks=True, inner=True)


def _grad(x0, y0, x1, y1) -> QLinearGradient:
    g = QLinearGradient(QPointF(x0, y0), QPointF(x1, y1))
    g.setColorAt(0.0, QColor(MONO_HI)); g.setColorAt(0.55, QColor(MONO_MID)); g.setColorAt(1.0, QColor(MONO_LO))
    return g


def _round_pen(brush, w: float) -> QPen:
    pen = QPen(brush, w)
    pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
    return pen


def paint_monogram(p: QPainter, x: float, y: float, size: float, optical: str | None = None) -> None:
    """Emblema 7x na caixa (x, y, size, size). ``optical``: "L" ou "S" (padrão: pelo tamanho)."""
    sp = _mono_spec(optical or mono_optical(size))
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.translate(x, y)
    p.scale(size / 48.0, size / 48.0)
    hexp = QPolygonF([QPointF(a, b) for a, b in HEX])
    p.setPen(QPen(QBrush(_grad(5, 2, 43, 46)), sp["edge"], Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin))
    p.setBrush(QColor(MONO_FILL))
    p.drawPolygon(hexp)
    if sp["inner"]:
        p.setPen(Qt.NoPen)
        p.setOpacity(0.18)
        p.setBrush(QBrush(_grad(10, 8, 38, 40)))
        p.drawPolygon(QPolygonF([QPointF(a, b) for a, b in HEX_IN]))
        p.setOpacity(1.0)
    if sp["ticks"]:
        p.setPen(_round_pen(QColor(MONO_TICK), 1.6))
        for (a, b), (c, d) in TICKS:
            p.drawLine(QPointF(a, b), QPointF(c, d))
    seven = QPainterPath(QPointF(*sp["seven"][0]))
    for pt in sp["seven"][1:]:
        seven.lineTo(QPointF(*pt))
    p.setBrush(Qt.NoBrush)
    p.setPen(_round_pen(QColor(MONO_SEVEN), sp["seven_w"]))
    p.drawPath(seven)
    p.setPen(_round_pen(QColor(MONO_X), sp["ex_w"]))
    for (a, b), (c, d) in sp["ex"]:
        p.drawLine(QPointF(a, b), QPointF(c, d))
    p.restore()


def paint_glow(p: QPainter, cx: float, cy: float, radius: float, alpha: float = 0.42) -> None:
    """Brilho violeta suave (o drop-shadow do emblema no site), barato: um gradiente radial."""
    g = QRadialGradient(QPointF(cx, cy), radius)
    c0 = QColor(GLOW); c0.setAlphaF(alpha)
    c1 = QColor(GLOW); c1.setAlphaF(0.0)
    g.setColorAt(0.0, c0); g.setColorAt(0.55, QColor(c0.red(), c0.green(), c0.blue(), int(255 * alpha * 0.35)))
    g.setColorAt(1.0, c1)
    p.save(); p.setPen(Qt.NoPen); p.setBrush(QBrush(g))
    p.drawEllipse(QPointF(cx, cy), radius, radius)
    p.restore()


def monogram_pixmap(size: int, dpr: float = 2.0, glow: bool = False) -> QPixmap:
    """Emblema em ``size`` px lógicos; ``glow`` acrescenta o brilho violeta (margem de 25 % em volta)."""
    pad = round(size * 0.25) if glow else 0
    full = size + 2 * pad
    pm = QPixmap(round(full * dpr), round(full * dpr))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.scale(dpr, dpr)
    p.setRenderHint(QPainter.Antialiasing)
    if glow:
        paint_glow(p, full / 2, full / 2, full / 2)
    paint_monogram(p, pad, pad, size)
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


# --------------------------------------------------------------------------------------- logotipo

def word_gradient(x0: float, x1: float) -> QLinearGradient:
    """Gradiente horizontal do "7x" (violet-200 -> purple-500)."""
    g = QLinearGradient(QPointF(x0, 0), QPointF(x1, 0))
    g.setColorAt(0.0, QColor(WORD_FROM)); g.setColorAt(1.0, QColor(WORD_TO))
    return g


def wordmark_width(height: float, weight: float, cut: float = 0.0, track: float = 0.0) -> float:
    return wordmark_parts(weight, cut, track)[2].width() * height / CAP


def paint_wordmark(p: QPainter, left: float, top: float, height: float, ink: str = INK,
                   weight: float = 14.0, cut: float = 0.0, track: float = 0.0) -> float:
    """Desenha "7xAi" com altura de maiúsculas ``height`` a partir de (left, top). Devolve a largura."""
    main, acc, box = wordmark_parts(weight, cut, track)
    k = height / CAP
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.translate(left - box.left() * k, top)
    p.scale(k, k)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(ink))
    p.drawPath(main)
    ab = acc.boundingRect()
    p.setBrush(QBrush(word_gradient(ab.left(), ab.right())))
    p.drawPath(acc)
    p.restore()
    return box.width() * k


def wordmark_pixmap(height: int, dpr: float = 2.0, **style) -> QPixmap:
    st = {**HEADER, **style}
    w = int(math.ceil(wordmark_width(height, st["weight"], st["cut"], st["track"]))) + 2
    pm = QPixmap(int(w * dpr), int(math.ceil(height * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    paint_wordmark(p, 1, 0, height, **st)
    p.end()
    return pm


def lockup_pixmap(height: int, dpr: float = 2.0, glow: bool = True) -> QPixmap:
    """Emblema + logotipo lado a lado (cabeçalho). ``height``: altura do emblema; o logotipo tem 0,56 dela.
    O brilho violeta (``glow``) se espalha para os lados; em cima e embaixo ele já se apagou na borda do desenho,
    então o cabeçalho só cresce 2 px de cada lado."""
    cap = round(height * 0.56)
    gap = round(height * 0.34)
    pad_x = round(height * 0.2) if glow else 0
    pad_y = 2 if glow else 0
    ww = int(math.ceil(wordmark_width(cap, HEADER["weight"], HEADER["cut"], HEADER["track"]))) + 2
    w, h = pad_x + height + gap + ww + 2, height + 2 * pad_y
    pm = QPixmap(round(w * dpr), round(h * dpr))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.scale(dpr, dpr)
    p.setRenderHint(QPainter.Antialiasing)
    if glow:
        paint_glow(p, pad_x + height / 2, h / 2, height * 0.62, 0.45)
    paint_monogram(p, pad_x, pad_y, height, "L")
    paint_wordmark(p, pad_x + height + gap, pad_y + (height - cap) / 2, cap, **HEADER)
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


# ------------------------------------------------------------------------------------------- ícone

def optical_class(size: float) -> str:
    return mono_optical(size)


def paint_icon(p: QPainter, size: float) -> None:
    """Ícone do app em ``size`` px: o emblema 7x ocupando o quadro (sem placa de fundo)."""
    paint_monogram(p, 0, 0, size)


def app_icon_pixmap(size: int = 256, dpr: float = 1.0) -> QPixmap:
    """Ícone com ``size`` px lógicos (tamanho óptico escolhido por ``size``), nítido em ``dpr``."""
    pm = QPixmap(round(size * dpr), round(size * dpr))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.scale(dpr, dpr)
    paint_icon(p, size)
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


# --------------------------------------------------------------------------------------------- SVG

def _d(path: QPainterPath, tf=None) -> str:
    from PySide6.QtGui import QTransform
    parts = []
    for poly in path.toSubpathPolygons(tf or QTransform()):
        pts = [f"{pt.x():.2f} {pt.y():.2f}" for pt in poly]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        parts.append("M" + " L".join(pts) + "Z")
    return "".join(parts)


def _pts(points) -> str:
    return " ".join(f"{a:g},{b:g}" for a, b in points)


def icon_svg(optical: str = "L") -> str:
    """SVG do emblema (só formas, sem fonte), na grade de 48. L: completo; S: tamanho pequeno (<= 32 px)."""
    sp = _mono_spec("S" if optical in ("S", "P") else "L")
    px = {"L": 256, "M": 48, "S": 24, "P": 16}.get(optical, 256)
    stop = (f'<stop offset="0" stop-color="{MONO_HI}"/><stop offset="0.55" stop-color="{MONO_MID}"/>'
            f'<stop offset="1" stop-color="{MONO_LO}"/>')
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="{px}" height="{px}">',
           "  <title>7xAi</title>",
           "  <defs>",
           f'    <linearGradient id="edge" x1="5" y1="2" x2="43" y2="46" gradientUnits="userSpaceOnUse">{stop}'
           "</linearGradient>",
           f'    <linearGradient id="facet" x1="10" y1="8" x2="38" y2="40" gradientUnits="userSpaceOnUse">{stop}'
           "</linearGradient>",
           "  </defs>",
           f'  <polygon points="{_pts(HEX)}" fill="{MONO_FILL}" stroke="url(#edge)" stroke-width="{sp["edge"]:g}"/>']
    if sp["inner"]:
        out.append(f'  <polygon points="{_pts(HEX_IN)}" fill="url(#facet)" opacity="0.18"/>')
    if sp["ticks"]:
        d = "".join(f"M{a:g} {b:g}L{c:g} {d_:g}" for (a, b), (c, d_) in TICKS)
        out.append(f'  <path d="{d}" stroke="{MONO_TICK}" stroke-width="1.6" stroke-linecap="round"/>')
    out.append(f'  <polyline points="{_pts(sp["seven"])}" fill="none" stroke="{MONO_SEVEN}" '
               f'stroke-width="{sp["seven_w"]:g}" stroke-linecap="round" stroke-linejoin="round"/>')
    d = "".join(f"M{a:g} {b:g}L{c:g} {d_:g}" for (a, b), (c, d_) in sp["ex"])
    out.append(f'  <path d="{d}" stroke="{MONO_X}" stroke-width="{sp["ex_w"]:g}" stroke-linecap="round"/>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def wordmark_svg(height: float = 100.0, ink: str = INK, **style) -> str:
    from PySide6.QtGui import QTransform
    st = {**HEADER, **style}
    main, acc, box = wordmark_parts(st["weight"], st["cut"], st["track"])
    k = height / CAP
    tf = QTransform(k, 0, 0, k, -box.left() * k, 0)
    w = box.width() * k
    ab = tf.mapRect(acc.boundingRect())
    defs = f"""<defs>
    <linearGradient id="word" x1="{ab.left():.2f}" y1="0" x2="{ab.right():.2f}" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{WORD_FROM}"/><stop offset="1" stop-color="{WORD_TO}"/>
    </linearGradient>
  </defs>"""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {height:g}" '
            f'width="{w:.0f}" height="{height:g}">\n  <title>7xAi</title>\n  {defs}\n  '
            f'<path fill="{ink}" fill-rule="evenodd" d="{_d(main, tf)}"/>\n  '
            f'<path fill="url(#word)" fill-rule="evenodd" d="{_d(acc, tf)}"/>\n</svg>\n')


def video_mark_png(path: str | Path, cap: int = 128) -> Path:
    """Marca d'água do vídeo antes e depois (``mobilia/video/res/marca_7xai.png``): o logotipo com maiúsculas de
    ``cap`` px e 2 px de margem (o motor do vídeo não depende do Qt e só lê o PNG)."""
    from PySide6.QtGui import QImage
    st = dict(HEADER)
    w = int(math.ceil(wordmark_width(cap, st["weight"], st["cut"], st["track"]))) + 4
    img = QImage(w, cap + 4, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    paint_wordmark(p, 2, 2, cap, **st)
    p.end()
    path = Path(path)
    img.save(str(path))
    return path


# --------------------------------------------------------------------------------------------- ICO

ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def _ico_dib(img) -> bytes:
    """Imagem do .ico no formato clássico (BMP 32 bits BGRA + máscara AND), o mais compatível
    com o Explorer, o Inno Setup e editores de recurso; só o 256 px vai em PNG."""
    import struct

    from PySide6.QtGui import QImage

    img = img.convertToFormat(QImage.Format_ARGB32)          # BGRA na memória (little-endian)
    w, h = img.width(), img.height()
    stride = w * 4
    raw = bytes(img.constBits())[: stride * h]
    xor = b"".join(raw[y * stride:(y + 1) * stride] for y in range(h - 1, -1, -1))   # de baixo para cima
    mask_stride = ((w + 31) // 32) * 4
    mask = bytearray()
    for y in range(h - 1, -1, -1):
        row = bytearray(mask_stride)
        for x in range(w):
            if raw[y * stride + x * 4 + 3] == 0:                     # transparente -> bit 1 na máscara AND
                row[x // 8] |= 0x80 >> (x % 8)
        mask += row
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, len(xor) + len(mask), 0, 0, 0, 0)
    return header + xor + bytes(mask)


def write_ico(path: str | Path, sizes=ICO_SIZES) -> Path:
    """Grava um .ico com uma imagem por tamanho (cada uma no seu tamanho óptico):
    BMP 32 bits até 128 px e PNG em 256 px, como nos ícones do próprio Windows."""
    import struct

    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    blobs = []
    for s in sizes:
        img = app_icon_pixmap(s).toImage()
        if s >= 256:
            ba = QByteArray(); buf = QBuffer(ba); buf.open(QIODevice.WriteOnly)
            img.save(buf, "PNG")
            blobs.append((s, bytes(ba.data())))
        else:
            blobs.append((s, _ico_dib(img)))
    head = struct.pack("<HHH", 0, 1, len(blobs))
    offset = 6 + 16 * len(blobs)
    dirs, data = b"", b""
    for s, blob in blobs:
        dirs += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(blob), offset + len(data))
        data += blob
    path = Path(path)
    path.write_bytes(head + dirs + data)
    return path


def export_assets(out_dir: str | Path) -> list[Path]:
    """Gera icon.ico + icon.png (256) + SVGs da marca em ``out_dir`` (usado no build)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    files = [write_ico(out / "icon.ico")]
    app_icon_pixmap(256).save(str(out / "icon.png"))
    files.append(out / "icon.png")
    for name, text in (("logo.svg", icon_svg("L")), ("logo_m.svg", icon_svg("M")),
                       ("logo_small.svg", icon_svg("S")), ("logo_16.svg", icon_svg("P")),
                       ("wordmark.svg", wordmark_svg())):
        (out / name).write_text(text, encoding="utf-8")
        files.append(out / name)
    return files


if __name__ == "__main__":  # python -m mobilia.ui.brand packaging/assets
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication

    _app = QGuiApplication.instance() or QGuiApplication([])
    for f in export_assets(sys.argv[1] if len(sys.argv) > 1 else "packaging/assets"):
        print(f)
