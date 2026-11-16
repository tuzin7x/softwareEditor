"""Marca 7xAi: letras desenhadas em vetor (sem depender de fonte) e o ícone do aplicativo.

Fonte única do logotipo: a janela desenha a marca com estas formas, e o ícone do executável
(``packaging/assets/icon.ico``) e os SVGs da marca são gerados a partir delas
(``python -m mobilia.ui.brand <pasta>``).

Construção (unidades de desenho, y para baixo, linha de base em y = CAP):
  * altura das maiúsculas CAP = 100, altura-x XH = 70;
  * hastes retas, terminais cortados a prumo (traço de desenho de arquitetura);
  * o "x" é o acento da marca, em latão/champanhe ("7×": sete vezes). Nos tamanhos grandes
    o traço "/" do x passa por baixo do "\\" (uma fresta fina, como uma junta de marcenaria).

Tamanhos ópticos do ícone (cada um desenhado para a sua grade, não só reduzido):
  * L  (>= 64 px): "7xAi" com a fresta no x;
  * M  (32-63 px): "7xAi" mais encorpado, ocupando mais o quadro;
  * S  (< 32 px):  monograma "7x" alinhado à grade de 16 px.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QPolygonF, QRadialGradient, QTransform)

CAP = 100.0
XH = 70.0

# Cores da marca (o tema reaproveita as mesmas).
IVORY = "#F4EFE7"
BRASS_HI = "#EAD5A8"
BRASS = "#CDAE76"
BRASS_LO = "#A8854F"
TILE_TOP = "#2B2B30"
TILE_BOTTOM = "#111113"


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
    """(tinta "7 A i", acento "x", caixa) em unidades de desenho. ``weight``: espessura das hastes."""
    w = weight
    p7 = glyph_7(w, 0.0)
    x = 63.0 + track
    px = glyph_x(w, x, width=60.0 + (w - 18) * 0.6, cut=cut)
    x = px.boundingRect().right() + 4.0 + track
    pA = glyph_A(w, x, width=72.0 + (w - 18) * 0.8)
    x = pA.boundingRect().right() + 9.0 + (w - 18) * 0.2 + track
    pi = glyph_i(w, x)
    ink = _merge(p7, pA, pi)
    return ink, px, ink.boundingRect().united(px.boundingRect())


# Monograma "7x" da variante pequena (20-31 px), numa grade de 16 x 16.
SMALL_7 = [(2.5, 4), (8.5, 4), (6.1, 12), (4.0, 12), (6.15, 6), (2.5, 6)]
SMALL_X_A = [(8.5, 6), (10.4, 6), (13.6, 12), (11.7, 12)]
SMALL_X_B = [(11.7, 6), (13.6, 6), (10.4, 12), (8.5, 12)]

# 16 px: desenhado pixel a pixel (sem meio-tom nas hastes). "#" = pixel aceso; colunas 0-15.
PIXELS_16_7 = ["..#####.........",   # linha 4
               "..#####.........",
               ".....##.........",
               "....##..........",
               "....##..........",
               "...##...........",
               "...##...........",
               "...##..........."]   # linha 11
PIXELS_16_X = ["........#....#..",   # linha 6
               "........##..##..",
               ".........####...",
               ".........####...",
               "........##..##..",
               "........#....#.."]   # linha 11


def _pixel_path(rows: list[str], top: int) -> QPainterPath:
    path = QPainterPath()
    for j, row in enumerate(rows):
        i = 0
        while i < len(row):
            if row[i] == "#":
                k = i
                while k < len(row) and row[k] == "#":
                    k += 1
                path.addRect(QRectF(i, top + j, k - i, 1))
                i = k
            else:
                i += 1
    return path.simplified()

# Estilos da marca. HEADER: cabeçalho da janela (fino, espaçado).
HEADER = dict(weight=14.0, cut=3.0, track=6.0)


# ----------------------------------------------------------------------------------------- pintura

def brass_gradient(y0: float, y1: float, bright: bool = False) -> QLinearGradient:
    """Latão escovado. ``bright``: sem o tom escuro no pé (tamanhos pequenos, onde ele apagaria o x)."""
    g = QLinearGradient(QPointF(0, y0), QPointF(0, y1))
    g.setColorAt(0.0, QColor(BRASS_HI))
    g.setColorAt(0.55 if not bright else 1.0, QColor(BRASS))
    if not bright:
        g.setColorAt(1.0, QColor(BRASS_LO))
    return g


def wordmark_width(height: float, weight: float, cut: float = 0.0, track: float = 0.0) -> float:
    return wordmark_parts(weight, cut, track)[2].width() * height / CAP


def paint_wordmark(p: QPainter, left: float, top: float, height: float, ink: str = IVORY,
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
    p.setBrush(QBrush(brass_gradient(CAP - XH, CAP)))
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


def optical_class(size: float) -> str:
    return "L" if size >= 64 else ("M" if size >= 32 else "S")


def _icon_layout(size: float):
    """(tinta, acento, transformação desenho->pixels) do ícone grande/médio."""
    big = size >= 64
    main, acc, box = wordmark_parts(19.0 if big else 22.0, cut=5.0 if size >= 64 else 0.0)
    k = size * (0.70 if big else 0.80) / box.width()
    ox = (size - box.width() * k) / 2 - box.left() * k
    oy = (size - CAP * k) / 2
    return main, acc, QTransform(k, 0, 0, k, ox, oy)


def _tile(p: QPainter, size: float, radius: float, hairline: float = 0.0) -> None:
    """Placa grafite. ``hairline``: espessura do filete de latão (0 = sem filete)."""
    r = QRectF(0, 0, size, size)
    g = QLinearGradient(r.topLeft(), r.bottomLeft())
    g.setColorAt(0.0, QColor(TILE_TOP))
    g.setColorAt(1.0, QColor(TILE_BOTTOM))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(g))
    p.drawRoundedRect(r, radius, radius)
    # brilho difuso no alto (luz de estúdio)
    sheen = QRadialGradient(QPointF(size * 0.30, -size * 0.10), size * 0.95)
    c0 = QColor(BRASS_HI); c0.setAlphaF(0.10)
    c1 = QColor(BRASS_HI); c1.setAlphaF(0.0)
    sheen.setColorAt(0.0, c0); sheen.setColorAt(1.0, c1)
    p.setBrush(QBrush(sheen))
    p.drawRoundedRect(r, radius, radius)
    if hairline > 0:
        lw = hairline
        edge = QLinearGradient(r.topLeft(), r.bottomLeft())
        e0 = QColor(BRASS_HI); e0.setAlphaF(0.60)
        e1 = QColor(BRASS); e1.setAlphaF(0.16)
        edge.setColorAt(0.0, e0); edge.setColorAt(1.0, e1)
        p.setPen(QPen(QBrush(edge), lw)); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r.adjusted(lw / 2, lw / 2, -lw / 2, -lw / 2), radius - lw / 2, radius - lw / 2)


def paint_icon(p: QPainter, size: float) -> None:
    """Ícone do app em ``size`` px, escolhendo o tamanho óptico."""
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    if optical_class(size) == "S":
        k = size / 16.0
        p.scale(k, k)
        _tile(p, 16, 3.6, hairline=_small_hairline(size))
        seven, ex = small_parts(size)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(IVORY))
        p.drawPath(seven)
        p.setBrush(QBrush(brass_gradient(6, 12, bright=True)))
        p.drawPath(ex)
        p.restore()
        return
    _tile(p, size, size * 0.225, hairline=max(1.0, size / 128))
    main, acc, tf = _icon_layout(size)
    p.setTransform(tf, True)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(IVORY))
    p.drawPath(main)
    p.setBrush(QBrush(brass_gradient(CAP - XH, CAP, bright=size < 64)))
    p.drawPath(acc)
    p.restore()


def _small_hairline(size: float) -> float:
    """Filete de ~1 px real (em unidades da grade de 16)."""
    return 0.0 if size < 20 else 16.0 / size


def small_parts(size: float) -> tuple[QPainterPath, QPainterPath]:
    """(7, x) do monograma pequeno, na grade de 16: pixel a pixel até 19 px, vetorial acima."""
    if size < 20:
        return _pixel_path(PIXELS_16_7, 4), _pixel_path(PIXELS_16_X, 6)
    return _poly(SMALL_7), _poly(SMALL_X_A).united(_poly(SMALL_X_B))


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

def _d(path: QPainterPath, tf: QTransform | None = None) -> str:
    parts = []
    for poly in path.toSubpathPolygons(tf or QTransform()):
        pts = [f"{pt.x():.2f} {pt.y():.2f}" for pt in poly]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        parts.append("M" + " L".join(pts) + "Z")
    return "".join(parts)


def _svg_tile(size: float, radius: float, hairline: float) -> str:
    lw = hairline
    out = [f'<rect width="{size:g}" height="{size:g}" rx="{radius:g}" fill="url(#tile)"/>',
           f'<rect width="{size:g}" height="{size:g}" rx="{radius:g}" fill="url(#sheen)"/>']
    if hairline > 0:
        out.append(f'<rect x="{lw / 2:g}" y="{lw / 2:g}" width="{size - lw:g}" height="{size - lw:g}" '
                   f'rx="{radius - lw / 2:g}" fill="none" stroke="url(#edge)" stroke-width="{lw:g}"/>')
    return "\n  ".join(out)


def _svg_defs(size: float, brass_y0: float, brass_y1: float, bright: bool = False) -> str:
    mid = ('<stop offset="1" stop-color="{0}"/>'.format(BRASS) if bright else
           '<stop offset="0.55" stop-color="{0}"/><stop offset="1" stop-color="{1}"/>'.format(BRASS, BRASS_LO))
    return f"""<defs>
    <linearGradient id="tile" x1="0" y1="0" x2="0" y2="{size:g}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{TILE_TOP}"/><stop offset="1" stop-color="{TILE_BOTTOM}"/>
    </linearGradient>
    <radialGradient id="sheen" cx="{size * 0.30:g}" cy="{-size * 0.10:g}" r="{size * 0.95:g}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{BRASS_HI}" stop-opacity="0.10"/><stop offset="1" stop-color="{BRASS_HI}" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="edge" x1="0" y1="0" x2="0" y2="{size:g}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{BRASS_HI}" stop-opacity="0.60"/><stop offset="1" stop-color="{BRASS}" stop-opacity="0.16"/>
    </linearGradient>
    <linearGradient id="brass" x1="0" y1="{brass_y0:.2f}" x2="0" y2="{brass_y1:.2f}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{BRASS_HI}"/>{mid}
    </linearGradient>
  </defs>"""


def icon_svg(optical: str = "L") -> str:
    """SVG do ícone (só caminhos, sem fonte).

    L: mestre 256 px (>= 64 px); M: 48 px (32-63 px); S: 24 px (20-31 px); P: 16 px (pixel a pixel)."""
    if optical in ("S", "P"):
        px = 24 if optical == "S" else 16
        seven, ex = small_parts(px)
        body = (f'<path fill="{IVORY}" d="{_d(seven)}"/>\n  '
                f'<path fill="url(#brass)" d="{_d(ex)}"/>')
        return _svg_doc(16, _svg_defs(16, 6, 12, bright=True), _svg_tile(16, 3.6, _small_hairline(px)), body,
                        f"7xAi ({px} px)", px)
    size = 256 if optical == "L" else 48
    main, acc, tf = _icon_layout(size)
    y0 = tf.map(QPointF(0, CAP - XH)).y(); y1 = tf.map(QPointF(0, CAP)).y()
    body = (f'<path fill="{IVORY}" fill-rule="evenodd" d="{_d(main, tf)}"/>\n  '
            f'<path fill="url(#brass)" fill-rule="evenodd" d="{_d(acc, tf)}"/>')
    return _svg_doc(size, _svg_defs(size, y0, y1, bright=size < 64),
                    _svg_tile(size, size * 0.225, max(1.0, size / 128)), body, "7xAi")


def wordmark_svg(height: float = 100.0, ink: str = IVORY, **style) -> str:
    st = {**HEADER, **style}
    main, acc, box = wordmark_parts(st["weight"], st["cut"], st["track"])
    k = height / CAP
    tf = QTransform(k, 0, 0, k, -box.left() * k, 0)
    w = box.width() * k
    defs = f"""<defs>
    <linearGradient id="brass" x1="0" y1="{(CAP - XH) * k:.2f}" x2="0" y2="{height:.2f}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{BRASS_HI}"/><stop offset="0.55" stop-color="{BRASS}"/><stop offset="1" stop-color="{BRASS_LO}"/>
    </linearGradient>
  </defs>"""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {height:g}" '
            f'width="{w:.0f}" height="{height:g}">\n  <title>7xAi</title>\n  {defs}\n  '
            f'<path fill="{ink}" fill-rule="evenodd" d="{_d(main, tf)}"/>\n  '
            f'<path fill="url(#brass)" fill-rule="evenodd" d="{_d(acc, tf)}"/>\n</svg>\n')


def _svg_doc(size: int, defs: str, tile: str, body: str, title: str, px: int | None = None) -> str:
    px = px or size
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{px}" '
            f'height="{px}">\n  <title>{title}</title>\n  {defs}\n  {tile}\n  {body}\n</svg>\n')


# --------------------------------------------------------------------------------------------- ICO

ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def write_ico(path: str | Path, sizes=ICO_SIZES) -> Path:
    """Grava um .ico com uma imagem PNG por tamanho (cada uma no seu tamanho óptico)."""
    import struct

    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    blobs = []
    for s in sizes:
        ba = QByteArray(); buf = QBuffer(ba); buf.open(QIODevice.WriteOnly)
        app_icon_pixmap(s).toImage().save(buf, "PNG")
        blobs.append((s, bytes(ba.data())))
    head = struct.pack("<HHH", 0, 1, len(blobs))
    offset = 6 + 16 * len(blobs)
    dirs, data = b"", b""
    for s, png in blobs:
        dirs += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
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
