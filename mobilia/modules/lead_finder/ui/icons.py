"""Ícones de linha do Lead Finder (grade de 16, mesmo traço dos ícones do 7xAi, desenhados em QPainter)."""
from __future__ import annotations

from math import cos, radians, sin

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from mobilia.ui import theme

_CACHE: dict = {}


def _draw(p: QPainter, name: str) -> None:
    line = lambda x1, y1, x2, y2: p.drawLine(QPointF(x1, y1), QPointF(x2, y2))  # noqa: E731
    if name == "search":
        p.drawEllipse(QPointF(7, 7), 4.3, 4.3); line(10.2, 10.2, 13.8, 13.8)
    elif name == "phone":
        path = QPainterPath(QPointF(4.2, 2.5)); path.lineTo(6.4, 2.5); path.lineTo(7.4, 5.4); path.lineTo(6, 6.6)
        path.cubicTo(6.8, 8.4, 7.6, 9.2, 9.4, 10); path.lineTo(10.6, 8.6); path.lineTo(13.5, 9.6); path.lineTo(13.5, 11.8)
        path.cubicTo(13.5, 12.8, 12.7, 13.6, 11.7, 13.5); path.cubicTo(6.5, 13, 3, 9.5, 2.5, 4.3)
        path.cubicTo(2.4, 3.3, 3.2, 2.5, 4.2, 2.5); p.drawPath(path)
    elif name == "whatsapp":
        path = QPainterPath(); path.addEllipse(QRectF(2.5, 2.3, 11, 11)); p.drawPath(path)
        line(3.4, 10.9, 2.3, 13.8); line(2.3, 13.8, 5.4, 12.8)
        inner = QPainterPath(QPointF(6.2, 5.6)); inner.lineTo(6.9, 5.6); inner.lineTo(7.3, 6.8); inner.lineTo(6.8, 7.3)
        inner.cubicTo(7.2, 8.2, 7.8, 8.8, 8.7, 9.2); inner.lineTo(9.2, 8.7); inner.lineTo(10.4, 9.1); inner.lineTo(10.4, 9.8)
        inner.cubicTo(10.4, 10.3, 9.9, 10.6, 9.4, 10.5); inner.cubicTo(7.5, 10.1, 6, 8.6, 5.6, 6.7)
        inner.cubicTo(5.5, 6.1, 5.8, 5.6, 6.2, 5.6); p.drawPath(inner)
    elif name == "user":
        p.drawEllipse(QPointF(8, 5.6), 2.8, 2.8)
        path = QPainterPath(QPointF(2.8, 13.5)); path.cubicTo(3.2, 10.3, 5.3, 9.3, 8, 9.3)
        path.cubicTo(10.7, 9.3, 12.8, 10.3, 13.2, 13.5); p.drawPath(path)
    elif name == "globe":
        p.drawEllipse(QPointF(8, 8), 5.6, 5.6); p.drawEllipse(QRectF(5.4, 2.4, 5.2, 11.2))
        line(2.4, 8, 13.6, 8)
    elif name == "building":
        p.drawRect(QRectF(3, 2.5, 7, 11)); p.drawRect(QRectF(10, 6.5, 3.5, 7))
        for y in (5, 7.5, 10):
            line(5, y, 5.8, y); line(7.2, y, 8, y)
    elif name == "pin":
        path = QPainterPath(QPointF(8, 14)); path.cubicTo(4, 9.5, 3.4, 7.8, 3.4, 6.2)
        path.cubicTo(3.4, 3.7, 5.4, 1.9, 8, 1.9); path.cubicTo(10.6, 1.9, 12.6, 3.7, 12.6, 6.2)
        path.cubicTo(12.6, 7.8, 12, 9.5, 8, 14); p.drawPath(path); p.drawEllipse(QPointF(8, 6.2), 1.8, 1.8)
    elif name == "funnel":
        path = QPainterPath(QPointF(2.5, 3)); path.lineTo(13.5, 3); path.lineTo(9.3, 8.4); path.lineTo(9.3, 13)
        path.lineTo(6.7, 11.6); path.lineTo(6.7, 8.4); path.closeSubpath(); p.drawPath(path)
    elif name == "chart":
        line(2.5, 13.5, 13.5, 13.5)
        for x, h in ((4, 4), (7, 7), (10, 5.5), (13, 9.5)):
            line(x, 13.5, x, 13.5 - h)
    elif name == "gear":
        p.drawEllipse(QPointF(8, 8), 2.2, 2.2); p.drawEllipse(QPointF(8, 8), 4.4, 4.4)
        for a in range(0, 360, 45):
            c, s = cos(radians(a)), sin(radians(a))
            line(8 + 4.4 * c, 8 + 4.4 * s, 8 + 6 * c, 8 + 6 * s)
    elif name == "message":
        path = QPainterPath(QPointF(2.5, 3.5)); path.lineTo(13.5, 3.5); path.lineTo(13.5, 10.5); path.lineTo(7, 10.5)
        path.lineTo(4, 13.2); path.lineTo(4, 10.5); path.lineTo(2.5, 10.5); path.closeSubpath(); p.drawPath(path)
        line(5, 6, 11, 6); line(5, 8.2, 9.2, 8.2)
    elif name == "bell":
        path = QPainterPath(QPointF(3.2, 11.5)); path.lineTo(12.8, 11.5); path.cubicTo(11.6, 10.3, 11.6, 9, 11.6, 7.2)
        path.cubicTo(11.6, 4.8, 10, 3.2, 8, 3.2); path.cubicTo(6, 3.2, 4.4, 4.8, 4.4, 7.2)
        path.cubicTo(4.4, 9, 4.4, 10.3, 3.2, 11.5); p.drawPath(path); line(7, 13.4, 9, 13.4); line(8, 2, 8, 3.2)
    elif name == "note":
        p.drawRoundedRect(QRectF(3, 2.5, 10, 11), 1.5, 1.5)
        for y in (5.5, 8, 10.5):
            line(5.2, y, 10.8, y)
    elif name == "link":
        p.drawRoundedRect(QRectF(1.8, 6, 7, 4), 2, 2); p.drawRoundedRect(QRectF(7.2, 6, 7, 4), 2, 2)
    elif name == "star":
        path = QPainterPath()
        for i in range(10):
            r = 5.8 if i % 2 == 0 else 2.6
            a = radians(-90 + i * 36)
            pt = QPointF(8 + r * cos(a), 8.4 + r * sin(a))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        path.closeSubpath(); p.drawPath(path)
    elif name == "x":
        line(4, 4, 12, 12); line(12, 4, 4, 12)
    elif name == "plus":
        line(8, 3, 8, 13); line(3, 8, 13, 8)
    elif name == "download":
        line(8, 2.5, 8, 10); line(5, 7, 8, 10); line(8, 10, 11, 7)
        path = QPainterPath(QPointF(2.5, 9.5)); path.lineTo(2.5, 13.5); path.lineTo(13.5, 13.5); path.lineTo(13.5, 9.5)
        p.drawPath(path)
    elif name == "refresh":
        p.drawArc(QRectF(3, 3, 10, 10), 60 * 16, 280 * 16); line(11.5, 2.5, 11.5, 5.2); line(11.5, 5.2, 8.8, 5.2)
    elif name == "trash":
        line(2.5, 4.5, 13.5, 4.5); line(6, 4.5, 6.5, 2.5); line(6.5, 2.5, 9.5, 2.5); line(9.5, 2.5, 10, 4.5)
        path = QPainterPath(QPointF(4, 4.5)); path.lineTo(4.8, 13.5); path.lineTo(11.2, 13.5); path.lineTo(12, 4.5)
        p.drawPath(path)
    elif name == "copy":
        p.drawRoundedRect(QRectF(5, 5, 8.5, 8.5), 1.5, 1.5)
        path = QPainterPath(QPointF(3, 10.5)); path.lineTo(2.5, 10.5); path.lineTo(2.5, 2.5); path.lineTo(10.5, 2.5)
        path.lineTo(10.5, 3); p.drawPath(path)
    elif name == "block":
        p.drawEllipse(QPointF(8, 8), 5.5, 5.5); line(4.2, 11.8, 11.8, 4.2)
    elif name == "video":
        p.drawRoundedRect(QRectF(2, 4, 9, 8), 1.5, 1.5)
        path = QPainterPath(QPointF(11, 7)); path.lineTo(14, 5); path.lineTo(14, 11); path.lineTo(11, 9); p.drawPath(path)
    elif name == "sparkle":
        path = QPainterPath(QPointF(7, 2)); path.quadTo(7.6, 6.4, 12, 7); path.quadTo(7.6, 7.6, 7, 12)
        path.quadTo(6.4, 7.6, 2, 7); path.quadTo(6.4, 6.4, 7, 2); p.drawPath(path)
        line(12.5, 10.5, 12.5, 14); line(10.75, 12.25, 14.25, 12.25)
    elif name == "check":
        line(3.5, 8.5, 6.5, 11.5); line(6.5, 11.5, 12.5, 4.5)
    elif name == "folder":
        path = QPainterPath(QPointF(2, 4)); path.lineTo(6, 4); path.lineTo(7.5, 5.5); path.lineTo(14, 5.5)
        path.lineTo(14, 13); path.lineTo(2, 13); path.closeSubpath(); p.drawPath(path)
    else:
        p.drawEllipse(QPointF(8, 8), 2, 2)


def pixmap(name: str, size: int = 16, color: str = theme.TEXT_2, dpr: float = 2.0) -> QPixmap:
    pm = QPixmap(int(size * dpr), int(size * dpr)); pm.setDevicePixelRatio(dpr); pm.fill(Qt.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    p.scale(size / 16.0, size / 16.0)
    pen = QPen(QColor(color)); pen.setWidthF(1.4); pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.NoBrush)
    _draw(p, name)
    p.end()
    return pm


def icon(name: str, color: str = theme.TEXT_2, size: int = 16) -> QIcon:
    key = (name, color, size)
    if key not in _CACHE:
        _CACHE[key] = QIcon(pixmap(name, size, color))
    return _CACHE[key]
