"""Tema visual (escuro, moderno) e ícones de linha desenhados em QPainter (sem arquivos externos)."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

ACCENT = "#E0A060"        # terracota suave
BG = "#14161A"
SURFACE = "#1C1F25"
SURFACE_2 = "#252932"
BORDER = "#323743"
TEXT = "#ECEDEF"
MUTED = "#9AA0AC"
DANGER = "#E26D6D"
OK = "#6FCF97"

QSS = f"""
* {{ font-family: "Segoe UI Variable", "Segoe UI", "Inter", "Noto Sans", sans-serif; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QLabel#title {{ font-size: 22px; font-weight: 600; }}
QLabel#subtitle {{ color: {MUTED}; font-size: 13px; }}
QLabel#section {{ color: {MUTED}; font-size: 12px; font-weight: 600; letter-spacing: 1px; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#notice {{ color: {ACCENT}; background: #2a241d; border: 1px solid #5a4a38; border-radius: 8px;
                 padding: 8px 10px; }}
QFrame#card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}
QFrame#drop {{ background: {SURFACE}; border: 2px dashed {BORDER}; border-radius: 16px; }}
QFrame#drop[hover="true"] {{ border-color: {ACCENT}; }}
QPushButton {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 10px; padding: 9px 16px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #5c6270; border-color: #2a2e37; }}
QPushButton#primary {{ background: {ACCENT}; color: #1a1205; border: none; font-weight: 600; padding: 11px 22px; }}
QPushButton#primary:hover {{ background: #ecb173; }}
QPushButton#primary:disabled {{ background: #5a4a38; color: #2a2116; }}
QPushButton#danger {{ border-color: #5b3434; color: {DANGER}; }}
QToolButton#room {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 10px;
                    font-size: 13px; }}
QToolButton#room:hover {{ border-color: #6b5b4b; }}
QToolButton#room:checked {{ border: 2px solid {ACCENT}; background: #2a241d; }}
QProgressBar {{ background: {SURFACE_2}; border: none; border-radius: 6px; height: 10px; text-align: center;
               }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}
QComboBox {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 10px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_2}; selection-background-color: #3a3126; }}
QStatusBar {{ background: {SURFACE}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QTextEdit {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 8px; font-family: monospace; }}
"""


def _pen(color: str, w: float) -> QPen:
    p = QPen(QColor(color))
    p.setWidthF(w)
    p.setCapStyle(Qt.RoundCap)
    p.setJoinStyle(Qt.RoundJoin)
    return p


def room_icon(kind: str, size: int = 56, color: str = TEXT) -> QIcon:
    """Ícone de linha para cada tipo de ambiente (desenhado em vetor)."""
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(size * 2 / 48.0, size * 2 / 48.0)
    p.setPen(_pen(color, 2.0))
    p.setBrush(Qt.NoBrush)
    if kind == "sala":            # sofá
        p.drawRoundedRect(QRectF(8, 20, 32, 12), 3, 3)
        p.drawRoundedRect(QRectF(5, 24, 6, 12), 2, 2)
        p.drawRoundedRect(QRectF(37, 24, 6, 12), 2, 2)
        p.drawLine(QPointF(10, 36), QPointF(38, 36))
        p.drawLine(QPointF(11, 36), QPointF(11, 40)); p.drawLine(QPointF(37, 36), QPointF(37, 40))
    elif kind == "quarto":        # cama
        p.drawRoundedRect(QRectF(6, 14, 8, 24), 2, 2)
        p.drawRoundedRect(QRectF(6, 26, 36, 8), 2, 2)
        p.drawRoundedRect(QRectF(16, 20, 9, 6), 2, 2)
        p.drawLine(QPointF(42, 34), QPointF(42, 39)); p.drawLine(QPointF(8, 38), QPointF(8, 39))
    elif kind == "cozinha":       # panela + fogão
        p.drawRoundedRect(QRectF(8, 30, 32, 10), 2, 2)
        path = QPainterPath(QPointF(13, 18)); path.lineTo(35, 18); path.lineTo(33, 28); path.lineTo(15, 28)
        path.closeSubpath(); p.drawPath(path)
        p.drawLine(QPointF(9, 18), QPointF(13, 18)); p.drawLine(QPointF(35, 18), QPointF(39, 18))
        p.drawLine(QPointF(20, 12), QPointF(20, 9)); p.drawLine(QPointF(28, 12), QPointF(28, 9))
    elif kind == "banheiro":      # banheira
        p.drawRoundedRect(QRectF(6, 24, 36, 10), 4, 4)
        p.drawLine(QPointF(11, 34), QPointF(9, 39)); p.drawLine(QPointF(37, 34), QPointF(39, 39))
        p.drawLine(QPointF(12, 24), QPointF(12, 12)); p.drawArc(QRectF(12, 9, 8, 6), 0, 180 * 16)
        p.drawLine(QPointF(20, 12), QPointF(20, 15))
    elif kind == "escritorio":    # mesa + monitor
        p.drawLine(QPointF(6, 30), QPointF(42, 30))
        p.drawLine(QPointF(9, 30), QPointF(9, 40)); p.drawLine(QPointF(39, 30), QPointF(39, 40))
        p.drawRoundedRect(QRectF(16, 12, 16, 11), 2, 2)
        p.drawLine(QPointF(24, 23), QPointF(24, 27)); p.drawLine(QPointF(20, 27), QPointF(28, 27))
    else:                         # sala de jantar: mesa + cadeiras
        p.drawLine(QPointF(12, 24), QPointF(36, 24))
        p.drawLine(QPointF(15, 24), QPointF(15, 38)); p.drawLine(QPointF(33, 24), QPointF(33, 38))
        p.drawLine(QPointF(5, 16), QPointF(5, 38)); p.drawLine(QPointF(5, 30), QPointF(10, 30))
        p.drawLine(QPointF(43, 16), QPointF(43, 38)); p.drawLine(QPointF(38, 30), QPointF(43, 30))
        p.drawEllipse(QPointF(24, 19), 3, 3)
    p.end()
    return QIcon(pm)


def app_icon(size: int = 256) -> QPixmap:
    """Logotipo: casa estilizada com uma poltrona dentro, sobre fundo arredondado."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    s = size / 48.0
    p.scale(s, s)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(SURFACE_2))
    p.drawRoundedRect(QRectF(1, 1, 46, 46), 11, 11)
    p.setPen(_pen(ACCENT, 3.0))
    path = QPainterPath(QPointF(9, 23)); path.lineTo(24, 10); path.lineTo(39, 23)
    p.drawPath(path)
    p.drawLine(QPointF(13, 20), QPointF(13, 38)); p.drawLine(QPointF(35, 20), QPointF(35, 38))
    p.setPen(_pen(TEXT, 2.4))
    p.drawRoundedRect(QRectF(18, 26, 12, 6), 2, 2)
    p.drawLine(QPointF(17, 32), QPointF(17, 35)); p.drawLine(QPointF(31, 32), QPointF(31, 35))
    p.drawLine(QPointF(17, 28), QPointF(17, 32)); p.drawLine(QPointF(31, 28), QPointF(31, 32))
    p.end()
    return pm
