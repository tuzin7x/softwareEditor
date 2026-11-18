"""Visual do Lead Finder derivado dos tokens do 7xAi (``mobilia.ui.theme``), sem alterar o tema global.

A folha de estilo daqui é aplicada só à página do módulo e completa o QSS global com o que ele não cobre
(tabelas, abas, listas, campos de data). Cartões "vidro": gradiente translúcido, filete claro e brilho suave
na cor de destaque — Qt não desfoca o que está atrás, então o efeito é simulado com camadas translúcidas.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from mobilia.ui import theme
from mobilia.ui.widgets import on_click

from ..models import STATUS_KEYS

# cores de estado do CRM (as que não existem no tema são tons frios discretos, só nas etiquetas)
STATUS_COLOR = {
    "nao_contatado": theme.MUTED, "contato_iniciado": theme.ACCENT_HI, "aguardando": theme.NOTICE_TEXT,
    "interessado": theme.OK, "demo_enviada": "#B9A3E3", "negociacao": "#8FB3D9", "convertido": "#7FD1AE",
    "sem_interesse": theme.DISABLED, "nao_contatar": theme.DANGER,
}
assert set(STATUS_COLOR) == set(STATUS_KEYS)
WA_COLOR = theme.OK
PRIORITY_COLOR = {0: theme.DISABLED, 1: theme.MUTED, 2: theme.NOTICE_TEXT, 3: theme.ACCENT_HI}


def rgba(hex_color: str, alpha: float) -> str:
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha:.3f})"


QSS = f"""
QFrame#lfGlass {{
  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255,255,255,0.050), stop:1 rgba(255,255,255,0.014));
  border: 1px solid rgba(255,255,255,0.075); border-radius: 16px; }}
QFrame#lfGlassHero {{
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {rgba(theme.ACCENT, 0.14)},
                              stop:0.55 rgba(255,255,255,0.03), stop:1 rgba(255,255,255,0.012));
  border: 1px solid {rgba(theme.ACCENT, 0.32)}; border-radius: 18px; }}
QFrame#lfKpi {{
  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255,255,255,0.055), stop:1 rgba(255,255,255,0.012));
  border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; }}
QLabel#lfKpiValue {{ font-family: {theme.FONT_DISPLAY}; font-size: 28px; font-weight: 500; color: {theme.TEXT}; }}
QLabel#lfKpiLabel {{ color: {theme.MUTED}; font-size: 11px; font-weight: 600; }}
QLabel#lfHeroTitle {{ font-family: {theme.FONT_DISPLAY}; font-size: 26px; font-weight: 500; }}
QLabel#lfBig {{ font-family: {theme.FONT_DISPLAY}; font-size: 20px; font-weight: 500; }}
QLabel#lfMono {{ font-family: {theme.FONT_MONO}; color: {theme.TEXT_2}; font-size: 12px; }}
QLabel#lfFieldName {{ color: {theme.MUTED}; font-size: 11px; font-weight: 600; }}
QLabel#lfFieldValue {{ color: {theme.TEXT}; font-size: 13px; }}
QLabel#lfSource {{ color: {theme.MUTED}; font-size: 11px; }}
QLabel#lfWarn {{ color: {theme.NOTICE_TEXT}; font-size: 12px; }}
QLabel#lfDanger {{ color: {theme.DANGER}; font-size: 12px; }}
QLabel#lfOk {{ color: {theme.OK}; font-size: 12px; }}

QTableView#lfTable {{
  background: {theme.SURFACE}; alternate-background-color: #18181B; border: 1px solid {theme.HAIRLINE};
  border-radius: 12px; gridline-color: transparent; selection-background-color: {theme.SELECTED_BG};
  selection-color: {theme.TEXT}; }}
QTableView#lfTable::item {{ padding: 0 8px; border-bottom: 1px solid {theme.HAIRLINE}; }}
QTableView#lfTable::item:selected {{ background: {theme.SELECTED_BG}; color: {theme.TEXT}; }}
QTableView#lfTable::item:hover {{ background: {rgba(theme.ACCENT, 0.05)}; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: {theme.SURFACE_2}; color: {theme.TEXT_2}; border: none;
  border-bottom: 1px solid {theme.HAIRLINE_STRONG}; padding: 9px 8px; font-size: 11px; font-weight: 600; }}
QHeaderView::section:hover {{ color: {theme.TEXT}; }}
QTableCornerButton::section {{ background: {theme.SURFACE_2}; border: none; }}

QListWidget#lfList {{ background: transparent; border: none; outline: none; }}
QListWidget#lfList::item {{ border-bottom: 1px solid {theme.HAIRLINE}; padding: 8px 4px; color: {theme.TEXT_2}; }}
QListWidget#lfList::item:selected {{ background: {theme.SELECTED_BG}; color: {theme.TEXT}; border-radius: 8px; }}
QListWidget#lfList::item:hover {{ background: {rgba(theme.ACCENT, 0.05)}; }}

QDateTimeEdit, QSpinBox, QDoubleSpinBox {{ background: {theme.SURFACE_2}; border: 1px solid {theme.HAIRLINE_STRONG};
  border-radius: 10px; padding: 6px 10px; min-height: 24px; color: {theme.TEXT}; }}
QDateTimeEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {theme.ACCENT}; }}
QSplitter::handle {{ background: transparent; width: 10px; }}
QScrollArea#lfScroll, QScrollArea#lfScroll > QWidget > QWidget {{ background: transparent; border: none; }}

QPushButton#lfWa {{ background: {rgba(WA_COLOR, 0.12)}; color: {WA_COLOR}; border: 1px solid {rgba(WA_COLOR, 0.45)};
  border-radius: 10px; font-weight: 600; padding: 0 14px; }}
QPushButton#lfWa:hover {{ background: {rgba(WA_COLOR, 0.22)}; border-color: {WA_COLOR}; }}
QPushButton#lfWa:disabled {{ color: {theme.DISABLED}; background: {theme.SURFACE_2}; border-color: {theme.HAIRLINE}; }}
QPushButton#lfSmall {{ min-height: 28px; padding: 0 10px; font-size: 12px; border-radius: 8px; }}
"""


def glass(kind: str = "lfGlass", glow: bool = False) -> QFrame:
    f = QFrame()
    f.setObjectName(kind)
    if glow:
        fx = QGraphicsDropShadowEffect(f)
        c = QColor(theme.ACCENT); c.setAlphaF(0.16)
        fx.setColor(c); fx.setBlurRadius(42); fx.setOffset(0, 10)
        f.setGraphicsEffect(fx)
    return f


def card(title: str = "", glow: bool = False, margins=(18, 16, 18, 16)) -> tuple[QFrame, QVBoxLayout]:
    f = glass(glow=glow)
    lay = QVBoxLayout(f)
    lay.setContentsMargins(*margins)
    lay.setSpacing(10)
    if title:
        t = QLabel(title.upper()); t.setObjectName("section"); theme.tracked(t)
        lay.addWidget(t)
    return f, lay


def label(text: str = "", name: str = "body", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    lb.setObjectName(name)
    lb.setWordWrap(wrap)
    if wrap:
        lb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    return lb


def button(text: str, fn=None, kind: str = "", icon=None, tip: str = "", small: bool = False) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    elif small:
        b.setObjectName("lfSmall")
    if icon is not None:
        b.setIcon(icon)
    b.setCursor(Qt.PointingHandCursor)
    if tip:
        b.setToolTip(tip)
    if small and kind:
        b.setMinimumHeight(30); b.setStyleSheet("min-height: 30px; font-size: 12px;")
    if fn is not None:
        on_click(b, fn)
    return b


def row(*widgets, spacing: int = 8, stretch_at: int | None = None) -> QHBoxLayout:
    lay = QHBoxLayout(); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(spacing)
    for i, w in enumerate(widgets):
        if w is None:
            lay.addStretch(1)
        elif isinstance(w, QWidget):
            lay.addWidget(w, 1 if i == stretch_at else 0)
        else:
            lay.addLayout(w)
    return lay


def pill_html(text: str, color: str) -> str:
    return (f'<span style="color:{color}; background:{rgba(color, 0.14)}; border-radius:8px;">'
            f'&nbsp;{text}&nbsp;</span>')
