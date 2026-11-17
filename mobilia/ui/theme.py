"""Tema visual do 7xAi — "estúdio premium escuro".

Grafite quente + latão/champanhe, filetes de 1 px, cantos generosos e a foto como protagonista.
Tipografia: Inter (texto) e Inter Display (títulos e números), embutidas em ``ui/fonts``
(SIL Open Font License, ver ``fonts/OFL.txt``); sem elas cai para Segoe UI.

Os ícones de linha são desenhados em QPainter (sem arquivos externos) na mesma linguagem da
marca: traço fino de desenho de arquitetura, terminais retos e uma linha de piso sob cada móvel.
Contraste do texto (WCAG): ver ``CONTRAST`` — tudo >= 4.5:1 sobre as superfícies onde aparece.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QIcon, QImage, QPainter, QPainterPath, QPalette, QPen,
                           QPixmap)

from . import brand

# ------------------------------------------------------------------------------------------ paleta
BG = "#0F0F11"            # fundo da janela (grafite quase preto)
SURFACE = "#161619"       # cartões
SURFACE_2 = "#1E1E22"     # campos, botões secundários
SURFACE_3 = "#27272C"     # hover / menus
TILE = "#1A1A1E"          # ladrilhos de ambiente
STAGE = "#0B0B0D"         # palco da foto/vídeo
HAIRLINE = "#29292E"      # filetes de 1 px
HAIRLINE_STRONG = "#3A3A41"
TEXT = brand.IVORY        # #F4EFE7 marfim
TEXT_2 = "#CBC5BB"        # texto secundário
MUTED = "#A39D94"         # legendas
DISABLED = "#8E8981"
ACCENT = brand.BRASS      # #CDAE76 latão
ACCENT_HI = brand.BRASS_HI
ACCENT_LO = brand.BRASS_LO
ON_ACCENT = "#17130C"
SELECTED_BG = "#221E17"   # latão a ~8% sobre o cartão
ACCENT_LINE = "#5E5038"   # latão a ~35%
OK = "#8FCB9B"            # sálvia
DANGER = "#EB8F84"
NOTICE_TEXT = "#E9D3A6"
TRACK = "#45454D"         # trilho da barra de progresso (visível já em 0 %)
PENDING_MARK = "#6E6A63"  # contorno das etapas pendentes: 3.2:1 sobre o cartão (WCAG 1.4.11)

# compatibilidade com nomes antigos
BORDER = HAIRLINE_STRONG

# Contraste WCAG medido (texto sobre fundo), ver README do design.
CONTRAST = {
    "TEXT/SURFACE": 15.8, "TEXT_2/SURFACE": 10.5, "MUTED/SURFACE": 6.7, "MUTED/SURFACE_2": 6.2,
    "ACCENT/SURFACE": 8.5, "OK/SURFACE": 9.6, "DANGER/SURFACE": 7.6, "NOTICE_TEXT/SELECTED_BG": 11.3,
    "ON_ACCENT/ACCENT": 8.7, "DISABLED/SURFACE_2": 4.8,
}

FONT_UI = '"Inter", "Segoe UI Variable Text", "Segoe UI", "Noto Sans", sans-serif'
FONT_DISPLAY = '"Inter Display", "Inter", "Segoe UI Variable Display", "Segoe UI", sans-serif'
FONT_MONO = '"Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace'

_HERE = Path(__file__).resolve().parent
_RES = _HERE / "res"


def _res(name: str) -> str:
    return (_RES / name).as_posix()


QSS = f"""
* {{ font-family: {FONT_UI}; font-size: 13px; color: {TEXT}; outline: none; }}
QMainWindow, QWidget#root, QDialog {{ background: {BG}; }}
QMessageBox {{ background: {SURFACE}; }}
QMessageBox QLabel {{ color: {TEXT}; }}
QToolTip {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {HAIRLINE_STRONG}; border-radius: 6px;
            padding: 6px 8px; }}

/* tipografia */
QLabel#title {{ font-family: {FONT_DISPLAY}; font-size: 22px; font-weight: 500; }}
QLabel#subtitle, QLabel#tagline, QLabel#body {{ color: {TEXT_2}; font-size: 13px; }}
QLabel#headline {{ font-family: {FONT_DISPLAY}; font-size: 24px; font-weight: 500; }}
QLabel#section {{ color: {TEXT_2}; font-size: 11px; font-weight: 600; }}
QLabel#stepnum {{ color: {ACCENT}; font-family: {FONT_DISPLAY}; font-size: 12px; font-weight: 500; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#hint, QLabel#caption {{ color: {MUTED}; font-size: 12px; }}
QLabel#filename {{ font-size: 15px; font-weight: 600; }}
QLabel#result {{ font-family: {FONT_DISPLAY}; font-size: 26px; font-weight: 500; }}
QLabel#privacy {{ color: {MUTED}; font-size: 12px; }}
QLabel#trust {{ color: {TEXT_2}; font-size: 12px; }}
QLabel#stagecap {{ color: {MUTED}; font-size: 12px; font-style: italic; }}
QFrame#chip {{ background: {SURFACE}; border: 1px solid {ACCENT_LINE}; border-radius: 15px; }}
QLabel#chiptext {{ color: {TEXT_2}; font-size: 12px; }}
QLabel#pct {{ font-family: {FONT_DISPLAY}; }}
QLabel#notice {{ color: {NOTICE_TEXT}; background: {SELECTED_BG}; border: 1px solid {ACCENT_LINE};
                 border-radius: 10px; padding: 10px 12px; font-size: 12px; }}
QLabel#seal {{ color: {TEXT_2}; font-size: 12px; }}
QFrame#sealbox {{ background: {SURFACE_2}; border: 1px solid {HAIRLINE}; border-radius: 10px; }}
QFrame#vrule {{ background: {HAIRLINE_STRONG}; border: none; }}
QFrame#rule {{ background: {HAIRLINE}; border: none; }}

/* cartões e palco */
QFrame#card {{ background: {SURFACE}; border: 1px solid {HAIRLINE}; border-radius: 16px; }}
QFrame#stage {{ background: {STAGE}; border: 1px solid {HAIRLINE}; border-radius: 16px; }}
QFrame#drop {{ border: 1px solid {HAIRLINE}; border-radius: 16px;
               background: qradialgradient(cx:0.5, cy:0.42, radius:0.75, fx:0.5, fy:0.42,
                                           stop:0 #1B1B1F, stop:1 {STAGE}); }}
QFrame#drop[hover="true"] {{ border: 1px solid {ACCENT};
               background: qradialgradient(cx:0.5, cy:0.42, radius:0.75, fx:0.5, fy:0.42,
                                           stop:0 #241F17, stop:1 {STAGE}); }}
QFrame#drop[filled="true"] {{ background: {STAGE}; }}

/* botões */
QPushButton {{ background: {SURFACE_2}; border: 1px solid {HAIRLINE_STRONG}; border-radius: 10px;
               padding: 0 16px; min-height: 38px; font-weight: 500; }}
QPushButton:hover {{ background: {SURFACE_3}; border-color: #4C4C54; }}
QPushButton:pressed {{ background: #19191C; }}
QPushButton:disabled {{ color: {DISABLED}; background: {SURFACE_2}; border-color: {HAIRLINE}; }}
QPushButton#primary {{ color: {ON_ACCENT}; font-size: 14px; font-weight: 600; min-height: 46px; border-radius: 12px;
                       border: 1px solid #DCC394;
                       background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #DDC08A, stop:1 {ACCENT}); }}
QPushButton#primary:hover {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {ACCENT_HI}, stop:1 #D6B97F); }}
QPushButton#primary:pressed {{ background: {ACCENT_LO}; }}
QPushButton#primary:disabled {{ color: {DISABLED}; background: {SURFACE_2}; border: 1px solid {HAIRLINE}; }}
QPushButton#outline {{ background: transparent; color: {ACCENT_HI}; border: 1px solid {ACCENT_LINE}; }}
QPushButton#outline:hover {{ background: {SELECTED_BG}; border-color: {ACCENT}; }}
QPushButton#ghost {{ background: transparent; border: 1px solid transparent; color: {TEXT_2}; }}
QPushButton#ghost:hover {{ background: {SURFACE_2}; color: {TEXT}; border-color: {HAIRLINE}; }}
QPushButton#danger {{ background: transparent; border: 1px solid #4A302D; color: {DANGER}; }}
QPushButton#danger:hover {{ background: #211716; border-color: {DANGER}; }}
QPushButton#danger:disabled {{ color: {DISABLED}; border-color: {HAIRLINE}; }}

/* ladrilhos de ambiente */
QToolButton#room {{ background: {TILE}; border: 1px solid {HAIRLINE}; border-radius: 12px;
                    padding: 12px 4px 10px 4px; color: {TEXT_2}; font-size: 12px; }}
QToolButton#room:hover {{ background: #202025; border-color: {HAIRLINE_STRONG}; color: {TEXT}; }}
QToolButton#room:checked {{ background: {SELECTED_BG}; border: 1px solid {ACCENT}; color: {TEXT}; font-weight: 600; }}

/* redesenho: chips (ambiente, evento, formato) e seletor segmentado (versões) */
QToolButton#chip {{ background: {TILE}; border: 1px solid {HAIRLINE}; border-radius: 10px; padding: 6px 4px;
                    color: {TEXT_2}; font-size: 12px; }}
QToolButton#chip:hover {{ background: #202025; border-color: {HAIRLINE_STRONG}; color: {TEXT}; }}
QToolButton#chip:checked {{ background: {SELECTED_BG}; border: 1px solid {ACCENT}; color: {TEXT}; font-weight: 600; }}
QToolButton#chip:disabled {{ color: {DISABLED}; border-color: {HAIRLINE}; background: {SURFACE}; }}
QToolButton#seg {{ background: {SURFACE_2}; border: 1px solid {HAIRLINE_STRONG}; padding: 0; min-height: 34px;
                   color: {TEXT_2}; font-weight: 500; border-radius: 0; }}
QToolButton#seg[pos="first"] {{ border-top-left-radius: 10px; border-bottom-left-radius: 10px; }}
QToolButton#seg[pos="last"] {{ border-top-right-radius: 10px; border-bottom-right-radius: 10px; }}
QToolButton#seg:hover {{ background: {SURFACE_3}; color: {TEXT}; }}
QToolButton#seg:checked {{ background: {SELECTED_BG}; border-color: {ACCENT}; color: {ACCENT_HI}; font-weight: 600; }}
QPushButton#link {{ background: transparent; border: none; color: {ACCENT_HI}; padding: 0; min-height: 22px;
                    text-align: left; font-weight: 500; }}
QPushButton#link:hover {{ color: {TEXT}; text-decoration: underline; }}
QLineEdit, QPlainTextEdit {{ background: {SURFACE_2}; border: 1px solid {HAIRLINE_STRONG}; border-radius: 10px;
                             padding: 8px 10px; color: {TEXT}; selection-background-color: {ACCENT};
                             selection-color: {ON_ACCENT}; }}
QLineEdit {{ min-height: 22px; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {TRACK}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT_HI}; border: 2px solid {SURFACE}; width: 14px; height: 14px;
                              margin: -7px 0; border-radius: 9px; }}
QFrame#taskbox {{ background: {SURFACE_2}; border: 1px solid {HAIRLINE}; border-radius: 12px; }}
QFrame#stepper {{ background: transparent; }}
QLabel#stepdot {{ color: {MUTED}; font-size: 12px; }}

/* qualidade */
QComboBox {{ background: {SURFACE_2}; border: 1px solid {HAIRLINE_STRONG}; border-radius: 10px;
             padding: 0 12px; min-height: 38px; }}
QComboBox:hover {{ border-color: #4C4C54; }}
QComboBox:on {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 34px; subcontrol-origin: padding; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url({_res("chevron.png")}); width: 12px; height: 8px; }}
QComboBox QAbstractItemView {{ background: {SURFACE_3}; border: 1px solid {HAIRLINE_STRONG}; padding: 4px;
                               selection-background-color: {SELECTED_BG}; selection-color: {TEXT}; }}
QComboBox QAbstractItemView::item {{ min-height: 32px; padding: 0 10px; border-radius: 6px; }}
QComboBox QAbstractItemView::item:selected {{ background: {SELECTED_BG}; color: {ACCENT_HI}; }}

/* progresso */
QProgressBar {{ background: {TRACK}; border: none; border-radius: 3px; min-height: 6px; max-height: 6px; }}
QProgressBar::chunk {{ border-radius: 3px;
                       background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_LO}, stop:1 {ACCENT_HI}); }}

/* caixa de seleção */
QCheckBox {{ color: {TEXT_2}; spacing: 10px; font-size: 12px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid #55555D; background: {SURFACE_2}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url({_res("check.png")}); }}

/* rodapé */
QStatusBar {{ background: {BG}; color: {MUTED}; border-top: 1px solid {HAIRLINE}; font-size: 12px;
              min-height: 30px; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {MUTED}; font-size: 12px; padding-right: 20px; }}

QTextEdit {{ background: {STAGE}; border: 1px solid {HAIRLINE}; border-radius: 10px; padding: 8px;
             font-family: {FONT_MONO}; font-size: 12px; color: {TEXT_2}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
QScrollBar::handle:vertical {{ background: {HAIRLINE_STRONG}; border-radius: 3px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""


# ------------------------------------------------------------------------------------ fontes/paleta

_fonts_loaded = False


def load_fonts() -> None:
    """Registra as fontes embutidas (idempotente; precisa de QGuiApplication)."""
    global _fonts_loaded
    if _fonts_loaded:
        return
    _fonts_loaded = True
    for f in sorted((_HERE / "fonts").glob("*.otf")):
        QFontDatabase.addApplicationFont(str(f))


def assets_ok() -> bool:
    """Fontes e imagens da interface presentes (o autoteste do .exe falha se o build esquecê-las)."""
    load_fonts()
    fonts = "Inter" in QFontDatabase.families() and "Inter Display" in QFontDatabase.families()
    return fonts and all((_RES / n).is_file() for n in ("chevron.png", "check.png"))


def palette() -> QPalette:
    pal = QPalette()
    roles = {
        QPalette.Window: BG, QPalette.WindowText: TEXT, QPalette.Base: SURFACE_2, QPalette.AlternateBase: SURFACE,
        QPalette.Text: TEXT, QPalette.Button: SURFACE_2, QPalette.ButtonText: TEXT, QPalette.ToolTipBase: SURFACE_3,
        QPalette.ToolTipText: TEXT, QPalette.Highlight: ACCENT, QPalette.HighlightedText: ON_ACCENT,
        QPalette.PlaceholderText: MUTED, QPalette.Link: ACCENT_HI, QPalette.BrightText: "#FFFFFF",
        QPalette.Mid: HAIRLINE_STRONG, QPalette.Dark: STAGE, QPalette.Light: SURFACE_3,
    }
    for role, c in roles.items():
        pal.setColor(role, QColor(c))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(DISABLED))
    return pal


def apply(app) -> None:
    """Tema completo no QApplication: estilo Fusion, paleta escura, fontes e folha de estilo."""
    load_fonts()
    try:  # barra de título escura no Windows 11 (Qt >= 6.8)
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except AttributeError:
        pass
    app.setStyle("Fusion")
    app.setPalette(palette())
    f = QFont("Inter")
    f.setPixelSize(13)
    f.setHintingPreference(QFont.PreferNoHinting)
    app.setFont(f)
    app.setStyleSheet(QSS)
    app.setWindowIcon(window_icon())


def tracked(label, spacing: float = 1.4):
    """Espaçamento entre letras (rótulos em caixa alta); QSS não tem letter-spacing."""
    f = label.font()
    f.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    label.setFont(f)
    return label


# ------------------------------------------------------------------------------------------- ícones

def _pen(color: str | QColor, w: float, cap=Qt.SquareCap) -> QPen:
    p = QPen(QColor(color))
    p.setWidthF(w)
    p.setCapStyle(cap)
    p.setJoinStyle(Qt.MiterJoin)
    return p


def _canvas(size: int, dpr: float = 2.0) -> tuple[QPixmap, QPainter]:
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    return pm, p


def _draw_room(p: QPainter, kind: str, color: str) -> None:
    """Elevação do móvel numa grade de 48 (piso em y = 38)."""
    p.setPen(_pen(color, 1.7))
    p.setBrush(Qt.NoBrush)
    line = lambda x1, y1, x2, y2: p.drawLine(QPointF(x1, y1), QPointF(x2, y2))  # noqa: E731
    rect = lambda x, y, w, h, r=0: p.drawRoundedRect(QRectF(x, y, w, h), r, r)  # noqa: E731
    if kind == "sala":            # sofá
        rect(10, 17, 28, 10, 2)
        rect(6, 22, 6, 12, 1.5); rect(36, 22, 6, 12, 1.5)
        line(12, 30, 36, 30)
        line(9, 34, 9, 38); line(39, 34, 39, 38)
    elif kind == "quarto":        # cama com cabeceira e travesseiro
        rect(6, 13, 6, 25)
        rect(12, 25, 30, 7, 1.5)
        rect(15, 20, 9, 5, 1.5)
        line(40, 32, 40, 38)
    elif kind == "cozinha":       # bancada com portas, panela e pendente
        line(6, 24, 42, 24)
        rect(8, 24, 32, 14)
        line(24, 27, 24, 38)
        line(20, 30, 20, 32); line(28, 30, 28, 32)
        rect(12, 17, 10, 7, 1); line(10, 18.5, 12, 18.5); line(22, 18.5, 24, 18.5)
        line(33, 4, 33, 10)
        path = QPainterPath(QPointF(29, 15)); path.lineTo(30.5, 10); path.lineTo(35.5, 10); path.lineTo(37, 15)
        path.closeSubpath(); p.drawPath(path)
    elif kind == "banheiro":      # banheira com pés e misturador
        path = QPainterPath(QPointF(6, 24)); path.lineTo(42, 24); path.lineTo(40, 31)
        path.quadTo(39, 34, 35, 34); path.lineTo(13, 34); path.quadTo(9, 34, 8, 31); path.closeSubpath()
        p.drawPath(path)
        line(12, 34, 10, 38); line(36, 34, 38, 38)
        line(11, 24, 11, 13); line(11, 13, 17, 13); line(17, 13, 17, 16)
    elif kind == "escritorio":    # mesa, monitor e luminária
        line(6, 27, 42, 27)
        line(9, 27, 9, 38); line(39, 27, 39, 38)
        rect(14, 12, 16, 10, 1); line(22, 22, 22, 27)
        line(35, 27, 35, 18); line(35, 18, 39, 14)
        path = QPainterPath(QPointF(37, 12)); path.lineTo(42, 17); p.drawPath(path)
    elif kind == "varanda":       # guarda-corpo, espreguiçadeira e vaso com folhas
        line(4, 20, 44, 20)
        for x in (8, 14, 20, 26, 32, 38):
            line(x, 20, x, 38)
        rect(9, 28, 18, 5, 1.5); line(9, 28, 6, 22); line(11, 33, 11, 38); line(25, 33, 25, 38)
        rect(33, 31, 7, 7, 1)
        p.drawArc(QRectF(30, 22, 8, 10), 0, 180 * 16); p.drawArc(QRectF(35, 20, 8, 12), 0, 180 * 16)
    elif kind == "fachada":       # casa: telhado, porta e janela
        path = QPainterPath(QPointF(5, 22)); path.lineTo(24, 7); path.lineTo(43, 22); p.drawPath(path)
        line(9, 19, 9, 38); line(39, 19, 39, 38)
        rect(14, 26, 7, 12, 0.8)
        rect(26, 25, 9, 7, 0.8); line(30.5, 25, 30.5, 32)
        line(33, 10, 33, 15); line(33, 10, 37, 10); line(37, 10, 37, 18)
    else:                         # sala de jantar: mesa, cadeiras e pendente
        line(13, 25, 35, 25)
        line(16, 25, 16, 38); line(32, 25, 32, 38)
        line(6, 15, 6, 38); line(6, 30, 11, 30); line(11, 30, 11, 38)
        line(42, 15, 42, 38); line(37, 30, 42, 30); line(37, 30, 37, 38)
        line(24, 4, 24, 11)
        p.drawArc(QRectF(19, 11, 10, 8), 0, 180 * 16); line(19, 15, 29, 15)
    # linha de piso: o mesmo filete em todos os ícones
    c = QColor(color); c.setAlphaF(0.55)
    p.setPen(_pen(c, 1.2))
    line(3, 38.5, 45, 38.5)


def room_pixmap(kind: str, size: int = 40, color: str = TEXT_2, dpr: float = 2.0) -> QPixmap:
    pm, p = _canvas(size, dpr)
    p.scale(size / 48.0, size / 48.0)
    _draw_room(p, kind, color)
    p.end()
    return pm


def _draw_function(p: QPainter, kind: str, color: str) -> None:
    """Ícones das funções do redesenho (grade de 48), no traço dos ícones de ambiente."""
    p.setPen(_pen(color, 1.7, Qt.RoundCap))
    p.setBrush(Qt.NoBrush)
    line = lambda x1, y1, x2, y2: p.drawLine(QPointF(x1, y1), QPointF(x2, y2))  # noqa: E731
    rect = lambda x, y, w, h, r=0: p.drawRoundedRect(QRectF(x, y, w, h), r, r)  # noqa: E731
    soft = QColor(color); soft.setAlphaF(0.55)
    if kind == "mudar_estilo":        # paleta de pintor + pincel
        path = QPainterPath(QPointF(24, 8))
        path.cubicTo(36, 8, 43, 16, 42, 24); path.cubicTo(41, 30, 34, 29, 32, 32); path.cubicTo(30, 35, 33, 40, 26, 40)
        path.cubicTo(14, 40, 6, 33, 6, 24); path.cubicTo(6, 15, 13, 8, 24, 8)
        p.drawPath(path)
        for x, y in ((15, 19), (23, 15), (31, 18), (14, 28)):
            p.drawEllipse(QPointF(x, y), 2.4, 2.4)
    elif kind == "reforma":           # sofá novo + setas de renovação
        rect(12, 24, 24, 8, 2); rect(8, 28, 5, 9, 1.5); rect(35, 28, 5, 9, 1.5); line(13, 34, 35, 34)
        p.drawArc(QRectF(15, 5, 18, 14), 20 * 16, 140 * 16); p.drawArc(QRectF(15, 5, 18, 14), 200 * 16, 140 * 16)
        line(31, 7, 32, 10.5); line(32, 10.5, 28.5, 11)
        line(17, 17, 16, 13.5); line(16, 13.5, 19.5, 13)
    elif kind == "redesign_simples":  # varinha com brilhos
        line(9, 39, 30, 18)
        p.setPen(_pen(color, 1.5, Qt.RoundCap))
        for cx, cy, r in ((35, 13, 5.5), (17, 12, 3.2), (38, 30, 3.2)):
            line(cx, cy - r, cx, cy + r); line(cx - r, cy, cx + r, cy)
    elif kind == "mobiliar":          # cômodo vazio em perspectiva recebendo um sofá
        p.setPen(_pen(soft, 1.2))
        line(4, 6, 14, 14); line(44, 6, 34, 14); line(4, 42, 14, 32); line(44, 42, 34, 32)
        p.setPen(_pen(color, 1.5, Qt.RoundCap))
        rect(14, 14, 20, 18)
        rect(17, 24, 14, 5, 1.2); line(17, 29, 17, 31); line(31, 29, 31, 31)
        line(38, 3, 38, 9); line(35, 6, 41, 6)
    elif kind == "editar_area":       # seleção tracejada + pincel
        pen = _pen(soft, 1.4); pen.setDashPattern([2.5, 2.5]); p.setPen(pen)
        rect(5, 10, 26, 22, 3)
        p.setPen(_pen(color, 1.7, Qt.RoundCap))
        line(42, 8, 28, 26)
        path = QPainterPath(QPointF(28, 26)); path.cubicTo(24, 25, 21, 29, 21, 33); path.lineTo(19, 36)
        path.cubicTo(25, 36, 30, 32, 28, 26); p.drawPath(path)
    elif kind == "fachada":
        _draw_room(p, "fachada", color)
        return
    elif kind == "evento":            # balões e bandeirolas
        p.drawEllipse(QRectF(9, 9, 12, 15)); p.drawEllipse(QRectF(22, 6, 12, 15))
        path = QPainterPath(QPointF(15, 24)); path.cubicTo(13, 30, 18, 33, 16, 40); p.drawPath(path)
        path = QPainterPath(QPointF(28, 21)); path.cubicTo(26, 28, 31, 32, 28, 40); p.drawPath(path)
        p.setPen(_pen(soft, 1.3))
        path = QPainterPath(QPointF(30, 30)); path.quadTo(37, 33, 45, 30); p.drawPath(path)
        for x in (34, 40):
            tri = QPainterPath(QPointF(x - 2.5, 31.5)); tri.lineTo(x, 37); tri.lineTo(x + 2.5, 31.5); p.drawPath(tri)
    elif kind == "texto":             # descrição vira imagem
        rect(6, 8, 36, 30, 4)
        line(12, 15, 22, 15); line(17, 15, 17, 26)
        p.setPen(_pen(soft, 1.4, Qt.RoundCap))
        line(26, 17, 36, 17); line(26, 22, 36, 22); line(12, 31, 36, 31)
        return
    else:
        _draw_room(p, "sala", color)
        return
    c = QColor(color); c.setAlphaF(0.55)
    if kind in ("reforma", "redesign_simples"):
        p.setPen(_pen(c, 1.2)); line(3, 42.5, 45, 42.5)


def function_pixmap(kind: str, size: int = 40, color: str = TEXT_2, dpr: float = 2.0) -> QPixmap:
    pm, p = _canvas(size, dpr)
    p.scale(size / 48.0, size / 48.0)
    _draw_function(p, kind, color)
    p.end()
    return pm


def function_icon(kind: str, size: int = 40, color: str = TEXT_2, on_color: str = ACCENT_HI) -> QIcon:
    icon = QIcon()
    off = function_pixmap(kind, size, color)
    on = function_pixmap(kind, size, on_color)
    for mode in (QIcon.Normal, QIcon.Active, QIcon.Selected):
        icon.addPixmap(off, mode, QIcon.Off)
        icon.addPixmap(on, mode, QIcon.On)
    return icon


def room_icon(kind: str, size: int = 40, color: str = TEXT_2, on_color: str = ACCENT_HI) -> QIcon:
    """Ícone de linha do ambiente: marfim apagado; latão quando o ladrilho está marcado."""
    icon = QIcon()
    off = room_pixmap(kind, size, color)
    on = room_pixmap(kind, size, on_color)
    for mode in (QIcon.Normal, QIcon.Active, QIcon.Selected):
        icon.addPixmap(off, mode, QIcon.Off)
        icon.addPixmap(on, mode, QIcon.On)
    icon.addPixmap(room_pixmap(kind, size, TEXT), QIcon.Active, QIcon.Off)
    return icon


def line_icon(name: str, size: int = 16, color: str = TEXT) -> QIcon:
    """Ícones pequenos dos botões, grade de 16."""
    pm, p = _canvas(size)
    p.scale(size / 16.0, size / 16.0)
    p.setPen(_pen(color, 1.4, Qt.RoundCap))
    p.setBrush(Qt.NoBrush)
    line = lambda x1, y1, x2, y2: p.drawLine(QPointF(x1, y1), QPointF(x2, y2))  # noqa: E731
    if name in ("import", "export"):
        line(8, 2.5, 8, 10)
        if name == "import":
            line(5, 5.5, 8, 2.5); line(8, 2.5, 11, 5.5)
        else:
            line(5, 7, 8, 10); line(8, 10, 11, 7)
        path = QPainterPath(QPointF(2.5, 9.5)); path.lineTo(2.5, 13.5); path.lineTo(13.5, 13.5); path.lineTo(13.5, 9.5)
        p.drawPath(path)
    elif name == "folder":
        path = QPainterPath(QPointF(2, 4)); path.lineTo(6, 4); path.lineTo(7.5, 5.5); path.lineTo(14, 5.5)
        path.lineTo(14, 13); path.lineTo(2, 13); path.closeSubpath(); p.drawPath(path)
    elif name == "copy":
        p.drawRoundedRect(QRectF(5, 5, 8.5, 8.5), 1.5, 1.5)
        path = QPainterPath(QPointF(3, 10.5)); path.lineTo(2.5, 10.5); path.lineTo(2.5, 2.5); path.lineTo(10.5, 2.5)
        path.lineTo(10.5, 3); p.drawPath(path)
    elif name == "plus":
        line(8, 3, 8, 13); line(3, 8, 13, 8)
    elif name == "arrow":
        line(3, 8, 13, 8); line(9, 4, 13, 8); line(13, 8, 9, 12)
    elif name == "check":
        line(3.5, 8.5, 6.5, 11.5); line(6.5, 11.5, 12.5, 4.5)
    elif name == "shield":
        path = QPainterPath(QPointF(8, 2)); path.lineTo(13, 4); path.lineTo(13, 8)
        path.cubicTo(13, 11, 10.5, 13, 8, 14); path.cubicTo(5.5, 13, 3, 11, 3, 8)
        path.lineTo(3, 4); path.closeSubpath(); p.drawPath(path)
        line(5.8, 8.2, 7.4, 9.8); line(7.4, 9.8, 10.4, 6.4)
    elif name == "chip":  # processador (privacidade: "processado no seu computador")
        p.drawRoundedRect(QRectF(4, 4, 8, 8), 1.2, 1.2)
        for k in (6, 8, 10):
            line(k, 2, k, 4); line(k, 12, k, 14); line(2, k, 4, k); line(12, k, 14, k)
    elif name == "clock":
        p.drawEllipse(QPointF(8, 8), 5.5, 5.5); line(8, 5, 8, 8); line(8, 8, 10.5, 9.5)
    elif name == "wifi-off":
        p.drawArc(QRectF(2, 4, 12, 12), 30 * 16, 120 * 16); p.drawArc(QRectF(4.5, 6.5, 7, 7), 30 * 16, 120 * 16)
        p.drawEllipse(QPointF(8, 12), 0.6, 0.6); line(3, 3, 13, 13)
    elif name == "refresh":
        p.drawArc(QRectF(3, 3, 10, 10), 60 * 16, 280 * 16); line(11.5, 2.5, 11.5, 5.2); line(11.5, 5.2, 8.8, 5.2)
    elif name == "photo":
        p.drawRoundedRect(QRectF(2, 3, 12, 10), 1.5, 1.5)
        path = QPainterPath(QPointF(2.5, 11.5)); path.lineTo(6, 8); path.lineTo(8.5, 10.5); path.lineTo(10.5, 8.5)
        path.lineTo(13.5, 11.5); p.drawPath(path)
        p.drawEllipse(QPointF(10.5, 6), 1.1, 1.1)
    elif name == "brush":         # pincel (editar área)
        line(13.5, 2.5, 7.5, 8.5)
        path = QPainterPath(QPointF(7.5, 8.5)); path.cubicTo(5, 8, 3.5, 10, 3.5, 12.5); path.lineTo(2.5, 13.5)
        path.cubicTo(5.5, 13.5, 8, 12, 7.5, 8.5); p.drawPath(path)
    elif name == "eraser":
        path = QPainterPath(QPointF(6, 13.5)); path.lineTo(2.5, 10); path.lineTo(9.5, 3); path.lineTo(13.5, 7)
        path.lineTo(7, 13.5); path.closeSubpath(); p.drawPath(path)
        line(5, 7.5, 9, 11.5); line(6, 13.5, 13.5, 13.5)
    elif name == "undo":
        line(3, 6, 6, 3); line(3, 6, 6, 9)
        path = QPainterPath(QPointF(3, 6)); path.lineTo(9.5, 6); path.cubicTo(12, 6, 13.5, 7.8, 13.5, 9.8)
        path.cubicTo(13.5, 11.8, 12, 13.5, 9.5, 13.5); path.lineTo(6, 13.5); p.drawPath(path)
    elif name == "trash":
        line(2.5, 4.5, 13.5, 4.5); line(6, 4.5, 6.5, 2.5); line(6.5, 2.5, 9.5, 2.5); line(9.5, 2.5, 10, 4.5)
        path = QPainterPath(QPointF(4, 4.5)); path.lineTo(4.8, 13.5); path.lineTo(11.2, 13.5); path.lineTo(12, 4.5)
        p.drawPath(path)
    elif name == "video":
        p.drawRoundedRect(QRectF(2, 4, 9, 8), 1.5, 1.5)
        path = QPainterPath(QPointF(11, 7)); path.lineTo(14, 5); path.lineTo(14, 11); path.lineTo(11, 9)
        p.drawPath(path)
    elif name == "expand":        # ampliar 4K
        line(2.5, 2.5, 6.5, 2.5); line(2.5, 2.5, 2.5, 6.5); line(13.5, 13.5, 9.5, 13.5); line(13.5, 13.5, 13.5, 9.5)
        line(2.5, 2.5, 6.5, 6.5); line(13.5, 13.5, 9.5, 9.5)
        line(13.5, 2.5, 9.5, 2.5); line(13.5, 2.5, 13.5, 6.5); line(2.5, 13.5, 6.5, 13.5); line(2.5, 13.5, 2.5, 9.5)
    elif name == "save":
        path = QPainterPath(QPointF(2.5, 2.5)); path.lineTo(11, 2.5); path.lineTo(13.5, 5); path.lineTo(13.5, 13.5)
        path.lineTo(2.5, 13.5); path.closeSubpath(); p.drawPath(path)
        p.drawRect(QRectF(5, 2.5, 5, 3.5)); p.drawRect(QRectF(5, 9, 6, 4.5))
    elif name == "sparkle":       # gerar com IA
        path = QPainterPath(QPointF(7, 2)); path.quadTo(7.6, 6.4, 12, 7); path.quadTo(7.6, 7.6, 7, 12)
        path.quadTo(6.4, 7.6, 2, 7); path.quadTo(6.4, 6.4, 7, 2); p.drawPath(path)
        line(12.5, 10.5, 12.5, 14); line(10.75, 12.25, 14.25, 12.25)
    elif name == "back":
        line(13, 8, 3, 8); line(7, 4, 3, 8); line(3, 8, 7, 12)
    elif name == "settings":      # avançado
        p.drawEllipse(QPointF(8, 8), 2.2, 2.2)
        for a in range(0, 360, 45):
            from math import cos, radians, sin
            c, s = cos(radians(a)), sin(radians(a))
            line(8 + 4.2 * c, 8 + 4.2 * s, 8 + 6 * c, 8 + 6 * s)
        p.drawEllipse(QPointF(8, 8), 4.2, 4.2)
    p.end()
    icon = QIcon(pm)
    return icon


def status_pixmap(state: str, size: int = 18) -> QPixmap:
    """Marcadores da linha do tempo de etapas: pending / current / done."""
    pm, p = _canvas(size)
    c = QPointF(size / 2, size / 2)
    if state == "done":
        p.setPen(Qt.NoPen)
        bg = QColor(OK); bg.setAlphaF(0.16)
        p.setBrush(bg); p.drawEllipse(c, size / 2 - 1, size / 2 - 1)
        p.setPen(_pen(OK, 1.6, Qt.RoundCap))
        k = size / 18.0
        p.drawLine(QPointF(5.5 * k, 9.3 * k), QPointF(8 * k, 11.8 * k))
        p.drawLine(QPointF(8 * k, 11.8 * k), QPointF(12.6 * k, 6.6 * k))
    elif state == "current":
        halo = QColor(ACCENT); halo.setAlphaF(0.22)
        p.setPen(Qt.NoPen); p.setBrush(halo); p.drawEllipse(c, size / 2 - 1, size / 2 - 1)
        p.setBrush(QColor(ACCENT_HI)); p.drawEllipse(c, size * 0.2, size * 0.2)
    else:
        p.setPen(_pen(PENDING_MARK, 1.4)); p.setBrush(QColor(SURFACE))
        p.drawEllipse(c, size * 0.24, size * 0.24)
    p.end()
    return pm


def empty_state_art(width: int = 168, height: int = 120) -> QPixmap:
    """Ilustração do estado vazio: um cômodo vazio em perspectiva, à espera da foto."""
    pm = QPixmap(int(width * 2), int(height * 2)); pm.setDevicePixelRatio(2.0); pm.fill(Qt.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    frame = QRectF(10.5, 8.5, width - 21, height - 21)
    p.setPen(_pen(HAIRLINE_STRONG, 1.0)); p.setBrush(QColor(SURFACE))
    p.drawRoundedRect(frame, 10, 10)
    # parede do fundo e arestas do cômodo
    back = QRectF(frame.left() + frame.width() * 0.30, frame.top() + frame.height() * 0.24,
                  frame.width() * 0.40, frame.height() * 0.42)
    brass = QColor(ACCENT); brass.setAlphaF(0.85)
    soft = QColor(ACCENT); soft.setAlphaF(0.45)
    p.setPen(_pen(soft, 1.0))
    inset = frame.adjusted(1, 1, -1, -1)
    for a, b in ((inset.topLeft(), back.topLeft()), (inset.topRight(), back.topRight()),
                 (inset.bottomLeft(), back.bottomLeft()), (inset.bottomRight(), back.bottomRight())):
        p.drawLine(a, b)
    p.setPen(_pen(brass, 1.2)); p.setBrush(Qt.NoBrush)
    p.drawRect(back)
    # janela na parede do fundo
    win = QRectF(back.left() + back.width() * 0.60, back.top() + back.height() * 0.18,
                 back.width() * 0.26, back.height() * 0.42)
    p.setPen(_pen(soft, 1.0)); p.drawRect(win); p.drawLine(QPointF(win.center().x(), win.top()), QPointF(win.center().x(), win.bottom()))
    # selo "+" (adicionar foto)
    badge_c = QPointF(frame.right() - 4, frame.bottom() - 4)
    p.setPen(Qt.NoPen); p.setBrush(QColor(BG)); p.drawEllipse(badge_c, 15, 15)
    p.setBrush(QColor(ACCENT)); p.drawEllipse(badge_c, 12, 12)
    p.setPen(_pen(ON_ACCENT, 1.8, Qt.RoundCap))
    p.drawLine(QPointF(badge_c.x() - 5, badge_c.y()), QPointF(badge_c.x() + 5, badge_c.y()))
    p.drawLine(QPointF(badge_c.x(), badge_c.y() - 5), QPointF(badge_c.x(), badge_c.y() + 5))
    p.end()
    return pm


# -------------------------------------------------------------------------------------------- marca

def app_icon(size: int = 256, dpr: float = 1.0) -> QPixmap:
    """Ícone 7xAi (placa grafite com "7xAi"; "7x" abaixo de 32 px). Ver ``brand``."""
    return brand.app_icon_pixmap(size, dpr)


def dark_title_bar(widget) -> None:
    """Barra de título escura no Windows 10 20H1+/11 (DWMWA_USE_IMMERSIVE_DARK_MODE). Nada fora do Windows."""
    import sys

    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = ctypes.c_void_p(int(widget.winId()))
        on = ctypes.c_int(1)
        for attr in (20, 19):  # 20 = Windows 11 / 10 20H1+; 19 = builds antigas do Windows 10
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(on), ctypes.sizeof(on)) == 0:
                break
    except Exception:  # noqa: BLE001 — cosmético
        pass


def window_icon() -> QIcon:
    icon = QIcon()
    for s in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        icon.addPixmap(brand.app_icon_pixmap(s))
    return icon


def render_res(out: Path = _RES) -> None:
    """Regera as imagens da folha de estilo (seta do combo e visto do checkbox), em 2x."""
    out.mkdir(parents=True, exist_ok=True)
    img = QImage(24, 16, QImage.Format_ARGB32_Premultiplied); img.fill(Qt.transparent)
    p = QPainter(img); p.setRenderHint(QPainter.Antialiasing)
    p.setPen(_pen(TEXT_2, 2.4, Qt.RoundCap))
    p.drawLine(QPointF(5, 5), QPointF(12, 11.5)); p.drawLine(QPointF(12, 11.5), QPointF(19, 5))
    p.end(); img.save(str(out / "chevron.png"))
    img = QImage(32, 32, QImage.Format_ARGB32_Premultiplied); img.fill(Qt.transparent)
    p = QPainter(img); p.setRenderHint(QPainter.Antialiasing)
    p.setPen(_pen(ON_ACCENT, 3.2, Qt.RoundCap))
    p.drawLine(QPointF(8, 16.5), QPointF(13.5, 22)); p.drawLine(QPointF(13.5, 22), QPointF(24, 10))
    p.end(); img.save(str(out / "check.png"))
