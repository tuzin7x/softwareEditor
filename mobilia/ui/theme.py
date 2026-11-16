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
QProgressBar {{ background: #2A2A30; border: none; border-radius: 2px; min-height: 4px; max-height: 4px; }}
QProgressBar::chunk {{ border-radius: 2px;
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
    elif name == "photo":
        p.drawRoundedRect(QRectF(2, 3, 12, 10), 1.5, 1.5)
        path = QPainterPath(QPointF(2.5, 11.5)); path.lineTo(6, 8); path.lineTo(8.5, 10.5); path.lineTo(10.5, 8.5)
        path.lineTo(13.5, 11.5); p.drawPath(path)
        p.drawEllipse(QPointF(10.5, 6), 1.1, 1.1)
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
        p.setPen(_pen(HAIRLINE_STRONG, 1.4)); p.setBrush(QColor(SURFACE))
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
