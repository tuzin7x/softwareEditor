"""Peças da tela de redesenho por IA: cartões de função e de estilo, chips, seletor segmentado, comparador antes/depois
(arrastar), editor de máscara (pincel/borracha), grade de versões em geração e faixa de miniaturas.

Nada aqui importa numpy/PIL: o executável da interface leva só o PySide6 (o motor roda em outro processo)."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QKeySequence, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QGridLayout, QHBoxLayout, QSizePolicy, QToolButton,
                               QWidget)

from . import theme
from .widgets import paint_chip

# ------------------------------------------------------------------------------------------- textos

FUNCTION_INFO: dict[str, tuple[str, str]] = {
    # chave: (título, descrição curta) — títulos curtos para caber no cartão
    "mudar_estilo": ("Mudar estilo", "Mesmo layout, novos materiais, cores e peças"),
    "reforma": ("Reforma completa", "Móveis e layout novos, a arquitetura fica"),
    "redesign_simples": ("Redesign simples", "Um refresh leve, quase o mesmo ambiente"),
    "mobiliar": ("Mobiliar vazio", "Cômodo vazio vira ambiente decorado"),
    "editar_area": ("Editar área", "Pinte uma parte da foto e troque só ela"),
    "fachada": ("Fachada", "Nova cara para a frente da casa"),
    "evento": ("Decoração de evento", "Aniversário, casamento, Natal e mais"),
    "texto": ("Texto para ambiente", "Descreva e crie do zero, sem foto"),
}
FUNCTION_ORDER = ("mudar_estilo", "reforma", "redesign_simples", "mobiliar", "editar_area", "fachada", "evento",
                  "texto")

# paleta de cada estilo (só para o cartão: quatro amostras de material/cor)
STYLE_SWATCHES: dict[str, tuple[str, str, str, str]] = {
    "moderno": ("#E9E4DC", "#8C8780", "#2B2B2B", "#B08A5B"),
    "contemporaneo": ("#ECE8E1", "#B9B1A6", "#5E6B73", "#C9A27A"),
    "escandinavo": ("#F3F0EA", "#D9CFC1", "#B7A68E", "#8FA39A"),
    "minimalista": ("#F6F5F2", "#E2DFD9", "#BDB8B0", "#3A3A3A"),
    "industrial": ("#3B3B3B", "#9C5A3C", "#7A6A5B", "#9EA3A6"),
    "classico": ("#EFE6D8", "#C8B48F", "#7A5C3E", "#2F3E4E"),
    "luxo": ("#1F2B2E", "#C9A45C", "#E8E2D6", "#5B2333"),
    "boho": ("#E8D9C4", "#C67B4E", "#8E9B6B", "#D9A441"),
    "japandi": ("#EDE6DA", "#C9B79C", "#7D6A55", "#2E2B28"),
    "rustico": ("#D8C3A5", "#9C6B43", "#5A4632", "#8A8F7A"),
    "costeiro": ("#F4F1EA", "#A9C6D3", "#3E6E8E", "#D8C8A8"),
    "tropical": ("#F0E9DC", "#2F6B4F", "#C46A3F", "#D9B26F"),
    "mid_century": ("#E9DFCB", "#B5652E", "#2F5D62", "#D4A24C"),
    "farmhouse": ("#F2EFE8", "#CFC6B6", "#7A7F75", "#3B3833"),
}
STYLE_HINTS: dict[str, str] = {
    "moderno": "Linhas retas, madeira clara, preto fosco e tecidos neutros",
    "contemporaneo": "Atual e sofisticado: curvas suaves, tons neutros e metal",
    "escandinavo": "Claro e aconchegante: madeira clara, lã, linho e plantas",
    "minimalista": "Pouca coisa, tudo no lugar: branco, cinza e formas simples",
    "industrial": "Tijolo, concreto, couro e metal escuro",
    "classico": "Molduras, simetria, tecidos nobres e madeira escura",
    "luxo": "Veludo, mármore, latão e cores profundas",
    "boho": "Fibras naturais, terracota, estampas e muitas plantas",
    "japandi": "Japonês + escandinavo: calma, madeira, linho e tons de terra",
    "rustico": "Madeira de demolição, pedra e texturas naturais",
    "costeiro": "Casa de praia: branco, azul, linho e fibras",
    "tropical": "Brasilidade: folhagens, palhinha, madeira e cores quentes",
    "mid_century": "Anos 50/60: pés palito, nogueira, mostarda e petróleo",
    "farmhouse": "Fazenda moderna: branco quente, madeira e ferro",
}


def _font(px: int, weight=QFont.Normal, family: str | None = None) -> QFont:
    f = QFont(family or "Inter")
    f.setPixelSize(px); f.setWeight(weight)
    return f


def fit_rect(src: QSize, box: QRectF) -> QRectF:
    """Retângulo com a proporção de ``src`` centrado e contido em ``box``."""
    if src.width() <= 0 or src.height() <= 0 or box.width() <= 0 or box.height() <= 0:
        return QRectF()
    k = min(box.width() / src.width(), box.height() / src.height())
    w, h = src.width() * k, src.height() * k
    return QRectF(box.left() + (box.width() - w) / 2, box.top() + (box.height() - h) / 2, w, h)


def cover_pixmap(pm: QPixmap, w: int, h: int, radius: float = 8.0, dpr: float = 1.0) -> QPixmap:
    """Recorta ``pm`` preenchendo ``w x h`` (como object-fit: cover), com cantos arredondados."""
    W, H = max(1, int(w * dpr)), max(1, int(h * dpr))
    out = QPixmap(W, H); out.fill(Qt.transparent)
    if pm is None or pm.isNull():
        return out
    sc = pm.scaled(W, H, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    p = QPainter(out); p.setRenderHint(QPainter.Antialiasing)
    clip = QPainterPath(); clip.addRoundedRect(QRectF(0, 0, W, H), radius * dpr, radius * dpr)
    p.setClipPath(clip)
    p.drawPixmap(int((W - sc.width()) / 2), int((H - sc.height()) / 2), sc)
    p.end()
    out.setDevicePixelRatio(dpr)
    return out


# ---------------------------------------------------------------------------------------- botões

class CardButton(QAbstractButton):
    """Cartão clicável e marcável, pintado à mão (fundo, filete, latão quando marcado)."""

    RADIUS = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self._hover = False

    def enterEvent(self, e):  # noqa: N802
        self._hover = True; self.update(); super().enterEvent(e)

    def leaveEvent(self, e):  # noqa: N802
        self._hover = False; self.update(); super().leaveEvent(e)

    def _paint_frame(self, p: QPainter) -> QRectF:
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        on = self.isChecked()
        bg = QColor(theme.SELECTED_BG if on else ("#202025" if self._hover else theme.TILE))
        border = QColor(theme.ACCENT if on else (theme.HAIRLINE_STRONG if self._hover else theme.HAIRLINE))
        if not self.isEnabled():
            bg, border = QColor(theme.SURFACE), QColor(theme.HAIRLINE)
        p.setPen(QPen(border, 1.0)); p.setBrush(bg)
        p.drawRoundedRect(r, self.RADIUS, self.RADIUS)
        if self.hasFocus() and not on:
            c = QColor(theme.ACCENT); c.setAlphaF(0.5)
            p.setPen(QPen(c, 1.0)); p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(1.5, 1.5, -1.5, -1.5), self.RADIUS - 1.5, self.RADIUS - 1.5)
        return r


class FunctionCard(CardButton):
    """Cartão de função: ícone de linha, título e uma linha de descrição."""

    def __init__(self, key: str, title: str, desc: str, parent=None):
        super().__init__(parent)
        self.key, self.title, self.desc = key, title, desc
        self.setAccessibleName(title)
        self.setToolTip(f"{title}: {desc}")
        self._pix = {s: theme.function_pixmap(key, 30, c) for s, c in
                     (("off", theme.TEXT_2), ("on", theme.ACCENT_HI), ("dis", theme.DISABLED))}
        self.setMinimumSize(150, 108)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(108)

    def set_compact(self, compact: bool) -> None:
        self.setFixedHeight(92 if compact else 108)

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = self._paint_frame(p)
        pad = 14
        state = "dis" if not self.isEnabled() else ("on" if self.isChecked() else "off")
        p.drawPixmap(QPointF(r.left() + pad, r.top() + pad - 2), self._pix[state])
        p.setFont(_font(13, QFont.DemiBold))
        p.setPen(QColor(theme.TEXT if self.isEnabled() else theme.DISABLED))
        title_y = r.top() + pad + 34
        p.drawText(QRectF(r.left() + pad, title_y, r.width() - 2 * pad, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   self.title)
        p.setFont(_font(11))
        p.setPen(QColor(theme.MUTED if self.isEnabled() else theme.DISABLED))
        p.drawText(QRectF(r.left() + pad, title_y + 19, r.width() - 2 * pad, r.bottom() - title_y - 22),
                   Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, self.desc)
        p.end()


class StyleCard(CardButton):
    """Cartão de estilo: quadro de amostras (cores/materiais do estilo) e o nome."""

    def __init__(self, key: str, label: str, parent=None):
        super().__init__(parent)
        self.key, self.label = key, label
        self.swatches = STYLE_SWATCHES.get(key, ("#DDD", "#AAA", "#777", "#444"))
        self.setAccessibleName(label)
        self.setToolTip(f"{label}: {STYLE_HINTS.get(key, '')}")
        self.setMinimumSize(96, 84)
        self.setFixedHeight(84)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_compact(self, compact: bool) -> None:
        self.setFixedHeight(74 if compact else 84)

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = self._paint_frame(p)
        sw = QRectF(r.left() + 6, r.top() + 6, r.width() - 12, r.height() - 34)
        clip = QPainterPath(); clip.addRoundedRect(sw, 7, 7)
        p.save(); p.setClipPath(clip)
        # quadro de materiais: uma faixa grande (base) + três amostras
        a, b, c, d = (QColor(x) for x in self.swatches)
        p.fillRect(sw, a)
        w3 = sw.width() * 0.46
        p.fillRect(QRectF(sw.right() - w3, sw.top(), w3, sw.height() * 0.58), b)
        p.fillRect(QRectF(sw.right() - w3, sw.top() + sw.height() * 0.58, w3 * 0.5, sw.height() * 0.42), c)
        p.fillRect(QRectF(sw.right() - w3 * 0.5, sw.top() + sw.height() * 0.58, w3 * 0.5, sw.height() * 0.42), d)
        g = QLinearGradient(sw.topLeft(), sw.bottomLeft())
        g.setColorAt(0, QColor(255, 255, 255, 18)); g.setColorAt(1, QColor(0, 0, 0, 40))
        p.fillRect(sw, g)
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 20), 1)); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(sw, 7, 7)
        p.setFont(_font(12, QFont.DemiBold if self.isChecked() else QFont.Medium))
        p.setPen(QColor(theme.TEXT if self.isChecked() else theme.TEXT_2))
        fm = QFontMetrics(p.font())
        name = fm.elidedText(self.label, Qt.ElideRight, int(r.width() - 16))
        p.drawText(QRectF(r.left() + 8, sw.bottom() + 4, r.width() - 16, r.bottom() - sw.bottom() - 6),
                   Qt.AlignLeft | Qt.AlignVCenter, name)
        if self.isChecked():          # selo de marcado no canto do quadro
            cc = QPointF(sw.right() - 11, sw.top() + 11)
            p.setPen(Qt.NoPen); p.setBrush(QColor(theme.ACCENT)); p.drawEllipse(cc, 8, 8)
            p.setPen(QPen(QColor(theme.ON_ACCENT), 1.6, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(cc.x() - 3.5, cc.y() + 0.2), QPointF(cc.x() - 1, cc.y() + 2.8))
            p.drawLine(QPointF(cc.x() - 1, cc.y() + 2.8), QPointF(cc.x() + 3.8, cc.y() - 2.6))
        p.end()


def chip(text: str, icon=None, icon_size: int = 22, height: int = 38) -> QToolButton:
    b = QToolButton(); b.setObjectName("chip"); b.setCheckable(True); b.setText(text)
    b.setCursor(Qt.PointingHandCursor)
    b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); b.setFixedHeight(height)
    if icon is not None:
        b.setIcon(icon); b.setIconSize(QSize(icon_size, icon_size))
        b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
    else:
        b.setToolButtonStyle(Qt.ToolButtonTextOnly)
    return b


class Segmented(QWidget):
    """Seletor segmentado (ex.: número de versões 1–4). ``changed`` emite o valor escolhido."""

    changed = Signal(object)

    def __init__(self, options: list[tuple[object, str]], value=None, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        self.group = QButtonGroup(self); self.group.setExclusive(True)
        self.buttons: dict[object, QToolButton] = {}
        for i, (val, label) in enumerate(options):
            b = QToolButton(); b.setObjectName("seg"); b.setCheckable(True); b.setText(label)
            b.setCursor(Qt.PointingHandCursor)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setProperty("pos", "first" if i == 0 else "last" if i == len(options) - 1 else "mid")
            b.clicked.connect(lambda *_a, v=val: self.set_value(v, emit=True))
            self.group.addButton(b); self.buttons[val] = b
            lay.addWidget(b)
        self._value = None
        self.set_value(value if value is not None else options[0][0])

    def value(self):
        return self._value

    def set_value(self, value, emit: bool = False) -> None:
        if value not in self.buttons:
            return
        self._value = value
        self.buttons[value].setChecked(True)
        if emit:
            self.changed.emit(value)


# ---------------------------------------------------------------------------------- antes / depois

class BeforeAfterView(QWidget):
    """Comparador antes/depois: a versão nova inteira e a foto original à esquerda do divisor; arrastar move o
    divisor. Sem foto "antes" (função texto), mostra só a imagem."""

    RADIUS = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.before: QPixmap | None = None
        self.after: QPixmap | None = None
        self.split = 0.5
        self._cache: dict[str, QPixmap] = {}
        self._cache_key = None
        self.after_label = "DEPOIS"
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.SizeHorCursor)
        self.setMouseTracking(False)

    def set_images(self, before: str | None, after: str | None) -> None:
        self.before = QPixmap(before) if before else None
        if self.before is not None and self.before.isNull():
            self.before = None
        self.after = QPixmap(after) if after else None
        if self.after is not None and self.after.isNull():
            self.after = None
        self._cache_key = None
        self.setCursor(Qt.SizeHorCursor if self.has_before() else Qt.ArrowCursor)
        self.update()

    def has_before(self) -> bool:
        return self.before is not None and self.after is not None

    def set_split(self, t: float) -> None:
        self.split = max(0.0, min(1.0, float(t)))
        self.update()

    def image_rect(self) -> QRectF:
        ref = self.after or self.before
        if ref is None:
            return QRectF()
        return fit_rect(ref.size(), QRectF(self.rect()).adjusted(4, 4, -4, -4))

    def _scaled(self, r: QRectF) -> dict[str, QPixmap]:
        dpr = self.devicePixelRatioF()
        key = (int(r.width()), int(r.height()), dpr, id(self.before), id(self.after))
        if key != self._cache_key:
            self._cache = {}
            for name, pm in (("before", self.before), ("after", self.after)):
                if pm is not None:
                    s = pm.scaled(int(r.width() * dpr), int(r.height() * dpr), Qt.IgnoreAspectRatio,
                                  Qt.SmoothTransformation)
                    s.setDevicePixelRatio(dpr)
                    self._cache[name] = s
            self._cache_key = key
        return self._cache

    def _x_to_split(self, x: float) -> None:
        r = self.image_rect()
        if r.width() > 0:
            self.set_split((x - r.left()) / r.width())

    def mousePressEvent(self, e):  # noqa: N802
        if self.has_before() and e.button() == Qt.LeftButton:
            self._x_to_split(e.position().x())

    def mouseMoveEvent(self, e):  # noqa: N802
        if self.has_before() and e.buttons() & Qt.LeftButton:
            self._x_to_split(e.position().x())

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = self.image_rect()
        if r.isEmpty():
            p.end(); return
        sc = self._scaled(r)
        clip = QPainterPath(); clip.addRoundedRect(r, self.RADIUS, self.RADIUS)
        p.save(); p.setClipPath(clip)
        if "after" in sc:
            p.drawPixmap(r.topLeft(), sc["after"])
        x = r.left() + r.width() * self.split
        if self.has_before() and "before" in sc:
            p.setClipRect(QRectF(r.left(), r.top(), x - r.left(), r.height()), Qt.IntersectClip)
            p.drawPixmap(r.topLeft(), sc["before"])
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 22), 1)); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), self.RADIUS, self.RADIUS)
        if self.has_before():
            # divisor + alça
            p.setPen(QPen(QColor(theme.TEXT), 2)); p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            c = QPointF(x, r.center().y())
            p.setPen(QPen(QColor(255, 255, 255, 60), 1)); p.setBrush(QColor(10, 10, 12, 200)); p.drawEllipse(c, 19, 19)
            p.setPen(QPen(QColor(theme.ACCENT_HI), 1.8, Qt.SolidLine, Qt.RoundCap))
            for sgn in (-1, 1):
                p.drawLine(QPointF(c.x() + sgn * 4, c.y() - 5), QPointF(c.x() + sgn * 9, c.y()))
                p.drawLine(QPointF(c.x() + sgn * 9, c.y()), QPointF(c.x() + sgn * 4, c.y() + 5))
            if self.split > 0.12:
                paint_chip(p, r.left() + 14, r.top() + 14, "ANTES", dot=theme.MUTED)
            if self.split < 0.88:
                paint_chip(p, r.right() - 14, r.top() + 14, self.after_label, align_right=True)
        else:
            paint_chip(p, r.left() + 14, r.top() + 14, self.after_label)
        p.end()


# ---------------------------------------------------------------------------------- máscara

class MaskEditor(QWidget):
    """Pintar a área a editar sobre a imagem: pincel e borracha, tamanho do pincel, desfazer e limpar.

    A máscara tem o tamanho da imagem (branco = editar) e é salva em PNG para ``design --mask``. O pincel é medido
    em pixels da tela (o círculo do cursor corresponde ao que será pintado)."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.base: QPixmap | None = None
        self.mask: QImage | None = None
        self.overlay: QImage | None = None
        self.brush_px = 48
        self.erasing = False
        self._undo: list[tuple[QImage, QImage]] = []
        self._last: QPointF | None = None
        self._cursor: QPointF | None = None
        self._cache_key = None
        self._scaled: QPixmap | None = None
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        QShortcut(QKeySequence.Undo, self, activated=self.undo)

    # --- estado
    def set_image(self, path: str) -> bool:
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self.base = pm
        self.mask = QImage(pm.size(), QImage.Format_Grayscale8); self.mask.fill(0)
        self.overlay = QImage(pm.size(), QImage.Format_ARGB32_Premultiplied); self.overlay.fill(Qt.transparent)
        self._undo.clear(); self._cache_key = None
        self.update(); self.changed.emit()
        return True

    def set_brush(self, px: int) -> None:
        self.brush_px = max(4, min(400, int(px)))
        self.update()

    def set_erasing(self, erasing: bool) -> None:
        self.erasing = bool(erasing)

    def clear(self) -> None:
        if self.mask is None:
            return
        self._push_undo()
        self.mask.fill(0); self.overlay.fill(Qt.transparent)
        self.update(); self.changed.emit()

    def undo(self) -> None:
        if not self._undo:
            return
        self.mask, self.overlay = self._undo.pop()
        self.update(); self.changed.emit()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def coverage(self) -> float:
        """Fração aproximada da imagem pintada (0..1), por uma redução 64x64 da máscara."""
        if self.mask is None:
            return 0.0
        small = self.mask.scaled(64, 64, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        small = small.convertToFormat(QImage.Format_Grayscale8)
        bpl = small.bytesPerLine()
        data = bytes(small.constBits())[: bpl * 64]
        on = sum(1 for y in range(64) for v in data[y * bpl: y * bpl + 64] if v > 8)
        return on / 4096.0

    def has_mask(self) -> bool:
        return self.coverage() > 0.0

    def save_mask(self, path: str) -> bool:
        return self.mask is not None and self.mask.save(str(path), "PNG")

    # --- geometria
    def image_rect(self) -> QRectF:
        if self.base is None:
            return QRectF()
        return fit_rect(self.base.size(), QRectF(self.rect()).adjusted(4, 4, -4, -4))

    def scale(self) -> float:
        r = self.image_rect()
        return r.width() / self.base.width() if self.base is not None and r.width() > 0 else 1.0

    def to_image(self, pt: QPointF) -> QPointF:
        r = self.image_rect()
        k = self.scale()
        return QPointF((pt.x() - r.left()) / k, (pt.y() - r.top()) / k)

    # --- pintura
    def _push_undo(self) -> None:
        if self.mask is not None:
            self._undo.append((self.mask.copy(), self.overlay.copy()))
            del self._undo[:-20]

    def stroke(self, a: QPointF, b: QPointF) -> None:
        """Traço de ``a`` a ``b`` em coordenadas do widget (usado pelo mouse e pelos testes)."""
        if self.mask is None:
            return
        w = self.brush_px / self.scale()
        ia, ib = self.to_image(a), self.to_image(b)
        for img, color in ((self.mask, QColor(255, 255, 255)), (self.overlay, QColor(theme.ACCENT))):
            p = QPainter(img); p.setRenderHint(QPainter.Antialiasing)
            if self.erasing:
                if img is self.mask:
                    color = QColor(0, 0, 0)
                else:
                    p.setCompositionMode(QPainter.CompositionMode_Clear)
            pen = QPen(color, w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            if ia == ib:
                p.setPen(Qt.NoPen); p.setBrush(color); p.drawEllipse(ia, w / 2, w / 2)
            else:
                p.drawLine(ia, ib)
            p.end()
        self.update()

    def mousePressEvent(self, e):  # noqa: N802
        if e.button() != Qt.LeftButton or self.mask is None:
            return
        self._push_undo()
        self._last = e.position()
        self.stroke(self._last, self._last)

    def mouseMoveEvent(self, e):  # noqa: N802
        self._cursor = e.position()
        if self._last is not None and e.buttons() & Qt.LeftButton:
            self.stroke(self._last, e.position())
            self._last = e.position()
        self.update()

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self._last is not None:
            self._last = None
            self.changed.emit()

    def leaveEvent(self, e):  # noqa: N802
        self._cursor = None; self.update(); super().leaveEvent(e)

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = self.image_rect()
        if self.base is None or r.isEmpty():
            p.end(); return
        dpr = self.devicePixelRatioF()
        key = (int(r.width()), int(r.height()), dpr, id(self.base))
        if key != self._cache_key:
            self._scaled = self.base.scaled(int(r.width() * dpr), int(r.height() * dpr), Qt.IgnoreAspectRatio,
                                            Qt.SmoothTransformation)
            self._scaled.setDevicePixelRatio(dpr)
            self._cache_key = key
        clip = QPainterPath(); clip.addRoundedRect(r, 12, 12)
        p.save(); p.setClipPath(clip)
        p.drawPixmap(r.topLeft(), self._scaled)
        if self.overlay is not None:
            p.setOpacity(0.55)
            p.drawImage(r, self.overlay)
            p.setOpacity(1.0)
        p.restore()
        p.setPen(QPen(QColor(255, 255, 255, 22), 1)); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
        paint_chip(p, r.left() + 14, r.top() + 14, "BORRACHA" if self.erasing else "PINCEL",
                   dot=theme.MUTED if self.erasing else theme.ACCENT)
        if self._cursor is not None and r.contains(self._cursor):
            rad = self.brush_px / 2
            p.setPen(QPen(QColor(0, 0, 0, 140), 3)); p.drawEllipse(self._cursor, rad, rad)
            p.setPen(QPen(QColor(theme.TEXT), 1.2)); p.drawEllipse(self._cursor, rad, rad)
        p.end()


# ---------------------------------------------------------------------------------- versões

class VariantSlot(QWidget):
    """Uma versão na grade de geração: espaço reservado ("Gerando a versão 2…") até a imagem chegar."""

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.pm: QPixmap | None = None
        self.active = False
        self.setMinimumSize(120, 90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, path: str | None) -> None:
        pm = QPixmap(path) if path else None
        self.pm = pm if pm is not None and not pm.isNull() else None
        self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self.pm is not None:
            img = cover_pixmap(self.pm, int(r.width()), int(r.height()), 12, self.devicePixelRatioF())
            p.drawPixmap(r.topLeft(), img)
            p.setPen(QPen(QColor(255, 255, 255, 22), 1)); p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r, 12, 12)
            paint_chip(p, r.left() + 12, r.top() + 12, f"VERSÃO {self.index + 1}")
        else:
            p.setPen(QPen(QColor(theme.ACCENT_LINE if self.active else theme.HAIRLINE), 1))
            p.setBrush(QColor(theme.SURFACE)); p.drawRoundedRect(r, 12, 12)
            p.setFont(_font(12, QFont.Medium))
            p.setPen(QColor(theme.TEXT_2 if self.active else theme.MUTED))
            txt = f"Gerando a versão {self.index + 1}…" if self.active else f"Versão {self.index + 1}"
            p.drawText(r, Qt.AlignCenter, txt)
        p.end()


class VariantGrid(QWidget):
    """Grade das N versões durante a geração (1, 1x2 ou 2x2)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QGridLayout(self); self.grid.setContentsMargins(0, 0, 0, 0); self.grid.setSpacing(12)
        self.slots: list[VariantSlot] = []

    def reset(self, n: int) -> None:
        for s in self.slots:
            s.setParent(None); s.deleteLater()
        self.slots = []
        n = max(1, int(n))
        cols = 1 if n == 1 else 2
        for i in range(n):
            s = VariantSlot(i); self.slots.append(s)
            self.grid.addWidget(s, i // cols, i % cols)
        self.set_active(0)

    def set_active(self, index: int) -> None:
        for i, s in enumerate(self.slots):
            s.active = (i == index and s.pm is None)
            s.update()

    def set_image(self, index: int, path: str) -> None:
        if 0 <= index < len(self.slots):
            self.slots[index].set_image(path)
            nxt = next((i for i, s in enumerate(self.slots) if s.pm is None), -1)
            self.set_active(nxt)

    def filled(self) -> int:
        return sum(1 for s in self.slots if s.pm is not None)


class ThumbButton(CardButton):
    """Miniatura de uma versão (faixa abaixo do comparador)."""

    RADIUS = 10

    def __init__(self, path: str, label: str, parent=None):
        super().__init__(parent)
        self.path, self.label = path, label
        self.pm = QPixmap(path)
        self.badge = ""
        self.setToolTip(label)
        self.setAccessibleName(label)
        self.setFixedSize(132, 92)

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        img = cover_pixmap(self.pm, int(r.width()), int(r.height()), self.RADIUS - 1, self.devicePixelRatioF())
        p.drawPixmap(r.topLeft(), img)
        if not self.isChecked():
            p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 70 if not self._hover else 20))
            p.drawRoundedRect(r, self.RADIUS - 1, self.RADIUS - 1)
        p.setPen(QPen(QColor(theme.ACCENT if self.isChecked() else theme.HAIRLINE_STRONG), 2 if self.isChecked() else 1))
        p.setBrush(Qt.NoBrush); p.drawRoundedRect(r, self.RADIUS - 1, self.RADIUS - 1)
        paint_chip(p, r.left() + 6, r.bottom() - 30, self.label, dot=theme.ACCENT if self.isChecked() else None)
        if self.badge:
            paint_chip(p, r.right() - 6, r.top() + 6, self.badge, dot=None, align_right=True)
        p.end()


class ThumbStrip(QWidget):
    """Faixa de miniaturas das versões; ``selected`` emite o índice escolhido."""

    selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QHBoxLayout(self); self.lay.setContentsMargins(0, 0, 0, 0); self.lay.setSpacing(10)
        self.group = QButtonGroup(self); self.group.setExclusive(True)
        self.buttons: list[ThumbButton] = []
        self.setFixedHeight(92)

    def set_items(self, paths: list[str], labels: list[str] | None = None) -> None:
        for b in self.buttons:
            self.group.removeButton(b); b.setParent(None); b.deleteLater()
        self.buttons = []
        while self.lay.count():
            self.lay.takeAt(0)
        self.lay.addStretch(1)
        for i, path in enumerate(paths):
            b = ThumbButton(path, (labels[i] if labels else f"VERSÃO {i + 1}"))
            b.clicked.connect(lambda *_a, k=i: self.select(k, emit=True))
            self.group.addButton(b); self.buttons.append(b)
            self.lay.addWidget(b)
        self.lay.addStretch(1)

    def select(self, index: int, emit: bool = False) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)
            if emit:
                self.selected.emit(index)

    def set_badge(self, index: int, text: str) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].badge = text
            self.buttons[index].update()


def widget_center(w: QWidget) -> QPoint:
    return QPoint(w.width() // 2, w.height() // 2)
