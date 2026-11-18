"""Peças visuais compartilhadas pelas telas do 7xAi (redesenho por IA e o modo legado de vídeo de montagem)."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStackedWidget,
                               QStatusBar, QStyle, QStyleOption, QVBoxLayout, QWidget)

from . import theme

PHOTO_FILTER = "Fotos (*.jpg *.jpeg *.png *.webp *.tif *.tiff *.heic)"


def on_click(button, fn, *args, **kwargs) -> None:
    """Liga ``clicked`` a ``fn(*args, **kwargs)`` sem repassar os argumentos do sinal.

    ``clicked(bool)`` entrega ``checked=False``: ligado direto a um método com parâmetros opcionais, esse ``False``
    vira o primeiro parâmetro (o erro antigo do "modo False"). Aqui os argumentos do sinal são sempre descartados."""
    button.clicked.connect(lambda *_signal_args: fn(*args, **kwargs))

def human_size(nbytes: int) -> str:
    """Tamanho de arquivo legível ("850 KB", "12.4 MB"); vazio quando desconhecido."""
    if nbytes <= 0:
        return ""
    if nbytes < 1_000_000:
        return f"{max(1, round(nbytes / 1000))} KB"
    return f"{nbytes / 1e6:.1f} MB"


def paint_chip(p: QPainter, x: float, y: float, text: str, dot: str | None = theme.ACCENT,
               align_right: bool = False) -> QRectF:
    """Etiqueta translúcida sobre a imagem (ex.: "FOTO ORIGINAL"). Devolve o retângulo desenhado."""
    f = QFont(p.font())
    f.setPixelSize(11); f.setWeight(QFont.DemiBold)
    f.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
    fm = QFontMetrics(f)
    pad, dot_w = 11, (14 if dot else 0)
    w = pad * 2 + dot_w + fm.horizontalAdvance(text)
    h = 26
    r = QRectF(x - w if align_right else x, y, w, h)
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(255, 255, 255, 26), 1))
    p.setBrush(QColor(10, 10, 12, 196))
    p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), h / 2, h / 2)
    if dot:
        p.setPen(Qt.NoPen); p.setBrush(QColor(dot))
        p.drawEllipse(QPointF(r.left() + pad + 3, r.center().y()), 3, 3)
    p.setFont(f); p.setPen(QColor(theme.TEXT))
    p.drawText(r.adjusted(pad + dot_w, 0, -pad + 2, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
    p.restore()
    return r


class ImageView(QLabel):
    """Mostra uma imagem inteira, preservando a proporção, com cantos arredondados e etiqueta opcional."""

    RADIUS = 10

    def __init__(self, text: str = "", caption: str = ""):
        super().__init__(text)
        self._pm: QPixmap | None = None
        self._scaled: QPixmap | None = None
        self.caption = caption
        self.corner_hint = ""
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, path: str) -> bool:
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self._pm = pm
        self._rescale()
        return True

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pm is None:
            return
        dpr = self.devicePixelRatioF()
        target = self._pm.size().scaled(self.size() - QSize(8, 8), Qt.KeepAspectRatio)
        if target.isEmpty():
            return
        img = self._pm.scaled(target * dpr, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        out = QPixmap(img.size()); out.fill(Qt.transparent)
        p = QPainter(out); p.setRenderHint(QPainter.Antialiasing)
        clip = QPainterPath(); clip.addRoundedRect(QRectF(out.rect()), self.RADIUS * dpr, self.RADIUS * dpr)
        p.setClipPath(clip); p.drawPixmap(0, 0, img); p.end()
        out.setDevicePixelRatio(dpr)
        self._scaled = out
        self.update()

    def paintEvent(self, e):  # noqa: N802
        if self._scaled is None:
            return super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        sz = self._scaled.deviceIndependentSize()
        r = QRectF((self.width() - sz.width()) / 2, (self.height() - sz.height()) / 2, sz.width(), sz.height())
        p.drawPixmap(r.topLeft(), self._scaled)
        p.setPen(QPen(QColor(255, 255, 255, 22), 1)); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), self.RADIUS, self.RADIUS)
        if self.caption:
            paint_chip(p, r.left() + 14, r.top() + 14, self.caption)
        if self.corner_hint:
            paint_chip(p, r.right() - 14, r.bottom() - 40, self.corner_hint, dot=None, align_right=True)
        p.end()


class DropZone(QFrame):
    """Área de importação: clique ou arraste uma foto."""

    HEADLINE = "Comece pela foto do ambiente vazio"
    TRUST = "Sua foto fica intacta: só os móveis surgem dentro dela."
    TIP = "Dica: fotografe o cômodo inteiro, na altura do peito, com a luz natural do dia."

    def __init__(self, on_file, headline: str | None = None, trust: str | None = None, tip: str | None = None):
        super().__init__()
        self._texts = (headline or self.HEADLINE, trust or self.TRUST, tip or self.TIP)
        self.setObjectName("drop")
        self.setProperty("filled", False)
        self.on_file = on_file
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        self.view = ImageView(caption="FOTO ORIGINAL")
        self.view.corner_hint = "Clique para trocar a foto"
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_empty())
        self.stack.addWidget(self.view)
        lay.addWidget(self.stack)

    def _build_empty(self) -> QWidget:
        w = QWidget()
        w.setAttribute(Qt.WA_TranslucentBackground)
        col = QVBoxLayout(w)
        col.setContentsMargins(24, 24, 24, 18)
        col.addStretch(3)
        art = QLabel(); art.setPixmap(theme.empty_state_art()); art.setAlignment(Qt.AlignCenter)
        self.hint = QLabel(self._texts[0])
        self.hint.setObjectName("headline"); self.hint.setAlignment(Qt.AlignCenter)
        body = QLabel("Arraste a imagem para cá ou escolha um arquivo.")
        body.setObjectName("body"); body.setAlignment(Qt.AlignCenter)
        btn = QPushButton("  Escolher foto"); btn.setObjectName("outline")
        btn.setIcon(theme.line_icon("import", 16, theme.ACCENT_HI)); btn.setIconSize(QSize(16, 16))
        btn.setCursor(Qt.PointingHandCursor); btn.setMinimumWidth(176)
        on_click(btn, self.mousePressEvent, None)
        formats = QLabel("JPG  ·  PNG  ·  WEBP  ·  TIFF  ·  HEIC")
        formats.setObjectName("caption"); formats.setAlignment(Qt.AlignCenter)
        theme.tracked(formats, 0.6)
        trust = QWidget(); tl = QHBoxLayout(trust); tl.setContentsMargins(0, 0, 0, 0); tl.setSpacing(8)
        shield = QLabel(); shield.setPixmap(theme.line_icon("shield", 16, theme.OK).pixmap(16, 16))
        tt = QLabel(self._texts[1]); tt.setObjectName("trust")
        tl.addStretch(1); tl.addWidget(shield); tl.addWidget(tt); tl.addStretch(1)
        tip = QLabel(self._texts[2])
        tip.setObjectName("caption"); tip.setAlignment(Qt.AlignCenter); tip.setWordWrap(True)
        col.addWidget(art); col.addSpacing(26)
        col.addWidget(self.hint); col.addSpacing(10)
        col.addWidget(body); col.addSpacing(24)
        col.addWidget(btn, 0, Qt.AlignHCenter); col.addSpacing(14)
        col.addWidget(formats)
        col.addStretch(4)
        col.addWidget(trust); col.addSpacing(8)
        col.addWidget(tip)
        return w

    def show_photo(self, path: str) -> bool:
        ok = self.view.set_image(path)
        self.stack.setCurrentIndex(1 if ok else 0)
        self._set_filled(ok)
        return ok

    def show_empty(self) -> None:
        self.stack.setCurrentIndex(0)
        self._set_filled(False)

    def _set_filled(self, filled: bool) -> None:
        self.setProperty("filled", filled); self.style().unpolish(self); self.style().polish(self)

    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setProperty("hover", True); self.style().polish(self)

    def dragLeaveEvent(self, e):  # noqa: N802
        self.setProperty("hover", False); self.style().polish(self)

    def dropEvent(self, e):  # noqa: N802
        self.setProperty("hover", False); self.style().polish(self)
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self.on_file(u.toLocalFile())
                break

    def mousePressEvent(self, e):  # noqa: N802
        path, _ = QFileDialog.getOpenFileName(self, "Importar foto", "", PHOTO_FILTER)
        if path:
            self.on_file(path)



class StepHeader(QWidget):
    """Cabeçalho de passo: número em violeta, título espaçado e um visto quando concluído."""

    def __init__(self, num: str, title: str):
        super().__init__()
        row = QHBoxLayout(self); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(10)
        n = QLabel(num); n.setObjectName("stepnum"); theme.tracked(n, 0.8)
        t = QLabel(title); t.setObjectName("section"); theme.tracked(t)
        self.title_label = t
        self.done = QLabel(); self.done.setPixmap(theme.status_pixmap("done", 18)); self.done.setFixedSize(18, 18)
        self.done.setToolTip("Concluído")
        self.done.setVisible(False)
        row.addWidget(n); row.addWidget(t); row.addStretch(1); row.addWidget(self.done)

    def set_done(self, done: bool) -> None:
        self.done.setVisible(done)

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)


class StatusBar(QStatusBar):
    """Rodapé: a mensagem alinhada à margem da janela (o QStatusBar fixa 6 px à esquerda)."""

    INDENT = 32
    DOT_W = 16

    def __init__(self):
        super().__init__()
        self.dot = theme.MUTED          # estado da GPU: verde pronta, violeta modo CPU, vermelho erro
        self.dot_tip = ""

    def set_dot(self, color: str, tip: str = "") -> None:
        self.dot = color; self.dot_tip = tip
        self.update()

    def paintEvent(self, e):  # noqa: N802
        msg = self.currentMessage()
        p = QPainter(self)
        opt = QStyleOption(); opt.initFrom(self)
        self.style().drawPrimitive(QStyle.PE_PanelStatusBar, opt, p, self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        halo = QColor(self.dot); halo.setAlphaF(0.22)
        cy = self.height() / 2
        p.setBrush(halo); p.drawEllipse(QPointF(self.INDENT + 4, cy), 5, 5)
        p.setBrush(QColor(self.dot)); p.drawEllipse(QPointF(self.INDENT + 4, cy), 3, 3)
        left = self.INDENT + self.DOT_W
        right = self.width() - 12
        for w in self.findChildren(QWidget):
            if w.parent() is self and w.isVisible() and w.geometry().left() > self.INDENT:
                right = min(right, w.geometry().left() - 16)
        if msg:
            p.setPen(self.palette().windowText().color())
            text = self.fontMetrics().elidedText(msg, Qt.ElideMiddle, max(0, right - left))
            p.drawText(left, 0, right - left, self.height(), Qt.AlignLeft | Qt.AlignVCenter, text)
        p.end()


class Backdrop(QWidget):
    """Fundo da janela na identidade 7x Intelligence: halo violeta e grade técnica (``theme.paint_backdrop``).

    O desenho é feito uma única vez num QPixmap por tamanho de janela; cada repintura só copia o recorte."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        self._cache: QPixmap | None = None

    def resizeEvent(self, e):  # noqa: N802
        self._cache = None
        super().resizeEvent(e)

    def paintEvent(self, e):  # noqa: N802
        dpr = self.devicePixelRatioF() or 1.0
        if self._cache is None or self._cache.deviceIndependentSize().toSize() != self.size() \
                or self._cache.devicePixelRatio() != dpr:
            self._cache = theme.backdrop_pixmap(self.width(), self.height(), dpr)
        p = QPainter(self)
        r = e.rect()
        p.drawPixmap(r, self._cache, QRectF(r.x() * dpr, r.y() * dpr, r.width() * dpr, r.height() * dpr).toRect())
        p.end()


def rule() -> QFrame:
    r = QFrame(); r.setObjectName("rule"); r.setFixedHeight(1)
    return r


def tracked_label(text: str) -> QLabel:
    lb = QLabel(text); lb.setObjectName("section")
    return theme.tracked(lb)



_rule = rule
_tracked = tracked_label
