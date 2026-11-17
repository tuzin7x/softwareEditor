"""Peças visuais do Lead Finder: indicador animado, pilha com transição suave, chips e gráficos simples."""
from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QGraphicsOpacityEffect, QHBoxLayout, QLabel, QSizePolicy, QStackedWidget,
                               QToolButton, QVBoxLayout, QWidget)

from mobilia.ui import theme

from . import style


class KpiCard(QWidget):
    """Indicador com número animado (sobe até o valor em ~0,6 s)."""

    clicked = Signal()

    def __init__(self, title: str, accent: str = theme.ACCENT_HI, hint: str = ""):
        super().__init__()
        frame = style.glass("lfKpi")
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.addWidget(frame)
        lay = QVBoxLayout(frame); lay.setContentsMargins(16, 12, 16, 12); lay.setSpacing(2)
        self.value_label = QLabel("0"); self.value_label.setObjectName("lfKpiValue")
        t = QLabel(title.upper()); t.setObjectName("lfKpiLabel"); theme.tracked(t, 1.0)
        bar = QWidget(); bar.setFixedHeight(2)
        bar.setStyleSheet(f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {accent}, stop:1 transparent);"
                          "border-radius: 1px;")
        lay.addWidget(bar); lay.addSpacing(6); lay.addWidget(self.value_label); lay.addWidget(t)
        if hint:
            self.setToolTip(hint)
        self.value = 0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(600)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(lambda v: self.value_label.setText(f"{int(v):,}".replace(",", ".")))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, value: int, animate: bool = True) -> None:
        old, self.value = self.value, int(value)
        self._anim.stop()
        if animate and old != self.value:
            self._anim.setStartValue(old); self._anim.setEndValue(self.value); self._anim.start()
        else:
            self.value_label.setText(f"{self.value:,}".replace(",", "."))

    def mousePressEvent(self, e):  # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(e)


class FadeStack(QStackedWidget):
    """``QStackedWidget`` com transição de opacidade ao trocar de página."""

    def fade_to(self, index: int) -> None:
        if index == self.currentIndex():
            return
        self.setCurrentIndex(index)
        w = self.currentWidget()
        eff = QGraphicsOpacityEffect(w); w.setGraphicsEffect(eff)
        anim = QPropertyAnimation(eff, b"opacity", w)
        anim.setDuration(220); anim.setStartValue(0.0); anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.finished.connect(lambda: w.setGraphicsEffect(None))
        anim.start()


class FlowLayoutWidget(QWidget):
    """Chips em linhas que quebram (layout simples por colunas fixas)."""

    def __init__(self, columns: int = 3):
        super().__init__()
        self.columns = columns
        self.lay = QVBoxLayout(self); self.lay.setContentsMargins(0, 0, 0, 0); self.lay.setSpacing(6)
        self.items: list[QWidget] = []

    def clear(self) -> None:
        self.items.clear()
        while self.lay.count():
            it = self.lay.takeAt(0)
            if it.layout():
                while it.layout().count():
                    w = it.layout().takeAt(0).widget()
                    if w is not None:
                        w.deleteLater()
            elif it.widget():
                it.widget().deleteLater()

    def set_items(self, widgets: list[QWidget]) -> None:
        self.clear()
        self.items = list(widgets)
        for i in range(0, len(widgets), self.columns):
            r = QHBoxLayout(); r.setSpacing(6)
            for w in widgets[i:i + self.columns]:
                r.addWidget(w, 1)
            for _ in range(self.columns - len(widgets[i:i + self.columns])):
                r.addStretch(1)
            self.lay.addLayout(r)


def toggle_chip(text: str, checked: bool = False, tip: str = "") -> QToolButton:
    b = QToolButton(); b.setObjectName("chip"); b.setCheckable(True); b.setChecked(checked); b.setText(text)
    b.setCursor(Qt.PointingHandCursor); b.setToolButtonStyle(Qt.ToolButtonTextOnly)
    b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); b.setFixedHeight(32)
    if tip:
        b.setToolTip(tip)
    return b


class FunnelChart(QWidget):
    """Funil de vendas: barras centradas, largura proporcional, com taxa de conversão entre etapas."""

    def __init__(self):
        super().__init__()
        self.stages: list[tuple[str, int]] = []
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_data(self, stages: list[tuple[str, int]]) -> None:
        self.stages = stages
        self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        n = len(self.stages)
        if not n:
            return
        top = max((v for _, v in self.stages), default=0) or 1
        h = self.height() / n
        label_w = 170.0
        w_avail = max(40.0, self.width() - label_w - 70)
        f = QFont(self.font()); f.setPixelSize(12); p.setFont(f)
        for i, (name, v) in enumerate(self.stages):
            y = i * h + 4
            bh = h - 8
            frac = v / top
            bw = max(6.0, w_avail * frac) if v else 6.0
            x = label_w + (w_avail - bw) / 2
            c = QColor(theme.ACCENT); c.setAlphaF(0.30 + 0.60 * (1 - i / max(1, n - 1)))
            p.setPen(Qt.NoPen); p.setBrush(c if v else QColor(theme.TRACK))
            p.drawRoundedRect(QRectF(x, y, bw, bh), 6, 6)
            p.setPen(QColor(theme.TEXT_2))
            p.drawText(QRectF(0, y, label_w - 12, bh), Qt.AlignVCenter | Qt.AlignRight, name)
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(label_w + w_avail + 8, y, 62, bh), Qt.AlignVCenter | Qt.AlignLeft,
                       f"{v:,}".replace(",", "."))
            if i:
                prev = self.stages[i - 1][1]
                rate = f"{100 * v / prev:.0f}%" if prev else "—"
                p.setPen(QColor(theme.MUTED))
                p.drawText(QRectF(x, y, bw, bh), Qt.AlignCenter, rate if bw > 40 else "")
        p.end()


class BarList(QWidget):
    """Barras horizontais com rótulo (distribuição geográfica, fontes)."""

    def __init__(self, color: str = theme.ACCENT, empty_text: str = "Sem dados ainda."):
        super().__init__()
        self.items: list[tuple[str, int]] = []
        self.color = color
        self.empty_text = empty_text
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def set_data(self, items) -> None:
        self.items = [(str(a), int(b)) for a, b in items]
        self.setMinimumHeight(max(60, 26 * len(self.items) + 8))
        self.updateGeometry(); self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        f = QFont(self.font()); f.setPixelSize(12); p.setFont(f)
        if not self.items:
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, self.empty_text)
            return
        top = max(v for _, v in self.items) or 1
        lw = min(170.0, self.width() * 0.42)
        for i, (name, v) in enumerate(self.items):
            y = 4 + i * 26
            p.setPen(QColor(theme.TEXT_2))
            p.drawText(QRectF(0, y, lw - 10, 20), Qt.AlignVCenter | Qt.AlignLeft,
                       p.fontMetrics().elidedText(name, Qt.ElideRight, int(lw - 10)))
            avail = self.width() - lw - 50
            track = QRectF(lw, y + 6, avail, 8)
            p.setPen(Qt.NoPen); p.setBrush(QColor(theme.SURFACE_3)); p.drawRoundedRect(track, 4, 4)
            p.setBrush(QColor(self.color)); p.drawRoundedRect(QRectF(lw, y + 6, max(4.0, avail * v / top), 8), 4, 4)
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(lw + avail + 6, y, 44, 20), Qt.AlignVCenter | Qt.AlignLeft, f"{v:,}".replace(",", "."))
        p.end()


class ScoreBar(QWidget):
    def __init__(self):
        super().__init__()
        self.value: int | None = None
        self.setFixedHeight(10)

    def set_value(self, v: int | None) -> None:
        self.value = v
        self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0, 1, 0, -1)
        p.setPen(Qt.NoPen); p.setBrush(QColor(theme.TRACK)); p.drawRoundedRect(r, 4, 4)
        if self.value:
            c = QColor(style.PRIORITY_COLOR[3 if self.value >= 70 else 2 if self.value >= 40 else 1])
            p.setBrush(c); p.drawRoundedRect(QRectF(r.left(), r.top(), r.width() * self.value / 100, r.height()), 4, 4)
        p.setPen(QPen(QColor(255, 255, 255, 18), 1)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(r, 4, 4)
        p.end()
