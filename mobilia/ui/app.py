"""Janela principal do 7xAi (PySide6).

Fluxo único e simples: 1) importar a foto  2) escolher o ambiente  3) gerar  4) assistir e exportar.
"""
from __future__ import annotations

import html
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QMimeData, QPointF, QRectF, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QListView, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QStackedWidget, QStatusBar, QStyle, QStyleOption, QToolButton,
                               QVBoxLayout, QWidget)

from .. import APP_NAME, __version__, bootstrap, diagnostics, runtime
from ..jobs import ROOM_TYPES, STAGES, JobRequest
from . import brand, theme
from .runner import HardwareProbe, JobRunner, QueuedJob

PHOTO_FILTER = "Fotos (*.jpg *.jpeg *.png *.webp *.tif *.tiff *.heic)"
TAGLINE = "Decoração virtual em vídeo, a partir de uma única foto"


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

    def __init__(self, on_file):
        super().__init__()
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
        self.hint = QLabel("Comece pela foto do ambiente vazio")
        self.hint.setObjectName("headline"); self.hint.setAlignment(Qt.AlignCenter)
        body = QLabel("Arraste a imagem para cá ou escolha um arquivo.")
        body.setObjectName("body"); body.setAlignment(Qt.AlignCenter)
        btn = QPushButton("  Escolher foto"); btn.setObjectName("outline")
        btn.setIcon(theme.line_icon("import", 16, theme.ACCENT_HI)); btn.setIconSize(QSize(16, 16))
        btn.setCursor(Qt.PointingHandCursor); btn.setMinimumWidth(176)
        btn.clicked.connect(lambda: self.mousePressEvent(None))
        formats = QLabel("JPG  ·  PNG  ·  WEBP  ·  TIFF  ·  HEIC")
        formats.setObjectName("caption"); formats.setAlignment(Qt.AlignCenter)
        theme.tracked(formats, 0.6)
        trust = QWidget(); tl = QHBoxLayout(trust); tl.setContentsMargins(0, 0, 0, 0); tl.setSpacing(8)
        shield = QLabel(); shield.setPixmap(theme.line_icon("shield", 16, theme.OK).pixmap(16, 16))
        tt = QLabel("Sua foto fica intacta: só os móveis surgem dentro dela."); tt.setObjectName("trust")
        tl.addStretch(1); tl.addWidget(shield); tl.addWidget(tt); tl.addStretch(1)
        tip = QLabel("Dica: fotografe o cômodo inteiro, na altura do peito, com a luz natural do dia.")
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


class PosterView(ImageView):
    """Último quadro do vídeo com um botão de play: aparece até o player mostrar o primeiro quadro
    (ou no lugar dele, se o sistema não tiver backend de vídeo). Clicar abre o vídeo no player do sistema."""

    def __init__(self, on_click):
        super().__init__()
        self.on_click = on_click
        self.corner_hint = "Clique para assistir"
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, e):  # noqa: N802
        self.on_click()

    def paintEvent(self, e):  # noqa: N802
        super().paintEvent(e)
        if self._scaled is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        p.setPen(QPen(QColor(255, 255, 255, 60), 1)); p.setBrush(QColor(10, 10, 12, 190))
        p.drawEllipse(c, 34, 34)
        tri = QPainterPath(QPointF(c.x() - 9, c.y() - 14)); tri.lineTo(c.x() + 15, c.y()); tri.lineTo(c.x() - 9, c.y() + 14)
        tri.closeSubpath()
        p.setPen(Qt.NoPen); p.setBrush(QColor(theme.ACCENT_HI)); p.drawPath(tri)
        p.end()


class VideoView(QWidget):
    """Player em loop do vídeo gerado (QtMultimedia), com pôster até o primeiro quadro chegar."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack)
        self.poster = PosterView(self.open_external)
        self.stack.addWidget(self.poster)
        self.player = None
        self.widget = None
        try:
            from PySide6.QtMultimedia import QMediaPlayer
            from PySide6.QtMultimediaWidgets import QVideoWidget

            self.widget = QVideoWidget()
            self.player = QMediaPlayer(self)
            self.player.setVideoOutput(self.widget)
            self.player.setLoops(QMediaPlayer.Infinite)
            self.stack.addWidget(self.widget)
            self.widget.videoSink().videoFrameChanged.connect(self._first_frame)
        except Exception:  # sem backend de mídia: o pôster fica no lugar (clique abre o player do sistema)
            self.player = None
        self.source: str | None = None

    def _first_frame(self, *_):
        if self.widget is not None and self.stack.currentWidget() is not self.widget:
            self.stack.setCurrentWidget(self.widget)

    def play(self, path: str, poster: str | None = None) -> None:
        self.source = path
        self.stack.setCurrentWidget(self.poster)
        if poster:
            self.poster.set_image(poster)
        if self.player is not None:
            self.player.setSource(QUrl.fromLocalFile(path))
            self.player.play()

    def open_external(self) -> None:
        if self.source:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.source))

    def stop(self) -> None:
        if self.player is not None:
            self.player.stop()


class StageList(QWidget):
    """Linha do tempo das etapas: marcador (pendente / atual / feita) ligado por um filete."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self._pix = {s: theme.status_pixmap(s) for s in ("pending", "current", "done")}
        self._lay = lay
        self.rows = {}
        self.marks = {}
        self.order: list[str] = []
        self.set_stages([(k, label) for k, label, _ in STAGES])

    def set_stages(self, stages) -> None:
        """Troca a lista de etapas (o motor anuncia o modo B, que tem etapas próprias)."""
        stages = [(str(k), str(label)) for k, label in stages]
        if [k for k, _ in stages] == self.order and all(self.rows[k][1] == lb for k, lb in stages):
            return
        while self._lay.count():
            it = self._lay.takeAt(0)
            sub = it.layout()
            if sub is not None:
                while sub.count():
                    w = sub.takeAt(0).widget()
                    if w is not None:
                        w.deleteLater()
        self.rows, self.marks = {}, {}
        for key, label in stages:
            row = QHBoxLayout(); row.setSpacing(12)
            mark = QLabel(); mark.setFixedSize(18, 18); mark.setPixmap(self._pix["pending"])
            lb = QLabel(label); lb.setObjectName("muted")
            row.addWidget(mark); row.addWidget(lb, 1)
            self._lay.addLayout(row)
            self.rows[key] = (lb, label)
            self.marks[key] = mark
        self.order = [k for k, _ in stages]
        self._idx = -1
        self.reset()

    def _style(self, key: str, state: str) -> None:
        lb, _ = self.rows[key]
        self.marks[key].setPixmap(self._pix[state])
        color, weight = {"done": (theme.TEXT_2, 400), "current": (theme.TEXT, 600),
                         "pending": (theme.MUTED, 400)}[state]
        lb.setStyleSheet(f"color: {color}; font-weight: {weight};")

    def set_current(self, stage: str) -> None:
        if stage not in self.rows:
            return
        self._idx = self.order.index(stage)
        for i, k in enumerate(self.order):
            self._style(k, "done" if i < self._idx else "current" if i == self._idx else "pending")
        self.update()

    def reset(self) -> None:
        self._idx = -1
        for k in self.order:
            self._style(k, "pending")
        self.update()

    def paintEvent(self, e):  # noqa: N802
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        marks = [self.marks[k].geometry() for k in self.order]
        for i in range(len(marks) - 1):
            a, b = marks[i], marks[i + 1]
            color = QColor(theme.OK if i < self._idx else theme.HAIRLINE_STRONG)
            if i < self._idx:
                color.setAlphaF(0.45)
            p.setPen(QPen(color, 1.2))
            x = a.center().x() + 0.5
            p.drawLine(QPointF(x, a.bottom() - 1), QPointF(x, b.top() + 2))
        p.end()


class StepHeader(QWidget):
    """Cabeçalho de passo: número em latão, título espaçado e um visto quando concluído."""

    def __init__(self, num: str, title: str):
        super().__init__()
        row = QHBoxLayout(self); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(10)
        n = QLabel(num); n.setObjectName("stepnum"); theme.tracked(n, 0.8)
        t = QLabel(title); t.setObjectName("section"); theme.tracked(t)
        self.done = QLabel(); self.done.setPixmap(theme.status_pixmap("done", 18)); self.done.setFixedSize(18, 18)
        self.done.setToolTip("Concluído")
        self.done.setVisible(False)
        row.addWidget(n); row.addWidget(t); row.addStretch(1); row.addWidget(self.done)

    def set_done(self, done: bool) -> None:
        self.done.setVisible(done)


class StatusBar(QStatusBar):
    """Rodapé: a mensagem alinhada à margem da janela (o QStatusBar fixa 6 px à esquerda)."""

    INDENT = 32
    DOT_W = 16

    def __init__(self):
        super().__init__()
        self.dot = theme.MUTED          # estado da GPU: sálvia pronta, latão modo CPU, vermelho erro
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


def _rule() -> QFrame:
    r = QFrame(); r.setObjectName("rule"); r.setFixedHeight(1)
    return r


def _tracked(text: str) -> QLabel:
    lb = QLabel(text); lb.setObjectName("section")
    return theme.tracked(lb)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        theme.load_fonts()
        self.setWindowTitle(f"{APP_NAME}")
        self.setWindowIcon(theme.window_icon())
        self.resize(1360, 820)
        self.photo_path: str | None = None
        self.room: str | None = None
        self.runner = JobRunner(self)
        self.runner.progress.connect(self._on_progress)
        self.runner.forecast.connect(self._on_forecast)
        self.runner.preview.connect(self._on_preview)
        self.runner.finished.connect(self._on_finished)
        self.runner.mode.connect(self._on_mode)
        self.last_request: JobRequest | None = None
        self.started_at = 0.0
        self.last_output: str | None = None
        self.last_job: QueuedJob | None = None
        self.last_diagnostic: Path | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.last_preview: str | None = None
        self._gaps: list[tuple[object, int, int]] = []   # (espaçador, normal, compacto)
        self._compact: bool | None = None

        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        self.outer = outer
        outer.setContentsMargins(32, 20, 32, 20)
        outer.setSpacing(18)
        outer.addLayout(self._build_header())

        body = QHBoxLayout(); body.setSpacing(20)
        outer.addLayout(body, 1)
        # ---- esquerda: foto / vídeo (o palco)
        self.left = QStackedWidget()
        self.drop = DropZone(self.load_photo)
        self.live = ImageView(caption="PRÉVIA AO VIVO")
        lcard = QFrame(); lcard.setObjectName("stage")
        ll = QVBoxLayout(lcard); ll.setContentsMargins(14, 14, 14, 12); ll.setSpacing(10); ll.addWidget(self.live)
        lcap = QLabel("Prévia ao vivo: os móveis surgem dentro da sua foto, que continua intacta.")
        lcap.setObjectName("stagecap"); lcap.setAlignment(Qt.AlignCenter)
        ll.addWidget(lcap)
        self.video = VideoView()
        vcard = QFrame(); vcard.setObjectName("stage")
        vl = QVBoxLayout(vcard); vl.setContentsMargins(18, 16, 18, 18); vl.setSpacing(14)
        vhead = QHBoxLayout()
        vtag = _tracked("VÍDEO FINAL"); vtag.setStyleSheet(f"color: {theme.ACCENT_HI};")
        self.video_name = QLabel(""); self.video_name.setObjectName("caption")
        vhead.addWidget(vtag); vhead.addStretch(1); vhead.addWidget(self.video_name)
        vcap = QLabel("Foto original, preservada pixel a pixel: só os móveis e as sombras são novos.")
        vcap.setObjectName("stagecap"); vcap.setAlignment(Qt.AlignCenter)
        vl.addLayout(vhead); vl.addWidget(self.video, 1); vl.addWidget(vcap)
        self.left.addWidget(self.drop); self.left.addWidget(lcard); self.left.addWidget(vcard)
        body.addWidget(self.left, 1)
        # ---- direita: passos / progresso / resultado
        self.right = QStackedWidget()
        self.right.setFixedWidth(392)
        self.right.addWidget(self._build_setup())
        self.right.addWidget(self._build_progress())
        self.right.addWidget(self._build_result())
        body.addWidget(self.right)

        sb = StatusBar()
        self.setStatusBar(sb)
        sb.setSizeGripEnabled(False)
        self.sb = sb
        ver = QLabel(f"{APP_NAME}  {__version__}")
        sb.addPermanentWidget(ver)
        sb.showMessage("Detectando GPU…")
        self.probe = HardwareProbe(self)
        self.probe.done.connect(self._on_hardware)
        self._update_ready()
        self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """Tamanho inicial que cabe na tela (até 1360x820): em notebooks 1366x768 ou com escala de 125%
        a janela já abre no modo compacto, sem passar da área útil (barra de tarefas descontada)."""
        w, h = 1360, 820
        scr = self.screen()
        g = scr.availableGeometry() if scr is not None else None
        if g is not None and g.width() > 0 and g.height() > 0:
            w, h = min(w, g.width() - 32), min(h, g.height() - 48)    # moldura e barra de título
        # o modo é escolhido ANTES do primeiro resize: o mínimo do layout normal (~725 px de altura)
        # não deixaria a janela abrir menor que isso numa tela baixa
        self.set_compact(h < self.COMPACT_BELOW)
        self.resize(max(w, 640), max(h, 480))

    # ------------------------------------------------------------------ páginas
    def _build_header(self) -> QHBoxLayout:
        head = QHBoxLayout(); head.setSpacing(0); head.setContentsMargins(2, 6, 0, 2)
        mark = QLabel(); mark.setPixmap(brand.wordmark_pixmap(30)); mark.setToolTip(APP_NAME)
        mark.setAccessibleName(APP_NAME)
        rule = QFrame(); rule.setObjectName("vrule"); rule.setFixedSize(1, 22)
        tag = QLabel(TAGLINE); tag.setObjectName("tagline")
        lock = QLabel(); lock.setPixmap(theme.line_icon("chip", 14, theme.MUTED).pixmap(14, 14))
        priv = QLabel("Processado no seu computador"); priv.setObjectName("privacy")
        priv.setToolTip("A foto e o vídeo não saem do seu PC: tudo é gerado aqui, na sua placa de vídeo.")
        head.addWidget(mark, 0, Qt.AlignVCenter)
        head.addSpacing(22); head.addWidget(rule, 0, Qt.AlignVCenter); head.addSpacing(22)
        head.addWidget(tag, 0, Qt.AlignVCenter); head.addStretch(1)
        head.addWidget(lock, 0, Qt.AlignVCenter); head.addSpacing(8); head.addWidget(priv, 0, Qt.AlignVCenter)
        return head

    def _gap(self, lay, normal: int, compact: int | None = None) -> None:
        """Espaço vertical que encolhe no modo compacto (notebooks de 768 px de altura)."""
        lay.addSpacing(normal)
        self._gaps.append((lay.itemAt(lay.count() - 1).spacerItem(), normal,
                           round(normal * 0.5) if compact is None else compact))

    def _build_setup(self) -> QWidget:
        w = QFrame(); w.setObjectName("card")
        lay = QVBoxLayout(w); lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(0)
        self.step_photo = StepHeader("01", "FOTO DO AMBIENTE")
        self.btn_import = QPushButton("  Importar foto…")
        self.btn_import.setToolTip("Escolher a foto do ambiente, vazio ou mobiliado (JPG, PNG, WEBP, TIFF ou HEIC). "
                                   "O 7xAi detecta sozinho: vazio recebe móveis; mobiliado começa vazio e os "
                                   "próprios móveis da foto voltam um a um")
        self.btn_import.setIcon(theme.line_icon("photo", 16, theme.TEXT_2)); self.btn_import.setIconSize(QSize(16, 16))
        self.btn_import.clicked.connect(lambda: self.drop.mousePressEvent(None))
        self.photo_label = QLabel("Nenhuma foto selecionada"); self.photo_label.setObjectName("hint")
        self.step_room = StepHeader("02", "TIPO DE AMBIENTE")
        grid = QGridLayout(); grid.setSpacing(8)
        self.room_group = QButtonGroup(self); self.room_group.setExclusive(True)
        self.room_buttons = {}
        for i, (key, label) in enumerate(ROOM_TYPES.items()):
            b = QToolButton(); b.setObjectName("room"); b.setCheckable(True)
            b.setText(label); b.setIcon(theme.room_icon(key)); b.setIconSize(QSize(40, 40))
            b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon); b.setMinimumSize(104, 90)
            b.setFixedHeight(90)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self.select_room(k))
            self.room_group.addButton(b); self.room_buttons[key] = b
            grid.addWidget(b, i // 3, i % 3)
        self.step_quality = StepHeader("03", "QUALIDADE")
        self.quality = QComboBox(); self.quality.setView(QListView())
        self.quality.addItem("Final (recomendado com GPU)", "final")
        self.quality.addItem("Prévia rápida", "previa")
        self.quality.addItem("Automática", "auto")
        self.ready_hint = QLabel(""); self.ready_hint.setObjectName("hint"); self.ready_hint.setAlignment(Qt.AlignCenter)
        self.btn_generate = QPushButton("Gerar vídeo"); self.btn_generate.setObjectName("primary")
        self.btn_generate.setCursor(Qt.PointingHandCursor)
        # clicked(bool) mandaria checked=False como "mode": sempre chama sem argumentos
        self.btn_generate.clicked.connect(lambda _checked=False: self.generate())
        g = lambda n: self._gap(lay, n)  # noqa: E731
        lay.addWidget(self.step_photo); g(14)
        lay.addWidget(self.btn_import); g(10); lay.addWidget(self.photo_label)
        g(22); lay.addWidget(_rule()); g(22)
        lay.addWidget(self.step_room); g(14); lay.addLayout(grid)
        g(22); lay.addWidget(_rule()); g(22)
        lay.addWidget(self.step_quality); g(14); lay.addWidget(self.quality)
        lay.addStretch(1)
        lay.addWidget(self.ready_hint); g(12)
        lay.addWidget(self.btn_generate)
        self._cards = [lay]
        return w

    def _build_progress(self) -> QWidget:
        w = QFrame(); w.setObjectName("card")
        lay = QVBoxLayout(w); lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(0)
        head = QHBoxLayout()
        s = _tracked("GERANDO O VÍDEO")
        self.elapsed = QLabel(""); self.elapsed.setObjectName("hint")
        head.addWidget(s); head.addStretch(1); head.addWidget(self.elapsed)
        self.bar = QProgressBar(); self.bar.setRange(0, 1000); self.bar.setTextVisible(False)
        self.pct = QLabel(); self.pct.setObjectName("pct"); self._set_pct(0.0)
        self.msg = QLabel("Preparando…"); self.msg.setWordWrap(True); self.msg.setObjectName("body")
        # previsão do tempo total (controle de orçamento do render, PT7): aparece após o quadro final
        self.forecast = QLabel(""); self.forecast.setObjectName("hint"); self.forecast.setWordWrap(True); self.forecast.hide()
        # aviso "primeira execução — mais lenta" (estado da instalação, ver bootstrap.first_run_notice)
        self.notice = QLabel(""); self.notice.setObjectName("notice"); self.notice.setWordWrap(True)
        self.notice.hide()
        self.stages = StageList()
        self.btn_cancel = QPushButton("Cancelar"); self.btn_cancel.setObjectName("danger")
        self.btn_cancel.clicked.connect(self.cancel)
        g = lambda n: self._gap(lay, n)  # noqa: E731
        lay.addLayout(head); g(18)
        lay.addWidget(self.pct); g(10); lay.addWidget(self.bar)
        g(14); lay.addWidget(self.msg)
        g(8); lay.addWidget(self.forecast)
        g(16); lay.addWidget(self.notice)
        g(20); lay.addWidget(_rule()); g(20)
        lay.addWidget(self.stages)
        self._cards.append(lay)
        lay.addStretch(1)
        lay.addWidget(self.btn_cancel)
        return w

    def _build_result(self) -> QWidget:
        w = QFrame(); w.setObjectName("card")
        lay = QVBoxLayout(w); lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(0)
        head = QHBoxLayout(); head.setSpacing(10)
        badge = QLabel(); badge.setPixmap(theme.status_pixmap("done", 18)); badge.setFixedSize(18, 18)
        s = _tracked("VÍDEO PRONTO"); s.setStyleSheet(f"color: {theme.OK};")
        head.addWidget(badge); head.addWidget(s); head.addStretch(1)
        self.result_title = QLabel("Seu vídeo está pronto"); self.result_title.setObjectName("result")
        self.result_title.setWordWrap(True)
        self.result_info = QLabel(""); self.result_info.setObjectName("hint"); self.result_info.setWordWrap(True)
        self.result_meta = QLabel(""); self.result_meta.setObjectName("hint")
        seal = QFrame(); seal.setObjectName("sealbox")
        sl = QHBoxLayout(seal); sl.setContentsMargins(12, 11, 12, 11); sl.setSpacing(10)
        sico = QLabel(); sico.setPixmap(theme.line_icon("check", 16, theme.ACCENT_HI).pixmap(16, 16))
        sico.setAlignment(Qt.AlignTop); sico.setFixedWidth(16)
        stext = QLabel(self.SEAL_A)
        stext.setObjectName("seal"); stext.setWordWrap(True)
        self.seal_text = stext
        # aviso do portão de qualidade do modo B (vídeo gerado a pedido com a sala vazia reprovada)
        self.result_warning = QLabel(""); self.result_warning.setObjectName("notice")
        self.result_warning.setWordWrap(True); self.result_warning.hide()
        sl.addWidget(sico); sl.addWidget(stext, 1)
        # narrativa de designer (PT12): por que cada peça foi escolhida e onde ficou
        self.notes_title = _tracked("ESCOLHAS DO DESIGNER")
        self.result_notes = QLabel(""); self.result_notes.setObjectName("body"); self.result_notes.setWordWrap(True)
        self.result_notes.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result_notes.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        st = _tracked("TEMPO POR ETAPA")
        self.result_times = QLabel(""); self.result_times.setObjectName("muted"); self.result_times.setWordWrap(True)
        self.result_times.setTextFormat(Qt.RichText)
        self.result_times.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        # observações abaixo da tabela (ex.: "primeira execução"), em rótulo próprio para quebrar linha sem corte
        self.result_times_note = QLabel(""); self.result_times_note.setObjectName("hint")
        self.result_times_note.setWordWrap(True); self.result_times_note.hide()
        self.btn_export = QPushButton("  Exportar MP4…"); self.btn_export.setObjectName("primary")
        self.btn_export.setIcon(theme.line_icon("export", 16, theme.ON_ACCENT)); self.btn_export.setIconSize(QSize(16, 16))
        self.btn_export.clicked.connect(self.export)
        self.btn_folder = QPushButton("  Abrir pasta"); self.btn_folder.clicked.connect(self.open_folder)
        self.btn_folder.setIcon(theme.line_icon("folder", 16, theme.TEXT_2))
        self.btn_diag = QPushButton("  Copiar diagnóstico")
        self.btn_diag.setIcon(theme.line_icon("copy", 16, theme.TEXT_2))
        self.btn_diag.setToolTip("Cria um .zip com o relatório de tempos e os registros desta geração "
                                 "e copia o arquivo para a área de transferência.")
        self.btn_diag.clicked.connect(lambda: self.copy_diagnostic())
        row = QHBoxLayout(); row.setSpacing(8); row.addWidget(self.btn_folder, 1); row.addWidget(self.btn_diag, 1)
        self.chk_diag_photo = QCheckBox("Incluir a foto no diagnóstico")
        self.btn_new = QPushButton("  Começar com outra foto"); self.btn_new.setObjectName("ghost")
        self.btn_new.setIcon(theme.line_icon("plus", 16, theme.TEXT_2))
        self.btn_new.clicked.connect(self.new_photo)
        g = lambda n: self._gap(lay, n)  # noqa: E731
        lay.addLayout(head); g(12)
        lay.addWidget(self.result_title); g(8)
        lay.addWidget(self.result_info); lay.addSpacing(4); lay.addWidget(self.result_meta)
        g(16); lay.addWidget(self.result_warning); lay.addWidget(seal)
        # escolhas do designer + tempos numa única área rolável: nada é cortado nem sobreposto
        details = QWidget(); details.setObjectName("details")
        dl = QVBoxLayout(details); dl.setContentsMargins(0, 0, 6, 0); dl.setSpacing(0)
        self._dgap_notes = QWidget(); self._dgap_notes.setFixedHeight(10)
        dl.addWidget(self.notes_title); dl.addWidget(self._dgap_notes); dl.addWidget(self.result_notes)
        self._dgap_times = QWidget(); self._dgap_times.setFixedHeight(22); dl.addWidget(self._dgap_times)
        dl.addWidget(st); dl.addSpacing(10); dl.addWidget(self.result_times)
        dl.addSpacing(6); dl.addWidget(self.result_times_note); dl.addStretch(1)
        self.notes_scroll = QScrollArea(); self.notes_scroll.setWidgetResizable(True)
        self.notes_scroll.setFrameShape(QFrame.NoFrame); self.notes_scroll.setWidget(details)
        self.notes_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.notes_scroll.setMinimumHeight(120)
        self.notes_scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        g(22); lay.addWidget(self.notes_scroll, 1); g(16)
        g(16)
        lay.addWidget(self.btn_export); g(10); lay.addLayout(row)
        g(12); lay.addWidget(self.chk_diag_photo)
        g(14); lay.addWidget(_rule()); g(10)
        lay.addWidget(self.btn_new)
        self._cards.append(lay)
        return w

    SEAL_A = ("Fora dos móveis e das sombras, a sua foto foi preservada pixel a pixel "
              "(verificado em cada quadro).")
    SEAL_B = ("O vídeo termina idêntico à sua foto, na resolução do vídeo (verificado byte a byte). A sala vazia "
              "do início foi reconstruída pelo 7xAi só onde havia móveis: é imagem ilustrativa.")
    NOTES_A = "ESCOLHAS DO DESIGNER"

    def _set_pct(self, frac: float) -> None:
        self.pct.setText(f'<span style="font-size:60px; font-weight:300;">{frac * 100:.0f}</span>'
                         f'<span style="font-size:34px; font-weight:300; color:{theme.ACCENT};">%</span>')

    # ------------------------------------------------------------------ ações
    COMPACT_BELOW = 790   # px lógicos: 1366x768 maximizado (barra de tarefas incluída) fica abaixo

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self.set_compact(self.height() < self.COMPACT_BELOW)

    def set_compact(self, compact: bool) -> None:
        """Layout compacto para telas baixas: menos respiro, ladrilhos menores (tudo continua visível)."""
        if compact == self._compact or not hasattr(self, "stages"):
            return
        self._compact = compact
        for sp, normal, small in self._gaps:
            sp.changeSize(0, small if compact else normal)
        m = 16 if compact else 24
        for lay in self._cards:
            lay.setContentsMargins(m, m, m, m)
            lay.invalidate()
        self.outer.setContentsMargins(*((24, 10, 24, 10) if compact else (32, 20, 32, 20)))
        self.outer.setSpacing(12 if compact else 18)
        for b in self.room_buttons.values():
            b.setFixedHeight(70 if compact else 90)
            b.setIconSize(QSize(30, 30) if compact else QSize(40, 40))
        self.stages.layout().setSpacing(6 if compact else 10)

    def showEvent(self, e):  # noqa: N802
        super().showEvent(e)
        theme.dark_title_bar(self)
        if not getattr(self, "_probed", False):
            self._probed = True
            self.probe.start()

    def load_photo(self, path: str) -> None:
        if not Path(path).is_file() or not self.drop.show_photo(path):
            QMessageBox.warning(self, APP_NAME, "Não foi possível abrir essa imagem.")
            return
        self.photo_path = path
        size = self.drop.view._pm.size() if self.drop.view._pm is not None else None
        dims = f"   ·   {size.width()} × {size.height()} px" if size else ""
        fm = QFontMetrics(self.photo_label.font())
        name = fm.elidedText(Path(path).name, Qt.ElideMiddle, 330 - fm.horizontalAdvance(dims))
        self.photo_label.setText(name + dims)
        self.photo_label.setToolTip(path)
        self.btn_import.setText("  Trocar foto…")
        self.left.setCurrentIndex(0)
        self._update_ready()

    def select_room(self, key: str) -> None:
        self.room = key
        self.room_buttons[key].setChecked(True)
        self._update_ready()

    def _update_ready(self) -> None:
        self.btn_generate.setEnabled(bool(self.photo_path and self.room) and not self.runner.busy())
        self.step_photo.set_done(bool(self.photo_path))
        self.step_room.set_done(bool(self.room))
        self.step_quality.set_done(self.quality.currentIndex() >= 0)   # sempre há um padrão válido
        if not self.photo_path:
            hint = "Comece importando a foto do ambiente (vazio ou mobiliado)."
        elif not self.room:
            hint = "Agora escolha o tipo de ambiente."
        else:
            hint = "Tudo pronto. O vídeo é gerado aqui, na sua GPU."
        self.ready_hint.setText(hint)

    def generate(self, mode: str = "auto", accept_weak_plate: bool = False) -> None:
        """Gera o vídeo. ``mode``/``accept_weak_plate`` só vêm da tela de aviso do portão de qualidade (a tela
        inicial pede apenas a foto e o tipo de ambiente; o modo é detectado pela foto)."""
        if not (self.photo_path and self.room):
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        work = runtime.runtime_paths().jobs_dir / f"{stamp}_{Path(self.photo_path).stem}"
        out = work / f"{Path(self.photo_path).stem}_{self.room}.mp4"
        req = JobRequest(self.photo_path, self.room, str(out), quality=self.quality.currentData(),
                         work_dir=str(work), mode=mode, accept_weak_plate=accept_weak_plate)
        self.last_request = req
        self.stages.set_stages([(k, label) for k, label, _ in STAGES])
        self.stages.reset(); self.bar.setValue(0); self._set_pct(0.0); self.msg.setText("Iniciando o motor…")
        self.elapsed.setText("")
        self.forecast.setText(""); self.forecast.hide()
        self.live.set_image(self.photo_path)
        self.last_preview = None
        notice = self.first_run_notice()
        self.notice.setText(notice or ""); self.notice.setVisible(bool(notice))
        self.left.setCurrentIndex(1); self.right.setCurrentIndex(1)
        self.started_at = time.monotonic(); self.timer.start(1000)
        self.runner.submit(req)
        self._update_ready()

    @staticmethod
    def first_run_notice() -> str | None:
        """Texto do aviso de primeira execução (None nas execuções seguintes)."""
        try:
            return bootstrap.first_run_notice(bootstrap.load_state(runtime.runtime_paths().state_file))
        except Exception:  # aviso é cosmético: nunca impede a geração
            return None

    def cancel(self) -> None:
        self.btn_cancel.setEnabled(False)
        self.msg.setText("Cancelando…")
        self.runner.cancel()

    def export(self) -> None:
        if not self.last_output:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Exportar vídeo", Path(self.last_output).name, "Vídeo MP4 (*.mp4)")
        if dest:
            shutil.copyfile(self.last_output, dest)
            self.statusBar().showMessage(f"Vídeo exportado: {dest}", 8000)
            self.statusBar().setToolTip(dest)

    def open_folder(self) -> None:
        if self.last_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_output).parent)))

    def copy_diagnostic(self, include_photo: bool | None = None, job: QueuedJob | None = None) -> Path | None:
        """Gera o zip de diagnóstico do último trabalho (sem a foto, salvo se marcado) e copia o
        arquivo para a área de transferência (colar no e-mail/WhatsApp anexa o zip)."""
        job = job or self.last_job
        work = job.work_dir if job else None
        if not work or not Path(work).is_dir():
            self.statusBar().showMessage("Nenhum trabalho para diagnosticar.", 6000)
            return None
        if include_photo is None:
            include_photo = self.chk_diag_photo.isChecked()
        dest = runtime.runtime_paths().home / "diagnosticos" / f"diagnostico_{Path(work).name}.zip"
        try:
            path = diagnostics.make_diagnostic_zip(work, dest, include_photo=include_photo,
                                                   app_version=__version__)
        except OSError as e:
            self.statusBar().showMessage(f"Não foi possível criar o diagnóstico: {e}", 10000)
            return None
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(path))])
        mime.setText(str(path))
        QApplication.clipboard().setMimeData(mime)
        self.last_diagnostic = path
        self.statusBar().showMessage(f"Diagnóstico copiado ({'com' if include_photo else 'sem'} a foto): "
                                     f"{path.name}. Cole no e-mail ou no WhatsApp.", 15000)
        self.statusBar().setToolTip(str(path))
        return path

    def new_photo(self) -> None:
        self.video.stop()
        self.photo_path = None
        self.photo_label.setText("Nenhuma foto selecionada")
        self.photo_label.setToolTip("")
        self.btn_import.setText("  Importar foto…")
        self.drop.show_empty()
        self.left.setCurrentIndex(0); self.right.setCurrentIndex(0)
        self._update_ready()

    # ------------------------------------------------------------------ eventos do worker
    def _tick(self) -> None:
        s = int(time.monotonic() - self.started_at)
        self.elapsed.setText(f"Tempo decorrido: {s // 60:d} min {s % 60:02d} s")

    def _on_progress(self, stage: str, frac: float, message: str) -> None:
        self.bar.setValue(int(frac * 1000)); self._set_pct(frac)
        self.msg.setText(message); self.stages.set_current(stage)

    def _on_mode(self, mode: str, stages: list) -> None:
        if stages:
            self.stages.set_stages(stages)
        if mode == "B":
            self.statusBar().showMessage("Foto mobiliada: o vídeo começa com a sala vazia e os móveis da própria "
                                         "foto voltam um a um.", 10000)

    def _on_forecast(self, total_s: float, message: str) -> None:
        self.forecast.setText(message or f"Previsão: ≈ {int(round(total_s))} s no total")
        self.forecast.show()

    def _on_preview(self, path: str) -> None:
        if path and Path(path).exists():
            self.live.set_image(path)
            self.last_preview = path

    def _on_finished(self, job: QueuedJob) -> None:
        self.timer.stop()
        self.btn_cancel.setEnabled(True)
        self.last_job = job
        self._update_ready()
        if job.status == "pronto" and job.output and Path(job.output).exists():
            self.last_output = job.output
            size = human_size(Path(job.output).stat().st_size)
            secs = int(time.monotonic() - self.started_at) if self.started_at else 0
            took = f"{secs // 60:d} min {secs % 60:02d} s" if secs >= 60 else (f"{secs} s" if secs >= 1 else "")
            self.result_info.setText(Path(job.output).name)
            self.result_info.setToolTip(job.output)
            meta = ["MP4"] + ([size] if size else []) + ([f"gerado em {took}"] if took else [])
            self.result_meta.setText("  ·  ".join(meta))
            self.video_name.setText(Path(job.output).name)
            times = self._times_text(job)
            self.result_times.setText(self._times_html(times))
            note = "\n".join(l for l in times.splitlines() if ": " not in l)
            self.result_times_note.setText(note); self.result_times_note.setVisible(bool(note))
            # altura mínima da tabela (linhas + folga do "Total"): a narrativa rola, os tempos não são cortados
            notes = self._designer_notes(job)
            rep = self._report(job)
            mode_b = (job.mode or rep.get("modo", {}).get("modo")) == "B"
            self.seal_text.setText(self.SEAL_B if mode_b else self.SEAL_A)
            self.notes_title.setText(rep.get("titulo_narrativa") or self.NOTES_A)
            self.result_warning.setText(job.warning or rep.get("aviso") or "")
            self.result_warning.setVisible(bool(job.warning or rep.get("aviso")))
            self.result_notes.setText("\n".join(f"•  {n}" for n in notes))
            for wdg in (self.notes_title, self._dgap_notes, self.result_notes, self._dgap_times):
                wdg.setVisible(bool(notes))
            self.left.setCurrentIndex(2); self.right.setCurrentIndex(2)
            poster = self.last_preview if self.last_preview and Path(self.last_preview).exists() else self.photo_path
            self.video.play(job.output, poster)
            return
        self.left.setCurrentIndex(0); self.right.setCurrentIndex(0)
        if job.status == "reprovado":
            self._on_gate_rejected(job)
            return
        if job.status == "cancelado":
            self.statusBar().showMessage("Geração cancelada.", 6000)
            return
        self.show_error(job.error or "Erro desconhecido.", job.detail)

    # ------------------------------------------------------------------ portão de qualidade (modo B)
    GATE_CHOICES = {"forcar": "Gerar mesmo assim", "modo_a": "Usar o modo sala vazia → mobiliada",
                    "outra": "Escolher outra foto"}

    def _on_gate_rejected(self, job: QueuedJob) -> None:
        """Sala vazia reprovada: nenhum MP4 foi entregue. Mostra o aviso e as alternativas."""
        choice = self.ask_gate_choice(job)
        if choice == "forcar":
            self.generate(mode="B", accept_weak_plate=True)
        elif choice == "modo_a":
            self.generate(mode="A")
        elif choice == "outra":
            self.new_photo()
        else:
            self.statusBar().showMessage("Nenhum vídeo gerado: a sala vazia não passou na verificação.", 10000)

    def ask_gate_choice(self, job: QueuedJob) -> str | None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(APP_NAME)
        box.setText("A sala vazia não ficou boa o suficiente.")
        box.setInformativeText(job.warning or "A verificação automática reprovou a sala vazia do início do vídeo.")
        buttons = {box.addButton(label, QMessageBox.ActionRole): key for key, label in self.GATE_CHOICES.items()}
        box.addButton("Fechar", QMessageBox.RejectRole)
        box.exec()
        return buttons.get(box.clickedButton())

    @staticmethod
    def _report(job: QueuedJob) -> dict:
        try:
            return json.loads(Path(job.report).read_text(encoding="utf-8")) if job.report else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _times_text(job: QueuedJob) -> str:
        if not job.report:
            return ""
        try:
            rep = json.loads(Path(job.report).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        return diagnostics.summary_text(rep)

    @staticmethod
    def _designer_notes(job: QueuedJob) -> list[str]:
        """Narrativa de designer do trabalho: report.json (``narrativa``) ou, sem ela, layout.json."""
        srcs = []
        if job.report:
            srcs.append((Path(job.report), lambda d: d.get("narrativa") or (d.get("layout") or {}).get("notes")))
        if job.work_dir:
            srcs.append((Path(job.work_dir) / "layout.json", lambda d: d.get("notes")))
        for path, pick in srcs:
            try:
                notes = pick(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, AttributeError):
                continue
            if notes:
                return [str(n) for n in notes]
        return []

    @staticmethod
    def _times_html(text: str) -> str:
        """Tabela "etapa ....... tempo" a partir de ``diagnostics.summary_text``."""
        if not text:
            return ""
        rows = []
        for line in text.splitlines():
            label, sep, value = line.rpartition(": ")
            if not sep:
                continue
            total = label == "Total"
            color = theme.TEXT if total else theme.MUTED
            vcolor = theme.TEXT if total else theme.TEXT_2
            weight = "600" if total else "400"
            pad = "10px 0 0 0" if total else "3px 0"
            rows.append(f'<tr><td style="color:{color}; font-weight:{weight}; padding:{pad};">{html.escape(label)}</td>'
                        f'<td align="right" style="color:{vcolor}; font-weight:{weight}; padding:{pad};">'
                        f'{html.escape(value)}</td></tr>')
        out = f'<table width="100%" cellspacing="0" cellpadding="0">{"".join(rows)}</table>'
        return out   # as linhas sem "etapa: tempo" vão para ``result_times_note``

    def show_error(self, message: str, detail: str = "") -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(APP_NAME)
        box.setText("Não foi possível gerar o vídeo.")
        box.setInformativeText(message)
        if detail:
            box.setDetailedText(detail)
        diag = None
        if self.last_job is not None and self.last_job.work_dir and Path(self.last_job.work_dir).is_dir():
            diag = box.addButton("Copiar diagnóstico", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if diag is not None and box.clickedButton() is diag:
            self.copy_diagnostic()

    def _on_hardware(self, info) -> None:
        if not info:
            self.sb.set_dot(theme.DANGER)
            self.statusBar().showMessage("GPU: não detectada (o motor ainda não está instalado ou não respondeu).")
            return
        prof = info.get("profile", {})
        gpus = info.get("gpus") or []
        gpu = max(gpus, key=lambda g: g.get("vram_total_mb", 0)) if gpus else None
        g = f"{gpu['name']} · {gpu['vram_total_mb'] / 1024:.0f} GB" if gpu else "sem GPU NVIDIA (modo CPU)"
        self.statusBar().showMessage(f"GPU: {g}     ·     Perfil: {prof.get('name', '?')} — "
                                     f"{prof.get('description', '')}")
        self.sb.set_dot(theme.OK if gpu else theme.ACCENT)
        if gpu is None:
            self.quality.setCurrentIndex(1)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--worker", "worker"):
        from ..cli import main as cli_main

        return cli_main(argv[1:])
    selftest = "--selftest" in argv
    if selftest:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if "--version" in argv:
        print(f"{APP_NAME} {__version__}")
        return 0
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    theme.apply(app)
    if not selftest and runtime.uses_embedded_runtime():
        from .setup_dialog import ensure_runtime

        if not ensure_runtime(None):
            return 1
    win = MainWindow()
    if selftest:
        win.show()
        app.processEvents()
        ok = win.windowTitle() == APP_NAME and not theme.app_icon().isNull() and theme.assets_ok()
        print("selftest ok" if ok else "selftest FALHOU")
        return 0 if ok else 2
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
