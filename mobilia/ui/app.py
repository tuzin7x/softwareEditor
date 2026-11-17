"""Janela principal do 7xAi (PySide6): redesenho de ambientes por IA.

Fluxo: 1) foto  2) função (mudar estilo, reforma, mobiliar, editar área, fachada, evento, texto)  3) ambiente e
estilo  4) gerar N versões (o motor roda em outro processo: ``mobilia.cli design --json``)  5) resultados com
comparador antes/depois, "Editar área", "Ampliar 4K" (``upscale``), "Vídeo antes e depois" (``showcase``), salvar e
abrir a pasta. O vídeo de montagem antigo (modos A/B) fica em "Avançado (vídeo de montagem, legado)" (``legacy.py``).

Regra dos botões: todo ``clicked`` é ligado por ``widgets.on_click`` (descarta o ``checked`` do sinal) e os pedidos
ao motor são validados em ``DesignRequest.validate`` antes de virar linha de comando.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QListView, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QSlider, QStackedWidget, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__, bootstrap, runtime
from . import brand, theme
from .design_widgets import (FUNCTION_INFO, FUNCTION_ORDER, STYLE_HINTS, BeforeAfterView, FunctionCard, MaskEditor,
                             Segmented, StyleCard, ThumbStrip, VariantGrid, chip, cover_pixmap)
from .legacy import StageList
from .modules import ModuleHost
from .runner import CliResult, CliTask, HardwareProbe
from .widgets import (PHOTO_FILTER, DropZone, ImageView, StatusBar, StepHeader, human_size, on_click, paint_chip,
                      rule, tracked_label)

__all__ = ["MainWindow", "DesignRequest", "DesignOutcome", "CATALOG", "main", "upscale_args", "showcase_args",
           # reexportados (código antigo importava daqui)
           "PHOTO_FILTER", "DropZone", "ImageView", "StatusBar", "StepHeader", "human_size", "paint_chip"]

TAGLINE = "Redesenho de ambientes por IA, a partir de uma única foto"

# Mesmas chaves e rótulos de ``mobilia.design.presets.catalog()`` (conferido em tests/test_ui.py). A interface não
# importa o pacote do motor: ele traz numpy/OpenCV/PyTorch, que ficam fora do executável da interface.
CATALOG: dict[str, dict[str, str]] = {
    "funcoes": {"mudar_estilo": "Mudar estilo", "reforma": "Reforma completa", "redesign_simples": "Redesign simples",
                "mobiliar": "Mobiliar ambiente vazio", "editar_area": "Editar área", "fachada": "Fachada",
                "evento": "Decoração de evento", "texto": "Texto para ambiente"},
    "estilos": {"moderno": "Moderno", "contemporaneo": "Contemporâneo", "escandinavo": "Escandinavo",
                "minimalista": "Minimalista", "industrial": "Industrial", "classico": "Clássico", "luxo": "Luxo",
                "boho": "Boho", "japandi": "Japandi", "rustico": "Rústico", "costeiro": "Costeiro",
                "tropical": "Tropical brasileiro", "mid_century": "Mid-century", "farmhouse": "Fazenda moderna"},
    "ambientes": {"sala": "Sala de estar", "quarto": "Quarto", "cozinha": "Cozinha", "banheiro": "Banheiro",
                  "escritorio": "Escritório", "sala_jantar": "Sala de jantar", "varanda": "Varanda",
                  "fachada": "Fachada"},
    "eventos": {"aniversario": "Aniversário", "casamento": "Casamento", "natal": "Natal",
                "festa_infantil": "Festa infantil", "cha_de_bebe": "Chá de bebê", "reveillon": "Réveillon"},
}
ROOM_SHORT = {"sala": "Sala", "quarto": "Quarto", "cozinha": "Cozinha", "banheiro": "Banheiro",
              "escritorio": "Escritório", "sala_jantar": "Jantar", "varanda": "Varanda", "fachada": "Fachada"}
ASPECTS = {"paisagem": "Paisagem", "retrato": "Retrato", "quadrado": "Quadrado"}
QUALITIES = ("final", "rapido")
NEEDS_PHOTO = {k for k in CATALOG["funcoes"] if k != "texto"}
VIDEO_FORMATS = {"16x9": ("Horizontal 16:9", "1920 × 1080 · YouTube, TV e apresentações"),
                 "9x16": ("Vertical 9:16", "1080 × 1920 · Reels, Stories, TikTok e WhatsApp")}
DESIGN_STAGES = [("preparo", "Preparando a foto"), ("analise", "Analisando o ambiente"),
                 ("geracao", "Gerando as versões"), ("final", "Finalizando")]
TEXT_STAGES = [("preparo", "Preparando"), ("geracao", "Gerando as versões"), ("final", "Finalizando")]


# =============================================================================================== pedido

@dataclass
class DesignRequest:
    """Pedido ao motor (``mobilia.cli design``). ``validate`` barra valores fora do catálogo e tipos errados (um
    ``False`` vindo de um sinal ``clicked(bool)`` nunca vira ``--n False`` nem ``--funcao False``)."""

    funcao: str
    estilo: str | None
    ambiente: str
    n: int
    out_dir: str
    photo: str | None = None
    prompt: str = ""
    evento: str | None = None
    mask: str | None = None
    formato: str = "paisagem"
    quality: str = "final"
    seed: int | None = None

    def validate(self) -> "DesignRequest":
        def check(cond, msg):
            if not cond:
                raise ValueError(msg)

        check(isinstance(self.funcao, str) and self.funcao in CATALOG["funcoes"], f"Função inválida: {self.funcao!r}")
        check(self.estilo is None or (isinstance(self.estilo, str) and self.estilo in CATALOG["estilos"]),
              f"Estilo inválido: {self.estilo!r}")
        check(isinstance(self.ambiente, str) and self.ambiente in CATALOG["ambientes"],
              f"Ambiente inválido: {self.ambiente!r}")
        check(type(self.n) is int and 1 <= self.n <= 4, f"Número de versões inválido: {self.n!r}")
        check(self.evento is None or (isinstance(self.evento, str) and self.evento in CATALOG["eventos"]),
              f"Evento inválido: {self.evento!r}")
        check(isinstance(self.formato, str) and self.formato in ASPECTS, f"Formato inválido: {self.formato!r}")
        check(self.quality in QUALITIES, f"Qualidade inválida: {self.quality!r}")
        check(self.seed is None or type(self.seed) is int, f"Semente inválida: {self.seed!r}")
        check(isinstance(self.prompt, str), "Descrição inválida")
        check(isinstance(self.out_dir, str) and bool(self.out_dir), "Pasta de saída inválida")
        if self.funcao in NEEDS_PHOTO:
            check(isinstance(self.photo, str) and Path(self.photo).is_file(), "Escolha a foto do ambiente.")
        if self.funcao == "editar_area":
            check(isinstance(self.mask, str) and Path(self.mask).is_file(), "Pinte a área a editar.")
        if self.funcao == "evento":
            check(self.evento is not None, "Escolha o evento.")
        return self

    def args(self) -> list[str]:
        """Argumentos de ``mobilia.cli design`` (sem o nome do subcomando)."""
        self.validate()
        a = ["--funcao", self.funcao, "--ambiente", self.ambiente, "--n", str(self.n), "--out-dir", self.out_dir,
             "--quality", self.quality]
        if self.funcao in NEEDS_PHOTO:
            a += ["--photo", self.photo]
        if self.estilo:
            a += ["--estilo", self.estilo]
        if self.evento:
            a += ["--evento", self.evento]
        if self.mask and self.funcao == "editar_area":
            a += ["--mask", self.mask]
        if self.funcao == "texto":
            a += ["--formato", self.formato]
        if self.prompt.strip():
            a += ["--prompt", self.prompt.strip()]
        if self.seed is not None:
            a += ["--seed", str(self.seed)]
        return a + ["--json"]


@dataclass
class DesignOutcome:
    """Versões prontas de um pedido (evento ``done`` do motor)."""

    request: DesignRequest
    outputs: list[str]
    before: str | None
    report: str | None
    size: tuple[int, int] | None
    seconds: float

    @property
    def folder(self) -> str:
        return str(Path(self.outputs[0]).parent) if self.outputs else self.request.out_dir


def upscale_args(image: str, out: str) -> list[str]:
    return ["--image", str(image), "--out", str(out), "--json"]


def showcase_args(before: str, afters: list[str], out: str, fmt: str) -> list[str]:
    if fmt not in VIDEO_FORMATS:
        raise ValueError(f"Formato de vídeo inválido: {fmt!r}")
    if not afters or not all(isinstance(a, str) for a in afters):
        raise ValueError("Escolha ao menos uma versão para o vídeo.")
    return ["--before", str(before), "--after", *[str(a) for a in afters], "--out", str(out), "--format", fmt,
            "--json"]


# =============================================================================================== peças

class Stepper(QWidget):
    """Passos do fluxo no topo: número (ou visto), rótulo e filete; clicar volta a um passo já alcançado."""

    step_clicked = Signal(int)
    LABELS = ("Foto", "Função", "Ajustes", "Gerar", "Resultados")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current = 0
        self.reachable = {0}
        self.done: set[int] = set()
        self.setFixedHeight(34)
        self._hit: list[tuple[QRectF, int]] = []
        self.setMouseTracking(True)

    def set_state(self, current: int, reachable: set[int], done: set[int]) -> None:
        self.current, self.reachable, self.done = current, set(reachable), set(done)
        self.update()

    def mousePressEvent(self, e):  # noqa: N802
        for r, i in self._hit:
            if r.contains(e.position()) and i in self.reachable and i != self.current:
                self.step_clicked.emit(i)
                return

    def mouseMoveEvent(self, e):  # noqa: N802
        hand = any(r.contains(e.position()) and i in self.reachable and i != self.current for r, i in self._hit)
        self.setCursor(Qt.PointingHandCursor if hand else Qt.ArrowCursor)

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        f = QFont("Inter"); f.setPixelSize(12); f.setWeight(QFont.Medium)
        fn = QFont("Inter Display"); fn.setPixelSize(11); fn.setWeight(QFont.DemiBold)
        fm = QFontMetrics(f)
        n = len(self.LABELS)
        widths = [26 + 8 + fm.horizontalAdvance(t) for t in self.LABELS]
        gap = max(24.0, (self.width() - sum(widths)) / max(1, n - 1)) if n > 1 else 0
        gap = min(gap, 72.0)
        total = sum(widths) + gap * (n - 1)
        x = (self.width() - total) / 2
        cy = self.height() / 2
        self._hit = []
        for i, (label, w) in enumerate(zip(self.LABELS, widths)):
            c = QPointF(x + 11, cy)
            cur, dn, reach = i == self.current, i in self.done and i != self.current, i in self.reachable
            if cur:
                p.setPen(Qt.NoPen); p.setBrush(QColor(theme.ACCENT)); p.drawEllipse(c, 11, 11)
                p.setPen(QColor(theme.ON_ACCENT))
            elif dn:
                bg = QColor(theme.OK); bg.setAlphaF(0.16)
                p.setPen(Qt.NoPen); p.setBrush(bg); p.drawEllipse(c, 11, 11)
            else:
                p.setPen(QPen(QColor(theme.HAIRLINE_STRONG), 1)); p.setBrush(QColor(theme.SURFACE))
                p.drawEllipse(c, 10.5, 10.5)
                p.setPen(QColor(theme.TEXT_2 if reach else theme.MUTED))
            if dn:
                p.setPen(QPen(QColor(theme.OK), 1.6, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(QPointF(c.x() - 4, c.y() + 0.3), QPointF(c.x() - 1, c.y() + 3.2))
                p.drawLine(QPointF(c.x() - 1, c.y() + 3.2), QPointF(c.x() + 4.4, c.y() - 3))
            else:
                p.setFont(fn)
                p.drawText(QRectF(c.x() - 11, c.y() - 11, 22, 22), Qt.AlignCenter, str(i + 1))
            p.setFont(f)
            p.setPen(QColor(theme.TEXT if cur else theme.TEXT_2 if reach else theme.MUTED))
            tr = QRectF(x + 30, 0, w - 30, self.height())
            p.drawText(tr, Qt.AlignLeft | Qt.AlignVCenter, label)
            self._hit.append((QRectF(x - 4, 0, w + 8, self.height()), i))
            if i < n - 1:
                lc = QColor(theme.OK if i in self.done else theme.HAIRLINE_STRONG)
                if i in self.done:
                    lc.setAlphaF(0.45)
                p.setPen(QPen(lc, 1.2))
                p.drawLine(QPointF(x + w + 10, cy), QPointF(x + w + gap - 10, cy))
            x += w + gap
        p.end()


def _section(text: str) -> QLabel:
    lb = tracked_label(text)
    lb.setContentsMargins(0, 0, 0, 0)
    return lb


def _scroll(inner: QWidget) -> QScrollArea:
    sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame)
    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sc.setWidget(inner)
    sc.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
    return sc


def _button(text: str, icon: str | None = None, kind: str | None = None, icon_color: str = theme.TEXT_2) -> QPushButton:
    b = QPushButton(("  " + text) if icon else text)
    if kind:
        b.setObjectName(kind)
    if icon:
        b.setIcon(theme.line_icon(icon, 16, icon_color)); b.setIconSize(QSize(16, 16))
    b.setCursor(Qt.PointingHandCursor)
    return b


class Page(QFrame):
    """Cartão do painel direito: cabeçalho fixo, corpo rolável e rodapé fixo (botões principais sempre visíveis)."""

    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        self.outer = QVBoxLayout(self); self.outer.setContentsMargins(24, 22, 24, 22); self.outer.setSpacing(0)
        self.head = QVBoxLayout(); self.head.setSpacing(0)
        self.inner = QWidget(); self.inner.setObjectName("pagebody")
        self.body = QVBoxLayout(self.inner); self.body.setContentsMargins(0, 0, 0, 0); self.body.setSpacing(0)
        self.scroll = _scroll(self.inner)
        # com a barra de rolagem visível, o corpo recua para ela não cobrir os cartões; sem ela, o corpo usa a
        # largura toda e fica alinhado com os botões do rodapé
        self.scroll.verticalScrollBar().rangeChanged.connect(
            lambda _lo, hi: self.body.setContentsMargins(0, 0, 14 if hi > 0 else 0, 0))
        self.foot = QVBoxLayout(); self.foot.setSpacing(10)
        self.outer.addLayout(self.head)
        self.outer.addWidget(self.scroll, 1)
        self.outer.addSpacing(14)
        self.outer.addLayout(self.foot)

    def set_compact(self, compact: bool) -> None:
        m = 16 if compact else 24
        self.outer.setContentsMargins(m, m - 2, m, m - 2)


# =============================================================================================== janela

class MainWindow(QMainWindow):
    P_PHOTO, P_FUNC, P_STYLE, P_PROGRESS, P_RESULTS, P_EDIT, P_VIDEO = range(7)
    L_DROP, L_TEXT, L_PROGRESS, L_RESULT, L_MASK = range(5)
    COMPACT_BELOW = 790

    def __init__(self):
        super().__init__()
        theme.load_fonts()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(theme.window_icon())
        self.photo_path: str | None = None
        self.funcao: str | None = None
        self.estilo: str = "moderno"
        self.ambiente: str = "sala"
        self.evento: str = "aniversario"
        self.formato: str = "paisagem"
        self.result: DesignOutcome | None = None
        self.selected = 0
        self.edit_base: str | None = None
        self.last_request: DesignRequest | None = None
        self.last_task: CliResult | None = None       # último trabalho de ação (ampliar / vídeo)
        self.upscaled: dict[str, str] = {}             # versão -> arquivo 4K
        self.videos: list[str] = []
        self.legacy_window = None
        self._back_page = self.P_STYLE
        self._compact: bool | None = None
        self.started_at = 0.0

        self.design = CliTask(self)
        self.design.progress.connect(self._on_progress)
        self.design.preview.connect(self._on_preview)
        self.design.finished.connect(self._on_design_finished)
        self.side = CliTask(self)
        self.side.progress.connect(self._on_side_progress)
        self.side.finished.connect(self._on_side_finished)
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)

        self.modules = ModuleHost(self)                # módulos opcionais (mobilia/modules); sem eles, nada muda
        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root)
        outer = QVBoxLayout(root); self.outer = outer
        outer.setContentsMargins(32, 18, 32, 18); outer.setSpacing(14)
        outer.addLayout(self._build_header())
        self.stepper = Stepper(); self.stepper.step_clicked.connect(self._on_step_clicked)
        view = self.modules.design_layout(outer)       # == outer quando não há módulos
        view.addWidget(self.stepper)
        body = QHBoxLayout(); body.setSpacing(20); view.addLayout(body, 1)
        self.left = QStackedWidget()
        self.left.addWidget(self._build_drop())
        self.left.addWidget(self._build_text_stage())
        self.left.addWidget(self._build_progress_stage())
        self.left.addWidget(self._build_result_stage())
        self.left.addWidget(self._build_mask_stage())
        body.addWidget(self.left, 1)
        self.right = QStackedWidget(); self.right.setFixedWidth(424)
        self.pages: list[Page] = []
        for build in (self._page_photo, self._page_function, self._page_style, self._page_progress,
                      self._page_results, self._page_edit, self._page_video):
            pg = build(); self.pages.append(pg); self.right.addWidget(pg)
        body.addWidget(self.right)

        sb = StatusBar(); self.setStatusBar(sb); sb.setSizeGripEnabled(False); self.sb = sb
        sb.addPermanentWidget(QLabel(f"{APP_NAME}  {__version__}"))
        sb.showMessage("Detectando GPU…")
        self.probe = HardwareProbe(self); self.probe.done.connect(self._on_hardware)
        self.go(self.P_PHOTO)
        self._fit_to_screen()

    # ------------------------------------------------------------------------------------------ montagem
    def _build_header(self) -> QHBoxLayout:
        head = QHBoxLayout(); head.setSpacing(0); head.setContentsMargins(2, 4, 0, 0)
        mark = QLabel(); mark.setPixmap(brand.wordmark_pixmap(30)); mark.setToolTip(APP_NAME)
        mark.setAccessibleName(APP_NAME)
        vr = QFrame(); vr.setObjectName("vrule"); vr.setFixedSize(1, 22)
        tag = QLabel(TAGLINE); tag.setObjectName("tagline")
        lock = QLabel(); lock.setPixmap(theme.line_icon("chip", 14, theme.MUTED).pixmap(14, 14))
        priv = QLabel("Processado no seu computador"); priv.setObjectName("privacy")
        priv.setToolTip("A foto e as imagens não saem do seu PC: tudo é gerado aqui, na sua placa de vídeo.")
        self.btn_legacy = _button("Avançado (vídeo de montagem, legado)", "settings", "ghost")
        self.btn_legacy.setToolTip("Abre o modo antigo: vídeo em que os móveis 3D entram na foto (modos A/B).")
        on_click(self.btn_legacy, self.open_legacy)
        head.addWidget(mark, 0, Qt.AlignVCenter)
        head.addSpacing(22); head.addWidget(vr, 0, Qt.AlignVCenter); head.addSpacing(22)
        self.modules.add_nav(head, tag)
        head.addWidget(tag, 0, Qt.AlignVCenter); head.addStretch(1)
        head.addWidget(self.btn_legacy, 0, Qt.AlignVCenter); head.addSpacing(18)
        head.addWidget(lock, 0, Qt.AlignVCenter); head.addSpacing(8); head.addWidget(priv, 0, Qt.AlignVCenter)
        return head

    def _stage(self) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame(); card.setObjectName("stage")
        lay = QVBoxLayout(card); lay.setContentsMargins(16, 14, 16, 14); lay.setSpacing(10)
        return card, lay

    def _build_drop(self) -> QWidget:
        self.drop = DropZone(self.load_photo, headline="Comece pela foto do ambiente",
                             trust="A foto não sai do seu computador: tudo é gerado aqui.",
                             tip="Vale cômodo vazio ou mobiliado, interno ou fachada. Fotografe o ambiente inteiro, "
                                 "na altura do peito, com luz natural.")
        return self.drop

    def _build_text_stage(self) -> QWidget:
        card, lay = self._stage()
        lay.addStretch(3)
        art = QLabel(); art.setPixmap(theme.function_pixmap("texto", 96, theme.ACCENT)); art.setAlignment(Qt.AlignCenter)
        h = QLabel("Criar a partir de um texto"); h.setObjectName("headline"); h.setAlignment(Qt.AlignCenter)
        b = QLabel("Sem foto: o 7xAi cria o ambiente do zero com o tipo de cômodo, o estilo e a sua descrição.")
        b.setObjectName("body"); b.setAlignment(Qt.AlignCenter); b.setWordWrap(True)
        self.text_summary = QLabel(""); self.text_summary.setObjectName("caption"); self.text_summary.setAlignment(Qt.AlignCenter)
        lay.addWidget(art); lay.addSpacing(22); lay.addWidget(h); lay.addSpacing(10); lay.addWidget(b)
        lay.addSpacing(16); lay.addWidget(self.text_summary)
        lay.addStretch(4)
        return card

    def _build_progress_stage(self) -> QWidget:
        card, lay = self._stage()
        head = QHBoxLayout()
        t = tracked_label("VERSÕES"); t.setStyleSheet(f"color: {theme.ACCENT_HI};")
        self.grid_caption = QLabel(""); self.grid_caption.setObjectName("caption")
        head.addWidget(t); head.addStretch(1); head.addWidget(self.grid_caption)
        self.variants = VariantGrid()
        cap = QLabel("Cada versão aparece aqui assim que fica pronta.")
        cap.setObjectName("stagecap"); cap.setAlignment(Qt.AlignCenter)
        lay.addLayout(head); lay.addWidget(self.variants, 1); lay.addWidget(cap)
        return card

    def _build_result_stage(self) -> QWidget:
        card, lay = self._stage()
        head = QHBoxLayout()
        t = tracked_label("ANTES E DEPOIS"); t.setStyleSheet(f"color: {theme.ACCENT_HI};")
        self.compare_name = QLabel(""); self.compare_name.setObjectName("caption")
        head.addWidget(t); head.addStretch(1); head.addWidget(self.compare_name)
        self.compare = BeforeAfterView()
        self.thumbs = ThumbStrip(); self.thumbs.selected.connect(self.select_variant)
        self.compare_cap = QLabel("Arraste sobre a imagem para comparar com a foto original.")
        self.compare_cap.setObjectName("stagecap"); self.compare_cap.setAlignment(Qt.AlignCenter)
        lay.addLayout(head); lay.addWidget(self.compare, 1); lay.addWidget(self.compare_cap); lay.addWidget(self.thumbs)
        return card

    def _build_mask_stage(self) -> QWidget:
        card, lay = self._stage()
        head = QHBoxLayout()
        t = tracked_label("EDITAR ÁREA"); t.setStyleSheet(f"color: {theme.ACCENT_HI};")
        self.mask_name = QLabel(""); self.mask_name.setObjectName("caption")
        head.addWidget(t); head.addStretch(1); head.addWidget(self.mask_name)
        self.mask = MaskEditor(); self.mask.changed.connect(self._update_edit_ready)
        cap = QLabel("Pinte sobre o que quer trocar. Fora da área pintada, a imagem fica idêntica.")
        cap.setObjectName("stagecap"); cap.setAlignment(Qt.AlignCenter)
        lay.addLayout(head); lay.addWidget(self.mask, 1); lay.addWidget(cap)
        return card

    # --- páginas do painel direito
    def _page_photo(self) -> Page:
        pg = Page()
        self.step_photo = StepHeader("01", "FOTO DO AMBIENTE")
        pg.head.addWidget(self.step_photo); pg.head.addSpacing(16)
        intro = QLabel("Envie a foto de um cômodo, vazio ou mobiliado, ou da fachada. O 7xAi cria versões novas do "
                       "mesmo ambiente: paredes, janelas, portas e perspectiva ficam; móveis, materiais e estilo "
                       "mudam.")
        intro.setObjectName("body"); intro.setWordWrap(True)
        self.btn_import = _button("Importar foto…", "photo")
        on_click(self.btn_import, self.choose_photo)
        self.photo_label = QLabel("Nenhuma foto selecionada"); self.photo_label.setObjectName("hint")
        pg.body.addWidget(intro); pg.body.addSpacing(20)
        pg.body.addWidget(self.btn_import); pg.body.addSpacing(10); pg.body.addWidget(self.photo_label)
        pg.body.addSpacing(22); pg.body.addWidget(rule()); pg.body.addSpacing(18)
        nt = _section("SEM FOTO?")
        nb = QLabel("Descreva o ambiente e o 7xAi cria do zero."); nb.setObjectName("hint"); nb.setWordWrap(True)
        self.btn_text_mode = _button("Criar a partir de um texto", "sparkle", "outline", theme.ACCENT_HI)
        on_click(self.btn_text_mode, self.choose_function, "texto")
        pg.body.addWidget(nt); pg.body.addSpacing(8); pg.body.addWidget(nb); pg.body.addSpacing(12)
        pg.body.addWidget(self.btn_text_mode)
        pg.body.addStretch(1)
        self.photo_ready = QLabel(""); self.photo_ready.setObjectName("hint"); self.photo_ready.setAlignment(Qt.AlignCenter)
        self.btn_photo_next = _button("Continuar", kind="primary")
        on_click(self.btn_photo_next, self.go, self.P_FUNC)
        pg.foot.addWidget(self.photo_ready); pg.foot.addWidget(self.btn_photo_next)
        return pg

    def _page_function(self) -> Page:
        pg = Page()
        pg.head.addWidget(StepHeader("02", "O QUE VOCÊ QUER FAZER")); pg.head.addSpacing(14)
        grid = QGridLayout(); grid.setSpacing(10)
        self.function_group = QButtonGroup(self); self.function_group.setExclusive(True)
        self.function_cards: dict[str, FunctionCard] = {}
        for i, key in enumerate(FUNCTION_ORDER):
            title, desc = FUNCTION_INFO[key]
            c = FunctionCard(key, title, desc)
            on_click(c, self.choose_function, key)
            self.function_group.addButton(c); self.function_cards[key] = c
            grid.addWidget(c, i // 2, i % 2)
        pg.body.addLayout(grid)
        pg.body.addSpacing(12)
        self.func_hint = QLabel(""); self.func_hint.setObjectName("hint"); self.func_hint.setWordWrap(True)
        pg.body.addWidget(self.func_hint)
        pg.body.addStretch(1)
        self.btn_func_back = _button("Trocar a foto", "back", "ghost")
        on_click(self.btn_func_back, self.go, self.P_PHOTO)
        pg.foot.addWidget(self.btn_func_back)
        return pg

    def _page_style(self) -> Page:
        pg = Page()
        self.style_header = StepHeader("03", "AMBIENTE E ESTILO")
        pg.head.addWidget(self.style_header); pg.head.addSpacing(10)
        frow = QHBoxLayout(); frow.setSpacing(8)
        self.func_label = QLabel(""); self.func_label.setObjectName("hint")
        self.btn_change_func = QPushButton("Trocar"); self.btn_change_func.setObjectName("link")
        self.btn_change_func.setCursor(Qt.PointingHandCursor)
        on_click(self.btn_change_func, self.go, self.P_FUNC)
        frow.addWidget(self.func_label); frow.addWidget(self.btn_change_func); frow.addStretch(1)
        pg.head.addLayout(frow); pg.head.addSpacing(14)
        b = pg.body
        # ambiente
        self.room_box = QWidget(); rl = QVBoxLayout(self.room_box); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        rl.addWidget(_section("TIPO DE AMBIENTE")); rl.addSpacing(10)
        rg = QGridLayout(); rg.setSpacing(8)
        self.room_group = QButtonGroup(self); self.room_group.setExclusive(True)
        self.room_chips = {}
        for i, key in enumerate(k for k in CATALOG["ambientes"] if k != "fachada"):
            c = chip(ROOM_SHORT[key], theme.room_icon(key, 24), 24, 62)
            c.setToolTip(CATALOG["ambientes"][key])
            on_click(c, self.select_room, key)
            self.room_group.addButton(c); self.room_chips[key] = c
            rg.addWidget(c, i // 4, i % 4)
        rl.addLayout(rg); rl.addSpacing(20)
        b.addWidget(self.room_box)
        # evento
        self.event_box = QWidget(); el = QVBoxLayout(self.event_box); el.setContentsMargins(0, 0, 0, 0); el.setSpacing(0)
        el.addWidget(_section("EVENTO")); el.addSpacing(10)
        eg = QGridLayout(); eg.setSpacing(8)
        self.event_group = QButtonGroup(self); self.event_group.setExclusive(True)
        self.event_chips = {}
        for i, (key, label) in enumerate(CATALOG["eventos"].items()):
            c = chip(label)
            on_click(c, self.select_event, key)
            self.event_group.addButton(c); self.event_chips[key] = c
            eg.addWidget(c, i // 3, i % 3)
        el.addLayout(eg); el.addSpacing(20)
        b.addWidget(self.event_box)
        # formato (texto)
        self.aspect_box = QWidget(); al = QVBoxLayout(self.aspect_box); al.setContentsMargins(0, 0, 0, 0); al.setSpacing(0)
        al.addWidget(_section("FORMATO DA IMAGEM")); al.addSpacing(10)
        self.aspect_seg = Segmented(list(ASPECTS.items()), "paisagem")
        self.aspect_seg.changed.connect(self.select_aspect)
        al.addWidget(self.aspect_seg); al.addSpacing(20)
        b.addWidget(self.aspect_box)
        # estilo
        self.style_box = QWidget(); sl = QVBoxLayout(self.style_box); sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(0)
        srow = QHBoxLayout(); srow.addWidget(_section("ESTILO")); srow.addStretch(1)
        self.style_name = QLabel(""); self.style_name.setObjectName("hint"); srow.addWidget(self.style_name)
        sl.addLayout(srow); sl.addSpacing(10)
        sg = QGridLayout(); sg.setSpacing(8)
        self.style_group = QButtonGroup(self); self.style_group.setExclusive(True)
        self.style_cards: dict[str, StyleCard] = {}
        for i, (key, label) in enumerate(CATALOG["estilos"].items()):
            c = StyleCard(key, label)
            on_click(c, self.select_style, key)
            self.style_group.addButton(c); self.style_cards[key] = c
            sg.addWidget(c, i // 3, i % 3)
        sl.addLayout(sg); sl.addSpacing(20)
        b.addWidget(self.style_box)
        # descrição
        self.prompt_title = _section("DETALHES (OPCIONAL)")
        self.prompt = QPlainTextEdit(); self.prompt.setFixedHeight(68)
        self.prompt.setPlaceholderText("Ex.: sofá verde-musgo, piso de madeira clara, muitas plantas")
        b.addWidget(self.prompt_title); b.addSpacing(10); b.addWidget(self.prompt)
        b.addStretch(1)
        # rodapé: versões + qualidade + gerar
        vrow = QHBoxLayout(); vrow.setSpacing(12)
        vl = QLabel("Versões"); vl.setObjectName("muted")
        self.n_seg = Segmented([(1, "1"), (2, "2"), (3, "3"), (4, "4")], 4)
        self.n_seg.setFixedWidth(176)
        self.n_seg.changed.connect(lambda *_a: self._update_style_ready())
        self.chk_fast = QCheckBox("Prévia rápida")
        self.chk_fast.setToolTip("Resolução menor e menos passos: bom para testar ideias. Use a qualidade final "
                                 "para entregar.")
        vrow.addWidget(vl); vrow.addWidget(self.n_seg); vrow.addStretch(1); vrow.addWidget(self.chk_fast)
        self.style_ready = QLabel(""); self.style_ready.setObjectName("hint"); self.style_ready.setAlignment(Qt.AlignCenter)
        self.style_ready.setWordWrap(True)
        self.btn_generate = _button("Gerar 4 versões", "sparkle", "primary", theme.ON_ACCENT)
        on_click(self.btn_generate, self.generate)
        pg.foot.addLayout(vrow); pg.foot.addWidget(self.style_ready); pg.foot.addWidget(self.btn_generate)
        return pg

    def _page_progress(self) -> Page:
        pg = Page()
        head = QHBoxLayout()
        self.progress_title = tracked_label("GERANDO AS VERSÕES")
        self.elapsed = QLabel(""); self.elapsed.setObjectName("hint")
        head.addWidget(self.progress_title); head.addStretch(1); head.addWidget(self.elapsed)
        pg.head.addLayout(head); pg.head.addSpacing(18)
        self.pct = QLabel(); self.pct.setObjectName("pct"); self._set_pct(0.0)
        self.bar = QProgressBar(); self.bar.setRange(0, 1000); self.bar.setTextVisible(False)
        self.msg = QLabel("Preparando…"); self.msg.setObjectName("body"); self.msg.setWordWrap(True)
        self.progress_summary = QLabel(""); self.progress_summary.setObjectName("hint"); self.progress_summary.setWordWrap(True)
        self.notice = QLabel(""); self.notice.setObjectName("notice"); self.notice.setWordWrap(True); self.notice.hide()
        self.stages = StageList(); self.stages.set_stages(DESIGN_STAGES)
        b = pg.body
        b.addWidget(self.pct); b.addSpacing(10); b.addWidget(self.bar); b.addSpacing(14); b.addWidget(self.msg)
        b.addSpacing(6); b.addWidget(self.progress_summary)
        b.addSpacing(16); b.addWidget(self.notice)
        b.addSpacing(20); b.addWidget(rule()); b.addSpacing(20)
        b.addWidget(self.stages); b.addStretch(1)
        self.btn_cancel = _button("Cancelar", kind="danger")
        on_click(self.btn_cancel, self.cancel)
        pg.foot.addWidget(self.btn_cancel)
        return pg

    def _page_results(self) -> Page:
        pg = Page()
        head = QHBoxLayout(); head.setSpacing(10)
        badge = QLabel(); badge.setPixmap(theme.status_pixmap("done", 18)); badge.setFixedSize(18, 18)
        s = tracked_label("VERSÕES PRONTAS"); s.setStyleSheet(f"color: {theme.OK};")
        head.addWidget(badge); head.addWidget(s); head.addStretch(1)
        pg.head.addLayout(head); pg.head.addSpacing(12)
        self.result_title = QLabel(""); self.result_title.setObjectName("result"); self.result_title.setWordWrap(True)
        self.result_info = QLabel(""); self.result_info.setObjectName("hint"); self.result_info.setWordWrap(True)
        pg.head.addWidget(self.result_title); pg.head.addSpacing(6); pg.head.addWidget(self.result_info)
        pg.head.addSpacing(18)
        b = pg.body
        self.selected_label = _section("VERSÃO 1 DE 4")
        b.addWidget(self.selected_label); b.addSpacing(12)
        self.btn_save = _button("Salvar imagem…", "save", "primary", theme.ON_ACCENT)
        on_click(self.btn_save, self.save_image)
        b.addWidget(self.btn_save); b.addSpacing(10)
        grid = QGridLayout(); grid.setSpacing(8)
        self.btn_edit = _button("Editar área", "brush")
        self.btn_edit.setToolTip("Pinte uma parte desta versão e troque só ela (ex.: o sofá, o quadro, o piso).")
        on_click(self.btn_edit, self.start_edit_from_result)
        self.btn_upscale = _button("Ampliar 4K", "expand")
        self.btn_upscale.setToolTip("Amplia esta versão para 3840 px no lado maior.")
        on_click(self.btn_upscale, self.upscale_selected)
        self.btn_video = _button("Vídeo antes e depois", "video")
        self.btn_video.setToolTip("Vídeo curto: a foto original se transforma nas versões escolhidas (MP4).")
        on_click(self.btn_video, self.go, self.P_VIDEO)
        self.btn_folder = _button("Abrir pasta", "folder")
        on_click(self.btn_folder, self.open_folder)
        for i, w in enumerate((self.btn_edit, self.btn_upscale, self.btn_video, self.btn_folder)):
            grid.addWidget(w, i // 2, i % 2)
        b.addLayout(grid)
        # trabalho de ação em andamento / concluído (ampliar, vídeo)
        b.addSpacing(16)
        self.taskbox = QFrame(); self.taskbox.setObjectName("taskbox")
        tl = QVBoxLayout(self.taskbox); tl.setContentsMargins(14, 12, 14, 12); tl.setSpacing(8)
        trow = QHBoxLayout()
        self.task_title = QLabel(""); self.task_title.setStyleSheet("font-weight: 600;")
        self.task_pct = QLabel(""); self.task_pct.setObjectName("hint")
        trow.addWidget(self.task_title); trow.addStretch(1); trow.addWidget(self.task_pct)
        self.task_bar = QProgressBar(); self.task_bar.setRange(0, 1000); self.task_bar.setTextVisible(False)
        self.task_msg = QLabel(""); self.task_msg.setObjectName("hint"); self.task_msg.setWordWrap(True)
        arow = QHBoxLayout(); arow.setSpacing(8)
        self.btn_task_open = _button("Abrir", "arrow"); on_click(self.btn_task_open, self.open_task_output)
        self.btn_task_export = _button("Exportar…", "export"); on_click(self.btn_task_export, self.export_task_output)
        self.btn_task_cancel = _button("Cancelar", kind="danger"); on_click(self.btn_task_cancel, self.cancel_side)
        arow.addWidget(self.btn_task_open, 1); arow.addWidget(self.btn_task_export, 1); arow.addWidget(self.btn_task_cancel, 1)
        tl.addLayout(trow); tl.addWidget(self.task_bar); tl.addWidget(self.task_msg); tl.addLayout(arow)
        self.taskbox.hide()
        b.addWidget(self.taskbox)
        b.addSpacing(18); b.addWidget(rule()); b.addSpacing(14)
        b.addWidget(_section("NÃO GOSTOU? TENTE DE NOVO")); b.addSpacing(10)
        row = QHBoxLayout(); row.setSpacing(8)
        self.btn_again = _button("Novas versões", "refresh")
        self.btn_again.setToolTip("Mesmas escolhas, outras sementes: versões diferentes.")
        on_click(self.btn_again, self.generate_again)
        self.btn_restyle = _button("Outro estilo", "back")
        on_click(self.btn_restyle, self.go, self.P_STYLE)
        row.addWidget(self.btn_again, 1); row.addWidget(self.btn_restyle, 1)
        b.addLayout(row)
        b.addStretch(1)
        self.btn_new = _button("Começar com outra foto", "plus", "ghost")
        on_click(self.btn_new, self.new_photo)
        pg.foot.addWidget(self.btn_new)
        return pg

    def _page_edit(self) -> Page:
        pg = Page()
        pg.head.addWidget(StepHeader("03", "EDITAR ÁREA")); pg.head.addSpacing(12)
        b = pg.body
        intro = QLabel("Pinte sobre o que quer trocar e diga o que deve aparecer no lugar.")
        intro.setObjectName("body"); intro.setWordWrap(True)
        b.addWidget(intro); b.addSpacing(18)
        b.addWidget(_section("FERRAMENTA")); b.addSpacing(10)
        self.tool_seg = Segmented([("pincel", "Pincel"), ("borracha", "Borracha")], "pincel")
        self.tool_seg.changed.connect(self.set_tool)
        b.addWidget(self.tool_seg); b.addSpacing(14)
        srow = QHBoxLayout(); srow.setSpacing(12)
        sl = QLabel("Tamanho"); sl.setObjectName("muted")
        self.brush_slider = QSlider(Qt.Horizontal); self.brush_slider.setRange(8, 200); self.brush_slider.setValue(48)
        self.brush_value = QLabel("48 px"); self.brush_value.setObjectName("hint"); self.brush_value.setFixedWidth(48)
        self.brush_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.brush_slider.valueChanged.connect(self.set_brush_size)
        srow.addWidget(sl); srow.addWidget(self.brush_slider, 1); srow.addWidget(self.brush_value)
        b.addLayout(srow); b.addSpacing(12)
        urow = QHBoxLayout(); urow.setSpacing(8)
        self.btn_undo = _button("Desfazer", "undo"); on_click(self.btn_undo, self.mask.undo)
        self.btn_clear = _button("Limpar", "trash"); on_click(self.btn_clear, self.mask.clear)
        urow.addWidget(self.btn_undo, 1); urow.addWidget(self.btn_clear, 1)
        b.addLayout(urow)
        b.addSpacing(20); b.addWidget(rule()); b.addSpacing(18)
        b.addWidget(_section("O QUE COLOCAR NA ÁREA")); b.addSpacing(10)
        self.edit_prompt = QPlainTextEdit(); self.edit_prompt.setFixedHeight(68)
        self.edit_prompt.setPlaceholderText("Ex.: poltrona de couro caramelo; mesa de centro redonda de madeira")
        b.addWidget(self.edit_prompt); b.addSpacing(14)
        b.addWidget(_section("ESTILO")); b.addSpacing(10)
        self.edit_style = QComboBox(); self.edit_style.setView(QListView())
        self.edit_style.addItem("Seguir o estilo da imagem", None)
        for key, label in CATALOG["estilos"].items():
            self.edit_style.addItem(label, key)
        b.addWidget(self.edit_style)
        b.addStretch(1)
        vrow = QHBoxLayout(); vrow.setSpacing(12)
        vl = QLabel("Versões"); vl.setObjectName("muted")
        self.edit_n = Segmented([(1, "1"), (2, "2"), (3, "3"), (4, "4")], 2)
        self.edit_n.setFixedWidth(176)
        vrow.addWidget(vl); vrow.addWidget(self.edit_n); vrow.addStretch(1)
        self.edit_ready = QLabel(""); self.edit_ready.setObjectName("hint"); self.edit_ready.setAlignment(Qt.AlignCenter)
        self.btn_edit_generate = _button("Gerar na área pintada", "sparkle", "primary", theme.ON_ACCENT)
        on_click(self.btn_edit_generate, self.generate_edit)
        self.btn_edit_back = _button("Voltar", "back", "ghost")
        on_click(self.btn_edit_back, self.leave_edit)
        pg.foot.addLayout(vrow); pg.foot.addWidget(self.edit_ready); pg.foot.addWidget(self.btn_edit_generate)
        pg.foot.addWidget(self.btn_edit_back)
        return pg

    def _page_video(self) -> Page:
        pg = Page()
        pg.head.addWidget(StepHeader("05", "VÍDEO ANTES E DEPOIS")); pg.head.addSpacing(12)
        b = pg.body
        intro = QLabel("A foto original se revela nas versões escolhidas, com transições suaves e um leve "
                       "movimento de câmera. MP4 pronto para WhatsApp e Instagram.")
        intro.setObjectName("body"); intro.setWordWrap(True)
        b.addWidget(intro); b.addSpacing(18)
        b.addWidget(_section("FORMATO")); b.addSpacing(10)
        self.format_group = QButtonGroup(self); self.format_group.setExclusive(True)
        self.format_chips = {}
        frow = QHBoxLayout(); frow.setSpacing(8)
        for key, (title, desc) in VIDEO_FORMATS.items():
            c = chip(f"{title}\n{desc.split(' · ')[0]}", height=58)
            c.setAccessibleName(title)
            c.setToolTip(desc)
            on_click(c, self.select_video_format, key)
            self.format_group.addButton(c); self.format_chips[key] = c
            frow.addWidget(c)
        b.addLayout(frow); b.addSpacing(8)
        self.format_desc = QLabel(""); self.format_desc.setObjectName("hint"); self.format_desc.setWordWrap(True)
        b.addWidget(self.format_desc); b.addSpacing(18)
        b.addWidget(_section("VERSÕES NO VÍDEO")); b.addSpacing(10)
        self.video_list = QWidget(); self.video_list_lay = QVBoxLayout(self.video_list)
        self.video_list_lay.setContentsMargins(0, 0, 0, 0); self.video_list_lay.setSpacing(8)
        self.video_checks: list[QCheckBox] = []
        b.addWidget(self.video_list)
        b.addStretch(1)
        self.video_ready = QLabel(""); self.video_ready.setObjectName("hint"); self.video_ready.setAlignment(Qt.AlignCenter)
        self.btn_video_generate = _button("Gerar vídeo", "video", "primary", theme.ON_ACCENT)
        on_click(self.btn_video_generate, self.generate_video)
        self.btn_video_back = _button("Voltar aos resultados", "back", "ghost")
        on_click(self.btn_video_back, self.go, self.P_RESULTS)
        pg.foot.addWidget(self.video_ready); pg.foot.addWidget(self.btn_video_generate); pg.foot.addWidget(self.btn_video_back)
        self.video_format = "16x9"
        self.select_video_format("16x9")
        return pg

    def _set_pct(self, frac: float) -> None:
        self.pct.setText(f'<span style="font-size:60px; font-weight:300;">{frac * 100:.0f}</span>'
                         f'<span style="font-size:34px; font-weight:300; color:{theme.ACCENT};">%</span>')

    # ------------------------------------------------------------------------------------------ tela
    def _fit_to_screen(self) -> None:
        w, h = 1360, 840
        scr = self.screen()
        g = scr.availableGeometry() if scr is not None else None
        if g is not None and g.width() > 0 and g.height() > 0:
            w, h = min(w, g.width() - 32), min(h, g.height() - 48)
        self.set_compact(h < self.COMPACT_BELOW)
        self.resize(max(w, 640), max(h, 480))

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self.set_compact(self.height() < self.COMPACT_BELOW)

    def set_compact(self, compact: bool) -> None:
        """Telas baixas (1366x768): menos respiro e cartões menores; o corpo de cada página rola."""
        if compact == self._compact:
            return
        self._compact = compact
        self.outer.setContentsMargins(*((24, 10, 24, 10) if compact else (32, 18, 32, 18)))
        self.outer.setSpacing(10 if compact else 14)
        self.modules.set_compact(compact)
        for pg in self.pages:
            pg.set_compact(compact)
        for c in list(self.function_cards.values()) + list(self.style_cards.values()):
            c.set_compact(compact)
        self.stages.layout().setSpacing(6 if compact else 10)

    def showEvent(self, e):  # noqa: N802
        super().showEvent(e)
        theme.dark_title_bar(self)
        if not getattr(self, "_probed", False):
            self._probed = True
            self.probe.start()

    def closeEvent(self, e):  # noqa: N802
        for t in (self.design, self.side):
            if t.busy():
                t.cancel()
        super().closeEvent(e)

    # ------------------------------------------------------------------------------------------ navegação
    def page(self) -> int:
        return self.right.currentIndex()

    def go(self, page: int) -> None:
        """Mostra uma página (e o palco correspondente). Páginas que dependem de algo ainda não feito são barradas."""
        if type(page) is not int or not 0 <= page < self.right.count():
            raise ValueError(f"Página inválida: {page!r}")
        if self.design.busy() and page != self.P_PROGRESS:
            page = self.P_PROGRESS
        if page in (self.P_RESULTS, self.P_VIDEO) and self.result is None:
            page = self.P_STYLE if self.funcao else self.P_FUNC
        if page == self.P_STYLE and not self.funcao:
            page = self.P_FUNC
        if page == self.P_STYLE and self.funcao in NEEDS_PHOTO and not self.photo_path:
            page = self.P_PHOTO
        if page == self.P_STYLE and self.funcao == "editar_area":
            page = self.P_EDIT
        if page == self.P_EDIT and not self.edit_base:
            page = self.P_FUNC
        self.right.setCurrentIndex(page)
        if page == self.P_PROGRESS:
            self.left.setCurrentIndex(self.L_PROGRESS)
        elif page in (self.P_RESULTS, self.P_VIDEO):
            self.left.setCurrentIndex(self.L_RESULT)
        elif page == self.P_EDIT:
            self.left.setCurrentIndex(self.L_MASK)
        elif self.funcao == "texto" and page == self.P_STYLE or (page == self.P_FUNC and not self.photo_path):
            self.left.setCurrentIndex(self.L_TEXT if self.funcao == "texto" else self.L_DROP)
        else:
            self.left.setCurrentIndex(self.L_DROP)
        if page == self.P_FUNC:
            self._update_function_page()
        elif page == self.P_STYLE:
            self._update_style_page()
        elif page == self.P_VIDEO:
            self._update_video_page()
        elif page == self.P_RESULTS:
            self._update_result_actions()
        self._update_photo_page()
        self._update_stepper()

    def _update_stepper(self) -> None:
        page = self.page()
        cur = {self.P_PHOTO: 0, self.P_FUNC: 1, self.P_STYLE: 2, self.P_EDIT: 2, self.P_PROGRESS: 3,
               self.P_RESULTS: 4, self.P_VIDEO: 4}[page]
        reach = {0, 1}
        if self.funcao and (self.funcao not in NEEDS_PHOTO or self.photo_path):
            reach.add(2)
        if self.design.busy():
            reach = {3}
        if self.result is not None and not self.design.busy():
            reach.add(4)
        done = set()
        if self.photo_path or self.funcao == "texto":
            done.add(0)
        if self.funcao:
            done.add(1)
        if self.result is not None:
            done |= {2, 3}
        self.stepper.set_state(cur, reach, done)

    def _on_step_clicked(self, index: int) -> None:
        target = {0: self.P_PHOTO, 1: self.P_FUNC, 2: self.P_STYLE, 4: self.P_RESULTS}.get(int(index))
        if target is not None:
            self.go(target)

    # ------------------------------------------------------------------------------------------ passo 1: foto
    def choose_photo(self) -> None:
        self.drop.mousePressEvent(None)

    def load_photo(self, path: str) -> bool:
        if not isinstance(path, str) or not Path(path).is_file() or not self.drop.show_photo(path):
            self.warn("Não foi possível abrir essa imagem.")
            return False
        self.photo_path = path
        self.result = None
        self.upscaled.clear()
        if self.funcao == "texto":
            self.funcao = None
            self.function_group.setExclusive(False)
            for c in self.function_cards.values():
                c.setChecked(False)
            self.function_group.setExclusive(True)
        pm = self.drop.view._pm
        dims = f"   ·   {pm.width()} × {pm.height()} px" if pm is not None else ""
        fm = QFontMetrics(self.photo_label.font())
        self.photo_label.setText(fm.elidedText(Path(path).name, Qt.ElideMiddle, 350 - fm.horizontalAdvance(dims)) + dims)
        self.photo_label.setToolTip(path)
        self.btn_import.setText("  Trocar foto…")
        self.go(self.P_FUNC if self.page() in (self.P_PHOTO, self.P_RESULTS, self.P_VIDEO) else self.page())
        return True

    def _update_photo_page(self) -> None:
        self.step_photo.set_done(bool(self.photo_path))
        self.btn_photo_next.setEnabled(bool(self.photo_path))
        self.photo_ready.setText("Tudo certo. Agora escolha o que fazer com a foto." if self.photo_path else
                                 "Importe uma foto, ou crie um ambiente a partir de um texto.")

    def new_photo(self) -> None:
        self.photo_path = None
        self.result = None
        self.funcao = None
        self.edit_base = None
        self.upscaled.clear()
        self.function_group.setExclusive(False)
        for c in self.function_cards.values():
            c.setChecked(False)
        self.function_group.setExclusive(True)
        self.photo_label.setText("Nenhuma foto selecionada"); self.photo_label.setToolTip("")
        self.btn_import.setText("  Importar foto…")
        self.drop.show_empty()
        self.taskbox.hide()
        self.go(self.P_PHOTO)

    # ------------------------------------------------------------------------------------------ passo 2: função
    def _update_function_page(self) -> None:
        for key, c in self.function_cards.items():
            ok = key not in NEEDS_PHOTO or bool(self.photo_path)
            c.setEnabled(ok)
            c.setChecked(key == self.funcao)
        self.func_hint.setText("" if self.photo_path else
                               "Para as funções sobre a foto, importe a foto primeiro. Sem foto, use "
                               "“Texto para ambiente”.")

    def choose_function(self, key: str) -> None:
        if not isinstance(key, str) or key not in CATALOG["funcoes"]:
            raise ValueError(f"Função inválida: {key!r}")
        if key in NEEDS_PHOTO and not self.photo_path:
            self.go(self.P_PHOTO)
            self.statusBar().showMessage("Importe a foto do ambiente primeiro.", 6000)
            return
        self.funcao = key
        self.function_cards[key].setChecked(True)
        if key == "fachada":
            self.ambiente = "fachada"
        elif self.ambiente == "fachada":
            self.ambiente = "sala"
        if key == "editar_area":
            self.start_edit(self.photo_path, back=self.P_FUNC)
            return
        self.go(self.P_STYLE)

    # ------------------------------------------------------------------------------------------ passo 3: ajustes
    def _update_style_page(self) -> None:
        f = self.funcao
        self.func_label.setText(f"Função: {CATALOG['funcoes'].get(f, '')}")
        self.room_box.setVisible(f != "fachada")
        self.event_box.setVisible(f == "evento")
        self.style_box.setVisible(f != "evento")
        self.aspect_box.setVisible(f == "texto")
        self.style_header.set_title("AMBIENTE E EVENTO" if f == "evento" else
                                    "ESTILO DA FACHADA" if f == "fachada" else "AMBIENTE E ESTILO")
        if f == "texto":
            self.prompt_title.setText("DESCRIÇÃO DO AMBIENTE")
            self.prompt.setPlaceholderText("Ex.: quarto de casal amplo, cabeceira de palhinha, luz de fim de tarde, "
                                           "janela grande com vista para árvores")
        elif f == "evento":
            self.prompt_title.setText("DETALHES DA DECORAÇÃO (OPCIONAL)")
            self.prompt.setPlaceholderText("Ex.: balões dourados e brancos, mesa de doces, painel com flores")
        else:
            self.prompt_title.setText("DETALHES (OPCIONAL)")
            self.prompt.setPlaceholderText("Ex.: sofá verde-musgo, piso de madeira clara, muitas plantas")
        if self.ambiente in self.room_chips:
            self.room_chips[self.ambiente].setChecked(True)
        self.event_chips[self.evento].setChecked(True)
        self.style_cards[self.estilo].setChecked(True)
        self.aspect_seg.set_value(self.formato)
        self.text_summary.setText(self._summary())
        self._update_style_ready()

    def _summary(self) -> str:
        parts = [CATALOG["ambientes"].get(self.ambiente, "")]
        if self.funcao == "evento":
            parts.append(CATALOG["eventos"][self.evento])
        else:
            parts.append(CATALOG["estilos"][self.estilo])
        if self.funcao == "texto":
            parts.append(ASPECTS[self.formato])
        return "  ·  ".join(p for p in parts if p)

    def _update_style_ready(self) -> None:
        n = self.n_seg.value()
        self.btn_generate.setText(f"  Gerar {n} {'versão' if n == 1 else 'versões'}")
        busy = self.design.busy()
        ok = bool(self.funcao) and (self.funcao not in NEEDS_PHOTO or bool(self.photo_path))
        if self.funcao == "texto" and not self.prompt.toPlainText().strip():
            hint = "Dica: descreva o ambiente que você imagina."
        else:
            hint = f"{self._summary()}  ·  gerado aqui, na sua GPU"
        self.style_ready.setText(hint)
        self.style_name.setText(CATALOG["estilos"].get(self.estilo, ""))
        self.style_name.setToolTip(STYLE_HINTS.get(self.estilo, ""))
        self.btn_generate.setEnabled(ok and not busy)

    def select_room(self, key: str) -> None:
        if not isinstance(key, str) or key not in CATALOG["ambientes"]:
            raise ValueError(f"Ambiente inválido: {key!r}")
        self.ambiente = key
        if key in self.room_chips:
            self.room_chips[key].setChecked(True)
        self.text_summary.setText(self._summary())
        self._update_style_ready()

    def select_style(self, key: str) -> None:
        if not isinstance(key, str) or key not in CATALOG["estilos"]:
            raise ValueError(f"Estilo inválido: {key!r}")
        self.estilo = key
        self.style_cards[key].setChecked(True)
        self.text_summary.setText(self._summary())
        self._update_style_ready()

    def select_event(self, key: str) -> None:
        if not isinstance(key, str) or key not in CATALOG["eventos"]:
            raise ValueError(f"Evento inválido: {key!r}")
        self.evento = key
        self.event_chips[key].setChecked(True)
        self._update_style_ready()

    def select_aspect(self, key) -> None:
        if key not in ASPECTS:
            raise ValueError(f"Formato inválido: {key!r}")
        self.formato = key
        self.aspect_seg.set_value(key)
        self.text_summary.setText(self._summary())
        self._update_style_ready()

    def set_variations(self, n: int) -> None:
        if type(n) is not int or not 1 <= n <= 4:
            raise ValueError(f"Número de versões inválido: {n!r}")
        self.n_seg.set_value(n)
        self._update_style_ready()

    # ------------------------------------------------------------------------------------------ passo 4: gerar
    def _new_out_dir(self, tag: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = runtime.runtime_paths().jobs_dir
        d = base / f"design_{stamp}_{tag}"
        k = 2
        while d.exists():
            d = base / f"design_{stamp}_{tag}_{k}"; k += 1
        return str(d)

    def build_request(self) -> DesignRequest:
        f = self.funcao
        estilo = None if f == "evento" else self.estilo
        tag = f if f != "evento" else f"evento_{self.evento}"
        return DesignRequest(funcao=f, estilo=estilo, ambiente="fachada" if f == "fachada" else self.ambiente,
                             n=int(self.n_seg.value()), out_dir=self._new_out_dir(f"{tag}_{estilo or ''}".rstrip("_")),
                             photo=self.photo_path if f in NEEDS_PHOTO else None,
                             prompt=self.prompt.toPlainText(), evento=self.evento if f == "evento" else None,
                             formato=self.formato, quality="rapido" if self.chk_fast.isChecked() else "final")

    def generate(self) -> None:
        """Gera as versões com as escolhas da tela (sem parâmetros: nada do sinal do botão entra no pedido)."""
        if self.design.busy() or not self.funcao or self.funcao == "editar_area":
            return
        try:
            req = self.build_request().validate()
        except ValueError as e:
            self.warn(str(e))
            return
        self._back_page = self.P_STYLE
        self.submit(req)

    def generate_again(self) -> None:
        if self.last_request is None or self.design.busy():
            return
        r = self.last_request
        req = DesignRequest(**{**r.__dict__, "seed": None, "out_dir": self._new_out_dir(Path(r.out_dir).name.split("_", 3)[-1])})
        if r.funcao == "editar_area":
            self._back_page = self.P_EDIT
        self.submit(req)

    def submit(self, req: DesignRequest) -> None:
        req.validate()
        self.last_request = req
        self.variants.reset(req.n)
        texto = req.funcao == "texto"
        self.stages.set_stages(TEXT_STAGES if texto else DESIGN_STAGES)
        self.stages.reset()
        self.bar.setValue(0); self._set_pct(0.0); self.msg.setText("Iniciando o motor…")
        self.elapsed.setText("")
        n = req.n
        what = CATALOG["eventos"][req.evento] if req.evento else (CATALOG["estilos"].get(req.estilo or "", "")
                                                                  if req.funcao != "editar_area" else "Área pintada")
        self.progress_title.setText("GERANDO A VERSÃO" if n == 1 else f"GERANDO {n} VERSÕES")
        self.progress_summary.setText(f"{CATALOG['funcoes'][req.funcao]}  ·  {what}  ·  "
                                      f"{CATALOG['ambientes'][req.ambiente]}")
        self.grid_caption.setText(f"0 de {n} prontas")
        notice = self.first_run_notice()
        self.notice.setText(notice or ""); self.notice.setVisible(bool(notice))
        self.btn_cancel.setEnabled(True)
        self.started_at = time.monotonic(); self.timer.start(1000)
        self.design.start("design", req.args())
        self.go(self.P_PROGRESS)

    @staticmethod
    def first_run_notice() -> str | None:
        """Primeira geração por IA neste PC: os modelos são carregados do disco pela primeira vez."""
        try:
            jobs = runtime.runtime_paths().jobs_dir
            if jobs.is_dir() and any(jobs.glob("design_*/design.json")):
                return None
            st = bootstrap.load_state(runtime.runtime_paths().state_file)
        except Exception:  # noqa: BLE001 — aviso cosmético
            return None
        if not st.get("done"):
            return None
        return ("Primeira geração neste computador: os modelos de IA ainda vão ser carregados pela primeira vez "
                "e isso pode levar alguns minutos. As próximas são bem mais rápidas.")

    def cancel(self) -> None:
        if self.design.busy():
            self.btn_cancel.setEnabled(False)
            self.msg.setText("Cancelando…")
            self.design.cancel()

    def _tick(self) -> None:
        s = int(time.monotonic() - self.started_at)
        self.elapsed.setText(f"{s // 60:d} min {s % 60:02d} s")

    def _on_progress(self, stage: str, frac: float, message: str) -> None:
        self.bar.setValue(int(frac * 1000)); self._set_pct(frac)
        if message:
            self.msg.setText(message)
        self.stages.set_current(stage)

    def _on_preview(self, path: str, index: int) -> None:
        if path and Path(path).exists():
            self.variants.set_image(index, path)
            n = len(self.variants.slots)
            self.grid_caption.setText(f"{self.variants.filled()} de {n} prontas")

    def _on_design_finished(self, res: CliResult) -> None:
        self.timer.stop()
        self.btn_cancel.setEnabled(True)
        req = self.last_request
        outs = [o for o in res.outputs if Path(o).is_file()]
        if res.status == "pronto" and outs and req is not None:
            before = res.done.get("before")
            before = before if before and Path(before).is_file() else None
            if before is None and req.funcao in NEEDS_PHOTO:
                before = req.photo if req.funcao != "editar_area" else self.edit_base
            size = res.done.get("size")
            self.result = DesignOutcome(req, outs, before, res.done.get("report"),
                                        tuple(size) if isinstance(size, (list, tuple)) and len(size) == 2 else None,
                                        (res.finished_at or time.time()) - res.started_at)
            self.upscaled = {}
            self.taskbox.hide()
            self._show_result()
            return
        self.go(self._back_page)
        if res.status == "cancelado":
            self.statusBar().showMessage("Geração cancelada.", 6000)
            return
        if res.status == "pronto":
            self.show_error("O motor terminou sem entregar nenhuma imagem.", "\n".join(res.log[-40:]))
            return
        self.show_error(res.error or "Erro desconhecido.", res.detail)

    # ------------------------------------------------------------------------------------------ passo 5: resultados
    def _show_result(self) -> None:
        r = self.result
        req = r.request
        if req.funcao == "editar_area":
            title = "Área editada"
        elif req.funcao == "evento":
            title = CATALOG["eventos"][req.evento]
        else:
            title = CATALOG["estilos"].get(req.estilo or "", "")
        room = CATALOG["ambientes"][req.ambiente]
        self.result_title.setText(f"{title} · {room}" if req.funcao != "fachada" else f"{title} · Fachada")
        took = int(round(r.seconds))
        took_s = f"{took // 60} min {took % 60:02d} s" if took >= 60 else f"{took} s"
        dims = f"{r.size[0]} × {r.size[1]} px" if r.size else ""
        k = len(r.outputs)
        info = [CATALOG["funcoes"][req.funcao], f"{k} {'versão' if k == 1 else 'versões'}"] + ([dims] if dims else [])
        if took >= 1:
            info.append(f"em {took_s}")
        self.result_info.setText("  ·  ".join(info))
        self.thumbs.set_items(r.outputs)
        self.compare.after_label = "DEPOIS"
        self.compare_cap.setText("Arraste sobre a imagem para comparar com a foto original." if r.before else
                                 "Criado a partir do texto: não há foto “antes” para comparar.")
        self.select_variant(0)
        self.compare.set_split(0.5)
        self.go(self.P_RESULTS)

    def select_variant(self, index: int) -> None:
        if self.result is None or type(index) is not int or not 0 <= index < len(self.result.outputs):
            return
        self.selected = index
        path = self.result.outputs[index]
        self.thumbs.select(index)
        self.compare.set_images(self.result.before, path)
        k = len(self.result.outputs)
        self.selected_label.setText(f"VERSÃO {index + 1} DE {k}")
        self.compare_name.setText(Path(path).name)
        self._update_result_actions()

    def selected_output(self) -> str | None:
        if self.result is None:
            return None
        return self.result.outputs[self.selected]

    def _update_result_actions(self) -> None:
        busy = self.side.busy()
        has = self.result is not None
        self.btn_upscale.setEnabled(has and not busy)
        self.btn_video.setEnabled(has and not busy and bool(self.result and self.result.before))
        self.btn_video.setToolTip("Vídeo curto: a foto original se transforma nas versões escolhidas (MP4)."
                                  if not has or self.result.before else
                                  "Sem foto original (criado do texto): não há “antes” para o vídeo.")
        self.btn_edit.setEnabled(has)
        self.btn_again.setEnabled(has and not self.design.busy())
        up = self.upscaled.get(self.selected_output() or "")
        self.btn_upscale.setText("  4K pronto" if up else "  Ampliar 4K")
        self.btn_save.setText("  Salvar imagem 4K…" if up else "  Salvar imagem…")
        if has:
            for i, o in enumerate(self.result.outputs):
                self.thumbs.set_badge(i, "4K" if o in self.upscaled else "")

    def ask_save_path(self, suggested: str, filter_: str) -> str:
        dest, _ = QFileDialog.getSaveFileName(self, "Salvar", suggested, filter_)
        return dest

    def save_image(self) -> str | None:
        src = self.selected_output()
        if not src:
            return None
        src = self.upscaled.get(src, src)
        dest = self.ask_save_path(str(Path.home() / Path(src).name), "Imagem JPEG (*.jpg *.jpeg);;Imagem PNG (*.png)")
        if not dest:
            return None
        try:
            if Path(dest).suffix.lower() == ".png":
                from PySide6.QtGui import QImage
                if not QImage(src).save(dest, "PNG"):
                    raise OSError("falha ao gravar o PNG")
            else:
                shutil.copyfile(src, dest)
        except OSError as e:
            self.warn(f"Não foi possível salvar a imagem: {e}")
            return None
        self.statusBar().showMessage(f"Imagem salva: {dest}", 8000)
        return dest

    def open_url(self, path: str) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_folder(self) -> None:
        if self.result is not None:
            self.open_url(self.result.folder)

    # --- editar área
    def start_edit_from_result(self) -> None:
        src = self.selected_output()
        if src:
            self.start_edit(src, back=self.P_RESULTS)

    def start_edit(self, base: str | None, back: int | None = None) -> None:
        if not base or not Path(base).is_file() or not self.mask.set_image(base):
            self.warn("Não foi possível abrir a imagem para editar.")
            return
        self.edit_base = base
        self._edit_back = back if back is not None else self.P_FUNC
        self.mask_name.setText(Path(base).name)
        self.tool_seg.set_value("pincel"); self.set_tool("pincel")
        self.edit_prompt.setPlainText("")
        self.go(self.P_EDIT)
        self._update_edit_ready()

    def leave_edit(self) -> None:
        back = getattr(self, "_edit_back", self.P_FUNC)
        if back == self.P_FUNC and self.funcao == "editar_area":
            self.funcao = None
        self.edit_base = None
        self.go(back)

    def set_tool(self, tool) -> None:
        if tool not in ("pincel", "borracha"):
            raise ValueError(f"Ferramenta inválida: {tool!r}")
        self.mask.set_erasing(tool == "borracha")
        self.mask.update()

    def set_brush_size(self, px: int) -> None:
        self.mask.set_brush(int(px))
        self.brush_value.setText(f"{int(px)} px")

    def _update_edit_ready(self) -> None:
        has = self.mask.has_mask()
        self.btn_edit_generate.setEnabled(has and not self.design.busy())
        self.btn_undo.setEnabled(self.mask.can_undo())
        self.btn_clear.setEnabled(has)
        self.edit_ready.setText("Pronto: a área pintada será refeita." if has else
                                "Pinte a área que quer trocar na imagem ao lado.")

    def generate_edit(self) -> None:
        if self.design.busy() or not self.edit_base or not self.mask.has_mask():
            return
        out_dir = self._new_out_dir("editar_area")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        mask_path = str(Path(out_dir) / "mascara.png")
        if not self.mask.save_mask(mask_path):
            self.warn("Não foi possível gravar a máscara.")
            return
        req = DesignRequest(funcao="editar_area", estilo=self.edit_style.currentData(), ambiente=self.ambiente
                            if self.ambiente in CATALOG["ambientes"] else "sala",
                            n=int(self.edit_n.value()), out_dir=out_dir, photo=self.edit_base,
                            prompt=self.edit_prompt.toPlainText(), mask=mask_path,
                            quality="rapido" if self.chk_fast.isChecked() else "final")
        try:
            req.validate()
        except ValueError as e:
            self.warn(str(e))
            return
        self._back_page = self.P_EDIT
        self.submit(req)

    # --- ampliar 4K / vídeo (trabalhos de ação)
    def upscale_selected(self) -> None:
        src = self.selected_output()
        if not src or self.side.busy():
            return
        if src in self.upscaled and Path(self.upscaled[src]).is_file():
            self._show_task_done("Imagem 4K pronta", self.upscaled[src], "Já ampliada: use “Salvar imagem 4K…”.")
            return
        out = str(Path(src).with_name(Path(src).stem + "_4k.jpg"))
        self._start_side("upscale", upscale_args(src, out), "Ampliando para 4K", tag=src)

    def _update_video_page(self) -> None:
        for c in self.video_checks:
            c.setParent(None); c.deleteLater()
        self.video_checks = []
        if self.result is None:
            return
        for i, o in enumerate(self.result.outputs):
            c = QCheckBox(f"Versão {i + 1}  ·  {Path(o).name}")
            c.setChecked(True)
            c.setIcon(theme_icon_from(o)); c.setIconSize(QSize(54, 36))
            c.toggled.connect(lambda *_a: self._update_video_ready())
            self.video_checks.append(c); self.video_list_lay.addWidget(c)
        self.format_chips[self.video_format].setChecked(True)
        self._update_video_ready()

    def select_video_format(self, key: str) -> None:
        if key not in VIDEO_FORMATS:
            raise ValueError(f"Formato de vídeo inválido: {key!r}")
        self.video_format = key
        self.format_chips[key].setChecked(True)
        self.format_desc.setText("Ideal para " + VIDEO_FORMATS[key][1].split(" · ", 1)[1] + ".")

    def video_selection(self) -> list[str]:
        if self.result is None:
            return []
        return [o for o, c in zip(self.result.outputs, self.video_checks) if c.isChecked()]

    def _update_video_ready(self) -> None:
        sel = self.video_selection()
        ok = bool(sel) and self.result is not None and bool(self.result.before) and not self.side.busy()
        self.btn_video_generate.setEnabled(ok)
        k = len(sel)
        secs = min(20, max(10, 4 + 3 * k))
        self.video_ready.setText(f"{k} {'versão' if k == 1 else 'versões'}  ·  cerca de {secs} s de vídeo" if k else
                                 "Escolha ao menos uma versão.")

    def generate_video(self) -> None:
        if self.result is None or self.side.busy() or not self.result.before:
            return
        sel = self.video_selection()
        if not sel:
            return
        out = str(Path(self.result.folder) / f"antes_depois_{self.video_format}.mp4")
        try:
            args = showcase_args(self.result.before, sel, out, self.video_format)
        except ValueError as e:
            self.warn(str(e))
            return
        self._start_side("showcase", args, "Gerando o vídeo antes e depois")
        self.go(self.P_RESULTS)

    def _start_side(self, kind: str, args: list[str], title: str, tag: object = None) -> None:
        self.side.start(kind, args, tag=tag)
        self.task_title.setText(title); self.task_pct.setText("0%"); self.task_bar.setValue(0); self.task_bar.show()
        self.task_msg.setText("Iniciando o motor…")
        self.btn_task_open.hide(); self.btn_task_export.hide(); self.btn_task_cancel.show(); self.btn_task_cancel.setEnabled(True)
        self.taskbox.show()
        self._update_result_actions()

    def cancel_side(self) -> None:
        if self.side.busy():
            self.btn_task_cancel.setEnabled(False)
            self.task_msg.setText("Cancelando…")
            self.side.cancel()

    def _on_side_progress(self, stage: str, frac: float, message: str) -> None:
        self.task_bar.setValue(int(frac * 1000)); self.task_pct.setText(f"{frac * 100:.0f}%")
        if message:
            self.task_msg.setText(message)

    def _show_task_done(self, title: str, path: str, msg: str) -> None:
        self.task_title.setText(title); self.task_pct.setText("")
        self.task_bar.hide(); self.task_msg.setText(msg)
        self.btn_task_cancel.hide()
        self.btn_task_open.show(); self.btn_task_export.show()
        self._task_output = path
        self.taskbox.show()

    def _on_side_finished(self, res: CliResult) -> None:
        self.last_task = res
        out = res.output
        if res.status == "pronto" and out and Path(out).is_file():
            if res.kind == "upscale":
                if isinstance(res.tag, str):
                    self.upscaled[res.tag] = out
                size = res.done.get("size")
                dims = f"{size[0]} × {size[1]} px" if isinstance(size, (list, tuple)) and len(size) == 2 else ""
                self._show_task_done("Imagem 4K pronta", out, "  ·  ".join(x for x in (Path(out).name, dims) if x))
                self.btn_task_open.setText("  Abrir imagem")
            else:
                self.videos.append(out)
                sz = human_size(Path(out).stat().st_size)
                dur = res.done.get("duration_s")
                dur = f"{float(dur):.0f} s" if isinstance(dur, (int, float)) and dur > 0 else ""
                self._show_task_done("Vídeo pronto", out, "  ·  ".join(x for x in (Path(out).name, dur, sz) if x))
                self.btn_task_open.setText("  Assistir")
            self.statusBar().showMessage(f"Pronto: {Path(out).name}", 10000)
            self.statusBar().setToolTip(out)
        else:
            self.taskbox.hide()
            if res.status == "cancelado":
                self.statusBar().showMessage("Cancelado.", 6000)
            else:
                what = "ampliar a imagem" if res.kind == "upscale" else "gerar o vídeo"
                self.show_error(res.error or "O motor terminou sem entregar o arquivo.", res.detail,
                                title=f"Não foi possível {what}.")
        self._update_result_actions()
        if self.page() == self.P_VIDEO:
            self._update_video_ready()

    def open_task_output(self) -> None:
        p = getattr(self, "_task_output", None)
        if p:
            self.open_url(p)

    def export_task_output(self) -> str | None:
        p = getattr(self, "_task_output", None)
        if not p:
            return None
        flt = "Vídeo MP4 (*.mp4)" if p.lower().endswith(".mp4") else "Imagem JPEG (*.jpg *.jpeg)"
        dest = self.ask_save_path(str(Path.home() / Path(p).name), flt)
        if not dest:
            return None
        try:
            shutil.copyfile(p, dest)
        except OSError as e:
            self.warn(f"Não foi possível exportar: {e}")
            return None
        self.statusBar().showMessage(f"Exportado: {dest}", 8000)
        return dest

    # ------------------------------------------------------------------------------------------ legado
    def open_legacy(self) -> None:
        """Abre o vídeo de montagem antigo (modos A/B) numa janela própria, com a foto atual já carregada."""
        from .legacy import LegacyWindow

        if self.legacy_window is None:
            self.legacy_window = LegacyWindow()
        w = self.legacy_window
        if self.photo_path and w.photo_path != self.photo_path:
            w.load_photo(self.photo_path)
        w.show(); w.raise_(); w.activateWindow()
        return w

    # ------------------------------------------------------------------------------------------ avisos
    def warn(self, message: str) -> None:
        QMessageBox.warning(self, APP_NAME, message)

    def show_error(self, message: str, detail: str = "", title: str = "Não foi possível gerar as versões.") -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(APP_NAME)
        box.setText(title)
        box.setInformativeText(message)
        if detail:
            box.setDetailedText(detail)
        box.addButton(QMessageBox.Ok)
        box.exec()

    def _on_hardware(self, info) -> None:
        if not info:
            self.sb.set_dot(theme.DANGER)
            self.statusBar().showMessage("GPU: não detectada (o motor ainda não está instalado ou não respondeu).")
            return
        gpus = info.get("gpus") or []
        gpu = max(gpus, key=lambda g: g.get("vram_total_mb", 0)) if gpus else None
        g = f"{gpu['name']} · {gpu['vram_total_mb'] / 1024:.0f} GB" if gpu else "sem GPU NVIDIA (modo CPU, bem mais lento)"
        self.statusBar().showMessage(f"GPU: {g}")
        self.sb.set_dot(theme.OK if gpu else theme.ACCENT)
        if gpu is None:
            self.chk_fast.setChecked(True)


def theme_icon_from(path: str):
    from PySide6.QtGui import QIcon, QPixmap
    return QIcon(cover_pixmap(QPixmap(path), 54, 36, 5, 2.0))


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
