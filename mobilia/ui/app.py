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
from .updater_dialog import attach_updater
from .widgets import (PHOTO_FILTER, Backdrop, DropZone, ImageView, StatusBar, StepHeader, human_size, on_click,
                      paint_chip, rule, tracked_label)

__all__ = ["MainWindow", "DesignRequest", "DesignOutcome", "CATALOG", "main", "upscale_args", "showcase_args",
           # reexportados (código antigo importava daqui)
           "PHOTO_FILTER", "DropZone", "ImageView", "StatusBar", "StepHeader", "human_size", "paint_chip"]

TAGLINE = "Redesenho de ambientes por IA, a partir de uma única foto"

# Mesmas chaves e rótulos de ``mobilia.design.presets.catalog()`` (conferido em tests/test_ui.py). A interface não
# importa o pacote do motor: ele traz numpy/OpenCV/PyTorch, que ficam fora do executável da interface.
CATALOG: dict[str, dict[str, str]] = {
    "funcoes": {"mudar_estilo": "Mudar estilo", "reforma": "Reforma completa", "redesign_simples": "Redesign simples",
                "mobiliar": "Mobiliar ambiente vazio", "editar_area": "Editar área", "fachada": "Fachada",
                "evento": "Decoração de evento", "texto": "Texto para ambiente", "melhorar": "Melhorar foto",
                "esvaziar": "Esvaziar ambiente"},
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
# "Vídeo: móveis entrando" (modo B a partir do "Esvaziar ambiente"): o formato da foto ou um recorte
FURNITURE_FORMATS = {"original": ("Formato da foto", "Sem recorte · o vídeo termina exatamente na sua foto"),
                     **VIDEO_FORMATS}
# ambientes da interface -> tipos de ambiente do motor de vídeo (``jobs.ROOM_TYPES``)
VIDEO_ROOMS = {"sala": "sala", "quarto": "quarto", "cozinha": "cozinha", "banheiro": "banheiro",
               "escritorio": "escritorio", "sala_jantar": "sala_jantar", "varanda": "sala", "fachada": "sala"}
# funções sem estilo (o "Esvaziar ambiente" só pede o tipo de ambiente)
NO_STYLE = ("evento", "melhorar", "esvaziar")
DESIGN_STAGES = [("preparo", "Preparando a foto"), ("analise", "Analisando o ambiente"),
                 ("geracao", "Gerando as versões"), ("final", "Finalizando")]
TEXT_STAGES = [("preparo", "Preparando"), ("geracao", "Gerando as versões"), ("final", "Finalizando")]
ENHANCE_STAGES = [("preparo", "Preparando a foto"), ("analise", "Melhorando a foto"), ("final", "Finalizando")]
# "Melhorar foto": etapas que o usuário pode desligar (chave -> (rótulo, dica)); todas ligadas por padrão
ENHANCE_STEPS: dict[str, tuple[str, str]] = {
    "exposicao": ("Luz e janelas (HDR natural)", "Clareia as sombras e recupera as janelas, sem cara de filtro."),
    "branco": ("Balanço de branco", "Tira o amarelado da lâmpada e o azulado da sombra (teto e paredes brancos)."),
    "verticais": ("Endireitar verticais", "Paredes, portas e armários retos; recorta um pouco as bordas."),
    "ceu": ("Céu azul", "Troca céu nublado ou estourado por um céu azul limpo (fachadas e varandas)."),
    "nitidez": ("Nitidez e menos ruído", "Detalhe mais definido e menos granulado de celular."),
}
ENHANCE_FLAGS = tuple(f"sem_{k}" for k in ENHANCE_STEPS) + ("ampliar_4k",)
# funções que podem rodar em lote (as que precisam de pintura ou não usam foto ficam fora)
BATCH_FUNCS = ("melhorar", "mudar_estilo", "redesign_simples", "reforma", "mobiliar", "fachada")
EXPORT_SIDES = {"2048": "2048 px (WhatsApp e portais)", "4k": "4K (3840 px)"}
ILLUSTRATIVE = {k for k in CATALOG["funcoes"] if k != "melhorar"}   # == presets.ILLUSTRATIVE (tests/test_ui.py)


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
    opcoes: tuple = ()                # "Melhorar foto": etapas desligadas (``sem_ceu``...) e ``ampliar_4k``

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
        if self.funcao == "melhorar" and self.mask is not None:
            check(isinstance(self.mask, str) and Path(self.mask).is_file(), "Máscara de remoção inválida.")
        check(isinstance(self.opcoes, (tuple, list)) and all(o in ENHANCE_FLAGS for o in self.opcoes),
              f"Opções inválidas: {self.opcoes!r}")
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
        if self.mask and self.funcao in ("editar_area", "melhorar"):
            a += ["--mask", self.mask]
        if self.funcao == "melhorar":
            a += ["--" + o.replace("_", "-") for o in self.opcoes]
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
    aviso: str | None = None          # o motor mudou algo por conta própria (ex.: ambiente vazio -> mobiliado)
    funcao: str | None = None         # a função aplicada (pode diferir da pedida)

    @property
    def folder(self) -> str:
        return str(Path(self.outputs[0]).parent) if self.outputs else self.request.out_dir


def export_args(images: list[str], out_dir: str, lado: str, rotulo: bool) -> list[str]:
    if lado not in EXPORT_SIDES:
        raise ValueError(f"Tamanho inválido: {lado!r}")
    if not images or not all(isinstance(i, str) for i in images):
        raise ValueError("Nada para exportar.")
    return ["--images", *images, "--out-dir", str(out_dir), "--lado", lado,
            "--rotulo" if rotulo else "--sem-rotulo", "--json"]


@dataclass
class BatchRequest:
    """Pedido de lote (``mobilia.cli design-lote``): várias fotos do mesmo imóvel, a mesma função."""

    photos: list
    imovel: str
    funcao: str
    out_dir: str
    estilo: str | None = None
    ambiente: str = "sala"
    opcoes: tuple = ()
    exportar: str | None = None        # None | "2048" | "4k"
    rotulo: bool = False

    def validate(self) -> "BatchRequest":
        if not isinstance(self.photos, list) or not self.photos or not all(
                isinstance(p, str) and Path(p).is_file() for p in self.photos):
            raise ValueError("Adicione as fotos do imóvel.")
        if self.funcao not in BATCH_FUNCS:
            raise ValueError(f"Função inválida para lote: {self.funcao!r}")
        if self.funcao != "melhorar" and self.estilo not in CATALOG["estilos"]:
            raise ValueError("Escolha um estilo.")
        if self.ambiente not in CATALOG["ambientes"]:
            raise ValueError(f"Ambiente inválido: {self.ambiente!r}")
        if not all(o in ENHANCE_FLAGS for o in self.opcoes):
            raise ValueError(f"Opções inválidas: {self.opcoes!r}")
        if self.exportar is not None and self.exportar not in EXPORT_SIDES:
            raise ValueError(f"Tamanho inválido: {self.exportar!r}")
        if not isinstance(self.out_dir, str) or not self.out_dir:
            raise ValueError("Pasta de saída inválida")
        return self

    def args(self) -> list[str]:
        self.validate()
        a = ["--fotos", *self.photos, "--imovel", self.imovel or "", "--funcao", self.funcao,
             "--ambiente", self.ambiente, "--out-dir", self.out_dir]
        if self.funcao != "melhorar":
            a += ["--estilo", self.estilo]
        else:
            a += ["--" + o.replace("_", "-") for o in self.opcoes]
        if self.exportar:
            a += ["--exportar", self.exportar, "--rotulo" if self.rotulo else "--sem-rotulo"]
        return a + ["--json"]


def upscale_args(image: str, out: str) -> list[str]:
    return ["--image", str(image), "--out", str(out), "--json"]


def furniture_video_args(photo: str, empty: str | None, room: str, out: str, fmt: str, work_dir: str,
                         accept_weak: bool = False, fabric: bool = True) -> list[str]:
    """Argumentos de ``mobilia.cli run`` para o vídeo "móveis entrando" (modo B): começa no cômodo vazio
    escolhido (``empty``), os móveis reais da foto entram peça por peça e o vídeo termina na foto."""
    if fmt not in FURNITURE_FORMATS:
        raise ValueError(f"Formato de vídeo inválido: {fmt!r}")
    if not isinstance(photo, str) or not Path(photo).is_file():
        raise ValueError("Foto original não encontrada.")
    a = ["--photo", str(photo), "--room", VIDEO_ROOMS.get(room, "sala"), "--out", str(out), "--mode", "B",
         "--quality", "final", "--work-dir", str(work_dir), "--formato", fmt]
    if empty:
        a += ["--sala-vazia", str(empty)]
    if accept_weak:
        a.append("--aceitar-placa-fraca")
    if not fabric:
        a.append("--sem-tecido")
    return a + ["--json"]


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
        f = theme.font(12, QFont.Medium)
        fn = theme.font(11, QFont.DemiBold, theme.MONO_FAMILY)
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
    P_PHOTO, P_FUNC, P_STYLE, P_PROGRESS, P_RESULTS, P_EDIT, P_VIDEO, P_ENHANCE, P_BATCH, P_FVIDEO = range(10)
    L_DROP, L_TEXT, L_PROGRESS, L_RESULT, L_MASK, L_BATCH = range(6)
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
        self.batch = CliTask(self)                     # lote (várias fotos de um imóvel)
        self.batch.progress.connect(self._on_batch_progress)
        self.batch.item.connect(self._on_batch_item)
        self.batch.finished.connect(self._on_batch_finished)
        self.batch_photos: list[str] = []
        self.batch_result: CliResult | None = None
        self.mask_mode = "editar"                      # editar | remover (o pincel do "Melhorar foto")
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)

        self.modules = ModuleHost(self)                # módulos opcionais (mobilia/modules); sem eles, nada muda
        root = Backdrop(); self.setCentralWidget(root)
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
        self.left.addWidget(self._build_batch_stage())
        body.addWidget(self.left, 1)
        self.right = QStackedWidget(); self.right.setFixedWidth(424)
        self.pages: list[Page] = []
        for build in (self._page_photo, self._page_function, self._page_style, self._page_progress,
                      self._page_results, self._page_edit, self._page_video, self._page_enhance, self._page_batch,
                      self._page_furniture_video):
            pg = build(); self.pages.append(pg); self.right.addWidget(pg)
        body.addWidget(self.right)

        sb = StatusBar(); self.setStatusBar(sb); sb.setSizeGripEnabled(False); self.sb = sb
        sb.addPermanentWidget(QLabel(f"{APP_NAME}  {__version__}"))
        # atualização automática (GitHub Releases): só no executável; ver updater_dialog.py
        self.updater = attach_updater(self, sb, jobs=(self.design, self.side, self.batch))
        sb.showMessage("Detectando GPU…")
        self.probe = HardwareProbe(self); self.probe.done.connect(self._on_hardware)
        self.go(self.P_PHOTO)
        self._fit_to_screen()

    # ------------------------------------------------------------------------------------------ montagem
    def _build_header(self) -> QHBoxLayout:
        head = QHBoxLayout(); head.setSpacing(0); head.setContentsMargins(2, 4, 0, 0)
        mark = QLabel(); mark.setPixmap(brand.lockup_pixmap(34)); mark.setToolTip(APP_NAME)
        mark.setAccessibleName(APP_NAME)
        vr = QFrame(); vr.setObjectName("vrule"); vr.setFixedSize(1, 22)
        tag = QLabel(TAGLINE); tag.setObjectName("tagline")
        lock = QLabel(); lock.setPixmap(theme.line_icon("chip", 14, theme.MUTED).pixmap(14, 14))
        priv = QLabel("Processado no seu computador"); priv.setObjectName("privacy")
        priv.setToolTip("A foto e as imagens não saem do seu PC: tudo é gerado aqui, na sua placa de vídeo.")
        self.privacy_widgets = (lock, priv)     # escondidos em módulos que usam a internet (ModuleSpec.local)
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
        self.mask_title = t
        self.mask_name = QLabel(""); self.mask_name.setObjectName("caption")
        head.addWidget(t); head.addStretch(1); head.addWidget(self.mask_name)
        self.mask = MaskEditor(); self.mask.changed.connect(self._on_mask_changed)
        cap = QLabel("Pinte sobre o que quer trocar. Fora da área pintada, a imagem fica idêntica.")
        cap.setObjectName("stagecap"); cap.setAlignment(Qt.AlignCenter); cap.setWordWrap(True)
        self.mask_cap = cap
        lay.addLayout(head); lay.addWidget(self.mask, 1); lay.addWidget(cap)
        return card

    def _set_mask_mode(self, mode: str) -> None:
        """O mesmo editor de máscara serve à "Editar área" (o que trocar) e ao "Melhorar foto" (o que apagar)."""
        self.mask_mode = mode
        if mode == "remover":
            self.mask_title.setText("REMOVER OBJETOS (OPCIONAL)")
            self.mask_cap.setText("Pinte o que deve sumir da foto (fios, lixo, objetos pessoais). Sem pintura, "
                                  "nada é removido.")
        else:
            self.mask_title.setText("EDITAR ÁREA")
            self.mask_cap.setText("Pinte sobre o que quer trocar. Fora da área pintada, a imagem fica idêntica.")

    def _on_mask_changed(self) -> None:
        if self.mask_mode == "remover":
            self._update_enhance_ready()
        else:
            self._update_edit_ready()

    def _build_batch_stage(self) -> QWidget:
        from PySide6.QtWidgets import QListWidget

        card, lay = self._stage()
        head = QHBoxLayout()
        t = tracked_label("FOTOS DO IMÓVEL"); t.setStyleSheet(f"color: {theme.ACCENT_HI};")
        self.batch_caption = QLabel(""); self.batch_caption.setObjectName("caption")
        head.addWidget(t); head.addStretch(1); head.addWidget(self.batch_caption)
        self.batch_list = QListWidget(); self.batch_list.setObjectName("batchlist")
        self.batch_list.setViewMode(QListWidget.IconMode); self.batch_list.setIconSize(QSize(176, 118))
        self.batch_list.setResizeMode(QListWidget.Adjust); self.batch_list.setMovement(QListWidget.Static)
        self.batch_list.setSpacing(8); self.batch_list.setWordWrap(True)
        self.batch_list.setStyleSheet(
            f"QListWidget#batchlist {{ background: transparent; border: none; color: {theme.TEXT_2}; }}"
            f"QListWidget#batchlist::item {{ border-radius: 8px; padding: 4px; }}"
            f"QListWidget#batchlist::item:selected {{ background: {theme.SELECTED_BG}; color: {theme.TEXT}; }}")
        self.batch_list.itemDoubleClicked.connect(lambda it: self.open_url(it.data(Qt.UserRole)) if it.data(Qt.UserRole) else None)
        cap = QLabel("Cada foto mostra o resultado assim que fica pronta. Dois cliques abrem a imagem.")
        cap.setObjectName("stagecap"); cap.setAlignment(Qt.AlignCenter); cap.setWordWrap(True)
        lay.addLayout(head); lay.addWidget(self.batch_list, 1); lay.addWidget(cap)
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
        pg.body.addSpacing(18); pg.body.addWidget(_section("VÁRIAS FOTOS DO MESMO IMÓVEL?")); pg.body.addSpacing(8)
        self.btn_batch_mode = _button("Lote: processar várias fotos", "photo", "outline", theme.ACCENT_HI)
        self.btn_batch_mode.setToolTip("Escolha as fotos de um imóvel e aplique a mesma função em todas "
                                       "(ex.: Melhorar foto). As saídas vão para uma pasta com o nome do imóvel.")
        on_click(self.btn_batch_mode, self.go, self.P_BATCH)
        pg.body.addWidget(self.btn_batch_mode)
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
        self.chk_fast = QCheckBox("Prévia rápida (resolução menor)")
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
        self.result_notice = QLabel(""); self.result_notice.setObjectName("notice"); self.result_notice.setWordWrap(True)
        self.result_notice.setStyleSheet(f"color: {theme.ACCENT}; padding-top: 6px;")
        self.result_notice.hide()
        pg.head.addWidget(self.result_notice)
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
        self.btn_fvideo = _button("Vídeo: móveis entrando", "video", "primary", theme.ON_ACCENT)
        self.btn_fvideo.setToolTip("Vídeo que começa neste cômodo vazio, os móveis reais da sua foto entram peça "
                                   "por peça e termina exatamente na sua foto (MP4).")
        on_click(self.btn_fvideo, self.go, self.P_FVIDEO)
        self.btn_fvideo.hide()
        b.addWidget(self.btn_fvideo); b.addSpacing(10)
        self.btn_export = _button("Exportar para anúncio", "export")
        self.btn_export.setToolTip("JPEG pronto para WhatsApp e portais (2048 px ou 4K), com o selo "
                                   "“Imagem ilustrativa” opcional.")
        on_click(self.btn_export, self.export_listing)
        for i, w in enumerate((self.btn_edit, self.btn_upscale, self.btn_video, self.btn_folder)):
            grid.addWidget(w, i // 2, i % 2)
        grid.addWidget(self.btn_export, 2, 0, 1, 2)
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

    def _page_furniture_video(self) -> Page:
        pg = Page()
        pg.head.addWidget(StepHeader("05", "VÍDEO: MÓVEIS ENTRANDO")); pg.head.addSpacing(12)
        b = pg.body
        intro = QLabel("O vídeo começa no cômodo vazio escolhido, os móveis reais da sua foto entram peça por peça "
                       "(a roupa de cama e as almofadas assentam no pouso) e termina exatamente na sua foto.")
        intro.setObjectName("body"); intro.setWordWrap(True)
        b.addWidget(intro); b.addSpacing(18)
        b.addWidget(_section("FORMATO")); b.addSpacing(10)
        self.fformat_group = QButtonGroup(self); self.fformat_group.setExclusive(True)
        self.fformat_chips = {}
        frow = QHBoxLayout(); frow.setSpacing(8)
        for key, (title, desc) in FURNITURE_FORMATS.items():
            c = chip(f"{title}\n{desc.split(' · ')[0]}", height=58)
            c.setAccessibleName(title); c.setToolTip(desc)
            on_click(c, self.select_furniture_format, key)
            self.fformat_group.addButton(c); self.fformat_chips[key] = c
            frow.addWidget(c)
        b.addLayout(frow); b.addSpacing(8)
        self.fformat_desc = QLabel(""); self.fformat_desc.setObjectName("hint"); self.fformat_desc.setWordWrap(True)
        b.addWidget(self.fformat_desc); b.addSpacing(18)
        b.addWidget(_section("CÔMODO VAZIO DO INÍCIO")); b.addSpacing(10)
        self.fvideo_base = QLabel(""); self.fvideo_base.setObjectName("hint"); self.fvideo_base.setWordWrap(True)
        b.addWidget(self.fvideo_base); b.addSpacing(12)
        self.chk_fabric = QCheckBox("Tecido assentando (roupa de cama, almofadas, cortinas)")
        self.chk_fabric.setChecked(True)
        self.chk_fabric.setToolTip("Um movimento pequeno e suave do tecido quando a peça pousa. O último quadro "
                                   "continua idêntico à foto.")
        b.addWidget(self.chk_fabric)
        b.addStretch(1)
        self.fvideo_ready = QLabel(""); self.fvideo_ready.setObjectName("hint"); self.fvideo_ready.setAlignment(Qt.AlignCenter)
        self.fvideo_ready.setWordWrap(True)
        self.btn_fvideo_generate = _button("Gerar vídeo com os móveis entrando", "video", "primary", theme.ON_ACCENT)
        on_click(self.btn_fvideo_generate, self.generate_furniture_video)
        self.btn_fvideo_back = _button("Voltar aos resultados", "back", "ghost")
        on_click(self.btn_fvideo_back, self.go, self.P_RESULTS)
        pg.foot.addWidget(self.fvideo_ready); pg.foot.addWidget(self.btn_fvideo_generate)
        pg.foot.addWidget(self.btn_fvideo_back)
        self.furniture_format = "original"
        self.select_furniture_format("original")
        return pg

    def _enhance_checks(self, lay, prefix: str) -> dict[str, QCheckBox]:
        checks = {}
        for key, (label, tip) in ENHANCE_STEPS.items():
            c = QCheckBox(label); c.setChecked(True); c.setToolTip(tip)
            c.setAccessibleName(f"{prefix} {label}")
            lay.addWidget(c); checks[key] = c
        return checks

    def _page_enhance(self) -> Page:
        pg = Page()
        pg.head.addWidget(StepHeader("03", "MELHORAR FOTO")); pg.head.addSpacing(12)
        b = pg.body
        intro = QLabel("Correções de fotógrafo profissional, sem mudar nada do imóvel: luz, cor, linhas retas, céu e "
                       "nitidez. Desmarque o que não quiser.")
        intro.setObjectName("body"); intro.setWordWrap(True)
        b.addWidget(intro); b.addSpacing(16)
        b.addWidget(_section("AJUSTES")); b.addSpacing(8)
        box = QVBoxLayout(); box.setSpacing(6)
        self.enhance_checks = self._enhance_checks(box, "Melhorar:")
        self.chk_enh_4k = QCheckBox("Entregar em 4K (3840 px)")
        self.chk_enh_4k.setToolTip("Amplia o resultado com o “Ampliar 4K”. Mais lento; para WhatsApp, 2048 px bastam.")
        box.addWidget(self.chk_enh_4k)
        b.addLayout(box)
        b.addSpacing(16); b.addWidget(rule()); b.addSpacing(14)
        b.addWidget(_section("REMOVER OBJETOS (OPCIONAL)")); b.addSpacing(8)
        rh = QLabel("Pinte na foto ao lado o que deve sumir."); rh.setObjectName("hint"); rh.setWordWrap(True)
        b.addWidget(rh); b.addSpacing(8)
        self.enh_tool = Segmented([("pincel", "Pincel"), ("borracha", "Borracha")], "pincel")
        self.enh_tool.changed.connect(self.set_tool)
        b.addWidget(self.enh_tool); b.addSpacing(10)
        srow = QHBoxLayout(); srow.setSpacing(12)
        sl = QLabel("Tamanho"); sl.setObjectName("muted")
        self.enh_brush = QSlider(Qt.Horizontal); self.enh_brush.setRange(8, 200); self.enh_brush.setValue(32)
        self.enh_brush.valueChanged.connect(lambda v: self.mask.set_brush(int(v)))
        srow.addWidget(sl); srow.addWidget(self.enh_brush, 1)
        b.addLayout(srow); b.addSpacing(8)
        urow = QHBoxLayout(); urow.setSpacing(8)
        self.btn_enh_undo = _button("Desfazer", "undo"); on_click(self.btn_enh_undo, self.mask.undo)
        self.btn_enh_clear = _button("Limpar", "trash"); on_click(self.btn_enh_clear, self.mask.clear)
        urow.addWidget(self.btn_enh_undo, 1); urow.addWidget(self.btn_enh_clear, 1)
        b.addLayout(urow)
        b.addStretch(1)
        self.enhance_ready = QLabel(""); self.enhance_ready.setObjectName("hint")
        self.enhance_ready.setAlignment(Qt.AlignCenter); self.enhance_ready.setWordWrap(True)
        self.btn_enhance = _button("Melhorar foto", "sparkle", "primary", theme.ON_ACCENT)
        on_click(self.btn_enhance, self.generate_enhance)
        self.btn_enh_back = _button("Trocar a função", "back", "ghost")
        on_click(self.btn_enh_back, self.leave_enhance)
        pg.foot.addWidget(self.enhance_ready); pg.foot.addWidget(self.btn_enhance); pg.foot.addWidget(self.btn_enh_back)
        return pg

    def _page_batch(self) -> Page:
        from PySide6.QtWidgets import QLineEdit

        pg = Page()
        pg.head.addWidget(StepHeader("01", "LOTE DE FOTOS")); pg.head.addSpacing(12)
        b = pg.body
        intro = QLabel("Todas as fotos de um imóvel de uma vez, com a mesma função. As imagens prontas vão para "
                       "uma pasta com o nome do imóvel.")
        intro.setObjectName("body"); intro.setWordWrap(True)
        b.addWidget(intro); b.addSpacing(14)
        b.addWidget(_section("IMÓVEL")); b.addSpacing(8)
        self.batch_name = QLineEdit(); self.batch_name.setPlaceholderText("Ex.: Apto 302 - Rua das Flores")
        b.addWidget(self.batch_name); b.addSpacing(10)
        frow = QHBoxLayout(); frow.setSpacing(8)
        self.btn_batch_add = _button("Adicionar fotos…", "photo"); on_click(self.btn_batch_add, self.choose_batch_photos)
        self.btn_batch_clear = _button("Limpar", "trash"); on_click(self.btn_batch_clear, self.clear_batch)
        frow.addWidget(self.btn_batch_add, 2); frow.addWidget(self.btn_batch_clear, 1)
        b.addLayout(frow); b.addSpacing(14)
        b.addWidget(_section("FUNÇÃO")); b.addSpacing(8)
        self.batch_func = QComboBox(); self.batch_func.setView(QListView())
        for k in BATCH_FUNCS:
            self.batch_func.addItem(CATALOG["funcoes"][k], k)
        self.batch_func.currentIndexChanged.connect(lambda *_a: self._update_batch_page())
        b.addWidget(self.batch_func); b.addSpacing(8)
        self.batch_style = QComboBox(); self.batch_style.setView(QListView())
        for k, lab in CATALOG["estilos"].items():
            self.batch_style.addItem(lab, k)
        self.batch_room = QComboBox(); self.batch_room.setView(QListView())
        for k, lab in CATALOG["ambientes"].items():
            if k != "fachada":
                self.batch_room.addItem(lab, k)
        self.batch_gen_box = QWidget(); gl = QHBoxLayout(self.batch_gen_box); gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(8); gl.addWidget(self.batch_room, 1); gl.addWidget(self.batch_style, 1)
        b.addWidget(self.batch_gen_box)
        self.batch_enh_box = QWidget(); el = QVBoxLayout(self.batch_enh_box); el.setContentsMargins(0, 0, 0, 0)
        el.setSpacing(4)
        self.batch_checks = self._enhance_checks(el, "Lote:")
        b.addWidget(self.batch_enh_box); b.addSpacing(12)
        self.chk_batch_export = QCheckBox("Exportar para anúncio (JPEG 2048 px)"); self.chk_batch_export.setChecked(True)
        self.chk_batch_label = QCheckBox("Selo “Imagem ilustrativa”")
        self.chk_batch_export.toggled.connect(lambda *_a: self._update_batch_page())
        b.addWidget(self.chk_batch_export); b.addWidget(self.chk_batch_label)
        b.addSpacing(12)
        self.batch_bar = QProgressBar(); self.batch_bar.setRange(0, 1000); self.batch_bar.setTextVisible(False)
        self.batch_msg = QLabel(""); self.batch_msg.setObjectName("hint"); self.batch_msg.setWordWrap(True)
        b.addWidget(self.batch_bar); b.addWidget(self.batch_msg)
        b.addStretch(1)
        self.btn_batch_run = _button("Processar fotos", "sparkle", "primary", theme.ON_ACCENT)
        on_click(self.btn_batch_run, self.run_batch)
        arow = QHBoxLayout(); arow.setSpacing(8)
        self.btn_batch_cancel = _button("Cancelar", kind="danger"); on_click(self.btn_batch_cancel, self.cancel_batch)
        self.btn_batch_folder = _button("Abrir pasta", "folder"); on_click(self.btn_batch_folder, self.open_batch_folder)
        arow.addWidget(self.btn_batch_cancel, 1); arow.addWidget(self.btn_batch_folder, 1)
        self.btn_batch_back = _button("Voltar", "back", "ghost"); on_click(self.btn_batch_back, self.go, self.P_PHOTO)
        pg.foot.addWidget(self.btn_batch_run); pg.foot.addLayout(arow); pg.foot.addWidget(self.btn_batch_back)
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
        if self.batch.busy() and page != self.P_BATCH:
            page = self.P_BATCH
        if page in (self.P_RESULTS, self.P_VIDEO, self.P_FVIDEO) and self.result is None:
            page = self.P_STYLE if self.funcao else self.P_FUNC
        if page == self.P_FVIDEO and self.result is not None and self.result.request.funcao != "esvaziar":
            page = self.P_RESULTS
        if page == self.P_STYLE and not self.funcao:
            page = self.P_FUNC
        if page == self.P_STYLE and self.funcao in NEEDS_PHOTO and not self.photo_path:
            page = self.P_PHOTO
        if page == self.P_STYLE and self.funcao == "editar_area":
            page = self.P_EDIT
        if page == self.P_STYLE and self.funcao == "melhorar":
            page = self.P_ENHANCE
        if page == self.P_EDIT and not self.edit_base:
            page = self.P_FUNC
        if page == self.P_ENHANCE and not self.photo_path:
            page = self.P_PHOTO
        if page != self.right.currentIndex():
            self.statusBar().clearMessage()          # nada de "Geração cancelada" velho em outra tela
        self.right.setCurrentIndex(page)
        if page == self.P_ENHANCE:
            self._enter_enhance()
        elif page == self.P_EDIT and self.mask_mode != "editar":
            self._set_mask_mode("editar")
            if self.edit_base:
                self.mask.set_image(self.edit_base)
        if page == self.P_PROGRESS:
            self.left.setCurrentIndex(self.L_PROGRESS)
        elif page == self.P_BATCH:
            self.left.setCurrentIndex(self.L_BATCH)
            self._update_batch_page()
        elif page in (self.P_RESULTS, self.P_VIDEO, self.P_FVIDEO):
            self.left.setCurrentIndex(self.L_RESULT)
        elif page in (self.P_EDIT, self.P_ENHANCE):
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
        elif page == self.P_FVIDEO:
            self._update_furniture_video_page()
        elif page == self.P_RESULTS:
            self._update_result_actions()
        self._update_photo_page()
        self._update_stepper()

    def _update_stepper(self) -> None:
        page = self.page()
        cur = {self.P_PHOTO: 0, self.P_FUNC: 1, self.P_STYLE: 2, self.P_EDIT: 2, self.P_PROGRESS: 3,
               self.P_RESULTS: 4, self.P_VIDEO: 4, self.P_ENHANCE: 2, self.P_BATCH: 0, self.P_FVIDEO: 4}[page]
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
        self.go(self.P_FUNC if self.page() in (self.P_PHOTO, self.P_RESULTS, self.P_VIDEO, self.P_FVIDEO)
                else self.page())
        return True

    def _update_photo_page(self) -> None:
        self.step_photo.set_done(bool(self.photo_path))
        self.btn_photo_next.setEnabled(bool(self.photo_path))
        self.photo_ready.setText("Tudo certo. Agora escolha o que fazer com a foto." if self.photo_path else
                                 "Importe uma foto, ou crie um ambiente a partir de um texto.")

    def new_photo(self) -> None:
        self._enhance_photo = None
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
        if key == "melhorar":
            self._enhance_photo = None            # foto nova no pincel (máscara limpa)
            self.go(self.P_ENHANCE)
            return
        self.go(self.P_STYLE)

    # ------------------------------------------------------------------------------------------ passo 3: ajustes
    def _update_style_page(self) -> None:
        f = self.funcao
        self.func_label.setText(f"Função: {CATALOG['funcoes'].get(f, '')}")
        self.room_box.setVisible(f != "fachada")
        self.event_box.setVisible(f == "evento")
        self.style_box.setVisible(f not in ("evento", "esvaziar"))
        self.prompt_title.setVisible(f != "esvaziar"); self.prompt.setVisible(f != "esvaziar")
        self.aspect_box.setVisible(f == "texto")
        self.style_header.set_title("AMBIENTE E EVENTO" if f == "evento" else
                                    "ESTILO DA FACHADA" if f == "fachada" else
                                    "TIPO DE AMBIENTE" if f == "esvaziar" else "AMBIENTE E ESTILO")
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
        card = self.style_cards[self.estilo]
        QTimer.singleShot(0, lambda: self.pages[self.P_STYLE].scroll.ensureWidgetVisible(card, 0, 24))
        self.aspect_seg.set_value(self.formato)
        self.text_summary.setText(self._summary())
        self._update_style_ready()

    def _summary(self) -> str:
        parts = [CATALOG["ambientes"].get(self.ambiente, "")]
        if self.funcao == "evento":
            parts.append(CATALOG["eventos"][self.evento])
        elif self.funcao == "esvaziar":
            parts.append("Cômodo vazio")
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
        estilo = None if f in NO_STYLE else self.estilo
        tag = f if f != "evento" else f"evento_{self.evento}"
        return DesignRequest(funcao=f, estilo=estilo, ambiente="fachada" if f == "fachada" else self.ambiente,
                             n=int(self.n_seg.value()), out_dir=self._new_out_dir(f"{tag}_{estilo or ''}".rstrip("_")),
                             photo=self.photo_path if f in NEEDS_PHOTO else None,
                             prompt="" if f == "esvaziar" else self.prompt.toPlainText(),
                             evento=self.evento if f == "evento" else None,
                             formato=self.formato, quality="rapido" if self.chk_fast.isChecked() else "final")

    def generate(self) -> None:
        """Gera as versões com as escolhas da tela (sem parâmetros: nada do sinal do botão entra no pedido)."""
        if self.design.busy() or self.batch.busy() or not self.funcao or self.funcao == "editar_area":
            return
        if self.funcao == "melhorar":
            self.generate_enhance()
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
        elif r.funcao == "melhorar":
            self._back_page = self.P_ENHANCE
        self.submit(req)

    def submit(self, req: DesignRequest) -> None:
        req.validate()
        self.last_request = req
        self.variants.reset(req.n)
        texto = req.funcao == "texto"
        self.stages.set_stages(TEXT_STAGES if texto else ENHANCE_STAGES if req.funcao == "melhorar" else DESIGN_STAGES)
        self.stages.reset()
        self.bar.setValue(0); self._set_pct(0.0); self.msg.setText("Iniciando o motor…")
        self.elapsed.setText("")
        n = req.n
        what = CATALOG["eventos"][req.evento] if req.evento else (
            "Área pintada" if req.funcao == "editar_area" else "Luz, cor e linhas" if req.funcao == "melhorar"
            else "Cômodo vazio" if req.funcao == "esvaziar"
            else CATALOG["estilos"].get(req.estilo or "", ""))
        self.progress_title.setText("MELHORANDO A FOTO" if req.funcao == "melhorar" else
                                    "GERANDO A VERSÃO" if n == 1 else f"GERANDO {n} VERSÕES")
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
            aviso = res.done.get("aviso")
            applied = res.done.get("funcao")
            self.result = DesignOutcome(req, outs, before, res.done.get("report"),
                                        tuple(size) if isinstance(size, (list, tuple)) and len(size) == 2 else None,
                                        (res.finished_at or time.time()) - res.started_at,
                                        aviso if isinstance(aviso, str) and aviso else None,
                                        applied if applied in CATALOG["funcoes"] else None)
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
        elif req.funcao == "melhorar":
            title = "Foto melhorada"
        elif req.funcao == "esvaziar":
            title = "Cômodo vazio"
        elif req.funcao == "evento":
            title = CATALOG["eventos"][req.evento]
        else:
            title = CATALOG["estilos"].get(req.estilo or "", "")
        room = CATALOG["ambientes"][req.ambiente]
        self.result_title.setText(f"{title} · {room}" if req.funcao != "fachada" else f"{title} · Fachada")
        took = int(round(r.seconds))
        took_s = f"{took // 60} min {took % 60:02d} s" if took >= 60 else f"{took} s"
        k = len(r.outputs)
        # sem "768 × 576 px" aqui: a resolução de trabalho não é a de entrega (o "Ampliar 4K" leva a 3840 px)
        info = [CATALOG["funcoes"][r.funcao or req.funcao], f"{k} {'versão' if k == 1 else 'versões'}"]
        if took >= 1:
            info.append(f"em {took_s}")
        self.result_info.setText("  ·  ".join(info))
        self.result_notice.setText(r.aviso or "")
        self.result_notice.setVisible(bool(r.aviso))
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
        self.btn_export.setEnabled(has and not busy)
        esvaziar = has and self.result.request.funcao == "esvaziar"
        self.btn_fvideo.setVisible(esvaziar)
        self.btn_fvideo.setEnabled(esvaziar and not busy)
        melhorar = has and self.result.request.funcao == "melhorar"
        self.btn_again.setEnabled(has and not self.design.busy() and not melhorar)
        self.btn_again.setToolTip("O “Melhorar foto” dá sempre o mesmo resultado: mude os ajustes em “Outro ajuste”."
                                  if melhorar else "Mesmas escolhas, outras sementes: versões diferentes.")
        self.btn_restyle.setText("  Outro ajuste" if melhorar else "  Outro ambiente" if esvaziar else "  Outro estilo")
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

    # --- melhorar foto
    def _enter_enhance(self) -> None:
        """Pincel de remoção sobre a foto atual (a máscara pintada continua se a foto é a mesma)."""
        same = self.mask_mode == "remover" and getattr(self, "_enhance_photo", None) == self.photo_path
        self._set_mask_mode("remover")
        if not same and self.photo_path:
            if self.mask.set_image(self.photo_path):
                self._enhance_photo = self.photo_path
                self.mask_name.setText(Path(self.photo_path).name)
        self.mask.set_brush(int(self.enh_brush.value()))
        self.mask.set_erasing(self.enh_tool.value() == "borracha")
        self._update_enhance_ready()

    def leave_enhance(self) -> None:
        self.go(self.P_FUNC)

    def enhance_options(self) -> tuple:
        opts = tuple(f"sem_{k}" for k, c in self.enhance_checks.items() if not c.isChecked())
        return opts + (("ampliar_4k",) if self.chk_enh_4k.isChecked() else ())

    def _update_enhance_ready(self) -> None:
        has = self.mask.has_mask()
        busy = self.design.busy() or self.batch.busy()
        self.btn_enhance.setEnabled(bool(self.photo_path) and not busy)
        self.btn_enh_undo.setEnabled(self.mask.can_undo())
        self.btn_enh_clear.setEnabled(has)
        self.btn_enhance.setText("  Melhorar e remover objetos" if has else "  Melhorar foto")
        self.enhance_ready.setText("A área pintada será apagada antes dos ajustes." if has else
                                   "Pronto. Leva segundos (o céu e a remoção usam IA).")

    def generate_enhance(self) -> None:
        if self.design.busy() or self.batch.busy() or not self.photo_path:
            return
        out_dir = self._new_out_dir("melhorar")
        mask_path = None
        if self.mask_mode == "remover" and self.mask.has_mask():
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            mask_path = str(Path(out_dir) / "remover.png")
            if not self.mask.save_mask(mask_path):
                self.warn("Não foi possível gravar a máscara.")
                return
        req = DesignRequest(funcao="melhorar", estilo=None, ambiente=self.ambiente if self.ambiente in
                            CATALOG["ambientes"] else "sala", n=1, out_dir=out_dir, photo=self.photo_path,
                            mask=mask_path, opcoes=self.enhance_options())
        try:
            req.validate()
        except ValueError as e:
            self.warn(str(e))
            return
        self._back_page = self.P_ENHANCE
        self.submit(req)

    # --- exportar para anúncio
    def ask_export_options(self, label_default: bool) -> tuple[str, bool] | None:
        """Tamanho (2048 px / 4K) e selo "Imagem ilustrativa" (padrão: ligado no redesenho, desligado no
        "Melhorar foto"). ``None`` = cancelado."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox

        dlg = QDialog(self); dlg.setWindowTitle("Exportar para anúncio")
        lay = QVBoxLayout(dlg); lay.setSpacing(10)
        lay.addWidget(_section("TAMANHO"))
        seg = Segmented(list(EXPORT_SIDES.items()), "2048")
        lay.addWidget(seg)
        chk = QCheckBox("Selo discreto “Imagem ilustrativa”"); chk.setChecked(label_default)
        chk.setToolTip("Recomendado quando a foto mostra móveis ou decoração criados por IA.")
        lay.addWidget(chk)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return None
        return str(seg.value()), chk.isChecked()

    def export_listing(self) -> None:
        src = self.selected_output()
        if not src or self.side.busy() or self.result is None:
            return
        src = self.upscaled.get(src, src)
        func = self.result.funcao or self.result.request.funcao
        opt = self.ask_export_options(func in ILLUSTRATIVE)
        if not opt:
            return
        lado, rotulo = opt
        try:
            args = export_args([src], str(Path(self.result.folder) / "anuncio"), lado, rotulo)
        except ValueError as e:
            self.warn(str(e))
            return
        self._start_side("exportar", args, "Exportando para o anúncio")

    # --- lote
    def choose_batch_photos(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Fotos do imóvel", str(Path.home()), PHOTO_FILTER)
        if files:
            self.add_batch_photos(list(files))

    def add_batch_photos(self, paths: list[str]) -> int:
        from PySide6.QtGui import QIcon, QPixmap
        from PySide6.QtWidgets import QListWidgetItem

        if self.batch.busy():
            return 0
        k = 0
        for p in paths:
            if not isinstance(p, str) or not Path(p).is_file() or p in self.batch_photos:
                continue
            self.batch_photos.append(p)
            it = QListWidgetItem(QIcon(cover_pixmap(QPixmap(p), 176, 118, 6, 2.0)), f"{Path(p).name}\nna fila")
            it.setToolTip(p)
            self.batch_list.addItem(it)
            k += 1
        self.batch_result = None
        self._update_batch_page()
        return k

    def clear_batch(self) -> None:
        if self.batch.busy():
            return
        self.batch_photos = []
        self.batch_list.clear()
        self.batch_result = None
        self.batch_bar.setValue(0); self.batch_msg.setText("")
        self._update_batch_page()

    def _update_batch_page(self) -> None:
        func = self.batch_func.currentData()
        self.batch_gen_box.setVisible(func != "melhorar")
        self.batch_room.setVisible(func not in ("melhorar", "fachada"))
        self.batch_enh_box.setVisible(func == "melhorar")
        if getattr(self, "_batch_func_prev", None) != func:
            self._batch_func_prev = func
            self.chk_batch_label.setChecked(func in ILLUSTRATIVE)
        self.chk_batch_label.setEnabled(self.chk_batch_export.isChecked())
        busy = self.batch.busy()
        n = len(self.batch_photos)
        self.btn_batch_run.setText(f"  Processar {n} {'foto' if n == 1 else 'fotos'}" if n else "  Processar fotos")
        self.btn_batch_run.setEnabled(n > 0 and not busy and not self.design.busy())
        self.btn_batch_add.setEnabled(not busy); self.btn_batch_clear.setEnabled(not busy and n > 0)
        self.batch_func.setEnabled(not busy)
        self.btn_batch_cancel.setVisible(busy)
        done = self.batch_result is not None and bool(self.batch_result.done.get("folder"))
        self.btn_batch_folder.setVisible(done)
        self.batch_caption.setText(f"{n} {'foto' if n == 1 else 'fotos'}" if n else "")
        if not busy and not done and not self.batch_msg.text():
            self.batch_msg.setText("Adicione as fotos do imóvel." if not n else "")

    @staticmethod
    def _safe_folder(name: str) -> str:
        import re

        s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name or "")
        s = re.sub(r"\s+", " ", s).strip(" .")
        return s[:80]

    def batch_request(self) -> BatchRequest:
        name = self.batch_name.text().strip()
        folder = self._safe_folder(name) or f"Imóvel {datetime.now().strftime('%Y-%m-%d %H%M')}"
        base = runtime.runtime_paths().jobs_dir
        d = base / folder
        k = 2
        while d.exists():
            d = base / f"{folder} ({k})"; k += 1
        func = self.batch_func.currentData()
        opts = tuple(f"sem_{key}" for key, c in self.batch_checks.items() if not c.isChecked())
        return BatchRequest(photos=list(self.batch_photos), imovel=name, funcao=func, out_dir=str(d),
                            estilo=self.batch_style.currentData(),
                            ambiente="fachada" if func == "fachada" else self.batch_room.currentData(),
                            opcoes=opts if func == "melhorar" else (),
                            exportar="2048" if self.chk_batch_export.isChecked() else None,
                            rotulo=self.chk_batch_label.isChecked())

    def run_batch(self) -> None:
        if self.batch.busy() or self.design.busy() or not self.batch_photos:
            return
        try:
            req = self.batch_request().validate()
        except ValueError as e:
            self.warn(str(e))
            return
        self.batch_out = req.out_dir
        self.batch_result = None
        for i in range(self.batch_list.count()):
            it = self.batch_list.item(i)
            it.setText(f"{Path(self.batch_photos[i]).name}\nna fila"); it.setData(Qt.UserRole, None)
        self.batch_bar.setValue(0); self.batch_msg.setText("Iniciando o motor…")
        self.batch.start("design-lote", req.args())
        self._update_batch_page()

    def cancel_batch(self) -> None:
        if self.batch.busy():
            self.batch_msg.setText("Cancelando…")
            self.batch.cancel()

    def _on_batch_progress(self, stage: str, frac: float, message: str) -> None:
        self.batch_bar.setValue(int(frac * 1000))
        if message:
            self.batch_msg.setText(message)

    def _on_batch_item(self, ev: dict) -> None:
        from PySide6.QtGui import QIcon, QPixmap

        try:
            i = int(ev.get("index"))
        except (TypeError, ValueError):
            return
        it = self.batch_list.item(i)
        if it is None:
            return
        name = Path(str(ev.get("photo") or "")).name
        st = ev.get("status")
        if st == "processando":
            it.setText(f"{name}\nprocessando…")
        elif st == "pronto":
            outs = [o for o in (ev.get("outputs") or []) if isinstance(o, str) and Path(o).is_file()]
            it.setText(f"{name}\npronta")
            if outs:
                it.setIcon(QIcon(cover_pixmap(QPixmap(outs[0]), 176, 118, 6, 2.0)))
                it.setData(Qt.UserRole, outs[0])
        elif st == "erro":
            it.setText(f"{name}\nerro: {str(ev.get('message') or '')[:60]}")

    def _on_batch_finished(self, res: CliResult) -> None:
        self.batch_result = res
        if res.status == "pronto":
            itens = res.done.get("itens") or []
            ok = sum(1 for i in itens if isinstance(i, dict) and i.get("status") == "pronto")
            self.batch_bar.setValue(1000)
            extra = "  ·  pasta “anuncio” pronta" if res.done.get("anuncio") else ""
            self.batch_msg.setText(f"{ok} de {len(itens) or ok} fotos prontas{extra}.")
            self.statusBar().showMessage(f"Lote pronto: {res.done.get('folder', '')}", 10000)
        elif res.status == "cancelado":
            self.batch_msg.setText("Lote cancelado.")
        else:
            self.batch_msg.setText("O lote falhou.")
            self.show_error(res.error or "Erro desconhecido.", res.detail, title="Não foi possível processar o lote.")
        self._update_batch_page()

    def open_batch_folder(self) -> None:
        if self.batch_result is not None and self.batch_result.done.get("folder"):
            self.open_url(self.batch_result.done["folder"])

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
        # versões já ampliadas entram em 4K: quadros de 1080p nítidos em vez de uma imagem de 768 px esticada
        sel = [self.upscaled[o] if o in self.upscaled and Path(self.upscaled[o]).is_file() else o for o in sel]
        try:
            args = showcase_args(self.result.before, sel, out, self.video_format)
        except ValueError as e:
            self.warn(str(e))
            return
        self._start_side("showcase", args, "Gerando o vídeo antes e depois")
        self.go(self.P_RESULTS)

    # --- vídeo "móveis entrando" (modo B a partir do cômodo vazio escolhido)
    def select_furniture_format(self, key: str) -> None:
        if key not in FURNITURE_FORMATS:
            raise ValueError(f"Formato de vídeo inválido: {key!r}")
        self.furniture_format = key
        self.fformat_chips[key].setChecked(True)
        d = FURNITURE_FORMATS[key][1].split(" · ", 1)[1]
        self.fformat_desc.setText(d[:1].upper() + d[1:] + ".")

    def _update_furniture_video_page(self) -> None:
        sel = self.selected_output()
        self.fvideo_base.setText(f"Versão {self.selected + 1}  ·  {Path(sel).name}" if sel else "")
        self.fformat_chips[self.furniture_format].setChecked(True)
        ok = bool(sel) and self.result is not None and bool(self.result.request.photo) and not self.side.busy()
        self.btn_fvideo_generate.setEnabled(ok)
        self.fvideo_ready.setText("Leva alguns minutos: separa os móveis da foto, monta cada entrada e confere "
                                  "todos os quadros." if ok else "Aguarde o trabalho em andamento.")

    def generate_furniture_video(self, accept_weak: bool = False) -> None:
        if self.result is None or self.side.busy() or accept_weak not in (True, False):
            return
        req = self.result.request
        sel = self.selected_output()
        if not sel or not req.photo:
            return
        folder = Path(self.result.folder)
        stamp = datetime.now().strftime("%H%M%S")
        out = str(folder / f"moveis_entrando_{stamp}.mp4")
        try:
            args = furniture_video_args(req.photo, sel, req.ambiente, out, self.furniture_format,
                                        str(folder / f"video_moveis_{stamp}"), accept_weak=accept_weak,
                                        fabric=self.chk_fabric.isChecked())
        except ValueError as e:
            self.warn(str(e))
            return
        self._start_side("run", args, "Gerando o vídeo com os móveis entrando")
        self.go(self.P_RESULTS)

    def ask_accept_weak(self, message: str) -> bool:
        """O portão de qualidade reprovou a sala vazia: pergunta se gera mesmo assim."""
        box = QMessageBox(QMessageBox.Warning, "Sala vazia abaixo da qualidade mínima", message,
                          QMessageBox.Yes | QMessageBox.No, self)
        box.button(QMessageBox.Yes).setText("Gerar mesmo assim")
        box.button(QMessageBox.No).setText("Agora não")
        return box.exec() == QMessageBox.Yes

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
        if res.kind == "run" and res.status == "pronto" and not out and res.done.get("warning"):
            # modo B: a sala vazia não passou no portão (sem MP4): oferece gerar mesmo assim
            self.taskbox.hide()
            self._update_result_actions()
            if self.ask_accept_weak(str(res.done["warning"])):
                self.generate_furniture_video(accept_weak=True)
            return
        if res.status == "pronto" and out and Path(out).is_file():
            if res.kind == "upscale":
                if isinstance(res.tag, str):
                    self.upscaled[res.tag] = out
                size = res.done.get("size")
                dims = f"{size[0]} × {size[1]} px" if isinstance(size, (list, tuple)) and len(size) == 2 else ""
                self._show_task_done("Imagem 4K pronta", out, "  ·  ".join(x for x in (Path(out).name, dims) if x))
                self.btn_task_open.setText("  Abrir imagem")
                self.btn_task_export.setText("  Exportar…")
            elif res.kind == "exportar":
                lab = "com selo “Imagem ilustrativa”" if res.done.get("rotulo") else "sem selo"
                self._show_task_done("Pronto para o anúncio", out, "  ·  ".join((Path(out).name, lab)))
                self.btn_task_open.setText("  Abrir imagem")
            else:
                self.videos.append(out)
                sz = human_size(Path(out).stat().st_size)
                dur = res.done.get("duration_s")
                dur = f"{float(dur):.0f} s" if isinstance(dur, (int, float)) and dur > 0 else ""
                if res.kind == "run" and res.done.get("warning"):
                    dur = "com aviso: confira o início"
                self._show_task_done("Vídeo pronto", out, "  ·  ".join(x for x in (Path(out).name, dur, sz) if x))
                self.btn_task_open.setText("  Assistir")
                self.btn_task_export.setText("  Salvar…")
            self.statusBar().showMessage(f"Pronto: {Path(out).name}", 10000)
            self.statusBar().setToolTip(out)
        else:
            self.taskbox.hide()
            if res.status == "cancelado":
                self.statusBar().showMessage("Cancelado.", 6000)
            else:
                what = {"upscale": "ampliar a imagem", "exportar": "exportar a imagem"}.get(res.kind, "gerar o vídeo")
                self.show_error(res.error or "O motor terminou sem entregar o arquivo.", res.detail,
                                title=f"Não foi possível {what}.")
        self._update_result_actions()
        if self.page() == self.P_VIDEO:
            self._update_video_ready()
        elif self.page() == self.P_FVIDEO:
            self._update_furniture_video_page()

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
        from . import modules as _modules
        mods_ok = not _modules.failed            # um módulo que não importou (ex.: faltou no .exe) reprova
        print(f"módulos: {', '.join(s.id for s in win.modules.specs) or 'nenhum'}")
        print("selftest ok" if ok and mods_ok else "selftest FALHOU")
        return 0 if ok and mods_ok else (2 if not ok else 3)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
