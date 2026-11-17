"""Registro de módulos opcionais da janela principal (abas de primeiro nível, ex.: um futuro "Lead Finder").

Como funciona
-------------
* Um módulo é um subpacote de :mod:`mobilia.modules` (ex.: ``mobilia/modules/lead_finder/``) que, ao ser
  importado, chama :func:`register` com um :class:`ModuleSpec`.
* :func:`load` importa :mod:`mobilia.modules`, que descobre e importa cada subpacote. Um módulo que falhar ao
  importar é registrado em :data:`failed` e ignorado: a janela principal abre normalmente.
* :class:`ModuleHost` é o único ponto de contato com ``MainWindow``. **Sem nenhum módulo registrado ele não faz
  nada**: a janela é montada exatamente como antes (mesmos widgets, mesmos layouts).
* Com módulos, aparece um seletor no cabeçalho ("Ambientes" + um botão por módulo) e o conteúdo da janela vira
  uma pilha: a página 0 é o fluxo atual de redesenho; cada módulo ganha uma página criada só na primeira vez em
  que é aberto (``factory`` preguiçosa).

Regras para módulos
-------------------
* A interface não importa numpy/PIL/torch/cv2 nem QtWebEngine (o .exe da interface não os leva).
* A página recebe a ``MainWindow`` só para usar a barra de status (``window.statusBar()``) e como pai Qt; não deve
  mexer no estado do redesenho.
* Métodos opcionais na página: ``on_show()`` e ``on_hide()``, chamados ao entrar e sair do módulo.

Exemplo (documentação, não é um módulo real) em ``docs/PARALLEL_DEVELOPMENT.md``.
"""
from __future__ import annotations

import logging
import re
import traceback
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

__all__ = ["ModuleSpec", "ModuleHost", "register", "unregister", "registered", "get", "load", "failed",
           "DESIGN_ID", "DESIGN_TITLE"]

log = logging.getLogger(__name__)

DESIGN_ID = "design"          # id reservado: o fluxo principal de redesenho (página 0)
DESIGN_TITLE = "Ambientes"
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class ModuleSpec:
    """Descrição de um módulo da janela principal.

    ``id``: identificador estável (``[a-z][a-z0-9_]*``; ``"design"`` é reservado).
    ``title``: rótulo curto no seletor do cabeçalho (ex.: ``"Lead Finder"``).
    ``factory``: recebe a ``MainWindow`` e devolve o ``QWidget`` da página; chamada uma única vez, na 1ª abertura.
    ``order``: posição no seletor (menor vem antes; empate pelo título).
    ``tooltip``: dica opcional do botão.
    """

    id: str
    title: str
    factory: Callable[[QMainWindow], QWidget]
    order: int = 100
    tooltip: str = ""


_REGISTRY: dict[str, ModuleSpec] = {}
failed: list[tuple[str, str]] = []      # (módulo, traceback) de módulos que não carregaram ou não abriram
_loaded = False


def register(spec: ModuleSpec) -> ModuleSpec:
    """Registra um módulo. Levanta ``ValueError`` para id inválido, reservado ou repetido."""
    if not isinstance(spec, ModuleSpec):
        raise TypeError("register() espera um ModuleSpec")
    if not _ID_RE.match(spec.id or "") or spec.id == DESIGN_ID:
        raise ValueError(f"id de módulo inválido: {spec.id!r}")
    if spec.id in _REGISTRY:
        raise ValueError(f"módulo já registrado: {spec.id!r}")
    if not callable(spec.factory):
        raise TypeError(f"factory do módulo {spec.id!r} não é chamável")
    if not (spec.title or "").strip():
        raise ValueError(f"módulo {spec.id!r} sem título")
    _REGISTRY[spec.id] = spec
    return spec


def unregister(module_id: str) -> None:
    """Remove um módulo do registro (usado em testes)."""
    _REGISTRY.pop(module_id, None)


def get(module_id: str) -> ModuleSpec | None:
    return _REGISTRY.get(module_id)


def registered() -> list[ModuleSpec]:
    """Módulos registrados, ordenados por ``(order, title)``."""
    return sorted(_REGISTRY.values(), key=lambda s: (s.order, s.title.casefold()))


def load() -> list[ModuleSpec]:
    """Importa :mod:`mobilia.modules` (uma vez) e devolve os módulos registrados. Nunca levanta exceção."""
    global _loaded
    if not _loaded:
        _loaded = True
        try:
            from .. import modules as _pkg
            _pkg.discover()
        except Exception:  # noqa: BLE001 - um módulo quebrado não pode derrubar a janela
            failed.append(("mobilia.modules", traceback.format_exc()))
            log.exception("falha ao carregar mobilia.modules")
    return registered()


def _reset_for_tests() -> None:
    global _loaded
    _REGISTRY.clear()
    failed.clear()
    _loaded = True      # testes não descobrem módulos do disco


class ModuleHost:
    """Liga os módulos registrados à ``MainWindow``. Sem módulos, todos os métodos são inócuos."""

    def __init__(self, window: QMainWindow):
        self.window = window
        self.specs: list[ModuleSpec] = load()
        self.views: QStackedWidget | None = None
        self.nav = None
        self.current = DESIGN_ID
        self.pages: dict[str, QWidget] = {}
        self._design_lay: QVBoxLayout | None = None

    @property
    def active(self) -> bool:
        return bool(self.specs)

    def design_layout(self, outer: QVBoxLayout) -> QVBoxLayout:
        """Layout onde a janela monta o fluxo de redesenho. Sem módulos é o próprio ``outer``."""
        if not self.active:
            return outer
        self.views = QStackedWidget()
        outer.addWidget(self.views, 1)
        page = QWidget()
        lay = QVBoxLayout(page); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(outer.spacing())
        self.views.addWidget(page)
        self.pages[DESIGN_ID] = page
        self._design_lay = lay
        return lay

    def add_nav(self, head: QHBoxLayout, tagline: QLabel | None = None) -> None:
        """Insere o seletor de módulos no cabeçalho (esconde a frase de efeito para caber em 1280 px)."""
        if not self.active:
            return
        from .design_widgets import Segmented
        options = [(DESIGN_ID, DESIGN_TITLE)] + [(s.id, s.title) for s in self.specs]
        self.nav = Segmented(options, DESIGN_ID)
        self.nav.setObjectName("modulenav")
        for s in self.specs:
            if s.tooltip:
                self.nav.buttons[s.id].setToolTip(s.tooltip)
        self.nav.changed.connect(self.show)
        if tagline is not None:
            tagline.hide()
        head.addWidget(self.nav, 0, Qt.AlignVCenter)

    def show(self, module_id: str) -> None:
        """Mostra o fluxo de redesenho (``"design"``) ou a página de um módulo (criada na 1ª vez)."""
        if self.views is None or module_id == self.current:
            return
        spec = None if module_id == DESIGN_ID else get(module_id)
        if module_id != DESIGN_ID and spec is None:
            return
        if module_id not in self.pages:
            self.pages[module_id] = self._build(spec)
            self.views.addWidget(self.pages[module_id])
        old = self.pages.get(self.current)
        if old is not None and hasattr(old, "on_hide"):
            self._safe(old.on_hide)
        self.current = module_id
        page = self.pages[module_id]
        self.views.setCurrentWidget(page)
        legacy = getattr(self.window, "btn_legacy", None)
        if legacy is not None:
            legacy.setVisible(module_id == DESIGN_ID)
        if self.nav is not None and self.nav.value() != module_id:
            self.nav.set_value(module_id)
        if hasattr(page, "on_show"):
            self._safe(page.on_show)

    def set_compact(self, compact: bool) -> None:
        if self._design_lay is not None:
            self._design_lay.setSpacing(10 if compact else 14)

    # ------------------------------------------------------------------------------------------ interno
    def _build(self, spec: ModuleSpec) -> QWidget:
        try:
            page = spec.factory(self.window)
            if not isinstance(page, QWidget):
                raise TypeError(f"factory de {spec.id!r} não devolveu um QWidget")
            return page
        except Exception:  # noqa: BLE001
            failed.append((spec.id, traceback.format_exc()))
            log.exception("falha ao abrir o módulo %s", spec.id)
            box = QWidget()
            lay = QVBoxLayout(box)
            msg = QLabel(f"Não foi possível abrir “{spec.title}”. Os detalhes estão no log do aplicativo.")
            msg.setObjectName("notice"); msg.setWordWrap(True); msg.setAlignment(Qt.AlignCenter)
            lay.addStretch(1); lay.addWidget(msg); lay.addStretch(1)
            return box

    @staticmethod
    def _safe(fn) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001
            log.exception("erro no gancho do módulo")

