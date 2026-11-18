"""Página do Lead Finder dentro da janela do 7xAi (criada na primeira abertura da aba)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from mobilia.ui import theme
from mobilia.ui.design_widgets import Segmented

from .. import VERSION
from . import icons, style
from .context import AppContext
from .dashboard_view import DashboardView
from .leads_view import LeadsView
from .messages_view import MessagesView
from .parts import FadeStack
from .search_view import SearchView
from .settings_view import SettingsView

TABS = (("search", "Pesquisar"), ("leads", "Leads"), ("messages", "Mensagens"), ("dashboard", "Painel"),
        ("settings", "Configurações"))


class LeadFinderPage(QWidget):
    def __init__(self, window, base: Path | None = None, http=None):
        super().__init__()
        self.main_window = window          # não use self.window: é um método do QWidget
        self.setObjectName("leadFinder")
        self.setStyleSheet(style.QSS)
        self.ctx = AppContext(window, base, http)

        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(14)
        head = QHBoxLayout(); head.setSpacing(14)
        mark = QWidget(); ml = QVBoxLayout(mark); ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(0)
        title = style.label("Lead Finder", "headline")
        sub = style.label("Imobiliárias de todo o Brasil em fontes autorizadas · CRM local · WhatsApp com revisão",
                          "subtitle")
        ml.addWidget(title); ml.addWidget(sub)
        head.addWidget(mark, 1)
        self.nav = Segmented(list(TABS), "search")
        self.nav.setMinimumWidth(560)
        for key, ic in (("search", "search"), ("leads", "user"), ("messages", "message"), ("dashboard", "chart"),
                        ("settings", "gear")):
            self.nav.buttons[key].setIcon(icons.icon(ic, theme.TEXT_2))
            self.nav.buttons[key].setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.nav.changed.connect(self.show_tab)
        head.addWidget(self.nav, 0, Qt.AlignVCenter)
        lay.addLayout(head)

        self.views = {"search": SearchView(self.ctx), "leads": LeadsView(self.ctx),
                      "messages": MessagesView(self.ctx), "dashboard": DashboardView(self.ctx),
                      "settings": SettingsView(self.ctx)}
        self.stack = FadeStack()
        for key, _ in TABS:
            # cada aba numa área rolável: em telas baixas (1280×720) a janela não cresce, a aba rola
            area = QScrollArea(); area.setObjectName("lfScroll"); area.setWidgetResizable(True)
            area.setFrameShape(QScrollArea.NoFrame)
            area.setWidget(self.views[key])
            self.stack.addWidget(area)
        lay.addWidget(self.stack, 1)
        self.current = "search"

        self.ctx.open_lead.connect(self._open_lead)
        self.ctx.compose_for.connect(lambda _id: self.show_tab("messages"))
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.ctx.shutdown)
        self._purged = False

    def show_tab(self, key: str) -> None:
        if key not in self.views:
            return
        self.current = key
        if self.nav.value() != key:
            self.nav.set_value(key)
        self.stack.fade_to([k for k, _ in TABS].index(key))
        view = self.views[key]
        if hasattr(view, "on_show"):
            view.on_show()

    def _open_lead(self, lead_id: int) -> None:
        self.show_tab("leads")
        self.views["leads"].show_lead(lead_id)

    # ganchos do registro de módulos
    def on_show(self) -> None:
        self.ctx.status(f"Lead Finder {VERSION} · dados comerciais guardados só neste computador")
        if not self._purged:   # prazos de guarda das licenças (ex.: Google, 30 dias) aplicados ao abrir
            self._purged = True
            self.ctx.db.call(lambda s: s.purge_expired(),
                             on_done=lambda n: self.ctx.leads_changed.emit() if n else None)
        view = self.views[self.current]
        if hasattr(view, "on_show"):
            view.on_show()

    def on_hide(self) -> None:
        pass
