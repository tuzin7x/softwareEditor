"""Estado compartilhado pelas telas do Lead Finder: banco, configurações, rede e sinais entre as telas."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from .. import sources as srcmod
from ..config import Secrets, Settings, data_dir
from ..net import HttpClient
from ..storage import DB_NAME, Store
from ..tasks import DbService, Dispatcher, Task


class AppContext(QObject):
    leads_changed = Signal()          # dados de leads mudaram (pesquisa, CRM, exclusão)
    open_lead = Signal(int)           # pedir para mostrar uma empresa na aba Leads
    compose_for = Signal(int)         # abrir o gerador de mensagens para uma empresa
    settings_changed = Signal()

    def __init__(self, window, base: Path | None = None, http: HttpClient | None = None):
        super().__init__()
        self.window = window
        self.base = Path(base) if base is not None else data_dir()
        self.base.mkdir(parents=True, exist_ok=True)
        self.dispatcher = Dispatcher()
        self.settings = Settings.load(self.base)
        self.secrets = Secrets(self.base)
        self.http = http or HttpClient()
        self._apply_rate_limits()
        self.db = DbService(lambda: Store(self.base / DB_NAME), self.dispatcher)
        self.tasks: list[Task] = []
        # abrir URLs passa por aqui (os testes trocam esta função)
        self.open_url = lambda url: QDesktopServices.openUrl(QUrl(url))

    @property
    def cache_dir(self) -> Path:
        return self.base / "cache"

    def _apply_rate_limits(self) -> None:
        overpass = urlsplit(self.settings.overpass_url).hostname or "overpass-api.de"
        self.http.min_interval.update({overpass: 2.0, "brasilapi.com.br": 1.2, "places.googleapis.com": 0.2,
                                       "api.anthropic.com": 1.0})

    def sources(self) -> dict:
        return srcmod.build(self.http, self.settings, self.secrets, self.base)

    def save_settings(self) -> None:
        self.settings.save(self.base)
        self._apply_rate_limits()
        self.settings_changed.emit()

    def status(self, message: str, ms: int = 8000) -> None:
        try:
            self.window.statusBar().showMessage(message, ms)
        except Exception:  # noqa: BLE001 - janela sem barra de status (uso isolado)
            pass

    # ------------------------------------------------- operações pagas (Google): uma de cada vez
    def begin_paid(self, owner: str) -> bool:
        """Reserva exclusiva para operações pagas. Pesquisa e "Atualizar dados do Google" não rodam juntas:
        assim o limite mensal lido do banco já inclui tudo o que foi enviado antes. Só a thread da interface usa."""
        if getattr(self, "_paid_owner", ""):
            self.status("Outra operação paga do Google está em andamento; aguarde ela terminar.")
            return False
        self._paid_owner = owner
        return True

    def end_paid(self, owner: str) -> None:
        if getattr(self, "_paid_owner", "") == owner:
            self._paid_owner = ""

    def run(self, fn, name: str = "lead-finder-task", **callbacks) -> Task:
        task = Task(fn, self.dispatcher, name=name, **callbacks)
        self.tasks = [t for t in self.tasks if t.running()]
        self.tasks.append(task)
        return task.start()

    def shutdown(self) -> None:
        for t in self.tasks:
            t.cancel()
        self.db.stop()
