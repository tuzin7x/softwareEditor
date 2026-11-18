"""Fixtures dos testes do Lead Finder: sem internet (transporte HTTP falso com respostas gravadas e fictícias),
dados do app numa pasta temporária (``MOBILIA_HOME``) e Qt sem tela."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FIX = Path(__file__).parent / "fixtures"


def fixture_json(name: str):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


class FakeTransport:
    """Substitui a rede: cada rota é (método, regex da URL) -> (status, corpo, cabeçalhos) ou função(req)."""

    def __init__(self):
        self.routes: list[tuple[str, re.Pattern, object]] = []
        self.requests: list = []

    def add(self, method: str, pattern: str, response) -> FakeTransport:
        self.routes.append((method.upper(), re.compile(pattern), response))
        return self

    def __call__(self, req, timeout, max_bytes):
        from mobilia.modules.lead_finder.net import Response
        self.requests.append(req)
        for method, pat, resp in self.routes:
            if method == req.get_method() and pat.search(req.full_url):
                if callable(resp):
                    resp = resp(req)
                if isinstance(resp, BaseException):
                    raise resp
                status, body, headers = resp if len(resp) == 3 else (*resp, {})
                if not isinstance(body, (bytes, bytearray)):
                    body = json.dumps(body).encode("utf-8") if not isinstance(body, str) else body.encode("utf-8")
                headers = {k.lower(): v for k, v in {"content-type": "application/json", **headers}.items()}
                return Response(status, headers, bytes(body))
        raise AssertionError(f"requisição inesperada (o teste não pode usar a internet): {req.get_method()} "
                             f"{req.full_url}")


@pytest.fixture()
def transport():
    return FakeTransport()


@pytest.fixture()
def http(transport):
    from mobilia.modules.lead_finder.net import HttpClient
    return HttpClient(transport=transport, sleeper=lambda s: None)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("MOBILIA_HOME", str(h))
    monkeypatch.delenv("LEADFINDER_GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return h


@pytest.fixture()
def store(tmp_path):
    from mobilia.modules.lead_finder.storage import Store
    s = Store(tmp_path / "leads.sqlite3")
    yield s
    s.close()


@pytest.fixture(scope="session")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from mobilia.ui import theme
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion"); app.setStyleSheet(theme.QSS)
    yield app


def wait_until(app, cond, timeout: float = 10.0) -> bool:
    """Processa eventos do Qt até ``cond()`` ser verdadeira (as tarefas rodam em threads)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(cond())


@pytest.fixture()
def no_modal(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: 0))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: 0))
