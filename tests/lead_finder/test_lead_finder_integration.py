"""Integração do Lead Finder com a janela principal do 7xAi: pasta de dados, persistência entre execuções e
banco fora da thread da interface. Sem internet (nenhuma pesquisa é feita)."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_data_dir_is_inside_app_data_on_windows(tmp_path, monkeypatch):
    """No Windows o banco fica em %LOCALAPPDATA%\\MobiliaStudio\\lead_finder (a mesma pasta do motor)."""
    from mobilia.modules.lead_finder import config
    from mobilia.modules.lead_finder.storage import DB_NAME
    monkeypatch.delenv("MOBILIA_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setattr(sys, "platform", "win32")
    assert config.data_dir() == tmp_path / "Local" / "MobiliaStudio" / "lead_finder"
    assert DB_NAME.endswith(".sqlite3")


def _open_window(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    import mobilia.modules.lead_finder  # noqa: F401 - garante o registro
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setenv("MOBILIA_WORKER", json.dumps([sys.executable, str(ROOT / "tests/fake_design_worker.py")]))
    from mobilia.ui.app import MainWindow
    w = MainWindow(); w.resize(1280, 720); w.show(); qapp.processEvents()
    w.modules.show("lead_finder"); qapp.processEvents()
    return w, w.modules.pages["lead_finder"]


def test_leads_persist_across_restart_and_db_runs_off_ui_thread(qapp, home, monkeypatch):
    from mobilia.modules.lead_finder.models import Lead

    ui_thread = threading.get_ident()
    w, page = _open_window(qapp, monkeypatch)
    try:
        assert page.ctx.base == home / "lead_finder"
        seen = {}

        def add(store):
            seen["thread"] = threading.get_ident()
            return store.add_manual(Lead(source="manual", source_ref="", name="Imobiliária Exemplo Fictícia",
                                         city="Campinas", uf="SP", phone="(19) 3200-0000"))
        lead_id = page.ctx.db.call(add).result(timeout=10)
        assert lead_id and seen["thread"] != ui_thread                 # o banco nunca roda na thread da UI
        page.ctx.shutdown()
    finally:
        w.close()
    assert (home / "lead_finder").is_dir()
    assert any(p.suffix == ".sqlite3" for p in (home / "lead_finder").iterdir())

    # "reinício": nova janela, novo contexto, mesmo MOBILIA_HOME
    w2, page2 = _open_window(qapp, monkeypatch)
    try:
        assert page2 is not page
        rows, total = page2.ctx.db.call(lambda s: s.page({}, limit=10)).result(timeout=10)
        assert total == 1 and rows[0]["name"] == "Imobiliária Exemplo Fictícia"
        page2.ctx.shutdown()
    finally:
        w2.close()
