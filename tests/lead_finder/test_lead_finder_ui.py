"""Interface do Lead Finder (offscreen, sem internet): registro na janela do 7xAi, pesquisa em segundo plano,
tabela, botão OPEN WHATSAPP com revisão e confirmação, CRM, mensagens, painel e configurações."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import unquote

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QMainWindow  # noqa: E402

from mobilia.modules.lead_finder.models import Lead  # noqa: E402

from .conftest import fixture_json, wait_until  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "mobilia/modules/lead_finder"


# ------------------------------------------------------------------------------------------ registro

def test_module_is_registered_lazily_and_light():
    """Importar o pacote só registra a aba: nada da interface do módulo nem bibliotecas do motor."""
    code = ("import sys; import mobilia.modules.lead_finder as lf; from mobilia.ui import modules as m; "
            "s = m.get('lead_finder'); "
            "assert (s.id, s.title, s.order) == ('lead_finder', 'Lead Finder', 10), s; "
            "assert callable(s.factory); "
            "bad = [n for n in sys.modules if n.split('.')[0] in ('numpy', 'PIL', 'torch', 'cv2', 'diffusers') "
            "or n.startswith('PySide6.QtWebEngine') or n.startswith('mobilia.modules.lead_finder.ui')]; "
            "assert not bad, bad; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60,
                       env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"})
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_module_code_has_no_engine_imports_or_secrets():
    for f in sorted(PKG.rglob("*.py")):
        text = f.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            for n in names:
                assert n.split(".")[0] not in {"numpy", "PIL", "torch", "cv2", "diffusers", "requests"}, (f, n)
                assert not n.startswith("PySide6.QtWebEngine"), (f, n)
        assert "mobilia.design" not in text and "mobilia.video" not in text, f
        assert "AIzaSy" not in text and "sk-ant-api" not in text, f          # nenhuma chave no código
    assert not [p for p in PKG.rglob("*") if p.is_file() and p.suffix not in (".py", ".pyc")]  # sem 'datas' no spec


# --------------------------------------------------------------------------------------------- página

@pytest.fixture()
def page(qapp, tmp_path, transport, home):
    from mobilia.modules.lead_finder.net import HttpClient
    from mobilia.modules.lead_finder.ui.page import LeadFinderPage
    from mobilia.ui.widgets import StatusBar
    win = QMainWindow(); win.setStatusBar(StatusBar())
    http = HttpClient(transport=transport, sleeper=lambda s: None)
    p = LeadFinderPage(win, base=tmp_path / "lf", http=http)
    win.setCentralWidget(p); win.resize(1440, 900); win.show()
    p.opened = []
    p.ctx.open_url = p.opened.append
    p.on_show(); qapp.processEvents()
    yield p
    p.ctx.shutdown()
    win.close(); win.deleteLater(); qapp.processEvents()


def db(page, fn, *args):
    return page.ctx.db.call(fn, *args).result(timeout=10)


def seed(page):
    leads = [Lead("osm", "node/1", "Imobiliária Exemplo Alfa", category="Imobiliária", city="Campinas", uf="SP",
                  phone="19 3200-0001", whatsapp="19 99900-0001", website="https://alfa-exemplo.com.br"),
             Lead("cnpj_aberto", "11222333000181", "Beta Construtora Fictícia", category="Construtora",
                  city="Santos", uf="SP", phone="13 3222-1111", cnpj="11222333000181", situacao="ATIVA"),
             Lead("osm", "node/3", "Gama Imóveis de Teste", category="Imobiliária", city="Niterói", uf="RJ",
                  phone="21 98888-7777")]
    from mobilia.modules.lead_finder import ops
    db(page, ops.ingest_and_score, leads)
    rows, _ = db(page, lambda s: s.page(sort="name", desc=False))
    return {r["name"]: r["id"] for r in rows}


def leads_loaded(qapp, page, n):
    view = page.views["leads"]
    return wait_until(qapp, lambda: len(view.model.rows) == n)


def test_page_builds_all_tabs_and_is_native(page, qapp):
    assert list(page.nav.buttons) == ["search", "leads", "messages", "dashboard", "settings"]
    for key in page.nav.buttons:
        page.nav.buttons[key].click(); qapp.processEvents()
        assert page.current == key
    assert page.minimumSizeHint().height() < 500                       # cabe em 1280×720 (abas roláveis)
    from mobilia.ui import theme
    assert theme.ACCENT in page.styleSheet() or "rgba(205, 174, 118" in page.styleSheet()  # tokens do tema


def test_table_whatsapp_button_review_open_and_confirm(page, qapp):
    ids = seed(page)
    page.show_tab("leads")
    view = page.views["leads"]
    view.sort_key, view.sort_desc = "name", False
    view.load()
    assert leads_loaded(qapp, page, 3)
    model = view.model
    row = next(i for i, r in enumerate(model.rows) if r["name"] == "Imobiliária Exemplo Alfa")
    beta = next(i for i, r in enumerate(model.rows) if r["name"].startswith("Beta"))
    from mobilia.modules.lead_finder.ui.table import COL, can_whatsapp
    assert can_whatsapp(model.rows[row]) and not can_whatsapp(model.rows[beta])   # fixo sem confirmação: sem botão
    view.table.scrollTo(model.index(row, COL["whatsapp"]))
    rect = view.table.visualRect(model.index(row, COL["whatsapp"]))
    btn = view.wa_delegate.button_rect(rect)
    QTest.mouseClick(view.table.viewport(), Qt.LeftButton, Qt.NoModifier,
                     QPoint(int(btn.center().x()), int(btn.center().y())))
    assert wait_until(qapp, lambda: getattr(page.ctx, "last_dialog", None) is not None)
    dlg = page.ctx.last_dialog
    assert "Imobiliária Exemplo Alfa" in dlg.editor.toPlainText()        # personalizada com o nome real
    assert dlg.lead["wa_status"] == "confirmado"
    dlg.btn_open.click()
    assert wait_until(qapp, lambda: len(page.opened) == 1)
    url = page.opened[0]
    assert url.startswith("https://wa.me/5519999000001?text=")
    assert unquote(url.split("text=", 1)[1]) == dlg.editor.toPlainText().strip()
    lid = ids["Imobiliária Exemplo Alfa"]
    assert wait_until(qapp, lambda: any(e["kind"] == "whatsapp_aberto" for e in db(page, lambda s: s.get(lid))["events"]))
    d = db(page, lambda s: s.get(lid))
    assert d["status"] == "nao_contatado" and not any(e["kind"] == "mensagem_enviada" for e in d["events"])
    assert dlg.pages.currentIndex() == 1                                 # pergunta se enviou de fato
    dlg._confirm_sent()
    assert wait_until(qapp, lambda: db(page, lambda s: s.get(lid))["status"] == "contato_iniciado")


def test_whatsapp_for_unconfirmed_mobile_warns(page, qapp):
    ids = seed(page)
    from mobilia.modules.lead_finder.ui.dialogs import open_whatsapp
    page.ctx.last_dialog = None
    open_whatsapp(page.ctx, page, ids["Gama Imóveis de Teste"])
    assert wait_until(qapp, lambda: page.ctx.last_dialog is not None)
    dlg = page.ctx.last_dialog
    assert dlg.lead["wa_status"] == "provavel"
    warn = [w.text() for w in dlg.findChildren(type(dlg.counter)) if "não foi confirmado" in w.text()]
    assert warn
    dlg.editor.setPlainText("")
    dlg.btn_open.click(); qapp.processEvents()
    assert page.opened == [] and "Escreva" in dlg.error.text()          # sem texto não abre
    dlg.reject()


def test_do_not_contact_blocks_whatsapp(page, qapp):
    ids = seed(page)
    lid = ids["Imobiliária Exemplo Alfa"]
    db(page, lambda s: s.set_status(lid, "nao_contatar", "pedido"))
    from mobilia.modules.lead_finder.ui.dialogs import open_whatsapp
    page.ctx.last_dialog = None
    open_whatsapp(page.ctx, page, lid)
    wait_until(qapp, lambda: False, 0.3)
    assert page.ctx.last_dialog is None and page.opened == []
    page.show_tab("leads")
    view = page.views["leads"]
    view.f_hide.setChecked(False)
    assert leads_loaded(qapp, page, 3)
    from mobilia.modules.lead_finder.ui.table import can_whatsapp
    assert not can_whatsapp(next(r for r in view.model.rows if r["id"] == lid))


def test_detail_panel_crm_actions(page, qapp, no_modal):
    ids = seed(page)
    lid = ids["Gama Imóveis de Teste"]
    page._open_lead(lid)
    detail = page.views["leads"].detail
    assert wait_until(qapp, lambda: detail.lead is not None and detail.lead["id"] == lid)
    assert detail.btn_wa.isEnabled() and "Gama" in detail.title.text()
    detail.note_edit.setPlainText("Ligar para a gerente.")
    detail._add_note()
    assert wait_until(qapp, lambda: detail.notes.count() == 1)
    detail.status.setCurrentIndex(detail.status.findData("interessado"))
    assert wait_until(qapp, lambda: db(page, lambda s: s.get(lid))["status"] == "interessado")
    detail.rem_text.setText("enviar vídeo")
    detail._add_reminder()
    assert wait_until(qapp, lambda: detail.reminders.count() == 1)
    assert db(page, lambda s: s.get(lid))["next_followup"]
    detail._event("demo_enviada")
    assert wait_until(qapp, lambda: db(page, lambda s: s.get(lid))["status"] == "demo_enviada")
    detail.status.setCurrentIndex(detail.status.findData("nao_contatar"))   # confirmação aceita (no_modal)
    assert wait_until(qapp, lambda: db(page, lambda s: s.get(lid))["status"] == "nao_contatar")
    assert wait_until(qapp, lambda: not detail.btn_wa.isEnabled())


def test_search_runs_in_background_and_consolidates(page, qapp, transport):
    threads = []

    def reply(req):
        threads.append(threading.current_thread().name)
        return 200, fixture_json("overpass_campinas.json")
    transport.add("POST", r"/api/interpreter", reply)
    view = page.views["search"]
    assert view.uf_buttons["SP"].isChecked() and view.kw_buttons["Imobiliária"].isChecked()
    assert wait_until(qapp, lambda: view.source_rows["osm"][0].isChecked())       # disponibilidade em 2º plano
    assert not view.source_rows["google_places"][0].isEnabled()
    view.btn_search.click()
    assert wait_until(qapp, lambda: view.btn_search.isEnabled() and "Concluído" in view.step.text(), 15)
    assert threads and "MainThread" not in threads
    assert view.k_new.value == 3
    assert db(page, lambda s: s.page()[1]) == 3
    view.btn_search.click()                                              # repetir: nada duplica
    assert wait_until(qapp, lambda: view.btn_search.isEnabled() and "Concluído" in view.step.text()
                      and view.k_updated.value == 3, 15)
    assert db(page, lambda s: s.page()[1]) == 3 and view.k_new.value == 0
    assert db(page, lambda s: s.recent_searches(5))[0]["params"]["fontes"] == ["osm"]


def test_search_cancel(page, qapp, transport):
    gate = threading.Event()
    transport.add("POST", r"/api/interpreter", lambda req: (gate.wait(10), (200, {"elements": []}))[1])
    view = page.views["search"]
    assert wait_until(qapp, lambda: view.source_rows["osm"][0].isChecked())
    view.btn_search.click()
    assert wait_until(qapp, lambda: view.btn_cancel.isEnabled())
    view.btn_cancel.click()
    assert wait_until(qapp, lambda: "cancelada" in view.step.text(), 5)
    assert view.btn_search.isEnabled()
    gate.set()


def test_search_validation_messages(page, qapp):
    view = page.views["search"]
    view._set_all_kw(False)
    view.btn_search.click(); qapp.processEvents()
    assert "palavra-chave" in view.estimate.text()
    view._set_all_kw(True)
    view.scope.set_value("municipios", emit=True)
    assert "onde pesquisar" in view.estimate.text().lower()


def test_custom_keyword_saved(page, qapp):
    view = page.views["search"]
    view.kw_input.setText("Imóveis na planta")
    view.save_keyword()
    assert wait_until(qapp, lambda: "Imóveis na planta" in view.kw_buttons)
    assert view.kw_buttons["Imóveis na planta"].isChecked()
    assert db(page, lambda s: s.custom_keywords()) == ["Imóveis na planta"]


def test_dashboard_empty_and_with_data(page, qapp):
    page.show_tab("dashboard")
    dash = page.views["dashboard"]
    assert wait_until(qapp, lambda: dash.data is not None)
    assert dash.data["values"]["found"] == 0 and not dash.empty.isHidden()
    seed(page)
    dash.data = None
    dash.refresh()
    assert wait_until(qapp, lambda: dash.data is not None and dash.data["values"]["found"] == 3)
    assert dash.empty.isHidden() and len(dash.funnel.stages) == 7
    dash.grab()                                                           # desenha sem erro


def test_messages_compose_for_lead_and_templates(page, qapp):
    ids = seed(page)
    view = page.views["messages"]
    page.ctx.compose_for.emit(ids["Imobiliária Exemplo Alfa"])
    assert wait_until(qapp, lambda: view.lead is not None)
    assert page.current == "messages"
    assert "Imobiliária Exemplo Alfa" in view.editor.toPlainText()
    assert view.btn_wa.isEnabled() and not view.btn_ai.isVisible()      # IA desligada por padrão
    view.tone.set_value("direto", emit=True)
    assert view.current_template()["tone"] == "direto"
    view.generate_ai()
    assert "desativada" in view.info.text()


def test_settings_keep_keys_out_of_settings_file(page, qapp):
    view = page.views["settings"]
    page.show_tab("settings")
    view.g_key.setText("AIzaFAKEKEY0000000000000")
    view._save_google_key()
    view.g_enabled.setChecked(True)
    view.g_per_search.setValue(7)
    view.save()
    base = page.ctx.base
    settings = json.loads((base / "settings.json").read_text(encoding="utf-8"))
    assert settings["google_enabled"] is True and settings["google_max_requests_per_search"] == 7
    assert "AIzaFAKE" not in (base / "settings.json").read_text(encoding="utf-8")
    assert "AIzaFAKE" in (base / "secrets.json").read_text(encoding="utf-8")
    assert "AIza…000" in view.g_key_info.text() and view.g_key.text() == ""
    assert wait_until(qapp, lambda: page.views["search"].source_rows["google_places"][0].isEnabled())


def test_export_from_ui(page, qapp, tmp_path):
    seed(page)
    out = tmp_path / "exp.csv"
    page.views["leads"]._export(str(out))
    assert wait_until(qapp, lambda: out.exists())
    text = out.read_text(encoding="utf-8-sig")
    assert "Imobiliária Exemplo Alfa" in text and text.splitlines()[0].startswith("empresa;")


def test_studio_picker_links_project(page, qapp, home):
    from mobilia import runtime
    jobs = runtime.runtime_paths().jobs_dir
    p = jobs / "design_20261117-101010_mobiliar"; p.mkdir(parents=True)
    (p / "design.json").write_text(json.dumps({"funcao": "mobiliar", "ambiente": "quarto", "saidas": []}),
                                   encoding="utf-8")
    (p / "antes_depois_16x9.mp4").write_bytes(b"x")
    ids = seed(page)
    from mobilia.modules.lead_finder.ui.dialogs import StudioPickerDialog
    lid = ids["Gama Imóveis de Teste"]
    dlg = StudioPickerDialog(page.ctx, lid, page)
    assert wait_until(qapp, lambda: dlg.list.count() == 1)
    dlg.what.setCurrentIndex(1)                                         # o vídeo
    dlg._link()
    assert wait_until(qapp, lambda: len(db(page, lambda s: s.get(lid))["media"]) == 1)
    assert db(page, lambda s: s.get(lid))["media"][0]["kind"] == "video"


# --------------------------------------------------------------------------------- janela principal

def test_main_window_opens_lead_finder_tab(qapp, home, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    import mobilia.modules.lead_finder  # noqa: F401 - garante o registro
    from mobilia.ui import modules as m
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setenv("MOBILIA_WORKER", json.dumps([sys.executable, str(ROOT / "tests/fake_design_worker.py")]))
    assert m.get("lead_finder") is not None
    from mobilia.ui.app import MainWindow
    w = MainWindow(); w.resize(1280, 720); w.show(); qapp.processEvents()
    try:
        assert "lead_finder" in w.modules.nav.buttons
        w.modules.nav.buttons["lead_finder"].click(); qapp.processEvents()
        page = w.modules.pages["lead_finder"]
        from mobilia.modules.lead_finder.ui.page import LeadFinderPage
        assert isinstance(page, LeadFinderPage) and not [f for f in m.failed if f[0] == "lead_finder"]
        assert w.height() <= 760                                        # a janela não cresceu além da tela
        assert m.get("lead_finder").local is False                      # usa a internet
        assert not any(x.isVisible() for x in w.privacy_widgets)        # sem o selo "Processado no seu computador"
        w.modules.nav.buttons["design"].click(); qapp.processEvents()
        assert w.modules.current == "design" and w.btn_legacy.isVisible()
        assert all(x.isVisible() for x in w.privacy_widgets)
        page.ctx.shutdown()
    finally:
        w.close()
