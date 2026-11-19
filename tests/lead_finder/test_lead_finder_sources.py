"""Fontes autorizadas com respostas gravadas (fictícias): OpenStreetMap, Google Places, base aberta de CNPJ,
BrasilAPI, análise de site com robots.txt e a orquestração da pesquisa com limites de custo."""
from __future__ import annotations

import json
import zipfile
from urllib.parse import unquote_plus

import pytest

from mobilia.modules.lead_finder.config import Secrets, Settings
from mobilia.modules.lead_finder.models import GeoScope, SearchQuery
from mobilia.modules.lead_finder.net import Cancelled, CancelToken
from mobilia.modules.lead_finder.search import Budget, run_search
from mobilia.modules.lead_finder.site_check import analyze_html, check_site
from mobilia.modules.lead_finder.sources import build
from mobilia.modules.lead_finder.sources.brasilapi import format_cnpj, valid_cnpj
from mobilia.modules.lead_finder.sources.cnpj_open import import_open_data, index_info
from mobilia.modules.lead_finder.sources.osm import accent_regex, build_query

from .conftest import fixture_json


class Ctx(CancelToken):
    """Contexto de tarefa sem Qt (coleta o que a pesquisa emite)."""

    def __init__(self):
        super().__init__()
        self.items, self.messages = [], []

    def emit(self, item):
        self.items.append(item)

    def progress(self, frac, msg=""):
        self.messages.append((frac, msg))


def sources(http, tmp_path, **settings):
    s = Settings(**settings)
    sec = Secrets(tmp_path, env={})
    return build(http, s, sec, tmp_path), s, sec


def q(kind="municipios", keywords=("Imobiliária",), srcs=("osm",), **kw):
    scope = GeoScope(kind=kind, municipalities=[("Campinas", "SP")] if kind == "municipios" else [],
                     ufs=kw.pop("ufs", ["SP"]), region=kw.pop("region", ""))
    return SearchQuery(scope=scope, keywords=list(keywords), sources=list(srcs), **kw)


# ----------------------------------------------------------------------------------------------- OSM

def test_overpass_query_is_area_scoped_and_accent_tolerant():
    ql = build_query("SP", "Campinas", ["office=estate_agent"], "imobili")
    assert 'area["ISO3166-2"="BR-SP"]' in ql and '["name"="Campinas"]' in ql and "map_to_area->.a" in ql
    assert 'nwr["office"="estate_agent"](area.a)' in ql and "out center tags" in ql
    assert accent_regex("imove") == "[ií]m[oóôõ]v[eéê]"
    assert '"Sao \\"X\\""' in build_query("SP", 'Sao "X"', [], "")       # aspas escapadas


def test_osm_fetch_parses_and_dedupes(http, transport, tmp_path):
    body = {}

    def reply(req):
        body["ql"] = unquote_plus(req.data.decode().split("data=", 1)[1])
        return 200, fixture_json("overpass_campinas.json")
    transport.add("POST", r"/api/interpreter", reply)
    srcs, _, _ = sources(http, tmp_path)
    osm = srcs["osm"]
    query = q()
    units = osm.plan(query)
    assert [u.label for u in units] == ["Campinas - SP"] and units[0].cost_units == 0
    leads = list(osm.fetch(units[0], query, CancelToken()))
    assert "Campinas" in body["ql"]
    assert [lead.name for lead in leads] == ["Imobiliária Exemplo Alfa", "Beta Imóveis de Teste",
                                             "Construtora Gama Fictícia"]
    alfa, beta, gama = leads
    assert alfa.source_ref == "node/1000001" and alfa.whatsapp == "+55 19 99900-0001"
    assert alfa.address == "Rua Fictícia 100, Centro" and alfa.cep == "13000000" and alfa.uf == "SP"
    assert beta.phone == "(19) 99900-0002" and beta.city == "Campinas"          # 1º de "a;b"; cidade da área
    assert gama.category == "Construtora" and alfa.category == "Imobiliária"
    assert osm.policy.exportable and "OpenStreetMap" in osm.policy.attribution


def test_osm_plan_by_state_and_region(http, tmp_path):
    srcs, _, _ = sources(http, tmp_path)
    assert [u.params["uf"] for u in srcs["osm"].plan(q("regiao", region="S"))] == ["PR", "RS", "SC"]
    assert len(srcs["osm"].plan(q("pais"))) == 27
    assert srcs["osm"].estimate(q("uf", ufs=["SP", "RJ"])).cost_usd == 0


# -------------------------------------------------------------------------------------------- Google

def test_google_requires_key_and_enable(http, tmp_path):
    srcs, s, sec = sources(http, tmp_path)
    assert srcs["google_places"].availability()[0] is False
    s.google_enabled = True
    assert "chave" in srcs["google_places"].availability()[1].lower()
    sec.set("google_places", "AIzaFAKEKEY0000000000000")
    assert srcs["google_places"].availability()[0] is True


def test_google_text_search_parse_mask_and_budget(http, transport, tmp_path):
    seen = []

    def reply(req):
        seen.append((json.loads(req.data), {k.lower(): v for k, v in req.header_items()}))
        return 200, fixture_json("google_textsearch.json")
    transport.add("POST", r"places\.googleapis\.com/v1/places:searchText", reply)
    srcs, s, sec = sources(http, tmp_path, google_enabled=True, google_max_pages=2)
    sec.set("google_places", "AIzaFAKEKEY0000000000000")
    g = srcs["google_places"]
    query = q(srcs=("google_places",))
    unit = g.plan(query)[0]
    assert unit.params["text"] == "Imobiliária em Campinas - SP, Brasil" and unit.cost_units == 2
    budget = Budget(per_search=1, monthly_cap=100, used_month=0)
    leads = list(g.fetch(unit, query, CancelToken(), budget=budget))
    assert len(seen) == 1 and budget.denied == 1                      # 2ª página barrada pelo limite
    body, headers = seen[0]
    assert body["includedType"] == "real_estate_agency" and body["regionCode"] == "BR"
    assert headers["x-goog-api-key"] == "AIzaFAKEKEY0000000000000"
    assert "places.nationalPhoneNumber" in headers["x-goog-fieldmask"] and "nextPageToken" in headers["x-goog-fieldmask"]
    assert [lead.name for lead in leads] == ["Alfa Exemplo Imóveis", "Delta Negócios Imobiliários"]  # fechada fora
    assert leads[0].city == "Campinas" and leads[0].uf == "SP" and leads[0].cep == "13000000"
    assert g.policy.store_days == 30 and not g.policy.exportable


def test_google_estimate_respects_caps(http, tmp_path):
    srcs, s, _ = sources(http, tmp_path, google_max_requests_per_search=5, google_price_per_1000_usd=35.0)
    est = srcs["google_places"].estimate(q("uf", keywords=("Imobiliária", "Construtora", "Incorporadora"),
                                           ufs=["SP", "RJ"], srcs=("google_places",)))
    assert est.units == 6 and est.paid_requests == 5 and est.cost_usd == pytest.approx(5 * 35 / 1000)
    assert "puladas" in est.note


# ----------------------------------------------------------------------------------------- BrasilAPI

def test_cnpj_check_digits():
    assert valid_cnpj("11.222.333/0001-81") and valid_cnpj("00000000000191")
    assert not valid_cnpj("11222333000182") and not valid_cnpj("11111111111111") and not valid_cnpj("123")
    assert format_cnpj("11222333000181") == "11.222.333/0001-81"


def test_brasilapi_lookup_minimizes_personal_data(http, transport, tmp_path):
    transport.add("GET", r"brasilapi\.com\.br/api/cnpj/v1/11222333000181", (200, fixture_json("brasilapi_cnpj.json")))
    srcs, _, _ = sources(http, tmp_path)
    lead = srcs["brasilapi"].lookup("11.222.333/0001-81")
    assert lead.name == "Alfa Exemplo Imoveis" and lead.legal_name.startswith("ALFA EXEMPLO")
    assert lead.situacao == "ATIVA" and lead.cnae == "6821801" and lead.category == "Imobiliária"
    assert lead.phone == "1932000001" and lead.uf == "SP" and lead.city == "Campinas"
    dumped = json.dumps(lead.to_dict(), ensure_ascii=False)
    assert "PESSOA FICTICIA" not in dumped and "contato@" not in dumped and "50000" not in dumped
    with pytest.raises(Exception, match="inválido"):
        srcs["brasilapi"].lookup("11222333000100")


# -------------------------------------------------------------------------------------- CNPJ aberto

def _estab(basico, ordem, dv, fantasia, situacao, cnae, sec, uf, muni, ddd="19", tel="32000001", email="x@y.com"):
    cols = [basico, ordem, dv, "1", fantasia, situacao, "20200101", "00", "", "", "20150310", cnae, sec, "RUA",
            "FICTICIA", "10", "SALA 2", "CENTRO", "13000000", uf, muni, ddd, tel, "", "", "", "", email, "", ""]
    return ";".join(f'"{c}"' for c in cols)


@pytest.fixture()
def receita_folder(tmp_path):
    d = tmp_path / "receita"; d.mkdir()
    rows = [
        _estab("11222333", "0001", "81", "ALFA EXEMPLO IMOVEIS", "02", "6821801", "", "SP", "6291"),
        _estab("22333444", "0001", "00", "", "02", "4711302", "6822600,4110700", "SP", "6291"),   # CNAE secundário
        _estab("33444555", "0001", "00", "PADARIA FICTICIA", "02", "1091101", "", "SP", "6291"),   # fora do setor
        _estab("44555666", "0001", "00", "IMOB BAIXADA", "08", "6821801", "", "SP", "6291"),       # baixada
        _estab("55666777", "0001", "00", "CARIOCA IMOVEIS", "02", "6821802", "", "RJ", "6001"),
        _estab("66777888", "0001", "00", "", "02", "6821801", "", "SP", "7071"),                   # MEI
    ]
    with zipfile.ZipFile(d / "Estabelecimentos0.zip", "w") as z:
        z.writestr("K3241.K03200Y0.D41109.ESTABELE", ("\n".join(rows) + "\n").encode("latin-1"))
    with zipfile.ZipFile(d / "Empresas0.zip", "w") as z:
        z.writestr("K3241.K03200Y0.D41109.EMPRECSV", ('"11222333";"ALFA EXEMPLO NEGOCIOS IMOBILIARIOS LTDA";"2062";"49";"1000,00";"01";""\n'
                                                      '"22333444";"BETA SUPERMERCADO E IMOVEIS LTDA";"2062";"49";"0";"01";""\n'
                                                      '"66777888";"JOAO FICTICIO DA SILVA";"2135";"50";"0";"01";""\n'
                                                      ).encode("latin-1"))
    with zipfile.ZipFile(d / "Municipios.zip", "w") as z:
        z.writestr("F.K03200$Z.D41109.MUNICCSV", '"6291";"CAMPINAS"\n"6001";"RIO DE JANEIRO"\n"7071";"SANTOS"\n'
                   .encode("latin-1"))
    return d


def test_cnpj_import_keeps_only_active_real_estate(receita_folder, tmp_path):
    idx = tmp_path / "data" / "cnpj_index.sqlite3"
    ctx = Ctx()
    rep = import_open_data(receita_folder, idx, ["SP"], ctx, ctx.progress)
    assert rep.rows_read == 6 and rep.kept == 3 and rep.with_legal_name == 3
    info = index_info(idx)
    assert info["rows"] == 3 and info["ufs"] == "SP"
    import sqlite3
    con = sqlite3.connect(idx)
    cols = [r[1] for r in con.execute("PRAGMA table_info(estab)")]
    assert "email" not in cols                                         # minimização: e-mail não é importado
    con.close()
    # reimportar outra UF não apaga SP
    rep2 = import_open_data(receita_folder, idx, ["RJ"], Ctx(), None)
    assert rep2.kept == 1 and index_info(idx)["rows"] == 4 and index_info(idx)["ufs"] == "RJ,SP"


def test_cnpj_import_cancel_keeps_previous_index(receita_folder, tmp_path):
    idx = tmp_path / "cnpj_index.sqlite3"
    import_open_data(receita_folder, idx, ["SP"], Ctx(), None)
    ctx = Ctx(); ctx.cancel()
    with pytest.raises(Cancelled):
        import_open_data(receita_folder, idx, [], ctx, None)
    assert index_info(idx)["rows"] == 3


def test_cnpj_search_filters(receita_folder, tmp_path, http):
    import_open_data(receita_folder, tmp_path / "cnpj_index.sqlite3", [], Ctx(), None)
    srcs, _, _ = sources(http, tmp_path)
    src = srcs["cnpj_aberto"]
    assert src.availability()[0]
    query = q(srcs=("cnpj_aberto",))
    leads = list(src.fetch(src.plan(query)[0], query, Ctx()))
    assert sorted(lead.cnpj for lead in leads) == ["11222333000181", "22333444000100"]
    alfa = next(lead for lead in leads if lead.cnpj == "11222333000181")
    assert alfa.name == "Alfa Exemplo Imoveis" and alfa.phone == "1932000001" and alfa.situacao == "ATIVA"
    beta = next(lead for lead in leads if lead.cnpj == "22333444000100")
    assert beta.name.startswith("Beta Supermercado") and beta.category == "Administradora de imóveis"
    # empresário individual só com a opção marcada
    q2 = q("uf", srcs=("cnpj_aberto",), ufs=["SP"], include_individuals=True)
    names = [lead.name for lead in src.fetch(src.plan(q2)[0], q2, Ctx())]
    assert "Joao Ficticio Da Silva" in names
    q3 = q("uf", srcs=("cnpj_aberto",), ufs=["RJ"], keywords=("Construtora",))
    assert list(src.fetch(src.plan(q3)[0], q3, Ctx())) == []


def test_cnpj_source_unavailable_without_index(http, tmp_path):
    srcs, _, _ = sources(http, tmp_path)
    ok, why = srcs["cnpj_aberto"].availability()
    assert not ok and "Importe" in why


# ------------------------------------------------------------------------------------------- pesquisa

def test_run_search_emits_batches_and_skips_unavailable(http, transport, tmp_path):
    transport.add("POST", r"/api/interpreter", (200, fixture_json("overpass_campinas.json")))
    srcs, _, _ = sources(http, tmp_path)
    ctx = Ctx()
    summary = run_search(ctx, q(srcs=("osm", "google_places", "cnpj_aberto")), srcs)
    assert summary["osm"]["found"] == 3 and summary["google_places"]["skipped"] and summary["cnpj_aberto"]["skipped"]
    kinds = [i[0] for i in ctx.items]
    assert kinds.count("batch") == 1 and kinds.count("skipped") == 2
    assert ctx.messages[-1][0] == 1.0


def test_run_search_unit_error_does_not_stop_other_units(http, transport, tmp_path):
    calls = {"n": 0}

    def reply(req):
        calls["n"] += 1
        return (504, {"remark": "timeout"}) if "BR-RJ" in unquote_plus(req.data.decode()) else \
            (200, fixture_json("overpass_campinas.json"))
    transport.add("POST", r"/api/interpreter", reply)
    srcs, _, _ = sources(http, tmp_path)
    ctx = Ctx()
    summary = run_search(ctx, q("uf", ufs=["RJ", "SP"]), srcs)
    assert summary["osm"]["found"] == 3 and len(summary["osm"]["errors"]) == 1
    assert any(i[0] == "unit_error" for i in ctx.items)


def test_run_search_google_budget_and_usage(http, transport, tmp_path):
    transport.add("POST", r"places\.googleapis\.com", (200, {**fixture_json("google_textsearch.json"),
                                                             "nextPageToken": None}))
    srcs, s, sec = sources(http, tmp_path, google_enabled=True, google_max_requests_per_search=10,
                           google_monthly_request_cap=12)
    sec.set("google_places", "AIzaFAKEKEY0000000000000")
    ctx = Ctx()
    query = q("uf", ufs=["SP", "RJ", "MG"], keywords=("Imobiliária", "Construtora"), srcs=("google_places",))
    summary = run_search(ctx, query, srcs, google_used_month=10)      # restam 2 no mês
    assert len(transport.requests) == 2 and summary["google_places"]["paid"] == 2
    usage = [i for i in ctx.items if i[0] == "usage"]
    assert sum(u[2] for u in usage) == 2 and summary["google_places"]["skipped"].startswith("Limite de custo")


def test_run_search_cancel(http, transport, tmp_path):
    transport.add("POST", r"/api/interpreter", (200, fixture_json("overpass_campinas.json")))
    srcs, _, _ = sources(http, tmp_path)
    ctx = Ctx(); ctx.cancel()
    with pytest.raises(Cancelled):
        run_search(ctx, q("pais"), srcs)
    assert transport.requests == []


# ------------------------------------------------------------------------------------------ site

PAGE = """<html><head><title>Alfa</title><script>var imovel_vazio = 1;</script></head><body>
<h1>Imóveis à venda e para alugar</h1><p>Apartamento 2 quartos, 70 m². Casa com suíte. Imóvel desocupado,
pronto para morar. Lançamento na planta!</p><a href="https://wa.me/5519999000001?text=Oi">WhatsApp</a></body></html>"""


def test_analyze_html_counts_terms_and_whatsapp():
    r = analyze_html(PAGE, "https://alfa-exemplo.com.br/", ["+5519999000001"])
    assert r["ok"] and r["whatsapp_match"] == "+5519999000001"
    assert r["totals"]["listings"] >= 4 and r["totals"]["empty"] >= 2 and r["totals"]["launch"] >= 1
    assert "imovel_vazio" not in json.dumps(r["terms"])                 # script ignorado
    assert "PAGE" not in r and len(json.dumps(r)) < 2000                 # não guarda o conteúdo da página


def test_check_site_respects_robots(http, transport):
    transport.add("GET", r"bloqueado\.test/robots\.txt", (200, "User-agent: *\nDisallow: /", {"content-type": "text/plain"}))
    r = check_site(http, "https://bloqueado.test", [])
    assert not r["ok"] and "robots.txt" in r["summary"]
    assert [req.full_url for req in transport.requests] == ["https://bloqueado.test/robots.txt"]


def test_check_site_reads_home_page_when_allowed(http, transport):
    transport.add("GET", r"alfa-exemplo\.test/robots\.txt", (404, ""))
    transport.add("GET", r"alfa-exemplo\.test/$", (200, PAGE, {"content-type": "text/html; charset=utf-8"}))
    r = check_site(http, "alfa-exemplo.test", ["(19) 99900-0001"])
    assert r["ok"] and r["whatsapp_match"] == "+5519999000001"


def test_check_site_skips_social_profiles(http):
    r = check_site(http, "https://instagram.com/alfa", [])
    assert not r["ok"] and "rede social" in r["summary"]
