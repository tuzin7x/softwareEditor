"""Overpass no teste real (Campinas): a consulta única da cidade falhou nos três servidores públicos (502/504/tempo
esgotado; 403) e uma pequena funcionou. Pedaços pequenos, troca automática de servidor, mensagem honesta (a
Overpass não tem chave) e um User-Agent descritivo."""
from __future__ import annotations

from urllib.parse import unquote_plus

from mobilia.modules.lead_finder.net import CancelToken, NetError, Response, _status_message
from mobilia.modules.lead_finder.search import run_search
from mobilia.modules.lead_finder.sources.osm import OVERPASS_ENDPOINTS, build_queries, endpoints

from .conftest import fixture_json
from .test_lead_finder_sources import Ctx, q, sources


def test_city_query_is_split_into_small_pieces():
    pieces = build_queries("SP", "Campinas", ["office=estate_agent", "office=real_estate"], "imobili", 30)
    assert len(pieces) == 5                                       # 2 etiquetas + nome em office/shop/craft
    for ql in pieces:
        assert ql.count("  nwr[") == 1 and "[timeout:30]" in ql and '["name"="Campinas"]' in ql


def test_fallback_across_public_servers(http, transport, tmp_path):
    hits = []

    def by_host(status):
        def reply(req):
            hits.append((req.full_url, unquote_plus(req.data.decode())))
            return status
        return reply
    transport.add("POST", r"overpass-api\.de", by_host((504, {"remark": "timeout"})))
    transport.add("POST", r"maps\.mail\.ru", by_host((403, "<html>forbidden</html>")))
    transport.add("POST", r"private\.coffee", by_host((200, fixture_json("overpass_campinas.json"))))
    transport.add("POST", r"openstreetmap\.fr", by_host((502, "")))
    srcs, _, _ = sources(http, tmp_path)
    ctx = Ctx()
    summary = run_search(ctx, q(), srcs)
    assert summary["osm"]["found"] == 3 and not summary["osm"]["errors"]
    assert any("private.coffee" in u for u, _ in hits)
    # o servidor que funcionou passa a ser o primeiro dos pedaços seguintes
    first_ok = next(i for i, (u, _) in enumerate(hits) if "private.coffee" in u)
    assert all("private.coffee" in u for u, _ in hits[first_ok:])
    ua = transport.requests[0].get_header("User-agent")
    assert ua.startswith("7xAi-LeadFinder/") and "Overpass" in ua and "@" not in ua


def test_all_servers_refusing_gives_honest_message(http, transport, tmp_path):
    transport.add("POST", r"/api/interpreter", (403, "<html>forbidden</html>"))
    srcs, _, _ = sources(http, tmp_path)
    ctx = Ctx()
    summary = run_search(ctx, q(), srcs)
    err = summary["osm"]["errors"][0]
    assert "recusaram ou limitaram" in err and "não usa chave" in err and "confira a chave" not in err
    assert "403" in err
    # dois pedaços seguidos sem nenhum servidor: não insiste nos outros pedaços
    assert len(transport.requests) <= 2 * len(OVERPASS_ENDPOINTS)


def test_endpoints_order_and_no_duplicates():
    eps = endpoints("https://meu.overpass/api/interpreter", OVERPASS_ENDPOINTS[2])
    assert eps[0] == OVERPASS_ENDPOINTS[2] and eps[1] == "https://meu.overpass/api/interpreter"
    assert len(eps) == len(set(eps)) == len(OVERPASS_ENDPOINTS) + 1


def test_403_message_mentions_key_only_when_a_key_was_sent(http, transport):
    r = Response(403, {"content-type": "text/html"}, b"")
    assert "confira a chave" in _status_message(r, "places.googleapis.com", keyed=True)
    msg = _status_message(r, "overpass-api.de", keyed=False)
    assert "chave" not in msg.replace("não usa chave", "") and "recusou ou limitou" in msg
    transport.add("GET", r"servicodados\.ibge", (403, ""))
    try:
        http.get_json("https://servicodados.ibge.gov.br/api/v1/x", ctx=CancelToken())
    except NetError as e:
        assert "confira a chave" not in str(e) and e.status == 403
    else:
        raise AssertionError("esperava NetError")
