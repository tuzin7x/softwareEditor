"""Núcleo sem Qt: telefones e wa.me, geografia, palavras-chave, mensagens, consolidação, configurações e HTTP."""
from __future__ import annotations

import json
import threading
import time
from urllib.parse import unquote

import pytest

from mobilia.modules.lead_finder import dedupe, geo, keywords as kw, messages as msg, phone as ph
from mobilia.modules.lead_finder.config import Secrets, Settings, mask
from mobilia.modules.lead_finder.net import Cancelled, CancelToken, HttpClient, NetError, redact

from .conftest import fixture_json

# ------------------------------------------------------------------------------------------- telefone


@pytest.mark.parametrize("raw,kind,e164", [
    ("(11) 98765-4321", ph.MOBILE, "+5511987654321"),
    ("+55 11 98765-4321", ph.MOBILE, "+5511987654321"),
    ("011 98765 4321", ph.MOBILE, "+5511987654321"),
    ("0 21 11 3456 7890", ph.LANDLINE, "+551134567890"),       # código de operadora
    ("19 3200-0001", ph.LANDLINE, "+551932000001"),
    ("11 8765-4321", ph.MOBILE, "+5511987654321"),              # celular antigo: nono dígito
    ("0800 123 4567", ph.SERVICE, ""),
    ("4004-1234", ph.SERVICE, ""),
    ("20 3333-4444", ph.INVALID, ""),                           # DDD inexistente
    ("11 1234-5678", ph.INVALID, ""),
    ("", ph.INVALID, ""),
])
def test_phone_normalize(raw, kind, e164):
    info = ph.normalize(raw)
    assert info.kind == kind and info.e164 == e164


def test_phone_national_and_note():
    info = ph.normalize("11 8765-4321")
    assert info.national == "(11) 98765-4321" and "nono dígito" in info.note
    assert ph.normalize("1932000001").national == "(19) 3200-0001"


def test_wa_me_url_is_official_and_prefilled():
    url = ph.wa_me_url("(11) 98765-4321", "Olá! Tudo bem? & 100%")
    assert url.startswith("https://wa.me/5511987654321?text=")
    assert unquote(url.split("text=", 1)[1]) == "Olá! Tudo bem? & 100%"
    assert ph.wa_me_url("+55 11 98765-4321") == "https://wa.me/5511987654321"
    with pytest.raises(ValueError):
        ph.wa_me_url("0800 123 4567", "oi")
    with pytest.raises(ValueError):
        ph.wa_me_url("abc", "oi")


def test_numbers_in_official_whatsapp_links():
    html = ('<a href="https://wa.me/5519999000001?text=oi">fale</a>'
            '<a href="https://api.whatsapp.com/send?phone=551932000002&amp;text=x">x</a>'
            '<a href="https://example.com/5511987654321">não é link de whatsapp</a>')
    assert ph.numbers_in_wa_links(html) == ["+5519999000001", "+551932000002"]


# -------------------------------------------------------------------------------------------- geografia

def test_regions_cover_all_states_once():
    all_ufs = [u for _, ufs in geo.REGIONS.values() for u in ufs]
    assert sorted(all_ufs) == sorted(geo.UFS) and len(set(all_ufs)) == 27


def test_scope_ufs_and_fold():
    from mobilia.modules.lead_finder.models import GeoScope
    assert len(geo.scope_ufs(GeoScope("pais"))) == 27
    assert geo.scope_ufs(GeoScope("regiao", region="S")) == ["PR", "RS", "SC"]
    assert geo.scope_ufs(GeoScope("uf", ufs=["SP", "XX"])) == ["SP"]
    assert geo.scope_ufs(GeoScope("municipios", municipalities=[("Campinas", "SP"), ("Niterói", "RJ")])) == ["RJ", "SP"]
    assert geo.fold("  São  Paulo! ") == "sao paulo"
    assert geo.uf_from_text("são paulo") == "SP" and geo.uf_from_text("rj") == "RJ" and geo.uf_from_text("x") == ""


def test_municipalities_from_ibge_are_cached(http, transport, tmp_path):
    transport.add("GET", r"servicodados\.ibge\.gov\.br/api/v1/localidades/estados/SP/municipios",
                  (200, fixture_json("ibge_sp.json")))
    names = geo.fetch_municipalities(http, "sp", tmp_path)
    assert names[0] == "Campinas" and "São Paulo" in names
    assert geo.fetch_municipalities(http, "SP", tmp_path) == names      # do cache: sem nova requisição
    assert len(transport.requests) == 1
    assert geo.match_municipality("sao paulo", names) == "São Paulo"
    assert geo.match_municipality("Gotham", names) == ""
    with pytest.raises(ValueError):
        geo.fetch_municipalities(http, "XX", tmp_path)


# ---------------------------------------------------------------------------------------- palavras-chave

def test_all_requested_default_keywords_exist_and_have_rules():
    expected = {"Imobiliária", "Loja de imóveis", "Corretora de imóveis", "Agência imobiliária", "Imóveis à venda",
                "Venda de imóveis", "Aluguel de imóveis", "Administração de imóveis", "Consultoria imobiliária",
                "Corretagem imobiliária", "Empreendimentos imobiliários", "Lançamentos imobiliários", "Construtora",
                "Incorporadora", "Corretores de imóveis", "Imóveis de luxo", "Imóveis comerciais"}
    assert set(kw.DEFAULT_KEYWORDS) == expected
    for k in kw.DEFAULT_KEYWORDS:
        cnaes, tags, stem, _ = kw.rule(k)
        assert cnaes and all(c in kw.CNAE for c in cnaes), k
        assert tags or stem, k


def test_keyword_mappings():
    assert kw.cnaes_for(["Imobiliária", "Construtora"]) == ["6821801", "6821802", "6822600", "4120400"]
    tags, rx = kw.osm_filters(["Imobiliária", "Imóveis na planta"])
    assert "office=estate_agent" in tags and "imobili" in rx and "imoveis" in rx
    assert kw.custom_terms(["Imobiliária", "Imóveis na Planta"]) == ["imoveis na planta"]
    assert kw.classify("Qualquer", cnae="4110700") == "Incorporadora"
    assert kw.classify("Construtora Gama") == "Construtora"
    assert kw.classify("Alfa Negócios", hints=("estate_agent",)) == "Imobiliária"
    assert kw.classify("Padaria") == "Outro"
    assert kw.validate_custom("  imóveis   na planta ") == "imóveis na planta"
    for bad in ("", "ab", "<script>", "x" * 70):
        with pytest.raises(ValueError):
            kw.validate_custom(bad)


# ------------------------------------------------------------------------------------------- mensagens

def test_templates_render_deterministically_without_inventing():
    t = msg.BUILTIN_TEMPLATES[0]["body"]
    ctx = msg.Context(empresa="Alfa Exemplo Imóveis")
    a, b = msg.render(t, ctx), msg.render(t, ctx)
    assert a == b
    assert a == ("Olá! Tudo bem? Sou da 7x Intelligence. Desenvolvemos uma ferramenta de IA para transformar fotos "
                 "de imóveis em apresentações visuais e vídeos profissionais. Gostaria de mostrar uma demonstração "
                 "para a Alfa Exemplo Imóveis. Posso enviar?")
    assert "Sou Ana, da 7x" in msg.render(t, msg.Context(empresa="X", meu_nome="Ana"))
    assert msg.render(t, ctx, optout=True).endswith(msg.OPTOUT)
    assert "sua imobiliária" in msg.render(t, msg.Context())          # sem nome: genérico, nada inventado
    for tpl in msg.BUILTIN_TEMPLATES:
        msg.validate(tpl["body"])
        assert "{" not in msg.render(tpl["body"], ctx)
    assert {t["tone"] for t in msg.BUILTIN_TEMPLATES} == {"profissional", "direto"}


def test_template_validation_and_optional_segments():
    assert msg.render("Oi[[ de {cidade}]]!", msg.Context()) == "Oi!"
    assert msg.render("Oi[[ de {cidade}]]!", msg.Context(cidade="Santos")) == "Oi de Santos!"
    with pytest.raises(ValueError):
        msg.validate("Olá {fulano}")
    with pytest.raises(ValueError):
        msg.validate("Olá [[ aberto")


def test_ai_generation_uses_only_given_facts(http, transport):
    sent = {}

    def reply(req):
        sent["headers"] = {k.lower(): v for k, v in req.header_items()}
        sent["body"] = json.loads(req.data)
        return 200, {"content": [{"type": "text", "text": "Olá, equipe da Alfa! Podemos mostrar uma demonstração?"}],
                     "stop_reason": "end_turn"}
    transport.add("POST", r"api\.anthropic\.com/v1/messages", reply)
    out = msg.generate_with_ai(http, "sk-ant-teste-000000000000", "claude-opus-5-5",
                               msg.Context(empresa="Alfa", cidade="Campinas"), "direto", "base", "Imobiliária")
    assert out.startswith("Olá, equipe da Alfa")
    body = sent["body"]
    assert body["model"] == "claude-opus-5-5" and "não invente" in body["system"]
    assert "Alfa" in body["messages"][0]["content"] and "Campinas" in body["messages"][0]["content"]
    assert sent["headers"]["x-api-key"].startswith("sk-ant-") and sent["headers"]["anthropic-version"]
    with pytest.raises(NetError):
        msg.generate_with_ai(http, "", "m", msg.Context(), "direto", "x")


def test_ai_refusal_is_reported(http, transport):
    transport.add("POST", r"api\.anthropic\.com", (200, {"content": [], "stop_reason": "refusal"}))
    with pytest.raises(NetError, match="recusou"):
        msg.generate_with_ai(http, "k" * 20, "m", msg.Context(empresa="A"), "direto", "x")


# ----------------------------------------------------------------------------------------- consolidação

def test_dedupe_keys_and_merge_priority():
    assert dedupe.domain_of("https://www.Alfa.com.br/contato") == "alfa.com.br"
    assert dedupe.is_generic("instagram.com") and dedupe.is_generic("m.facebook.com")
    assert not dedupe.is_generic("alfa.com.br")
    assert dedupe.name_key("Imobiliária Alfa LTDA", "Campinas", "sp") == dedupe.name_key("imobiliaria alfa", "CAMPINAS", "SP")
    assert dedupe.name_key("Al", "X", "SP") == ""
    merged, origin = dedupe.merge([("osm", {"name": "Alfa OSM", "phone": "1"}, "2026-01-01"),
                                   ("cnpj_aberto", {"name": "ALFA OFICIAL"}, "2025-01-01"),
                                   ("manual", {"phone": "2"}, "2024-01-01")])
    assert merged == {"name": "ALFA OFICIAL", "phone": "2"} and origin == {"name": "cnpj_aberto", "phone": "manual"}


# ------------------------------------------------------------------------------------ configurações

def test_settings_roundtrip_and_type_safety(tmp_path):
    s = Settings(sender_name="Ana", google_max_requests_per_search=7)
    s.save(tmp_path)
    assert Settings.load(tmp_path).sender_name == "Ana"
    raw = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    raw.update(google_max_requests_per_search="muito", google_price_per_1000_usd=40, desconhecido=1)
    (tmp_path / "settings.json").write_text(json.dumps(raw), encoding="utf-8")
    loaded = Settings.load(tmp_path)
    assert loaded.google_max_requests_per_search == 20 and loaded.google_price_per_1000_usd == 40.0
    assert not any("key" in k or "secret" in k for k in raw)          # nenhuma chave em settings.json


def test_secrets_file_env_priority_and_mask(tmp_path):
    sec = Secrets(tmp_path, env={})
    assert sec.get("google_places") == "" and sec.origin("google_places") == ""
    sec.set("google_places", "AIzaFAKE000000000000000000")
    assert sec.get("google_places") == "AIzaFAKE000000000000000000" and sec.origin("google_places") == "arquivo"
    env = Secrets(tmp_path, env={"LEADFINDER_GOOGLE_API_KEY": "AIzaENV"})
    assert env.get("google_places") == "AIzaENV" and env.origin("google_places") == "ambiente"
    sec.set("google_places", "")
    assert sec.get("google_places") == ""
    assert mask("AIzaFAKE000000000000000000") == "AIza…000" and mask("") == ""


# -------------------------------------------------------------------------------------------- HTTP

def test_redact_hides_keys():
    text = redact("GET https://x.com/a?key=AIzaSECRET123&q=1 token=abc sk-ant-api03-SECRETSECRET AIzaSyD1234567890123")
    assert "SECRET" not in text and "abc" not in text and "1234567890123" not in text and "q=1" in text


def test_http_retries_on_429_then_succeeds(http, transport):
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        return (429, {"error": "lento"}, {"Retry-After": "1"}) if calls["n"] < 3 else (200, {"ok": True})
    transport.add("GET", r"example\.test/api", flaky)
    assert http.get_json("https://example.test/api") == {"ok": True}
    assert calls["n"] == 3


def test_http_non_retryable_error_has_clean_message(http, transport):
    transport.add("POST", r"example\.test/keyed", (403, {"error": {"message": "API key AIzaSyD1234567890123456 inválida"}}))
    with pytest.raises(NetError) as e:
        http.post_json("https://example.test/keyed?key=AIzaSECRETSECRETSECRET", {})
    assert e.value.status == 403 and "AIzaSyD1234567890123456" not in str(e.value) and "Acesso negado" in str(e.value)


def test_http_network_failure_is_friendly(http, transport):
    transport.add("GET", r"example\.test/down", OSError("connection refused"))
    with pytest.raises(NetError, match="Sem conexão"):
        http.get_json("https://example.test/down")


def test_http_cancel_returns_immediately(transport):
    gate = threading.Event()
    transport.add("GET", r"example\.test/slow", lambda req: (gate.wait(5), (200, {}))[1])
    client = HttpClient(transport=transport)
    tok = CancelToken()
    threading.Timer(0.1, tok.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        client.get_json("https://example.test/slow", ctx=tok)
    assert time.monotonic() - t0 < 2.0
    gate.set()


def test_http_rate_limit_per_host(transport):
    waits = []
    transport.add("GET", r"slow\.test", (200, {}))
    client = HttpClient(transport=transport, min_interval={"slow.test": 2.0}, sleeper=waits.append)
    client.get_json("https://slow.test/a"); client.get_json("https://slow.test/b")
    assert waits and 1.5 < waits[0] <= 2.0
