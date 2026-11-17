"""Banco local: migrações, consolidação de fontes, CRM (estados, histórico, notas, lembretes), não contatar,
prazos de guarda por licença, exportação, painel, pontuação e vínculo com o AI Studio."""
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from mobilia.modules.lead_finder import ops
from mobilia.modules.lead_finder.export import write_csv
from mobilia.modules.lead_finder.models import Lead
from mobilia.modules.lead_finder.scoring import score_lead
from mobilia.modules.lead_finder.storage import MIGRATIONS, SCHEMA_VERSION, Store


def L(source="osm", ref="node/1", name="Imobiliária Alfa", **kw):
    base = dict(city="Campinas", uf="SP", category="Imobiliária")
    base.update(kw)
    return Lead(source=source, source_ref=ref, name=name, **base)


def only(store) -> dict:
    rows, total = store.page()
    assert total == 1
    return store.get(rows[0]["id"])


# ---------------------------------------------------------------------------------------- migrações

def test_fresh_database_is_at_latest_version(store):
    assert store.version == SCHEMA_VERSION == len(MIGRATIONS)
    tables = {r[0] for r in store.con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"leads", "lead_sources", "events", "notes", "reminders", "suppression", "templates", "keywords",
            "api_usage", "searches", "media_links"} <= tables


def test_migration_from_v1_keeps_data(tmp_path):
    path = tmp_path / "old.sqlite3"
    con = sqlite3.connect(path)
    con.executescript(MIGRATIONS[0])
    con.execute("PRAGMA user_version=1")
    con.execute("INSERT INTO leads(name, created_at) VALUES ('Antiga', '2026-01-01')")
    con.commit(); con.close()
    s = Store(path)
    assert s.version == SCHEMA_VERSION
    assert s.page()[1] == 1 and s.page()[0][0]["name"] == "Antiga"
    s.con.execute("SELECT * FROM media_links")
    s.close()
    assert Store(path).version == SCHEMA_VERSION                       # idempotente


def test_newer_database_is_refused(tmp_path):
    path = tmp_path / "future.sqlite3"
    con = sqlite3.connect(path); con.execute(f"PRAGMA user_version={SCHEMA_VERSION + 5}"); con.close()
    with pytest.raises(RuntimeError, match="mais nova"):
        Store(path)


# ------------------------------------------------------------------------------------- consolidação

def test_same_company_from_three_sources_becomes_one_record(store):
    s1 = store.ingest([L(phone="(19) 99900-0001", website="https://www.alfa-exemplo.com.br")])
    s2 = store.ingest([L("cnpj_aberto", "11222333000181", "ALFA EXEMPLO IMOVEIS", phone="19999000001",
                         cnpj="11222333000181", situacao="ATIVA")])
    s3 = store.ingest([L("google_places", "ChIJTESTE01", "Alfa Exemplo Imóveis", phone="+55 19 3200-0001",
                         website="alfa-exemplo.com.br/contato")])
    assert (s1["novos"], s2["consolidados"], s3["consolidados"]) == (1, 1, 1)
    d = only(store)
    assert d["sources"] == "cnpj_aberto,google_places,osm" and len(d["source_records"]) == 3
    assert d["cnpj"] == "11222333000181" and d["origin"]["name"] == "cnpj_aberto"
    assert d["restricted"] == 1                                         # tem dado do Google
    assert [e["kind"] for e in d["events"]].count("fonte") == 2


def test_same_source_record_updates_instead_of_duplicating(store):
    store.ingest([L(phone="19 3200-0001")])
    st = store.ingest([L(phone="19 3200-0009")])
    assert st == {"novos": 0, "consolidados": 0, "atualizados": 1, "ignorados_nao_contatar": 0}
    assert only(store)["phone_e164"] == "+551932000009"


def test_different_cnpj_never_merges(store):
    store.ingest([L("cnpj_aberto", "11222333000181", "Alfa", cnpj="11222333000181", phone="1932000001")])
    store.ingest([L("cnpj_aberto", "00000000000191", "Outra", cnpj="00000000000191", phone="1932000001")])
    assert store.page()[1] == 2


def test_generic_domains_do_not_merge(store):
    store.ingest([L(ref="node/1", name="Alfa", website="https://instagram.com/alfa", city="A")])
    store.ingest([L(ref="node/2", name="Beta", website="https://instagram.com/beta", city="B")])
    assert store.page()[1] == 2


def test_whatsapp_status_rules(store):
    store.ingest([L(ref="node/1", name="Celular", phone="19 99900-0001", city="A")])
    store.ingest([L(ref="node/2", name="Fixo", phone="19 3200-0002", city="B")])
    store.ingest([L(ref="node/3", name="Confirmado", phone="19 3200-0003", whatsapp="19 99900-0003", city="C")])
    by = {r["name"]: r for r in store.page()[0]}
    assert (by["Celular"]["wa_status"], by["Celular"]["wa_number"]) == ("provavel", "+5519999000001")
    assert (by["Fixo"]["wa_status"], by["Fixo"]["wa_number"]) == ("nao_verificado", "")
    assert by["Confirmado"]["wa_status"] == "confirmado"
    store.set_whatsapp(by["Fixo"]["id"], "confirmado")                  # a pessoa conversou: fixo com WhatsApp
    assert store.get(by["Fixo"]["id"])["wa_number"] == "+551932000002"
    store.set_whatsapp(by["Celular"]["id"], "sem_whatsapp")
    assert store.get(by["Celular"]["id"])["wa_number"] == ""


# ---------------------------------------------------------------------------------------------- CRM

def test_opening_whatsapp_is_not_sending(store):
    store.ingest([L(phone="19 99900-0001")])
    lid = only(store)["id"]
    store.add_event(lid, "whatsapp_aberto", "Apresentação")
    d = store.get(lid)
    assert d["status"] == "nao_contatado" and d["last_contact"] == ""
    store.add_event(lid, "mensagem_enviada", "Apresentação")
    d = store.get(lid)
    assert d["status"] == "contato_iniciado" and d["last_contact"]
    kinds = [e["kind"] for e in d["events"]]
    assert "whatsapp_aberto" in kinds and "mensagem_enviada" in kinds
    store.add_event(lid, "resposta_recebida")
    assert store.get(lid)["status"] == "aguardando"
    store.add_event(lid, "demo_enviada")
    assert store.get(lid)["status"] == "demo_enviada"


def test_all_statuses_and_history(store):
    from mobilia.modules.lead_finder.models import STATUS_KEYS
    store.ingest([L(phone="19 99900-0001")])
    lid = only(store)["id"]
    for st in STATUS_KEYS:
        store.set_status(lid, st)
    assert store.get(lid)["status"] == "nao_contatar"
    with pytest.raises(ValueError):
        store.set_status(lid, "inventado")
    assert sum(1 for e in store.get(lid)["events"] if e["kind"] == "status") == len(STATUS_KEYS) - 1


def test_do_not_contact_blocks_and_survives_new_searches(store):
    store.ingest([L(phone="19 99900-0001", cnpj="11222333000181", website="https://alfa-exemplo.com.br")])
    lid = only(store)["id"]
    rid = store.add_reminder(lid, (datetime.now(timezone.utc) + timedelta(days=2)).isoformat())
    store.set_status(lid, "nao_contatar", "pediu por telefone")
    d = store.get(lid)
    assert d["next_followup"] == "" and all(r["done"] for r in d["reminders"]) and rid
    for kind in ("whatsapp_aberto", "mensagem_enviada", "demo_enviada"):
        with pytest.raises(PermissionError):
            store.add_event(lid, kind)
    with pytest.raises(PermissionError):
        store.add_reminder(lid, datetime.now(timezone.utc).isoformat())
    store.delete_lead(lid, do_not_contact=True)
    assert store.suppression_count() >= 3
    # nova pesquisa traz a mesma empresa por outra fonte: não volta como contatável
    st = store.ingest([L("google_places", "ChIJX", "Alfa Exemplo", phone="+55 19 99900-0001")])
    assert st["ignorados_nao_contatar"] == 1 and store.page()[1] == 0
    st = store.ingest([L("osm", "node/9", "Alfa", cnpj="11.222.333/0001-81", city="Outra")])
    assert st["ignorados_nao_contatar"] == 1


def test_lifting_do_not_contact_removes_suppression(store):
    store.ingest([L(phone="19 99900-0001")])
    lid = only(store)["id"]
    store.set_status(lid, "nao_contatar")
    assert store.suppression_count() == 1
    store.set_status(lid, "sem_interesse", "a empresa autorizou")
    assert store.suppression_count() == 0
    assert any(e["kind"] == "contato_liberado" for e in store.get(lid)["events"])


def test_notes_reminders_and_followup(store):
    store.ingest([L()])
    lid = only(store)["id"]
    store.add_note(lid, "Falar com a gerente na sexta.")
    with pytest.raises(ValueError):
        store.add_note(lid, "   ")
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(days=3)).replace(microsecond=0).isoformat()
    r1 = store.add_reminder(lid, past, "ligar")
    store.add_reminder(lid, future, "enviar vídeo")
    with pytest.raises(ValueError):
        store.add_reminder(lid, "amanhã")
    d = store.get(lid)
    assert d["notes"][0]["text"].startswith("Falar") and d["next_followup"] == past
    assert store.page({"followup_due": True})[1] == 1
    assert [r["text"] for r in store.reminders_due(7)] == ["ligar", "enviar vídeo"]
    store.complete_reminder(r1)
    assert store.get(lid)["next_followup"] == future
    assert store.page({"followup_due": True})[1] == 0


def test_filters_sort_pagination_and_text_search(store):
    leads = [L(ref=f"node/{i}", name=f"Imobiliária {i:03d}", city="Campinas" if i % 2 else "Santos",
               phone=f"19 99900-{i:04d}") for i in range(1, 251)]
    store.ingest(leads)
    rows, total = store.page({"text": "santos"}, "name", False, 0, 50)
    assert total == 125 and len(rows) == 50 and rows[0]["name"] == "Imobiliária 002"
    rows2, _ = store.page({"text": "santos"}, "name", False, 50, 50)
    assert rows2[0]["name"] == "Imobiliária 102"
    assert store.page({"text": "imobiliaria 250"})[1] == 1                 # sem acento encontra com acento
    assert store.page({"text": "19999000017"})[1] == 1                    # busca por telefone
    rows, _ = store.page(sort="name", desc=True, limit=1)
    assert rows[0]["name"] == "Imobiliária 250"
    assert store.page(sort="'; DROP TABLE leads; --")[1] == 250           # chave inválida: ordem padrão
    assert store.page({"source": "osm", "uf": "SP", "has_whatsapp": True})[1] == 250
    assert store.facets() == {"ufs": ["SP"], "sources": ["osm"], "total": 250}


def test_manual_edit_wins_and_manual_lead(store):
    store.ingest([L(phone="19 3200-0001")])
    lid = only(store)["id"]
    store.edit_fields(lid, {"phone": "19 99900-0077", "inventado": "x"})
    d = store.get(lid)
    assert d["phone_e164"] == "+5519999000077" and d["origin"]["phone"] == "manual"
    mid = store.add_manual(Lead(source="manual", source_ref="", name="Indicação Beta", city="Santos", uf="SP"))
    assert store.get(mid)["sources"] == "manual"


def test_keywords_and_templates(store):
    store.add_keyword("Imóveis na planta")
    with pytest.raises(ValueError):
        store.add_keyword("imóveis na planta")
    with pytest.raises(ValueError):
        store.add_keyword("Imobiliária")
    assert store.custom_keywords() == ["Imóveis na planta"]
    store.delete_keyword("Imóveis na planta")
    store.save_template("Meu modelo", "direto", "Oi {empresa}!")
    store.save_template("Meu modelo", "profissional", "Olá {empresa}!")
    assert store.templates() == [{"id": 1, "name": "Meu modelo", "tone": "profissional", "body": "Olá {empresa}!"}]
    with pytest.raises(ValueError):
        store.save_template("", "direto", "x")


def test_usage_counters(store):
    store.record_usage("google_places", 3, 0.105)
    store.record_usage("google_places", 2, 0.07)
    assert store.usage_month("google_places") == (5, pytest.approx(0.175))
    assert store.usage_today("anthropic") == (0, 0.0)


# ----------------------------------------------------------------------------------------- licenças

def test_google_data_is_purged_after_30_days_but_place_id_stays(store):
    store.ingest([L("google_places", "ChIJTESTE01", "Só Google Imóveis", phone="+55 19 99900-0001")])
    lid = only(store)["id"]
    rec = store.get(lid)["source_records"][0]
    exp = datetime.fromisoformat(rec["expires_at"])
    assert timedelta(days=29) < exp - datetime.now(timezone.utc) <= timedelta(days=30)
    assert store.purge_expired() == 0
    store.con.execute("UPDATE lead_sources SET expires_at='2000-01-01T00:00:00+00:00'"); store.con.commit()
    assert store.purge_expired() == 1
    d = store.get(lid)
    assert d["needs_refresh"] == 1 and d["phone_e164"] == "" and "expirados" in d["name"]
    assert store.google_refs(lid) == ["ChIJTESTE01"]                    # place_id pode ficar
    assert d["source_records"][0]["purged"] == 1


def test_export_excludes_google_fields_and_adds_attribution(store, tmp_path):
    store.ingest([L(phone="19 3200-0001", website="https://alfa-exemplo.com.br")])
    store.ingest([L("google_places", "ChIJTESTE01", "Alfa", phone="19 3200-0001", website="alfa-exemplo.com.br",
                    address="Endereço que veio do Google")])
    store.ingest([L("google_places", "ChIJTESTE02", "Só Google", phone="+55 21 99999-0002", city="Rio", uf="RJ")])
    rows, attributions, omitted = store.export_rows()
    by = {r["cidade"]: r for r in rows}
    assert by["Campinas"]["empresa"] == "Imobiliária Alfa" and by["Campinas"]["endereco"] == ""
    assert by["Campinas"]["fontes"] == "osm" and "ChIJTESTE01" in by["Campinas"]["observacao"]
    only_google = [r for r in rows if "ChIJTESTE02" in r["observacao"]][0]
    assert only_google["empresa"] == "" and only_google["telefone"] == "" and only_google["cidade"] == ""
    assert omitted >= 4 and any("OpenStreetMap" in a for a in attributions)
    out = write_csv(tmp_path / "leads.csv", rows, Store.EXPORT_COLUMNS, attributions)
    text = out.read_text(encoding="utf-8-sig")
    assert "Endereço que veio do Google" not in text and "99999-0002" not in text
    assert list(csv.reader(text.splitlines(), delimiter=";"))[0][0] == "empresa"
    assert (tmp_path / "leads_fontes.txt").read_text(encoding="utf-8").count("OpenStreetMap") == 1


def test_csv_formula_injection_is_neutralized(tmp_path):
    out = write_csv(tmp_path / "x.csv", [{"empresa": "=HYPERLINK(\"http://mal\")", "telefone": "+55 11"}],
                    ("empresa", "telefone"), [])
    text = out.read_text(encoding="utf-8-sig")
    assert "'=HYPERLINK" in text and ";'+55 11" in text                # "+55 11" não é número puro


def test_delete_lead_removes_everything(store):
    store.ingest([L()])
    lid = only(store)["id"]
    store.add_note(lid, "x"); store.link_media(lid, "/tmp/design_x", "projeto", "Projeto X")
    store.delete_lead(lid)
    for t in ("leads", "lead_sources", "events", "notes", "media_links"):
        assert store.con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] == 0, t
    assert store.suppression_count() == 0


# ------------------------------------------------------------------------------------ pontuação

def test_scoring_is_explained_and_never_invents(store):
    store.ingest([L("cnpj_aberto", "11222333000181", "Alfa", category="Incorporadora", cnpj="11222333000181",
                    situacao="ATIVA", phone="19 99900-0001", website="https://alfa-exemplo.com.br")])
    lid = only(store)["id"]
    total, prio, items = score_lead(store.get(lid))
    crit = {i["criterio"]: i for i in items}
    assert crit["Site oficial"]["pontos"] == 20 and "alfa-exemplo.com.br" in crit["Site oficial"]["evidencia"]
    assert crit["Segmento"]["pontos"] == 20 and crit["Cadastro ativo"]["pontos"] == 5
    assert crit["Análise do site"]["avaliado"] is False and crit["Análise do site"]["pontos"] == 0
    assert all(i["justificativa"] for i in items) and total == sum(i["pontos"] for i in items) and prio == 2
    # sem dados: pontuação baixa, nada positivo inventado
    store.ingest([L(ref="node/2", name="Sem Dados", city="X", category="")])
    bare = [r for r in store.page()[0] if r["name"] == "Sem Dados"][0]
    t2, p2, items2 = score_lead(bare)
    assert t2 == 0 and p2 == 1 and all(i["pontos"] == 0 for i in items2)


def test_site_check_confirms_whatsapp_and_raises_score(store):
    store.ingest([L(phone="19 3200-0001", website="https://alfa-exemplo.com.br")])
    lid = only(store)["id"]
    ops.rescore(store, [lid])
    before = store.get(lid)["score"]
    store.save_site_check(lid, {"url": "https://alfa-exemplo.com.br/", "ok": True, "summary": "ok",
                                "totals": {"listings": 12, "empty": 3, "launch": 1, "staging": 0},
                                "terms": {"listings": {"venda": 12}, "empty": {"desocupado": 3}, "launch": {}},
                                "whatsapp_numbers": ["+5519999000001"], "whatsapp_match": ""})
    ops.rescore(store, [lid])
    d = store.get(lid)
    assert d["wa_status"] == "confirmado" and d["wa_number"] == "+5519999000001" and "site" in d["sources"]
    assert d["score"] > before and json.loads(d["score_detail"])
    assert any(e["kind"] == "site_analisado" for e in d["events"])


# --------------------------------------------------------------------------------------- painel

def test_dashboard_with_no_data(store):
    d = store.dashboard()
    assert d["values"] == {"found": 0, "qualified": 0, "contacted": 0, "interested": 0, "demo": 0,
                           "negotiation": 0, "converted": 0}
    assert d["ufs"] == [] and d["reminders"] == [] and d["overdue"] == 0


def test_dashboard_funnel_counts(store):
    statuses = ["nao_contatado", "contato_iniciado", "interessado", "demo_enviada", "negociacao", "convertido",
                "sem_interesse", "nao_contatar"]
    for i, st in enumerate(statuses):
        store.ingest([L(ref=f"node/{i}", name=f"Empresa {i}", city="Campinas" if i < 5 else "Rio",
                        uf="SP" if i < 5 else "RJ", phone=f"19 99900-00{i:02d}")])
    ids = {r["name"]: r["id"] for r in store.page()[0]}
    for i, st in enumerate(statuses):
        if st != "nao_contatado":
            store.set_status(ids[f"Empresa {i}"], st)
    store.save_score(ids["Empresa 0"], 80, 3, [])
    v = store.dashboard(60)["values"]
    assert v == {"found": 8, "qualified": 1, "contacted": 6, "interested": 4, "demo": 3, "negotiation": 2,
                 "converted": 1}
    d = store.dashboard()
    assert dict((u, n) for u, n in d["ufs"]) == {"SP": 5, "RJ": 3} and d["dnc"] == 1 and d["lost"] == 1


# ------------------------------------------------------------------------------------ AI Studio

def test_studio_projects_listing_and_link(tmp_path, store):
    from mobilia.modules.lead_finder import studio
    jobs = tmp_path / "jobs"
    p = jobs / "design_20261117-101010_mudar_estilo_japandi"; p.mkdir(parents=True)
    (p / "sala_1.jpg").write_bytes(b"jpg"); (p / "antes.jpg").write_bytes(b"jpg")
    (p / "antes_depois_9x16.mp4").write_bytes(b"mp4")
    (p / "design.json").write_text(json.dumps({"funcao": "mudar_estilo", "estilo": "japandi", "ambiente": "sala",
                                               "saidas": [str(p / "sala_1.jpg")]}), encoding="utf-8")
    (jobs / "design_sem_relatorio").mkdir()
    projects = studio.list_projects(jobs)
    assert len(projects) == 1
    pr = projects[0]
    assert pr.title == "Mudar estilo · Japandi · Sala" and pr.videos[0].endswith("antes_depois_9x16.mp4")
    assert pr.thumb.endswith("sala_1.jpg")
    assert studio.list_projects(tmp_path / "nao_existe") == []
    store.ingest([L()])
    lid = only(store)["id"]
    store.link_media(lid, pr.videos[0], "video", pr.title)
    store.link_media(lid, pr.videos[0], "video", pr.title)          # repetido: ignorado
    d = store.get(lid)
    assert len(d["media"]) == 1 and d["media"][0]["kind"] == "video"
    store.unlink_media(d["media"][0]["id"])
    assert store.get(lid)["media"] == [] and (p / "antes_depois_9x16.mp4").exists()   # arquivo intacto


def test_public_api_for_main_session(home):
    from mobilia.modules.lead_finder import api
    from mobilia.modules.lead_finder.config import data_dir
    s = Store(data_dir() / "leads.sqlite3")
    s.ingest([L(name="Alfa Exemplo", phone="19 99900-0001")]); s.close()
    found = api.search_leads("alfa")
    assert found and found[0][1] == "Alfa Exemplo — Campinas/SP"
    api.link_output(found[0][0], "/caminho/antes_depois_16x9.mp4", kind="video")
    assert api.links_for(found[0][0])[0]["path"] == "/caminho/antes_depois_16x9.mp4"
    with pytest.raises(ValueError):
        api.link_output(found[0][0], "/x", kind="outro")
