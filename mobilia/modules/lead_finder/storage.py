"""Banco local do Lead Finder (SQLite, com migrações) — leads consolidados, origem de cada dado, CRM e uso de API.

Regras:

* **Uma thread só** usa uma instância de :class:`Store` (a thread do banco, ``tasks.DbService``); a interface nunca
  toca o banco diretamente.
* **Origem de tudo:** cada fonte que trouxe uma empresa fica em ``lead_sources`` (fonte, id na fonte, data,
  consulta e os campos fornecidos). Os campos exibidos são recalculados a partir delas (``dedupe.merge``).
* **Licenças:** campos de fontes com prazo de guarda (Google: 30 dias) são apagados por :meth:`purge_expired`;
  :meth:`export_rows` só usa fontes exportáveis.
* **Não contatar:** marcar uma empresa como "Não contatar" grava telefone, CNPJ e domínio numa lista de supressão;
  novas pesquisas não recriam a empresa como contatável.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import dedupe, phone as ph
from .geo import fold
from .models import (DO_NOT_CONTACT, FUNNEL_RANK, LEAD_FIELDS, STATUS_KEYS, WA_CONFIRMED, WA_LIKELY, WA_NONE,
                     WA_UNKNOWN, Lead, now_iso, parse_iso)
from .sources import policy as source_policy

DB_NAME = "leads.sqlite3"

MIGRATIONS: tuple[str, ...] = (
    # v1 — estrutura inicial
    """
    CREATE TABLE leads(
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL DEFAULT '', legal_name TEXT DEFAULT '', category TEXT DEFAULT '',
      address TEXT DEFAULT '', city TEXT DEFAULT '', uf TEXT DEFAULT '', cep TEXT DEFAULT '',
      phone TEXT DEFAULT '', phone2 TEXT DEFAULT '', whatsapp TEXT DEFAULT '', website TEXT DEFAULT '',
      cnpj TEXT DEFAULT '', cnae TEXT DEFAULT '', situacao TEXT DEFAULT '', natureza TEXT DEFAULT '',
      phone_e164 TEXT DEFAULT '', phone_kind TEXT DEFAULT '', wa_number TEXT DEFAULT '',
      wa_status TEXT DEFAULT 'nao_verificado', wa_user TEXT DEFAULT '', domain TEXT DEFAULT '',
      sources TEXT DEFAULT '', origin TEXT DEFAULT '{}', name_key TEXT DEFAULT '',
      search_text TEXT DEFAULT '', restricted INTEGER DEFAULT 0, needs_refresh INTEGER DEFAULT 0,
      status TEXT DEFAULT 'nao_contatado', priority INTEGER DEFAULT 0, priority_user INTEGER DEFAULT 0,
      score INTEGER, score_detail TEXT DEFAULT '', site_check TEXT DEFAULT '',
      last_contact TEXT DEFAULT '', next_followup TEXT DEFAULT '', last_fetched TEXT DEFAULT '',
      created_at TEXT, updated_at TEXT);
    CREATE INDEX ix_leads_status ON leads(status);
    CREATE INDEX ix_leads_loc ON leads(uf, city);
    CREATE INDEX ix_leads_cnpj ON leads(cnpj);
    CREATE INDEX ix_leads_phone ON leads(phone_e164);
    CREATE INDEX ix_leads_domain ON leads(domain);
    CREATE INDEX ix_leads_namekey ON leads(name_key);
    CREATE TABLE lead_sources(
      id INTEGER PRIMARY KEY, lead_id INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
      source TEXT NOT NULL, source_ref TEXT NOT NULL, fetched_at TEXT, query TEXT DEFAULT '',
      payload TEXT DEFAULT '{}', expires_at TEXT DEFAULT '', purged INTEGER DEFAULT 0,
      UNIQUE(source, source_ref));
    CREATE INDEX ix_ls_lead ON lead_sources(lead_id);
    CREATE TABLE events(
      id INTEGER PRIMARY KEY, lead_id INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
      at TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT DEFAULT '', payload TEXT DEFAULT '{}');
    CREATE INDEX ix_events_lead ON events(lead_id, at);
    CREATE TABLE notes(
      id INTEGER PRIMARY KEY, lead_id INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
      at TEXT NOT NULL, text TEXT NOT NULL);
    CREATE TABLE reminders(
      id INTEGER PRIMARY KEY, lead_id INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
      due_at TEXT NOT NULL, text TEXT DEFAULT '', done INTEGER DEFAULT 0, created_at TEXT);
    CREATE INDEX ix_rem_due ON reminders(done, due_at);
    CREATE TABLE suppression(
      id INTEGER PRIMARY KEY, kind TEXT NOT NULL, value TEXT NOT NULL, reason TEXT DEFAULT '', at TEXT,
      UNIQUE(kind, value));
    CREATE TABLE templates(
      id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, tone TEXT DEFAULT 'profissional', body TEXT NOT NULL,
      updated_at TEXT);
    CREATE TABLE keywords(id INTEGER PRIMARY KEY, text TEXT UNIQUE NOT NULL, created_at TEXT);
    CREATE TABLE api_usage(id INTEGER PRIMARY KEY, source TEXT NOT NULL, at TEXT NOT NULL, units INTEGER,
      cost_usd REAL DEFAULT 0);
    CREATE INDEX ix_usage ON api_usage(source, at);
    CREATE TABLE searches(id INTEGER PRIMARY KEY, at TEXT, params TEXT, summary TEXT DEFAULT '{}');
    """,
    # v2 — vínculos com o AI Studio (projetos e vídeos gerados)
    """
    CREATE TABLE media_links(
      id INTEGER PRIMARY KEY, lead_id INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
      path TEXT NOT NULL, kind TEXT DEFAULT 'projeto', title TEXT DEFAULT '', linked_at TEXT,
      UNIQUE(lead_id, path));
    """,
)
SCHEMA_VERSION = len(MIGRATIONS)

SORTABLE = {
    "name": "name COLLATE NOCASE", "city": "uf, city COLLATE NOCASE", "address": "address COLLATE NOCASE",
    "phone": "phone_e164", "whatsapp": "wa_status", "website": "domain", "cnpj": "cnpj",
    "category": "category", "source": "sources, last_fetched", "priority": "max(priority, priority_user)",
    "status": "status", "score": "coalesce(score, -1)", "updated": "updated_at", "followup": "next_followup",
}
DASH_STAGES = (("found", "Encontradas"), ("qualified", "Qualificadas"), ("contacted", "Contatos iniciados"),
               ("interested", "Interessados"), ("demo", "Demonstrações"), ("negotiation", "Negociações"),
               ("converted", "Clientes convertidos"))


def _j(v) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def _days_from_now(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).replace(microsecond=0).isoformat()


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys=ON")
        if str(path) != ":memory:":
            self.con.execute("PRAGMA journal_mode=WAL")
        self.migrate()

    def close(self) -> None:
        self.con.close()

    # ------------------------------------------------------------------------------------- migrações
    @property
    def version(self) -> int:
        return self.con.execute("PRAGMA user_version").fetchone()[0]

    def migrate(self) -> int:
        v = self.version
        if v > SCHEMA_VERSION:
            raise RuntimeError(f"Banco do Lead Finder é de uma versão mais nova ({v}); atualize o 7xAi.")
        for i in range(v, SCHEMA_VERSION):
            try:   # cada migração é atômica: estrutura + número da versão na mesma transação
                self.con.executescript(f"BEGIN;\n{MIGRATIONS[i]}\nPRAGMA user_version={i + 1};\nCOMMIT;")
            except sqlite3.Error:
                if self.con.in_transaction:
                    self.con.execute("ROLLBACK")
                raise
        return self.version

    # --------------------------------------------------------------------------------- supressão
    def _suppressed(self, cnpj: str, e164s: list[str], domain: str) -> bool:
        keys = [("cnpj", cnpj)] if cnpj else []
        keys += [("phone", e) for e in e164s if e]
        if domain and not dedupe.is_generic(domain):
            keys.append(("domain", domain))
        for k, v in keys:
            if self.con.execute("SELECT 1 FROM suppression WHERE kind=? AND value=?", (k, v)).fetchone():
                return True
        return False

    def _suppress_lead(self, row, reason: str) -> None:
        at = now_iso()
        vals = [("cnpj", row["cnpj"]), ("phone", row["phone_e164"]), ("phone", row["wa_number"])]
        p2 = ph.normalize(row["phone2"])
        vals.append(("phone", p2.e164))
        if row["domain"] and not dedupe.is_generic(row["domain"]):
            vals.append(("domain", row["domain"]))
        self.con.executemany("INSERT OR IGNORE INTO suppression(kind, value, reason, at) VALUES (?,?,?,?)",
                             [(k, v, reason, at) for k, v in vals if v])

    def _unsuppress_lead(self, row) -> None:
        p2 = ph.normalize(row["phone2"])
        for k, v in (("cnpj", row["cnpj"]), ("phone", row["phone_e164"]), ("phone", row["wa_number"]),
                     ("phone", p2.e164), ("domain", row["domain"])):
            if v:
                self.con.execute("DELETE FROM suppression WHERE kind=? AND value=?", (k, v))

    # ----------------------------------------------------------------------------------- ingestão
    def ingest(self, leads: list[Lead]) -> dict:
        """Grava registros vindos de uma fonte, consolidando com o que já existe. Devolve contagens."""
        stats = {"novos": 0, "consolidados": 0, "atualizados": 0, "ignorados_nao_contatar": 0}
        with self.con:
            for lead in leads:
                info = self._keys(lead)
                lead_id = self._find(lead, info)
                if lead_id is None and self._suppressed(info["cnpj"], info["phones"], info["domain"]):
                    stats["ignorados_nao_contatar"] += 1
                    continue
                pol = source_policy(lead.source)
                expires = _days_from_now(pol.store_days) if pol.store_days else ""
                created = lead_id is None
                if created:
                    t = now_iso()
                    lead_id = self.con.execute("INSERT INTO leads(name, created_at, updated_at) VALUES (?,?,?)",
                                               (lead.name, t, t)).lastrowid
                    self._event(lead_id, "criado", f"{lead.name} · {lead.query}".strip(" ·"),
                                {"fonte": lead.source})
                existing = self.con.execute("SELECT lead_id FROM lead_sources WHERE source=? AND source_ref=?",
                                            (lead.source, lead.source_ref)).fetchone()
                payload = _j(lead.fields())
                if existing:
                    self.con.execute("UPDATE lead_sources SET fetched_at=?, query=?, payload=?, expires_at=?, "
                                     "purged=0 WHERE source=? AND source_ref=?",
                                     (lead.fetched_at, lead.query, payload, expires, lead.source, lead.source_ref))
                    stats["atualizados"] += 1
                else:
                    self.con.execute("INSERT INTO lead_sources(lead_id, source, source_ref, fetched_at, query, "
                                     "payload, expires_at) VALUES (?,?,?,?,?,?,?)",
                                     (lead_id, lead.source, lead.source_ref, lead.fetched_at, lead.query, payload,
                                      expires))
                    if created:
                        stats["novos"] += 1
                    else:
                        stats["consolidados"] += 1
                        self._event(lead_id, "fonte", f"Também encontrada em {lead.source}", {"fonte": lead.source})
                self._recompute(lead_id)
        return stats

    def _keys(self, lead: Lead) -> dict:
        phones = [ph.normalize(p).e164 for p in (lead.whatsapp, lead.phone, lead.phone2)]
        domain = dedupe.domain_of(lead.website)
        cnpj = "".join(c for c in lead.cnpj if c.isdigit())
        return {"cnpj": cnpj if len(cnpj) == 14 else "", "phones": [p for p in phones if p],
                "domain": "" if dedupe.is_generic(domain) else domain,
                "name_key": dedupe.name_key(lead.name, lead.city, lead.uf)}

    def _find(self, lead: Lead, info: dict) -> int | None:
        r = self.con.execute("SELECT lead_id FROM lead_sources WHERE source=? AND source_ref=?",
                             (lead.source, lead.source_ref)).fetchone()
        if r:
            return r[0]
        if info["cnpj"]:
            r = self.con.execute("SELECT id FROM leads WHERE cnpj=? LIMIT 1", (info["cnpj"],)).fetchone()
            if r:
                return r[0]
        for e in info["phones"]:
            r = self.con.execute("SELECT id, cnpj FROM leads WHERE phone_e164=? OR wa_number=? LIMIT 1",
                                 (e, e)).fetchone()
            if r and not (info["cnpj"] and r[1] and r[1] != info["cnpj"]):
                return r[0]
        if info["domain"]:
            r = self.con.execute("SELECT id, cnpj FROM leads WHERE domain=? LIMIT 1", (info["domain"],)).fetchone()
            if r and not (info["cnpj"] and r[1] and r[1] != info["cnpj"]):
                return r[0]
        if info["name_key"]:
            r = self.con.execute("SELECT id, cnpj FROM leads WHERE name_key=? LIMIT 1", (info["name_key"],)).fetchone()
            if r and not (info["cnpj"] and r[1] and r[1] != info["cnpj"]):
                return r[0]
        return None

    def _recompute(self, lead_id: int) -> None:
        rows = self.con.execute("SELECT source, payload, fetched_at, purged FROM lead_sources WHERE lead_id=?",
                                (lead_id,)).fetchall()
        records = [(r["source"], json.loads(r["payload"] or "{}"), r["fetched_at"]) for r in rows]
        merged, origin = dedupe.merge(records)
        cur = self.con.execute("SELECT * FROM leads WHERE id=?", (lead_id,)).fetchone()
        restricted = any(not source_policy(s).exportable and p for s, p, _ in records)
        needs_refresh = int(not merged.get("name") and any(r["purged"] for r in rows))
        name = merged.get("name") or ("(dados expirados — atualize)" if needs_refresh else cur["name"])
        p1, p2 = ph.normalize(merged.get("phone", "")), ph.normalize(merged.get("phone2", ""))
        if not p1.valid and p2.valid:
            p1, p2 = p2, p1
        wa_src = ph.normalize(merged.get("whatsapp", ""))
        wa_user = cur["wa_user"]
        if wa_user == WA_NONE:
            wa_num, wa_status = "", WA_NONE
        elif wa_src.valid:
            wa_num, wa_status = wa_src.e164, WA_CONFIRMED
        elif wa_user == WA_CONFIRMED and (p1.valid or p2.valid):
            wa_num, wa_status = (p1 if p1.valid else p2).e164, WA_CONFIRMED
        else:
            mobile = p1 if p1.kind == ph.MOBILE else p2 if p2.kind == ph.MOBILE else None
            wa_num, wa_status = (mobile.e164, WA_LIKELY) if mobile else ("", WA_UNKNOWN)
        domain = dedupe.domain_of(merged.get("website", ""))
        vals = {k: merged.get(k, "") for k in LEAD_FIELDS}
        vals["name"] = name
        cnpj = "".join(c for c in vals["cnpj"] if c.isdigit())
        vals["cnpj"] = cnpj if len(cnpj) == 14 else ""
        sources = sorted({r["source"] for r in rows})
        search = fold(" ".join([name, vals["legal_name"], vals["city"], vals["uf"], vals["category"],
                                vals["address"], domain, cnpj, p1.digits, p2.digits]))
        vals.update(phone_e164=p1.e164, phone_kind=p1.kind if (p1.valid or merged.get("phone")) else "",
                    wa_number=wa_num, wa_status=wa_status, domain=domain, sources=",".join(sources),
                    origin=_j(origin), name_key=dedupe.name_key(name, vals["city"], vals["uf"]),
                    search_text=search, restricted=int(restricted), needs_refresh=needs_refresh,
                    last_fetched=max((r["fetched_at"] or "" for r in rows), default=""), updated_at=now_iso())
        sets = ", ".join(f"{k}=?" for k in vals)
        self.con.execute(f"UPDATE leads SET {sets} WHERE id=?", (*vals.values(), lead_id))

    # ------------------------------------------------------------------------------------ consulta
    def _where(self, f: dict) -> tuple[str, list]:
        w, a = ["1=1"], []
        f = f or {}
        if f.get("text"):
            for tok in fold(f["text"]).split()[:6]:
                w.append("search_text LIKE ?"); a.append(f"%{tok}%")
        if f.get("status"):
            w.append("status=?"); a.append(f["status"])
        elif f.get("hide_dnc"):
            w.append("status != ?"); a.append(DO_NOT_CONTACT)
        if f.get("uf"):
            w.append("uf=?"); a.append(f["uf"])
        if f.get("city"):
            w.append("city=?"); a.append(f["city"])
        if f.get("source"):
            w.append("(',' || sources || ',') LIKE ?"); a.append(f"%,{f['source']},%")
        if f.get("priority"):
            w.append("max(priority, priority_user)=?"); a.append(int(f["priority"]))
        if f.get("has_phone"):
            w.append("phone_e164 != ''")
        if f.get("has_whatsapp"):
            w.append("wa_number != ''")
        if f.get("has_site"):
            w.append("domain != ''")
        if f.get("followup_due"):
            w.append("next_followup != '' AND next_followup <= ?"); a.append(now_iso())
        if f.get("ids"):
            ids = [int(i) for i in f["ids"]]
            w.append(f"id IN ({','.join('?' * len(ids))})"); a.extend(ids)
        return " AND ".join(w), a

    def page(self, filters: dict | None = None, sort: str = "updated", desc: bool = True, offset: int = 0,
             limit: int = 100) -> tuple[list[dict], int]:
        where, args = self._where(filters or {})
        total = self.con.execute(f"SELECT count(*) FROM leads WHERE {where}", args).fetchone()[0]
        order = SORTABLE.get(sort, SORTABLE["updated"])
        direction = "DESC" if desc else "ASC"
        order_sql = ", ".join(f"{part.strip()} {direction}" for part in order.split(","))
        rows = self.con.execute(f"SELECT * FROM leads WHERE {where} ORDER BY {order_sql}, id {direction} "
                                "LIMIT ? OFFSET ?", (*args, max(1, min(1000, limit)), max(0, offset))).fetchall()
        return [dict(r) for r in rows], total

    def ids(self, filters: dict | None = None) -> list[int]:
        where, args = self._where(filters or {})
        return [r[0] for r in self.con.execute(f"SELECT id FROM leads WHERE {where}", args)]

    def get(self, lead_id: int) -> dict | None:
        r = self.con.execute("SELECT * FROM leads WHERE id=?", (lead_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["source_records"] = [dict(x) for x in self.con.execute(
            "SELECT source, source_ref, fetched_at, query, expires_at, purged FROM lead_sources WHERE lead_id=? "
            "ORDER BY fetched_at DESC", (lead_id,))]
        d["events"] = [dict(x) for x in self.con.execute(
            "SELECT at, kind, detail FROM events WHERE lead_id=? ORDER BY at DESC, id DESC LIMIT 200", (lead_id,))]
        d["notes"] = [dict(x) for x in self.con.execute(
            "SELECT id, at, text FROM notes WHERE lead_id=? ORDER BY at DESC, id DESC", (lead_id,))]
        d["reminders"] = [dict(x) for x in self.con.execute(
            "SELECT id, due_at, text, done FROM reminders WHERE lead_id=? ORDER BY done, due_at", (lead_id,))]
        d["media"] = [dict(x) for x in self.con.execute(
            "SELECT id, path, kind, title, linked_at FROM media_links WHERE lead_id=? ORDER BY linked_at DESC",
            (lead_id,))]
        d["origin"] = json.loads(d.get("origin") or "{}")
        return d

    def facets(self) -> dict:
        return {
            "ufs": [r[0] for r in self.con.execute("SELECT DISTINCT uf FROM leads WHERE uf != '' ORDER BY uf")],
            "sources": sorted({s for (v,) in self.con.execute("SELECT DISTINCT sources FROM leads")
                               for s in (v or "").split(",") if s}),
            "total": self.con.execute("SELECT count(*) FROM leads").fetchone()[0],
        }

    # ----------------------------------------------------------------------------------------- CRM
    def _event(self, lead_id: int, kind: str, detail: str = "", payload: dict | None = None) -> None:
        self.con.execute("INSERT INTO events(lead_id, at, kind, detail, payload) VALUES (?,?,?,?,?)",
                         (lead_id, now_iso(), kind, detail, _j(payload or {})))

    def add_event(self, lead_id: int, kind: str, detail: str = "", payload: dict | None = None) -> None:
        """Registra uma ação. ``mensagem_enviada`` só quando a pessoa confirma que enviou."""
        with self.con:
            row = self._row(lead_id)
            if kind in ("whatsapp_aberto", "mensagem_enviada", "demo_enviada") and row["status"] == DO_NOT_CONTACT:
                raise PermissionError("Esta empresa pediu para não ser contatada.")
            self._event(lead_id, kind, detail, payload)
            if kind == "mensagem_enviada":
                upd = {"last_contact": now_iso()}
                if row["status"] == "nao_contatado":
                    upd["status"] = "contato_iniciado"
                    self._event(lead_id, "status", "Não contatado → Contato iniciado")
                self._update(lead_id, upd)
            elif kind == "demo_enviada":
                upd = {"last_contact": now_iso()}
                if FUNNEL_RANK.get(row["status"], 0) < FUNNEL_RANK["demo_enviada"] and row["status"] != "sem_interesse":
                    upd["status"] = "demo_enviada"
                    self._event(lead_id, "status", "→ Demonstração enviada")
                self._update(lead_id, upd)
            elif kind == "resposta_recebida":
                upd = {"last_contact": now_iso()}
                if row["status"] in ("nao_contatado", "contato_iniciado"):
                    upd["status"] = "aguardando"
                self._update(lead_id, upd)

    def _row(self, lead_id: int):
        row = self.con.execute("SELECT * FROM leads WHERE id=?", (lead_id,)).fetchone()
        if row is None:
            raise KeyError(f"lead {lead_id} não existe")
        return row

    def _update(self, lead_id: int, values: dict) -> None:
        if values:
            values = {**values, "updated_at": now_iso()}
            self.con.execute(f"UPDATE leads SET {', '.join(f'{k}=?' for k in values)} WHERE id=?",
                             (*values.values(), lead_id))

    def set_status(self, lead_id: int, status: str, note: str = "") -> None:
        if status not in STATUS_KEYS:
            raise ValueError(f"status inválido: {status}")
        from .models import STATUS_LABEL
        with self.con:
            row = self._row(lead_id)
            if row["status"] == status:
                return
            self._update(lead_id, {"status": status})
            self._event(lead_id, "status", f"{STATUS_LABEL[row['status']]} → {STATUS_LABEL[status]}"
                        + (f" · {note}" if note else ""))
            if status == DO_NOT_CONTACT:
                self._suppress_lead(row, note or "pedido de não contato")
                self._event(lead_id, "nao_contatar", note)
                self.con.execute("UPDATE reminders SET done=1 WHERE lead_id=? AND done=0", (lead_id,))
                self._update(lead_id, {"next_followup": ""})
            elif row["status"] == DO_NOT_CONTACT:
                self._unsuppress_lead(row)
                self._event(lead_id, "contato_liberado", note)

    def add_note(self, lead_id: int, text: str) -> int:
        text = (text or "").strip()
        if not text:
            raise ValueError("Nota vazia.")
        with self.con:
            self._row(lead_id)
            nid = self.con.execute("INSERT INTO notes(lead_id, at, text) VALUES (?,?,?)",
                                   (lead_id, now_iso(), text[:4000])).lastrowid
            self._event(lead_id, "nota", text[:120])
            self._update(lead_id, {})
        return nid

    def delete_note(self, note_id: int) -> None:
        with self.con:
            self.con.execute("DELETE FROM notes WHERE id=?", (note_id,))

    def add_reminder(self, lead_id: int, due_at: str, text: str = "") -> int:
        if parse_iso(due_at) is None:
            raise ValueError("Data do lembrete inválida.")
        with self.con:
            row = self._row(lead_id)
            if row["status"] == DO_NOT_CONTACT:
                raise PermissionError("Esta empresa pediu para não ser contatada.")
            rid = self.con.execute("INSERT INTO reminders(lead_id, due_at, text, created_at) VALUES (?,?,?,?)",
                                   (lead_id, due_at, text.strip()[:500], now_iso())).lastrowid
            self._event(lead_id, "lembrete_criado", f"{due_at[:16].replace('T', ' ')} {text}".strip())
            self._refresh_followup(lead_id)
        return rid

    def complete_reminder(self, reminder_id: int) -> None:
        with self.con:
            r = self.con.execute("SELECT lead_id, text FROM reminders WHERE id=?", (reminder_id,)).fetchone()
            if r is None:
                return
            self.con.execute("UPDATE reminders SET done=1 WHERE id=?", (reminder_id,))
            self._event(r[0], "lembrete_concluido", r[1])
            self._refresh_followup(r[0])

    def _refresh_followup(self, lead_id: int) -> None:
        r = self.con.execute("SELECT min(due_at) FROM reminders WHERE lead_id=? AND done=0", (lead_id,)).fetchone()
        self._update(lead_id, {"next_followup": r[0] or ""})

    def reminders_due(self, days_ahead: int = 7, limit: int = 30) -> list[dict]:
        until = _days_from_now(days_ahead)
        return [dict(r) for r in self.con.execute(
            "SELECT r.id, r.lead_id, r.due_at, r.text, l.name, l.city, l.uf FROM reminders r "
            "JOIN leads l ON l.id = r.lead_id WHERE r.done=0 AND r.due_at <= ? AND l.status != ? "
            "ORDER BY r.due_at LIMIT ?", (until, DO_NOT_CONTACT, limit))]

    def set_whatsapp(self, lead_id: int, state: str) -> None:
        """Confirmação manual: ``confirmado`` (conversou pelo WhatsApp), ``sem_whatsapp`` ou ``""`` (desfazer)."""
        if state not in ("", WA_CONFIRMED, WA_NONE):
            raise ValueError(state)
        with self.con:
            self._row(lead_id)
            self._update(lead_id, {"wa_user": state})
            self._recompute(lead_id)
            self._event(lead_id, "editado", f"WhatsApp: {state or 'automático'}")

    def set_priority(self, lead_id: int, priority: int) -> None:
        with self.con:
            self._row(lead_id)
            self._update(lead_id, {"priority_user": max(0, min(3, int(priority)))})

    def edit_fields(self, lead_id: int, values: dict) -> None:
        """Correção manual (fonte ``manual``, prioridade máxima na consolidação)."""
        clean = {k: str(v).strip() for k, v in values.items() if k in LEAD_FIELDS}
        with self.con:
            self._row(lead_id)
            ref = f"lead:{lead_id}"
            cur = self.con.execute("SELECT payload FROM lead_sources WHERE source='manual' AND source_ref=?",
                                   (ref,)).fetchone()
            payload = {**(json.loads(cur[0]) if cur else {}), **clean}
            payload = {k: v for k, v in payload.items() if v}
            self.con.execute("INSERT INTO lead_sources(lead_id, source, source_ref, fetched_at, query, payload) "
                             "VALUES (?,?,?,?,?,?) ON CONFLICT(source, source_ref) DO UPDATE SET payload=excluded.payload,"
                             " fetched_at=excluded.fetched_at", (lead_id, "manual", ref, now_iso(), "edição", _j(payload)))
            self._recompute(lead_id)
            self._event(lead_id, "editado", ", ".join(sorted(clean)))

    def add_manual(self, lead: Lead) -> int:
        lead.source, lead.source_ref = "manual", lead.source_ref or f"novo:{now_iso()}:{lead.name}"
        self.ingest([lead])
        return self.con.execute("SELECT lead_id FROM lead_sources WHERE source='manual' AND source_ref=?",
                                (lead.source_ref,)).fetchone()[0]

    def save_score(self, lead_id: int, score: int, priority: int, detail: list[dict]) -> None:
        with self.con:
            self._update(lead_id, {"score": int(score), "priority": int(priority), "score_detail": _j(detail)})

    def save_site_check(self, lead_id: int, result: dict) -> None:
        """Guarda a análise do site; um WhatsApp publicado no site oficial vira fonte ``site`` (confirmado)."""
        with self.con:
            self._row(lead_id)
            self._update(lead_id, {"site_check": _j(result)})
            wa = result.get("whatsapp_match") or (result.get("whatsapp_numbers") or [""])[0]
            if wa:
                ref = f"site:{lead_id}"
                self.con.execute(
                    "INSERT INTO lead_sources(lead_id, source, source_ref, fetched_at, query, payload) VALUES "
                    "(?,?,?,?,?,?) ON CONFLICT(source, source_ref) DO UPDATE SET payload=excluded.payload, "
                    "fetched_at=excluded.fetched_at",
                    (lead_id, "site", ref, now_iso(), result.get("url", ""), _j({"whatsapp": wa})))
            self._recompute(lead_id)
            self._event(lead_id, "site_analisado", result.get("summary", ""))

    def delete_lead(self, lead_id: int, do_not_contact: bool = False) -> None:
        """Exclui a empresa e todo o histórico (pedido de exclusão, LGPD). Com ``do_not_contact``, telefone/CNPJ/
        domínio ficam na lista de supressão para não voltarem em pesquisas futuras."""
        with self.con:
            row = self._row(lead_id)
            if do_not_contact:
                self._suppress_lead(row, "exclusão com pedido de não contato")
            self.con.execute("DELETE FROM leads WHERE id=?", (lead_id,))

    def delete_all(self) -> None:
        with self.con:
            for t in ("media_links", "events", "notes", "reminders", "lead_sources", "leads", "searches",
                      "api_usage"):
                self.con.execute(f"DELETE FROM {t}")

    def suppression_count(self) -> int:
        return self.con.execute("SELECT count(*) FROM suppression").fetchone()[0]

    def clear_suppression(self) -> None:
        with self.con:
            self.con.execute("DELETE FROM suppression")

    # --------------------------------------------------------------------------------- AI Studio
    def link_media(self, lead_id: int, path: str, kind: str = "projeto", title: str = "") -> None:
        with self.con:
            self._row(lead_id)
            self.con.execute("INSERT OR IGNORE INTO media_links(lead_id, path, kind, title, linked_at) "
                             "VALUES (?,?,?,?,?)", (lead_id, str(path), kind, title, now_iso()))
            self._event(lead_id, "midia_vinculada", title or Path(path).name, {"path": str(path)})

    def unlink_media(self, link_id: int) -> None:
        with self.con:
            r = self.con.execute("SELECT lead_id, title, path FROM media_links WHERE id=?", (link_id,)).fetchone()
            if r:
                self.con.execute("DELETE FROM media_links WHERE id=?", (link_id,))
                self._event(r[0], "midia_removida", r[1] or Path(r[2]).name)

    # ------------------------------------------------------------------------------- uso e custo
    def record_usage(self, source: str, units: int, cost_usd: float = 0.0) -> None:
        with self.con:
            self.con.execute("INSERT INTO api_usage(source, at, units, cost_usd) VALUES (?,?,?,?)",
                             (source, now_iso(), int(units), float(cost_usd)))

    def usage_since(self, source: str, since_iso: str) -> tuple[int, float]:
        r = self.con.execute("SELECT coalesce(sum(units),0), coalesce(sum(cost_usd),0) FROM api_usage "
                             "WHERE source=? AND at >= ?", (source, since_iso)).fetchone()
        return int(r[0]), float(r[1])

    def usage_month(self, source: str) -> tuple[int, float]:
        start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return self.usage_since(source, start.isoformat())

    def usage_today(self, source: str) -> tuple[int, float]:
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return self.usage_since(source, start.isoformat())

    def log_search(self, params: dict, summary: dict) -> int:
        with self.con:
            return self.con.execute("INSERT INTO searches(at, params, summary) VALUES (?,?,?)",
                                    (now_iso(), _j(params), _j(summary))).lastrowid

    def recent_searches(self, limit: int = 10) -> list[dict]:
        return [{"at": r[0], "params": json.loads(r[1]), "summary": json.loads(r[2])} for r in self.con.execute(
            "SELECT at, params, summary FROM searches ORDER BY id DESC LIMIT ?", (limit,))]

    # ---------------------------------------------------------------------------- modelos/palavras
    def templates(self) -> list[dict]:
        return [dict(r) for r in self.con.execute("SELECT id, name, tone, body FROM templates ORDER BY name")]

    def save_template(self, name: str, tone: str, body: str) -> None:
        name, body = (name or "").strip(), (body or "").strip()
        if not name or not body:
            raise ValueError("Dê um nome e escreva o texto do modelo.")
        with self.con:
            self.con.execute("INSERT INTO templates(name, tone, body, updated_at) VALUES (?,?,?,?) ON CONFLICT(name) "
                             "DO UPDATE SET tone=excluded.tone, body=excluded.body, updated_at=excluded.updated_at",
                             (name[:80], tone, body[:4000], now_iso()))

    def delete_template(self, name: str) -> None:
        with self.con:
            self.con.execute("DELETE FROM templates WHERE name=?", (name,))

    def custom_keywords(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT text FROM keywords ORDER BY text COLLATE NOCASE")]

    def add_keyword(self, text: str) -> None:
        from .keywords import is_builtin, validate_custom
        k = validate_custom(text)
        if is_builtin(k):
            raise ValueError("Essa palavra-chave já existe na lista padrão.")
        with self.con:
            if self.con.execute("SELECT 1 FROM keywords WHERE lower(text)=lower(?)", (k,)).fetchone():
                raise ValueError("Essa palavra-chave já está salva.")
            self.con.execute("INSERT INTO keywords(text, created_at) VALUES (?,?)", (k, now_iso()))

    def delete_keyword(self, text: str) -> None:
        with self.con:
            self.con.execute("DELETE FROM keywords WHERE text=?", (text,))

    # ------------------------------------------------------------------------------- licenças
    def purge_expired(self) -> int:
        """Apaga campos de fontes cujo prazo de guarda acabou (ex.: Google, 30 dias). Mantém o identificador."""
        now = now_iso()
        rows = self.con.execute("SELECT id, lead_id, source FROM lead_sources WHERE purged=0 AND expires_at != '' "
                                "AND expires_at <= ?", (now,)).fetchall()
        with self.con:
            for r in rows:
                self.con.execute("UPDATE lead_sources SET payload='{}', purged=1, query='' WHERE id=?", (r["id"],))
            for lead_id in {r["lead_id"] for r in rows}:
                self._recompute(lead_id)
        return len(rows)

    def google_refs(self, lead_id: int) -> list[str]:
        return [r[0] for r in self.con.execute(
            "SELECT source_ref FROM lead_sources WHERE lead_id=? AND source='google_places'", (lead_id,))]

    # --------------------------------------------------------------------------------- exportação
    EXPORT_COLUMNS = ("empresa", "razao_social", "categoria", "endereco", "cidade", "uf", "cep", "telefone",
                      "telefone2", "whatsapp", "site", "cnpj", "status", "prioridade", "pontuacao",
                      "ultimo_contato", "proximo_acompanhamento", "fontes", "consultado_em", "observacao")

    def export_rows(self, filters: dict | None = None) -> tuple[list[dict], list[str], int]:
        """Linhas exportáveis (só campos de fontes que permitem reutilização). Devolve (linhas, atribuições,
        quantidade de campos omitidos por licença)."""
        from .models import PRIORITY_LABEL, STATUS_LABEL
        where, args = self._where(filters or {})
        out, attributions, omitted = [], set(), 0
        for row in self.con.execute(f"SELECT * FROM leads WHERE {where} ORDER BY uf, city, name", args).fetchall():
            recs = self.con.execute("SELECT source, payload, fetched_at, source_ref FROM lead_sources WHERE lead_id=?",
                                    (row["id"],)).fetchall()
            ok = [(r["source"], json.loads(r["payload"] or "{}"), r["fetched_at"]) for r in recs
                  if source_policy(r["source"]).exportable]
            blocked = [r for r in recs if not source_policy(r["source"]).exportable]
            merged, _ = dedupe.merge(ok)
            full, _ = dedupe.merge([(r["source"], json.loads(r["payload"] or "{}"), r["fetched_at"]) for r in recs])
            omitted += sum(1 for k, v in full.items() if v and not merged.get(k))
            for s, _, _ in ok:
                if source_policy(s).attribution:
                    attributions.add(source_policy(s).attribution)
            wa = ph.normalize(merged.get("whatsapp", ""))
            note = ""
            if blocked:
                note = ("dados do Google não exportáveis pelos termos de uso; place_id: "
                        + ", ".join(r["source_ref"] for r in blocked if r["source"] == "google_places"))
            out.append({
                "empresa": merged.get("name", ""), "razao_social": merged.get("legal_name", ""),
                "categoria": merged.get("category", ""), "endereco": merged.get("address", ""),
                "cidade": merged.get("city", ""), "uf": merged.get("uf", ""), "cep": merged.get("cep", ""),
                "telefone": ph.normalize(merged.get("phone", "")).national if merged.get("phone") else "",
                "telefone2": ph.normalize(merged.get("phone2", "")).national if merged.get("phone2") else "",
                "whatsapp": wa.national if wa.valid else "", "site": merged.get("website", ""),
                "cnpj": merged.get("cnpj", ""), "status": STATUS_LABEL.get(row["status"], row["status"]),
                "prioridade": PRIORITY_LABEL.get(max(row["priority"], row["priority_user"]), ""),
                "pontuacao": "" if row["score"] is None else row["score"],
                "ultimo_contato": row["last_contact"], "proximo_acompanhamento": row["next_followup"],
                "fontes": ", ".join(sorted({s for s, _, _ in ok})),
                "consultado_em": max((f for _, _, f in ok), default=""), "observacao": note})
        return out, sorted(attributions), omitted

    # --------------------------------------------------------------------------------- painel
    def dashboard(self, qualified_threshold: int = 60) -> dict:
        c = self.con
        by_status = dict(c.execute("SELECT status, count(*) FROM leads GROUP BY status").fetchall())
        total = sum(by_status.values())
        dnc = by_status.get(DO_NOT_CONTACT, 0)
        rank = lambda r: sum(n for s, n in by_status.items() if FUNNEL_RANK.get(s, -1) >= r)  # noqa: E731
        contacted = c.execute("SELECT count(*) FROM leads WHERE status NOT IN ('nao_contatado', ?) OR id IN "
                              "(SELECT lead_id FROM events WHERE kind='mensagem_enviada')", (DO_NOT_CONTACT,)
                              ).fetchone()[0]
        demo = c.execute("SELECT count(*) FROM leads WHERE status IN ('demo_enviada','negociacao','convertido') OR "
                         "id IN (SELECT lead_id FROM events WHERE kind='demo_enviada')").fetchone()[0]
        values = {
            "found": total,
            "qualified": c.execute("SELECT count(*) FROM leads WHERE score >= ? AND status != ?",
                                   (qualified_threshold, DO_NOT_CONTACT)).fetchone()[0],
            "contacted": contacted, "interested": rank(2), "demo": demo, "negotiation": rank(4),
            "converted": by_status.get("convertido", 0),
        }
        return {
            "values": values, "stages": DASH_STAGES, "by_status": by_status, "dnc": dnc,
            "lost": by_status.get("sem_interesse", 0),
            "ufs": c.execute("SELECT uf, count(*) n FROM leads WHERE uf != '' GROUP BY uf ORDER BY n DESC, uf "
                             "LIMIT 12").fetchall(),
            "cities": c.execute("SELECT city || ' - ' || uf, count(*) n FROM leads WHERE city != '' GROUP BY city, uf "
                                "ORDER BY n DESC LIMIT 8").fetchall(),
            "sources": sorted(((s, n) for s, n in self._source_counts().items()), key=lambda x: -x[1]),
            "unscored": c.execute("SELECT count(*) FROM leads WHERE score IS NULL").fetchone()[0],
            "with_whatsapp": c.execute("SELECT count(*) FROM leads WHERE wa_number != '' AND status != ?",
                                       (DO_NOT_CONTACT,)).fetchone()[0],
            "reminders": self.reminders_due(7, 12),
            "overdue": c.execute("SELECT count(*) FROM reminders r JOIN leads l ON l.id=r.lead_id WHERE r.done=0 AND "
                                 "r.due_at <= ? AND l.status != ?", (now_iso(), DO_NOT_CONTACT)).fetchone()[0],
        }

    def _source_counts(self) -> dict:
        out: dict = {}
        for s, n in self.con.execute("SELECT source, count(DISTINCT lead_id) FROM lead_sources GROUP BY source"):
            out[s] = n
        return out
