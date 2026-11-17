"""Operações compostas executadas na thread do banco (recebem a ``Store`` como primeiro argumento)."""
from __future__ import annotations

from .scoring import score_lead


def rescore(store, ids: list[int] | None = None, only_missing: bool = False) -> int:
    """Recalcula a pontuação (critérios verificáveis) dos leads indicados, ou de todos."""
    q = "SELECT * FROM leads"
    args: list = []
    conds = []
    if ids:
        conds.append(f"id IN ({','.join('?' * len(ids))})"); args.extend(int(i) for i in ids)
    if only_missing:
        conds.append("score IS NULL")
    if conds:
        q += " WHERE " + " AND ".join(conds)
    rows = store.con.execute(q, args).fetchall()
    for r in rows:
        total, prio, detail = score_lead(dict(r))
        store.save_score(r["id"], total, prio, detail)
    return len(rows)


def ingest_and_score(store, leads) -> dict:
    stats = store.ingest(leads)
    rescore(store, only_missing=True)
    return stats


def site_targets(store, ids: list[int], limit: int) -> list[tuple[int, str, list[str]]]:
    """(id, site, telefones) dos leads selecionados que têm site próprio."""
    from .dedupe import is_generic
    out = []
    for i in ids:
        r = store.con.execute("SELECT id, website, domain, phone, phone2, wa_number FROM leads WHERE id=?",
                              (int(i),)).fetchone()
        if r and r["website"] and not is_generic(r["domain"]):
            out.append((r["id"], r["website"], [p for p in (r["phone"], r["phone2"], r["wa_number"]) if p]))
        if len(out) >= limit:
            break
    return out


def cnpj_targets(store, ids: list[int]) -> list[tuple[int, str]]:
    out = []
    for i in ids:
        r = store.con.execute("SELECT id, cnpj FROM leads WHERE id=?", (int(i),)).fetchone()
        if r and r["cnpj"]:
            out.append((r["id"], r["cnpj"]))
    return out
