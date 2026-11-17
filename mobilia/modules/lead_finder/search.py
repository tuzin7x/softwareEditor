"""Orquestração de uma pesquisa: percorre as fontes escolhidas, respeita limites de custo e entrega lotes.

Roda numa thread de trabalho (``tasks.Task``). Não grava nada: entrega ``("batch", fonte, [Lead])`` e
``("usage", fonte, unidades, custo)`` pelo ``ctx.emit``; a interface repassa ao banco (``DbService``).
"""
from __future__ import annotations

from .models import SearchQuery
from .net import Cancelled, NetError
from .sources import Source, SourceUnavailable

BATCH = 400


class Budget:
    """Limite de requisições pagas: por pesquisa e no mês (já usado no mês vem do banco)."""

    def __init__(self, per_search: int, monthly_cap: int, used_month: int):
        self.left_search = max(0, int(per_search))
        self.left_month = max(0, int(monthly_cap) - int(used_month))
        self.used = 0
        self.denied = 0

    def __call__(self, units: int = 1) -> bool:
        if units > self.left_search or units > self.left_month:
            self.denied += units
            return False
        self.left_search -= units
        self.left_month -= units
        self.used += units
        return True


def _flush_usage(ctx, src: Source, budget: Budget | None, st: dict) -> None:
    """Emite o uso pago ainda não registrado. A unidade é consumida antes da requisição sair, então uma consulta
    cancelada no meio também conta (preferimos superestimar a estourar o limite mensal)."""
    if budget is None or budget.used <= st["paid"]:
        return
    n = budget.used - st["paid"]
    ctx.emit(("usage", src.id, n, n * src.settings.google_price_per_1000_usd / 1000))
    st["paid"] = budget.used


def run_search(ctx, query: SearchQuery, sources: dict[str, Source], google_used_month: int = 0) -> dict:
    summary: dict[str, dict] = {}
    chosen = [sources[s] for s in query.sources if s in sources]
    plans = []
    for src in chosen:
        ok, why = src.availability()
        st = summary.setdefault(src.id, {"units": 0, "found": 0, "errors": [], "skipped": "", "paid": 0})
        if not ok:
            st["skipped"] = why
            ctx.emit(("skipped", src.id, why))
            continue
        units = src.plan(query)
        st["units"] = len(units)
        plans.append((src, units))
    total = sum(len(u) for _, u in plans) or 1
    done = 0
    for src, units in plans:
        st = summary[src.id]
        budget = None
        if src.id == "google_places":
            s = src.settings
            budget = Budget(s.google_max_requests_per_search, s.google_monthly_request_cap, google_used_month)
        for unit in units:
            ctx.check()
            ctx.progress(done / total, f"{src.title}: {unit.label}")
            batch = []
            try:
                it = src.fetch(unit, query, ctx, budget=budget) if budget is not None else src.fetch(unit, query, ctx)
                for lead in it:
                    batch.append(lead)
                    if len(batch) >= BATCH:
                        st["found"] += len(batch)
                        ctx.emit(("batch", src.id, batch)); batch = []
            except Cancelled:
                raise
            except (NetError, SourceUnavailable) as e:
                st["errors"].append(f"{unit.label}: {e}")
                ctx.emit(("unit_error", src.id, unit.label, str(e)))
            finally:
                # roda também no cancelamento e em erros inesperados: o que já chegou é entregue e toda
                # requisição paga já enviada (unidade consumida do limite) é contabilizada no uso do mês
                if batch:
                    st["found"] += len(batch)
                    ctx.emit(("batch", src.id, batch))
                _flush_usage(ctx, src, budget, st)
            done += 1
            if budget is not None and (budget.left_search <= 0 or budget.left_month <= 0):
                rest = len(units) - units.index(unit) - 1
                if rest:
                    st["skipped"] = (f"Limite de custo atingido: {rest} consulta(s) não feita(s). "
                                     "Ajuste os limites em Configurações.")
                    ctx.emit(("skipped", src.id, st["skipped"]))
                    done += rest
                break
    ctx.progress(1.0, "Pesquisa concluída")
    return summary
