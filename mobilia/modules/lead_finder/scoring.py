"""Qualificação comercial por critérios verificáveis, cada um com a sua justificativa.

Só pontua o que está nos dados (fonte informada) ou na análise do site oficial. Critério sem dado aparece como
"não avaliado", nunca como positivo ou negativo. Total de 0 a 100; prioridade: Alta ≥ 70, Média ≥ 40, Baixa.
"""
from __future__ import annotations

import json

from . import dedupe
from .models import WA_CONFIRMED, WA_LIKELY
from .sources import TITLES

SEGMENT_POINTS = {
    "Incorporadora": (20, "Incorporadoras vendem lançamentos: imagens e vídeos de ambientes decorados são centrais."),
    "Empreendimentos": (20, "Empreendimentos e lançamentos dependem de apresentação visual de unidades."),
    "Construtora": (16, "Construtoras apresentam unidades prontas e vazias: boa aderência à ambientação virtual."),
    "Imobiliária": (15, "Imobiliárias anunciam muitos imóveis e podem usar ambientação e vídeos nos anúncios."),
    "Compra e venda de imóveis": (15, "Venda de imóveis se beneficia de fotos ambientadas."),
    "Loteadora": (8, "Loteamentos usam menos fotos de interiores."),
    "Imobiliária (aluguel)": (12, "Locação: imóveis vazios entre contratos podem ser ambientados virtualmente."),
    "Administradora de imóveis": (10, "Administradoras lidam com imóveis vagos entre locações."),
    "Aluguel de imóveis próprios": (6, "Aluguel de imóveis próprios: carteira pequena, aderência menor."),
}


def _item(label: str, points: int, reason: str, evidence: str = "", evaluated: bool = True) -> dict:
    return {"criterio": label, "pontos": points, "justificativa": reason, "evidencia": evidence,
            "avaliado": evaluated}


def score_lead(lead: dict) -> tuple[int, int, list[dict]]:
    """``lead`` = linha do banco (dict). Devolve (pontuação, prioridade 1-3, detalhes)."""
    origin = lead.get("origin") or {}
    if isinstance(origin, str):
        origin = json.loads(origin or "{}")
    src = lambda f: TITLES.get(origin.get(f, ""), origin.get(f, "fonte"))  # noqa: E731
    items: list[dict] = []
    domain = lead.get("domain") or ""
    if domain and not dedupe.is_generic(domain):
        items.append(_item("Site oficial", 20, "Tem site próprio: presença digital estabelecida.",
                           f"{domain} (informado por {src('website')})"))
    elif domain and dedupe.is_social(domain):
        items.append(_item("Presença digital", 8, "Só perfil em rede social informado.", domain))
    else:
        items.append(_item("Site oficial", 0, "Nenhum site informado pelas fontes.", evaluated=bool(lead.get("sources"))))

    if lead.get("phone_e164"):
        items.append(_item("Telefone comercial", 10, "Telefone válido informado pela fonte.",
                           f"{lead.get('phone_kind') or 'telefone'} (fonte: {src('phone')})"))
    else:
        items.append(_item("Telefone comercial", 0, "Nenhum telefone válido nas fontes."))

    if lead.get("wa_status") == WA_CONFIRMED:
        items.append(_item("WhatsApp", 10, "WhatsApp confirmado (fonte, site oficial ou você).", lead.get("wa_number", "")))
    elif lead.get("wa_status") == WA_LIKELY:
        items.append(_item("WhatsApp", 4, "Número de celular: WhatsApp provável, ainda não confirmado.",
                           lead.get("wa_number", "")))
    else:
        items.append(_item("WhatsApp", 0, "Sem número de WhatsApp conhecido."))

    cat = lead.get("category") or ""
    pts, why = SEGMENT_POINTS.get(cat, (0, "Segmento fora do foco imobiliário ou não identificado."))
    items.append(_item("Segmento", pts, why, cat or "—", evaluated=bool(cat)))

    if (lead.get("situacao") or "").upper() == "ATIVA":
        items.append(_item("Cadastro ativo", 5, "CNPJ com situação cadastral ativa na Receita.",
                           f"CNPJ {lead.get('cnpj', '')}"))

    check = lead.get("site_check") or ""
    check = json.loads(check) if isinstance(check, str) and check else (check or {})
    if check.get("ok"):
        t = check.get("totals", {})
        terms = check.get("terms", {})
        def top(k):
            return ", ".join(f"“{w}” ×{n}" for w, n in sorted(terms.get(k, {}).items(), key=lambda x: -x[1])[:3])
        n = t.get("listings", 0)
        items.append(_item("Anúncios de imóveis", 15 if n >= 10 else 8 if n >= 3 else 0,
                           "Site com anúncios de imóveis." if n >= 3 else "Poucas menções a anúncios no site.",
                           top("listings") or "nenhuma menção"))
        e = t.get("empty", 0)
        items.append(_item("Imóveis vazios / prontos", 10 if e >= 3 else 5 if e else 0,
                           "Menções a imóveis vazios, desocupados ou prontos: oportunidade de ambientação virtual."
                           if e else "Sem menções a imóveis vazios na página principal.", top("empty") or "—"))
        lc = t.get("launch", 0)
        if lc:
            items.append(_item("Lançamentos", 5, "Site menciona lançamentos/empreendimentos.", top("launch")))
        if t.get("staging"):
            items.append(_item("Já usa recursos visuais", 0,
                               "O site já menciona decorado virtual/tour: avalie como cliente maduro ou concorrência.",
                               top("staging")))
    else:
        items.append(_item("Análise do site", 0,
                           check.get("summary") or "Site ainda não analisado (use “Analisar site”).", evaluated=False))

    total = max(0, min(100, sum(i["pontos"] for i in items)))
    priority = 3 if total >= 70 else 2 if total >= 40 else 1
    return total, priority, items
