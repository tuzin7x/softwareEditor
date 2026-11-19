"""Palavras-chave de pesquisa e como cada fonte as interpreta.

* Google Places: a palavra-chave entra no texto da busca ("imobiliária em Campinas - SP").
* OpenStreetMap: vira etiquetas oficiais (``office=estate_agent``...) e um filtro de nome.
* Base aberta de CNPJ: vira códigos CNAE (atividade econômica oficial) e, nas palavras personalizadas, filtro
  pelo nome da empresa.

A categoria exibida na tabela sai de regras verificáveis (CNAE, etiquetas, nome); nada é inventado.
"""
from __future__ import annotations

import re

from .geo import fold

DEFAULT_KEYWORDS: tuple[str, ...] = (
    "Imobiliária", "Loja de imóveis", "Corretora de imóveis", "Agência imobiliária", "Imóveis à venda",
    "Venda de imóveis", "Aluguel de imóveis", "Administração de imóveis", "Consultoria imobiliária",
    "Corretagem imobiliária", "Empreendimentos imobiliários", "Lançamentos imobiliários", "Construtora",
    "Incorporadora", "Corretores de imóveis", "Imóveis de luxo", "Imóveis comerciais",
)

# CNAE 2.3 (subclasses) do setor
CNAE: dict[str, str] = {
    "6821801": "Corretagem na compra e venda e avaliação de imóveis",
    "6821802": "Corretagem no aluguel de imóveis",
    "6822600": "Gestão e administração da propriedade imobiliária",
    "6810201": "Compra e venda de imóveis próprios",
    "6810202": "Aluguel de imóveis próprios",
    "6810203": "Loteamento de imóveis próprios",
    "4110700": "Incorporação de empreendimentos imobiliários",
    "4120400": "Construção de edifícios",
}
CNAE_CATEGORY = {
    "6821801": "Imobiliária", "6821802": "Imobiliária (aluguel)", "6822600": "Administradora de imóveis",
    "6810201": "Compra e venda de imóveis", "6810202": "Aluguel de imóveis próprios", "6810203": "Loteadora",
    "4110700": "Incorporadora", "4120400": "Construtora",
}
CORE_CNAES = ("6821801", "6821802", "6822600")

# palavra-chave padrão (dobrada) -> (CNAEs, etiquetas OSM, raiz para filtro de nome, tipo Google)
_RULES: dict[str, tuple[tuple[str, ...], tuple[str, ...], str, str]] = {
    "imobiliaria": (CORE_CNAES, ("office=estate_agent",), "imobili", "real_estate_agency"),
    "loja de imoveis": (("6821801",), ("office=estate_agent", "shop=estate_agent"), "imove", "real_estate_agency"),
    "corretora de imoveis": (("6821801", "6821802"), ("office=estate_agent",), "corret", "real_estate_agency"),
    "agencia imobiliaria": (("6821801",), ("office=estate_agent",), "imobili", "real_estate_agency"),
    "imoveis a venda": (("6821801", "6810201"), ("office=estate_agent",), "imove", "real_estate_agency"),
    "venda de imoveis": (("6821801", "6810201"), ("office=estate_agent",), "imove", "real_estate_agency"),
    "aluguel de imoveis": (("6821802", "6822600"), ("office=estate_agent",), "alug", "real_estate_agency"),
    "administracao de imoveis": (("6822600",), ("office=estate_agent", "office=property_management"), "administ",
                                 "real_estate_agency"),
    "consultoria imobiliaria": (("6821801",), ("office=estate_agent",), "consult", "real_estate_agency"),
    "corretagem imobiliaria": (("6821801", "6821802"), ("office=estate_agent",), "corret", "real_estate_agency"),
    "empreendimentos imobiliarios": (("4110700", "6810203"), ("office=estate_agent",), "empreend", ""),
    "lancamentos imobiliarios": (("4110700",), (), "lancament", ""),
    "construtora": (("4120400",), ("office=construction_company",), "construtora", ""),
    "incorporadora": (("4110700",), (), "incorpora", ""),
    "corretores de imoveis": (("6821801", "6821802"), ("office=estate_agent",), "corret", "real_estate_agency"),
    "imoveis de luxo": (("6821801",), ("office=estate_agent",), "imove", "real_estate_agency"),
    "imoveis comerciais": (("6821801", "6821802"), ("office=estate_agent",), "imove", "real_estate_agency"),
}


def rule(keyword: str) -> tuple[tuple[str, ...], tuple[str, ...], str, str] | None:
    return _RULES.get(fold(keyword))


def is_builtin(keyword: str) -> bool:
    return fold(keyword) in _RULES


def validate_custom(keyword: str) -> str:
    """Palavra-chave personalizada limpa; levanta ``ValueError`` se for vazia ou estranha."""
    k = re.sub(r"\s+", " ", (keyword or "").strip())
    if len(k) < 3 or len(k) > 60:
        raise ValueError("Use de 3 a 60 caracteres.")
    if not re.search(r"\w", k) or re.search(r"[<>\"{}\\]", k):
        raise ValueError("Use só letras, números e pontuação simples.")
    return k


def cnaes_for(keywords: list[str]) -> list[str]:
    out: list[str] = []
    for k in keywords:
        r = rule(k)
        for c in (r[0] if r else ()):
            if c not in out:
                out.append(c)
    return out


def custom_terms(keywords: list[str]) -> list[str]:
    """Palavras personalizadas (não padrão), dobradas, para filtrar pelo nome."""
    return [fold(k) for k in keywords if not is_builtin(k) and fold(k)]


def osm_filters(keywords: list[str]) -> tuple[list[str], str]:
    """(etiquetas ``chave=valor`` sem repetição, regex de nome em minúsculas sem acento)."""
    tags: list[str] = []
    stems: list[str] = []
    for k in keywords:
        r = rule(k)
        if r:
            tags.extend(t for t in r[1] if t not in tags)
            if r[2] and r[2] not in stems:
                stems.append(r[2])
        else:
            f = fold(k)
            if f and f not in stems:
                stems.append(re.escape(f))
    return tags, "|".join(stems)


def google_type(keyword: str) -> str:
    r = rule(keyword)
    return r[3] if r else ""


_CATEGORY_RULES = (
    (r"incorporad", "Incorporadora"),
    (r"construtor|construc|engenharia", "Construtora"),
    (r"lancament|empreendiment|loteament|urbaniz", "Empreendimentos"),
    (r"administra|condominio|gestao", "Administradora de imóveis"),
    (r"alug|locac|locador", "Imobiliária (aluguel)"),
    (r"imobili|imove|corret|realty|real estate|broker", "Imobiliária"),
)


def classify(name: str = "", cnae: str = "", hints: tuple[str, ...] = ()) -> str:
    """Categoria por regras verificáveis: CNAE oficial primeiro; depois etiquetas da fonte e o nome."""
    if cnae and cnae[:7] in CNAE_CATEGORY:
        return CNAE_CATEGORY[cnae[:7]]
    text = fold(" ".join((name, *hints)))
    for pat, cat in _CATEGORY_RULES:
        if re.search(pat, text):
            return cat
    if any(h in ("estate_agent", "real_estate_agency") for h in hints):
        return "Imobiliária"
    return "Outro"
