"""Consolidação de registros de fontes diferentes: chaves de duplicidade e escolha de cada campo.

Duas empresas são a mesma quando têm (em ordem de confiança) o mesmo registro na fonte, o mesmo CNPJ, o mesmo
telefone, o mesmo domínio de site próprio ou o mesmo nome (sem sufixos societários) na mesma cidade.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

from .geo import fold
from .sources import PRIORITY

# domínios que não identificam uma empresa (redes sociais, portais, encurtadores)
SOCIAL = ("facebook.com", "instagram.com", "linkedin.com", "youtube.com", "tiktok.com", "twitter.com", "x.com",
          "linktr.ee", "wa.me", "whatsapp.com", "threads.net", "pinterest.com")
PORTALS = ("zapimoveis.com.br", "vivareal.com.br", "imovelweb.com.br", "olx.com.br", "chavesnamao.com.br",
           "quintoandar.com.br", "google.com", "goo.gl", "g.page", "business.site", "sites.google.com",
           "wixsite.com", "bit.ly", "creci.org.br")
_LEGAL = re.compile(r"\b(ltda|me|epp|eireli|s ?a|s ?s|mei|cia|limitada|sociedade|empresarial)\b")


def domain_of(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        u = "http://" + u
    try:
        host = (urlsplit(u).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def is_generic(domain: str) -> bool:
    return not domain or any(domain == g or domain.endswith("." + g) for g in SOCIAL + PORTALS)


def is_social(domain: str) -> bool:
    return any(domain == g or domain.endswith("." + g) for g in SOCIAL)


def name_key(name: str, city: str, uf: str) -> str:
    n = _LEGAL.sub(" ", fold(name))
    n = re.sub(r"\s+", " ", n).strip()
    if len(n) < 3 or not city:
        return ""
    return f"{n}|{fold(city)}|{(uf or '').upper()}"


def merge(records: list[tuple[str, dict, str]]) -> tuple[dict, dict]:
    """``records`` = [(fonte, campos, data)]. Devolve (campos consolidados, fonte de cada campo).

    Para cada campo vence a fonte de maior prioridade (manual > oficial > site > Google > OSM); empate: a mais
    recente."""
    ordered = sorted(records, key=lambda r: (PRIORITY.get(r[0], 0), r[2] or ""), reverse=True)
    out: dict = {}
    origin: dict = {}
    for src, fields, _ in ordered:
        for k, v in fields.items():
            if v and k not in out:
                out[k] = v
                origin[k] = src
    return out, origin
