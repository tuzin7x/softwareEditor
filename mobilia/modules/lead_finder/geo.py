"""Geografia do Brasil: estados, regiões e municípios (API de localidades do IBGE, com cache local).

Os municípios vêm do serviço oficial de dados do IBGE (gratuito, sem chave) e ficam guardados em
``<dados do app>/lead_finder/cache/ibge_<UF>.json`` para funcionar sem internet depois da primeira consulta.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

REGIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "N": ("Norte", ("AC", "AM", "AP", "PA", "RO", "RR", "TO")),
    "NE": ("Nordeste", ("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE")),
    "CO": ("Centro-Oeste", ("DF", "GO", "MS", "MT")),
    "SE": ("Sudeste", ("ES", "MG", "RJ", "SP")),
    "S": ("Sul", ("PR", "RS", "SC")),
}
REGION_NAME = {k: v[0] for k, v in REGIONS.items()}

UF_NAME = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia", "CE": "Ceará",
    "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão", "MG": "Minas Gerais",
    "MS": "Mato Grosso do Sul", "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco",
    "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RO": "Rondônia",
    "RR": "Roraima", "RS": "Rio Grande do Sul", "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo",
    "TO": "Tocantins",
}
UFS = tuple(sorted(UF_NAME))
UF_REGION = {uf: r for r, (_, ufs) in REGIONS.items() for uf in ufs}
CAPITALS = {
    "AC": "Rio Branco", "AL": "Maceió", "AM": "Manaus", "AP": "Macapá", "BA": "Salvador", "CE": "Fortaleza",
    "DF": "Brasília", "ES": "Vitória", "GO": "Goiânia", "MA": "São Luís", "MG": "Belo Horizonte",
    "MS": "Campo Grande", "MT": "Cuiabá", "PA": "Belém", "PB": "João Pessoa", "PE": "Recife", "PI": "Teresina",
    "PR": "Curitiba", "RJ": "Rio de Janeiro", "RN": "Natal", "RO": "Porto Velho", "RR": "Boa Vista",
    "RS": "Porto Alegre", "SC": "Florianópolis", "SE": "Aracaju", "SP": "São Paulo", "TO": "Palmas",
}
_NAME_TO_UF = {}

IBGE_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"


def fold(text: str) -> str:
    """Minúsculas, sem acentos e com espaços simples (para comparar nomes de cidades e empresas)."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def uf_from_text(text: str) -> str:
    """Sigla da UF a partir da sigla ou do nome do estado ("SP", "São Paulo", "sao paulo"); vazio se não achar."""
    t = (text or "").strip()
    if t.upper() in UF_NAME:
        return t.upper()
    if not _NAME_TO_UF:
        _NAME_TO_UF.update({fold(n): uf for uf, n in UF_NAME.items()})
    return _NAME_TO_UF.get(fold(t), "")


def scope_ufs(scope) -> list[str]:
    """UFs cobertas por um :class:`GeoScope`."""
    if scope.kind == "pais":
        return list(UFS)
    if scope.kind == "regiao":
        return list(REGIONS.get(scope.region, ("", ()))[1])
    if scope.kind == "uf":
        return [u for u in scope.ufs if u in UF_NAME]
    return sorted({u for _, u in scope.municipalities if u in UF_NAME})


def cache_file(cache_dir: Path, uf: str) -> Path:
    return Path(cache_dir) / f"ibge_{uf}.json"


def cached_municipalities(cache_dir: Path, uf: str) -> list[str]:
    f = cache_file(cache_dir, uf)
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return [str(n) for n in data if isinstance(n, str)]
    except (OSError, ValueError):
        return []


def fetch_municipalities(http, uf: str, cache_dir: Path, ctx=None) -> list[str]:
    """Lista oficial de municípios da UF (IBGE). Usa o cache local; só vai à rede se não houver cache."""
    uf = uf.upper()
    if uf not in UF_NAME:
        raise ValueError(f"UF inválida: {uf}")
    names = cached_municipalities(cache_dir, uf)
    if names:
        return names
    data = http.get_json(IBGE_URL.format(uf=uf), ctx=ctx)
    names = sorted({str(m["nome"]) for m in data if isinstance(m, dict) and m.get("nome")}, key=fold)
    if names:
        f = cache_file(cache_dir, uf)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    return names


def match_municipality(name: str, known: list[str]) -> str:
    """Nome oficial que corresponde a ``name`` (ignora acentos e caixa); vazio se não existir."""
    k = fold(name)
    for n in known:
        if fold(n) == k:
            return n
    return ""
