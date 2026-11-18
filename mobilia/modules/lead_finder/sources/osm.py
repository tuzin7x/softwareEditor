"""OpenStreetMap pela Overpass API: gratuita, sem chave, dados sob a licença ODbL.

Uso permitido: consultar, guardar e exportar, com atribuição "© colaboradores do OpenStreetMap" (bases derivadas
publicadas ficam sob ODbL). A política de uso da Overpass pede consultas moderadas e um User-Agent identificável:
uma consulta por área, com intervalo mínimo entre elas.

Cobertura: o OSM depende de voluntários; no Brasil muitas imobiliárias não estão mapeadas ou não têm telefone.

Teste real (Campinas): a consulta única da cidade (todas as etiquetas + a busca por nome com regex) falhou nos três
servidores públicos (502/504/tempo esgotado no overpass-api.de e no mail.ru; 403 no openstreetmap.fr), enquanto uma
consulta PEQUENA funcionou. Por isso: a área é consultada em pedaços (uma etiqueta ou uma chave da busca por nome
por pedido, com ``[timeout]`` curto, que os servidores aceitam mais fácil), cada pedaço tenta os servidores públicos
em ordem (o último que funcionou, o das configurações e os de ``OVERPASS_ENDPOINTS``) com uma
espera crescente entre eles, e a mensagem de erro diz a verdade: a Overpass não tem chave — o servidor recusou ou
limitou o pedido.
"""
from __future__ import annotations

import re
from urllib.parse import quote

from urllib.parse import urlsplit

from .. import keywords as kw
from ..geo import UF_NAME, scope_ufs
from ..models import Lead, SearchQuery, SourcePolicy
from ..net import NetError
from .base import Estimate, Source, Unit

# servidores públicos da Overpass (sem chave), na ordem de preferência
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
)
PIECE_TIMEOUT_CITY = 30     # s declarados na consulta de um pedaço de cidade (pedidos curtos passam na fila)
PIECE_TIMEOUT_STATE = 90
SWITCH_BACKOFF_S = 2.0      # espera antes do 2º servidor (cresce a cada troca)

_ACCENTS = {"a": "aáàâã", "e": "eéê", "i": "ií", "o": "oóôõ", "u": "uúü", "c": "cç"}


def accent_regex(stems: str) -> str:
    """"imove|corret" -> regex que também casa "imóveis" (a Overpass não ignora acentos)."""
    return "".join(f"[{_ACCENTS[ch]}]" if ch in _ACCENTS else ch for ch in stems)


def _q(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_query(uf: str, municipality: str, tags: list[str], name_regex: str, timeout: int = 90) -> str:
    if municipality:
        area = (f'area["ISO3166-2"="BR-{uf}"]["boundary"="administrative"]->.uf;\n'
                f'rel(area.uf)["boundary"="administrative"]["admin_level"="8"]["name"="{_q(municipality)}"];\n'
                f"map_to_area->.a;\n")
    else:
        area = f'area["ISO3166-2"="BR-{uf}"]["boundary"="administrative"]->.a;\n'
    parts = []
    for t in tags:
        k, v = t.split("=", 1)
        parts.append(f'  nwr["{_q(k)}"="{_q(v)}"](area.a);')
    if name_regex:
        rx = _q(accent_regex(name_regex))
        for key in ("office", "shop", "craft"):
            parts.append(f'  nwr["name"~"{rx}",i]["{key}"](area.a);')
    body = "\n".join(parts)
    return f"[out:json][timeout:{timeout}];\n{area}(\n{body}\n);\nout center tags 2000;\n"


def build_queries(uf: str, municipality: str, tags: list[str], name_regex: str, timeout: int = 30) -> list[str]:
    """A consulta da área em PEDAÇOS: uma etiqueta por pedido e uma chave (office/shop/craft) da busca por nome por
    pedido. Cada pedaço é pequeno e rápido; somados, o mesmo resultado de :func:`build_query`."""
    out = [build_query(uf, municipality, [t], "", timeout) for t in tags]
    if name_regex:
        for key in ("office", "shop", "craft"):
            out.append(_name_query(uf, municipality, name_regex, key, timeout))
    return out


def _name_query(uf: str, municipality: str, name_regex: str, key: str, timeout: int) -> str:
    full = build_query(uf, municipality, [], name_regex, timeout)
    lines = [ln for ln in full.splitlines() if not ln.startswith("  nwr[") or f'["{key}"]' in ln]
    return "\n".join(lines) + "\n"


def endpoints(configured: str | None = None, preferred: str | None = None) -> list[str]:
    """Servidores a tentar: o último que funcionou nesta busca, o das configurações e os públicos (sem repetir)."""
    out: list[str] = []
    for u in (preferred, configured, *OVERPASS_ENDPOINTS):
        if u and u.startswith("https://") and u not in out:
            out.append(u)
    return out


def _host(url: str) -> str:
    return urlsplit(url).hostname or url


def _why(e: NetError) -> str:
    if e.status == 403:
        return "recusou (403)"
    if e.status == 429:
        return "limite de uso (429)"
    if e.status:
        return f"erro {e.status}"
    return "tempo esgotado ou sem conexão"


def _first(tags: dict, *keys: str) -> str:
    for k in keys:
        v = (tags.get(k) or "").strip()
        if v:
            return v.split(";")[0].strip()
    return ""


class OsmSource(Source):
    id = "osm"
    title = "OpenStreetMap"
    description = "Mapa colaborativo aberto (Overpass API). Gratuito, sem chave; cobertura variável."
    policy = SourcePolicy(store_days=None, exportable=True,
                          attribution="© colaboradores do OpenStreetMap (ODbL 1.0)",
                          terms_url="https://www.openstreetmap.org/copyright")

    def availability(self):
        if self.settings is not None and not self.settings.osm_enabled:
            return False, "Desativada nas configurações."
        return True, "Gratuita · uma consulta por área"

    def plan(self, query: SearchQuery) -> list[Unit]:
        tags, rx = kw.osm_filters(query.keywords)
        if not tags and not rx:
            return []
        if query.scope.kind == "municipios":
            return [Unit(f"{m} - {u}", {"uf": u, "municipality": m, "tags": tags, "rx": rx}, 0)
                    for m, u in query.scope.municipalities]
        return [Unit(UF_NAME[u], {"uf": u, "municipality": "", "tags": tags, "rx": rx}, 0)
                for u in scope_ufs(query.scope)]

    def estimate(self, query: SearchQuery) -> Estimate:
        n = len(self.plan(query))
        return Estimate(n, 0, 0.0, f"{n} consulta(s) gratuita(s) ao OpenStreetMap")

    _preferred: str | None = None

    def _post(self, ql: str, ctx, timeout: int):
        """Um pedaço da consulta, tentando os servidores públicos em ordem (com espera crescente entre eles)."""
        conf = self.settings.overpass_url if self.settings else None
        fails: list[str] = []
        for i, url in enumerate(endpoints(conf, self._preferred)):
            if i:
                self.http._wait(ctx, SWITCH_BACKOFF_S * i)
            try:
                data = self.http.post_form(url, "data=" + quote(ql, safe=""), ctx=ctx, timeout=timeout + 15,
                                           retries=1)
            except NetError as e:
                if e.status == 400:        # consulta inválida: os outros servidores diriam o mesmo
                    raise
                fails.append(f"{_host(url)}: {_why(e)}")
                continue
            if isinstance(data, dict) and "elements" not in data and data.get("remark"):
                fails.append(f"{_host(url)}: {str(data['remark'])[:80]}")      # tempo/memória esgotados lá
                continue
            self._preferred = url
            return data
        raise NetError("Os servidores públicos do OpenStreetMap (Overpass) recusaram ou limitaram a consulta ("
                       + "; ".join(fails) + "). A Overpass não usa chave: os servidores gratuitos ficam "
                       "sobrecarregados; tente de novo em alguns minutos ou escolha uma área menor.", 0, False)

    def fetch(self, unit: Unit, query: SearchQuery, ctx):
        p = unit.params
        timeout = PIECE_TIMEOUT_CITY if p["municipality"] else PIECE_TIMEOUT_STATE
        pieces = build_queries(p["uf"], p["municipality"], p["tags"], p["rx"], timeout)
        seen: set[str] = set()
        failed: list[str] = []
        ok = 0
        for ql in pieces:
            if len(failed) >= 2 and not ok:
                failed.append("")         # todos os servidores fora duas vezes seguidas: não insiste nos outros
                break
            try:
                data = self._post(ql, ctx, timeout)
            except NetError as e:
                if e.status == 400:
                    raise
                failed.append(str(e))
                continue
            ok += 1
            yield from self._parse(data, unit, p, seen)
        failed = [f for f in failed if f] or failed
        if failed:
            # o que chegou dos outros pedaços já foi entregue; a falha aparece no resumo da busca
            msg = next(f for f in reversed(failed) if f) if any(failed) else "Falha na consulta."
            raise NetError(msg if not ok else f"{len(pieces) - ok} de {len(pieces)} partes da consulta falharam "
                           f"(as outras chegaram). {msg}")

    def _parse(self, data, unit: Unit, p: dict, seen: set[str]):
        for el in data.get("elements", []) if isinstance(data, dict) else []:
            ref = f"{el.get('type')}/{el.get('id')}"
            tags = el.get("tags") or {}
            name = _first(tags, "name", "brand", "operator")
            if not name or ref in seen:
                continue
            seen.add(ref)
            street = " ".join(x for x in (_first(tags, "addr:street"), _first(tags, "addr:housenumber")) if x)
            address = ", ".join(x for x in (street, _first(tags, "addr:suburb")) if x)
            hints = tuple(v for k, v in tags.items() if k in ("office", "shop", "craft"))
            yield Lead(
                source=self.id, source_ref=ref, name=name,
                category=kw.classify(name, hints=hints),
                address=address, city=_first(tags, "addr:city") or p["municipality"], uf=p["uf"],
                cep=re.sub(r"\D", "", _first(tags, "addr:postcode")),
                phone=_first(tags, "phone", "contact:phone"),
                phone2=_first(tags, "contact:mobile", "mobile"),
                whatsapp=_first(tags, "contact:whatsapp", "whatsapp"),
                website=_first(tags, "website", "contact:website", "url"),
                cnpj=re.sub(r"\D", "", _first(tags, "ref:CNPJ", "ref:cnpj")),
                query=unit.label,
                extra={"osm_tags": {k: tags[k] for k in ("office", "shop", "craft") if k in tags}})
