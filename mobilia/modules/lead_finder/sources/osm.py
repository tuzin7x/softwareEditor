"""OpenStreetMap pela Overpass API: gratuita, sem chave, dados sob a licença ODbL.

Uso permitido: consultar, guardar e exportar, com atribuição "© colaboradores do OpenStreetMap" (bases derivadas
publicadas ficam sob ODbL). A política de uso da Overpass pede consultas moderadas e um User-Agent identificável:
uma consulta por área, com intervalo mínimo entre elas.

Cobertura: o OSM depende de voluntários; no Brasil muitas imobiliárias não estão mapeadas ou não têm telefone.
"""
from __future__ import annotations

import re
from urllib.parse import quote

from .. import keywords as kw
from ..geo import UF_NAME, scope_ufs
from ..models import Lead, SearchQuery, SourcePolicy
from .base import Estimate, Source, Unit

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

    def fetch(self, unit: Unit, query: SearchQuery, ctx):
        p = unit.params
        timeout = 60 if p["municipality"] else 120
        ql = build_query(p["uf"], p["municipality"], p["tags"], p["rx"], timeout)
        url = self.settings.overpass_url if self.settings else "https://overpass-api.de/api/interpreter"
        data = self.http.post_form(url, "data=" + quote(ql, safe=""), ctx=ctx, timeout=timeout + 15)
        seen: set[str] = set()
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
