"""Google Places API (New) — Text Search. Precisa da chave do próprio usuário (Google Cloud, faturamento ativo).

Termos do Google Maps Platform (resumo aplicado aqui, confira sempre o texto atual):

* o ``place_id`` pode ser guardado sem limite;
* os demais dados (nome, endereço, telefone, site) **não** podem formar uma base permanente: ficam no máximo
  30 dias no banco local e depois são apagados (o lead fica com o ``place_id`` e pode ser atualizado);
* esses dados **não** são exportados para CSV.

Custo: cada requisição com telefone e site entra na faixa "Text Search Enterprise" (preço por 1.000 configurável).
Há limite por pesquisa e limite mensal, contados no banco local. Requisições pagas **não** têm novas tentativas
automáticas: cada envio real passa pelo limite e é registrado; uma falha aparece para o usuário, que decide repetir.
"""
from __future__ import annotations

import re

from .. import keywords as kw
from ..geo import UF_NAME, scope_ufs, uf_from_text
from ..models import Lead, SearchQuery, SourcePolicy
from ..net import NetError
from .base import Estimate, Source, SourceUnavailable, Unit

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
DETAILS_URL = "https://places.googleapis.com/v1/places/{id}"
_FIELDS = ("id", "displayName", "formattedAddress", "addressComponents", "nationalPhoneNumber",
           "internationalPhoneNumber", "websiteUri", "types", "primaryTypeDisplayName", "businessStatus")
SEARCH_MASK = ",".join(f"places.{f}" for f in _FIELDS) + ",nextPageToken"
DETAILS_MASK = ",".join(_FIELDS)


def parse_place(p: dict, query_label: str = "") -> Lead | None:
    pid = p.get("id")
    name = (p.get("displayName") or {}).get("text", "").strip()
    if not pid or not name or p.get("businessStatus") == "CLOSED_PERMANENTLY":
        return None
    city = uf = cep = ""
    for c in p.get("addressComponents") or []:
        types = c.get("types") or []
        if "administrative_area_level_2" in types:
            city = c.get("longText", "")
        elif "administrative_area_level_1" in types:
            uf = uf_from_text(c.get("shortText", "")) or uf_from_text(c.get("longText", ""))
        elif "postal_code" in types:
            cep = re.sub(r"\D", "", c.get("longText", ""))
    types = tuple(p.get("types") or ())
    primary = (p.get("primaryTypeDisplayName") or {}).get("text", "")
    return Lead(source="google_places", source_ref=pid, name=name,
                category=kw.classify(name, hints=types + ((primary,) if primary else ())),
                address=p.get("formattedAddress", ""), city=city, uf=uf, cep=cep,
                phone=p.get("internationalPhoneNumber") or p.get("nationalPhoneNumber") or "",
                website=p.get("websiteUri", ""), query=query_label,
                extra={"types": list(types)})


class GooglePlacesSource(Source):
    id = "google_places"
    title = "Google Places"
    description = "Empresas do Google Maps pela API oficial. Precisa de chave própria; consultas pagas."
    policy = SourcePolicy(store_days=30, exportable=False, attribution="Dados do Google Maps",
                          permanent_fields=("source_ref",),
                          terms_url="https://cloud.google.com/maps-platform/terms")

    def _key(self) -> str:
        return self.secrets.get("google_places") if self.secrets is not None else ""

    def availability(self):
        if self.settings is not None and not self.settings.google_enabled:
            return False, "Desativada. Ative e informe a chave nas configurações."
        if not self._key():
            return False, "Sem chave de API. Informe a sua chave nas configurações."
        return True, "Paga por requisição · com limites de custo"

    def localities(self, query: SearchQuery) -> list[str]:
        if query.scope.kind == "municipios":
            return [f"{m} - {u}" for m, u in query.scope.municipalities]
        return [UF_NAME[u] for u in scope_ufs(query.scope)]

    def plan(self, query: SearchQuery) -> list[Unit]:
        pages = max(1, min(3, self.settings.google_max_pages if self.settings else 1))
        return [Unit(f"{k} · {loc}", {"text": f"{k} em {loc}, Brasil", "type": kw.google_type(k), "pages": pages},
                     pages)
                for loc in self.localities(query) for k in query.keywords]

    def estimate(self, query: SearchQuery) -> Estimate:
        units = self.plan(query)
        cap = self.settings.google_max_requests_per_search if self.settings else 20
        paid = min(cap, sum(u.cost_units for u in units))
        price = self.settings.google_price_per_1000_usd if self.settings else 35.0
        note = (f"Até {paid} requisição(ões) paga(s) ao Google (≈ US$ {paid * price / 1000:.2f})"
                + (f"; {sum(u.cost_units for u in units) - paid} além do limite por pesquisa serão puladas"
                   if sum(u.cost_units for u in units) > paid else ""))
        return Estimate(len(units), paid, paid * price / 1000, note)

    def _headers(self, mask: str) -> dict:
        key = self._key()
        if not key:
            raise SourceUnavailable("Sem chave de API do Google.")
        return {"X-Goog-Api-Key": key, "X-Goog-FieldMask": mask}

    def fetch(self, unit: Unit, query: SearchQuery, ctx, budget=None):
        body = {"textQuery": unit.params["text"], "languageCode": "pt-BR", "regionCode": "BR", "pageSize": 20}
        if unit.params.get("type"):
            body["includedType"] = unit.params["type"]
        for _page in range(unit.params.get("pages", 1)):
            if budget is not None and not budget(1):
                return
            # paga: sem novas tentativas automáticas (cada tentativa real consome e registra uma unidade do limite)
            data = self.http.post_json(SEARCH_URL, body, ctx=ctx, headers=self._headers(SEARCH_MASK), retries=0)
            for p in data.get("places", []) if isinstance(data, dict) else []:
                lead = parse_place(p, unit.label)
                if lead is not None:
                    yield lead
            token = data.get("nextPageToken") if isinstance(data, dict) else None
            if not token:
                return
            body = {**body, "pageToken": token}

    def details(self, place_id: str, ctx=None) -> Lead | None:
        """Atualiza um lugar pelo ``place_id`` (Place Details: também é cobrado)."""
        if not re.fullmatch(r"[\w-]{10,300}", place_id or ""):
            raise NetError("Identificador do Google inválido.")
        data = self.http.get_json(DETAILS_URL.format(id=place_id), ctx=ctx, headers=self._headers(DETAILS_MASK),
                                  retries=0)
        return parse_place(data, "atualização") if isinstance(data, dict) else None
