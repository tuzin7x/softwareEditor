"""Fontes de dados autorizadas do Lead Finder.

| id              | fonte                                   | custo          | guarda / exportação          |
|-----------------|-----------------------------------------|----------------|------------------------------|
| ``osm``         | OpenStreetMap (Overpass API)            | gratuita       | sem limite / sim (ODbL)      |
| ``cnpj_aberto`` | Receita Federal, dados abertos (local)  | gratuita       | sem limite / sim             |
| ``google_places``| Google Places API (New), chave própria | paga           | 30 dias / não                |
| ``brasilapi``   | BrasilAPI, consulta de CNPJ             | gratuita       | sem limite / sim             |
| ``manual``      | digitado pelo usuário                   | —              | sem limite / sim             |
"""
from __future__ import annotations

from ..models import SourcePolicy
from .base import Estimate, Source, SourceUnavailable, Unit
from .brasilapi import BrasilApiSource
from .cnpj_open import CnpjOpenSource
from .google_places import GooglePlacesSource
from .osm import OsmSource

SEARCH_SOURCES = (OsmSource, CnpjOpenSource, GooglePlacesSource)
ALL_SOURCES = SEARCH_SOURCES + (BrasilApiSource,)

MANUAL_POLICY = SourcePolicy(store_days=None, exportable=True, attribution="")
SITE_POLICY = SourcePolicy(store_days=None, exportable=True, attribution="Site oficial da empresa")

POLICIES: dict[str, SourcePolicy] = {cls.id: cls.policy for cls in ALL_SOURCES}
POLICIES["manual"] = MANUAL_POLICY
POLICIES["site"] = SITE_POLICY
TITLES: dict[str, str] = {cls.id: cls.title for cls in ALL_SOURCES}
TITLES.update({"manual": "Manual", "site": "Site oficial"})
# prioridade na consolidação de um campo (maior vence): dado oficial > site da empresa > Google > OSM
PRIORITY = {"manual": 100, "brasilapi": 80, "cnpj_aberto": 70, "site": 60, "google_places": 50, "osm": 40}


def policy(source_id: str) -> SourcePolicy:
    return POLICIES.get(source_id, SourcePolicy(store_days=30, exportable=False))


def build(http, settings, secrets, data_dir) -> dict[str, Source]:
    return {cls.id: cls(http=http, settings=settings, secrets=secrets, data_dir=data_dir) for cls in ALL_SOURCES}


__all__ = ["Source", "Unit", "Estimate", "SourceUnavailable", "OsmSource", "CnpjOpenSource", "GooglePlacesSource",
           "BrasilApiSource", "SEARCH_SOURCES", "ALL_SOURCES", "POLICIES", "TITLES", "PRIORITY", "policy", "build"]
