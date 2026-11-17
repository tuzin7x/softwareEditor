"""Contrato das fontes de dados. A pesquisa (aqui) é separada do armazenamento (``storage.py``): cada fonte
declara a sua :class:`~..models.SourcePolicy` e o banco aplica prazos de guarda e regras de exportação."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from ..models import Lead, SearchQuery, SourcePolicy


@dataclass
class Unit:
    """Uma consulta à fonte (ex.: uma área no OSM, uma palavra-chave numa cidade no Google)."""

    label: str
    params: dict = field(default_factory=dict)
    cost_units: int = 1        # requisições pagas estimadas (0 = gratuita)


@dataclass
class Estimate:
    units: int
    paid_requests: int
    cost_usd: float
    note: str = ""


class SourceUnavailable(Exception):
    """A fonte não pode ser usada agora (sem chave, sem índice importado...). Mensagem pronta para o usuário."""


class Source:
    id = ""
    title = ""
    description = ""
    online = True
    policy = SourcePolicy()

    def __init__(self, http=None, settings=None, secrets=None, data_dir=None):
        self.http = http
        self.settings = settings
        self.secrets = secrets
        self.data_dir = data_dir

    def availability(self) -> tuple[bool, str]:
        """(pode usar, explicação curta)."""
        return True, ""

    def plan(self, query: SearchQuery) -> list[Unit]:
        raise NotImplementedError

    def estimate(self, query: SearchQuery) -> Estimate:
        units = self.plan(query)
        return Estimate(len(units), 0, 0.0)

    def fetch(self, unit: Unit, query: SearchQuery, ctx) -> Iterator[Lead]:
        raise NotImplementedError
