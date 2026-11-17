"""Tipos do Lead Finder: empresa encontrada, pesquisa, estados do CRM e políticas de uso das fontes.

Só biblioteca padrão (nada de Qt aqui): estes tipos são usados pelos workers, pelo banco e pelos testes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# ------------------------------------------------------------------------------------------ CRM

STATUSES: tuple[tuple[str, str], ...] = (
    ("nao_contatado", "Não contatado"),
    ("contato_iniciado", "Contato iniciado"),
    ("aguardando", "Aguardando resposta"),
    ("interessado", "Interessado"),
    ("demo_enviada", "Demonstração enviada"),
    ("negociacao", "Negociação"),
    ("convertido", "Cliente convertido"),
    ("sem_interesse", "Sem interesse"),
    ("nao_contatar", "Não contatar"),
)
STATUS_LABEL = dict(STATUSES)
STATUS_KEYS = tuple(k for k, _ in STATUSES)
DO_NOT_CONTACT = "nao_contatar"
# posição de cada estado no funil (estados fora do funil ficam com -1)
FUNNEL_RANK = {"nao_contatado": 0, "contato_iniciado": 1, "aguardando": 1, "interessado": 2, "demo_enviada": 3,
               "negociacao": 4, "convertido": 5, "sem_interesse": 1, "nao_contatar": -1}

EVENT_LABEL = {
    "criado": "Empresa encontrada",
    "fonte": "Nova fonte consolidada",
    "status": "Status alterado",
    "nota": "Nota",
    "whatsapp_aberto": "WhatsApp aberto (mensagem ainda não confirmada)",
    "mensagem_enviada": "Mensagem enviada (confirmada por você)",
    "mensagem_copiada": "Mensagem copiada",
    "resposta_recebida": "Resposta recebida",
    "demo_enviada": "Demonstração enviada",
    "lembrete_criado": "Lembrete criado",
    "lembrete_concluido": "Lembrete concluído",
    "site_analisado": "Site analisado",
    "cnpj_consultado": "CNPJ consultado",
    "midia_vinculada": "Projeto do AI Studio vinculado",
    "midia_removida": "Vínculo com o AI Studio removido",
    "nao_contatar": "Pedido de não contato registrado",
    "contato_liberado": "Pedido de não contato removido",
    "pontuacao": "Pontuação recalculada",
    "editado": "Dados editados manualmente",
}

PRIORITY_LABEL = {0: "—", 1: "Baixa", 2: "Média", 3: "Alta"}

WA_UNKNOWN, WA_LIKELY, WA_CONFIRMED, WA_NONE = "nao_verificado", "provavel", "confirmado", "sem_whatsapp"
WA_LABEL = {WA_UNKNOWN: "Não verificado", WA_LIKELY: "Provável (celular)", WA_CONFIRMED: "Confirmado",
            WA_NONE: "Sem WhatsApp"}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# ------------------------------------------------------------------------------------- fontes

@dataclass(frozen=True)
class SourcePolicy:
    """O que a licença/termos de uma fonte permitem fazer com os dados dela.

    ``store_days``: por quantos dias os campos podem ficar guardados no banco local (``None`` = sem limite).
    ``exportable``: se os campos podem sair do programa (CSV). ``attribution``: texto obrigatório na exportação.
    ``permanent_fields``: campos que podem ficar guardados sem limite mesmo quando ``store_days`` existe
    (ex.: o ``place_id`` do Google).
    """

    store_days: int | None = None
    exportable: bool = True
    attribution: str = ""
    permanent_fields: tuple[str, ...] = ()
    terms_url: str = ""


LEAD_FIELDS = ("name", "legal_name", "category", "address", "city", "uf", "cep", "phone", "phone2", "whatsapp",
               "website", "cnpj", "cnae", "situacao", "natureza")


@dataclass
class Lead:
    """Uma empresa vinda de UMA fonte (registro bruto, antes de consolidar com as outras fontes)."""

    source: str                      # id da fonte ("osm", "google_places", "cnpj_aberto", "brasilapi", "manual")
    source_ref: str                  # id estável na fonte (place_id, "node/123", CNPJ...)
    name: str
    legal_name: str = ""
    category: str = ""
    address: str = ""
    city: str = ""
    uf: str = ""
    cep: str = ""
    phone: str = ""
    phone2: str = ""
    whatsapp: str = ""               # só quando a própria fonte informa (ex.: contact:whatsapp no OSM)
    website: str = ""
    cnpj: str = ""
    cnae: str = ""
    situacao: str = ""
    natureza: str = ""
    query: str = ""                  # palavra-chave/local que trouxe o registro
    fetched_at: str = field(default_factory=now_iso)
    extra: dict = field(default_factory=dict)

    def fields(self) -> dict:
        return {k: getattr(self, k) for k in LEAD_FIELDS if getattr(self, k)}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeoScope:
    """Abrangência geográfica: ``pais`` | ``regiao`` | ``uf`` | ``municipios``."""

    kind: str = "uf"
    region: str = ""                                   # sigla da região (N, NE, CO, SE, S)
    ufs: list[str] = field(default_factory=list)
    municipalities: list[tuple[str, str]] = field(default_factory=list)   # (nome, UF)

    def describe(self) -> str:
        from .geo import REGION_NAME, UF_NAME
        if self.kind == "pais":
            return "Brasil inteiro"
        if self.kind == "regiao":
            return f"Região {REGION_NAME.get(self.region, self.region)}"
        if self.kind == "uf":
            return ", ".join(UF_NAME.get(u, u) for u in self.ufs) or "nenhum estado"
        return ", ".join(f"{m} - {u}" for m, u in self.municipalities) or "nenhum município"


@dataclass
class SearchQuery:
    scope: GeoScope
    keywords: list[str]
    sources: list[str]
    include_individuals: bool = False   # CNPJ: incluir empresário individual/MEI (dados mais pessoais)
