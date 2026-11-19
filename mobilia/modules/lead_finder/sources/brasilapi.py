"""BrasilAPI — consulta pública e gratuita de um CNPJ (dados abertos da Receita Federal).

Usada para **enriquecer** um lead que já tem CNPJ (não faz busca por cidade). Por minimização (LGPD), o quadro
de sócios (``qsa``), o e-mail e o capital social **não** são guardados.
"""
from __future__ import annotations

import re

from .. import keywords as kw
from ..geo import uf_from_text
from ..models import Lead, SourcePolicy
from ..net import NetError
from .base import Source

URL = "https://brasilapi.com.br/api/cnpj/v1/{cnpj}"


def cnpj_digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def valid_cnpj(cnpj: str) -> bool:
    """Dígitos verificadores do CNPJ (só números)."""
    d = cnpj_digits(cnpj)
    if len(d) != 14 or d == d[0] * 14:
        return False
    def dv(base: str) -> int:
        w = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2][-len(base):]
        r = sum(int(a) * b for a, b in zip(base, w)) % 11
        return 0 if r < 2 else 11 - r
    return dv(d[:12]) == int(d[12]) and dv(d[:13]) == int(d[13])


def format_cnpj(cnpj: str) -> str:
    d = cnpj_digits(cnpj)
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}" if len(d) == 14 else cnpj


def parse(data: dict) -> Lead:
    cnpj = cnpj_digits(str(data.get("cnpj", "")))
    street = " ".join(x for x in (data.get("descricao_tipo_de_logradouro") or "", data.get("logradouro") or "",
                                  data.get("numero") or "") if x).strip()
    address = ", ".join(x for x in (street, data.get("complemento") or "", data.get("bairro") or "") if x)
    cnae = str(data.get("cnae_fiscal") or "")
    phones = [p for p in (data.get("ddd_telefone_1"), data.get("ddd_telefone_2")) if p]
    name = (data.get("nome_fantasia") or "").strip() or (data.get("razao_social") or "").strip()
    return Lead(source="brasilapi", source_ref=cnpj, name=name.title() if name.isupper() else name,
                legal_name=(data.get("razao_social") or "").strip(),
                category=kw.classify(name, cnae=cnae), address=address.title(),
                city=(data.get("municipio") or "").title(), uf=uf_from_text(data.get("uf") or ""),
                cep=cnpj_digits(data.get("cep") or ""), phone=phones[0] if phones else "",
                phone2=phones[1] if len(phones) > 1 else "", cnpj=cnpj, cnae=cnae,
                situacao=(data.get("descricao_situacao_cadastral") or "").upper(),
                natureza=data.get("natureza_juridica") or "", query="consulta de CNPJ",
                extra={"cnae_descricao": data.get("cnae_fiscal_descricao") or "",
                       "inicio_atividade": data.get("data_inicio_atividade") or ""})


class BrasilApiSource(Source):
    id = "brasilapi"
    title = "BrasilAPI (CNPJ)"
    description = "Consulta pública de CNPJ (dados abertos da Receita). Gratuita; só enriquece leads com CNPJ."
    policy = SourcePolicy(store_days=None, exportable=True,
                          attribution="Receita Federal do Brasil — dados abertos, via BrasilAPI",
                          terms_url="https://brasilapi.com.br/")

    def lookup(self, cnpj: str, ctx=None) -> Lead:
        d = cnpj_digits(cnpj)
        if not valid_cnpj(d):
            raise NetError("CNPJ inválido.")
        data = self.http.get_json(URL.format(cnpj=d), ctx=ctx)
        if not isinstance(data, dict) or not data.get("cnpj"):
            raise NetError("CNPJ não encontrado.")
        return parse(data)
