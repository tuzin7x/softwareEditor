"""Telefones brasileiros: normalização, tipo (fixo/celular/serviço) e link oficial ``wa.me``.

Regras (plano de numeração da Anatel):

* nacional = DDD (2 dígitos, lista oficial) + assinante;
* celular: 9 dígitos começando com 9; fixo: 8 dígitos começando com 2 a 5;
* números de celular antigos com 8 dígitos (começando com 6 a 9) ganharam o nono dígito em todo o país até 2016:
  o 9 é acrescentado e isso fica anotado em ``note``;
* 0800/0300/4004 etc. são números de serviço: não têm WhatsApp de conversa e não geram botão.

Nenhuma regra aqui afirma que um número tem WhatsApp: celular só vira "provável". A confirmação vem da fonte
(ex.: link de WhatsApp no site oficial) ou do usuário.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote

VALID_DDD = frozenset({
    11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 27, 28, 31, 32, 33, 34, 35, 37, 38, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 51, 53, 54, 55, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73, 74, 75, 77, 79, 81, 82, 83, 84, 85, 86, 87,
    88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99,
})

_SERVICE = re.compile(r"^(0?(300|500|800|900)\d{6,8}|[34]00[0-9]\d{4})$")

MOBILE, LANDLINE, SERVICE, INVALID = "celular", "fixo", "servico", "invalido"


@dataclass(frozen=True)
class PhoneInfo:
    raw: str
    kind: str                 # celular | fixo | servico | invalido
    e164: str = ""            # +5511987654321
    ddd: str = ""
    note: str = ""

    @property
    def valid(self) -> bool:
        return self.kind in (MOBILE, LANDLINE)

    @property
    def digits(self) -> str:
        """Número para o ``wa.me`` (só dígitos, com o 55)."""
        return self.e164.lstrip("+")

    @property
    def national(self) -> str:
        if not self.valid:
            return self.raw.strip()
        sub = self.e164[5:]
        mid = 5 if len(sub) == 9 else 4
        return f"({self.ddd}) {sub[:mid]}-{sub[mid:]}"


def normalize(raw: str | None) -> PhoneInfo:
    """Normaliza um telefone brasileiro escrito de qualquer jeito ("(11) 9 8765-4321", "+55 11 ...", "011...")."""
    text = (raw or "").strip()
    d = re.sub(r"\D", "", text)
    if not d:
        return PhoneInfo(text, INVALID)
    if _SERVICE.match(d):
        return PhoneInfo(text, SERVICE, note="número de serviço (0800/4004...)")
    if d.startswith("55") and len(d) in (12, 13):        # código do país
        d = d[2:]
    if d.startswith("0") and len(d) in (11, 12, 13, 14):  # 0 + DDD (+ operadora: 0 XX DDD ...)
        d = d[1:]
        if len(d) in (12, 13) and int(d[2:4]) in VALID_DDD:
            d = d[2:]                                     # remove o código da operadora
    if len(d) not in (10, 11):
        return PhoneInfo(text, INVALID, note="quantidade de dígitos inválida")
    ddd, sub = int(d[:2]), d[2:]
    if ddd not in VALID_DDD:
        return PhoneInfo(text, INVALID, note="DDD inexistente")
    note = ""
    if len(sub) == 9:
        if sub[0] != "9":
            return PhoneInfo(text, INVALID, note="celular deve começar com 9")
        kind = MOBILE
    elif sub[0] in "2345":
        kind = LANDLINE
    elif sub[0] in "6789":
        sub, kind, note = "9" + sub, MOBILE, "nono dígito acrescentado (número antigo de 8 dígitos)"
    else:
        return PhoneInfo(text, INVALID, note="prefixo inválido")
    return PhoneInfo(text, kind, f"+55{ddd}{sub}", str(ddd), note)


def wa_me_url(phone: PhoneInfo | str, text: str = "") -> str:
    """Link oficial de conversa do WhatsApp (``https://wa.me/<número>?text=<mensagem>``).

    Levanta ``ValueError`` para números inválidos ou de serviço. O link só abre a conversa: o envio é sempre
    manual, feito pela pessoa no próprio WhatsApp."""
    info = phone if isinstance(phone, PhoneInfo) else normalize(phone)
    if not info.valid:
        raise ValueError(f"telefone inválido para WhatsApp: {info.note or info.kind}")
    url = f"https://wa.me/{info.digits}"
    if text.strip():
        url += "?text=" + quote(text.strip(), safe="")
    return url


def numbers_in_wa_links(html: str) -> list[str]:
    """Números presentes em links oficiais de WhatsApp (wa.me / api.whatsapp.com) de uma página."""
    found: list[str] = []
    for m in re.finditer(r"(?:wa\.me/|api\.whatsapp\.com/send/?\?(?:[^\"'<>\s]*?&(?:amp;)?)?phone=)\+?(\d{10,13})",
                         html, re.I):
        info = normalize(m.group(1))
        if info.valid and info.e164 not in found:
            found.append(info.e164)
    return found
