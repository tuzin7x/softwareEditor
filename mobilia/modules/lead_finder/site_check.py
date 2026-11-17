"""Análise do site oficial de uma empresa, para qualificar a oportunidade com evidências verificáveis.

Limites (por educação e por lei):

* só o **endereço principal** do site informado pela fonte (uma página), iniciada por ação do usuário;
* respeita o ``robots.txt`` do site (se ele proibir, a página não é lida);
* nada de login, formulários, captchas ou páginas de portais de terceiros;
* guarda **só** as contagens de termos, os números de WhatsApp publicados em links oficiais e o endereço lido —
  não guarda o conteúdo da página.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from . import dedupe
from .geo import fold
from .models import now_iso
from .net import USER_AGENT, NetError
from .phone import normalize, numbers_in_wa_links

TERMS = {
    "listings": ("imoveis a venda", "comprar", "alugar", "aluguel", "venda", "apartamento", "casa", "terreno",
                 "sala comercial", "codigo do imovel", "ref ", "dormitorio", "quartos", "suite", "m2", "m²"),
    "empty": ("imovel vazio", "imovel desocupado", "desocupado", "sem mobilia", "nao mobiliado", "vazio",
              "pronto para morar", "entrega imediata", "planta"),
    "launch": ("lancamento", "lancamentos", "em construcao", "breve lancamento", "na planta", "empreendimento",
               "decorado"),
    "staging": ("decorado virtual", "home staging", "ambientacao virtual", "tour virtual", "tour 360",
                "decoracao virtual", "render"),
}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1
        if tag == "a":
            href = dict(attrs).get("href") or ""
            if href:
                self.links.append(href)

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _count(text: str, terms: tuple[str, ...]) -> dict[str, int]:
    out = {}
    for t in terms:
        n = len(re.findall(r"(?<!\w)" + re.escape(fold(t)) + r"(?!\w)", text)) if t != "m²" else text.count("m²")
        if n:
            out[t] = n
    return out


def analyze_html(html: str, url: str, lead_phones: list[str]) -> dict:
    p = _Text()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001 - HTML quebrado: usa o que deu para ler
        pass
    text = fold(" ".join(p.parts))
    found = {k: _count(text, v) for k, v in TERMS.items()}
    wa = numbers_in_wa_links(" ".join(p.links) + " " + html[:400000])
    match = next((n for n in wa if n in set(lead_phones)), "")
    total = {k: sum(v.values()) for k, v in found.items()}
    summary = (f"{total['listings']} menções a anúncios, {total['empty']} a imóveis vazios/prontos, "
               f"{total['launch']} a lançamentos" + (f", {len(wa)} link(s) de WhatsApp" if wa else ""))
    return {"url": url, "checked_at": now_iso(), "ok": True, "terms": found, "totals": total,
            "whatsapp_numbers": wa, "whatsapp_match": match, "summary": summary}


def check_site(http, website: str, lead_phones: list[str], ctx=None) -> dict:
    """Lê a página principal do site (se o robots.txt permitir) e devolve as evidências encontradas."""
    domain = dedupe.domain_of(website)
    if not domain or dedupe.is_generic(domain):
        return {"url": website, "checked_at": now_iso(), "ok": False,
                "summary": "Sem site próprio (rede social ou portal não são analisados)."}
    parts = urlsplit(website if "://" in website else "https://" + website)
    base = f"{parts.scheme or 'https'}://{parts.netloc}"
    url = base + (parts.path or "/")
    rp = RobotFileParser()
    try:
        r = http.request("GET", urljoin(base, "/robots.txt"), ctx=ctx, timeout=15, ok=(200, 401, 403, 404, 410))
        if r.status in (401, 403):
            rp.disallow_all = True
        elif r.status == 200:
            rp.parse(r.text(200_000).splitlines())
        else:
            rp.allow_all = True
    except NetError as e:
        return {"url": url, "checked_at": now_iso(), "ok": False, "summary": f"Site inacessível: {e}"}
    if not rp.can_fetch(USER_AGENT, url):
        return {"url": url, "checked_at": now_iso(), "ok": False,
                "summary": "O robots.txt do site não permite a leitura automática; nada foi lido."}
    try:
        page = http.request("GET", url, ctx=ctx, timeout=20, headers={"Accept": "text/html"})
    except NetError as e:
        return {"url": url, "checked_at": now_iso(), "ok": False, "summary": f"Site inacessível: {e}"}
    if "html" not in page.headers.get("content-type", "text/html"):
        return {"url": url, "checked_at": now_iso(), "ok": False, "summary": "A página não é HTML."}
    phones = [normalize(p).e164 for p in lead_phones]
    return analyze_html(page.text(1_500_000), url, [p for p in phones if p])
