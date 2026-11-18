"""Mensagens comerciais: modelos determinísticos (padrão) e geração opcional por IA, sempre revisada pela pessoa.

Modelos usam marcadores ``{empresa}``, ``{cidade}``, ``{meu_nome}`` e ``{minha_empresa}``. Um trecho entre
``[[ ]]`` some quando algum marcador dele está vazio (ex.: ``[[ em {cidade}]]``). Nada sobre a empresa é
inventado: só entram dados que estão no cadastro.

IA (opcional): API de mensagens da Anthropic, chamada direto por HTTPS (``urllib``) porque o executável da
interface não pode ganhar dependências novas. A chave é do usuário (variável ``ANTHROPIC_API_KEY`` ou
configurações), nunca vai no executável. Há limite diário de chamadas.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .net import NetError

OPTOUT = "Se preferir não receber contato, é só me avisar."

BUILTIN_TEMPLATES: tuple[dict, ...] = (
    {"name": "Apresentação · profissional", "tone": "profissional", "body":
        "Olá! Tudo bem? Sou[[ {meu_nome},]] da {minha_empresa}. Desenvolvemos uma ferramenta de IA para "
        "transformar fotos de imóveis em apresentações visuais e vídeos profissionais. Gostaria de mostrar uma "
        "demonstração para a {empresa}. Posso enviar?"},
    {"name": "Apresentação · direto", "tone": "direto", "body":
        "Oi, pessoal da {empresa}! [[Aqui é {meu_nome}. ]]Somos da {minha_empresa}: nossa IA transforma fotos de "
        "imóveis vazios em ambientes decorados e vídeos antes e depois em segundos. Quer ver um exemplo com um "
        "imóvel de vocês?"},
    {"name": "Acompanhamento · profissional", "tone": "profissional", "body":
        "Olá, equipe da {empresa}! [[Aqui é {meu_nome}, da {minha_empresa}. ]]Retomo o contato sobre a "
        "demonstração da nossa ferramenta de imagens e vídeos de imóveis com IA. Ficou alguma dúvida? Posso "
        "preparar um exemplo com um imóvel de vocês."},
    {"name": "Acompanhamento · direto", "tone": "direto", "body":
        "Oi! Passando para saber se conseguiram ver a demonstração da {minha_empresa}. Posso montar um exemplo "
        "com um imóvel da {empresa}?"},
    {"name": "Demonstração enviada", "tone": "profissional", "body":
        "Olá! Conforme combinamos, segue a demonstração da {minha_empresa} para a {empresa}: fotos do imóvel "
        "transformadas em ambientes decorados e vídeo antes e depois. Fico à disposição para conversar."},
)
TONES = (("profissional", "Profissional"), ("direto", "Direto"))
_PLACEHOLDER = re.compile(r"\{(\w+)\}")
KNOWN = ("empresa", "cidade", "meu_nome", "minha_empresa")


@dataclass
class Context:
    empresa: str = ""
    cidade: str = ""
    meu_nome: str = ""
    minha_empresa: str = "7x Intelligence"


def placeholders(body: str) -> list[str]:
    return _PLACEHOLDER.findall(body or "")


def validate(body: str) -> None:
    unknown = [p for p in placeholders(body) if p not in KNOWN]
    if unknown:
        raise ValueError("Marcadores desconhecidos: " + ", ".join("{" + p + "}" for p in unknown)
                         + ". Use {empresa}, {cidade}, {meu_nome} ou {minha_empresa}.")
    if body.count("[[") != body.count("]]"):
        raise ValueError("Trechos opcionais precisam abrir com [[ e fechar com ]].")


def render(body: str, ctx: Context, optout: bool = False) -> str:
    """Preenche o modelo de forma determinística (mesma entrada, mesmo texto)."""
    validate(body)
    values = {k: (getattr(ctx, k) or "").strip() for k in KNOWN}
    if not values["empresa"]:
        values["empresa"] = "sua imobiliária"

    def optional(m: re.Match) -> str:
        seg = m.group(1)
        return "" if any(not values.get(p) for p in placeholders(seg)) else seg
    text = re.sub(r"\[\[(.*?)\]\]", optional, body, flags=re.S)
    text = _PLACEHOLDER.sub(lambda m: values.get(m.group(1), ""), text)
    text = re.sub(r"[ \t]+", " ", text).replace(" ,", ",").replace(" .", ".").strip()
    if optout and OPTOUT not in text:
        text = f"{text}\n\n{OPTOUT}"
    return text


def template_for(name: str, custom: list[dict]) -> dict | None:
    for t in list(custom) + list(BUILTIN_TEMPLATES):
        if t["name"] == name:
            return t
    return None


# ---------------------------------------------------------------------------------------- IA (opcional)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
SYSTEM_PROMPT = (
    "Você escreve mensagens curtas de primeiro contato comercial pelo WhatsApp, em português do Brasil, para "
    "apresentar o 7xAi (software da 7x Intelligence que transforma fotos de imóveis em ambientes decorados e "
    "vídeos com inteligência artificial) a empresas do setor imobiliário.\n"
    "Regras: use somente os fatos fornecidos sobre a empresa; não invente características, números, prêmios, "
    "imóveis, nomes de pessoas nem resultados; não prometa preços nem prazos; no máximo 450 caracteres; sem "
    "hashtags; no máximo um emoji; termine com uma pergunta simples. Responda só com o texto da mensagem.")


def ai_prompt(ctx: Context, tone: str, base_text: str, category: str = "") -> str:
    facts = [f"Nome da empresa: {ctx.empresa or 'não informado'}"]
    if ctx.cidade:
        facts.append(f"Cidade: {ctx.cidade}")
    if category:
        facts.append(f"Segmento (pelo cadastro): {category}")
    if ctx.meu_nome:
        facts.append(f"Quem escreve: {ctx.meu_nome}, da {ctx.minha_empresa}")
    tone_txt = "direto e próximo" if tone == "direto" else "profissional e cordial"
    return ("Fatos conhecidos (use apenas estes):\n- " + "\n- ".join(facts)
            + f"\n\nTom: {tone_txt}.\nTexto de referência (pode reescrever):\n{base_text}")


def generate_with_ai(http, api_key: str, model: str, ctx: Context, tone: str, base_text: str, category: str = "",
                     cancel=None) -> str:
    """Pede uma variação da mensagem à API da Anthropic. O resultado é só um rascunho para revisão."""
    if not api_key:
        raise NetError("Configure uma chave de API da Anthropic para usar a geração por IA.")
    body = {
        "model": model or "claude-opus-5-5",
        "max_tokens": 16000,
        "output_config": {"effort": "low"},
        "fallbacks": "default",
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": ai_prompt(ctx, tone, base_text, category)}],
    }
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
               "anthropic-beta": "server-side-fallback-2026-07-01"}
    data = http.post_json(ANTHROPIC_URL, body, ctx=cancel, headers=headers, timeout=120)
    if not isinstance(data, dict):
        raise NetError("Resposta inesperada da API de IA.")
    if data.get("stop_reason") == "refusal":
        raise NetError("A IA recusou este pedido. Use um modelo determinístico.")
    text = "".join(b.get("text", "") for b in data.get("content", []) if isinstance(b, dict) and b.get("type") == "text")
    text = text.strip().strip('"').strip()
    if not text:
        raise NetError("A IA não devolveu texto.")
    return text[:1200]
