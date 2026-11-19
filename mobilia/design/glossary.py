"""Texto livre do usuário em português -> inglês para o prompt do gerador.

O codificador de texto do SDXL foi treinado em inglês: "mesa de centro de madeira clara com flores" vira ruído
para ele (a "Editar área" devolvia livros e vasos soltos, sem a mesa, porque só "vaso"/"livros" se parecem com
palavras que ele conhece). Aqui um glossário de decoração (móveis, materiais, cores, adjetivos de forma e tamanho)
traduz por frases, da mais longa para a mais curta, sem rede nem modelo. Palavras que não estão no glossário (nomes
próprios, termos já em inglês) passam como estão; os conectivos ("de", "do", "da") somem quando ligam substantivo e
material ("mesa de madeira" -> "mesa madeira" -> "wood table" depois da troca de ordem).

A ordem do português (substantivo + adjetivo) é invertida para a do inglês (adjetivo + substantivo) dentro de cada
trecho entre vírgulas/"com"/"e": "mesa baixa retangular de madeira clara" -> "low rectangular light wood table".
"""
from __future__ import annotations

import re
import unicodedata

# frases (minúsculas, sem acento) -> inglês. "N" = substantivo (vai para o fim do trecho), "A" = adjetivo/material
_PHRASES: dict[str, tuple[str, str]] = {
    # ---- móveis e objetos (substantivos)
    "mesa de centro": ("coffee table", "N"), "mesinha de centro": ("coffee table", "N"),
    "mesa lateral": ("side table", "N"), "mesinha lateral": ("side table", "N"), "mesa de canto": ("side table", "N"),
    "mesa de jantar": ("dining table", "N"), "mesa de cabeceira": ("nightstand", "N"),
    "criado mudo": ("nightstand", "N"), "criado-mudo": ("nightstand", "N"),
    "mesa de escritorio": ("desk", "N"), "escrivaninha": ("desk", "N"), "mesa": ("table", "N"),
    "sofa de canto": ("sectional sofa", "N"), "sofa retratil": ("reclining sofa", "N"), "sofa": ("sofa", "N"),
    "poltrona": ("armchair", "N"), "cadeira de balanco": ("rocking chair", "N"), "cadeira": ("chair", "N"),
    "cadeiras": ("chairs", "N"), "banqueta": ("bar stool", "N"), "banquetas": ("bar stools", "N"),
    "banco": ("bench", "N"), "puff": ("pouf", "N"), "pufe": ("pouf", "N"), "pufes": ("poufs", "N"),
    "cama de casal": ("double bed", "N"), "cama": ("bed", "N"), "cabeceira": ("headboard", "N"),
    "rack": ("tv console", "N"), "painel de tv": ("tv wall panel", "N"), "televisao": ("television", "N"),
    "aparador": ("sideboard", "N"), "buffet": ("sideboard", "N"), "estante": ("bookshelf", "N"),
    "prateleira": ("shelf", "N"), "prateleiras": ("shelves", "N"), "nicho": ("wall niche", "N"),
    "armario": ("cabinet", "N"), "armarios": ("cabinets", "N"), "guarda-roupa": ("wardrobe", "N"),
    "guarda roupa": ("wardrobe", "N"), "comoda": ("dresser", "N"), "bancada": ("countertop", "N"),
    "ilha": ("kitchen island", "N"), "balcao": ("counter", "N"), "pia": ("sink", "N"), "cuba": ("sink", "N"),
    "espelho": ("mirror", "N"), "tapete": ("rug", "N"), "cortina": ("curtain", "N"), "cortinas": ("curtains", "N"),
    "persiana": ("blinds", "N"), "luminaria de piso": ("floor lamp", "N"), "luminaria de mesa": ("table lamp", "N"),
    "luminaria pendente": ("pendant lamp", "N"), "pendente": ("pendant lamp", "N"), "pendentes": ("pendant lamps", "N"),
    "luminaria": ("lamp", "N"), "abajur": ("table lamp", "N"), "lustre": ("chandelier", "N"),
    "arandela": ("wall sconce", "N"), "quadro": ("framed wall art", "N"), "quadros": ("framed wall art", "N"),
    "almofada": ("cushion", "N"), "almofadas": ("cushions", "N"), "manta": ("throw blanket", "N"),
    "vaso de flores": ("vase with flowers", "N"), "vaso": ("vase", "N"), "vasos": ("vases", "N"),
    "flores": ("flowers", "N"), "flor": ("flower", "N"), "planta": ("plant", "N"), "plantas": ("plants", "N"),
    "livro": ("book", "N"), "livros": ("books", "N"), "bandeja": ("tray", "N"), "velas": ("candles", "N"),
    "vela": ("candle", "N"), "cesto": ("basket", "N"), "cesta": ("basket", "N"), "lareira": ("fireplace", "N"),
    "parede": ("wall", "N"), "piso": ("floor", "N"), "chao": ("floor", "N"), "teto": ("ceiling", "N"),
    "revestimento": ("wall cladding", "N"), "papel de parede": ("wallpaper", "N"), "painel": ("panel", "N"),
    "ripado": ("wood slat panel", "N"), "porta": ("door", "N"), "janela": ("window", "N"),
    "box": ("glass shower enclosure", "N"), "chuveiro": ("shower", "N"), "banheira": ("bathtub", "N"),
    "vaso sanitario": ("toilet", "N"), "toalha": ("towel", "N"), "toalhas": ("towels", "N"),
    "fogao": ("stove", "N"), "geladeira": ("refrigerator", "N"), "forno": ("oven", "N"),
    "coifa": ("range hood", "N"), "escultura": ("sculpture", "N"), "objetos decorativos": ("decor objects", "N"),
    "decoracao": ("decor", "N"),
    "mesas de cabeceira": ("nightstands", "N"), "mesas laterais": ("side tables", "N"), "mesas": ("tables", "N"),
    "poltronas": ("armchairs", "N"), "sofas": ("sofas", "N"), "tapetes": ("rugs", "N"), "luz": ("light", "N"),
    "iluminacao": ("lighting", "N"), "manha": ("morning", "A"), "tarde": ("afternoon", "A"),
    "noite": ("evening", "A"), "janelas": ("windows", "N"), "portas": ("doors", "N"),
    "sala de estar": ("living room", "N"), "sala de jantar": ("dining room", "N"), "sala": ("living room", "N"),
    "quarto de casal": ("master bedroom", "N"), "quarto": ("bedroom", "N"), "cozinha": ("kitchen", "N"),
    "banheiro": ("bathroom", "N"), "lavabo": ("powder room", "N"), "varanda": ("balcony", "N"),
    "escritorio": ("home office", "N"), "fachada": ("facade", "N"), "lareira a lenha": ("wood fireplace", "N"),
    # ---- materiais e acabamentos (adjetivos)
    "madeira clara": ("light wood", "A"), "madeira escura": ("dark wood", "A"), "madeira": ("wood", "A"),
    "carvalho": ("oak", "A"), "nogueira": ("walnut", "A"), "freijo": ("freijo wood", "A"),
    "marmore branco": ("white marble", "A"), "marmore": ("marble", "A"), "granito": ("granite", "A"),
    "pedra": ("stone", "A"), "concreto": ("concrete", "A"), "cimento queimado": ("polished concrete", "A"),
    "vidro": ("glass", "A"), "metal": ("metal", "A"), "ferro": ("iron", "A"), "aco": ("steel", "A"),
    "latao": ("brass", "A"), "dourado": ("gold", "A"), "dourada": ("gold", "A"), "cobre": ("copper", "A"),
    "couro": ("leather", "A"), "linho": ("linen", "A"), "veludo": ("velvet", "A"), "tecido": ("fabric", "A"),
    "palha": ("cane", "A"), "rattan": ("rattan", "A"), "vime": ("wicker", "A"), "junco": ("rattan", "A"),
    "ceramica": ("ceramic", "A"), "porcelanato": ("porcelain tile", "A"), "azulejo": ("tile", "A"),
    "tijolo": ("brick", "A"), "tijolinho": ("brick", "A"), "boucle": ("boucle", "A"), "la": ("wool", "A"),
    "laca": ("lacquered", "A"), "laqueado": ("lacquered", "A"), "fosco": ("matte", "A"), "fosca": ("matte", "A"),
    "brilhante": ("glossy", "A"), "estofado": ("upholstered", "A"), "estofada": ("upholstered", "A"),
    # ---- cores
    "branco": ("white", "A"), "branca": ("white", "A"), "brancos": ("white", "A"), "brancas": ("white", "A"),
    "preto": ("black", "A"), "preta": ("black", "A"), "pretos": ("black", "A"), "pretas": ("black", "A"),
    "cinza": ("gray", "A"), "cinza claro": ("light gray", "A"), "cinza escuro": ("dark gray", "A"),
    "bege": ("beige", "A"), "marrom": ("brown", "A"), "caramelo": ("caramel", "A"), "creme": ("cream", "A"),
    "off white": ("off-white", "A"), "off-white": ("off-white", "A"), "verde": ("green", "A"),
    "verde oliva": ("olive green", "A"), "verde musgo": ("moss green", "A"), "azul": ("blue", "A"),
    "azul marinho": ("navy blue", "A"), "vermelho": ("red", "A"), "vermelha": ("red", "A"),
    "amarelo": ("yellow", "A"), "amarela": ("yellow", "A"), "laranja": ("orange", "A"), "rosa": ("pink", "A"),
    "terracota": ("terracotta", "A"), "mostarda": ("mustard", "A"), "nude": ("nude beige", "A"),
    "roxo": ("purple", "A"), "lilas": ("lilac", "A"), "colorido": ("colorful", "A"), "colorida": ("colorful", "A"),
    # ---- forma, tamanho, estilo
    "baixo": ("low", "A"), "baixa": ("low", "A"), "alto": ("tall", "A"), "alta": ("tall", "A"),
    "grande": ("large", "A"), "grandes": ("large", "A"), "pequeno": ("small", "A"), "pequena": ("small", "A"),
    "pequenos": ("small", "A"), "pequenas": ("small", "A"), "redondo": ("round", "A"), "redonda": ("round", "A"),
    "retangular": ("rectangular", "A"), "quadrado": ("square", "A"), "quadrada": ("square", "A"),
    "oval": ("oval", "A"), "curvo": ("curved", "A"), "curva": ("curved", "A"), "moderno": ("modern", "A"),
    "moderna": ("modern", "A"), "classico": ("classic", "A"), "classica": ("classic", "A"),
    "rustico": ("rustic", "A"), "rustica": ("rustic", "A"), "minimalista": ("minimalist", "A"),
    "elegante": ("elegant", "A"), "aconchegante": ("cozy", "A"), "claro": ("light", "A"), "clara": ("light", "A"),
    "escuro": ("dark", "A"), "escura": ("dark", "A"), "suspenso": ("floating", "A"), "suspensa": ("floating", "A"),
    "embutido": ("built-in", "A"), "embutida": ("built-in", "A"), "novo": ("new", "A"), "nova": ("new", "A"),
    "bonito": ("beautiful", "A"), "bonita": ("beautiful", "A"), "natural": ("natural", "A"),
    "naturais": ("natural", "A"), "verdes": ("green", "A"),
}
# ligam trechos (cada trecho tem a própria ordem adjetivo + substantivo)
_JOINERS = {"com": "with", "e": "and", "sobre": "on", "em cima": "on top", "ao lado": "next to",
            "perto": "near", "na frente": "in front", "atras": "behind", "no lugar": "instead", "em": "in",
            "para": "for", "sem": "without"}
_DROP = {"de", "do", "da", "dos", "das", "um", "uma", "uns", "umas", "o", "a", "os", "as", "no", "na", "nos",
         "nas", "bem", "muito", "muita", "estilo", "tipo", "cor", "tom"}
_COUNT = {"dois": "two", "duas": "two", "tres": "three", "quatro": "four", "cinco": "five", "seis": "six"}


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


_AMBIGUOUS = {"a", "e", "o", "box", "rack", "buffet", "rattan", "metal", "oval", "natural", "nude", "puff", "sofa",
              "em", "no", "off", "white", "off-white"}
_ENGLISH = {"the", "a", "an", "with", "and", "of", "on", "in", "for", "wooden", "wood", "table", "chair", "light", "dark",
            "large", "small", "white", "black", "round", "low", "modern", "plant", "plants", "flowers", "rug", "lamp"}


def _is_portuguese(words: list[str], accented: bool = False) -> bool:
    """Mais palavras do glossário português que palavras inglesas comuns (empate sem nenhuma das duas: acento
    decide, ex.: "sofá")."""
    pt = sum(1 for w in words if w not in _AMBIGUOUS and (w in _DROP or w in _JOINERS or w in _COUNT or w in _PHRASES))
    en = sum(1 for w in words if w in _ENGLISH)
    return pt > en or (pt == en == 0 and accented)


def _segment(words: list[str]) -> str:
    """Um trecho sem conectivos: adjetivos/materiais primeiro, depois os substantivos."""
    adj, nouns, other = [], [], []
    i = 0
    while i < len(words):
        for n in (3, 2, 1):
            key = " ".join(words[i:i + n])
            if len(words[i:i + n]) == n and key in _PHRASES:
                en, kind = _PHRASES[key]
                (nouns if kind == "N" else adj).append(en)
                i += n
                break
        else:
            w = words[i]
            if w in _COUNT:
                other.append(_COUNT[w])
            elif w not in _DROP:
                other.append(w)          # termo desconhecido (ou já em inglês): fica
            i += 1
    return " ".join(other + adj + nouns)


def to_english(text: str) -> str:
    """Traduz o texto livre do usuário (português) para o vocabulário do gerador; texto já em inglês volta igual
    (só espaços normalizados)."""
    text = " ".join((text or "").split())
    if not text:
        return ""
    folded = _fold(text)
    words = re.findall(r"[a-z0-9\-]+|[,.;]", folded)
    if not _is_portuguese([w for w in words if w not in ",.;"], accented=folded != text.lower()):
        return text
    out: list[str] = []
    seg: list[str] = []
    i = 0

    def flush():
        if seg:
            s = _segment(seg)
            if s:
                out.append(s)
            seg.clear()

    while i < len(words):
        w = words[i]
        two = " ".join(words[i:i + 2])
        if w in ",.;":
            flush()
            out.append(",")
            i += 1
        elif two in _JOINERS:
            flush()
            out.append(_JOINERS[two])
            i += 2
        elif w in _JOINERS:
            flush()
            out.append(_JOINERS[w])
            i += 1
        else:
            seg.append(w)
            i += 1
    flush()
    s = " ".join(out).replace(" ,", ",")
    s = re.sub(r"(,\s*)+", ", ", s).strip(" ,")
    s = re.sub(r"\b(with|and|on|in|for|next to|near|behind|without)\s+(with|and)\b", r"\1", s)
    return s


__all__ = ["to_english"]
