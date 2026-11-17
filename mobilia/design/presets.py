"""Funções, estilos e ambientes do redesenho de ambientes por IA (o fluxo principal do 7xAi).

Cada **função** é um preset do mesmo motor (SDXL + ControlNet de profundidade, ``engine.py``): muda a força da
difusão sobre a foto, o peso e a duração do controle de profundidade, o que fica travado (janelas, portas, teto) e
como o prompt é montado. Cada **estilo** traz o vocabulário de materiais, cores e peças em inglês (o idioma em que o
gerador foi treinado) e uma lista curta de paletas: cada variação usa uma, para as N opções não saírem iguais. Cada
**ambiente** traz o conjunto completo de móveis que um decorador poria ali (o "sala com um vaso e dois quadros" é
exatamente o que estes prompts evitam).

Os textos para o usuário (rótulos) ficam em português; os prompts ficam em inglês. O CLIP do SDXL lê só os primeiros
77 tokens de cada prompt: por isso a ordem é *ambiente + móveis -> estilo -> fotografia* e os textos são curtos.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

# ============================================================================================ ambientes
ROOMS: dict[str, dict] = {
    "sala": {
        "label": "Sala de estar",
        "en": "living room",
        "furniture": ("a large sofa against the wall, two armchairs, a coffee table on a large area rug, side tables "
                      "with table lamps, a floor lamp, large framed wall art above the sofa, potted plants in the "
                      "corners, curtains, throw pillows"),
        "empty_negative": "",
    },
    "quarto": {
        "label": "Quarto",
        "en": "bedroom",
        "furniture": ("a double bed with an upholstered headboard against the wall, layered bedding and pillows, two "
                      "nightstands with table lamps, a large area rug under the bed, a bench at the foot of the bed, "
                      "framed art above the bed, curtains, a plant"),
        "empty_negative": "",
    },
    "cozinha": {
        "label": "Cozinha",
        "en": "kitchen",
        "furniture": ("full height kitchen cabinets, stone countertops, a kitchen island with bar stools, pendant "
                      "lights above the island, a backsplash, built-in appliances, open shelves with ceramics, herbs "
                      "in pots"),
        "empty_negative": "",
    },
    "banheiro": {
        "label": "Banheiro",
        "en": "bathroom",
        "furniture": ("a floating vanity with a vessel sink, a large mirror with sconces, a walk-in glass shower, "
                      "wall tiles, folded towels, a plant, bath accessories"),
        "empty_negative": "",
    },
    "escritorio": {
        "label": "Escritório",
        "en": "home office",
        "furniture": ("a large wooden desk, an ergonomic office chair, a bookshelf full of books and decor, a desk "
                      "lamp, an area rug, framed wall art, a lounge armchair, a plant, curtains"),
        "empty_negative": "",
    },
    "sala_jantar": {
        "label": "Sala de jantar",
        "en": "dining room",
        "furniture": ("a dining table with six upholstered chairs, a statement pendant chandelier above the table, a "
                      "sideboard against the wall, a large area rug, framed wall art, a centerpiece vase, curtains"),
        "empty_negative": "",
    },
    "varanda": {
        "label": "Varanda",
        "en": "balcony terrace",
        "furniture": ("an outdoor lounge sofa with cushions, two lounge chairs, a low outdoor coffee table, an outdoor "
                      "rug, large potted plants and planters, warm string lights"),
        "empty_negative": "",
    },
    "fachada": {
        "label": "Fachada",
        "en": "house exterior facade",
        "furniture": ("redesigned facade cladding, new window frames, a front door, exterior wall lights, "
                      "landscaping with plants and shrubs, a clean walkway"),
        "empty_negative": "",
    },
}

# ============================================================================================ estilos
@dataclass(frozen=True)
class Style:
    key: str
    label: str
    prompt: str                       # materiais, cores e peças típicas (inglês)
    palettes: tuple[str, ...]         # uma por variação (gira entre as N opções)
    negative: str = ""                # o que destoa deste estilo
    exterior: str = ""                # vocabulário para fachada (vazio: usa ``prompt``)


STYLES: dict[str, Style] = {s.key: s for s in (
    Style("moderno", "Moderno",
          "modern interior design, clean lines, low profile furniture, matte black metal accents, warm oak wood, "
          "neutral textured fabrics",
          ("warm white and greige palette", "charcoal and walnut palette", "soft beige and black accents"),
          "ornate, vintage, cluttered",
          "modern house, flat roof lines, large glass windows, white stucco and wood cladding"),
    Style("contemporaneo", "Contemporâneo",
          "contemporary interior design, curved sofa, sculptural lighting, natural stone, bouclé and linen, "
          "layered neutral textures",
          ("ivory and taupe palette", "sand and terracotta accents", "cream and soft gray palette"),
          "rustic, dated, cluttered",
          "contemporary house, mixed materials, stone and wood cladding, large windows"),
    Style("escandinavo", "Escandinavo",
          "scandinavian interior design, light oak wood, white walls, light gray fabric sofa, wool rug, rattan and "
          "linen, cozy hygge, simple functional furniture",
          ("white and light oak palette", "soft gray and sage green accents", "warm white and pale blue accents"),
          "dark wood, ornate, heavy drapes, cluttered",
          "scandinavian house, white wood cladding, black window frames, gabled roof"),
    Style("minimalista", "Minimalista",
          "minimalist interior design, few carefully chosen pieces, uncluttered, clean surfaces, hidden storage, "
          "monochrome, soft diffused light",
          ("all white palette", "warm beige monochrome palette", "light gray monochrome palette"),
          "clutter, many decorations, patterns, ornate",
          "minimalist house, smooth white render, clean geometric volumes"),
    Style("industrial", "Industrial",
          "industrial loft interior design, exposed brick wall, black steel and reclaimed wood, leather sofa, "
          "edison bulb pendant lights, concrete textures",
          ("charcoal and cognac leather palette", "rust and black palette", "concrete gray and warm wood palette"),
          "pastel, ornate, floral",
          "industrial style building, dark metal cladding, brick, black steel frame windows"),
    Style("classico", "Clássico",
          "classic elegant interior design, wall moldings and wainscoting, tufted sofa, carved wood furniture, "
          "crystal chandelier, symmetrical arrangement, silk curtains",
          ("cream and gold palette", "navy and ivory palette", "soft sage and cream palette"),
          "industrial, concrete, rustic",
          "classic neoclassical house, columns, cornices, symmetrical facade"),
    Style("luxo", "Luxo",
          "luxury high end interior design, marble surfaces, velvet sofa, brass and gold accents, designer "
          "lighting, plush rug, statement art, five star hotel quality",
          ("ivory, gold and black palette", "emerald green velvet and brass palette", "taupe and champagne palette"),
          "cheap, plastic, cluttered, worn",
          "luxury modern villa, marble and glass, warm architectural lighting"),
    Style("boho", "Boho",
          "bohemian interior design, rattan and wicker furniture, macrame, layered patterned rugs, many green "
          "plants, warm earthy textiles, floor cushions",
          ("terracotta and cream palette", "mustard and olive palette", "warm sand and rust palette"),
          "cold, sterile, chrome",
          "bohemian house, warm stucco, arches, many plants"),
    Style("japandi", "Japandi",
          "japandi interior design, low wooden furniture, light ash wood, paper lantern lighting, linen, "
          "wabi-sabi ceramics, calm natural minimalism",
          ("warm beige and light wood palette", "muted olive and oak palette", "soft charcoal and natural wood"),
          "clutter, bright colors, ornate",
          "japandi house, natural timber slats, low roof, zen garden"),
    Style("rustico", "Rústico",
          "rustic interior design, solid reclaimed wood furniture, exposed wooden beams, natural stone, wool and "
          "leather, warm cozy cabin feel",
          ("warm brown and cream palette", "olive and natural wood palette", "rust and oak palette"),
          "chrome, glossy, ultra modern",
          "rustic country house, natural stone walls, wooden beams, warm lights"),
    Style("costeiro", "Costeiro",
          "coastal interior design, white and sand tones, light blue accents, linen slipcovered sofa, jute rug, "
          "rattan, whitewashed wood, breezy beach house",
          ("white and sea blue palette", "sand and soft aqua palette", "white and natural jute palette"),
          "dark, heavy, industrial",
          "coastal beach house, white siding, blue shutters, light wood deck"),
    Style("tropical", "Tropical brasileiro",
          "brazilian tropical modern interior design, freijo wood, cane and rattan furniture, large tropical plants, "
          "terracotta and green, natural stone, warm daylight",
          ("green and terracotta palette", "freijo wood and cream palette", "olive and caramel palette"),
          "cold, gray, sterile",
          "brazilian tropical modernist house, wood brise soleil, exposed concrete, lush tropical garden"),
    Style("mid_century", "Mid-century",
          "mid-century modern interior design, tapered wooden legs, walnut furniture, iconic lounge chair, "
          "globe pendant lights, retro graphic accents",
          ("mustard and walnut palette", "teal and walnut palette", "burnt orange and cream palette"),
          "ornate, rustic, cluttered",
          "mid-century modern house, low flat roof, large windows, wood and stone"),
    Style("farmhouse", "Fazenda moderna",
          "modern farmhouse interior design, shiplap walls, white oak, black metal fixtures, linen slipcovered sofa, "
          "woven baskets, warm and inviting",
          ("white and black accents palette", "cream and natural oak palette", "soft sage and white palette"),
          "glossy, ultra modern, chrome",
          "modern farmhouse, white board and batten siding, black windows, metal roof"),
)}

# ============================================================================================ eventos
EVENTS: dict[str, dict] = {
    "aniversario": {"label": "Aniversário",
                    "prompt": ("decorated for an elegant birthday party, balloon garland arch, a dessert table with a "
                               "birthday cake, flower arrangements, string lights, festive table setting")},
    "casamento": {"label": "Casamento",
                  "prompt": ("decorated for an elegant wedding reception, white flower arrangements, draped fabric, "
                             "candles, a long banquet table with fine table setting, warm string lights")},
    "natal": {"label": "Natal",
              "prompt": ("decorated for christmas, a tall decorated christmas tree with warm lights, garlands, wrapped "
                         "presents, stockings, candles, cozy festive atmosphere")},
    "festa_infantil": {"label": "Festa infantil",
                       "prompt": ("decorated for a children's party, colorful balloon arch, a themed dessert table "
                                  "with cake, paper decorations, playful festive setup")},
    "cha_de_bebe": {"label": "Chá de bebê",
                    "prompt": ("decorated for a baby shower, pastel balloon garland, a dessert table with cake, soft "
                               "flower arrangements, delicate festive decor")},
    "reveillon": {"label": "Réveillon",
                  "prompt": ("decorated for a new year's eve party, white and gold decor, gold balloons, champagne "
                             "glasses, candles, elegant festive table, warm lights")},
}

# ============================================================================================ funções
@dataclass(frozen=True)
class FunctionPreset:
    """Parâmetros de uma função. ``steps`` é o total do agendador do Lightning (os passos efetivos são
    ``steps × strength``); ``control_end`` é a fração do caminho em que a profundidade deixa de guiar."""

    key: str
    label: str
    strength: float
    control_scale: float
    control_end: float
    guidance: float = 1.8
    steps: int = 8
    # grupos semânticos (``analysis/segmentation.GROUPS``) cujos pixels originais voltam por cima no fim e que o
    # gerador não refaz (arquitetura travada): janelas e portas; no "Mobiliar" também persianas/cortinas e o teto
    lock: tuple[str, ...] = ("window", "door")
    # grupos que o gerador não refaz (latentes da foto repostos a cada passo) mas que NÃO são colados de volta pixel
    # a pixel. Vazio por padrão: manter o teto nos latentes deixou uma faixa borrada na junção teto/parede
    keep: tuple[str, ...] = ()
    needs_mask: bool = False          # "Editar área": máscara do usuário
    needs_photo: bool = True
    furnish: bool = True              # o prompt pede o conjunto completo de móveis do ambiente
    keep_layout: bool = False         # o prompt pede manter a disposição
    exterior: bool = False
    event: bool = False


FUNCTIONS: dict[str, FunctionPreset] = {f.key: f for f in (
    FunctionPreset("mudar_estilo", "Mudar estilo", strength=0.8, control_scale=0.65, control_end=0.8,
                   guidance=1.8, keep_layout=True),
    FunctionPreset("reforma", "Reforma completa", strength=1.0, control_scale=0.45, control_end=0.5,
                   guidance=2.0),
    FunctionPreset("redesign_simples", "Redesign simples", strength=0.45, control_scale=0.8, control_end=1.0,
                   guidance=1.5, keep_layout=True),
    # profundidade fraca e curta: a da sala VAZIA puxa o gerador para um piso vazio (com 0,5/0,5 metade das versões
    # saía com um tapete e uma cadeira); a estrutura fica presa pelas janelas/portas mantidas nos latentes
    FunctionPreset("mobiliar", "Mobiliar ambiente vazio", strength=1.0, control_scale=0.35, control_end=0.35,
                   guidance=2.8, lock=("window", "door", "window_covering")),
    FunctionPreset("editar_area", "Editar área", strength=1.0, control_scale=0.35, control_end=0.6,
                   guidance=1.8, lock=(), needs_mask=True, furnish=False),
    FunctionPreset("fachada", "Fachada", strength=0.72, control_scale=0.7, control_end=0.85, guidance=1.8,
                   lock=(), exterior=True),
    # 0,62 não punha nem a árvore de Natal: a decoração precisa de espaço para aparecer
    FunctionPreset("evento", "Decoração de evento", strength=0.8, control_scale=0.55, control_end=0.6,
                   guidance=1.8, furnish=False, keep_layout=True, event=True),
    FunctionPreset("texto", "Texto para ambiente", strength=1.0, control_scale=0.0, control_end=0.0, guidance=2.0,
                   lock=(), needs_photo=False),
)}

# ============================================================================================ prompts
PHOTO_TAIL = ("professional interior design magazine photograph, natural daylight, realistic materials, "
              "sharp focus, high detail, photorealistic")
EXTERIOR_TAIL = "professional architectural photograph, daylight, realistic materials, sharp focus, photorealistic"

NEGATIVE_BASE = ("cgi, 3d render, cartoon, illustration, painting, plastic, waxy, oversaturated, lowres, blurry, "
                 "noise, distorted perspective, warped walls, deformed furniture, floating furniture, extra window, "
                 "extra door, people, text, watermark, messy, cluttered")
NEGATIVE_EMPTY = "empty room, unfurnished, bare floor, sparse, single vase, lonely object in the middle of the room"


@dataclass(frozen=True)
class DesignPrompt:
    positive: str
    negative: str

    def key(self) -> str:
        import hashlib

        return hashlib.sha1((self.positive + "\n" + self.negative).encode("utf-8")).hexdigest()[:16]


def _norm(key: str, table: dict, kind: str) -> str:
    k = (key or "").strip().lower().replace("-", "_").replace(" ", "_")
    k = {"contemporâneo": "contemporaneo", "clássico": "classico", "rústico": "rustico", "tropical_brasileiro":
         "tropical", "brasileiro": "tropical", "escritório": "escritorio", "sala_de_jantar": "sala_jantar",
         "jantar": "sala_jantar", "aniversário": "aniversario", "chá_de_bebê": "cha_de_bebe",
         "redesign": "redesign_simples", "reforma_completa": "reforma", "editar": "editar_area",
         "mobiliar_ambiente_vazio": "mobiliar", "texto_para_ambiente": "texto", "decoracao_de_evento": "evento",
         "decoração_de_evento": "evento"}.get(k, k)
    if k not in table:
        raise ValueError(f"{kind} inválido: {key!r} (opções: {', '.join(table)})")
    return k


def function(key: str) -> FunctionPreset:
    return FUNCTIONS[_norm(key, FUNCTIONS, "Função")]


def style(key: str) -> Style:
    return STYLES[_norm(key, STYLES, "Estilo")]


def room(key: str) -> dict:
    return ROOMS[_norm(key, ROOMS, "Ambiente")]


def norm_room(key: str) -> str:
    return _norm(key, ROOMS, "Ambiente")


def norm_style(key: str) -> str:
    return _norm(key, STYLES, "Estilo")


def norm_event(key: str) -> str:
    return _norm(key, EVENTS, "Evento")


def build_prompt(funcao: str, estilo: str | None = None, ambiente: str = "sala", variation: int = 0,
                 extra: str = "", evento: str | None = None) -> DesignPrompt:
    """Prompt positivo e negativo para uma variação. ``extra`` é o texto livre do usuário (vai na frente: é o que
    ele mais quer). ``variation`` escolhe a paleta do estilo (gira entre as opções)."""
    f = function(funcao)
    rk = norm_room("fachada" if f.exterior else ambiente)
    r = ROOMS[rk]
    st = STYLES[norm_style(estilo)] if estilo else None
    palette = st.palettes[variation % len(st.palettes)] if st else ""
    extra = " ".join((extra or "").split())
    parts: list[str] = []
    if extra:
        parts.append(extra)
    neg = [NEGATIVE_BASE]
    if f.exterior:
        parts.append(f"photo of a {r['en']}")
        if st:
            parts.append(st.exterior or st.prompt)
        parts += [palette, r["furniture"], EXTERIOR_TAIL]
        neg.append("interior, furniture, indoor")
    elif f.event:
        ev = EVENTS[norm_event(evento or (estilo if estilo in EVENTS else "aniversario"))]
        parts += [f"photo of a {r['en']} {ev['prompt']}", "same room, same walls and windows", PHOTO_TAIL]
    elif f.key == "editar_area":
        if st:
            parts += [st.prompt.split(",")[0], palette]
        parts.append(f"inside a {r['en']}, seamless, matching lighting and perspective, {PHOTO_TAIL}")
        neg.append("seam, visible border, mismatched lighting")
    else:
        adj = st.prompt.split(",")[0].replace(" interior design", "").strip() + " " if st else ""
        lead = {"mobiliar": f"photo of a fully furnished {adj}{r['en']} with {r['furniture']}",
                "reforma": f"photo of a beautifully redesigned {adj}{r['en']} with {r['furniture']}"}.get(
            f.key, f"photo of a {adj}{r['en']}, {r['furniture']}")
        parts.append(lead)
        if st:
            parts += [", ".join(x.strip() for x in st.prompt.split(",")[1:]), palette]
        parts.append(PHOTO_TAIL)
        if f.furnish:
            neg.append(NEGATIVE_EMPTY)
        if f.key == "mobiliar":     # o volume é da sala vazia: nada de arquitetura nova
            neg.append("fireplace, radiator, column, new wall, built-in niche")
    if st and st.negative:
        neg.append(st.negative)
    pos = ", ".join(p.strip(", ") for p in parts if p and p.strip())
    return DesignPrompt(pos, ", ".join(neg))


def with_overrides(preset: FunctionPreset, **kw) -> FunctionPreset:
    """Cópia do preset com os campos dados (``None`` = mantém)."""
    return replace(preset, **{k: v for k, v in kw.items() if v is not None})


# ============================================================================================ qualidade
@dataclass(frozen=True)
class QualityProfile:
    key: str
    long_side: int        # lado maior da imagem gerada (múltiplos de 64)
    steps: int            # passos do agendador (Lightning)
    max_pixels: int = field(default=0)


def quality_profile(quality: str = "final", device: str = "cuda") -> QualityProfile:
    """GPU: ``final`` = 1024 px (resolução nativa do SDXL), 8 passos; ``rapido`` = 896 px, 6 passos. CPU (só
    desenvolvimento e máquinas sem placa de vídeo): 768/6 e 640/5."""
    q = "rapido" if str(quality).lower() in ("rapido", "rápido", "previa", "prévia", "fast") else "final"
    gpu = str(device).startswith("cuda")
    if gpu:
        return QualityProfile(q, 1024, 8) if q == "final" else QualityProfile(q, 896, 6)
    return QualityProfile(q, 768, 6) if q == "final" else QualityProfile(q, 640, 5)


def work_size(width: int, height: int, long_side: int, multiple: int = 64) -> tuple[int, int]:
    """Tamanho de trabalho com a proporção da foto, lado maior ``long_side``, ambos múltiplos de ``multiple`` e
    com área mínima de 0,4 MP para fotos muito alongadas (o SDXL degrada abaixo disso)."""
    s = long_side / max(width, height)
    w, h = width * s, height * s
    min_area = 0.4 * long_side * long_side * 0.75
    if w * h < min_area:
        k = (min_area / (w * h)) ** 0.5
        w, h = w * k, h * k
    w = max(multiple, int(round(w / multiple)) * multiple)
    h = max(multiple, int(round(h / multiple)) * multiple)
    return w, h


def catalog() -> dict:
    """Listas para a interface (chave -> rótulo em português)."""
    return {"funcoes": {k: f.label for k, f in FUNCTIONS.items()},
            "estilos": {k: s.label for k, s in STYLES.items()},
            "ambientes": {k: r["label"] for k, r in ROOMS.items()},
            "eventos": {k: e["label"] for k, e in EVENTS.items()}}


__all__ = ["ROOMS", "STYLES", "EVENTS", "FUNCTIONS", "Style", "FunctionPreset", "DesignPrompt", "QualityProfile",
           "function", "style", "room", "norm_room", "norm_style", "norm_event", "build_prompt", "with_overrides",
           "quality_profile", "work_size", "catalog", "NEGATIVE_BASE", "NEGATIVE_EMPTY"]
