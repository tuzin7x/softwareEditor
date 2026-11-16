"""Narrativa de designer (PT12, item 5): as escolhas do layout explicadas em português.

Gera as frases de ``Layout.notes``, exibidas na tela de conclusão da interface. Cada frase é
derivada só do que o layout de fato contém (medidas finais das peças, paredes, apoios) e do que a
análise mediu na foto (tons, piso, pé-direito, área livre visível): nada é inventado. Exemplo::

    Sala clara, carpete claro, paredes de tom médio em tons quentes, pé-direito 2,45 m,
    11 m² de piso livre visível.
    Estilo moderno, com destaque em terracota, madeira nogueira e metal preto.
    Escolhi um sofá de tecido claro de 1,57 m na parede P0, de frente para a câmera.
    ...

Determinística (mesmo layout -> mesmas frases) e barata (sem render nem projeção de imagem).
"""
from __future__ import annotations

import math

import numpy as np

from . import catalog
from .model import Layout, PlacedItem
from .style import STYLE_LABELS, StyleProfile, is_bright_room

ROLE_LABELS = {
    "sofa": "sofá", "armchair": "poltrona", "coffee_table": "mesa de centro", "rug": "tapete",
    "side_table": "mesa lateral", "floor_lamp": "luminária de piso", "plant_large": "planta grande",
    "plant_medium": "planta média", "plant_small": "planta pequena", "wall_art": "quadro",
    "tv_console": "rack/aparador", "pillows": "almofadas", "vase": "vaso", "bowl": "fruteira",
    "books": "livros", "bed": "cama", "nightstand": "criado-mudo", "table_lamp": "abajur",
    "dresser": "cômoda", "throw": "manta", "dining_table": "mesa de jantar", "dining_chair": "cadeira",
    "shelf": "estante", "desk": "escrivaninha", "desk_chair": "cadeira de trabalho", "bar_stool": "banqueta",
    "kitchen_decor": "utensílio decorativo", "basket": "cesto", "wall_mirror": "espelho",
}
# plural (para "2 poltronas", "4 cadeiras")
_PLURAL = {"sofá": "sofás", "mesa de centro": "mesas de centro", "mesa lateral": "mesas laterais",
           "luminária de piso": "luminárias de piso", "planta grande": "plantas grandes",
           "planta média": "plantas médias", "planta pequena": "plantas pequenas", "rack/aparador": "racks/aparadores",
           "criado-mudo": "criados-mudos", "mesa de jantar": "mesas de jantar",
           "cadeira de trabalho": "cadeiras de trabalho", "utensílio decorativo": "utensílios decorativos"}
# gênero gramatical do rótulo (para "encostado/encostada", "um/uma")
_FEM = {"poltrona", "mesa de centro", "mesa lateral", "luminária de piso", "planta grande", "planta média",
        "planta pequena", "almofadas", "fruteira", "cama", "cama de casal", "cama de solteiro", "cômoda", "manta", "mesa de jantar", "cadeira",
        "estante", "escrivaninha", "cadeira de trabalho", "banqueta"}

ROOM_LABELS = {"sala": ("Sala", True), "quarto": ("Quarto", False), "sala_jantar": ("Sala de jantar", True),
               "escritorio": ("Escritório", False), "cozinha": ("Cozinha", True), "banheiro": ("Banheiro", False)}
ANCHOR_ROLES = ("sofa", "bed", "dining_table", "desk")
# peças citadas uma a uma (as demais entram no "completam")
_FEATURED = set(ANCHOR_ROLES) | {"rug", "coffee_table", "armchair", "wall_art", "dining_chair"}

# tecido CC0 do tapete (``catalog.RUG_TEXTURES``) -> como um designer o descreveria
RUG_FABRICS = {"poly_wool_herringbone": "lã espinha de peixe", "curly_teddy_natural": "bouclé natural",
               "hessian_230": "juta", "knitted_fleece": "tricô de lã"}
RUG_DESIGN_LABELS = {"filete": "com filete na borda", "borda": "com borda larga", "borda_dupla": "com borda dupla",
                     "liso": "liso"}
# piso na narrativa (mais curto que ``style.FLOOR_LABELS``)
FLOOR_NAMES = {"madeira": "piso de madeira", "carpete": "carpete", "piso_frio": "piso frio", "cimento": "cimento queimado"}
# móveis sob os quais um quadro "fica centrado" (plantas, luminárias e cadeiras não contam)
BELOW_ART_ROLES = {"sofa", "bed", "tv_console", "desk", "dresser", "desk_storage", "shelf", "armchair",
                   "side_table", "nightstand", "ottoman", "dining_table"}
ART_STYLE_LABELS = {"formas": "formas orgânicas", "campos": "campos de cor", "arcos": "arcos e disco"}


# ----------------------------------------------------------------------------- formatação

def fmt_num(v: float, nd: int = 2) -> str:
    """Número em português: vírgula decimal, sem zeros à direita (1.50 -> "1,5"; 2.0 -> "2")."""
    s = f"{float(v):.{nd}f}".rstrip("0").rstrip(".")
    return (s if s not in ("-0", "") else "0").replace(".", ",")


def fmt_m(v: float) -> str:
    return f"{fmt_num(v)} m"


def fmt_dims(w: float, h: float) -> str:
    """Largura × altura (ou comprimento): em cm se as duas forem menores que 1 m."""
    if max(w, h) < 1.0:
        return f"{int(round(w * 100))} × {int(round(h * 100))} cm"
    return f"{fmt_num(w)} × {fmt_num(h)} m"


def label(role: str) -> str:
    return ROLE_LABELS.get(role, role.replace("_", " "))


def _art(lbl: str) -> str:
    return "uma" if lbl in _FEM else "um"


def _count(lbl: str, n: int) -> str:
    if n == 1:
        return lbl
    return f"{n} {_PLURAL.get(lbl, lbl + 's')}"


def join_pt(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " e " + parts[-1]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


# ----------------------------------------------------------------------------- geometria simples

def _axes(it: PlacedItem) -> tuple[np.ndarray, np.ndarray]:
    """(eixo lateral u, frente f) do item no piso."""
    y = math.radians(it.yaw_deg)
    return np.array([math.cos(y), math.sin(y)]), np.array([math.sin(y), -math.cos(y)])


def _local(it: PlacedItem, xy) -> tuple[float, float]:
    """Coordenadas (lateral, frente) de um ponto do piso no referencial do item."""
    u, f = _axes(it)
    d = np.asarray(xy, float) - np.asarray(it.position[:2], float)
    return float(d @ u), float(d @ f)


def _inside(it: PlacedItem, xy, tol: float = 0.0) -> bool:
    lx, ly = _local(it, xy)
    return abs(lx) <= it.size[0] / 2 + tol and abs(ly) <= it.size[1] / 2 + tol


def front_feet_on_rug(seat: PlacedItem, rug: PlacedItem) -> bool:
    """Os pés dianteiros do assento (8 cm atrás da borda da frente, nas pontas e no meio) estão no tapete."""
    u, f = _axes(seat)
    c = np.asarray(seat.position[:2], float)
    return all(_inside(rug, c + u * lx * seat.size[0] / 2 + f * (seat.size[1] / 2 - 0.08))
               for lx in (-0.85, 0.0, 0.85))


def front_gap(anchor: PlacedItem, other: PlacedItem) -> float:
    """Vão (m) entre a frente do ``anchor`` e o ponto mais próximo da base de ``other``."""
    _, f = _axes(anchor)
    poly = np.asarray(other.footprint(), float) - np.asarray(anchor.position[:2], float)
    return float(np.min(poly @ f)) - anchor.size[1] / 2


def item_below_art(art: PlacedItem, items: list[PlacedItem]) -> PlacedItem | None:
    """Móvel de piso logo abaixo do quadro (centro do quadro sobre a largura do móvel, junto à parede)."""
    best, bd = None, 1e9
    for it in items:
        if it.mount != "floor" or it.role not in BELOW_ART_ROLES or it.size[2] >= art.position[2]:
            continue
        lx, ly = _local(it, art.position[:2])
        if abs(lx) <= it.size[0] / 2 and -(it.size[1] / 2 + 0.35) <= ly <= it.size[1] / 2:
            d = abs(lx)
            if d < bd:
                best, bd = it, d
    return best


# ----------------------------------------------------------------------------- descrições

def upholstery_desc(it: PlacedItem) -> str:
    """Material/tom da peça pelo catálogo (tags e tom medidos nas texturas)."""
    try:
        spec = catalog.get(it.asset_id)
    except KeyError:
        return ""
    fem = label(it.role) in _FEM
    if "couro" in spec.tags:
        return "de couro escuro" if spec.tone == "escuro" else "de couro"
    if it.role in ("sofa", "armchair", "ottoman"):
        return {"claro": "de tecido claro", "escuro": "de tecido escuro", "madeira": "de madeira e tecido",
                "colorido": "estampad" + ("a" if fem else "o")}.get(spec.tone, "")
    return {"madeira": "de madeira", "claro": "clar" + ("a" if fem else "o"),
            "escuro": "escur" + ("a" if fem else "o")}.get(spec.tone, "")


def room_sentence(room_type: str, profile: StyleProfile, ceiling: float, ceiling_observed: bool,
                  free_area: float) -> str:
    name, fem = ROOM_LABELS.get(room_type, (_cap(room_type.replace("_", " ")), True))
    if is_bright_room(profile):
        tom = "clara" if fem else "claro"
    elif profile.wall_tone == "escuro":
        tom = "escura" if fem else "escuro"
    else:
        tom = "de luz média"
    piso_tom = {"claro": "claro", "medio": "de tom médio", "escuro": "escuro"}.get(profile.floor_tone, "")
    parede = {"claro": "claras", "medio": "de tom médio", "escuro": "escuras"}.get(profile.wall_tone, "")
    temp = "quentes" if profile.warmth > 0.25 else ("frios" if profile.warmth < -0.15 else "neutros")
    pd = f"pé-direito {fmt_m(ceiling)}" if ceiling_observed else f"pé-direito estimado em {fmt_m(ceiling)}"
    area = f"{fmt_num(free_area, 0 if free_area >= 10 else 1)} m² de piso livre visível"
    piso = FLOOR_NAMES.get(profile.floor_material, profile.floor_material)
    return f"{name} {tom}, {piso} {piso_tom}, paredes {parede} em tons {temp}, {pd}, {area}."


def style_sentence(profile: StyleProfile) -> str:
    P = profile.palette
    extra = [f"destaque em {P['accent_name']}"]
    if P.get("wood_name"):
        extra.append(f"madeira {P['wood_name']}")
    if P.get("metal_name"):
        extra.append(f"metal {P['metal_name']}")
    return f"Estilo {STYLE_LABELS.get(profile.style, profile.style)}, com {join_pt(extra)}."


def anchor_sentence(anc: PlacedItem, ctx) -> str:
    lbl = label(anc.role)
    fem = lbl in _FEM
    desc, extra = upholstery_desc(anc), ""
    dim = fmt_m(anc.size[0])
    if anc.role == "bed":
        lbl = "cama de casal" if anc.size[0] >= 1.2 else "cama de solteiro"
        desc, extra = "", ", com roupa de cama de linho"
        dim = fmt_dims(anc.size[0], anc.size[1])
    elif "round" in anc.asset_id.lower():
        desc = ("redond" + ("a" if fem else "o") + " " + desc).strip()
        dim = f"{fmt_m(anc.size[0])} de diâmetro"
    if anc.wall_id is not None:
        onde = f"encostad{'a' if fem else 'o'} na parede P{anc.wall_id}"
    else:
        onde = f"solt{'a' if fem else 'o'} no ambiente"
    f = float(np.asarray(anc.front) @ ctx.dir_to_camera(np.asarray(anc.position[:2])))
    vista = "de frente para a câmera" if f > 0.5 else ("de perfil para a câmera" if f > -0.3 else "de costas para a câmera")
    nome = f"{lbl} {desc}".strip()
    if anc.role == "dining_table":          # mesa não tem frente: a vista não diz nada
        return f"Escolhi {_art(lbl)} {nome} de {dim}{extra}, {onde}."
    return f"Escolhi {_art(lbl)} {nome} de {dim}{extra}, {onde}, {vista}."


def rug_sentence(rug: PlacedItem, items: list[PlacedItem], anc: PlacedItem | None) -> str:
    p = rug.params or {}
    fab = RUG_FABRICS.get(str(p.get("texture_id", "")), "")
    des = RUG_DESIGN_LABELS.get(str(p.get("design", "")), "")
    w, d = max(rug.size[0], rug.size[1]), min(rug.size[0], rug.size[1])
    s = f"Tapete de {fmt_dims(w, d)}"
    if fab:
        s += f" em {fab}"
    if des:
        s += f" {des}"
    seats = [it for it in items if it.role in ("sofa", "armchair") and it.mount == "floor"]
    on = [it for it in seats if front_feet_on_rug(it, rug)]
    if anc is not None and anc.role == "bed" and _inside(rug, anc.position[:2], tol=rug.size[0]):
        s += ", sob a metade de baixo da cama"
    elif seats and len(on) == len(seats):
        s += ", com os pés dianteiros de todos os assentos sobre ele" if len(seats) > 1 \
            else f", com os pés dianteiros do {label(seats[0].role)} sobre ele"
    elif on:
        s += f", com os pés dianteiros do {label(on[0].role)} sobre ele"
    tables = [it for it in items if it.role in ("coffee_table", "dining_table") and _inside(rug, it.position[:2])]
    if tables:
        s += f"; {label(tables[0].role)} inteira sobre o tapete" if all(
            _inside(rug, q) for q in tables[0].footprint()) else ""
    return s + "."


def coffee_sentence(ct: PlacedItem, anc: PlacedItem | None) -> str:
    s = f"Mesa de centro de {fmt_m(max(ct.size[0], ct.size[1]))}"
    if anc is not None and anc.role == "sofa":
        ratio = max(ct.size[0], ct.size[1]) / max(anc.size[0], 1e-6)
        gap = front_gap(anc, ct)
        s += f" ({int(round(ratio * 100))}% da largura do sofá)"
        if 0.0 < gap < 1.5:
            s += f" a {int(round(gap * 100))} cm da frente do sofá"
    return s + "."


def art_sentence(arts: list[PlacedItem], items: list[PlacedItem]) -> str:
    a0 = arts[0]
    p = a0.params or {}
    zc = a0.position[2] + a0.size[2] / 2
    if len(arts) == 2 and abs(arts[0].size[0] - arts[1].size[0]) < 1e-3:
        s = f"Par de quadros de {fmt_dims(a0.size[0], a0.size[2])}"
    elif len(arts) > 1:
        s = f"{len(arts)} quadros"
    else:
        s = f"Quadro de {fmt_dims(a0.size[0], a0.size[2])}"
    if a0.asset_id == "proc:canvas_art":
        est = ART_STYLE_LABELS.get(str(p.get("art_style", "")), "")
        mold = "em tela" if p.get("frame") == "tela" else "com moldura fina"
        s += f", pintura{'s' if len(arts) > 1 else ''} abstrata{'s' if len(arts) > 1 else ''}"
        s += f" ({est}) {mold}" if est else f" {mold}"
    elif a0.asset_id.startswith("ph:"):
        s += ", com moldura clássica"
    below = item_below_art(a0, items)
    xs = [a.position for a in arts]
    if len(arts) > 1:
        mid = np.mean(np.asarray(xs, float), axis=0)
        cen = PlacedItem(a0.id, a0.role, a0.asset_id, list(mid), a0.yaw_deg, 1.0, list(a0.size), "wall")
        below = item_below_art(cen, items) or below
    if below is not None:
        s += f", centrad{'os' if len(arts) > 1 else 'o'} sobre {'a' if label(below.role) in _FEM else 'o'} " \
             f"{label(below.role)}"
    elif a0.wall_id is not None:
        s += f" na parede P{a0.wall_id}"
    return s + f", com o centro a {fmt_m(zc)} do piso."


def decor_sentence(items: list[PlacedItem]) -> str:
    """Decoração por apoio; apoios do mesmo papel com a mesma decoração viram "sobre cada"."""
    by_parent: dict[str, list[PlacedItem]] = {}
    for it in items:
        if it.mount == "on_top" and it.parent:
            by_parent.setdefault(it.parent, []).append(it)
    ids = {it.id: it for it in items}
    groups: dict[tuple, list[str]] = {}        # (papel do apoio, conteúdo) -> ids dos apoios
    for pid in sorted(by_parent, key=lambda k: (ids[k].order if k in ids else 0, k)):
        par = ids.get(pid)
        if par is None:
            continue
        roles: dict[str, int] = {}
        for it in by_parent[pid]:
            r = "pilha de livros" if it.asset_id == "proc:book_stack" else label(it.role)
            roles[r] = roles.get(r, 0) + 1
        groups.setdefault((par.role, tuple(roles.items())), []).append(pid)
    parts = []
    for (prole, content), pids in groups.items():
        coisas = join_pt([_count(r, n) for r, n in content])
        fem = label(prole) in _FEM
        if prole in ("coffee_table", "side_table", "nightstand", "dining_table", "desk", "tv_console",
                     "dresser", "shelf"):
            prep = "sobre cada" if len(pids) > 1 else ("sobre a" if fem else "sobre o")
        else:
            prep = "em cada" if len(pids) > 1 else ("na" if fem else "no")
        parts.append(f"{coisas} {prep} {label(prole)}")
    return (_cap(join_pt(parts)) + ".") if parts else ""


def others_sentence(items: list[PlacedItem]) -> str:
    counts: dict[str, int] = {}
    for it in items:
        if it.mount == "on_top" or it.role in _FEATURED:
            continue
        lbl = label(it.role)
        counts[lbl] = counts.get(lbl, 0) + 1
    if not counts:
        return ""
    return "Completam o ambiente: " + join_pt([_count(k, n) for k, n in counts.items()]) + "."


def seats_sentence(items: list[PlacedItem], anc: PlacedItem | None) -> str:
    arm = [it for it in items if it.role == "armchair"]
    chairs = [it for it in items if it.role == "dining_chair"]
    out = []
    if arm:
        d = upholstery_desc(arm[0])
        out.append(f"{_cap(_count('poltrona', len(arm)))} {d}".strip()
                   + (" voltada" + ("s" if len(arm) > 1 else "") + " para o sofá" if anc is not None
                      and anc.role == "sofa" and all(
                          float(np.asarray(a.front) @ (np.asarray(anc.position[:2]) - np.asarray(a.position[:2]))) > 0
                          for a in arm) else ""))
    if chairs:
        out.append(f"{_cap(_count('cadeira', len(chairs)))} em volta da mesa")
    return ("; ".join(out) + ".") if out else ""


def designer_notes(layout: Layout, ctx, profile: StyleProfile, missing: list[str] | None = None,
                   warnings: list[str] | None = None) -> list[str]:
    """Frases da narrativa (na ordem em que um designer explicaria o projeto)."""
    items = list(layout.items)
    room = ctx.room
    notes = [room_sentence(layout.room_type, profile, float(ctx.ceiling), bool(getattr(room, "ceiling_observed", True)),
                           float(ctx.free_area)),
             style_sentence(profile)]
    anc = next((it for r in ANCHOR_ROLES for it in items if it.role == r), None)
    if anc is not None:
        notes.append(anchor_sentence(anc, ctx))
    rug = next((it for it in items if it.role == "rug"), None)
    if rug is not None:
        notes.append(rug_sentence(rug, items, anc))
    ct = next((it for it in items if it.role == "coffee_table"), None)
    if ct is not None:
        notes.append(coffee_sentence(ct, anc))
    notes.append(seats_sentence(items, anc))
    arts = [it for it in items if it.role == "wall_art"]
    if arts:
        groups: dict[int | None, list[PlacedItem]] = {}
        for a in arts:
            groups.setdefault(a.wall_id, []).append(a)
        for g in groups.values():
            notes.append(art_sentence(g, items))
    notes.append(others_sentence(items))
    notes.append(decor_sentence(items))
    if missing:
        notes.append("Não coube na parte livre visível da foto: " + join_pt(missing) + ".")
    if warnings:
        notes.append("Atenção: " + "; ".join(warnings))
    return [n for n in notes if n]
