"""Catálogo de assets 3D por papel (role) e tipo de ambiente.

Fontes:
- ``ph:<id>``  -> Poly Haven (CC0, uso comercial livre). Dimensões oficiais da API (mm -> m).
- ``proc:<k>`` -> assets procedurais gerados no Blender (mobilia/assets/procedural.py),
                  para itens que a Poly Haven não tem (tapete, luminária de piso, cama...).

``size`` = (largura X, profundidade Y, altura Z) do asset normalizado, frente para -Y.
``yaw_fix`` corrige assets cuja frente original não aponta para -Y (verificado no import).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    role: str
    size: tuple[float, float, float]
    styles: tuple[str, ...] = ("moderno",)
    tone: str = "neutro"            # claro | escuro | madeira | neutro | colorido
    yaw_fix: float = 0.0
    scalable: bool = False          # procedurais aceitam dimensões arbitrárias
    tags: tuple[str, ...] = field(default_factory=tuple)


def _ph(pid, role, mm, styles=("moderno",), tone="neutro", yaw_fix=0.0, tags=()):
    return AssetSpec(f"ph:{pid}", role, tuple(round(v / 1000.0, 3) for v in mm), tuple(styles), tone,
                     yaw_fix, False, tuple(tags))


def _proc(kind, role, size, styles=("moderno",), tone="neutro", tags=()):
    return AssetSpec(f"proc:{kind}", role, tuple(size), tuple(styles), tone, 0.0, True, tuple(tags))


CATALOG: list[AssetSpec] = [
    # --- assentos
    # tag "albedo_escuro": textura difusa quase preta (mediana sRGB < 0,2, medida nos arquivos 1k:
    # sofá 02 0,06, almofada da poltrona moderna 01 0,06, pufe 01 0,15). Em sala clara são excluídos
    # (``style.rank_assets``): viram um "buraco" preto na foto.
    _ph("sofa_02", "sofa", (1807, 817, 709), ("moderno", "classico"), "escuro", tags=("couro", "albedo_escuro")),
    _ph("Sofa_01", "sofa", (1571, 658, 796), ("classico",), "claro"),
    _ph("sofa_03", "sofa", (2731, 925, 1118), ("classico",), "madeira"),
    _ph("modern_arm_chair_01", "armchair", (820, 986, 1022), ("moderno", "industrial"), "escuro",
        tags=("albedo_escuro",)),
    _ph("mid_century_lounge_chair", "armchair", (1008, 1190, 1169), ("moderno",), "madeira"),
    _ph("ArmChair_01", "armchair", (848, 765, 1065), ("classico",), "claro"),
    _ph("Ottoman_01", "ottoman", (884, 621, 624), ("moderno", "classico"), "escuro", tags=("albedo_escuro",)),
    # --- mesas
    _ph("modern_coffee_table_01", "coffee_table", (1201, 600, 389), ("moderno",), "madeira", yaw_fix=90.0),
    _ph("modern_coffee_table_02", "coffee_table", (1199, 1199, 369), ("moderno",), "madeira"),
    _ph("coffee_table_round_01", "coffee_table", (1301, 1301, 491), ("moderno",), "claro"),
    _ph("side_table_01", "side_table", (550, 450, 550), ("moderno", "escandinavo"), "madeira"),
    _ph("modern_wooden_cabinet", "tv_console", (2440, 520, 680), ("moderno",), "madeira"),
    _ph("round_wooden_table_02", "dining_table", (796, 796, 746), ("classico", "escandinavo"), "madeira"),
    _ph("wooden_table_02", "dining_table", (1134, 706, 799), ("rustico",), "madeira"),
    _ph("dining_chair_02", "dining_chair", (433, 576, 973), ("moderno",), "neutro"),
    _ph("painted_wooden_chair_01", "dining_chair", (431, 539, 956), ("rustico", "escandinavo"), "claro"),
    _ph("bar_chair_round_01", "bar_stool", (483, 485, 750), ("moderno", "industrial"), "madeira"),
    _ph("painted_wooden_nightstand", "nightstand", (504, 508, 615), ("rustico", "escandinavo"), "claro"),
    _ph("ClassicNightstand_01", "nightstand", (567, 424, 699), ("classico",), "madeira"),
    _ph("GothicCommode_01", "dresser", (1201, 580, 1212), ("classico",), "madeira"),
    _ph("vintage_wooden_drawer_01", "desk_storage", (1715, 1006, 1090), ("classico",), "madeira"),
    # --- armazenamento
    _ph("wooden_display_shelves_01", "shelf", (1077, 371, 1556), ("moderno", "escandinavo"), "madeira", yaw_fix=90.0),
    _ph("steel_frame_shelves_03", "shelf_large", (2351, 721, 2311), ("industrial",), "madeira"),
    # --- plantas e decoração
    _ph("potted_plant_01", "plant_large", (588, 636, 1350), ("moderno", "escandinavo", "classico")),
    _ph("potted_plant_02", "plant_medium", (701, 657, 841), ("moderno", "escandinavo", "classico")),
    _ph("potted_plant_04", "plant_small", (170, 185, 267), ("moderno", "escandinavo")),
    _ph("ceramic_vase_01", "vase", (204, 206, 401), ("moderno",), "claro"),
    _ph("ceramic_vase_02", "vase", (219, 221, 309), ("moderno",), "claro"),
    _ph("ceramic_vase_04", "vase", (183, 181, 342), ("moderno",), "claro"),
    # quadros: molduras douradas com pintura holandesa real escaneada (CC0) no estilo clássico; nos
    # demais, ``proc:canvas_art`` (pintura abstrata gerada na paleta, tamanho livre). Os quadros
    # modernos 01/02 da Poly Haven ficaram de fora: o 01 tem um cartão cinza de modelo ("Ray Homes")
    # no lugar da arte e o 02 tem a imagem esticada nas bordas (ver ``PLACEHOLDER_ART``).
    _ph("fancy_picture_frame_01", "wall_art", (603, 20, 464), ("classico",), "madeira"),
    _ph("fancy_picture_frame_02", "wall_art", (661, 89, 771), ("classico",), "madeira"),
    _ph("throw_pillows_01", "pillows", (940, 452, 450), ("moderno", "boho"), "colorido"),
    _ph("wooden_bowl_01", "bowl", (313, 309, 93), ("moderno", "rustico"), "madeira"),
    _ph("wicker_basket_02", "basket", (347, 254, 200), ("boho", "escandinavo"), "madeira"),
    _ph("book_encyclopedia_set_01", "books", (551, 163, 237), ("classico", "moderno"), "colorido"),
    _ph("ornate_mirror_01", "wall_mirror", (486, 25, 744), ("classico",), "claro"),
    _ph("jug_01", "kitchen_decor", (275, 167, 211), ("rustico", "classico"), "claro"),
    _ph("pot_enamel_01", "kitchen_decor", (262, 180, 178), ("rustico",), "claro"),
    # --- procedurais (dimensões nominais; o layout pode redimensionar)
    _proc("rug", "rug", (2.4, 1.7, 0.012), ("moderno", "escandinavo", "classico", "boho")),
    _proc("floor_lamp", "floor_lamp", (0.42, 0.42, 1.62), ("moderno", "escandinavo", "industrial")),
    _proc("table_lamp", "table_lamp", (0.28, 0.28, 0.48), ("moderno", "escandinavo")),
    _proc("bed", "bed", (1.68, 2.12, 0.95), ("moderno", "escandinavo")),
    _proc("curtain", "curtain", (1.6, 0.12, 2.5), ("moderno", "escandinavo", "classico")),
    _proc("desk", "desk", (1.4, 0.7, 0.75), ("moderno", "escandinavo")),
    _proc("throw_blanket", "throw", (1.2, 0.6, 0.06), ("moderno", "escandinavo", "boho")),
    # quadro com pintura abstrata gerada (tela de 3,5 cm ou moldura fina + passe-partout); o layout
    # escolhe largura e altura pelas regras de projeto (``solver._art_sizes``)
    _proc("canvas_art", "wall_art", (0.8, 0.035, 0.6), ("moderno", "escandinavo", "industrial", "boho", "rustico"),
          "colorido"),
    # pilha de 3 livros deitados (~30 x 23 x 3 cm cada, girados e deslocados): decoração de tampo
    _proc("book_stack", "books", (0.33, 0.26, 0.085),
          ("moderno", "escandinavo", "classico", "boho", "industrial", "rustico"), "colorido"),
]


# Ativos da Poly Haven que NÃO entram no catálogo por parecerem modelo de CG na foto.
PLACEHOLDER_ART: dict[str, str] = {
    "ph:hanging_picture_frame_01": "cartão cinza de modelo (\"Ray Homes\") no lugar da arte",
    "ph:hanging_picture_frame_02": "imagem esticada nas bordas da moldura",
}


@dataclass(frozen=True)
class TextureSpec:
    """Textura CC0 de tecido da Poly Haven para o tapete (baixada e cacheada como os modelos)."""
    tex_id: str
    tile_m: float        # lado da amostra fotografada (m), da API ("dimensions")
    lum_mean: float      # luminância LINEAR média do mapa difuso (medida no arquivo 2k)
    sheen: float = 0.7   # brilho de fibra (lã/pelúcia alto, juta baixo)


# Tapete texturizado por estilo: trama fotográfica tingida pela paleta + desenho tecido discreto.
RUG_TEXTURES: dict[str, TextureSpec] = {
    "moderno": TextureSpec("poly_wool_herringbone", 0.272, 0.184, 0.7),
    "escandinavo": TextureSpec("curly_teddy_natural", 0.333, 0.676, 0.85),
    "industrial": TextureSpec("hessian_230", 0.268, 0.21, 0.15),
    "classico": TextureSpec("knitted_fleece", 0.268, 0.079, 0.6),
}
RUG_DESIGNS: dict[str, str] = {"moderno": "filete", "escandinavo": "liso", "industrial": "borda",
                               "classico": "borda_dupla"}


# Decoração cujo asset depende do móvel de apoio: papel -> {papel do apoio: asset_id}; "*" vale
# para os demais apoios. A fileira de enciclopédias em pé só faz sentido numa estante; em mesas
# de centro, escrivaninhas, racks etc. vai a pilha de livros deitados.
DECOR_BY_PARENT: dict[str, dict[str, str]] = {
    "books": {"shelf": "ph:book_encyclopedia_set_01", "*": "proc:book_stack"},
}


def decor_asset(role: str, parent_role: str) -> str | None:
    """asset_id da decoração ``role`` sobre um apoio ``parent_role`` (None = escolha por estilo)."""
    table = DECOR_BY_PARENT.get(role)
    if not table:
        return None
    return table.get(parent_role, table.get("*"))


def by_role(role: str) -> list[AssetSpec]:
    return [a for a in CATALOG if a.role == role]


def get(asset_id: str) -> AssetSpec:
    for a in CATALOG:
        if a.asset_id == asset_id:
            return a
    raise KeyError(asset_id)


# Programa de móveis por ambiente: (papel, obrigatório?, quantidade máx.)
ROOM_PROGRAMS: dict[str, list[tuple[str, bool, int]]] = {
    "sala": [("rug", True, 1), ("sofa", True, 1), ("coffee_table", True, 1), ("armchair", False, 2),
             ("side_table", False, 1), ("floor_lamp", False, 1), ("plant_large", False, 1),
             ("plant_medium", False, 1), ("wall_art", False, 1), ("tv_console", False, 1),
             ("pillows", False, 1), ("vase", False, 1), ("bowl", False, 1), ("books", False, 1)],
    "quarto": [("rug", True, 1), ("bed", True, 1), ("nightstand", True, 2), ("table_lamp", False, 2),
               ("dresser", False, 1), ("plant_medium", False, 1), ("wall_art", False, 1),
               ("armchair", False, 1), ("throw", False, 1)],
    "sala_jantar": [("rug", False, 1), ("dining_table", True, 1), ("dining_chair", True, 4),
                    ("plant_large", False, 1), ("wall_art", False, 1), ("vase", False, 1),
                    ("shelf", False, 1)],
    "escritorio": [("rug", False, 1), ("desk", True, 1), ("armchair", True, 1), ("shelf", False, 1),
                   ("floor_lamp", False, 1), ("plant_large", False, 1), ("wall_art", False, 1),
                   ("books", False, 1)],
    "cozinha": [("bar_stool", False, 3), ("dining_table", False, 1), ("dining_chair", False, 2),
                ("plant_medium", False, 1), ("bowl", False, 1), ("kitchen_decor", False, 2),
                ("vase", False, 1)],
    "banheiro": [("plant_small", False, 2), ("basket", False, 1), ("vase", False, 1),
                 ("wall_mirror", False, 1), ("plant_medium", False, 1)],
}
