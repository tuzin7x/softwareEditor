"""Superfícies de apoio dos móveis (onde a decoração "on_top" pode repousar).

Os valores dos assets Poly Haven foram MEDIDOS nas malhas reais (glTF 1k) por ray casting
vertical no Blender (grade de 1 cm), no referencial normalizado do projeto: origem no centro
da base do bounding box, frente voltada para -Y, X = largura (como ``catalog.AssetSpec.size``).
- ``seat``: assento de sofás/poltronas/cadeiras. ``z`` = altura do assento; ``rect`` =
  (x0, y0, x1, y1) da área útil do assento, com y1 = face frontal do encosto.
- ``top``: tampo plano de mesas/cômodas. ``rects`` = retângulos planos utilizáveis.

Observação: ``modern_coffee_table_01`` e ``wooden_display_shelves_01`` têm o eixo longo em Y
no glTF original; os valores abaixo já estão girados 90° para casar com o tamanho do catálogo,
e por isso esses dois assets PRECISAM de ``yaw_fix=90`` no catálogo (sem ele o render gira o
móvel 90° em relação à pegada do layout e a decoração fica metade no ar).
O tampo de ``modern_coffee_table_02`` é ripado (frestas finas): tratado como tampo cheio.
O topo de ``wooden_display_shelves_01`` é a prateleira de cima (z 1.542); os montantes das
pontas sobem até 1.556 fora do retângulo útil.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class Support:
    kind: str                 # "seat" | "top"
    z: float                  # altura da superfície (m, a partir da base do móvel)
    rects: tuple[Rect, ...]   # áreas utilizáveis no referencial local do móvel
    coverage: float = 1.0     # fração da área útil que é material (< 1 em tampo ripado)

    @property
    def back_y(self) -> float:
        """Para assentos: y da face frontal do encosto (maior y do retângulo do assento)."""
        return max(r[3] for r in self.rects)


MEASURED: dict[str, Support] = {
    # assentos (z do assento; retângulo até a frente do encosto)
    "ph:sofa_02": Support("seat", 0.406, ((-0.729, -0.404, 0.731, 0.186),)),
    "ph:Sofa_01": Support("seat", 0.344, ((-0.701, -0.324, 0.689, 0.206),)),
    "ph:sofa_03": Support("seat", 0.560, ((-0.971, -0.458, 0.969, 0.042),)),
    "ph:modern_arm_chair_01": Support("seat", 0.544, ((-0.315, -0.488, 0.315, 0.132),)),
    "ph:mid_century_lounge_chair": Support("seat", 0.465, ((-0.319, -0.590, 0.321, 0.040),)),
    "ph:ArmChair_01": Support("seat", 0.497, ((-0.319, -0.378, 0.311, 0.282),)),
    "ph:Ottoman_01": Support("top", 0.620, ((-0.407, -0.306, 0.403, 0.304),)),
    "ph:dining_chair_02": Support("seat", 0.450, ((-0.202, -0.283, 0.198, 0.067),)),
    "ph:painted_wooden_chair_01": Support("seat", 0.447, ((-0.191, -0.265, 0.199, 0.185),)),
    "ph:bar_chair_round_01": Support("seat", 0.689, ((-0.237, -0.238, 0.233, 0.242),)),
    # tampos
    "ph:modern_coffee_table_01": Support("top", 0.390, ((-0.561, -0.270, 0.559, 0.270),)),
    "ph:modern_coffee_table_02": Support("top", 0.369, ((-0.550, -0.550, 0.550, 0.550),), coverage=0.6),
    "ph:coffee_table_round_01": Support("top", 0.491, ((-0.401, -0.401, 0.399, 0.399),)),
    "ph:side_table_01": Support("top", 0.551, ((-0.235, -0.195, 0.235, 0.195),)),
    "ph:modern_wooden_cabinet": Support("top", 0.680, ((-1.180, -0.230, 1.180, 0.230),)),
    "ph:round_wooden_table_02": Support("top", 0.746, ((-0.278, -0.288, 0.282, 0.282),)),
    "ph:wooden_table_02": Support("top", 0.800, ((-0.527, -0.323, 0.523, 0.327),)),
    "ph:painted_wooden_nightstand": Support("top", 0.616, ((-0.222, -0.214, 0.218, 0.216),)),
    "ph:ClassicNightstand_01": Support("top", 0.700, ((-0.244, -0.182, 0.246, 0.178),)),
    "ph:GothicCommode_01": Support("top", 1.212, ((-0.561, -0.260, 0.559, 0.260),)),
    "ph:vintage_wooden_drawer_01": Support("top", 1.091, ((-0.818, -0.373, 0.822, 0.477),)),
    "ph:wooden_display_shelves_01": Support("top", 1.542, ((-0.499, -0.156, 0.501, 0.154),)),
    "ph:steel_frame_shelves_03": Support("top", 2.302, ((-1.136, -0.331, 1.134, 0.329),)),
}

# Proporções usadas quando um asset novo ainda não foi medido (altura do assento / altura total).
_SEAT_RATIO = {"sofa": 0.57, "armchair": 0.50, "dining_chair": 0.47, "bar_stool": 0.92, "desk_chair": 0.47}
_SEAT_DEPTH_RATIO = {"sofa": 0.72, "armchair": 0.65, "dining_chair": 0.62, "bar_stool": 1.0, "desk_chair": 0.62}

# Recuo mínimo da base de um objeto "on_top" em relação à borda do tampo (m): nada encosta
# na quina nem fica com a beirada no ar.
ON_TOP_INSET = 0.02

BED_MATTRESS_TOP = 0.56        # altura padrão do colchão da cama procedural (m)
BED_HEADBOARD_DEPTH = 0.08     # espessura da cabeceira procedural (m)


def support_for(asset_id: str, role: str, size: tuple[float, float, float] | list[float],
                params: dict | None = None) -> Support | None:
    """Superfície de apoio de um móvel com o tamanho final ``size`` (w, d, h).

    Para assets medidos, os valores são escalados proporcionalmente se ``size`` diferir
    do tamanho medido (só acontece com procedurais). Retorna None se o papel não serve de apoio.
    """
    w, d, h = (float(v) for v in size)
    params = params or {}
    if asset_id in MEASURED:
        return MEASURED[asset_id]
    if asset_id.startswith("proc:") and role in ("bed", "desk"):
        # medida exata do gerador procedural (ex.: itens repousam no edredom, não no colchão)
        try:
            from ..assets import support_surface

            sup = support_surface(asset_id[5:], (w, d, h), params)
        except Exception:
            sup = None
        if sup is not None:
            return Support("seat" if role == "bed" else "top", float(sup["z"]),
                           tuple(tuple(float(v) for v in r) for r in sup["rects"]))
    if role == "bed":
        z = float(params.get("mattress_top", BED_MATTRESS_TOP))
        hd = float(params.get("headboard_depth", BED_HEADBOARD_DEPTH))
        return Support("seat", z, ((-w / 2 + 0.06, -d / 2 + 0.04, w / 2 - 0.06, d / 2 - hd - 0.02),))
    if role in _SEAT_RATIO:
        z = _SEAT_RATIO[role] * h
        dep = _SEAT_DEPTH_RATIO[role] * d
        arm = 0.12 * w if role in ("sofa", "armchair") else 0.02
        return Support("seat", z, ((-w / 2 + arm, -d / 2, w / 2 - arm, -d / 2 + dep),))
    if role in ("coffee_table", "side_table", "nightstand", "dresser", "tv_console", "desk", "dining_table",
                "shelf", "shelf_large", "desk_storage", "ottoman"):
        m = 0.04
        return Support("top", h, ((-w / 2 + m, -d / 2 + m, w / 2 - m, d / 2 - m),))
    return None


def rotated_extent(w: float, d: float, dyaw_deg: float) -> tuple[float, float]:
    """Largura/profundidade do retângulo alinhado ao apoio que envolve uma base w x d girada
    ``dyaw_deg`` em relação ao móvel de apoio."""
    c, s = abs(math.cos(math.radians(dyaw_deg))), abs(math.sin(math.radians(dyaw_deg)))
    return c * w + s * d, s * w + c * d


def fits_on(support: Support, lx: float, ly: float, w: float, d: float, margin: float = 0.01) -> bool:
    """True se a base (w x d, alinhada ao móvel) centrada em (lx, ly) cabe num retângulo de apoio.

    ``margin`` > 0 tolera esse tanto de beirada para fora; ``margin`` < 0 exige recuo da borda."""
    for x0, y0, x1, y1 in support.rects:
        if (lx - w / 2 >= x0 - margin and lx + w / 2 <= x1 + margin and
                ly - d / 2 >= y0 - margin and ly + d / 2 <= y1 + margin):
            return True
    return False
