"""Regras de decoração, restrições rígidas e função de pontuação do layout.

RESTRIÇÕES RÍGIDAS (qualquer violação descarta a configuração): caixa 3D inteira dentro do
quadro (margem de 1,2% da largura); base só em células FREE (até 25% UNKNOWN, nunca BLOCKED/
OUTSIDE ou fora da grade); peças de piso sem sobreposição (tapete pode ficar por baixo); nada a
menos de 1,2 m da câmera; área de abertura de portas (0,9 m) livre; nenhuma peça esconde mais de
10% de uma janela na imagem (portas: só peças com mais de 0,95 m contam integralmente);
quadros rentes à parede, longe de janelas/portas e sem colidir com móveis.

PONTUAÇÃO (maior = melhor) = soma dos termos abaixo × peso (``Weights``, valor entre colchetes):
- complete [1.0]       + importância (``IMPORTANCE``) de cada item do programa presente.
- facing [1.5]         frente do assento/cama/cômoda para a câmera (``FACING_W`` por papel);
                       sofá de costas é penalizado 2,5×; poltrona de perfil já é ótimo.
- wall_backing [2.0]   fração da traseira encostada numa parede (``BACKING_W`` por papel).
- relations [1.0]      relações de projeto: mesa de centro a 45 cm do sofá, centrada e paralela;
                       poltronas voltadas para a mesa a ~45 cm; tapete sob os pés dianteiros do
                       sofá/poltronas e com a mesa de centro inteira sobre ele; tapete sob os 2/3
                       inferiores da cama; criados-mudos colados nas laterais da cama; cadeiras
                       voltadas para a mesa com 45 cm livres atrás para puxar; mesa lateral junto
                       ao sofá/cama; luminária junto a um assento; plantas em cantos (isoladas no
                       meio da parede perdem pontos); peças altas coladas umas nas outras (-3).
- awkward_gap [1.5]    vãos "inúteis" de 3–50 cm entre móveis volumosos sem relação de projeto.
- walkway [6.0]        fração do piso visível alcançável com passagem de 0,6 m a partir do lado
                       da câmera/portas (pode passar por piso não observado).
- seat_access [1.5]    cada assento/cama precisa de acesso pela frente ou pelas laterais.
- occlusion [6.0]      peças escondidas por peças mais próximas (pintor, baixa resolução): acima
                       de 30% (50% plantas); quadros acima de 8% e cômodas/estantes acima de 15%
                       pesam mais.
- floor_visible [4.0]  piso visível que continua aparecendo (amplitude) abaixo de 55%.
- balance [3.0]        centro de massa visual do grupo principal fora de 0,5 ± 0,08 da largura.
- foreground [4.0]     ocupação da faixa inferior (14%) da foto; peças a menos de 2,6 m da
                       câmera; acessórios à frente do grupo principal ou ocupando >5% do quadro.
- edge_margin [0.8]    folga até a borda do quadro menor que 3,5% da largura.
- window_cover [6.0]   fração de janela/porta escondida (mesmo abaixo do limite rígido).
- prominence [3.0]     âncora (sofá/cama/mesa) longe do centro horizontal ou pequena (<3%).
- asset_pref [0.8]     adequação do asset ao estilo e tons da foto (``style.asset_preference``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from . import geom
from .context import LayoutContext
from .state import Geo, GeoCache, Placement

# ----------------------------------------------------------------------------- papéis
SEATING = {"sofa", "armchair", "dining_chair", "bar_stool", "desk_chair", "ottoman"}
BULKY = {"sofa", "armchair", "bed", "dining_table", "desk", "tv_console", "dresser", "shelf",
         "shelf_large", "desk_storage", "coffee_table"}
WALL_ROLES = {"wall_art", "wall_mirror"}
ON_TOP_ROLES = {"pillows", "vase", "bowl", "books", "throw", "table_lamp", "kitchen_decor"}
FLAT_ROLES = {"rug"}
PLANTS = {"plant_large", "plant_medium", "plant_small"}
TALL_ROLES = {"floor_lamp", "plant_large", "plant_medium", "shelf", "shelf_large"}
STORAGE = {"dresser", "shelf", "shelf_large", "tv_console", "desk_storage"}

# Peso da importância de cada papel (completude, oclusão, penalidades por ausência).
IMPORTANCE: dict[str, float] = {
    "sofa": 10, "bed": 10, "dining_table": 8, "desk": 8, "coffee_table": 5, "rug": 4, "armchair": 3,
    "dresser": 3, "nightstand": 3, "desk_chair": 3, "dining_chair": 2.5, "wall_art": 2.5,
    "tv_console": 1.5, "side_table": 2, "floor_lamp": 2, "plant_large": 2, "plant_medium": 1.6,
    "plant_small": 1, "shelf": 2, "shelf_large": 2, "desk_storage": 2, "bar_stool": 2, "wall_mirror": 2,
    "basket": 1, "vase": 0.8, "bowl": 0.6, "books": 0.6, "pillows": 1.0, "throw": 0.8, "table_lamp": 1.5,
    "kitchen_decor": 0.5, "ottoman": 1.0,
}

# Peso de "mostrar a frente para a câmera" por papel.
FACING_W: dict[str, float] = {"sofa": 2.0, "armchair": 1.0, "ottoman": 0.3, "bed": 1.5, "tv_console": 0.8,
                              "dresser": 1.0, "shelf": 1.0, "shelf_large": 1.0, "desk_storage": 0.6,
                              "desk": 0.6, "nightstand": 0.3, "side_table": 0.2, "wall_mirror": 0.0}
# Peso de "encostado na parede" por papel.
BACKING_W: dict[str, float] = {"sofa": 1.2, "bed": 2.0, "tv_console": 1.0, "dresser": 1.0, "shelf": 1.0,
                               "shelf_large": 1.0, "desk_storage": 0.8, "desk": 0.8, "nightstand": 0.5,
                               "basket": 0.3}

# ----------------------------------------------------------------------------- medidas (m)
WALL_GAP = 0.03            # folga entre o encosto e a parede (2–5 cm)
COFFEE_GAP = 0.45          # frente do sofá -> borda da mesa de centro
SEAT_TABLE_GAP = 0.45      # frente da poltrona -> borda da mesa de centro
SIDE_GAP = 0.05            # mesa lateral / criado-mudo ao lado do assento/cama
WALKWAY = 0.60             # largura mínima de passagem "quando possível"
DOOR_CLEARANCE = 0.90      # área livre na frente de portas (definida em context.py)
MIN_CAMERA_DIST = 1.2      # nenhuma peça a menos disso da câmera (horizontal)
NEAR_CAMERA_SOFT = 2.6     # abaixo disso a peça "invade" o primeiro plano (penalidade suave)
MAX_UNKNOWN_FRAC = 0.25    # fração máxima da base sobre piso não observado
MAX_WINDOW_COVER = 0.10    # fração máxima de uma janela/porta escondida por uma peça (na imagem)
ART_CENTER_Z = 1.50        # altura do centro de quadros/espelhos
ART_ABOVE_GAP = 0.20       # folga mínima entre o móvel e a moldura acima dele
OVERLAP_TOL = 0.005        # interpenetração tolerada entre bases (m)
RUG_SIZES = ((2.0, 1.4), (2.4, 1.7), (3.0, 2.0))   # tamanhos comerciais (comprimento x largura)
_EIGHT = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Weights:
    complete: float = 1.0
    facing: float = 1.5
    wall_backing: float = 2.0
    relations: float = 1.0
    awkward_gap: float = 1.5
    walkway: float = 6.0
    seat_access: float = 1.5
    occlusion: float = 6.0
    floor_visible: float = 4.0
    balance: float = 3.0
    foreground: float = 4.0
    edge_margin: float = 0.8
    window_cover: float = 6.0
    prominence: float = 3.0
    asset_pref: float = 0.8


WEIGHTS = Weights()


@dataclass
class Score:
    total: float
    terms: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)
    occlusion: dict = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return not self.violations


# ----------------------------------------------------------------------------- restrições rígidas

def item_violations(p: Placement, g: Geo) -> list[str]:
    """Restrições rígidas de uma peça isolada (mensagens curtas, para log/testes)."""
    v = []
    if not g.inside:
        v.append(f"{p.key}: fora do quadro da foto")
    if g.cam_dist < MIN_CAMERA_DIST:
        v.append(f"{p.key}: a menos de {MIN_CAMERA_DIST} m da câmera")
    if p.mount == "floor":
        if g.bad > 0:
            v.append(f"{p.key}: sobre área bloqueada/fora da sala")
        if g.unknown > MAX_UNKNOWN_FRAC:
            v.append(f"{p.key}: {g.unknown:.0%} sobre piso não observado")
        if p.role not in FLAT_ROLES and g.door_hit:
            v.append(f"{p.key}: invade a abertura de uma porta")
    if p.role not in FLAT_ROLES and p.mount != "wall" and g.window_cover > MAX_WINDOW_COVER:
        v.append(f"{p.key}: esconde {g.window_cover:.0%} de uma janela/porta na foto")
    return v


def near(ga: Geo, gb: Geo, margin: float = 0.0) -> bool:
    """Teste rápido pelos círculos envolventes das bases (falso = separadas por mais que ``margin``)."""
    return math.hypot(ga.center[0] - gb.center[0], ga.center[1] - gb.center[1]) <= ga.radius + gb.radius + margin


def gap_between(ga: Geo, gb: Geo, cap: float = 9.0) -> float:
    """Distância entre bases; devolve ``cap`` se os círculos envolventes já estão mais longe que isso."""
    if not near(ga, gb, cap):
        return cap
    return geom.polygon_distance(ga.poly, gb.poly)


def floor_overlap(a: Placement, ga: Geo, b: Placement, gb: Geo) -> float:
    """Interpenetração (m) entre bases de peças de piso (tapete não conta)."""
    if a.mount != "floor" or b.mount != "floor" or a.role in FLAT_ROLES or b.role in FLAT_ROLES:
        return 0.0
    if not near(ga, gb):
        return 0.0
    return max(0.0, geom.obb_penetration(ga.poly, gb.poly))


def box_overlap_3d(a: Placement, ga: Geo, b: Placement, gb: Geo) -> bool:
    """Caixas 3D se interceptam (usado para quadros na parede contra móveis)."""
    if a.z >= b.z + b.h - 1e-3 or b.z >= a.z + a.h - 1e-3:
        return False
    if not near(ga, gb):
        return False
    return geom.obb_penetration(ga.poly, gb.poly) > OVERLAP_TOL


def _pair_msg(a: Placement, ga: Geo, b: Placement, gb: Geo) -> str | None:
    if floor_overlap(a, ga, b, gb) > OVERLAP_TOL:
        return f"{a.key} sobrepõe {b.key}"
    wa, wb = a.mount == "wall", b.mount == "wall"
    if (wa or wb) and a.role not in FLAT_ROLES and b.role not in FLAT_ROLES and a.parent != b.key and b.parent != a.key:
        if box_overlap_3d(a, ga, b, gb):
            return f"{a.key} colide com {b.key}"
    return None


def pair_violations(items: list[Placement], geos: list[Geo], others: list[Placement] | None = None,
                    ogeos: list[Geo] | None = None) -> list[str]:
    """Sobreposições entre peças (todas contra todas, ou ``items`` contra ``others``)."""
    v = []
    if others is None:
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                m = _pair_msg(items[i], geos[i], items[j], geos[j])
                if m:
                    v.append(m)
        return v
    for a, ga in zip(items, geos):
        for b, gb in zip(others, ogeos):
            m = _pair_msg(a, ga, b, gb)
            if m:
                v.append(m)
    return v


# ----------------------------------------------------------------------------- termos de pontuação

def facing_reward(role: str, f: float) -> float:
    if role == "sofa":
        return f if f >= 0 else 2.5 * f
    if role in ("armchair", "ottoman"):
        return 0.4 if f >= -0.25 else 0.4 + 2.0 * (f + 0.25)
    if role == "bed":
        return 0.4 if f >= -0.3 else 0.4 + 2.0 * (f + 0.3)
    if role == "desk":
        return 0.3 if f >= -0.5 else 0.3 + (f + 0.5)
    return min(f, 0.5) if f >= 0 else 1.5 * f


def _seat_front_strip(p: Placement, depth: float = 0.15) -> np.ndarray:
    """Pontos da faixa dianteira de um assento (onde ficam os pés da frente)."""
    lx = np.linspace(-p.w / 2 + 0.05, p.w / 2 - 0.05, 9)
    ly = np.linspace(-p.d / 2, -p.d / 2 + depth, 3)
    LX, LY = np.meshgrid(lx, ly)
    return geom.local_to_world(p.x, p.y, p.yaw, LX.ravel(), LY.ravel())


def _inside_frac(pts: np.ndarray, poly: np.ndarray) -> float:
    return float(np.mean(geom.points_in_convex(pts, poly)))


def relation_terms(items: list[Placement], geos: dict[str, Geo], ctx: LayoutContext) -> float:
    """Relações de projeto entre peças (ver docstring do módulo)."""
    by: dict[str, list[Placement]] = {}
    for p in items:
        by.setdefault(p.role, []).append(p)
    R = 0.0
    sofa = by.get("sofa", [None])[0]
    ct = by.get("coffee_table", [None])[0]
    rug = by.get("rug", [None])[0]
    seats = [p for p in items if p.role in ("sofa", "armchair", "desk_chair", "dining_chair", "bed", "ottoman")]
    if sofa is not None and ct is not None:
        f, u = sofa.front, sofa.lateral
        gap = float(np.min((geos[ct.key].poly - sofa.xy) @ f)) - sofa.d / 2
        R -= min(abs(gap - COFFEE_GAP) / 0.10, 4.0)
        R -= min(abs((ct.xy - sofa.xy) @ u) / 0.25, 4.0)
        R -= abs(math.sin(math.radians(2 * (ct.yaw - sofa.yaw))))
    for a in by.get("armchair", []):
        if a.group != "conversa" or ct is None:
            continue
        to_ct = ct.xy - a.xy
        dist = float(np.linalg.norm(to_ct))
        facing = float(a.front @ (to_ct / (dist + 1e-9)))
        R -= max(0.0, 0.75 - facing) * 2.0
        g = geom.polygon_distance(geos[a.key].poly, geos[ct.key].poly)
        R -= min(abs(g - SEAT_TABLE_GAP) / 0.25, 3.0)
        if sofa is not None and np.linalg.norm(a.xy - sofa.xy) > 3.4:
            R -= 1.0
    if rug is not None:
        rpoly = geos[rug.key].poly
        if sofa is not None and rug.group == "conversa":
            R += 1.2 * min(1.0, _inside_frac(_seat_front_strip(sofa), rpoly) / 0.7)
            for a in by.get("armchair", []):
                if a.group == "conversa":
                    R += 0.6 * min(1.0, _inside_frac(_seat_front_strip(a), rpoly) / 0.6) - 0.3
            if ct is not None:
                # mesa de centro inteira sobre o tapete (e de preferência com ~25 cm de sobra)
                full = _inside_frac(geom.sample_obb(ct.x, ct.y, ct.w, ct.d, ct.yaw, step=0.05, shrink=0), rpoly)
                R += 0.8 if full >= 0.999 else -0.5 - 3.0 * (1.0 - full)
                inner = geom.sample_obb(ct.x, ct.y, ct.w + 0.5, ct.d + 0.5, ct.yaw, step=0.1, shrink=0)
                R += 0.6 * _inside_frac(inner, rpoly)
        bed = by.get("bed", [None])[0]
        if bed is not None and rug.group == "cama":
            # tapete sob os 2/3 inferiores da cama e sobrando dos lados
            lx = np.linspace(-bed.w / 2 + 0.05, bed.w / 2 - 0.05, 7)
            ly = np.linspace(-bed.d / 2 + 0.05, bed.d / 6, 5)
            LX, LY = np.meshgrid(lx, ly)
            lower = geom.local_to_world(bed.x, bed.y, bed.yaw, LX.ravel(), LY.ravel())
            R += 1.5 * _inside_frac(lower, rpoly)
            side = geom.local_to_world(bed.x, bed.y, bed.yaw, [-bed.w / 2 - 0.35, bed.w / 2 + 0.35], [-bed.d / 4] * 2)
            R += 0.5 * _inside_frac(side, rpoly)
        table = by.get("dining_table", [None])[0]
        if table is not None and rug.group == "jantar":
            pts = geom.sample_obb(table.x, table.y, table.w + 1.0, table.d + 1.0, table.yaw, step=0.15, shrink=0)
            R += 1.5 * _inside_frac(pts, rpoly)
    for st in by.get("side_table", []):
        best = -1.0
        for s in seats:
            if s.mount != "floor":
                continue
            g = gap_between(geos[st.key], geos[s.key], 2.0)
            if g <= 0.12:   # junto ao braço do sofá/cama vale mais que junto à poltrona
                best = max(best, 0.9 if s.role in ("sofa", "bed") else 0.4)
            else:
                best = max(best, -min(g, 2.0) / 2.0)
        R += best
    for lp in by.get("floor_lamp", []):
        cands = [gap_between(geos[lp.key], geos[s.key], 2.0) for s in seats if s.mount == "floor"]
        g = min(cands) if cands else 3.0
        R += 0.7 if g <= 0.6 else -0.5 * min(g, 2.0) / 2.0
    tv = by.get("tv_console", [None])[0]
    if tv is not None and sofa is not None:
        to_s = sofa.xy - tv.xy
        dist = float(np.linalg.norm(to_s))
        R += 0.8 * float(tv.front @ (to_s / (dist + 1e-9)))
        R += 0.4 if 1.8 <= dist <= 4.2 else -0.4
    bed = by.get("bed", [None])[0]
    if bed is not None:
        for ns in by.get("nightstand", []):
            g = geom.polygon_distance(geos[ns.key].poly, geos[bed.key].poly)
            back = abs((ns.xy - bed.xy) @ bed.front + bed.d / 2 - ns.d / 2)
            R += 0.8 if (g <= 0.12 and back <= 0.12) else -0.5
    for table_role, chair_role in (("dining_table", "dining_chair"), ("desk", "desk_chair")):
        table = by.get(table_role, [None])[0]
        if table is None:
            continue
        for ch in by.get(chair_role, []):
            to_t = table.xy - ch.xy
            facing = float(ch.front @ (to_t / (np.linalg.norm(to_t) + 1e-9)))
            g = geom.polygon_distance(geos[ch.key].poly, geos[table.key].poly)
            R += 0.5 if (facing > 0.8 and g <= 0.15) else -0.5
            # espaço para puxar a cadeira: faixa de 45 cm atrás dela livre de outras peças
            zone = geom.obb_corners(*(ch.xy - ch.front * (ch.d / 2 + 0.225)), ch.w, 0.45, ch.yaw)
            for q in items:
                if q.mount == "floor" and q.role not in FLAT_ROLES and q.key != ch.key and q.group != ch.group:
                    if geom.obb_penetration(zone, geos[q.key].poly) > 0.01:
                        R -= 1.0
    for p in items:
        if p.role in PLANTS and p.mount == "floor":
            dc = min((float(np.linalg.norm(p.xy - c.point)) for c in ctx.corners), default=9.0)
            near_end = min((gap_between(geos[p.key], geos[q.key], 0.5)
                            for q in items if q.role in BULKY and q.mount == "floor"), default=9.0)
            R += 0.6 if dc <= 0.75 else (0.3 if near_end <= 0.2 else -0.4)
    # peças altas encostadas umas nas outras (luminária colada na planta etc.) poluem a foto
    tall = [p for p in items if p.mount == "floor" and p.role in TALL_ROLES]
    for i in range(len(tall)):
        for j in range(i + 1, len(tall)):
            if gap_between(geos[tall[i].key], geos[tall[j].key], 0.5) < 0.35:
                R -= 3.0
    return R


def walkway_terms(items: list[Placement], geos: dict[str, Geo], ctx: LayoutContext) -> tuple[float, int]:
    """Fração do piso livre alcançável com passagem >= WALKWAY e nº de assentos inacessíveis.

    A pessoa pode passar por piso não observado (UNKNOWN, ex.: fora do quadro, onde estava o
    fotógrafo), mas a fração é medida só sobre o piso visível (FREE).
    """
    occ = np.zeros(ctx.walk_free.shape, np.uint8)
    for p in items:
        if p.mount != "floor" or p.role in FLAT_ROLES:
            continue
        q = np.round(ctx.to_walk_px(geos[p.key].poly) * 16).astype(np.int32)
        cv2.fillConvexPoly(occ, q, 1, lineType=cv2.LINE_8, shift=4)
    walk = (ctx.walk_ok & (occ == 0)).astype(np.uint8)
    free = ctx.walk_free & (occ == 0)
    total = int(free.sum())
    if total == 0:
        return 0.0, 0
    r = max(1, int(round((WALKWAY / 2 - 0.02) / ctx.walk_cell)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    passable = cv2.erode(walk, k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if passable.sum() == 0:
        return 0.0, sum(1 for p in items if p.role in SEATING)
    lab, n = ndimage.label(passable, structure=_EIGHT)
    pcells = passable.astype(bool)
    dist = ctx.walk_cam_dist
    dmin = dist[pcells].min()
    seeds = set(np.unique(lab[pcells & (dist <= dmin + 0.6)]).tolist())
    for dz in ctx.door_zones:
        ix, iy, ok = ctx.walk_index(dz.mean(0)[None])
        if ok[0] and lab[iy[0], ix[0]] > 0:
            seeds.add(int(lab[iy[0], ix[0]]))
    seeds.discard(0)
    reach = np.isin(lab, sorted(seeds)) & pcells
    area = (cv2.dilate(reach.astype(np.uint8), k) > 0) & (walk > 0)
    frac = float((area & free).sum()) / total
    blocked = 0
    for p in items:
        if p.role not in ("sofa", "armchair", "bed", "desk_chair") or p.mount != "floor":
            continue
        if p.role == "bed":
            lx, ly = [-p.w / 2 - 0.3, p.w / 2 + 0.3, 0.0], [0.0, 0.0, -p.d / 2 - 0.3]
        else:   # faixa à frente do assento, incluindo as pontas (acesso pelas laterais da mesa)
            lx = [-p.w / 2 - 0.25, -p.w / 4, 0.0, p.w / 4, p.w / 2 + 0.25]
            ly = [-p.d / 2 - 0.25] * 5
        pts = geom.local_to_world(p.x, p.y, p.yaw, lx, ly)
        ix, iy, ok = ctx.walk_index(pts)
        if not any(ok[i] and area[iy[i], ix[i]] for i in range(len(ix))):
            blocked += 1
    return frac, blocked


def image_terms(items: list[Placement], geos: dict[str, Geo], ctx: LayoutContext) -> dict:
    """Métricas na imagem (baixa resolução): oclusão por peça, piso visível, equilíbrio, 1º plano."""
    vis_items = [p for p in items if p.role not in FLAT_ROLES and p.mount != "on_top"]
    order = sorted(vis_items, key=lambda p: geos[p.key].depth)       # mais próximo primeiro
    covered = np.zeros((ctx.lh, ctx.lw), bool)
    occl: dict[str, float] = {}
    for p in order:
        x0, y0, m = geos[p.key].low
        if m.size == 0:
            occl[p.key] = 0.0
            continue
        sub = covered[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        tot = int(m.sum())
        occl[p.key] = float((m & sub).sum()) / tot if tot else 0.0
        sub |= m
    floor_items = np.zeros_like(covered)
    for p in vis_items:
        if p.mount == "floor":
            x0, y0, m = geos[p.key].low
            if m.size:
                floor_items[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] |= m
    fl = ctx.floor_low
    floor_vis = float((fl & ~floor_items).sum()) / max(1, int(fl.sum()))
    heavy = np.zeros_like(covered)
    for p in vis_items:
        if p.mount == "floor" and (p.group or p.role in BULKY):
            x0, y0, m = geos[p.key].low
            if m.size:
                heavy[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] |= m
    ys, xs = np.nonzero(heavy)
    cx = float(xs.mean() + 0.5) / ctx.lw if len(xs) else 0.5
    band = floor_items[int(ctx.lh * 0.86):]
    fg = float(band.mean()) if band.size else 0.0
    return dict(occl=occl, floor_visible=floor_vis, centroid_x=cx, foreground=fg)


def _depth(poly: np.ndarray, ctx: LayoutContext) -> float:
    """Menor profundidade (ao longo da direção da câmera, no piso) dos cantos da base."""
    return float(np.min((poly - ctx.cam_xy) @ ctx.forward2))


def accessory_front(items: list[Placement], geos: dict[str, Geo], ctx: LayoutContext) -> float:
    """Acessórios (fora do grupo principal) à frente do grupo ou enormes no quadro poluem o 1º plano."""
    grp = [p for p in items if p.group and p.mount == "floor" and p.role not in FLAT_ROLES]
    if not grp:
        return 0.0
    front = min(_depth(geos[p.key].poly, ctx) for p in grp)
    pen = 0.0
    img_area = float(ctx.W * ctx.H)
    for p in items:
        if p.group or p.mount != "floor" or p.role in FLAT_ROLES:
            continue
        g = geos[p.key]
        pen += max(0.0, front - 0.15 - _depth(g.poly, ctx)) * min(1.5, p.h) * 0.8
        pen += max(0.0, g.hull_area / img_area - 0.05) * 6.0
    return pen


def anchor_of(items: list[Placement]) -> Placement | None:
    for role in ("sofa", "bed", "dining_table", "desk"):
        for p in items:
            if p.role == role:
                return p
    return None


def score_layout(items: list[Placement], cache: GeoCache, ctx: LayoutContext, prefs: dict[str, float] | None = None,
                 weights: Weights = WEIGHTS, check_hard: bool = True) -> Score:
    """Pontuação completa de um conjunto de peças (ver docstring do módulo)."""
    prefs = prefs or {}
    geos = {p.key: cache.geo(p) for p in items}
    viol: list[str] = []
    if check_hard:
        for p in items:
            viol += item_violations(p, geos[p.key])
        viol += pair_violations(items, [geos[p.key] for p in items])
    T: dict[str, float] = {}
    T["complete"] = weights.complete * sum(IMPORTANCE.get(p.role, 1.0) for p in items)
    fac = 0.0
    back = 0.0
    for p in items:
        if p.mount != "floor":
            continue
        if p.role in FACING_W:
            f = float(p.front @ ctx.dir_to_camera(p.xy))
            fac += FACING_W[p.role] * facing_reward(p.role, f)
        if p.role in BACKING_W:
            frac, _ = ctx.wall_contact(geos[p.key].poly, -p.front)
            back += BACKING_W[p.role] * frac
    T["facing"] = weights.facing * fac
    T["wall_backing"] = weights.wall_backing * back
    T["relations"] = weights.relations * relation_terms(items, geos, ctx)
    gap_pen = 0.0
    exempt = {frozenset(("sofa", "coffee_table")), frozenset(("armchair", "coffee_table")),
              frozenset(("sofa", "armchair")), frozenset(("dining_table", "dining_chair")),
              frozenset(("desk", "desk_chair"))}
    bulky = [p for p in items if p.role in BULKY and p.mount == "floor"]
    for i in range(len(bulky)):
        for j in range(i + 1, len(bulky)):
            a, b = bulky[i], bulky[j]
            if frozenset((a.role, b.role)) in exempt and a.group == b.group:
                continue
            g = gap_between(geos[a.key], geos[b.key], 0.6)
            if 0.03 < g < 0.5:
                gap_pen += (0.5 - g) / 0.47
    T["awkward_gap"] = -weights.awkward_gap * gap_pen
    frac, blocked = walkway_terms(items, geos, ctx)
    T["walkway"] = -weights.walkway * max(0.0, 0.92 - frac) / 0.92
    T["seat_access"] = -weights.seat_access * blocked
    im = image_terms(items, geos, ctx)
    occ_pen = 0.0
    for p in items:
        o = im["occl"].get(p.key)
        if o is None:
            continue
        if p.mount == "wall":       # quadro/espelho escondido atrás de peças estraga a composição
            occ_pen += IMPORTANCE.get(p.role, 1.0) / 2.5 * max(0.0, o - 0.08)
            continue
        if p.role in STORAGE:       # cômoda/estante/rack escondidos atrás de poltronas parecem erro
            occ_pen += IMPORTANCE.get(p.role, 1.0) / 2.5 * max(0.0, o - 0.15)
            continue
        allowed = 0.5 if p.role in PLANTS else 0.3
        occ_pen += IMPORTANCE.get(p.role, 1.0) / 5.0 * max(0.0, o - allowed)
    T["occlusion"] = -weights.occlusion * occ_pen
    T["floor_visible"] = -weights.floor_visible * max(0.0, 0.55 - im["floor_visible"]) / 0.55
    T["balance"] = -weights.balance * max(0.0, abs(im["centroid_x"] - 0.5) - 0.08)
    near_cam = sum(max(0.0, NEAR_CAMERA_SOFT - geos[p.key].cam_dist) * min(1.5, p.h)
                   for p in items if p.mount == "floor" and p.role not in FLAT_ROLES)
    T["foreground"] = -weights.foreground * (im["foreground"] + 0.35 * near_cam
                                             + accessory_front(items, geos, ctx))
    em = 0.0
    lim = 0.035 * ctx.W
    for p in items:
        if p.mount == "on_top":
            continue
        em += max(0.0, 1.0 - geos[p.key].margin / lim) * (0.5 if p.role in FLAT_ROLES else 1.0)
    T["edge_margin"] = -weights.edge_margin * em
    T["window_cover"] = -weights.window_cover * sum(geos[p.key].window_cover for p in items
                                                     if p.role not in FLAT_ROLES and p.mount != "wall")
    anc = anchor_of(items)
    if anc is not None:
        g = geos[anc.key]
        if g.hull_area > 0:
            cxa = float(g.hull[:, 0].mean()) / ctx.W
            area = g.hull_area / (ctx.W * ctx.H)
            T["prominence"] = -weights.prominence * (4.0 * max(0.0, abs(cxa - 0.5) - 0.22)
                                                     + 30.0 * max(0.0, 0.03 - area))
    T["asset_pref"] = weights.asset_pref * sum(prefs.get(p.asset_id, 2.0) - 2.0 for p in items
                                               if p.asset_id.startswith("ph:"))
    total = float(sum(T.values()))
    if viol:
        total -= 1000.0 * len(viol)
    return Score(total, {k: round(v, 4) for k, v in T.items()}, viol, im["occl"])


# ----------------------------------------------------------------------------- validação final

def validate_layout(layout, ctx: LayoutContext, cache: GeoCache | None = None) -> list[str]:
    """Verifica as restrições rígidas de um ``Layout`` pronto. Lista vazia = tudo certo."""
    from .surfaces import fits_on, support_for

    cache = cache or GeoCache(ctx)
    items = [Placement.from_item(it) for it in layout.items]
    geos = [cache.geo(p) for p in items]
    v = []
    for p, g in zip(items, geos):
        v += item_violations(p, g)
    v += pair_violations(items, geos)
    by_key = {p.key: p for p in items}
    for p in items:
        if p.mount == "on_top":
            par = by_key.get(p.parent or "")
            if par is None:
                v.append(f"{p.key}: apoio '{p.parent}' inexistente")
                continue
            sup = support_for(par.asset_id, par.role, (par.w, par.d, par.h), par.params)
            if sup is None:
                v.append(f"{p.key}: '{par.key}' não tem superfície de apoio")
                continue
            if abs(p.z - (par.z + sup.z)) > 0.005:
                v.append(f"{p.key}: altura {p.z:.3f} diferente do apoio {par.z + sup.z:.3f}")
            lxy = geom.world_to_local(par.x, par.y, par.yaw, p.xy[None])[0]
            dyaw = math.radians(p.yaw - par.yaw)
            ew = abs(math.cos(dyaw)) * p.w + abs(math.sin(dyaw)) * p.d
            ed = abs(math.sin(dyaw)) * p.w + abs(math.cos(dyaw)) * p.d
            if sup.kind == "top" and not fits_on(sup, lxy[0], lxy[1], ew, ed, margin=0.02):
                v.append(f"{p.key}: base fora do tampo de {par.key}")
            if sup.kind == "seat" and not fits_on(sup, lxy[0], lxy[1], min(ew, 0.1), min(ed, 0.1), margin=0.02):
                v.append(f"{p.key}: fora do assento de {par.key}")
        if p.mount == "wall":
            if p.wall_id is None:
                v.append(f"{p.key}: sem parede")
                continue
            w = ctx.wall_by_id(p.wall_id)
            s = float((p.xy - w.p0) @ w.direction)
            dist = float((p.xy - w.p0) @ w.normal)
            if s - p.w / 2 < -0.01 or s + p.w / 2 > w.length + 0.01:
                v.append(f"{p.key}: além da extensão da parede")
            if not (0.0 <= dist - p.d / 2 <= 0.03):
                v.append(f"{p.key}: não está rente à parede ({dist - p.d / 2:.3f} m)")
            if not w.window_free(s - p.w / 2, s + p.w / 2, p.z, p.z + p.h, margin=0.05):
                v.append(f"{p.key}: sobre janela/porta")
            if p.z + p.h > ctx.ceiling - 0.05:
                v.append(f"{p.key}: encosta no teto")
    return v
