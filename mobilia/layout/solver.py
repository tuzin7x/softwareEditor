"""Motor de layout automático: a "IA" que escolhe e posiciona os móveis para ESTA foto.

Fluxo de ``plan_layout``:
1. contexto espacial (sala + grade + câmera) e análise de estilo/paleta da foto;
2. grupo principal do cômodo (planners.py): configurações enumeradas, filtradas por restrições
   rígidas e ordenadas pela pontuação (rules.py); as melhores são refinadas por têmpera simulada;
3. peças secundárias (mesa lateral, luminária, plantas, cômoda, estante...) em encaixes de parede,
   canto e "ao lado" das âncoras, escolhidas gulosamente pela pontuação global;
4. polimento conjunto (têmpera simulada) de grupo + secundárias;
5. quadros na parede (acima do sofá/cama/aparador quando possível), decoração sobre os móveis;
6. ordem da coreografia de montagem e validação final.

Determinístico: toda aleatoriedade vem de ``np.random.default_rng(seed)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..analysis.room import BLOCKED, RoomModel
from . import catalog, geom
from .catalog import AssetSpec
from .context import LayoutContext
from .entrances import plan_entrances
from .narrative import ROLE_LABELS, designer_notes
from .model import Layout
from .planners import (PLANNERS, BancadaPlanner, CozinhaMesaPlanner, GroupCand, GroupPlanner)
from .rules import (ART_ABOVE_GAP, ART_CENTER_Z, ON_TOP_ROLES, SIDE_GAP, WALL_GAP, WALL_ROLES,
                    Score, item_violations, pair_violations, score_layout, validate_layout)
from .slots import Slot, adjacent_slots, corner_slots, free_slots, wall_pose, wall_slots
from .state import GeoCache, Placement
from .style import StyleProfile, analyze_style, asset_preference, proc_params, rank_assets
from .surfaces import ON_TOP_INSET, Support, fits_on, support_for

# Onde cada papel secundário pode ser encaixado.
SLOT_KINDS: dict[str, tuple[str, ...]] = {
    "side_table": ("lado",), "floor_lamp": ("lado", "canto"),
    "plant_large": ("canto", "lado", "parede"), "plant_medium": ("canto", "lado", "parede"),
    "plant_small": ("canto", "lado", "parede"), "tv_console": ("parede",), "dresser": ("parede",),
    "shelf": ("parede",), "shelf_large": ("parede",), "desk_storage": ("parede",), "armchair": ("canto", "parede"),
    "basket": ("parede", "canto", "lado"), "vase": ("parede", "canto", "lado"), "ottoman": ("livre",),
    "nightstand": ("lado",),
}
ADJ_ANCHORS: dict[str, tuple[str, ...]] = {
    "side_table": ("sofa", "armchair", "bed"),
    "floor_lamp": ("sofa", "armchair", "desk", "bed"),
    "plant_large": ("sofa", "tv_console", "dresser", "shelf", "desk", "armchair", "side_table"),
    "plant_medium": ("sofa", "tv_console", "dresser", "shelf", "desk", "armchair", "side_table", "nightstand"),
    "plant_small": ("basket", "vase", "dresser", "plant_medium"),
    "basket": ("plant_medium", "plant_small"),
    "vase": ("basket", "plant_medium"),
    "nightstand": ("bed",),
}
# Peças pequenas que, sem parede ou canto disponível, podem ficar soltas no piso.
FREE_FALLBACK = {"plant_large", "plant_medium", "plant_small", "basket", "vase", "floor_lamp"}
# Decoração sobre móveis: papel -> possíveis apoios (em ordem de preferência).
DECOR_TARGETS: dict[str, tuple[str, ...]] = {
    "pillows": ("sofa", "armchair"), "throw": ("bed", "sofa"),
    "table_lamp": ("nightstand", "side_table", "desk", "dresser"),
    "books": ("coffee_table", "desk", "tv_console", "nightstand", "side_table", "dresser", "shelf"),
    "vase": ("coffee_table", "dining_table", "side_table", "dresser", "tv_console", "desk"),
    "bowl": ("coffee_table", "dining_table", "tv_console"),
    "kitchen_decor": ("dining_table",),
}
# Papéis que viram procedurais redimensionáveis (asset proc:<tipo>).
PROC_KIND = {"rug": "rug", "bed": "bed", "desk": "desk", "floor_lamp": "floor_lamp", "table_lamp": "table_lamp",
             "throw": "throw_blanket", "curtain": "curtain"}
GROUP_ROLES = {
    "conversa": {"sofa", "coffee_table", "armchair", "rug"},
    "cama": {"bed", "nightstand", "rug"},
    "jantar": {"dining_table", "dining_chair", "rug"},
    "copa": {"dining_table", "dining_chair"},
    "trabalho": {"desk", "rug"},
    "bancada": {"bar_stool", "dining_table", "dining_chair"},
}
ART_ABOVE = ("sofa", "bed", "tv_console", "desk", "dresser", "desk_storage", "shelf")
ART_MIN_VIEW_COS = 0.45        # cos do ângulo entre a normal da parede e a direção da câmera (~63°)
ART_PAIR_GAP = 0.08            # vão entre os dois quadros de um díptico (m)
ART_DEPTH = 0.035              # profundidade do quadro gerado (tela de 3,5 cm / moldura)
CHOREO_CLASS = {"rug": 0}
for _r in ("sofa", "bed", "dining_table", "desk", "tv_console", "dresser", "shelf", "shelf_large", "desk_storage"):
    CHOREO_CLASS[_r] = 1
for _r in ("armchair", "coffee_table", "side_table", "nightstand", "dining_chair", "bar_stool", "desk_chair",
           "ottoman"):
    CHOREO_CLASS[_r] = 2
for _r in ("floor_lamp", "table_lamp", "plant_large", "plant_medium", "plant_small", "basket"):
    CHOREO_CLASS[_r] = 3
for _r in ("wall_art", "wall_mirror", "curtain"):
    CHOREO_CLASS[_r] = 4



@dataclass
class SearchBudget:
    group_candidates: int = 3      # grupos principais levados até o fim
    group_steps: int = 70          # passos de têmpera no grupo isolado
    polish_steps: int = 100        # passos de têmpera com as peças secundárias
    max_slot_evals: int = 36       # candidatos avaliados com a pontuação completa por peça


class Solver:
    def __init__(self, ctx: LayoutContext, profile: StyleProfile, room_type: str, seed: int = 0,
                 budget: SearchBudget | None = None):
        if room_type not in catalog.ROOM_PROGRAMS:
            raise ValueError(f"Tipo de ambiente desconhecido: {room_type}")
        self.ctx = ctx
        self.profile = profile
        self.room_type = room_type
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.budget = budget or SearchBudget()
        self.cache = GeoCache(ctx)
        self.program = [(r, req, n) for r, req, n in catalog.ROOM_PROGRAMS[room_type]]
        self.prefs = {a.asset_id: asset_preference(a, profile) for a in catalog.CATALOG}
        self.slot_cache: dict[str, list] = {}

    # ------------------------------------------------------------------ utilidades p/ planners
    def spec(self, asset_id: str) -> AssetSpec:
        return catalog.get(asset_id)

    def in_program(self, role: str) -> bool:
        return any(r == role for r, _, _ in self.program)

    def ranked(self, role: str, k: int) -> list[AssetSpec]:
        return rank_assets(role, self.profile)[:k]

    def make(self, key: str, spec: AssetSpec, x: float, y: float, yaw: float, group: str = "",
             wall_id: int | None = None, role: str | None = None) -> Placement:
        w, d, h = spec.size
        return Placement(key, role or spec.role, spec.asset_id, float(w), float(d), float(h), float(x), float(y), 0.0,
                         float(yaw) % 360.0, "floor", wall_id, None, {}, group)

    def make_proc(self, key: str, role: str, size, x: float, y: float, yaw: float, group: str = "",
                  wall_id: int | None = None) -> Placement:
        w, d, h = (float(v) for v in size)
        return Placement(key, role, f"proc:{PROC_KIND[role]}", w, d, h, float(x), float(y), 0.0, float(yaw) % 360.0,
                         "floor", wall_id, None, proc_params(role, self.profile, (w, d, h)), group)

    def violations(self, items: list[Placement]) -> list[str]:
        geos = [self.cache.geo(p) for p in items]
        v = []
        for p, g in zip(items, geos):
            v += item_violations(p, g)
            if v:
                return v
        return pair_violations(items, geos)

    def score(self, items: list[Placement]) -> Score:
        return score_layout(items, self.cache, self.ctx, self.prefs)

    # ------------------------------------------------------------------ fluxo principal
    def solve(self) -> Layout:
        planner = self._planner()
        cands = planner.candidates(self) if planner is not None else []
        if not cands:
            cands = [GroupCand({}, [], 0.0)]
            planner = None
        results = []
        for gc in cands[: self.budget.group_candidates]:
            self.slot_cache = {}
            items, _ = self._refine_group(planner, gc)
            items = self._place_wall_art(items)
            items = self._place_secondaries(items)
            items = self._polish(items)
            results.append((self.score(items).total, items))
        results.sort(key=lambda t: -t[0])
        items = results[0][1]
        items = self._place_decor(items)
        return self._finalize(items)

    def _planner(self) -> GroupPlanner | None:
        if self.room_type == "cozinha":
            bp = BancadaPlanner()
            if bp.islands(self):
                return bp
            return CozinhaMesaPlanner()
        cls = PLANNERS.get(self.room_type)
        return cls() if cls is not None else None

    # ------------------------------------------------------------------ grupo principal
    def _refine_group(self, planner, gc: GroupCand) -> tuple[list[Placement], dict]:
        if planner is None or not gc.items:
            return list(gc.items), gc.params
        cur_p, cur_items = gc.params, gc.items
        cur = self.score(cur_items).total
        best = (cur, cur_items, cur_p)
        n = self.budget.group_steps
        for k in range(n):
            T = 0.6 * (0.03 / 0.6) ** (k / max(1, n - 1))
            prm = planner.mutate(self, cur_p, self.rng)
            items = planner.build(self, prm)
            if items is None:
                continue
            sc = self.score(items)
            if not sc.feasible:
                continue
            if sc.total >= cur or self.rng.random() < math.exp((sc.total - cur) / T):
                cur, cur_items, cur_p = sc.total, items, prm
                if cur > best[0]:
                    best = (cur, cur_items, cur_p)
        return list(best[1]), best[2]

    # ------------------------------------------------------------------ peças secundárias
    def _secondary_roles(self, items: list[Placement]) -> list[tuple[str, int]]:
        groups = {p.group for p in items}
        handled = set()
        for g in groups:
            handled |= GROUP_ROLES.get(g, set())
        if self.room_type == "cozinha":
            handled |= {"bar_stool", "dining_table", "dining_chair"}
        out = []
        for role, req, n in self.program:
            if role in handled or role in WALL_ROLES or role == "rug":
                continue
            if role in ON_TOP_ROLES and not (role == "vase" and self.room_type == "banheiro"):
                continue
            out.append((role, n))
        return out

    def _slots_for(self, role: str, spec: AssetSpec, items: list[Placement]) -> list[Slot]:
        w, d, _ = spec.size
        kinds = SLOT_KINDS.get(role, ("parede", "canto"))
        out: list[Slot] = []
        if "lado" in kinds:
            for a in items:
                if a.role in ADJ_ANCHORS.get(role, ()) and a.mount == "floor":
                    gap = SIDE_GAP if role in ("side_table", "nightstand") else 0.08
                    out += adjacent_slots(a, w, d, gap=gap, align=("fundo",) if a.role != "armchair" else ("fundo", "centro"))
        if "canto" in kinds:
            out += corner_slots(self.ctx, w, d, diagonal=role in ("armchair", "plant_large", "plant_medium", "floor_lamp"))
        if "parede" in kinds:
            out += wall_slots(self.ctx, w, d, gap=WALL_GAP, step=0.1)
        if "livre" in kinds:
            out += free_slots(self.ctx, step=0.3)
        return out

    def _try_slots(self, role: str, spec: AssetSpec, key: str, items: list[Placement],
                   slots: list[Slot]) -> list[tuple[Placement, Slot]]:
        """Candidatos viáveis (restrições rígidas isoladas + sem sobreposição)."""
        ok = []
        others = [p for p in items if p.mount == "floor"]
        ogeos = [self.cache.geo(q) for q in others]
        for sl in slots:
            p = self.make(key, spec, sl.x, sl.y, sl.yaw, wall_id=sl.wall_id, role=role)
            if spec.scalable:
                p.params = proc_params(role, self.profile, spec.size)
            g = self.cache.geo(p)
            if item_violations(p, g):
                continue
            if self._overlaps(p, g, others, ogeos):
                continue
            p.slot = sl.kind
            ok.append((p, sl))
        return ok

    def _overlaps(self, p: Placement, g, others: list[Placement], ogeos: list) -> bool:
        return bool(pair_violations([p], [g], others, ogeos))

    def _place_secondaries(self, items: list[Placement]) -> list[Placement]:
        items = list(items)
        for role, n in self._secondary_roles(items):
            specs = self.ranked(role, 2)
            if not specs:
                continue
            for i in range(n):
                key = f"{role}_{sum(1 for p in items if p.role == role)}"
                base = self.score(items).total
                best = None
                all_cands = []
                for spec in specs:
                    cands = self._try_slots(role, spec, key, items, self._slots_for(role, spec, items))
                    if not cands and role in FREE_FALLBACK:     # sem parede/canto viável: piso livre
                        cands = self._try_slots(role, spec, key, items, free_slots(self.ctx, step=0.3))
                    all_cands += cands
                    if len(cands) > self.budget.max_slot_evals:
                        pri = [c for c in cands if c[1].kind in ("lado", "canto")]
                        rest = [c for c in cands if c[1].kind not in ("lado", "canto")]
                        stride = max(1, len(rest) // max(1, self.budget.max_slot_evals - len(pri)))
                        cands = pri + rest[::stride]
                    for p, sl in cands:
                        sc = self.score(items + [p])
                        if sc.feasible and (best is None or sc.total > best[0]):
                            best = (sc.total, p, sl)
                if best is None or best[0] <= base:
                    break
                _, p, sl = best
                items.append(p)
                self.slot_cache[p.key] = [(q, t) for q, t in all_cands if q.asset_id == p.asset_id]
        return items

    def _reslot(self, items: list[Placement], key: str) -> list[Placement] | None:
        """Move a peça ``key`` para outro encaixe (da lista calculada quando ela foi posicionada)."""
        cands = self.slot_cache.get(key) or []
        if not cands:
            return None
        others = [q for q in items if q.key != key]
        fl = [q for q in others if q.mount == "floor"]
        fg = [self.cache.geo(q) for q in fl]
        for _ in range(6):
            p, sl = cands[int(self.rng.integers(len(cands)))]
            if self._overlaps(p, self.cache.geo(p), fl, fg):
                continue
            return others + [p]
        return None

    def _polish(self, items: list[Placement]) -> list[Placement]:
        """Têmpera simulada sobre as peças secundárias (o grupo e os quadros ficam fixos)."""
        cur_items = items
        cur = self.score(cur_items).total
        best = (cur, cur_items)
        sec_keys = [p.key for p in items if not p.group and p.mount == "floor"]
        if not sec_keys:
            return items
        n = self.budget.polish_steps
        for k in range(n):
            T = 0.5 * (0.02 / 0.5) ** (k / max(1, n - 1))
            key = sec_keys[int(self.rng.integers(len(sec_keys)))]
            if self.rng.random() < 0.35:
                cand = self._reslot(cur_items, key)
            else:
                cand = []
                for p in cur_items:
                    if p.key == key:
                        if p.wall_id is not None and p.slot == "parede":
                            w = self.ctx.wall_by_id(p.wall_id)
                            s = float((p.xy - w.p0) @ w.direction) + float(self.rng.normal(0, 0.12))
                            x, y, yaw = wall_pose(w, s, p.d, WALL_GAP)
                            p = p.moved(x=x, y=y, yaw=yaw)
                        else:
                            p = p.moved(x=p.x + float(self.rng.normal(0, 0.07)),
                                        y=p.y + float(self.rng.normal(0, 0.07)))
                    cand.append(p)
            if not cand:
                continue
            sc = self.score(cand)
            if not sc.feasible:
                continue
            if sc.total >= cur or self.rng.random() < math.exp((sc.total - cur) / T):
                cur, cur_items = sc.total, cand
                if cur > best[0] + 1e-9:
                    best = (cur, cur_items)
        return list(best[1])

    # ------------------------------------------------------------------ quadros e espelho
    def _art_sets(self, role: str, below: Placement | None) -> list[list[tuple[AssetSpec, tuple]]]:
        """Arranjos candidatos (lista de (asset, tamanho)) para um quadro/espelho.

        Assets de tamanho fixo entram sozinhos (ou em par de dois DIFERENTES: duas cópias da mesma
        pintura denunciam o CG); o quadro gerado (``scalable``) recebe tamanhos pelas regras de
        projeto de ``_art_sizes``, inclusive em díptico."""
        out = []
        specs = self.ranked(role, 2)
        for s in specs:
            if s.scalable:
                out += [[(s, sz) for sz in sizes] for sizes in self._art_sizes(below)]
            else:
                out.append([(s, tuple(s.size))])
        fixed = [s for s in specs if not s.scalable]
        if role == "wall_art" and len(fixed) == 2:
            out.append([(fixed[0], tuple(fixed[0].size)), (fixed[1], tuple(fixed[1].size))])
        return out

    def _art_sizes(self, below: Placement | None) -> list[list[tuple[float, float, float]]]:
        """Tamanhos do quadro gerado pelas regras de projeto (largura x profundidade x altura, m).

        - sobre um móvel: o conjunto ocupa ~2/3 da largura do móvel (0,5–1,8 m): um quadro
          paisagem (4:3; 3:2 a partir de 1,2 m) ou um díptico de retratos 3:4 com 8 cm de vão;
          a altura cabe entre o móvel + ``ART_ABOVE_GAP`` e o teto − 25 cm;
        - parede livre: retrato 60 x 80 cm, paisagem 90 x 65 cm ou díptico 45 x 60 cm.
        """
        D = ART_DEPTH
        r2 = lambda v: round(float(v), 2)   # noqa: E731 — tamanhos em centímetros inteiros
        if below is None:
            return [[(0.6, D, 0.8)], [(0.9, D, 0.65)], [(0.45, D, 0.6), (0.45, D, 0.6)]]
        tw = float(np.clip(below.w * 2.0 / 3.0, 0.5, 1.8))
        hmax = self.ctx.ceiling - 0.25 - (below.h + ART_ABOVE_GAP)
        if hmax < 0.3:
            return []
        out = []
        w1 = tw
        h1 = w1 / (1.5 if w1 >= 1.2 else 4.0 / 3.0)
        k = min(1.0, hmax / h1, 1.0 / h1)
        out.append([(r2(w1 * k), D, r2(h1 * k))])
        w2 = (tw - ART_PAIR_GAP) / 2
        if w2 >= 0.3:
            h2 = w2 * 4.0 / 3.0
            k = min(1.0, hmax / h2, 1.0 / h2)
            out.append([(r2(w2 * k), D, r2(h2 * k))] * 2)
        return out

    def _place_wall_art(self, items: list[Placement]) -> list[Placement]:
        roles = [r for r, _, _ in self.program if r in WALL_ROLES]
        for role in roles:
            items = self._place_one_art(items, role)
        return items

    def _art_placements(self, role: str, arr: list[tuple[AssetSpec, tuple]], w, s: float,
                        zc: float) -> list[Placement]:
        gapx = ART_PAIR_GAP if len(arr) > 1 else 0.0
        tot_w = sum(sz[0] for _, sz in arr) + gapx * (len(arr) - 1)
        ah = max(sz[2] for _, sz in arr)
        arts = []
        x0 = s - tot_w / 2
        style_seed = self.seed * 7 + int(round(tot_w * 100))
        for i, (a, (aw, ad, ahh)) in enumerate(arr):
            sc_ = x0 + aw / 2
            x0 += aw + gapx
            c = w.point(sc_) + w.normal * (ad / 2 + 0.005)
            params = {}
            if a.scalable:
                params = proc_params(role, self.profile, (aw, ad, ahh), seed=self.seed * 1009 + 17 * i + 1,
                                     style_seed=style_seed)
            arts.append(Placement(f"{role}_{i}", role, a.asset_id, aw, ad, ahh, float(c[0]), float(c[1]),
                                  float(zc - ah / 2 + (ah - ahh) / 2), geom.yaw_facing(*w.normal), "wall", w.id,
                                  None, params, ""))
        return arts

    def _place_one_art(self, items: list[Placement], role: str) -> list[Placement]:
        ctx = self.ctx
        base_items = list(items)
        base = self.score(base_items).total
        best = None
        below_cands = []
        for p in items:
            if p.role in ART_ABOVE and p.mount == "floor":
                frac, wid = ctx.wall_contact(self.cache.geo(p).poly, -p.front, tol=0.12)
                if wid is not None and frac > 0.5:
                    w = ctx.wall_by_id(wid)
                    below_cands.append((p, w, float((p.xy - w.p0) @ w.direction)))
        blocked_near = self._fixture_walls() if role == "wall_mirror" else {}
        options = []          # (arranjo, parede, s, z do centro, bônus)
        for p, w, s in below_cands:
            for arr in self._art_sets(role, p):
                tot_w = sum(sz[0] for _, sz in arr) + ART_PAIR_GAP * (len(arr) - 1)
                ah = max(sz[2] for _, sz in arr)
                ratio = tot_w / p.w
                # base 20–25 cm acima do móvel; o centro não passa da altura dos olhos sem precisar
                zc = max(min(ART_CENTER_Z, p.h + ART_ABOVE_GAP + 0.05 + ah / 2), p.h + ART_ABOVE_GAP + ah / 2)
                options.append((arr, w, s, zc, 3.0 - 4.0 * abs(ratio - 2.0 / 3.0)))
        for arr in self._art_sets(role, None):
            tot_w = sum(sz[0] for _, sz in arr) + ART_PAIR_GAP * (len(arr) - 1)
            for w in ctx.walls:
                for s0, s1 in w.intervals(w.vis_art & (w.view_cos > ART_MIN_VIEW_COS), min_len=tot_w + 0.2):
                    for s in np.arange(s0 + tot_w / 2 + 0.1, s1 - tot_w / 2 - 0.1 + 1e-9, 0.2):
                        bonus = 0.0
                        if role == "wall_mirror" and w.id in blocked_near:
                            bonus = 1.5 if any(a <= s <= b for a, b in blocked_near[w.id]) else 0.0
                        options.append((arr, w, float(s), ART_CENTER_Z, bonus))
        for arr, w, s, zc, bonus in options:
            ah = max(sz[2] for _, sz in arr)
            if zc + ah / 2 > ctx.ceiling - 0.12:
                continue
            arts = self._art_placements(role, arr, w, s, zc)
            if not self._art_ok(arts, w, base_items):
                continue
            cand = base_items + arts
            sc = self.score(cand)
            if not sc.feasible:
                continue
            total = sc.total + bonus
            if best is None or total > best[0]:
                best = (total, cand)
        if best is None or best[0] <= base:
            return items
        return best[1]

    def _art_ok(self, arts: list[Placement], w, items: list[Placement]) -> bool:
        for a in arts:
            s = float((a.xy - w.p0) @ w.direction)
            if s - a.w / 2 < 0.1 or s + a.w / 2 > w.length - 0.1:
                return False
            if not w.window_free(s - a.w / 2, s + a.w / 2, a.z, a.z + a.h, margin=0.1):
                return False
            i = int(np.clip(np.searchsorted(w.s, s), 0, len(w.s) - 1))
            if w.view_cos[i] < ART_MIN_VIEW_COS:      # parede vista muito de lado: quadro vira um risco
                return False
        return True

    def _fixture_walls(self) -> dict[int, list[tuple[float, float]]]:
        """Trechos de parede com louças/bancadas (BLOCKED) logo à frente — bons para espelho."""
        out: dict[int, list[tuple[float, float]]] = {}
        for w in self.ctx.walls:
            hits = []
            for s in w.s:
                pts = np.array([w.point(s) + w.normal * dd for dd in (0.15, 0.3, 0.45)])
                if np.any(self.ctx.states(pts) == BLOCKED):
                    hits.append(float(s))
            if hits:
                out[w.id] = [(min(hits), max(hits))]
        return out

    # ------------------------------------------------------------------ decoração sobre móveis
    def _place_decor(self, items: list[Placement]) -> list[Placement]:
        """Decoração sobre os móveis: almofadas no assento, abajures nos criados-mudos, composição
        de objetos (livros, fruteira, vaso) espalhada sobre mesas — sempre na altura da superfície."""
        items = list(items)
        used: dict[str, list[tuple[float, float, float, float]]] = {}
        requests = []
        for role, req, n in self.program:
            if role not in DECOR_TARGETS or (role == "vase" and self.room_type == "banheiro"):
                continue
            if role in PROC_KIND:
                specs = catalog.by_role(role)
                if not specs:
                    continue
                size, asset_id = specs[0].size, f"proc:{PROC_KIND[role]}"
            elif role in catalog.DECOR_BY_PARENT:      # asset escolhido pelo apoio (abaixo)
                asset_id = catalog.decor_asset(role, "*")
                size = catalog.get(asset_id).size
            else:
                specs = self.ranked(role, 1)
                if not specs:
                    continue
                size, asset_id = specs[0].size, specs[0].asset_id
            requests.append((role, n, tuple(float(v) for v in size), asset_id))
        # objetos maiores primeiro: os menores se encaixam nos espaços que sobram
        requests.sort(key=lambda r: -(r[2][0] * r[2][1]))
        for role, n, size, asset_id in requests:
            placed = 0
            for target_role in DECOR_TARGETS[role]:
                parents = sorted([p for p in items if p.role == target_role and p.mount == "floor"], key=lambda q: q.key)
                for par in parents:
                    if placed >= n:
                        break
                    if role == "table_lamp" and any(q.parent == par.key and q.role == "table_lamp" for q in items):
                        continue
                    sup = support_for(par.asset_id, par.role, (par.w, par.d, par.h), par.params)
                    if sup is None:
                        continue
                    aid = catalog.decor_asset(role, par.role) or asset_id
                    sz = size if aid == asset_id else tuple(float(v) for v in catalog.get(aid).size)
                    spot = self._decor_spot(role, par, sup, sz, used.setdefault(par.key, []))
                    if spot is None:
                        continue
                    lx, ly = spot
                    c = geom.local_to_world(par.x, par.y, par.yaw, lx, ly)
                    w_, d_, h_ = sz
                    key = f"{role}_{sum(1 for p in items if p.role == role)}"
                    p = Placement(key, role, aid, w_, d_, h_, float(c[0]), float(c[1]), float(par.z + sup.z),
                                  par.yaw, "on_top", None, par.key,
                                  proc_params(role, self.profile, sz) if aid.startswith("proc:") else {}, "")
                    if item_violations(p, self.cache.geo(p)):
                        used[par.key].pop()
                        continue
                    items.append(p)
                    placed += 1
                if placed >= n:
                    break
        return items

    def _decor_spot(self, role: str, par: Placement, sup: Support, size, used: list) -> tuple[float, float] | None:
        w, d, _ = (float(v) for v in size)
        x0, y0, x1, y1 = max(sup.rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rd = y1 - y0
        if sup.kind == "seat" and w > (x1 - x0) + 0.02:
            return None     # ex.: jogo de almofadas mais largo que o assento da poltrona (sobraria no ar)
        if sup.kind == "seat":
            # almofadas: centradas e encostadas no encosto; manta: no pé da cama
            if role == "throw" and par.role == "bed":
                spots = [(0.0, y0 + d / 2 + 0.18)]
            else:
                dep = min(d, 0.8 * rd)
                spots = [(cx, y1 - dep / 2)]
        elif role == "table_lamp":
            spots = [(cx, cy + min(0.05, max(0.0, (rd - d) / 2 - ON_TOP_INSET))), (cx, cy)]
        else:
            # grade de posições no tampo (recuada ON_TOP_INSET da borda); escolhe a que mais se
            # afasta dos objetos já colocados
            m = ON_TOP_INSET
            xs = np.linspace(x0 + m + w / 2, x1 - m - w / 2, 7) if x1 - x0 > w + 2 * m else np.array([cx])
            ys = np.linspace(y0 + m + d / 2, y1 - m - d / 2, 3) if y1 - y0 > d + 2 * m else np.array([cy])
            cand = [(float(a), float(b)) for b in ys for a in xs]
            if not used:
                cand.sort(key=lambda q: (abs(q[0] - (cx - 0.22 * (x1 - x0))) + abs(q[1] - cy), q))
            else:
                ctrs = [((u[0] + u[2]) / 2, (u[1] + u[3]) / 2) for u in used]
                cand.sort(key=lambda q: (-min(math.hypot(q[0] - a, q[1] - b) for a, b in ctrs), q))
            spots = cand
        for lx, ly in spots:
            if sup.kind == "top" and not fits_on(sup, lx, ly, w, d, margin=-ON_TOP_INSET + 1e-6):
                continue
            if role == "throw" and par.role == "bed" and not fits_on(sup, lx, ly, w, d):
                continue
            box = (lx - w / 2, ly - d / 2, lx + w / 2, ly + d / 2)
            if any(not (box[2] <= u[0] + 0.03 or u[2] <= box[0] + 0.03 or box[3] <= u[1] + 0.03 or u[3] <= box[1] + 0.03)
                   for u in used):
                continue
            used.append(box)
            return lx, ly
        return None

    # ------------------------------------------------------------------ finalização
    def _finalize(self, items: list[Placement]) -> Layout:
        ctx = self.ctx
        dist = {p.key: float(np.linalg.norm(p.xy - ctx.cam_xy)) for p in items}

        def key(p: Placement):
            cls = 5 if p.mount == "on_top" and p.role != "table_lamp" else CHOREO_CLASS.get(p.role, 3)
            if p.role == "table_lamp":
                cls = 3
            return (cls, -dist[p.key], p.key)

        ordered = sorted(items, key=key)
        # a decoração nunca surge antes do seu apoio
        pos = {p.key: i for i, p in enumerate(ordered)}
        for p in list(ordered):
            if p.parent and pos.get(p.parent, -1) > pos[p.key]:
                ordered.remove(p)
                ordered.insert(pos[p.parent] + 1, p)
                pos = {q.key: i for i, q in enumerate(ordered)}
        placed = [p.to_item(i) for i, p in enumerate(ordered)]
        # entrada de cada peça na animação (PT11): o deslize só com caminho varrido livre
        entr = plan_entrances(ordered, ctx)
        for it in placed:
            e = entr.get(it.id, {})
            it.motion = str(e.get("motion", ""))
            if it.motion == "glide":
                it.motion_dir = list(e["motion_dir"])
                it.motion_dist = float(e["motion_dist"])
        sc = self.score(items)
        layout = Layout(room_type=self.room_type, style=self.profile.style, palette=self.profile.palette,
                        items=placed, score=round(sc.total, 3), notes=[])
        missing = []
        for role, req, n in self.program:
            got = sum(1 for p in items if p.role == role)
            if got == 0:
                missing.append(ROLE_LABELS.get(role, role) + (" (essencial)" if req else ""))
        # narrativa de designer (PT12): as escolhas explicadas, exibida na interface
        layout.notes = designer_notes(layout, ctx, self.profile, missing, validate_layout(layout, ctx, self.cache))
        return layout


def plan_layout(room: RoomModel, grid: np.ndarray, room_type: str, photo_rgb8: np.ndarray, seed: int = 0,
                seg=None) -> Layout:
    """Escolhe e posiciona os móveis para a foto (determinístico para uma mesma semente)."""
    ctx = LayoutContext(room, grid)
    profile = analyze_style(photo_rgb8, seg, ctx)
    return Solver(ctx, profile, room_type, seed).solve()
