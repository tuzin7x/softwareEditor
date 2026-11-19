"""Planejadores por tipo de ambiente: o "grupo principal" de cada cômodo.

Cada planejador descreve o grupo funcional por PARÂMETROS (onde fica a âncora, folgas,
ângulos, tamanho do tapete...) e sabe:
- ``candidates``: enumerar configurações iniciais (já filtradas e ordenadas pela pontuação);
- ``build``: transformar parâmetros em peças posicionadas (relações garantidas por construção:
  mesa de centro centrada a ~45 cm do sofá, criados-mudos ladeando a cama, cadeiras voltadas
  para a mesa...); devolve None se alguma restrição rígida falhar;
- ``mutate``: perturbar os parâmetros (busca local / têmpera simulada).

Grupos: sala -> conversa (sofá, mesa de centro, poltronas, tapete); quarto -> cama (cama,
criados-mudos, tapete); sala de jantar -> jantar (mesa, cadeiras, tapete); escritório ->
trabalho (escrivaninha, cadeira, tapete); cozinha -> bancada (banquetas na ilha) ou mesa com
cadeiras; banheiro -> sem grupo (só decoração junto às paredes).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from ..analysis.room import BLOCKED, OUTSIDE
from . import geom
from .rules import COFFEE_GAP, RUG_SIZES, SEAT_TABLE_GAP, SIDE_GAP, WALL_GAP
from .slots import free_slots, wall_pose
from .state import Placement

BED_SIZES = ((1.68, 2.12, 1.00), (1.48, 2.02, 0.95), (1.98, 2.12, 1.05), (1.08, 2.02, 0.90))  # queen, casal, king, solteiro
BED_FALLBACK = 3          # solteiro só quando nenhuma cama de casal cabe no espaço visível
DESK_SIZES = ((1.40, 0.70, 0.75), (1.20, 0.60, 0.75), (1.60, 0.75, 0.75))
# Configurações de poltronas ao redor da mesa de centro (ângulo polar no referencial do sofá:
# 0° = lateral +X do sofá, 90° = em frente ao sofá, 180° = lateral -X).
ARM_CONFIGS = ((), (0.0,), (180.0,), (-20.0,), (200.0,), (0.0, 180.0), (-20.0, 200.0), (90.0,),
               (55.0, 125.0), (35.0,), (145.0,), (0.0, 125.0), (180.0, 55.0))
RUG_TRIALS = ((1, 0.0), (0, 0.0), (1, 0.3), (1, -0.3), (0, 0.3), (0, -0.3), (2, 0.0))


@dataclass
class GroupCand:
    params: dict
    items: list[Placement]
    score: float


def _anchor_list(S, w: float, d: float, *, allow_free: bool, wall_step: float = 0.1,
                 overhang_frac: float = 0.15, free_step: float = 0.3, min_face: float | None = 0.3) -> list[tuple]:
    out: list[tuple] = []
    for wall in S.ctx.walls:
        ov = overhang_frac * w
        lo, hi = w / 2 - ov, wall.length - w / 2 + ov
        if hi < lo:
            if wall.length < 0.6 * w:
                continue
            lo = hi = wall.length / 2
        n = int(math.floor((hi - lo) / wall_step + 1e-9)) + 1
        for k in range(n):
            out.append(("wall", wall.id, round(lo + k * wall_step, 3)))
    if allow_free:
        for sl in free_slots(S.ctx, step=free_step, yaws=(0.0, 90.0, 180.0, 270.0), face_camera=False):
            if min_face is not None:
                f = geom.front_of(sl.yaw) @ S.ctx.dir_to_camera(np.array([sl.x, sl.y]))
                if f < min_face:
                    continue
            out.append(("free", round(sl.x, 3), round(sl.y, 3), sl.yaw))
    return out


def _anchor_pose(S, anchor: tuple, d: float, gap: float = WALL_GAP) -> tuple[float, float, float, int | None]:
    if anchor[0] == "wall":
        wall = S.ctx.wall_by_id(anchor[1])
        x, y, yaw = wall_pose(wall, anchor[2], d, gap)
        return x, y, yaw, wall.id
    return float(anchor[1]), float(anchor[2]), float(anchor[3]), None


def _nms(cands: list[tuple[float, dict, Placement]], k: int, min_dist: float = 0.35) -> list[tuple[float, dict]]:
    """Mantém os k melhores candidatos espacialmente distintos (supressão de não-máximos)."""
    cands = sorted(cands, key=lambda t: -t[0])
    kept: list[tuple[float, dict, Placement]] = []
    for sc, prm, p in cands:
        dup = any(np.hypot(p.x - q.x, p.y - q.y) < min_dist and abs(geom.wrap_deg(p.yaw - q.yaw)) < 10
                  for _, _, q in kept)
        if not dup:
            kept.append((sc, prm, p))
        if len(kept) >= k:
            break
    return [(sc, prm) for sc, prm, _ in kept]


def _mutate_anchor(params: dict, rng, sigma_s: float = 0.12, sigma_xy: float = 0.1) -> None:
    a = params["anchor"]
    if a[0] == "wall":
        params["anchor"] = ("wall", a[1], round(a[2] + float(rng.normal(0, sigma_s)), 3))
    else:
        params["anchor"] = ("free", round(a[1] + float(rng.normal(0, sigma_xy)), 3),
                            round(a[2] + float(rng.normal(0, sigma_xy)), 3), a[3])


def _rect_extent(w: float, d: float, e_local: np.ndarray) -> float:
    """Distância do centro à borda de um retângulo w x d ao longo da direção local unitária e."""
    ex, ey = abs(e_local[0]), abs(e_local[1])
    tx = (w / 2) / ex if ex > 1e-9 else math.inf
    ty = (d / 2) / ey if ey > 1e-9 else math.inf
    return min(tx, ty)


class GroupPlanner:
    name = ""
    keep = 4

    def candidates(self, S) -> list[GroupCand]:
        return [GroupCand({}, [], 0.0)]

    def build(self, S, params: dict) -> list[Placement] | None:
        return []

    def mutate(self, S, params: dict, rng) -> dict:
        return copy.deepcopy(params)

    # utilidades -----------------------------------------------------------------------
    @staticmethod
    def finish(S, items: list[Placement]) -> list[Placement] | None:
        """None se alguma peça violar restrições rígidas isoladas ou houver sobreposição."""
        if S.violations(items):
            return None
        return items


# ============================================================================ sala de estar
class SalaPlanner(GroupPlanner):
    name = "conversa"

    def _quick_rug(self, S, prm: dict) -> tuple[dict, list[Placement]] | None:
        """Primeiro tapete viável numa lista curta de tamanhos/deslocamentos laterais."""
        for ri, lat in RUG_TRIALS:
            q = dict(prm, rug=ri, rug_lat=lat)
            items = self.build(S, q)
            if items is not None:
                return q, items
        return None

    def _best_rug(self, S, prm: dict) -> list[tuple[float, dict]]:
        out = []
        q = dict(prm, rug=None)
        items = self.build(S, q)
        if items is not None:
            out.append((S.score(items).total, q))
        for ri in range(len(RUG_SIZES)):
            for rin in (0.15, 0.25):
                for lat in (0.0, -0.3, 0.3):
                    q = dict(prm, rug=ri, rug_in=rin, rug_lat=lat)
                    items = self.build(S, q)
                    if items is not None:
                        out.append((S.score(items).total, q))
        return out

    def candidates(self, S) -> list[GroupCand]:
        sofas = S.ranked("sofa", 3)
        cts = S.ranked("coffee_table", 3)
        arm = S.ranked("armchair", 1)
        base_cands = []
        for sofa in sofas:
            w, d, _ = sofa.size
            for anc in _anchor_list(S, w, d, allow_free=True):
                for ct in cts:          # primeira mesa de centro que couber (ordem de preferência)
                    prm = dict(sofa=sofa.asset_id, anchor=anc, ct=ct.asset_id, ct_gap=COFFEE_GAP, arms=[],
                               rug=None, rug_in=0.2, rug_lat=0.0)
                    items = self.build(S, prm)
                    if items is None:
                        prm["ct_gap"] = 0.38
                        items = self.build(S, prm)
                    if items is not None:
                        base_cands.append((S.score(items).total, prm, items[0]))
                        break
        if not base_cands:      # nem sofá + mesa cabem: sofá sozinho (a mesa fica de fora)
            for sofa in sofas:
                w, d, _ = sofa.size
                for anc in _anchor_list(S, w, d, allow_free=True):
                    prm = dict(sofa=sofa.asset_id, anchor=anc, ct=None, ct_gap=COFFEE_GAP, arms=[], rug=None,
                               rug_in=0.2, rug_lat=0.0)
                    items = self.build(S, prm)
                    if items is not None:
                        base_cands.append((S.score(items).total, prm, items[0]))
        anchors = _nms(base_cands, 6)
        out: list[tuple[float, dict]] = []
        for _, base in anchors:
            ct_opts = [base["ct"]] + [c.asset_id for c in cts if c.asset_id != base["ct"]][:1] if base["ct"] else [None]
            for ct in ct_opts:
                trials = []
                for cfg in ARM_CONFIGS:
                    if cfg and (not arm or ct is None):
                        continue
                    prm = dict(base, ct=ct,
                               arms=[dict(asset=arm[0].asset_id, theta=t, gap=SEAT_TABLE_GAP, twist=0.0) for t in cfg])
                    r = self._quick_rug(S, prm)
                    if r is None:
                        items = self.build(S, dict(prm, rug=None))
                        if items is None:
                            continue
                        prm = dict(prm, rug=None)
                    else:
                        prm, items = r
                    trials.append((S.score(items).total, prm))
                trials.sort(key=lambda t: -t[0])
                kept_counts: dict[int, int] = {}
                for _, prm in trials:
                    n = len(prm["arms"])
                    if kept_counts.get(n, 0) >= 1 or sum(kept_counts.values()) >= 2:
                        continue
                    kept_counts[n] = kept_counts.get(n, 0) + 1
                    out += self._best_rug(S, prm)
        out.sort(key=lambda t: -t[0])
        res, seen = [], set()
        for sc, prm in out:
            sig = (prm["sofa"], prm["anchor"], len(prm["arms"]))
            if sig in seen:
                continue
            seen.add(sig)
            res.append(GroupCand(prm, self.build(S, prm), sc))
            if len(res) >= self.keep:
                break
        return res

    def build(self, S, prm: dict) -> list[Placement] | None:
        sofa_spec = S.spec(prm["sofa"])
        w, d, h = sofa_spec.size
        x, y, yaw, wid = _anchor_pose(S, prm["anchor"], d)
        sofa = S.make("sofa_0", sofa_spec, x, y, yaw, group=self.name, wall_id=wid)
        f, u = sofa.front, sofa.lateral
        items = [sofa]
        if prm["ct"] is None:
            return self.finish(S, items)
        ct_spec = S.spec(prm["ct"])
        cw, cd, _ = ct_spec.size
        cc = sofa.xy + f * (d / 2 + prm["ct_gap"] + cd / 2)
        ct = S.make("coffee_table_0", ct_spec, cc[0], cc[1], yaw, group=self.name)
        items.append(ct)
        for i, a in enumerate(prm["arms"]):
            spec = S.spec(a["asset"])
            aw, ad, _ = spec.size
            th = math.radians(a["theta"])
            e_loc = np.array([math.cos(th), math.sin(th)])            # (u, f)
            e = u * e_loc[0] + f * e_loc[1]
            r = _rect_extent(cw, cd, e_loc) + a["gap"] + ad / 2
            ac = cc + e * r
            ayaw = geom.yaw_facing(*(-e)) + a["twist"]
            items.append(S.make(f"armchair_{i}", spec, ac[0], ac[1], ayaw, group=self.name))
        if prm["rug"] is not None:
            L, Wd = RUG_SIZES[prm["rug"]]
            rc = sofa.xy + f * (d / 2 - prm["rug_in"] + Wd / 2) + u * prm["rug_lat"]
            items.append(S.make_proc("rug_0", "rug", (L, Wd, 0.012), rc[0], rc[1], yaw, group=self.name))
        return self.finish(S, items)

    def mutate(self, S, prm: dict, rng) -> dict:
        p = copy.deepcopy(prm)
        r = rng.random()
        if r < 0.3:
            _mutate_anchor(p, rng)
        elif r < 0.45:
            p["ct_gap"] = float(np.clip(p["ct_gap"] + rng.normal(0, 0.03), 0.38, 0.52))
        elif r < 0.75 and p["arms"]:
            a = p["arms"][int(rng.integers(len(p["arms"])))]
            a["theta"] = float(a["theta"] + rng.normal(0, 8))
            a["gap"] = float(np.clip(a["gap"] + rng.normal(0, 0.05), 0.35, 0.65))
            a["twist"] = float(np.clip(a["twist"] + rng.normal(0, 6), -20, 20))
        elif r < 0.85 and len(S.ranked("armchair", 3)) > 1 and p["arms"]:
            opts = [s.asset_id for s in S.ranked("armchair", 3)]
            new = opts[int(rng.integers(len(opts)))]
            for a in p["arms"]:
                a["asset"] = new
        else:
            if p["rug"] is not None:
                p["rug_in"] = float(np.clip(p["rug_in"] + rng.normal(0, 0.04), 0.08, 0.35))
                p["rug_lat"] = float(np.clip(p["rug_lat"] + rng.normal(0, 0.1), -0.5, 0.5))
                if rng.random() < 0.3:
                    p["rug"] = int(np.clip(p["rug"] + rng.choice([-1, 1]), 0, len(RUG_SIZES) - 1))
        return p


# ============================================================================ quarto
class QuartoPlanner(GroupPlanner):
    name = "cama"

    def candidates(self, S) -> list[GroupCand]:
        ns = S.ranked("nightstand", 1)
        base = []
        for bi, (w, d, h) in enumerate(BED_SIZES):
            if bi >= BED_FALLBACK and base:
                break
            anchors = _anchor_list(S, w, d, allow_free=False, overhang_frac=0.0)
            if not S.ctx.walls:     # nenhuma parede detectada: cama solta, com os pés para a câmera
                anchors = _anchor_list(S, w, d, allow_free=True, overhang_frac=0.0, min_face=0.3)
            for anc in anchors:
                prm = dict(bed=bi, anchor=anc, ns=ns[0].asset_id if ns else None, sides=(-1, 1), ns_gap=SIDE_GAP,
                           rug=None, rug_head=0.7, rug_lat=0.0)
                items = self.build(S, prm)
                if items is None:
                    prm["sides"] = (1,)
                    items = self.build(S, prm)
                if items is None:
                    prm["sides"] = (-1,)
                    items = self.build(S, prm)
                if items is None:
                    prm["sides"] = ()
                    items = self.build(S, prm)
                if items is None:
                    continue
                base.append((S.score(items).total, prm, items[0]))
        anchors = _nms(base, 8)
        out = []
        for _, prm in anchors:
            for ri in (None, 0, 1, 2):
                for head in (0.6, 0.8):
                    q = dict(prm, rug=ri, rug_head=head)
                    items = self.build(S, q)
                    if items is not None:
                        out.append((S.score(items).total, q))
                    if ri is None:
                        break
        out.sort(key=lambda t: -t[0])
        res, seen = [], set()
        for sc, prm in out:
            sig = (prm["bed"], prm["anchor"])
            if sig in seen:
                continue
            seen.add(sig)
            res.append(GroupCand(prm, self.build(S, prm), sc))
            if len(res) >= self.keep:
                break
        return res

    def build(self, S, prm: dict) -> list[Placement] | None:
        w, d, h = BED_SIZES[prm["bed"]]
        x, y, yaw, wid = _anchor_pose(S, prm["anchor"], d, gap=0.02)
        bed = S.make_proc("bed_0", "bed", (w, d, h), x, y, yaw, group=self.name, wall_id=wid)
        f, u = bed.front, bed.lateral
        items = [bed]
        if prm["ns"]:
            spec = S.spec(prm["ns"])
            nw, nd, _ = spec.size
            for i, sg in enumerate(sorted(prm["sides"])):
                c = bed.xy + u * sg * (w / 2 + prm["ns_gap"] + nw / 2) + f * (-(d / 2) + nd / 2)
                items.append(S.make(f"nightstand_{i}", spec, c[0], c[1], yaw, group=self.name, wall_id=wid))
        if prm["rug"] is not None:
            L, Wd = RUG_SIZES[prm["rug"]]
            rc = bed.xy + f * (-(d / 2) + prm["rug_head"] + Wd / 2) + u * prm["rug_lat"]
            items.append(S.make_proc("rug_0", "rug", (L, Wd, 0.012), rc[0], rc[1], yaw, group=self.name))
        return self.finish(S, items)

    def mutate(self, S, prm: dict, rng) -> dict:
        p = copy.deepcopy(prm)
        r = rng.random()
        if r < 0.45:
            _mutate_anchor(p, rng, sigma_s=0.1)
        elif r < 0.65:
            p["ns_gap"] = float(np.clip(p["ns_gap"] + rng.normal(0, 0.02), 0.02, 0.12))
        elif p["rug"] is not None:
            p["rug_head"] = float(np.clip(p["rug_head"] + rng.normal(0, 0.06), 0.45, 0.95))
            p["rug_lat"] = float(np.clip(p["rug_lat"] + rng.normal(0, 0.05), -0.2, 0.2))
            if rng.random() < 0.3:
                p["rug"] = int(np.clip(p["rug"] + rng.choice([-1, 1]), 0, len(RUG_SIZES) - 1))
        return p


# ============================================================================ sala de jantar
def _chair_layout(table_spec, layout: str) -> list[tuple[float, float, float]]:
    """(lx, ly, yaw_local) das cadeiras ao redor da mesa (referencial local da mesa)."""
    tw, td, _ = table_spec.size
    round_table = abs(tw - td) < 0.05
    pos = []
    if layout == "2+2" and not round_table and tw >= 1.0:
        for side in (-1, 1):
            for off in (-tw / 4, tw / 4):
                pos.append((off, side * td / 2, side))
    elif layout == "4":
        pos = [(0.0, -td / 2, -1), (0.0, td / 2, 1), (-tw / 2, 0.0, -2), (tw / 2, 0.0, 2)]
    elif layout == "2":
        pos = [(0.0, -td / 2, -1), (0.0, td / 2, 1)]
    elif layout == "2l":
        pos = [(-tw / 2, 0.0, -2), (tw / 2, 0.0, 2)]
    return pos


class JantarPlanner(GroupPlanner):
    name = "jantar"
    layouts = ("2+2", "4")
    chair_role = "dining_chair"
    max_chairs = 4

    def candidates(self, S) -> list[GroupCand]:
        tables = S.ranked("dining_table", 2)
        chairs = S.ranked(self.chair_role if self.chair_role != "desk_chair" else "dining_chair", 1)
        base = []
        for t in tables:
            for sl in free_slots(S.ctx, step=0.2, yaws=(0.0, 90.0), face_camera=False):
                for lay in self.layouts:
                    if len(_chair_layout(t, lay)) == 0:
                        continue
                    prm = dict(table=t.asset_id, x=round(sl.x, 3), y=round(sl.y, 3), yaw=sl.yaw, layout=lay,
                               chair=chairs[0].asset_id if chairs else None, gap=0.03, rug=None)
                    items = self.build(S, prm)
                    if items is None:
                        continue
                    base.append((S.score(items).total, prm, items[0]))
        anchors = _nms(base, 8, min_dist=0.3)
        out = []
        rugs = (None, 0, 1, 2) if S.in_program("rug") else (None,)
        for _, prm in anchors:
            for ri in rugs:
                for rot in (0.0, 90.0):
                    q = dict(prm, rug=ri, rug_rot=rot)
                    items = self.build(S, q)
                    if items is not None:
                        out.append((S.score(items).total, q))
                    if ri is None:
                        break
        out.sort(key=lambda t: -t[0])
        res, seen = [], set()
        for sc, prm in out:
            sig = (prm["table"], prm["x"], prm["y"], prm["yaw"])
            if sig in seen:
                continue
            seen.add(sig)
            res.append(GroupCand(prm, self.build(S, prm), sc))
            if len(res) >= self.keep:
                break
        return res

    def build(self, S, prm: dict) -> list[Placement] | None:
        t_spec = S.spec(prm["table"])
        table = S.make("dining_table_0", t_spec, prm["x"], prm["y"], prm["yaw"], group=self.name)
        items = [table]
        if prm["chair"]:
            c_spec = S.spec(prm["chair"])
            cw, cd, _ = c_spec.size
            for i, (lx, ly, side) in enumerate(_chair_layout(t_spec, prm["layout"])[: self.max_chairs]):
                if abs(side) == 1:
                    off = np.array([lx, ly + side * (prm["gap"] + cd / 2)])
                    fdir = np.array([0.0, -side])
                else:
                    sg = 1 if side > 0 else -1
                    off = np.array([lx + sg * (prm["gap"] + cd / 2), ly])
                    fdir = np.array([-sg, 0.0])
                p = geom.local_to_world(table.x, table.y, table.yaw, off[0], off[1])
                fw = geom.local_to_world(0, 0, table.yaw, fdir[0], fdir[1])
                items.append(S.make(f"{self.chair_role}_{i}", c_spec, p[0], p[1], geom.yaw_facing(*fw),
                                    group=self.name, role=self.chair_role))
        if prm.get("rug") is not None:
            L, Wd = RUG_SIZES[prm["rug"]]
            tw, td, _ = t_spec.size
            if tw < td:
                L, Wd = Wd, L
            items.append(S.make_proc("rug_0", "rug", (L, Wd, 0.012), table.x, table.y,
                                     table.yaw + prm.get("rug_rot", 0.0), group=self.name))
        return self.finish(S, items)

    def mutate(self, S, prm: dict, rng) -> dict:
        p = copy.deepcopy(prm)
        r = rng.random()
        if r < 0.6:
            p["x"] = round(p["x"] + float(rng.normal(0, 0.08)), 3)
            p["y"] = round(p["y"] + float(rng.normal(0, 0.08)), 3)
        elif r < 0.75:
            p["gap"] = float(np.clip(p["gap"] + rng.normal(0, 0.02), 0.01, 0.1))
        elif p.get("rug") is not None:
            p["rug"] = int(np.clip(p["rug"] + rng.choice([-1, 1]), 0, len(RUG_SIZES) - 1))
        return p


class CozinhaMesaPlanner(JantarPlanner):
    """Cozinha sem ilha: mesa pequena com duas cadeiras."""
    name = "copa"
    layouts = ("2", "2l")
    max_chairs = 2


# ============================================================================ escritório
class EscritorioPlanner(GroupPlanner):
    name = "trabalho"

    def candidates(self, S) -> list[GroupCand]:
        chair = S.ranked("dining_chair", 1)
        base = []
        for di, (w, d, h) in enumerate(DESK_SIZES):
            for anc in _anchor_list(S, w, d, allow_free=True, overhang_frac=0.0, min_face=None, free_step=0.35):
                prm = dict(desk=di, anchor=anc, chair=chair[0].asset_id if chair else None, gap=0.05, rug=None)
                items = self.build(S, prm)
                if items is None:
                    continue
                base.append((S.score(items).total, prm, items[0]))
        anchors = _nms(base, 8)
        out = []
        for _, prm in anchors:
            for ri in ((None, 0) if S.in_program("rug") else (None,)):
                q = dict(prm, rug=ri)
                items = self.build(S, q)
                if items is not None:
                    out.append((S.score(items).total, q))
        out.sort(key=lambda t: -t[0])
        res, seen = [], set()
        for sc, prm in out:
            sig = (prm["desk"], prm["anchor"])
            if sig in seen:
                continue
            seen.add(sig)
            res.append(GroupCand(prm, self.build(S, prm), sc))
            if len(res) >= self.keep:
                break
        return res

    def build(self, S, prm: dict) -> list[Placement] | None:
        w, d, h = DESK_SIZES[prm["desk"]]
        x, y, yaw, wid = _anchor_pose(S, prm["anchor"], d, gap=0.02)
        desk = S.make_proc("desk_0", "desk", (w, d, h), x, y, yaw, group=self.name, wall_id=wid)
        items = [desk]
        f = desk.front
        if prm["chair"]:
            spec = S.spec(prm["chair"])
            cw, cd, _ = spec.size
            c = desk.xy + f * (d / 2 + prm["gap"] + cd / 2)
            items.append(S.make("desk_chair_0", spec, c[0], c[1], yaw + 180.0, group=self.name, role="desk_chair"))
        if prm.get("rug") is not None:
            L, Wd = RUG_SIZES[prm["rug"]]
            rc = desk.xy + f * (-(d / 2) + 0.08 + Wd / 2)
            items.append(S.make_proc("rug_0", "rug", (L, Wd, 0.012), rc[0], rc[1], yaw, group=self.name))
        return self.finish(S, items)

    def mutate(self, S, prm: dict, rng) -> dict:
        p = copy.deepcopy(prm)
        if rng.random() < 0.7:
            _mutate_anchor(p, rng)
        else:
            p["gap"] = float(np.clip(p["gap"] + rng.normal(0, 0.03), 0.02, 0.2))
        return p


# ============================================================================ cozinha com ilha
class BancadaPlanner(GroupPlanner):
    """Banquetas ao longo de uma ilha/península detectada (região BLOCKED cercada de piso livre)."""
    name = "bancada"

    def islands(self, S) -> list[dict]:
        import cv2

        ctx = S.ctx
        blk = (ctx.grid == BLOCKED).astype(np.uint8)
        n, lab = cv2.connectedComponents(blk, connectivity=8)
        out_mask = cv2.dilate((ctx.grid == OUTSIDE).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        res = []
        centers = ctx.cell_centers()
        for k in range(1, n):
            m = lab == k
            if m.sum() * ctx.cell ** 2 < 0.4 or (m & out_mask).any():
                continue
            pts = centers[m].astype(np.float32)
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
            rw += ctx.cell
            rh += ctx.cell
            res.append(dict(center=(float(cx), float(cy)), size=(float(rw), float(rh)), angle=float(ang)))
        return res

    def edges(self, isl: dict) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """(ponto médio, normal para fora, comprimento) das 4 arestas do retângulo da ilha."""
        cx, cy = isl["center"]
        rw, rh = isl["size"]
        a = math.radians(isl["angle"])
        ex, ey = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        c = np.array([cx, cy])
        return [(c + ex * rw / 2, ex, rh), (c - ex * rw / 2, -ex, rh), (c + ey * rh / 2, ey, rw),
                (c - ey * rh / 2, -ey, rw)]

    def candidates(self, S) -> list[GroupCand]:
        stools = S.ranked("bar_stool", 1)
        if not stools:
            return []
        out = []
        for ii, isl in enumerate(self.islands(S)):
            for ei, (mid, nrm, L) in enumerate(self.edges(isl)):
                for n in (3, 2):
                    prm = dict(island=ii, edge=ei, n=n, stool=stools[0].asset_id, gap=0.12, shift=0.0)
                    items = self.build(S, prm)
                    if items is not None:
                        out.append((S.score(items).total, prm))
                        break
        out.sort(key=lambda t: -t[0])
        return [GroupCand(p, self.build(S, p), sc) for sc, p in out[: self.keep]]

    def build(self, S, prm: dict) -> list[Placement] | None:
        isl = self.islands(S)[prm["island"]]
        mid, nrm, L = self.edges(isl)[prm["edge"]]
        spec = S.spec(prm["stool"])
        sw, sd, _ = spec.size
        n = prm["n"]
        spacing = max(sw + 0.1, 0.6)
        if (n - 1) * spacing + sw > L + 0.1:
            return None
        t = np.array([-nrm[1], nrm[0]])
        items = []
        for i in range(n):
            off = (i - (n - 1) / 2) * spacing + prm["shift"]
            c = mid + nrm * (prm["gap"] + sd / 2) + t * off
            items.append(S.make(f"bar_stool_{i}", spec, c[0], c[1], geom.yaw_facing(*(-nrm)), group=self.name))
        return self.finish(S, items)

    def mutate(self, S, prm: dict, rng) -> dict:
        p = copy.deepcopy(prm)
        p["shift"] = float(np.clip(p["shift"] + rng.normal(0, 0.05), -0.3, 0.3))
        p["gap"] = float(np.clip(p["gap"] + rng.normal(0, 0.02), 0.05, 0.25))
        return p


PLANNERS = {"sala": SalaPlanner, "quarto": QuartoPlanner, "sala_jantar": JantarPlanner,
            "escritorio": EscritorioPlanner, "cozinha": None, "banheiro": GroupPlanner}
