"""Geração de posições candidatas ("encaixes") para as peças.

- parede: costas encostadas na parede (folga de 2–5 cm), frente para dentro da sala;
- canto: encaixada entre duas paredes (alinhada a uma delas ou na diagonal);
- ao lado: nas laterais de uma peça âncora (sofá, poltrona, cama, mesa), alinhada pelo fundo;
- livre: pontos do piso livre (peças soltas como plantas ou mesas de jantar).

Cada candidato é um ``Slot`` com pose (x, y, yaw) e metadados para depuração e para manter
peças "presas" à âncora quando a âncora se move durante a otimização.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..analysis.room import FREE
from . import geom
from .context import LayoutContext
from .rules import SIDE_GAP, WALL_GAP
from .state import Placement


@dataclass
class Slot:
    x: float
    y: float
    yaw: float
    kind: str                     # parede | canto | lado | livre
    wall_id: int | None = None
    s: float | None = None        # posição ao longo da parede
    attach: tuple | None = None   # (chave da âncora, lado, alinhamento, folga): peça "presa" à âncora


def wall_pose(wall, s: float, d: float, gap: float = WALL_GAP) -> tuple[float, float, float]:
    """Centro e yaw de uma peça de profundidade ``d`` encostada na parede na posição ``s``."""
    c = wall.point(s) + wall.normal * (gap + d / 2.0)
    return float(c[0]), float(c[1]), geom.yaw_facing(*wall.normal)


def wall_slots(ctx: LayoutContext, w: float, d: float, gap: float = WALL_GAP, step: float = 0.1,
               overhang: float = 0.0, need_visible: bool = True) -> list[Slot]:
    out = []
    for wall in ctx.walls:
        lo, hi = w / 2 - overhang, wall.length - w / 2 + overhang
        if hi < lo:
            continue
        n = max(1, int(math.floor((hi - lo) / step)) + 1)
        for s in np.linspace(lo, hi, n) if n > 1 else [0.5 * (lo + hi)]:
            if need_visible:
                i = int(np.clip(np.searchsorted(wall.s, s), 0, len(wall.s) - 1))
                if not wall.vis_base[i]:
                    continue
            x, y, yaw = wall_pose(wall, float(s), d, gap)
            out.append(Slot(x, y, yaw, "parede", wall.id, float(s)))
    return out


def corner_slots(ctx: LayoutContext, w: float, d: float, gap: float = 0.04, diagonal: bool = True) -> list[Slot]:
    out = []
    for c in ctx.corners:
        n1, n2 = c.normals
        for a, b, wid in ((n1, n2, c.walls[0]), (n2, n1, c.walls[1])):
            p = c.point + a * (gap + d / 2) + b * (gap + w / 2)
            out.append(Slot(float(p[0]), float(p[1]), geom.yaw_facing(*a), "canto", wid))
        if diagonal:
            yaw = geom.yaw_facing(*c.bisector)
            f, u = geom.front_of(yaw), geom.lateral_of(yaw)
            h1 = abs(w / 2 * (u @ n1)) + abs(d / 2 * (f @ n1))
            h2 = abs(w / 2 * (u @ n2)) + abs(d / 2 * (f @ n2))
            p = c.point + n1 * (gap + h1) + n2 * (gap + h2)
            out.append(Slot(float(p[0]), float(p[1]), yaw, "canto", None))
    return out


def adjacent_pose(anchor: Placement, w: float, d: float, sigma: float, align: str,
                  gap: float) -> tuple[float, float, float]:
    """Pose de uma peça w x d ao lado (sigma = -1/+1) da âncora, alinhada pelo fundo/frente/centro."""
    lx = sigma * (anchor.w / 2 + gap + w / 2)
    if align == "fundo":
        ly = anchor.d / 2 - d / 2
    elif align == "frente":
        ly = -anchor.d / 2 + d / 2
    else:
        ly = 0.0
    p = geom.local_to_world(anchor.x, anchor.y, anchor.yaw, lx, ly)
    return float(p[0]), float(p[1]), anchor.yaw


def adjacent_slots(anchor: Placement, w: float, d: float, gap: float = SIDE_GAP,
                   align: tuple[str, ...] = ("fundo",)) -> list[Slot]:
    """Laterais de ``anchor``: alinhada pelo fundo (encostada na mesma parede) ou pelo centro."""
    out = []
    for sigma in (-1.0, 1.0):
        for al in align:
            x, y, yaw = adjacent_pose(anchor, w, d, sigma, al, gap)
            out.append(Slot(x, y, yaw, "lado", anchor.wall_id, attach=(anchor.key, sigma, al, gap)))
    return out


def free_slots(ctx: LayoutContext, step: float = 0.3, yaws: tuple[float, ...] | None = None,
               face_camera: bool = True) -> list[Slot]:
    """Pontos do piso livre numa grade de ``step`` m; yaw voltado para a câmera e/ou eixos da sala."""
    centers = ctx.cell_centers()[ctx.free]
    if len(centers) == 0:
        return []
    lo, hi = centers.min(0), centers.max(0)
    xs = np.arange(lo[0] + step / 2, hi[0], step)
    ys = np.arange(lo[1] + step / 2, hi[1], step)
    out = []
    for y in ys:
        for x in xs:
            p = np.array([x, y])
            if ctx.states(p[None])[0] != FREE:
                continue
            cands = list(yaws or ())
            if face_camera:
                t = ctx.dir_to_camera(p)
                cands.append(round(geom.yaw_facing(*t), 1))
            for yaw in cands:
                out.append(Slot(float(x), float(y), float(yaw), "livre"))
    return out


def axis_yaws_facing_camera(ctx: LayoutContext, xy: np.ndarray, min_dot: float = 0.15) -> list[float]:
    """Yaws alinhados aos eixos da sala (0/90/180/270) cuja frente aparece para a câmera."""
    t = ctx.dir_to_camera(xy)
    out = []
    for yaw in (0.0, 90.0, 180.0, 270.0):
        if geom.front_of(yaw) @ t >= min_dot:
            out.append(yaw)
    return out
