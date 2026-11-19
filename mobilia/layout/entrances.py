"""Entradas cinematográficas dos itens (PT11): qual movimento cada peça usa na montagem.

Por papel (``choreography.entrance_for``): móveis pesados caem; estantes, cômodas e a luminária de
piso surgem do chão; poltronas, cadeiras, mesas laterais e pufes deslizam; tapete desenrola;
plantas crescem; quadros descem rente à parede; decoração pequena cai.

O deslize só vale com um caminho varrido livre, validado aqui com a mesma geometria do layout:
a base (OBB) amostrada a cada ``SWEEP_STEP`` m do início até o repouso precisa ficar só em células
FREE/UNKNOWN, nunca cruzar uma parede-proxy (com as extensões das quinas, como no render) e nunca
invadir um móvel de piso que entra antes dela, além do que já encosta nele em repouso. A direção é a do lado da
borda mais próxima da imagem: o eixo horizontal da câmera no piso e, depois, o eixo lateral da
própria peça (rente à parede dela) no mesmo sentido; depois o lado oposto. Cadeiras de mesa tentam
antes entrar "empurradas" para a mesa. Sem caminho livre, a peça cai (``drop``).
"""
from __future__ import annotations

import math

import numpy as np

from ..analysis.room import BLOCKED, OUTSIDE
from ..render import choreography as ch
from . import geom
from .context import LayoutContext
from .state import Placement

SWEEP_STEP = 0.04          # m entre amostras do caminho
TOUCH_TOL = 0.015          # m: penetração extra tolerada com vizinhos (arredondamentos)


def camera_lateral(ctx: LayoutContext) -> np.ndarray:
    """Eixo horizontal da imagem (para a direita) projetado no piso, unitário."""
    r = np.asarray(ctx.R[:2, 0], float)
    n = float(np.linalg.norm(r))
    return r / n if n > 1e-9 else np.array([1.0, 0.0])


def _wall_segments(ctx: LayoutContext) -> list[tuple[np.ndarray, np.ndarray]]:
    walls = [{"p0": list(w.p0), "p1": list(w.p1), "normal": list(w.normal)} for w in ctx.walls]
    out = []
    for w, (e0, e1) in zip(ctx.walls, geom.wall_extensions(walls)):
        out.append((np.asarray(w.p0, float) - w.direction * e0, np.asarray(w.p1, float) + w.direction * e1))
    return out


def _crosses_wall(poly: np.ndarray, walls) -> bool:
    for a, b in walls:
        for i in range(len(poly)):
            if geom.segments_intersect(poly[i], poly[(i + 1) % len(poly)], a, b):
                return True
    return False


def path_free(p: Placement, direction, dist: float, others: list[Placement], ctx: LayoutContext,
              walls=None) -> bool:
    """O deslize de ``dist`` m ao longo de ``direction`` (sentido do movimento) até a pose final de
    ``p`` é livre: base só em FREE/UNKNOWN, sem cruzar paredes, sem invadir outro móvel de piso."""
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    walls = _wall_segments(ctx) if walls is None else walls
    rest = p.corners()
    near: list[tuple[np.ndarray, float]] = []
    for o in others:
        if o.key == p.key or o.mount != "floor" or o.role == "rug":
            continue
        oc = o.corners()
        # só o que pode ser alcançado pelo caminho (círculos envolventes com folga)
        reach = dist + 0.5 * math.hypot(p.w, p.d) + 0.5 * math.hypot(o.w, o.d) + 0.05
        mid = p.xy - d * dist / 2
        if float(np.linalg.norm(o.xy - mid)) > reach:
            continue
        near.append((oc, max(0.0, geom.obb_penetration(rest, oc))))
    n = max(1, int(math.ceil(dist / SWEEP_STEP)))
    for k in range(n + 1):
        s = dist * k / n
        cx, cy = p.x - d[0] * s, p.y - d[1] * s
        st = ctx.states(geom.sample_obb(cx, cy, p.w, p.d, p.yaw, step=0.025, shrink=0.012))
        if np.any((st == BLOCKED) | (st == OUTSIDE)):
            return False
        poly = geom.obb_corners(cx, cy, p.w, p.d, p.yaw)
        # base encolhida 1 cm: encostar na parede em repouso não conta como atravessá-la
        if _crosses_wall(geom.obb_corners(cx, cy, max(p.w - 0.02, 0.01), max(p.d - 0.02, 0.01), p.yaw), walls):
            return False
        for oc, pen0 in near:
            if geom.obb_penetration(poly, oc) > pen0 + TOUCH_TOL:
                return False
    return True


def glide_candidates(p: Placement, ctx: LayoutContext) -> list[tuple[np.ndarray, str, tuple]]:
    """Direções (sentido do movimento), rótulo e distâncias a tentar, em ordem de preferência."""
    r = camera_lateral(ctx)
    uv, z = ctx.project(np.array([[p.x, p.y, p.z + p.h / 2]]))
    left = bool(z[0] <= 0 or uv[0, 0] < ctx.W / 2)
    # vem da borda mais próxima: da esquerda, anda para a direita da imagem
    near_side, far_side = ("esquerda", "direita") if left else ("direita", "esquerda")
    toward = r if left else -r
    # o eixo lateral da própria peça (paralelo à parede em que ela encosta), no sentido que também
    # vem da borda mais próxima: numa câmera girada em relação à sala, o eixo da câmera entra na
    # parede do fundo, e o da peça desliza rente a ela
    lat = np.asarray(p.lateral, float)
    lat = lat if float(lat @ toward) >= 0 else -lat
    out = []
    if p.role in ch.PUSH_IN_ROLES:
        out.append((np.asarray(p.front, float), "frente", (ch.PUSH_IN_DIST,)))
    out += [(toward, near_side, ch.GLIDE_DISTS), (lat, near_side, ch.GLIDE_DISTS),
            (-toward, far_side, ch.GLIDE_DISTS), (-lat, far_side, ch.GLIDE_DISTS)]
    return out


def plan_glide(p: Placement, others: list[Placement], ctx: LayoutContext, walls=None) -> dict | None:
    """Primeiro caminho livre (direção, lado, distância) para o deslize, ou None."""
    walls = _wall_segments(ctx) if walls is None else walls
    for d, side, dists in glide_candidates(p, ctx):
        for dist in dists:
            if path_free(p, d, dist, others, ctx, walls):
                return {"motion": "glide", "motion_dir": [round(float(d[0]), 4), round(float(d[1]), 4)],
                        "motion_dist": float(dist), "lado": side}
    return None


def plan_entrances(items: list[Placement], ctx: LayoutContext) -> dict[str, dict]:
    """{chave: {"motion", ["motion_dir", "motion_dist", "lado"]}} para cada item do layout.

    ``items`` na ordem da coreografia: o caminho do deslize é validado contra os móveis que já
    estão na cena quando a peça entra (os anteriores). Os seguintes só começam depois de ~55% da
    duração dela, quando o deslize (60% da duração, ease-out) já está a menos de 1 mm do fim."""
    walls = _wall_segments(ctx)
    out: dict[str, dict] = {}
    for i, p in enumerate(items):
        m = ch.entrance_for(p.role, p.mount)
        if m == "glide":
            g = plan_glide(p, items[:i], ctx, walls)
            out[p.key] = g if g is not None else {"motion": "drop", "bloqueado": "glide"}
        elif m == "emerge" and p.mount != "floor":
            out[p.key] = {"motion": "drop"}
        else:
            out[p.key] = {"motion": m}
    return out
