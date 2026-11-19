"""Geometria auxiliar do layout: caixas orientadas (OBB) no piso, caixas 3D e polígonos.

Convenções (iguais a ``mobilia/layout/model.py``): referencial da sala com Z para cima,
piso em Z = 0, metros; ``yaw`` em graus, anti-horário visto de cima; a frente do asset
aponta para -Y local, ou seja, para (sin(yaw), -cos(yaw)) após o giro. O eixo X local
("lateral") vira (cos(yaw), sin(yaw)).
"""
from __future__ import annotations

import math

import numpy as np


def front_of(yaw_deg: float) -> np.ndarray:
    """Direção (2D, unitária) para onde a frente do asset aponta."""
    y = math.radians(yaw_deg)
    return np.array([math.sin(y), -math.cos(y)])


def lateral_of(yaw_deg: float) -> np.ndarray:
    """Eixo X local do asset no referencial da sala (2D, unitário)."""
    y = math.radians(yaw_deg)
    return np.array([math.cos(y), math.sin(y)])


def yaw_facing(dx: float, dy: float) -> float:
    """Yaw (graus) que faz a frente do asset apontar para a direção (dx, dy)."""
    return math.degrees(math.atan2(dx, -dy))


def wrap_deg(a: float) -> float:
    """Ângulo em graus no intervalo (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def local_to_world(cx: float, cy: float, yaw_deg: float, lx, ly) -> np.ndarray:
    """Converte coordenadas locais (lx, ly) do asset em coordenadas da sala."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    lx = np.asarray(lx, float)
    ly = np.asarray(ly, float)
    return np.stack([cx + c * lx - s * ly, cy + s * lx + c * ly], -1)


def world_to_local(cx: float, cy: float, yaw_deg: float, xy: np.ndarray) -> np.ndarray:
    """Converte pontos da sala (N, 2) para o referencial local do asset."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    d = np.asarray(xy, float) - np.array([cx, cy])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def obb_corners(cx: float, cy: float, w: float, d: float, yaw_deg: float) -> np.ndarray:
    """4 cantos (anti-horário) da base de largura ``w`` (X local) e profundidade ``d`` (Y local)."""
    hw, hd = w / 2.0, d / 2.0
    return local_to_world(cx, cy, yaw_deg, [-hw, hw, hw, -hw], [-hd, -hd, hd, hd])


def box_corners_3d(cx: float, cy: float, z: float, w: float, d: float, h: float, yaw_deg: float) -> np.ndarray:
    """8 cantos (N, 3) da caixa 3D: 4 da base (z) e 4 do topo (z + h)."""
    c2 = obb_corners(cx, cy, w, d, yaw_deg)
    lo = np.column_stack([c2, np.full(4, z)])
    hi = np.column_stack([c2, np.full(4, z + h)])
    return np.vstack([lo, hi])


def sample_obb(cx: float, cy: float, w: float, d: float, yaw_deg: float, step: float = 0.025,
               shrink: float = 0.01) -> np.ndarray:
    """Pontos regulares (N, 2) cobrindo a base (incluindo a borda), encolhida de ``shrink``."""
    hw, hd = max(w / 2.0 - shrink, 0.005), max(d / 2.0 - shrink, 0.005)
    nx = max(2, int(math.ceil(2 * hw / step)) + 1)
    ny = max(2, int(math.ceil(2 * hd / step)) + 1)
    lx, ly = np.meshgrid(np.linspace(-hw, hw, nx), np.linspace(-hd, hd, ny))
    return local_to_world(cx, cy, yaw_deg, lx.ravel(), ly.ravel())


def _axes(poly: np.ndarray) -> np.ndarray:
    e = poly[[*range(1, len(poly)), 0]] - poly
    n = np.stack([-e[:, 1], e[:, 0]], -1)
    return n / (np.sqrt((n * n).sum(1, keepdims=True)) + 1e-12)


def obb_penetration(a: np.ndarray, b: np.ndarray) -> float:
    """Profundidade de interpenetração (m) de dois polígonos convexos pelo teorema do eixo separador.

    Retorna <= 0 quando estão separados (o valor negativo é a folga ao longo do melhor eixo,
    um limite inferior da distância real). Implementação em Python puro: para polígonos de
    4 vértices é ~6x mais rápida que a versão vetorizada (arrays minúsculos).
    """
    A = a.tolist()
    B = b.tolist()
    best = math.inf
    for P in (A, B):
        n = len(P)
        for i in range(n):
            x0, y0 = P[i]
            x1, y1 = P[(i + 1) % n]
            nx, ny = y0 - y1, x1 - x0
            L = math.hypot(nx, ny)
            if L < 1e-12:
                continue
            nx /= L
            ny /= L
            pa = [x * nx + y * ny for x, y in A]
            pb = [x * nx + y * ny for x, y in B]
            ov = min(max(pa), max(pb)) - max(min(pa), min(pb))
            if ov < best:
                best = ov
    return float(best)


def bounding_radius(poly: np.ndarray) -> float:
    c = poly.mean(0)
    return float(np.sqrt(((poly - c) ** 2).sum(1)).max())


def may_touch(a: np.ndarray, b: np.ndarray, margin: float = 0.0) -> bool:
    """Teste rápido por círculos envolventes (falso = certamente separados por mais que ``margin``)."""
    ca, cb = a.mean(0), b.mean(0)
    d = float(np.hypot(*(ca - cb)))
    return d <= bounding_radius(a) + bounding_radius(b) + margin


def polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distância euclidiana mínima entre dois polígonos convexos (0 se interceptam)."""
    if obb_penetration(a, b) > 0:
        return 0.0
    return float(min(_pts_to_edges(a, b).min(), _pts_to_edges(b, a).min()))


def _pts_to_edges(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distâncias (N, M) dos pontos p às arestas do polígono q."""
    s0 = q
    s1 = q[[*range(1, len(q)), 0]]
    d = s1 - s0                                             # (M, 2)
    L2 = np.maximum((d * d).sum(1), 1e-12)                  # (M,)
    r = p[:, None, :] - s0[None, :, :]                      # (N, M, 2)
    t = np.clip((r * d[None]).sum(-1) / L2[None], 0.0, 1.0)
    proj = s0[None] + t[..., None] * d[None]
    return np.sqrt(((p[:, None, :] - proj) ** 2).sum(-1))


def point_segment_distance(p: np.ndarray, s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
    """Distância de pontos (N, 2) ao segmento s0-s1."""
    p = np.atleast_2d(p)
    d = s1 - s0
    L2 = float(d @ d)
    if L2 < 1e-12:
        return np.linalg.norm(p - s0, axis=1)
    t = np.clip(((p - s0) @ d) / L2, 0.0, 1.0)
    return np.linalg.norm(p - (s0 + t[:, None] * d), axis=1)


def point_polygon_distance(p: np.ndarray, poly: np.ndarray) -> float:
    """Distância de um ponto 2D a um polígono convexo (0 se estiver dentro)."""
    p = np.asarray(p, float)
    if point_in_convex(p, poly):
        return 0.0
    return float(_pts_to_edges(p[None], poly).min())


def points_in_convex(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Máscara (N,) de pontos dentro de um polígono convexo anti-horário."""
    e = poly[[*range(1, len(poly)), 0]] - poly                  # (M, 2)
    r = np.asarray(pts, float)[:, None, :] - poly[None]         # (N, M, 2)
    cross = e[None, :, 0] * r[..., 1] - e[None, :, 1] * r[..., 0]
    return np.all(cross >= -1e-9, axis=1)


def point_in_convex(p: np.ndarray, poly: np.ndarray) -> bool:
    """Ponto dentro de polígono convexo anti-horário."""
    return bool(points_in_convex(np.asarray(p, float)[None], poly)[0])


def segments_intersect(p0, p1, q0, q1, eps: float = 1e-9) -> bool:
    """Interseção própria de dois segmentos 2D (toques nas extremidades não contam)."""
    p0, p1, q0, q1 = (np.asarray(v, float) for v in (p0, p1, q0, q1))

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1, o2 = orient(p0, p1, q0), orient(p0, p1, q1)
    o3, o4 = orient(q0, q1, p0), orient(q0, q1, p1)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Envoltória convexa (anti-horária) por cadeia monotônica de Andrew."""
    pts = np.unique(np.round(np.asarray(points, float), 9), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Recorte de Sutherland-Hodgman de um polígono convexo por outro (ambos anti-horários)."""
    out = [np.asarray(p, float) for p in subject]
    clip = np.asarray(clip, float)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break

        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

        def inter(p, q):
            d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            d2 = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
            t = d1 / (d1 - d2) if abs(d1 - d2) > 1e-12 else 0.0
            return p + (q - p) * t

        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
    return np.array(out) if out else np.zeros((0, 2))


def intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    """Área da interseção de dois polígonos convexos anti-horários."""
    if len(a) < 3 or len(b) < 3:
        return 0.0
    return polygon_area(clip_convex(a, b))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return poly if signed >= 0 else poly[::-1].copy()


def wall_extensions(walls: list[dict], ext: float = 0.35, step: float = 0.01,
                    tol: float = 0.01) -> list[list[float]]:
    """Quanto cada parede-proxy pode ser estendida além de cada ponta (fecha as quinas).

    A extensão só pode passar POR TRÁS das outras paredes. Se avançasse para o lado de dentro
    da sala diante de outra parede visível (quinas que não são de 90°), o proxy taparia os
    móveis encostados nela (ex.: um quadro cortado pela quina).
    """
    out = []
    for w in walls:
        p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
        L = float(np.linalg.norm(p1 - p0))
        d = (p1 - p0) / max(L, 1e-9)
        ends = []
        for base, sign in ((p0, -1.0), (p1, 1.0)):
            allowed = 0.0
            s = step
            while s <= ext + 1e-9:
                q = base + sign * d * s
                blocked = False
                for o in walls:
                    if o is w:
                        continue
                    a, b = np.asarray(o["p0"], float), np.asarray(o["p1"], float)
                    Lo = float(np.linalg.norm(b - a))
                    do = (b - a) / max(Lo, 1e-9)
                    t = float(np.dot(q - a, do))
                    in_front = float(np.dot(q - a, np.asarray(o["normal"], float))) > tol
                    if in_front and -tol <= t <= Lo + tol:
                        blocked = True
                        break
                if blocked:
                    break
                allowed = s
                s += step
            ends.append(round(allowed, 3))
        out.append(ends)
    return out
