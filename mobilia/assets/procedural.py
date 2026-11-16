"""Assets 3D procedurais gerados no Blender (roda DENTRO de um processo com ``bpy``).

Uso:
    from mobilia.assets.procedural import build_asset, animate_rug_unroll
    root = build_asset("bed", {"size": [1.68, 2.12, 0.95]}, "bed_0")
    rug = build_asset("rug", {"pattern": "berber", "fringe": True}, "rug_0")
    animate_rug_unroll(rug, 1, 48)

Contrato do asset (igual aos assets da Poly Haven normalizados):
- ``build_asset`` devolve um EMPTY raiz na origem do mundo, sem rotação/escala;
- origem = centro da base do bounding box (piso em Z = 0), FRENTE para -Y,
  dimensões reais em metros iguais a ``params["size"]`` = (largura X, profundidade Y,
  altura Z) — padrão: tamanho do catálogo (``mobilia/layout/catalog.py``);
- as peças são filhos do EMPTY, só com translação (a origem de cada peça é o seu
  pivô de animação) e propriedades customizadas:
    obj["mobilia_part"]    nome da peça ("frame", "shade", ...)
    obj["assembly_order"]  ordem em que a peça aparece (0 = primeira)
    obj["assembly_motion"] drop | rise | slide | unroll | unfold | grow | settle
- pivôs: base da peça para drop/rise/grow/settle; topo para unfold (cortina, que
  "desce" a partir do varão); ponto de encaixe para slide; o tapete (unroll) é
  animado por shape keys com ``animate_rug_unroll``.

Abajures e luminárias ficam APAGADOS (nenhum material emite luz): a iluminação da
foto original não pode mudar.

Parâmetros por tipo (todos opcionais; cores em sRGB "#RRGGBB" ou tupla 0..1). Os nomes
usados pelo motor de layout (``pattern_color``, ``metal_color``, ``frame_color``,
``linen_color``, ``top_color``, ``leg_color``...) são traduzidos por ``normalize_params``:
- comuns: ``size`` [l, p, a] (ou ``width``/``depth``/``height``), ``seed``;
- rug: ``pattern`` (solid_border | berber | stripes | persian), ``color``, ``accent``,
  ``fringe`` (bool), ``fringe_length``, ``fringe_color``, ``unroll_axis`` (x | y);
- floor_lamp: ``metal`` (black | graphite | brass | chrome | matte_black | white),
  ``shade_color``, ``shade_shape`` (drum | tapered);
- table_lamp: ``base_material`` (ceramic | wood | brass | black | chrome),
  ``base_color``, ``metal``, ``shade_color``, ``shade_shape``;
- bed: ``frame`` (upholstered | wood), ``frame_color`` (tom da madeira/laca),
  ``fabric`` (linen | velvet | boucle), ``fabric_color``, ``duvet_color``, ``sheet_color``,
  ``pillow_colors`` (lista), ``pillows`` (2..4), ``headboard`` (channel | panel),
  ``legs`` (wood | black), ``wood``, ``mattress_top``, ``headboard_depth``;
- curtain: ``color``, ``sheer`` (bool), ``panels`` (1 | 2), ``coverage`` (fração do
  varão coberta por painel), ``metal``;
- desk: ``wood`` (oak | walnut), ``top_color`` (tom da madeira), ``legs`` (wood | black |
  painted | metal), ``leg_finish`` (acabamento de metal), ``leg_color``, ``paint_color``,
  ``drawer`` (bool);
- throw_blanket: ``color``, ``accent``, ``pattern`` (solid | stripes), ``tassels`` (bool);
- book_stack: ``count`` (2 | 3 livros deitados), ``colors`` (capas, de baixo para cima),
  ``page_color`` (corte das páginas). Cada livro é uma peça ``book_<i>`` (ordem i, "drop").
"""
from __future__ import annotations

import json
import math
import zlib
from collections.abc import Callable, Sequence

import bpy
import numpy as np

from . import (ASSEMBLY_MOTIONS, DUVET_GAP, DUVET_THICKNESS, DUVET_TURN, MATTRESS_EDGE_RADIUS, MATTRESS_HEIGHT,
               PROCEDURAL_KINDS, bed_dimensions)
from . import materials as M

Array = np.ndarray

# =========================================================================== ruído determinístico


def _hash01(ix: Array, iy: Array, seed: int) -> Array:
    """Hash inteiro -> [0, 1) (determinístico, independente de plataforma)."""
    x = (np.asarray(ix, np.int64) & 0xFFFFFFFF).astype(np.uint64)
    y = (np.asarray(iy, np.int64) & 0xFFFFFFFF).astype(np.uint64)
    h = x * np.uint64(0x9E3779B1) + y * np.uint64(0x85EBCA77) + np.uint64((seed * 0xC2B2AE3D) & 0xFFFFFFFF)
    h ^= h >> np.uint64(15)
    h *= np.uint64(0x2C1B3C6D)
    h ^= h >> np.uint64(12)
    h *= np.uint64(0x297A2D39)
    h ^= h >> np.uint64(15)
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


def _noise2(x: Array, y: Array, seed: int = 0) -> Array:
    """Value noise 2D suave (interpolação quíntica), saída em [-1, 1]."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    ix, iy = x0.astype(np.int64), y0.astype(np.int64)
    a = _hash01(ix, iy, seed)
    b = _hash01(ix + 1, iy, seed)
    c = _hash01(ix, iy + 1, seed)
    d = _hash01(ix + 1, iy + 1, seed)
    v = a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
    return v * 2.0 - 1.0


def _fbm2(x: Array, y: Array, seed: int = 0, octaves: int = 3, gain: float = 0.5) -> Array:
    total = np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape)
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        total += amp * _noise2(np.asarray(x) * freq, np.asarray(y) * freq, seed + 101 * o)
        norm += amp
        amp *= gain
        freq *= 2.03
    return total / norm


def _smoothstep(e0: float, e1: float, x: Array) -> Array:
    t = np.clip((np.asarray(x, np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _normalize(v: Array) -> Array:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _spaced(segments: Sequence[tuple[float, float, float]]) -> Array:
    """Coordenadas crescentes a partir de trechos (início, fim, passo máximo)."""
    pts = []
    for a, b, step in segments:
        if b <= a:
            continue
        n = max(1, int(math.ceil((b - a) / step)))
        pts.append(np.linspace(a, b, n + 1))
    out = np.unique(np.round(np.concatenate(pts), 9))
    keep = np.concatenate([[True], np.diff(out) > 1e-6])
    return out[keep]


def _catmull(points: Sequence[Sequence[float]], samples: int = 12) -> Array:
    """Spline Catmull-Rom (centrípeta simples) passando pelos pontos de controle."""
    P = np.asarray(points, np.float64)
    P = np.vstack([P[0] * 2 - P[1], P, P[-1] * 2 - P[-2]])
    out = []
    t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2][None])
    return np.vstack(out)


def _rot_x(deg: float) -> Array:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def _rot_y(deg: float) -> Array:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def _rot_z(deg: float) -> Array:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


# =========================================================================== acumulador de malha


class MeshBuilder:
    """Acumula vértices/faces (com UV por canto e índice de material) e cria a malha."""

    def __init__(self) -> None:
        self._verts: list[Array] = []
        self._faces: list[tuple[Array, Array, Array]] = []
        self.count = 0

    def add_verts(self, V: Array) -> int:
        V = np.asarray(V, np.float64).reshape(-1, 3)
        base = self.count
        self._verts.append(V)
        self.count += len(V)
        return base

    def add_faces(self, F: Array, UV: Array, mat: int | Array = 0) -> None:
        F = np.asarray(F, np.int64)
        UV = np.asarray(UV, np.float64).reshape(F.shape[0], F.shape[1], 2)
        mats = np.broadcast_to(np.asarray(mat, np.int64), (F.shape[0],)).copy()
        self._faces.append((F, UV, mats))

    def grid(self, P: Array, UV: Array, mat: int = 0, close_u: bool = False, flip: bool = False) -> int:
        """Grade de quads. P (nv, nu, 3); UV (nv, nu[+1 se fechada em u], 2)."""
        nv, nu, _ = P.shape
        base = self.add_verts(P.reshape(-1, 3))
        nj = nu if close_u else nu - 1
        i, j = np.meshgrid(np.arange(nv - 1), np.arange(nj), indexing="ij")
        i, j = i.ravel(), j.ravel()
        j1 = (j + 1) % nu
        F = np.stack([i * nu + j, i * nu + j1, (i + 1) * nu + j1, (i + 1) * nu + j], 1) + base
        UVg = np.asarray(UV, np.float64)
        uvc = np.stack([UVg[i, j], UVg[i, j + 1], UVg[i + 1, j + 1], UVg[i + 1, j]], 1)
        if flip:
            F, uvc = F[:, ::-1], uvc[:, ::-1]
        self.add_faces(F, uvc, mat)
        return base

    def vertices(self) -> Array:
        return np.concatenate(self._verts) if self._verts else np.zeros((0, 3))

    def transform(self, R: Array | None = None, t: Sequence[float] = (0.0, 0.0, 0.0), start: int = 0) -> None:
        """Aplica rotação/translação aos vértices a partir do índice ``start``."""
        V = self.vertices()
        sub = V[start:]
        if R is not None:
            sub = sub @ np.asarray(R).T
        V[start:] = sub + np.asarray(t)
        self._verts = [V]

    def extend(self, other: "MeshBuilder", mat: int | None = None, R: Array | None = None,
               t: Sequence[float] = (0.0, 0.0, 0.0)) -> int:
        """Copia outra malha para esta (com rotação/translação e material opcionais)."""
        V = other.vertices()
        if R is not None:
            V = V @ np.asarray(R).T
        base = self.add_verts(V + np.asarray(t))
        for F, UV, mt in other._faces:
            self.add_faces(F + base, UV, mt if mat is None else mat)
        return base

    def set_vertices(self, V: Array) -> None:
        """Substitui todas as posições (mesma quantidade e ordem de vértices)."""
        V = np.asarray(V, np.float64).reshape(-1, 3)
        if len(V) != self.count:
            raise ValueError("quantidade de vértices diferente")
        self._verts = [V]

    def offset_uv(self, off: Sequence[float]) -> None:
        """Desloca todas as UVs (em metros)."""
        self._faces = [(F, UV + np.asarray(off, np.float64), mt) for F, UV, mt in self._faces]

    def bounds(self, start: int = 0) -> tuple[Array, Array]:
        V = self.vertices()[start:]
        return V.min(0), V.max(0)

    def to_mesh(self, name: str, origin: Sequence[float] = (0.0, 0.0, 0.0)) -> bpy.types.Mesh:
        V = self.vertices() - np.asarray(origin, np.float64)
        sizes, loops, uvs, mats = [], [], [], []
        for F, UV, mt in self._faces:
            sizes.append(np.full(len(F), F.shape[1], np.int64))
            loops.append(F.ravel())
            uvs.append(UV.reshape(-1, 2))
            mats.append(mt)
        S = np.concatenate(sizes)
        L = np.concatenate(loops)
        me = bpy.data.meshes.new(name)
        me.vertices.add(len(V))
        me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        me.loops.add(len(L))
        me.loops.foreach_set("vertex_index", L.astype(np.int32))
        me.polygons.add(len(S))
        me.polygons.foreach_set("loop_start", (np.cumsum(S) - S).astype(np.int32))
        me.polygons.foreach_set("material_index", np.concatenate(mats).astype(np.int32))
        me.update(calc_edges=True)
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.concatenate(uvs).astype(np.float32).ravel())
        me.shade_smooth()
        me.validate(clean_customdata=False)
        me.update()
        return me


# =========================================================================== primitivas


def _box_axis(h: float, r: float, nb: int, step: float, guard: float) -> Array:
    """Coordenadas "virtuais" de um eixo da caixa arredondada (com linhas de guarda)."""
    inner = h - r
    if r > 1e-9:
        band = inner + r * np.tan(np.linspace(0.0, math.pi / 4, nb + 1)[1:])
    else:
        band = np.zeros(0)
    if inner <= 1e-7:
        flat = np.array([0.0])
    else:
        n = max(1, int(math.ceil(2 * inner / step)))
        flat = np.linspace(-inner, inner, n + 1)
        if r > 1e-9 and 2 * guard < inner:
            flat = np.union1d(flat, [-inner + guard, inner - guard])
    c = np.concatenate([-band[::-1], flat, band])
    c = np.unique(np.round(c, 10))
    keep = np.concatenate([[True], np.diff(c) > 1e-7])
    return c[keep]


_BOX_SIDES = (  # (eixo fixo, lado, eixo a, eixo b) com a×b apontando para fora
    (0, 1, 1, 2), (0, 0, 2, 1), (1, 1, 2, 0), (1, 0, 0, 2), (2, 1, 0, 1), (2, 0, 1, 0),
)
_SIDE_NAMES = ("+x", "-x", "+y", "-y", "+z", "-z")


def rounded_box(mb: MeshBuilder, center: Sequence[float], size: Sequence[float], radius: float,
                mat: int | dict = 0, step: float | Sequence[float] = 0.05, seg: int = 4,
                guard: float = 0.0025, uv: str = "box", uv_swap: bool = False,
                deform: Callable[[Array], Array] | None = None) -> int:
    """Caixa com TODAS as arestas arredondadas (raio ``radius``) — chanfro suave real.

    Linhas de guarda perto das tangências mantêm as faces planas sem degradê de
    sombreamento. ``mat`` pode ser um dict por lado ("+x", "-z", ...).
    ``uv="xy"`` projeta todas as faces no plano XY (usado no tapete).
    """
    h = np.asarray(size, np.float64) / 2.0
    r = float(min(radius, *h))
    steps = np.broadcast_to(np.asarray(step, np.float64), (3,))
    nb = max(1, int(math.ceil(seg / 2)))
    axes = [_box_axis(h[k], r, nb, steps[k], guard) for k in range(3)]
    n = [len(a) for a in axes]
    idx = -np.ones(n, np.int64)
    surf = np.zeros(n, bool)
    surf[[0, -1], :, :] = True
    surf[:, [0, -1], :] = True
    surf[:, :, [0, -1]] = True
    ii = np.argwhere(surf)
    idx[tuple(ii.T)] = np.arange(len(ii))
    Vv = np.stack([axes[0][ii[:, 0]], axes[1][ii[:, 1]], axes[2][ii[:, 2]]], 1)
    inner = np.clip(Vv, -(h - r), h - r)
    d = Vv - inner
    P = inner + r * _normalize(d) if r > 1e-9 else Vv
    if deform is not None:
        P = deform(P)
    P = P + np.asarray(center, np.float64)
    base = mb.add_verts(P)
    for side, (fa, hi, aa, ab) in enumerate(_BOX_SIDES):
        sl = [slice(None)] * 3
        sl[fa] = n[fa] - 1 if hi else 0
        G = idx[tuple(sl)]  # 2D (eixos restantes em ordem crescente)
        rest = [k for k in range(3) if k != fa]
        if rest.index(aa) == 1:  # garante ordem (a, b)
            G = G.T
        na, nbb = G.shape
        i, j = np.meshgrid(np.arange(na - 1), np.arange(nbb - 1), indexing="ij")
        i, j = i.ravel(), j.ravel()
        F = np.stack([G[i, j], G[i + 1, j], G[i + 1, j + 1], G[i, j + 1]], 1)
        Pf = P[F]
        if uv == "xy":
            UV = Pf[..., [1, 0]] if uv_swap else Pf[..., [0, 1]]
        else:
            UV = Pf[..., rest]
        m = mat.get(_SIDE_NAMES[side], mat.get("side", 0)) if isinstance(mat, dict) else mat
        mb.add_faces(F + base, UV, m)
    return base


def lathe(mb: MeshBuilder, profile: Sequence[Sequence[float]], mat: int = 0, segments: int = 48,
          center: Sequence[float] = (0.0, 0.0, 0.0), sharp: Sequence[int] = ()) -> int:
    """Sólido de revolução em torno de Z a partir do perfil (r, z).

    Pontos com r = 0 viram polos (leque de triângulos). Índices em ``sharp`` viram
    quinas vivas (anel duplicado, normais separadas).
    """
    prof = np.asarray(profile, np.float64)
    rows: list[tuple[float, float]] = []
    connect: list[bool] = []
    for k, (r, z) in enumerate(prof):
        rows.append((r, z))
        connect.append(True)
        if k in sharp and 0 < k < len(prof) - 1:
            connect[-1] = False
            rows.append((r, z))
            connect.append(True)
    connect[-1] = False
    seglen = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff([p[0] for p in rows]), np.diff([p[1] for p in rows])))])
    th = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    cx, cy, cz = center
    start = mb.count
    ids: list[Array] = []
    for r, z in rows:
        if r < 1e-7:
            b = mb.add_verts([[cx, cy, cz + z]])
            ids.append(np.array([b]))
        else:
            b = mb.add_verts(np.stack([cx + r * np.cos(th), cy + r * np.sin(th), np.full(segments, cz + z)], 1))
            ids.append(np.arange(b, b + segments))
    j = np.arange(segments)
    j1 = (j + 1) % segments
    uj = th / (2 * math.pi)
    uj1 = np.where(j1 == 0, 1.0, th[j1] / (2 * math.pi))
    for k in range(len(rows) - 1):
        if not connect[k]:
            continue
        a, b = ids[k], ids[k + 1]
        ra, rb = rows[k][0], rows[k + 1][0]
        circ = 2 * math.pi * max(ra, rb, 1e-4)
        va, vb = seglen[k], seglen[k + 1]
        if len(a) > 1 and len(b) > 1:
            F = np.stack([a[j], a[j1], b[j1], b[j]], 1)
            UV = np.stack([np.stack([uj * circ, np.full(segments, va)], 1),
                           np.stack([uj1 * circ, np.full(segments, va)], 1),
                           np.stack([uj1 * circ, np.full(segments, vb)], 1),
                           np.stack([uj * circ, np.full(segments, vb)], 1)], 1)
        elif len(a) == 1 and len(b) > 1:
            F = np.stack([np.full(segments, a[0]), b[j1], b[j]], 1)
            UV = np.stack([np.stack([uj * circ, np.full(segments, va)], 1),
                           np.stack([uj1 * circ, np.full(segments, vb)], 1),
                           np.stack([uj * circ, np.full(segments, vb)], 1)], 1)
        elif len(a) > 1 and len(b) == 1:
            F = np.stack([a[j], a[j1], np.full(segments, b[0])], 1)
            UV = np.stack([np.stack([uj * circ, np.full(segments, va)], 1),
                           np.stack([uj1 * circ, np.full(segments, va)], 1),
                           np.stack([uj * circ, np.full(segments, vb)], 1)], 1)
        else:
            continue
        mb.add_faces(F, UV, mat)
    return start


def round_profile(points: Sequence[Sequence[float]], radii: float | Sequence[float], arc_seg: int = 4) -> list:
    """Arredonda os cantos de uma polilinha (r, z) com arcos de raio dado (filetes)."""
    P = np.asarray(points, np.float64)
    R = np.broadcast_to(np.asarray(radii, np.float64), (len(P),))
    out = [P[0]]
    for k in range(1, len(P) - 1):
        r = R[k]
        a, b, c = P[k - 1], P[k], P[k + 1]
        u1, u2 = _normalize(a - b), _normalize(c - b)
        cosang = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        ang = math.acos(cosang)
        if r <= 1e-9 or ang > math.pi - 1e-3 or ang < 1e-3:
            out.append(b)
            continue
        t = r / math.tan(ang / 2)
        t = min(t, 0.49 * np.linalg.norm(a - b), 0.49 * np.linalg.norm(c - b))
        p1, p2 = b + u1 * t, b + u2 * t
        bis = _normalize(u1 + u2)
        rr = t * math.tan(ang / 2)
        cen = b + bis * math.hypot(t, rr)
        a1 = math.atan2(*(p1 - cen)[::-1])
        a2 = math.atan2(*(p2 - cen)[::-1])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        for s in np.linspace(0.0, 1.0, arc_seg + 1):
            ang_s = a1 + da * s
            out.append(cen + rr * np.array([math.cos(ang_s), math.sin(ang_s)]))
    out.append(P[-1])
    return [tuple(p) for p in out]


def tube(mb: MeshBuilder, path: Sequence[Sequence[float]], radius: float | Sequence[float], mat: int = 0,
         seg: int = 12, closed: bool = False, caps: bool = True, up: Sequence[float] | None = None,
         ellipse: tuple[float, float] = (1.0, 1.0)) -> int:
    """Tubo ao longo de um caminho (quadros de transporte paralelo); seção elíptica opcional."""
    P = np.asarray(path, np.float64)
    n = len(P)
    r = np.broadcast_to(np.asarray(radius, np.float64), (n,))
    if closed:
        T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    else:
        T = np.gradient(P, axis=0)
    T = _normalize(T)
    ref = np.asarray(up, np.float64) if up is not None else (
        np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0]))
    N = np.zeros_like(P)
    N[0] = _normalize(ref - np.dot(ref, T[0]) * T[0])
    for i in range(1, n):
        v = N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
        N[i] = _normalize(v) if np.linalg.norm(v) > 1e-9 else N[i - 1]
    B = np.cross(T, N)
    if closed:  # distribui a torção residual para fechar o anel sem costura
        v = N[-1] - np.dot(N[-1], T[0]) * T[0]
        v = _normalize(v)
        ang = math.atan2(np.dot(np.cross(v, N[0]), T[0]), np.dot(v, N[0]))
        tw = np.linspace(0.0, ang, n + 1)[:-1]
        c, s = np.cos(tw)[:, None], np.sin(tw)[:, None]
        N, B = N * c + B * s, B * c - N * s
    phi = np.linspace(0.0, 2 * math.pi, seg, endpoint=False)
    ex, ey = ellipse
    ring = P[:, None, :] + r[:, None, None] * (np.cos(phi)[None, :, None] * ex * N[:, None, :]
                                              + np.sin(phi)[None, :, None] * ey * B[:, None, :])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    circ = 2 * math.pi * float(np.mean(r)) * max(ex, ey)
    U = np.concatenate([phi, [2 * math.pi]]) / (2 * math.pi) * circ
    if closed:
        ring_c = np.concatenate([ring, ring[:1]], 0)
        arc = np.concatenate([arc, [arc[-1] + np.linalg.norm(P[0] - P[-1])]])
        UV = np.stack(np.meshgrid(U, arc, indexing="xy"), -1)
        base = mb.grid(ring_c[:-1], UV, mat, close_u=True)
        # fecha o anel ligando a última linha à primeira
        j = np.arange(seg)
        j1 = (j + 1) % seg
        a = base + (n - 1) * seg
        F = np.stack([a + j, a + j1, base + j1, base + j], 1)
        uvc = np.stack([UV[n - 1, j], UV[n - 1, j + 1], UV[n, j + 1], UV[n, j]], 1)
        mb.add_faces(F, uvc, mat)
        return base
    UV = np.stack(np.meshgrid(U, arc, indexing="xy"), -1)
    base = mb.grid(ring, UV, mat, close_u=True)
    if caps:
        j = np.arange(seg)
        j1 = (j + 1) % seg
        for end in (0, n - 1):
            c = mb.add_verts(P[end][None])
            ringi = base + end * seg
            if end == 0:
                F = np.stack([np.full(seg, c), ringi + j1, ringi + j], 1)
            else:
                F = np.stack([np.full(seg, c), ringi + j, ringi + j1], 1)
            uvc = np.zeros((seg, 3, 2))
            mb.add_faces(F, uvc + np.array([0.0, arc[end]]), mat)
    return base


def shell(mb: MeshBuilder, P: Array, th: Array | float, UV: Array, mats: Sequence[int] = (0, 0, 0),
          rim_seg: int = 4, normals: Array | None = None) -> int:
    """Casca espessa a partir de uma superfície média em grade (tecidos, mantas).

    Gera face superior, inferior e uma borda meia-cana (arredondada) em todo o
    contorno. ``th`` pode variar por vértice (bainhas mais finas, volume irregular).
    """
    nv, nu, _ = P.shape
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    N = _normalize(np.cross(du, dv)) if normals is None else _normalize(normals)
    th = np.broadcast_to(np.asarray(th, np.float64), (nv, nu))
    h = th[..., None] / 2.0
    top, bot = P + N * h, P - N * h
    bt = mb.grid(top, UV, mats[0])
    bb = mb.grid(bot, UV, mats[1], flip=True)
    I = np.concatenate([np.zeros(nu, int), np.arange(1, nv), np.full(nu - 1, nv - 1), np.arange(nv - 2, 0, -1)])
    J = np.concatenate([np.arange(nu), np.full(nv - 1, nu - 1), np.arange(nu - 2, -1, -1), np.zeros(nv - 2, int)])
    out = np.zeros((len(I), 3))
    out[I == 0] -= dv[0, J[I == 0]]
    out[I == nv - 1] += dv[nv - 1, J[I == nv - 1]]
    out[J == 0] -= du[I[J == 0], 0]
    out[J == nu - 1] += du[I[J == nu - 1], nu - 1]
    Nb = N[I, J]
    out = _normalize(out - np.sum(out * Nb, 1, keepdims=True) * Nb)
    Pb, hb = P[I, J], h[I, J]
    L = len(I)
    rows = [bt + I * nu + J]
    for k in range(1, rim_seg):
        al = math.pi / 2 - math.pi * k / rim_seg
        R = Pb + hb * (Nb * math.sin(al) + out * math.cos(al))
        b = mb.add_verts(R)
        rows.append(np.arange(b, b + L))
    rows.append(bb + I * nu + J)
    uvb = UV[I, J]
    k = np.arange(L)
    k1 = (k + 1) % L
    faces = []
    for a, b in zip(rows[:-1], rows[1:]):
        faces.append(np.stack([a[k], b[k], b[k1], a[k1]], 1))
    F = np.concatenate(faces)
    uvc = np.tile(np.stack([uvb[k], uvb[k], uvb[k1], uvb[k1]], 1), (len(faces), 1, 1))
    # orientação: a normal da borda deve apontar para fora
    V = mb.vertices()
    q = F[nu // 2]
    nq = np.cross(V[q[1]] - V[q[0]], V[q[3]] - V[q[0]])
    if np.dot(nq, out[nu // 2]) < 0:
        F, uvc = F[:, ::-1], uvc[:, ::-1]
    mb.add_faces(F, uvc, mats[2])
    return bt


def pillow_surface(mb: MeshBuilder, w: float, d: float, t: float, mat: int, seed: int,
                   ns: int = 44, nt: int = 34, ears: float = 0.05, puff: float = 0.36,
                   wrinkle: float = 0.005) -> int:
    """Almofada/travesseiro: duas faces estufadas unidas por uma costura viva no contorno,
    lados levemente côncavos e "orelhas" nos cantos. Deitada (espessura em Z), centrada."""
    s = np.sin(np.linspace(-1.0, 1.0, ns) * math.pi / 2)
    tt = np.sin(np.linspace(-1.0, 1.0, nt) * math.pi / 2)
    S, T = np.meshgrid(s, tt)
    x = w / 2 * S * (1 - ears * (1 - T ** 2))
    y = d / 2 * T * (1 - ears * (1 - S ** 2))
    f = np.clip((1 - S ** 2) * (1 - T ** 2), 0.0, 1.0) ** puff
    edge = 1.0 - f
    seam = 0.25 * t * edge ** 3 * _fbm2(x * 5, y * 5, seed + 7, 2) * 0.2
    x = x + 0.004 * edge ** 2 * _noise2(y * 9, x * 3, seed + 3)
    y = y + 0.004 * edge ** 2 * _noise2(x * 9, y * 3, seed + 4)
    zt = seam + t / 2 * f * (1 + 0.07 * _fbm2(x * 4, y * 4, seed, 2)) + wrinkle * edge ** 1.5 * _fbm2(x * 22, y * 22, seed + 1, 2)
    zb = seam - t / 2 * f * (1 + 0.07 * _fbm2(x * 4, y * 4, seed + 2, 2)) + wrinkle * edge ** 1.5 * _fbm2(x * 22, y * 22, seed + 5, 2)
    UV = np.stack([x, y], -1)
    b = mb.grid(np.stack([x, y, zt], -1), UV, mat)
    mb.grid(np.stack([x, y, zb], -1), UV, mat, flip=True)
    return b


# =========================================================================== montagem do asset


def resolve_size(kind: str, params: dict) -> tuple[float, float, float]:
    """Dimensões finais (l, p, a): ``params["size"]`` sobre o tamanho do catálogo."""
    from ..layout.catalog import get as catalog_get

    w, d, h = catalog_get(f"proc:{kind}").size
    size = params.get("size")
    if size is not None:
        vals = [float(v) for v in size]
        if len(vals) != 3:
            raise ValueError("size deve ter 3 valores (largura, profundidade, altura)")
        if min(vals) < 0:
            raise ValueError(f"Dimensões inválidas para {kind}: {tuple(vals)}")
        w, d, h = (v if v > 0 else dflt for v, dflt in zip(vals, (w, d, h)))  # 0 = padrão
    w = float(params.get("width", w))
    d = float(params.get("depth", d))
    h = float(params.get("height", h))
    if min(w, d, h) <= 0:
        raise ValueError(f"Dimensões inválidas para {kind}: {(w, d, h)}")
    return (w, d, h)


# Nomes de padrão de tapete aceitos (inclusive os do motor de layout, em português)
_RUG_PATTERN_ALIASES = {
    "solid_border": "solid_border", "borda": "solid_border", "border": "solid_border", "liso": "solid_border",
    "solid": "solid_border", "berber": "berber", "geometrico": "berber", "geométrico": "berber",
    "geometric": "berber", "berbere": "berber", "stripes": "stripes", "listras": "stripes",
    "listrado": "stripes", "persian": "persian", "persa": "persian", "oriental": "persian",
}


def normalize_params(kind: str, params: dict) -> dict:
    """Traduz os parâmetros vindos do layout (cores da paleta) para os nomes canônicos.

    - rug: ``pattern`` em português (geometrico, listras, borda, persa), ``pattern_color``
      (= ``accent``) e ``thickness`` (= altura);
    - luminárias: ``metal_color`` -> acabamento de metal mais próximo;
    - bed: ``frame_color`` (tom da madeira -> estrutura de madeira), ``linen_color``,
      ``accent_color``, ``mattress_top``, ``headboard_depth``;
    - desk: ``top_color`` (tom do tampo) e ``leg_color`` (metal/pintura das pernas).
    ``light_on`` é ignorado de propósito: luminárias ficam apagadas (a luz da foto não muda).
    """
    p = dict(params)
    if kind == "rug":
        if "pattern" in p:
            key = str(p["pattern"]).strip().lower()
            if key not in _RUG_PATTERN_ALIASES:
                raise ValueError(f"Padrão de tapete desconhecido: {p['pattern']!r}")
            p["pattern"] = _RUG_PATTERN_ALIASES[key]
        if "pattern_color" in p and "accent" not in p:
            p["accent"] = p["pattern_color"]
        if "thickness" in p:
            p["height"] = float(p["thickness"])
    if kind in ("floor_lamp", "table_lamp") and "metal_color" in p and "metal" not in p:
        p["metal"] = M.metal_finish_for_color(p["metal_color"])
    if kind == "bed":
        if "frame_color" in p and "frame" not in p:
            p["frame"] = "wood"
        lc = p.get("linen_color")
        if lc is not None:
            p.setdefault("duvet_color", lc)
            p.setdefault("sheet_color", lc)
        if "accent_color" in p and "pillow_colors" not in p:
            p["pillow_colors"] = [lc or "#f2efe9", lc or "#f2efe9", p["accent_color"], p["accent_color"]]
    if kind == "desk":
        if "leg_color" in p and "legs" not in p:
            fin = M.metal_finish_for_color(p["leg_color"])
            p["legs"] = "metal"
            p["leg_finish"] = "matte_black" if fin == "black" else fin
    return p


def wood_or_lacquer(color, grain_axis: str, seed: int, species: str | None = None) -> bpy.types.Material:
    """Madeira no tom pedido; tons muito claros (ex.: "laca branca") viram laca acetinada."""
    rgb = M.parse_color(color, "#a47a50")
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    if lum > 0.5:
        return M.painted(color, "satin", seed)
    sp = species or ("walnut" if lum < 0.09 else "oak")
    return M.wood(sp, grain_axis, seed, tint=color)


class _Asset:
    """EMPTY raiz + peças; normaliza o conjunto para as dimensões exatas pedidas."""

    def __init__(self, kind: str, name: str, size: tuple[float, float, float], params: dict, seed: int,
                 collection: bpy.types.Collection):
        self.kind, self.name, self.size, self.params, self.seed = kind, name, size, params, seed
        self.collection = collection
        root = bpy.data.objects.new(name, None)
        root.empty_display_type = "PLAIN_AXES"
        root.empty_display_size = 0.25
        collection.objects.link(root)
        self.name = root.name  # o Blender acrescenta sufixo se o nome já existir
        root["mobilia_kind"] = kind
        root["mobilia_asset_id"] = f"proc:{kind}"
        root["mobilia_size"] = list(size)
        root["mobilia_params"] = json.dumps(params, ensure_ascii=False, default=str)
        root["mobilia_procedural"] = True
        self.root = root
        self.parts: list[bpy.types.Object] = []

    def part(self, part: str, mb: MeshBuilder, materials: Sequence[bpy.types.Material], order: int,
             motion: str, pivot: Sequence[float] = (0.0, 0.0, 0.0)) -> bpy.types.Object:
        if motion not in ASSEMBLY_MOTIONS:
            raise ValueError(f"Movimento de montagem inválido: {motion}")
        me = mb.to_mesh(f"{self.name}.{part}", origin=pivot)
        for m in materials:
            me.materials.append(m)
        obj = bpy.data.objects.new(f"{self.name}.{part}", me)
        self.collection.objects.link(obj)
        obj.location = tuple(float(v) for v in pivot)
        obj.parent = self.root
        obj["mobilia_part"] = part
        obj["assembly_order"] = int(order)
        obj["assembly_motion"] = motion
        self.parts.append(obj)
        return obj

    def finalize(self) -> None:
        """Ajusta o conjunto (escala por eixo, ~1%) para o bounding box exato pedido."""
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        data = []
        for obj in self.parts:
            co = np.empty(len(obj.data.vertices) * 3, np.float32)
            obj.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3).astype(np.float64)
            loc = np.asarray(obj.location, np.float64)
            data.append((obj, co, loc))
            lo = np.minimum(lo, (co + loc).min(0))
            hi = np.maximum(hi, (co + loc).max(0))
        target = np.asarray(self.size, np.float64)
        S = target / np.maximum(hi - lo, 1e-9)
        c0 = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        for obj, co, loc in data:
            obj.data.vertices.foreach_set("co", (co * S).astype(np.float32).ravel())
            obj.data.update()
            obj.location = tuple((loc - c0) * S)
        self.root["mobilia_fit_scale"] = [round(float(v), 5) for v in S]


def build_asset(kind: str, params: dict | None, name: str, *,
                collection: bpy.types.Collection | None = None) -> bpy.types.Object:
    """Constrói um asset procedural e devolve o EMPTY raiz (veja a docstring do módulo).

    ``kind`` aceita o tipo puro ("bed") ou o asset_id do catálogo ("proc:bed").
    """
    kind = kind.split(":", 1)[1] if kind.startswith("proc:") else kind
    if kind not in _BUILDERS:
        raise ValueError(f"Tipo de asset procedural desconhecido: {kind!r} (válidos: {', '.join(PROCEDURAL_KINDS)})")
    p = normalize_params(kind, dict(params or {}))
    size = resolve_size(kind, p)
    seed = int(p.get("seed", zlib.crc32(name.encode("utf-8")) & 0xFFFF))
    coll = collection or bpy.context.scene.collection
    asset = _Asset(kind, name, size, p, seed, coll)
    _BUILDERS[kind](asset)
    asset.finalize()
    asset.root["mobilia_parts"] = len(asset.parts)
    return asset.root


# =========================================================================== tapete


def _rug_body(mb: MeshBuilder, Lx: float, Ly: float, T: float, dense: str, long_axis: str,
              seed: int = 0) -> None:
    step = (0.01, 0.05, 1.0) if dense == "x" else (0.05, 0.01, 1.0)

    def wobble(P: Array) -> Array:  # contorno levemente irregular (tapete real não é régua)
        P = P.copy()
        ex = _smoothstep(Lx / 2 - 0.03, Lx / 2, np.abs(P[:, 0]))
        ey = _smoothstep(Ly / 2 - 0.03, Ly / 2, np.abs(P[:, 1]))
        P[:, 0] += np.sign(P[:, 0]) * ex * 0.0016 * _fbm2(P[:, 1] * 7.0, np.sign(P[:, 0]) * 3.0, seed, 2)
        P[:, 1] += np.sign(P[:, 1]) * ey * 0.0016 * _fbm2(P[:, 0] * 7.0, np.sign(P[:, 1]) * 5.0, seed + 1, 2)
        return P

    rounded_box(mb, (0.0, 0.0, T / 2), (Lx, Ly, T), T / 2,
                mat={"+z": 0, "-z": 1, "side": 0}, step=step, seg=4, guard=0.002, uv="xy",
                uv_swap=(long_axis == "y"), deform=wobble)
    # UVs começam em 0 (as bordas do desenho dependem disso)
    mb.offset_uv((Ly / 2, Lx / 2) if long_axis == "y" else (Lx / 2, Ly / 2))


def _rug_fringe(mb: MeshBuilder, Lx: float, Ly: float, fl: float, long_axis: str, seed: int) -> None:
    rng = np.random.default_rng(seed + 11)
    across = Ly if long_axis == "x" else Lx
    half_len = (Lx if long_axis == "x" else Ly) / 2
    n = max(4, int(across / 0.0085))
    ys = np.linspace(-across / 2 + 0.008, across / 2 - 0.008, n)
    ry, rz = 0.0017, 0.0009
    for end in (-1.0, 1.0):
        for k, yv in enumerate(ys):
            ln = fl * (0.9 + 0.1 * rng.random()) if k % 7 else fl
            drift = rng.normal(0.0, 0.0025)
            s = np.linspace(0.0, 1.0, 6)
            xs = end * (half_len - 0.016 + (ln + 0.016) * s)
            yy = yv + drift * s ** 1.5 + 0.0008 * np.sin(s * 7 + k)
            zz = np.full_like(s, rz)
            pts = np.stack([xs, yy, zz], 1) if long_axis == "x" else np.stack([yy, xs, zz], 1)
            tube(mb, pts, np.linspace(1.0, 0.8, 6) * ry, 0, seg=6, caps=True, up=(0.0, 0.0, 1.0),
                 ellipse=(rz / ry, 1.0))


def _build_rug(a: _Asset) -> None:
    W, D, T = a.size
    p = a.params
    pattern = str(p.get("pattern", "solid_border"))
    if pattern not in M.RUG_PATTERNS:
        raise ValueError(f"Padrão de tapete desconhecido: {pattern!r}")
    fringe = bool(p.get("fringe", pattern in ("berber", "persian")))
    long_axis = "x" if W >= D else "y"
    dense = str(p.get("unroll_axis", long_axis)).lower()
    if dense not in ("x", "y"):
        raise ValueError("unroll_axis deve ser 'x' ou 'y'")
    fl = min(float(p.get("fringe_length", 0.06)), 0.12 * max(W, D)) if fringe else 0.0
    Lx = W - (2 * fl if long_axis == "x" else 0.0)
    Ly = D - (2 * fl if long_axis == "y" else 0.0)
    Lu, Lv = (Lx, Ly) if long_axis == "x" else (Ly, Lx)
    mats = [M.rug_wool(pattern, p.get("color"), p.get("accent"), Lu, Lv, a.seed),
            M.rug_wool(pattern, p.get("color"), p.get("accent"), Lu, Lv, a.seed, back=True)]
    mb = MeshBuilder()
    _rug_body(mb, Lx, Ly, T, dense, long_axis, a.seed)
    a.part("rug_body", mb, mats, 0, "unroll")
    if fringe:
        fm = MeshBuilder()
        _rug_fringe(fm, Lx, Ly, fl, long_axis, a.seed)
        a.part("rug_fringe", fm, [M.rug_fringe(p.get("fringe_color"))], 0, "unroll")
    a.root["mobilia_rug"] = json.dumps(dict(thickness=T, long_axis=long_axis, dense_axis=dense,
                                            body=[Lx, Ly], fringe=fl, pattern=pattern, seed=a.seed))


# =========================================================================== luminárias


def _bulb_profile(z0: float, scale: float = 1.0) -> list:
    pts = [(0.0135, 0.0), (0.0145, 0.012), (0.017, 0.028), (0.026, 0.048), (0.0302, 0.066),
           (0.0295, 0.083), (0.024, 0.097), (0.013, 0.106)]
    sp = _catmull(pts, 6)
    prof = [(0.0, z0)] + [(r * scale, z0 + z * scale) for r, z in sp] + [(0.0, z0 + 0.109 * scale)]
    return prof


def _shade(mb: MeshBuilder, mat_fabric: int, mat_metal: int, z0: float, z1: float, r_bot: float,
           r_top: float, spider: bool = True) -> None:
    """Cúpula: parede fina de tecido com bordas arredondadas, frisos e aranha superior."""
    rb = 0.0026      # raio do friso (aro de arame encapado)
    wall = 0.0022
    cb = np.array([r_bot - rb, z0 + rb])
    ct = np.array([r_top - rb, z1 - rb])
    d = _normalize(ct - cb)
    nrm = np.array([d[1], -d[0]])  # para fora
    prof = []
    for t in np.linspace(0.0, 1.0, 14):           # face externa (baixo -> cima)
        prof.append(cb + (ct - cb) * t + nrm * wall / 2)
    for t in np.linspace(0.0, math.pi, 6)[1:-1]:   # dobra superior
        v = nrm * math.cos(t) + d * math.sin(t)
        prof.append(ct + v * wall / 2)
    for t in np.linspace(1.0, 0.0, 14):           # face interna (cima -> baixo)
        prof.append(cb + (ct - cb) * t - nrm * wall / 2)
    for t in np.linspace(math.pi, 2 * math.pi, 6)[1:-1]:
        v = nrm * math.cos(t) + d * math.sin(t)
        prof.append(cb + v * wall / 2)
    prof.append(prof[0])
    lathe(mb, prof, mat_fabric, segments=96)
    for c in (cb, ct):
        circ = [(c[0] * math.cos(t), c[0] * math.sin(t), c[1]) for t in np.linspace(0, 2 * math.pi, 96, endpoint=False)]
        tube(mb, circ, rb, mat_fabric, seg=10, closed=True)
    if spider:
        zt = ct[1] - 0.0005
        lathe(mb, round_profile([(0.0, zt - 0.006), (0.013, zt - 0.006), (0.013, zt + 0.002), (0.0, zt + 0.002)],
                                0.0015, 2), mat_metal, segments=24)
        for k in range(3):
            ang = 2 * math.pi * k / 3 + 0.3
            dirv = np.array([math.cos(ang), math.sin(ang), 0.0])
            pts = [tuple(dirv * 0.012 + [0, 0, zt - 0.002]), tuple(dirv * (ct[0] - 0.002) + [0, 0, zt - 0.002])]
            tube(mb, pts, 0.0017, mat_metal, seg=8)


def _build_floor_lamp(a: _Asset) -> None:
    W, D, H = a.size
    p = a.params
    finish = str(p.get("metal", "black"))
    Ds = min(W, D)
    shape = str(p.get("shade_shape", "drum"))
    shade_h = float(np.clip(0.72 * Ds, 0.18, 0.42))
    zb = H - shade_h
    r_bot = Ds / 2
    r_top = r_bot if shape == "drum" else r_bot * 0.78
    m_metal = M.metal(finish, "Z", a.seed)
    m_metal_r = M.metal(finish, "RADIAL", a.seed)
    # --- base pesada (disco torneado com filetes) + feltro
    R = min(0.15, 0.34 * Ds + 0.0)
    hb = 0.028
    base = MeshBuilder()
    prof = round_profile([(0.0, 0.0015), (R - 0.004, 0.0015), (R, 0.006), (R, hb - 0.006), (R - 0.012, hb),
                          (0.03, hb + 0.004), (0.016, hb + 0.006), (0.016, hb + 0.022), (0.0, hb + 0.022)],
                         [0, 0.002, 0.003, 0.004, 0.006, 0.01, 0.002, 0.002, 0], 4)
    lathe(base, prof, 0, segments=96)
    lathe(base, [(0.0, 0.0), (R - 0.012, 0.0), (R - 0.012, 0.0016), (0.0, 0.0016)], 1, segments=64, sharp=(1, 2))
    a.part("base", base, [m_metal_r, M.felt()], 0, "drop", pivot=(0.0, 0.0, 0.0))
    # --- haste com luva de emenda
    z_sock_top = zb + 0.3 * shade_h
    z_pole_top = z_sock_top - 0.055
    pole = MeshBuilder()
    rp = 0.0105
    lathe(pole, round_profile([(0.0, hb + 0.01), (rp, hb + 0.01), (rp, z_pole_top), (0.0, z_pole_top)], 0.0015, 2),
          0, segments=32)
    zc = 0.5 * (hb + z_pole_top)
    lathe(pole, round_profile([(0.0, zc - 0.012), (rp + 0.0025, zc - 0.012), (rp + 0.0025, zc + 0.012),
                               (0.0, zc + 0.012)], 0.0012, 2), 0, segments=32)
    # soquete
    lathe(pole, round_profile([(0.0, z_pole_top - 0.004), (0.0175, z_pole_top), (0.0175, z_sock_top),
                               (0.012, z_sock_top), (0.012, z_sock_top + 0.004), (0.0, z_sock_top + 0.004)],
                              [0, 0.002, 0.002, 0.001, 0.001, 0], 2), 0, segments=32)
    a.part("pole", pole, [m_metal], 1, "rise", pivot=(0.0, 0.0, hb + 0.01))
    # --- cúpula + lâmpada + armação (harpa e aranha)
    sh = MeshBuilder()
    _shade(sh, 0, 1, zb, H, r_bot, r_top, spider=True)
    lathe(sh, _bulb_profile(z_sock_top + 0.002), 2, segments=32)
    zt = H - 0.0026 - 0.002
    for sgn in (-1.0, 1.0):
        pts = [(sgn * 0.0175, 0.0, z_sock_top - 0.04), (sgn * 0.045, 0.0, z_sock_top + 0.01),
               (sgn * 0.045, 0.0, zt - 0.05), (sgn * 0.012, 0.0, zt - 0.006)]
        path = _catmull(pts, 10)
        tube(sh, path, 0.0016, 1, seg=8)
    tube(sh, [(-0.0175, 0, z_sock_top - 0.04), (0.0175, 0, z_sock_top - 0.04)], 0.002, 1, seg=8)
    a.part("shade", sh, [M.lamp_shade(p.get("shade_color"), a.seed), m_metal, M.opal_bulb()], 2, "drop",
           pivot=(0.0, 0.0, zb))


def _build_table_lamp(a: _Asset) -> None:
    W, D, H = a.size
    p = a.params
    Ds = min(W, D)
    finish = str(p.get("metal", "brass"))
    shape = str(p.get("shade_shape", "tapered"))
    base_kind = str(p.get("base_material", "ceramic"))
    fin_h = 0.016
    shade_h = float(np.clip(0.4 * H, 0.12, 0.3))
    z_st = H - fin_h
    z_sb = z_st - shade_h
    r_bot = Ds / 2
    r_top = r_bot if shape == "drum" else r_bot * 0.8
    m_metal = M.metal(finish, "Z", a.seed)
    if base_kind == "ceramic":
        m_body = M.ceramic(p.get("base_color"), "gloss", a.seed)
    elif base_kind == "wood":
        m_body = M.wood(str(p.get("wood", "walnut")), "Z", a.seed)
    elif base_kind in ("brass", "black", "chrome"):
        m_body = M.metal(base_kind, "Z", a.seed)
    else:
        raise ValueError(f"Material de base desconhecido: {base_kind!r}")
    # corpo em forma de moringa: proporções relativas à altura
    hb = z_sb - 0.02
    rb = min(0.3 * Ds, 0.085)
    ctrl = [(0.64 * rb, 0.006), (0.7 * rb, 0.04 * hb), (0.93 * rb, 0.2 * hb), (rb, 0.38 * hb),
            (0.92 * rb, 0.6 * hb), (0.62 * rb, 0.8 * hb), (0.36 * rb, 0.93 * hb), (0.3 * rb, hb)]
    body = _catmull(ctrl, 10)
    foot = round_profile([(0.0, 0.0), (0.6 * rb, 0.0), (0.64 * rb, 0.006)], [0, 0.003, 0], 3)
    prof = foot[:-1] + [tuple(q) for q in body] + [(0.3 * rb - 0.004, hb), (0.0, hb)]
    base = MeshBuilder()
    lathe(base, prof, 0, segments=72)
    # pescoço e soquete (metal)
    z_sock = z_sb + 0.35 * shade_h
    lathe(base, round_profile([(0.0, hb - 0.002), (0.022, hb - 0.002), (0.022, hb + 0.006), (0.009, hb + 0.012),
                               (0.009, z_sock - 0.045), (0.016, z_sock - 0.04), (0.016, z_sock),
                               (0.011, z_sock), (0.011, z_sock + 0.004), (0.0, z_sock + 0.004)],
                              [0, 0.002, 0.003, 0.002, 0.001, 0.002, 0.002, 0.001, 0.001, 0], 2),
          1, segments=32)
    a.part("base", base, [m_body, m_metal], 0, "drop", pivot=(0.0, 0.0, 0.0))
    sh = MeshBuilder()
    _shade(sh, 0, 1, z_sb, z_st, r_bot, r_top, spider=True)
    lathe(sh, _bulb_profile(z_sock + 0.002, 0.85), 2, segments=32)
    for sgn in (-1.0, 1.0):  # harpa
        pts = [(sgn * 0.016, 0.0, z_sock - 0.035), (sgn * 0.05, 0.0, z_sock + 0.01),
               (sgn * 0.05, 0.0, z_st - 0.05), (sgn * 0.012, 0.0, z_st - 0.007)]
        tube(sh, _catmull(pts, 10), 0.0015, 1, seg=8)
    lathe(sh, round_profile([(0.0, z_st - 0.003), (0.006, z_st - 0.003), (0.006, z_st + 0.002), (0.011, z_st + 0.006),
                             (0.011, H - 0.004), (0.0, H)], [0, 0.001, 0.002, 0.003, 0.004, 0], 3), 1, segments=32)
    a.part("shade", sh, [M.lamp_shade(p.get("shade_color"), a.seed), m_metal, M.opal_bulb()], 1, "drop",
           pivot=(0.0, 0.0, z_sb))


# =========================================================================== escrivaninha


def _taper(z0: float, z1: float, k_bottom: float, axes: tuple[int, ...] = (0, 1)):
    def f(P: Array) -> Array:
        P = P.copy()
        t = np.clip((P[:, 2] - z0) / (z1 - z0), 0.0, 1.0)
        s = k_bottom + (1.0 - k_bottom) * t
        for ax in axes:
            P[:, ax] *= s
        return P
    return f


def _build_desk(a: _Asset) -> None:
    W, D, H = a.size
    p = a.params
    species = str(p.get("wood", "oak"))
    legs_kind = str(p.get("legs", "wood"))
    if legs_kind not in ("wood", "black", "painted", "metal"):
        raise ValueError(f"Tipo de pernas desconhecido: {legs_kind!r}")
    finish = "matte_black" if legs_kind == "black" else str(p.get("leg_finish", "matte_black"))
    metal_legs = legs_kind in ("black", "metal")
    drawer = bool(p.get("drawer", True))
    tt = float(np.clip(0.035 * H, 0.02, 0.032))
    e = 0.03
    s_top = 0.03 if metal_legs else 0.042
    s_bot = 0.03 if metal_legs else 0.03
    ah, at = 0.085, 0.02
    top_color = p.get("top_color")

    def top_wood(axis: str, k: int) -> bpy.types.Material:
        return (wood_or_lacquer(top_color, axis, a.seed + k, species if "wood" in p else None)
                if top_color is not None else M.wood(species, axis, a.seed + k))

    m_top = top_wood("X", 0)
    if metal_legs:      # estrutura de aço (pintura eletrostática ou metal escovado)
        m_leg, m_ax, m_ay = (M.metal(finish, ax, a.seed) for ax in ("Z", "X", "Y"))
    elif legs_kind == "painted":  # estrutura pintada (ex.: branca) com tampo de madeira
        m_leg = m_ax = m_ay = M.painted(p.get("paint_color", "#efece6"), "satin", a.seed)
    else:
        m_leg, m_ax, m_ay = (top_wood(ax, k) for k, ax in enumerate(("Z", "X", "Y"), 1))
    zt0 = H - tt
    # --- pernas (afinando para baixo)
    legs = MeshBuilder()
    lx, ly = W / 2 - e - s_top / 2, D / 2 - e - s_top / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            rounded_box(legs, (sx * lx, sy * ly, zt0 / 2), (s_top, s_top, zt0), 0.0035 if not metal_legs else 0.002,
                        0, step=0.1, seg=4, deform=_taper(-zt0 / 2, zt0 / 2, s_bot / s_top))
    a.part("legs", legs, [m_leg], 0, "rise", pivot=(0.0, 0.0, 0.0))
    # --- cintas (aventais) entre as pernas
    apron = MeshBuilder()
    za = zt0 - ah / 2
    inner_x = 2 * lx - s_top + 0.004
    inner_y = 2 * ly - s_top + 0.004
    yo = D / 2 - e - at / 2 - 0.004
    xo = W / 2 - e - at / 2 - 0.004
    rounded_box(apron, (0.0, yo, za), (inner_x, at, ah), 0.002, 0, step=0.2, seg=2)        # traseira
    for sx in (-1, 1):
        rounded_box(apron, (sx * xo, 0.0, za), (at, inner_y, ah), 0.002, 1, step=0.2, seg=2)
    dw = min(0.5, 0.36 * W) if drawer else 0.0
    gap = 0.0025
    if drawer:
        seg_w = (inner_x - dw) / 2 - gap
        for sx in (-1, 1):
            cx = sx * (dw / 2 + gap + seg_w / 2)
            rounded_box(apron, (cx, -yo, za), (seg_w, at, ah), 0.002, 0, step=0.2, seg=2)
        rounded_box(apron, (0.0, -yo + at / 2 + 0.004, zt0 - 0.006), (dw + 2 * gap + 0.02, 0.012, 0.012), 0.001, 0,
                    step=0.2, seg=2)
        rounded_box(apron, (0.0, -yo + at / 2 + 0.004, zt0 - ah + 0.006), (dw + 2 * gap + 0.02, 0.012, 0.012), 0.001,
                    0, step=0.2, seg=2)
    else:
        rounded_box(apron, (0.0, -yo, za), (inner_x, at, ah), 0.002, 0, step=0.2, seg=2)
    a.part("apron", apron, [m_ax, m_ay], 1, "rise", pivot=(0.0, 0.0, zt0 - ah))
    # --- tampo
    top = MeshBuilder()
    rounded_box(top, (0.0, 0.0, H - tt / 2), (W, D, tt), 0.0045, 0, step=0.15, seg=4)
    a.part("top", top, [m_top], 2, "drop", pivot=(0.0, 0.0, zt0))
    # --- gaveta (frente + caixa + puxador)
    if drawer:
        dr = MeshBuilder()
        fh = ah - 2 * gap
        rounded_box(dr, (0.0, -yo, za), (dw, at, fh), 0.0025, 0, step=0.2, seg=4)
        depth = min(0.42, D - 2 * e - 0.08)
        bx_y = -yo + at / 2 + depth / 2
        for sx in (-1, 1):
            rounded_box(dr, (sx * (dw / 2 - 0.035), bx_y, za - 0.004), (0.012, depth, fh - 0.02), 0.001, 1, step=0.3, seg=2)
        rounded_box(dr, (0.0, bx_y, za - fh / 2 + 0.012), (dw - 0.07, depth, 0.008), 0.001, 1, step=0.3, seg=2)
        prof = round_profile([(0.0, 0.0), (0.0055, 0.0), (0.0055, 0.008), (0.012, 0.016), (0.012, 0.022), (0.0, 0.023)],
                             [0, 0.001, 0.003, 0.003, 0.004, 0], 3)
        kb = MeshBuilder()
        lathe(kb, prof, 2, segments=32)
        # eixo do puxador (+Z local) apontando para fora da frente da gaveta (-Y)
        dr.extend(kb, R=_rot_x(90.0), t=(0.0, -yo - at / 2 + 0.0005, za))
        m_front = m_ax if legs_kind == "painted" else top_wood("X", 4)
        a.part("drawer", dr, [m_front, m_ay, M.metal("brass", "RADIAL", a.seed)], 3,
               "slide", pivot=(0.0, -yo, za))


# =========================================================================== cortina


def _build_curtain(a: _Asset) -> None:
    W, D, H = a.size
    p = a.params
    sheer = bool(p.get("sheer", False))
    panels = int(p.get("panels", 2))
    if panels not in (1, 2):
        raise ValueError("panels deve ser 1 ou 2")
    coverage = float(p.get("coverage", 0.32 if panels == 2 else 1.0))
    finish = str(p.get("metal", "black"))
    m_rod = M.metal(finish, "X", a.seed)
    m_rod_r = M.metal(finish, "RADIAL", a.seed)
    rr = 0.0125
    fin_r = 0.021
    z_r = H - fin_r
    y_r = -D / 2 + 0.058
    # --- varão, ponteiras, suportes e argolas
    rod = MeshBuilder()
    x_end = W / 2 - 2 * fin_r + 0.004
    tube(rod, [(-x_end, y_r, z_r), (x_end, y_r, z_r)], rr, 0, seg=24, up=(0.0, 0.0, 1.0))
    finial = MeshBuilder()
    prof = round_profile([(0.0, 0.0), (0.009, 0.0), (0.009, 0.006), (0.011, 0.009)], [0, 0.001, 0.002, 0], 2)
    ball = _catmull([(0.011, 0.009), (0.018, 0.014), (fin_r, 0.024), (0.017, 0.035), (0.0, 0.042)], 6)
    lathe(finial, prof + [tuple(q) for q in ball[1:]], 1, segments=32)
    for sx in (-1, 1):  # eixo do torneado (+Z local) apontando para fora do varão
        rod.extend(finial, R=_rot_y(90.0 * sx), t=(sx * (x_end - 0.004), y_r, z_r))
    plate = MeshBuilder()
    lathe(plate, round_profile([(0.0, 0.0), (0.024, 0.0), (0.024, 0.006), (0.016, 0.01), (0.0, 0.01)],
                               [0, 0.001, 0.002, 0.002, 0], 2), 1, segments=32)
    yb = D / 2
    for bx in (-(W / 2 - 0.16), W / 2 - 0.16):  # suportes de parede (espelho + braço + berço)
        rod.extend(plate, R=_rot_x(90.0), t=(bx, yb, z_r + 0.004))
        tube(rod, [(bx, yb - 0.008, z_r + 0.004), (bx, y_r + 0.004, z_r + 0.004)], 0.0055, 0, seg=12)
        cradle = [(bx, y_r + (rr + 0.003) * math.sin(t), z_r - (rr + 0.003) * math.cos(t))
                  for t in np.linspace(-1.9, 1.9, 12)]
        tube(rod, cradle, 0.0025, 0, seg=8)
    # --- painéis
    usable = W - 4 * fin_r
    Wp = max(0.18, min(1.0, coverage) * usable)
    if panels == 1:
        Wp = min(Wp, usable / 1.1)  # a barra se abre ~9%: o painel continua sob o varão
        spans = [(-Wp / 2, Wp / 2, None)]
    else:
        Wp = min(Wp, usable / 2)
        spans = [(-usable / 2, -usable / 2 + Wp, "left"), (usable / 2 - Wp, usable / 2, "right")]
    z_top = z_r - rr - 0.03
    m_fab = M.fabric(p.get("color", "#e9e4da"), "linen", coords="UV", seed=a.seed,
                     translucency=0.35 if not sheer else 0.0, sheer=sheer)
    for pi, (x0, x1, fixed) in enumerate(spans):
        mb = MeshBuilder()
        rings = _curtain_panel(mb, x0, x1, y_r, z_top, D, a.seed + 31 * pi, sheer, fixed)
        for xr in rings:  # argola no varão + gancho até o cabeçalho do tecido
            circ = [(xr, y_r + 0.021 * math.sin(t), z_r - 0.006 + 0.021 * math.cos(t))
                    for t in np.linspace(0, 2 * math.pi, 20, endpoint=False)]
            tube(rod, circ, 0.0024, 0, seg=8, closed=True)
            tube(rod, [(xr, y_r, z_r - 0.026), (xr, y_r, z_top + 0.004)], 0.0012, 0, seg=6)
        name = "panel" if panels == 1 else f"panel_{fixed}"
        a.part(name, mb, [m_fab], 1, "unfold", pivot=((x0 + x1) / 2, y_r, z_top))
    a.part("rod", rod, [m_rod, m_rod_r], 0, "slide", pivot=(0.0, y_r, z_r))


def _curtain_panel(mb: MeshBuilder, x0: float, x1: float, y_r: float, z_top: float, D: float, seed: int,
                   sheer: bool, fixed: str | None) -> list[float]:
    """Painel com ondas (wave): dobras regulares no cabeçalho que, descendo, variam de
    largura e profundidade, serpenteiam e se abrem perto da barra (como tecido real)."""
    Wp = x1 - x0
    n_f = max(2, int(round(Wp / 0.12)))
    amp_max = D / 2 - 0.006 - abs(y_r)
    A0, A1 = 0.62 * amp_max, 0.8 * amp_max
    hgt = z_top
    nu = n_f * 18 + 1
    av = np.linspace(0.0, 1.0, nu)
    bv = _spaced([(0.0, 0.12, 0.012), (0.12, hgt - 0.25, 0.03), (hgt - 0.25, hgt, 0.012)])
    Aa, Bb = np.meshgrid(av, bv)  # (nv, nu)
    t = Bb / hgt                  # 0 no topo, 1 na barra
    rng = np.random.default_rng(seed)
    mix_t = _smoothstep(0.02, 0.45, t)
    # largura das dobras varia (fase deformada, monotônica) e as cristas serpenteiam
    warp = (0.22 / n_f) * mix_t * _fbm2(Aa * n_f * 0.45, Bb * 0.7, seed, 2)
    warp += (0.08 / n_f) * mix_t * np.sin(2 * math.pi * (Aa * 1.3 + rng.random()))
    ph = 2 * math.pi * n_f * (Aa + warp)
    # profundidade por dobra (interpolação suave entre dobras vizinhas)
    amp_k = 0.6 + 0.75 * rng.random(n_f * 2 + 3)
    kf = np.clip(ph / math.pi, 0.0, len(amp_k) - 2.001)
    k = kf.astype(int)
    fr = kf - k
    fr = fr * fr * (3 - 2 * fr)
    ampv = amp_k[k] * (1 - fr) + amp_k[k + 1] * fr
    flare = _smoothstep(0.82, 1.0, t)
    amp = (A0 + (A1 - A0) * t ** 0.7) * (1.0 + (ampv - 1.0) * mix_t) * (1.0 + 0.12 * flare)
    shape = np.sin(ph) + 0.16 * np.sin(2 * ph + 0.7)
    yv = amp * shape
    # a dobra mais funda encosta na frente do volume pedido; atrás, folga para a parede
    yv = y_r + yv * ((y_r + D / 2 - 0.003) / max(-yv.min(), 1e-6))
    yv = np.minimum(yv, D / 2 - 0.008)
    spread = 1.0 + 0.04 * _smoothstep(0.15, 0.85, t) + 0.05 * flare
    xc = (x0 + x1) / 2
    xv = xc + (Aa - 0.5) * Wp * spread
    # a borda externa do painel fica no lugar (cai a prumo junto à parede lateral)
    if fixed == "left":
        xv = xv + (x0 - xv[:, :1]) * (1 - Aa) ** 4
    elif fixed == "right":
        xv = xv + (x1 - xv[:, -1:]) * Aa ** 4
    # barra pousada no piso: leve ondulação vertical só acima dela
    zv = z_top - Bb
    P = np.stack([xv, yv, zv], -1)
    # u = comprimento de arco ao longo da largura (o trançado acompanha as dobras)
    seg = np.linalg.norm(np.diff(P, axis=1), axis=-1)
    U = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(seg, 1)], 1)
    UV = np.stack([U, z_top - zv], -1)
    hem = 0.075
    th = np.where((Bb > hgt - hem) | (Bb < 0.06), 0.0045, 0.0015)
    if sheer:
        mb.grid(P, UV, 0)
        idx = np.flatnonzero(Bb[:, 0] > hgt - hem)
        shell(mb, P[idx[0] - 1:], 0.0045, UV[idx[0] - 1:], (0, 0, 0), rim_seg=2)
        idx = np.flatnonzero(Bb[:, 0] < 0.06)
        shell(mb, P[: idx[-1] + 2], 0.0045, UV[: idx[-1] + 2], (0, 0, 0), rim_seg=2)
    else:
        shell(mb, P, th, UV, (0, 0, 0), rim_seg=2)
    # argolas nos cruzamentos de zero do topo (onde o tecido passa sob o varão)
    xs_top = xv[0]
    return [float(np.interp(j / (2 * n_f), av, xs_top)) for j in range(0, 2 * n_f + 1)]


# =========================================================================== manta


def _build_throw(a: _Asset) -> None:
    """Manta de tricô dobrada em camadas (zigue-zague), macia, com franjas nas pontas."""
    W, D, H = a.size
    p = a.params
    tassels = bool(p.get("tassels", True))
    tl_len = 0.05 if tassels else 0.0
    N = 3 if H <= 0.075 else 4
    tl = H / N
    th = tl * 0.8
    Wl = W - 2 * tl_len
    Ye = D / 2 - tl / 2 - th / 2 - 0.005
    # caminho (y, z) da linha média dobrada em zigue-zague
    path: list[tuple[float, float]] = []
    for k in range(N):
        z = tl * (k + 0.5)
        dirn = -1.0 if k % 2 == 0 else 1.0
        y_from = Ye * 0.8 if k == 0 else -dirn * Ye
        y_to = dirn * Ye if k < N - 1 else dirn * Ye * 0.88
        ys = np.linspace(y_from, y_to, max(3, int(abs(y_to - y_from) / 0.01)))
        path += [(y, z) for y in ys]
        if k < N - 1:
            cz = z + tl / 2
            cy = dirn * Ye
            for ang in np.linspace(-math.pi / 2, math.pi / 2, 16)[1:-1]:
                path.append((cy + dirn * (tl / 2) * math.cos(ang), cz + (tl / 2) * math.sin(ang)))
    path_arr = np.asarray(path)
    seg = np.linalg.norm(np.diff(path_arr, axis=0), axis=1)
    path_arr = path_arr[np.concatenate([[True], seg > 1e-5])]
    m = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1))])
    xs = _spaced([(-Wl / 2, Wl / 2, 0.015)])
    X, Mm = np.meshgrid(xs, m)
    Y = np.repeat(path_arr[:, 0:1], len(xs), 1)
    Z = np.repeat(path_arr[:, 1:2], len(xs), 1)
    s = a.seed
    layer = np.clip(np.round(Z / tl - 0.5), 0, N - 1)
    top = layer == N - 1
    # maciez: ondulação comum às camadas, mais solta na de cima; dobras irregulares
    wave = 0.0025 * _fbm2(X * 3.0, Y * 3.0, s, 2)
    loose = 0.003 * _fbm2(X * 4.5 + 7.0, Y * 4.5, s + 4, 2) * top * _smoothstep(0.0, 0.05, Ye - np.abs(Y))
    Z = Z + wave + np.maximum(loose, -0.0015) - 0.003 * (1 - (2 * X / Wl) ** 2)
    fold_wob = 0.008 * _noise2(X * 3.5, layer * 3.1, s + 1)
    Y = Y + fold_wob * _smoothstep(Ye * 0.75, Ye, np.abs(Y))
    X = X + 0.008 * _noise2(layer * 2.3, Y * 2.0, s + 2)
    P = np.stack([X, Y, Z], -1)
    thv = th * (1.0 + 0.1 * _fbm2(X * 5, Mm * 5, s + 3, 2))
    UV = np.stack([X + Wl / 2, Mm], -1)
    mb = MeshBuilder()
    shell(mb, P, thv, UV, (0, 0, 0), rim_seg=5)
    mats = [M.knit(p.get("color"), p.get("accent"), str(p.get("pattern", "solid")), s, width=Wl)]
    if tassels:
        mats.append(M.fabric(p.get("color", "#c9b49b"), "cotton", coords="UV", seed=s))
        rng = np.random.default_rng(s)
        for k in range(N):
            zk = tl * (k + 0.5)
            ys = np.arange(-Ye + 0.03, Ye - 0.02, 0.04)
            for sx in (-1.0, 1.0):
                for yv in ys:
                    ln = tl_len * (0.85 + 0.15 * rng.random())
                    droop = min(zk - 0.005, 0.035)
                    tt = np.linspace(0.0, 1.0, 8)
                    xp = sx * (Wl / 2 - 0.012 + (ln + 0.012) * tt)
                    zp = np.maximum(zk - droop * tt ** 2, 0.004)
                    yp = yv + rng.normal(0, 0.004) * tt ** 2
                    rad = 0.004 * (1.0 - 0.45 * tt) * (1.0 + 0.25 * np.exp(-((tt - 0.18) / 0.06) ** 2))
                    tube(mb, np.stack([xp, yp, zp], 1), rad, 1, seg=8)
    a.part("throw", mb, mats, 0, "settle", pivot=(0.0, 0.0, 0.0))


# =========================================================================== cama


def _build_bed(a: _Asset) -> None:
    W, D, H = a.size
    p = a.params
    s = a.seed
    frame_style = str(p.get("frame", "upholstered"))
    if frame_style not in ("upholstered", "wood"):
        raise ValueError(f"Estrutura de cama desconhecida: {frame_style!r}")
    wood_frame = frame_style == "wood"
    fab_kind = str(p.get("fabric", "linen"))
    m_fab = M.fabric(p.get("fabric_color", "#b9b1a4"), fab_kind, coords="UV", seed=s)
    m_duvet = M.bed_linen(p.get("duvet_color", "#f2efe9"), 1.0, s)
    m_sheet = M.bed_linen(p.get("sheet_color", "#f4f2ee"), 0.6, s + 1)
    legs_kind = str(p.get("legs", "wood"))
    frame_color = p.get("frame_color", "#a47a50")
    species = p.get("wood")

    def wmat(axis: str, k: int) -> bpy.types.Material:
        return (wood_or_lacquer(frame_color, axis, s + k, species) if wood_frame or "frame_color" in p
                else M.wood(str(species or "oak"), axis, s + k))

    m_leg = M.metal("matte_black", "Z", s) if legs_kind == "black" else wmat("Z", 0)
    dims = bed_dimensions(a.size, {**p, "frame": frame_style})
    hb_t, leg_h, pz1 = dims["hb_t"], dims["leg_h"], dims["deck_z"]
    mh = MATTRESS_HEIGHT
    if wood_frame:
        _bed_wood_frame(a, W, D, H, hb_t, leg_h, pz1, wmat, m_leg)
    else:
        _bed_upholstered_frame(a, W, D, H, hb_t, leg_h, pz1, m_fab, m_leg, legs_kind,
                               str(p.get("headboard", "channel")))
    # --- colchão (com lençol de elástico): a largura deixa o edredom dobrado caber em W
    mx, my0, my1 = dims["mx"], dims["my0"], dims["my1"]
    mz1 = pz1 + mh
    rm = MATTRESS_EDGE_RADIUS
    hb_y0 = D / 2 - hb_t
    mat_mb = MeshBuilder()

    def crown(P: Array) -> Array:  # tampo levemente abaulado (colchão real não é tábua)
        P = P.copy()
        top = _smoothstep(0.0, mh / 2, P[:, 2])
        P[:, 2] += 0.012 * top * (1 - (P[:, 0] / mx) ** 2).clip(0) * (1 - (2 * P[:, 1] / (my1 - my0)) ** 4).clip(0)
        return P

    rounded_box(mat_mb, (0.0, (my0 + my1) / 2, pz1 + mh / 2), (2 * mx, my1 - my0, mh), rm, 0, step=0.06, seg=6,
                deform=crown)
    a.part("mattress", mat_mb, [m_sheet], 2, "drop", pivot=(0.0, (my0 + my1) / 2, pz1))
    # --- edredom com dobra na cabeceira
    y_fold = dims["y_fold"]
    duv = MeshBuilder()
    _duvet(duv, mx, my0, y_fold, dims["cuff"], mz1, rm, pz1, s)
    a.part("duvet", duv, [m_duvet], 3, "settle", pivot=(0.0, (my0 + y_fold) / 2, mz1))
    # --- travesseiros e almofadas
    colors = list(p.get("pillow_colors", []))
    n_p = int(np.clip(int(p.get("pillows", 4)), 2, 4))
    _pillows(a, n_p, colors, mx, my1, mz1, hb_y0, H, s)


def _bed_upholstered_frame(a: _Asset, W: float, D: float, H: float, hb_t: float, leg_h: float, pz1: float,
                           m_fab: bpy.types.Material, m_leg: bpy.types.Material, legs_kind: str,
                           style: str) -> None:
    """Base e cabeceira estofadas (gomos verticais ou painel) sobre pés curtos."""
    hb_y0 = D / 2 - hb_t
    legs = MeshBuilder()
    lx, ly0, ly1 = W / 2 - 0.09, -D / 2 + 0.09, hb_y0 - 0.06
    for sx in (-1, 1):
        for yy in (ly0, ly1):
            if legs_kind == "black":
                rounded_box(legs, (sx * lx, yy, leg_h / 2 + 0.001), (0.03, 0.03, leg_h + 0.002), 0.002, 0, step=0.1)
            else:
                lathe(legs, round_profile([(0.0, 0.0), (0.016, 0.0), (0.022, leg_h + 0.004), (0.0, leg_h + 0.004)],
                                          [0, 0.002, 0.001, 0], 2), 0, segments=24,
                      center=(sx * lx, yy, 0.0))
    a.part("legs", legs, [m_leg], 0, "rise", pivot=(0.0, 0.0, 0.0))
    frame = MeshBuilder()
    fy0, fy1 = -D / 2, hb_y0 + 0.01
    rounded_box(frame, (0.0, (fy0 + fy1) / 2, (leg_h + pz1) / 2), (W - 0.006, fy1 - fy0, pz1 - leg_h), 0.03, 0,
                step=0.15, seg=6)
    a.part("frame", frame, [m_fab], 0, "rise", pivot=(0.0, (fy0 + fy1) / 2, leg_h))
    head = MeshBuilder()
    hz0, hz1 = leg_h, H
    hyc = D / 2 - hb_t / 2
    if style == "channel":
        n_ch = max(3, int(round(W / 0.21)))
        cw = W / n_ch
        rr = min(0.045, hb_t * 0.5, cw * 0.4)

        def bulge(P: Array) -> Array:
            P = P.copy()
            xr = np.clip(P[:, 0] / (cw / 2), -1, 1)
            zr = (P[:, 2] - (-(hz1 - hz0) / 2)) / (hz1 - hz0)
            front = np.clip(-P[:, 1] / (hb_t / 2), 0, 1)
            P[:, 1] -= 0.012 * (1 - xr ** 2) * front * _smoothstep(0.0, 0.08, zr) * _smoothstep(1.0, 0.92, zr)
            return P

        for k in range(n_ch):
            cx = -W / 2 + cw * (k + 0.5)
            rounded_box(head, (cx, hyc, (hz0 + hz1) / 2), (cw + 0.002, hb_t, hz1 - hz0), rr, 0,
                        step=(0.03, 0.05, 0.05), seg=8, deform=bulge)
    elif style == "panel":
        def bulge_p(P: Array) -> Array:
            P = P.copy()
            xr = np.clip(P[:, 0] / (W / 2), -1, 1)
            zr = (P[:, 2] + (hz1 - hz0) / 2) / (hz1 - hz0)
            front = np.clip(-P[:, 1] / (hb_t / 2), 0, 1)
            P[:, 1] -= 0.015 * (1 - xr ** 4) * front * np.clip(np.sin(np.clip(zr, 0, 1) * math.pi), 0, 1) ** 0.5
            return P

        rounded_box(head, (0.0, hyc, (hz0 + hz1) / 2), (W, hb_t, hz1 - hz0), min(0.045, hb_t * 0.5), 0,
                    step=(0.05, 0.05, 0.05), seg=8, deform=bulge_p)
    else:
        raise ValueError(f"Cabeceira desconhecida: {style!r}")
    a.part("headboard", head, [m_fab], 1, "rise", pivot=(0.0, hyc, hz0))


def _bed_wood_frame(a: _Asset, W: float, D: float, H: float, hb_t: float, leg_h: float, pz1: float,
                    wmat: Callable[[str, int], bpy.types.Material], m_leg: bpy.types.Material) -> None:
    """Estrutura de madeira maciça: pés-colunas nos cantos, travessas laterais e do pé,
    cabeceira de tábuas horizontais com frestas finas (veio acompanha cada peça)."""
    hb_y0 = D / 2 - hb_t
    post = 0.055
    rail_t = 0.028
    rail_h = pz1 - leg_h + 0.012
    rz = leg_h + rail_h / 2 - 0.006
    legs = MeshBuilder()
    for sx in (-1, 1):
        for yy in (-D / 2 + post / 2, hb_y0 - post / 2):
            rounded_box(legs, (sx * (W / 2 - post / 2), yy, pz1 / 2), (post, post, pz1), 0.005, 0, step=0.2,
                        seg=4, deform=_taper(-pz1 / 2, pz1 / 2, 0.82))
    a.part("legs", legs, [m_leg], 0, "rise", pivot=(0.0, 0.0, 0.0))
    frame = MeshBuilder()
    ry0, ry1 = -D / 2 + post - 0.004, hb_y0 - post + 0.004
    for sx in (-1, 1):  # travessas laterais (veio em Y)
        rounded_box(frame, (sx * (W / 2 - rail_t / 2 - 0.006), (ry0 + ry1) / 2, rz), (rail_t, ry1 - ry0, rail_h),
                    0.004, 0, step=0.2, seg=4)
    rounded_box(frame, (0.0, -D / 2 + rail_t / 2 + 0.006, rz), (W - 2 * post + 0.008, rail_t, rail_h), 0.004, 1,
                step=0.2, seg=4)  # travessa do pé (veio em X)
    rounded_box(frame, (0.0, (ry0 + ry1) / 2, pz1 - 0.012), (W - 2 * rail_t - 0.02, ry1 - ry0 - 0.02, 0.018),
                0.002, 1, step=0.3, seg=2)  # estrado (só aparece nas frestas)
    a.part("frame", frame, [wmat("Y", 1), wmat("X", 2)], 0, "rise", pivot=(0.0, 0.0, leg_h))
    head = MeshBuilder()
    z0 = pz1 - 0.06
    n = max(3, int(round((H - z0) / 0.16)))
    gap = 0.007
    ph = (H - z0 - gap * (n - 1)) / n
    for k in range(n):  # tábuas da cabeceira, apoiadas nos pés traseiros
        zc = z0 + ph / 2 + k * (ph + gap)
        rounded_box(head, (0.0, D / 2 - hb_t / 2, zc), (W, hb_t, ph), 0.005, 0, step=0.2, seg=4)
    a.part("headboard", head, [wmat("X", 3)], 1, "rise", pivot=(0.0, D / 2 - hb_t / 2, z0))




def _duvet_map(av: Array, bv: Array, g: dict) -> tuple[Array, Array]:
    """Superfície média do edredom (camada de baixo) no referencial da cama + normal."""
    a0, Rb, zc, yf0, Lc = g["a0"], g["Rb"], g["zc"], g["yf0"], g["Lc"]
    lb = Rb * math.pi / 2
    sx = np.where(av < 0, -1.0, 1.0)
    dx = np.abs(av) - a0
    dyf = yf0 - bv

    def drape(ell):
        beta = np.clip(ell / Rb, 0.0, math.pi / 2)
        hh = Rb * np.sin(beta)
        z = zc + Rb * np.cos(beta) - np.maximum(ell - lb, 0.0)
        return hh, z, np.sin(beta), np.cos(beta)

    P = np.zeros(av.shape + (3,))
    N = np.zeros(av.shape + (3,))
    top = (dx <= 0) & (dyf <= 0)
    P[top] = np.stack([av[top], bv[top], np.full(top.sum(), zc + Rb)], -1)
    N[top] = (0.0, 0.0, 1.0)
    side = (dx > 0) & (dyf <= 0)
    hh, z, nh, nz = drape(dx[side])
    P[side] = np.stack([sx[side] * (a0 + hh), bv[side], z], -1)
    N[side] = np.stack([sx[side] * nh, np.zeros_like(nh), nz], -1)
    foot = (dx <= 0) & (dyf > 0)
    hh, z, nh, nz = drape(dyf[foot])
    P[foot] = np.stack([av[foot], yf0 - hh, z], -1)
    N[foot] = np.stack([np.zeros_like(nh), -nh, nz], -1)
    cor = (dx > 0) & (dyf > 0)
    u = dx[cor] / Lc
    v = dyf[cor] / Lc
    rho = np.maximum(u, v)
    psi = np.where(u >= v, (math.pi / 4) * v / np.maximum(u, 1e-12), math.pi / 2 - (math.pi / 4) * u / np.maximum(v, 1e-12))
    ell = rho * Lc
    hh, z, nh, nz = drape(ell)
    # canto: o tecido que sobra abre em cone (mais largo perto da barra)
    hh = hh + g["flare"] * np.maximum(ell - lb, 0.0) * np.sin(2 * psi) ** 2
    P[cor] = np.stack([sx[cor] * (a0 + hh * np.cos(psi)), yf0 - hh * np.sin(psi), z], -1)
    N[cor] = np.stack([sx[cor] * nh * np.cos(psi), -nh * np.sin(psi), nz], -1)
    return P, N


def _duvet(mb: MeshBuilder, mx: float, my0: float, y_fold: float, cuff: float, mz1: float, rm: float,
           pz1: float, seed: int) -> None:
    """Edredom: cai sobre as laterais e o pé com cantos em cone, dobra (meia-volta) e aba
    sobre si mesmo na cabeceira, volume irregular, amassados e pregas verticais."""
    th0 = DUVET_THICKNESS
    gap = DUVET_GAP
    Rb = rm + gap + th0 / 2
    zc = mz1 - rm
    a0 = mx - rm
    yf0 = my0 + rm
    hang = max(0.05, (zc - (pz1 + 0.04)))
    Lc = Rb * math.pi / 2 + hang
    g = dict(a0=a0, Rb=Rb, zc=zc, yf0=yf0, Lc=Lc, flare=0.12)
    Ru = DUVET_TURN * th0
    amax = a0 + Lc
    a_coords = _spaced([(-amax, -a0 - 0.02, 0.016), (-a0 - 0.02, -a0 + Rb * 1.7, 0.007),
                        (-a0 + Rb * 1.7, a0 - Rb * 1.7, 0.018), (a0 - Rb * 1.7, a0 + 0.02 + Rb * 1.6, 0.007),
                        (a0 + 0.02 + Rb * 1.6, amax, 0.016)])
    a_coords = np.unique(np.concatenate([a_coords, [-a0, a0]]))
    b0 = yf0 - Lc
    b_turn = y_fold + math.pi * Ru
    b_end = b_turn + cuff
    b_coords = _spaced([(b0, yf0 - Rb * 1.7, 0.016), (yf0 - Rb * 1.7, yf0 + 0.03, 0.007),
                        (yf0 + 0.03, y_fold - 0.03, 0.018), (y_fold - 0.03, b_turn + 0.02, 0.005),
                        (b_turn + 0.02, b_end, 0.014)])
    b_coords = np.unique(np.concatenate([b_coords, [yf0, y_fold, b_turn]]))
    A, B = np.meshgrid(a_coords, b_coords)
    base_b = np.where(B <= y_fold, B, np.where(B <= b_turn, y_fold, y_fold - (B - b_turn)))
    P, N = _duvet_map(A, base_b, g)
    # dobra (meia-volta) e aba sobre a camada de baixo
    turn = (B > y_fold) & (B <= b_turn)
    beta = np.clip((B - y_fold) / Ru, 0.0, math.pi)
    Ty = np.array([0.0, 1.0, 0.0])
    P = np.where(turn[..., None], P + Ty * (Ru * np.sin(beta))[..., None] + N * (Ru * (1 - np.cos(beta)))[..., None], P)
    cuffm = B > b_turn
    P = np.where(cuffm[..., None], P + N * (2 * Ru), P)
    # volume e amassados: ondulação larga no topo, rugas finas, pregas verticais nas quedas
    hang_w = 1.0 - _smoothstep(0.0, 0.5, N[..., 2])
    # o ruído usa a coordenada da camada de baixo: a aba acompanha o relevo sob ela
    # (camadas nunca se atravessam); só a ruga fina é própria de cada camada
    n_top = (0.011 * _fbm2(A * 2.4, base_b * 2.1, seed, 3)
             + 0.007 * (1.0 - np.abs(_fbm2(A * 5.0 + 3.1, base_b * 3.4, seed + 5, 2))) ** 4
             + 0.0015 * _fbm2(A * 19.0, B * 17.0, seed + 1, 2))
    along = np.where(np.abs(N[..., 0]) > np.abs(N[..., 1]), base_b, A)
    depth = np.clip((zc - P[..., 2]) / max(hang, 1e-3), 0.0, 1.0)
    n_side = (0.003 + 0.011 * depth) * np.abs(_fbm2(along * 6.5, P[..., 2] * 1.5, seed + 2, 2))
    disp = n_top * (1 - hang_w) + n_side * hang_w
    disp = disp * (1.0 - 0.6 * turn)
    P = P + N * disp[..., None]
    # espessura: estufado irregular, bainha mais fina no contorno
    edge = np.minimum.reduce([A + amax, amax - A, B - b0, b_end - B])
    loft = 1.0 + 0.2 * _fbm2(A * 3.5, B * 3.5, seed + 3, 2)
    th = th0 * loft * (0.32 + 0.68 * _smoothstep(0.0, 0.07, edge))
    UV = np.stack([A, B], -1)
    shell(mb, P, th, UV, (0, 0, 0), rim_seg=5)


def _pillows(a: _Asset, n_p: int, colors: list, mx: float, my1: float, mz1: float, hb_y0: float, H: float,
             seed: int) -> None:
    """Travesseiros encostados na cabeceira (+ almofadas decorativas, conforme ``n_p``)."""
    width_avail = 2 * mx
    pair = width_avail >= 1.2
    xs = [-width_avail / 4, width_avail / 4] if pair else [0.0]
    dflt = ["#f2efe9", "#f2efe9", "#d9cfc1", "#d9cfc1"]
    cols = (colors + dflt)[:4]
    order = 4
    lean_y = hb_y0 + 0.01      # superfície onde a fileira de trás se apoia
    head_room = H - mz1 - 0.025

    def place(mb: MeshBuilder, tilt: float, y_back: float, x: float) -> tuple[Array, Array]:
        # reclina o travesseiro até ele caber abaixo do topo da cabeceira
        V0 = mb.vertices().copy()
        for tl in np.arange(tilt, 14.0, -4.0):
            V = V0 @ _rot_x(float(tl)).T
            if V[:, 2].max() - V[:, 2].min() <= H - mz1 - 0.012:
                break
        mb.set_vertices(V)
        lo, hi = mb.bounds()
        mb.transform(None, (x, y_back - hi[1], mz1 - lo[2] - 0.006))
        return mb.bounds()

    if n_p == 4:
        # almofadas decorativas retangulares encostadas na cabeceira (cabem abaixo do topo)
        ch = float(np.clip(head_room / 1.03, 0.25, 0.42))
        cw = min(0.6, width_avail / len(xs) - 0.08)
        fronts = []
        for i, x in enumerate(xs):
            mb = MeshBuilder()
            pillow_surface(mb, cw, ch, 0.16, 0, seed + 40 + i, ears=0.045, puff=0.32)
            lo, hi = place(mb, 90.0 - 13.0, lean_y, x)
            fronts.append(lo[1] + 0.35 * (hi[1] - lo[1]))
            a.part(f"pillow_back_{i}", mb, [M.bed_linen(cols[2 + i % 2], 0.7, seed + 50 + i)], order, "drop",
                   pivot=(x, (lo[1] + hi[1]) / 2, lo[2]))
        lean_y = min(fronts)
        order += 1
    # travesseiros de dormir, mais deitados, apoiados na fileira de trás
    pw = min(0.68, width_avail / len(xs) - 0.06)
    tilt = 42.0 if n_p == 4 else 62.0
    fronts = []
    for i, x in enumerate(xs):
        mb = MeshBuilder()
        pillow_surface(mb, pw, 0.46, 0.13, 0, seed + 60 + i, ears=0.05, puff=0.4)
        lo, hi = place(mb, tilt, lean_y, x * 0.985)
        fronts.append(lo[1])
        a.part(f"pillow_{i}", mb, [M.bed_linen(cols[i % 2], 0.8, seed + 70 + i)], order, "drop",
               pivot=(x, (lo[1] + hi[1]) / 2, lo[2]))
    order += 1
    if n_p == 3:  # almofada lombar à frente, no centro
        mb = MeshBuilder()
        pillow_surface(mb, min(0.55, width_avail * 0.4), 0.3, 0.12, 0, seed + 80, ears=0.05, puff=0.33)
        lo, hi = place(mb, 58.0, min(fronts) + 0.07, 0.0)
        a.part("pillow_lumbar", mb, [M.bed_linen(cols[2], 0.6, seed + 81)], order, "settle",
               pivot=(0.0, (lo[1] + hi[1]) / 2, lo[2]))


# =========================================================================== pilha de livros

# Proporções de cada livro em relação ao de baixo (largura, profundidade, espessura): o maior
# fica embaixo, como um designer empilha. Giros (graus) e deslocamentos (fração de l/p) leves.
_BOOK_REL = ((1.0, 1.0, 1.1), (0.93, 0.88, 0.95), (0.84, 0.8, 0.85))
_BOOK_COVERS = ("#33414f", "#b8a68b", "#7c3f30")


def _book_stack_layout(W: float, D: float, H: float, n: int, seed: int) -> list[dict]:
    """Dimensões, giro e posição de cada livro para que a pilha ocupe EXATAMENTE (W, D, H)."""
    rng = np.random.default_rng(seed)
    rel = _BOOK_REL[:n]
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    ang = [sgn * rng.uniform(0.5, 2.5)] + [(-sgn if i % 2 else sgn) * rng.uniform(4.0, 9.0) for i in range(1, n)]
    off = [(0.0, 0.0)] + [(rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04)) for _ in range(1, n)]
    tsum = sum(r[2] for r in rel)
    bw, bd = W, D
    for _ in range(12):     # ponto fixo: ajusta o livro de baixo até a pilha caber no envelope
        pts = []
        for (rw, rd, _), a, (ox, oy) in zip(rel, ang, off):
            hx, hy = rw * bw / 2, rd * bd / 2
            c = np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]]) @ _rot_z(a).T
            pts.append(c[:, :2] + [ox * W, oy * D])
        P = np.concatenate(pts)
        ex, ey = np.ptp(P[:, 0]), np.ptp(P[:, 1])
        cx, cy = (P[:, 0].max() + P[:, 0].min()) / 2, (P[:, 1].max() + P[:, 1].min()) / 2
        bw, bd = bw * W / ex, bd * D / ey
    books, z = [], 0.0
    for i, ((rw, rd, rt), a, (ox, oy)) in enumerate(zip(rel, ang, off)):
        t = H * rt / tsum
        books.append(dict(w=rw * bw, d=rd * bd, t=t, yaw=a, x=ox * W - cx, y=oy * D - cy, z=z,
                          spine=-1.0 if i % 2 == 0 else 1.0))
        z += t
    return books


def _book(mb: MeshBuilder, w: float, d: float, t: float, spine: float) -> None:
    """Livro de capa dura deitado (base em z = 0, centrado): duas capas, lombada no lado
    ``spine`` (-1 = -Y, 1 = +Y) e o miolo de páginas recuado nos outros três lados."""
    c = float(np.clip(0.1 * t, 0.0018, 0.003))       # espessura da capa
    inset = min(0.004, 0.03 * d)                     # recuo do miolo em relação às capas
    r = 0.45 * c
    for zc in (c / 2, t - c / 2):                    # capas de baixo e de cima
        rounded_box(mb, (0.0, 0.0, zc), (w, d, c), r, 0, step=0.1, seg=2)
    rounded_box(mb, (0.0, spine * (d / 2 - c / 2), t / 2), (w, c, t), r, 0, step=0.1, seg=2)   # lombada
    pd = d - c - inset
    rounded_box(mb, (0.0, spine * (d / 2 - c - pd / 2 + 0.0002), t / 2),
                (w - 2 * inset, pd + 0.0004, t - 2 * c + 0.0006), 0.0008, 1, step=0.1, seg=2)


def _build_book_stack(a: _Asset) -> None:
    """Pilha de 2–3 livros deitados, levemente girados e deslocados, capas na paleta."""
    W, D, H = a.size
    p = a.params
    n = int(p.get("count", 3))
    if n not in (2, 3):
        raise ValueError(f"Pilha de livros aceita 2 ou 3 livros, não {n}")
    colors = list(p.get("colors") or [])
    pages = M.paper_edges(p.get("page_color"), a.seed)
    for i, b in enumerate(_book_stack_layout(W, D, H, n, a.seed)):
        cover = colors[i] if i < len(colors) else _BOOK_COVERS[i]
        mb = MeshBuilder()
        _book(mb, b["w"], b["d"], b["t"], b["spine"])
        mb.transform(_rot_z(b["yaw"]), (b["x"], b["y"], b["z"]))
        a.part(f"book_{i}", mb, [M.fabric(cover, "cotton", seed=a.seed + i), pages], i, "drop",
               pivot=(b["x"], b["y"], b["z"]))


# =========================================================================== registro

_BUILDERS: dict[str, Callable[[_Asset], None]] = {
    "rug": _build_rug,
    "floor_lamp": _build_floor_lamp,
    "table_lamp": _build_table_lamp,
    "bed": _build_bed,
    "curtain": _build_curtain,
    "desk": _build_desk,
    "throw_blanket": _build_throw,
    "book_stack": _build_book_stack,
}
assert set(_BUILDERS) == set(PROCEDURAL_KINDS)


# =========================================================================== tapete: desenrolar


def _ease(t: Array) -> Array:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rug_roll_positions(co: Array, axis: int, sign: float, thickness: float, s_flat: float, s_end: float,
                       pitch: float, r0: float) -> Array:
    """Posições de um tapete parcialmente enrolado (espiral de Arquimedes sobre o piso).

    ``co`` = coordenadas de repouso (tapete plano) no referencial do asset. Material com
    coordenada s <= ``s_flat`` fica plano; o restante forma o rolo, que toca o piso em
    s_flat e tem o avesso para fora (o lado do pelo fica para dentro, como no real).
    """
    out = co.copy()
    s = sign * co[:, axis]
    w = co[:, 2] - thickness / 2
    R = max(s_end - s_flat, 0.0)
    b = pitch / math.pi
    r_out = math.sqrt(r0 * r0 + b * R)
    roll = s > s_flat
    u = s[roll] - s_flat
    r_u = np.sqrt(r0 * r0 + b * np.maximum(R - u, 0.0))
    theta = (2 * math.pi / pitch) * (r_out - r_u)
    phi = -math.pi / 2 + theta
    rho = r_u - w[roll]
    zc = r_out + thickness / 2
    out[roll, axis] = sign * (s_flat + rho * np.cos(phi))
    out[roll, 2] = zc + rho * np.sin(phi)
    return out


def _rug_parts(root: bpy.types.Object) -> list[bpy.types.Object]:
    return [c for c in root.children if c.get("mobilia_part") in ("rug_body", "rug_fringe")]


def _rebuild_rug_body(root: bpy.types.Object, axis: str) -> None:
    info = json.loads(root["mobilia_rug"])
    body = next(c for c in root.children if c.get("mobilia_part") == "rug_body")
    Lx, Ly = info["body"]
    mb = MeshBuilder()
    _rug_body(mb, Lx, Ly, info["thickness"], axis, info["long_axis"], int(info.get("seed", 0)))
    old = body.data
    me = mb.to_mesh(old.name, origin=(0.0, 0.0, 0.0))
    for m in old.materials:
        me.materials.append(m)
    S = np.asarray(root.get("mobilia_fit_scale", [1.0, 1.0, 1.0]), np.float64)
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    me.vertices.foreach_set("co", (co.reshape(-1, 3) * S).astype(np.float32).ravel())
    body.data = me
    bpy.data.meshes.remove(old)
    info["dense_axis"] = axis
    root["mobilia_rug"] = json.dumps(info)


def animate_rug_unroll(root: bpy.types.Object, frame_start: int, frame_end: int, *, axis: str | None = None,
                       reverse: bool = False, substeps: int = 1) -> None:
    """Anima o tapete desenrolando-se no piso, de ``frame_start`` (rolo) a ``frame_end`` (plano).

    Técnica: shape keys por quadro calculadas em Python (espiral de Arquimedes que rola
    sem deslizar), com pesos lineares entre quadros vizinhos — funciona com motion blur
    de deformação do Cycles. Antes de ``frame_start`` o tapete fica enrolado; em
    ``frame_end`` ele está exatamente plano (forma base), com toda a geometria em Z >= 0.

    ``axis``: eixo do desenrolar ("x" ou "y"; padrão = comprimento do tapete);
    ``reverse``: começa pela ponta +eixo em vez da ponta -eixo;
    ``substeps``: shape keys por quadro (2 deixa o motion blur mais fiel).
    """
    if "mobilia_rug" not in root:
        raise ValueError("O objeto não é um tapete procedural do 7xAi")
    if frame_end <= frame_start:
        raise ValueError("frame_end deve ser maior que frame_start")
    info = json.loads(root["mobilia_rug"])
    axis = (axis or info["long_axis"]).lower()
    if axis not in ("x", "y"):
        raise ValueError("axis deve ser 'x' ou 'y'")
    if axis != info["dense_axis"]:
        _rebuild_rug_body(root, axis)
    parts = _rug_parts(root)
    T = float(info["thickness"]) * float(root.get("mobilia_fit_scale", [1, 1, 1])[2])
    ax = 0 if axis == "x" else 1
    sign = -1.0 if reverse else 1.0
    rest = []
    for obj in parts:
        if obj.data.shape_keys is not None:
            obj.shape_key_clear()
        co = np.empty(len(obj.data.vertices) * 3, np.float32)
        obj.data.vertices.foreach_get("co", co)
        loc = np.asarray(obj.location, np.float64)
        rest.append((obj, co.reshape(-1, 3).astype(np.float64) + loc, loc))
    all_s = np.concatenate([sign * r[1][:, ax] for r in rest])
    s_min, s_end = float(all_s.min()), float(all_s.max())
    pitch = 1.07 * T
    r0 = max(0.014, 1.6 * T)
    b = pitch / math.pi
    r = r0
    for _ in range(30):  # rolo encostado na ponta inicial, dentro da pegada do tapete
        r = math.sqrt(r0 * r0 + b * max(s_end - (s_min + r + T / 2), 0.0))
    s0 = s_min + r + T / 2
    n_keys = max(1, int(round((frame_end - frame_start) * max(1, int(substeps)))))
    times = np.linspace(frame_start, frame_end, n_keys + 1)
    keys_per_obj: dict[str, list] = {}
    for obj, co, loc in rest:
        obj.shape_key_add(name="Basis", from_mix=False)
        keys_per_obj[obj.name] = []
    for k in range(n_keys):
        e = float(_ease(np.array((times[k] - frame_start) / (frame_end - frame_start))))
        s_flat = s0 + (s_end - s0) * e
        for obj, co, loc in rest:
            pos = rug_roll_positions(co, ax, sign, T, s_flat, s_end, pitch, r0)
            kb = obj.shape_key_add(name=f"desenrolar_{k:04d}", from_mix=False)
            kb.data.foreach_set("co", (pos - loc).astype(np.float32).ravel())
            kb.value = 0.0
            keys_per_obj[obj.name].append(kb.name)
    for obj, co, loc in rest:
        key = obj.data.shape_keys
        key.use_relative = True
        ad = key.animation_data or key.animation_data_create()
        act = bpy.data.actions.new(f"{obj.name}.desenrolar")
        ad.action = act
        names = keys_per_obj[obj.name]
        for k, nm in enumerate(names):
            pts = []
            if k > 0:
                pts.append((times[k - 1], 0.0))
            pts.append((times[k], 1.0))
            pts.append((times[k + 1], 0.0))
            fc = act.fcurve_ensure_for_datablock(key, f'key_blocks["{nm}"].value')
            fc.keyframe_points.add(len(pts))
            fc.keyframe_points.foreach_set("co", np.asarray(pts, np.float32).ravel())
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
            fc.update()
    root["mobilia_unroll"] = json.dumps(dict(frame_start=frame_start, frame_end=frame_end, axis=axis,
                                             reverse=reverse, keys=n_keys))
