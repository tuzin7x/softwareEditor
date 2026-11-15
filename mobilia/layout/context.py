"""Contexto espacial do layout: sala + grade de ocupação + câmera fixa da foto.

Pré-calcula tudo o que a busca de posicionamento consulta milhares de vezes:
- estado das células da grade (FREE/UNKNOWN/BLOCKED/OUTSIDE) para pontos do piso;
- projeção de caixas 3D na imagem (a câmera é a da foto e nunca se move);
- paredes com os trechos visíveis na foto, cantos, janelas/portas projetadas;
- zonas de abertura de portas e a máscara (em baixa resolução) do piso visível.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..analysis.room import BLOCKED, FREE, OUTSIDE, UNKNOWN, RoomModel
from . import geom

NEAR_Z = 0.15            # m — pontos mais próximos que isso da câmera são inválidos
LOWRES_WIDTH = 192       # largura da imagem de baixa resolução usada nas métricas de imagem


@dataclass
class OpeningInfo:
    kind: str                      # "window" | "door"
    wall: int
    s0: float
    s1: float
    z0: float
    z1: float
    corners3d: np.ndarray          # (4, 3) retângulo na parede
    quad_img: np.ndarray           # polígono projetado e recortado à imagem (anti-horário)
    area_img: float
    door_zone: np.ndarray | None   # polígono (4, 2) da área de abertura da porta no piso

    @property
    def bbox_img(self) -> tuple[float, float, float, float]:
        if len(self.quad_img) < 3:
            return (0.0, 0.0, 0.0, 0.0)
        q = self.quad_img
        return (float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max()))


@dataclass
class WallInfo:
    id: int
    p0: np.ndarray
    p1: np.ndarray
    direction: np.ndarray
    normal: np.ndarray             # aponta para dentro da sala
    length: float
    height: float
    openings: list[OpeningInfo] = field(default_factory=list)
    s: np.ndarray = field(default_factory=lambda: np.zeros(0))          # amostras ao longo da parede
    vis_art: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))   # visível a 0,9–2,1 m
    vis_base: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # base (piso) visível
    view_cos: np.ndarray = field(default_factory=lambda: np.zeros(0))   # quão frontal a câmera vê

    def point(self, s: float) -> np.ndarray:
        return self.p0 + self.direction * s

    def intervals(self, mask: np.ndarray, min_len: float = 0.3) -> list[tuple[float, float]]:
        """Intervalos contínuos [s0, s1] onde ``mask`` é verdadeira."""
        out = []
        start = None
        for i, ok in enumerate(mask):
            if ok and start is None:
                start = self.s[i]
            if (not ok or i == len(mask) - 1) and start is not None:
                end = self.s[i] if ok else self.s[i - 1]
                if end - start >= min_len:
                    out.append((float(start), float(end)))
                start = None
        return out

    def window_free(self, s0: float, s1: float, z0: float, z1: float, margin: float = 0.08) -> bool:
        """True se o retângulo [s0,s1]x[z0,z1] na parede não toca nenhuma abertura (com folga)."""
        for o in self.openings:
            if s1 > o.s0 - margin and s0 < o.s1 + margin and z1 > o.z0 - margin and z0 < o.z1 + margin:
                return False
        return True


@dataclass
class Corner:
    point: np.ndarray              # vértice interno (x, y)
    walls: tuple[int, int]
    normals: tuple[np.ndarray, np.ndarray]
    bisector: np.ndarray           # direção para dentro da sala


class LayoutContext:
    """Consultas geométricas rápidas sobre a sala e a câmera da foto."""

    def __init__(self, room: RoomModel, grid: np.ndarray, image_margin_frac: float = 0.012):
        self.room = room
        self.grid = np.asarray(grid)
        self.cell = float(room.grid_cell)
        self.origin = np.asarray(room.grid_origin, float)
        self.ny, self.nx = self.grid.shape
        intr = room.intr()
        self.W, self.H = int(intr.width), int(intr.height)
        self.fx, self.fy, self.cx, self.cy = float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)
        self.margin_px = max(4.0, image_margin_frac * self.W)
        M = room.M
        self.R = M[:3, :3].copy()
        self.t = M[:3, 3].copy()
        self.Minv = np.linalg.inv(M)
        self.cam_pos = self.t.copy()
        self.cam_xy = self.t[:2].copy()
        self.cam_h = float(room.camera_height)
        f = self.R[:, 2]
        self.forward3 = f / np.linalg.norm(f)
        self.forward2 = f[:2] / (np.linalg.norm(f[:2]) + 1e-12)
        self.ceiling = float(room.ceiling_height)
        self.image_poly = np.array([[0, 0], [self.W, 0], [self.W, self.H], [0, self.H]], float)
        self.free = self.grid == FREE
        self.lw = LOWRES_WIDTH
        self.lh = max(8, int(round(self.H * self.lw / self.W)))
        self.low_scale = self.lw / self.W
        self.walls = self._build_walls()
        self.openings = [o for w in self.walls for o in w.openings]
        self.door_zones = [o.door_zone for o in self.openings if o.door_zone is not None]
        self.corners = self._build_corners()
        self.floor_low = self._floor_lowres()
        self.free_area = float(self.free.sum()) * self.cell ** 2
        self._build_walk_grid()

    # ------------------------------------------------------------------ projeção
    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pontos da sala (N, 3) -> pixels (N, 2) e profundidade de câmera z (N,)."""
        p = np.asarray(pts, float) @ self.Minv[:3, :3].T + self.Minv[:3, 3]
        z = p[:, 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.fx * p[:, 0] / zs + self.cx
        v = self.fy * p[:, 1] / zs + self.cy
        return np.stack([u, v], -1), z

    def rays_room(self, uv: np.ndarray) -> np.ndarray:
        """Direções (N, 3) no referencial da sala para pixels contínuos (N, 2)."""
        d = np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy, np.ones(len(uv))], -1)
        return d @ self.R.T

    def in_image(self, uv: np.ndarray, z: np.ndarray, margin: float | None = None) -> np.ndarray:
        m = self.margin_px if margin is None else margin
        return (z > NEAR_Z) & (uv[:, 0] >= m) & (uv[:, 0] <= self.W - m) & (uv[:, 1] >= m) & (uv[:, 1] <= self.H - m)

    def border_margin(self, uv: np.ndarray, z: np.ndarray) -> float:
        """Menor distância (px) dos pontos à borda da imagem; negativa se algum ponto sai dela."""
        if np.any(z <= NEAR_Z):
            return -1e6
        m = np.minimum.reduce([uv[:, 0], self.W - uv[:, 0], uv[:, 1], self.H - uv[:, 1]])
        return float(m.min())

    # ------------------------------------------------------------------ grade
    def cell_index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ij = np.floor((np.asarray(xy, float) - self.origin) / self.cell).astype(int)
        ix, iy = ij[..., 0], ij[..., 1]
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, ok

    def states(self, xy: np.ndarray) -> np.ndarray:
        """Estado da grade para pontos (N, 2); fora da grade conta como OUTSIDE."""
        ix, iy, ok = self.cell_index(xy)
        out = np.full(len(ix), OUTSIDE, np.uint8)
        out[ok] = self.grid[iy[ok], ix[ok]]
        return out

    def footprint_states(self, cx: float, cy: float, w: float, d: float, yaw: float,
                         shrink: float = 0.01) -> tuple[float, float]:
        """Fração das células tocadas pela base em estados proibidos (BLOCKED/OUTSIDE) e em UNKNOWN.

        A base (encolhida de ``shrink`` para tolerar arredondamentos) é rasterizada na grade;
        toda célula tocada conta (conservador). Fora da grade conta como OUTSIDE.
        """
        poly = geom.obb_corners(cx, cy, max(w - 2 * shrink, 0.01), max(d - 2 * shrink, 0.01), yaw)
        g = self.to_grid_px(poly)
        x0 = int(math.floor(g[:, 0].min())) - 1
        y0 = int(math.floor(g[:, 1].min())) - 1
        x1 = int(math.ceil(g[:, 0].max())) + 1
        y1 = int(math.ceil(g[:, 1].max())) + 1
        m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        q = np.round((g - [x0, y0]) * 16).astype(np.int32)
        cv2.fillConvexPoly(m, q, 1, lineType=cv2.LINE_8, shift=4)
        ys, xs = np.nonzero(m)
        gx, gy = xs + x0, ys + y0
        ok = (gx >= 0) & (gx < self.nx) & (gy >= 0) & (gy < self.ny)
        st = np.full(len(gx), OUTSIDE, np.uint8)
        st[ok] = self.grid[gy[ok], gx[ok]]
        n = max(1, len(st))
        bad = float(np.count_nonzero((st == BLOCKED) | (st == OUTSIDE))) / n
        unk = float(np.count_nonzero(st == UNKNOWN)) / n
        return bad, unk

    def to_grid_px(self, xy: np.ndarray) -> np.ndarray:
        """Coordenadas contínuas de pixel na grade (centro da célula = inteiro)."""
        return (np.asarray(xy, float) - self.origin) / self.cell - 0.5

    def cell_centers(self) -> np.ndarray:
        yy, xx = np.mgrid[0:self.ny, 0:self.nx]
        return np.stack([self.origin[0] + (xx + 0.5) * self.cell, self.origin[1] + (yy + 0.5) * self.cell], -1)

    def camera_distance(self, poly: np.ndarray) -> float:
        return geom.point_polygon_distance(self.cam_xy, poly)

    def dir_to_camera(self, xy: np.ndarray) -> np.ndarray:
        d = self.cam_xy - np.asarray(xy, float)
        return d / (np.linalg.norm(d) + 1e-12)

    # ------------------------------------------------------------------ paredes e aberturas
    def _build_walls(self) -> list[WallInfo]:
        walls: list[WallInfo] = []
        for w in self.room.walls:
            p0, p1 = np.asarray(w.p0, float), np.asarray(w.p1, float)
            L = float(np.linalg.norm(p1 - p0))
            d = (p1 - p0) / max(L, 1e-9)
            wi = WallInfo(w.id, p0, p1, d, np.asarray(w.normal, float), L, float(w.height))
            walls.append(wi)
        for wi in walls:
            n = max(2, int(math.ceil(wi.length / 0.05)) + 1)
            wi.s = np.linspace(0.0, wi.length, n)
            pts = wi.p0 + wi.s[:, None] * wi.direction
            occl = np.array([self._occluded_by_other_wall(p, wi.id, walls) for p in pts])
            z_art = (0.9, min(2.1, self.ceiling - 0.1))
            vis_art = np.ones(n, bool)
            for z in z_art:
                uv, zc = self.project(np.column_stack([pts, np.full(n, z)]))
                vis_art &= self.in_image(uv, zc)
            uv, zc = self.project(np.column_stack([pts, np.zeros(n)]))
            vis_base = self.in_image(uv, zc)
            to_cam = self.cam_xy - pts
            to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-12
            wi.view_cos = to_cam @ wi.normal
            front = wi.view_cos > 0.05
            wi.vis_art = vis_art & ~occl & front
            wi.vis_base = vis_base & ~occl & front
        for o in self.room.openings:
            if o.wall >= len(walls):
                continue
            wi = walls[o.wall]
            a, b = wi.point(o.s0), wi.point(o.s1)
            c3 = np.array([[a[0], a[1], o.z0], [b[0], b[1], o.z0], [b[0], b[1], o.z1], [a[0], a[1], o.z1]])
            quad = self._project_poly_clipped(c3)
            zone = None
            if o.kind == "door":
                s0, s1 = o.s0 - 0.05, o.s1 + 0.05
                depth = 0.9
                zone = np.array([wi.point(s0), wi.point(s1), wi.point(s1) + wi.normal * depth,
                                 wi.point(s0) + wi.normal * depth])
                zone = geom.ensure_ccw(zone)
            wi.openings.append(OpeningInfo(o.kind, o.wall, float(o.s0), float(o.s1), float(o.z0), float(o.z1),
                                           c3, quad, geom.polygon_area(quad) if len(quad) >= 3 else 0.0, zone))
        return walls

    def _occluded_by_other_wall(self, p: np.ndarray, wid: int, walls: list[WallInfo]) -> bool:
        """True se o segmento câmera->p cruza outra parede antes de chegar em p."""
        q = p + (self.cam_xy - p) * 0.02     # recua 2% para não contar a própria quina
        for w in walls:
            if w.id == wid:
                continue
            if geom.segments_intersect(self.cam_xy, q, w.p0, w.p1):
                return True
        return False

    def _project_poly_clipped(self, pts3: np.ndarray) -> np.ndarray:
        """Projeta um polígono 3D (subdividido para lidar com pontos atrás da câmera) e recorta à imagem."""
        dense = []
        for i in range(len(pts3)):
            a, b = pts3[i], pts3[(i + 1) % len(pts3)]
            for t in np.linspace(0, 1, 12, endpoint=False):
                dense.append(a + (b - a) * t)
        dense = np.array(dense)
        uv, z = self.project(dense)
        uv = uv[z > NEAR_Z]
        if len(uv) < 3:
            return np.zeros((0, 2))
        hull = geom.convex_hull(uv)
        if len(hull) < 3:
            return np.zeros((0, 2))
        return geom.clip_convex(hull, self.image_poly)

    def _build_corners(self) -> list[Corner]:
        out: list[Corner] = []
        ws = self.walls
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a, b = ws[i], ws[j]
                if abs(a.direction @ b.direction) > 0.5:
                    continue
                A = np.column_stack([a.direction, -b.direction])
                if abs(np.linalg.det(A)) < 1e-6:
                    continue
                ta, tb = np.linalg.solve(A, b.p0 - a.p0)
                p = a.p0 + a.direction * ta
                # as duas paredes precisam terminar perto do vértice (canto em "L")
                if min(abs(ta), abs(ta - a.length)) > 0.4 or min(abs(tb), abs(tb - b.length)) > 0.4:
                    continue
                bis = a.normal + b.normal
                bis = bis / (np.linalg.norm(bis) + 1e-12)
                probe = p + bis * 0.35
                if (probe - a.p0) @ a.normal <= 0 or (probe - b.p0) @ b.normal <= 0:
                    continue          # canto reentrante visto de dentro: não serve para apoiar peças
                if self.states(probe[None])[0] in (BLOCKED, OUTSIDE):
                    continue
                out.append(Corner(p, (a.id, b.id), (a.normal, b.normal), bis))
        return out

    def _floor_lowres(self) -> np.ndarray:
        """Máscara (lh, lw) dos pixels de baixa resolução que mostram piso livre (FREE)."""
        u = (np.arange(self.lw) + 0.5) / self.low_scale
        v = (np.arange(self.lh) + 0.5) / self.low_scale
        uu, vv = np.meshgrid(u, v)
        uv = np.column_stack([uu.ravel(), vv.ravel()])
        d = self.rays_room(uv)
        down = d[:, 2] < -1e-6
        t = np.where(down, -self.cam_h / np.where(down, d[:, 2], -1.0), -1.0)
        hit = self.cam_pos[:2] + d[:, :2] * t[:, None]
        st = self.states(hit)
        m = down & (t > 0) & (st == FREE)
        return m.reshape(self.lh, self.lw)

    def _build_walk_grid(self, factor: int = 2) -> None:
        """Grade de circulação em resolução menor (2 x CELL): livre / caminhável."""
        self.walk_factor = factor
        self.walk_cell = self.cell * factor
        ny, nx = self.ny // factor, self.nx // factor
        g = self.grid[: ny * factor, : nx * factor].reshape(ny, factor, nx, factor)
        free = (g == FREE).mean(axis=(1, 3)) >= 0.5
        walk = ((g == FREE) | (g == UNKNOWN)).mean(axis=(1, 3)) >= 0.75
        self.walk_free = free
        self.walk_ok = walk | free
        yy, xx = np.mgrid[0:ny, 0:nx]
        centers = np.stack([self.origin[0] + (xx + 0.5) * self.walk_cell,
                            self.origin[1] + (yy + 0.5) * self.walk_cell], -1)
        self.walk_cam_dist = np.linalg.norm(centers - self.cam_xy, axis=-1)

    def to_walk_px(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy, float) - self.origin) / self.walk_cell - 0.5

    def walk_index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ij = np.floor((np.asarray(xy, float) - self.origin) / self.walk_cell).astype(int)
        ix, iy = ij[..., 0], ij[..., 1]
        ok = (ix >= 0) & (ix < self.walk_free.shape[1]) & (iy >= 0) & (iy < self.walk_free.shape[0])
        return ix, iy, ok

    def wall_by_id(self, wid: int) -> WallInfo:
        for w in self.walls:
            if w.id == wid:
                return w
        raise KeyError(wid)

    def wall_contact(self, poly: np.ndarray, back_dir: np.ndarray, tol: float = 0.1) -> tuple[float, int | None]:
        """Fração da aresta traseira de uma base encostada numa parede paralela (e id da parede)."""
        best, best_id = 0.0, None
        ctr = poly.mean(0)
        for w in self.walls:
            if w.normal @ back_dir > -0.97:      # a traseira precisa olhar para a parede
                continue
            # aresta traseira: os dois cantos mais distantes no sentido de back_dir
            proj = (poly - ctr) @ back_dir
            idx = np.argsort(-proj)[:2]
            pts = np.linspace(poly[idx[0]], poly[idx[1]], 11)
            sd = (pts - w.p0) @ w.normal
            s = (pts - w.p0) @ w.direction
            ok = (sd > -0.03) & (sd < tol) & (s >= -0.02) & (s <= w.length + 0.02)
            frac = float(ok.mean())
            if frac > best:
                best, best_id = frac, w.id
        return best, best_id
