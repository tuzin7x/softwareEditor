"""Estado de trabalho da busca: peças posicionadas e a avaliação geométrica de cada uma.

``Placement`` é a versão mutável de ``model.PlacedItem`` usada durante a otimização.
``GeoCache.geo`` calcula (e memoriza) tudo o que depende só da pose de uma peça:
base no piso, caixa 3D projetada na foto, margem até a borda da imagem, células da grade
ocupadas, distância à câmera, invasão de área de porta e cobertura de janelas na imagem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from . import geom
from .context import NEAR_Z, LayoutContext
from .model import PlacedItem

DOOR_COVER_MIN_H = 0.95    # peças mais baixas que isso escondem pouco de uma porta na foto


@dataclass
class Placement:
    key: str
    role: str
    asset_id: str
    w: float
    d: float
    h: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    mount: str = "floor"          # floor | wall | on_top
    wall_id: int | None = None
    parent: str | None = None
    params: dict = field(default_factory=dict)
    group: str = ""               # grupo funcional (ex.: "conversa", "cama", "jantar")
    slot: str = ""                # descrição do encaixe usado (depuração)

    def pose_key(self) -> tuple:
        return (self.asset_id, round(self.w, 4), round(self.d, 4), round(self.h, 4), round(self.x, 4),
                round(self.y, 4), round(self.z, 4), round(self.yaw % 360.0, 3), self.mount)

    @property
    def front(self) -> np.ndarray:
        return geom.front_of(self.yaw)

    @property
    def lateral(self) -> np.ndarray:
        return geom.lateral_of(self.yaw)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def corners(self) -> np.ndarray:
        return geom.obb_corners(self.x, self.y, self.w, self.d, self.yaw)

    def box(self) -> np.ndarray:
        return geom.box_corners_3d(self.x, self.y, self.z, self.w, self.d, self.h, self.yaw)

    def moved(self, **kw) -> "Placement":
        return replace(self, **kw)

    def to_item(self, order: int) -> PlacedItem:
        return PlacedItem(id=self.key, role=self.role, asset_id=self.asset_id,
                          position=[round(self.x, 4), round(self.y, 4), round(self.z, 4)],
                          yaw_deg=round(geom.wrap_deg(self.yaw), 3), scale=1.0,
                          size=[round(self.w, 4), round(self.d, 4), round(self.h, 4)], mount=self.mount,
                          wall_id=self.wall_id, parent=self.parent, order=order, params=dict(self.params))

    @staticmethod
    def from_item(it: PlacedItem) -> "Placement":
        return Placement(it.id, it.role, it.asset_id, float(it.size[0]), float(it.size[1]), float(it.size[2]),
                         float(it.position[0]), float(it.position[1]), float(it.position[2]), float(it.yaw_deg),
                         it.mount, it.wall_id, it.parent, dict(it.params))


@dataclass
class Geo:
    poly: np.ndarray              # base no piso (4, 2)
    box: np.ndarray               # caixa 3D (8, 3)
    uv: np.ndarray                # cantos projetados (8, 2)
    inside: bool                  # caixa inteira dentro da imagem (com margem)
    margin: float                 # px até a borda (negativo = cortada)
    hull: np.ndarray              # silhueta convexa na imagem, recortada ao quadro
    hull_area: float
    depth: float                  # distância 3D da câmera ao centro da caixa (m)
    cam_dist: float               # distância horizontal mínima da base à câmera (m)
    bad: float                    # fração da base em células BLOCKED/OUTSIDE
    unknown: float                # fração da base em células UNKNOWN
    door_hit: bool                # invade a área de abertura de alguma porta
    window_cover: float           # maior fração (0..1) de janela (ou porta, se a peça for alta) coberta
    low: tuple                    # (x0, y0, máscara bool) em baixa resolução
    center: tuple = (0.0, 0.0)    # centro da base (x, y)
    radius: float = 0.0           # raio do círculo envolvente da base


class GeoCache:
    """Memoriza ``Geo`` por pose (a busca reavalia as mesmas poses muitas vezes)."""

    def __init__(self, ctx: LayoutContext, max_items: int = 200_000):
        self.ctx = ctx
        self._cache: dict[tuple, Geo] = {}
        self._max = max_items

    def geo(self, p: Placement) -> Geo:
        k = p.pose_key()
        g = self._cache.get(k)
        if g is None:
            if len(self._cache) >= self._max:
                self._cache.clear()
            g = self._compute(p)
            self._cache[k] = g
        return g

    def _compute(self, p: Placement) -> Geo:
        ctx = self.ctx
        poly = p.corners()
        box = p.box()
        uv, zc = ctx.project(box)
        inside = bool(np.all(ctx.in_image(uv, zc)))
        margin = ctx.border_margin(uv, zc)
        ok = zc > NEAR_Z
        if ok.sum() >= 3:
            h = cv2.convexHull(uv[ok].astype(np.float32), clockwise=False).reshape(-1, 2).astype(np.float64)
            hull = geom.ensure_ccw(h) if len(h) >= 3 else np.zeros((0, 2))
            if len(hull) >= 3 and not inside:
                hull = geom.clip_convex(hull, ctx.image_poly)
        else:
            hull = np.zeros((0, 2))
        area = geom.polygon_area(hull) if len(hull) >= 3 else 0.0
        ctr = box.mean(0)
        depth = float(np.linalg.norm(ctr - ctx.cam_pos))
        cam_dist = ctx.camera_distance(poly)
        if p.mount == "floor":
            bad, unk = ctx.footprint_states(p.x, p.y, p.w, p.d, p.yaw)
            door = any(geom.obb_penetration(poly, dz) > 0.01 for dz in ctx.door_zones)
        else:
            bad, unk, door = 0.0, 0.0, False
        cover = 0.0
        if area > 0 and p.role != "rug":
            hb = (hull[:, 0].min(), hull[:, 1].min(), hull[:, 0].max(), hull[:, 1].max())
            for o in ctx.openings:
                if o.area_img <= 1.0:
                    continue
                qb = o.bbox_img
                if hb[2] <= qb[0] or qb[2] <= hb[0] or hb[3] <= qb[1] or qb[3] <= hb[1]:
                    continue
                frac = geom.intersection_area(hull, o.quad_img) / o.area_img
                if o.kind == "door" and p.h + p.z < DOOR_COVER_MIN_H:
                    frac *= 0.3          # peça baixa diante de porta distante: só uma faixa da porta some
                cover = max(cover, frac)
        low = self._raster(hull)
        center = (float(p.x), float(p.y))
        radius = 0.5 * math.hypot(p.w, p.d)
        return Geo(poly, box, uv, inside, margin, hull, area, depth, cam_dist, bad, unk, door, cover, low, center,
                   radius)

    def _raster(self, hull: np.ndarray) -> tuple:
        ctx = self.ctx
        if len(hull) < 3:
            return (0, 0, np.zeros((0, 0), bool))
        pts = hull * ctx.low_scale - 0.5
        x0 = max(0, int(math.floor(pts[:, 0].min())))
        y0 = max(0, int(math.floor(pts[:, 1].min())))
        x1 = min(ctx.lw - 1, int(math.ceil(pts[:, 0].max())))
        y1 = min(ctx.lh - 1, int(math.ceil(pts[:, 1].max())))
        if x1 < x0 or y1 < y0:
            return (0, 0, np.zeros((0, 0), bool))
        m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        q = np.round((pts - [x0, y0]) * 16).astype(np.int32)
        cv2.fillConvexPoly(m, q, 1, lineType=cv2.LINE_8, shift=4)
        return (x0, y0, m.astype(bool))
