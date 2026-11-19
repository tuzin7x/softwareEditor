"""Visualizações de depuração da análise (overlay de geometria e mapa visto de cima)."""
from __future__ import annotations

import cv2
import numpy as np

from .room import BLOCKED, FREE, OUTSIDE, RoomModel
from .segmentation import GROUP_IDS, SegmentationResult

GROUP_COLORS = {
    "floor": (60, 180, 75), "wall": (230, 190, 255), "ceiling": (255, 250, 200), "window": (0, 130, 200),
    "window_covering": (70, 240, 240), "door": (245, 130, 48), "structure": (128, 0, 0),
    "fixture": (170, 110, 40), "furniture": (240, 50, 230), "ceiling_fixture": (255, 225, 25),
}


def seg_overlay(rgb8: np.ndarray, seg: SegmentationResult, alpha: float = 0.45) -> np.ndarray:
    out = rgb8.astype(np.float32).copy()
    for g, col in GROUP_COLORS.items():
        m = seg.groups == GROUP_IDS[g]
        out[m] = out[m] * (1 - alpha) + np.array(col, np.float32) * alpha
    return out.astype(np.uint8)


def _draw_poly(img, uv, color, thick=2, closed=False):
    pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], closed, color, thick, cv2.LINE_AA)


def _clip_segment_front(p, q, room: RoomModel, near=0.05):
    """Recorta o segmento 3D (sala) para a parte à frente da câmera."""
    Minv = room.cam_from_room()
    pc = Minv[:3, :3] @ p + Minv[:3, 3]; qc = Minv[:3, :3] @ q + Minv[:3, 3]
    if pc[2] < near and qc[2] < near:
        return None
    if pc[2] < near:
        t = (near - pc[2]) / (qc[2] - pc[2]); p = p + (q - p) * t
    elif qc[2] < near:
        t = (near - qc[2]) / (pc[2] - qc[2]); q = q + (p - q) * t
    return p, q


def geometry_overlay(rgb8: np.ndarray, room: RoomModel, seg: SegmentationResult | None = None) -> np.ndarray:
    img = seg_overlay(rgb8, seg, 0.25) if seg is not None else rgb8.copy()
    img = np.ascontiguousarray(img)

    def line3d(p, q, color, thick=2, n=24):
        r = _clip_segment_front(np.asarray(p, float), np.asarray(q, float), room)
        if r is None:
            return
        p, q = r
        pts = np.linspace(p, q, n)
        uv, z = room.project_room_points(pts)
        _draw_poly(img, uv[z > 0], color, thick)

    # grade de 1 m no piso, limitada à área da grade de ocupação
    x0, y0 = room.grid_origin
    ny, nx = room.grid_shape
    x1, y1 = x0 + nx * room.grid_cell, y0 + ny * room.grid_cell
    for x in np.arange(np.ceil(x0), x1, 1.0):
        line3d((x, max(y0, 0.3), 0), (x, y1, 0), (255, 255, 255), 1)
    for y in np.arange(np.ceil(max(y0, 0.3)), y1, 1.0):
        line3d((x0, y, 0), (x1, y, 0), (255, 255, 255), 1)
    # paredes: base, topo e verticais
    for w in room.walls:
        a = np.array([*w.p0, 0.0]); b = np.array([*w.p1, 0.0])
        h = np.array([0, 0, room.ceiling_height])
        line3d(a, b, (255, 60, 60), 3)
        line3d(a + h, b + h, (255, 60, 60), 2)
        line3d(a, a + h, (255, 60, 60), 2); line3d(b, b + h, (255, 60, 60), 2)
        mid = (a + b) / 2 + np.array([0, 0, 1.2])
        uv, z = room.project_room_points(mid[None])
        if z[0] > 0:
            cv2.putText(img, f"P{w.id}", tuple(int(v) for v in uv[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 60, 60), 2, cv2.LINE_AA)
    for o in room.openings:
        w = room.walls[o.wall]
        d = w.direction; base = np.asarray(w.p0)
        corners = []
        for s, z in ((o.s0, o.z0), (o.s1, o.z0), (o.s1, o.z1), (o.s0, o.z1), (o.s0, o.z0)):
            xy = base + d * s
            corners.append([xy[0], xy[1], z])
        col = (0, 160, 255) if o.kind == "window" else (255, 160, 0)
        for i in range(4):
            line3d(corners[i], corners[i + 1], col, 3, 8)
    txt = (f"h_cam={room.camera_height:.2f}m  teto={room.ceiling_height:.2f}m  "
           f"FOVx={room.quality.get('fov_x_deg', 0):.1f}  paredes={len(room.walls)}  "
           f"escala={room.scale_factor:.2f}")
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, txt, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def topdown_map(room: RoomModel, grid: np.ndarray, px_per_cell: int = 4,
                footprints: list | None = None) -> np.ndarray:
    """Mapa visto de cima: piso livre (cinza claro), obstáculos, paredes, câmera e frustum."""
    ny, nx = grid.shape
    img = np.full((ny, nx, 3), 40, np.uint8)
    img[grid == FREE] = (200, 200, 200)
    img[grid == BLOCKED] = (200, 80, 80)
    img[grid == OUTSIDE] = (15, 15, 15)
    img = np.ascontiguousarray(np.flipud(img))
    img = cv2.resize(img, (nx * px_per_cell, ny * px_per_cell), interpolation=cv2.INTER_NEAREST)
    x0, y0 = room.grid_origin; c = room.grid_cell

    def to_px(xy):
        xy = np.asarray(xy, float)
        u = (xy[..., 0] - x0) / c * px_per_cell
        v = (ny - (xy[..., 1] - y0) / c) * px_per_cell
        return np.stack([u, v], -1)

    for w in room.walls:
        _draw_poly(img, to_px(np.array([w.p0, w.p1])), (255, 60, 60), 3)
    for o in room.openings:
        w = room.walls[o.wall]
        a = np.asarray(w.p0) + w.direction * o.s0; b = np.asarray(w.p0) + w.direction * o.s1
        _draw_poly(img, to_px(np.array([a, b])), (0, 160, 255) if o.kind == "window" else (255, 160, 0), 5)
    # câmera + frustum horizontal
    intr = room.intr()
    cam = to_px(np.array([0.0, 0.0]))
    M = room.M
    for u in (0, intr.width):
        ray = np.array([(u - intr.cx) / intr.fx, 0.0, 1.0])
        rr = M[:3, :3] @ ray
        end = rr[:2] / (np.linalg.norm(rr[:2]) + 1e-9) * 6.0
        _draw_poly(img, np.stack([cam, to_px(end)]), (255, 255, 0), 1)
    cv2.circle(img, tuple(int(v) for v in cam), 6, (255, 255, 0), -1)
    if footprints:
        for fp in footprints:
            poly = to_px(np.asarray(fp["polygon"]))
            _draw_poly(img, poly, fp.get("color", (60, 220, 60)), 2, closed=True)
            ctr = poly.mean(0)
            cv2.putText(img, fp.get("label", ""), (int(ctr[0]) - 20, int(ctr[1])), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img
