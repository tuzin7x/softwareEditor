"""Visualizações de depuração do layout.

- ``layout_topdown``: mapa visto de cima (grade de ocupação + paredes + câmera) com a base de
  cada peça, a seta da frente e o número da ordem de montagem;
- ``layout_overlay``: caixas 3D de cada peça projetadas sobre a foto (aramado) com rótulos —
  mostra exatamente o que a câmera da foto vai ver;
- ``save_layout_debug``: grava as duas imagens (PNG/JPG) numa pasta.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..analysis.debug_viz import topdown_map
from ..analysis.room import RoomModel
from . import geom
from .model import Layout, PlacedItem

ROLE_COLORS = {
    "rug": (175, 135, 255), "sofa": (255, 110, 50), "armchair": (255, 150, 60), "dining_chair": (255, 150, 60),
    "desk_chair": (255, 150, 60), "bar_stool": (255, 150, 60), "ottoman": (255, 150, 60), "bed": (255, 110, 50),
    "coffee_table": (60, 190, 255), "side_table": (60, 190, 255), "dining_table": (60, 190, 255),
    "desk": (60, 190, 255), "nightstand": (60, 190, 255), "tv_console": (255, 215, 50), "dresser": (255, 215, 50),
    "shelf": (255, 215, 50), "shelf_large": (255, 215, 50), "desk_storage": (255, 215, 50),
    "floor_lamp": (255, 255, 150), "table_lamp": (255, 255, 150), "plant_large": (70, 215, 70),
    "plant_medium": (70, 215, 70), "plant_small": (70, 215, 70), "basket": (200, 170, 120),
    "wall_art": (255, 80, 200), "wall_mirror": (255, 80, 200),
}
DECOR_COLOR = (230, 230, 230)

SHORT = {"sofa": "sofá", "armchair": "poltrona", "coffee_table": "centro", "rug": "tapete",
         "side_table": "lateral", "floor_lamp": "lumin.", "plant_large": "planta G", "plant_medium": "planta M",
         "plant_small": "planta P", "wall_art": "quadro", "tv_console": "rack", "pillows": "almof.",
         "vase": "vaso", "bowl": "fruteira", "books": "livros", "bed": "cama", "nightstand": "criado",
         "table_lamp": "abajur", "dresser": "cômoda", "throw": "manta", "dining_table": "mesa",
         "dining_chair": "cadeira", "shelf": "estante", "desk": "escrivaninha", "desk_chair": "cadeira",
         "bar_stool": "banqueta", "kitchen_decor": "jarra", "basket": "cesto", "wall_mirror": "espelho"}

_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def _color(it: PlacedItem) -> tuple[int, int, int]:
    return ROLE_COLORS.get(it.role, DECOR_COLOR)


def _label(it: PlacedItem) -> str:
    return f"{it.order}:{SHORT.get(it.role, it.role)}"


def _put_text(img, text, org, color, scale=0.5, thick=1):
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def layout_topdown(room: RoomModel, grid: np.ndarray, layout: Layout, px_per_cell: int = 6) -> np.ndarray:
    """Mapa visto de cima (RGB) com as bases das peças, setas de frente e ordem de montagem."""
    floor_items = [it for it in layout.items if it.mount == "floor"]
    floor_items.sort(key=lambda it: (it.role != "rug", it.order))
    fps = [dict(polygon=it.footprint(), color=_color(it), label="") for it in floor_items]
    img = np.ascontiguousarray(topdown_map(room, grid, px_per_cell, fps))
    ny = grid.shape[0]
    x0, y0 = room.grid_origin
    c = room.grid_cell

    def to_px(xy):
        xy = np.asarray(xy, float)
        return np.stack([(xy[..., 0] - x0) / c * px_per_cell, (ny - (xy[..., 1] - y0) / c) * px_per_cell], -1)

    for it in floor_items:
        ctr = np.array(it.position[:2])
        if it.role != "rug":
            f = np.array(it.front)
            a = to_px(ctr)
            b = to_px(ctr + f * min(0.45, max(it.size[1] * 0.6, 0.2)))
            cv2.arrowedLine(img, tuple(int(v) for v in a), tuple(int(v) for v in b), _color(it), 2, cv2.LINE_AA,
                            tipLength=0.3)
        p = to_px(ctr)
        _put_text(img, _label(it), (p[0] - 18, p[1] - 4), (255, 255, 255), 0.42)
    for it in layout.items:
        if it.mount == "wall":
            poly = to_px(np.asarray(it.footprint()))
            cv2.polylines(img, [np.round(poly).astype(np.int32)], True, _color(it), 3, cv2.LINE_AA)
            _put_text(img, _label(it), poly.mean(0) + np.array([4, -6]), _color(it), 0.4)
        elif it.mount == "on_top":
            p = to_px(np.array(it.position[:2]))
            cv2.circle(img, tuple(int(v) for v in p), 3, DECOR_COLOR, -1, cv2.LINE_AA)
    head = f"{layout.room_type} | estilo {layout.style} | {len(layout.items)} pecas | score {layout.score:.1f}"
    bar = np.zeros((26, img.shape[1], 3), np.uint8)
    _put_text(bar, head, (6, 18), (255, 255, 255), 0.5)
    return np.vstack([bar, img])


def _project_edges(room: RoomModel, box: np.ndarray, sx: float, sy: float):
    Minv = room.cam_from_room()
    intr = room.intr()
    out = []
    for i, j in _EDGES:
        p, q = box[i], box[j]
        pc = Minv[:3, :3] @ p + Minv[:3, 3]
        qc = Minv[:3, :3] @ q + Minv[:3, 3]
        near = 0.05
        if pc[2] < near and qc[2] < near:
            continue
        if pc[2] < near:
            pc = pc + (qc - pc) * (near - pc[2]) / (qc[2] - pc[2])
        elif qc[2] < near:
            qc = qc + (pc - qc) * (near - qc[2]) / (pc[2] - qc[2])
        uv = intr.project(np.stack([pc, qc]))
        out.append(uv * np.array([sx, sy]))
    return out


def layout_overlay(photo_rgb8: np.ndarray, room: RoomModel, layout: Layout, thickness: int | None = None,
                   show_openings: bool = True) -> np.ndarray:
    """Caixas 3D projetadas sobre a foto (RGB), de trás para frente, com rótulos."""
    img = np.ascontiguousarray(photo_rgb8[..., :3].copy())
    H, W = img.shape[:2]
    intr = room.intr()
    sx, sy = W / intr.width, H / intr.height
    th = thickness or max(1, int(round(W / 640)))
    cam = np.asarray(room.M)[:3, 3]
    if show_openings:
        for o in room.openings:
            w = room.walls[o.wall]
            d = w.direction
            base = np.asarray(w.p0)
            pts = []
            for s, z in ((o.s0, o.z0), (o.s1, o.z0), (o.s1, o.z1), (o.s0, o.z1)):
                xy = base + d * s
                pts.append([xy[0], xy[1], z])
            pts = np.array(pts)
            uv, z = room.project_room_points(pts)
            if np.all(z > 0.05):
                q = np.round(uv * np.array([sx, sy])).astype(np.int32)
                cv2.polylines(img, [q], True, (0, 160, 255) if o.kind == "window" else (255, 160, 0), th,
                              cv2.LINE_AA)
    items = sorted(layout.items, key=lambda it: -float(np.linalg.norm(np.array(it.position) - cam)))
    for it in items:
        box = geom.box_corners_3d(it.position[0], it.position[1], it.position[2], it.size[0], it.size[1],
                                  it.size[2], it.yaw_deg)
        col = _color(it)
        t = th if it.mount != "on_top" else max(1, th - 1)
        for seg in _project_edges(room, box, sx, sy):
            a, b = np.round(seg).astype(int)
            cv2.line(img, tuple(a), tuple(b), col, t, cv2.LINE_AA)
        if it.role != "rug":
            # seta da frente na base
            ctr = np.array([it.position[0], it.position[1], it.position[2] + 0.01])
            f = np.array([*it.front, 0.0])
            tip = ctr + f * max(0.15, it.size[1] * 0.6)
            uv, z = room.project_room_points(np.stack([ctr, tip]))
            if np.all(z > 0.05):
                a, b = np.round(uv * np.array([sx, sy])).astype(int)
                cv2.arrowedLine(img, tuple(a), tuple(b), col, t, cv2.LINE_AA, tipLength=0.25)
    for it in items:
        top = np.array([it.position[0], it.position[1], it.position[2] + it.size[2]])
        uv, z = room.project_room_points(top[None])
        if z[0] > 0.05:
            p = uv[0] * np.array([sx, sy])
            _put_text(img, _label(it), (p[0] - 20, p[1] - 6), _color(it), 0.45 * W / 960, max(1, th - 1))
    head = f"{layout.room_type} | estilo {layout.style} | score {layout.score:.1f}"
    _put_text(img, head, (10, int(30 * W / 960)), (255, 255, 255), 0.7 * W / 960, max(1, th))
    return img


def save_layout_debug(layout: Layout, room: RoomModel, grid: np.ndarray, photo_rgb8: np.ndarray,
                      out_dir: str | Path = "/tmp/claude-0/layout", prefix: str = "layout") -> dict[str, str]:
    """Grava mapa visto de cima (PNG) e aramado sobre a foto (JPG). Retorna os caminhos."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    td = layout_topdown(room, grid, layout)
    ov = layout_overlay(photo_rgb8, room, layout)
    p_td = out / f"{prefix}_topo.png"
    p_ov = out / f"{prefix}_foto.jpg"
    cv2.imwrite(str(p_td), cv2.cvtColor(td, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(p_ov), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"topo": str(p_td), "foto": str(p_ov)}
