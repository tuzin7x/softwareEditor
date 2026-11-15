"""Malha de oclusão para objetos que JÁ existem na foto (bancadas, armários, porta aberta...).

As paredes/piso/teto da sala-proxy já ocultam o que fica atrás deles. Objetos não
planares — ilha de cozinha, geladeira, folha de porta, coluna, vaso sanitário — viram
uma malha triangulada a partir da nuvem de pontos do MoGe-2, recuada levemente ao
longo do raio da câmera. No Blender essa malha é um *shadow catcher*: oculta os móveis
que estiverem atrás dela e recebe as sombras dos móveis que estiverem na frente.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import GeometryResult
from .room import RoomModel
from .segmentation import SegmentationResult

OCCLUDER_GROUPS = ("fixture", "structure", "furniture", "door")


def occluder_mesh(geo: GeometryResult, seg: SegmentationResult, room: RoomModel, step: int = 4,
                  max_rel_jump: float = 0.035, push: float = 0.012,
                  groups: tuple[str, ...] = OCCLUDER_GROUPS) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (vértices no referencial da sala, faces triangulares). Vazio se não houver objetos."""
    if seg.groups.shape != geo.depth.shape:
        raise ValueError("segmentação e geometria precisam ter a mesma resolução")
    m = np.zeros(seg.groups.shape, bool)
    for g in groups:
        m |= seg.mask(g)
    m = ndimage.binary_dilation(m, iterations=2) & geo.mask & (geo.depth > 0.1)
    if m.sum() < 400:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    H, W = m.shape
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    sub_m = m[np.ix_(ys, xs)]
    P = geo.points[np.ix_(ys, xs)].astype(np.float64)
    D = geo.depth[np.ix_(ys, xs)].astype(np.float64)
    # recua ao longo do raio (evita que o móvel encostado seja "cortado" por imprecisão)
    P = P * (1.0 + push)
    idx = -np.ones(sub_m.shape, np.int64)
    idx[sub_m] = np.arange(int(sub_m.sum()))
    M = room.M
    verts = P[sub_m] @ M[:3, :3].T + M[:3, 3]
    faces = []
    a = idx[:-1, :-1]; b = idx[:-1, 1:]; c = idx[1:, :-1]; d = idx[1:, 1:]
    da, db, dc, dd = D[:-1, :-1], D[:-1, 1:], D[1:, :-1], D[1:, 1:]
    stack = np.stack([da, db, dc, dd], -1)
    rel = (stack.max(-1) - stack.min(-1)) / np.maximum(stack.min(-1), 1e-6)
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0) & (rel < max_rel_jump * step)
    faces = np.concatenate([np.stack([a[ok], c[ok], b[ok]], 1), np.stack([b[ok], c[ok], d[ok]], 1)])
    if len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    # remove vértices não usados
    used = np.unique(faces)
    remap = -np.ones(len(verts), np.int64)
    remap[used] = np.arange(len(used))
    return verts[used].astype(np.float32), remap[faces].astype(np.int64)
