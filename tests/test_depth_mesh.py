"""Malha de oclusão de objetos existentes na foto."""
import numpy as np

from mobilia.analysis.depth_mesh import occluder_mesh
from mobilia.analysis.room import build_room_model
from mobilia.analysis.segmentation import GROUP_IDS

from test_room import synthetic_room


def test_occluder_mesh_follows_existing_object():
    geo, seg = synthetic_room()
    # marca um "armário" existente: um bloco de pixels da parede do fundo vira 'fixture'
    H, W = seg.groups.shape
    wall = seg.groups == GROUP_IDS["wall"]
    ys, xs = np.nonzero(wall)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    box = np.zeros_like(wall)
    box[cy - 15:cy + 15, cx - 20:cx + 20] = True
    seg.groups[box & wall] = GROUP_IDS["fixture"]
    rgb = np.full((H, W, 3), 180, np.uint8)
    room, _ = build_room_model(geo, seg, rgb)
    verts, faces = occluder_mesh(geo, seg, room, step=2)
    assert len(verts) > 50 and len(faces) > 50
    assert faces.max() < len(verts)
    # os vértices ficam ligeiramente ATRÁS da superfície real (recuo ao longo do raio)
    cam = room.M[:3, 3]
    d_mesh = np.linalg.norm(verts - cam, axis=1)
    pts = geo.points[box & wall] @ room.M[:3, :3].T + room.M[:3, 3]
    d_real = np.linalg.norm(pts - cam, axis=1)
    assert np.median(d_mesh) > np.median(d_real) * 1.005


def test_no_mesh_for_empty_room():
    geo, seg = synthetic_room()
    rgb = np.full(seg.groups.shape + (3,), 180, np.uint8)
    room, _ = build_room_model(geo, seg, rgb)
    verts, faces = occluder_mesh(geo, seg, room)
    assert len(verts) == 0 and len(faces) == 0
