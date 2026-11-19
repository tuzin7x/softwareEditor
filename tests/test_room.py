"""Reconstrução da sala em uma cena sintética com geometria conhecida."""
import numpy as np

from mobilia.analysis.geometry import GeometryResult, Intrinsics
from mobilia.analysis.room import FREE, build_room_model
from mobilia.analysis.segmentation import GROUP_IDS, SegmentationResult


def synthetic_room(W=320, H=240, cam_h=1.5, room=(4.0, 5.0, 2.7), pitch_deg=-8.0, yaw_deg=20.0):
    """Sala retangular: x em [-2, 2], y em [0.5, 5.5] (sala), câmera em (0.3, 0, cam_h)."""
    fx = 0.8 * W
    K = Intrinsics(W, H, fx, fx, W / 2, H / 2)
    rays = K.rays()                                   # câmera OpenCV
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    # câmera -> mundo (Z para cima): frente = +Y do mundo girado por yaw, inclinado por pitch
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    cv2w = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)      # x->x, z->y, -y->z
    R = Rz @ Rx @ cv2w
    d = rays @ R.T
    o = np.array([0.3, 0.0, cam_h])
    xmin, xmax, ymin, ymax, zt = -2.0, 2.0, -0.5, 5.5, room[2]
    planes = [("floor", np.array([0, 0, 1.0]), 0.0), ("ceiling", np.array([0, 0, 1.0]), zt),
              ("wall", np.array([1.0, 0, 0]), xmin), ("wall", np.array([1.0, 0, 0]), xmax),
              ("wall", np.array([0, 1.0, 0]), ymax), ("wall", np.array([0, 1.0, 0]), ymin)]
    t_best = np.full(d.shape[:2], np.inf)
    lab = np.full(d.shape[:2], 255, np.uint8)
    for name, n, c in planes:
        den = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (c - o @ n) / den
        t[(t <= 0) | ~np.isfinite(t)] = np.inf
        P = o + d * t[..., None]
        inside = (P[..., 0] >= xmin - 1e-6) & (P[..., 0] <= xmax + 1e-6) & (P[..., 1] >= ymin - 1e-6) & \
                 (P[..., 1] <= ymax + 1e-6) & (P[..., 2] >= -1e-6) & (P[..., 2] <= zt + 1e-6)
        t[~inside] = np.inf
        better = t < t_best
        t_best[better] = t[better]
        lab[better] = GROUP_IDS[name]
    pts_cam = rays * (t_best[..., None] * np.linalg.norm(rays, axis=-1, keepdims=True)) / np.linalg.norm(rays, axis=-1, keepdims=True)
    depth = pts_cam[..., 2]
    # normais na câmera
    nw = np.zeros_like(pts_cam)
    nw[lab == GROUP_IDS["floor"]] = (0, 0, 1)
    nw[lab == GROUP_IDS["ceiling"]] = (0, 0, -1)
    P = o + d * t_best[..., None]
    for name, n, c in planes[2:]:
        m = (lab == GROUP_IDS["wall"]) & (np.abs(P @ n - c) < 1e-4)
        inward = n if c < 0.5 else -n
        nw[m] = inward
    n_cam = nw @ R
    geo = GeometryResult(pts_cam.astype(np.float32), depth.astype(np.float32), n_cam.astype(np.float32),
                         np.isfinite(depth), K, "sintetico")
    labels = np.zeros(lab.shape, np.uint8)
    seg = SegmentationResult(labels, lab, {}, "sintetico")
    return geo, seg


def test_synthetic_room_reconstruction():
    geo, seg = synthetic_room()
    rgb = np.full(geo.depth.shape + (3,), 180, np.uint8)
    room, dbg = build_room_model(geo, seg, rgb)
    assert abs(room.camera_height - 1.5) < 0.03
    assert abs(room.ceiling_height - 2.7) < 0.06
    assert room.scale_factor == 1.0
    # paredes encontradas devem ser ortogonais entre si (sala de Manhattan)
    normals = [np.asarray(w.normal) for w in room.walls]
    assert len(normals) >= 2
    for a in normals:
        for b in normals:
            dot = abs(float(a @ b))
            assert dot < 0.05 or dot > 0.99
    assert (dbg["grid"] == FREE).sum() > 200


def test_room_json_roundtrip():
    geo, seg = synthetic_room()
    rgb = np.full(geo.depth.shape + (3,), 180, np.uint8)
    room, _ = build_room_model(geo, seg, rgb)
    from mobilia.analysis.room import RoomModel

    r2 = RoomModel.from_json(room.to_json())
    assert r2.walls[0].p0 == room.walls[0].p0 and r2.camera_height == room.camera_height
    # projeção de um ponto do piso e volta (consistência câmera <-> sala)
    uv, z = r2.project_room_points(np.array([[0.0, 3.0, 0.0]]))
    assert z[0] > 0 and 0 <= uv[0, 0] <= 320 and 0 <= uv[0, 1] <= 240
