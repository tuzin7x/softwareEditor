"""Casamento de câmera: pontos 3D renderizados pelo Blender caem exatamente onde a
projeção do modelo da sala prevê (mesmo código de câmera usado em produção)."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

SCRIPT = textwrap.dedent(r'''
import json, sys, math
import bpy, mathutils
import numpy as np
from mobilia.render import blender_worker as bw
spec = json.load(open(sys.argv[1]))
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 8
sc.render.resolution_x, sc.render.resolution_y = spec["width"], spec["height"]
sc.render.resolution_percentage = 100
sc.render.film_transparent = True
sc.view_settings.view_transform = "Standard"
sc.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
sc.render.image_settings.color_depth = "32"
bw.setup_camera(sc, spec)
for i, p in enumerate(spec["points"]):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=p)
    m = bpy.data.materials.new(f"e{i}"); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0, 0, 0, 1)
    b.inputs["Emission Color"].default_value = (1, 1, 1, 1); b.inputs["Emission Strength"].default_value = 30
    bpy.context.active_object.data.materials.append(m)
sc.render.filepath = spec["out"]
bpy.ops.render.render(write_still=True)
''')


def test_camera_projection_matches_room_model(tmp_path: Path):
    from scipy import ndimage

    from mobilia.analysis.geometry import Intrinsics
    from mobilia.composite.compositor import load_render_passes

    W, H = 240, 180
    intr = Intrinsics(W, H, 200.0, 200.0, 131.0, 84.0)          # ponto principal fora do centro
    # câmera: altura 1.4 m, olhando para +Y com yaw 25° e pitch -10°
    yaw, pitch = np.radians(25), np.radians(-10)
    cv2w = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Rx @ cv2w
    M[:3, 3] = [0, 0, 1.4]
    # pontos gerados a partir de pixels conhecidos (garante que estão no quadro)
    uvz = np.array([[30.0, 25.0, 3.0], [200.0, 40.0, 4.5], [70.0, 150.0, 2.5], [180.0, 140.0, 5.0], [120.0, 90.0, 3.7]])
    pc0 = np.stack([(uvz[:, 0] - intr.cx) / intr.fx * uvz[:, 2], (uvz[:, 1] - intr.cy) / intr.fy * uvz[:, 2], uvz[:, 2]], 1)
    pts = pc0 @ M[:3, :3].T + M[:3, 3]
    spec = {"width": W, "height": H, "intrinsics": intr.to_dict(), "cam_to_room": M.tolist(),
            "points": pts.tolist(), "out": str(tmp_path / "cam.exr")}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    scr = tmp_path / "cam_render.py"
    scr.write_text(SCRIPT)
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(scr), str(sp)], cwd=root, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(root)})
    assert r.returncode == 0, r.stderr[-2000:]
    rgba, _ = load_render_passes(tmp_path / "cam.exr")
    lum = rgba[..., :3].mean(-1)
    lab, n = ndimage.label(lum > 0.3)
    found = np.array(ndimage.center_of_mass(lum, lab, range(1, n + 1)))[:, ::-1] + 0.5   # (u, v), borda
    Minv = np.linalg.inv(M)
    pc = pts @ Minv[:3, :3].T + Minv[:3, 3]
    expected = intr.project(pc)
    assert n == len(pts)
    for e in expected:
        dist = np.min(np.linalg.norm(found - e, axis=1))
        assert dist < 0.35, (e, found)
