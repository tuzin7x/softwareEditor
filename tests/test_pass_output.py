"""PT3: o EXR de cada quadro sai do nó File Output só com Combined + Shadow Catcher, um arquivo de
uma execução anterior com o mesmo nome nunca passa por quadro recém-renderizado, e o mix de fade
só fica ligado durante o fade-in."""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = r'''
import json, os, sys, time
from pathlib import Path
import bpy
from mobilia.render import blender_worker as bw

out = Path(sys.argv[1])
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 2
sc.cycles.use_denoising = False
sc.render.resolution_x, sc.render.resolution_y = 32, 24
sc.render.film_transparent = True
im = sc.render.image_settings
im.media_type = "MULTI_LAYER_IMAGE"; im.color_depth = "16"; im.exr_codec = "ZIP"
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.location = (0, -4, 2); cam.rotation_euler = (1.1, 0, 0)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
bpy.ops.mesh.primitive_plane_add(size=6); bpy.context.object.is_shadow_catcher = True
sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN")); sc.collection.objects.link(sun)
sc.view_layers[0].cycles.use_pass_shadow_catcher = True
frames = out / "frames"; frames.mkdir(parents=True, exist_ok=True)
res = {}
# 1) caminho normal: o nó grava o quadro no nome pedido
po = {"node": bw.setup_pass_output(sc, frames)}
res["node_ok"] = po["node"] is not None
bw.render_frame(sc, frames / "f_00001.exr", po)
res["normal_node_kept"] = po["node"] is not None
# 2) restos de uma execução anterior (mesmo nome e um candidato do glob) e um nó que grava em
#    outro lugar: o quadro tem de ser regravado agora, não reaproveitado
(frames / "f_00002.exr").write_bytes(b"velho")
(frames / "f_000020002.exr").write_bytes(b"velho")
for q in ("f_00002.exr", "f_000020002.exr"):
    os.utime(frames / q, (time.time() - 3600, time.time() - 3600))
po["node"].directory = str(out / "outro")
bw.render_frame(sc, frames / "f_00002.exr", po)
res["fallback_node"] = po["node"] is None
# 3) mix de fade: ligado só em torno do fade-in; depois a saída volta ao shader original
from mobilia.render import choreography as ch
cube = bpy.data.objects["Cube"]
m = bpy.data.materials.new("m"); m.use_nodes = True; cube.data.materials.append(m)
bw._wrap_fade(m)
orig = m.node_tree.nodes["Principled BSDF"].outputs["BSDF"]
surf = m.node_tree.nodes[m["mobilia_fade_out"]].inputs["Surface"]
rt = {"parts": [cube], "plans": {cube.name: ch.PartPlan(cube.name, "drop", 1.0, 1.0, {"height": 0.4})}}
mats = bw.fade_materials([rt])
state = {}
for f in (13, 26, 60, 200):          # 24 fps: antes, no fade, assentando (opaco), parado
    bw.update_fade_shaders(mats, f, 24)
    state[f] = surf.links[0].from_socket == orig
res["fade_original"] = state
res["fade_changed_again"] = bw.update_fade_shaders(mats, 200, 24)
print("RESULT:" + json.dumps(res))
'''


@pytest.mark.skipif(importlib.util.find_spec("bpy") is None, reason="bpy não está instalado")
def test_pass_output_only_used_passes_and_never_stale(tmp_path: Path):
    import OpenEXR

    from mobilia.composite.compositor import load_render_passes

    env = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_SCRIPT), str(tmp_path)],
                          cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    res = json.loads(lines[-1][len("RESULT:"):])
    fade = res.pop("fade_original")
    assert res == {"node_ok": True, "normal_node_kept": True, "fallback_node": True,
                   "fade_changed_again": 0}, res
    # quadro 26 (t ≈ 1,04 s) em fade-in: mix ligado; depois do fade (60, 200) e antes da peça
    # aparecer (13), a saída é o shader original (sem BSDF transparente nas sombras)
    assert fade == {"13": True, "26": False, "60": True, "200": True}, fade
    frames = tmp_path / "frames"
    # quadro normal: só os dois passes que a composição usa
    f = OpenEXR.File(str(frames / "f_00001.exr"), separate_channels=True)
    names = {n for p in f.parts for n in p.channels}
    assert not any("Noisy" in n for n in names) and {"Combined.R", "Shadow Catcher.R"} <= names, names
    rgba, sh = load_render_passes(frames / "f_00001.exr")
    assert rgba.shape == (24, 32, 4) and sh is not None and rgba[..., 3].max() > 0.5
    # quadro do fallback: EXR novo e legível (não o resto da execução anterior)
    assert (frames / "f_00002.exr").read_bytes()[:4] == b"\x76\x2f\x31\x01"   # assinatura do OpenEXR
    rgba2, sh2 = load_render_passes(frames / "f_00002.exr")
    assert np.array_equal(rgba2[..., 3] > 0.5, rgba[..., 3] > 0.5)
