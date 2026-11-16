"""PT11a: o corte no plano do piso do "surgir do chão" (render real do Blender, CPU).

Um cubo com metade abaixo de z = 0 e o material embrulhado por ``_wrap_fade``: com o mix ligado,
o alfa é idêntico ao de um cubo que só tem a metade de cima, com e sem o catcher do piso (o corte
vale sozinho, e o catcher também esconde o que está abaixo dele). Sem o mix e sem o piso, a metade
de baixo aparece (o teste morde). O mix fica ligado enquanto a peça surge e volta ao shader
original quando ela assenta."""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPT = r'''
import json, sys
from pathlib import Path
import bpy
import numpy as np
from mobilia.render import blender_worker as bw
from mobilia.render import choreography as ch

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 8
sc.cycles.use_denoising = False; sc.cycles.seed = 3
sc.render.resolution_x, sc.render.resolution_y = 64, 48
sc.render.film_transparent = True
sc.render.image_settings.file_format = "OPEN_EXR"; sc.render.image_settings.color_depth = "32"
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.location = (0, -4, 0.6); cam.rotation_euler = (1.45, 0, 0)
sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN")); sc.collection.objects.link(sun)
sun.data.energy = 3.0
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))     # z de -0,5 a 0,5: metade abaixo do piso
cube = bpy.context.object
m = bpy.data.materials.new("m"); m.use_nodes = True; cube.data.materials.append(m)
bw._wrap_fade(m)
cube[bw.FADE_PROP] = 1.0
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.25)); ref = bpy.context.object
ref.scale = (1, 1, 0.5); ref.data.materials.append(bpy.data.materials.new("r"))
bpy.ops.mesh.primitive_plane_add(size=20); floor = bpy.context.object; floor.is_shadow_catcher = True

def alpha(name):
    p = out / f"{name}.exr"
    sc.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(p))
    a = np.asarray(img.pixels[:], np.float32).reshape(48, 64, 4)[::-1, :, 3].copy()
    bpy.data.images.remove(img)
    return a

res = {}
for floor_on in (False, True):
    floor.hide_render = not floor_on
    ref.hide_render, cube.hide_render = False, True
    a_ref = alpha(f"ref_{floor_on}")
    ref.hide_render, cube.hide_render = True, False
    bw.set_fade_shader(m, True)
    a_clip = alpha(f"clip_{floor_on}")
    bw.set_fade_shader(m, False)
    a_raw = alpha(f"raw_{floor_on}")
    res[str(floor_on)] = {"clip_vs_ref": float(np.abs(a_clip - a_ref).max()),
                          "raw_vs_ref": float(np.abs(a_raw - a_ref).max()),
                          "raw_extra_px": int(((a_raw > 0.5) & (a_ref < 0.01)).sum())}
# o mix fica ligado enquanto a peça surge do chão e volta ao original no repouso
plan = ch.PartPlan(cube.name, "emerge", 1.0, 1.0, {"emerge_depth": 0.51})
mats = bw.fade_materials([{"parts": [cube], "plans": {cube.name: plan}}])
state = {}
for f in (30, 36, 120):
    bw.update_fade_shaders(mats, f, 24)
    surf = m.node_tree.nodes[m["mobilia_fade_out"]].inputs["Surface"]
    state[f] = surf.links[0].from_socket.node.type
res["mix_state"] = state
print("RESULT:" + json.dumps(res))
'''


@pytest.mark.skipif(importlib.util.find_spec("bpy") is None, reason="bpy não está instalado")
def test_emerge_floor_clip_is_exact(tmp_path: Path):
    env = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_SCRIPT), str(tmp_path)],
                          cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    res = json.loads(lines[-1][len("RESULT:"):])
    no_floor, floor = res["False"], res["True"]
    # sem piso nenhum: só o corte do shader; nada abaixo de z = 0 aparece
    assert no_floor["clip_vs_ref"] == 0.0, no_floor
    assert no_floor["raw_extra_px"] > 50, no_floor                 # sem o corte, a metade de baixo aparece
    # com o catcher do piso: idem, e o catcher sozinho também esconde
    assert floor["clip_vs_ref"] == 0.0 and floor["raw_vs_ref"] == 0.0, floor
    # quadros 30 e 36 (24 fps): subindo (mix ligado); 120: em repouso, shader original
    assert res["mix_state"] == {"30": "MIX_SHADER", "36": "MIX_SHADER", "120": "BSDF_PRINCIPLED"}, res
