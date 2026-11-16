"""Peça em fade-in com muitas faces sobrepostas (render real do Blender, CPU).

Uma pilha de 12 placas com o material embrulhado por ``_wrap_fade`` e fade 0: com o limite de
transparência da sala (8), o Cycles encerra o raio como opaco e a peça "invisível" aparece com
alfa 1 (o teste morde). ``update_fade_shaders(..., sc)`` sobe o limite enquanto o mix está ligado,
e a pilha some; com o mix desligado (peça em repouso) o limite volta ao da sala."""
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
cam.location = (0, -4, 0.5); cam.rotation_euler = (1.5708, 0, 0)
m = bpy.data.materials.new("m"); m.use_nodes = True
parts = []
for i in range(12):
    bpy.ops.mesh.primitive_plane_add(size=0.6, location=(0, i * 0.02, 0.5), rotation=(1.5708, 0, 0))
    o = bpy.context.object; o.data.materials.append(m); o[bw.FADE_PROP] = 0.0; parts.append(o)
bw._wrap_fade(m)
sc.cycles.transparent_max_bounces = bw.TRANSPARENT_BOUNCES

def alpha(name):
    p = out / f"{name}.exr"
    sc.render.filepath = str(p)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(p))
    a = np.asarray(img.pixels[:], np.float32).reshape(48, 64, 4)[..., 3].copy()
    bpy.data.images.remove(img)
    return float(a.max())

res = {}
bw.set_fade_shader(m, True)
res["sem_ajuste"] = alpha("sem_ajuste")             # limite da sala: raio encerrado como opaco
# drop começando no quadro 2 (t = 1/24 s): no quadro 2 a peça é visível com fade 0
plans = {o.name: ch.PartPlan(o.name, "drop", 1 / 24, 1.0, {"height": 0.3}) for o in parts}
mats = bw.fade_materials([{"parts": parts, "plans": plans}])
bw.update_fade_shaders(mats, 2, 24, sc)
res["limite_fade"] = sc.cycles.transparent_max_bounces
res["com_ajuste"] = alpha("com_ajuste")
bw.update_fade_shaders(mats, 200, 24, sc)          # em repouso: mix desligado, limite da sala
res["limite_repouso"] = sc.cycles.transparent_max_bounces
print("RESULT:" + json.dumps(res))
'''


@pytest.mark.skipif(importlib.util.find_spec("bpy") is None, reason="bpy não está instalado")
def test_fade_raises_transparent_bounces(tmp_path: Path):
    env = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_SCRIPT), str(tmp_path)],
                          cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    res = json.loads(lines[-1][len("RESULT:"):])
    assert res["sem_ajuste"] == 1.0, res                    # o defeito: fade 0 e alfa 1
    assert res["limite_fade"] >= 64 and res["com_ajuste"] == 0.0, res
    assert res["limite_repouso"] == 8, res
