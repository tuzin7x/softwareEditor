"""Decoração "on_top" (livros, vaso, fruteira, abajur, almofadas...) sempre inteira sobre o apoio.

Regressão do teste real do usuário: na sala da PoC os livros ficaram metade sobre a mesa de centro
e metade no ar. Causa: o glTF de ``modern_coffee_table_01`` vem com o eixo longo em Y (0.60 x 1.20)
e o catálogo não tinha ``yaw_fix=90``; o layout planejava sobre um tampo 1.20 x 0.60 enquanto o
render desenhava a mesa atravessada (0.60 de largura), então tudo que ficava a mais de 0.30 m do
centro no eixo longo pendia para fora. Além disso a decoração era encostada rente à borda.

Os tamanhos crus dos glTF ficam em ``fixtures/polyhaven_raw_sizes.json`` (medidos no Blender); o
teste lento confere o fixture e os tampos medidos contra as malhas reais em cache.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mobilia.layout import catalog, geom
from mobilia.layout.solver import DECOR_TARGETS, PROC_KIND, Solver
from mobilia.layout.surfaces import MEASURED, ON_TOP_INSET, fits_on, rotated_extent, support_for

FIX = Path(__file__).parent / "fixtures"
RAW = {k: v for k, v in json.loads((FIX / "polyhaven_raw_sizes.json").read_text(encoding="utf-8")).items()
       if not k.startswith("_")}


def rendered_size(asset_id: str, yaw_fix: float | None = None) -> tuple[float, float, float]:
    """(X, Y, Z) da malha como o worker a coloca na cena: glTF cru girado por ``yaw_fix``."""
    sx, sy, sz = RAW[asset_id]
    yf = catalog.get(asset_id).yaw_fix if yaw_fix is None else yaw_fix
    ex, ey = rotated_extent(sx, sy, yf)
    return ex, ey, sz


def corners_local(par_xy, par_yaw, xy, yaw, w, d) -> np.ndarray:
    """Cantos da base do filho no referencial local do apoio."""
    c = geom.obb_corners(xy[0], xy[1], w, d, yaw)
    return geom.world_to_local(par_xy[0], par_xy[1], par_yaw, c)


# ----------------------------------------------------------------------------- catálogo x malha

def test_every_polyhaven_asset_has_raw_size():
    ph = {a.asset_id for a in catalog.CATALOG if a.asset_id.startswith("ph:")}
    assert ph <= set(RAW), sorted(ph - set(RAW))


@pytest.mark.parametrize("asset_id", sorted(RAW))
def test_catalog_footprint_matches_rendered_mesh(asset_id):
    """A pegada do catálogo (que o layout usa) é a pegada que o render desenha (após yaw_fix)."""
    spec = catalog.get(asset_id)
    ex, ey, ez = rendered_size(asset_id)
    assert abs(ex - spec.size[0]) <= 0.02 and abs(ey - spec.size[1]) <= 0.02, (asset_id, (ex, ey), spec.size)
    assert abs(ez - spec.size[2]) <= 0.02


@pytest.mark.parametrize("asset_id", sorted(MEASURED))
def test_measured_support_inside_rendered_mesh(asset_id):
    """Cada área de apoio medida cabe na malha COMO RENDERIZADA (e na pegada do catálogo)."""
    sup = MEASURED[asset_id]
    spec = catalog.get(asset_id)
    ex, ey, ez = rendered_size(asset_id)
    assert sup.z <= ez + 0.005
    for x0, y0, x1, y1 in sup.rects:
        for hw, hd in ((ex / 2, ey / 2), (spec.size[0] / 2, spec.size[1] / 2)):
            assert -hw - 0.005 <= x0 < x1 <= hw + 0.005, (asset_id, sup.rects, hw)
            assert -hd - 0.005 <= y0 < y1 <= hd + 0.005, (asset_id, sup.rects, hd)


# ----------------------------------------------------------------------------- caso exato da PoC

POC_TABLE = {"asset_id": "ph:modern_coffee_table_01", "xy": (-1.7611, 2.9781), "yaw": 90.0, "size": (1.201, 0.6, 0.389)}
POC_CHILDREN = {   # layout.json da PoC (sala stafford, semente 0)
    "books_0": ("ph:book_encyclopedia_set_01", (-1.5726, 3.2616), 90.0, (0.551, 0.163)),
    "vase_0": ("ph:ceramic_vase_01", (-1.9281, 3.4351), 90.0, (0.204, 0.206)),
    "bowl_0": ("ph:wooden_bowl_01", (-1.7611, 2.7081), 90.0, (0.313, 0.309)),
}


def _inside_rendered_top(yaw_fix: float, key: str) -> bool:
    ex, ey, _ = rendered_size(POC_TABLE["asset_id"], yaw_fix)   # tampo desta mesa = bbox inteiro
    _, xy, yaw, (w, d) = POC_CHILDREN[key]
    c = corners_local(POC_TABLE["xy"], POC_TABLE["yaw"], xy, yaw, w, d)
    return bool(np.all(np.abs(c[:, 0]) <= ex / 2 + 1e-4) and np.all(np.abs(c[:, 1]) <= ey / 2 + 1e-4))


def test_poc_books_were_hanging_off_without_yaw_fix():
    """Reproduz o defeito: sem yaw_fix a mesa renderizada fica atravessada e os livros pendem."""
    assert not _inside_rendered_top(0.0, "books_0")
    assert not _inside_rendered_top(0.0, "vase_0")


def test_poc_decor_is_on_the_rendered_table_top():
    assert catalog.get(POC_TABLE["asset_id"]).yaw_fix in (90.0, -90.0)
    for key in POC_CHILDREN:
        assert _inside_rendered_top(catalog.get(POC_TABLE["asset_id"]).yaw_fix, key), key


def test_poc_coffee_table_decor_is_inset_from_the_edge():
    """O mesmo tampo, mesma ordem da PoC (fruteira, livros, vaso): tudo recuado da borda."""
    par = SimpleNamespace(role="coffee_table", key="coffee_table_0")
    sup = support_for(POC_TABLE["asset_id"], "coffee_table", POC_TABLE["size"])
    ex, ey, _ = rendered_size(POC_TABLE["asset_id"])
    used: list = []
    for role in ("bowl", "books", "vase"):
        spec = next(a for a in catalog.by_role(role) if a.asset_id == POC_CHILDREN[f"{role}_0"][0])
        w, d, _ = spec.size
        spot = Solver._decor_spot(None, role, par, sup, spec.size, used)
        assert spot is not None, role
        lx, ly = spot
        assert fits_on(sup, lx, ly, w, d, margin=-ON_TOP_INSET + 1e-6), (role, spot)
        assert abs(lx) + w / 2 <= ex / 2 - ON_TOP_INSET and abs(ly) + d / 2 <= ey / 2 - ON_TOP_INSET


# ----------------------------------------------------------------------------- propriedade genérica

def _parents(role: str):
    """(asset_id, tamanho, params) de todos os apoios possíveis de um papel."""
    out = []
    for a in catalog.by_role(role):
        if a.asset_id.startswith("ph:"):
            out.append((a.asset_id, a.size, {}))
        else:
            for k in (1.0, 0.8, 1.25):
                out.append((a.asset_id, (a.size[0] * k, a.size[1] * k, a.size[2]), {}))
    if not out and role in PROC_KIND:
        spec = catalog.by_role(role)[0]
        out.append((f"proc:{PROC_KIND[role]}", spec.size, {}))
    return out


def _children(role: str):
    specs = catalog.by_role(role)
    if role in PROC_KIND:
        return [(f"proc:{PROC_KIND[role]}", specs[0].size)]
    return [(a.asset_id, a.size) for a in specs]


COMBOS = sorted({(c, p) for c, ps in DECOR_TARGETS.items() for p in ps})


@pytest.mark.parametrize("child_role,parent_role", COMBOS)
def test_decor_spots_always_fit_the_support(child_role, parent_role):
    """Para TODO par (decoração, apoio) do catálogo, em vários giros do apoio e com até 4 objetos
    disputando o mesmo tampo: a base inteira do filho, girada para o referencial do apoio, fica
    dentro da área útil (tampos: recuada ON_TOP_INSET da borda; assentos: largura toda sobre o
    assento) e dentro da pegada do móvel."""
    checked = possible = 0
    for pid, psize, params in _parents(parent_role):
        sup = support_for(pid, parent_role, psize, params)
        if sup is None:
            continue
        for cid, csize in _children(child_role):
            w, d, _ = (float(v) for v in csize)
            r = max(sup.rects, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
            if sup.kind == "top":
                possible += w + 2 * ON_TOP_INSET <= r[2] - r[0] and d + 2 * ON_TOP_INSET <= r[3] - r[1]
            else:
                possible += w <= r[2] - r[0] + 0.02
            for pyaw in (0.0, 37.0, 90.0, 180.0, -90.0):
                par = SimpleNamespace(role=parent_role, key="p")
                used: list = []
                pxy = (1.3, 2.1)
                for _ in range(4):
                    spot = Solver._decor_spot(None, child_role, par, sup, csize, used)
                    if spot is None:
                        break
                    xy = geom.local_to_world(pxy[0], pxy[1], pyaw, *spot)
                    c = corners_local(pxy, pyaw, xy, pyaw, w, d)        # o filho herda o yaw do apoio
                    lo, hi = c.min(0), c.max(0)
                    assert np.all(np.abs(c[:, 0]) <= psize[0] / 2 + 1e-6), (pid, cid, pyaw, spot)
                    if sup.kind == "top":
                        assert np.all(np.abs(c[:, 1]) <= psize[1] / 2 + 1e-6), (pid, cid, pyaw, spot)
                        ok = any(lo[0] >= r[0] + ON_TOP_INSET - 1e-5 and hi[0] <= r[2] - ON_TOP_INSET + 1e-5 and
                                 lo[1] >= r[1] + ON_TOP_INSET - 1e-5 and hi[1] <= r[3] - ON_TOP_INSET + 1e-5
                                 for r in sup.rects)
                    else:
                        ok = any(lo[0] >= r[0] - 0.02 and hi[0] <= r[2] + 0.02 and r[1] <= spot[1] <= r[3]
                                 for r in sup.rects)
                    assert ok, (pid, cid, pyaw, spot, sup.rects)
                    checked += 1
                    if sup.kind == "seat" or child_role == "table_lamp":
                        break
    # se algum filho cabe geometricamente em algum apoio, o solver acha lugar para ele
    # (ex.: livros de 0.55 m não cabem em criado-mudo; almofadas de 0.94 m não cabem em poltrona)
    assert (checked > 0) == (possible > 0), (checked, possible)


def test_layouts_keep_decor_inside_support():
    """Layouts completos (sala real e sintéticas): decoração dentro do apoio, sem beirada no ar."""
    from test_layout import CASES, get_layout

    from mobilia.layout.state import Placement

    for name, room_type in CASES:
        lay = get_layout(name, room_type)
        items = {it.id: Placement.from_item(it) for it in lay.items}
        for p in items.values():
            if p.mount != "on_top":
                continue
            par = items[p.parent]
            sup = support_for(par.asset_id, par.role, (par.w, par.d, par.h), par.params)
            assert abs(p.z - (par.z + sup.z)) < 0.005
            if sup.kind != "top":
                continue
            ew, ed = rotated_extent(p.w, p.d, p.yaw - par.yaw)
            lxy = geom.world_to_local(par.x, par.y, par.yaw, p.xy[None])[0]
            assert fits_on(sup, lxy[0], lxy[1], ew, ed, margin=-ON_TOP_INSET + 1e-3), (name, room_type, p.key)
            if par.asset_id in RAW:
                ex, ey, _ = rendered_size(par.asset_id)
                assert abs(lxy[0]) + ew / 2 <= ex / 2 and abs(lxy[1]) + ed / 2 <= ey / 2, (name, p.key)


# ----------------------------------------------------------------------------- conferência nas malhas reais

def _cached_gltf(asset_id: str) -> Path | None:
    home = Path(os.environ.get("MOBILIA_HOME", Path.home() / "mobilia_data"))
    hits = sorted((home / "assets" / "polyhaven" / asset_id[3:] / "1k").glob("*.gltf"))
    return hits[0] if hits else None


@pytest.mark.slow
def test_raw_sizes_fixture_matches_gltf():
    """Recalcula no Blender o bbox cru de cada glTF em cache e, para os tampos que recebem
    decoração, confere por raio vertical que a área medida é plana na altura declarada."""
    bpy = pytest.importorskip("bpy")
    import mathutils
    from mathutils.bvhtree import BVHTree

    top_parents = {p for ps in DECOR_TARGETS.values() for p in ps}
    seen = 0
    for asset_id, raw in RAW.items():
        path = _cached_gltf(asset_id)
        if path is None:
            continue
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(path))
        bpy.context.view_layer.update()
        spec = catalog.get(asset_id)
        R = mathutils.Matrix.Rotation(math.radians(spec.yaw_fix), 4, "Z")
        verts, tris = [], []
        for o in bpy.data.objects:
            if o.type != "MESH":
                continue
            m = o.matrix_world
            off = len(verts)
            o.data.calc_loop_triangles()
            verts += [tuple(m @ v.co) for v in o.data.vertices]
            tris += [tuple(off + i for i in t.vertices) for t in o.data.loop_triangles]
        V = np.array(verts)
        assert np.allclose(V.max(0) - V.min(0), raw, atol=0.003), (asset_id, V.max(0) - V.min(0), raw)
        seen += 1
        sup = MEASURED.get(asset_id)
        if sup is None or sup.kind != "top" or spec.role not in top_parents:
            continue
        V = np.array([tuple(R @ mathutils.Vector(v)) for v in V])
        mn, mx = V.min(0), V.max(0)
        V -= [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mn[2]]
        bvh = BVHTree.FromPolygons([tuple(v) for v in V], tris)
        for x0, y0, x1, y1 in sup.rects:
            hits = []
            for y in np.arange(y0 + 0.005, y1, 0.02):
                for x in np.arange(x0 + 0.005, x1, 0.02):
                    loc = bvh.ray_cast(mathutils.Vector((x, y, mx[2] - mn[2] + 1)), mathutils.Vector((0, 0, -1)))[0]
                    hits.append(loc is not None and abs(loc.z - sup.z) < 0.01)
            # tampo ripado (coverage < 1): frestas entre as ripas
            assert np.mean(hits) >= min(0.97, sup.coverage), (asset_id, float(np.mean(hits)))
    if seen == 0:
        pytest.skip("nenhum glTF Poly Haven em cache")


_REST_SCRIPT = r'''
import json, sys
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
from mobilia.render import blender_worker as bw

path, yaw_fix = sys.argv[1], float(sys.argv[2])
bw._check_footprint = lambda item, meshes: None      # simula o defeito antigo (sem a correção automática)
table = {"id": "coffee_table_0", "role": "coffee_table", "asset_id": "ph:modern_coffee_table_01",
         "size": [1.201, 0.6, 0.389], "position": [0.0, 0.0, 0.0], "yaw_deg": 0.0, "yaw_fix": yaw_fix,
         "mount": "floor", "params": {}}
root, parts = bw.import_gltf_item(path, table)
bw.place_root(root, table)
items_rt = [{"item": table, "root": root, "parts": parts, "plans": {}}]
for key, x, kind, params, size in (("books_0", 0.35, "book_stack", {}, [0.33, 0.26, 0.085]),
                                   ("books_1", 0.0, "book_stack", {"count": 2}, [0.3, 0.24, 0.06])):
    it = {"id": key, "role": "books", "asset_id": f"proc:{kind}", "size": size, "position": [x, 0.0, 0.39],
          "yaw_deg": 0.0, "mount": "on_top", "parent": "coffee_table_0",
          "source": {"type": "procedural", "kind": kind, "params": dict(params, size=size)}}
    r, ps = bw.build_procedural_item(it)
    bw.place_root(r, it)
    items_rt.append({"item": it, "root": r, "parts": ps, "plans": {}})
removed = bw._check_rest_support(items_rt)
print("RESULT:" + json.dumps({"removed": removed, "kept": [rt["item"]["id"] for rt in items_rt],
                              "objects": sorted(o.name for o in bpy.data.objects if o.name.startswith("books"))}))
'''


@pytest.mark.slow
@pytest.mark.parametrize("yaw_fix", [0.0, 90.0])
def test_check_rest_support_removes_hanging_decor(yaw_fix):
    """Rede de segurança do worker: com a mesa de centro desenhada atravessada (yaw_fix=0 e sem a
    correção automática de pegada), a pilha de livros na ponta do tampo fica sem apoio e sai da
    cena; a do centro continua. Com o yaw_fix certo nada é removido."""
    import subprocess
    import sys
    import textwrap

    pytest.importorskip("bpy")
    path = _cached_gltf("ph:modern_coffee_table_01")
    if path is None:
        pytest.skip("glTF da mesa de centro não está em cache")
    env = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_REST_SCRIPT), str(path), str(yaw_fix)],
                          cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    out = json.loads(lines[-1][len("RESULT:"):])
    if yaw_fix == 0.0:
        assert [r["id"] for r in out["removed"]] == ["books_0"], out
        assert out["removed"][0]["apoio_acertos"] < 0.9
        assert "books_0" not in out["kept"] and "books_1" in out["kept"]
        assert not any(n.startswith("books_0") for n in out["objects"]), out["objects"]
    else:
        assert out["removed"] == [], out


_SLAT_SCRIPT = r'''
import json
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
from mobilia.render import blender_worker as bw

# tampo ripado sintético na altura medida da mesa ripada: ripas de 5 cm ao longo de Y, frestas de
# 6 cm, a fresta do meio centrada em x = 0; o tampo vai de x = -0.61 a 0.63
slats = []
for k in range(-5, 6):
    x0 = 0.03 + 0.11 * k
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x0 + 0.025, 0.0, 0.359), scale=(0.05, 1.2, 0.02))
    slats.append(bpy.context.active_object)
table = {"id": "coffee_table_0", "role": "coffee_table", "asset_id": "ph:modern_coffee_table_02",
         "size": [1.199, 1.199, 0.369], "position": [0.0, 0.0, 0.0], "yaw_deg": 0.0, "mount": "floor", "params": {}}
items_rt = [{"item": table, "root": None, "parts": slats, "plans": {}}]
for key, x, z in (("books_center", 0.0, 0.369), ("books_edge", 0.6, 0.369), ("books_float", -0.3, 0.40)):
    size = [0.33, 0.26, 0.085]
    it = {"id": key, "role": "books", "asset_id": "proc:book_stack", "size": size, "position": [x, 0.0, z],
          "yaw_deg": 0.0, "mount": "on_top", "parent": "coffee_table_0",
          "source": {"type": "procedural", "kind": "book_stack", "params": {"size": size}}}
    r, ps = bw.build_procedural_item(it)
    bw.place_root(r, it)
    items_rt.append({"item": it, "root": r, "parts": ps, "plans": {}})
bpy.context.view_layer.update()
bvh = bw._world_bvh(slats)
plain = bw.rest_support_hits(bvh, items_rt[1]["item"], 0.369)
removed = bw._check_rest_support(items_rt)
print("RESULT:" + json.dumps({"plain_center": plain, "removed": sorted(r["id"] for r in removed),
                              "kept": [rt["item"]["id"] for rt in items_rt[1:]]}))
'''


@pytest.mark.skipif(importlib.util.find_spec("bpy") is None, reason="bpy não está instalado")
def test_check_rest_support_slatted_top_bridges_gaps():
    """Tampo ripado: a pilha no meio, com várias linhas da grade caindo nas frestas, está apoiada e
    fica (a regra "fração de acertos × coverage" a removia); a que passa da borda externa e a que
    flutua 3 cm acima do tampo saem."""
    import subprocess
    import sys
    import textwrap

    env = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_SLAT_SCRIPT)],
                          cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    out = json.loads(lines[-1][len("RESULT:"):])
    assert out["plain_center"] < 0.9 * MEASURED["ph:modern_coffee_table_02"].coverage, out   # cenário discrimina
    assert out["removed"] == ["books_edge", "books_float"], out
    assert out["kept"] == ["books_center"], out
