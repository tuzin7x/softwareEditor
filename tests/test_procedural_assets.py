"""Testes dos assets procedurais (Blender).

Os assets rodam dentro do ``bpy``; por isso cada verificação acontece num
subprocesso Python isolado que constrói os assets, mede a geometria AVALIADA
(depsgraph) e devolve um relatório JSON para as asserções abaixo.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(importlib.util.find_spec("bpy") is None,
                                reason="bpy (Blender como módulo Python) não está instalado")

KINDS = ("rug", "floor_lamp", "table_lamp", "bed", "curtain", "desk", "throw_blanket", "book_stack", "canvas_art")
ART_COLORS = ["#b5603e", "#844b34", "#b6a99a", "#4a4440"]
MOTIONS = {"drop", "rise", "slide", "unroll", "unfold", "grow", "settle"}

# (rótulo, tipo, parâmetros) — padrões do catálogo + variações de tamanho/estilo
CASES: list[tuple[str, str, dict]] = [(k, k, {}) for k in KINDS] + [
    ("rug_stripes_3x2", "rug", {"size": [3.0, 2.0, 0.015], "pattern": "stripes"}),
    ("rug_berber_portrait", "rug", {"size": [1.2, 1.8, 0.01], "pattern": "berber", "fringe": True}),
    ("bed_panel_tall", "bed", {"size": [1.4, 2.05, 1.1], "pillows": 2, "headboard": "panel",
                               "legs": "black", "fabric": "velvet", "fabric_color": "#5b6b5e"}),
    ("bed_single", "bed", {"size": [0.95, 2.0, 0.85], "pillows": 3, "fabric": "boucle"}),
    ("curtain_sheer_wide", "curtain", {"size": [2.4, 0.14, 2.7], "panels": 1, "sheer": True,
                                       "metal": "brass"}),
    ("desk_walnut_metal", "desk", {"size": [1.2, 0.6, 0.74], "legs": "black", "drawer": False,
                                   "wood": "walnut"}),
    ("desk_painted", "desk", {"legs": "painted", "paint_color": "#f1efe9"}),
    ("floor_lamp_brass", "floor_lamp", {"size": [0.5, 0.5, 1.75], "metal": "brass",
                                        "shade_shape": "tapered"}),
    ("table_lamp_wood", "table_lamp", {"size": [0.35, 0.35, 0.6], "base_material": "wood",
                                       "shade_shape": "drum"}),
    ("throw_solid", "throw_blanket", {"size": [1.3, 0.45, 0.08], "pattern": "solid", "tassels": False}),
    ("book_stack_2", "book_stack", {"size": [0.3, 0.24, 0.06], "count": 2, "colors": ["#2f3e46", "#d8c3a5"]}),
    # parâmetros como o motor de layout envia (paleta sRGB 0..1, nomes em português)
    ("layout_rug_persa", "rug", {"size": [2.4, 1.7, 0.012], "color": [0.48, 0.2, 0.16],
                                 "pattern_color": [0.14, 0.2, 0.29], "pattern": "persa", "thickness": 0.012}),
    ("layout_rug_listras", "rug", {"color": [0.85, 0.82, 0.76], "pattern_color": [0.36, 0.3, 0.25],
                                   "pattern": "listras", "thickness": 0.012}),
    ("layout_bed_wood", "bed", {"size": [1.68, 2.12, 1.0], "frame_color": [0.643, 0.478, 0.314],
                                "linen_color": [0.94, 0.92, 0.89], "accent_color": [0.55, 0.42, 0.28],
                                "mattress_top": 0.56, "headboard_height": 1.0, "headboard_depth": 0.08}),
    ("layout_bed_laca", "bed", {"size": [1.08, 2.02, 0.9], "frame_color": [0.925, 0.914, 0.89],
                                "linen_color": [0.95, 0.94, 0.92], "accent_color": [0.6, 0.65, 0.56],
                                "mattress_top": 0.56}),
    ("layout_desk", "desk", {"size": [1.6, 0.75, 0.75], "top_color": [0.643, 0.478, 0.314],
                             "leg_color": [0.122, 0.122, 0.129]}),
    ("layout_floor_lamp", "floor_lamp", {"shade_color": [0.91, 0.88, 0.83], "metal_color": [0.69, 0.553, 0.341],
                                         "light_on": True}),
    ("layout_table_lamp", "table_lamp", {"base_color": [0.79, 0.64, 0.49], "shade_color": [0.93, 0.91, 0.86],
                                         "metal_color": [0.929, 0.922, 0.902], "light_on": True}),
    ("layout_book_stack", "book_stack", {"size": [0.33, 0.26, 0.085], "count": 3,
                                         "colors": [[0.2, 0.18, 0.16], [0.86, 0.83, 0.78], [0.55, 0.3, 0.2]]}),
    # quadros gerados (PT9a): tela de 3,5 cm e moldura fina com passe-partout, tamanhos livres
    ("canvas_tela_paisagem", "canvas_art", {"size": [1.2, 0.035, 0.8], "frame": "tela", "colors": ART_COLORS,
                                            "ground": "#ede9e1", "art_style": "campos", "seed": 3}),
    ("canvas_moldura_retrato", "canvas_art", {"size": [0.45, 0.035, 0.6], "frame": "moldura",
                                              "colors": ART_COLORS, "art_style": "formas",
                                              "frame_color": "#1f1f21", "mat_color": "#efeae3", "seed": 4}),
    ("canvas_moldura_madeira", "canvas_art", {"size": [0.6, 0.035, 0.6], "frame": "moldura", "art_style": "arcos",
                                              "frame_color": [0.64, 0.48, 0.31], "passe_partout": False}),
    # tapete texturizado (PT9c): mapas sintéticos gravados pelo subprocesso no lugar da textura CC0
    ("rug_textured", "rug", {"size": [2.4, 1.7, 0.012], "texture": "__synthetic__", "tile_m": 0.27,
                             "lum_mean": 0.2, "design": "filete", "color": [0.41, 0.3, 0.26],
                             "accent": [0.76, 0.72, 0.69]}),
]

_SCRIPT = r'''
import json, math, sys
import bpy
import numpy as np

bpy.ops.wm.read_factory_settings(use_empty=True)
from mathutils import Vector
from mobilia.assets import support_surface
from mobilia.assets.procedural import build_asset, animate_rug_unroll

CASES = json.loads(sys.argv[1])
out = {"cases": {}, "unroll": {}, "errors": {}}

import tempfile
from PIL import Image
_tmp = tempfile.mkdtemp(prefix="mob_tex_")
_rng = np.random.default_rng(0)
_synth = {}
for _k, _v in (("diff", 120), ("nor", 128), ("rough", 200)):
    _a = np.clip(_v + _rng.normal(0, 12, (64, 64, 3)), 0, 255).astype(np.uint8)
    if _k == "nor":
        _a[..., 2] = 255
    _synth[_k] = f"{_tmp}/{_k}.png"
    Image.fromarray(_a).save(_synth[_k])
for _c in CASES:
    if _c[2].get("texture") == "__synthetic__":
        _c[2]["texture"] = _synth


def evaluated_points(root):
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for c in root.children:
        ev = c.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3, np.float64)
        me.vertices.foreach_get("co", co)
        mw = np.asarray(c.matrix_world)
        pts.append(co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
        ev.to_mesh_clear()
    return np.concatenate(pts)


for label, kind, params in CASES:
    for o in list(bpy.context.scene.objects):  # cena limpa: o ray casting só vê este asset
        o.hide_viewport = True
    root = build_asset(kind, params, label)
    bpy.context.view_layer.update()
    pts = evaluated_points(root)
    parts = []
    for c in root.children:
        me = c.data
        idx = np.empty(len(me.polygons), np.int64)
        me.polygons.foreach_get("material_index", idx)
        pco = np.empty(len(me.vertices) * 3, np.float64)
        me.vertices.foreach_get("co", pco)
        pco = pco.reshape(-1, 3) + np.asarray(c.location)
        parts.append(dict(
            name=c.name, type=c.type, lo=pco.min(0).tolist(), hi=pco.max(0).tolist(),
            pivot=list(c.location),
            part=c.get("mobilia_part"), order=c.get("assembly_order"), motion=c.get("assembly_motion"),
            n_slots=len(c.material_slots),
            slots_filled=all(s.material is not None for s in c.material_slots),
            max_mat_index=int(idx.max()) if len(idx) else -1,
            n_polys=len(me.polygons),
            rot=list(c.rotation_euler), scale=list(c.scale),
        ))
    mats = {s.material for c in root.children for s in c.material_slots if s.material is not None}
    images = sorted({n.image.name for m in mats for n in m.node_tree.nodes
                     if n.type == "TEX_IMAGE" and n.image is not None})
    canvas = json.loads(root["mobilia_canvas"]) if "mobilia_canvas" in root else None
    art_uv = None
    if canvas is not None:      # extensão (m) e faixa de UV das faces com a pintura (material 0)
        c0 = root.children[0]
        me = c0.data
        uvl = me.uv_layers[0].data
        P, U = [], []
        for poly in me.polygons:
            if poly.material_index != 0:
                continue
            for li in poly.loop_indices:
                P.append(tuple(me.vertices[me.loops[li].vertex_index].co))
                U.append(tuple(uvl[li].uv))
        P, U = np.array(P), np.array(U)
        art_uv = dict(ext=(P.max(0) - P.min(0)).tolist(), uv_lo=U.min(0).tolist(), uv_hi=U.max(0).tolist(),
                      y=float(P[:, 1].mean() + c0.location.y))
    emissive = []
    for m in mats:
        for n in m.node_tree.nodes:
            if n.type == "EMISSION":
                emissive.append(m.name)
            if n.type == "BSDF_PRINCIPLED":
                es = n.inputs["Emission Strength"]
                if es.is_linked or es.default_value > 0.0:
                    emissive.append(m.name)
    mattress_top = None
    for c in root.children:
        if c.get("mobilia_part") == "mattress":
            co = np.empty(len(c.data.vertices) * 3)
            c.data.vertices.foreach_get("co", co)
            mattress_top = float(co.reshape(-1, 3)[:, 2].max() + c.location.z)
    sup = support_surface(kind, root["mobilia_size"], json.loads(root["mobilia_params"]))
    hits = []
    if sup is not None:
        dg = bpy.context.evaluated_depsgraph_get()
        x0, y0, x1, y1 = sup["rects"][0]
        for fx in (0.2, 0.5, 0.8):
            for fy in (0.2, 0.5, 0.8):
                o = Vector((x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy, 5.0))
                ok, loc, *_ = bpy.context.scene.ray_cast(dg, o, Vector((0, 0, -1)))
                hits.append(float(loc.z) if ok else None)
    out["cases"][label] = dict(
        emissive=sorted(set(emissive)), mattress_top=mattress_top, images=images, canvas=canvas, art_uv=art_uv,
        support=sup, support_hits=hits,
        kind=kind, root_type=root.type, root_name=root.name, root_parent=root.parent is not None,
        root_loc=list(root.location), size=list(root["mobilia_size"]),
        fit=list(root["mobilia_fit_scale"]), lo=pts.min(0).tolist(), hi=pts.max(0).tolist(),
        nan=bool(~np.isfinite(pts).all()), parts=parts)

# --- desenrolar do tapete
for label, params, kw in (("x", {"pattern": "berber", "fringe": True}, {}),
                          ("y_rev", {"pattern": "solid_border"}, {"axis": "y", "reverse": True})):
    root = build_asset("rug", params, f"rug_unroll_{label}")
    f0, f1 = 5, 41
    animate_rug_unroll(root, f0, f1, **kw)
    sc = bpy.context.scene
    frames = {}
    for f in (1, f0, f0 + 1, 10, 15, 20, 23, 30, 35, 38, 40, f1, f1 + 10):
        sc.frame_set(f)
        pts = evaluated_points(root)
        frames[str(f)] = dict(lo=pts.min(0).tolist(), hi=pts.max(0).tolist(),
                              nan=bool(~np.isfinite(pts).all()))
    out["unroll"][label] = dict(f0=f0, f1=f1, thickness=float(root["mobilia_size"][2]),
                                size=list(root["mobilia_size"]), frames=frames,
                                n_keys=[len(c.data.shape_keys.key_blocks) for c in root.children])

# --- erros esperados
for label, call in (("kind", lambda: build_asset("sofa", {}, "x")),
                    ("size", lambda: build_asset("rug", {"size": [1, -1, 0.01]}, "y")),
                    ("pattern", lambda: build_asset("rug", {"pattern": "xadrez"}, "z"))):
    try:
        call()
        out["errors"][label] = None
    except ValueError as e:
        out["errors"][label] = str(e)

print("RESULT:" + json.dumps(out))
'''


@pytest.fixture(scope="module")
def report() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_SCRIPT), json.dumps(CASES)],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert proc.returncode == 0 and lines, f"subprocesso bpy falhou:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT:"):])


@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_root_empty_with_mesh_parts(report, label):
    c = report["cases"][label]
    assert c["root_type"] == "EMPTY"
    assert not c["root_parent"]
    assert c["root_loc"] == [0.0, 0.0, 0.0]
    assert len(c["parts"]) >= 1
    assert all(p["type"] == "MESH" and p["n_polys"] > 0 for p in c["parts"])


@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_bbox_matches_size(report, label):
    c = report["cases"][label]
    lo, hi, size = c["lo"], c["hi"], c["size"]
    for k in range(3):
        dim = hi[k] - lo[k]
        assert abs(dim - size[k]) <= 0.02 * size[k], f"eixo {k}: {dim:.4f} m != {size[k]:.4f} m"
    assert abs(lo[2]) <= 0.002, f"base fora do piso: min Z = {lo[2]:.4f}"
    for k in range(2):
        center = (lo[k] + hi[k]) / 2
        assert abs(center) <= max(0.002, 0.01 * size[k]), f"centro do eixo {k} deslocado: {center:.4f}"


@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_builder_proportions_need_little_correction(report, label):
    # o ajuste final de bounding box deve ser um retoque (<= 5%), não uma deformação
    fit = report["cases"][label]["fit"]
    assert all(0.95 <= f <= 1.05 for f in fit), fit


@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_parts_have_assembly_properties(report, label):
    parts = report["cases"][label]["parts"]
    names = [p["part"] for p in parts]
    assert len(set(names)) == len(names), f"nomes de peça repetidos: {names}"
    for p in parts:
        assert isinstance(p["part"], str) and p["part"], p
        assert isinstance(p["order"], int) and p["order"] >= 0, p
        assert p["motion"] in MOTIONS, p
        # peças só com translação (o pivô é a origem); rotação/escala ficam para a animação
        assert p["rot"] == [0.0, 0.0, 0.0] and p["scale"] == [1.0, 1.0, 1.0], p


@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_materials_assigned(report, label):
    for p in report["cases"][label]["parts"]:
        assert p["n_slots"] >= 1 and p["slots_filled"], p
        assert 0 <= p["max_mat_index"] < p["n_slots"], p


@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_no_nan_vertices(report, label):
    assert not report["cases"][label]["nan"]


def test_assembly_orders_start_at_zero_and_motions_fit_kind(report):
    for label in KINDS:
        parts = report["cases"][label]["parts"]
        assert min(p["order"] for p in parts) == 0
    assert {p["motion"] for p in report["cases"]["rug"]["parts"]} == {"unroll"}
    assert any(p["motion"] == "unfold" for p in report["cases"]["curtain"]["parts"])


@pytest.mark.parametrize("label", ["x", "y_rev"])
def test_rug_unroll_rolled_at_start(report, label):
    u = report["unroll"][label]
    T, size = u["thickness"], u["size"]
    ax = 0 if label == "x" else 1
    for f in ("1", str(u["f0"])):  # antes e no quadro inicial: rolo compacto
        fr = u["frames"][f]
        length = fr["hi"][ax] - fr["lo"][ax]
        assert length < 0.25 * size[ax], f"quadro {f}: extensão {length:.3f} m — não está enrolado"
        assert fr["hi"][2] > 6 * T, f"quadro {f}: altura {fr['hi'][2]:.3f} m — rolo baixo demais"
    # o rolo fica dentro da pegada final do tapete
    fr = u["frames"][str(u["f0"])]
    assert fr["lo"][ax] >= -size[ax] / 2 - 0.002 and fr["hi"][ax] <= size[ax] / 2 + 0.002


@pytest.mark.parametrize("label", ["x", "y_rev"])
def test_rug_unroll_flat_at_end(report, label):
    u = report["unroll"][label]
    T, size = u["thickness"], u["size"]
    for f in (str(u["f1"]), str(u["f1"] + 10)):
        fr = u["frames"][f]
        assert fr["hi"][2] <= T + 0.002, f"quadro {f}: max Z {fr['hi'][2]:.4f} > espessura"
        for k in range(2):
            assert abs((fr["hi"][k] - fr["lo"][k]) - size[k]) <= 0.02 * size[k]


@pytest.mark.parametrize("label", ["x", "y_rev"])
def test_rug_unroll_never_below_floor_and_progresses(report, label):
    u = report["unroll"][label]
    ax = 0 if label == "x" else 1
    frames = sorted(u["frames"].items(), key=lambda kv: int(kv[0]))
    lengths = []
    for f, fr in frames:
        assert not fr["nan"]
        assert fr["lo"][2] >= -0.0005, f"quadro {f}: atravessa o piso (min Z {fr['lo'][2]:.4f})"
        lengths.append(fr["hi"][ax] - fr["lo"][ax])
    # só avança (tolerância de poucos mm para o último "cacho" que se assenta) e nunca
    # passa da pegada final mais que 1 cm
    assert all(b >= a - 0.005 for a, b in zip(lengths, lengths[1:])), lengths
    assert max(lengths) <= u["size"][ax] + 0.01, lengths
    assert all(n >= 2 for n in u["n_keys"])


def test_invalid_inputs_raise_value_error(report):
    errs = report["errors"]
    assert errs["kind"] and "desconhecido" in errs["kind"]
    assert errs["size"] and "inválidas" in errs["size"]
    assert errs["pattern"] and "desconhecido" in errs["pattern"].lower()


@pytest.mark.parametrize("label", [c[0] for c in CASES])
def test_lamps_and_everything_never_emit_light(report, label):
    # a iluminação da foto não pode mudar: nenhum material emite (nem com light_on=True)
    assert report["cases"][label]["emissive"] == []


def test_layout_parameters_are_honored(report):
    c = report["cases"]
    assert abs(c["layout_bed_wood"]["mattress_top"] - 0.56) <= 0.015
    assert abs(c["layout_bed_laca"]["mattress_top"] - 0.56) <= 0.015
    parts = {p["part"] for p in c["layout_bed_wood"]["parts"]}
    assert {"legs", "frame", "headboard", "mattress", "duvet"} <= parts


@pytest.mark.parametrize("label", ["desk", "layout_desk", "bed", "layout_bed_wood", "bed_single"])
def test_support_surface_matches_geometry(report, label):
    c = report["cases"][label]
    sup, hits = c["support"], c["support_hits"]
    assert sup is not None and len(hits) == 9 and all(h is not None for h in hits), hits
    tol = 0.002 if c["kind"] == "desk" else 0.025   # edredom tem relevo macio
    for h in hits:
        assert abs(h - sup["z"]) <= tol, (h, sup["z"])


@pytest.mark.parametrize("label", ["book_stack", "book_stack_2", "layout_book_stack"])
def test_book_stack_footprint(report, label):
    """Pilha de livros deitados: 2–3 livros de ~30 x 23 x 3 cm, de baixo para cima, cada um
    inteiro dentro da pegada do catálogo (é ela que o layout encaixa no tampo) e apoiado no de baixo."""
    c = report["cases"][label]
    W, D, H = c["size"]
    books = sorted(c["parts"], key=lambda p: p["order"])
    n = 2 if label == "book_stack_2" else 3
    assert [p["part"] for p in books] == [f"book_{i}" for i in range(n)]
    assert [p["order"] for p in books] == list(range(n))
    assert all(p["motion"] == "drop" for p in books)
    for k, p in enumerate(books):
        lo, hi = np.asarray(p["lo"]), np.asarray(p["hi"])
        ext = hi - lo
        assert 0.2 <= max(ext[0], ext[1]) <= 0.36 and 0.15 <= min(ext[0], ext[1]) <= 0.28, ext
        assert 0.02 <= ext[2] <= 0.045, ext
        # pegada do catálogo (origem no centro da base)
        assert np.all(lo[:2] >= [-W / 2 - 0.002, -D / 2 - 0.002]) and np.all(hi[:2] <= [W / 2 + 0.002, D / 2 + 0.002])
        assert abs(p["pivot"][2] - lo[2]) <= 1e-4      # pivô na base (a queda termina no repouso)
        if k:
            below = books[k - 1]
            assert abs(lo[2] - below["hi"][2]) <= 0.0015, (lo[2], below["hi"][2])   # sem vão nem interpenetração
    assert abs(books[0]["lo"][2]) <= 0.002 and abs(books[-1]["hi"][2] - H) <= 0.002


# ----------------------------------------------------------------------------- PT9: quadros e tapete


@pytest.mark.parametrize("label", ["canvas_tela_paisagem", "canvas_moldura_retrato", "canvas_moldura_madeira"])
def test_canvas_art_sizes(report, label):
    """Quadro gerado: tamanho livre exato, pintura na proporção da área pintada (sem esticar),
    UV 0..1 cobrindo só essa área, frente em -Y e imagem própria (gerada) no material."""
    c = report["cases"][label]
    W, D, H = c["size"]
    cv, uv = c["canvas"], c["art_uv"]
    assert cv is not None and uv is not None
    aw, ah = cv["art_w"], cv["art_h"]
    px_w, px_h = cv["art_px"]
    assert abs(px_w / px_h - aw / ah) <= 1.0 / min(px_w, px_h) + 1e-6, (px_w, px_h, aw, ah)
    assert abs(uv["ext"][0] - aw) <= 1e-3 and abs(uv["ext"][2] - ah) <= 1e-3
    assert np.allclose(uv["uv_lo"], [0, 0], atol=1e-6) and np.allclose(uv["uv_hi"], [1, 1], atol=1e-6)
    assert uv["y"] < 0, "a pintura fica na frente (-Y), longe da parede"
    if cv["frame"] == "tela":
        assert aw >= W - 0.01 and ah >= H - 0.01, "tela: a pintura ocupa a frente toda"
    else:
        assert aw < W - 2 * cv["bar"] + 1e-6 and ah < H - 2 * cv["bar"] + 1e-6
        if label == "canvas_moldura_retrato":
            assert cv["mat"] > 0.03, "passe-partout presente"
    assert any(".arte" in n for n in c["images"]) and any(".relevo" in n for n in c["images"])
    assert len(c["parts"]) == 1 and c["parts"][0]["motion"] == "slide"


def test_art_generator_deterministic():
    from mobilia.assets.art import ART_STYLES, art_resolution, crop_to_aspect, generate_art

    w, h = art_resolution(0.45, 0.6)
    assert abs(w / h - 0.75) < 0.01 and max(w, h) <= 1536
    for style in ART_STYLES:
        a, ra = generate_art(w, h, ART_COLORS, "#ede9e1", style, seed=7)
        b, rb = generate_art(w, h, ART_COLORS, "#ede9e1", style, seed=7)
        c, _ = generate_art(w, h, ART_COLORS, "#ede9e1", style, seed=8)
        assert a.shape == (h, w, 3) and a.dtype == np.uint8 and ra.shape == (h, w)
        assert np.array_equal(a, b) and np.array_equal(ra, rb), "mesma semente -> mesmos bytes"
        assert not np.array_equal(a, c), "outra semente -> outra pintura"
        # pintura de verdade: fundo + tintas da paleta, nada de cartão liso
        assert a.reshape(-1, 3).std(0).mean() > 12
        acc = np.array([0xb5, 0x60, 0x3e], float)
        assert np.min(np.abs(a.reshape(-1, 3).astype(float) - acc).sum(1)) < 60
    with pytest.raises(ValueError):
        generate_art(w, h, ART_COLORS, "#ffffff", "xadrez", 0)
    img = np.zeros((300, 400, 3), np.uint8)
    assert crop_to_aspect(img, 1.0).shape[:2] == (300, 300)
    assert crop_to_aspect(img, 2.0).shape[:2] == (200, 400)


def test_rug_uses_cc0_texture(report):
    c = report["cases"]["rug_textured"]
    assert any(n.startswith("diff") for n in c["images"]) and any(n.startswith("nor") for n in c["images"])
    assert not any(p["part"] == "rug_fringe" for p in c["parts"]), "tapete texturizado sem franja procedural"
    # sem textura volta ao desenho procedural (sem imagens)
    assert report["cases"]["rug"]["images"] == []
