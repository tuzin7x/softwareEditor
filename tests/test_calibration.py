"""PT8: luz e cor do CG iguais às da foto.

- ajuste das luzes (``render/calibration.py``): pixels estourados como limite inferior, sol
  ajustado à parte nas manchas, limite do ambiente pela claridade da sala, cor por canal;
- geometria do sol: telhado só de sombra e aberturas de sol das janelas tiradas das manchas;
- sala-proxy com "Is Camera Ray" (difuso para a câmera, emissão da foto para os outros raios);
- teste lento: calibração real na sala de teste.
"""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from mobilia.render import calibration as cal

FX = Path(__file__).parent / "fixtures"
HAS_BPY = importlib.util.find_spec("bpy") is not None


# ------------------------------------------------------------------------------- ajuste das luzes
def test_nnls_censored_respects_lower_bounds():
    """Linhas de limite inferior só empurram a solução para cima quando violadas."""
    A = np.array([[1.0], [1.0], [1.0]])
    b = np.array([1.0, 1.1, 0.9])
    assert cal.nnls_censored(A, b)[0] == pytest.approx(1.0)
    # pixels estourados em 1,5: a predição tem de chegar perto deles
    x = cal.nnls_censored(A, b, np.ones((3, 1)), np.full(3, 1.5))[0]
    assert x > 1.2
    # limites já satisfeitos não mudam nada
    assert cal.nnls_censored(A, b, np.ones((2, 1)), np.full(2, 0.5))[0] == pytest.approx(1.0)
    # sem linhas comuns (só limites): o menor x que satisfaz todos; e sempre x ≥ 0
    assert cal.nnls_censored(np.zeros((0, 1)), np.zeros(0), np.ones((2, 1)), np.array([0.8, 1.0]))[0] \
        == pytest.approx(1.0)
    assert cal.nnls_censored(A, -b)[0] == 0.0


def _synthetic_room(h=60, w=80, k=(1.5, 0.8, 2.0), cast=(1.08, 1.0, 0.9), clip=0.93):
    """Renders cinza de cada fonte e a "foto" = emissão + Σ kᵢ·Rᵢ (com um tom de cor nas luzes),
    estourada em ``clip``; a mancha de sol é um retângulo com gradiente."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = lambda a: np.repeat(a[..., None], 3, -1).astype(np.float32)  # noqa: E731
    E = g(0.22 + 0.06 * xx / w)
    R = {"windows": g(0.05 + 0.15 * np.exp(-xx / 30.0)), "ambient": g(np.full((h, w), 0.1)),
         "sun": g(np.full((h, w), 1e-3))}
    patch = np.zeros((h, w), bool)
    patch[30:42, 20:44] = True
    R["sun"][patch] = g(0.1 + 0.2 * (xx - 20) / 24)[patch]
    c = np.asarray(cast, np.float32)
    light = sum(kk * R[n] for kk, n in zip(k, ("windows", "ambient", "sun")))
    photo = E + light * c
    lum = photo @ cal.LUM
    sat = lum >= clip
    photo[sat] *= (clip / lum[sat])[:, None]
    normal = ~sat
    normal[:3] = False                        # uma faixa fora da máscara
    return R, E, photo, normal, sat, patch


def test_fit_lights_sun_on_patch_and_light_color():
    R, E, photo, normal, sat, patch = _synthetic_room()
    assert sat.any() and (patch & ~sat).any()          # parte da mancha estourada, parte não
    fit = cal.fit_lights(R, photo, normal, sat, emit=E, sun_patch=patch, ambient_max=0.7)
    c_lum = float(np.asarray((1.08, 1.0, 0.9)) @ cal.LUM)
    assert fit["window_power_per_m2"] == pytest.approx(1.5 * c_lum, rel=0.03)
    assert fit["ambient_strength"] == pytest.approx(0.8 * c_lum, rel=0.03)
    # o sol sai do excesso da mancha sobre a vizinhança, só nas manchas
    assert fit["sun_strength"] == pytest.approx(2.0 * c_lum, rel=0.1)
    assert fit["sol"]["ajustado_a_parte"] and fit["sol"]["pixels_limite_inferior"] > 0
    # os pixels estourados ficam acima do valor recortado na predição
    assert fit["limite_inferior_violado"] < 0.05
    # cor global por canal aplicada (normalizada pela luminância, limitada)
    g = np.asarray(fit["color_ratio"])
    assert g == pytest.approx(np.asarray((1.08, 1.0, 0.9)) / c_lum, abs=0.02)
    assert float(g @ cal.LUM) == pytest.approx(1.0, abs=0.02)
    pred = fit["pred"]
    assert pred.shape == photo.shape and fit["rel_rms"] < 0.02 and fit["emissao_fracao"] > 0.3


def test_sun_fit_robust_when_patch_is_clipped():
    """Mancha toda estourada: o sol tem de estourar ao menos metade do miolo da mancha (mediana
    da força exigida), não o pixel mais fraco da borda; se a vizinhança também estoura, a foto não
    mede o sol e o limite some. Pixels estourados nunca pesam mais que os normais no ajuste."""
    R, E, photo, normal, sat, patch = _synthetic_room(clip=0.70)
    assert not (patch & normal).any()                     # nenhum pixel normal na mancha
    R["sun"][patch.nonzero()[0][0], patch.nonzero()[1][0]] = 0.002   # um pixel quase sem sol no render
    fit = cal.fit_lights(R, photo, normal, sat, emit=E, sun_patch=patch, ambient_max=0.7)
    s = fit["sun_strength"]
    sol = fit["sol"]
    assert sol["forca_pixels_normais"] == 0.0 and s == pytest.approx(sol["forca_minima_estourados"], abs=1e-4)
    assert 0.5 < s < 2.0                                  # limite inferior plausível (a verdade é 2)
    # tudo estourado em volta da mancha também: nenhuma informação sobre o sol
    R2, E2, photo2, normal2, sat2, patch2 = _synthetic_room(clip=0.40)
    ring = np.zeros_like(patch2)
    ring[24:48, 14:50] = True
    sat2 |= ring
    normal2 &= ~ring
    photo2[ring] = np.minimum(photo2[ring], 0.40 / 1.0)
    fit2 = cal.fit_lights(R2, photo2, normal2, sat2, emit=E2, sun_patch=patch2, ambient_max=0.7)
    assert fit2["sun_strength"] < 0.3


def test_fit_lights_without_photo_patch_and_without_sun():
    R, E, photo, normal, sat, patch = _synthetic_room()
    # sem a máscara da foto: o sol é ajustado nos pixels que o sol do render ilumina
    fit = cal.fit_lights(R, photo, normal, sat, emit=E, sun_patch=None, ambient_max=0.7)
    assert fit["sun_strength"] > 1.5 and fit["sol"]["mascara_foto"] is False
    # sem render de sol: força 0, nada quebra
    R2 = {k: v for k, v in R.items() if k != "sun"}
    fit2 = cal.fit_lights(R2, photo, normal, sat, emit=E)
    assert fit2["sun_strength"] == 0.0 and not fit2["sol"]["ajustado_a_parte"]


def test_ambient_share_cap_is_adaptive():
    assert cal.ambient_share_limit(np.full(10, 0.05)) == pytest.approx(0.5)
    assert cal.ambient_share_limit(np.full(10, 0.8)) == pytest.approx(0.7)
    mid = cal.ambient_share_limit(np.full(10, 0.325))
    assert 0.5 < mid < 0.7
    # sala em que só o ambiente explicaria a luz: fica no limite e as janelas completam
    R, E, photo, normal, sat, _ = _synthetic_room(k=(0.1, 9.0, 0.0), cast=(1, 1, 1), clip=10.0)
    for lim in (0.5, 0.7):
        fit = cal.fit_lights({k: R[k] for k in ("windows", "ambient")}, photo, normal, None, emit=E, ambient_max=lim)
        assert fit["ambiente_limitado"]
        share = fit["ambient_strength"] * float(R["ambient"][normal][:, 0].mean()) / float(photo[normal].mean())
        assert share == pytest.approx(lim, rel=0.02)


def test_mask_png_levels():
    m8 = np.array([[0, 128, 255], [64, 191, 192]], np.uint8)
    normal, lower = cal.masks_from_png(m8)
    assert normal.tolist() == [[False, False, True], [False, False, True]]
    assert lower.tolist() == [[False, True, False], [True, True, False]]


# ------------------------------------------------------------------------------- geometria do sol
@pytest.fixture(scope="module")
def stafford(tmp_path_factory):
    """Cena da sala de teste (sem móveis) com o sol detectado na foto e as máscaras do PT8."""
    from PIL import Image

    from mobilia.analysis.room import RoomModel
    from mobilia.analysis.segmentation import SegmentationResult, labels_to_groups
    from mobilia.analysis.sunlight import detect_sun
    from mobilia.layout.model import Layout
    from mobilia.photo import load_photo
    from mobilia.render import scene_spec

    tmp = tmp_path_factory.mktemp("stafford")
    room = RoomModel.from_json((FX / "rooms/stafford/room.json").read_text(encoding="utf-8"))
    seg = SegmentationResult.load(FX / "rooms/stafford/seg.npz")
    seg.groups = labels_to_groups(seg.labels, seg.id2label)
    photo = load_photo(FX / "photos/sala_vazia_stafford.jpg").rgb8
    diag: dict = {}
    sun = detect_sun(room, seg, photo, diag=diag)
    assert sun is not None and diag["patch_mask"].any()
    W, H = 320, 240
    plate = np.asarray(Image.fromarray(photo).resize((W, H), Image.LANCZOS))
    Image.fromarray(plate).save(tmp / "plate.png")
    rs = lambda a: np.asarray(Image.fromarray(a).resize((W, H), Image.NEAREST))  # noqa: E731
    segp = SegmentationResult(rs(seg.labels), rs(seg.groups), seg.id2label, seg.model_id)
    spec = scene_spec.build_scene_spec(room, Layout("sala", "moderno", {}, []), plate, tmp / "plate.png", segp,
                                       tmp, tmp, samples=8, device="CPU", threads=4, sun=sun.to_dict(),
                                       sun_mask=diag["patch_mask"])
    return spec, tmp


def test_scene_spec_writes_three_level_mask_and_sun_patches(stafford):
    from PIL import Image

    spec, tmp = stafford
    lt = spec["lighting"]
    m = np.asarray(Image.open(lt["calib_mask_path"]))
    assert set(np.unique(m)) == {0, 128, 255}           # a foto tem pixels estourados nas paredes/teto
    assert (m == 255).mean() > 0.4 and (m == 128).any()
    sm = np.asarray(Image.open(lt["sun"]["mask_path"]))
    assert sm.shape == (spec["height"], spec["width"]) and (sm > 127).any()
    assert lt["photo_emission"] is True


def test_roof_covers_room_but_not_beyond_walls(stafford):
    from shapely.geometry import Point, Polygon

    spec, _ = stafford
    rings = cal.roof_polygons(spec["room"])
    roof = [Polygon(r) for r in rings]
    inside = lambda p: any(r.covers(Point(p)) for r in roof)  # noqa: E731
    cam = np.asarray(spec["cam_to_room"])[:2, 3]
    assert inside(cam)                                     # o lado da câmera (não fotografado) tem telhado
    for w in spec["room"]["walls"]:
        p0, p1, n = (np.asarray(w[k], float)[:2] for k in ("p0", "p1", "normal"))
        mid = 0.5 * (p0 + p1)
        assert inside(mid + 0.3 * n), w["id"]               # dentro da sala, diante de cada parede
        assert not inside(mid - 0.05 * n), w["id"]          # nunca atrás de uma parede fotografada


def test_sun_apertures_from_photo_patches(stafford):
    spec, _ = stafford
    from PIL import Image

    patch = np.asarray(Image.open(spec["lighting"]["sun"]["mask_path"])) > 127
    aps = cal.sun_apertures(spec, patch)
    assert aps and {a["estado"] for a in aps} <= {"parcial", "fechada", "sem_evidencia", "sem_sol"}
    lit = [a for a in aps if a["estado"] == "parcial"]
    assert lit, aps                                         # as manchas vêm de alguma janela
    for a in lit:
        s0, s1, z0, z1 = a["abertura"]
        a0, a1, b0, b1 = a["sol"]
        assert s0 <= a0 < a1 <= s1 and z0 <= b0 < b1 <= z1
        rects = cal.blocker_rects(a)
        inside = lambda r, s_, z: r[0] <= s_ <= r[1] and r[2] <= z <= r[3]  # noqa: E731
        # nenhuma persiana invade a abertura de sol...
        gs, gz = np.meshgrid(np.linspace(a0 + 1e-4, a1 - 1e-4, 25), np.linspace(b0 + 1e-4, b1 - 1e-4, 25))
        assert not any(inside(r, x, y) for r in rects for x, y in zip(gs.ravel(), gz.ravel()))
        # ... e o resto da janela, com as emendas, fica todo fechado
        gs, gz = np.meshgrid(np.linspace(s0, s1, 81), np.linspace(z0, z1, 81))
        for x, y in zip(gs.ravel(), gz.ravel()):
            if not (a0 <= x <= a1 and b0 <= y <= b1):
                assert any(inside(r, x, y) for r in rects), (x, y)
        # nas bordas da abertura da parede que a abertura de sol não toca, a persiana passa para trás
        # da parede (sem "lâmina" de sol rasante entre as duas)
        lo_s, hi_s = min(r[0] for r in rects), max(r[1] for r in rects)
        lo_z, hi_z = min(r[2] for r in rects), max(r[3] for r in rects)
        assert lo_z <= z0 - cal.BLOCKER_OUT + 1e-9 and hi_z >= z1 + cal.BLOCKER_OUT - 1e-9
        assert lo_s <= s0 - cal.BLOCKER_OUT + 1e-9 and hi_s >= s1 + cal.BLOCKER_OUT - 1e-9
    for a in aps:
        if a["estado"] == "fechada":
            s0, s1, z0, z1 = a["abertura"]
            O = cal.BLOCKER_OUT
            assert cal.blocker_rects(a) == [[s0 - O, s1 + O, z0 - O, z1 + O]]
        if a["estado"] in ("sem_evidencia", "sem_sol"):
            assert cal.blocker_rects(a) == []


def test_sun_aperture_recovers_known_rect(stafford):
    """Mancha sintética = projeção de uma abertura conhecida de uma janela no piso, vista pela
    câmera: a abertura estimada volta dentro da margem."""
    spec, _ = stafford
    W, H = spec["width"], spec["height"]
    d = np.asarray(spec["lighting"]["sun"]["direction"], float)
    room = spec["room"]
    wins = [i for i, o in enumerate(room["openings"]) if o["kind"] == "window"]
    best = None
    for i in wins:                     # a janela cuja projeção cai mais dentro da foto
        o = room["openings"][i]
        P0, dw, n = cal.window_frame(room, o)
        if d @ n > -0.05:
            continue
        rect = [o["s0"] + 0.3 * (o["s1"] - o["s0"]), o["s1"] - 0.1 * (o["s1"] - o["s0"]),
                o["z0"] + 0.05, o["z0"] + 0.45]
        gs, gz = np.meshgrid(np.linspace(rect[0], rect[1], 80), np.linspace(rect[2], rect[3], 80))
        Pw = P0 + gs.reshape(-1, 1) * dw + gz.reshape(-1, 1) * np.array([0, 0, 1.0])
        Q = Pw - d * (Pw[:, 2] / d[2])[:, None]
        uv, zc = cal.project_points(spec, Q)
        ok = (zc > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if best is None or ok.sum() > best[0]:
            best = (int(ok.sum()), i, rect, uv[ok])
    assert best is not None and best[0] > 500
    _, i, rect, uv = best
    patch = np.zeros((H, W), bool)
    patch[uv[:, 1].astype(int), uv[:, 0].astype(int)] = True
    ap = next(a for a in cal.sun_apertures(spec, patch) if a["janela"] == i)
    assert ap["estado"] == "parcial"
    assert np.allclose(ap["sol"], rect, atol=0.12), (ap["sol"], rect)


# ------------------------------------------------------------------------------- Blender
_PROXY_SCRIPT = r'''
import json, sys
import bpy
import numpy as np
from mobilia.render import blender_worker as bw

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 32
sc.cycles.use_denoising = False
sc.render.resolution_x, sc.render.resolution_y = 40, 30
sc.render.film_transparent = True
img = bpy.data.images.new("foto", 4, 4)
img.pixels.foreach_set([0.9, 0.5, 0.2, 1.0] * 16)
mat = bw.proxy_material("proxy_floor", img, 1.0, [0.5, 0.5, 0.5])
nt = mat.node_tree
res = {"emission_sampling": mat.cycles.emission_sampling}
mix = next(n for n in nt.nodes if n.type == "MIX_SHADER")
res["fac"] = mix.inputs["Fac"].links[0].from_socket.name
res["outro_raio"] = mix.inputs[1].links[0].from_node.type
res["raio_camera"] = mix.inputs[2].links[0].from_node.type
# o difuso (raio da câmera) não depende da foto: a cadeia de cor vem do nó RGB da cor média
def upstream(node, seen=None):
    seen = seen or set()
    for i in node.inputs:
        for l in i.links:
            if l.from_node.name not in seen:
                seen.add(l.from_node.name)
                upstream(l.from_node, seen)
    return seen
diff = mix.inputs[2].links[0].from_node
res["difuso_le_foto"] = any(nt.nodes[n].type == "TEX_IMAGE" for n in upstream(diff))
# cena: piso-proxy + cubo branco; nenhuma luz. Câmera de cima.
floor = bw._mesh_object("piso", [(-4, -4, 0), (4, -4, 0), (4, 4, 0), (-4, 4, 0)], [(0, 1, 2, 3)], mat)
cube_v = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (0.0, 1.0)]
cube_f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
white = bpy.data.materials.new("branco"); white.use_nodes = True
cube = bw._mesh_object("cubo", cube_v, cube_f, white)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.location = (0, -5, 4); cam.rotation_euler = (0.95, 0, 0)
world = bpy.data.worlds.new("W"); sc.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
def render(path):
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    c = bw._combined(bw.read_exr(path))
    return c
im = sc.render.image_settings
im.media_type = "MULTI_LAYER_IMAGE"; im.color_depth = "32"
out = sys.argv[1]
a = render(out + "/com.exr")
bw.set_photo_emission(0.0)
b = render(out + "/sem.exr")
# pixels do cubo: o centro da imagem; do piso: o canto inferior
cub = a[12:18, 17:23, :3].reshape(-1, 3).mean(0)
res["cubo_com_emissao"] = cub.tolist()
res["cubo_sem_emissao"] = b[12:18, 17:23, :3].reshape(-1, 3).mean(0).tolist()
res["piso_camera"] = a[26:30, 0:6, :3].reshape(-1, 3).mean(0).tolist()
print("RESULT:" + json.dumps(res))
'''


@pytest.mark.skipif(not HAS_BPY, reason="bpy não está instalado")
def test_proxy_emits_photo_only_for_non_camera_rays(tmp_path: Path):
    env = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_PROXY_SCRIPT), str(tmp_path)],
                          capture_output=True, text=True, env=env, timeout=300)
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert line, proc.stdout[-2000:] + proc.stderr[-2000:]
    r = json.loads(line[-1][7:])
    assert r["emission_sampling"] == "NONE"                 # fora da árvore de luzes
    assert r["fac"] == "Is Camera Ray" and r["outro_raio"] == "EMISSION" and r["raio_camera"] == "BSDF_DIFFUSE"
    assert r["difuso_le_foto"] is False                     # albedo uniforme para a câmera
    cub, dark, floor = (np.asarray(r[k]) for k in ("cubo_com_emissao", "cubo_sem_emissao", "piso_camera"))
    # o móvel recebe a luz da foto (com a cor dela) e nada quando a emissão é desligada
    assert cub.max() > 0.05 and cub[0] > cub[1] > cub[2]
    assert dark.max() < 1e-3
    # a câmera vê o piso difuso (sem luz nenhuma na cena), nunca a própria emissão
    assert floor.max() < 0.25 * cub.max()


@pytest.mark.slow
@pytest.mark.skipif(not HAS_BPY, reason="bpy não está instalado")
def test_room_calibration_matches_photo(stafford):
    """Calibração real (Blender na CPU) da sala de teste vazia: com a emissão da foto, o sol
    ajustado nas manchas e o mapa de ganho, o erro da luz na escala da sala fica ≤ 0,13."""
    spec, tmp = stafford
    env = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-m", "mobilia.render.blender_worker", str(tmp / "scene.json")],
                          capture_output=True, text=True, env=env, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-3000:]
    meta = json.loads((tmp / "render" / "frames.json").read_text(encoding="utf-8"))
    c = meta["calibration"]
    assert c["emissao_foto"] and 0.2 < c["emissao_fracao"] < 0.9
    assert c["rel_rms_ganho"] <= 0.13 and c["rel_rms"] <= 0.25
    assert c["sun_strength"] > 0 and c["sol"]["ajustado_a_parte"] and c["sol"]["aberturas"]
    adj = c["ajuste_cg"]
    assert adj["ganho"]["modo"] == "mapa" and 0.85 <= adj["ganho"]["min"] <= adj["ganho"]["max"] <= 1.2
    assert all(0.85 <= v <= 1.15 for v in c["color_ratio"])
    assert (tmp / "render" / "calib_pred.npy").exists()
