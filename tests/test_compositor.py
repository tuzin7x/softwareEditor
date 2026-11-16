"""Testes da composição: a foto original precisa sair bit a bit idêntica fora dos móveis/sombras."""
import numpy as np
import pytest

from mobilia.composite.compositor import Compositor, CompositeSettings


@pytest.fixture()
def plate():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    img[60:, :] = (180, 160, 140)  # "piso" liso
    return img


def _object_layer(h=120, w=160):
    rgba = np.zeros((h, w, 4), np.float32)
    rgba[50:80, 60:100, :3] = (0.2, 0.1, 0.05)
    rgba[50:80, 60:100, 3] = 1.0
    rgba[49, 60:100, 3] = 0.5                      # borda antisserrilhada
    rgba[49, 60:100, :3] = (0.1, 0.05, 0.025)
    shadow = np.ones((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot((yy - 82) / 6.0, (xx - 80) / 30.0)
    shadow[..., :] = (1.0 - 0.6 * np.exp(-d ** 2))[..., None]   # sombra suave sob o objeto
    return rgba, shadow


def test_empty_frame_is_exact_copy(plate):
    comp = Compositor(plate)
    out, st = comp.composite(1, None, None, None)
    assert np.array_equal(out, plate)
    assert st.changed_pixels == 0 and st.preserved_exact


def test_pixels_outside_influence_are_bit_exact(plate):
    comp = Compositor(plate)
    rgba, shadow = _object_layer()
    out, st = comp.composite(5, rgba, shadow, [0, 0, 160, 120])
    assert st.preserved_exact and st.max_diff_outside == 0
    # o objeto mudou os pixels dele
    assert not np.array_equal(out[60:75, 65:95], plate[60:75, 65:95])
    # longe do objeto (fora do raio de influência) tudo idêntico
    far = np.ones(plate.shape[:2], bool)
    r = int(CompositeSettings().influence_radius_frac * np.hypot(160, 120)) + 2
    far[max(0, 49 - r):80 + r, max(0, 60 - r):100 + r] = False
    assert np.array_equal(out[far], plate[far])


def test_shadow_darkens_floor_near_object(plate):
    comp = Compositor(plate, CompositeSettings(grain_strength=0.0))
    rgba, shadow = _object_layer()
    out, _ = comp.composite(5, rgba, shadow, [0, 0, 160, 120])
    below = (slice(81, 86), slice(70, 90))
    assert out[below].mean() < plate[below].mean() - 5


def test_outside_render_border_is_untouched(plate):
    comp = Compositor(plate)
    rgba, shadow = _object_layer()
    shadow[:, :40] = 0.0          # o Cycles grava 0 fora da região de render
    out, st = comp.composite(5, rgba, shadow, [40, 0, 160, 120])
    assert np.array_equal(out[:, :40], plate[:, :40])
    assert st.preserved_exact


def test_noise_below_eps_does_not_touch_pixels(plate):
    comp = Compositor(plate)
    rgba, _ = _object_layer()
    rng = np.random.default_rng(3)
    shadow = (1.0 + rng.normal(0, 0.001, size=(120, 160, 3))).astype(np.float32)
    out, st = comp.composite(5, rgba, shadow, [0, 0, 160, 120])
    obj = rgba[..., 3] > 0
    assert st.shadow_pixels == 0
    assert np.array_equal(out[~obj], plate[~obj])


def test_layer_accumulator_incremental_paste_is_seamless():
    from mobilia.composite.compositor import LayerAccumulator

    H, W = 60, 80
    acc = LayerAccumulator(H, W, feather_px=6)
    full = np.zeros((H, W, 4), np.float32)
    full[20:40, 10:30] = (0.2, 0.2, 0.2, 1.0)           # peça estática
    sh = np.ones((H, W, 3), np.float32)
    acc.update(full, sh, [0, 0, W, H], "full")
    assert np.allclose(acc.rgba, full)
    # quadro incremental: re-render só da região da peça móvel; fora dela o render
    # é zero (border do Cycles) e não pode apagar a peça estática
    inc = np.zeros_like(full)
    inc[30:50, 50:70] = (0.8, 0.1, 0.1, 1.0)
    acc.update(inc, sh, [40, 20, 80, 60], "incremental")
    assert np.allclose(acc.rgba[20:40, 10:30], full[20:40, 10:30])       # estática preservada
    assert np.allclose(acc.rgba[35:45, 60:70], inc[35:45, 60:70])         # interior da região: novo render
    # na borda interna da região a mistura é gradual (sem degrau)
    col = acc.rgba[40, 40:47, 3]
    assert col[0] < col[-1] or np.allclose(col, 0)
    # um quadro "full" substitui tudo
    acc.update(full, sh, [0, 0, W, H], "full")
    assert np.allclose(acc.rgba, full)


def test_unrendered_border_rows_are_neutral_not_black():
    """Linhas fora do border do Cycles chegam zeradas; não podem virar sombra preta."""
    from mobilia.composite.compositor import Compositor, mask_unrendered

    H, W = 60, 80
    photo = np.full((H, W, 3), 180, np.uint8)
    rgba = np.zeros((H, W, 4), np.float32)
    shadow = np.zeros((H, W, 3), np.float32)
    shadow[20:] = 1.0                       # região renderizada: sem sombra
    rgba[22:40, 30:50] = [0.2, 0.2, 0.2, 1.0]  # um móvel colado à borda do render
    shadow2 = mask_unrendered(rgba, shadow)
    assert (shadow2[:20] == 1.0).all() and (shadow[:20] == 0).all()   # não altera a entrada
    out, st = Compositor(photo).composite(1, rgba, shadow2, None)
    assert st.preserved_exact
    assert (out[:20] == photo[:20]).all()   # nada escurece acima da região renderizada
    # controle: sem a máscara, a mesma entrada escurece a foto (era o defeito da linha na parede)
    out_bad, _ = Compositor(photo).composite(1, rgba, shadow, None)
    assert (out_bad[:20] != photo[:20]).any()


def test_accumulator_shadow_fades_at_internal_border_edge():
    from mobilia.composite.compositor import LayerAccumulator

    H, W = 100, 100
    acc = LayerAccumulator(H, W, feather_px=4, shadow_feather_px=20)
    rgba = np.zeros((H, W, 4), np.float32)
    shadow = np.full((H, W, 3), 0.8, np.float32)
    acc.update(rgba, shadow, [0, 40, W, H], "full")
    col = acc.shadow[:, 50, 0]
    assert (col[:40] == 1.0).all()                      # fora do border: neutro
    assert abs(col[40] - 1.0) < 0.01                    # sem degrau na borda interna
    assert np.all(np.diff(col[40:61]) <= 1e-6)          # rampa monótona
    assert abs(col[70] - 0.8) < 1e-6 and abs(col[99] - 0.8) < 1e-6   # borda da imagem: sem rampa


# ---------------------------------------------------------------- PT4: EXR recortado no border
_CROP_SCRIPT = r'''
import json, sys
from pathlib import Path
import bpy
from mobilia.render import blender_worker as bw
from mobilia.render import border as bd

out = Path(sys.argv[1])
W, H = 97, 61                                      # ímpar: o border nunca cai em pixel inteiro
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 4
sc.cycles.use_denoising = True                   # o denoiser também vê só o recorte
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.render.film_transparent = True
im = sc.render.image_settings
im.media_type = "MULTI_LAYER_IMAGE"; im.color_depth = "16"; im.exr_codec = bd.EXR_CODEC
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.location = (0, -4, 2); cam.rotation_euler = (1.1, 0, 0)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
bpy.ops.mesh.primitive_plane_add(size=6); bpy.context.object.is_shadow_catcher = True
sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN")); sc.collection.objects.link(sun)
sc.view_layers[0].cycles.use_pass_shadow_catcher = True
po = {"node": bw.setup_pass_output(sc, out)}
res = []
for k, b in enumerate([(0.1337, 0.2222, 0.7771, 0.8613), (0.0, 0.31, 0.5049, 1.0), (0.41, 0.0, 1.0, 0.577)]):
    px, bb = bd.pixel_border(b, W, H)
    sc.render.use_border = True
    sc.render.border_min_x, sc.render.border_min_y, sc.render.border_max_x, sc.render.border_max_y = bb
    for crop in (False, True):
        sc.render.use_crop_to_border = crop
        bw.render_frame(sc, out / f"f{k}_{int(crop)}.exr", po)
    res.append(px)
print("RESULT:" + json.dumps({"node": po["node"] is not None, "borders": res}))
'''


@pytest.mark.skipif(__import__("importlib").util.find_spec("bpy") is None, reason="bpy não está instalado")
def test_cropped_exr_equals_full_frame(tmp_path):
    """Recorte do Cycles colado no quadro inteiro == EXR de quadro inteiro, byte a byte, e a
    composição sai idêntica (inclusive o retângulo exato renderizado: sem linha escura)."""
    import json
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    from mobilia.composite.compositor import load_render_passes, read_render_passes

    repo = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_CROP_SCRIPT), str(tmp_path)],
                          cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    res = json.loads(lines[-1][len("RESULT:"):])
    assert res["node"] is True
    H, W = 61, 97
    rng = np.random.default_rng(3)
    photo = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    for k, bpx in enumerate(res["borders"]):
        x0, y0, x1, y1 = bpx
        full_raw, full_sh_raw = read_render_passes(tmp_path / f"f{k}_0.exr")
        crop_raw, _ = read_render_passes(tmp_path / f"f{k}_1.exr")
        # o recorte tem exatamente o tamanho do border anotado
        assert crop_raw.shape[:2] == (y1 - y0, x1 - x0), (crop_raw.shape, bpx)
        # e o quadro inteiro tem pixels renderizados exatamente dentro do border anotado
        rendered = ~((full_sh_raw == 0).all(-1) & (full_raw == 0).all(-1))
        ys, xs = np.nonzero(rendered)
        assert [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] == bpx
        full = load_render_passes(tmp_path / f"f{k}_0.exr", bpx, (H, W))
        crop = load_render_passes(tmp_path / f"f{k}_1.exr", bpx, (H, W))
        assert np.array_equal(full[0], crop[0]) and np.array_equal(full[1], crop[1])
        assert full[0][..., 3].max() > 0.5            # o cubo aparece
        comp = Compositor(photo)
        a, sa = comp.composite(1, full[0], full[1], bpx)
        b, sb = comp.composite(1, crop[0], crop[1], bpx)
        assert np.array_equal(a, b) and sa.preserved_exact and sb.preserved_exact


def test_cropped_paste_matches_full_frame_synthetic(tmp_path):
    """Sem Blender: um EXR recortado (File Output, RLE) colado no border dá os mesmos passes
    que o EXR de quadro inteiro com zeros fora do border."""
    import OpenEXR

    from mobilia.composite.compositor import load_render_passes

    H, W = 50, 70
    bpx = [13, 7, 52, 41]
    x0, y0, x1, y1 = bpx
    rng = np.random.default_rng(5)
    rgba = np.zeros((H, W, 4), np.float32)
    sh = np.zeros((H, W, 3), np.float32)
    rgba[y0:y1, x0:x1] = rng.random((y1 - y0, x1 - x0, 4)).astype(np.float16)
    rgba[y0:y0 + 3, x0:x1] = 0.0                      # cantos sem móvel dentro do recorte
    sh[y0:y1, x0:x1] = (0.5 + rng.random((y1 - y0, x1 - x0, 3))).astype(np.float16)
    sh[y0 + 1, x0:x0 + 5] = 0.0                       # pixel "não renderizado" dentro do recorte

    def write(path, a, s):
        ch = {f"Combined.{c}": np.ascontiguousarray(a[..., i]).astype(np.float16) for i, c in enumerate("RGBA")}
        ch.update({f"Shadow Catcher.{c}": np.ascontiguousarray(s[..., i]).astype(np.float16)
                   for i, c in enumerate("RGB")})
        with OpenEXR.File({"compression": OpenEXR.RLE_COMPRESSION, "type": OpenEXR.scanlineimage}, ch) as f:
            f.write(str(path))

    write(tmp_path / "full.exr", rgba, sh)
    write(tmp_path / "crop.exr", rgba[y0:y1, x0:x1], sh[y0:y1, x0:x1])
    fa, fs = load_render_passes(tmp_path / "full.exr", bpx, (H, W))
    ca, cs = load_render_passes(tmp_path / "crop.exr", bpx, (H, W))
    assert ca.shape == (H, W, 4) and cs.shape == (H, W, 3)
    assert np.array_equal(fa, ca) and np.array_equal(fs, cs)
    assert (cs[:y0] == 1.0).all() and (ca[:y0] == 0.0).all()
    # tamanho que não bate com o border: erro claro, nunca um palpite de posição
    with pytest.raises(ValueError, match="não bate"):
        load_render_passes(tmp_path / "crop.exr", [x0, y0, x1 + 1, y1], (H, W))


def test_border_rounding_no_dark_line():
    """Regressão da "linha escura" de 1 px: o border entregue ao Blender vira exatamente o
    retângulo anotado (truncamento em float32 do Cycles), em qualquer tamanho e posição; e o
    recorte colado nunca escurece a foto fora dele."""
    from mobilia.composite.compositor import paste_crop
    from mobilia.render.border import blender_disprect, pixel_border

    rng = np.random.default_rng(11)
    sizes = [(97, 61), (640, 480), (1920, 1440), (2560, 1920), (3840, 2880), (1000, 750)]
    for W, H in sizes:
        for _ in range(400):
            a, c = np.sort(rng.random(2))
            d, e = np.sort(rng.random(2))
            b = (float(a), float(d), float(c), float(e))
            pb = pixel_border(b, W, H)
            if pb is None:
                continue
            px, bb = pb
            assert all(0.0 <= v <= 1.0 for v in bb)
            assert blender_disprect(bb, W, H) == px, (W, H, b)
            # o retângulo cobre (para fora) o border pedido: nada do móvel fica de fora
            assert px[0] <= b[0] * W + 1e-5 and px[2] >= b[2] * W - 1e-5
            assert px[1] <= (1 - b[3]) * H + 1e-5 and px[3] >= (1 - b[1]) * H - 1e-5
        # casos de borda: tela inteira e pixels exatamente inteiros
        assert pixel_border((0.0, 0.0, 1.0, 1.0), W, H)[0] == [0, 0, W, H]
        assert blender_disprect(pixel_border((0.0, 0.0, 1.0, 1.0), W, H)[1], W, H) == [0, 0, W, H]
        px, bb = pixel_border((10 / W, 20 / H, 30 / W, 40 / H), W, H)
        assert px == [10, H - 40, 30, H - 20] and blender_disprect(bb, W, H) == px
    assert pixel_border((0.5, 0.5, 0.5, 0.9), 100, 100) is None       # retângulo vazio

    # composição de um recorte colado: fora do border, os bytes da foto
    H, W = 60, 80
    photo = np.full((H, W, 3), 180, np.uint8)
    bpx = [20, 15, 61, 47]
    x0, y0, x1, y1 = bpx
    rgba_c = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
    rgba_c[5:20, 10:30] = [0.2, 0.2, 0.2, 1.0]
    sh_c = np.ones((y1 - y0, x1 - x0, 3), np.float32)
    rgba, sh = paste_crop(rgba_c, sh_c, bpx, (H, W))
    out, st = Compositor(photo).composite(1, rgba, sh, bpx)
    assert st.preserved_exact
    outside = np.ones((H, W), bool)
    outside[y0:y1, x0:x1] = False
    assert (out[outside] == photo[outside]).all()
    assert (out[y0 - 1] == photo[y0 - 1]).all() and (out[:, x1] == photo[:, x1]).all()
