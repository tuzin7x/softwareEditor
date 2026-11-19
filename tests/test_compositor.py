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


# ---------------------------------------------------------------- PT5: composição recortada
def _scene(H, W, kind, seed=0):
    """Camadas sintéticas de quadro inteiro (RGBA + sombra) com móveis em lugares difíceis."""
    rng = np.random.default_rng(seed)
    rgba = np.zeros((H, W, 4), np.float32)
    # sombra larga e texturizada (sol/luz rebatida) por toda a imagem: o desvio passa do eps até
    # o raio de influência, onde a borda do recorte faria diferença se a margem fosse curta
    shadow = (0.9 + 0.06 * rng.random((H, W, 3))).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    boxes = {"meio": [(110, 150, 70, 120)], "canto": [(0, 40, 0, 50)],
             "dois": [(60, 120, 20, 60), (150, 200, 250, 300)], "ponto": [(120, 121, 160, 161)],
             "borda_dir": [(90, 160, 290, 320)]}[kind]
    for y0, y1, x0, x1 in boxes:
        rgba[y0:y1, x0:x1, :3] = rng.random((y1 - y0, x1 - x0, 3)).astype(np.float32) * 0.5
        rgba[y0:y1, x0:x1, 3] = 1.0
        if y0 > 0:
            rgba[y0 - 1, x0:x1, 3] = 0.4                       # antisserrilhado
            rgba[y0 - 1, x0:x1, :3] = 0.1
            rgba[max(0, y0 - 2), x0:x1, 3] = 1e-4             # abaixo de alpha_eps
        d = np.hypot((yy - min(H - 1, y1 + 4)) / 8.0, (xx - (x0 + x1) / 2) / max(8.0, (x1 - x0) / 1.5))
        shadow *= (1.0 - 0.5 * np.exp(-d ** 2))[..., None].astype(np.float32)
    return rgba, shadow.astype(np.float32)


def test_crop_composite_bit_identical():
    """Compor só no recorte (border + raio de influência + margem dos gaussianos) dá o MESMO quadro,
    byte a byte, que a conta no quadro inteiro (protótipo: scratchpad/speed/crop_bench.py), com e
    sem border, com o desfoque de nitidez ligado, com móveis encostados na borda da imagem e com as
    camadas recortadas (``origin``)."""
    H, W = 240, 320
    rng = np.random.default_rng(1)
    plate = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    plate[150:] = (170, 150, 130)
    full_box = [0, 0, W, H]
    for sharp in (0.0, 0.8):
        comp = Compositor(plate)
        comp.plate.blur_sigma_px = sharp                 # força (ou desliga) o casamento de nitidez
        for kind in ("meio", "canto", "dois", "ponto", "borda_dir"):
            rgba, shadow = _scene(H, W, kind)
            for border in (None, [0, 30, W, H], [50, 20, 305, 230], [0, 0, W, H]):
                out, st = comp.composite(3, rgba, shadow, border)
                ref_st = type(st)(3)
                ref_sub = comp._composite_region(ref_st, rgba, shadow, border, full_box)   # quadro inteiro
                assert np.array_equal(out, ref_sub), (sharp, kind, border)
                assert (st.object_pixels, st.shadow_pixels, st.changed_pixels, st.preserved_exact) == \
                    (ref_st.object_pixels, ref_st.shadow_pixels, ref_st.changed_pixels, True)
                if kind in ("meio", "ponto") and border != [0, 0, W, H]:
                    assert (st.box[2] - st.box[0]) * (st.box[3] - st.box[1]) < W * H   # recortou mesmo
            # camadas só no retângulo do border (como chegam do EXR recortado), fora dele neutras
            bx = [40, 25, 310, 235] if kind != "canto" else [0, 0, 120, 90]
            x0, y0, x1, y1 = bx
            rg_full = np.zeros_like(rgba)
            sh_full = np.ones_like(shadow)
            rg_full[y0:y1, x0:x1] = rgba[y0:y1, x0:x1]
            sh_full[y0:y1, x0:x1] = shadow[y0:y1, x0:x1]
            a, sa = comp.composite(4, rg_full, sh_full, None)
            b, sb = comp.composite(4, rgba[y0:y1, x0:x1].copy(), shadow[y0:y1, x0:x1].copy(), None,
                                   origin=(x0, y0))
            assert np.array_equal(a, b) and sa == sb, (sharp, kind)
    # sem nenhum pixel de móvel: a própria foto (e nenhuma conta)
    out, st = Compositor(plate).composite(1, np.zeros((H, W, 4), np.float32), np.full((H, W, 3), 0.5, np.float32),
                                          None)
    assert np.array_equal(out, plate) and st.box is None and st.changed_pixels == 0


def test_verification_catches_crop_paste_bug():
    """A verificação independente (``verify.py``: array_equal fora da máscara de influência) passa
    nos quadros certos e pega um erro de 1 px fora do recorte e uma colagem do recorte deslocada."""
    from mobilia.composite.verify import MARGIN_PX, influence_mask, outside_mask_equal

    H, W = 240, 320
    rng = np.random.default_rng(2)
    plate = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    comp = Compositor(plate)
    radius = comp.influence_px + MARGIN_PX
    for kind in ("meio", "canto", "dois", "ponto", "borda_dir"):
        rgba, shadow = _scene(H, W, kind, seed=3)
        out, st = comp.composite(1, rgba, shadow, None)
        mask = influence_mask(rgba[..., 3], (0, 0), (H, W), radius)
        changed = (out != plate).any(-1)
        assert changed.any() and not (changed & ~mask).any(), kind      # superconjunto, sem falso alarme
        assert outside_mask_equal(out, plate, mask) == (True, 0, 0)
        assert (~mask).any()
    # 1 px errado fora do recorte da composição
    rgba, shadow = _scene(H, W, "meio", seed=3)
    out, st = comp.composite(1, rgba, shadow, None)
    mask = influence_mask(rgba[..., 3], (0, 0), (H, W), radius)
    x0, y0, x1, y1 = st.box
    ys, xs = np.nonzero(~mask)
    k = int(np.argmax((ys < y0) | (ys >= y1) | (xs < x0) | (xs >= x1)))
    bad = out.copy()
    bad[ys[k], xs[k], 1] ^= 1
    assert outside_mask_equal(bad, plate, mask) == (False, 1, 1)
    # recorte colado 1 px para baixo (erro de índice na colagem)
    sub = out[y0:y1, x0:x1]
    shifted = plate.copy()
    shifted[y0 + 1:min(H, y1 + 1), x0:x1] = sub[:min(H, y1 + 1) - y0 - 1]
    ok, n_bad, _ = outside_mask_equal(shifted, plate, mask)
    assert not ok and n_bad > 0


# ---------------------------------------------------------------- PT8: ajuste do CG (grade.py)
def _grade_calib(H, W, gain=None, curve=None, knee=0.8):
    """``frames.json["calibration"]`` mínimo com o ajuste do CG."""
    adj = {"versao": 1, "joelho": knee, "curva_tom": curve or {"ok": False, "motivo": "teste"},
           "ganho": {"modo": "identidade"}}
    if gain is not None:
        adj["ganho"] = {"modo": "mapa", "limites": [0.85, 1.2], "grade": np.asarray(gain).tolist(),
                        "forma": list(np.asarray(gain).shape)}
    return {"ajuste_cg": adj}


def test_gain_map_only_touches_cg():
    """O mapa de ganho, a curva de tom e o joelho mudam SÓ a camada CG: fora dos pixels com alfa
    o quadro é byte a byte o mesmo sem o ajuste (e a foto fora da influência); recortar não muda
    nada; a verificação independente passa."""
    from mobilia.composite.grade import CGGrade
    from mobilia.composite.verify import MARGIN_PX, influence_mask, outside_mask_equal

    H, W = 240, 320
    rng = np.random.default_rng(5)
    plate = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    plate[150:] = (170, 150, 130)
    yy, xx = np.mgrid[0:6, 0:8]
    gain = np.stack([0.86 + 0.33 * xx / 7, 1.18 - 0.3 * yy / 5, np.full((6, 8), 1.05)], -1)
    curve = {"ok": True, "x_log2": [-4.0, -2.0, -1.0], "y_log2": [-4.2, -1.8, -0.9]}
    grade = CGGrade.from_calibration(_grade_calib(H, W, gain, curve), (H, W))
    assert grade.gain.shape == (H, W, 3) and not grade.gain.flags.writeable
    comp = Compositor(plate)
    for kind in ("meio", "canto", "dois", "borda_dir"):
        rgba, shadow = _scene(H, W, kind, seed=8)
        rgba[..., :3] *= 2.5                                   # altas luzes acima do joelho
        a, sa = comp.composite(1, rgba, shadow, None)
        b, sb = comp.composite(1, rgba, shadow, None, grade=grade)
        cg = rgba[..., 3] > comp.s.alpha_eps
        assert not np.array_equal(a[cg], b[cg])                 # o CG mudou...
        # ... e nada fora dele (sombra e foto idênticas; o desfoque de nitidez espalha o CG 1–2 px)
        far = ~ndimage_dilate(cg, 4)
        assert np.array_equal(a[far], b[far]), kind
        assert sb.shadow_pixels == sa.shadow_pixels and sb.preserved_exact
        mask = influence_mask(rgba[..., 3], (0, 0), (H, W), comp.influence_px + MARGIN_PX)
        assert outside_mask_equal(b, plate, mask) == (True, 0, 0)
        # recorte (camadas só no border) = quadro inteiro, byte a byte, com o ajuste
        ref = comp._composite_region(type(sb)(1), rgba, shadow, None, [0, 0, W, H], grade)
        assert np.array_equal(b, ref), kind
    # joelho: nada passa de 1 e é contínuo; abaixo de 0,8 é a identidade
    from mobilia.composite.grade import knee
    x = np.linspace(0, 5, 501, dtype=np.float32)
    y = knee(x)
    assert np.all(np.diff(y) >= 0) and y.max() <= 1.0 and np.array_equal(y[x <= 0.8], x[x <= 0.8])
    assert knee(np.float32([1.0]))[0] == pytest.approx(0.8 + 0.2 * np.tanh(1.0))
    # trabalho antigo (sem ajuste_cg): nenhum ajuste, resultado de antes
    assert CGGrade.from_calibration({"rel_rms": 0.2}, (H, W)) is None


def ndimage_dilate(m, it):
    from scipy import ndimage

    return ndimage.binary_dilation(m, iterations=it)


def test_gain_map_fallback_bright_room():
    """Sala branca (quase tudo estourado): com pouca máscara o mapa vira ganho global; com
    pouquíssima, identidade. Mapa normal: suave, limitado e com o ganho global onde a máscara é rala."""
    from mobilia.composite.grade import GAIN_LIMITS, gain_grid, gain_map, upsample_grid

    h, w = 60, 80
    yy, xx = np.mgrid[0:h, 0:w]
    pred = np.full((h, w, 3), 0.4, np.float32)
    photo = (pred * np.stack([1.0 + 0.25 * xx / w, np.ones((h, w)), 0.9 * np.ones((h, w))], -1)).astype(np.float32)
    full = np.ones((h, w), bool)
    G, info = gain_map(photo, pred, full)
    assert info["modo"] == "mapa" and info["fracao_mascara"] == 1.0
    assert G.min() >= GAIN_LIMITS[0] and G.max() <= GAIN_LIMITS[1]
    assert G[..., 0][:, -1].mean() > G[..., 0][:, 0].mean() + 0.1          # segue o gradiente da foto
    assert np.allclose(G[..., 2], 0.9, atol=0.02) and np.allclose(G[..., 1], 1.0, atol=0.02)
    # grade 8× menor e de volta: o mapa de baixa frequência sobrevive
    back = upsample_grid(gain_grid(G), (h, w))
    assert np.abs(back - G).max() < 0.02
    # sala branca: só 3% do quadro com pixels não estourados -> ganho global
    few = np.zeros((h, w), bool)
    few[:12, :12] = True
    G2, info2 = gain_map(photo, pred, few)
    assert info2["modo"] == "global" and info2["fracao_mascara"] == pytest.approx(144 / (h * w), abs=1e-4)
    assert np.allclose(G2, G2[0, 0]) and G2[0, 0, 1] == pytest.approx(1.0, abs=1e-3)
    # sala branca em que o quadro ainda tem máscara, mas quase toda a superfície está estourada
    # (sobram cantos e rodapés): ganho global também
    lots = np.zeros((h, w), bool)
    lots[:, :16] = True                                              # 20% do quadro
    G4, info4 = gain_map(photo, pred, lots, lower=~lots)
    assert info4["modo"] == "global" and info4["fracao_superficies"] == pytest.approx(0.2)
    assert gain_map(photo, pred, lots, lower=lots & (yy < 5))[1]["modo"] == "mapa"
    # quase nada: identidade
    G3, info3 = gain_map(photo, pred, few & (xx < 3) & (yy < 3))
    assert info3["modo"] == "identidade" and np.all(G3 == 1.0)
    # no compositor: modo global é um ganho constante, sem mapa
    from mobilia.composite.grade import CGGrade, grade_data

    data = grade_data(photo, pred, few)
    g = CGGrade.from_calibration({"ajuste_cg": data}, (120, 160))
    assert g.gain is None and g.gain_const is not None and "grade" not in data["ganho"]


def test_tone_curve_identity_on_bad_fit():
    """Curva de tom: identidade com dados sem relação (resíduo/dispersão altos) ou poucos pixels;
    recupera uma resposta monotônica real; nunca achata o contraste nem desvia demais."""
    from mobilia.composite.grade import TONE_MAX_DEV, TONE_MIN_SLOPE, eval_log2, fit_tone_curve

    rng = np.random.default_rng(3)
    x = np.exp2(rng.uniform(-4, 0, 20000))
    # foto sem relação com a predição
    bad = fit_tone_curve(x, np.exp2(rng.uniform(-4, 0, 20000)))
    assert bad["ok"] is False
    # poucos pixels
    assert fit_tone_curve(x[:100], x[:100])["ok"] is False
    # identidade com ruído: não vale a pena aplicar
    same = fit_tone_curve(x, x * np.exp2(rng.normal(0, 0.1, x.size)))
    assert same["ok"] is False
    # resposta real (compressão das altas luzes + textura no y): a curva é aceita e a recupera
    lx = np.log2(x)
    ty = lx - 0.08 * np.clip(lx + 2, 0, None) ** 2 + 0.05
    good = fit_tone_curve(x, np.exp2(ty + rng.normal(0, 0.15, x.size)))
    assert good["ok"] is True, good
    probe = np.linspace(-3.5, -0.5, 30)
    want = probe - 0.08 * np.clip(probe + 2, 0, None) ** 2 + 0.05
    assert np.abs(eval_log2(probe, good) - want).max() < 0.08
    # monotônica, inclinação mínima, desvio limitado, e volta à identidade fora da faixa dos dados
    xs = np.linspace(-12, 4, 2000)
    ys = eval_log2(xs, good)
    assert np.all(np.diff(ys) > 0)
    assert np.all(np.diff(ys) / np.diff(xs) >= TONE_MIN_SLOPE - 1e-6)
    assert np.abs(ys - xs).max() <= TONE_MAX_DEV + 1e-9
    assert eval_log2(np.array([-12.0, 4.0]), good) == pytest.approx([-12.0, 4.0])
    # no CG: aplicada na cor sem pré-multiplicação (o alfa não muda o resultado da cor)
    from mobilia.composite.grade import CGGrade

    g = CGGrade(curve={"x_log2": good["x_log2"], "y_log2": good["y_log2"]}, knee_at=None)
    C = np.full((2, 2, 3), 0.3, np.float32)
    A = np.array([[1.0, 0.5], [0.25, 0.0]], np.float32)
    out = g.apply(C * A[..., None], A, [0, 0, 2, 2])
    unp = out[A > 0] / A[A > 0][:, None]
    assert np.allclose(unp, unp[0], atol=1e-6) and out[1, 1].sum() == 0


# ---------------------------------------------------------------- PT10: sombra de contato analítica
_CH, _CW = 120, 160


def _contact_meta(footprint=(0.4, 0.4), center=(0.0, 2.6), heights=0.0, cell=0.02):
    """Câmera a 1,5 m olhando +y, 35° para baixo; uma peça com a silhueta de baixo retangular."""
    from mobilia.composite import contact as ct

    a = np.radians(35.0)
    x_c = np.array([1.0, 0.0, 0.0])
    z_c = np.array([0.0, np.cos(a), -np.sin(a)])
    y_c = np.cross(z_c, x_c)
    M = np.eye(4)
    M[:3, :3] = np.column_stack([x_c, y_c, z_c])
    M[:3, 3] = [0.0, 0.0, 1.5]
    intr = {"width": _CW, "height": _CH, "fx": 130.0, "fy": 130.0, "cx": _CW / 2, "cy": _CH / 2}
    nx, ny = int(round(footprint[0] / cell)), int(round(footprint[1] / cell))
    h = np.full((ny, nx), float(heights)) if np.isscalar(heights) else np.asarray(heights, float)
    origin = (center[0] - (nx - 1) * cell / 2, center[1] - (ny - 1) * cell / 2)
    rec = ct.part_record("peca/base", "peca", origin, cell, h)
    return {"versao": 1, "width": _CW, "height": _CH, "camera": {"intrinsics": intr, "cam_to_room": M.tolist()},
            "celula_m": cell, "altura_max_m": ct.HEIGHT_MAX_M, "pecas": [rec]}


def _pose(dz=0.0, fade=1.0):
    from mobilia.composite import contact as ct

    D = np.eye(4)[:3]
    D[2, 3] = dz
    return [ct.pose_record(0, fade, D)]


def _floor_plate():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, size=(_CH, _CW, 3), dtype=np.uint8)
    img[40:, :] = (170, 150, 130)                       # piso liso
    return img


def _project(meta, pts):
    from mobilia.render.calibration import project_points

    spec = {"cam_to_room": meta["camera"]["cam_to_room"], "intrinsics": meta["camera"]["intrinsics"]}
    uv, z = project_points(spec, np.asarray(pts, float))
    assert (z > 0).all()
    return uv


def test_contact_homography_matches_projection():
    """A homografia piso→imagem é a mesma projeção da câmera usada no resto do motor; a silhueta
    codificada volta igual (mm)."""
    from mobilia.composite import contact as ct

    meta = _contact_meta(heights=np.array([[0.0, 0.0123], [np.nan, 0.5996]]), footprint=(0.04, 0.04))
    Hf = ct.floor_homography(meta["camera"]["intrinsics"], meta["camera"]["cam_to_room"])
    P = np.array([[0.3, 2.0, 0.0], [-0.5, 3.4, 0.0], [0.0, 2.6, 0.0]])
    q = np.column_stack([P[:, :2], np.ones(3)]) @ Hf.T
    assert np.allclose(q[:, :2] / q[:, 2:], _project(meta, P), atol=1e-9)
    h = ct.decode_heights(meta["pecas"][0]["alturas_mm"], 4)
    assert h[0] == 0.0 and h[1] == 0.012 and np.isnan(h[2]) and h[3] == 0.6
    m = ct.ContactModel.from_meta(meta)
    assert m.n_parts == 1 and len(m.parts[0].pts) == 3


def test_contact_shadow_inside_influence_only():
    """PT10: a sombra de contato escurece o piso junto à base, mas só dentro da região de
    influência dos móveis. Uma silhueta enorme (4 m × 4 m, cobrindo quase todo o piso da imagem)
    junto de um móvel pequeno: fora da máscara independente de ``verify.py`` o quadro continua a
    foto byte a byte."""
    from mobilia.composite import contact as ct
    from mobilia.composite.verify import MARGIN_PX, influence_mask, outside_mask_equal

    plate8 = _floor_plate()
    meta = _contact_meta(footprint=(4.0, 4.0), center=(0.0, 3.0), cell=0.04)
    model = ct.ContactModel.from_meta(meta)
    # a oclusão sozinha cobre boa parte do piso da imagem
    c = model.occlusion(_pose(), [0, 0, _CW, _CH])
    assert (c > 0.5).mean() > 0.3
    # móvel pequeno no meio do piso
    rgba = np.zeros((_CH, _CW, 4), np.float32)
    rgba[70:84, 72:88, :3] = 0.2
    rgba[70:84, 72:88, 3] = 1.0
    comp = Compositor(plate8, CompositeSettings(grain_strength=0.0))
    base, _ = comp.composite(3, rgba, None, None)
    out, st = comp.composite(3, rgba, None, None, contact=model.frame(_pose()))
    assert st.preserved_exact and st.contact_pixels > 0
    mask = influence_mask(rgba[..., 3], (0, 0), (_CH, _CW), comp.influence_px + MARGIN_PX)
    ok, n_bad, _ = outside_mask_equal(out, plate8, mask)
    assert ok and n_bad == 0
    assert not mask.all() and (c[~mask] > 0.5).any()      # havia sombra a aplicar fora da máscara
    # dentro da influência, o piso junto ao móvel escureceu em relação ao quadro sem contato
    near = (slice(84, 90), slice(70, 90))
    assert out[near].astype(float).mean() < base[near].astype(float).mean() - 20
    # o móvel em si não muda (alfa 1: a sombra só multiplica a foto atrás dele)
    assert np.array_equal(out[71:83, 73:87], base[71:83, 73:87])
    # força 0: exatamente o quadro sem sombra de contato
    off = Compositor(plate8, CompositeSettings(grain_strength=0.0, contact_strength=0.0))
    o0, _ = off.composite(3, rgba, None, None, contact=model.frame(_pose()))
    assert np.array_equal(o0, base)


def test_contact_shadow_fades_with_height():
    """Peça subindo/caindo: colada no piso a linha é escura e estreita; quanto mais alta, mais
    fraca e larga; invisível (fade 0) não faz sombra."""
    from mobilia.composite import contact as ct

    meta = _contact_meta(footprint=(0.3, 0.3), center=(0.0, 2.6), cell=0.015)
    # o campo lido num quadro 4× maior (≈ 0,5 cm por pixel no piso): a câmera escala junto
    big = ct.ContactModel(meta, (4 * _CH, 4 * _CW))
    # pixel do piso na borda da base e um a 12 cm dela
    uv = 4 * _project(meta, [[0.0, 2.6 - 0.15 - 0.0075, 0.0], [0.0, 2.6 - 0.15 - 0.12, 0.0]])
    box = [0, 0, 4 * _CW, 4 * _CH]
    near, far, peak = [], [], []
    for dz in (0.0, 0.05, 0.15, 0.35, 0.6):
        c = big.occlusion(_pose(dz), box)
        near.append(float(c[int(uv[0, 1]), int(uv[0, 0])]))
        far.append(float(c[int(uv[1, 1]), int(uv[1, 0])]))
        peak.append(float(c.max()))
    assert all(a > b for a, b in zip(peak, peak[1:])), peak
    assert peak[0] > 0.9 and peak[-1] < 0.1
    assert near[0] > near[2] > near[-1]
    # subindo, a sombra alarga: a 12 cm da base ela só aparece com a peça no alto (desfoque maior)
    assert far[0] < 0.005 and max(far[1:4]) > far[0] + 0.01, far
    # fade: proporcional; invisível não conta
    model = ct.ContactModel.from_meta(meta)
    box = [0, 0, _CW, _CH]
    c1 = model.occlusion(_pose(0.0, 1.0), box)
    c5 = model.occlusion(_pose(0.0, 0.5), box)
    assert c5.max() < c1.max() and model.occlusion(_pose(0.0, 0.0), box) is None
    assert model.frame([]) is None
    # na composição: mais escuro com a peça no piso que no alto
    plate8 = _floor_plate()
    rgba = np.zeros((_CH, _CW, 4), np.float32)
    corners = _project(meta, [[x, y, 0.0] for x in (-0.15, 0.15) for y in (2.45, 2.75)])
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    rgba[y0 - 12:y0 + 2, x0:x1, :3] = 0.3                  # o corpo da peça logo acima da base
    rgba[y0 - 12:y0 + 2, x0:x1, 3] = 1.0
    comp = Compositor(plate8, CompositeSettings(grain_strength=0.0))
    ring = (slice(y1 - 1, y1 + 4), slice(x0, x1))
    lum = []
    for dz in (0.0, 0.15, 0.6):
        out, st = comp.composite(1, rgba, None, None, contact=model.frame(_pose(dz)))
        assert st.preserved_exact
        lum.append(out[ring].astype(float).mean())
    assert lum[0] < lum[1] < lum[2] <= plate8[ring].astype(float).mean() + 1e-9


def test_contact_shadow_follows_part_motion():
    """A silhueta segue a matriz da peça (deslize lateral): a sombra anda junto. Abaixo do piso
    (surgir do chão, PT11) pesa pela profundidade: igual à mesma altura acima do piso, então a
    sombra só aparece quando a peça chega perto do repouso."""
    from mobilia.composite import contact as ct

    meta = _contact_meta(footprint=(0.3, 0.3), center=(0.0, 2.6), cell=0.015)
    model = ct.ContactModel.from_meta(meta)
    D = np.eye(4)[:3]
    D[0, 3] = 0.5
    box = [0, 0, _CW, _CH]
    c0 = model.occlusion(_pose(), box)
    c1 = model.occlusion([ct.pose_record(0, 1.0, D)], box)
    u0 = (c0.sum(0) * np.arange(_CW)).sum() / c0.sum()
    u1 = (c1.sum(0) * np.arange(_CW)).sum() / c1.sum()
    assert u1 > u0 + 10
    below = model.occlusion(_pose(-0.2), box)
    above = model.occlusion(_pose(0.2), box)
    assert np.allclose(below, above)
    assert below.max() < 0.5 * c0.max()
    deep = model.occlusion(_pose(-1.2), box)
    assert deep.max() < 0.02 * c0.max()
    assert np.abs(model.occlusion(_pose(-1e-4), box) - c0).max() < 0.02 * c0.max()


def test_contact_occlusion_restricted_equals_dense():
    """O campo só é lido no retângulo da grade projetada e, com ``where``, só nos pixels marcados
    (o compositor passa a região de influência): nesses pixels o valor é exatamente o da leitura
    densa de cada pixel do recorte; nos demais, 0. A composição com e sem a restrição é igual."""
    from scipy import ndimage

    from mobilia.composite import contact as ct

    meta = _contact_meta(footprint=(0.3, 0.3), center=(0.2, 2.4), cell=0.015)
    model = ct.ContactModel(meta, (2 * _CH, 2 * _CW))
    rng = np.random.default_rng(3)
    for dz in (0.0, 0.2, 0.5):
        poses = _pose(dz)
        field, (gx0, gy0, g) = model.floor_field(poses)
        for box in ([0, 0, 2 * _CW, 2 * _CH], [37, 101, 251, 240], [0, 0, 30, 20]):
            X0, Y0, X1, Y1 = box
            uu, vv = np.meshgrid(np.arange(X0, X1) + 0.5, np.arange(Y0, Y1) + 0.5)
            q = np.einsum("ij,jhw->ihw", model.Hinv, np.stack([uu, vv, np.ones_like(uu)]))
            front = q[2] > 1e-12
            wz = np.where(front, q[2], 1.0)
            cx = np.where(front, (q[0] / wz - gx0) / g, -10.0)
            cy = (q[1] / wz - gy0) / g
            ref = ndimage.map_coordinates(field, [cy, cx], order=1, mode="constant", cval=0.0)
            ref = np.where(front, np.clip(ref, 0.0, 1.0), 0.0).astype(np.float32)
            c = model.occlusion(poses, box)
            assert np.array_equal(c, ref)
            where = rng.random(ref.shape) < 0.4
            cw = model.occlusion(poses, box, where)
            assert np.array_equal(cw, np.where(where, ref, 0.0))
            if box[0] == 0 and box[2] > 100 and dz == 0.0:
                assert (c > 0.01).any() and (ref == 0).mean() > 0.5   # campo local: o corte economiza
    # na composição o resultado é o mesmo de ler o campo no recorte inteiro
    plate8 = np.repeat(np.repeat(_floor_plate(), 2, 0), 2, 1)
    rgba = np.zeros((2 * _CH, 2 * _CW, 4), np.float32)
    corners = 2 * _project(meta, [[x, y, 0.0] for x in (0.05, 0.35) for y in (2.25, 2.55)])
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, _ = np.ceil(corners.max(0)).astype(int)
    rgba[y0 - 20:y0 + 3, x0:x1, :3] = 0.3
    rgba[y0 - 20:y0 + 3, x0:x1, 3] = 1.0

    class _Dense:
        def __init__(self, fr):
            self.fr = fr

        def occlusion(self, box, where=None):
            return self.fr.occlusion(box)

    comp = Compositor(plate8, CompositeSettings(grain_strength=0.0))
    fr = model.frame(_pose())
    a, sa = comp.composite(2, rgba, None, None, contact=_Dense(fr))
    b, sb = comp.composite(2, rgba, None, None, contact=fr)
    assert sb.contact_pixels > 0 and sa.contact_pixels == sb.contact_pixels and np.array_equal(a, b)


_CONTACT_SCRIPT = r'''
import json, sys
import bpy, mathutils
import numpy as np
from mobilia.render import blender_worker as bw
from mobilia.render import choreography as ch

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

def box(name, size, loc):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return o

def item(iid, role, mount, parts, pos):
    root = bpy.data.objects.new(iid, None)
    sc.collection.objects.link(root)
    for p in parts:
        p.parent = root
        p["rest_location"] = list(p.location)
    root.location = pos
    it = {"id": iid, "role": role, "mount": mount, "start": 0.5, "duration": 1.0, "size": [0.8, 0.5, 0.44]}
    plans = {p.name: p2 for p, p2 in zip(parts, ch.plan_parts(it, [p.name for p in parts]))}
    return {"item": it, "root": root, "parts": parts, "plans": plans}

legs = [box(f"mesa_leg{i}", (0.04, 0.04, 0.4), (sx * 0.35, sy * 0.2, 0.2))
        for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)])]
top = box("mesa_top", (0.8, 0.5, 0.04), (0.0, 0.0, 0.42))
rts = [item("mesa", "coffee_table", "floor", legs + [top], (1.0, 2.0, 0.0)),
       item("quadro", "wall_art", "wall", [box("quadro_tela", (0.5, 0.02, 0.4), (0, 0, 0))], (0, 3, 1.2)),
       item("tapete", "rug", "floor", [box("tapete_base", (2.0, 1.4, 0.01), (0, 0, 0.005))], (1, 2, 0))]
fps, n = 24, 72
bw.bake_animation(sc, rts, fps, n)
sc.frame_set(n)
bpy.context.view_layer.update()
a = np.radians(35.0)
x_c, z_c = np.array([1.0, 0, 0]), np.array([0, np.cos(a), -np.sin(a)])
M = np.eye(4)
M[:3, :3] = np.column_stack([x_c, np.cross(z_c, x_c), z_c])
M[:3, 3] = [1.0, 0.0, 1.5]
spec = {"width": 160, "height": 120, "intrinsics": {"fx": 130, "fy": 130, "cx": 80, "cy": 60},
        "cam_to_room": M.tolist()}
meta, refs = bw.contact_model(rts, spec)
rest = bw.contact_poses(refs, n, fps)
f_mid = 16                                  # queda da base em andamento
sc.frame_set(f_mid)
bpy.context.view_layer.update()
mid = bw.contact_poses(refs, f_mid, fps)
dz = [rts[0]["plans"][p.name].pose((f_mid - 1) / fps).dz for p, _ in [(r[1], 0) for r in refs]]
print("RESULT:" + json.dumps({"meta": meta, "rest": rest, "mid": mid, "dz": dz}))
'''


@pytest.mark.skipif(__import__("importlib").util.find_spec("bpy") is None, reason="bpy não está instalado")
def test_worker_contact_model_records_floor_parts(tmp_path):
    """Worker (bpy real): a silhueta de baixo das peças de piso vai para o modelo (pés colados no
    piso; tampo a 0,40 m), quadro e tapete ficam de fora, e a pose de cada quadro é a matriz da
    peça relativa ao repouso (identidade no repouso, a queda da animação no meio)."""
    import json
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    from mobilia.composite import contact as ct

    repo = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_CONTACT_SCRIPT), str(tmp_path)],
                          cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    res = json.loads(lines[-1][len("RESULT:"):])
    meta = res["meta"]
    ids = [p["id"] for p in meta["pecas"]]
    assert sorted(i.split("/")[1] for i in ids) == ["mesa_leg0", "mesa_leg1", "mesa_leg2", "mesa_leg3", "mesa_top"]
    model = ct.ContactModel.from_meta(meta)
    for rec, part in zip(meta["pecas"], model.parts):
        z = part.pts[:, 2]
        if rec["id"].endswith("top"):
            assert abs(z.min() - 0.40) < 0.002 and len(z) >= 0.9 * (0.8 / ct.CELL_M) * (0.5 / ct.CELL_M)
        else:
            assert abs(z.max()) < 0.002 and 4 <= len(z) <= 16
            assert abs(part.pts[:, 0].mean() - (1.0 + np.sign(part.pts[:, 0].mean() - 1.0) * 0.35)) < 0.01
    ident = np.array(ct.IDENTITY)
    assert len(res["rest"]) == 5 and all(np.allclose(r[2:], ident, atol=1e-5) and r[1] == 1.0 for r in res["rest"])
    assert res["mid"]
    lift = []
    for r in res["mid"]:
        pts = model.parts[r[0]].pts
        moved = pts @ np.array(r[2:]).reshape(3, 4).T
        lift.append(float((moved[:, 2] - pts[:, 2]).mean()))
        assert abs(lift[-1] - res["dz"][r[0]]) < 0.01, (lift[-1], res["dz"][r[0]])
    assert max(lift) > 0.05


# ---------------------------------------------------------------- PT6: render em camadas
def _two_objects(H, W):
    """Duas peças separadas (A parada, B "dinâmica"), cada uma com a sua sombra suave no piso."""
    rgba_a = np.zeros((H, W, 4), np.float32)
    rgba_b = np.zeros((H, W, 4), np.float32)
    rgba_a[100:150, 40:110] = (0.25, 0.12, 0.06, 1.0)
    rgba_a[99, 40:110] = (0.1, 0.05, 0.02, 0.45)
    rgba_b[110:160, 200:280] = (0.08, 0.2, 0.3, 1.0)
    rgba_b[109, 200:280] = (0.03, 0.08, 0.12, 0.5)
    yy, xx = np.mgrid[0:H, 0:W]
    sh_a = 1.0 - 0.55 * np.exp(-(((yy - 152) / 5.0) ** 2 + ((xx - 75) / 30.0) ** 2))
    sh_b = 1.0 - 0.55 * np.exp(-(((yy - 162) / 5.0) ** 2 + ((xx - 240) / 34.0) ** 2))
    return (rgba_a, np.repeat(sh_a[..., None], 3, -1).astype(np.float32),
            rgba_b, np.repeat(sh_b[..., None], 3, -1).astype(np.float32))


def _layer_plate(H, W, seed=4):
    rng = np.random.default_rng(seed)
    plate = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    ramp = (np.arange(95, H) - 95)[:, None, None] // 20
    plate[95:] = np.clip(np.array([175, 158, 138])[None, None, :] + ramp, 0, 255).astype(np.uint8)
    return plate


def test_layered_equals_single_layer_when_static():
    """Sem movimento, marco (peça A) + camada dinâmica (peça B) reproduzem o quadro único (A e B no
    mesmo render) dentro de 1/255, com o grão e a nitidez da câmera ligados; com a camada
    dinâmica vazia o resultado é o próprio marco, byte a byte."""
    from mobilia.composite.compositor import LayerAccumulator

    H, W = 240, 320
    plate = _layer_plate(H, W)
    rgba_a, sh_a, rgba_b, sh_b = _two_objects(H, W)
    for sharp in (0.0, 0.7):
        comp = Compositor(plate)
        comp.plate.blur_sigma_px = sharp
        single, st1 = comp.composite(9, rgba_a + rgba_b, sh_a * sh_b, None)
        base = LayerAccumulator.static_base(comp, 9, rgba_a, sh_a, source="m_00009.exr")
        layered, st2 = LayerAccumulator.compose_dynamic(comp, 9, base, rgba_b, sh_b)
        d = np.abs(single.astype(np.int16) - layered.astype(np.int16))
        assert d.max() <= 1, (sharp, int(d.max()))
        assert st1.preserved_exact and st2.preserved_exact
        # o marco sozinho é um quadro comum (mesmos bytes de uma composição só de A)
        only_a, _ = comp.composite(9, rgba_a, sh_a, None)
        assert np.array_equal(base.out8, only_a)
        # camada dinâmica vazia: o quadro é o marco
        empty, _ = LayerAccumulator.compose_dynamic(comp, 10, base, np.zeros_like(rgba_b), np.ones_like(sh_b))
        assert np.array_equal(empty, base.out8)
        # e sem marco a camada dinâmica é o quadro único dela
        alone, _ = LayerAccumulator.compose_dynamic(comp, 9, None, rgba_b, sh_b)
        assert np.array_equal(alone, comp.composite(9, rgba_b, sh_b, None)[0])


def test_layered_plate_bit_exact():
    """A camada dinâmica sobre o marco: fora da máscara de influência dela a saída é o marco byte a
    byte; fora da UNIÃO das máscaras (marco ∪ dinâmica) é a foto byte a byte (a máscara só do
    dinâmico não basta: a verificação a acusaria). Com camadas recortadas (``origin``) e com a
    peça dinâmica na frente da parada e sombreando-a, o resultado é o mesmo."""
    from mobilia.composite.compositor import LayerAccumulator
    from mobilia.composite.verify import MARGIN_PX, influence_mask, outside_mask_equal

    H, W = 240, 320
    plate = _layer_plate(H, W, seed=5)
    comp = Compositor(plate)
    radius = comp.influence_px + MARGIN_PX
    rgba_a, sh_a, rgba_b, sh_b = _two_objects(H, W)
    # B passa na frente de A (cobre parte dela) e projeta sombra sobre ela (catcher no render)
    rgba_b[120:150, 90:200] = (0.08, 0.2, 0.3, 1.0)
    sh_b[100:150, 40:90] *= 0.8
    base = LayerAccumulator.static_base(comp, 3, rgba_a, sh_a)
    out, st = LayerAccumulator.compose_dynamic(comp, 3, base, rgba_b, sh_b)
    m_a = influence_mask(rgba_a[..., 3], (0, 0), (H, W), radius)
    m_b = influence_mask(rgba_b[..., 3], (0, 0), (H, W), radius)
    assert st.preserved_exact
    assert outside_mask_equal(out, base.out8, m_b) == (True, 0, 0)
    assert outside_mask_equal(out, plate, m_a | m_b) == (True, 0, 0)
    assert (~(m_a | m_b)).any()
    ok_b, n_bad, _ = outside_mask_equal(out, plate, m_b)
    assert not ok_b and n_bad > 0                       # o marco está fora da máscara do dinâmico
    # a sombra de B escureceu a peça A do marco (onde B não a cobre)
    assert out[105:118, 45:85].mean() < base.out8[105:118, 45:85].mean() - 3
    # camadas recortadas no border (como vêm do EXR): mesmos bytes
    bx = [180, 95, 300, 175]
    x0, y0, x1, y1 = bx
    rg = np.zeros_like(rgba_b)
    sb = np.ones_like(sh_b)
    rg[y0:y1, x0:x1] = rgba_b[y0:y1, x0:x1]
    sb[y0:y1, x0:x1] = sh_b[y0:y1, x0:x1]
    full, _ = LayerAccumulator.compose_dynamic(comp, 4, base, rg, sb)
    crop, _ = LayerAccumulator.compose_dynamic(comp, 4, base, rgba_b[y0:y1, x0:x1].copy(),
                                               sh_b[y0:y1, x0:x1].copy(), origin=(x0, y0))
    assert np.array_equal(full, crop)
    # o marco é só leitura: nenhuma composição dinâmica pode mexer nele
    with pytest.raises(ValueError):
        base.out8[0, 0, 0] = 1


def test_blend_frames_keeps_equal_pixels():
    """Transição entre marcos: onde as duas composições coincidem (a foto) os bytes não mudam."""
    from mobilia.composite.compositor import blend_frames

    rng = np.random.default_rng(6)
    a = rng.integers(0, 256, size=(20, 30, 3), dtype=np.uint8)
    b = a.copy()
    b[5:10, 5:12] = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        o = blend_frames(a, b, w)
        same = (a == b).all(-1)
        assert np.array_equal(o[same], a[same])
    assert np.array_equal(blend_frames(a, b, 0.0), a) and np.array_equal(blend_frames(a, b, 1.0), b)
    mid = blend_frames(a, b, 0.5).astype(int)
    assert np.abs(mid - (a.astype(int) + b.astype(int)) / 2).max() <= 0.5 + 1e-9


def test_contact_shadow_of_dynamic_layer_skips_milestone_cg():
    """A sombra de contato da camada dinâmica só escurece pixels em que o marco é foto: sobre o CG
    do marco (p. ex. o tapete) vale a sombra do render, como no quadro único."""
    from mobilia.composite.compositor import LayerAccumulator

    H, W = 240, 320
    plate = _layer_plate(H, W, seed=7)
    comp = Compositor(plate, CompositeSettings(grain_strength=0.0))
    comp.plate.blur_sigma_px = 0.0
    rug = np.zeros((H, W, 4), np.float32)
    rug[150:200, 60:160] = (0.3, 0.1, 0.1, 1.0)
    base = LayerAccumulator.static_base(comp, 1, rug, np.ones((H, W, 3), np.float32))
    obj = np.zeros((H, W, 4), np.float32)
    obj[120:160, 100:200] = (0.1, 0.1, 0.1, 1.0)

    class _C:                       # oclusão de contato uniforme 1 em todo o recorte
        def occlusion(self, box, where=None):
            return np.ones((box[3] - box[1], box[2] - box[0]), np.float32)

    out, st = LayerAccumulator.compose_dynamic(comp, 2, base, obj, np.ones((H, W, 3), np.float32), contact=_C())
    assert st.contact_pixels > 0
    on_rug = (slice(162, 170), slice(70, 95))
    on_floor = (slice(162, 170), slice(170, 195))
    assert np.array_equal(out[on_rug], base.out8[on_rug])
    assert out[on_floor].mean() < plate[on_floor].mean() - 10


_LAYER_SCRIPT = r'''
import json, sys
from pathlib import Path
import bpy
from mobilia.render import blender_worker as bw

out = Path(sys.argv[1])
W, H, spp = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = spp
sc.cycles.use_denoising = True
sc.cycles.seed = 3
sc.render.film_transparent = True
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.view_settings.view_transform = "Standard"
sc.view_layers[0].cycles.use_pass_shadow_catcher = True
im = sc.render.image_settings
im.media_type = "MULTI_LAYER_IMAGE"
im.color_depth = "32"

def mat(c):
    m = bpy.data.materials.new("m")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*c, 1)
    return m

bpy.ops.mesh.primitive_plane_add(size=20)
floor = bpy.context.object
floor.data.materials.append(mat((0.6, 0.6, 0.6)))
floor.is_shadow_catcher = True
bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.2, 0.3, 0.5))
a = bpy.context.object
a.name = "A_corpo"
a.scale = (1.4, 0.8, 1.0)
a.data.materials.append(mat((0.7, 0.25, 0.2)))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(0.55, -0.55, 0.35))
b = bpy.context.object
b.name = "B_bola"
b.data.materials.append(mat((0.2, 0.3, 0.75)))
bpy.ops.object.light_add(type="SUN", rotation=(0.7, 0.35, 0.4))
bpy.context.object.data.energy = 3
bpy.ops.object.camera_add(location=(0, -6, 3.2), rotation=(1.12, 0, 0))
sc.camera = bpy.context.object
w = bpy.data.worlds.new("w")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[1].default_value = 0.4
rts = [{"item": {"id": "A"}, "parts": [a]}, {"item": {"id": "B"}, "parts": [b]}]

def render(name):
    sc.render.filepath = str(out / name)
    bpy.ops.render.render(write_still=True)

bw.set_layer_roles(rts, None)
render("unico.exr")
bw.set_layer_roles(rts, frozenset({"A_corpo"}))                          # marco: só A
render("marco.exr")
n = bw.set_layer_roles(rts, frozenset({"B_bola"}), frozenset({"A_corpo"}))   # camada de B, A catcher
render("dinamico.exr")
print("RESULT:" + json.dumps({"trocas": n, "catcher": bool(a.is_shadow_catcher), "cam_b": bool(b.visible_camera)}))
'''


@pytest.mark.slow
@pytest.mark.skipif(__import__("importlib").util.find_spec("bpy") is None, reason="bpy não está instalado")
def test_layered_blender_matches_single_render(tmp_path):
    """Cycles real (320 px): marco (peça A) + camada dinâmica (peça B com A como *shadow catcher*)
    compostos reproduzem o render único das duas peças: diferença média < 2/255 nos pixels de CG e
    das sombras, zero fora da máscara de influência (a foto)."""
    import json
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    from mobilia.composite.compositor import LayerAccumulator, read_render_passes
    from mobilia.composite.verify import MARGIN_PX, influence_mask

    W, H = 320, 240
    repo = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_LAYER_SCRIPT), str(tmp_path), str(W), str(H), "64"],
                          cwd=repo, env=env, capture_output=True, text=True, timeout=900)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    res = json.loads(lines[-1][len("RESULT:"):])
    assert res == {"trocas": 2, "catcher": True, "cam_b": True}
    yy, xx = np.mgrid[0:H, 0:W]
    plate = np.stack([120 + 60 * yy / H, 110 + 50 * xx / W, 100 + 0 * xx], -1).astype(np.uint8)
    comp = Compositor(plate)
    one = read_render_passes(tmp_path / "unico.exr")
    ma = read_render_passes(tmp_path / "marco.exr")
    dy = read_render_passes(tmp_path / "dinamico.exr")
    single, _ = comp.composite(1, one[0], one[1], None)
    base = LayerAccumulator.static_base(comp, 1, ma[0], ma[1])
    layered, st = LayerAccumulator.compose_dynamic(comp, 1, base, dy[0], dy[1])
    assert st.preserved_exact
    radius = comp.influence_px + MARGIN_PX
    mask = influence_mask(ma[0][..., 3], (0, 0), (H, W), radius) | influence_mask(dy[0][..., 3], (0, 0), (H, W), radius)
    assert np.array_equal(layered[~mask], plate[~mask]) and np.array_equal(single[~mask], plate[~mask])
    touched = (single != plate).any(-1) | (layered != plate).any(-1)
    d = np.abs(single.astype(np.int16) - layered.astype(np.int16))[touched]
    assert touched.sum() > 0.05 * W * H
    assert d.mean() < 2.0, d.mean()
    # a sombra de B sobre A aparece no composto em camadas (A do marco escurece onde B a sombreia)
    sh_on_a = (dy[1].mean(-1) < 0.9) & (ma[0][..., 3] > 0.99)
    assert sh_on_a.any()
    assert layered[sh_on_a].mean() < base.out8[sh_on_a].mean()
