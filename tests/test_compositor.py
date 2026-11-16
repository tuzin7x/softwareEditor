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
