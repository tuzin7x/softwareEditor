"""Controle do orçamento de tempo do render (PT7): função pura, escala do CG e reamostragem."""
import math

import numpy as np
import pytest

from mobilia.render import border as bd
from mobilia.render import budget as bg

SPL = {"final": 128, "marco": 64, "dinamico": 32}


def _plan(n_dyn=150, dyn_px=350_000, marcos=(1_400_000,) * 4):
    return bg.RenderPlan(tuple(marcos), (dyn_px,) * n_dyn, SPL["marco"], SPL["dinamico"])


# GPU de referência (RTX 5070 Laptop, estimativa do PLANO_V2): ~1,4 MP × 128 amostras em ~12 s
GPU = 1.4e6 * 128 / 11.0
GPU_OH = bg.Overhead(0.06, 0.03 / 1e6)


def _check_invariants(st: bg.Settings, plan: bg.RenderPlan):
    lay = st.layers(SPL)
    assert lay["final"] == SPL["final"]                       # nunca reduz o quadro final
    assert lay["dinamico"] <= lay["marco"] <= lay["final"]
    assert bg.DYN_SPP_MIN <= st.spp_dinamico <= plan.spp_dinamico
    assert bg.MARCO_SPP_MIN <= st.spp_marcos <= plan.spp_marcos
    assert bg.SCALE_MIN - 1e-9 <= st.escala_cg <= 1.0


def test_small_photo_fits_without_cuts():
    """Foto do usuário (~0,57 MP): o recorte dinâmico é pequeno e tudo cabe com o perfil."""
    plan = _plan(dyn_px=100_000, marcos=(400_000,) * 4)
    st = bg.choose_settings(GPU, GPU_OH, plan, budget=60.0)
    assert st.cortes == [] and st.cabe
    assert (st.spp_dinamico, st.escala_cg, st.spp_marcos) == (32, 1.0, 64)
    assert st.previsto_s <= 60.0 and st.previsto_s == st.sem_corte_s
    assert st.layers(SPL) == SPL


def test_large_photo_cuts_dynamic_samples_first():
    """1920 × 1440: as amostras dinâmicas caem (entre 12 e o perfil) e nada mais muda."""
    plan = _plan()
    full = bg.forecast(GPU, GPU_OH, plan, 32, 1.0, 64)
    st = bg.choose_settings(GPU, GPU_OH, plan, budget=full * 0.8)
    assert st.cortes == ["spp_dinamico"] and st.cabe
    assert bg.DYN_SPP_MIN <= st.spp_dinamico < 32
    assert st.escala_cg == 1.0 and st.spp_marcos == 64
    assert st.previsto_s <= full * 0.8
    # o maior valor que cabe: uma amostra a mais estoura
    assert bg.forecast(GPU, GPU_OH, plan, st.spp_dinamico + 1, 1.0, 64) > full * 0.8
    _check_invariants(st, plan)


def test_scale_only_after_dynamic_samples_at_minimum():
    plan = _plan()
    t12 = bg.forecast(GPU, GPU_OH, plan, 12, 1.0, 64)
    t08 = bg.forecast(GPU, GPU_OH, plan, 12, 0.8, 64)
    st = bg.choose_settings(GPU, GPU_OH, plan, budget=(t12 + t08) / 2)
    assert st.cortes == ["spp_dinamico", "escala_cg"] and st.cabe
    assert st.spp_dinamico == 12 and bg.SCALE_MIN <= st.escala_cg < 1.0 and st.spp_marcos == 64
    assert st.render_pct == round(st.escala_cg * 100)
    assert st.previsto_s <= (t12 + t08) / 2
    _check_invariants(st, plan)


def test_slow_gpu_cuts_everything_in_order():
    """GPU lenta: dinâmico no mínimo, escala 0,8, marcos cortados até caber (ou até 40)."""
    slow = GPU / 6
    plan = _plan()
    t_min_dyn = bg.forecast(slow, GPU_OH, plan, 12, 0.8, 64)
    t_all_min = bg.forecast(slow, GPU_OH, plan, 12, 0.8, 40)
    st = bg.choose_settings(slow, GPU_OH, plan, budget=(t_min_dyn + t_all_min) / 2)
    assert st.cortes == ["spp_dinamico", "escala_cg", "spp_marcos"] and st.cabe
    assert (st.spp_dinamico, st.escala_cg) == (12, 0.8) and 40 <= st.spp_marcos < 64
    # mais lenta ainda: tudo no mínimo, sem caber (o render segue assim mesmo)
    st2 = bg.choose_settings(slow / 4, GPU_OH, plan, budget=t_all_min)
    assert (st2.spp_dinamico, st2.escala_cg, st2.spp_marcos) == (12, 0.8, 40) and not st2.cabe
    assert st2.layers(SPL)["final"] == 128
    # orçamento já estourado antes do restante (análise lenta): mínimo, nunca erro
    st3 = bg.choose_settings(GPU, GPU_OH, plan, budget=-5.0)
    assert (st3.spp_dinamico, st3.escala_cg, st3.spp_marcos) == (12, 0.8, 40) and not st3.cabe


def test_more_budget_never_gives_worse_settings():
    plan = _plan()
    prev = None
    for b in np.linspace(5, 200, 60):
        st = bg.choose_settings(GPU / 3, GPU_OH, plan, float(b))
        q = (st.spp_marcos, st.escala_cg, st.spp_dinamico)
        if prev is not None:
            assert q[0] >= prev[0] and q[1] >= prev[1] - 1e-9
            if q[:2] == prev[:2]:
                assert q[2] >= prev[2]
        if st.cabe:
            assert st.previsto_s <= b + 1e-6
        prev = q


def test_profile_below_minimums_is_kept():
    """Prévia na CPU (dinâmico 12, marcos 24): não há o que cortar nas amostras; só a escala."""
    plan = bg.RenderPlan((200_000,), (50_000,) * 40, 24, 12)
    st = bg.choose_settings(1e5, 0.5, plan, budget=1.0)
    assert (st.spp_dinamico, st.escala_cg, st.spp_marcos) == (12, 0.8, 24)
    assert st.cortes == ["escala_cg"] and not st.cabe
    assert st.layers({"final": 24, "marco": 24, "dinamico": 12}) == {"final": 24, "marco": 24, "dinamico": 12}


def test_no_dynamic_renders_goes_to_milestones():
    plan = bg.RenderPlan((1_000_000,) * 3, (), 64, 32)
    t40 = bg.forecast(GPU, GPU_OH, plan, 32, 1.0, 40)
    st = bg.choose_settings(GPU, GPU_OH, plan, budget=t40 + 0.01)
    assert st.cortes == ["spp_marcos"]
    assert st.spp_dinamico == 32 and st.escala_cg == 1.0 and 40 <= st.spp_marcos < 64 and st.cabe


def test_measure_from_final_frame_row():
    row = {"camada": "final", "area_px": 1_000_000, "spp": 128, "amostras": 100, "amostragem_s": 8.0,
           "denoise_s": 0.5, "gravacao_s": 0.1, "parede_s": 9.0}
    p = bg.measure(row)
    assert p.throughput == pytest.approx(1e6 * 100 / 8.0)
    assert p.overhead.fixed_s == pytest.approx(0.4) and p.overhead.per_px_s == pytest.approx(0.6 / 1e6)
    assert not p.estimado
    # sem tempo de amostragem: estimativa conservadora pela parede
    p2 = bg.measure({"area_px": 1000, "spp": 10, "parede_s": 1.0})
    assert p2.estimado and p2.throughput == pytest.approx(1000 * 10 / 0.9)
    assert bg.measure({"area_px": 0, "spp": 10, "parede_s": 1.0}) is None


def test_budget_spec_env(monkeypatch):
    monkeypatch.delenv(bg.ENV_ORCAMENTO, raising=False)
    s = bg.budget_spec(1000.0)
    assert s["ativo"] and s["total_s"] == 120.0 and s["t_clique"] == 1000.0
    monkeypatch.setenv(bg.ENV_ORCAMENTO, "0")
    assert not bg.budget_spec(None)["ativo"]
    monkeypatch.setenv(bg.ENV_ORCAMENTO, "90,5")
    assert bg.budget_spec(None)["total_s"] == 90.5


def test_format_duration():
    assert bg.format_duration(100) == "≈ 1 min 40 s"
    assert bg.format_duration(119) == "≈ 2 min"
    assert bg.format_duration(42.4) == "≈ 42 s"


# ---------------------------------------------------------------- escala do CG (border + reamostragem)
@pytest.mark.parametrize("W,H,pct", [(640, 480, 80), (641, 481, 80), (1920, 1440, 87), (481, 641, 85)])
def test_scaled_border_covers_full_border(W, H, pct):
    b = (0.21, 0.13, 0.67, 0.58)
    full_px, _ = bd.pixel_border(b, W, H)
    r = bd.scaled_border(b, W, H, pct)
    sc_px, esc, bb = r
    Ws, Hs = bd.scaled_resolution(W, H, pct)
    assert esc["tamanho"] == [Ws, Hs] and esc["k"] == pytest.approx(W / Ws)
    # o retângulo reduzido é o que o Blender renderiza com esse border
    assert bd.blender_disprect(bb, Ws, Hs) == esc["border"]
    # e cobre (com no máximo ~k px de folga) o retângulo do quadro inteiro
    k = esc["k"]
    assert sc_px[0] <= full_px[0] + 1 and sc_px[2] >= full_px[2] - 1
    assert sc_px[1] <= full_px[1] + 1 and sc_px[3] >= full_px[3] - 1
    assert sc_px[2] - sc_px[0] <= full_px[2] - full_px[0] + 2 * k + 2


def test_resample_crop_identity_and_linear():
    from mobilia.composite.compositor import resample_crop

    rng = np.random.default_rng(0)
    a = rng.random((30, 40, 4)).astype(np.float32)
    esc = {"border": [10, 5, 50, 35], "k": 1.0, "o": [0.0, 0.0]}
    assert np.array_equal(resample_crop(a, esc, [10, 5, 50, 35]), a)
    # campo linear nas coordenadas do quadro: a interpolação o reproduz no interior
    W, H, pct = 641, 481, 80
    b = (0.2, 0.2, 0.7, 0.7)
    full_px, esc, _ = bd.scaled_border(b, W, H, pct)
    bx0, by0, bx1, by1 = esc["border"]
    k, (ox, oy) = esc["k"], esc["o"]
    xs = ox + k * (np.arange(bx0, bx1) + 0.5)                 # centro de cada pixel reduzido, no quadro
    ys = oy + k * (np.arange(by0, by1) + 0.5)
    f = lambda x, y: 0.003 * x + 0.002 * y                     # noqa: E731
    crop = np.repeat(f(xs[None, :], ys[:, None])[..., None], 3, axis=2).astype(np.float32)
    out = resample_crop(crop, esc, full_px)
    X = np.arange(full_px[0], full_px[2]) + 0.5
    Y = np.arange(full_px[1], full_px[3]) + 0.5
    ref = f(X[None, :], Y[:, None])
    inner = (slice(3, -3), slice(3, -3))
    assert out.shape[:2] == ref.shape
    assert np.abs(out[..., 0][inner] - ref[inner]).max() < 1e-4


def test_scaled_layer_composite_preserves_plate(tmp_path):
    """Camada dinâmica reduzida + reamostrada: a placa fica byte a byte fora da máscara."""
    from mobilia.composite.compositor import Compositor, layers_from_passes, resample_crop
    from mobilia.composite.verify import influence_mask, outside_mask_equal

    H, W = 120, 160
    rng = np.random.default_rng(1)
    plate8 = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    rgba_full = np.zeros((H, W, 4), np.float32)
    rgba_full[40:70, 60:100] = (0.3, 0.2, 0.1, 1.0)
    sh_full = np.ones((H, W, 3), np.float32)
    sh_full[70:74, 60:100] = 0.7
    b = (50 / W, 1 - 85 / H, 112 / W, 1 - 30 / H)
    full_px, esc, _ = bd.scaled_border(b, W, H, 80)
    bx0, by0, bx1, by1 = esc["border"]
    k, (ox, oy) = esc["k"], esc["o"]
    xs = np.clip(np.floor(ox + k * (np.arange(bx0, bx1) + 0.5)).astype(int), 0, W - 1)
    ys = np.clip(np.floor(oy + k * (np.arange(by0, by1) + 0.5)).astype(int), 0, H - 1)
    rgba_c, sh_c = rgba_full[ys][:, xs], sh_full[ys][:, xs]
    rgba, sh, origin = layers_from_passes(resample_crop(rgba_c, esc, full_px), resample_crop(sh_c, esc, full_px),
                                          full_px, (H, W))
    comp = Compositor(plate8, seed=7)
    out8, st = comp.composite(1, rgba, sh, None, origin=origin)
    assert st.preserved_exact
    mask = influence_mask(rgba[..., 3], origin, (H, W), comp_radius(W, H))
    ok, n_bad, _ = outside_mask_equal(out8, plate8, mask)
    assert ok and n_bad == 0
    # fora do retângulo coberto pelo recorte reduzido, nada muda
    out_mask = np.ones((H, W), bool)
    out_mask[full_px[1]:full_px[3], full_px[0]:full_px[2]] = False
    assert math.isclose(float(np.abs(out8[out_mask].astype(int) - plate8[out_mask]).max()), 0.0)


def comp_radius(W, H):
    """Raio da máscara independente, como no ``StreamComposer``."""
    from mobilia.composite.compositor import CompositeSettings
    from mobilia.composite.verify import MARGIN_PX

    return max(8.0, CompositeSettings().influence_radius_frac * math.hypot(W, H)) + MARGIN_PX


def test_dynamic_jobs_follows_worker_reuse_rule():
    from mobilia.render import layers as lay

    fr = lay.LayerFrame
    sig_a, sig_b, sig_t = ("m1", "a"), ("m1", "b"), ("m0", "t")
    frames = [fr(1, "foto"), fr(2, "dinamico", "m1", frozenset({"p"}), sig_a),
              fr(3, "dinamico", "m1", frozenset({"p"}), sig_a),          # mesmo render: reaproveitado
              fr(4, "dinamico", "m1", frozenset({"p"}), sig_b),
              fr(5, "foto"), fr(6, "dinamico", "m1", frozenset({"p"}), sig_b),   # depois da foto: de novo
              fr(7, "estatico", "m2", frozenset(), ("estatico", "m2"), ("m1", frozenset({"q"}), sig_t, 0.75)),
              fr(8, "estatico", "m2", frozenset(), ("estatico", "m2"), ("m1", frozenset({"q"}), sig_t, 0.5))]
    plan = lay.LayerPlan([], frames)
    jobs = lay.dynamic_jobs(plan)
    assert [(j[0], j[2]) for j in jobs] == [(2, sig_a), (4, sig_b), (6, sig_b), (7, sig_t)]
    assert [j[0] for j in lay.dynamic_jobs(plan, {4, 7})] == [4, 7]
