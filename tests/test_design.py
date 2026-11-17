"""Redesenho de ambientes (``mobilia/design``), rápidos: o SDXL é trocado por um gerador falso.

Garantias: presets completos (funções, 12+ estilos, ambientes) e prompts coerentes; força efetiva dos poucos passos
do Lightning perto da pedida; trava da arquitetura (janelas/portas originais por cima, rampa sem halo); "Editar
área" pixel-idêntico fora da máscara; escolha entre candidatos; contrato da CLI ``design`` (eventos JSON)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mobilia.analysis.segmentation import GROUP_IDS
from mobilia.design import api, archlock as AL, presets as P, quality as Q
from mobilia.design.engine import _chunk_ids, plan_steps

WIN, DOOR, FLOOR, WALL, CEIL, FURN = (GROUP_IDS[k] for k in ("window", "door", "floor", "wall", "ceiling",
                                                             "furniture"))


# ============================================================================ apoio
class FakeEngine:
    """Gerador falso: devolve a imagem de entrada com cor trocada dentro da máscara (e algo fora, para provar que a
    trava e a composição é que garantem os pixels originais)."""

    device = "cpu"

    def __init__(self):
        self.calls = []
        self.load_s = 0.0

    def embeddings(self, prompts):
        self.prompts = list(prompts)
        return [None] * len(prompts)

    def generate(self, image, mask, control, prompt, seed, strength, steps, guidance, control_scale, control_end,
                 callback=None):
        self.calls.append(dict(shape=image.shape, mask=mask.copy(), control=None if control is None else
                               control.copy(), prompt=prompt, seed=seed, strength=strength, steps=steps,
                               guidance=guidance, control_scale=control_scale, control_end=control_end))
        for k in range(1, 4):
            if callback:
                callback(k, 3)
        out = image.astype(np.int16)
        out = (255 - out) // 2 + (seed % 7) * 3          # tudo muda (fora da máscara também)
        out[mask] = np.array([40 + seed % 50, 120, 200])
        return np.clip(out, 0, 255).astype(np.uint8)


def _room(W=320, H=240, furnished=True):
    """Foto sintética: parede, piso, teto e uma janela na parede (com 'vista' texturizada)."""
    rng = np.random.default_rng(0)
    rgb = np.zeros((H, W, 3), np.uint8)
    g = np.full((H, W), WALL, np.uint8)
    rgb[:] = (200, 190, 180)
    g[: H // 6] = CEIL
    rgb[: H // 6] = (235, 235, 235)
    g[int(H * 0.65):] = FLOOR
    rgb[int(H * 0.65):] = (150, 110, 80)
    y0, y1, x0, x1 = H // 4, H // 2 + 10, W // 2, W // 2 + W // 4
    g[y0:y1, x0:x1] = WIN
    rgb[y0:y1, x0:x1] = rng.integers(60, 255, (y1 - y0, x1 - x0, 3), dtype=np.uint8)
    if furnished:            # um sofá encostado na parede: o ambiente não é "vazio"
        g[int(H * 0.5):int(H * 0.8), int(W * 0.05):int(W * 0.4)] = FURN
        rgb[int(H * 0.5):int(H * 0.8), int(W * 0.05):int(W * 0.4)] = (90, 90, 100)
    depth = np.tile(np.linspace(6.0, 2.0, H, dtype=np.float32)[:, None], (1, W))   # piso embaixo = perto
    return rgb, g, depth, (y0, y1, x0, x1)


@pytest.fixture()
def room():
    return _room()


class FakeSegmenter:
    """OneFormer falso da conferência pós-geração (nota/reparo): sem janela nova nenhuma."""

    def __init__(self, device="cpu", long_side=512):
        pass

    def groups(self, rgb):
        return np.full(rgb.shape[:2], WALL, np.uint8)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _no_real_segmenter(monkeypatch):
    monkeypatch.setattr(Q, "Segmenter", FakeSegmenter)


# ============================================================================ presets
def test_catalog_has_all_functions_styles_and_rooms():
    c = P.catalog()
    assert set(c["funcoes"]) == {"mudar_estilo", "reforma", "redesign_simples", "mobiliar", "editar_area",
                                 "fachada", "evento", "texto"}
    assert len(c["estilos"]) >= 12
    for k in ("moderno", "contemporaneo", "escandinavo", "minimalista", "industrial", "classico", "luxo", "boho",
              "japandi", "rustico", "costeiro", "tropical"):
        assert k in c["estilos"]
    assert {"sala", "quarto", "cozinha", "banheiro", "escritorio", "sala_jantar", "varanda", "fachada"} <= set(
        c["ambientes"])
    assert {"aniversario", "casamento", "natal"} <= set(c["eventos"])
    # rótulos em português, sem chaves cruas
    assert c["funcoes"]["mobiliar"] == "Mobiliar ambiente vazio"
    assert c["estilos"]["tropical"] == "Tropical brasileiro"


def test_function_strengths_are_ordered():
    f = P.FUNCTIONS
    assert f["redesign_simples"].strength < f["mudar_estilo"].strength < f["reforma"].strength
    assert 0.35 <= f["redesign_simples"].strength <= 0.5
    assert 0.55 <= f["mudar_estilo"].strength <= 0.85
    assert f["reforma"].strength >= 0.8 and f["reforma"].control_scale < f["mudar_estilo"].control_scale
    assert set(f["mobiliar"].lock) >= {"window", "door"} and f["mobiliar"].control_scale < 0.5
    assert f["mudar_estilo"].lock == ("window", "door") and f["editar_area"].lock == ()


@pytest.mark.parametrize("funcao", ["mudar_estilo", "reforma", "redesign_simples", "mobiliar", "texto"])
def test_prompts_for_every_style_and_room(funcao):
    for st in P.STYLES:
        for rk in ("sala", "quarto", "cozinha", "banheiro", "escritorio", "sala_jantar", "varanda"):
            p = P.build_prompt(funcao, st, rk, variation=2)
            assert P.ROOMS[rk]["en"] in p.positive
            assert P.ROOMS[rk]["furniture"].split(",")[0] in p.positive        # o conjunto de móveis vai junto
            assert P.STYLES[st].palettes[2 % len(P.STYLES[st].palettes)] in p.positive
            assert "extra window" in p.negative and "cgi" in p.negative
            assert P.NEGATIVE_EMPTY in p.negative
            assert "photorealistic" in p.positive


def test_style_adjective_leads_and_variations_change_palette():
    p0 = P.build_prompt("mudar_estilo", "escandinavo", "quarto", 0)
    p1 = P.build_prompt("mudar_estilo", "escandinavo", "quarto", 1)
    assert p0.positive.startswith("photo of a scandinavian bedroom")
    assert p0.positive != p1.positive and p0.key() != p1.key()


def test_free_text_goes_first_and_events_and_facade():
    p = P.build_prompt("reforma", "luxo", "sala", 0, extra="  sofá   verde  ")
    assert p.positive.startswith("sofá verde, ")
    e = P.build_prompt("evento", None, "sala", 0, evento="natal")
    assert "christmas tree" in e.positive and P.NEGATIVE_EMPTY not in e.negative
    f = P.build_prompt("fachada", "tropical", "sala", 0)
    assert "facade" in f.positive and "interior" in f.negative
    m = P.build_prompt("mobiliar", "moderno", "sala", 0)
    assert "fully furnished" in m.positive and "fireplace" in m.negative


def test_aliases_and_invalid_keys():
    assert P.function("Reforma completa").key == "reforma"
    assert P.style("Contemporâneo").key == "contemporaneo"
    assert P.norm_room("sala de jantar") == "sala_jantar"
    with pytest.raises(ValueError, match="Estilo inválido"):
        P.style("gótico")
    with pytest.raises(ValueError, match="Função inválid"):
        P.function("voar")


@pytest.mark.parametrize("w,h,side", [(960, 640, 1024), (1920, 1440, 768), (1080, 1920, 1024), (3000, 800, 1024),
                                      (640, 640, 768)])
def test_work_size_multiples_of_64_keep_aspect(w, h, side):
    W, H = P.work_size(w, h, side)
    assert W % 64 == 0 and H % 64 == 0
    assert abs(W / H - w / h) / (w / h) < 0.12
    assert max(W, H) <= side * 1.4 and W * H >= 0.25 * side * side


def test_quality_profiles():
    assert P.quality_profile("final", "cuda").long_side == 1024
    assert P.quality_profile("rapido", "cuda").steps < P.quality_profile("final", "cuda").steps
    cpu = P.quality_profile("final", "cpu")
    assert cpu.long_side == 768 and cpu.steps <= 6


# ============================================================================ passos do Lightning
@pytest.mark.parametrize("steps", [5, 6, 8])
@pytest.mark.parametrize("s", [0.4, 0.45, 0.6, 0.68, 0.72, 0.8, 0.85, 0.9, 1.0])
def test_plan_steps_effective_strength_close(steps, s):
    tot, s_eff, n = plan_steps(steps, s)
    assert abs(s_eff - s) <= 0.07 or s >= 0.97
    assert int(tot * min(1.0, s_eff + 1e-4)) == n          # o que o diffusers roda de fato
    assert n >= 2 and tot >= n


def test_plan_steps_distinguishes_strengths_that_used_to_collide():
    # com 6 passos o diffusers começava 0,68 / 0,72 / 0,82 no mesmo ruído (0,67)
    effs = {round(plan_steps(6, s)[1], 3) for s in (0.68, 0.72, 0.82)}
    assert len(effs) == 3


# ============================================================================ prompts longos
class _Tok:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __call__(self, text, truncation=False, add_special_tokens=False):
        return {"input_ids": [10 + (len(w) % 50) for w in text.split()]}


def test_chunk_ids_blocks_of_77_with_bos_eos():
    ch = _chunk_ids(_Tok(), " ".join(["word"] * 160))
    assert len(ch) == 3 and all(len(c) == 77 for c in ch)
    assert ch[0][0] == 1 and ch[0][76] == 2 and ch[2][0] == 1
    assert ch[2][1 + 10] == 2 and ch[2][-1] == 0                 # 160 = 75 + 75 + 10
    assert len(_chunk_ids(_Tok(), " ".join(["w"] * 1000))) == 3   # teto de blocos
    assert _chunk_ids(_Tok(), "")[0][:3] == [1, 2, 0]


# ============================================================================ trava / máscaras
def test_feather_alpha_inside_ramp_never_leaks_outside():
    m = np.zeros((100, 100), bool)
    m[30:70, 30:70] = True
    a = AL.feather_alpha(m, 8, inside=1.0)
    assert np.all(a[~m] == 0)
    assert a[50, 50] == 1.0 and 0 < a[31, 50] < 1
    b = AL.feather_alpha(m, 8, inside=0.0)                # rampa toda por fora
    assert np.all(b[m] == 1.0) and b[25, 50] > 0 and np.all(b[:20] == 0)
    c = AL.feather_alpha(m, 8, inside=0.5)
    assert np.all(np.diff(c[50, :50]) >= -1e-6)          # monotônica até o centro
    assert np.all(AL.feather_alpha(np.zeros((5, 5), bool), 3) == 0)


def test_composite_exact_at_alpha_extremes():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 256, (2, 40, 40, 3), dtype=np.uint8)
    al = np.zeros((40, 40), np.float32)
    al[10:20] = 1
    al[20:25] = 0.5
    out = AL.composite(a, b, al)
    assert np.array_equal(out[:10], a[:10]) and np.array_equal(out[10:20], b[10:20])
    assert np.array_equal(out[25:], a[25:])


def test_lock_openings_restores_window_and_leaves_rest(room):
    rgb, g, _, (y0, y1, x0, x1) = room
    gen = np.full_like(rgb, 30)
    lock = AL.lock_mask(g, ("window", "door"))
    out, info = AL.lock_openings(gen, rgb, lock, color_match=False)
    assert info["componentes"] == 1
    assert np.array_equal(out[y0 + 6:y1 - 6, x0 + 6:x1 - 6], rgb[y0 + 6:y1 - 6, x0 + 6:x1 - 6])
    far = np.ones(lock.shape, bool)
    far[y0 - 10:y1 + 10, x0 - 10:x1 + 10] = False
    assert np.array_equal(out[far], gen[far])


def test_lock_color_match_is_bounded(room):
    rgb, g, _, (y0, y1, x0, x1) = room
    gen = rgb.copy()
    gen[y0:y1, x0:x1] = np.clip(rgb[y0:y1, x0:x1].astype(int) - 80, 0, 255)     # gerador escureceu a janela
    out, _ = AL.lock_openings(gen, rgb, AL.lock_mask(g), color_match=True)
    core = (slice(y0 + 8, y1 - 8), slice(x0 + 8, x1 - 8))
    d = rgb[core].astype(float).mean() - out[core].astype(float).mean()
    assert 1 < d < 45                       # escureceu um pouco, sem virar a imagem gerada
    assert np.corrcoef(out[core].ravel(), rgb[core].ravel())[0, 1] > 0.95   # mesma vista


def test_generation_mask_and_clean_mask():
    lk = np.zeros((200, 200), bool)
    lk[50:150, 50:150] = True
    lk[5, 5] = True                                       # ruído de segmentação
    clean = AL.clean_mask(lk)
    assert not clean[5, 5] and clean[100, 100]
    gm = AL.generation_mask(clean)
    assert not gm[100, 100] and gm[50, 100] and gm[0, 0]  # miolo fixo, contorno refeito
    assert AL.generation_mask(np.zeros((10, 10), bool)).all()


def test_user_mask_formats_and_bbox():
    gray = np.zeros((60, 80), np.uint8)
    gray[10:30, 20:50] = 255
    m = AL.user_mask(gray)
    assert m.sum() == 20 * 30
    rgba = np.zeros((60, 80, 4), np.uint8)
    rgba[10:30, 20:50] = (255, 255, 255, 255)
    assert np.array_equal(AL.user_mask(rgba), m)
    alpha_only = np.zeros((60, 80, 4), np.uint8)
    alpha_only[10:30, 20:50, 3] = 255
    assert np.array_equal(AL.user_mask(alpha_only), m)
    assert AL.user_mask(gray, (160, 120)).shape == (120, 160)
    x0, y0, x1, y1 = AL.mask_bbox(m)
    assert x0 <= 20 and y0 <= 10 and x1 >= 50 and y1 >= 30 and x1 <= 80 and y1 <= 60


# ============================================================================ redesign (gerador falso)
def test_redesign_locks_openings_and_uses_preset(room, tmp_path):
    rgb, g, depth, (y0, y1, x0, x1) = room
    eng = FakeEngine()
    events = []
    res = api.redesign(rgb, "mudar_estilo", "escandinavo", "sala", n=3, seed=5, out_dir=tmp_path, engine=eng,
                       depth=depth, groups=g, progress=events.append, long_side=320)
    assert len(res.outputs) == 3 and all(Path(p).is_file() for p in res.outputs)
    assert Path(res.before).is_file() and Path(res.report).is_file()
    W, H = res.size
    assert W % 64 == 0 and H % 64 == 0
    c = eng.calls[0]
    f = P.FUNCTIONS["mudar_estilo"]
    assert c["strength"] == f.strength and c["control_scale"] == f.control_scale
    assert c["control"] is not None and c["control"].shape == (H, W)
    assert c["control"][-1].mean() > c["control"][0].mean()      # perto (embaixo) = claro
    # o gerador não refaz o miolo da janela; o resultado traz a janela original
    sx, sy = W / rgb.shape[1], H / rgb.shape[0]
    cy, cx = int((y0 + y1) / 2 * sy), int((x0 + x1) / 2 * sx)
    assert not c["mask"][cy, cx] and c["mask"][2, 2]
    base = np.asarray(Image.open(res.before))
    img = res.images[0]
    core = (slice(int(y0 * sy) + 6, int(y1 * sy) - 6), slice(int(x0 * sx) + 6, int(x1 * sx) - 6))
    assert np.abs(img[core].astype(int) - base[core].astype(int)).mean() < 25     # só o casamento de cor
    assert len({cl["seed"] for cl in eng.calls}) == 3
    assert len({cl["prompt"].positive for cl in eng.calls}) == 3              # paletas diferentes
    # eventos
    prog = [e["fraction"] for e in events if e["event"] == "progress"]
    assert prog == sorted(prog) and 0 <= prog[0] and prog[-1] <= 1
    assert [e["index"] for e in events if e["event"] == "preview"] == [0, 1, 2]
    done = res.to_event()
    assert done["event"] == "done" and done["outputs"] == res.outputs and done["output"] == res.outputs[0]
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    assert rep["funcao"] == "mudar_estilo" and rep["trava"]["grupos"] == ["window", "door"]


def test_redesign_staging_masks_and_keep_override(room, tmp_path):
    rgb, g, depth, (y0, y1, x0, x1) = room
    eng = FakeEngine()
    res = api.redesign(rgb, "mobiliar", "moderno", "quarto", n=1, seed=1, out_dir=tmp_path / "m", engine=eng,
                       depth=depth, groups=g, long_side=320)
    mask = eng.calls[0]["mask"]
    H, W = mask.shape
    sy, sx = H / rgb.shape[0], W / rgb.shape[1]
    assert not mask[int((y0 + y1) / 2 * sy), int((x0 + x1) / 2 * sx)]      # miolo da janela fica
    assert mask[int(H * 0.9), 10] and mask[2, W // 2]                        # piso e teto são gerados
    assert eng.calls[0]["strength"] == 1.0
    assert "fully furnished" in eng.calls[0]["prompt"].positive and "bedroom" in eng.calls[0]["prompt"].positive
    assert res.funcao == "mobiliar"
    # ``keep``: o teto não é refeito pelo gerador, mas também não é colado de volta pixel a pixel
    eng2 = FakeEngine()
    res2 = api.redesign(rgb, "mobiliar", "moderno", "quarto", n=1, seed=1, out_dir=tmp_path / "k", engine=eng2,
                        depth=depth, groups=g, long_side=320, overrides={"keep": ("ceiling",)})
    assert not eng2.calls[0]["mask"][2, W // 2]
    base = np.asarray(Image.open(res2.before))
    assert np.abs(res2.images[0][2].astype(int) - base[2].astype(int)).mean() > 20


def test_redesign_without_lock_skips_segmentation(room, tmp_path, monkeypatch):
    rgb, _, depth, _ = room
    monkeypatch.setattr(api, "estimate_groups", lambda *a, **k: pytest.fail("não devia segmentar"))
    eng = FakeEngine()
    api.redesign(rgb, "fachada", "moderno", n=1, seed=2, out_dir=tmp_path, engine=eng, depth=depth, long_side=256)
    assert eng.calls[0]["mask"].all()


def test_redesign_rejects_wrong_entry_points(room, tmp_path):
    rgb, g, depth, _ = room
    with pytest.raises(ValueError):
        api.redesign(rgb, "editar_area", "moderno", engine=FakeEngine(), depth=depth, groups=g, out_dir=tmp_path)
    with pytest.raises(ValueError, match="estilo"):
        api.redesign(rgb, "mudar_estilo", None, engine=FakeEngine(), depth=depth, groups=g, out_dir=tmp_path)


def test_redesign_candidates_keep_best(room, tmp_path):
    rgb, g, depth, _ = room
    eng = FakeEngine()
    # nota falsa: a "mobília" é proporcional ao canal vermelho médio da imagem (a semente muda a cor)
    def fake_groups(img):
        out = np.full(img.shape[:2], WALL, np.uint8)
        k = int(img[..., 0].mean()) % 100
        out[: max(1, k * img.shape[0] // 100)] = FURN
        return out

    res = api.redesign(rgb, "mobiliar", "boho", n=2, candidates=4, seed=3, out_dir=tmp_path, engine=eng,
                       depth=depth, groups=g, long_side=256, rank_fn=fake_groups)
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    notas = [n["nota"] for n in rep["candidatos"]["notas"]]
    assert len(eng.calls) == 4 and len(res.outputs) == 2
    assert sorted(notas, reverse=True)[:2] == [notas[k] for k in rep["candidatos"]["escolhidos"]]
    assert all(Path(p).parent == tmp_path for p in res.outputs)


def test_quality_score_rewards_furniture_and_punishes_new_windows():
    g = np.full((100, 100), WALL, np.uint8)
    lock = np.zeros((100, 100), bool)
    lock[10:30, 10:30] = True
    g[10:30, 10:30] = WIN
    sparse = Q.score(g, lock)
    full = g.copy()
    full[60:100, 20:80] = FURN
    s_full = Q.score(full, lock)
    assert s_full["nota"] > sparse["nota"] and s_full["mobiliado"] > 0.2
    extra = full.copy()
    extra[40:60, 70:95] = WIN                                       # janela inventada
    s_extra = Q.score(extra, lock)
    assert s_extra["aberturas_novas"] > 0 and s_extra["nota"] < s_full["nota"]


# ============================================================================ editar área / texto
def test_edit_area_pixel_identical_outside_mask(room, tmp_path):
    rgb, _, depth, _ = room
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[150:220, 40:140] = 255
    mpath = tmp_path / "m.png"
    Image.fromarray(mask).save(mpath)
    eng = FakeEngine()
    res = api.edit_area(rgb, str(mpath), prompt="round wooden coffee table", n=2, seed=4, out_dir=tmp_path / "o",
                        engine=eng, depth=depth)
    assert res.size == (rgb.shape[1], rgb.shape[0])
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    g = rep["margem_px"]
    assert 0 < g <= 0.012 * np.hypot(*rgb.shape[:2]) + 1
    import cv2

    outside = cv2.dilate((mask > 0).astype(np.uint8), AL._disk(g)) == 0
    for img in res.images:
        assert img.shape == rgb.shape
        assert np.array_equal(img[outside], rgb[outside])
        assert np.abs(img[170:200, 60:120].astype(int) - rgb[170:200, 60:120].astype(int)).mean() > 20
    c = eng.calls[0]
    assert "round wooden coffee table" in c["prompt"].positive and c["strength"] >= 0.97
    assert c["shape"][0] % 64 == 0 and c["shape"][1] % 64 == 0
    with pytest.raises(ValueError, match="vazia"):
        api.edit_area(rgb, np.zeros(rgb.shape[:2], np.uint8), "x", engine=eng, depth=depth, out_dir=tmp_path)
    with pytest.raises(ValueError, match="Descreva"):
        api.edit_area(rgb, mask, "", engine=eng, depth=depth, out_dir=tmp_path)


def test_text_to_design(tmp_path):
    eng = FakeEngine()
    res = api.text_to_design("sala aconchegante com lareira", "japandi", "sala", n=2, seed=9, out_dir=tmp_path,
                             engine=eng, aspect="9:16")
    W, H = res.size
    assert H > W and W % 64 == 0 and len(res.outputs) == 2
    c = eng.calls[0]
    assert c["strength"] == 1.0 and c["control"] is None and c["mask"].all()
    assert c["prompt"].positive.startswith("sala aconchegante com lareira")


def test_depth_control_near_is_bright_and_invalid_is_far():
    z = np.tile(np.linspace(1, 5, 50, dtype=np.float32)[:, None], (1, 40))
    z[:5] = 0
    c = api.depth_control(z, (64, 80))
    assert c.shape == (80, 64) and c.min() >= 0 and c.max() <= 1
    assert c[2].mean() < 0.05 and c[10].mean() > c[60].mean()


# ============================================================================ CLI
def _cli(monkeypatch, capsys, args):
    from mobilia import cli

    eng = FakeEngine()
    monkeypatch.setattr(api, "get_engine", lambda *a, **k: eng)
    monkeypatch.setattr(api, "estimate_depth", lambda rgb, dev, **k: np.tile(
        np.linspace(6, 2, rgb.shape[0], dtype=np.float32)[:, None], (1, rgb.shape[1])))
    monkeypatch.setattr(api, "estimate_groups", lambda rgb, dev, **k: _room(rgb.shape[1], rgb.shape[0])[1])
    code = cli.main(args)
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip().startswith("{")]
    return code, lines, eng


def test_cli_design_json_contract(monkeypatch, capsys, tmp_path):
    rgb, _, _, _ = _room(400, 300)
    photo = tmp_path / "foto.jpg"
    Image.fromarray(rgb).save(photo)
    code, ev, eng = _cli(monkeypatch, capsys, ["design", "--photo", str(photo), "--funcao", "mudar_estilo",
                                                "--estilo", "industrial", "--ambiente", "escritorio", "--n", "2",
                                                "--seed", "7", "--out-dir", str(tmp_path / "out"), "--json",
                                                "--quality", "rapido"])
    assert code == 0
    kinds = [e["event"] for e in ev]
    assert kinds[-1] == "done" and kinds.count("preview") == 2 and "progress" in kinds
    for e in ev:
        if e["event"] == "progress":
            assert {"stage", "fraction", "message"} <= set(e)
    done = ev[-1]
    assert len(done["outputs"]) == 2 and all(Path(p).is_file() for p in done["outputs"])
    assert done["funcao"] == "mudar_estilo" and done["estilo"] == "industrial" and done["ambiente"] == "escritorio"
    assert done["seeds"][0] == 7 and Path(done["before"]).is_file() and Path(done["report"]).is_file()
    assert "home office" in eng.calls[0]["prompt"].positive


def test_cli_design_edit_text_and_errors(monkeypatch, capsys, tmp_path):
    rgb, _, _, _ = _room(400, 300)
    photo = tmp_path / "foto.jpg"
    Image.fromarray(rgb).save(photo)
    mask = np.zeros((300, 400), np.uint8)
    mask[200:280, 50:200] = 255
    Image.fromarray(mask).save(tmp_path / "m.png")
    code, ev, _ = _cli(monkeypatch, capsys, ["design", "--photo", str(photo), "--funcao", "editar_area", "--mask",
                                             str(tmp_path / "m.png"), "--prompt", "tapete persa", "--n", "1",
                                             "--out-dir", str(tmp_path / "e"), "--json"])
    assert code == 0 and ev[-1]["event"] == "done" and ev[-1]["funcao"] == "editar_area"
    code, ev, _ = _cli(monkeypatch, capsys, ["design", "--funcao", "texto", "--estilo", "boho", "--n", "1",
                                             "--out-dir", str(tmp_path / "t"), "--json"])
    assert code == 0 and ev[-1]["event"] == "done"
    code, ev, _ = _cli(monkeypatch, capsys, ["design", "--photo", str(tmp_path / "nao_existe.jpg"), "--json"])
    assert code == 1 and ev[-1]["event"] == "error" and "não encontrada" in ev[-1]["message"]
    code, ev, _ = _cli(monkeypatch, capsys, ["design", "--photo", str(photo), "--funcao", "editar_area", "--json"])
    assert code == 1 and "Pinte" in ev[-1]["message"]
    code, ev, _ = _cli(monkeypatch, capsys, ["design", "--photo", str(photo), "--estilo", "gotico", "--json"])
    assert code == 1 and "Estilo inválido" in ev[-1]["message"]
    code, ev, eng = _cli(monkeypatch, capsys, ["design", "--photo", str(photo), "--funcao", "evento", "--evento",
                                               "natal", "--n", "1", "--out-dir", str(tmp_path / "v"), "--json"])
    assert code == 0 and "christmas" in eng.calls[0]["prompt"].positive and ev[-1]["estilo"] == "natal"


# ============================================================================ pesos e ampliador
def test_design_weights_in_prefetch(monkeypatch):
    from mobilia.analysis import weights as Wt

    assert Wt.ESRGAN_WEIGHT in Wt.DESIGN_WEIGHTS
    assert {w.key for w in Wt.SDXL_WEIGHTS} <= {w.key for w in Wt.DESIGN_WEIGHTS}
    assert Wt.ESRGAN_WEIGHT.licenca == "BSD-3-Clause" and len(Wt.ESRGAN_WEIGHT.sha256) == 64
    got = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: "/x")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "/x")
    monkeypatch.setattr(Wt, "ensure_weight", lambda w, offline=False: got.append(w.key))
    Wt.prefetch_models()
    assert {"sdxl_gerador", "sdxl_texto", "sdxl_vae", "sdxl_controlnet_profundidade", "realesrgan_x4plus"} <= set(got)
    assert len(got) == len(set(got))
    got.clear()
    Wt.prefetch_models(design=False)
    assert "realesrgan_x4plus" not in got


def test_upscaler_tiles_without_seams():
    torch = pytest.importorskip("torch")
    from torch.nn import functional as F

    from mobilia.design.upscale import Upscaler, _build

    up = object.__new__(Upscaler)
    up.device, up.dtype, up.tile = "cpu", torch.float32, 32
    up.net = lambda x: F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (70, 90, 3), dtype=np.uint8)
    out = up(img, pad=8)
    ref = up.net(torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255)
    ref = (ref[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
    assert out.shape == (280, 360, 3)
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1
    # a arquitetura bate com as chaves do arquivo oficial (params_ema do RRDBNet x4plus)
    keys = set(_build().state_dict())
    assert {"conv_first.weight", "body.22.rdb3.conv5.bias", "conv_up2.weight", "conv_last.bias"} <= keys


def test_cli_ampliar_contract(monkeypatch, capsys, tmp_path):
    from mobilia import cli
    from mobilia.design import upscale as U

    img = tmp_path / "v.jpg"
    Image.fromarray(np.full((64, 96, 3), 120, np.uint8)).save(img)

    def fake(image, out_path=None, long_side=3840, device="auto", progress=None, **k):
        progress(1.0)
        Image.fromarray(np.zeros((256, 384, 3), np.uint8)).save(out_path)
        return None, {"para": [384, 256], "s": 0.1}

    monkeypatch.setattr(U, "upscale", fake)
    code = cli.main(["ampliar", "--image", str(img), "--out", str(tmp_path / "o.jpg"), "--json"])
    ev = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert code == 0 and ev[0]["event"] == "progress" and ev[-1] == {"event": "done", "output": str(
        tmp_path / "o.jpg"), "size": [384, 256], "s": 0.1}
    code = cli.main(["ampliar", "--image", str(tmp_path / "x.jpg"), "--out", str(tmp_path / "o.jpg"), "--json"])
    assert code == 1 and json.loads(capsys.readouterr().out.splitlines()[-1])["event"] == "error"


# ============================================================================ rodada 2: correções gerais
def test_empty_room_switches_style_change_to_staging(tmp_path):
    rgb, g, depth, _ = _room(furnished=False)
    assert api.furniture_coverage(g) < P.EMPTY_FURNITURE_FRAC
    for funcao in ("mudar_estilo", "redesign_simples", "reforma"):
        eng = FakeEngine()
        events = []
        res = api.redesign(rgb, funcao, "costeiro", "sala", n=1, seed=1, out_dir=tmp_path / funcao, engine=eng,
                           depth=depth, groups=g, long_side=256, progress=events.append)
        c = eng.calls[0]
        mob = P.FUNCTIONS["mobiliar"]
        assert res.funcao == "mobiliar" and c["strength"] == mob.strength
        assert "fully furnished coastal living room" in c["prompt"].positive
        done = res.to_event()
        assert done["funcao_pedida"] == funcao and "vazio" in done["aviso"]
        assert any("vazio" in e.get("message", "") for e in events)
    # com móveis, nada muda; e ``auto_empty=False`` respeita a função pedida
    rgb2, g2, depth2, _ = _room()
    res = api.redesign(rgb2, "mudar_estilo", "costeiro", n=1, seed=1, out_dir=tmp_path / "x", engine=FakeEngine(),
                       depth=depth2, groups=g2, long_side=256)
    assert res.funcao == "mudar_estilo" and "aviso" not in res.to_event()
    res = api.redesign(rgb, "mudar_estilo", "costeiro", n=1, seed=1, out_dir=tmp_path / "y", engine=FakeEngine(),
                       depth=depth, groups=g, long_side=256, auto_empty=False)
    assert res.funcao == "mudar_estilo"


def test_prompts_curtains_only_with_windows_and_palette_yields_to_user_text():
    p = P.build_prompt("reforma", "classico", "sala", 0, windows=False)
    assert "curtain" not in p.positive and "soft natural light" in p.positive
    assert "window on a plain wall" in p.negative
    q = P.build_prompt("reforma", "classico", "sala", 0, windows=True)
    assert "curtains framing the existing windows" in q.positive and "existing windows" in q.positive
    e = P.build_prompt("editar_area", "mid_century", "sala", 1, extra="poltrona de couro caramelo")
    assert not any(pal in e.positive for pal in P.STYLES["mid_century"].palettes)
    assert e.positive.startswith("poltrona de couro caramelo") and "mid-century" in e.positive
    b = P.build_prompt("mudar_estilo", "costeiro", "banheiro", 0)
    assert "shiplap" in b.positive and "sofa" not in b.positive.split("a floating vanity")[0]
    k = P.build_prompt("mudar_estilo", "tropical", "sala", 0, open_kitchen=True)
    assert P.OPEN_KITCHEN in k.positive
    assert "letters" in p.negative


def test_bright_curtain_becomes_lock_and_dark_one_does_not():
    rgb = np.full((100, 100, 3), 120, np.uint8)
    cov = np.zeros((100, 100), bool)
    cov[10:60, 10:30] = True
    rgb[10:60, 10:30] = 250                    # cortina com luz atravessando
    cov[10:60, 60:80] = True
    rgb[10:60, 60:80] = 60                     # cortina escura
    m = AL.bright_coverings(cov, rgb)
    assert m[30, 20] and not m[30, 70]


def test_doorway_view_is_locked_next_to_the_door_leaf():
    H, W = 120, 160
    z = np.full((H, W), 3.0, np.float32)       # parede a 3 m
    door = np.zeros((H, W), bool)
    door[30:110, 60:75] = True                 # folha da porta
    z[30:110, 75:100] = 7.0                    # o cômodo do lado, visto pelo vão
    z[10:20, 130:150] = 9.0                    # algo fundo longe da porta (não entra)
    v = AL.doorway_views(door, z)
    assert v[70, 85] and not v[15, 140] and not v[70, 20]


def test_big_door_and_bathroom_doors_lock_only_their_frame():
    lock = np.zeros((100, 100), bool)
    door = np.zeros((100, 100), bool)
    door[10:90, 20:70] = True                  # 40% da imagem: box de vidro
    lock |= door
    capped, n = AL.cap_doors(lock, door)
    assert n == 1 and not capped[50, 45] and capped[10, 45] and capped[50, 20]
    small = np.zeros((100, 100), bool)
    small[40:70, 80:90] = True
    capped, n = AL.cap_doors(small.copy(), small)
    assert n == 0 and capped[55, 85]
    capped, n = AL.cap_doors(small.copy(), small, all_frames=True)
    assert n == 1 and not capped[55, 85]


def test_new_openings_ignore_existing_ones_and_curtains_beside_them():
    orig = np.zeros((100, 100), bool)
    orig[20:50, 20:40] = True
    gen = np.full((100, 100), WALL, np.uint8)
    gen[20:50, 20:40] = WIN                    # a janela original
    gen[20:50, 42:47] = GROUP_IDS["window_covering"]   # cortina ao lado dela
    assert not AL.new_openings(gen, orig, (100, 100)).any()
    gen[30:60, 70:95] = WIN                    # janela inventada numa parede lisa
    new = AL.new_openings(gen, orig, (100, 100))
    assert new[45, 80] and not new[30, 30]


def test_redesign_repairs_invented_window(tmp_path):
    rgb, g, depth, _ = _room()
    eng = FakeEngine()

    def fake_groups(img):
        out = np.full(img.shape[:2], WALL, np.uint8)
        h, w = out.shape
        out[int(h * 0.2):int(h * 0.45), int(w * 0.05):int(w * 0.3)] = WIN     # parede lisa à esquerda
        return out

    res = api.redesign(rgb, "reforma", "luxo", n=1, seed=2, out_dir=tmp_path, engine=eng, depth=depth, groups=g,
                       long_side=256, rank_fn=fake_groups)
    assert len(eng.calls) == 2                               # versão + reparo
    rep_call = eng.calls[1]
    assert "plain painted wall" in rep_call["prompt"].positive and "window" in rep_call["prompt"].negative
    H, W = rep_call["mask"].shape
    assert rep_call["mask"][int(H * 0.3), int(W * 0.15)] and not rep_call["mask"][int(H * 0.9), int(W * 0.9)]
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    assert rep["reparos"] and Path(res.outputs[0]).is_file()
    # sem reparo pedido: uma chamada só
    eng2 = FakeEngine()
    api.redesign(rgb, "reforma", "luxo", n=1, seed=2, out_dir=tmp_path / "s", engine=eng2, depth=depth, groups=g,
                 long_side=256, rank_fn=fake_groups, repair=False)
    assert len(eng2.calls) == 1


def test_appliances_are_pasted_back_from_labels(tmp_path):
    from mobilia.analysis.segmentation import SegmentationResult, labels_to_groups

    rgb, _, depth, _ = _room()
    H, W = rgb.shape[:2]
    names = {0: "wall", 1: "floor", 2: "refrigerator, icebox", 3: "windowpane, window", 4: "sofa, couch, lounge",
             5: "cabinet"}
    labels = np.zeros((H, W), np.uint8)
    labels[int(H * 0.65):] = 1
    labels[60:200, 10:60] = 2                  # geladeira
    labels[60:120, 160:240] = 3
    labels[130:200, 100:150] = 4
    seg = SegmentationResult(labels, labels_to_groups(labels, names), names)
    rgb[60:200, 10:60] = (230, 230, 235)
    eng = FakeEngine()
    res = api.redesign(rgb, "mudar_estilo", "industrial", "cozinha", n=1, seed=3, out_dir=tmp_path, engine=eng,
                       depth=depth, groups=seg, long_side=320)
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    assert rep["trava"]["eletrodomesticos"] > 0.05
    Wr, Hr = res.size
    base = np.asarray(Image.open(res.before))
    sx, sy = Wr / W, Hr / H
    core = (slice(int(80 * sy), int(180 * sy)), slice(int(20 * sx), int(50 * sx)))
    assert np.abs(res.images[0][core].astype(int) - base[core].astype(int)).mean() < 30     # só o casamento de cor


def test_anchor_tone_pulls_cast_back_but_keeps_style_color():
    ref = np.full((64, 64, 3), (200, 200, 200), np.uint8)
    yellow = np.full((64, 64, 3), (230, 200, 120), np.uint8)
    out = AL.anchor_tone(yellow, ref)
    assert out[..., 2].mean() > yellow[..., 2].mean()        # menos amarelo
    assert out[..., 0].mean() - out[..., 2].mean() > 20      # mas ainda quente (âncora parcial)
    same = AL.anchor_tone(ref, ref)
    assert np.abs(same.astype(int) - ref.astype(int)).max() <= 2
