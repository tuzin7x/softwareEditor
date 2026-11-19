"""Redesenho de ambientes (``mobilia/design``), rápidos: o SDXL é trocado por um gerador falso.

Garantias: presets completos (funções, 12+ estilos, ambientes) e prompts coerentes; força efetiva dos poucos passos
do Lightning perto da pedida; trava da arquitetura (janelas/portas originais por cima, rampa sem halo); "Editar
área" pixel-idêntico fora da máscara; escolha entre candidatos; contrato da CLI ``design`` (eventos JSON)."""
from __future__ import annotations
import dataclasses

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
        mask = np.asarray(mask)
        if mask.dtype != bool:          # máscara suave (peso em [0, 1]): "gera" onde o peso passa de 0,5
            mask = mask > 0.5
        self.calls.append(dict(shape=image.shape, image=np.asarray(image).copy(), mask=mask.copy(), control=None if control is None else
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
                                 "fachada", "evento", "texto", "melhorar"}
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
    assert f["redesign_simples"].strength < f["mudar_estilo"].strength <= f["reforma"].strength
    assert 0.35 <= f["redesign_simples"].strength <= 0.5
    assert 0.78 <= f["mudar_estilo"].strength <= 0.9 and f["mudar_estilo"].control_end <= 0.75
    assert f["reforma"].strength >= 0.85 and f["reforma"].structural and not f["mudar_estilo"].structural
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
    assert p.positive.startswith("green sofa, ")          # texto do usuário traduzido (glossary.py)
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
    assert c["shape"][0] % 8 == 0 and c["shape"][1] % 8 == 0
    with pytest.raises(ValueError, match="vazia"):
        api.edit_area(rgb, np.zeros(rgb.shape[:2], np.uint8), "x", engine=eng, depth=depth, out_dir=tmp_path)
    with pytest.raises(ValueError, match="Descreva"):
        api.edit_area(rgb, mask, "", engine=eng, depth=depth, out_dir=tmp_path)


def test_text_to_design(tmp_path):
    eng = FakeEngine()
    res = api.text_to_design("sala aconchegante com lareira", "japandi", "sala", n=2, seed=9, out_dir=tmp_path,
                             engine=eng, aspect="9:16")
    W, H = res.size
    assert H > W and W % 8 == 0 and abs(W / H - 9 / 16) < 0.01 and len(res.outputs) == 2
    c = eng.calls[0]
    assert c["strength"] == 1.0 and c["control"] is None and c["mask"].all()
    assert c["prompt"].positive.startswith("cozy living room with fireplace")     # português -> inglês


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

    def no_lama(*a, **k):
        raise RuntimeError("sem removedor nos testes")
    monkeypatch.setattr("mobilia.removal.plate.load_lama", no_lama)
    monkeypatch.setattr("mobilia.analysis.weights.missing_design_models", lambda: [])
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
    pytest.importorskip("huggingface_hub")        # prefetch_models importa o huggingface_hub (não está no CI)
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
    assert e.positive.startswith("a leather caramel armchair") and "mid-century" in e.positive
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


# ============================================================================ correções da avaliação (rodada 3)
def test_portuguese_free_text_is_translated_for_the_generator():
    from mobilia.design.glossary import to_english

    assert to_english("mesa de centro baixa retangular de madeira clara com livros e um vaso de cerâmica") == \
        "low rectangular light wood coffee table with books and ceramic vase"
    assert to_english("sofá de linho bege com almofadas verdes") == "linen beige sofa with green cushions"
    assert to_english("a wooden coffee table with flowers") == "a wooden coffee table with flowers"   # já inglês
    assert to_english("sofá") == "sofa" and to_english("") == ""
    p = P.build_prompt("editar_area", None, "sala", 0, "mesa de centro de madeira com flores")
    assert p.positive.startswith("a wood coffee table with flowers") and "living room" in p.positive
    assert "empty floor" in p.negative


def test_soft_mask_is_kept_soft_for_the_pipeline():
    from mobilia.design.engine import mask_u8

    hard, soft = mask_u8(np.array([[True, False]]))
    assert not soft and hard.tolist() == [[255, 0]]
    m, soft = mask_u8(np.array([[1.0, 0.5, 0.0]], np.float32))
    assert soft and m.tolist() == [[255, 128, 0]]


def _plane_room(H=120, W=160):
    """Profundidade sintética: parede do fundo (z = 5) e piso (disparidade linear em y) com um 'sofá' perto."""
    ys = np.arange(H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    z = np.full((H, W), 5.0, np.float32)
    floor = ys >= H // 2
    disp_floor = 0.2 + (ys - H // 2) / (H // 2) * 0.6             # 1/z: de 0,2 (fundo) a 0,8 (perto)
    z[floor] = 1.0 / disp_floor[floor]
    g = np.full((H, W), WALL, np.uint8)
    g[floor] = FLOOR
    sofa = np.zeros((H, W), bool)
    sofa[H // 3:H * 3 // 4, W // 4:W * 11 // 16] = True
    z_sofa = z.copy()
    z_sofa[sofa] = 1.6                                            # o sofá está na frente da parede e do piso
    g_sofa = g.copy()
    g_sofa[sofa] = FURN
    return z, z_sofa, g_sofa, sofa


def test_structural_depth_removes_furniture_and_restores_planes():
    from mobilia.design import structure as S

    z, z_sofa, g, sofa = _plane_room()
    out = S.structural_depth(z_sofa, sofa, g)
    inner = np.zeros_like(sofa)
    inner[46:84, 48:102] = True
    err = np.abs(1 / out[inner] - 1 / z[inner]) / (1 / z[inner])
    assert np.median(err) < 0.08                                     # parede e piso atrás do sofá voltam
    assert np.all(out[sofa] >= z_sofa[sofa] - 1e-3)                  # nada mais perto que o próprio móvel
    assert np.array_equal(out[~sofa], z_sofa[~sofa])                 # fora do buraco nada muda
    hole = S.movable_mask(g, grow_frac=0.0)
    assert hole[60, 70] and not hole[10, 10]


def test_reforma_uses_structural_depth_and_prefills_the_furniture(tmp_path):
    z, z_sofa, g, sofa = _plane_room(240, 320)
    rgb = np.full((240, 320, 3), 200, np.uint8)
    rgb[g == FLOOR] = (140, 100, 70)
    rgb[sofa] = (40, 40, 160)                                       # sofá azul
    eng = FakeEngine()
    res = api.redesign(rgb, "reforma", "japandi", "sala", n=1, seed=1, out_dir=tmp_path / "r", engine=eng,
                       depth=z_sofa, groups=g, long_side=320, repair=False)
    c = eng.calls[0]
    H, W = c["shape"][:2]
    ctl_plain = api.depth_control(z_sofa, (W, H))
    yy, xx = int(0.54 * H), int(0.45 * W)
    assert c["control"][yy, xx] < ctl_plain[yy, xx] - 0.1            # a silhueta do sofá saiu do controle
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    assert rep["trava"]["profundidade_estrutural"] and rep["trava"]["estrutural_moveis"] > 0.05
    px = c["image"][yy, xx].astype(int)                             # o sofá azul não semeia a geração
    assert px[2] - px[0] < 60
    # mudar estilo mantém a disposição: profundidade da foto como está
    eng2 = FakeEngine()
    api.redesign(rgb, "mudar_estilo", "japandi", "sala", n=1, seed=1, out_dir=tmp_path / "m", engine=eng2,
                 depth=z_sofa, groups=g, long_side=320, repair=False)
    assert abs(eng2.calls[0]["control"][yy, xx] - ctl_plain[yy, xx]) < 0.05


def test_edit_area_soft_mask_structural_control_and_translation(room, tmp_path):
    rgb, _, depth, _ = room
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[160:220, 60:150] = 255
    d = depth.copy()
    d[160:220, 60:150] = 1.0                                        # o objeto antigo, bem perto
    seen = {}

    class Eng(FakeEngine):
        def generate(self, image, mask, *a, **k):
            seen["mask"] = np.asarray(mask).copy()
            return super().generate(image, mask, *a, **k)

    eng = Eng()
    res = api.edit_area(rgb, mask, prompt="mesa de centro de madeira com flores", n=1, seed=4,
                        out_dir=tmp_path / "e", engine=eng, depth=d)
    w = seen["mask"]
    assert w.dtype != bool and w.max() == pytest.approx(1.0) and ((w > 0.05) & (w < 0.95)).any()   # rampa suave
    c = eng.calls[0]
    assert "coffee table" in c["prompt"].positive and c["control"] is not None
    H, W = c["shape"][:2]
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    x0, y0, x1, y1 = rep["recorte"]
    cy, cx = int((190 - y0) / (y1 - y0) * H), int((105 - x0) / (x1 - x0) * W)
    ring = c["control"][max(0, cy - 40):cy + 40, :]
    assert c["control"][cy, cx] < ring.max() + 1e-3 and c["control"][cy, cx] < 0.95   # sem a silhueta do objeto
    # o recorte tem contexto: bem maior que a área pintada
    assert (x1 - x0) * (y1 - y0) > 3 * (90 * 60)


def test_missing_opening_is_restored_and_ceiling_kept(tmp_path):
    rgb, g, depth, (y0, y1, x0, x1) = _room()
    H, W = rgb.shape[:2]
    COV = GROUP_IDS["window_covering"]
    g = g.copy()
    g[y0:y1, 20:70] = COV                                   # uma persiana fechada (não travada no "Mudar estilo")
    rgb[y0:y1, 20:70] = (90, 60, 30)
    eng = FakeEngine()

    def gen_groups(img):                                    # a persiana virou parede na imagem gerada
        out = np.full(img.shape[:2], WALL, np.uint8)
        out[: img.shape[0] // 6] = CEIL
        return out

    res = api.redesign(rgb, "mudar_estilo", "industrial", "sala", n=1, seed=2, out_dir=tmp_path, engine=eng,
                       depth=depth, groups=g, long_side=320, rank_fn=gen_groups)
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    assert rep["reparos"]["0"]["aberturas_perdidas"] > 0
    base = np.asarray(Image.open(res.before))
    Wr, Hr = res.size
    sx, sy = Wr / W, Hr / H
    core = (slice(int((y0 + 8) * sy), int((y1 - 8) * sy)), slice(int(28 * sx), int(62 * sx)))
    assert np.abs(res.images[0][core].astype(int) - base[core].astype(int)).mean() < 30
    # teto: a foto volta (só com o casamento de cor), a junção com a parede fica com a imagem gerada
    assert rep["reparos"]["0"]["teto"] > 0.05
    top = (slice(2, int(H / 6 * sy) - 8), slice(10, Wr - 10))
    assert np.abs(res.images[0][top].astype(int) - base[top].astype(int)).mean() < 30
    assert P.FUNCTIONS["mobiliar"].keep_ceiling and not P.FUNCTIONS["reforma"].keep_ceiling


def test_ceiling_keep_mask_spares_new_pendant_lights():
    src = np.full((100, 100), WALL, np.uint8)
    src[:30] = CEIL
    gen = src.copy()
    gen[0:25, 45:55] = GROUP_IDS["ceiling_fixture"]                 # um pendente novo
    m = AL.ceiling_keep_mask(src, gen, (100, 100))
    assert m[10, 20] and not m[10, 50] and not m[29, 20]
    gen2 = gen.copy()
    gen2[15:30] = WALL                                              # a parede gerada sobe mais: lá não cola
    m2 = AL.ceiling_keep_mask(src, gen2, (100, 100))
    assert m2[5, 20] and not m2[20, 20]


def test_refine_openings_finds_window_behind_glass():
    g = np.full((100, 100), WALL, np.uint8)
    g[10:90, 40:90] = DOOR                                          # box de vidro inteiro = "porta"
    rgb = np.zeros((200, 200, 3), np.uint8)

    def seg_fn(crop):                                               # o recorte ampliado mostra a janela
        out = np.full(crop.shape[:2], DOOR, np.uint8)
        h, w = out.shape
        out[int(h * 0.2):int(h * 0.4), int(w * 0.3):int(w * 0.7)] = WIN
        return out

    g2, changed = api.refine_openings(rgb, g, seg_fn)
    assert changed > 0 and (g2 == WIN).sum() == changed and np.all(g2[g != DOOR] == g[g != DOOR])
    lock = AL.lock_mask(g2, ("window", "door"))
    lock2, _ = AL.cap_doors(lock, AL.lock_mask(g2, ("door",)), all_frames=True)    # banheiro: só a moldura
    assert lock2[(g2 == WIN)].mean() > 0.9                          # mas a janela continua travada


def test_quality_score_punishes_lost_openings_and_oversized_furniture():
    g = np.full((100, 100), WALL, np.uint8)
    lock = np.zeros((100, 100), bool)
    src = np.zeros((100, 100), bool)
    src[10:30, 10:30] = True
    kept = g.copy()
    kept[10:30, 10:30] = WIN
    kept[60:90, 40:60] = FURN
    a = Q.score(kept, lock, source_openings=src)
    lost = kept.copy()
    lost[10:30, 10:30] = WALL
    b = Q.score(lost, lock, source_openings=src)
    assert b["aberturas_perdidas"] > 0.9 and b["nota"] < a["nota"]
    giant = kept.copy()
    giant[40:100, 0:100] = FURN
    c = Q.score(giant, lock, source_openings=src)
    assert c["maior_movel"] > Q.OVERSIZE_FRAC and c["nota"] < a["nota"]


def test_style_prompts_are_distinct_and_ceiling_negatives():
    esc = P.build_prompt("mudar_estilo", "escandinavo", "sala", 0)
    mod = P.build_prompt("mudar_estilo", "moderno", "sala", 0)
    w1 = set(esc.positive.replace(",", " ").split())
    w2 = set(mod.positive.replace(",", " ").split())
    assert len(w1 ^ w2) > 25                                       # vocabulário bem diferente
    assert "beams" not in P.STYLES["rustico"].prompt
    mob = P.build_prompt("mobiliar", "rustico", "sala", 0)
    assert "coffered ceiling" in mob.negative and "oversized furniture" in mob.negative
    assert P.FUNCTIONS["mudar_estilo"].tone_ab < 0.2 and P.FUNCTIONS["mudar_estilo"].strength >= 0.78


def test_new_furniture_where_photo_had_furniture_is_not_an_invented_opening():
    src = np.full((100, 100), WALL, np.uint8)
    src[40:100, 50:100] = GROUP_IDS["fixture"]                      # armário/espelho na foto
    gen = src.copy()
    gen[10:30, 10:30] = WIN                                         # janela numa parede lisa: inventada
    gen[50:90, 60:90] = DOOR                                        # box de vidro onde havia o armário: móvel novo
    orig = np.zeros((100, 100), bool)
    new = AL.new_openings(gen, orig, (100, 100), source_groups=src)
    assert new[20, 20] and not new[70, 75]
    assert AL.new_openings(gen, orig, (100, 100))[70, 75]           # sem os grupos da foto: os dois contam


def test_movable_mask_keeps_open_kitchen_fixtures():
    from mobilia.design import structure as S

    g = np.full((60, 80), WALL, np.uint8)
    g[30:50, 5:30] = FURN
    g[10:40, 50:75] = GROUP_IDS["fixture"]
    assert S.movable_mask(g, grow_frac=0)[20, 60] and S.movable_mask(g, grow_frac=0)[40, 10]
    keep = S.movable_mask(g, grow_frac=0, fixtures=False)
    assert not keep[20, 60] and keep[40, 10]


def test_repair_stays_off_the_ceiling(tmp_path):
    rgb, g, depth, _ = _room()
    eng = FakeEngine()

    def gen_groups(img):                                   # janela inventada que encosta no teto
        out = np.full(img.shape[:2], WALL, np.uint8)
        h, w = out.shape
        out[: h // 6] = CEIL
        out[0:int(h * 0.45), int(w * 0.05):int(w * 0.3)] = WIN
        return out

    api.redesign(rgb, "mobiliar", "moderno", "sala", n=1, seed=2, out_dir=tmp_path, engine=eng, depth=depth,
                 groups=g, long_side=256, rank_fn=gen_groups)
    rep_call = eng.calls[1]
    H, W = rep_call["mask"].shape
    assert rep_call["mask"][int(H * 0.3), int(W * 0.15)] and not rep_call["mask"][2, int(W * 0.15)]


def test_kitchen_reforma_keeps_cabinets_in_the_structural_depth(tmp_path):
    z, z_sofa, g, sofa = _plane_room(240, 320)
    g = g.copy()
    g[sofa] = GROUP_IDS["fixture"]                                  # bancada/armário em vez de sofá
    rgb = np.full((240, 320, 3), 200, np.uint8)
    eng = FakeEngine()
    api.redesign(rgb, "reforma", "industrial", "cozinha", n=1, seed=1, out_dir=tmp_path, engine=eng,
                 depth=z_sofa, groups=g, long_side=320, repair=False, auto_empty=False)
    c = eng.calls[0]
    H, W = c["shape"][:2]
    yy, xx = int(0.54 * H), int(0.45 * W)
    assert abs(c["control"][yy, xx] - api.depth_control(z_sofa, (W, H))[yy, xx]) < 0.05


# ============================================================================ rodada 2 do teste na GPU (notebook 8 GB)
from mobilia.design import memplan as MP  # noqa: E402


def test_memory_plan_8gb_laptop_fits_in_7gb():
    """RTX 5070 Laptop (8 GB) com ~650 MB de RAM livre: geração a 1024 com CFG, refino a 1280 SEM CFG, OneFormer
    fora da GPU, codificadores de texto em bf16 — nada passa de ~7 GB (o refino a 1536 com CFG transbordava)."""
    p = MP.plan("cuda", vram_total_mb=8188, vram_free_mb=7600, ram_free_mb=650)
    assert p.gen_long_side == 1024 and not p.offload
    # teste real (a45fe5c): o refino a 1280 px passou de 8 GB de pico -> placas < 12 GB ficam sem refino
    assert not p.refine and p.refine_long_side == 0
    assert p.peak_gen_mb <= 7168 and p.budget_mb <= 7168
    assert p.seg_off_gpu and p.te_dtype == "bfloat16"
    # o caso que transbordou no teste (1536 px) nunca é escolhido numa placa de 8 GB
    assert MP.REFINE_SIDES_TIGHT == ()
    big = MP.plan("cuda", 12288, 11800, 16000)
    assert big.refine_long_side == 1536 and big.refine_cfg and big.te_dtype == "float32"
    small = MP.plan("cuda", 6144, 5800, 16000)
    assert small.offload and not small.refine
    cpu = MP.plan("cpu", None, None, 8000)
    assert cpu.gen_long_side == 768 and not cpu.refine and not cpu.offload
    # VRAM já ocupada por outro programa: o orçamento segue a livre
    busy = MP.plan("cuda", 8188, 6600, 8000)
    assert busy.budget_mb <= 6344 and (busy.offload or not busy.refine or busy.peak_refine_mb <= busy.budget_mb)


def test_memory_plan_calibrated_to_measured_peak():
    """Teste real no notebook (8 GB): a estimativa antiga (6985 MB contra orçamento 6757) ligava o offload e
    desligava o refino, mas o pico medido por chamada foi 5813 MB. Calibrado: a geração cabe sem offload e o refino
    é leve (≤ 1280 px, sem CFG) só se a estimativa couber; senão desligado e anotado."""
    est = MP.torch_peak_mb(1024, 768, cfg=True)
    assert abs(est - MP.MEASURED_GEN_PEAK_MB) / MP.MEASURED_GEN_PEAK_MB < 0.05
    nb = MP.plan("cuda", vram_total_mb=8188, vram_free_mb=7013, ram_free_mb=650)    # números do teste real
    assert nb.budget_mb == 6757 and not nb.offload and nb.peak_gen_mb <= nb.budget_mb
    assert not nb.refine and nb.refine_long_side == 0
    tight = MP.plan("cuda", 8188, 6600, 8000)
    assert not tight.offload and not tight.refine and any("sem folga" in x for x in tight.notes)


def test_refine_spill_detector():
    assert MP.refine_spilled(1.5, 1024 * 768, 20.0, 1536 * 1024)          # 18–23 s/passo medido no notebook
    assert not MP.refine_spilled(1.5, 1024 * 768, 1.2, 1280 * 960)         # refino sem CFG, sem transbordar


class TimedEngine(FakeEngine):
    """Gerador falso que registra tamanho e tempo por passo como o real (para o detector de transbordo)."""

    device = "cuda"

    def __init__(self, plan, refine_s=0.5):
        super().__init__()
        self.plan = plan
        self.refine_s = refine_s
        self.log = []

    def generate(self, image, mask, control, prompt, seed, strength, steps, guidance, control_scale, control_end,
                 callback=None):
        out = super().generate(image, mask, control, prompt, seed, strength, steps, guidance, control_scale,
                               control_end, callback)
        h, w = image.shape[:2]
        s = self.refine_s if strength < 0.5 else 1.0
        self.log.append({"tamanho": [w, h], "s_por_passo": s, "passos_efetivos": 3, "s": 3 * s})
        return out


def test_redesign_uses_memory_plan_on_gpu(tmp_path, monkeypatch):
    rgb, g, depth, _ = _room(960, 640)
    # plano com refino leve forçado (o caminho do refino continua válido para placas maiores)
    plan8 = dataclasses.replace(MP.plan("cuda", 8188, 7600, 650), refine=True, refine_long_side=1280,
                                refine_cfg=False, peak_refine_mb=6000)
    eng = TimedEngine(plan8, refine_s=0.6)
    monkeypatch.setattr(api, "_refine_spilled", lambda e: MP.refine_spilled(
        e.log[-2]["s_por_passo"], np.prod(e.log[-2]["tamanho"]), e.log[-1]["s_por_passo"], np.prod(e.log[-1]["tamanho"])))
    res = api.redesign(rgb, "mudar_estilo", "moderno", n=2, seed=3, out_dir=tmp_path, engine=eng, depth=depth,
                       groups=g, candidates=2, repair=False)
    gen = [c for c in eng.calls if c["strength"] > 0.5]
    ref = [c for c in eng.calls if c["strength"] < 0.5]
    assert max(gen[0]["shape"][:2]) == 1024 and gen[0]["guidance"] > 1.0
    assert len(ref) == 2 and max(ref[0]["shape"][:2]) == 1280 and ref[0]["guidance"] == 1.0    # sem CFG
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    assert rep["plano_memoria"]["refine_long_side"] == 1280 and not rep["refino_desligado_por_memoria"]
    assert max(res.size) == 1280
    # refino lento demais (transbordou para a RAM): a segunda versão sai sem ele, no mesmo tamanho final
    eng2 = TimedEngine(plan8, refine_s=25.0)
    monkeypatch.setattr(api, "_refine_spilled", lambda e: MP.refine_spilled(
        e.log[-2]["s_por_passo"], np.prod(e.log[-2]["tamanho"]), e.log[-1]["s_por_passo"], np.prod(e.log[-1]["tamanho"])))
    logs = []
    res2 = api.redesign(rgb, "mudar_estilo", "moderno", n=2, seed=3, out_dir=tmp_path / "b", engine=eng2,
                        depth=depth, groups=g, candidates=2, repair=False, progress=logs.append)
    assert len([c for c in eng2.calls if c["strength"] < 0.5]) == 1
    assert all(im.shape[:2] == (res2.size[1], res2.size[0]) for im in res2.images)
    assert any(ev.get("event") == "log" and "memória" in ev.get("message", "") for ev in logs)
    assert json.loads(Path(res2.report).read_text(encoding="utf-8"))["refino_desligado_por_memoria"]


def test_refine_spilled_reads_engine_calls():
    class E:
        calls = [{"tamanho": [1024, 768], "s_por_passo": 1.5}, {"tamanho": [1536, 1024], "s_por_passo": 20.0}]
    assert api._refine_spilled(E())
    E.calls = E.calls[:1]
    assert not api._refine_spilled(E())


def test_repair_has_no_flat_patch_with_hard_border():
    """Reparo de abertura inventada: mesmo se o gerador devolver um retângulo cinza chapado, a área entra por
    Poisson + rampa larga — sem degrau de luz na borda (o "retângulo cinza num canto de parede")."""
    rng = np.random.default_rng(1)
    H, W = 240, 320
    wall = np.clip(np.array([205, 192, 175]) + rng.normal(0, 4, (H, W, 3)), 0, 255).astype(np.uint8)
    wall = (wall.astype(np.float32) * np.linspace(0.85, 1.05, W)[None, :, None]).clip(0, 255).astype(np.uint8)
    new = np.zeros((H, W), bool)
    new[40:140, 200:300] = True                                     # janela inventada perto do canto

    class Flat(FakeEngine):
        def generate(self, image, mask, *a, **k):
            super().generate(image, mask, *a, **k)
            out = np.asarray(image).copy()
            out[np.asarray(mask) > 0.5] = (128, 128, 128)
            return out

    eng = Flat()
    lock = np.zeros((H, W), bool)
    out, info = api._repair_openings(eng, wall, new, lock, None, "sala", 1, 6, True, gen_size=(160, 120))
    c = eng.calls[0]
    assert c["shape"][:2] == (120, 160)                     # gera no tamanho da geração, não no do refino
    assert c["strength"] >= 0.9
    m = cv2_dilate(new, int(0.02 * np.hypot(H, W)))
    naive = wall.copy()
    naive[m] = (128, 128, 128)
    assert AL.seam_step(naive, m) > 30
    assert AL.seam_step(out, m) < 3 and info["degrau_borda"] < 3


def cv2_dilate(m, r):
    import cv2

    return cv2.dilate(m.astype(np.uint8), AL._disk(r)).astype(bool)


def test_poisson_blend_handles_mask_on_the_image_border():
    base = np.full((60, 80, 3), 200, np.uint8)
    top = np.full((60, 80, 3), 90, np.uint8)
    m = np.zeros((60, 80), bool)
    m[0:30, 50:80] = True                                            # encosta em duas bordas
    out = AL.poisson_blend(base, top, m)
    assert out.shape == base.shape and np.array_equal(out[~m], base[~m])
    assert np.abs(out[m].astype(int) - 200).mean() < 8              # gradiente nulo: vira a cor da borda


def test_ceiling_restore_covers_invented_soffit_island():
    src = np.full((100, 100), WALL, np.uint8)
    src[:40] = CEIL
    gen = src.copy()
    gen[10:22, 30:70] = WALL                                         # face de um rebaixo no meio do teto
    m = AL.ceiling_keep_mask(src, gen, (100, 100))
    assert m[15, 50] and m[5, 50]
    gen2 = src.copy()
    gen2[25:40] = WALL                                               # a parede só sobe a partir da junção
    assert not AL.ceiling_keep_mask(src, gen2, (100, 100))[30, 50]


def test_edit_area_keeps_photo_aspect_and_object_prompt(tmp_path):
    """Notebook: 960×640 (3:2) gerado a 1024×768 (4:3), força 1, sem ControlNet, e uma "mini-sala" no retângulo.
    Agora: recorte e geração com a proporção da foto, profundidade ligada e prompt só do objeto."""
    rgb, _, depth, _ = _room(960, 640)
    mask = np.zeros((640, 960), np.uint8)
    mask[380:560, 300:620] = 255
    eng = FakeEngine()
    eng.device = "cuda"                                              # os padrões da GPU
    res = api.edit_area(rgb, mask, prompt="mesa de centro redonda de madeira", n=1, seed=2, out_dir=tmp_path,
                        engine=eng, depth=depth)
    c = eng.calls[0]
    h, w = c["shape"][:2]
    assert max(w, h) == 1024 and abs(w / h - 1.5) < 0.01 and w % 8 == 0 and h % 8 == 0
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    x0, y0, x1, y1 = rep["recorte"]
    assert abs((x1 - x0) / (y1 - y0) - 1.5) < 0.01
    assert x0 <= 300 and y0 <= 380 and x1 >= 620 and y1 >= 560
    fp = P.function("editar_area")
    assert c["steps"] == fp.steps and c["control"] is not None and c["control_scale"] == fp.control_scale > 0
    pos, neg = c["prompt"].positive, c["prompt"].negative
    assert pos.startswith("a round wood coffee table") and "single" in pos and "magazine" not in pos
    assert "entire room" in neg and "sofa" in neg and "new window" in neg
    s = P.build_prompt("editar_area", None, "sala", 0, "sofá cinza")
    assert "sofa" not in s.negative.split("entire room")[1] and s.positive.startswith("a gray sofa")


def test_edit_crop_box_aspect_and_bounds():
    m = np.zeros((640, 960), bool)
    m[600:640, 900:960] = True                                       # canto da foto
    x0, y0, x1, y1 = api.edit_crop_box(m, 1.5, min_side=480)
    assert 0 <= x0 and 0 <= y0 and x1 <= 960 and y1 <= 640 and abs((x1 - x0) / (y1 - y0) - 1.5) < 0.01
    assert x1 == 960 and y1 == 640 and (x1 - x0) >= 480
    big = np.ones((640, 960), bool)
    assert api.edit_crop_box(big, 1.5) == (0, 0, 960, 640)
    p = np.zeros((900, 600), bool)
    p[400:500, 200:300] = True                                       # foto retrato
    x0, y0, x1, y1 = api.edit_crop_box(p, 600 / 900, min_side=450)
    assert abs((x1 - x0) / (y1 - y0) - 600 / 900) < 0.01 and (y1 - y0) >= 449


def test_text_room_comes_from_the_prompt_and_format_is_kept(tmp_path):
    eng = FakeEngine()
    res = api.text_to_design("quarto aconchegante", "escandinavo", "sala", n=1, seed=1, out_dir=tmp_path,
                             engine=eng, aspect="16x9")
    assert res.ambiente == "quarto"
    pos = eng.calls[0]["prompt"].positive
    assert "bedroom" in pos and "living room" not in pos
    W, H = res.size
    assert abs(W / H - 16 / 9) < 0.01
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    assert rep["ambiente_pedido"] == "sala" and rep["ambiente_do_texto"] == "quarto" and rep["formato"] == "16:9"
    res = api.text_to_design("ambiente aconchegante", "moderno", "cozinha", n=1, seed=1, out_dir=tmp_path / "b",
                             engine=FakeEngine(), aspect="retrato")
    assert res.ambiente == "cozinha" and res.size[1] > res.size[0]
    with pytest.raises(ValueError, match="Formato"):
        api.text_to_design("sala", "moderno", "sala", n=1, engine=FakeEngine(), out_dir=tmp_path / "c",
                           aspect="panorama")
    assert P.infer_room("Sala de jantar com mesa para 8") == "sala_jantar"
    assert P.infer_room("quarto de casal com banheiro") == "quarto"
    assert P.infer_room("mesa de madeira") is None


def test_event_prompts_lead_with_key_objects():
    p = P.build_prompt("evento", None, "sala", 0, evento="natal")
    assert p.positive.startswith("a tall green christmas tree") and "fairy lights" in p.positive
    assert p.positive.count("christmas tree") >= 2 and "plain wall" in p.positive    # árvore reforçada, fora da janela
    assert "undecorated" in p.negative
    for k, ev in P.EVENTS.items():
        assert ev["key"] and P.build_prompt("evento", None, "sala", 0, evento=k).positive.startswith(ev["key"])
    f = P.function("evento")
    assert f.strength >= 0.85 and f.guidance >= 2.2


def test_every_design_model_is_prefetched(monkeypatch):
    from mobilia.analysis import weights as Wt
    from mobilia.config import GEOMETRY_MODEL_ID, GEOMETRY_MODEL_ID_SMALL, SEGMENTATION_MODEL_ID

    got = []
    huggingface_hub = pytest.importorskip("huggingface_hub")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **k: got.append(repo))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, *a, **k: got.append(repo))
    monkeypatch.setattr(Wt, "ensure_weight", lambda w, offline=False: got.append(
        w.repo_id if isinstance(w, Wt.HubWeight) else w.filename))
    Wt.prefetch_models()
    for m in Wt.design_models():
        assert m in got, m
    for m in (GEOMETRY_MODEL_ID, GEOMETRY_MODEL_ID_SMALL, SEGMENTATION_MODEL_ID, Wt.ESRGAN_WEIGHT.filename,
              Wt.SDXL_GEN_WEIGHT.repo_id, Wt.SDXL_TEXT_WEIGHT.repo_id, Wt.SDXL_VAE_WEIGHT.repo_id,
              Wt.SDXL_CN_DEPTH_WEIGHT.repo_id):
        assert m in Wt.design_models()


def test_oneformer_loads_without_safetensors_conversion(monkeypatch):
    transformers = pytest.importorskip("transformers")

    from mobilia.analysis import weights as Wt

    seen = {}

    class FakeModel:
        config = type("C", (), {"id2label": {0: "wall"}})()

        def eval(self):
            return self

        def to(self, *a, **k):
            return self

    def fake_model(repo, **k):
        seen.update(k)
        return FakeModel()

    monkeypatch.setattr(transformers.OneFormerForUniversalSegmentation, "from_pretrained", staticmethod(fake_model))
    monkeypatch.setattr(transformers.OneFormerProcessor, "from_pretrained", staticmethod(lambda repo, **k: object()))
    monkeypatch.setattr(Wt, "ensure_local", lambda repo: 0.0)
    monkeypatch.setattr(Q, "Segmenter", Q.Segmenter.__wrapped__ if hasattr(Q.Segmenter, "__wrapped__") else
                        _REAL_SEGMENTER)
    s = Q.Segmenter("cpu")
    assert seen.get("use_safetensors") is False
    s.park()                                                         # na CPU não faz nada
    assert not s.parked
    import os

    assert os.environ.get("DISABLE_SAFETENSORS_CONVERSION") == "1"


_REAL_SEGMENTER = Q.Segmenter


def test_cli_uses_installed_models_cache(tmp_path):
    from mobilia import runtime

    hub = tmp_path / "models" / "huggingface" / "hub" / "models--x--y"
    hub.mkdir(parents=True)
    env = runtime.use_installed_models({"MOBILIA_HOME": str(tmp_path)})
    assert env["HF_HOME"] == str(tmp_path / "models" / "huggingface")
    assert env["DISABLE_SAFETENSORS_CONVERSION"] == "1"
    keep = runtime.use_installed_models({"MOBILIA_HOME": str(tmp_path), "HF_HOME": "/outro"})
    assert keep["HF_HOME"] == "/outro"
    empty = runtime.use_installed_models({"MOBILIA_HOME": str(tmp_path / "nada")})
    assert "HF_HOME" not in empty


def test_upscaler_weights_download_on_demand(monkeypatch, tmp_path):
    from mobilia.analysis import weights as Wt

    calls = []
    monkeypatch.setattr(Wt, "ensure_file_weight", lambda w, offline=False, progress=None, **k: calls.append(
        (offline, progress)) or tmp_path / "x.pth")
    monkeypatch.delenv("MOBILIA_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    prog = lambda d, t: None  # noqa: E731
    Wt.esrgan_path(progress=prog)
    assert calls[-1] == (False, prog)                                # online: baixa se faltar
    monkeypatch.setenv("MOBILIA_OFFLINE", "1")
    Wt.esrgan_path()
    assert calls[-1][0] is True


def test_fetch_reports_progress_and_offline_message(monkeypatch, tmp_path):
    import io
    import urllib.request

    from mobilia.analysis import weights as Wt

    data = b"x" * (3 * (1 << 20) + 5)

    class Resp(io.BytesIO):
        headers = {"Content-Length": str(len(data))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    from mobilia import netfast

    monkeypatch.setattr(netfast, "urlopen", lambda req, timeout=0: Resp(data))      # downloads via netfast
    seen = []
    Wt._fetch("https://example.invalid/w.pth", tmp_path / "w.pth", progress=lambda d, t: seen.append((d, t)))
    assert (tmp_path / "w.pth").read_bytes() == data and seen[-1] == (len(data), len(data)) and len(seen) == 4
    spec = Wt.FileWeight(key="t", filename="t.pth", subdir="t", sha256="0" * 64, size=1, urls=("https://x",),
                         label="Teste", licenca="MIT")
    with pytest.raises(Wt.WeightsError, match="não está instalado"):
        Wt.ensure_file_weight(spec, root=tmp_path, offline=True)

    def no_net(url, dest):
        raise urllib.error.URLError("offline")
    with pytest.raises(Wt.WeightsError, match="sem internet"):
        Wt.ensure_file_weight(spec, root=tmp_path, fetch=no_net)


def test_ceiling_paste_has_no_halo_around_a_dot():
    """Teto bege da foto sobre teto branco gerado com um pontinho (segmentado como luminária): antes ficava um
    buraco redondo com halo claro em volta do ponto. Agora o ponto não é luminária e o tom é casado por Poisson."""
    H, W = 120, 200
    src = np.full((H, W), WALL, np.uint8)
    src[:50] = CEIL
    gen_g = src.copy()
    gen_g[20:23, 100:103] = GROUP_IDS["ceiling_fixture"]
    original = np.zeros((H, W, 3), np.uint8)
    original[:] = (200, 190, 175)
    generated = np.zeros((H, W, 3), np.uint8)
    generated[:] = (210, 205, 200)
    generated[:50] = (238, 236, 232)
    m = AL.ceiling_keep_mask(src, gen_g, (W, H))
    assert m[21, 101]                                                 # o pontinho não abre buraco
    out, _ = AL.paste_ceiling(generated, original, m)
    L = out[8:36, 70:132].astype(float).mean(axis=2)
    assert L.std() < 1.5                                               # em volta do ponto: uniforme, sem halo


def test_christmas_tree_is_not_an_invented_window(tmp_path):
    """No teste, a árvore de Natal com luzes era segmentada como janela e o reparo a apagava (sobravam presentes)."""
    gen = np.full((100, 100), WALL, np.uint8)
    yy, xx = np.mgrid[0:100, 0:100]
    tree = (yy > 20) & (yy < 80) & (np.abs(xx - 50) < (yy - 20) * 0.4)        # triângulo
    gen[tree] = WIN
    assert not AL.new_openings(gen, np.zeros((100, 100), bool), (100, 100)).any()
    gen[10:60, 5:30] = WIN                                                    # janela retangular inventada
    assert AL.new_openings(gen, np.zeros((100, 100), bool), (100, 100))[30, 15]
    assert not P.function("evento").repair
    rgb, g, depth, _ = _room()
    eng = FakeEngine()
    api.redesign(rgb, "evento", None, "sala", evento="natal", n=1, seed=1, out_dir=tmp_path, engine=eng, depth=depth,
                 groups=g, rank_fn=lambda im: np.full(im.shape[:2], WIN, np.uint8))
    assert len(eng.calls) == 1                                                # nenhuma chamada de reparo


def test_edit_area_pastes_only_the_new_object(room, tmp_path):
    """Retângulo sobre um tapete estampado: o gerador refazia o tapete como piso liso e a borda do retângulo
    aparecia. Com a segmentação, só o objeto novo (+ sombra) entra; o tapete em volta continua o da foto."""
    rgb, _, depth, _ = room
    H, W = rgb.shape[:2]
    rng = np.random.default_rng(3)
    rug = rng.integers(60, 200, (H, W, 3), dtype=np.uint8)
    rgb = rgb.copy()
    rgb[150:, :] = rug[150:, :]                                       # "tapete" estampado embaixo
    mask = np.zeros((H, W), np.uint8)
    mask[150:230, 60:260] = 255

    def seg(img):                                                     # objeto = a área azul que o falso gerou
        out = np.full(img.shape[:2], FLOOR, np.uint8)
        blue = (img[..., 2] > 190) & (img[..., 1] > 100) & (img[..., 1] < 140)
        ys, xs = np.nonzero(blue)
        if len(xs):
            cy, cx = int(ys.mean()), int(xs.mean())
            h, w = img.shape[:2]
            out[cy - h // 12:cy + h // 12, cx - w // 12:cx + w // 12] = FURN   # mesa no meio da área
        return out

    eng = FakeEngine()
    res = api.edit_area(rgb, mask, prompt="mesa de centro redonda", n=1, seed=2, out_dir=tmp_path, engine=eng,
                        depth=depth, seg_fn=seg)
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    assert rep["colagem"] == ["objeto"]
    img = res.images[0]
    corner = (slice(155, 165), slice(66, 90))                         # canto da área, longe da mesa
    assert np.abs(img[corner].astype(int) - rgb[corner].astype(int)).mean() < 6   # tapete da foto
    cy, cx = 190, 160
    assert np.abs(img[cy - 3:cy + 3, cx - 3:cx + 3].astype(int) - rgb[cy - 3:cy + 3, cx - 3:cx + 3].astype(int)).mean() > 20
    # sem objeto reconhecido: a área inteira, como antes
    res2 = api.edit_area(rgb, mask, prompt="mesa", n=1, seed=2, out_dir=tmp_path / "b", engine=FakeEngine(),
                         depth=depth, seg_fn=lambda im: np.full(im.shape[:2], FLOOR, np.uint8))
    assert json.loads(Path(res2.report).read_text(encoding="utf-8"))["colagem"] == ["area"]


def test_split_target_separates_touching_furniture():
    objs = np.zeros((200, 300), bool)
    objs[40:120, 10:150] = True                                       # sofá (sai bastante da área)
    objs[118:170, 120:220] = True                                     # mesa encostada no sofá
    area = np.zeros((200, 300), bool)
    area[100:190, 100:240] = True
    ids = np.zeros((200, 300), np.uint8)
    ids[40:120, 10:150] = 7                                           # rótulo "sofa"
    ids[118:170, 120:220] = 9                                         # rótulo "coffee table"
    target, protect = api._split_target(objs, area, ids=ids)
    assert target[150, 170] and not target[80, 60] and protect[80, 60] and not protect[150, 170]
    sep = objs.copy()
    sep[110:130] = False                                              # sem rótulos: separados só se não encostam
    target, protect = api._split_target(sep, area)
    assert target[150, 170] and protect[80, 60]


# ------------------------------------------------- teste real na GPU (rodada 3): "Editar área" sem borrão nem mesa translúcida
def test_object_only_pastes_opaque_object_and_keeps_neighbour_furniture():
    """A mesa nova entra OPACA (os pixels do gerador, só a borda suavizada; o Poisson no objeto deixava um "tronco"
    semitransparente) e o braço do sofá que o retângulo pegou fica idêntico à foto, também no modo área inteira."""
    H, W = 200, 300
    rng = np.random.default_rng(1)
    photo = rng.integers(90, 170, (H, W, 3), dtype=np.uint8)            # tapete estampado
    plate = photo.copy()
    g = np.full((H, W, 3), 60, np.uint8)                                  # gerador: fundo liso escuro...
    g[90:150, 120:200] = (180, 40, 30)                                    # ... e uma mesa vermelha
    area = np.zeros((H, W), bool); area[60:180, 60:260] = True
    old = np.zeros((H, W), bool); old[100:150, 130:190] = True
    new = np.zeros((H, W), bool); new[90:150, 120:200] = True
    sofa = np.zeros((H, W), bool); sofa[60:120, 60:95] = True              # braço do sofá dentro do retângulo
    out, kp, mode = api._object_only(g, photo, plate, area, old, new, protect=sofa)
    assert mode == "objeto"
    assert np.array_equal(out[115:125, 150:170], g[115:125, 150:170])   # miolo da mesa = gerador (opaco)
    assert kp[115:125, 150:170].max() == 0.0
    final = AL.composite(out, photo, kp)
    assert np.abs(final[70:110, 65:90].astype(int) - photo[70:110, 65:90].astype(int)).max() <= 1   # sofá intacto
    # o objeto antigo sai crescido só um pouco: longe dele (e do novo) a foto continua
    assert np.array_equal(final[170:178, 230:255], photo[170:178, 230:255])
    # sem objeto novo reconhecido (modo área): o sofá ainda fica o da foto
    out2, kp2, mode2 = api._object_only(g, photo, plate, area, old, np.zeros((H, W), bool), protect=sofa)
    assert mode2 == "area" and kp2 is not None and kp2[80:100, 70:85].min() == 1.0


def test_edit_area_removes_only_old_object_not_the_sofa(room, tmp_path):
    """O removedor roda SÓ no objeto antigo (crescido ~1%), nunca no móvel vizinho, e a geração parte da foto sem
    ele (nítida), não de um borrão da área inteira."""
    rgb, _, depth, _ = room
    H, W = rgb.shape[:2]
    mask = np.zeros((H, W), np.uint8); mask[130:220, 60:250] = 255
    holes = []

    def inpainter(img, hole):
        holes.append(np.asarray(hole, bool).copy())
        out = img.copy(); out[hole] = (0, 255, 0)
        return out

    def seg(img):
        out = np.full(img.shape[:2], FLOOR, np.uint8)
        h, w = img.shape[:2]
        out[int(h * .30):int(h * .62), int(w * .02):int(w * .42)] = FURN    # sofá (sai bastante do retângulo)
        out[int(h * .58):int(h * .75), int(w * .55):int(w * .75)] = FURN    # mesa de centro dentro da área
        return out

    eng = FakeEngine()
    res = api.edit_area(rgb, mask, prompt="mesa de centro redonda", n=1, seed=2, out_dir=tmp_path, engine=eng,
                        depth=depth, seg_fn=seg, inpainter=inpainter)
    assert len(holes) == 1                                       # nada de pré-preencher a área inteira
    hole = holes[0]
    sofa = seg(np.zeros((hole.shape[0], hole.shape[1], 3), np.uint8)) == FURN
    sofa[:, int(hole.shape[1] * .5):] = False
    assert not (hole & sofa).any()
    assert hole.sum() < 0.6 * (eng.calls[0]["mask"].sum())        # só a mesa (+1%), não o retângulo
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    assert rep["preenchimento"] == "remocao"



# ------------------------------------------------- teste real na GPU (rodada 3): Natal sem árvore; móvel cortado no canto
def _xmas(tree: bool, H=240, W=320):
    import cv2

    img = np.full((240, 320, 3), (205, 195, 185), np.uint8)
    img[168:] = (150, 110, 80)
    img[150:200, 40:120] = (180, 30, 40)                                  # presentes e meias vermelhas
    if tree:
        pts = np.array([[288, 40], [258, 200], [316, 200]], np.int32)    # pinheiro em pé no canto (fora da janela)
        cv2.fillPoly(img, [pts], (30, 110, 45))
        for y, x in ((90, 288), (120, 282), (150, 292), (170, 276), (185, 300)):
            cv2.circle(img, (x, y), 2, (255, 225, 140), -1)               # pisca-pisca
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST) if (H, W) != (240, 320) else img


def test_tree_score_detects_christmas_tree_and_ignores_photo_green():
    from mobilia.design.quality import tree_score

    yes, no = tree_score(_xmas(True)), tree_score(_xmas(False))
    assert yes["arvore"] > 0.8 and yes["luzes"] > 0 and no["arvore"] < 0.1
    rug = _xmas(False); rug[200:235, 20:300] = (30, 110, 45)               # tapete verde deitado: não é árvore
    assert tree_score(rug)["arvore"] < 0.5
    ex = np.zeros((240, 320), bool); ex[:, 250:] = True                    # verde que já estava na foto
    assert tree_score(_xmas(True), exclude=ex)["arvore"] < 0.1


def test_christmas_candidates_ranked_by_tree(tmp_path):
    """Na GPU o Natal gera +2 candidatos e a escolha prefere os que têm árvore (antes nenhuma versão tinha)."""
    rgb, g, depth, _ = _room(320, 240)

    class XmasEngine(FakeEngine):
        device = "cuda"
        plan = MP.plan("cpu")

        def generate(self, image, mask, control, prompt, seed, *a, **kw):
            super().generate(image, mask, control, prompt, seed, *a, **kw)
            return _xmas(len(self.calls) == 3, *image.shape[:2])          # só o 3º candidato tem árvore

    eng = XmasEngine()
    res = api.redesign(rgb, "evento", None, evento="natal", n=1, seed=5, out_dir=tmp_path, engine=eng,
                       depth=depth, groups=g, rank_fn=lambda im: np.full(im.shape[:2], WALL, np.uint8),
                       refine=False, tone=False)
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    assert len(eng.calls) == 3 and rep["candidatos"]["escolhidos"] == [2]
    assert rep["candidatos"]["notas"][2]["arvore_arvore"] > 0.8


def test_score_penalizes_cut_furniture_at_border():
    from mobilia.design.quality import cut_pieces, score

    g = np.full((240, 320), FLOOR, np.uint8)
    g[100:200, 80:220] = FURN                                             # sofá inteiro no meio
    lock = np.zeros_like(g, bool)
    clean = score(g, lock)
    cut = g.copy(); cut[200:240, 290:320] = FURN                          # pedaço cortado no canto de baixo
    bad = score(cut, lock)
    assert bad["cortados"] > 0 and clean["cortados"] == 0 and bad["nota"] < clean["nota"]
    src = np.zeros_like(lock); src[200:240, 290:320] = True               # já estava na foto: não pesa
    assert cut_pieces(cut, src) == 0.0
    big = g.copy(); big[60:240, 0:200] = FURN                              # móvel grande saindo do quadro: normal
    assert cut_pieces(big) == 0.0
    p = P.build_prompt("mobiliar", "moderno", "sala", 0)
    assert "cut off" in p.negative
