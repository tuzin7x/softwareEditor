"""Correções da rodada 2 da sala vazia do modo B (rápidos, sem modelos).

- protegidos (janela/porta atrás do móvel) também na estratégia "imagem" e sem modelo da sala;
- véu/fumaça por mancha (``haze_metric``), inclusive onde não há plano medido;
- material do piso contra o piso visível sem tapete (``floor_material_metric``), e o tapete fora do contexto do piso;
- regra do produto: planejado mantido ou planejado/louça que ficou na foto reprova a placa;
- VRAM do SDXL: mínimo que cabe o UNet inteiro e troca da LaMa/OneFormer para a CPU durante a geração.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from mobilia.analysis.segmentation import GROUP_IDS
from mobilia.removal import plate as rp
from mobilia.removal import plate_score as ps

from test_plate import _inst_set, _piece, _render, _room

G_FLOOR, G_WALL, G_WIN = GROUP_IDS["floor"], GROUP_IDS["wall"], GROUP_IDS["window"]


# ------------------------------------------------------------------------------------------- protegidos
def _window_scene():
    W, H = 480, 360
    room = _room(W, H)
    truth, groups, _ = _render(room, W, H)
    groups = groups.copy()
    win = np.zeros((H, W), bool)
    win[90:240, 150:300] = True
    groups[win] = G_WIN
    photo = truth.copy()
    photo[win] = (180, 210, 240)
    sofa = np.zeros((H, W), bool)
    sofa[180:300, 120:330] = True
    photo[sofa] = (60, 70, 90)
    return room, photo, groups, win, sofa


@pytest.mark.parametrize("strategy", rp.STRATEGIES)
def test_window_behind_furniture_is_protected_in_every_strategy(strategy):
    """Revisão de código (rodada 2): na estratégia "imagem" nenhum pixel mascarado era "protegido", e o refino SDXL
    repintava a janela escondida pelo sofá. Agora as duas estratégias rotulam igual."""
    room, photo, groups, win, sofa = _window_scene()
    seg = SimpleNamespace(groups=groups)
    res = rp.build_plate(photo, [_piece(sofa, "solto", "sofa")], seg, room, inpaint=rp.telea_inpaint,
                         strategy=strategy, assess_quality=False)
    ref = rp.build_plate(photo, [_piece(sofa, "solto", "sofa")], seg, room, inpaint=rp.telea_inpaint,
                         strategy="planos", assess_quality=False)
    prot = res.mask & (res.labels == rp.PROTECTED)
    assert prot.sum() > 500
    if strategy == "geometria":
        # a geometria ainda estende a abertura pela parede logo abaixo dela (até o peitoril/chão, ver
        # ``geofill.continue_openings``): contém os protegidos das outras, sempre perto da janela
        assert not (res.mask & ref.mask & (ref.labels == rp.PROTECTED) & ~prot).any()
    else:
        assert np.array_equal(prot, res.mask & (ref.labels == rp.PROTECTED))
    assert (prot & ~rp._dil(win, 10)).sum() == 0          # só perto da janela


def test_window_behind_furniture_is_protected_without_room_model():
    _, photo, groups, win, sofa = _window_scene()
    res = rp.build_plate(photo, [_piece(sofa, "solto", "sofa")], SimpleNamespace(groups=groups), None,
                         inpaint=rp.telea_inpaint, assess_quality=False)
    prot = res.mask & (res.labels == rp.PROTECTED)
    assert prot.sum() > 500 and (prot & ~rp._dil(win, 10)).sum() == 0
    # e o refino SDXL não toca nesses pixels (por padrão, ``refine_protected_mask=False``)
    from mobilia.gen import sdxl as G

    noisy = lambda im, m, c, p, s, st: np.random.default_rng(s).integers(0, 255, im.shape, dtype=np.uint8)  # noqa
    seen = []
    G.refine_plate(photo, res, SimpleNamespace(groups=groups), None, None, noisy, None,
                   cfg=G.RefineConfig(seeds=1, work_side=480, min_crop_frac=1.0), on_seed=seen.append)
    assert seen and all(np.array_equal(sr.plate[prot], res.plate[prot]) for sr in seen)


# ------------------------------------------------------------------------------------------- véu / fumaça
def _wall_scene(H=300, W=400, seed=0):
    """Parede com grão e rampa de luz; máscara retangular no meio; tudo "parede 0"."""
    rng = np.random.default_rng(seed)
    xs = np.arange(W)[None].repeat(H, 0).astype(np.float32)
    base = 170.0 - 25.0 * xs / W
    img = np.clip(base[..., None] + rng.normal(0, 2.5, (H, W, 3)), 0, 255)
    mask = np.zeros((H, W), bool)
    mask[60:250, 90:330] = True
    labels = np.zeros((H, W), np.int32)
    groups = np.full((H, W), G_WALL, np.uint8)
    return img, mask, labels, groups


def _haze(img, mask, labels, groups):
    plate = np.clip(img, 0, 255).round().astype(np.uint8)
    ref = ps.plane_reference(plate, mask, labels, groups)
    return ps.haze_metric(plate, mask, labels, groups, ref)


def test_haze_clean_fill_and_light_ramp_are_not_veil():
    img, mask, labels, groups = _wall_scene()
    assert _haze(img, mask, labels, groups)["severidade"] == 0.0


def test_haze_dark_smooth_blob_is_veil_even_without_measured_plane():
    img, mask, labels, groups = _wall_scene()
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.exp(-(((yy - 155) / 45.0) ** 2 + ((xx - 210) / 60.0) ** 2))
    # fumaça escura e lisa (sem o grão da parede), localizada, dentro do preenchimento
    smoke = np.where(mask[..., None], (170.0 - 25.0 * xx / W)[..., None] - 32.0 * blob[..., None], img)
    for lbl in (labels, np.full_like(labels, rp.OTHER)):         # com plano medido e "sem plano"
        h = _haze(smoke, mask, lbl, groups)
        assert h["severidade"] > 0.6, h


def test_haze_bright_reflection_that_keeps_texture_is_not_veil():
    """Reflexo da janela num piso encerado: mancha clara de verdade, mas com o grão do material."""
    img, mask, labels, groups = _wall_scene()
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.exp(-(((yy - 155) / 45.0) ** 2 + ((xx - 210) / 30.0) ** 2))
    rng = np.random.default_rng(3)
    glare = img + (20.0 * blob)[..., None] + rng.normal(0, 4.0, img.shape) * blob[..., None]
    assert _haze(glare, mask, labels, groups)["severidade"] < 0.6


# ------------------------------------------------------------------------------------------- material do piso
def _floor_scene(H=300, W=400, seed=1):
    """Piso de "tábuas" (listras verticais com grão) em tudo; máscara no meio; rótulo piso."""
    rng = np.random.default_rng(seed)
    xs = np.arange(W)[None].repeat(H, 0)
    planks = 120.0 + 18.0 * ((xs // 12) % 2) + rng.normal(0, 3.0, (H, W))
    img = np.repeat(planks[..., None], 3, 2)
    mask = np.zeros((H, W), bool)
    mask[80:260, 100:300] = True
    labels = np.full((H, W), rp.FLOOR, np.int32)
    groups = np.full((H, W), G_FLOOR, np.uint8)
    return img, mask, labels, groups


def _u8(a):
    return np.clip(a, 0, 255).round().astype(np.uint8)


def test_floor_material_same_floor_passes_blurred_floor_fails():
    import cv2

    img, mask, labels, groups = _floor_scene()
    assert ps.floor_material_metric(_u8(img), mask, labels, groups)["severidade"] == 0.0
    blur = img.copy()
    blur[mask] = cv2.GaussianBlur(img, (0, 0), 8)[mask]          # tábuas viraram borrão marrom
    m = ps.floor_material_metric(_u8(blur), mask, labels, groups)
    assert m["severidade"] > 0.6 and m["energia_log2"] < -1.4, m


def test_floor_reference_excludes_rug():
    """Sob um tapete grande, a referência é o piso em volta dele, não a estampa do tapete."""
    img, mask, labels, groups = _floor_scene()
    H, W = mask.shape
    rug = np.zeros((H, W), bool)
    rug[30:290, 40:360] = True
    rug &= ~mask
    plain = img.copy()
    plain[~rug] = 128.0 + np.random.default_rng(2).normal(0, 1.0, (int((~rug).sum()), 3))  # piso liso de verdade
    rng = np.random.default_rng(5)
    plain[rug] = (rng.integers(0, 2, (int(rug.sum()), 1)) * 120 + 60).astype(float)   # estampa forte do tapete
    fill_plain = plain.copy()
    # preenchimento liso (certo: o piso de baixo é o liso visível em volta do tapete)
    with_rug = ps.floor_material_metric(_u8(fill_plain), mask, labels, groups, rug=rug)
    assert with_rug["severidade"] < 0.3, with_rug


def test_rug_is_never_floor_context_in_plane_fill():
    """O grupo "piso" da segmentação inclui o tapete; o preenchimento do piso não pode copiar a cor dele."""
    room = _room()
    img, groups, box = _render(room)
    H, W = groups.shape
    floor = box.label == rp.FLOOR
    rug = floor & (np.arange(H)[:, None] > 300)                   # faixa de tapete vermelho na frente
    img = img.copy()
    img[rug] = (200, 30, 30)
    sofa = np.zeros((H, W), bool)
    sofa[255:300, 150:330] = True                                 # o sofá fica logo atrás do tapete
    names = {0: "floor", 1: "rug", 2: "wall"}
    lbl = np.where(rug, 1, np.where(floor, 0, 2)).astype(np.uint8)
    seg = SimpleNamespace(groups=groups, labels=lbl, id2label=names)
    r = rp.build_plate(img, [_piece(sofa, "solto", "sofa")], seg, room, inpaint=rp.telea_inpaint,
                       assess_quality=False)
    filled = r.plate[r.mask & floor & ~rp._dil(rug, 6)].astype(float)
    red = (filled[:, 0] - filled[:, 1]) > 60
    assert red.mean() < 0.05


# ------------------------------------------------------------------------------------------- regra do produto
def test_leftover_fixtures_reports_undetected_toilet_only_outside_mask():
    H, W = 200, 200
    lbl = np.zeros((H, W), np.uint8)
    lbl[100:160, 20:70] = 5                                       # vaso 7,5%
    lbl[20:40, 150:170] = 6                                       # cuba pequena (1%): abaixo do mínimo
    seg = SimpleNamespace(labels=lbl, id2label={0: "wall", 5: "toilet", 6: "sink"})
    mask = np.zeros((H, W), bool)
    left = rp.leftover_fixtures(seg, mask)
    assert [x["rotulo"] for x in left] == ["toilet"] and left[0]["area_pct"] == pytest.approx(7.5, abs=0.1)
    mask[100:160, 20:70] = True                                   # removido: nada sobra
    assert rp.leftover_fixtures(seg, mask) == []
    keep = np.zeros((H, W), bool)
    keep[100:160, 20:70] = True                                   # mantido de propósito (já tem motivo próprio)
    assert rp.leftover_fixtures(seg, np.zeros((H, W), bool), keep) == []
    assert rp.leftover_fixtures(SimpleNamespace(groups=None), mask) == []   # sem rótulos: não mede
    sliver = np.zeros((H, W), np.uint8)
    sliver[:, 100:104] = 5                                        # tira de 4 px (2% da foto) na borda de um armário
    assert rp.leftover_fixtures(SimpleNamespace(labels=sliver, id2label={5: "cabinet"}), np.zeros((H, W), bool)) == []


def test_kept_builtins_never_approve(monkeypatch):
    """Banheiro da avaliação: aprovado com vaso, cuba e gabinete mantidos. No modo B a sala é crua: placa com
    planejados mantidos reprova (fica só para "gerar mesmo assim"), com o motivo."""
    room = _room()
    img, groups, _ = _render(room)
    H, W = groups.shape
    loose = np.zeros((H, W), bool)
    loose[250:320, 60:140] = True
    built = np.zeros((H, W), bool)
    built[150:240, 250:400] = True
    insts = [_piece(loose, "solto", "chair"), _piece(built, "planejado", "toilet")]
    calls = []

    def fake_found(seg2, pan2, mask, **kw):                        # a crua "reencontra" o móvel: reprova
        calls.append(1)
        return [{"rotulo": "cabinet", "area_pct": 3.0}] if len(calls) == 1 else []

    monkeypatch.setattr(rp, "redetected_furniture", fake_found)
    r = rp.build_plate_gated(img, _inst_set(insts), SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                             redetect=lambda p: (None, None), alternatives=False)
    kept = [t for t in r.info["tentativas"] if t["planejados_mantidos"]]
    assert kept and all(not t["aprovada"] for t in kept)
    assert all(any("não está crua" in m for m in t["motivos"]) for t in kept)
    assert not (r.kept_builtins and r.quality.ok)


def test_fail_quality_caps_score_and_is_idempotent():
    q = rp.PlateQuality(90.0, True, {}, {}, 0.1, raw_score=90.0)
    rp.fail_quality(q, "x")
    rp.fail_quality(q, "x")
    assert not q.ok and q.reasons == ["x"] and q.score == pytest.approx(54.0)


# ------------------------------------------------------------------------------------------- VRAM do SDXL
def test_sdxl_vram_minimum_fits_whole_unet():
    from mobilia.hardware import SDXL_VRAM_MIN_MB

    unet_fp16_mb = 5.14e9 / 2**20                                 # LICENCAS_V3.md
    assert SDXL_VRAM_MIN_MB >= unet_fp16_mb + 300 + 1000         # + ControlNet + ativações (estimativa)


def test_sdxl_recheck_counts_vram_freed_by_swap(monkeypatch):
    from mobilia import pipeline_b as pb

    monkeypatch.setattr(pb, "cuda_free_mb", lambda dev: 5800)
    use, off, why, free = pb.sdxl_recheck("cuda", False)
    assert not use and free == 5800 and "mínimo" in why
    use, off, why, free = pb.sdxl_recheck("cuda", False, reclaim_mb=900)      # LaMa + OneFormer vão para a CPU
    assert use and off and free == 6700 and "offload" in why
    use, off, why, free = pb.sdxl_recheck("cuda", False, reclaim_mb=1300)
    assert use and not off and free == 7100


def test_vram_swap_moves_models_out_during_generate_and_back():
    pytest.importorskip("torch")                  # a troca de VRAM chama o torch (não está no CI)
    from mobilia import pipeline_b as pb

    moves = []

    class Mod:
        def __init__(self, name):
            self.name = name

        def to(self, dev):
            moves.append((self.name, dev))
            return self

        def parameters(self):
            return []

        def buffers(self):
            return []

    sw = pb.VramSwap("cuda", [Mod("lama"), None, Mod("oneformer")])
    seen = []

    def gen(*a, **k):
        seen.append(list(moves))
        return "ok"

    assert sw.wrap(gen)(1, 2) == "ok"
    assert seen[0] == [("lama", "cpu"), ("oneformer", "cpu")]
    assert moves[-2:] == [("lama", "cuda"), ("oneformer", "cuda")]

    def boom(*a, **k):
        raise RuntimeError("falta de memória")

    moves.clear()
    with pytest.raises(RuntimeError):
        sw.wrap(boom)()
    assert moves[-2:] == [("lama", "cuda"), ("oneformer", "cuda")]     # volta mesmo com erro
    cpu = pb.VramSwap("cpu", [Mod("lama")])
    assert cpu.wrap(gen) is gen and cpu.reclaim_mb() == 0
