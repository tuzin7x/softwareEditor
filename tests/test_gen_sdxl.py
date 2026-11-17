"""PT-V3-4: refino SDXL da sala vazia e nota contínua do portão (rápidos: difusão substituída por stubs).

Garantias testadas: fora da máscara e nas janelas/portas visíveis o resultado é ``array_equal`` à placa LaMa;
gerador que falha ou não melhora = fallback para a LaMa; a semente escolhida é a de melhor nota; a nota cai de
forma monotônica com névoa e fantasma sintéticos; a integração no portão e no pipeline registra tudo no relatório.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from mobilia.analysis.segmentation import GROUP_IDS
from mobilia.gen import sdxl as G
from mobilia.removal import plate as rp
from mobilia.removal import plate_score as ps
from test_plate import _piece, _render, _room

G_WIN = GROUP_IDS["window"]
W, H = 480, 360


def _scene():
    """Sala sintética vazia (verdade), a mesma com um "sofá" escuro no piso e um "quadro" na parede (foto) e a
    placa LaMa feita pelo Telea (inpainter clássico, com borrão e sombra do sofá)."""
    room = _room(W, H)
    truth, groups, box = _render(room, W, H)
    photo = truth.copy()
    sofa = np.zeros((H, W), bool)
    sofa[250:330, 120:330] = True
    photo[sofa] = (60, 70, 90)
    photo[250:262, 120:330] = (200, 200, 205)          # encosto claro: estrutura interna da peça
    art = np.zeros((H, W), bool)
    art[110:170, 300:400] = True
    photo[art] = (30, 90, 40)
    pieces = [_piece(sofa, "solto", "sofa"), _piece(art, "solto", "painting")]
    seg = SimpleNamespace(groups=groups)
    res = rp.build_plate(photo, pieces, seg, room, inpaint=rp.telea_inpaint)
    return room, truth, photo, groups, box, seg, res


@pytest.fixture(scope="module")
def scene():
    return _scene()


def _truth_generator(truth, fog=0.0, seed_fog=None, calls=None):
    """Stub do SDXL que devolve a sala vazia verdadeira (na resolução de trabalho), opcionalmente com névoa."""
    def gen(image, mask, control, prompt, seed, strength):
        import cv2

        if calls is not None:
            calls.append((image.shape, seed, strength, prompt.text))
        t = cv2.resize(truth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
        f = fog if seed_fog is None else seed_fog.get(seed, 0.0)
        if f > 0:
            t = (1 - f) * cv2.GaussianBlur(t, (0, 0), 6) + f * 245.0
        return np.clip(t, 0, 255).round().astype(np.uint8)

    return gen


def _cfg(**kw):
    c = G.RefineConfig(seeds=2, work_side=max(W, H), min_crop_frac=1.0)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ------------------------------------------------------------------------------------------- invariantes
def test_outside_mask_and_windows_are_untouched(scene):
    room, truth, photo, groups, box, seg, res = scene
    groups = groups.copy()
    win = np.zeros((H, W), bool)
    win[60:200, 20:100] = True                          # janela visível (fora da máscara)
    groups[win & ~res.mask] = G_WIN
    seg2 = SimpleNamespace(groups=groups)
    labels = res.labels.copy()
    prot_in = res.mask & (np.arange(W)[None] < 140)     # pedaço mascarado rotulado "protegido"
    labels[prot_in] = rp.PROTECTED
    res2 = rp.PlateResult(res.plate.copy(), res.mask, res.core, res.effect, res.shadow, res.light, labels,
                          res.quality, dict(res.info))
    seen = []

    def noisy(image, mask, control, prompt, seed, strength):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 255, image.shape, dtype=np.uint8)

    for refine_prot in (True, False):
        out = G.refine_plate(photo, res2, seg2, room, None, noisy, None, seed=3,
                             cfg=_cfg(refine_protected_mask=refine_prot), on_seed=seen.append)
        assert np.array_equal(out.plate[~res.mask], res.plate[~res.mask])
        for sr in seen:
            assert np.array_equal(sr.plate[~res.mask], res.plate[~res.mask])   # fora da máscara
            assert np.array_equal(sr.plate[win], res.plate[win])               # janela visível
            if not refine_prot:
                assert np.array_equal(sr.plate[prot_in], res.plate[prot_in])
            # a rampa é por dentro: o 1º pixel dentro da máscara ainda é quase a LaMa
            edge = res.mask & ~rp._ero(res.mask, 1)
            assert np.abs(sr.plate[edge].astype(int) - res.plate[edge].astype(int)).mean() < 40
        seen.clear()


def test_generator_failure_falls_back_to_lama(scene):
    room, truth, photo, groups, box, seg, res = scene

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory (simulado)")

    out = G.refine_plate(photo, res, seg, room, None, boom, None, cfg=_cfg())
    assert np.array_equal(out.plate, res.plate)
    info = out.info["refino_sdxl"]
    assert not info["usado"] and "LaMa" in info["motivo"] and "memory" in info["erros"][0]
    out = G.refine_plate(photo, res, seg, room, None, None, None, cfg=_cfg())
    assert np.array_equal(out.plate, res.plate) and out.info["refino_sdxl"]["motivo"] == "sem gerador"


def test_no_improvement_keeps_lama(scene):
    room, truth, photo, groups, box, seg, res = scene

    def black(image, *a):
        return np.zeros_like(image)

    out = G.refine_plate(photo, res, seg, room, None, black, None, cfg=_cfg())
    info = out.info["refino_sdxl"]
    assert not info["usado"] and ("não melhorou" in info["motivo"] or "falha dura" in info["motivo"])
    assert np.array_equal(out.plate, res.plate)
    assert len(info["sementes"]) == 2 and all(s["nota"] is not None for s in info["sementes"])


def test_seed_selection_picks_clean_seed(scene):
    room, truth, photo, groups, box, seg, res = scene
    cfg = _cfg(seeds=3, floor_detail="sdxl")      # escolha pela textura do próprio SDXL (névoa no piso)
    seeds = [3 * 1000 + 17 * k + 1 for k in range(3)]
    calls = []
    gen = _truth_generator(truth, seed_fog={seeds[0]: 0.6, seeds[1]: 0.0, seeds[2]: 0.35}, calls=calls)
    out = G.refine_plate(photo, res, seg, room, None, gen, None, seed=3, cfg=cfg)
    info = out.info["refino_sdxl"]
    assert len(calls) == 3 and all(c[2] == cfg.strength for c in calls)
    assert "empty" in calls[0][3]
    notas = {s["semente"]: s["nota_sem_teto"] for s in info["sementes"]}
    assert notas[seeds[1]] > max(notas[seeds[2]], notas[seeds[0]])
    assert info["usado"] and info["semente"] == seeds[1]
    assert out.quality.raw_score > res.quality.raw_score
    assert np.array_equal(out.plate[~res.mask], res.plate[~res.mask])
    assert out.mask is res.mask                                     # só a placa e a nota mudam


def test_seed_with_new_object_is_never_chosen(scene, monkeypatch):
    """O OneFormer acha uma porta nova (ou um móvel) na placa refinada: a semente é descartada e fica a LaMa."""
    room, truth, photo, groups, box, seg, res = scene
    door = np.zeros((H, W), bool)
    door[262:320, 200:240] = True                       # dentro da máscara do sofá, longe de qualquer abertura
    assert (door & res.mask).sum() > 0.8 * door.sum()

    def redetect(plate):
        g = groups.copy()
        if not np.array_equal(plate, res.plate):
            g[door] = GROUP_IDS["door"]
        return SimpleNamespace(groups=g), None

    monkeypatch.setattr(rp, "redetected_furniture", lambda *a, **k: [])
    base = rp.assess(photo, res.plate, res.mask, res.labels, groups, box, redetect, core=res.core)
    res.quality = base
    out = G.refine_plate(photo, res, seg, room, None, _truth_generator(truth), redetect, cfg=_cfg())
    info = out.info["refino_sdxl"]
    assert not info["usado"] and "objeto novo" in info["motivo"]
    assert all("porta/janela nova" in " ".join(s["motivos"]) for s in info["sementes"])
    assert np.array_equal(out.plate, res.plate)


# ------------------------------------------------------------------------------------------- nota contínua
def _fogged(truth, plate_mask, f):
    import cv2

    if f <= 0:
        return truth.copy()
    t = truth.astype(np.float32)
    fog = (1 - f) * cv2.GaussianBlur(t, (0, 0), 1 + 8 * f) + f * 245.0
    out = truth.copy()
    out[plate_mask] = np.clip(fog, 0, 255).round().astype(np.uint8)[plate_mask]
    return out


def _ghosted(truth, photo, plate_mask, a):
    out = truth.copy()
    g = (1 - a) * truth.astype(np.float32) + a * photo.astype(np.float32)
    out[plate_mask] = np.clip(g, 0, 255).round().astype(np.uint8)[plate_mask]
    return out


def test_score_is_monotonic_in_fog_and_ghost(scene):
    room, truth, photo, groups, box, seg, res = scene
    m = res.mask

    def q(plate):
        return rp.assess(photo, plate, m, res.labels, groups, box, None, core=res.core)

    clean = q(_fogged(truth, m, 0.0))
    fogs = [q(_fogged(truth, m, f)) for f in (0.08, 0.2, 0.4)]
    ghosts = [q(_ghosted(truth, photo, m, a)) for a in (0.1, 0.25, 0.5)]
    raw = lambda x: x.raw_score  # noqa: E731
    assert clean.ok and clean.score >= 80, (clean.score, clean.components, clean.reasons)
    assert raw(clean) > raw(fogs[0]) > raw(fogs[1]) > raw(fogs[2])
    assert raw(clean) > raw(ghosts[0]) > raw(ghosts[1]) > raw(ghosts[2])
    fog_sev = [x.realism["nevoa"]["severidade"] for x in fogs]
    ghost_sev = [x.realism["fantasma"]["severidade"] for x in ghosts]
    assert fog_sev == sorted(fog_sev) and fog_sev[-1] > 0.5
    assert ghost_sev == sorted(ghost_sev) and ghost_sev[-1] > 0.5
    # o pior caso reprova e a nota continua informativa (contínua, abaixo de 60, não um teto fixo)
    for bad in (fogs[-1], ghosts[-1]):
        assert not bad.ok and bad.score < 60
    assert len({round(x.score, 1) for x in fogs + ghosts}) == 6
    assert "nevoa" in fogs[-1].components and "fantasma" in ghosts[-1].components
    d = fogs[-1].to_dict()
    assert {"nota_sem_teto", "pontos_perdidos", "realismo", "aberturas_novas"} <= set(d)


def test_ghost_ignores_fill_that_matches_photo_background(scene):
    """Onde a "peça" da foto já mostra o próprio fundo (máscara folgada, vão entre pés), a placa certa é igual à
    foto: isso não é fantasma."""
    room, truth, photo, groups, box, seg, res = scene
    ref = ps.plane_reference(truth, res.mask, res.labels, groups)
    g = ps.ghost_metric(truth, truth, res.mask, res.core, ref, labels=res.labels)
    assert g["severidade"] < 0.1


def test_new_openings_far_from_existing_windows():
    mask = np.zeros((100, 100), bool)
    mask[40:90, 10:90] = True
    groups = np.zeros((100, 100), np.uint8)
    groups[0:30, 0:20] = G_WIN                              # janela da foto
    g2 = groups.copy()
    g2[60:85, 50:70] = GROUP_IDS["door"]                    # porta nova no meio do preenchimento
    g2[30:42, 2:18] = G_WIN                                # continuação da janela (perto): não conta
    out = ps.new_openings(SimpleNamespace(groups=g2), mask, groups)
    assert [o["rotulo"] for o in out] == ["porta"]


# ------------------------------------------------------------------------------------------- partes
def test_prompt_and_materials():
    rng = np.random.default_rng(0)
    H2, W2 = 200, 300
    img = np.zeros((H2, W2, 3), np.uint8)
    groups = np.full((H2, W2), 255, np.uint8)
    img[:100] = (235, 235, 232)
    groups[:100] = GROUP_IDS["wall"]
    stripes = (np.sin(np.arange(W2) / 3.0)[None] * 25).astype(np.float32)
    wood = np.array([150, 105, 70], np.float32)[None, None] + stripes[..., None] + rng.normal(0, 1, (100, W2, 3))
    img[100:] = np.clip(wood, 0, 255).astype(np.uint8)
    groups[100:] = GROUP_IDS["floor"]
    mask = np.zeros((H2, W2), bool)
    mask[80:160, 100:200] = True
    d = G.describe_surfaces(img, mask, groups)
    assert d["wall_color"] == "white" and d["floor_kind"] == "wooden plank floor"       # só para o relatório
    p = G.build_prompt(d, "quarto")
    assert "empty unfurnished bedroom" in p.text and "white" in p.text and d["floor_color"] in p.text
    # o prompt nunca nomeia o tipo do piso: "tiled"/"plank" viravam ordem (placas com rejunte sobre porcelanato)
    for word in ("tile", "tiled", "plank", "wooden", "slate", "marble"):
        assert word not in p.text.lower()
    for word in ("furniture", "new window", "new door", "power outlet", "fog", "text"):
        assert word in p.negative
    assert len(p.negative.split(",")) <= 32                             # cabe nos 77 tokens do CLIP
    assert G.build_prompt(G.describe_surfaces(img, mask, None)).text.count("floor") >= 1


def test_empty_room_depth_ignores_furniture(scene):
    """O controle de profundidade dentro da máscara vem da caixa vazia: um "sofá" bem perto da câmera no MoGe
    não aparece no mapa (a profundidade continua a do piso)."""
    room, truth, photo, groups, box, seg, res = scene
    depth = box.t.copy().astype(np.float32)
    depth[res.core] = 0.8                                             # o móvel, perto da câmera
    c = G.empty_room_depth((H, W), res.mask, room, depth)
    c_true = G.empty_room_depth((H, W), res.mask, room, box.t.astype(np.float32))
    sel = rp._ero(res.core, 4)
    assert np.abs(c[sel] - c_true[sel]).max() < 0.05
    assert c.min() >= 0 and c.max() <= 1
    col = c[:, W // 2]
    assert col[H - 5] > col[H // 2 + 20]                              # piso perto (embaixo) mais claro
    # sem modelo da sala: o buraco vira interpolação suave do MoGe (sem o móvel)
    c2 = G.empty_room_depth((H, W), res.mask, None, depth)
    assert np.isfinite(c2).all() and np.abs(c2[sel] - c_true[sel]).mean() < 0.25


def test_crop_and_work_size():
    m = np.zeros((600, 900), bool)
    m[500:560, 800:880] = True
    y0, y1, x0, x1 = G.crop_box(m, 0.5)
    assert y0 <= 500 and y1 >= 560 and x0 <= 800 and x1 >= 880
    assert max(y1 - y0, x1 - x0) >= 450 and y1 <= 600 and x1 <= 900
    w, h = G.work_size(y1 - y0, x1 - x0, 768)
    assert w % 8 == 0 and h % 8 == 0 and max(w, h) <= 776


# ------------------------------------------------------------------------------------------- portão + pipeline
def test_gate_runs_refine_after_each_attempt_and_records_it(monkeypatch):
    room = _room(W, H)
    img, groups, _ = _render(room, W, H)
    m = np.zeros((H, W), bool)
    m[250:320, 60:140] = True
    insts = [_piece(m, "solto", "sofa")]
    inst_set = SimpleNamespace(instances=insts, removal=lambda keep=False: insts)
    seen = []

    def refine(r, name):
        seen.append(name)
        r.info["refino_sdxl"] = {"usado": True, "semente": 42, "motivo": "teste", "tempo_s": 0.1}
        return r

    r = rp.build_plate_gated(img, inst_set, SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                             refine=refine)
    assert seen and seen[0] == "crua_por_plano"
    t = r.report()["placa"]["tentativas"][0]
    assert t["refino_sdxl"]["usado"] and t["refino_sdxl"]["semente"] == 42 and t["refino_sdxl"]["nota_lama"] is not None
    assert r.report()["placa"]["refino_sdxl"]["semente"] == 42


def test_sdxl_choice_env_and_profile(monkeypatch):
    from mobilia import pipeline_b as pb
    from mobilia.hardware import GpuInfo, HardwareReport

    cpu = HardwareReport(os="t", cpu="t", cpu_threads=8, ram_gb=32, gpus=[], torch_cuda=False)
    monkeypatch.setenv(G.ENV_SDXL, "0")
    assert pb.sdxl_choice(cpu)[0] is False
    monkeypatch.setenv(G.ENV_SDXL, "auto")
    use, _, why = pb.sdxl_choice(cpu)
    assert not use and "GPU" in why
    gpu = HardwareReport(os="t", cpu="t", cpu_threads=8, ram_gb=32, torch_cuda=True,
                         gpus=[GpuInfo("RTX 5070 Laptop GPU", 8151, 7400, "591.10")])
    monkeypatch.setattr("mobilia.analysis.weights.sdxl_installed", lambda *a, **k: True)
    use, off, why = pb.sdxl_choice(gpu)
    assert use and off                                               # 8 GB: offload (OneFormer divide a VRAM)
    monkeypatch.setattr("mobilia.analysis.weights.sdxl_installed", lambda *a, **k: False)
    assert pb.sdxl_choice(gpu)[0] is False
    monkeypatch.setenv(G.ENV_SDXL, "1")
    monkeypatch.setattr("mobilia.analysis.weights.sdxl_installed", lambda *a, **k: True)
    use, off, why = pb.sdxl_choice(cpu)
    assert use and not off and "forçado" in why


# ------------------------------------------------------------------------------------------- ponta a ponta
class _StubRefiner:
    """Gerador sem modelo: devolve a imagem de partida (o "início misto" da sala vazia) levemente borrada."""

    def __init__(self):
        self.calls, self.released = [], False

    def __call__(self, image, mask, control, prompt, seed, strength):
        import cv2

        self.calls.append({"tamanho": [image.shape[1], image.shape[0]], "semente": seed})
        assert control.shape == mask.shape and 0 <= control.min() and control.max() <= 1
        return cv2.GaussianBlur(image, (0, 0), 0.7)

    def release(self):
        self.released = True


@pytest.mark.parametrize("sdxl_on", [True, False])
def test_mode_b_pipeline_with_stubbed_sdxl(tmp_path, monkeypatch, sdxl_on):
    """Pipeline do modo B com o perfil "B com SDXL" (gerador stub): o refino roda depois da LaMa, o relatório
    registra semente/notas e o vídeo continua passando em todas as verificações §3.5. Com o SDXL desligado, o
    pipeline segue igual ao de antes (só LaMa) e o relatório diz por quê."""
    import json
    from pathlib import Path

    from mobilia import pipeline, pipeline_b
    from mobilia.jobs import ProgressEmitter
    from mobilia.video import encoder as E
    from test_pipeline_mode_b import ROOMS, _fast_models, _fixture_analysis, _job, _tiny_quality

    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(E.ENV_CODEC, "libx264")
    monkeypatch.setenv("MOBILIA_MODO_B_MKV", "0")
    _tiny_quality(monkeypatch, pipeline, 384)
    _fixture_analysis(monkeypatch, "sala_sofas_brancos")
    _fast_models(monkeypatch, gate_ok=True)
    stub = _StubRefiner()
    if sdxl_on:
        monkeypatch.setattr(pipeline_b, "sdxl_choice", lambda hw: (True, True, "teste"))
        monkeypatch.setattr(pipeline_b, "load_refiner",
                            lambda dev, off: (stub, G.RefineConfig(seeds=2, work_side=256)))
    else:
        monkeypatch.setenv(G.ENV_SDXL, "0")
    res = pipeline.run_job(_job(tmp_path, ROOMS / "sala_sofas_brancos" / "photo.jpg"), ProgressEmitter())
    rep = json.loads(Path(res.report_path).read_text(encoding="utf-8"))
    assert rep["verificacao_modo_b"]["ok"]
    sd = rep["refino_sdxl"]
    if sdxl_on:
        assert sd["ligado"] and len(stub.calls) == 2 and stub.released
        assert len(sd["sementes"]) == 2 and all(s["nota"] is not None for s in sd["sementes"])
        assert sd["nota_lama"] is not None and sd["resultado"]
        assert rep["placa"]["tentativas"][0]["refino_sdxl"] is not None
        assert rep["semente_sdxl"] == sd["semente"]
    else:
        assert not sd["ligado"] and not sd["usado"] and "desligado" in sd["motivo"] and not stub.calls


# ------------------------------------------------------------------------------------------- correções rodada 1
def test_error_while_scoring_a_seed_falls_back_to_lama(scene):
    """Falta de VRAM no OneFormer da nota (ou qualquer erro depois da geração) não derruba o trabalho."""
    room, truth, photo, groups, box, seg, res = scene
    res.quality = rp.assess(photo, res.plate, res.mask, res.labels, groups, box, None, core=res.core)

    def redetect(plate):
        raise RuntimeError("CUDA out of memory (simulado, na nota)")

    out = G.refine_plate(photo, res, seg, room, None, _truth_generator(truth), redetect, cfg=_cfg())
    assert np.array_equal(out.plate, res.plate)
    info = out.info["refino_sdxl"]
    assert not info["usado"] and "memory" in info["motivo"] and "LaMa" in info["motivo"]


def test_gate_survives_refine_exception(monkeypatch):
    room = _room(W, H)
    img, groups, _ = _render(room, W, H)
    m = np.zeros(groups.shape, bool)
    m[250:320, 60:140] = True
    insts = [_piece(m, "solto", "sofa")]
    inst_set = SimpleNamespace(instances=insts, removal=lambda keep=False: insts)

    def refine(r, name):
        raise MemoryError("sem memória (simulado)")

    r = rp.build_plate_gated(img, inst_set, SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                             refine=refine, alternatives=False)
    assert r.quality is not None
    t = r.info["tentativas"][0]["refino_sdxl"]
    assert t["usado"] is False and "MemoryError" in t["motivo"]


def test_objects_already_in_lama_do_not_discard_seeds(scene, monkeypatch):
    """A reconferência acha "toalha" na própria placa LaMa: isso não é culpa do SDXL e não descarta as sementes."""
    room, truth, photo, groups, box, seg, res = scene
    monkeypatch.setattr(rp, "redetected_furniture", lambda *a, **k: [{"rotulo": "towel", "area_pct": 1.0}])
    redetect = lambda p: (SimpleNamespace(groups=groups), None)  # noqa: E731
    res.quality = rp.assess(photo, res.plate, res.mask, res.labels, groups, box, redetect, core=res.core)
    out = G.refine_plate(photo, res, seg, room, None, _truth_generator(truth), redetect, cfg=_cfg())
    info = out.info["refino_sdxl"]
    assert "objeto novo" not in info["motivo"]
    assert G.new_objects(out.quality, res.quality) == []
    q_new = SimpleNamespace(redetected=[{"rotulo": "towel"}, {"rotulo": "chair"}], openings=[])
    assert G.new_objects(q_new, res.quality) == ["chair"]


def _grid(plate, sel, step=34, val=35):
    """Rejunte inventado: retas diagonais finas (antisserrilhadas) em duas direções, só dentro de ``sel``."""
    import cv2

    h, w = plate.shape[:2]
    lines = np.zeros((h, w), np.uint8)
    for k in range(-2 * max(h, w), 2 * max(h, w), step):
        cv2.line(lines, (k, 0), (k + h // 2, h), 255, 1, cv2.LINE_AA)
        cv2.line(lines, (k, h), (k + h // 2 + h, 0), 255, 1, cv2.LINE_AA)
    a = (lines.astype(np.float32) / 255.0) * sel
    out = plate.astype(np.float32) - val * a[..., None]
    return np.clip(out, 0, 255).round().astype(np.uint8)


def test_invented_grout_on_plain_floor_is_a_hard_failure(scene):
    """Rejunte inventado sobre piso liso: a nota média quase não muda, mas a falha dura reprova e a semente
    com rejunte nunca troca a LaMa."""
    room, truth, photo, groups, box, seg, res = scene
    floor = res.mask & (res.labels == rp.FLOOR)
    clean = rp.assess(photo, truth, res.mask, res.labels, groups, box, None, core=res.core)
    tiled = rp.assess(photo, _grid(truth, floor), res.mask, res.labels, groups, box, None, core=res.core)
    assert clean.hard_detail["estrutura_nova"] < 0.2 and clean.hard_detail["material"] < 0.2
    assert max(tiled.hard_detail["estrutura_nova"], tiled.hard_detail["material"]) > 0.6
    assert not tiled.ok and any("estrutura inventada" in r or "material" in r for r in tiled.reasons)
    # o SDXL que desenha placas é descartado mesmo com a nota maior que a de uma LaMa ruim
    res.quality = rp.assess(photo, res.plate, res.mask, res.labels, groups, box, None, core=res.core)

    def tiles(image, mask, *a):
        import cv2

        t = cv2.resize(truth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        return _grid(t, np.ones(t.shape[:2], bool), step=30, val=50)

    out = G.refine_plate(photo, res, seg, room, None, tiles, None, cfg=_cfg(floor_detail="sdxl", scope="tudo"))
    assert not out.info["refino_sdxl"]["usado"]
    assert np.array_equal(out.plate, res.plate)


def test_floor_texture_comes_from_lama_tone_from_sdxl():
    import cv2

    rng = np.random.default_rng(0)
    h, w = 120, 160
    lama = np.clip(120 + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8)        # textura do piso visível
    gen = np.full((h, w, 3), 160, np.uint8)
    gen[::10] = 60                                                                     # rejunte inventado
    sel = np.zeros((h, w), bool)
    sel[20:100, 20:140] = True
    out = G.floor_texture_from_lama(gen, lama, sel, sigma=6.0)
    assert np.array_equal(out[~sel], gen[~sel])
    inner = np.zeros_like(sel)
    inner[40:80, 40:120] = True
    hp = lambda a: a.astype(np.float32) - cv2.GaussianBlur(a.astype(np.float32), (0, 0), 6.0)  # noqa: E731
    # a textura dentro é a da LaMa (não as linhas do SDXL), o tom é o do SDXL (com as linhas diluídas)
    assert np.corrcoef(hp(out)[inner][:, 0], hp(lama)[inner][:, 0])[0, 1] > 0.9
    assert abs(float(out[inner].mean()) - float(gen[inner].mean())) < 6


def test_contour_of_removed_painting_is_caught(scene):
    """O quadro vira um retângulo um pouco mais claro que a parede: a borda fica exatamente no contorno da
    máscara (fora das outras medidas). A medida do contorno pega; a sala verdadeira não."""
    room, truth, photo, groups, box, seg, res = scene
    art = np.zeros((H, W), bool)
    art[110:170, 300:400] = True
    ghost = truth.copy()
    sel = res.mask & rp._dil(art, 3)
    ghost[sel] = np.clip(ghost[sel].astype(int) + 14, 0, 255).astype(np.uint8)
    c_true = ps.contour_metric(truth, res.mask, res.core, res.labels, groups)
    c_ghost = ps.contour_metric(ghost, res.mask, res.core, res.labels, groups)
    assert c_true["severidade"] < 0.2, c_true
    assert c_ghost["severidade"] > 0.6, c_ghost
    q = rp.assess(photo, ghost, res.mask, res.labels, groups, box, None, core=res.core)
    assert any("contorno" in r for r in q.reasons) and q.hard >= c_ghost["severidade"] - 1e-6
