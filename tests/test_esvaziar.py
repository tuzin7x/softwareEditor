"""0.9.1: "Esvaziar ambiente", sala vazia "ia" do modo B e tecido assentando no pouso (rápidos, sem modelos)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mobilia.analysis.segmentation import GROUP_IDS, UNKNOWN
from mobilia.design import empty as E
from mobilia.layers2d import choreo as C
from mobilia.layers2d import layers as L
from mobilia.removal import plate as rp

G = GROUP_IDS
H, W = 192, 256


def _scene():
    """Sala sintética: parede em cima, piso embaixo, janela na parede, sofá (móvel), armário (planejado) e um
    tapete (rótulo ADE20K "rug", grupo do piso)."""
    g = np.full((H, W), G["wall"], np.uint8)
    g[110:] = G["floor"]
    g[20:80, 20:70] = G["window"]
    g[80:140, 90:170] = G["furniture"]          # sofá
    g[40:150, 200:250] = G["fixture"]           # armário planejado
    labels = np.zeros((H, W), np.uint8)
    labels[150:180, 80:180] = 28                # tapete
    g[150:180, 80:180] = G["floor"]
    id2label = {0: "wall", 3: "floor", 28: "rug", 8: "window"}
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:110] = (200, 195, 185)
    rgb[110:] = (140, 110, 80)
    rgb[g == G["window"]] = (230, 240, 255)
    rgb[g == G["furniture"]] = (60, 60, 160)
    rgb[g == G["fixture"]] = (90, 70, 50)
    rgb[150:180, 80:180] = (170, 40, 40)
    return rgb, g, labels, id2label


class FakeEngine:
    """Gerador falso: dentro da máscara pinta parede/piso lisos pela linha (y < 110 parede); conta as chamadas."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, image, mask, control, prompt, seed, strength, steps, guidance, control_scale, control_end,
                 callback=None):
        self.calls.append({"strength": strength, "mask": mask.copy(), "control": control is not None,
                           "prompt": prompt, "init": image.copy()})
        if callback:
            callback(steps, steps)
        out = image.copy()
        h = image.shape[0]
        ys = np.arange(h)[:, None]
        wall = np.broadcast_to(ys < int(110 * h / H), mask.shape)
        out[mask & wall] = (200, 195, 185)
        out[mask & ~wall] = (140, 110, 80)
        return out


# ===================================================================================== máscara
def test_empty_mask_takes_furniture_rug_and_contact_shadow_but_keeps_the_house():
    rgb, g, labels, names = _scene()
    m = E.empty_mask(g, labels, names, (W, H))
    assert m[90:130, 100:160].all()                       # sofá inteiro
    assert m[150:180, 85:175].all()                       # tapete (rótulo, não grupo)
    # sombra de contato: abaixo do sofá, sobre o piso
    assert m[145, 120] and not m[60, 120]
    assert not m[30:70, 30:60].any()                      # janela fica
    assert not m[60:130, 210:240].any()                   # armário planejado fica
    m2 = E.empty_mask(g, None, None, (W, H))
    assert not m2[160:175, 90:170].all()                  # sem rótulos o tapete não é reconhecido


def test_empty_mask_curtains_optional_and_unknown_near_furniture():
    rgb, g, labels, names = _scene()
    g = g.copy()
    g[20:80, 72:82] = G["window_covering"]
    g[70:80, 120:130] = UNKNOWN                          # objeto "não classificado" em cima do sofá
    keep = E.empty_mask(g, labels, names, (W, H))
    out = E.empty_mask(g, labels, names, (W, H), keep_curtains=False)
    assert not keep[30:70, 74:80].any() and out[30:70, 74:80].all()
    assert keep[72:78, 122:128].all()


def test_empty_score_penalises_leftover_furniture_and_new_openings():
    rgb, g, labels, names = _scene()
    hole = E.empty_mask(g, labels, names, (W, H))
    opening = E.openings_mask(g, (W, H))
    clean = g.copy(); clean[hole & (np.arange(H)[:, None] < 110)] = G["wall"]; clean[hole & (np.arange(H)[:, None] >= 110)] = G["floor"]
    dirty = clean.copy(); dirty[85:125, 100:150] = G["furniture"]
    newwin = clean.copy(); newwin[60:105, 100:160] = G["window"]
    s_clean = E.empty_score(clean, hole, opening, rgb, src_groups=g)
    s_dirty = E.empty_score(dirty, hole, opening, rgb, src_groups=g)
    s_win = E.empty_score(newwin, hole, opening, rgb, src_groups=g)
    assert s_clean["movel_restante"] == 0 and s_dirty["movel_restante"] > 0.1
    assert s_win["aberturas_novas"] > 0
    assert s_clean["nota"] > s_dirty["nota"] and s_clean["nota"] > s_win["nota"]


def test_finish_keeps_photo_outside_and_restores_openings():
    rgb, g, labels, names = _scene()
    hole = np.zeros((H, W), bool); hole[50:150, 40:180] = True
    lock = E.openings_mask(g, (W, H))
    gen = np.full_like(rgb, 7)
    out = E.finish(gen, rgb, hole, lock)
    assert np.array_equal(out[~hole], rgb[~hole])
    core_win = lock & hole
    core_win[:, :] &= np.pad(np.ones((H - 6, W - 6), bool), 3)
    inner = np.zeros_like(core_win); inner[56:76, 46:66] = True
    assert np.abs(out[inner].astype(int) - rgb[inner].astype(int)).max() <= 2     # janela da foto, não o gerado


# ===================================================================================== função de design
def _analysis(geo=True):
    rgb, g, labels, names = _scene()
    depth = np.tile(np.linspace(4.0, 2.0, H, dtype=np.float32)[:, None], (1, W))
    fill = (lambda img, hole: E.soft_prefill(img, hole)) if geo else None
    return rgb, E.EmptyAnalysis(depth, g, labels, names, fill, {"teste": True})


def test_empty_room_with_fake_engine_alternates_starts_and_ranks(tmp_path):
    rgb, an = _analysis()
    eng = FakeEngine()
    calls = {"n": 0}

    def rank_fn(img):
        # o 1º candidato "ainda tem" móvel: a escolha tem de preferir o outro
        calls["n"] += 1
        gg = an.groups.copy()
        gg[an.groups == G["furniture"]] = G["wall"]
        if calls["n"] == 1:
            gg[80:140, 90:170] = G["furniture"]
        return gg, None

    events = []
    res = E.empty_room(rgb, n=1, candidates=2, seed=3, out_dir=tmp_path, engine=eng, analysis=an,
                       rank_fn=rank_fn, long_side=256, progress=events.append)
    assert [c["strength"] for c in eng.calls] == [E.STRENGTH_GEO, E.STRENGTH_SOFT]
    assert all(c["control"] for c in eng.calls)
    assert "furniture" in eng.calls[0]["prompt"].negative and "empty room" in eng.calls[0]["prompt"].positive
    rep = json.loads(Path(res.report).read_text(encoding="utf-8"))
    assert rep["candidatos"]["partidas"] == ["geometria", "suave"] and rep["candidatos"]["escolhidos"] == [1]
    assert res.funcao == "esvaziar" and len(res.outputs) == 1 and Path(res.outputs[0]).is_file()
    assert Path(res.before).is_file() and Path(rep["mascara"]).is_file()
    # fora da máscara, a foto byte a byte (a foto de teste já tem o tamanho de trabalho: 256 × 192)
    hole = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(rep["mascara"])) > 127
    img = res.images[0]
    assert rep["tamanho"] == [W, H] and np.array_equal(img[~hole], rgb[~hole])
    # o sofá sumiu: dentro dele, a cor da parede/piso do gerador falso
    assert np.abs(img[100, 130].astype(int) - np.array([200, 195, 185])).max() < 12
    ev = res.to_event()
    assert ev["funcao"] == "esvaziar" and ev["outputs"] == res.outputs
    assert any(e.get("event") == "preview" for e in events)


def test_empty_room_without_geometry_uses_soft_start_and_rejects_empty_photo(tmp_path):
    rgb, an = _analysis(geo=False)
    eng = FakeEngine()
    E.empty_room(rgb, n=2, candidates=2, seed=1, out_dir=tmp_path / "a", engine=eng, analysis=an,
                 rank_fn=lambda im: (an.groups, None), long_side=256)
    assert [c["strength"] for c in eng.calls] == [E.STRENGTH_SOFT] * 2
    vazio = E.EmptyAnalysis(an.depth, np.where(an.groups == G["furniture"], G["wall"], an.groups).astype(np.uint8))
    with pytest.raises(ValueError, match="vazio"):
        E.empty_room(rgb, n=1, out_dir=tmp_path / "b", engine=eng, analysis=vazio, long_side=256)


def test_esvaziar_is_a_catalogued_function_and_redesign_refuses_it():
    from mobilia.design import api, presets
    f = presets.function("esvaziar")
    assert f.label == "Esvaziar ambiente" and "esvaziar" in presets.ILLUSTRATIVE
    assert presets.catalog()["funcoes"]["esvaziar"] == "Esvaziar ambiente"
    with pytest.raises(ValueError, match="empty_room"):
        api.redesign(np.zeros((64, 64, 3), np.uint8), "esvaziar", engine=FakeEngine())


def test_cli_design_esvaziar_routes_to_empty_room(tmp_path, monkeypatch, capsys):
    from mobilia import cli
    from mobilia.design import api
    seen = {}

    def fake_empty(photo, **kw):
        seen.update(kw, photo=photo)
        o = tmp_path / "esvaziar_1.jpg"
        __import__("PIL.Image", fromlist=["Image"]).fromarray(np.zeros((8, 8, 3), np.uint8)).save(o)
        return api.DesignResult([str(o)], None, None, "esvaziar", None, "sala", [1], (8, 8))

    monkeypatch.setattr(api, "empty_room", fake_empty)
    photo = str(Path(__file__).parent / "fixtures/rooms/sala_sofas_brancos/photo.jpg")
    assert cli.main(["design", "--funcao", "esvaziar", "--photo", photo, "--n", "2", "--out-dir", str(tmp_path),
                     "--tirar-cortinas", "--partida", "suave", "--json"]) == 0
    done = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert done["event"] == "done" and done["funcao"] == "esvaziar"
    assert seen["n"] == 2 and seen["keep_curtains"] is False and seen["prefill"] == "suave"


# ===================================================================================== modo B: placa "ia"
def test_plate_filler_crops_generates_and_keeps_photo_outside():
    rgb, g, labels, names = _scene()
    mask = E.empty_mask(g, labels, names, (W, H))
    init = E.soft_prefill(rgb, mask)
    eng = FakeEngine()
    f = E.PlateFiller(eng, "cpu", long_side=128, steps=4)
    depth = np.full((H, W), 3.0, np.float32)
    plate, info = f(rgb, init, mask, g, depth, seed=9)
    assert np.array_equal(plate[~mask], rgb[~mask])
    assert info["origem"] == "gerada" and max(info["tamanho_geracao"]) <= 128 and len(eng.calls) == 1
    assert eng.calls[0]["strength"] == E.STRENGTH_PLATE and eng.calls[0]["control"]
    # o cômodo vazio escolhido pelo usuário dispensa o gerador
    given = np.full((96, 128, 3), 200, np.uint8)
    f2 = E.PlateFiller(FakeEngine(), "cpu", given=given)
    plate2, info2 = f2(rgb, init, mask, g, None)
    assert info2["origem"].startswith("escolhida") and np.array_equal(plate2[~mask], rgb[~mask])


@pytest.fixture(scope="module")
def sofas():
    from tests.conftest import load_room_fixture
    return load_room_fixture("sala_sofas_brancos")


def _inst(fx):
    from mobilia.pipeline_b import to_video_resolution
    from mobilia.removal import instances as ri
    from mobilia.removal.instances import PanopticResult
    h, w = fx.photo.shape[:2]
    pan = PanopticResult(fx.panoptic_ids, fx.panoptic_segments)
    geo = SimpleNamespace(points=np.zeros((h, w, 3), np.float32), mask=np.ones((h, w), bool))
    seg_v, pan_v, _ = to_video_resolution(fx.seg, pan, geo, w, h)
    return ri.detect_instances(fx.photo, seg_v, pan_v, sam=None, room=fx.room), seg_v


def test_gate_tries_ia_first_and_falls_back_to_geometry(sofas):
    inst, seg = _inst(sofas)
    seen = []

    def good(photo, plate, mask, groups, depth, seed):
        seen.append(mask.mean())
        return plate, {"origem": "falso"}

    r = rp.build_plate_gated(sofas.photo, inst, seg, sofas.room, None, rp.telea_inpaint, None, ai_fill=good)
    names = [a["estrategia"] for a in r.info["tentativas"]]
    assert names[0] == "ia" and len(seen) == 1
    assert np.array_equal(r.plate[~r.mask], sofas.photo[~r.mask])
    if r.quality is not None and r.quality.ok:
        assert r.info["estrategia_portao"] == "ia" and names == ["ia"]

    def boom(*a, **k):
        raise RuntimeError("sem memória (teste)")

    r2 = rp.build_plate_gated(sofas.photo, inst, seg, sofas.room, None, rp.telea_inpaint, None, ai_fill=boom)
    t = r2.info["tentativas"]
    assert t[0]["estrategia"] == "ia" and "sem memória" in t[0]["erro"]
    assert t[1]["estrategia"] == "crua_geometria" and r2.info["estrategia_portao"] != "ia"
    assert np.array_equal(r2.plate[~r2.mask], sofas.photo[~r2.mask])


def test_plate_choice_rules(monkeypatch):
    from mobilia import pipeline_b as pb
    monkeypatch.setattr("mobilia.design.empty.design_engine_available", lambda dev="auto": (True, "ok"))
    req = SimpleNamespace(plate="auto", empty_image=None)
    monkeypatch.setenv("MOBILIA_PLACA", "auto")
    assert pb.plate_choice(req, "cpu")[0] is True                       # padrão: IA quando há gerador
    monkeypatch.setenv("MOBILIA_PLACA", "geometria")
    assert pb.plate_choice(req, "cpu")[0] is False
    assert pb.plate_choice(SimpleNamespace(plate="ia", empty_image=None), "cpu")[0] is True
    assert pb.plate_choice(SimpleNamespace(plate="geometria", empty_image="x.jpg"), "cpu")[0] is False
    monkeypatch.setattr("mobilia.design.empty.design_engine_available", lambda dev="auto": (False, "sem pesos"))
    ok, why = pb.plate_choice(SimpleNamespace(plate="ia", empty_image=None), "cpu")
    assert not ok and "geometria" in why
    assert pb.plate_choice(SimpleNamespace(plate="ia", empty_image="vazio.jpg"), "cpu")[0] is True


def test_job_request_validates_plate_and_empty_image(tmp_path):
    from mobilia.jobs import JobRequest
    photo = str(Path(__file__).parent / "fixtures/rooms/sala_sofas_brancos/photo.jpg")
    JobRequest(photo, "sala", str(tmp_path / "o.mp4"), plate="ia").validate()
    with pytest.raises(ValueError):
        JobRequest(photo, "sala", str(tmp_path / "o.mp4"), plate="lama").validate()
    with pytest.raises(FileNotFoundError):
        JobRequest(photo, "sala", str(tmp_path / "o.mp4"), empty_image=str(tmp_path / "nao.jpg")).validate()


# ===================================================================================== tecido assentando
HB, WB = 240, 400


def _bed_stack():
    from tests.test_choreo import _build
    rects = [("bed", 110, 210, 120, 300, (200, 200, 220), "solto"),
             ("pillow", 100, 130, 140, 200, (240, 240, 240), "solto"),
             ("table", 60, 120, 20, 80, (120, 90, 60), "solto")]
    photo, res, st = _build(rects)
    rng = np.random.default_rng(3)                     # textura na roupa de cama (o deslocamento fica visível)
    for lay in st.layers:
        if lay.label == "bed":
            tex = rng.normal(0, 25, lay.color.shape).astype(np.float32)
            lay.color[:] = np.clip(lay.color + tex * lay.alpha[..., None], 0, 255 * lay.alpha[..., None])
    return photo, res, st


def test_fabric_settles_on_landing_and_last_frames_are_exact():
    photo, res, st = _bed_stack()
    ch = C.plan(st)
    kinds = {st.layers[i].label: f.kind for i, f in ch.fabric.items()}
    assert kinds.get("bed") == "drape" and kinds.get("pillow") == "squash" and "table" not in kinds
    H_ = st.photo_shape[0]
    for i, fs in ch.fabric.items():
        e = next(e for e in ch.entrances if i in e.members)
        assert abs((fs.t0 + fs.dur) - e.end) < 1e-6 and C.FABRIC_DUR[0] - 1e-6 <= fs.dur <= C.FABRIC_DUR[1] + 1e-6
        mags = []
        for f in range(ch.n_anim):
            q = C.placements_at(ch, st, f).get(i)
            if q is not None and q.warp is not None:
                mags.append(float(np.abs(q.warp).max()))
                assert f / ch.fps >= fs.t0 - 1e-6 and f / ch.fps < e.end
            elif q is not None and f / ch.fps >= e.end:
                assert q.warp is None
        assert mags and max(mags) <= C.FABRIC_AMP * H_ * 1.2 + 1e-3 and max(mags) > 0.3
    frames = list(C.render_frames(st, ch, photo))
    assert np.array_equal(frames[-1], photo)
    landed = L.composite(st, range(len(st.layers)))
    assert np.array_equal(frames[ch.n_anim - 1], landed)
    # durante o assentamento o quadro difere do pouso (o efeito existe), só dentro da caixa da cama
    bed = next(i for i, l in enumerate(st.layers) if l.label == "bed")
    fs = ch.fabric[bed]
    fmid = int(np.ceil((fs.t0 + 0.15 * fs.dur) * ch.fps))
    pl = C.placements_at(ch, st, fmid)
    still = {k: L.Placement(dx=q.dx, dy=q.dy, shade=q.shade, contact=q.contact, alpha_mul=q.alpha_mul,
                            scale=q.scale, anchor=q.anchor) for k, q in pl.items()}
    diff = np.abs(L.composite(st, pl).astype(int) - L.composite(st, still).astype(int)).max(axis=2) > 0
    x0, y0, x1, y1 = st.layers[bed].box
    assert diff.any() and not diff[:, :max(0, x0 - 2)].any()
    # verificação da coreografia continua limpa
    v = C.verify(st, ch)
    assert v["oculto_visivel_px"] == 0


def test_fabric_can_be_turned_off_and_compositor_backends_agree():
    from mobilia.layers2d import compose as K
    photo, res, st = _bed_stack()
    off = C.plan(st, fabric=False)
    assert off.fabric == {}
    ch = C.plan(st)
    comp = K.FrameCompositor(st, backend="numpy")
    for f, (fr, pl) in enumerate(K.master_frames(st, ch, photo, comp, with_placements=True)):
        if pl is not None and any(q.warp is not None for q in pl.values()):
            assert np.array_equal(fr, L.composite(st, pl))
            assert any(K.moving(q) for q in pl.values())
    assert np.array_equal(fr, photo)


def test_warp_layer_keeps_premultiplied_invariant():
    a = np.zeros((20, 30), np.float32); a[5:15, 5:25] = 1.0
    c = np.zeros((20, 30, 3), np.float32); c[a > 0] = (100, 150, 200)
    d = np.zeros((20, 30, 2), np.float32); d[..., 1] = 1.7; d[..., 0] = -0.6
    a2, c2, sh2 = L.warp_layer(a, c, None, d)
    assert sh2 is None and (c2 <= 255 * a2[..., None] + 1e-4).all() and a2.max() <= 1.0 + 1e-6
    assert a2[16, 10] > 0 and a2[5, 10] < 1.0                  # desceu
