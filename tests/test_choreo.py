"""PT-V3-6: coreografia 2D sem partes ocultas (rápidos, sem modelos)."""
from types import SimpleNamespace

import numpy as np
import pytest

from mobilia.layers2d import choreo as C
from mobilia.layers2d import layers as L
from mobilia.removal import plate as rp

H, W = 240, 400


def _floor(seed):
    rng = np.random.default_rng(seed)
    return np.full((H, W, 3), (150.0, 140.0, 130.0)) + rng.normal(0, 4.0, (H, W, 3))


def _build(rects, seed=0):
    """``rects`` de trás para frente: (rótulo, y0, y1, x0, x1, cor, tipo). A máscara de cada peça é só a parte
    visível na foto (o que fica atrás de uma peça da frente não existe)."""
    img = _floor(seed)
    visible = []
    for k, (lab, y0, y1, x0, x1, col, kind) in enumerate(rects):
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = True
        img[m] = col
        for v in visible:
            v &= ~m
        visible.append(m)
    photo = np.clip(img, 0, 255).round().astype(np.uint8)
    pieces = [SimpleNamespace(id=k, label=r[0], kind=r[6], alpha=visible[k].astype(np.float32), mask=visible[k],
                              touches=()) for k, r in enumerate(rects)]
    mask = rp.core_mask(pieces, (H, W))
    plate = photo.copy()
    pt = np.clip(_floor(seed + 1), 0, 255).round().astype(np.uint8)
    plate[mask] = pt[mask]
    res = SimpleNamespace(plate=plate, mask=mask, shadow=np.zeros((H, W), bool), effect=np.zeros((H, W), bool))
    return photo, res, L.build_layers(photo, res, pieces)


ROOM = [
    ("rug", 170, 236, 120, 330, (40, 80, 160), "solto"),
    ("chair", 70, 150, 200, 260, (90, 60, 30), "solto"),        # atrás do sofa (parte de baixo escondida)
    ("sofa", 120, 220, 160, 300, (180, 30, 30), "solto"),
    ("cushion", 104, 136, 200, 240, (230, 210, 40), "solto"),    # apoiada no sofa
    ("armchair", 100, 200, 340, 400, (60, 120, 60), "solto"),    # encosta na borda direita
    ("table", 60, 120, 40, 120, (200, 200, 210), "solto"),       # isolada
]


@pytest.fixture(scope="module")
def room():
    photo, res, st = _build(ROOM)
    ch = C.plan(st)
    v = C.verify(st, ch)
    return photo, res, st, ch, v


def _ent(st, ch, label):
    return next(e for e in ch.entrances if any(st.layers[i].label == label for i in e.members))


def _track(st, ch, label):
    i = next(k for k, l in enumerate(st.layers) if l.label == label)
    return [C.placements_at(ch, st, f).get(i) for f in range(ch.n_anim)]


def test_acceptance_no_hidden_no_fade_timing(room):
    photo, res, st, ch, v = room
    assert v["oculto_visivel_px"] == 0 and v["quadros_com_oculto"] == 0
    assert v["opacidade_parcial"] == 0
    assert v["borda_vindo_de_cima"] == 0
    assert TOTAL_OK(v["duracao_s"])
    assert v["vazio_inicial_s"] <= 0.5
    assert v["hold_final_s"] >= 1.0
    assert ch.warnings == []


def TOTAL_OK(d):
    return C.TOTAL_MIN - 1e-6 <= d <= C.TOTAL_MAX + 1e-6


def test_styles_follow_the_rules(room):
    _, _, st, ch, _ = room
    assert _ent(st, ch, "rug").style == "desenrolar"
    cush, sofa = _ent(st, ch, "cushion"), _ent(st, ch, "sofa")
    assert cush is sofa                                   # almofada entra junto com o sofá (cobre parte dele)
    arm = _ent(st, ch, "armchair")
    assert arm.style == "deslizar" and arm.direction == (1, 0)
    tab = _ent(st, ch, "table")
    assert tab.style == "queda" and 0 < tab.amp <= C.DROP_MAX * H + 1e-6
    chair = _ent(st, ch, "chair")
    assert chair.style in ("subir_atras", "escala", "emergir")
    # da frente para trás: a cadeira só começa depois de o sofá pousar; o tapete depois de quem o cobre
    assert chair.start >= sofa.motion_end
    assert _ent(st, ch, "rug").start >= sofa.motion_end


def test_edge_piece_never_comes_from_above_and_drop_bounces(room):
    _, _, st, ch, _ = room
    arm = [q for q in _track(st, ch, "armchair") if q is not None]
    assert arm and all(q.dy == 0 for q in arm) and arm[0].dx > 0 and arm[-1].dx == 0
    tab = [q for q in _track(st, ch, "table") if q is not None]
    dys = [q.dy for q in tab]
    assert dys[0] < 0 and -dys[0] <= C.DROP_MAX * H + 1
    land = dys.index(0)
    assert any(d < 0 for d in dys[land:])                  # quique
    assert dys[-1] == 0
    sh = [q.shade for q in tab[:land + 1]]
    assert all(b >= a - 1e-9 for a, b in zip(sh, sh[1:])) and sh[-1] == pytest.approx(1.0)   # sombra cresce
    assert all(q.contact == 0 for q in tab if q.dy != 0)  # o delta de contato fica parado: só no pouso


def test_frames_start_on_plate_and_end_on_photo(room):
    photo, res, st, ch, _ = room
    frames = list(C.render_frames(st, ch, photo))
    assert len(frames) == ch.n_frames
    assert np.array_equal(frames[0], st.plate)
    assert np.array_equal(frames[-1], photo)
    hold = int(round(C.HOLD_END * ch.fps))
    assert all(np.array_equal(f, photo) for f in frames[-hold:])
    assert np.abs(frames[ch.n_anim - 1].astype(int) - photo).max() <= L.RECON_TOL
    for f in (0, ch.n_anim // 2, ch.n_anim, ch.n_frames - 1):
        assert np.array_equal(C.frame_at(st, ch, photo, f), frames[f])
    # fora da máscara de remoção e das caixas em movimento, todo quadro é a foto
    for f in range(0, ch.n_anim, 3):
        pl = C.placements_at(ch, st, f)
        keep = ~res.mask
        for i, q in pl.items():
            x0, y0, x1, y1 = st.layers[i].box
            keep[max(0, y0 + q.dy):max(0, y1 + q.dy), max(0, x0 + q.dx):max(0, x1 + q.dx)] = False
        assert np.array_equal(frames[f][keep], photo[keep]), f


def test_hidden_metric_is_not_vacuous(room):
    _, _, st, ch, _ = room
    hard, bands = ch._hard, ch._bands
    c = next(k for k, l in enumerate(st.layers) if l.label == "chair")
    front = {k: L.Placement() for k, l in enumerate(st.layers) if l.label in ("sofa", "cushion")}
    assert len(bands[c]) > 0
    assert C.uncovered(st, hard, bands, c, {c: L.Placement()}) > 0                 # sem o sofá: corte à vista
    assert C.uncovered(st, hard, bands, c, {c: L.Placement(), **front}) == 0
    assert C.uncovered(st, hard, bands, c, {c: L.Placement(dy=-30), **front}) > 0   # subiu demais
    a = next(k for k, l in enumerate(st.layers) if l.label == "armchair")
    assert C.uncovered(st, hard, bands, a, {a: L.Placement(dx=-10)}) > 0        # saiu da borda: corte à vista


def test_scaled_placement_is_valid_premultiplied():
    photo, res, st = _build(ROOM)
    i = next(k for k, l in enumerate(st.layers) if l.label == "table")
    x0, y0, x1, y1 = st.layers[i].box
    base = L.composite(st, {i: L.Placement(scale=1.0)})
    assert np.array_equal(base, L.composite(st, [i]))
    sc = L.composite(st, {i: L.Placement(scale=0.94, anchor=((x0 + x1) / 2, y1 - 1))})
    changed = np.abs(sc.astype(int) - base).max(axis=2) > 0
    ys, xs = np.nonzero(changed)
    assert ys.size and ys.max() < y1 and xs.min() >= x0 - 1 and xs.max() <= x1    # encolhe em direção à base


def test_kitchen_builtins_first_with_their_own_moves():
    rects = [
        ("cabinet", 20, 80, 80, 240, (220, 220, 215), "planejado"),      # aéreo
        ("cabinet", 140, 220, 80, 240, (200, 190, 170), "planejado"),    # base
        ("countertop", 128, 140, 72, 248, (60, 60, 60), "planejado"),
        ("chair", 160, 230, 280, 320, (90, 60, 30), "solto"),
    ]
    photo, res, st = _build(rects)
    ch = C.plan(st)
    v = C.verify(st, ch)
    assert v["oculto_visivel_px"] == 0 and v["opacidade_parcial"] == 0
    roles = {e.role: e for e in ch.entrances}
    base, top, upper = roles["base"], roles["bancada"], roles["aereo"]
    assert base.style == "emergir" and base.clip == "piso"
    assert top.style == "pousar" and top.start >= base.motion_end            # bancada pousa sobre a base
    assert upper.style == "descer" and upper.direction == (0, -1)
    chair = _ent(st, ch, "chair")
    assert chair.start > max(base.start, top.start, upper.start)             # planejados antes dos soltos
    # subindo do chão: nada aparece abaixo da linha do piso da base
    b = next(k for k, l in enumerate(st.layers) if l.kind == "planejado" and l.box[1] > 100 and l.label == "cabinet")
    mid = int((base.start + 0.3 * base.dur) * ch.fps)
    q = C.placements_at(ch, st, mid)[b]
    assert q.dy > 0 and set(np.unique(q.alpha_mul)) <= {0.0, 1.0}
    fr = L.composite(st, {b: q})
    below = np.zeros((H, W), bool)
    below[221:, :] = True
    assert np.array_equal(fr[below], st.plate[below])


def test_occlusion_cycle_is_resolved_without_hidden_pixels():
    photo, res, st = _build(ROOM)
    c = next(k for k, l in enumerate(st.layers) if l.label == "chair")
    s = next(k for k, l in enumerate(st.layers) if l.label == "sofa")
    assert st.front[s, c]
    st.front[c, s] = True                                   # relação errada: ciclo sofa <-> cadeira
    ch = C.plan(st)
    v = C.verify(st, ch)
    assert ch.stats["ciclos_quebrados"] + ch.stats["ciclos_em_bloco"] >= 1
    assert v["oculto_visivel_px"] == 0


def test_empty_stack_is_just_the_photo():
    photo = np.full((H, W, 3), 120, np.uint8)
    res = SimpleNamespace(plate=photo.copy(), mask=np.zeros((H, W), bool), shadow=np.zeros((H, W), bool),
                          effect=np.zeros((H, W), bool))
    st = L.build_layers(photo, res, [])
    ch = C.plan(st)
    assert ch.entrances == [] and ch.warnings
    frames = list(C.render_frames(st, ch, photo))
    assert len(frames) == ch.n_frames and all(np.array_equal(f, photo) for f in frames)


# ------------------------------------------------------------------------------------------- fixtures reais
@pytest.mark.parametrize("name", ["sala_sofas_brancos", "nice_terrazzo"])
def test_fixture_acceptance(room_fixture, name):
    """Aceite PT-V3-6 nas fotos reais (placa Telea; a coreografia não depende da qualidade da placa)."""
    from test_layers2d import _stack

    f, s, r, st = _stack(room_fixture, name)
    ch = C.plan(st)
    v = C.verify(st, ch)
    assert v["oculto_visivel_px"] == 0, ch.report(st)
    assert v["opacidade_parcial"] == 0
    assert v["borda_vindo_de_cima"] == 0
    assert TOTAL_OK(v["duracao_s"]) and v["vazio_inicial_s"] <= 0.5 and v["hold_final_s"] >= 1.0
    assert sorted(i for e in ch.entrances for i in e.members) == list(range(len(st.layers)))
    assert ch.stats["estilos"].get("surgir", 0) == 0
    assert np.array_equal(C.frame_at(st, ch, f.photo, 0), st.plate)
    assert np.array_equal(C.frame_at(st, ch, f.photo, ch.n_frames - 1), f.photo)
    rep = ch.report(st)["coreografia"]
    assert rep["entradas"] and rep["oculto_visivel_px"] == 0
    # nada de "fantasma": cada peça na tela está inteira (sem multiplicador) ou recortada de forma binária
    for k in range(0, ch.n_anim, 4):
        for i, q in C.placements_at(ch, st, k).items():
            e = next(e for e in ch.entrances if i in e.members)
            if q.alpha_mul is not None and e.style != "desenrolar":
                assert set(np.unique(q.alpha_mul)) <= {0.0, 1.0}
