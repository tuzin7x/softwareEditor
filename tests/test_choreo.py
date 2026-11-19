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
    # da frente para trás: a cadeira só começa depois de o sofá pousar; o tapete (completado por baixo do sofá)
    # desenrola ANTES de o sofá pousar em cima dele
    assert chair.start >= sofa.motion_end
    rug = _ent(st, ch, "rug")
    assert rug.start < sofa.start and rug.motion_end <= sofa.motion_end


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
    assert v["oculto_independente_px"] == 0, v["oculto_independente_pares"]
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


# ------------------------------------------------------------------------------------- revisão PT-V3-6
def _hidden_with(st, ch, front):
    """"Pixel oculto visível" medido com outra relação de oclusão (a original, sem arestas tiradas)."""
    import dataclasses
    hard = ch._hard
    bands = C.cut_bands(dataclasses.replace(st, front=front), hard)
    tot = 0
    for f in range(0, ch.n_anim, 2):
        pl = C.placements_at(ch, st, f)
        for i in pl:
            e = next(e for e in ch.entrances if i in e.members)
            tot += C.uncovered(st, hard, bands, i, pl, (e.clip, e.profile) if e.clip else None)
    return tot


def test_undermount_sink_enters_with_its_countertop():
    """A borda da bancada cobre a cuba de verdade: a cuba não pode escalar/subir sozinha dentro do furo."""
    rects = [
        ("cabinet", 150, 220, 60, 300, (200, 190, 170), "planejado"),
        ("countertop", 120, 150, 50, 310, (60, 60, 60), "planejado"),
        ("sink", 126, 146, 140, 220, (235, 235, 240), "planejado"),     # a parte que aparece no furo
    ]
    photo, res, st = _build(rects)
    sink = next(k for k, l in enumerate(st.layers) if l.label == "sink")
    top = next(k for k, l in enumerate(st.layers) if l.label == "countertop")
    st.front[top, sink], st.front[sink, top] = True, False          # a bancada cobre a borda da cuba
    front = st.front.copy()
    ch = C.plan(st)
    assert _ent(st, ch, "sink") is _ent(st, ch, "countertop")
    assert C.verify(st, ch)["oculto_visivel_px"] == 0
    assert _hidden_with(st, ch, front) == 0


def test_true_cycle_edge_is_not_dropped_when_block_is_as_good():
    """Ciclo real (vaso do aparador na frente do sofá, sofá na frente do aparador): tirar a aresta mais fraca
    mostrava o recorte do vaso no sofá; o bloco rígido é seguro nas duas leituras."""
    rects = [
        ("table", 120, 170, 20, 90, (200, 200, 210), "solto"),
        ("sofa", 110, 220, 70, 300, (180, 30, 30), "solto"),
        ("plant", 80, 140, 60, 100, (30, 160, 30), "solto"),
    ]
    photo, res, st = _build(rects)
    front = st.front.copy()
    ch = C.plan(st)
    assert C.verify(st, ch)["oculto_visivel_px"] == 0
    assert _hidden_with(st, ch, front) == 0


def test_small_piece_ties_at_bottom_edge_go_in_front():
    """Manta jogada no encosto que desce até a borda da foto junto com a poltrona: sem relação nenhuma, a poltrona
    entrava com um buraco no formato da manta. A peça pequena fica na frente."""
    rects = [
        ("armchair", 120, 240, 100, 300, (230, 230, 225), "solto"),
        ("blanket", 140, 240, 160, 220, (120, 120, 130), "solto"),
    ]
    photo, res, st = _build(rects)
    hard = C._abs_hard(st)
    front, _ = L.occlusion_graph([l.label for l in st.layers], hard, None)
    a = next(k for k, l in enumerate(st.layers) if l.label == "armchair")
    b = next(k for k, l in enumerate(st.layers) if l.label == "blanket")
    assert front[b, a] and not front[a, b]


# ------------------------------------------------------------------------- defeitos da avaliação do PT-V3-8
def _dining(depth=True):
    """Mesa de jantar com duas cadeiras na frente, na MESMA profundidade na borda comum (o caso da Nice): o encosto
    e as pernas das cadeiras "mordem" o retângulo da mesa."""
    photo = np.clip(_floor(3), 0, 255).round().astype(np.uint8)
    table = np.zeros((H, W), bool)
    table[90:120, 80:320] = True                      # tampo
    table[120:200, 90:104] = True                     # pernas
    table[120:200, 296:310] = True
    chairs = []
    for x0 in (120, 230):
        c = np.zeros((H, W), bool)
        c[70:150, x0:x0 + 50] = True                  # encosto + assento (na frente do tampo)
        c[150:215, x0:x0 + 8] = True                  # pernas
        c[150:215, x0 + 42:x0 + 50] = True
        chairs.append(c)
    img = photo.astype(np.float64)
    img[table] = (200, 150, 90)
    for c in chairs:
        img[c] = (60, 60, 70)
    for c in chairs:
        table &= ~c
    photo = np.clip(img, 0, 255).round().astype(np.uint8)
    masks = [("table", table)] + [("chair", c) for c in chairs]
    pieces = [SimpleNamespace(id=k, label=lab, kind="solto", alpha=m.astype(np.float32), mask=m, touches=())
              for k, (lab, m) in enumerate(masks)]
    mask = rp.core_mask(pieces, (H, W))
    plate = photo.copy()
    pt = np.clip(_floor(4), 0, 255).round().astype(np.uint8)
    plate[mask] = pt[mask]
    res = SimpleNamespace(plate=plate, mask=mask, shadow=np.zeros((H, W), bool), effect=np.zeros((H, W), bool))
    dep = np.full((H, W), 4.2, np.float32) if depth else None
    return photo, res, L.build_layers(photo, res, pieces, dep)


def test_touching_pieces_at_same_depth_get_an_order():
    """Mesa x cadeiras com a mesma profundidade na borda comum: o dono do contorno (a cadeira entra na envoltória
    da mesa) põe as cadeiras na frente, e a mesa nunca aparece com o buraco delas."""
    photo, res, st = _dining()
    t = next(k for k, l in enumerate(st.layers) if l.label == "table")
    chairs = [k for k, l in enumerate(st.layers) if l.label == "chair"]
    for c in chairs:
        assert st.front[c, t] and not st.front[t, c]
        assert st.evidence[c, t]                                  # evidência clara (sem o grafo)
    cov = st.stats["oclusao_cobertura_evidencia"]                 # cobertura relatada (não superestimar)
    assert cov["com_evidencia"] >= len(chairs) and 0 < cov["fracao"] <= 1
    assert cov["com_evidencia"] <= cov["bordas_grafo"]
    ch = C.plan(st)
    v = C.verify(st, ch)
    assert v["oculto_visivel_px"] == 0 and v["oculto_independente_px"] == 0
    te = next(e for e in ch.entrances if t in e.members)
    for c in chairs:
        ce = next(e for e in ch.entrances if c in e.members)
        assert ce is te or ce.motion_end <= te.start + 1e-6       # cadeira pousada antes (ou bloco rígido)


def test_every_shared_boundary_gets_an_order_without_depth():
    photo, res, st = _dining(depth=False)
    hard = C._abs_hard(st)
    cues = L.occlusion_cues([l.label for l in st.layers], hard, None)
    assert cues
    for (i, j) in cues:
        assert st.front[i, j] or st.front[j, i]


def test_independent_check_catches_missing_relation_and_plan_repairs_it():
    """O grafo sem a relação mesa x cadeira (o defeito da Nice): ``verify`` diz 0, a verificação independente
    acusa o corte da mesa à vista, ``plan`` refaz o plano e a verificação do vídeo (§3.5, regra 5b) recusa o que
    sobrar."""
    import dataclasses

    from mobilia.composite.verify import ModeBVerificationError, ModeBVerifier

    photo, res, st = _dining()
    t = next(k for k, l in enumerate(st.layers) if l.label == "table")
    broken = st.front.copy()
    broken[:, t] = False
    st_b = dataclasses.replace(st, front=broken)
    ch = C._plan_once(st_b)
    v = C.verify(st_b, ch)
    assert v["oculto_visivel_px"] == 0                            # o grafo não vê
    assert v["oculto_independente_px"] > 0 and v["oculto_independente_pares"]
    assert v["oculto_independente_pares"][0]["peca"].startswith("table")
    ver = ModeBVerifier(photo, st_b, res.mask, ch.n_frames)
    ver.n = ch.n_frames
    with pytest.raises(ModeBVerificationError, match="5b"):
        ver.finish(v)
    ch2 = C.plan(st_b)                                            # reparo pela verificação independente
    v2 = C.verify(st_b, ch2)
    assert v2["oculto_independente_px"] == 0 and v2["oculto_visivel_px"] == 0
    assert ch2.stats["reparos_independentes"] >= 1


def test_rug_is_completed_under_furniture_and_unrolls_first():
    """O tapete coberto pelo sofá é completado por baixo dele (cor do tapete, não a placa) e desenrola ANTES de o
    sofá pousar; a pilha completa continua igual à foto."""
    photo, res, st = _build(ROOM)
    r = next(k for k, l in enumerate(st.layers) if l.label == "rug")
    s = next(k for k, l in enumerate(st.layers) if l.label == "sofa")
    assert st.layers[r].amodal is not None and s in st.rug_first.get(r, set())
    assert st.stats["tapete_completado_pct"] > 0
    only_rug = L.composite(st, [r])
    under = (slice(176, 216), slice(170, 290))                   # sob o sofá, dentro do tapete
    rug_col = np.array([40, 80, 160])
    assert np.abs(only_rug[under].reshape(-1, 3).astype(int) - rug_col).max(axis=1).mean() < 25
    full = L.composite(st, range(len(st.layers)))
    assert np.abs(full.astype(int) - photo).max() <= L.RECON_TOL
    ch = C.plan(st)
    v = C.verify(st, ch)
    assert v["oculto_visivel_px"] == 0 and v["oculto_independente_px"] == 0
    rug, sofa = _ent(st, ch, "rug"), _ent(st, ch, "sofa")
    assert rug.motion_end <= sofa.motion_end and rug.start < sofa.start


def test_rug_fill_continues_the_pattern_without_blur():
    """Preenchimento por espelho: listras do tapete continuam do outro lado (o Telea faria uma mancha lisa)."""
    img = np.zeros((40, 80, 3), np.uint8)
    img[:, ::4] = 255                                             # listras verticais
    unk = np.zeros((40, 80), bool)
    unk[:, 50:] = True
    out = L.rug_mirror_fill(img, unk)
    assert out[:, 50:].std() > 0.8 * img[:, :50].std()
    assert set(np.unique(out[:, 50:])) <= {0, 255}


def test_wall_painting_at_top_edge_does_not_fall_from_the_ceiling():
    rects = [
        ("painting", 0, 50, 150, 230, (30, 90, 140), "solto"),       # cortado pelo topo da foto
        ("sofa", 130, 220, 100, 300, (180, 30, 30), "solto"),
    ]
    photo, res, st = _build(rects)
    ch = C.plan(st)
    v = C.verify(st, ch)
    assert v["oculto_visivel_px"] == 0 and v["oculto_independente_px"] == 0
    e = _ent(st, ch, "painting")
    assert e.style in ("escala", "deslizar")
    assert all(q.dy == 0 for q in _track(st, ch, "painting") if q is not None)
    if e.style == "escala":
        assert e.anchor[1] == 0.0 and e.scale0 <= 0.9


@pytest.mark.parametrize("name", ["sala_sofas_brancos", "nice_terrazzo"])
def test_fixture_choreography_defects_fixed(room_fixture, name):
    """Defeitos da avaliação final do PT-V3-8 nas fotos reais: peça encostada entra depois de quem a cobre (ou
    junto), tapete desenrola antes de quem está em cima dele, nada "surge", quadro no topo não cai do teto."""
    from test_layers2d import _stack

    f, s, r, st = _stack(room_fixture, name)
    ch = C.plan(st)
    v = C.verify(st, ch)
    assert v["oculto_independente_px"] == 0
    ent = {i: e for e in ch.entrances for i in e.members}
    for a in range(len(st.layers)):
        for b in range(len(st.layers)):
            if st.front[a, b] and ent[a] is not ent[b] and a not in st.rug_first.get(b, set()):
                assert ent[a].motion_end <= ent[b].start + 1e-6 or ent[b].style == "desenrolar", \
                    (st.layers[a].label, st.layers[b].label)
    for rg, clear in st.rug_first.items():
        for j in clear:
            if ent[j] is not ent[rg]:
                assert ent[rg].motion_end <= ent[j].motion_end + 1e-6
    assert all(e.style != "surgir" for e in ch.entrances)
    for e in ch.entrances:
        u = C._unit_union(ch._hard, e.members)
        if all(st.layers[i].label in L.WALL_HUNG_LABELS for i in e.members) and set(
                k for k, n in C._edge_contact(u).items() if n) == {"topo"}:
            assert e.style != "deslizar" or e.direction[1] == 0
    if name == "nice_terrazzo":
        t = next(k for k, l in enumerate(st.layers) if l.label == "table" and
                 sum(st.layers[c].label == "chair" and st.front[c, k] for c in range(len(st.layers))) >= 2)
        for c in range(len(st.layers)):
            if st.layers[c].label == "chair" and st.front[c, t]:
                assert ent[c] is ent[t] or ent[c].motion_end <= ent[t].start + 1e-6
    else:
        assert st.rug_first, "o tapete da sala típica é completado sob os móveis"
