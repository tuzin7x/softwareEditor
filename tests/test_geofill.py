"""Sala vazia refeita pela geometria (``removal/geofill.py``): cenas sintéticas, sem modelos.

Cada cena é renderizada duas vezes a partir de texturas no espaço da sala (piso de porcelanato com grade conhecida,
tábuas, parede lisa com gradiente de luz e rodapé, parede de painéis de madeira): a "verdade" vazia e a foto com um
"móvel" pintado por cima. O preenchimento tem de continuar a grade/as tábuas alinhadas, sem névoa, com a divisa
parede/piso reta, e nunca tocar fora da máscara nem nas aberturas protegidas.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from test_plate import _inst_set, _piece, _room

from mobilia.analysis.segmentation import GROUP_IDS
from mobilia.removal import geofill as gf
from mobilia.removal import plate as rp

G_FLOOR, G_WALL, G_CEIL, G_WIN = (GROUP_IDS[k] for k in ("floor", "wall", "ceiling", "window"))
W, H = 480, 360


def _scene(floor_fn, wall_fn, seed=0, noise=1.5):
    """Foto vazia sintética: textura do piso f(x, y) e da parede f(s, z) no espaço da sala (metros)."""
    rng = np.random.default_rng(seed)
    room = _room(W, H)
    box = rp.RoomBox(room, W, H)
    img = np.zeros((H, W, 3), np.float32)
    groups = np.full((H, W), 255, np.uint8)
    fl = box.label == rp.FLOOR
    vs, us = np.nonzero(fl)
    a, b, _ = box.coords(box.planes[rp.FLOOR], us, vs)
    img[vs, us] = floor_fn(a, b)
    groups[fl] = G_FLOOR
    wl = box.label == 0
    vs, us = np.nonzero(wl)
    img[vs, us] = wall_fn(box.S[vs, us], box.Z[vs, us], us)
    groups[wl] = G_WALL
    cl = box.label == rp.CEIL
    img[cl] = 238
    groups[cl] = G_CEIL
    img += rng.normal(0, noise, img.shape)
    return room, box, np.clip(img, 0, 255).round().astype(np.uint8), groups


def _tiles(period=0.5, grout=0.02):
    def f(x, y):
        gx = np.abs(((x + 0.013) / period) % 1.0 - 0.5) > 0.5 - grout / period
        gy = np.abs(((y + 0.007) / period) % 1.0 - 0.5) > 0.5 - grout / period
        col = np.tile([205.0, 195.0, 180.0], (len(x), 1))
        col[gx | gy] = [120.0, 115.0, 110.0]
        return col
    return f


def _planks(width=0.18):
    def f(x, y):
        k = np.floor(x / width).astype(int)
        tone = 140 + 18 * np.sin(k * 2.7)                    # cada tábua com seu tom
        seam = (x / width) % 1.0 < 0.06
        col = np.stack([tone, 0.75 * tone, 0.5 * tone], 1)
        col[seam] *= 0.6
        return col
    return f


def _plain_wall(baseboard=0.10):
    def f(s, z, u):
        shade = 1.0 - 0.18 * u / W                           # gradiente de luz da esquerda para a direita
        col = np.array([210.0, 196.0, 172.0])[None] * shade[:, None]
        col[z < baseboard] = [90.0, 62.0, 45.0]
        return col
    return f


def _panel_wall(period=0.30):
    def f(s, z, u):
        k = np.floor(s / period).astype(int)
        tone = 150 + 25 * np.cos(k * 1.9)
        groove = (s / period) % 1.0 < 0.07
        col = np.stack([tone, 0.72 * tone, 0.48 * tone], 1)
        col[groove] *= 0.55
        return col
    return f


def _sofa(box, h_frac=0.28):
    """Retângulo cobrindo a divisa parede/piso no meio da foto (sofá encostado na parede)."""
    m = np.zeros((H, W), bool)
    y_line = int(np.nonzero(box.label[:, W // 2] == rp.FLOOR)[0].min())
    m[y_line - int(h_frac * H):y_line + 70, 120:360] = True
    return m


def _fill(photo, groups, room, pieces, **kw):
    return rp.build_plate(photo, pieces, SimpleNamespace(groups=groups), room, inpaint=_no_lama,
                          strategy="geometria", assess_quality=kw.pop("assess", False), **kw)


def _no_lama(rgb8, mask):           # a estratégia "geometria" nunca chama o inpainter
    raise AssertionError("inpainter chamado na estratégia geometria")


def _paint_sofa(truth, m):
    photo = truth.copy()
    photo[m] = (70, 75, 95)
    return photo


# ------------------------------------------------------------------------------------------------ piso
def test_tiled_floor_grid_continues_aligned():
    room, box, truth, groups = _scene(_tiles(), _plain_wall())
    m = _sofa(box)
    photo = _paint_sofa(truth, m)
    r = _fill(photo, groups, room, [_piece(m, "solto", "sofa")])
    assert np.array_equal(r.plate[~r.mask], photo[~r.mask])
    info = r.info["planos"]["piso"]
    assert info["metodo"] == "grade" and info["periodo_px"][0] and info["periodo_px"][1], info
    fl = r.mask & (r.labels == rp.FLOOR)
    fl &= rp._ero(r.mask, 6)
    vs, us = np.nonzero(fl)
    a, b, _ = box.coords(box.planes[rp.FLOOR], us, vs)
    grout = _tiles()(a, b)[:, 0] < 150
    lum = r.plate[vs, us].astype(float).mean(1)
    # o rejunte do preenchimento cai onde o rejunte de verdade estaria (grade alinhada à visível)
    assert lum[grout].mean() < lum[~grout].mean() - 25, (lum[grout].mean(), lum[~grout].mean())
    corr = np.corrcoef(lum, truth[vs, us].astype(float).mean(1))[0, 1]
    assert corr > 0.5, corr


def test_plank_floor_keeps_direction_and_seams():
    room, box, truth, groups = _scene(_planks(), _plain_wall())
    m = _sofa(box)
    photo = _paint_sofa(truth, m)
    r = _fill(photo, groups, room, [_piece(m, "solto", "sofa")])
    fl = rp._ero(r.mask, 6) & (r.labels == rp.FLOOR)
    vs, us = np.nonzero(fl)
    a, _b, _ = box.coords(box.planes[rp.FLOOR], us, vs)
    lum = r.plate[vs, us].astype(float).mean(1)
    # tábuas correm ao longo de y: a cor varia com x (entre tábuas) e quase nada com y dentro da mesma tábua
    assert r.info["planos"]["piso"]["metodo"] == "listras", r.info["planos"]["piso"]
    k = np.floor(a / 0.18).astype(int)
    tl = truth[vs, us].astype(float).mean(1)
    ids = [i for i in np.unique(k) if (k == i).sum() > 40]
    within = np.mean([lum[k == i].std() for i in ids])
    within_t = np.mean([tl[k == i].std() for i in ids])
    # cada tábua continua com o próprio tom (variação dentro da tábua ~ a da verdade, bem menor que entre tábuas)
    assert within < within_t + 3.0 and within < 0.85 * lum.std(), (within, within_t, lum.std())
    corr = np.corrcoef(lum, tl)[0, 1]
    assert corr > 0.6, corr


def test_floor_light_has_no_fog_or_island():
    """Piso liso com degradê de luz: o preenchimento segue o degradê (sem véu claro no meio)."""
    def lit_floor(x, y):
        v = 160 - 10 * (y - 3.0)
        return np.stack([v, 0.95 * v, 0.88 * v], 1)
    room, box, truth, groups = _scene(lit_floor, _plain_wall(), noise=2.0)
    m = _sofa(box)
    photo = _paint_sofa(truth, m)
    r = _fill(photo, groups, room, [_piece(m, "solto", "sofa")])
    fl = rp._ero(r.mask, 4) & (r.labels == rp.FLOOR)
    d = np.abs(r.plate[fl].astype(float) - truth[fl].astype(float)).mean()
    assert d < 6.0, d
    # grão presente (sem borrão liso)
    hp = r.plate.astype(float).mean(2) - rp.cv2.GaussianBlur(r.plate.astype(float).mean(2), (0, 0), 1.0)
    assert hp[fl].std() > 0.4


# ------------------------------------------------------------------------------------------------ paredes
def test_plain_lit_wall_gradient_baseboard_and_straight_line():
    room, box, truth, groups = _scene(_tiles(), _plain_wall())
    m = _sofa(box)
    photo = _paint_sofa(truth, m)
    r = _fill(photo, groups, room, [_piece(m, "solto", "sofa")])
    info = r.info["planos"]["parede0"]
    assert info["metodo"] == "liso" and info["rodape_m"] == pytest.approx(0.10, abs=0.03), info
    wall = rp._ero(r.mask, 4) & (r.labels == 0) & (box.Z > 0.2)
    d = np.abs(r.plate[wall].astype(float) - truth[wall].astype(float)).mean(1)
    assert np.median(d) < 6.0 and np.percentile(d, 95) < 14.0, (np.median(d), np.percentile(d, 95))
    # o gradiente esquerda→direita continua (sem névoa: o preenchimento não é uma cor média)
    left = wall & (np.arange(W)[None] < 180)
    right = wall & (np.arange(W)[None] > 300)
    assert r.plate[left].mean() - r.plate[right].mean() > 8
    # rodapé reto: logo acima da divisa a cor é a do rodapé, e a divisa fica na linha projetada
    for x in (160, 240, 320):
        y = int(np.nonzero(box.label[:, x] == rp.FLOOR)[0].min())
        assert np.abs(r.plate[y - 3, x].astype(float) - (90, 62, 45)).max() < 30, (x, r.plate[y - 3, x])
        assert r.plate[y - 15, x].astype(float).mean() > 150


def test_textured_wall_panels_continue_with_same_period():
    room, box, truth, groups = _scene(_tiles(), _panel_wall())
    m = _sofa(box, 0.35)
    photo = _paint_sofa(truth, m)
    r = _fill(photo, groups, room, [_piece(m, "solto", "sofa")])
    assert r.info["planos"]["parede0"]["metodo"] == "listras", r.info["planos"]["parede0"]
    wall = rp._ero(r.mask, 4) & (r.labels == 0)
    vs, us = np.nonzero(wall)
    groove_true = ((box.S[vs, us] / 0.30) % 1.0) < 0.07
    lum = r.plate[vs, us].astype(float).mean(1)
    assert lum[groove_true].mean() < lum[~groove_true].mean() - 15


# ------------------------------------------------------------------------------------------------ regras
def test_protected_window_untouched_and_hidden_part_continues_glass_only_briefly():
    room, _box, truth, groups = _scene(_tiles(), _plain_wall())
    groups = groups.copy()
    win = np.zeros((H, W), bool)
    win[60:190, 170:310] = True
    groups[win] = G_WIN
    truth = truth.copy()
    truth[win] = (185, 215, 240)
    m = np.zeros((H, W), bool)
    m[150:300, 140:340] = True                       # sofá na frente da parte de baixo da janela
    photo = _paint_sofa(truth, m)
    r = _fill(photo, groups, room, [_piece(m, "solto", "sofa")])
    assert np.array_equal(r.plate[~r.mask], photo[~r.mask])          # janela visível intacta
    prot = r.mask & (r.labels == rp.PROTECTED)
    assert prot.sum() > 300
    # o vidro escondido não desce pela parede inteira: depois de alguns pixels é parede
    below = r.mask & (np.arange(H)[:, None] > 190 + int(0.03 * H) + 8) & (r.labels >= 0)
    if below.any():
        glass = np.abs(r.plate[below].astype(float) - (185, 215, 240)).max(1) < 20
        assert glass.mean() < 0.05


def test_no_new_colors_appear():
    """Nada é gerado: as cores do preenchimento ficam dentro da faixa do material visível (sem objeto novo)."""
    room, box, truth, groups = _scene(_tiles(), _plain_wall())
    m = _sofa(box)
    photo = _paint_sofa(truth, m)
    r = _fill(photo, groups, room, [_piece(m, "solto", "sofa")])
    vis = ~r.mask & np.isin(groups, (G_FLOOR, G_WALL))
    lab_v = rp._lab(photo)[vis]
    lab_f = rp._lab(r.plate)[r.mask]
    lo, hi = np.percentile(lab_v, 0.5, axis=0) - 6, np.percentile(lab_v, 99.5, axis=0) + 6
    out = ((lab_f < lo) | (lab_f > hi)).any(1)
    assert out.mean() < 0.01, out.mean()
    # nenhum resto do "sofá" (azul-acinzentado escuro)
    assert (np.abs(r.plate[r.mask].astype(int) - (70, 75, 95)).max(1) < 25).mean() < 0.002


def test_wall_floor_line_corrected_by_visible_boundary():
    """Caixa com a linha parede/piso 12 px acima da real: a divisa visível corrige a linha dentro da máscara."""
    room, box, truth, groups = _scene(_tiles(), _plain_wall(0.0))
    groups = groups.copy()
    # a segmentação diz que a parede vai 12 px mais para baixo do que a caixa projeta (caixa errada)
    for x in range(W):
        col = np.nonzero(box.label[:, x] == rp.FLOOR)[0]
        if col.size:
            y = col.min()
            groups[y:y + 12, x] = G_WALL
            truth[y:y + 12, x] = truth[max(0, y - 20), x]
    m = _sofa(box)
    lines = gf.floor_lines(m, groups, box)
    assert 0 in lines and lines[0][0] == pytest.approx(12, abs=2.5), lines
    photo = _paint_sofa(truth, m)
    r = _fill(photo, groups, room, [_piece(m, "solto", "sofa")])
    for x in (170, 240, 310):
        col = r.labels[:, x]
        y_box = int(np.nonzero(box.label[:, x] == rp.FLOOR)[0].min())
        y_new = int(np.nonzero(r.mask[:, x] & (col == rp.FLOOR))[0].min())
        assert abs(y_new - (y_box + 12)) <= 2, (x, y_box, y_new)


def test_gate_runs_geometry_first_and_never_loads_lama():
    room, box, truth, groups = _scene(_tiles(), _plain_wall())
    m = _sofa(box)
    photo = _paint_sofa(truth, m)
    inst_set = _inst_set([_piece(m, "solto", "sofa")])
    r = rp.build_plate_gated(photo, inst_set, SimpleNamespace(groups=groups), room, inpaint=_no_lama)
    names = [t["estrategia"] for t in r.info["tentativas"]]
    assert names[0] == "crua_geometria" and all(t["metodo"] == "geometria" for t in r.info["tentativas"]), names
    assert np.array_equal(r.plate[~r.mask], photo[~r.mask])


def test_quilt_respects_period_and_only_touches_hole():
    rng = np.random.default_rng(3)
    h, w, p = 160, 200, 16
    img = np.full((h, w, 3), 190.0, np.float32)
    img[::p] = 110
    img[:, ::p] = 110
    img += rng.normal(0, 2, img.shape)
    hole = np.zeros((h, w), bool)
    hole[40:120, 50:150] = True
    out, left, _info = gf.quilt(img, ~hole, hole, rng, period=(p, p))
    assert not left.any()
    assert np.array_equal(out[~hole], img[~hole])
    grout = np.zeros((h, w), bool)
    grout[::p] = True
    grout[:, ::p] = True
    assert out[hole & grout].mean() < out[hole & ~grout].mean() - 40


def test_fill_is_fast():
    import time

    room, box, truth, groups = _scene(_tiles(), _plain_wall())
    m = _sofa(box)
    photo = _paint_sofa(truth, m)
    t0 = time.perf_counter()
    _fill(photo, groups, room, [_piece(m, "solto", "sofa")])
    assert time.perf_counter() - t0 < 20.0


# ------------------------------------------------------------------------------------------------ revisão r1
G_CURT = GROUP_IDS["window_covering"]


def _q(raw, ok=False, hard=0.0, n_reasons=1, area=0.4):
    return SimpleNamespace(ok=ok, area_generated=area, raw_score=raw, score=0.6 * raw, hard=hard,
                           reasons=["x"] * n_reasons)


def test_rank_prefers_raw_score_over_rounded_hard_failure():
    """A tentativa de nota 17,6 com uma falha dura forte ganha da de nota 0,5 com falha dura menor."""
    a = _q(17.6, hard=0.9, n_reasons=3)
    b = _q(0.5, hard=0.4, n_reasons=2)
    assert rp.attempt_rank(a) > rp.attempt_rank(b)
    # aprovada sempre ganha; e área fora do limite perde
    assert rp.attempt_rank(_q(61, ok=True, n_reasons=0)) > rp.attempt_rank(_q(95, hard=0.0, n_reasons=1))
    assert rp.attempt_rank(_q(30, area=0.5)) > rp.attempt_rank(_q(80, area=0.9))


def test_piece_seen_through_doorway_stays_in_photo():
    """Peça mais longe que a parede da caixa (cama do quarto ao lado, vista pela porta) fica na foto: sem plano
    desta sala atrás dela, não há o que copiar. A peça encostada na parede (mesma profundidade) sai."""
    room, box, truth, groups = _scene(_tiles(), _plain_wall())
    depth = box.t.astype(np.float32).copy()
    y_line = int(np.nonzero(box.label[:, 100] == rp.FLOOR)[0].min())
    far = np.zeros((H, W), bool)
    far[y_line - 80:y_line - 20, 60:140] = True          # na parede, mas 1,5× mais longe
    depth[far] *= 1.5
    near = np.zeros((H, W), bool)
    near[y_line - 80:y_line - 20, 300:380] = True         # quadro/estante na parede
    depth[near] *= 0.97
    pf = SimpleNamespace(id=1, mask=far, alpha=far.astype(np.float32), kind="solto", label="bed")
    pn = SimpleNamespace(id=2, mask=near, alpha=near.astype(np.float32), kind="solto", label="shelf")
    assert rp.pieces_beyond_openings([pf, pn], box, depth, groups) == [1]
    photo = truth.copy()
    photo[far] = (140, 40, 60)
    photo[near] = (70, 75, 95)
    r = _fill(photo, groups, room, [pf, pn], depth=depth)
    assert r.info["pecas_alem_da_abertura"] == [1]
    assert not r.mask[far].any() and np.array_equal(r.plate[far], photo[far])
    assert r.mask[near].all()


def test_curtain_runs_straight_to_floor_line_and_table_in_front_of_it_goes():
    """Cortina escondida por móvel: todas as colunas descem até a linha do piso (barra de baixo reta, sem
    "estalactites"); o "outro" debaixo da cortina (piso da caixa além do visível) vira continuação da cortina."""
    room, box, truth, groups = _scene(_tiles(), _plain_wall())
    groups = groups.copy()
    truth = truth.copy()
    y_line = {x: int(np.nonzero(box.label[:, x] == rp.FLOOR)[0].min()) for x in range(W)}
    cur = np.zeros((H, W), bool)
    for x in range(160, 320):
        cur[40:y_line[x], x] = True
    groups[cur] = G_CURT
    xs = np.arange(W)[None]
    truth[cur] = np.stack([235 - 12 * (np.sin(xs / 6.0) > 0)] * 3, -1).repeat(H, 0)[cur]   # dobras verticais
    m = np.zeros((H, W), bool)
    m[150:y_line[240] + 40, 180:300] = True
    photo = _paint_sofa(truth, m)
    r = _fill(photo, groups, room, [_piece(m, "solto", "table")])
    assert np.array_equal(r.plate[~r.mask], photo[~r.mask])
    hid = r.mask & cur
    # a cortina refeita tem as dobras da cortina (claras), em toda a altura até a linha do piso
    for x in range(190, 290, 7):
        col = r.plate[150:y_line[x] - 2, x].astype(float).mean(1)
        assert col.min() > 200, (x, col.min())
    assert (np.abs(r.plate[hid].astype(int) - (70, 75, 95)).max(1) < 30).mean() < 0.01


def test_glass_seen_low_runs_to_floor_but_high_window_stops():
    """Vidro visto até perto do chão (porta de correr/janela de piso a teto) continua até a linha do piso; janela
    cuja parte de baixo vista fica bem acima de 0,85 m para na altura típica de peitoril (o resto é parede)."""
    room, box, truth, groups = _scene(_tiles(), _plain_wall())
    Z = box.Z
    for z_bottom, to_floor in ((0.55, True), (1.35, False)):
        g = groups.copy()
        t = truth.copy()
        win = np.zeros((H, W), bool)
        win[:, 170:310] = (box.label[:, 170:310] == 0) & (Z[:, 170:310] > 0.0) & (Z[:, 170:310] < 2.2)
        g[win] = G_WIN
        t[win] = (185, 215, 240)
        m = np.zeros((H, W), bool)
        m[:, 140:340] = (box.label[:, 140:340] == 0) & (Z[:, 140:340] < z_bottom) | (box.label[:, 140:340] == rp.FLOOR)
        m &= np.arange(H)[:, None] < int(0.95 * H)
        photo = _paint_sofa(t, m)
        r = _fill(photo, g, room, [_piece(m, "solto", "sofa")])
        low = r.mask & win & (Z < 0.3)
        glass = np.abs(r.plate[low].astype(float) - (185, 215, 240)).max(1) < 25
        if to_floor:
            assert glass.mean() > 0.9, glass.mean()
        else:
            assert glass.mean() < 0.05, glass.mean()


def test_mask_hygiene_holes_and_resting_objects():
    core = np.zeros((100, 100), bool)
    core[20:80, 20:80] = True
    core[45:50, 45:50] = False                             # etiqueta/vão no meio da peça
    out = rp.fill_mask_holes(core, 100)
    assert out[45:50, 45:50].all()
    groups = np.full((100, 100), G_WALL, np.uint8)
    groups[12:20, 40:48] = GROUP_IDS["furniture"]          # enfeite em cima da peça
    groups[40:50, 5:15] = GROUP_IDS["furniture"]           # objeto ao lado (não apoiado): fica
    rest = rp.resting_on_removed(out, groups, 500)
    assert rest[12:20, 40:48].all() and not rest[40:50, 5:15].any()


def test_wall_halo_grows_over_dark_frame_edge_only():
    """Borda escura/sombra de quadro logo fora da máscara entra; a parede lisa em volta não."""
    rng = np.random.default_rng(0)
    h, w = 200, 300
    photo = np.full((h, w, 3), 200, np.uint8)
    photo = np.clip(photo + rng.normal(0, 1.0, photo.shape), 0, 255).astype(np.uint8)
    m = np.zeros((h, w), bool)
    m[60:120, 100:200] = True
    halo = rp._dil(m, 4) & ~m
    photo[halo] = 150
    photo[m] = 40
    groups = np.full((h, w), G_WALL, np.uint8)
    label = np.zeros((h, w), np.int32)
    p = SimpleNamespace(mask=m, alpha=m.astype(np.float32))
    g = rp.wall_halo(photo, [p], groups, label, m)
    assert g[halo].mean() > 0.95
    assert not (g & ~rp._dil(m, 6)).any()


def test_wall_fill_has_no_step_at_false_box_split():
    """Uma parede real lisa com a caixa partida em duas: o campo de luz das paredes passa suave pela divisa falsa."""
    _room_m, box, truth, groups = _scene(_tiles(), _plain_wall(0.0))
    m = _sofa(box)
    photo = _paint_sofa(truth, m)
    lab = box.label.copy()
    ctx = gf._Ctx(photo, m, groups, box, lab, np.random.default_rng(0), rp.MAX_SIDE, None)
    split = lab == 0
    lab[split & (np.arange(W)[None] >= W // 2)] = 1             # parede "1" falsa na metade direita
    box.room.walls.append(SimpleNamespace(id=1))
    try:
        low = gf.wall_light_field(ctx)
    finally:
        box.room.walls.pop()
    y = int(np.nonzero(m[:, W // 2])[0].mean())
    row = low[y, W // 2 - 10:W // 2 + 10].mean(1)
    assert np.abs(np.diff(row)).max() < 2.0, row


def test_blind_wall_is_a_hard_veto():
    """Parede escondida sem nada à vista (atrás de planejados de ponta a ponta): reprova como falha dura."""
    room, box, truth, groups = _scene(_tiles(), _plain_wall())
    m = _sofa(box)
    photo = _paint_sofa(truth, m)
    r = _fill(photo, groups, room, [_piece(m, "solto", "sofa")])
    info = dict(r.info["planos"])
    info["parede7"] = {"metodo": "sem_contexto", "visivel_px": 0, "area_px": 0, "buraco_px": int(0.05 * H * W)}
    q = rp.assess(photo, r.plate, r.mask, r.labels, groups, box, None, info, core=r.core)
    assert not q.ok and any("sem nenhuma referência" in x for x in q.reasons)
    assert q.hard_detail.get("sem_referencia") == 1.0 and q.hard == 1.0
