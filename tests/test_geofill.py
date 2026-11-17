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
    below = r.mask & (np.arange(H)[:, None] > 190 + int(gf.SHORT_OPENING * H) + 8) & (r.labels >= 0)
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
