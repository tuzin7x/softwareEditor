"""PT-V3-5: camadas 2D e anti-morph (rápidos, sem modelos)."""
from types import SimpleNamespace

import numpy as np
import pytest

from mobilia.layers2d import layers as L
from mobilia.removal import instances as ri
from mobilia.removal import plate as rp

H, W = 120, 160


def _floor(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (150.0, 140.0, 130.0)) + rng.normal(0, 4.0, (H, W, 3))
    return img


def _soft_rect(y0, y1, x0, x1, blur=1.2):
    import cv2
    m = np.zeros((H, W), np.float32)
    m[y0:y1, x0:x1] = 1.0
    a = cv2.GaussianBlur(m, (0, 0), blur)
    a[a < 0.02] = 0.0
    return a, a > 0.5


def _piece(pid, label, alpha, hard, kind="solto"):
    return SimpleNamespace(id=pid, label=label, kind=kind, alpha=alpha, mask=hard, touches=())


def _scene(seed=0, shadow=True, light_far=False):
    """Piso com grão, sofá vermelho de borda suave, almofada amarela sobre o sofá, tapete embaixo, sombra sob o
    sofá. A "placa" é o piso sem nada (com outro grão, como um inpainting), igual à foto fora da máscara."""
    floor = _floor(seed)
    plate_true = _floor(seed + 1)
    a_rug, h_rug = _soft_rect(85, 118, 100, 155)
    a_sofa, h_sofa = _soft_rect(40, 90, 40, 120)
    a_cush, h_cush = _soft_rect(30, 46, 60, 80)
    img = floor.copy()
    sh = np.zeros((H, W), bool)
    if shadow:
        sh[90:106, 42:96] = True
        img[sh] *= 0.6
    img = (1 - a_rug[..., None]) * img + a_rug[..., None] * np.array([40.0, 80.0, 160.0])
    img = (1 - a_sofa[..., None]) * img + a_sofa[..., None] * np.array([180.0, 30.0, 30.0])
    img = (1 - a_cush[..., None]) * img + a_cush[..., None] * np.array([230.0, 210.0, 40.0])
    if light_far:
        img[5:15, 5:15] += 60.0
    photo = np.clip(img, 0, 255).round().astype(np.uint8)
    pieces = [_piece(1, "sofa", a_sofa, h_sofa), _piece(2, "cushion", a_cush, h_cush),
              _piece(3, "rug", a_rug, h_rug)]
    core = rp.core_mask(pieces, (H, W))
    effect = sh & ~core
    mask = core | rp._dil(effect, 3)
    if light_far:
        far = np.zeros((H, W), bool)
        far[3:17, 3:17] = True
        mask |= far
        effect = effect | far
    plate = photo.copy()
    pt = np.clip(plate_true, 0, 255).round().astype(np.uint8)
    plate[mask] = pt[mask]
    res = SimpleNamespace(plate=plate, mask=mask, shadow=effect & (photo.mean(2) < pt.mean(2)), effect=effect)
    return photo, res, pieces


def test_full_stack_is_photo_and_empty_is_plate():
    photo, res, pieces = _scene()
    st = L.build_layers(photo, res, pieces)
    full = L.composite(st, range(len(st.layers)))
    assert np.abs(full.astype(int) - photo).max() <= L.RECON_TOL
    empty = L.composite(st, [])
    assert np.array_equal(empty, st.plate)
    assert np.array_equal(st.plate[~res.mask], photo[~res.mask])
    assert st.stats["reconstrucao_max"] <= L.RECON_TOL
    assert L.morph_fraction(st, empty, full) == 0.0 and st.stats["morph_fora_pct"] == 0.0


def test_premultiplied_color_is_valid_and_shade_bounded():
    photo, res, pieces = _scene()
    st = L.build_layers(photo, res, pieces)
    for lay in st.layers:
        a = lay.alpha[..., None]
        assert (lay.color >= -1e-4).all() and (lay.color <= 255.0 * a + 1e-3).all()
        assert (lay.color[lay.alpha == 0] == 0).all()
        if lay.shade is not None:
            assert lay.shade.max() <= 1.0 and lay.shade.min() >= L.SHADE_MIN
            assert (lay.shade < 1).any(axis=2)[~lay.effect].sum() == 0     # só na região de efeito
    sofa = st.layers[st.index(1)]
    assert sofa.shade is not None and (sofa.shade < 0.8).any()            # a sombra sob o sofá é dele
    # a sombra fica escura em repouso e some sem o sofá
    only_sofa = L.composite(st, [st.index(1)])
    assert only_sofa[102, 70].mean() < st.plate[102, 70].mean() - 20


def test_plate_correction_far_from_pieces_kills_morph():
    """Luz longe de qualquer peça (o protótipo deixava esse resíduo entrar em fade a cada pouso): é corrigida na
    placa, então o pixel é igual no 1º e no último quadro."""
    photo, res, pieces = _scene(light_far=True)
    st = L.build_layers(photo, res, pieces)
    assert st.corrected[8, 8] and not st.dynamic[8, 8]
    assert np.array_equal(st.plate[8, 8], photo[8, 8])
    full = L.composite(st, range(len(st.layers)))
    empty = L.composite(st, [])
    changed = np.abs(full.astype(int) - empty).max(axis=2) > 0
    assert not (changed & ~st.dynamic).any()


def test_paint_order_rug_back_cushion_on_sofa():
    photo, res, pieces = _scene()
    st = L.build_layers(photo, res, pieces)
    order = [l.label for l in st.layers]
    assert order.index("rug") < order.index("sofa") < order.index("cushion")
    i_c, i_s = st.index(2), st.index(1)
    assert st.supports == {i_c: i_s}
    assert st.front[i_c, i_s] and st.front[i_s, st.index(3)]


def test_occlusion_uses_depth_and_falls_back_to_bottom():
    a = np.zeros((60, 80), bool)
    b = np.zeros((60, 80), bool)
    a[10:30, 10:40] = True      # termina mais acima
    b[20:50, 40:70] = True      # encosta em a e termina mais embaixo
    front, _ = L.occlusion_graph(["chair", "table"], [a, b])
    assert front[1, 0] and not front[0, 1]
    depth = np.full((60, 80), 5.0, np.float32)
    depth[a] = 2.0               # a está mais perto da câmera
    front, _ = L.occlusion_graph(["chair", "table"], [a, b], depth[::4, ::4])
    assert front[0, 1] and not front[1, 0]


def test_toggling_a_layer_only_changes_its_own_region():
    """Base do anti-morph entre quadros consecutivos: ligar/desligar uma peça só muda pixels da região dela."""
    photo, res, pieces = _scene()
    st = L.build_layers(photo, res, pieces)
    n = len(st.layers)
    for k in range(n):
        others = [i for i in range(n) if i != k]
        d = np.abs(L.composite(st, others).astype(int) - L.composite(st, range(n))).max(axis=2) > 0
        lay = st.layers[k]
        own = np.zeros((H, W), bool)
        x0, y0, x1, y1 = lay.box
        own[y0:y1, x0:x1] = lay.support | lay.ring | lay.effect
        assert not (d & ~own).any(), lay.label


def test_moving_piece_and_partial_reveal_stay_valid():
    photo, res, pieces = _scene()
    st = L.build_layers(photo, res, pieces)
    i = st.index(1)
    lay = st.layers[i]
    up = L.composite(st, {i: L.Placement(dx=0, dy=-25, shade=0.3, contact=0.0)})
    assert up.dtype == np.uint8 and up.shape == photo.shape
    # o sofá subiu: a linha de baixo dele mostra a placa (sem delta de contato)
    assert np.abs(up[88, 80].astype(int) - st.plate[88, 80]).max() <= 1
    off = L.composite(st, {i: L.Placement(dx=500, contact=0.0)})          # fora do quadro: não quebra
    assert np.array_equal(off, L.composite(st, []))
    rug = st.index(3)
    x0, y0, x1, y1 = st.layers[rug].box
    half = np.zeros((y1 - y0, x1 - x0), np.float32)
    half[:, : (x1 - x0) // 2] = 1.0
    f = L.composite(st, {rug: L.Placement(alpha_mul=half, contact=0.0)})
    assert np.abs(f[112, x1 - 5].astype(int) - st.plate[112, x1 - 5]).max() <= 1


def test_rejects_plate_that_differs_outside_mask():
    photo, res, pieces = _scene()
    bad = SimpleNamespace(plate=res.plate.copy(), mask=res.mask, shadow=res.shadow, effect=res.effect)
    bad.plate[0, 0] = bad.plate[0, 0] ^ 1
    with pytest.raises(ValueError):
        L.build_layers(photo, bad, pieces)


def test_no_pieces_gives_empty_stack():
    photo, res, _ = _scene(shadow=False)
    res0 = SimpleNamespace(plate=photo.copy(), mask=np.zeros((H, W), bool), shadow=np.zeros((H, W), bool))
    st = L.build_layers(photo, res0, [])
    assert st.layers == [] and np.array_equal(L.composite(st, []), photo)
    assert st.report()["camadas"]["n"] == 0


def test_cushion_on_blanket_inherits_sofa_support():
    """Almofada cuja base está sobre a manta (pequena) que está sobre o sofá: o apoio é o sofá e ela fica na
    frente dele (antes a regra "termina mais embaixo" punha o sofá na frente da almofada)."""
    sofa = np.zeros((H, W), bool)
    sofa[50:100, 20:140] = True
    blanket = np.zeros((H, W), bool)
    blanket[45:60, 30:110] = True
    cush = np.zeros((H, W), bool)
    cush[25:45, 40:70] = True
    sofa &= ~blanket
    order, front, sup = L.paint_order(["sofa", "blanket", "cushion"], [sofa, blanket, cush])
    assert sup == {1: 0, 2: 0}
    assert front[2, 0] and not front[0, 2]
    assert order.index(0) < order.index(2)


def test_effect_reached_by_far_piece_is_not_corrected_on_plate():
    """Luz da placa ao alcance do armário (planejado) mas com um vaso pequeno mais perto: continua efeito do
    armário (região dinâmica), não volta a ser a foto mobiliada na sala vazia."""
    photo, res, pieces = _scene(shadow=False)
    a_cab, h_cab = _soft_rect(5, 25, 30, 130)
    a_vase, h_vase = _soft_rect(26, 30, 132, 136)
    pieces = pieces + [_piece(4, "cabinet", a_cab, h_cab, kind="planejado"), _piece(5, "vase", a_vase, h_vase)]
    light = np.zeros((H, W), bool)
    light[27:40, 118:138] = True
    img = photo.astype(np.float32)
    img[light] += 50.0
    for a, c in ((a_cab, (90.0, 60.0, 40.0)), (a_vase, (20.0, 120.0, 20.0))):
        img = (1 - a[..., None]) * img + a[..., None] * np.array(c)
    photo = np.clip(img, 0, 255).round().astype(np.uint8)
    core = rp.core_mask(pieces, (H, W))
    effect = light & ~core
    mask = core | rp._dil(effect, 3)
    plate = photo.copy()
    pt = np.clip(_floor(7), 0, 255).round().astype(np.uint8)
    plate[mask] = pt[mask]
    res = SimpleNamespace(plate=plate, mask=mask, shadow=np.zeros((H, W), bool), effect=effect)
    st = L.build_layers(photo, res, pieces)
    assert not (st.corrected & effect).any()
    assert st.stats["placa_corrigida_fantasma_pct"] == 0.0
    cab = st.layers[st.index(4)]
    assert cab.effect.sum() > 0
    full = L.composite(st, range(len(st.layers)))
    assert np.abs(full.astype(int) - photo).max() <= L.RECON_TOL


def test_partial_reveal_masks_contact_delta():
    """Tapete meio desenrolado com contato ligado: o delta de contato não aparece na parte ainda enrolada."""
    photo, res, pieces = _scene()
    st = L.build_layers(photo, res, pieces)
    rug = st.index(3)
    lay = st.layers[rug]
    x0, y0, x1, y1 = lay.box
    half = np.zeros((y1 - y0, x1 - x0), np.float32)
    half[:, : (x1 - x0) // 2] = 1.0
    f = L.composite(st, {rug: L.Placement(alpha_mul=half, contact=1.0)})
    hidden = np.zeros(photo.shape[:2], bool)
    hidden[y0:y1, x0 + (x1 - x0) // 2:x1] = True
    assert np.array_equal(f[hidden], st.plate[hidden])


# ------------------------------------------------------------------------------------------- fixtures reais
_cache: dict = {}


def _stack(room_fixture, name):
    if name not in _cache:
        f = room_fixture(name)
        s = ri.detect_instances(f.photo, f.seg, ri.PanopticResult(f.panoptic_ids, f.panoptic_segments), sam=None,
                                geometry=f.geometry, room=f.room)
        depth = f.geometry["points"][..., 2].astype(np.float32)
        r = rp.build_plate(f.photo, s.removal(), f.seg, f.room, depth, rp.telea_inpaint, assess_quality=False)
        _cache[name] = (f, s, r, L.build_layers(f.photo, r, s.removal(), depth))
    return _cache[name]


@pytest.mark.parametrize("name", ["sala_sofas_brancos", "nice_terrazzo"])
def test_fixture_acceptance_reconstruction_and_morph(room_fixture, name):
    """Aceite PT-V3-5 nas fotos reais (placa Telea; a pilha não depende da qualidade da placa):
    pilha completa x foto <= 2 níveis e 0% de pixels fora de alpha ∪ efeito mudando do 1º ao último quadro."""
    f, s, r, st = _stack(room_fixture, name)
    assert len(st.layers) == len(s.removal())
    full = L.composite(st, range(len(st.layers)))
    empty = L.composite(st, [])
    assert np.abs(full.astype(int) - f.photo).max() <= L.RECON_TOL
    assert np.array_equal(empty, st.plate)
    assert np.array_equal(st.plate[~r.mask], f.photo[~r.mask])
    assert L.morph_fraction(st, empty, full) == 0.0
    # a medida da crítica (alpha < 0,05 e diferença > 12) agora só acontece perto das peças
    alpha = np.zeros(r.mask.shape, np.float32)
    for p in s.removal():
        alpha = np.maximum(alpha, p.alpha)
    crit = (np.abs(empty.astype(int) - f.photo).max(axis=2) > 12) & (alpha < 0.05)
    assert not (crit & ~st.dynamic).any()
    # a correção na placa é exceção (o que não pertence a nenhuma peça), não a regra
    assert st.stats["placa_corrigida_da_mascara_pct"] < 5.0
    rep = st.report()["camadas"]
    assert rep["n"] == len(st.layers) and rep["reconstrucao_max"] <= L.RECON_TOL


def test_fixture_paint_order_supports(room_fixture):
    _, _, _, st = _stack(room_fixture, "nice_terrazzo")
    order = [l.label for l in st.layers]
    assert order[0] == "rug" and order[1] == "rug"
    sup = {st.layers[i].label: st.layers[j].label for i, j in st.supports.items()}
    assert sup.get("pillow") == "bed" and sup.get("vase") == "table"
