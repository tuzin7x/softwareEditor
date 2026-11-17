"""PT-V3-3: máscara de efeitos e sala vazia por plano (rápidos, sem modelos: inpainter Telea do OpenCV)."""
from types import SimpleNamespace

import numpy as np
import pytest

from mobilia.analysis.room import Opening, RoomModel, Wall
from mobilia.analysis.segmentation import GROUP_IDS
from mobilia.removal import instances as ri
from mobilia.removal import plate as rp

G_FLOOR, G_WALL, G_CEIL, G_WIN = (GROUP_IDS[k] for k in ("floor", "wall", "ceiling", "window"))
BEIGE = np.array([205.0, 190.0, 165.0])
BASEBOARD = np.array([95.0, 60.0, 40.0])
FLOOR_RGB = np.array([150.0, 140.0, 130.0])
BB_H = 0.10


# ------------------------------------------------------------------------------------------- sala sintética
def _room(W=480, H=360, openings=()):
    """Câmera a 1,5 m olhando +y; parede de fundo em y = 6 (x de -6 a 6); teto a 2,6 m."""
    f = 0.9 * W
    M = np.eye(4)
    M[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    M[:3, 3] = [0, 0, 1.5]
    walls = [Wall(0, [-6.0, 6.0], [6.0, 6.0], [0.0, -1.0], 2.6, 1000, 0.01)]
    return RoomModel(dict(width=W, height=H, fx=f, fy=f, cx=W / 2, cy=H / 2), M.tolist(), 1.5, 2.6, True, walls,
                     list(openings), [0, 0], 0.05, [1, 1])


def _render(room, W=480, H=360, seed=0):
    """Foto sintética: piso cinza com grão, parede bege com gradiente de luz e rodapé marrom, teto branco."""
    rng = np.random.default_rng(seed)
    box = rp.RoomBox(room, W, H)
    img = np.zeros((H, W, 3), np.float32)
    groups = np.full((H, W), 255, np.uint8)
    fl = box.label == rp.FLOOR
    img[fl] = FLOOR_RGB
    groups[fl] = G_FLOOR
    wl = box.label == 0
    xs = np.arange(W)[None].repeat(H, 0)
    shade = (1.0 - 0.15 * xs / W)[..., None]
    img[wl] = (BEIGE * shade)[wl]
    bb = wl & (box.Z < BB_H)
    img[bb] = BASEBOARD
    groups[wl] = G_WALL
    cl = box.label == rp.CEIL
    img[cl] = 240
    groups[cl] = G_CEIL
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).round().astype(np.uint8), groups, box


def _piece(mask, kind="solto", label="cabinet"):
    return SimpleNamespace(mask=mask, alpha=mask.astype(np.float32), kind=kind, label=label)


def _wall_floor_row(box, x):
    """Linha da imagem (y) onde a parede encontra o piso na coluna x."""
    col = box.label[:, x]
    return int(np.nonzero(col == rp.FLOOR)[0].min())


# ------------------------------------------------------------------------------------------- testes sintéticos
def test_raycast_labels_synthetic_box():
    room = _room()
    box = rp.RoomBox(room, 480, 360)
    assert box.label[350, 240] == rp.FLOOR and box.label[5, 240] == rp.CEIL and box.label[150, 240] == 0
    # ponto na parede: z cresce para cima, s ao longo da parede
    assert 0.0 < box.Z[150, 240] < 2.6 and abs(box.S[150, 240] - 6.0) < 0.05


def test_plain_wall_with_baseboard_and_straight_line():
    """Parede quase toda escondida (§3.6): pintura lisa na cor medida, rodapé reto na altura certa e a linha
    parede/piso a <= 3 px da projetada (aceite PT-V3-3)."""
    room = _room()
    img, groups, box = _render(room)
    H, W = groups.shape
    yline = _wall_floor_row(box, 240)
    piece = np.zeros((H, W), bool)
    piece[60:yline + 25, 40:440] = True        # "armário" de ponta a ponta cobrindo a parede e o rodapé
    seg = SimpleNamespace(groups=groups)
    r = rp.build_plate(img, [_piece(piece, "planejado")], seg, room, inpaint=rp.telea_inpaint)
    assert np.array_equal(r.plate[~r.mask], img[~r.mask])
    assert r.info["planos"]["parede0"]["metodo"] == "liso"
    assert r.info["planos"]["parede0"]["rodape_m"] == pytest.approx(BB_H, abs=0.025)
    # rodapé redesenhado: logo acima da linha a cor é a do rodapé; bem acima, a da parede
    for x in (120, 240, 360):
        y = _wall_floor_row(box, x)
        assert np.abs(r.plate[y - 3, x].astype(float) - BASEBOARD).max() < 25
        assert np.abs(r.plate[y - 40, x].astype(float) - BEIGE * (1 - 0.15 * x / W)).max() < 20
    q = r.quality
    assert q.line["distancia_px"] <= 3.0 and q.line["continuidade"] >= 0.8
    assert q.per_plane["parede0"]["costura_de"] <= 3.0
    assert q.ok, q.reasons


def test_tile_grid_is_detected_and_repeated_aligned():
    rng = np.random.default_rng(1)
    H, W, p = 200, 260, 20
    img = np.full((H, W, 3), 200, np.uint8)
    img[::p] = 120
    img[:, ::p] = 120                                  # rejunte a cada 20 px
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    hole = np.zeros((H, W), bool)
    hole[50:150, 70:190] = True
    known = ~hole
    per = rp.detect_period(img[..., 0], known)
    assert per == (p, p)
    filled, left = rp.periodic_fill(img, known, hole, per)
    assert not left.any()
    grout = np.zeros((H, W), bool)
    grout[::p] = True
    grout[:, ::p] = True
    assert filled[hole & grout, 0].mean() < 135 and filled[hole & ~grout, 0].mean() > 185


def test_no_grid_on_plain_or_noisy_surface():
    rng = np.random.default_rng(2)
    g = np.clip(150 + rng.normal(0, 8, (200, 200)), 0, 255)
    known = np.ones((200, 200), bool)
    known[60:140, 60:140] = False
    assert rp.detect_period(g, known) is None
    assert rp.detect_period(np.full((200, 200), 128.0), known) is None


def test_harmonize_removes_tone_step_and_islands():
    H, W = 160, 200
    img = np.full((H, W, 3), 120, np.uint8)
    hole = np.zeros((H, W), bool)
    hole[40:120, 50:150] = True
    img[hole] = 140                                    # preenchimento mais claro (costura)
    img[60:100, 75:125] = 30                           # ilha escura grande no meio
    out = rp.harmonize(img, ~hole, hole, island_sigma=6.0, tol=2.0)
    assert np.array_equal(out[~hole], img[~hole])
    edge = hole & ~rp._ero(hole, 4)
    assert abs(out[edge].astype(float).mean() - 120) < 4
    assert out[70:90, 90:110].astype(float).mean() > 90    # ilha puxada para a faixa do anel


def test_push_pull_fills_unknown_smoothly():
    v = np.zeros((64, 64), np.float32)
    w = np.zeros((64, 64), np.float32)
    v[:, :8], w[:, :8] = 10, 1
    v[:, -8:], w[:, -8:] = 30, 1
    out = rp.push_pull(v, w)
    assert np.allclose(out[:, :8], 10) and np.allclose(out[:, -8:], 30)
    mid = out[32, 8:-8]
    assert 10 <= mid.min() and mid.max() <= 30 and mid[-1] > mid[0]


def test_effects_shadow_and_light_are_separated_and_must_touch_the_piece():
    H, W = 400, 600                                    # anel de efeitos = 1,2% da diagonal = 8 px
    groups = np.full((H, W), G_FLOOR, np.uint8)
    photo = np.full((H, W, 3), 150, np.uint8)
    piece = np.zeros((H, W), bool)
    piece[40:120, 100:160] = True
    photo[piece] = 60
    photo[120:150, 100:160] = 90                       # sombra de contato logo abaixo
    photo[60:110, 160:170] = 215                       # luz/reflexo colado à direita
    photo[170:190, 10:40] = 60                         # mancha longe: não é efeito da peça
    plate1 = np.full((H, W, 3), 150, np.uint8)         # placa provisória "limpa"
    core = rp.core_mask([_piece(piece)], (H, W))
    cand = rp.effect_candidates([_piece(piece)], groups, core)
    eff, dark, light = rp.verify_effects(photo, plate1, cand, core)
    assert dark[125:145, 110:150].mean() > 0.8
    assert light[70:100, 164:168].mean() > 0.6
    assert not eff[170:190, 10:40].any()
    assert not (dark & light).any()


def test_builtin_light_band_reaches_the_wall_below_upper_cabinet():
    H, W = 200, 300
    groups = np.full((H, W), G_WALL, np.uint8)
    up = np.zeros((H, W), bool)
    up[20:60, 50:250] = True                           # aéreo
    c_loose = rp.effect_candidates([_piece(up, "solto")], groups, rp.core_mask([_piece(up)], (H, W)))
    c_built = rp.effect_candidates([_piece(up, "planejado")], groups, rp.core_mask([_piece(up)], (H, W)))
    below = np.zeros((H, W), bool)
    below[70:82, 60:240] = True                        # backsplash iluminado pelo LED
    assert c_built[below].mean() > 0.9 and c_loose[below].mean() < 0.1


def test_window_opening_is_protected_never_gets_wall_texture():
    op = Opening("window", 0, 6.5, 8.5, 0.9, 2.2, [0, 0, 0, 0], [0.8, 0.8, 0.9], 0.8)
    room = _room(openings=[op])
    img, groups, box = _render(room)
    inwin = (box.label == 0) & (box.S >= 6.5) & (box.S <= 8.5) & (box.Z >= 0.9) & (box.Z <= 2.2)
    img[inwin] = (90, 140, 200)                        # vista azul pela janela
    groups[inwin] = G_WIN
    piece = np.zeros(groups.shape, bool)
    ys, xs = np.nonzero(inwin)
    piece[ys.min() + 20: ys.max() + 40, xs.min() + 10: xs.max() - 10] = True   # móvel na frente da janela
    r = rp.build_plate(img, [_piece(piece)], SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint)
    prot = r.mask & inwin
    assert (r.labels[prot] == rp.PROTECTED).mean() > 0.95
    assert r.info["planos"]["protegido"]["metodo"] == "lama-local"
    # a vista continua azulada (não virou parede bege)
    assert (r.plate[prot][:, 2].astype(int) - r.plate[prot][:, 0]).mean() > 30   # parede bege daria < 0


def test_clutter_on_wall_is_not_context():
    H, W = 400, 600
    img = np.full((H, W, 3), 200, np.uint8)
    img[190:200, 290:300] = 30                         # tomada
    surf = np.ones((H, W), bool)
    m = rp.clutter_mask(img, surf)
    assert m[192:198, 292:298].all() and m.mean() < 0.01


def test_gate_falls_back_to_keeping_builtins(monkeypatch):
    room = _room()
    img, groups, box = _render(room)
    H, W = groups.shape
    loose = np.zeros((H, W), bool)
    loose[250:320, 60:140] = True
    built = np.zeros((H, W), bool)
    built[150:240, 250:400] = True                     # 7,8% da foto: abaixo do limite de "mantidos por padrão"
    insts = [_piece(loose, "solto", "chair"), _piece(built, "planejado", "cabinet")]
    inst_set = SimpleNamespace(instances=insts,
                               removal=lambda keep=False: [i for i in insts if not (keep and i.kind == "planejado")])
    assert rp.keep_builtins_reason(inst_set, (H, W)) == ""
    calls = []

    def fake_redetect(plate_rgb):
        calls.append(1)
        return None, None

    def fake_found(seg2, pan2, mask, **kw):            # 1ª placa: "armário" reencontrado; 2ª: limpa
        return [{"rotulo": "cabinet", "area_pct": 3.0}] if len(calls) == 1 else []

    monkeypatch.setattr(rp, "redetected_furniture", fake_found)
    r = rp.build_plate_gated(img, inst_set, SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                             redetect=fake_redetect)
    names = [t["estrategia"] for t in r.info["tentativas"]]
    assert names[:2] == ["crua_por_plano", "planejados_mantidos"]    # a crua reprovou: tenta manter os planejados
    assert not r.quality.redetected
    rep = r.report()["placa"]
    assert rep["planejados_mantidos"] is r.kept_builtins
    if r.kept_builtins:                                  # mantidos: dito no relatório e no aviso
        assert not r.mask[built & ~rp._dil(loose, 30)].any()
        assert any("planejados mantidos" in w for w in r.warnings) and rep["cobertura"]["planejados_mantidos"]
    else:                                                # uma alternativa crua aprovou: tudo removido
        assert r.quality.ok and rep["cobertura"]["tudo_removido"]


def _inst_set(insts):
    return SimpleNamespace(instances=insts,
                           removal=lambda keep=False: [i for i in insts if not (keep and i.kind == "planejado")])


def test_keep_builtins_reason_thresholds():
    H, W = 100, 100
    def box(y0, y1, x0, x1):
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = True
        return m
    chair = _piece(box(60, 80, 0, 20), "solto", "chair")                      # 4%
    assert rp.keep_builtins_reason(_inst_set([chair]), (H, W)) == ""            # sem planejados
    small = _piece(box(0, 5, 50, 60), "planejado", "cabinet")                   # 0,5%: não muda nada
    assert rp.keep_builtins_reason(_inst_set([chair, small]), (H, W)) == ""
    big = _piece(box(0, 30, 40, 80), "planejado", "cabinet")                    # 12%: cozinha planejada
    assert "planejados cobrem 12%" in rp.keep_builtins_reason(_inst_set([chair, big]), (H, W))
    sofa = _piece(box(40, 80, 0, 100), "solto", "sofa")                         # 40% + 5% de estante
    shelf = _piece(box(0, 10, 50, 100), "planejado", "bookcase")
    assert "reconstruiria 45%" in rp.keep_builtins_reason(_inst_set([sofa, shelf]), (H, W))


def _big_builtin_scene():
    room = _room()
    img, groups, box = _render(room)
    H, W = groups.shape
    loose = np.zeros((H, W), bool)
    loose[250:320, 60:140] = True
    built = np.zeros((H, W), bool)
    built[100:240, 200:440] = True                     # 19% da foto
    insts = [_piece(loose, "solto", "chair"), _piece(built, "planejado", "cabinet")]
    return room, img, groups, loose, built, insts


def test_gate_tries_raw_room_first_even_with_large_builtins(monkeypatch):
    """§3.6 (rodada 1 de correções): no modo B os planejados também saem; a sala crua é sempre a 1ª tentativa e o
    risco ("planejados cobrem 19%") só vai para o relatório. Se a crua aprova, entrega a crua."""
    room, img, groups, loose, built, insts = _big_builtin_scene()
    monkeypatch.setattr(rp, "redetected_furniture", lambda *a, **k: [])
    r = rp.build_plate_gated(img, _inst_set(insts), SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                             redetect=lambda p: (None, None), alternatives=False)
    assert r.info["tentativas"][0]["estrategia"] == "crua_por_plano"
    rep = r.report()["placa"]
    assert "planejados cobrem" in rep["risco_planejados"] and rep["planejados_por_padrao"] is None
    if not r.kept_builtins:
        assert r.mask[built & ~rp._dil(loose, 30)].all()
        assert rep["cobertura"]["tudo_removido"] is True
    else:                                               # crua reprovada: o motivo é dito, nunca em silêncio
        assert rep["cobertura"]["planejados_mantidos"] == ["cabinet"] and rep["cobertura"]["motivo"]
        assert any("planejados mantidos" in w for w in r.warnings)


def test_gate_keeps_large_builtins_first_when_configured(monkeypatch):
    """Comportamento antigo, por configuração: planejados grandes ficam já na 1ª tentativa."""
    room, img, groups, loose, built, insts = _big_builtin_scene()
    monkeypatch.setattr(rp, "redetected_furniture", lambda *a, **k: [])
    monkeypatch.setattr(rp, "KEEP_BUILTINS_FIRST", True)
    r = rp.build_plate_gated(img, _inst_set(insts), SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                             redetect=lambda p: (None, None), alternatives=False)
    assert r.kept_builtins
    assert [t["estrategia"] for t in r.info["tentativas"]] == ["planejados_mantidos"]
    assert not r.mask[built & ~rp._dil(loose, 30)].any()
    assert "planejados cobrem" in r.report()["placa"]["planejados_por_padrao"]
    assert r.report()["placa"]["cobertura"]["planejados_mantidos"] == ["cabinet"]
    assert any("por padrão" in w for w in r.warnings)


def test_gate_warns_when_still_bad(monkeypatch):
    room = _room()
    img, groups, _ = _render(room)
    m = np.zeros(groups.shape, bool)
    m[250:320, 60:140] = True
    insts = [_piece(m, "solto", "sofa")]
    inst_set = SimpleNamespace(instances=insts, removal=lambda keep=False: insts)
    monkeypatch.setattr(rp, "redetected_furniture", lambda *a, **k: [{"rotulo": "sofa", "area_pct": 2.0}])
    r = rp.build_plate_gated(img, inst_set, SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                             redetect=lambda p: (None, None))
    assert not r.quality.ok and r.quality.score < 60
    assert any("modo A" in w for w in r.warnings)
    assert "móvel reaparece" in " ".join(r.quality.reasons)


def test_no_room_model_falls_back_to_image_space():
    img = np.full((120, 160, 3), 100, np.uint8)
    m = np.zeros((120, 160), bool)
    m[40:80, 50:110] = True
    groups = np.full((120, 160), G_FLOOR, np.uint8)
    r = rp.build_plate(img, [_piece(m)], SimpleNamespace(groups=groups), None, inpaint=rp.telea_inpaint)
    assert np.array_equal(r.plate[~r.mask], img[~r.mask])
    assert any("espaço da imagem" in w for w in r.warnings)


# ------------------------------------------------------------------------------------------- fixtures reais
@pytest.mark.parametrize("name", ["sala_sofas_brancos", "nice_terrazzo"])
def test_fixture_plate_is_photo_outside_mask(room_fixture, name):
    f = room_fixture(name)
    s = ri.detect_instances(f.photo, f.seg, ri.PanopticResult(f.panoptic_ids, f.panoptic_segments), sam=None,
                            geometry=f.geometry, room=f.room)
    depth = f.geometry["points"][..., 2].astype(np.float32)
    r = rp.build_plate(f.photo, s.removal(), f.seg, f.room, depth, rp.telea_inpaint, assess_quality=True)
    assert np.array_equal(r.plate[~r.mask], f.photo[~r.mask])
    # o Telea faz uma passada 1 pior que o LaMa, então a máscara de efeitos é maior aqui que no produto
    assert r.mask.mean() < r.core.mean() + 0.3 and r.core.sum() <= r.mask.sum()
    assert (r.effect & r.core).sum() == 0
    planes = r.info["planos"]
    assert "piso" in planes and any(k.startswith("parede") for k in planes)
    assert (r.labels[r.mask] == rp.FLOOR).mean() > 0.2
    rep = r.report()["placa"]
    assert set(rep["qualidade"]) >= {"nota", "aprovada", "por_plano", "linha_parede_piso", "motivos"}


def test_nice_window_pixels_are_protected(room_fixture):
    """Mesa alta na frente da janela de Nice: o vão da janela é protegido (o protótipo inventou ladrilhos ali)."""
    f = room_fixture("nice_terrazzo")
    s = ri.detect_instances(f.photo, f.seg, ri.PanopticResult(f.panoptic_ids, f.panoptic_segments), sam=None,
                            geometry=f.geometry, room=f.room)
    r = rp.build_plate(f.photo, s.removal(), f.seg, f.room, None, rp.telea_inpaint, assess_quality=False)
    win = np.zeros_like(r.mask)
    win[510:570, 960:1220] = True                      # guarda-corpo da sacada atrás da mesa alta
    sel = win & r.mask
    assert sel.sum() > 500
    assert (r.labels[sel] == rp.PROTECTED).mean() > 0.8


# ------------------------------------------------------------------------------------------- revisão PT-V3-3
def test_piece_resting_on_kept_object_is_not_filled_as_floor():
    """Copo sobre a bancada mantida: a caixa diz "piso" atrás dele, mas em volta só há bancada. Não pode virar
    uma mancha de piso no meio da bancada: vai para o espaço da imagem (contexto local)."""
    room = _room()
    img, groups, box = _render(room)
    H, W = groups.shape
    counter = np.zeros((H, W), bool)
    counter[230:330, 120:360] = True                        # bancada (objeto que fica) sobre o piso
    img[counter] = (40, 120, 60)
    groups[counter] = 255
    cup = np.zeros((H, W), bool)
    cup[260:290, 220:250] = True
    img[cup] = (250, 30, 30)
    r = rp.build_plate(img, [_piece(cup, label="vase")], SimpleNamespace(groups=groups), room,
                       inpaint=rp.telea_inpaint, assess_quality=False)
    assert r.info["planos"].get("apoiado_em_objeto_px", 0) > 0
    assert (r.labels[cup] == rp.OTHER).all()
    fill = r.plate[cup].astype(float).mean(0)
    assert np.abs(fill - (40, 120, 60)).max() < 25, fill          # cor da bancada, não do piso


def test_plane_without_visible_material_is_not_filled_by_plane_and_gate_rejects_unmeasured():
    """Piso todo coberto (nada do material à vista): o LaMa na vista retificada inventaria a cor. Os pixels vão
    para o espaço da imagem; e um plano preenchido que não pôde ser medido reprova a nota."""
    room = _room()
    img, groups, box = _render(room)
    H, W = groups.shape
    fl = box.label == rp.FLOOR
    groups[fl] = G_CEIL           # nenhum pixel de piso visível (rótulo de superfície, mas de outro material)
    m = np.zeros((H, W), bool)
    m[250:330, 150:330] = True
    out, labels, info = rp.fill_planes(rp.telea_inpaint, img, m, groups, box)
    assert info.get("piso", {}).get("metodo") == "sem_contexto"
    assert not (labels[m] == rp.FLOOR).any()
    # nota: um plano com área relevante e sem referência visível do material reprova
    lab = np.where(m, rp.FLOOR, labels)
    q = rp.assess(img, out, m, lab, groups, box)
    assert not q.ok and any("sem referência" in s for s in q.reasons)


def test_gate_rejects_fill_whose_mean_color_is_off():
    """A média de cor do preenchimento muito longe da do anel (aceite do plano: ΔE da média) reprova, mesmo com
    a costura local boa (mancha clara no meio do piso)."""
    room = _room()
    img, groups, box = _render(room)
    H, W = groups.shape
    m = np.zeros((H, W), bool)
    m[230:350, 100:380] = True
    plate = img.copy()
    yy, xx = np.nonzero(m)
    cy, cx = yy.mean(), xx.mean()
    rr = np.hypot((yy - cy) / 60.0, (xx - cx) / 140.0)
    glow = np.clip(1.0 - rr, 0, 1) ** 0.5 * 90.0                 # mancha clara no centro, borda casada
    plate[yy, xx] = np.clip(plate[yy, xx] + glow[:, None], 0, 255).astype(np.uint8)
    lab = np.where(m, box.label, box.label)
    q = rp.assess(img, plate, m, lab, groups, box)
    s = q.per_plane["piso"]
    assert s["costura_de"] < rp.LIMITS["costura_de_max"], s
    assert not q.ok and any("destoa" in t for t in q.reasons), (q.reasons, s)


def test_gate_alternatives_record_attempts_and_skip_when_only_area(monkeypatch):
    """PT-V3-8: reprovada por costura -> tenta LaMa na imagem e máscara larga (registradas em ``tentativas``);
    reprovada só pela área grande -> as alternativas não ajudam e não rodam."""
    room = _room()
    img, groups, _ = _render(room)
    m = np.zeros(groups.shape, bool)
    m[250:320, 60:140] = True
    insts = [_piece(m, "solto", "sofa")]
    inst_set = SimpleNamespace(instances=insts, removal=lambda keep=False: insts)
    orig = rp.assess
    reason = ["costura visível no(a) piso (ΔE 9.0)"]

    def bad(*a, **k):
        q = orig(*a, **k)
        q.ok, q.reasons, q.score = False, list(reason), 40.0
        return q

    monkeypatch.setattr(rp, "assess", bad)
    seen = []
    r = rp.build_plate_gated(img, inst_set, SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                             on_attempt=lambda name, res: seen.append((name, res.info["estrategia"])))
    assert seen == [("crua_por_plano", "planos"), ("lama_imagem", "imagem"), ("mascara_ampla", "planos")]
    t = r.report()["placa"]["tentativas"]
    assert [x["estrategia"] for x in t] == [s for s, _ in seen] and t[2]["dilatacao_nucleo"] > t[0]["dilatacao_nucleo"]
    assert np.array_equal(r.plate[~r.mask], img[~r.mask])            # toda tentativa respeita a regra absoluta
    assert any("modo A" in w for w in r.warnings)
    reason[:] = ["área reconstruída muito grande (80% da foto)"]
    seen.clear()
    rp.build_plate_gated(img, inst_set, SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                         on_attempt=lambda name, res: seen.append(name))
    assert seen == ["crua_por_plano"]
    # área acima do limite junto com outros motivos (foto lotada): as alternativas também não podem passar
    reason[:] = ["costura visível no(a) piso (ΔE 9.0)", "área reconstruída muito grande (80% da foto)"]
    seen.clear()
    rp.build_plate_gated(img, inst_set, SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                         on_attempt=lambda name, res: seen.append(name))
    assert seen == ["crua_por_plano"]


def test_image_strategy_keeps_photo_outside_mask():
    room = _room()
    img, groups, _ = _render(room)
    m = np.zeros(groups.shape, bool)
    m[250:320, 60:140] = True
    r = rp.build_plate(img, [_piece(m)], SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                       strategy="imagem")
    assert r.info["estrategia"] == "imagem" and np.array_equal(r.plate[~r.mask], img[~r.mask])
    assert r.quality is not None
    with pytest.raises(ValueError):
        rp.build_plate(img, [_piece(m)], SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                       strategy="xyz")


def test_gate_never_prefers_wide_mask_that_breaks_area_limit(monkeypatch):
    """A máscara larga que só reprova pela área (passou do limite) não vence uma placa dentro do limite com o
    mesmo número de motivos (caso real da sala de sofás brancos a 960 px)."""
    room = _room()
    img, groups, _ = _render(room)
    m = np.zeros(groups.shape, bool)
    m[250:320, 60:140] = True
    insts = [_piece(m, "solto", "sofa")]
    inst_set = SimpleNamespace(instances=insts, removal=lambda keep=False: insts)
    orig = rp.assess
    base = []

    def fake(photo, plate, mask, *a, **k):
        q = orig(photo, plate, mask, *a, **k)
        base.append(base[0] if base else int(mask.sum()))
        wide = mask.sum() > base[0] * 1.05            # só a máscara larga cresce (a 2ª tentativa usa a mesma)
        q.ok, q.score = False, 59.0
        q.raw_score = 90.0 if wide else 50.0
        q.area_generated = 0.72 if wide else 0.30
        q.reasons = ["área reconstruída muito grande (72% da foto)"] if wide else ["cor do preenchimento destoa no(a) piso (ΔE 11.6)"]
        return q

    monkeypatch.setattr(rp, "assess", fake)
    areas = {}
    r = rp.build_plate_gated(img, inst_set, SimpleNamespace(groups=groups), room, inpaint=rp.telea_inpaint,
                             on_attempt=lambda name, res: areas.update({name: res.quality.area_generated}))
    assert areas["mascara_ampla"] == 0.72 and areas["crua_por_plano"] == 0.30     # a máscara larga "melhorou" a nota
    assert r.info["estrategia_portao"] != "mascara_ampla" and r.quality.area_generated == 0.30


def test_gate_rejects_large_fill_without_measured_plane():
    """Robustez r1 (falsos aceites): geometria da sala não achada (HDR, retrato, piso todo coberto) deixa quase todo
    o preenchimento "sem plano", que só tem o limite frouxo de costura. Área gerada >= 30% com a maioria sem plano
    reprova; a mesma área medida por plano, ou uma área pequena sem plano, não é afetada por esta regra."""
    room = _room()
    img, groups, box = _render(room)
    H, W = groups.shape
    m = np.zeros((H, W), bool)
    m[60:300, 40:440] = True                                       # ~56% da foto
    plate = img.copy()                                             # preenchimento perfeito (cor/costura certas)
    lab = np.where(m, rp.OTHER, box.label)
    q = rp.assess(img, plate, m, lab, groups, box)
    assert q.unplanned > 0.99 and q.to_dict()["sem_plano_pct"] > 99
    assert not q.ok and any("geometria da sala" in s for s in q.reasons), q.reasons
    # mesma área com os planos da caixa: a regra não dispara
    q2 = rp.assess(img, plate, m, box.label, groups, box)
    assert not any("geometria da sala" in s for s in q2.reasons), q2.reasons
    # área pequena (6%) toda sem plano: regra não vale (ex.: bancada/vaidade de banheiro aprovada)
    s = np.zeros((H, W), bool)
    s[150:210, 150:320] = True
    q3 = rp.assess(img, plate, s, np.where(s, rp.OTHER, box.label), groups, box)
    assert not any("geometria da sala" in t for t in q3.reasons), q3.reasons


def test_fill_color_follows_light_gradient_not_whole_ring_mean():
    """Robustez r1 (falsas recusas): a luz do piso cai para o fundo da sala e o buraco encosta na parede, então o
    anel do piso fica quase todo do lado claro (frente). O preenchimento que segue a luz do lugar não destoa, embora a
    média do anel inteiro puxe para o claro; uma cor errada por igual no mesmo buraco continua reprovando."""
    room = _room()
    img, groups, box = _render(room)
    H, W = groups.shape
    fl = box.label == rp.FLOOR
    y0 = _wall_floor_row(box, W // 2)
    ys = np.arange(H)[:, None].repeat(W, 1)
    light = (np.clip((ys - y0) / (H - y0), 0, 1) * 90.0 - 45.0)[..., None]   # fundo escuro, frente clara
    img = img.astype(np.float32)
    img[fl] = (img + light)[fl]
    img = np.clip(img, 0, 255).round().astype(np.uint8)
    m = np.zeros((H, W), bool)
    m[y0 - 40:y0 + 36, 200:300] = True                               # móvel encostado na parede de fundo
    lab = box.label
    plate = img.copy()                                               # preenchimento = a própria luz do lugar
    q = rp.assess(img, plate, m, lab, groups, box)
    s = q.per_plane["piso"]
    assert s["preenchimento_global_de"] > rp.LIMITS["preenchimento_de_max"], s   # a média do anel engana
    assert s["preenchimento_de"] < s["preenchimento_global_de"], s              # referência local
    assert s["razao_material"] <= rp.LIMITS["relaxa_area_max"], s               # buraco pequeno x piso visível
    assert s["preenchimento_de"] <= rp.fill_de_limit(s["costura_de"], s["grao_log2"]), s   # costura boa relaxa
    assert not any("destoa" in t for t in q.reasons), q.reasons
    bad = img.copy()
    sel = m & fl
    bad[sel] = np.clip(bad[sel].astype(np.float32) + np.array([30.0, 0.0, -25.0]), 0, 255).astype(np.uint8)
    qb = rp.assess(img, bad, m, lab, groups, box)
    assert any("destoa no(a) piso" in t for t in qb.reasons), (qb.reasons, qb.per_plane)


def test_fill_limit_relaxes_only_with_good_seam_and_grain():
    lo, hi = rp.LIMITS["preenchimento_de_max"], rp.LIMITS["preenchimento_de_max_relaxado"]
    assert rp.fill_de_limit(1.5, 0.1) == hi
    assert rp.fill_de_limit(rp.LIMITS["costura_relaxa_de"], 0.1) == lo
    assert lo < rp.fill_de_limit(3.5, 0.1) < hi
    assert rp.fill_de_limit(1.5, 0.9) == lo                         # grão ruim não relaxa
    assert rp.fill_de_limit(1.5, 0.1, relax=False) == lo           # mancha forte no resto da placa


def test_extend_wall_panels_continues_panel_down_to_the_floor():
    H, W = 120, 160
    plate = np.full((H, W, 3), 230, np.uint8)            # parede clara
    plate[20:60, 40:120] = 60                            # painel escuro que ficou na foto
    plate[20:60:6, 40:120] = 90                          # com textura (linhas)
    panels = np.zeros((H, W), bool)
    panels[20:100, 40:120] = True                        # painel inteiro (a parte de baixo estava atrás da cama)
    mask = np.zeros((H, W), bool)
    mask[60:110, 30:130] = True                          # buraco da cama (o LaMa pôs parede clara)
    labels = np.zeros((H, W), np.int32)                  # parede 0 ...
    labels[100:] = rp.FLOOR                              # ... e piso a partir da linha 100
    before = plate.copy()
    n = rp.extend_wall_panels(plate, mask, labels, panels)
    assert n == 40 * 80
    assert plate[60:100, 40:120].mean() < 100            # continua escuro até o piso
    assert (plate[100:] == before[100:]).all()           # o piso não muda
    assert (plate[60:100, 30:40] == 230).all()           # nem a parede fora das colunas do painel
    assert (plate[~mask] == before[~mask]).all()


def test_wall_hung_piece_shadow_band_is_on_the_wall():
    """Quadro na parede: a faixa de sombra da moldura é procurada na parede (antes só no piso)."""
    room = _room()
    img, groups, box = _render(room)
    H, W = groups.shape
    art = np.zeros((H, W), bool)
    art[100:150, 200:280] = True
    core = rp.core_mask([_piece(art, "solto", "painting")], (H, W))
    cand = rp.effect_candidates([_piece(art, "solto", "painting")], groups, core)
    below = np.zeros((H, W), bool)
    below[155:160, 210:270] = True                     # parede logo abaixo da moldura
    assert (cand & below).sum() > 0.8 * below.sum()
    cand_chair = rp.effect_candidates([_piece(art, "solto", "chair")], groups, core)
    assert (cand & ~cand_chair).sum() > 0                # a peça pendurada tem faixa maior na parede
