"""PT-V3-2: instâncias de móveis do modo B e detecção automática do modo (rápidos, sem modelos)."""
import json
from pathlib import Path

import numpy as np
import pytest

from mobilia.analysis.room import RoomModel, Wall
from mobilia.analysis.segmentation import SegmentationResult, labels_to_groups
from mobilia.removal import instances as ri

# Peças do protótipo CPU na foto de Nice (OneFormer panoptic ∪ semântica + SAM 2.1): rótulo e caixa
# (x0, y0, x1, y1). Eram 18 antes da fusão: a mesa de jantar (357..622) saiu só na semântica, partida pela
# cadeira em 2 fragmentos, que o protótipo fundiu numa peça (as 17 caixas abaixo + 1 fragmento fundido).
NICE_PROTOTYPE = [
    ("pillow", (36, 450, 92, 524)), ("table", (91, 502, 170, 528)), ("chair", (393, 473, 486, 508)),
    ("rug", (48, 633, 335, 754)), ("rug", (734, 786, 1079, 845)), ("chair", (377, 509, 526, 804)),
    ("chair", (599, 508, 745, 809)), ("table", (975, 526, 1219, 785)), ("flower", (483, 390, 591, 468)),
    ("sofa", (1182, 743, 1279, 845)), ("vase", (531, 467, 575, 541)), ("armchair", (1182, 551, 1279, 814)),
    ("armchair", (777, 513, 987, 748)), ("lamp", (86, 412, 144, 506)), ("bed", (30, 291, 259, 682)),
    ("chair", (586, 470, 676, 524)), ("table", (357, 519, 622, 816)),
]
DINING_TABLE = (357, 519, 622, 816)


def _box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def _contains(outer, inner):
    return outer[0] <= inner[0] + 3 and outer[1] <= inner[1] + 3 and outer[2] >= inner[2] - 3 and outer[3] >= inner[3] - 3


def _from_fixture(f, **kw):
    pan = ri.PanopticResult(f.panoptic_ids, f.panoptic_segments)
    kw.setdefault("sam", None)
    return ri.detect_instances(f.photo, f.seg, pan, geometry=f.geometry, room=f.room, **kw)


# ----------------------------------------------------------------------------------------------- fotos reais
def test_nice_matches_prototype_pieces(room_fixture):
    s = _from_fixture(room_fixture("nice_terrazzo"))
    boxes = [(i.label, i.bbox) for i in s.instances]
    found = 0
    for lab, b in NICE_PROTOTYPE:
        # a peça do protótipo está entre as detectadas (ou dentro de uma peça fundida com o mesmo rótulo)
        if any(l == lab and (_box_iou(b, bb) >= 0.5 or _contains(bb, b)) for l, bb in boxes):
            found += 1
    # mesa de jantar presente (vem da semântica, fundida pela regra do fecho convexo)
    tables = [i for i in s.instances if i.label == "table" and _box_iou(i.bbox, DINING_TABLE) >= 0.8]
    assert len(tables) == 1 and tables[0].source == "semantic" and tables[0].merged_from
    found += len(tables[0].merged_from)       # o fragmento fundido conta como a 18ª peça do protótipo
    assert found >= 16, (found, boxes)
    # os 2 tapetes separados
    rugs = [i for i in s.instances if i.label == "rug"]
    assert len(rugs) == 2 and all(r.source == "panoptic-split" for r in rugs)
    assert not (rugs[0].mask & rugs[1].mask).any()
    # portas, janelas, cortinas, arandela, parede, piso e teto nunca viram peça
    assert not {i.label for i in s.instances} & {"door", "window", "curtain", "sconce", "wall", "floor", "ceiling"}
    assert all(i.kind == ri.KIND_LOOSE for i in s.instances)
    assert s.mode.mode == "B" and not s.mode.uncertain


def test_typical_listing_photo(room_fixture):
    f = room_fixture("sala_sofas_brancos")
    s = _from_fixture(f)
    labels = [i.label for i in s.instances]
    assert "sofa" in labels and "coffee table" in labels and "rug" in labels
    assert s.mode.mode == "B"
    # a luminária pendente (encosta no topo da foto) é luminária de teto: fica na foto
    assert not any(i.label == "lamp" and i.bbox[1] <= 2 for i in s.instances)
    sofa = max((i for i in s.instances if i.label == "sofa"), key=lambda i: i.area)
    sem_sofa = f.seg.labels == [k for k, v in f.seg.id2label.items() if v == "sofa"][0]
    inter = (sofa.mask & sem_sofa).sum() / max(1, sem_sofa.sum())
    assert inter > 0.7


def test_report_is_json_and_consistent(room_fixture):
    s = _from_fixture(room_fixture("nice_terrazzo"))
    r = json.loads(json.dumps(s.report(), ensure_ascii=False))
    assert r["n_pecas"] == len(s.instances) == r["n_soltos"] + r["n_planejados"]
    assert r["modo"]["modo"] == "B"
    assert 15 < r["area_remocao_pct"] < 35
    assert {"id", "rotulo", "tipo", "bbox", "area_pct"} <= set(r["pecas"][0])


def test_empty_stafford_is_mode_a():
    seg = SegmentationResult.load(Path(__file__).parent / "fixtures" / "rooms" / "stafford" / "seg.npz")
    pan = ri.PanopticResult(np.zeros(seg.labels.shape, np.uint8), np.zeros((0, 4)))
    d = ri.detect_mode(seg, pan)
    assert d.mode == "A" and not d.forced


# ------------------------------------------------------------------------------------- sintéticos (regras)
NAMES = {0: "wall", 3: "floor", 5: "ceiling", 8: "window ", 10: "cabinet", 14: "door", 15: "table",
         18: "curtain", 19: "chair", 23: "sofa", 28: "rug", 36: "lamp", 47: "sink", 70: "countertop",
         85: "chandelier"}


def _scene(h=200, w=300):
    lab = np.zeros((h, w), np.uint8)          # parede
    lab[:30] = 5                              # teto
    lab[140:] = 3                             # piso
    return lab


def _seg(lab):
    return SegmentationResult(lab, labels_to_groups(lab, NAMES), dict(NAMES), "teste")


def _pan(shape, items):
    """items: lista de (label_id, máscara)."""
    ids = np.zeros(shape, np.uint8)
    rows = []
    for k, (lid, m) in enumerate(items, start=1):
        ids[m] = k
        rows.append((k, lid, 0.9, int(m.sum())))
    return ri.PanopticResult(ids, np.array(rows, np.float64).reshape(-1, 4))


def _rect(shape, x0, y0, x1, y1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_split_far_parts_and_keep_close_parts():
    shape = (200, 300)
    two_rugs = _rect(shape, 10, 150, 90, 190) | _rect(shape, 200, 150, 290, 190)
    assert len(ri._split_far_parts(two_rugs)) == 2
    # cadeira partida pela perna da mesa (fenda de 3 px): continua uma peça só
    chair = _rect(shape, 100, 60, 140, 140) | _rect(shape, 143, 60, 180, 140)
    assert len(ri._split_far_parts(chair)) == 1


def test_semantic_component_missed_by_panoptic_is_added():
    lab = _scene()
    lab[100:150, 40:120] = 15                 # mesa só na semântica
    lab[90:150, 200:260] = 19                 # cadeira no panoptic e na semântica
    lab[40:100, 140:180] = 14                 # porta: fica
    seg = _seg(lab)
    pan = _pan(lab.shape, [(19, lab == 19), (14, lab == 14)])
    c = ri.gather_candidates(seg, pan)
    got = sorted((x.label, x.source) for x in c)
    assert got == [("chair", "panoptic"), ("table", "semantic")]


def test_sam_fallback_when_iou_low_and_accept_when_good():
    shape = (120, 160)
    m0 = _rect(shape, 40, 30, 100, 90)
    cand = ri.Candidate("sofa", 23, "panoptic", 0.9, m0)
    good = np.zeros((3,) + shape, np.float32)
    good[0] = _rect(shape, 41, 31, 101, 91) * 0.99
    good[1] = _rect(shape, 0, 0, 160, 120) * 0.99    # pega tudo
    bad = np.zeros((3,) + shape, np.float32)
    bad[:, 0:20, 0:20] = 0.99                        # nada a ver com a peça

    def fake(results):
        return lambda rgb, boxes: [results] * len(boxes)

    rgb = np.zeros(shape + (3,), np.uint8)
    (hard, alpha, iou, used), = ri.refine(rgb, [cand], fake(good))
    assert used and iou > 0.9 and abs(int(hard.sum()) - int(m0.sum())) < 200
    (hard, alpha, iou, used), = ri.refine(rgb, [cand], fake(bad))
    assert not used and iou < ri.SAM_MIN_IOU
    assert np.array_equal(hard, m0)           # ficou a máscara do OneFormer
    assert alpha[m0].min() > 0.5 and alpha.dtype == np.float32


def test_cleanup_keeps_largest_and_confirmed_legs_and_cuts_low_alpha():
    shape = (120, 160)
    body = _rect(shape, 40, 30, 100, 80)
    leg = _rect(shape, 45, 84, 50, 110)           # separada do corpo, mas confirmada pelo OneFormer
    stray = _rect(shape, 130, 10, 150, 30)        # bolha solta que o OneFormer não confirma
    prob = np.where(body | leg | stray, 0.95, 0.02).astype(np.float32)
    prob[np.ix_(range(25, 30), range(40, 100))] = 0.3   # halo suave junto à borda (fica)
    m0 = body | leg
    hard, alpha = ri._cleanup(prob, m0)
    assert hard[leg].all() and not hard[stray].any() and hard[body].all()
    assert alpha[stray].max() == 0 and (alpha[(alpha > 0)] >= ri.ALPHA_CUT).all()
    assert alpha[27, 60] == pytest.approx(0.3)
    assert alpha[0, 0] == 0


def test_merge_fragments_convex_hull_rule():
    shape = (200, 300)

    def inst(i, lab, src, m):
        return ri.FurnitureInstance(i, lab, 15, ri.KIND_LOOSE, src, 0.0, m, m.astype(np.float32))

    left = _rect(shape, 50, 100, 120, 160)
    right = _rect(shape, 140, 100, 200, 160)
    chair = _rect(shape, 118, 90, 142, 170)       # cadeira na frente separa as duas metades da mesa
    out = ri.merge_fragments([inst(0, "table", "semantic", left), inst(1, "table", "semantic", right),
                              inst(2, "chair", "panoptic", chair)])
    tables = [i for i in out if i.label == "table"]
    assert len(tables) == 1 and tables[0].merged_from == [1] and tables[0].mask[130, 180]
    # duas mesas distantes com piso entre elas: não funde
    far = _rect(shape, 230, 100, 290, 160)
    out = ri.merge_fragments([inst(0, "table", "semantic", left), inst(1, "table", "semantic", far)])
    assert len(out) == 2


def _inst(i, lab, m):
    return ri.FurnitureInstance(i, lab, 0, ri.KIND_LOOSE, "panoptic", 0.9, m, m.astype(np.float32))


def test_builtin_rules_image_space():
    lab = _scene()
    seg = _seg(lab)
    shape = lab.shape
    upper = _inst(0, "cabinet", _rect(shape, 20, 30, 120, 70))       # aéreo colado no teto
    sink = _inst(1, "sink", _rect(shape, 150, 100, 200, 120))
    base = _inst(2, "cabinet", _rect(shape, 140, 120, 210, 160))     # gabinete sob a pia
    base2 = _inst(3, "cabinet", _rect(shape, 210, 120, 260, 160))    # módulo vizinho do gabinete
    loose = _inst(4, "cabinet", _rect(shape, 30, 110, 80, 160))      # cômoda solta, longe de tudo
    sofa = _inst(5, "sofa", _rect(shape, 90, 100, 130, 160))
    insts = [upper, sink, base, base2, loose, sofa]
    ri.classify(insts, seg)
    kinds = {i.id: (i.kind, i.kind_reason) for i in insts}
    assert kinds[0] == (ri.KIND_BUILTIN, "encosta no teto")
    assert kinds[1][0] == ri.KIND_BUILTIN
    assert kinds[2] == (ri.KIND_BUILTIN, "junto de sink")
    assert kinds[3][0] == ri.KIND_BUILTIN
    assert kinds[4][0] == ri.KIND_LOOSE and kinds[5][0] == ri.KIND_LOOSE
    s = ri.InstanceSet(insts, shape)
    assert {i.id for i in s.removal(keep_builtins=True)} == {4, 5}   # recuo do portão de qualidade
    assert s.removal_mask(True).sum() < s.removal_mask().sum()


def _room():
    # parede em y = 3 (normal -y), de x=-2 a x=2; câmera na origem a 1,5 m olhando +y
    M = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]], float)
    return RoomModel(intrinsics={"width": 40, "height": 40, "fx": 20, "fy": 20, "cx": 20, "cy": 20},
                     cam_to_room=M.tolist(), camera_height=1.5, ceiling_height=2.6, ceiling_observed=False,
                     walls=[Wall(0, [-2, 3], [2, 3], [0, -1], 2.6, 1000, 0.01)], openings=[],
                     grid_origin=[-3, 0], grid_cell=0.05, grid_shape=[10, 10])


def _geometry_for(mask, x, z, y=2.95):
    """Pontos de câmera (OpenCV: x direita, y para baixo, z frente) para uma peça em (x, z_sala)."""
    h, w = mask.shape
    pts = np.full((h, w, 3), np.nan, np.float32)
    ys, xs = np.nonzero(mask)
    tx = np.interp(xs, [xs.min(), xs.max()], x)
    tz = np.interp(ys, [ys.min(), ys.max()], z[::-1])
    pts[ys, xs] = np.stack([tx, 1.5 - tz, np.full_like(tx, y)], -1)
    return {"points": pts, "stride": 1, "mask": np.isfinite(pts).all(-1)}


def test_builtin_rules_3d():
    room = _room()
    lab = np.zeros((40, 40), np.uint8)
    seg = _seg(lab)
    m = _rect((40, 40), 10, 10, 30, 20)
    hanging = _inst(0, "shelf", m)
    assert ri.builtin_reason(hanging, seg, [hanging], _geometry_for(m, (-0.5, 0.5), (1.4, 2.0)), room) \
        == "suspenso na parede 0"
    tall = _inst(0, "wardrobe", m)
    assert ri.builtin_reason(tall, seg, [tall], _geometry_for(m, (-0.5, 0.5), (0.0, 2.5), y=2.0), room) \
        == "vai até o teto"
    full = _inst(0, "cabinet", m)
    assert ri.builtin_reason(full, seg, [full], _geometry_for(m, (-1.95, 1.95), (0.0, 0.9)), room) \
        == "ocupa a parede 0 inteira"
    chest = _inst(0, "chest of drawers", m)
    assert ri.builtin_reason(chest, seg, [chest], _geometry_for(m, (-0.4, 0.4), (0.0, 0.9)), room) == ""


def test_ceiling_lamp_stays_floor_lamp_leaves():
    lab = _scene()
    lab[0:40, 140:160] = 36            # pendente no topo
    lab[60:150, 20:35] = 36            # luminária de chão
    seg = _seg(lab)
    pan = _pan(lab.shape, [(36, _rect(lab.shape, 140, 0, 160, 40)), (36, _rect(lab.shape, 20, 60, 35, 150))])
    c = ri.gather_candidates(seg, pan)
    assert len(c) == 1 and c[0].mask[100, 25]


@pytest.mark.parametrize("area,sig,major,expected", [
    (0.0, 0, 0.0, "A"), (0.01, 1, 0.009, "A"), (0.05, 1, 0.05, "B"), (0.025, 1, 0.02, "B"),
    (0.03, 3, 0.006, "B"), (0.015, 2, 0.02, "A"),
])
def test_decide_mode_thresholds(area, sig, major, expected):
    f = {"area_removivel": area, "pecas_significativas": sig, "maior_peca_grande": major,
         "rotulo_maior_peca": "sofa", "n_pecas": sig}
    d = ri.decide_mode(f)
    assert d.mode == expected and d.reason
    assert ri.decide_mode(f, override="A" if expected == "B" else "B").forced


def test_empty_room_with_door_window_curtain_is_mode_a():
    lab = _scene()
    lab[40:140, 20:80] = 14
    lab[40:110, 150:250] = 8
    lab[40:140, 250:290] = 18
    lab[0:20, 140:170] = 85
    seg = _seg(lab)
    pan = _pan(lab.shape, [(14, lab == 14), (8, lab == 8), (18, lab == 18), (85, lab == 85)])
    s = ri.detect_instances(np.zeros(lab.shape + (3,), np.uint8), seg, pan, sam=None)
    assert s.instances == [] and s.mode.mode == "A"
    assert s.removal_mask().sum() == 0


def test_auto_sam_without_weights_falls_back(monkeypatch, room_fixture):
    from mobilia.analysis import weights

    def missing(offline=False):
        raise weights.WeightsError("SAM 2.1 não instalado")

    monkeypatch.setattr(weights, "sam2_dir", missing)
    s = _from_fixture(room_fixture("nice_terrazzo"), sam="auto")
    assert any("SAM" in w for w in s.warnings)
    assert len(s.instances) >= 16 and not any(i.sam_used for i in s.instances)


def test_builtins_alone_need_large_area_for_mode_b():
    # corredor vazio com um armário embutido (3,5% da foto): continua modo A
    hall = {"area_removivel": 0.035, "area_soltos": 0.0, "pecas_significativas": 1, "maior_peca_grande": 0.0,
            "rotulo_maior_peca": "", "n_pecas": 1}
    assert ri.decide_mode(hall).mode == "A"
    # cozinha planejada completa sem nenhum móvel solto: modo B (cenário cru)
    kitchen = dict(hall, area_removivel=0.2, pecas_significativas=2)
    assert ri.decide_mode(kitchen).mode == "B"


def test_mode_features_only_loose_pieces_count_as_major():
    shape = (100, 100)
    f = ri.mode_features([(_rect(shape, 0, 0, 30, 30), "cabinet"), (_rect(shape, 50, 50, 60, 60), "chair")], shape)
    assert f["rotulo_maior_peca"] == "chair" and f["area_soltos"] == pytest.approx(0.01)
    assert f["area_removivel"] == pytest.approx(0.1)
