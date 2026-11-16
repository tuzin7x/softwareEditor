"""Testes da coreografia de montagem (tempos, poses e ordem das peças)."""
import math

from mobilia.render import choreography as ch


def test_hidden_before_and_rest_after():
    p = ch.PartPlan("x", "drop", start=1.0, duration=1.0, params={"height": 0.5})
    assert not p.pose(0.5).visible and p.pose(0.5).fade == 0
    end = p.pose(10.0)
    assert end.visible and end.key() == ch.REST.key()


def test_drop_starts_high_fades_in_and_lands():
    p = ch.PartPlan("x", "drop", start=0.0, duration=1.0, params={"height": 0.5})
    first = p.pose(0.0)
    assert math.isclose(first.dz, 0.5, abs_tol=1e-6) and first.fade == 0.0
    mid = p.pose(0.3)
    assert 0 < mid.dz < 0.5 and 0 < mid.fade <= 1
    landed = p.pose(0.56)
    assert abs(landed.dz) < 0.02 and landed.fade == 1.0


def test_drop_never_goes_below_floor():
    p = ch.PartPlan("x", "drop", start=0.0, duration=1.2, params={"height": 0.55})
    assert all(p.pose(t / 100).dz >= -1e-9 for t in range(0, 300))


def test_grow_ends_at_scale_one():
    p = ch.PartPlan("leaves", "grow", start=0.0, duration=1.0)
    assert abs(p.pose(5).sx - 1) < 1e-9
    assert p.pose(0.01).sx < 0.3


def test_schedule_orders_and_overlaps():
    items = [{"id": "b", "role": "sofa", "order": 2}, {"id": "a", "role": "rug", "order": 1},
             {"id": "c", "role": "vase", "order": 3}]
    out, end = ch.schedule_items(items, hold_start=0.8)
    assert [i["id"] for i in out] == ["a", "b", "c"]
    assert out[0]["start"] == 0.8
    assert out[0]["start"] < out[1]["start"] < out[0]["start"] + out[0]["duration"]   # sobreposição
    assert end > out[-1]["start"]


def test_parts_assemble_base_first():
    item = {"start": 1.0, "duration": 1.0, "role": "plant_large", "motion": "grow", "size": [0.6, 0.6, 1.3]}
    plans = {p.name: p for p in ch.plan_parts(item, ["plant_leaves", "plant_pot", "plant_dirt", "plant_stem"])}
    assert plans["plant_pot"].start < plans["plant_dirt"].start < plans["plant_stem"].start <= plans["plant_leaves"].start
    assert plans["plant_leaves"].motion == "grow" and plans["plant_pot"].motion == "drop"


def test_books_enter_in_at_most_three_waves():
    item = {"start": 0.0, "duration": 0.7, "role": "books", "motion": "drop", "size": [0.5, 0.2, 0.24]}
    names = [f"set_book{i:02d}" for i in range(1, 11)]
    plans = ch.plan_parts(item, names)
    starts = [p.start for p in sorted(plans, key=lambda p: p.name)]
    assert starts == sorted(starts)            # da esquerda para a direita, em levas contíguas
    assert len(set(starts)) == 3
    gaps = sorted(set(starts))
    assert all(abs((b - a) - ch.ON_TOP_STEP) < 1e-9 for a, b in zip(gaps, gaps[1:]))


def _airborne_time(plans, dt=0.001, tol=0.01):
    """Soma do tempo (s) em que ALGUMA peça visível está mais de ``tol`` m acima do repouso."""
    t0 = min(p.start for p in plans)
    t1 = max(p.end for p in plans)
    n = int((t1 - t0) / dt) + 2
    air = 0
    for k in range(n):
        t = t0 + k * dt
        if any(pp.visible and pp.dz > tol for pp in (p.pose(t) for p in plans)):
            air += 1
    return air * dt


def test_on_top_parts_airborne_window_short():
    """Livros/vaso/fruteira sobre o tampo: nada fica "meio no ar" por muito tempo."""
    enc = {"start": 3.0, "duration": 0.7, "role": "books", "motion": "drop", "mount": "on_top",
           "size": [0.551, 0.163, 0.237]}
    names = [f"book_encyclopedia_set_01_book{i:02d}" for i in range(1, 21)]
    stack = dict(enc, size=[0.33, 0.26, 0.085])
    tags = {f"stack.book_{i}": (i, "drop") for i in range(3)}
    vase = {"start": 1.0, "duration": 0.6, "role": "vase", "motion": "drop", "mount": "on_top",
            "size": [0.2, 0.2, 0.4]}
    cases = [ch.plan_parts(enc, names), ch.plan_parts(stack, list(tags), tags), ch.plan_parts(vase, ["vase"])]
    for plans in cases:
        assert len({p.start for p in plans}) <= ch.ON_TOP_WAVES
        assert max(p.params["height"] for p in plans) <= 0.12 + 1e-9
        assert _airborne_time(plans) < 0.35, [p.name for p in plans][:3]
    # a pilha cai de baixo para cima: o livro de cima nunca está abaixo do de baixo
    st = {p.name: p for p in cases[1]}
    assert st["stack.book_0"].start < st["stack.book_1"].start < st["stack.book_2"].start
    for k in range(0, 1500):
        t = 3.0 + k / 1000
        z = [st[f"stack.book_{i}"].pose(t) for i in range(3)]
        for lo, hi in zip(z, z[1:]):
            if lo.visible and hi.visible:
                assert hi.dz >= lo.dz - 0.003, (t, lo.dz, hi.dz)


def test_procedural_part_tags_take_priority():
    item = {"start": 0.0, "duration": 1.0, "role": "curtain", "motion": "slide", "size": [1.6, 0.12, 2.5]}
    tags = {"curtain_rod": (0, "slide"), "curtain_panel_l": (1, "unfold"), "curtain_panel_r": (1, "unfold")}
    plans = {p.name: p for p in ch.plan_parts(item, list(tags), tags)}
    assert plans["curtain_rod"].motion == "slide"
    assert plans["curtain_panel_l"].motion == "unfold"
    assert plans["curtain_panel_l"].start > plans["curtain_rod"].start
    p = plans["curtain_panel_l"]
    assert p.pose(p.start + 0.05).sz < 0.5 and abs(p.pose(p.start + 5).sz - 1) < 1e-9


def test_settle_lands_softly():
    p = ch.PartPlan("duvet", "settle", start=0.0, duration=0.8)
    assert p.pose(0.0).dz > 0 and p.pose(0.0).fade == 0.0
    assert all(p.pose(t / 100).dz >= -1e-9 for t in range(0, 200))
    assert p.pose(10).key() == ch.REST.key()


# ------------------------------------------------------------------ PT3: menos quadros sem perda visível

# Sala de teste (Stafford, 1920 px): papéis, tamanhos e ampliação da projeção de cada item
TEST_ROOM = [
    ("rug_0", "rug", [2.0, 1.4, 0.012], "floor", "unroll", 682.7),
    ("sofa_0", "sofa", [1.807, 0.817, 0.709], "floor", None, 562.1),
    ("armchair_0", "armchair", [0.82, 0.986, 1.022], "floor", None, 409.4),
    ("side_table_0", "side_table", [0.55, 0.45, 0.55], "floor", None, 657.6),
    ("coffee_table_0", "coffee_table", [1.201, 0.6, 0.389], "floor", None, 631.2),
    ("floor_lamp_0", "floor_lamp", [0.42, 0.42, 1.62], "floor", None, 361.9),
    ("plant_medium_0", "plant_medium", [0.701, 0.657, 0.841], "floor", "grow", 444.9),
    ("wall_art_1", "wall_art", [0.593, 0.015, 0.84], "wall", "slide", 331.1),
    ("wall_art_0", "wall_art", [0.593, 0.015, 0.84], "wall", "slide", 366.8),
    ("pillows_0", "pillows", [1.021, 0.627, 0.45], "on_top", None, 495.0),
    ("vase_0", "vase", [0.204, 0.206, 0.401], "on_top", None, 457.6),
    ("books_0", "books", [0.33, 0.26, 0.085], "on_top", None, 526.0),
    ("bowl_0", "bowl", [0.313, 0.309, 0.093], "on_top", None, 570.8),
]


def _room_items(px_scale: float = 1.0):
    out = []
    for order, (iid, role, size, mount, motion, ppm) in enumerate(TEST_ROOM):
        it = {"id": iid, "role": role, "size": size, "mount": mount, "order": order, "px_per_m": ppm * px_scale}
        if motion:
            it["motion"] = motion
        out.append(it)
    return out


def _old_end(items, hold_start=0.8, gap_factor=0.55, min_gap=0.32):
    """Fim da animação pela regra de antes do PT3 (sem limite de peças simultâneas)."""
    t, end = hold_start, hold_start
    for it in sorted(items, key=lambda i: i["order"]):
        dur, _ = ch.ROLE_TIMING.get(it["role"], (0.9, "drop"))
        end = max(end, t + dur * 2.2)
        small = it["role"] in ch.SMALL_ROLES
        t += max(min_gap * (0.75 if small else 1.0), dur * gap_factor * (0.7 if small else 1.0))
    return end


def test_pose_snaps_to_rest_subpixel():
    """Quando o resto do assentamento projeta < 0,3 px a pose é EXATAMENTE a de repouso, e nunca
    volta a mexer; antes disso, o último trecho visível continua intacto."""
    ext, ppm = (0.82, 0.99, 1.02), 600.0
    p = ch.PartPlan("x", "drop", start=1.0, duration=1.0, params={"height": 0.45, "size": 1.02},
                    px_per_m=ppm, extent=ext)
    T = p.duration * ch.SETTLE_FACTOR
    assert 0.55 < p.settle_time < T - 0.3                       # a cauda invisível é cortada
    assert p.end == 1.0 + T                                      # fim nominal (linha do tempo) igual
    t_snap = p.start + p.settle_time
    for k in range(0, 2000):
        t = t_snap + k * (p.end + 0.5 - t_snap) / 2000
        assert p.pose(t).key() == ch.REST.key()
        if t - p.start <= T:
            assert ch.pose_residual_px(p.raw_pose(t - p.start), ext, ppm) < ch.SNAP_PX
    # logo antes do corte o resíduo ainda é visível e a pose é a do movimento
    tb = p.settle_time - ch.SNAP_DT
    assert ch.pose_residual_px(p.raw_pose(tb), ext, ppm) >= ch.SNAP_PX
    assert p.pose(p.start + tb).key() != ch.REST.key()
    # câmera muito ampliada: nada é cortado (comportamento de antes)
    q = ch.PartPlan("x", "drop", start=1.0, duration=1.0, params={"height": 0.45}, px_per_m=1e9, extent=ext)
    assert q.settle_time == T
    ts = [1.0 + k / 500 for k in range(0, int(T * 500) + 1)]
    assert all(q.pose(t) == q.raw_pose(t - q.start) for t in ts)
    assert q.pose(1.0 + T + 1e-6) is ch.REST
    # fade incompleto nunca é repouso
    assert ch.pose_residual_px(ch.Pose(fade=0.99)) == math.inf


def test_pose_snap_reuses_tail_frames():
    """Os quadros da cauda ficam idênticos ao repouso (o worker os reaproveita): na sala de teste
    a 24 fps o último quadro com movimento chega bem antes do fim nominal da animação."""
    fps = 24
    sched, end = ch.schedule_items(_room_items())
    plans = []
    for it in sched:
        names = [f"{it['id']}.p{i}" for i in range(3 if it["role"] != "books" else 3)]
        tags = {n: (i, "drop") for i, n in enumerate(names)} if it["role"] == "books" else None
        plans += ch.plan_parts(it, names, tags)
    rest = ch.REST.key(4)

    def moving(f):
        return any(pl.pose((f - 1) / fps).visible and pl.pose((f - 1) / fps).key(4) != rest for pl in plans)

    last = max(f for f in range(1, int(end * fps) + 2) if moving(f))
    assert last / fps < end - 0.5


def test_max_concurrent_movers():
    """No máximo MAX_MOVERS itens em movimento ao mesmo tempo, mesmo com muitos itens pequenos."""
    items = _room_items() + [
        {"id": f"vase_{i}", "role": "vase", "size": [0.2, 0.2, 0.4], "mount": "on_top", "order": 20 + i,
         "px_per_m": 800.0} for i in range(8)] + [
        {"id": f"chair_{i}", "role": "dining_chair", "size": [0.5, 0.5, 0.9], "mount": "floor",
         "order": 40 + i, "px_per_m": 700.0} for i in range(6)]
    fast = {"gap_factor": 0.3, "min_gap": 0.15}                  # entradas bem mais rápidas que o padrão
    sched, end = ch.schedule_items(items, **fast)
    windows = [(i["start"], i["start"] + ch.item_motion_window(i)) for i in sched]
    assert ch.max_concurrent(windows) <= ch.MAX_MOVERS
    free, end_free = ch.schedule_items(items, max_movers=None, **fast)
    wf = [(i["start"], i["start"] + ch.item_motion_window(i)) for i in free]
    assert ch.max_concurrent(wf) > ch.MAX_MOVERS                 # o caso realmente exige o limite
    # a espera só começa onde o limite seria passado (antes disso, tudo igual)
    first = next(k for k, (a, b) in enumerate(zip(sched, free)) if a["start"] != b["start"])
    assert ch.max_concurrent(wf[:first + 1]) > ch.MAX_MOVERS
    assert all(a["start"] >= b["start"] for a, b in zip(sched, free))
    # o fim cresce só o necessário: nunca mais que a soma das esperas
    waits = sum(a["start"] - b["start"] for a, b in zip(sched, free))
    assert end_free <= end <= end_free + waits + 1e-9


def test_timeline_length_unchanged():
    """O corte da cauda e o limite de peças não mudam a linha do tempo da sala de teste."""
    for scale in (0.5, 1.0, 1.34):                                # 960, 1920 e 2560 px
        items = _room_items(scale)
        sched, end = ch.schedule_items(items)
        assert math.isclose(end, _old_end(items), abs_tol=1e-9)
        windows = [(i["start"], i["start"] + ch.item_motion_window(i)) for i in sched]
        assert ch.max_concurrent(windows) <= ch.MAX_MOVERS
        free, _ = ch.schedule_items(items, max_movers=None)
        assert [i["start"] for i in sched] == [i["start"] for i in free]
    for it in sched:
        for pl in ch.plan_parts(it, [it["id"]]):
            assert pl.end == pl.start + pl.duration * 2.2


# ---------------------------------------------------------------- PT6: marcos do render em camadas
def _layer_parts(sched, fps):
    """Peças sintéticas dos itens agendados (como o worker monta para ``layers.plan_layers``)."""
    from mobilia.render import layers as lay

    parts = []
    for it in sched:
        n = 3 if it["role"] in ("books", "plant_medium") else 2
        names = [f"{it['id']}.p{i}" for i in range(n)]
        tags = {nm: (i, "drop") for i, nm in enumerate(names)} if it["role"] == "books" else None
        ur = None
        plans = ch.plan_parts(it, names, tags)
        if it.get("motion") == "unroll":
            f0, f1 = max(1, round(it["start"] * fps)), round((it["start"] + it["duration"]) * fps)
            ur = (f0, f1)
            plans = [ch.PartPlan(p.name, "rise", it["start"], 0.25, {"depth": 0.0}) for p in plans]
        parts += [lay.LayerPart(p.name, it["id"], p, ur) for p in plans]
    return parts


def test_milestones_cover_all_items():
    """Marcos da coreografia (PT6): 3–5 na sala de teste, em ordem, cada um contém o anterior, todo
    item aparece em algum e o último (o quadro final) tem todos. No plano do worker, toda peça
    entra num marco, cada quadro dinâmico tem uma base já assentada e só renderiza peças fora
    dela, e o fim do vídeo é o marco final."""
    from mobilia.render import layers as lay

    sched, end = ch.schedule_items(_room_items())
    ms = ch.milestones(sched)
    ids = {it["id"] for it in sched}
    assert 3 <= len(ms) <= len(ch.MILESTONE_CLASSES)
    assert [m["t"] for m in ms] == sorted(m["t"] for m in ms)
    for a, b in zip(ms, ms[1:]):
        assert set(a["itens"]) < set(b["itens"]) and b["t"] - a["t"] >= ch.MILESTONE_MIN_GAP
    assert set(ms[-1]["itens"]) == ids and set().union(*(m["itens"] for m in ms)) == ids
    assert {c for m in ms for c in m["classes"]} == {ch.item_class(it) for it in sched}
    # o plano do render em camadas (Python puro, o worker só executa)
    fps = 24
    n = int(math.ceil(ch.timeline_length(end) * fps))
    parts = _layer_parts(sched, fps)
    plan = lay.plan_layers(parts, fps, n, [m["itens"] for m in ms])
    keys = {p.key for p in parts}
    assert set().union(*(m.parts for m in plan.milestones)) == keys
    final = [m for m in plan.milestones if m.final]
    assert len(final) == 1 and final[0].parts == keys and final[0] is plan.milestones[-1]
    assert 3 <= len(plan.milestones) <= len(ms) + 1
    by_id = {m.id: m for m in plan.milestones}
    for fr in plan.frames:
        if fr.kind == "foto":
            continue
        base = by_id.get(fr.base)
        if base is not None:
            assert base.frame <= fr.frame
            assert all(plan.settle[k] <= fr.frame for k in base.parts)       # base sempre parada
            assert not (fr.layer & base.parts)
        if fr.kind == "estatico":
            assert base is not None and not fr.layer
    tail = [fr for fr in plan.frames if fr.frame > final[0].frame + lay.CROSSFADE_FRAMES]
    assert tail and all(fr.kind == "estatico" and fr.base == final[0].id and not fr.trans for fr in tail)


def test_layer_plan_transitions_fade_out_old_composition():
    """Na troca de marco, ``CROSSFADE_FRAMES`` quadros levam a composição antiga (marco anterior +
    a camada dele, com as mesmas peças visíveis) com peso decrescente; nenhum quadro dinâmico deixa
    uma peça parada fora do marco por uma pausa inteira (vira marco implícito)."""
    from mobilia.render import layers as lay

    sched, end = ch.schedule_items(_room_items())
    fps = 24
    n = int(math.ceil(ch.timeline_length(end) * fps))
    parts = _layer_parts(sched, fps)
    plan = lay.plan_layers(parts, fps, n, [m["itens"] for m in ch.milestones(sched)])
    frames = plan.frames
    switches = [i for i in range(1, len(frames)) if frames[i].kind != "foto" and frames[i - 1].kind != "foto"
                and frames[i].base != frames[i - 1].base]
    assert switches
    for i in switches:
        ws = []
        for j in range(lay.CROSSFADE_FRAMES):
            fr = frames[i + j]
            assert fr.trans is not None and fr.trans[0] == frames[i - 1].base
            old_base = plan.milestone(fr.trans[0])
            ob = old_base.parts if old_base else frozenset()
            new_base = plan.milestone(fr.base)
            # as duas composições mostram as mesmas peças
            assert fr.trans[1] | ob == fr.layer | (new_base.parts if new_base else frozenset())
            ws.append(fr.trans[3])
        assert ws == sorted(ws, reverse=True) and 0 < ws[-1] < ws[0] < 1
        assert frames[i + lay.CROSSFADE_FRAMES].trans is None or \
            frames[i + lay.CROSSFADE_FRAMES].base != frames[i].base
    # pausas sem movimento nunca ficam com peça parada na camada dinâmica
    for fr in frames:
        if fr.kind == "dinamico":
            assert any(fr.frame < plan.settle[k] for k in fr.layer)
