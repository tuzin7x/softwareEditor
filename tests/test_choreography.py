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
