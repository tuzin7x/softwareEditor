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


def test_books_enter_one_by_one():
    item = {"start": 0.0, "duration": 0.7, "role": "books", "motion": "drop", "size": [0.5, 0.2, 0.24]}
    names = [f"set_book{i:02d}" for i in range(1, 11)]
    plans = ch.plan_parts(item, names)
    starts = [p.start for p in sorted(plans, key=lambda p: p.name)]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


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
