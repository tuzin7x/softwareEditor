"""Adaptador entre os parâmetros do layout e os geradores procedurais."""
from mobilia.render.scene_spec import normalize_procedural_params


def test_rug_pattern_and_colors_are_mapped():
    p = normalize_procedural_params("rug", {"color": (0.9, 0.88, 0.85), "pattern_color": (0.2, 0.2, 0.2),
                                            "pattern": "geometrico", "thickness": 0.012, "size": [2.4, 1.7, 0.012]})
    assert p["pattern"] == "berber"
    assert p["color"] == "#e6e0d9" and p["accent"] == "#333333"
    assert p["size"] == [2.4, 1.7, 0.012] and "thickness" not in p


def test_unknown_rug_pattern_falls_back_safely():
    assert normalize_procedural_params("rug", {"pattern": "persa"})["pattern"] == "persian"
    assert normalize_procedural_params("rug", {"pattern": "xyz"})["pattern"] == "solid_border"


def test_lamps_never_turn_on_and_metal_is_named():
    p = normalize_procedural_params("floor_lamp", {"shade_color": "#f0ece4", "metal_color": (0.75, 0.6, 0.3),
                                                   "light_on": True})
    assert "light_on" not in p and p["metal"] == "brass"
    assert normalize_procedural_params("table_lamp", {"metal_color": (0.05, 0.05, 0.05)})["metal"] == "matte_black"


def test_bed_and_desk_params():
    b = normalize_procedural_params("bed", {"frame_color": (0.7, 0.66, 0.6), "linen_color": "#f4f2ee",
                                            "accent_color": (0.4, 0.5, 0.45)})
    assert b["fabric_color"] and b["duvet_color"] == "#f4f2ee" and len(b["pillow_colors"]) == 2
    d = normalize_procedural_params("desk", {"top_color": (0.25, 0.15, 0.1), "leg_color": (0.1, 0.1, 0.1)})
    assert d == {"wood": "walnut", "legs": "black"}
