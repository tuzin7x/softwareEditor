"""Adaptador entre os parâmetros do layout e os geradores procedurais."""
from pathlib import Path

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


def test_wall_extension_never_passes_in_front_of_another_wall():
    """Caso real da sala de teste: a parede 2 chega oblíqua na ponta da parede 1."""
    from mobilia.render.scene_spec import wall_extensions

    walls = [
        {"id": 0, "p0": [-3.36, 5.02], "p1": [-3.36, 1.28], "normal": [1.0, 0.0]},
        {"id": 1, "p0": [-1.70, 5.10], "p1": [-3.24, 5.10], "normal": [0.0, -1.0]},
        {"id": 2, "p0": [-0.67, 5.87], "p1": [-1.73, 5.16], "normal": [0.56, -0.83]},
    ]
    ext = wall_extensions(walls)
    # parede 2 -> ponta p1: só até cruzar o plano da parede 1 (~0,1 m), não os 0,35 m de antes
    assert ext[2][1] < 0.15
    # quinas de 90° (parede 0 x parede 1) continuam fechadas com a extensão cheia
    assert ext[0][0] == 0.35 and ext[1][1] == 0.35


# ------------------------------------------------------------------ PT3: amostras por camada

FX = Path(__file__).parent / "fixtures"


def _rtx_8gb():
    from mobilia.hardware import GpuInfo, HardwareReport, choose_profile

    rep = HardwareReport("Windows", "cpu", 16, 16.0, [GpuInfo("NVIDIA GeForce RTX 5070 Laptop GPU", 8151,
                                                             compute_capability="12.0")], torch_cuda=True)
    return choose_profile(rep)


def test_profile_samples_per_layer():
    from mobilia.render.layers import LAYERS

    prof = _rtx_8gb()
    assert prof.name == "alta" and prof.render_samples == 128
    assert prof.layer_samples == {"final": 128, "marco": 64, "dinamico": 32}
    assert set(prof.layer_samples) == set(LAYERS)
    from mobilia.hardware import HardwareReport, choose_profile

    cpu = choose_profile(HardwareReport("Linux", "cpu", 4, 8.0))
    assert cpu.layer_samples["final"] == cpu.render_samples
    assert cpu.layer_samples["dinamico"] <= cpu.layer_samples["marco"] <= cpu.layer_samples["final"]


def test_final_quality_is_24fps_with_layers():
    from mobilia.jobs import JobRequest
    from mobilia.pipeline import _quality

    prof = _rtx_8gb()
    q = _quality(JobRequest("x.jpg", "sala", "y.mp4", quality="final"), prof)
    assert q["fps"] == 24 and q["samples"] == 128
    assert q["samples_layers"] == prof.layer_samples
    p = _quality(JobRequest("x.jpg", "sala", "y.mp4", quality="previa"), prof)
    assert p["fps"] == 24 and p["samples_layers"]["final"] == p["samples"]
    assert p["samples_layers"]["dinamico"] < p["samples"]


def test_scene_json_has_samples_per_layer(tmp_path, monkeypatch):
    """scene.json leva amostras por camada, mínimo adaptativo explícito e a ampliação de cada item."""
    import json

    import numpy as np
    from PIL import Image

    from mobilia.analysis.room import RoomModel
    from mobilia.analysis.segmentation import SegmentationResult
    from mobilia.layout.model import Layout, PlacedItem
    from mobilia.render import scene_spec

    room = RoomModel.from_json((FX / "rooms/stafford/room.json").read_text(encoding="utf-8"))
    seg = SegmentationResult.load(FX / "rooms/stafford/seg.npz")
    W, H = 320, 240
    plate = np.asarray(Image.open(FX / "photos/sala_vazia_stafford.jpg").convert("RGB").resize((W, H)))
    Image.fromarray(plate).save(tmp_path / "plate.png")
    rs = lambda a: np.asarray(Image.fromarray(a).resize((W, H), Image.NEAREST))  # noqa: E731
    seg = SegmentationResult(rs(seg.labels), rs(seg.groups), seg.id2label, seg.model_id)
    items = [PlacedItem("rug_0", "rug", "proc:rug", [-2.0, 3.27, 0.0], 90.0, size=[2.0, 1.4, 0.012], order=0),
             PlacedItem("lamp_0", "floor_lamp", "proc:floor_lamp", [-3.1, 4.17, 0.0], 90.0,
                        size=[0.42, 0.42, 1.62], order=1)]
    layout = Layout("sala", "moderno", {}, items)
    prof = _rtx_8gb()
    spec = scene_spec.build_scene_spec(room, layout, plate, tmp_path / "plate.png", seg, tmp_path, tmp_path,
                                       samples=prof.render_samples, samples_layers=prof.layer_samples)
    on_disk = json.loads((tmp_path / "scene.json").read_text(encoding="utf-8"))
    for s in (spec, on_disk):
        r = s["render"]
        assert r["fps"] == 24
        assert r["samples"] == 128
        assert r["samples_layers"] == {"final": 128, "marco": 64, "dinamico": 32}
        assert r["adaptive_min_samples"] == {"final": 32, "marco": 32, "dinamico": 8}
        assert abs(r["adaptive_threshold"] - 0.02) < 1e-12
        assert r["sample_ramp_frames"] >= 1
    # uma qualidade mais baixa (prévia/teste) nunca faz uma camada passar do quadro final
    small = scene_spec.build_scene_spec(room, layout, plate, tmp_path / "plate.png", seg, tmp_path, tmp_path,
                                        samples=8, samples_layers=prof.layer_samples)
    assert small["render"]["samples_layers"] == {"final": 8, "marco": 8, "dinamico": 8}
    # px_per_m: 1 cm de deslocamento em qualquer direção projeta no máximo px_per_m/100 pixels
    intr = room.intr().scaled(W, H)
    Minv = room.cam_from_room()
    for it, pi in zip(spec["items"], items):
        assert it["px_per_m"] > 0
        x, y, z = pi.position
        base = np.array([[x, y, z + pi.size[2] / 2]])
        uv0 = intr.project(base @ Minv[:3, :3].T + Minv[:3, 3])
        for d in np.eye(3) * 0.01:
            uv = intr.project((base + d) @ Minv[:3, :3].T + Minv[:3, 3])
            assert np.linalg.norm(uv - uv0) <= it["px_per_m"] * 0.01


def test_frame_samples_ramp_before_static_frames():
    from mobilia.render.layers import adaptive_min, frame_samples

    spl = {"final": 128, "marco": 64, "dinamico": 32}
    seq = ["dinamico"] * 6 + ["marco"] + ["dinamico"] * 2 + ["final"]
    s = frame_samples(seq, spl, ramp=3)
    assert s[:3] == [32, 32, 32]                       # longe do quadro parado: o mínimo da camada
    assert s[3] < s[4] < s[5] < s[6] == 64              # sobe até o marco
    assert s[7] < s[8] < s[9] == 128                    # rampa curta (só 2 dinâmicos) até o final
    assert frame_samples(seq, spl, ramp=0) == [spl[c] for c in seq]
    assert adaptive_min("dinamico", 32) == 8 and adaptive_min("final", 128) == 32 and adaptive_min("final", 16) == 16


def test_canvas_art_and_textured_rug_params():
    """PT9: o quadro gerado leva paleta/moldura/semente ao Blender; o tapete leva os caminhos da
    textura CC0 (e cai no desenho procedural quando não há textura)."""
    a = normalize_procedural_params("canvas_art", {
        "size": [0.45, 0.035, 0.6], "colors": [(0.71, 0.38, 0.24), "#3d3f42"], "ground": (0.93, 0.91, 0.87),
        "art_style": "formas", "frame": "moldura", "frame_color": (0.12, 0.12, 0.13), "mat_color": "#f3f0e9",
        "seed": 18})
    assert a["frame"] == "moldura" and a["art_style"] == "formas" and a["seed"] == 18
    assert a["colors"] == ["#b5613d", "#3d3f42"] and a["ground"].startswith("#")
    assert normalize_procedural_params("canvas_art", {"frame": "vidro"})["frame"] == "tela"
    tex = {"diff": "/x/d.jpg", "nor": "/x/n.jpg", "rough": "/x/r.jpg"}
    r = normalize_procedural_params("rug", {"pattern": "geometrico", "texture": tex, "tile_m": 0.27,
                                            "lum_mean": 0.18, "design": "filete", "texture_id": "poly_wool_herringbone"})
    assert r["texture"] == tex and r["design"] == "filete" and r["tile_m"] == 0.27
    assert "texture_id" not in r
    r2 = normalize_procedural_params("rug", {"pattern": "geometrico", "texture": None, "design": "filete"})
    assert "texture" not in r2 and r2["pattern"] == "berber"


def test_rug_texture_paths_offline_falls_back(tmp_path, monkeypatch):
    from mobilia.render import scene_spec as ss

    def boom(*a, **k):
        raise RuntimeError("sem rede")

    monkeypatch.setattr(ss, "ensure_texture", boom)
    assert ss.rug_texture_paths("poly_wool_herringbone", tmp_path) is None
    monkeypatch.setattr(ss, "ensure_texture", lambda t, d, r: {"diff": tmp_path / "d.jpg", "nor": tmp_path / "n.jpg"})
    assert ss.rug_texture_paths("x", tmp_path) == {"diff": str(tmp_path / "d.jpg"), "nor": str(tmp_path / "n.jpg")}
