"""Renderiza os assets procedurais num estúdio neutro (Cycles) para inspeção visual.

Uso (com o Python que tem ``bpy``):
    python scripts/preview_procedural_assets.py [--out /tmp/claude-0/proc_assets]
        [--samples 32] [--size 480] [--only rug,bed] [--no-unroll] [--device CPU|OPTIX|CUDA|AUTO]

Gera um PNG por asset (e variantes de tapete), uma folha de contato
(``contact_sheet.png``) e a faixa com 5 quadros do tapete desenrolando
(``rug_unroll_strip.png``). Estúdio: piso cinza-claro com fundo infinito,
luz de área suave, câmera 3/4 frontal.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

from mobilia.assets import procedural  # noqa: E402

# (rótulo, tipo, parâmetros, elevação da câmera em graus, azimute em graus)
VARIANTS: list[tuple[str, str, dict, float, float]] = [
    ("rug_solid_border", "rug", {"pattern": "solid_border"}, 38.0, 28.0),
    ("rug_berber", "rug", {"pattern": "berber", "fringe": True}, 38.0, 28.0),
    ("rug_stripes", "rug", {"pattern": "stripes", "color": "#c9bfae", "accent": "#6b5a4a"}, 38.0, 28.0),
    ("rug_persian", "rug", {"pattern": "persian"}, 45.0, 22.0),
    ("floor_lamp", "floor_lamp", {"metal": "black"}, 12.0, 30.0),
    ("table_lamp", "table_lamp", {"base_material": "ceramic", "base_color": "#b9a58c"}, 16.0, 30.0),
    ("bed", "bed", {}, 24.0, 32.0),
    ("curtain", "curtain", {}, 8.0, 24.0),
    ("desk", "desk", {"wood": "oak"}, 20.0, 32.0),
    ("throw_blanket", "throw_blanket", {"pattern": "stripes"}, 30.0, 30.0),
]


# --------------------------------------------------------------------------- estúdio


def _studio_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("estudio_piso")
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.42, 0.42, 0.42, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.65
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    return mat


def _cyclorama(back_y: float, radius: float = 1.6) -> bpy.types.Object:
    """Piso com curva suave até a parede de fundo (sem linha de horizonte)."""
    prof = [(-14.0, 0.0), (back_y, 0.0)]
    for t in np.linspace(0, math.pi / 2, 16)[1:]:
        prof.append((back_y + radius * math.sin(t), radius * (1 - math.cos(t))))
    prof.append((back_y + radius, 9.0))
    xs = (-14.0, 14.0)
    verts, faces = [], []
    for y, z in prof:
        for x in xs:
            verts.append((x, y, z))
    for i in range(len(prof) - 1):
        a = 2 * i
        faces.append((a, a + 1, a + 3, a + 2))
    me = bpy.data.meshes.new("ciclorama")
    me.from_pydata(verts, [], faces)
    me.shade_smooth()
    ob = bpy.data.objects.new("ciclorama", me)
    ob.data.materials.append(_studio_material())
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _back_wall(y: float) -> None:
    me = bpy.data.meshes.new("parede")
    me.from_pydata([(-4, y, 0), (4, y, 0), (4, y, 3.5), (-4, y, 3.5)], [], [(0, 3, 2, 1)])
    ob = bpy.data.objects.new("parede", me)
    mat = _studio_material()
    nt = mat.node_tree
    next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled").inputs["Base Color"].default_value = (
        0.6, 0.58, 0.55, 1.0)
    ob.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(ob)


def _area(name: str, loc, target, size: float, energy: float) -> None:
    light = bpy.data.lights.new(name, "AREA")
    light.size = size
    light.energy = energy
    ob = bpy.data.objects.new(name, light)
    ob.location = loc
    direction = Vector(target) - Vector(loc)
    ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(ob)


def configure_device(device: str) -> str:
    """Seleciona o dispositivo do Cycles: CPU, OPTIX, CUDA ou AUTO (OPTIX > CUDA > CPU)."""
    device = device.upper()
    if device == "CPU":
        return "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return "CPU"
    for kind in (("OPTIX", "CUDA") if device == "AUTO" else (device,)):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            return "GPU"
    print(f"aviso: dispositivo {device} indisponível, usando CPU", flush=True)
    return "CPU"


def setup_scene(size_px: int, samples: int, device: str, aspect: float = 1.0) -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = configure_device(device)
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces = 8
    sc.cycles.seed = 7
    sc.render.resolution_x = size_px
    sc.render.resolution_y = int(round(size_px / aspect))
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.new("estudio")
    bg = next(n for n in world.node_tree.nodes if n.bl_idname == "ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.5, 0.5, 0.52, 1.0)
    bg.inputs["Strength"].default_value = 0.18
    sc.world = world
    return sc


def _bbox(root: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    dg = bpy.context.evaluated_depsgraph_get()
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for c in root.children:
        ev = c.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        mw = np.asarray(c.matrix_world)
        co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        lo, hi = np.minimum(lo, co.min(0)), np.maximum(hi, co.max(0))
        ev.to_mesh_clear()
    return lo, hi


def frame_camera(lo: np.ndarray, hi: np.ndarray, elev_deg: float, azim_deg: float, lens: float = 50.0,
                 margin: float = 1.12, aspect: float = 1.0) -> bpy.types.Object:
    """Câmera 3/4 frontal enquadrando o bounding box (frente do asset = -Y)."""
    sc = bpy.context.scene
    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.lens = lens
    cam.data.sensor_fit = "HORIZONTAL"
    center = (lo + hi) / 2
    el, az = math.radians(elev_deg), math.radians(azim_deg)
    dirv = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    fov_x = 2 * math.atan(cam.data.sensor_width / (2 * lens))
    fov_y = 2 * math.atan(math.tan(fov_x / 2) / aspect)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    fwd = -dirv
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    dist = 0.5
    for c in corners:  # distância mínima para todos os cantos caberem no quadro
        rel = c - center
        x, y, z = rel @ right, rel @ up, rel @ fwd
        dist = max(dist, abs(x) / math.tan(fov_x / 2) * margin - z, abs(y) / math.tan(fov_y / 2) * margin - z)
    pos = center + dirv * dist
    cam.location = pos
    cam.rotation_euler = Vector(-dirv).to_track_quat("-Z", "Y").to_euler()
    return cam


def light_rig(center: np.ndarray, scale: float) -> None:
    s = max(scale, 0.6)
    # potências calibradas para a transformação "Standard" (branco difuso ~0.85)
    _area("chave", center + np.array([-2.2, -1.7, 2.4]) * s, center, 2.0 * s, 100.0 * s * s)
    _area("preenchimento", center + np.array([2.4, -1.6, 1.2]) * s, center, 2.5 * s, 18.0 * s * s)
    _area("recorte", center + np.array([0.8, 2.2, 2.6]) * s, center, 1.6 * s, 40.0 * s * s)


def render(path: Path) -> float:
    sc = bpy.context.scene
    sc.render.filepath = str(path)
    t = time.time()
    bpy.ops.render.render(write_still=True)
    return time.time() - t


# --------------------------------------------------------------------------- pranchas


def contact_sheet(images: list[tuple[str, Path]], out: Path, cols: int = 3) -> None:
    import cv2

    tiles = []
    for label, p in images:
        im = cv2.imread(str(p))
        if im is None:
            continue
        cv2.rectangle(im, (0, 0), (im.shape[1], 26), (32, 32, 32), -1)
        cv2.putText(im, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(im)
    if not tiles:
        return
    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(im, (w, h))
    cv2.imwrite(str(out), sheet)


def unroll_strip(out_dir: Path, size_px: int, samples: int, device: str, n_frames: int = 5) -> Path:
    import cv2

    aspect = 1.5
    sc = setup_scene(size_px, samples, device, aspect)
    _cyclorama(3.0)
    root = procedural.build_asset("rug", {"pattern": "berber", "fringe": True}, "tapete_desenrolar")
    f0, f1 = 1, 49
    procedural.animate_rug_unroll(root, f0, f1)
    sc.frame_start, sc.frame_end = f0, f1
    lo, hi = np.array([-1.2, -0.85, 0.0]), np.array([1.2, 0.85, 0.25])
    frame_camera(lo, hi, 30.0, 18.0, lens=42.0, margin=1.05, aspect=aspect)
    light_rig((lo + hi) / 2, 1.3)
    paths = []
    for i, f in enumerate(np.linspace(f0, f1, n_frames)):
        sc.frame_set(int(round(f)))
        p = out_dir / f"rug_unroll_{i}.png"
        render(p)
        paths.append(p)
    ims = [cv2.imread(str(p)) for p in paths]
    for im, p, f in zip(ims, paths, np.linspace(f0, f1, n_frames)):
        cv2.putText(im, f"quadro {int(round(f))}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1,
                    cv2.LINE_AA)
    strip = np.concatenate(ims, axis=1)
    out = out_dir / "rug_unroll_strip.png"
    cv2.imwrite(str(out), strip)
    return out


# --------------------------------------------------------------------------- principal


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prévia dos assets procedurais do Mobília Studio")
    ap.add_argument("--out", default="/tmp/claude-0/proc_assets")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--only", default="", help="rótulos separados por vírgula (ex.: bed,rug_berber)")
    ap.add_argument("--no-unroll", action="store_true")
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--device", default="CPU", help="CPU | OPTIX | CUDA | AUTO")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    done: list[tuple[str, Path]] = []
    for label, kind, params, elev, azim in VARIANTS:
        if only and label not in only and kind not in only:
            continue
        setup_scene(a.size, a.samples, a.device)
        root = procedural.build_asset(kind, params, label)
        lo, hi = _bbox(root)
        _cyclorama(hi[1] + 1.2 if kind != "curtain" else hi[1] + 2.0)
        if kind == "curtain":
            _back_wall(hi[1] + 0.0005)
        frame_camera(lo, hi, elev, azim)
        light_rig((lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2.2)
        p = out / f"{label}.png"
        dt = render(p)
        print(f"{label}: {dt:.1f} s -> {p}", flush=True)
        done.append((label, p))
    if not a.no_sheet and not only:
        contact_sheet(done, out / "contact_sheet.png")
        print(f"folha de contato -> {out / 'contact_sheet.png'}")
    if not a.no_unroll and (not only or "unroll" in only):
        strip = unroll_strip(out, max(240, a.size * 3 // 4), a.samples, a.device)
        print(f"desenrolar -> {strip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
