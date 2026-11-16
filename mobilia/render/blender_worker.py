"""Worker de render (roda em processo separado, com o módulo bpy = Blender 5.x).

Uso: python -m mobilia.render.blender_worker <scene.json>

Monta a cena 3D a partir da especificação gerada pelo pipeline:
- câmera casada à foto (intrínsecos + pose do MoGe/modelo da sala);
- "proxy" da sala (piso, paredes com recortes nas janelas, teto) como SHADOW CATCHER,
  com albedo projetado da própria foto (coordenada Window) para que a luz rebatida
  nos móveis tenha a cor real do ambiente;
- luzes de área nas janelas + luz ambiente, com intensidades CALIBRADAS por mínimos
  quadrados contra a foto (o ambiente renderizado sem móveis deve reproduzir a foto);
- móveis (glTF Poly Haven ou procedurais) normalizados e animados peça a peça;
- render quadro a quadro em EXR multicamada (Combined RGBA + Shadow Catcher), com
  região de render (border) restrita aos móveis e reaproveitamento de quadros estáticos.

Eventos JSON (uma linha por evento) vão para a saída padrão.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import bpy  # noqa: E402  (precisa vir antes de mathutils)
import mathutils
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

from mobilia.render import choreography as ch

FADE_PROP = "mobilia_fade"
# fração máxima da luz explicada pela "sala não fotografada" (resto vem das janelas)
AMBIENT_MAX_SHARE = 0.5


def emit(**ev) -> None:
    sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ============================================================================ configuração


def configure_device(sc, device: str) -> str:
    """Seleciona OptiX/CUDA quando há GPU NVIDIA; cai para CPU com aviso."""
    sc.cycles.device = "CPU"
    if device.upper() == "CPU":
        return "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return "CPU"
    order = ["OPTIX", "CUDA"] if device.upper() in ("AUTO", "OPTIX") else ["CUDA", "OPTIX"]
    for kind in order:
        try:
            prefs.compute_device_type = kind
            prefs.refresh_devices()
            gpus = [d for d in prefs.devices if d.type == kind]
        except Exception:
            gpus = []
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            sc.cycles.device = "GPU"
            emit(event="log", level="info", message=f"Render na GPU via {kind}: {', '.join(d.name for d in gpus)}")
            return kind
    emit(event="log", level="warning", message="Nenhuma GPU NVIDIA disponível para o Cycles: usando CPU.")
    return "CPU"


def configure_render(sc, spec: dict) -> str:
    r = spec["render"]
    sc.render.engine = "CYCLES"
    dev = configure_device(sc, r.get("device", "AUTO"))
    sc.render.resolution_x = spec["width"]
    sc.render.resolution_y = spec["height"]
    sc.render.resolution_percentage = 100
    sc.render.pixel_aspect_x = sc.render.pixel_aspect_y = 1.0
    sc.render.film_transparent = True
    sc.render.fps = int(r["fps"])
    sc.frame_start, sc.frame_end = 1, int(r["n_frames"])
    sc.cycles.samples = int(r["samples"])
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = float(r.get("adaptive_threshold", 0.015))
    sc.cycles.use_denoising = bool(r.get("denoise", True))
    for attr, val in (("denoiser", "OPENIMAGEDENOISE"), ("denoising_use_gpu", dev != "CPU"),
                      ("denoising_prefilter", "ACCURATE"), ("denoising_quality", "HIGH"),
                      ("denoising_input_passes", "RGB_ALBEDO_NORMAL")):
        try:
            setattr(sc.cycles, attr, val)
        except (AttributeError, TypeError, ValueError):
            pass
    sc.cycles.seed = int(r.get("seed", 7))
    sc.cycles.use_animated_seed = False          # ruído estável entre quadros
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 4
    sc.cycles.glossy_bounces = 4
    sc.cycles.transmission_bounces = 8
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.sample_clamp_indirect = 8.0
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.use_light_tree = True
    sc.render.use_motion_blur = bool(r.get("motion_blur", True))
    sc.render.motion_blur_shutter = float(r.get("shutter", 0.5))
    sc.render.use_persistent_data = True
    if r.get("threads"):
        sc.render.threads_mode = "FIXED"
        sc.render.threads = int(r["threads"])
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    vl = sc.view_layers[0]
    vl.cycles.use_pass_shadow_catcher = True
    vl.use_pass_z = False
    im = sc.render.image_settings
    im.media_type = "MULTI_LAYER_IMAGE"
    im.color_depth = "16"
    im.exr_codec = "ZIP"
    return dev


def setup_camera(sc, spec: dict):
    K = spec["intrinsics"]
    W, H = spec["width"], spec["height"]
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = K["fx"] * cam_data.sensor_width / W
    cam_data.shift_x = -(K["cx"] - W / 2.0) / max(W, H)
    cam_data.shift_y = (K["cy"] - H / 2.0) / max(W, H)
    cam_data.clip_start = 0.02
    cam_data.clip_end = 200.0
    M = np.asarray(spec["cam_to_room"], dtype=np.float64)
    R = M[:3, :3] @ np.diag([1.0, -1.0, -1.0])        # OpenCV -> câmera do Blender
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = M[:3, 3]
    cam.matrix_world = mathutils.Matrix(T.tolist())
    return cam


# ============================================================================ proxy da sala


def _node(nt, kind, loc=(0, 0)):
    n = nt.nodes.new(kind)
    n.location = loc
    return n


def proxy_material(name: str, photo_img, albedo_scale: float, mean_lin: list[float],
                   illuminant: list[float] | None = None):
    """Difuso com albedo = foto projetada pela câmera (coordenada Window) × escala.

    Fora do enquadramento (atrás/ao lado da câmera) usa a cor média da superfície.
    O material NÃO é emissivo (catchers emissivos diluem a própria sombra — validado).
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    tc = _node(nt, "ShaderNodeTexCoord", (-900, 0))
    tex = _node(nt, "ShaderNodeTexImage", (-650, 0))
    tex.image = photo_img
    tex.extension = "CLIP"
    tex.interpolation = "Linear"
    nt.links.new(tc.outputs["Window"], tex.inputs["Vector"])
    mean = _node(nt, "ShaderNodeRGB", (-650, -260))
    mean.outputs[0].default_value = (*mean_lin, 1.0)
    mix = _node(nt, "ShaderNodeMix", (-380, 0))
    mix.data_type = "RGBA"
    sock = {s.identifier: s for s in mix.inputs}
    nt.links.new(tex.outputs["Alpha"], sock["Factor_Float"])
    nt.links.new(mean.outputs[0], sock["A_Color"])
    nt.links.new(tex.outputs["Color"], sock["B_Color"])
    out_color = [s for s in mix.outputs if s.identifier == "Result_Color"][0]
    # A cor da foto já inclui a luz rebatida várias vezes nas paredes; usar a cor
    # saturada como albedo exageraria o "sangramento" de cor em uma sala fechada.
    hsv = _node(nt, "ShaderNodeHueSaturation", (-300, 120))
    hsv.inputs["Saturation"].default_value = 0.75
    nt.links.new(out_color, hsv.inputs["Color"])
    out_color = hsv.outputs["Color"]
    if illuminant is not None:
        # albedo "balanceado": cor da foto dividida pela cor do iluminante
        wb = _node(nt, "ShaderNodeVectorMath", (-240, 120))
        wb.operation = "MULTIPLY"
        nt.links.new(out_color, wb.inputs[0])
        wb.inputs[1].default_value = tuple(1.0 / max(1e-3, c) for c in illuminant)
        out_color = wb.outputs["Vector"]
    scale = _node(nt, "ShaderNodeVectorMath", (-180, 0))
    scale.operation = "SCALE"
    nt.links.new(out_color, scale.inputs[0])
    scale.inputs["Scale"].default_value = float(albedo_scale)
    bsdf = _node(nt, "ShaderNodeBsdfDiffuse", (40, 0))
    nt.links.new(scale.outputs["Vector"], bsdf.inputs["Color"])
    out = _node(nt, "ShaderNodeOutputMaterial", (260, 0))
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _mesh_object(name: str, verts, faces, mat) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def wall_with_holes(p0, d, n, length, height, holes, ext=0.35):
    """Malha de parede (quads) com recortes retangulares nas janelas.

    ``ext`` = extensão além das pontas (número ou [início, fim]); ver ``scene_spec.wall_extensions``.
    """
    e0, e1 = (float(ext), float(ext)) if np.isscalar(ext) else (float(ext[0]), float(ext[1]))
    s_cuts = sorted({-e0, length + e1} | {max(-e0, min(length + e1, v)) for h in holes for v in (h[0], h[1])})
    z_cuts = sorted({0.0, height} | {max(0.0, min(height, v)) for h in holes for v in (h[2], h[3])})
    verts, faces, idx = [], [], {}

    def vid(i, j):
        if (i, j) not in idx:
            s, z = s_cuts[i], z_cuts[j]
            p = np.asarray(p0) + np.asarray(d) * s
            # 1 mm para fora: o proxy fica exatamente atrás da superfície real visível
            p = p - np.asarray(n) * 0.001
            idx[(i, j)] = len(verts)
            verts.append((p[0], p[1], z))
        return idx[(i, j)]

    for i in range(len(s_cuts) - 1):
        for j in range(len(z_cuts) - 1):
            sc_, zc = (s_cuts[i] + s_cuts[i + 1]) / 2, (z_cuts[j] + z_cuts[j + 1]) / 2
            if any(h[0] < sc_ < h[1] and h[2] < zc < h[3] for h in holes):
                continue
            a, b, c, e = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            faces.append((a, b, c, e))
    return verts, faces


def build_room(sc, spec: dict, photo_img) -> list:
    room = spec["room"]
    k = float(spec.get("albedo_scale", 1.0))
    means = spec.get("albedo_mean", {})
    xmin, ymin, xmax, ymax = room["extent"]
    pad = 6.0
    objs = []
    illum = spec.get("illuminant")
    m_floor = proxy_material("proxy_floor", photo_img, k, means.get("floor", [0.3, 0.3, 0.3]), illum)
    floor = _mesh_object("proxy_floor", [(xmin - pad, ymin - pad, 0), (xmax + pad, ymin - pad, 0),
                                         (xmax + pad, ymax + pad, 0), (xmin - pad, ymax + pad, 0)],
                         [(0, 1, 2, 3)], m_floor)
    objs.append(floor)
    ch_h = float(room["ceiling_height"])
    m_ceil = proxy_material("proxy_ceiling", photo_img, k, means.get("ceiling", [0.6, 0.6, 0.6]), illum)
    ceil = _mesh_object("proxy_ceiling", [(xmin - pad, ymin - pad, ch_h), (xmin - pad, ymax + pad, ch_h),
                                          (xmax + pad, ymax + pad, ch_h), (xmax + pad, ymin - pad, ch_h)],
                        [(0, 1, 2, 3)], m_ceil)
    ceil.visible_shadow = False        # o teto estendido além da sala não pode tapar o sol da janela
    objs.append(ceil)
    m_wall = proxy_material("proxy_wall", photo_img, k, means.get("wall", [0.6, 0.6, 0.6]), illum)
    for w in room["walls"]:
        p0, p1 = np.asarray(w["p0"]), np.asarray(w["p1"])
        L = float(np.linalg.norm(p1 - p0))
        d = (p1 - p0) / max(L, 1e-9)
        n = np.asarray(w["normal"])
        holes = [(o["s0"], o["s1"], o["z0"], o["z1"]) for o in room["openings"]
                 if o["wall"] == w["id"] and o["kind"] == "window"]
        verts, faces = wall_with_holes(p0, d, n, L, ch_h, holes, ext=w.get("ext", 0.35))
        ob = _mesh_object(f"proxy_wall_{w['id']}", verts, faces, m_wall)
        # garante normal apontando para dentro da sala
        me = ob.data
        if len(me.polygons) and np.dot(np.asarray(me.polygons[0].normal)[:2], n) < 0:
            for poly in me.polygons:
                poly.flip()
        objs.append(ob)
    objs += closure_walls(spec, ch_h, means.get("wall", [0.6, 0.6, 0.6]), k)
    objs += build_occluders(spec, photo_img, k, means, illum)
    return objs


def build_occluders(spec: dict, photo_img, k: float, means: dict, illum) -> list:
    """Objetos que já existem na foto (bancadas, armários, porta aberta) como catchers:
    ocultam móveis atrás deles e recebem as sombras dos móveis na frente."""
    path = spec.get("occluders")
    if not path or not Path(path).exists():
        return []
    d = np.load(path)
    verts, faces = d["verts"].astype(np.float64), d["faces"].astype(np.int64)
    if len(faces) == 0:
        return []
    # orienta as faces para a câmera
    cam = np.asarray(spec["cam_to_room"], dtype=np.float64)[:3, 3]
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    flip = (n * (cam - v0)).sum(1) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    mat = proxy_material("proxy_occluders", photo_img, k, means.get("wall", [0.5, 0.5, 0.5]), illum)
    ob = _mesh_object("proxy_occluders", verts.tolist(), faces.tolist(), mat)
    return [ob]


def closure_walls(spec: dict, height: float, wall_mean: list[float], k: float) -> list:
    """Fecha a sala nos lados não fotografados (atrás/ao lado da câmera).

    Sem isso a luz ambiente entraria pelo "lado da câmera" como um flash, projetando
    sombras largas e irreais atrás dos móveis. Com a sala fechada, a luz vem das
    janelas e é rebatida pelas superfícies — como no ambiente real.
    """
    xmin, ymin, xmax, ymax = spec["room"]["extent"]
    m = 0.35
    xmin, ymin, xmax, ymax = xmin - m, min(ymin, -0.6) - m, xmax + m, ymax + m
    il = spec.get("illuminant") or [1.0, 1.0, 1.0]
    alb = [float(np.clip(c / max(1e-3, i) * k * 0.9, 0.05, 0.9)) for c, i in zip(wall_mean, il)]
    mat = bpy.data.materials.new("proxy_closure")
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*alb, 1.0)
    b.inputs["Roughness"].default_value = 1.0
    # emissão = luz ambiente vinda da parte da sala que não aparece na foto (calibrada);
    # essas paredes nunca aparecem na câmera, então a emissão não afeta pixels da foto
    b.inputs["Emission Color"].default_value = (*[float(c) for c in il], 1.0)
    b.inputs["Emission Strength"].default_value = 0.0
    quads = [((xmin, ymin), (xmax, ymin)), ((xmax, ymin), (xmax, ymax)),
             ((xmax, ymax), (xmin, ymax)), ((xmin, ymax), (xmin, ymin))]
    out = []
    for i, (a, c) in enumerate(quads):
        verts = [(a[0], a[1], 0.0), (c[0], c[1], 0.0), (c[0], c[1], height), (a[0], a[1], height)]
        ob = _mesh_object(f"proxy_closure_{i}", verts, [(0, 1, 2, 3)], mat)
        ob.visible_shadow = False      # o sol atravessa: a parede real com janela é a ajustada
        out.append(ob)
    return out


def set_catchers(objs, on: bool) -> None:
    for o in objs:
        o.is_shadow_catcher = on


# ============================================================================ luzes


def build_lights(sc, spec: dict) -> dict:
    lt = spec["lighting"]
    windows = []
    room = spec["room"]
    walls = {w["id"]: w for w in room["walls"]}
    for i, o in enumerate(room["openings"]):
        if o["kind"] != "window":
            continue
        w = walls[o["wall"]]
        p0, p1 = np.asarray(w["p0"]), np.asarray(w["p1"])
        d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        n = np.asarray(w["normal"])
        sc_ = 0.5 * (o["s0"] + o["s1"])
        width, height = max(0.2, o["s1"] - o["s0"]), max(0.2, o["z1"] - o["z0"])
        c = p0 + d * sc_ + n * 0.03
        ld = bpy.data.lights.new(f"window_{i}", "AREA")
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = width, height
        ld.color = tuple(lt.get("window_color", [1.0, 1.0, 1.0]))
        ob = bpy.data.objects.new(f"window_{i}", ld)
        sc.collection.objects.link(ob)
        ob.location = (float(c[0]), float(c[1]), 0.5 * (o["z0"] + o["z1"]))
        # luz de área emite ao longo de -Z local: -Z local = normal interna da parede
        z_axis = mathutils.Vector((-n[0], -n[1], 0.0)).normalized()
        x_axis = mathutils.Vector((d[0], d[1], 0.0)).normalized()
        y_axis = z_axis.cross(x_axis)
        ob.matrix_world = mathutils.Matrix.Translation(ob.location) @ mathutils.Matrix(
            ((x_axis.x, y_axis.x, z_axis.x, 0), (x_axis.y, y_axis.y, z_axis.y, 0),
             (x_axis.z, y_axis.z, z_axis.z, 0), (0, 0, 0, 1)))
        ob.visible_camera = False
        windows.append((ob, width * height))
    sun = None
    sun_spec = lt.get("sun")
    if sun_spec:
        sd = bpy.data.lights.new("sol", "SUN")
        sd.angle = math.radians(float(sun_spec.get("angle_deg", 0.6)))
        sd.color = tuple(sun_spec.get("color", [1.0, 0.96, 0.9]))
        sd.energy = 0.0
        sun = bpy.data.objects.new("sol", sd)
        sc.collection.objects.link(sun)
        d = mathutils.Vector(sun_spec["direction"]).normalized()
        # o sol emite ao longo de -Z local: -Z local = -d (do sol para a cena)
        sun.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    world = bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (*lt.get("world_color", [1.0, 1.0, 1.0]), 1.0)
    return {"windows": windows, "world_bg": bg, "sun": sun}


def set_light_strengths(lights: dict, window_power_per_m2: float, ambient_strength: float,
                        sun_strength: float = 0.0) -> None:
    for ob, area in lights["windows"]:
        ob.data.energy = float(window_power_per_m2 * area)
    if lights.get("sun") is not None:
        lights["sun"].data.energy = float(sun_strength)
    mat = bpy.data.materials.get("proxy_closure")
    if mat is not None:
        mat.node_tree.nodes.get("Principled BSDF").inputs["Emission Strength"].default_value = float(ambient_strength)
    # sala fechada: o "world" não entra; mantido em 0 para não haver vazamentos numéricos
    lights["world_bg"].inputs["Strength"].default_value = 0.0


# ============================================================================ calibração


def read_exr(path: str) -> dict:
    import OpenEXR

    f = OpenEXR.File(path, separate_channels=True)
    out = {}
    for part in f.parts:
        for name, c in part.channels.items():
            out[name] = np.asarray(c.pixels, dtype=np.float32)
    return out


def _combined(chs: dict) -> np.ndarray:
    base = [k for k in chs if k.endswith("Combined.R")][0][:-2]
    return np.stack([chs[base + ".R"], chs[base + ".G"], chs[base + ".B"], chs[base + ".A"]], -1)


def calibrate(sc, spec: dict, room_objs: list, lights: dict, out_dir: Path) -> dict:
    """Ajusta potência das janelas e do ambiente para que a sala renderizada SEM móveis
    reproduza a luminância da foto (mínimos quadrados não negativos, 2 parâmetros)."""
    from PIL import Image

    t0 = time.time()
    W = spec["width"]
    cw = int(spec["render"].get("calib_width", 240))
    pct = max(5, min(100, int(round(100 * cw / W))))
    old = (sc.render.resolution_percentage, sc.cycles.samples, sc.render.use_motion_blur)
    sc.render.resolution_percentage = pct
    sc.cycles.samples = 48
    sc.render.use_motion_blur = False
    set_catchers(room_objs, False)
    renders = []
    cases = ["windows", "ambient"] + (["sun"] if lights.get("sun") is not None else [])
    for case in cases:
        set_light_strengths(lights, 1.0 if case == "windows" else 0.0, 1.0 if case == "ambient" else 0.0,
                            1.0 if case == "sun" else 0.0)
        p = str(out_dir / f"calib_{case}.exr")
        sc.render.filepath = p
        sc.frame_set(1)
        bpy.ops.render.render(write_still=True)
        renders.append(_combined(read_exr(p))[..., :3])
    h, w = renders[0].shape[:2]
    photo = np.asarray(Image.open(spec["photo_path"]).convert("RGB").resize((w, h), Image.BILINEAR),
                       dtype=np.float32) / 255.0
    photo = np.where(photo <= 0.04045, photo / 12.92, ((photo + 0.055) / 1.055) ** 2.4)
    mask = np.asarray(Image.open(spec["lighting"]["calib_mask_path"]).convert("L").resize((w, h), Image.NEAREST)) > 127
    lum = np.array([0.2126, 0.7152, 0.0722], np.float32)
    A = np.stack([(r @ lum)[mask] for r in renders], 1)
    b = (photo @ lum)[mask]
    sol = _nnls(A, b)
    tot = A @ sol
    amb_share = float((A[:, 1] * sol[1]).sum() / max(1e-9, tot.sum()))
    if amb_share > AMBIENT_MAX_SHARE and A[:, 0].any():
        # refaz com o ambiente limitado (luz da parte não fotografada não pode dominar)
        amb = AMBIENT_MAX_SHARE * float(b.mean()) / max(1e-9, float(A[:, 1].mean()))
        others = [j for j in range(A.shape[1]) if j != 1]
        sub = _nnls(A[:, others], b - A[:, 1] * amb)
        sol = np.zeros(A.shape[1])
        sol[1] = amb
        sol[others] = sub
    pred = A @ sol
    rel_rms = float(np.sqrt(np.mean((pred - b) ** 2)) / max(1e-6, np.mean(b)))
    # balanço de cor: razão por canal entre foto e predição (aplicada às luzes)
    full = sum(sol[j] * renders[j] for j in range(len(renders)))
    ratio = photo[mask].sum(0) / np.maximum(full[mask].sum(0), 1e-6)
    ratio = ratio / max(1e-6, float(ratio @ lum))
    ratio = np.clip(ratio, 0.85, 1.15)
    sc.render.resolution_percentage, sc.cycles.samples, sc.render.use_motion_blur = old
    res = {"window_power_per_m2": float(sol[0]), "ambient_strength": float(sol[1]),
           "sun_strength": float(sol[2]) if len(sol) > 2 else 0.0,
           "color_ratio": ratio.tolist(), "rel_rms": rel_rms, "pixels": int(mask.sum()),
           "seconds": round(time.time() - t0, 1)}
    return res


def _nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mínimos quadrados não negativos (scipy), com fallback fechado para 2 incógnitas."""
    try:
        from scipy.optimize import nnls

        x, _ = nnls(A.astype(np.float64), b.astype(np.float64))
        return x
    except Exception:
        if A.shape[1] == 2:
            return _nnls2(A, b)
        raise


def _nnls2(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mínimos quadrados não negativos para 2 incógnitas (solução fechada por casos)."""
    best, best_err = np.zeros(2), np.inf
    cands = []
    try:
        cands.append(np.linalg.lstsq(A, b, rcond=None)[0])
    except np.linalg.LinAlgError:
        pass
    for j in range(2):
        a = A[:, j]
        x = np.zeros(2)
        x[j] = max(0.0, float(a @ b) / max(1e-9, float(a @ a)))
        cands.append(x)
    for x in cands:
        if np.all(x >= 0):
            err = float(np.sum((A @ x - b) ** 2))
            if err < best_err:
                best, best_err = x, err
    return best


# ============================================================================ itens


def _wrap_fade(mat) -> None:
    """Insere Mix(Transparente, original) controlado pela propriedade do objeto."""
    if mat is None or not mat.use_nodes or mat.get("mobilia_fade_wrapped"):
        return
    nt = mat.node_tree
    outs = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"]
    if not outs:
        return
    out = next((n for n in outs if n.is_active_output), outs[0])
    surf = out.inputs["Surface"]
    if not surf.links:
        return
    src = surf.links[0].from_socket
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "OBJECT"
    attr.attribute_name = FADE_PROP
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(attr.outputs["Fac"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(src, mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], surf)
    mat["mobilia_fade_wrapped"] = True


def _fix_glass(mat) -> None:
    """Vidro de quadro/vitrine: o Cycles bloqueia luz atrás de vidro refrativo (arte fica preta).
    Troca por vidro fino arquitetônico: transparente + reflexo de Fresnel."""
    if mat is None or not mat.use_nodes:
        return
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return
    tw = bsdf.inputs.get("Transmission Weight")
    refractive = tw is not None and not tw.links and tw.default_value >= 0.5
    # vidro em modo alpha-blend cuja textura de alfa não veio no pacote fica opaco/escuro
    named_glass = "glass" in mat.name.lower() or "vidro" in mat.name.lower()
    if not (refractive or named_glass):
        return
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    fres = nt.nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = 1.5
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl.inputs["Roughness"].default_value = 0.03
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fres.outputs["Fac"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(gl.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def _world_bbox(objs) -> tuple[mathutils.Vector, mathutils.Vector]:
    mn = mathutils.Vector((1e9, 1e9, 1e9))
    mx = mathutils.Vector((-1e9, -1e9, -1e9))
    for o in objs:
        if o.type != "MESH":
            continue
        for c in o.bound_box:
            w = o.matrix_world @ mathutils.Vector(c)
            mn = mathutils.Vector(map(min, mn, w))
            mx = mathutils.Vector(map(max, mx, w))
    return mn, mx


def import_gltf_item(path: str, item: dict) -> tuple[bpy.types.Object, list]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"glTF sem malhas: {path}")
    # assa as transformações nas malhas (cada peça fica com transform identidade)
    bpy.context.view_layer.update()
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.data.transform(mw)
        o.matrix_world = mathutils.Matrix.Identity(4)
    for o in new:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    yaw_fix = math.radians(float(item.get("yaw_fix", 0.0)))
    if yaw_fix:
        R = mathutils.Matrix.Rotation(yaw_fix, 4, "Z")
        for o in meshes:
            o.data.transform(R)
    return _normalize_parts(item["id"], meshes)


def _normalize_parts(item_id: str, meshes: list) -> tuple[bpy.types.Object, list]:
    """Origem do conjunto no centro da base; cada peça com pivô no centro da sua base."""
    bpy.context.view_layer.update()
    mn, mx = _world_bbox(meshes)
    base = mathutils.Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))
    root = bpy.data.objects.new(item_id, None)
    bpy.context.scene.collection.objects.link(root)
    root.empty_display_size = 0.2
    for o in meshes:
        o.data.transform(mathutils.Matrix.Translation(-base))
        pmn, pmx = _world_bbox([o])
        pivot = mathutils.Vector(((pmn.x + pmx.x) / 2, (pmn.y + pmx.y) / 2, pmn.z))
        o.data.transform(mathutils.Matrix.Translation(-pivot))
        o.parent = root
        o.location = pivot
        o["rest_location"] = list(pivot)
        o[FADE_PROP] = 1.0
        for slot in o.material_slots:
            _fix_glass(slot.material)
            _wrap_fade(slot.material)
    return root, meshes


def build_procedural_item(item: dict) -> tuple[bpy.types.Object, list]:
    from mobilia.assets.procedural import build_asset  # executa dentro do Blender

    src = item["source"]
    root = build_asset(src["kind"], dict(src.get("params", {})), item["id"])
    parts = [o for o in root.children_recursive if o.type == "MESH"]
    bpy.context.view_layer.update()
    for o in parts:
        o["rest_location"] = list(o.location)
        o[FADE_PROP] = 1.0
        for slot in o.material_slots:
            _fix_glass(slot.material)
            _wrap_fade(slot.material)
    return root, parts


def place_root(root, item: dict) -> None:
    root.location = tuple(item["position"])
    root.rotation_euler = (0.0, 0.0, math.radians(item.get("yaw_deg", 0.0)))
    s = float(item.get("scale", 1.0))
    root.scale = (s, s, s)


# ============================================================================ animação


def bake_animation(sc, items_rt: list[dict], fps: int, n_frames: int) -> None:
    prefs = bpy.context.preferences.edit
    prefs.keyframe_new_interpolation_type = "LINEAR"
    for rt in items_rt:
        it = rt["item"]
        s = float(it.get("scale", 1.0))
        unroll = rt.get("unroll")
        for part in rt["parts"]:
            plan = rt["plans"][part.name]
            rest = mathutils.Vector(part["rest_location"])
            f_first = max(1, int(math.floor(plan.start * fps)))
            f_last = min(n_frames, int(math.ceil(plan.end * fps)) + 1)
            # antes do início: invisível; depois do fim: pose final exata
            for f, pose in ((1, ch.HIDDEN), (f_first - 1, ch.HIDDEN)):
                if f >= 1:
                    _key_pose(part, rest, pose, s, f)
            for f in range(f_first, f_last + 1):
                t = (f - 1) / fps
                _key_pose(part, rest, plan.pose(t), s, f)
            if f_last < n_frames:
                _key_pose(part, rest, ch.REST, s, n_frames)
        if unroll is not None:
            unroll()


def _key_pose(part, rest, pose: "ch.Pose", s: float, f: int) -> None:
    part.location = (rest.x + pose.dx / s, rest.y + pose.dy / s, rest.z + pose.dz / s)
    part.rotation_euler = (pose.rx, pose.ry, pose.rz)
    part.scale = (pose.sx, pose.sy, pose.sz)
    part[FADE_PROP] = float(pose.fade)
    part.hide_render = not pose.visible
    part.keyframe_insert("location", frame=f)
    part.keyframe_insert("rotation_euler", frame=f)
    part.keyframe_insert("scale", frame=f)
    part.keyframe_insert(f'["{FADE_PROP}"]', frame=f)
    part.keyframe_insert("hide_render", frame=f)


# ============================================================================ render


def frame_signature(items_rt: list[dict], f: int, fps: int) -> tuple:
    sig = []
    for rt in items_rt:
        for part in rt["parts"]:
            plan = rt["plans"][part.name]
            for df in (-1, 0, 1):
                sig.append(plan.pose((f - 1 + df) / fps).key(4))
        if rt.get("unroll_range"):
            a, b = rt["unroll_range"]
            sig.append(("unroll", min(max(f, a - 1), b + 1)))
    return tuple(sig)


def any_visible(items_rt: list[dict], f: int, fps: int) -> bool:
    for rt in items_rt:
        for part in rt["parts"]:
            if rt["plans"][part.name].pose((f - 1) / fps).visible:
                return True
    return False


def dynamic_parts(items_rt: list[dict], f: int, fps: int) -> list:
    """Peças que mudam de aparência em torno do quadro f (movimento, fade ou desenrolar)."""
    rest_key = ch.REST.key(4)
    out = []
    for rt in items_rt:
        ur = rt.get("unroll_range")
        unrolling = ur is not None and ur[0] - 1 <= f <= ur[1] + 1
        for part in rt["parts"]:
            plan = rt["plans"][part.name]
            poses = [plan.pose((f - 1 + df) / fps) for df in (-1, 0, 1)]
            if all(not p.visible for p in poses):
                continue
            if unrolling or any(p.key(4) != rest_key for p in poses):
                out.append((rt, part, poses))
    return out


def _bbox_points(mn, mx, margin_m: float, sun_dir=None) -> list:
    pts = [mathutils.Vector((x, y, z)) for x in (mn.x, mx.x) for y in (mn.y, mx.y) for z in (mn.z, mx.z)]
    pts += [mathutils.Vector((x, y, 0.0)) for x in (mn.x - margin_m, mx.x + margin_m)
            for y in (mn.y - margin_m, mx.y + margin_m)]
    if sun_dir is not None and sun_dir[2] > 0.05:
        # pegada da sombra do sol no piso (projeção dos cantos superiores ao longo de -sol)
        for x in (mn.x, mx.x):
            for y in (mn.y, mx.y):
                t = mx.z / sun_dir[2]
                pts.append(mathutils.Vector((x - sun_dir[0] * t, y - sun_dir[1] * t, 0.0)))
    return pts


def _project_border(sc, cam, pts) -> tuple[float, float, float, float] | None:
    if not pts:
        return None
    xs, ys = [], []
    for p in pts:
        v = world_to_camera_view(sc, cam, p)
        if v.z <= 0.01:
            return (0.0, 0.0, 1.0, 1.0)      # ponto atrás da câmera: tela inteira por segurança
        xs.append(v.x)
        ys.append(v.y)
    pad = 0.02
    return (max(0.0, min(xs) - pad), max(0.0, min(ys) - pad), min(1.0, max(xs) + pad), min(1.0, max(ys) + pad))


def incremental_border(sc, cam, dyn: list, margin_m: float, sun_dir=None):
    """Região da tela que precisa ser re-renderizada: peças em movimento nas poses de
    f−1, f e f+1 (cobre a posição anterior, apagando-a) + margem de sombra."""
    pts = []
    for rt, part, poses in dyn:
        mn, mx = part["_rest_bbox_min"], part["_rest_bbox_max"]
        mn = mathutils.Vector(mn); mx = mathutils.Vector(mx)
        if part.data.shape_keys is not None:
            mx = mathutils.Vector((mx.x, mx.y, mx.z + 0.3))       # rolo do tapete
        s = float(rt["item"].get("scale", 1.0))
        for p in poses:
            grow = max(p.sx, p.sy, p.sz, 1.0)
            ctr = (mn + mx) / 2
            half = (mx - mn) / 2 * grow + mathutils.Vector((0.03, 0.03, 0.03))   # inclinações pequenas
            off = mathutils.Vector((p.dx, p.dy, p.dz)) * s
            pts += _bbox_points(ctr - half + off, ctr + half + off, margin_m, sun_dir)
    return _project_border(sc, cam, pts)


def compute_border(sc, cam, items_rt: list[dict], margin_m: float) -> tuple[float, float, float, float] | None:
    pts = []
    for rt in items_rt:
        for part in rt["parts"]:
            if part.hide_render:
                continue
            mn, mx = _world_bbox([part])
            for x in (mn.x, mx.x):
                for y in (mn.y, mx.y):
                    for z in (mn.z, mx.z):
                        pts.append(mathutils.Vector((x, y, z)))
            # margem no piso para incluir a sombra de contato / sombra das janelas
            for x in (mn.x - margin_m, mx.x + margin_m):
                for y in (mn.y - margin_m, mx.y + margin_m):
                    pts.append(mathutils.Vector((x, y, 0.0)))
                    # e na vertical: peças de parede (quadros) e altas (luminária) sombreiam a
                    # parede acima delas; sem isso o border corta a sombra logo acima da peça
                    pts.append(mathutils.Vector((x, y, mx.z + margin_m)))
    if not pts:
        return None
    xs, ys = [], []
    for p in pts:
        v = world_to_camera_view(sc, cam, p)
        if v.z <= 0.01:
            # ponto atrás da câmera: renderiza a tela inteira por segurança
            return (0.0, 0.0, 1.0, 1.0)
        xs.append(v.x)
        ys.append(v.y)
    pad = 0.02
    return (max(0.0, min(xs) - pad), max(0.0, min(ys) - pad), min(1.0, max(xs) + pad), min(1.0, max(ys) + pad))


def main(spec_path: str) -> None:
    t_start = time.time()
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out_dir = Path(spec["render"]["out_dir"])
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    dev = configure_render(sc, spec)
    cam = setup_camera(sc, spec)
    photo_img = bpy.data.images.load(spec["photo_path"])
    photo_img.colorspace_settings.name = "sRGB"
    room_objs = build_room(sc, spec, photo_img)
    lights = build_lights(sc, spec)
    lt = spec["lighting"]
    calib = None
    if lt.get("calibrate", True):
        emit(event="log", level="info", message="Calibrando a luz pela foto...")
        calib = calibrate(sc, spec, room_objs, lights, out_dir)
        wp, ws, sun_s = calib["window_power_per_m2"], calib["ambient_strength"], calib["sun_strength"]
        # a cor da luz vem do iluminante estimado da foto; a razão por canal é só diagnóstico
        emit(event="calibration", **calib)
    else:
        wp, ws = float(lt.get("window_power_per_m2", 10.0)), float(lt.get("ambient_strength", 0.3))
        sun_s = float((lt.get("sun") or {}).get("strength", 0.0))
    set_light_strengths(lights, wp, ws, sun_s)
    set_catchers(room_objs, True)

    # --- itens
    fps = int(spec["render"]["fps"])
    n_frames = int(spec["render"]["n_frames"])
    items_rt = []
    for item in spec["items"]:
        src = item["source"]
        if src["type"] == "gltf":
            root, parts = import_gltf_item(src["path"], item)
            tags = None
        else:
            root, parts = build_procedural_item(item)
            tags = {p.name: (int(p.get("assembly_order", 1)), str(p.get("assembly_motion", "")))
                    for p in parts}
        place_root(root, item)
        plans = {p.name: p for p in ch.plan_parts(item, [o.name for o in parts], tags)}
        rt = {"item": item, "root": root, "parts": parts, "plans": plans}
        if item.get("motion") == "unroll" and src["type"] == "procedural":
            try:
                from mobilia.assets.procedural import animate_rug_unroll

                f0 = max(1, int(round(item["start"] * fps)))
                f1 = int(round((item["start"] + item["duration"]) * fps))
                rt["unroll"] = (lambda r=root, a=f0, b=f1: animate_rug_unroll(r, a, b, substeps=2))
                rt["unroll_range"] = (f0, f1)
                # durante o desenrolar a peça já aparece (fade rápido) sem "esticar"
                for p in parts:
                    plans[p.name] = ch.PartPlan(p.name, "rise", item["start"], 0.25, {"depth": 0.0})
            except ImportError:
                pass
        items_rt.append(rt)
    bpy.context.view_layer.update()
    for rt in items_rt:
        for part in rt["parts"]:
            mn, mx = _world_bbox([part])
            part["_rest_bbox_min"] = list(mn)
            part["_rest_bbox_max"] = list(mx)
    bake_animation(sc, items_rt, fps, n_frames)
    emit(event="log", level="info", message=f"Cena montada: {len(items_rt)} itens, dispositivo {dev}")

    # --- render quadro a quadro
    W, H = spec["width"], spec["height"]
    margin = float(spec["render"].get("border_margin_m", 0.6))
    meta = {"frames": [], "calibration": calib, "device": dev, "width": W, "height": H}
    last_sig, last_src, last_border = None, None, None
    # Render incremental (só a região das peças em movimento) é EXPERIMENTAL: em testes
    # reduziu a área renderizada em ~14%, mas o sombreamento suave de peças junto à parede
    # pode ultrapassar a região e deixar emendas sutis. Qualidade primeiro: desligado.
    incremental = bool(spec["render"].get("incremental", False)) and not spec["render"].get("only_frames")
    margin_inc = float(spec["render"].get("incremental_margin_m", 0.8))
    full_renders_since = 0
    sun_spec = spec["lighting"].get("sun")
    sun_dir = sun_spec["direction"] if sun_spec else None
    only = set(spec["render"].get("only_frames") or [])
    to_render = [f for f in range(1, n_frames + 1) if any_visible(items_rt, f, fps) and (not only or f in only)]
    done = 0
    t_render = time.time()
    for f in range(1, n_frames + 1):
        if only and f not in only:
            continue
        if not any_visible(items_rt, f, fps):
            meta["frames"].append({"frame": f, "source": None, "border": None})
            continue
        sig = frame_signature(items_rt, f, fps)
        if sig == last_sig:
            meta["frames"].append({"frame": f, "source": last_src, "border": last_border, "reused": True})
            continue
        sc.frame_set(f)
        bpy.context.view_layer.update()
        dyn = dynamic_parts(items_rt, f, fps)
        mode = "full"
        if incremental and last_src is not None and dyn and full_renders_since < 1000:
            b = incremental_border(sc, cam, dyn, margin_inc, sun_dir)
            mode = "incremental"
        else:
            b = compute_border(sc, cam, items_rt, margin)
        if not dyn and last_src is not None and incremental:
            # nada em movimento mas a assinatura mudou (ex.: fim do motion blur): quadro limpo completo
            b = compute_border(sc, cam, items_rt, margin)
            mode = "full"
        if b is None:
            meta["frames"].append({"frame": f, "source": None, "border": None})
            continue
        sc.render.use_border = True
        sc.render.use_crop_to_border = False
        sc.render.border_min_x, sc.render.border_min_y, sc.render.border_max_x, sc.render.border_max_y = b
        name = f"f_{f:05d}.exr"
        sc.render.filepath = str(out_dir / "frames" / name)
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        # retângulo em pixels (origem no topo-esquerdo), igual ao usado pelo Cycles
        border_px = [int(math.floor(b[0] * W)), int(math.floor((1 - b[3]) * H)),
                     int(math.ceil(b[2] * W)), int(math.ceil((1 - b[1]) * H))]
        meta["frames"].append({"frame": f, "source": name, "border": border_px, "mode": mode})
        last_sig, last_src, last_border = sig, name, border_px
        done += 1
        emit(event="render_progress", done=done, frame=f, n_frames=n_frames,
             estimated_total=len(to_render), seconds=round(time.time() - t0, 2))
    meta["timings"] = {"total_s": round(time.time() - t_start, 1), "render_s": round(time.time() - t_render, 1),
                       "rendered_frames": done}
    (out_dir / "frames.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    emit(event="render_done", rendered=done, meta=str(out_dir / "frames.json"))


if __name__ == "__main__":
    main(sys.argv[1])
