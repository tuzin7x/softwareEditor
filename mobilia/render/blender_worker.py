"""Worker de render (roda em processo separado, com o módulo bpy = Blender 5.x).

Uso: python -m mobilia.render.blender_worker <scene.json>
     python -m mobilia.render.blender_worker --aquecimento <pasta>   (instalador: ``warmup_main``)

Monta a cena 3D a partir da especificação gerada pelo pipeline:
- câmera casada à foto (intrínsecos + pose do MoGe/modelo da sala);
- "proxy" da sala (piso, paredes com recortes nas janelas, teto) como SHADOW CATCHER,
  com albedo projetado da própria foto (coordenada Window) para que a luz rebatida
  nos móveis tenha a cor real do ambiente;
- para os raios que NÃO são da câmera a sala-proxy emite a própria foto (luz rebatida real nos
  móveis); para a câmera ela continua um catcher difuso (``proxy_material``);
- luzes de área nas janelas + luz ambiente + sol, com intensidades e cor CALIBRADAS contra a foto
  (a sala renderizada sem móveis deve reproduzir a foto; ver ``render/calibration.py``);
- móveis (glTF Poly Haven ou procedurais) normalizados e animados peça a peça;
- render quadro a quadro em EXR multicamada (Combined RGBA + Shadow Catcher), com
  região de render (border) restrita aos móveis, EXR recortado nessa região e
  reaproveitamento de quadros estáticos;
- render em camadas (PT6, padrão; ``render.camadas = false`` volta ao quadro único): marcos
  estáticos com muitas amostras (o quadro final primeiro) e, nos quadros com movimento, só a
  camada dinâmica — as peças fora do marco vigente — com as peças do marco como *shadow catcher*
  (``render_layered``; o plano é ``render/layers.py::plan_layers``).

Eventos JSON (uma linha por evento) vão para a saída padrão.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

T_PROCESS = time.perf_counter()   # antes do import do bpy: mede a carga do Blender

import bpy  # noqa: E402  (precisa vir antes de mathutils)
import mathutils
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

from mobilia.render import border as bd
from mobilia.render import budget as bg
from mobilia.render import calibration as cal
from mobilia.render import choreography as ch
from mobilia.render import layers as lay
from mobilia.telemetry import RenderStatsCollector

FADE_PROP = "mobilia_fade"
PHOTO_EMISSION_NODE = "photo_emission"
ROOF_NAME = "proxy_telhado"


def emit(**ev) -> None:
    sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------- telemetria do Cycles
# Um único coletor: o handler ``render_stats`` recebe as mensagens de fase do Cycles da chamada
# de render em andamento; ``timed_render`` abre e fecha a janela de cada chamada.
STATS = RenderStatsCollector()
DEVICE_INFO: dict = {"dispositivo": "CPU", "gpus": []}


def _on_render_stats(*args) -> None:
    STATS.feed(args[0] if args else None)


def install_stats_handler() -> None:
    hs = bpy.app.handlers.render_stats
    if _on_render_stats not in hs:
        hs.append(_on_render_stats)


def timed_render(sc, write_still: bool = True) -> dict:
    """``bpy.ops.render.render`` com as estatísticas da chamada (ver ``RenderStatsCollector``)."""
    STATS.start()
    bpy.ops.render.render(write_still=write_still)
    return STATS.finish(denoise=bool(sc.cycles.use_denoising))


def denoiser_info(sc, dev: str) -> dict:
    """Denoiser efetivo. O Cycles cai em silêncio para a CPU quando o OIDN não acha uma GPU
    compatível; aqui a mesma consulta que o Blender faz decide o que é informado."""
    if not sc.cycles.use_denoising:
        return {"denoiser": None, "denoiser_dispositivo": None}
    name = str(getattr(sc.cycles, "denoiser", "OPENIMAGEDENOISE"))
    want_gpu = bool(getattr(sc.cycles, "denoising_use_gpu", False))
    gpu_ok = False
    if dev != "CPU":
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            gpu_ok = bool(prefs.has_optixdenoiser_gpu_devices() if name == "OPTIX" else prefs.has_oidn_gpu_devices())
        except Exception:
            gpu_ok = False
    if name == "OPTIX":
        want_gpu = True
    return {"denoiser": name, "denoiser_dispositivo": "GPU" if (want_gpu and gpu_ok) else "CPU",
            "denoiser_gpu_pedida": want_gpu}


def warmup_kernels(sc) -> dict:
    """Render de ~16 px com denoiser, antes de montar a cena: carrega (ou compila, na primeira vez
    numa GPU nova) os kernels de render e do denoiser. O tempo é o custo de primeira execução
    que o instalador deve absorver (PT2); numa segunda execução ele cai para frações de segundo."""
    old = (sc.render.resolution_percentage, sc.cycles.samples, sc.render.use_motion_blur, sc.render.use_border)
    try:
        sc.render.resolution_percentage = max(1, int(round(100 * 16 / max(1, sc.render.resolution_x))))
        sc.cycles.samples = 1
        sc.render.use_motion_blur = False
        sc.render.use_border = False
        return timed_render(sc, write_still=False)
    except Exception as e:  # aquecimento é só medição: nunca derruba o render
        emit(event="log", level="warning", message=f"Aquecimento dos kernels falhou: {e}")
        return {"parede_s": 0.0, "kernels_s": 0.0, "erro": str(e)}
    finally:
        (sc.render.resolution_percentage, sc.cycles.samples, sc.render.use_motion_blur,
         sc.render.use_border) = old


# ============================================================================ configuração


def configure_device(sc, device: str, quiet: bool = False) -> str:
    """Seleciona OptiX/CUDA quando há GPU NVIDIA; cai para CPU com aviso (``quiet``: sem eventos)."""
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
            DEVICE_INFO.update(dispositivo=kind, gpus=[d.name for d in gpus])
            if not quiet:
                emit(event="log", level="info", message=f"Render na GPU via {kind}: {', '.join(d.name for d in gpus)}")
            return kind
    if not quiet:
        emit(event="log", level="warning", message="Nenhuma GPU NVIDIA disponível para o Cycles: usando CPU.")
    return "CPU"


def configure_denoiser(sc, dev: str, quiet: bool = False) -> str:
    """Denoiser na GPU sempre que possível, guiado por albedo e normal.

    OIDN na GPU quando o Cycles acha um dispositivo compatível; senão, em placa com OptiX, o
    denoiser do OptiX (também na GPU); a CPU só fica para quando não há GPU de render.
    Devolve o nome do denoiser escolhido."""
    name, use_gpu = "OPENIMAGEDENOISE", False
    if dev != "CPU":
        oidn_gpu = optix_gpu = False
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            oidn_gpu = bool(prefs.has_oidn_gpu_devices())
            optix_gpu = dev == "OPTIX" and bool(prefs.has_optixdenoiser_gpu_devices())
        except Exception:
            pass
        if oidn_gpu:
            use_gpu = True
        elif optix_gpu:
            name = "OPTIX"
        elif not quiet:
            emit(event="log", level="warning",
                 message="Nenhuma GPU compatível com o denoiser: o denoise roda na CPU (mais lento).")
    for attr, val in (("denoiser", name), ("denoising_use_gpu", use_gpu),
                      ("denoising_prefilter", "ACCURATE"), ("denoising_quality", "HIGH"),
                      ("denoising_input_passes", "RGB_ALBEDO_NORMAL")):
        try:
            setattr(sc.cycles, attr, val)
        except (AttributeError, TypeError, ValueError):
            pass
    return name


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
    # amostras do quadro final; o laço de quadros troca por camada (ver render/layers.py)
    sc.cycles.samples = int(r["samples"])
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = float(r.get("adaptive_threshold", lay.ADAPTIVE_THRESHOLD))
    amin = r.get("adaptive_min_samples") or lay.ADAPTIVE_MIN
    sc.cycles.adaptive_min_samples = min(int(r["samples"]), int(amin.get("final", lay.ADAPTIVE_MIN["final"])))
    sc.cycles.use_denoising = bool(r.get("denoise", True))
    configure_denoiser(sc, dev, quiet=r.get("device", "AUTO") == "CPU")
    sc.cycles.seed = int(r.get("seed", 7))
    sc.cycles.use_animated_seed = False          # ruído estável entre quadros
    # rebatimentos: difusos continuam em 4 (luz da sala nos móveis); brilho e transparência
    # além de 2 e 8 não aparecem num quadro de sala e custam caro (a calibração usa os mesmos)
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 4
    sc.cycles.glossy_bounces = 2
    sc.cycles.transmission_bounces = 8
    sc.cycles.transparent_max_bounces = 8
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
    # o File Output dos quadros copia estes ajustes; a calibração usa os mesmos (write_still)
    im.exr_codec = bd.exr_codec(r.get("exr_codec"))
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
                   illuminant: list[float] | None = None, emission: bool = True):
    """Material da sala-proxy, misturado por "Is Camera Ray" (``emission``):

    - raio da câmera: difuso. Para a câmera o material NÃO é emissivo (catchers emissivos diluem
      a própria sombra — validado), então o passe de sombra continua o mesmo. O albedo é a cor
      mediana da superfície × escala (ver o comentário abaixo); sem ``emission``, a foto projetada
      pela câmera (coordenada Window) × escala, como antes;
    - qualquer outro raio (rebatido nos móveis, reflexos, raios de sombra): ``Emission`` da própria
      foto linear, força 1. O móvel recebe a luz real refletida pela sala, com a cor e a
      distribuição da foto. A força é o termo fixo da calibração (``set_photo_emission``).

    Fora do enquadramento (atrás/ao lado da câmera) usa a cor média da superfície. A emissão não
    entra na árvore de luzes (``emission_sampling = 'NONE'``): a malha de oclusores tem centenas
    de milhares de triângulos e só é achada pelos raios rebatidos, o que basta para uma sala
    que emite em volta toda.
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
    photo_color = [s for s in mix.outputs if s.identifier == "Result_Color"][0]
    # Com a emissão da foto, o difuso só é visto por raios da câmera: no passe de sombra o albedo
    # se cancela (razão com e sem móveis no mesmo ponto), então ele só pesa nos renders da
    # calibração. Lá ele é UNIFORME por superfície (cor mediana): com a foto como albedo, a luz já
    # "pintada" na foto (manchas de sol, claridade perto da janela) entraria duas vezes e a razão
    # foto/render viraria 1/(irradiância do modelo), sem informação sobre a luz real.
    # Sem emissão (modo antigo) o albedo é a própria foto, que dá a cor da luz rebatida nos móveis.
    out_color = mean.outputs[0] if emission else photo_color
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
    out = _node(nt, "ShaderNodeOutputMaterial", (460, 0))
    if not emission:
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return mat
    em = _node(nt, "ShaderNodeEmission", (40, -220))
    em.name = PHOTO_EMISSION_NODE
    nt.links.new(photo_color, em.inputs["Color"])
    em.inputs["Strength"].default_value = 1.0
    lp = _node(nt, "ShaderNodeLightPath", (40, 260))
    sel = _node(nt, "ShaderNodeMixShader", (260, 0))
    # Fac = 1 (raio da câmera) -> 2º shader (difuso); demais raios -> 1º shader (emissão da foto)
    nt.links.new(lp.outputs["Is Camera Ray"], sel.inputs["Fac"])
    nt.links.new(em.outputs["Emission"], sel.inputs[1])
    nt.links.new(bsdf.outputs["BSDF"], sel.inputs[2])
    nt.links.new(sel.outputs["Shader"], out.inputs["Surface"])
    try:
        mat.cycles.emission_sampling = "NONE"
    except (AttributeError, TypeError):
        pass
    mat["mobilia_photo_emission"] = True
    return mat


def set_photo_emission(strength: float) -> None:
    """Força da emissão da foto em todos os materiais da sala-proxy (0 nos renders da calibração
    que medem uma luz por vez; 1 no resto)."""
    for m in bpy.data.materials:
        if m.get("mobilia_photo_emission") and m.node_tree is not None:
            n = m.node_tree.nodes.get(PHOTO_EMISSION_NODE)
            if n is not None:
                n.inputs["Strength"].default_value = float(strength)


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
    emission = photo_emission_on(spec)
    m_floor = proxy_material("proxy_floor", photo_img, k, means.get("floor", [0.3, 0.3, 0.3]), illum, emission)
    floor = _mesh_object("proxy_floor", [(xmin - pad, ymin - pad, 0), (xmax + pad, ymin - pad, 0),
                                         (xmax + pad, ymax + pad, 0), (xmin - pad, ymax + pad, 0)],
                         [(0, 1, 2, 3)], m_floor)
    objs.append(floor)
    ch_h = float(room["ceiling_height"])
    m_ceil = proxy_material("proxy_ceiling", photo_img, k, means.get("ceiling", [0.6, 0.6, 0.6]), illum, emission)
    ceil = _mesh_object("proxy_ceiling", [(xmin - pad, ymin - pad, ch_h), (xmin - pad, ymax + pad, ch_h),
                                          (xmax + pad, ymax + pad, ch_h), (xmax + pad, ymin - pad, ch_h)],
                        [(0, 1, 2, 3)], m_ceil)
    ceil.visible_shadow = False        # o teto estendido além da sala não pode tapar o sol da janela
    objs.append(ceil)
    m_wall = proxy_material("proxy_wall", photo_img, k, means.get("wall", [0.6, 0.6, 0.6]), illum, emission)
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
    objs += build_occluders(spec, photo_img, k, means, illum, emission)
    roof = build_roof(spec, ch_h)
    if roof is not None:
        objs.append(roof)
    return objs


def photo_emission_on(spec: dict) -> bool:
    return bool((spec.get("lighting") or {}).get("photo_emission", True))


def build_roof(spec: dict, height: float):
    """Telhado só de sombra acima do teto (``calibration.roof_polygons``): o sol entra só pelas
    janelas, nunca por cima das paredes. É catcher (não muda o passe de sombra) e invisível para
    a câmera e para os raios rebatidos."""
    try:
        rings = cal.roof_polygons(spec["room"])
    except Exception as e:  # geometria degenerada: sem telhado (comportamento antigo)
        emit(event="log", level="warning", message=f"Sem telhado de sombra: {e}")
        return None
    if not rings:
        return None
    verts, faces = [], []
    z = float(height) + 0.002
    for ring in rings:
        base = len(verts)
        verts += [(x, y, z) for x, y in ring]
        faces.append(tuple(range(base, base + len(ring))))
    mat = bpy.data.materials.new("proxy_roof")
    ob = _mesh_object(ROOF_NAME, verts, faces, mat)
    for attr in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission",
                 "visible_volume_scatter"):
        setattr(ob, attr, False)
    ob.visible_shadow = True
    return ob


def build_occluders(spec: dict, photo_img, k: float, means: dict, illum, emission: bool = True) -> list:
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
    mat = proxy_material("proxy_occluders", photo_img, k, means.get("wall", [0.5, 0.5, 0.5]), illum, emission)
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


def build_sun_blockers(spec: dict) -> tuple[list, list[dict]]:
    """"Persianas" só de sombra nas partes das janelas por onde o sol da foto não entra
    (``calibration.sun_apertures`` a partir das manchas de sol da foto). Sem elas o sol do render
    atravessa a abertura inteira e ilumina piso (e móveis) onde a foto não tem sol. São catchers
    (não mudam o passe de sombra), no plano da parede, um pouco para fora, invisíveis para a câmera e
    para os raios rebatidos: só bloqueiam o sol (a luz de área da janela fica 3 cm para dentro).
    Ficam 3 mm atrás do plano da parede-proxy e passam das bordas da abertura (``blocker_rects``):
    nenhuma "lâmina" de sol escapa pelo vão entre a parede e a persiana."""
    from PIL import Image

    sun = (spec.get("lighting") or {}).get("sun") or {}
    path = sun.get("mask_path")
    if not sun.get("direction") or not path or not Path(path).exists():
        return [], []
    patch = np.asarray(Image.open(path).convert("L")) > 127
    aps = cal.sun_apertures(spec, patch)
    objs = []
    for ap in aps:
        o = spec["room"]["openings"][ap["janela"]]
        P0, dw, n = cal.window_frame(spec["room"], o)
        for k, (a0, a1, b0, b1) in enumerate(cal.blocker_rects(ap)):
            verts = [tuple(P0 + dw * s_ + np.array([0.0, 0.0, z]) - n * 0.004)
                     for s_, z in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))]
            ob = _mesh_object(f"proxy_persiana_{ap['janela']}_{k}", verts, [(0, 1, 2, 3)],
                              bpy.data.materials.get("proxy_roof") or bpy.data.materials.new("proxy_roof"))
            for attr in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission",
                         "visible_volume_scatter"):
                setattr(ob, attr, False)
            ob.visible_shadow = True
            objs.append(ob)
    return objs, aps


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
    # cores-base (iluminante da foto): a calibração multiplica pelo ganho de cor por canal
    base = {"windows": [tuple(ob.data.color) for ob, _ in windows],
            "sun": tuple(sun.data.color) if sun is not None else None,
            "closure": tuple(spec.get("illuminant") or [1.0, 1.0, 1.0])}
    return {"windows": windows, "world_bg": bg, "sun": sun, "base_colors": base}


def apply_light_colors(lights: dict, gains) -> None:
    """Cor das luzes = cor-base × ganho por canal da calibração (janelas, sol e ambiente)."""
    g = [float(v) for v in gains]
    base = lights.get("base_colors") or {}
    for (ob, _), c in zip(lights["windows"], base.get("windows") or []):
        ob.data.color = tuple(ci * gi for ci, gi in zip(c, g))
    if lights.get("sun") is not None and base.get("sun") is not None:
        lights["sun"].data.color = tuple(ci * gi for ci, gi in zip(base["sun"], g))
    mat = bpy.data.materials.get("proxy_closure")
    if mat is not None and base.get("closure") is not None:
        b = mat.node_tree.nodes.get("Principled BSDF")
        b.inputs["Emission Color"].default_value = (*[ci * gi for ci, gi in zip(base["closure"], g)], 1.0)


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
    """Ajusta as luzes para que a sala renderizada SEM móveis reproduza a foto (PT8).

    Renders rápidos (``calib_width``) da sala vazia, um por fonte: a emissão da foto (termo fixo,
    força 1), as janelas, o ambiente (paredes não fotografadas) e o sol. ``calibration.fit_lights``
    decide as forças (pixels estourados como limite inferior, sol ajustado à parte nas manchas de
    sol da foto, limite do ambiente pela claridade da sala) e o ganho de cor por canal, que é
    aplicado nas luzes aqui. A predição da sala vazia com a luz final (``calib_pred.npy``) serve
    para o ajuste fino do CG no compositor: mapa de ganho residual e curva de tom
    (``composite/grade.py``, gravados em ``ajuste_cg``)."""
    from PIL import Image

    from mobilia.composite import grade as gr

    t0 = time.time()
    W = spec["width"]
    cw = int(spec["render"].get("calib_width", 240))
    pct = max(5, min(100, int(round(100 * cw / W))))
    old = (sc.render.resolution_percentage, sc.cycles.samples, sc.render.use_motion_blur)
    sc.render.resolution_percentage = pct
    sc.cycles.samples = 48
    sc.render.use_motion_blur = False
    set_catchers(room_objs, False)
    emission = photo_emission_on(spec)
    renders: dict[str, np.ndarray] = {}
    stats: list[dict] = []
    cases = (["emissao"] if emission else []) + ["windows", "ambient"] \
        + (["sun"] if lights.get("sun") is not None else [])
    for case in cases:
        set_light_strengths(lights, 1.0 if case == "windows" else 0.0, 1.0 if case == "ambient" else 0.0,
                            1.0 if case == "sun" else 0.0)
        set_photo_emission(1.0 if case == "emissao" else 0.0)
        p = str(out_dir / f"calib_{case}.exr")
        sc.render.filepath = p
        sc.frame_set(1)
        stats.append(timed_render(sc, write_still=True))
        renders[case] = _combined(read_exr(p))[..., :3]
    set_photo_emission(1.0 if emission else 0.0)
    h, w = renders["windows"].shape[:2]
    photo = np.asarray(Image.open(spec["photo_path"]).convert("RGB").resize((w, h), Image.BILINEAR),
                       dtype=np.float32) / 255.0
    photo = np.where(photo <= 0.04045, photo / 12.92, ((photo + 0.055) / 1.055) ** 2.4).astype(np.float32)
    lt = spec["lighting"]
    m8 = np.asarray(Image.open(lt["calib_mask_path"]).convert("L").resize((w, h), Image.NEAREST))
    normal, lower = cal.masks_from_png(m8)
    sun_patch = None
    sun_mask_path = (lt.get("sun") or {}).get("mask_path")
    if sun_mask_path and Path(sun_mask_path).exists():
        sun_patch = np.asarray(Image.open(sun_mask_path).convert("L").resize((w, h), Image.NEAREST)) > 127
    fit = cal.fit_lights({k: v for k, v in renders.items() if k != "emissao"}, photo, normal, lower,
                         emit=renders.get("emissao"), sun_patch=sun_patch)
    apply_light_colors(lights, fit["color_ratio"])
    pred = fit.pop("pred")
    np.save(out_dir / "calib_pred.npy", pred)
    grade, G = gr.grade_data(photo, pred, normal, lower, return_map=True)
    sc.render.resolution_percentage, sc.cycles.samples, sc.render.use_motion_blur = old
    # erro da luz depois do mapa de ganho: o que o CG de fato recebe
    fit["rel_rms_ganho"] = cal.lowpass_rel_rms((G * pred) @ cal.LUM, photo @ cal.LUM, normal)
    res = {**fit, "emissao_foto": emission, "cor_aplicada": True,
           "pixels": int(normal.sum()), "pixels_limite_inferior": int(lower.sum()),
           "pixels_total": int(normal.size), "width": int(w), "height": int(h),
           "predicao": "calib_pred.npy", "ajuste_cg": grade,
           "seconds": round(time.time() - t0, 1), "renders": stats}
    return res


# ============================================================================ itens


# Corte no plano do piso (PT11a, surgir do chão): enquanto o mix de fade está ligado, todo ponto da
# superfície abaixo de FLOOR_CLIP_Z (posição no mundo) vira transparente. Nada de móvel é
# desenhado abaixo do piso nem projeta sombra de lá, mesmo onde o catcher do piso (que já esconde
# o que está atrás dele) não cobre; o corte é exato no plano, sem faixa de dissolução.
FLOOR_CLIP_Z = -0.0002


def _wrap_fade(mat) -> None:
    """Insere Mix(Transparente, original) controlado pela propriedade do objeto e pelo corte do piso:
    Fac = fade × (z_mundo > FLOOR_CLIP_Z)."""
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
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    above = nt.nodes.new("ShaderNodeMath")
    above.operation = "GREATER_THAN"
    above.inputs[1].default_value = FLOOR_CLIP_Z
    fac = nt.nodes.new("ShaderNodeMath")
    fac.operation = "MULTIPLY"
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], above.inputs[0])
    nt.links.new(attr.outputs["Fac"], fac.inputs[0])
    nt.links.new(above.outputs["Value"], fac.inputs[1])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fac.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(src, mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], surf)
    mat["mobilia_fade_wrapped"] = True
    mat["mobilia_fade_mix"] = mix.name
    mat["mobilia_fade_out"] = out.name


def set_fade_shader(mat, on: bool) -> bool:
    """Liga/desliga o Mix(Transparente, original) de ``_wrap_fade``. Desligado, a saída do
    material volta a ser o shader original: sem BSDF transparente, os raios de sombra não
    precisam avaliar o material (a peça volta a ser opaca para o Cycles). Devolve se mudou."""
    nt = mat.node_tree
    mix, out = nt.nodes.get(mat.get("mobilia_fade_mix", "")), nt.nodes.get(mat.get("mobilia_fade_out", ""))
    if mix is None or out is None or not mix.inputs[2].links:
        return False
    surf = out.inputs["Surface"]
    want = mix.outputs["Shader"] if on else mix.inputs[2].links[0].from_socket
    cur = surf.links[0].from_socket if surf.links else None
    if cur == want:
        return False
    nt.links.new(want, surf)
    return True


def fade_materials(items_rt: list[dict]) -> dict:
    """{material: [(rt, peça), ...]} dos materiais com o mix de fade."""
    mats: dict = {}
    for rt in items_rt:
        for part in rt["parts"]:
            for slot in getattr(part, "material_slots", []):
                m = slot.material
                if m is not None and m.get("mobilia_fade_mix"):
                    mats.setdefault(m, []).append((rt, part))
    return mats


def needs_fade_mix(p: "ch.Pose") -> bool:
    """Pose que precisa do mix: em fade-in, ou com a peça abaixo do repouso (pode cruzar o piso)."""
    return p.fade < 1.0 or p.dz < 0.0


def update_fade_shaders(mats: dict, f: int, fps: int) -> int:
    """O mix transparente fica ligado só enquanto alguma peça do material está em fade-in ou abaixo
    da pose final (surgindo do chão: o corte do piso) em torno do quadro f (f−1..f+1); depois o
    material volta ao original. Devolve quantos mudaram."""
    changed = 0
    for m, users in mats.items():
        on = False
        for rt, part in users:
            plan = rt["plans"][part.name]
            for df in (-1, 0, 1):
                p = plan.pose((f - 1 + df) / fps)
                if p.visible and needs_fade_mix(p):
                    on = True
                    break
            if on:
                break
        changed += set_fade_shader(m, on)
    return changed


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
    _check_footprint(item, meshes)
    return _normalize_parts(item["id"], meshes)


def _check_footprint(item: dict, meshes: list) -> None:
    """A pegada do layout (catálogo) tem de casar com a malha: se a malha veio com X/Y trocados
    (``yaw_fix`` faltando no catálogo), gira 90° para não renderizar o móvel atravessado em
    relação ao que o layout planejou (decoração pendurada fora do tampo, colisões)."""
    size = item.get("size")
    if not size:
        return
    bpy.context.view_layer.update()
    mn, mx = _world_bbox(meshes)
    sx, sy = mx.x - mn.x, mx.y - mn.y
    s = float(item.get("scale", 1.0)) or 1.0
    w, d = float(size[0]) / s, float(size[1]) / s
    direct = abs(sx - w) + abs(sy - d)
    swapped = abs(sx - d) + abs(sy - w)
    if direct > 0.05 and swapped < 0.5 * direct:
        R = mathutils.Matrix.Rotation(math.radians(90.0), 4, "Z")
        for o in meshes:
            o.data.transform(R)
        emit(event="log", level="warning",
             message=f"{item.get('asset_id', item.get('id'))}: malha {sx:.3f}x{sy:.3f} m com eixos trocados em "
                     f"relação ao catálogo {w:.3f}x{d:.3f} m; girada 90° (corrigir yaw_fix no catálogo).")


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


def part_tops(parts: list) -> dict:
    """Topo (z no mundo, m) de cada peça na pose final: no surgir do chão cada peça começa inteira
    abaixo do piso."""
    bpy.context.view_layer.update()
    return {o.name: float(_world_bbox([o])[1].z) for o in parts}


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


# Rede de segurança da decoração "on_top": grade de raios sob a base do filho contra a malha do
# apoio, na pose final. Pelo menos REST_MIN_HITS dos pontos precisam encontrar o tampo a até
# REST_TOL m da base (em tampo ripado, ``Support.coverage`` < 1, a fresta entre ripas dentro do
# polígono de apoio conta como apoio); senão o item sai da cena.
REST_GRID = 5
REST_TOL = 0.008
REST_MIN_HITS = 0.9
GAP_REACH = 0.1         # m: maior fresta de tampo ripado que ainda conta como apoio


def _world_bvh(objs: list):
    """BVH (referencial do mundo) das malhas visíveis no render, na pose avaliada atual."""
    from mathutils.bvhtree import BVHTree

    dg = bpy.context.evaluated_depsgraph_get()
    verts, tris, off = [], [], 0
    for o in objs:
        if o.type != "MESH" or o.hide_render:
            continue
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3, np.float64)
        me.vertices.foreach_get("co", co)
        mw = np.asarray(ev.matrix_world, np.float64)
        co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        tri = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
        verts.append(co)
        tris.append(tri.reshape(-1, 3) + off)
        off += len(co)
        ev.to_mesh_clear()
    if not tris or off == 0:
        return None
    return BVHTree.FromPolygons(np.concatenate(verts).tolist(), np.concatenate(tris).tolist())


def _hull_2d(pts: np.ndarray) -> np.ndarray:
    """Fecho convexo 2D (cadeia monótona), vértices em sentido anti-horário."""
    P = sorted({(round(float(a), 6), round(float(b), 6)) for a, b in pts})
    if len(P) < 3:
        return np.asarray(P, np.float64).reshape(-1, 2)

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h

    lo, hi = half(P), half(reversed(P))
    return np.asarray(lo[:-1] + hi[:-1], np.float64)


def _inside_hull(hull: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Máscara dos pontos ``q`` (N×2) dentro do polígono convexo ``hull`` (anti-horário)."""
    if len(hull) < 3:
        return np.zeros(len(q), bool)
    a, b = hull, np.roll(hull, -1, axis=0)
    cross = ((b[:, 0] - a[:, 0])[None] * (q[:, 1:2] - a[:, 1][None])
             - (b[:, 1] - a[:, 1])[None] * (q[:, 0:1] - a[:, 0][None]))
    return np.all(cross >= -eps, axis=1)


def rest_support_hits(bvh, item: dict, z_top: float, n: int = REST_GRID, tol: float = REST_TOL,
                      gaps: bool = False) -> float:
    """Fração dos n×n pontos (grade sob a base do item, recuada 10% das bordas) apoiados no tampo:
    o raio de cima para baixo (começando 3 cm acima) encontra o apoio a até ``tol`` de ``z_top``.

    ``gaps`` (tampo ripado/vazado): um ponto que cai numa fresta conta como apoiado se fica dentro
    do fecho convexo dos acertos de uma grade mais densa (passo ~3 cm) que vai até ``GAP_REACH``
    além da base. Assim a fresta entre ripas (até ~GAP_REACH) não derruba o item, mas a parte da
    base que passa da borda externa do tampo continua sem apoio (não há acertos além dela)."""
    x, y, _ = (float(v) for v in item["position"])
    w, d = float(item["size"][0]), float(item["size"][1])
    yaw = math.radians(float(item.get("yaw_deg", 0.0)))
    c, s_ = math.cos(yaw), math.sin(yaw)
    down = mathutils.Vector((0.0, 0.0, -1.0))

    def hit(fu: float, fv: float) -> bool:
        if bvh is None:
            return False
        u, v = fu * w, fv * d
        o = mathutils.Vector((x + c * u - s_ * v, y + s_ * u + c * v, z_top + 0.03))
        loc = bvh.ray_cast(o, down, 0.03 + tol + 0.002)[0]
        return loc is not None and abs(loc.z - z_top) <= tol

    fr = np.linspace(-0.4, 0.4, n)
    grid = np.array([(fu, fv) for fu in fr for fv in fr], np.float64)
    ok = np.array([hit(fu, fv) for fu, fv in grid], bool)
    if gaps and not ok.all():
        du = np.linspace(-0.5 - GAP_REACH / w, 0.5 + GAP_REACH / w, int(math.ceil((w + 2 * GAP_REACH) / 0.03)) + 1)
        dv = np.linspace(-0.5 - GAP_REACH / d, 0.5 + GAP_REACH / d, int(math.ceil((d + 2 * GAP_REACH) / 0.03)) + 1)
        pts = np.array([(fu * w, fv * d) for fu in du for fv in dv if hit(fu, fv)], np.float64)
        if len(pts) >= 3:
            ok |= _inside_hull(_hull_2d(pts), grid * [w, d])
    return float(ok.mean())


def _check_rest_support(items_rt: list[dict], sc=None, n_frames: int | None = None) -> list[dict]:
    """Remove da cena a decoração "on_top" que, na pose final, não está apoiada no tampo do pai
    (ex.: mesa renderizada girada em relação ao layout). Devolve a lista de itens removidos
    (para o relatório). Assentos (sofá, poltrona, cama) são macios e ficam de fora do teste."""
    from mobilia.layout.surfaces import support_for

    sc = sc or bpy.context.scene
    if n_frames:
        sc.frame_set(int(n_frames))          # pose final: tudo em repouso
    bpy.context.view_layer.update()
    by_id = {rt["item"]["id"]: rt for rt in items_rt}
    bvhs: dict = {}
    removed = []
    for rt in list(items_rt):
        it = rt["item"]
        if it.get("mount") != "on_top":
            continue
        par_rt = by_id.get(it.get("parent") or "")
        if par_rt is None:
            continue
        par = par_rt["item"]
        sup = support_for(par.get("asset_id", ""), par.get("role", ""), par.get("size") or (1, 1, 1),
                          par.get("params") or {})
        if sup is None or sup.kind != "top":
            continue
        if par["id"] not in bvhs:
            bvhs[par["id"]] = _world_bvh(par_rt["parts"])
        z_top = float(it["position"][2])
        frac = rest_support_hits(bvhs[par["id"]], it, z_top, gaps=float(sup.coverage) < 1.0)
        need = REST_MIN_HITS
        if frac >= need:
            continue
        for o in list(rt["parts"]) + [rt["root"]]:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except (ReferenceError, RuntimeError):
                pass
        items_rt.remove(rt)
        rec = {"id": it["id"], "role": it.get("role"), "asset_id": it.get("asset_id"), "parent": par["id"],
               "apoio_acertos": round(frac, 3), "minimo": round(need, 3)}
        removed.append(rec)
        emit(event="log", level="warning",
             message=f"{it['id']} removido: só {frac:.0%} da base apoiada no tampo de {par['id']} "
                     f"(mínimo {need:.0%}); evitaria um objeto flutuando ou pendurado na borda.")
    return removed


# ============================================================================ sombra de contato (PT10)

# papéis de piso sem sombra de contato própria (o tapete é a própria superfície; a cortina pende)
CONTACT_SKIP_ROLES = ("rug", "curtain")


def underside_heights(part, cell: float, h_max: float):
    """Silhueta ortográfica de baixo de uma peça, na pose atual: numa grade de ``cell`` m sob a
    caixa da peça, a altura (m, z do mundo) do primeiro ponto da malha acima de cada célula, por um
    raio vertical de baixo para cima; NaN onde não há malha até ``h_max``. Devolve (alturas (ny,
    nx), centro da célula [0, 0] (x, y)) ou None (peça toda acima de ``h_max``)."""
    mn, mx = _world_bbox([part])
    if mn.z > h_max or mx.x <= mn.x or mx.y <= mn.y:
        return None
    bvh = _world_bvh([part])
    if bvh is None:
        return None
    nx = max(1, int(math.ceil((mx.x - mn.x) / cell)))
    ny = max(1, int(math.ceil((mx.y - mn.y) / cell)))
    x0 = (mn.x + mx.x) / 2 - (nx - 1) * cell / 2
    y0 = (mn.y + mx.y) / 2 - (ny - 1) * cell / 2
    z0 = min(-0.02, mn.z - 0.02)
    up = mathutils.Vector((0.0, 0.0, 1.0))
    reach = h_max - z0
    out = np.full((ny, nx), np.nan)
    for j in range(ny):
        y = y0 + j * cell
        for i in range(nx):
            loc, _n, _k, _d = bvh.ray_cast(mathutils.Vector((x0 + i * cell, y, z0)), up, reach)
            if loc is not None and loc.z <= h_max:
                out[j, i] = loc.z
    return out, (x0, y0)


def contact_model(items_rt: list[dict], spec: dict) -> tuple[dict | None, list]:
    """Modelo da sombra de contato (``frames.json["contact"]``, ``composite/contact.py``) na pose
    final (chamar com a cena em repouso) e, por peça registrada, (rt, peça, M_repouso⁻¹) para as
    poses de cada quadro (``contact_poses``). Só itens de piso."""
    from mobilia.composite import contact as ct

    recs, refs = [], []
    for rt in items_rt:
        it = rt["item"]
        if it.get("mount", "floor") != "floor" or it.get("role") in CONTACT_SKIP_ROLES:
            continue
        for part in rt["parts"]:
            try:
                res = underside_heights(part, ct.CELL_M, ct.HEIGHT_MAX_M)
            except ReferenceError:
                continue
            if res is None:
                continue
            rec = ct.part_record(f"{it['id']}/{part.name}", it["id"], res[1], ct.CELL_M, res[0])
            if rec is None:
                continue
            recs.append(rec)
            refs.append((rt, part, np.linalg.inv(np.asarray(part.matrix_world, np.float64))))
    if not recs:
        return None, []
    meta = {"versao": ct.VERSION, "width": int(spec["width"]), "height": int(spec["height"]),
            "camera": {"intrinsics": spec["intrinsics"], "cam_to_room": spec["cam_to_room"]},
            "celula_m": ct.CELL_M, "altura_max_m": ct.HEIGHT_MAX_M, "pecas": recs}
    return meta, refs


def contact_poses(refs: list, f: int, fps: int, keys=None) -> list:
    """Poses das peças de piso visíveis no quadro f (a cena já em ``frame_set(f)``):
    ``[peça, fade, D 3×4]`` com D = M_quadro · M_repouso⁻¹. ``keys``: só essas peças (nomes dos
    objetos; a camada de um quadro no render em camadas)."""
    from mobilia.composite import contact as ct

    out = []
    for k, (rt, part, inv_rest) in enumerate(refs):
        if keys is not None and part.name not in keys:
            continue
        pose = rt["plans"][part.name].pose((f - 1) / fps)
        if not pose.visible or pose.fade <= ct.MIN_AMP:
            continue
        D = np.asarray(part.matrix_world, np.float64) @ inv_rest
        out.append(ct.pose_record(k, pose.fade, D))
    return out


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


def _bbox_points(mn, mx, margin_m: float, sun_dir=None, vertical: bool = False) -> list:
    pts = [mathutils.Vector((x, y, z)) for x in (mn.x, mx.x) for y in (mn.y, mx.y) for z in (mn.z, mx.z)]
    pts += [mathutils.Vector((x, y, 0.0)) for x in (mn.x - margin_m, mx.x + margin_m)
            for y in (mn.y - margin_m, mx.y + margin_m)]
    if vertical:
        # sombra na parede acima da peça (quadros, luminárias, peças junto à parede)
        pts += [mathutils.Vector((x, y, mx.z + margin_m)) for x in (mn.x - margin_m, mx.x + margin_m)
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


def incremental_border(sc, cam, dyn: list, margin_m: float, sun_dir=None, vertical: bool = False):
    """Região da tela que precisa ser re-renderizada: peças em movimento nas poses de
    f−1, f e f+1 (cobre a posição anterior, apagando-a) + margem de sombra (no piso, na direção
    do sol e, com ``vertical``, também acima da peça)."""
    pts = []
    for rt, part, poses in dyn:
        mn, mx = part["_rest_bbox_min"], part["_rest_bbox_max"]
        mn = mathutils.Vector(mn); mx = mathutils.Vector(mx)
        if part.data.shape_keys is not None:
            mx = mathutils.Vector((mx.x, mx.y, mx.z + 0.3))       # rolo do tapete
        s = float(rt["item"].get("scale", 1.0))
        yaw = math.radians(float(rt["item"].get("yaw_deg", 0.0)))
        cy, sy = math.cos(yaw), math.sin(yaw)
        for p in poses:
            grow = max(p.sx, p.sy, p.sz, 1.0)
            ctr = (mn + mx) / 2
            half = (mx - mn) / 2 * grow + mathutils.Vector((0.03, 0.03, 0.03))   # inclinações pequenas
            # dx/dy da pose estão no referencial do item (girado de yaw); o bbox está no mundo
            # (a escala do item não entra: ``_key_pose`` divide por ela e a raiz multiplica de volta)
            off = mathutils.Vector((cy * p.dx - sy * p.dy, sy * p.dx + cy * p.dy, p.dz))
            lo, hi = ctr - half + off, ctr + half + off
            # nada abaixo do piso é desenhado (surgir do chão): a região para no plano z = 0
            lo.z, hi.z = max(lo.z, 0.0), max(hi.z, 0.0)
            pts += _bbox_points(lo, hi, margin_m, sun_dir, vertical)
    return _project_border(sc, cam, pts)


def compute_border(sc, cam, items_rt: list[dict], margin_m: float,
                   keys=None) -> tuple[float, float, float, float] | None:
    """Região de render de todas as peças visíveis (ou só das ``keys``, p. ex. as de um marco)."""
    pts = []
    for rt in items_rt:
        for part in rt["parts"]:
            if part.hide_render or (keys is not None and part.name not in keys):
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


# ---------------------------------------------------------------------------- saída dos passes
# Com o denoise ligado o Cycles acrescenta "Noisy Image" e "Noisy Shadow Catcher" ao resultado,
# e o EXR multicamada do ``write_still`` grava os dois (quase o dobro de canais para comprimir e
# escrever por quadro). Um nó File Output no compositor grava só o que a composição usa:
# Combined (RGBA, já sem ruído) e Shadow Catcher. Os valores são idênticos aos do write_still.
PASS_OUTPUTS = (("Combined", "Image"), ("Shadow Catcher", "Shadow Catcher"))


def setup_pass_output(sc, directory: Path):
    """Monta o compositor (Render Layers → File Output). Devolve o nó, ou None se não der."""
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ng = bpy.data.node_groups.new("mobilia_passes", "CompositorNodeTree")
            sc.compositing_node_group = ng
        rl = ng.nodes.new("CompositorNodeRLayers")
        rl.scene = sc
        rl.layer = sc.view_layers[0].name
        fo = ng.nodes.new("CompositorNodeOutputFile")
        fmt = fo.format
        fmt.media_type = "MULTI_LAYER_IMAGE"
        fmt.color_depth = sc.render.image_settings.color_depth
        fmt.exr_codec = sc.render.image_settings.exr_codec
        fo.directory = str(directory)
        fo.file_output_items.clear()
        for name, sock in PASS_OUTPUTS:
            fo.file_output_items.new("RGBA", name)
            ng.links.new(rl.outputs[sock], fo.inputs[name])
        sc.render.use_compositing = True
        try:
            sc.render.compositor_device = "CPU"      # só cópia de passes: sem contexto de GPU
        except (AttributeError, TypeError, ValueError):
            pass
        return fo
    except Exception as e:
        sc.render.use_compositing = False
        emit(event="log", level="warning",
             message=f"Saída só com os passes usados indisponível ({e}); gravando o EXR completo.")
        return None


def render_frame(sc, path: Path, out: dict) -> dict:
    """Renderiza o quadro atual em ``path`` (EXR com Combined + Shadow Catcher).

    ``out["node"]`` é o File Output (ou None: ``write_still`` com todos os passes). Se o arquivo
    não aparecer onde esperado, salva o resultado do render e volta ao ``write_still``."""
    sc.render.filepath = str(path)
    fo = out.get("node")
    if fo is None:
        return timed_render(sc, write_still=True)
    fo.file_name = path.stem
    # um quadro de uma execução anterior na mesma pasta não pode passar por "gravado agora"
    # (o write_still sempre sobrescrevia; o nó pode falhar em silêncio)
    path.unlink(missing_ok=True)
    t0 = time.time() - 2.0                    # folga para a resolução do mtime (FAT/NTFS)
    st = timed_render(sc, write_still=False)
    if not path.exists():
        cands = sorted((q for q in path.parent.glob(path.stem + "*.exr") if q.stat().st_mtime >= t0),
                       key=lambda q: q.stat().st_mtime)
        if cands:
            cands[-1].replace(path)
        else:
            bpy.data.images["Render Result"].save_render(str(path))
            out["node"] = None
            sc.render.use_compositing = False
            emit(event="log", level="warning",
                 message="O nó File Output não gravou o quadro; gravando o EXR completo daqui em diante.")
    return st


def add_frame(meta: dict, entry: dict) -> None:
    """Anota o quadro no ``frames.json`` e o anuncia na hora (evento ``frame_ready``): o motor
    compõe e codifica em fluxo enquanto o render segue (PT5). Quando o evento sai, o EXR do
    quadro já está completo no disco (``render_frame`` só volta depois de gravá-lo)."""
    meta["frames"].append(entry)
    emit(event="frame_ready", entry=entry)


def frame_layer(items_rt: list[dict], f: int, fps: int, dyn: list, final_sig: tuple) -> str:
    """Camada do quadro para a telemetria: "dinamico" (peça em movimento), "final" (cena completa
    em repouso, igual ao último quadro) ou "marco" (parado, com parte dos itens)."""
    if dyn:
        return "dinamico"
    return "final" if frame_signature(items_rt, f, fps) == final_sig else "marco"


def visible_items(items_rt: list[dict], f: int, fps: int) -> int:
    return sum(1 for rt in items_rt
               if any(rt["plans"][p.name].pose((f - 1) / fps).visible for p in rt["parts"]))


# ============================================================================ render em camadas (PT6)
# Visibilidade por raio (não é animada, ao contrário de ``hide_render``, que tem keyframes): uma
# peça fora do render atual some para a câmera, para as sombras e para a luz rebatida.
RAY_VISIBILITY = ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission",
                  "visible_volume_scatter", "visible_shadow")


def set_layer_roles(items_rt: list[dict], show, catchers=frozenset()) -> int:
    """Papel de cada peça no próximo render: em ``show`` (ou ``show`` None: todas) é CG normal; em
    ``catchers`` (as peças do marco num quadro dinâmico) fica na cena como *shadow catcher*: esconde
    o que passa atrás dela, recebe a sombra e a luz rebatida das peças novas e continua projetando
    sombra e rebatendo luz nelas; as demais somem de todos os raios. Devolve quantas mudaram."""
    changed = 0
    for rt in items_rt:
        for part in rt["parts"]:
            k = part.name
            cat = k in catchers
            on = show is None or k in show or cat
            diff = False
            for a in RAY_VISIBILITY:
                if hasattr(part, a) and getattr(part, a) != on:
                    setattr(part, a, on)
                    diff = True
            if bool(part.is_shadow_catcher) != cat:
                part.is_shadow_catcher = cat
                diff = True
            changed += diff
    return changed


def layer_parts(items_rt: list[dict]) -> list:
    """Peças da cena para ``layers.plan_layers`` (nome do objeto, item, plano, desenrolar)."""
    out = []
    for rt in items_rt:
        ur = rt.get("unroll_range")
        for part in rt["parts"]:
            out.append(lay.LayerPart(part.name, str(rt["item"]["id"]), rt["plans"][part.name],
                                     tuple(ur) if ur else None))
    return out


def layer_border(sc, cam, items_rt: list[dict], keys, f: int, fps: int, margin_m: float, sun_dir=None):
    """Região da camada dinâmica do quadro f: as peças ``keys`` nas poses de f−1, f e f+1 (desfoque
    de movimento) + margem de contato no piso e acima delas + projeção da sombra do sol."""
    dyn = []
    for rt in items_rt:
        for part in rt["parts"]:
            if part.name in keys:
                plan = rt["plans"][part.name]
                dyn.append((rt, part, [plan.pose((f - 1 + df) / fps) for df in (-1, 0, 1)]))
    return incremental_border(sc, cam, dyn, margin_m, sun_dir, vertical=True) if dyn else None


def _layer_render_count(plan, only: set) -> int:
    """Renders previstos do plano (marcos usados + camadas novas), para a barra de progresso."""
    n, prev = 0, set()
    used = set()
    for fr in plan.frames:
        if only and fr.frame not in only:
            continue
        cur = set()
        if fr.kind == "dinamico":
            cur.add(fr.sig)
            used.add(fr.base)
        elif fr.kind == "estatico":
            used.add(fr.base)
        if fr.trans:
            cur.add(fr.trans[2])
            used.add(fr.trans[0])
        n += len(cur - prev)
        prev = cur
    return n + len({u for u in used if u})


def budget_plan(sc, cam, items_rt: list[dict], plan, rest: list, only: set, fps: int, W: int, H: int,
                margin: float, dmargin: float, sun_dir) -> tuple[list[int], list[int]]:
    """Áreas (px, escala 1) do que falta renderizar depois do quadro final, para a previsão do
    PT7: cada marco de ``rest`` (border de todas as suas peças) e cada render da camada dinâmica
    (``layers.dynamic_jobs``: mesma região que o laço vai usar; renders vazios não contam).

    Volta a cena ao quadro em que estava: quem chama ainda lê ``matrix_world`` do quadro do
    marco recém-renderizado (``contact_poses`` do quadro final)."""
    marcos = []
    f0 = sc.frame_current
    try:
        for m in rest:
            sc.frame_set(m.frame)
            bpy.context.view_layer.update()
            pb = bd.pixel_border(compute_border(sc, cam, items_rt, margin, keys=m.parts) or (0, 0, 0, 0), W, H)
            if pb is not None:
                x0, y0, x1, y1 = pb[0]
                marcos.append((x1 - x0) * (y1 - y0))
    finally:
        if rest:
            sc.frame_set(f0)
            bpy.context.view_layer.update()
    dyn = []
    for f, keys, _sig, _base in lay.dynamic_jobs(plan, only):
        b = layer_border(sc, cam, items_rt, keys, f, fps, dmargin, sun_dir)
        pb = bd.pixel_border(b, W, H) if b is not None else None
        if pb is not None:
            x0, y0, x1, y1 = pb[0]
            dyn.append((x1 - x0) * (y1 - y0))
    return marcos, dyn


def decide_budget(cfg: dict, row: dict | None, marcos: list[int], dyn: list[int], spl: dict, motion_blur: bool,
                  t_ref: float) -> dict:
    """Decisão do PT7 (uma vez, depois do quadro final): mede a sonda, escolhe amostras/escala e
    devolve o registro do relatório (o mesmo vai no evento ``budget``)."""
    probe = bg.measure(row or {})
    t0 = cfg.get("t_clique")
    elapsed = time.time() - (float(t0) if t0 else t_ref)
    total = float(cfg.get("total_s", bg.BUDGET_S))
    pos = float(cfg.get("pos_render_s", bg.POS_RENDER_S))
    res = float(cfg.get("reserva_s", bg.RESERVA_S))
    avail = total - res - pos - elapsed
    info = {"ativo": True, "total_s": total, "decorrido_s": round(elapsed, 2), "pos_render_s": pos,
            "reserva_s": res, "disponivel_render_s": round(avail, 2), "marcos_restantes": len(marcos),
            "renders_dinamicos": len(dyn), "area_dinamica_media_px": round(sum(dyn) / len(dyn)) if dyn else 0,
            "spp_perfil": {"marco": int(spl["marco"]), "dinamico": int(spl["dinamico"])}}
    if probe is None:
        info.update(ativo=False, motivo="sem medida do quadro final")
        return info
    plan = bg.RenderPlan(tuple(marcos), tuple(dyn), int(spl["marco"]), int(spl["dinamico"]), motion_blur)
    st = bg.choose_settings(probe.throughput, probe.overhead, plan, avail)
    info.update({
        "sonda": {"amostras_por_s": round(probe.throughput), "custo_fixo_s": round(probe.overhead.fixed_s, 4),
                  "custo_por_mpx_s": round(probe.overhead.per_px_s * 1e6, 4), "area_px": probe.area_px,
                  "spp": probe.spp, "estimada": probe.estimado},
        "spp_dinamico": st.spp_dinamico, "escala_cg": st.escala_cg, "render_pct": st.render_pct,
        "spp_marcos": st.spp_marcos, "cortes": st.cortes, "cabe": st.cabe,
        "previsto_render_s": st.previsto_s, "sem_corte_render_s": st.sem_corte_s,
        "previsto_total_s": round(elapsed + st.previsto_s + pos, 1),
        "motivo": ("cabe sem cortes" if not st.cortes else
                   ("cortes: " + ", ".join(st.cortes)) + ("" if st.cabe else " (no mínimo, ainda acima do orçamento)")),
    })
    return info


def render_layered(sc, cam, spec: dict, items_rt: list[dict], meta: dict, contact_refs: list, fade_mats: dict,
                   pass_out: dict, out_dir: Path, spl: dict, amin: dict, sun_dir, only: set,
                   t_ref: float | None = None) -> dict:
    """Render em camadas (PT6). Marcos primeiro (o quadro final antes de todos: é também a medida
    de desempenho do PT7), depois os quadros do vídeo em ordem, cada um anunciado já pronto.

    Entradas do ``frames.json``: o quadro parado é o EXR do marco (``camada`` "marco"/"final"); o
    quadro com movimento tem o EXR da camada dinâmica e ``base`` (o marco: EXR, border, poses de
    contato); na troca de marco, ``transicao`` traz a composição antiga (camada + marco anteriores)
    e o ``peso`` dela. Um render cuja assinatura (marco + poses das peças da camada) é a do quadro
    anterior é reaproveitado.

    Orçamento de tempo (PT7, ``render.orcamento``): logo depois do quadro final, a medida dele e o
    plano do restante escolhem as amostras da camada dinâmica, a escala do CG em movimento
    (``resolution_percentage`` só nesses renders; a entrada leva ``escala`` e o compositor reamostra
    o recorte) e as amostras dos marcos (``budget.choose_settings``)."""
    rs = spec["render"]
    spl = dict(spl)
    t_ref = time.time() if t_ref is None else t_ref
    bcfg = dict(rs.get("orcamento") or {})
    scale = {"pct": 100}
    fps, n_frames = int(rs["fps"]), int(rs["n_frames"])
    W, H = spec["width"], spec["height"]
    crop = bool(rs.get("crop_to_border", True))
    margin = float(rs.get("border_margin_m", 0.6))
    dmargin = float(rs.get("dynamic_margin_m", lay.DYNAMIC_MARGIN_M))
    xfade = int(rs.get("crossfade_frames", lay.CROSSFADE_FRAMES))
    t_plan = time.perf_counter()
    groups = [m["itens"] for m in ch.milestones([it for it in spec["items"] if "start" in it])]
    plan = lay.plan_layers(layer_parts(items_rt), fps, n_frames, groups, xfade)
    t_plan = time.perf_counter() - t_plan
    frames = [fr for fr in plan.frames if not only or fr.frame in only]
    used = {fr.base for fr in frames if fr.kind != "foto"} | {fr.trans[0] for fr in frames if fr.trans}
    mss = [m for m in plan.milestones if m.id in used]
    order = sorted(mss, key=lambda m: (not m.final, m.frame))       # o quadro final primeiro
    total = _layer_render_count(plan, only)
    por_quadro: list[dict] = []
    layer_s = {"final": 0.0, "marco": 0.0, "dinamico": 0.0}
    done = 0

    def _render(name: str, border, camada: str, f: int, extra: dict) -> tuple[list | None, dict]:
        nonlocal done
        t_it = time.perf_counter()
        pb = bd.pixel_border(border, W, H) if border is not None else None
        if pb is None:
            return None, {}
        border_px, b = pb
        esc = None
        # EXR de quadro inteiro (crop_to_border desligado) não tem o recorte que o compositor reamostra
        pct = scale["pct"] if camada == "dinamico" and crop else 100
        if pct < 100:
            sb = bd.scaled_border(border, W, H, pct)
            if sb is None:
                return None, {}
            border_px, esc, b = sb
        sc.render.resolution_percentage = pct
        sc.render.use_border = True
        sc.render.use_crop_to_border = crop
        sc.render.border_min_x, sc.render.border_min_y, sc.render.border_max_x, sc.render.border_max_y = b
        spp = int(spl[camada])
        sc.cycles.samples = spp
        sc.cycles.adaptive_min_samples = min(spp, int(amin.get(camada, lay.adaptive_min(camada, spp))))
        fades = update_fade_shaders(fade_mats, f, fps)
        t_prep = time.perf_counter() - t_it
        st = render_frame(sc, out_dir / "frames" / name, pass_out)
        sc.render.resolution_percentage = 100
        done += 1
        rb = esc["border"] if esc else border_px          # pixels realmente renderizados
        area = max(0, rb[2] - rb[0]) * max(0, rb[3] - rb[1])
        if esc:
            extra = {**extra, "escala": pct / 100}
        wall = time.perf_counter() - t_it
        layer_s[camada] += wall
        por_quadro.append({"quadro": f, "camada": camada, **extra, "area_px": area,
                           "cobertura": round(area / max(1, W * H), 4), "spp": int(sc.cycles.samples),
                           "spp_min": int(sc.cycles.adaptive_min_samples), "materiais_fade_trocados": fades,
                           "amostras": st.get("amostras"), "amostragem_s": st.get("amostragem_s"),
                           "denoise_s": st.get("denoise_s"), "gravacao_s": st.get("gravacao_s"),
                           "sincronizacao_s": st.get("sincronizacao_s"), "kernels_s": st.get("kernels_s"),
                           "preparo_s": round(t_prep, 4), "render_s": st.get("parede_s"),
                           "parede_s": round(wall, 4), "mem_mb": st.get("mem_mb")})
        emit(event="render_progress", done=done, frame=f, n_frames=n_frames, camada=camada,
             estimated_total=max(total, done), seconds=round(st.get("parede_s") or 0.0, 2))
        if esc:
            st = {**st, "escala": esc}
        return border_px, st

    # ---- marcos (bases estáticas)
    def _budget_step(rest: list) -> float:
        """PT7: depois do quadro final, decide amostras/escala do restante (uma vez)."""
        t0 = time.perf_counter()
        n0 = len(por_quadro)
        marcos, dyn = budget_plan(sc, cam, items_rt, plan, rest, only, fps, W, H, margin, dmargin, sun_dir)
        info = decide_budget(bcfg, por_quadro[-1] if n0 else None, marcos, dyn, spl,
                             bool(rs.get("motion_blur", True)), t_ref)
        if info.get("ativo"):
            spl.update(bg.Settings(info["spp_dinamico"], info["escala_cg"], info["spp_marcos"], 0, 0, True)
                       .layers(spl))
            scale["pct"] = int(info["render_pct"])
            emit(event="log", level="info" if info["cabe"] else "warning",
                 message=f"Orçamento de tempo: previsão {bg.format_duration(info['previsto_total_s'])} no total "
                         f"(dinâmico {spl['dinamico']} amostras, escala {scale['pct']}%, marcos "
                         f"{spl['marco']} amostras; {info['motivo']})")
        info["estimativa_s"] = round(time.perf_counter() - t0, 3)
        meta["orcamento"] = info
        emit(event="budget", **info)
        return time.perf_counter()

    descs: dict[str, dict | None] = {}
    t_decided = None
    meta["orcamento"] = {"ativo": False, "motivo": "orçamento desligado (render.orcamento)"}
    for i, m in enumerate(order):
        sc.frame_set(m.frame)
        bpy.context.view_layer.update()
        set_layer_roles(items_rt, m.parts)
        b = compute_border(sc, cam, items_rt, margin, keys=m.parts)
        border_px, _ = _render(m.source, b, m.layer, m.frame,
                               {"marco": m.id, "itens": len(m.itens), "pecas": len(m.parts),
                                "pecas_em_movimento": 0})
        if i == 0 and bcfg.get("ativo"):
            if m.final:
                t_decided = _budget_step(order[1:])
            else:
                meta["orcamento"] = {"ativo": False, "motivo": "sem quadro final no plano (quadros selecionados)"}
        if border_px is None:
            descs[m.id] = None                 # nada do marco aparece na imagem: base = a foto
            continue
        d = {"source": m.source, "border": border_px, "frame": m.frame, "camada": m.layer}
        if contact_refs:
            d["contato"] = contact_poses(contact_refs, m.frame, fps, keys=m.parts)
        descs[m.id] = d
    meta["camadas"] = {
        "modo": "camadas", "transicao_quadros": xfade, "margem_dinamica_m": dmargin, "plano_s": round(t_plan, 4),
        "marcos": [{"id": m.id, "quadro": m.frame, "final": m.final, "implicito": m.implicit,
                    "itens": m.itens, "pecas": len(m.parts), "source": (descs.get(m.id) or {}).get("source"),
                    "border": (descs.get(m.id) or {}).get("border")} for m in sorted(mss, key=lambda m: m.frame)]}

    def _base(mid):
        d = descs.get(mid) if mid else None
        return None if d is None else {k: d[k] for k in ("source", "border", "frame", "contato") if k in d}

    # ---- quadros do vídeo, em ordem
    prev_renders: dict = {}
    prev_entry: dict | None = None
    for fr in frames:
        f = fr.frame
        cur: dict = {}
        if fr.kind == "foto":
            add_frame(meta, {"frame": f, "source": None, "border": None})
            prev_renders, prev_entry = {}, None
            continue
        need_set = fr.kind == "dinamico" or fr.trans is not None
        if need_set:
            sc.frame_set(f)
            bpy.context.view_layer.update()

        def _dyn(mid, keys, sig, prefix):
            """Render (ou reaproveitamento) da camada ``keys`` sobre o marco ``mid``."""
            if sig in prev_renders:
                cur[sig] = prev_renders[sig]
                return cur[sig]
            if sig in cur:
                return cur[sig]
            base_m = plan.milestone(mid)
            # peças do marco como catcher mesmo se o marco não aparece na imagem (ainda fazem sombra)
            cats = base_m.parts if base_m is not None else frozenset()
            set_layer_roles(items_rt, keys, cats)
            b = layer_border(sc, cam, items_rt, keys, f, fps, dmargin, sun_dir)
            name = f"{prefix}_{f:05d}.exr"
            moving = sum(1 for k in keys if f < plan.settle.get(k, 0))
            border_px, st = _render(name, b, "dinamico", f, {"base": mid, "pecas": len(keys),
                                                             "pecas_em_movimento": moving,
                                                             "transicao": prefix == "t"})
            rec = None if border_px is None else {
                "source": name, "border": border_px,
                "contato": contact_poses(contact_refs, f, fps, keys=keys) if contact_refs else None}
            if rec is not None and st.get("escala"):
                rec["escala"] = st["escala"]
            cur[sig] = rec
            return rec

        if fr.kind == "estatico":
            d = descs.get(fr.base)
            entry = None if d is None else {"frame": f, "source": d["source"], "border": d["border"],
                                            "mode": "full", "crop": crop, "camada": d["camada"]}
            if entry is not None and d.get("contato") is not None:
                entry["contato"] = d["contato"]
        else:
            rec = _dyn(fr.base, fr.layer, fr.sig, "f")
            if rec is None:                     # a camada não aparece na imagem: o quadro é o marco
                d = descs.get(fr.base) if fr.base else None
                entry = None if d is None else {"frame": f, "source": d["source"], "border": d["border"],
                                                "mode": "full", "crop": crop, "camada": d["camada"],
                                                **({"contato": d["contato"]} if d.get("contato") else {})}
            else:
                entry = {"frame": f, "source": rec["source"], "border": rec["border"], "mode": "full",
                         "crop": crop, "camada": "dinamico", "base": _base(fr.base)}
                if rec.get("escala"):
                    entry["escala"] = rec["escala"]
                if rec.get("contato") is not None:
                    entry["contato"] = rec["contato"]
        if entry is not None and fr.trans is not None:
            omid, okeys, osig, w_old = fr.trans
            orec = _dyn(omid, okeys, osig, "t")
            if orec is not None:
                entry["transicao"] = {"source": orec["source"], "border": orec["border"], "base": _base(omid),
                                      "peso": float(w_old)}
                if orec.get("escala"):
                    entry["transicao"]["escala"] = orec["escala"]
                if orec.get("contato") is not None:
                    entry["transicao"]["contato"] = orec["contato"]
        if entry is None:
            add_frame(meta, {"frame": f, "source": None, "border": None})
            prev_renders, prev_entry = cur, None
            continue
        body = {k: v for k, v in entry.items() if k != "frame"}
        if prev_entry is not None and body == prev_entry:
            add_frame(meta, {"frame": f, **body, "reused": True})
        else:
            add_frame(meta, entry)
        prev_entry = body
        prev_renders = cur
    set_layer_roles(items_rt, None)
    orc = meta.get("orcamento") or {}
    if t_decided is not None and orc.get("ativo"):
        orc["real_render_s"] = round(time.perf_counter() - t_decided, 2)
    return {"por_quadro": por_quadro, "layer_s": layer_s, "done": done, "total": total, "spl": spl}


def main(spec_path: str) -> None:
    t_start = time.time()
    t_main = time.perf_counter()
    tempos: dict[str, float] = {"importacao_bpy": round(t_main - T_PROCESS, 3)}
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out_dir = Path(spec["render"]["out_dir"])
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    install_stats_handler()
    sc = bpy.context.scene
    dev = configure_render(sc, spec)
    cam = setup_camera(sc, spec)
    dev_info = {"dispositivo": dev, "optix": dev == "OPTIX", "gpus": list(DEVICE_INFO.get("gpus") or []),
                "threads": int(sc.render.threads), "blender": bpy.app.version_string, **denoiser_info(sc, dev)}
    tempos["configuracao"] = round(time.perf_counter() - t_main, 3)
    emit(event="worker_ready", **dev_info)

    # --- aquecimento: kernels de render e do denoiser (custo de primeira execução, medido à parte)
    t = time.perf_counter()
    warm = warmup_kernels(sc)
    tempos["kernels"] = time.perf_counter() - t
    if warm.get("kernels_s", 0.0) > 2.0:
        emit(event="log", level="info",
             message=f"Kernels do Cycles carregados/compilados em {warm['kernels_s']:.0f} s (primeira execução)")

    t = time.perf_counter()
    photo_img = bpy.data.images.load(spec["photo_path"])
    photo_img.colorspace_settings.name = "sRGB"
    room_objs = build_room(sc, spec, photo_img)
    lights = build_lights(sc, spec)
    blockers, sun_aps = build_sun_blockers(spec)
    room_objs += blockers
    tempos["cena_montagem"] = time.perf_counter() - t
    lt = spec["lighting"]
    calib = None
    t = time.perf_counter()
    if lt.get("calibrate", True):
        emit(event="log", level="info", message="Calibrando a luz pela foto...")
        calib = calibrate(sc, spec, room_objs, lights, out_dir)
        if sun_aps:
            calib.setdefault("sol", {})["aberturas"] = sun_aps
        wp, ws, sun_s = calib["window_power_per_m2"], calib["ambient_strength"], calib["sun_strength"]
        # cor das luzes = iluminante da foto × ganho por canal (já aplicado na calibração); o
        # evento leva também o ajuste do CG (mapa de ganho, curva de tom) para a composição em fluxo
        emit(event="calibration", **calib)
    else:
        wp, ws = float(lt.get("window_power_per_m2", 10.0)), float(lt.get("ambient_strength", 0.3))
        sun_s = float((lt.get("sun") or {}).get("strength", 0.0))
    set_light_strengths(lights, wp, ws, sun_s)
    set_photo_emission(1.0 if photo_emission_on(spec) else 0.0)
    set_catchers(room_objs, True)
    tempos["calibracao"] = time.perf_counter() - t

    # --- itens
    t = time.perf_counter()
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
        tops = part_tops(parts) if ch.item_motion(item) == "emerge" else None
        plans = {p.name: p for p in ch.plan_parts(item, [o.name for o in parts], tags, tops)}
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
    removed = _check_rest_support(items_rt, sc, n_frames)
    tempos["cena_montagem"] += time.perf_counter() - t
    # silhuetas de baixo das peças de piso na pose final (a cena está em repouso no último quadro)
    t = time.perf_counter()
    contact, contact_refs = (None, [])
    if spec["render"].get("contact_shadow", True):
        sc.frame_set(n_frames)
        bpy.context.view_layer.update()
        contact, contact_refs = contact_model(items_rt, spec)
    tempos["sombra_contato"] = time.perf_counter() - t          # informativo: já dentro de cena_montagem
    tempos["cena_montagem"] += tempos["sombra_contato"]
    if contact is not None:
        emit(event="contact", **contact)
    emit(event="log", level="info", message=f"Cena montada: {len(items_rt)} itens, dispositivo {dev}")

    # --- render quadro a quadro
    W, H = spec["width"], spec["height"]
    margin = float(spec["render"].get("border_margin_m", 0.6))
    # EXR recortado no border (só os pixels renderizados vão ao disco)
    crop = bool(spec["render"].get("crop_to_border", True))
    meta = {"frames": [], "calibration": calib, "device": dev, "width": W, "height": H,
            "removed_items": removed, "exr": {"recorte": crop, "codec": sc.render.image_settings.exr_codec},
            "contact": contact}
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
    rs = spec["render"]
    # render em camadas (PT6) é o padrão; ``render.camadas = false`` (ou o incremental antigo) volta
    # ao quadro único: a cena inteira em todo quadro com movimento
    layered = bool(rs.get("camadas", True)) and not incremental
    spl = lay.clamp_layers(rs.get("samples_layers"), int(rs["samples"]))
    amin = rs.get("adaptive_min_samples") or lay.ADAPTIVE_MIN
    fade_mats = fade_materials(items_rt)
    pass_out = {"node": setup_pass_output(sc, out_dir / "frames")}
    meta["passes"] = [n for n, _ in PASS_OUTPUTS] if pass_out["node"] is not None else "todos"
    por_quadro: list[dict] = []
    layer_s = {"final": 0.0, "marco": 0.0, "dinamico": 0.0}
    done = 0
    t_render = time.time()
    t_loop = time.perf_counter()
    if layered:
        res = render_layered(sc, cam, spec, items_rt, meta, contact_refs, fade_mats, pass_out, out_dir, spl, amin,
                             sun_dir, only, t_ref=t_start)
        por_quadro, layer_s, done, spl = res["por_quadro"], res["layer_s"], res["done"], res["spl"]
        frames_iter = []
    else:
        meta["camadas"] = {"modo": "unica"}
        meta["orcamento"] = {"ativo": False, "motivo": "render de quadro único (o controle usa o plano em camadas)"}
        to_render = [f for f in range(1, n_frames + 1) if any_visible(items_rt, f, fps) and (not only or f in only)]
        final_sig = frame_signature(items_rt, n_frames, fps)
        # amostras por camada (final > marco > dinâmico), com rampa antes dos quadros parados
        planned, prev_sig = [], None
        for f in to_render:
            sig = frame_signature(items_rt, f, fps)
            if sig != prev_sig:
                planned.append((f, frame_layer(items_rt, f, fps, dynamic_parts(items_rt, f, fps), final_sig)))
            prev_sig = sig
        spp_plan = dict(zip((f for f, _ in planned),
                            lay.frame_samples([c for _, c in planned], spl,
                                              int(rs.get("sample_ramp_frames", lay.RAMP_FRAMES)))))
        frames_iter = range(1, n_frames + 1)
    for f in frames_iter:
        if only and f not in only:
            continue
        if not any_visible(items_rt, f, fps):
            add_frame(meta, {"frame": f, "source": None, "border": None})
            continue
        sig = frame_signature(items_rt, f, fps)
        if sig == last_sig:
            add_frame(meta, {"frame": f, "source": last_src, "border": last_border, "reused": True})
            continue
        t_it = time.perf_counter()
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
        # retângulo em pixels (origem no topo-esquerdo) EXATAMENTE igual ao que o Cycles renderiza;
        # com o recorte, o EXR tem só esse retângulo e a composição o cola no quadro inteiro
        pb = bd.pixel_border(b, W, H) if b is not None else None
        if pb is None:
            add_frame(meta, {"frame": f, "source": None, "border": None})
            continue
        border_px, b = pb
        sc.render.use_border = True
        sc.render.use_crop_to_border = crop
        sc.render.border_min_x, sc.render.border_min_y, sc.render.border_max_x, sc.render.border_max_y = b
        camada = frame_layer(items_rt, f, fps, dyn, final_sig)
        spp = int(spp_plan.get(f, spl[camada]))
        sc.cycles.samples = spp
        sc.cycles.adaptive_min_samples = min(spp, int(amin.get(camada, lay.adaptive_min(camada, spp))))
        fades = update_fade_shaders(fade_mats, f, fps)
        name = f"f_{f:05d}.exr"
        t_prep = time.perf_counter() - t_it
        st = render_frame(sc, out_dir / "frames" / name, pass_out)
        entry = {"frame": f, "source": name, "border": border_px, "mode": mode, "crop": crop}
        if contact_refs:
            entry["contato"] = contact_poses(contact_refs, f, fps)
        add_frame(meta, entry)
        last_sig, last_src, last_border = sig, name, border_px
        done += 1
        area = max(0, border_px[2] - border_px[0]) * max(0, border_px[3] - border_px[1])
        wall = time.perf_counter() - t_it
        layer_s[camada] += wall
        por_quadro.append({"quadro": f, "camada": camada, "itens": visible_items(items_rt, f, fps),
                           "pecas_em_movimento": len(dyn), "area_px": area,
                           "cobertura": round(area / max(1, W * H), 4), "spp": int(sc.cycles.samples),
                           "spp_min": int(sc.cycles.adaptive_min_samples), "materiais_fade_trocados": fades,
                           "amostras": st.get("amostras"), "amostragem_s": st.get("amostragem_s"),
                           "denoise_s": st.get("denoise_s"), "gravacao_s": st.get("gravacao_s"),
                           "sincronizacao_s": st.get("sincronizacao_s"), "kernels_s": st.get("kernels_s"),
                           "preparo_s": round(t_prep, 4), "render_s": st.get("parede_s"),
                           "parede_s": round(wall, 4), "mem_mb": st.get("mem_mb")})
        emit(event="render_progress", done=done, frame=f, n_frames=n_frames, camada=camada,
             estimated_total=len(to_render), seconds=round(st.get("parede_s") or 0.0, 2))
    loop_s = time.perf_counter() - t_loop
    tempos["render_final"] = layer_s["final"]
    tempos["render_marcos"] = layer_s["marco"]
    tempos["render_dinamico"] = layer_s["dinamico"]
    # assinaturas, quadros reaproveitados e só-foto: fora das camadas, mas dentro do laço
    tempos["laco_outros"] = max(0.0, loop_s - sum(layer_s.values()))
    frames = meta["frames"]
    meta["timings"] = {"total_s": round(time.time() - t_start, 1), "render_s": round(time.time() - t_render, 1),
                       "rendered_frames": done}
    meta["telemetria"] = {
        **dev_info,
        "tempos_s": {k: round(v, 3) for k, v in tempos.items()},
        "aquecimento": warm,
        "calibracao_renders": (calib or {}).get("renders", []),
        "por_quadro": por_quadro,
        "quadros": lay.frame_counts(frames, done),
        "samples_config": int(spec["render"]["samples"]),
        "samples_camadas": spl,
        "camadas": meta.get("camadas"),
        "orcamento": meta.get("orcamento"),
        "passes": meta["passes"],
        "exr": meta["exr"],
    }
    (out_dir / "frames.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    emit(event="render_done", rendered=done, meta=str(out_dir / "frames.json"))


# ============================================================================ aquecimento do instalador

WARMUP_PX = 64
WARMUP_SAMPLES = 8
# Ordem dos backends aquecidos: o render usa OptiX quando existe; o CUDA é o plano B do
# ``configure_device`` e o OIDN na GPU usa o CUDA de qualquer jeito.
WARMUP_KINDS = ("OPTIX", "CUDA")


def _warm_material(name: str, kind: str, img=None):
    """Materiais com os mesmos nós/recursos dos trabalhos: Principled com textura e relevo,
    vidro e cortina (translúcido + transparente). A sala-proxy usa o ``proxy_material`` real."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = _node(nt, "ShaderNodeOutputMaterial", (400, 0))
    if kind == "curtain":
        tr = _node(nt, "ShaderNodeBsdfTranslucent")
        tp = _node(nt, "ShaderNodeBsdfTransparent")
        mix = _node(nt, "ShaderNodeMixShader", (200, 0))
        mix.inputs["Fac"].default_value = 0.4
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(tp.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
        return mat
    bsdf = _node(nt, "ShaderNodeBsdfPrincipled", (100, 0))
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    if kind == "glass":
        for k, v in (("Transmission Weight", 1.0), ("Roughness", 0.05), ("IOR", 1.45)):
            if k in bsdf.inputs:
                bsdf.inputs[k].default_value = v
        return mat
    tex = _node(nt, "ShaderNodeTexImage", (-300, 100))
    tex.image = img
    noise = _node(nt, "ShaderNodeTexNoise", (-300, -150))
    bump = _node(nt, "ShaderNodeBump", (-100, -150))
    bump.inputs["Strength"].default_value = 0.2
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.45
    return mat


def build_warmup_scene(sc) -> None:
    """Cena mínima com os recursos que mudam o conjunto de kernels do Cycles/OptiX nos trabalhos:
    shadow catcher (com a emissão da foto por "Is Camera Ray"), filme transparente, desfoque de
    movimento (objeto animado), vidro, transparência, texturas de imagem, relevo, luz de área, sol
    e mundo."""
    img = bpy.data.images.new("aquecimento", 8, 8)
    img.pixels.foreach_set(np.tile(np.array([0.6, 0.5, 0.4, 1.0], np.float32), 64))
    cam_d = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    cam.location = (0.0, -3.0, 1.2)
    cam.rotation_euler = (math.radians(75), 0.0, 0.0)
    sc.camera = cam

    mesh = _mesh_object
    # sala-proxy igual à dos trabalhos: catcher difuso para a câmera, emissão da foto nos outros raios
    floor = mesh("piso", [(-3, -3, 0), (3, -3, 0), (3, 3, 0), (-3, 3, 0)], [(0, 1, 2, 3)],
                 proxy_material("piso", img, 1.0, [0.5, 0.5, 0.5]))
    floor.is_shadow_catcher = True
    cube_v = [(x, y, z) for x in (-0.3, 0.3) for y in (-0.3, 0.3) for z in (0.0, 0.6)]
    cube_f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    box = mesh("movel", cube_v, cube_f, _warm_material("movel", "principled", img))
    box.location = (-0.6, 0.0, 0.0)
    box.keyframe_insert("location", frame=1)
    box.location = (-0.4, 0.0, 0.15)
    box.keyframe_insert("location", frame=3)
    glass = mesh("vaso", cube_v, cube_f, _warm_material("vaso", "glass"))
    glass.scale = (0.4, 0.4, 0.8)
    glass.location = (0.5, 0.2, 0.0)
    mesh("cortina", [(-1, 1, 0), (1, 1, 0), (1, 1, 2), (-1, 1, 2)], [(0, 1, 2, 3)],
         _warm_material("cortina", "curtain"))
    ld = bpy.data.lights.new("janela", "AREA")
    ld.shape, ld.size, ld.size_y, ld.energy = "RECTANGLE", 1.0, 1.2, 300.0
    lo = bpy.data.objects.new("janela", ld)
    sc.collection.objects.link(lo)
    lo.location = (0.0, 2.0, 1.5)
    lo.rotation_euler = (math.radians(-90), 0.0, 0.0)
    sd = bpy.data.lights.new("sol", "SUN")
    sd.energy = 2.0
    so = bpy.data.objects.new("sol", sd)
    sc.collection.objects.link(so)
    so.rotation_euler = (math.radians(40), 0.0, math.radians(30))
    world = bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs["Color"].default_value = (0.3, 0.3, 0.3, 1.0)


def _warm_render(sc, out: Path, motion_blur: bool) -> dict:
    sc.render.use_motion_blur = motion_blur
    sc.frame_set(2)
    sc.render.filepath = str(out)
    st = timed_render(sc, write_still=True)
    return {k: st.get(k) for k in ("parede_s", "kernels_s", "sincronizacao_s", "amostras", "mem_mb")}


def warmup_main(out_dir: str) -> dict:
    """Aquecimento do instalador (PT2): renders de 64 × 64 com OptiX e com CUDA (ou CPU, sem GPU)
    na MESMA instalação do Blender que os trabalhos usam. Preenche os caches de kernels do driver
    NVIDIA (OptiX e CUDA) e do denoiser, para que o primeiro vídeo não pague a compilação.

    Para cada dispositivo: um render sem e outro com desfoque de movimento (o OptiX monta um
    pipeline diferente para cada caso) e um terceiro, já com os kernels prontos, como referência.
    Grava ``aquecimento.json`` em ``out_dir`` e emite ``warmup_done``."""
    t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    install_stats_handler()
    sc = bpy.context.scene
    spec = {"width": WARMUP_PX, "height": WARMUP_PX,
            "render": {"device": "CPU", "fps": 24, "n_frames": 3, "samples": WARMUP_SAMPLES,
                       "denoise": True, "motion_blur": True, "seed": 7}}
    configure_render(sc, spec)
    sc.render.use_persistent_data = False       # troca de dispositivo entre os renders
    build_warmup_scene(sc)
    emit(event="progress", fraction=0.05, message=f"Blender {bpy.app.version_string} pronto")
    kinds = []
    for kind in WARMUP_KINDS:
        if configure_device(sc, kind, quiet=True) == kind:
            kinds.append(kind)
    if not kinds:
        kinds = ["CPU"]
        emit(event="log", level="warning", message="Nenhuma GPU NVIDIA disponível para o Cycles: aquecendo a CPU.")
    results = []
    for i, kind in enumerate(kinds):
        dev = configure_device(sc, kind)
        configure_denoiser(sc, dev, quiet=True)
        emit(event="progress", fraction=round(0.1 + 0.85 * i / len(kinds), 3),
             message=f"Preparando os kernels de render ({kind}) — na primeira vez pode levar alguns minutos")
        res = {"dispositivo": dev, "gpus": list(DEVICE_INFO.get("gpus") or []) if dev != "CPU" else [],
               **denoiser_info(sc, dev)}
        try:
            t = time.perf_counter()
            renders = [_warm_render(sc, out / f"{kind.lower()}_{j}.exr", mb)
                       for j, mb in enumerate((False, True, True))]
            res.update(ok=True, parede_s=round(time.perf_counter() - t, 3), renders=renders,
                       kernels_s=round(sum(float(r.get("kernels_s") or 0.0) for r in renders), 3),
                       aquecido_s=renders[-1].get("parede_s"))
        except Exception as e:  # um backend com problema não impede o outro
            res.update(ok=False, erro=f"{e.__class__.__name__}: {e}")
            emit(event="log", level="warning", message=f"Aquecimento {kind} falhou: {e}")
        results.append(res)
    result = {"ok": any(r.get("ok") for r in results), "blender": bpy.app.version_string,
              "tamanho_px": WARMUP_PX, "amostras": WARMUP_SAMPLES, "dispositivos": results,
              "parede_s": round(time.perf_counter() - t0, 3)}
    (out / "aquecimento.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit(event="progress", fraction=1.0, message="Kernels de render prontos")
    emit(event="warmup_done", resultado=result)
    return result


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--aquecimento":
        warmup_main(sys.argv[2])
    else:
        main(sys.argv[1])
