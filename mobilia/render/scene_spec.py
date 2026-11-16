"""Gera a especificação JSON da cena para o worker do Blender."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ..analysis.room import RoomModel
from ..analysis.segmentation import SegmentationResult
from ..assets.polyhaven import cached_texture, ensure_model, ensure_texture
from ..layout import catalog
from ..layout.model import Layout
from ..photo import srgb_to_linear
from . import choreography as ch
from .border import EXR_CODEC
from .layers import ADAPTIVE_MIN, ADAPTIVE_THRESHOLD, RAMP_FRAMES, clamp_layers


_RUG_DESIGNS = ("liso", "borda", "filete", "listras", "borda_dupla")    # = materials.RUG_DESIGNS
_RUG_PATTERN_MAP = {"geometrico": "berber", "berbere": "berber", "berber": "berber", "listras": "stripes",
                    "stripes": "stripes", "persa": "persian", "persian": "persian", "borda": "solid_border", "liso": "solid_border",
                    "solid_border": "solid_border"}


def _rgb(c):
    """Cor do layout (sRGB 0..1, 0..255 ou hex) -> lista sRGB 0..1."""
    if c is None:
        return None
    if isinstance(c, str):
        h = c.lstrip("#")
        return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    v = [float(x) for x in c][:3]
    return [x / 255.0 for x in v] if max(v) > 1.0 else v


def _hex(c) -> str | None:
    v = _rgb(c)
    return None if v is None else "#" + "".join(f"{int(round(min(1, max(0, x)) * 255)):02x}" for x in v)


def _metal_finish(c) -> str:
    v = _rgb(c)
    if v is None:
        return "black"
    r, g, b = v
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if r > b * 1.35 and lum > 0.25:
        return "brass"
    if lum > 0.55:
        return "chrome"
    return "matte_black" if lum < 0.12 else "black"


def normalize_procedural_params(kind: str, params: dict) -> dict:
    """Adapta os parâmetros semânticos do layout (paleta/estilo) ao vocabulário dos
    geradores procedurais. Valores desconhecidos viram padrões seguros; luminárias
    ficam SEMPRE apagadas (a iluminação da foto não pode mudar)."""
    p = dict(params)
    out: dict = {k: p[k] for k in ("size", "width", "depth", "height", "seed") if k in p}
    if kind == "rug":
        out["pattern"] = _RUG_PATTERN_MAP.get(str(p.get("pattern", "solid_border")).lower(), "solid_border")
        out["color"] = _hex(p.get("color"))
        out["accent"] = _hex(p.get("accent", p.get("pattern_color")))
        if "fringe" in p:
            out["fringe"] = bool(p["fringe"])
        tex = p.get("texture")
        if isinstance(tex, dict) and tex.get("diff") and tex.get("nor"):
            out["texture"] = {k: str(v) for k, v in tex.items() if k in ("diff", "nor", "rough") and v}
            for k in ("tile_m", "lum_mean", "sheen"):
                if p.get(k) is not None:
                    out[k] = float(p[k])
            design = str(p.get("design", "borda"))
            out["design"] = design if design in _RUG_DESIGNS else "borda"
    elif kind in ("floor_lamp", "table_lamp"):
        out["metal"] = p["metal"] if p.get("metal") in ("black", "brass", "chrome", "matte_black") \
            else _metal_finish(p.get("metal_color"))
        out["shade_color"] = _hex(p.get("shade_color"))
        if kind == "table_lamp":
            out["base_color"] = _hex(p.get("base_color"))
    elif kind == "bed":
        out["fabric_color"] = _hex(p.get("fabric_color", p.get("frame_color")))
        lin = _hex(p.get("duvet_color", p.get("linen_color")))
        out["duvet_color"] = lin
        out["sheet_color"] = _hex(p.get("sheet_color", p.get("linen_color")))
        if p.get("accent_color") is not None:
            out["pillow_colors"] = [lin, _hex(p["accent_color"])]
    elif kind == "desk":
        top = _rgb(p.get("top_color"))
        out["wood"] = "walnut" if top is not None and (0.2126 * top[0] + 0.7152 * top[1] + 0.0722 * top[2]) < 0.3 else "oak"
        leg = _rgb(p.get("leg_color"))
        out["legs"] = "black" if leg is not None and max(leg) < 0.35 else "wood"
    elif kind in ("throw_blanket", "throw"):
        out["color"] = _hex(p.get("color"))
        if p.get("accent") is not None:
            out["accent"] = _hex(p["accent"])
        pat = str(p.get("pattern", "stripes")).lower()
        out["pattern"] = "stripes" if pat in ("stripes", "listras") else "solid"
    elif kind == "book_stack":
        out["colors"] = [_hex(c) for c in p.get("colors") or [] if c is not None]
        if p.get("count") in (2, 3):
            out["count"] = int(p["count"])
        if p.get("page_color") is not None:
            out["page_color"] = _hex(p["page_color"])
    elif kind == "canvas_art":
        out["frame"] = p["frame"] if p.get("frame") in ("tela", "moldura") else "tela"
        out["colors"] = [_hex(c) for c in p.get("colors") or [] if c is not None]
        out["ground"] = _hex(p.get("ground"))
        out["art_style"] = str(p.get("art_style", "formas"))
        out["frame_color"] = _hex(p.get("frame_color"))
        out["mat_color"] = _hex(p.get("mat_color"))
        if "passe_partout" in p:
            out["passe_partout"] = bool(p["passe_partout"])
    elif kind == "curtain":
        out["color"] = _hex(p.get("color"))
        for k in ("sheer", "panels", "coverage"):
            if k in p:
                out[k] = p[k]
    return {k: v for k, v in out.items() if v is not None}


def rug_texture_paths(tex_id: str, assets_dir: Path, texture_res: str = "1k") -> dict | None:
    """Caminhos dos mapas da textura CC0 do tapete (do cache; baixa se faltar, com UMA tentativa de
    rede). Sem rede, usa a textura em outra resolução já no cache (ex.: só 1k instalado e render em
    2k). ``None`` quando não há como obtê-la: o tapete volta ao desenho procedural."""
    tex_id = str(tex_id)
    try:
        return {k: str(v) for k, v in ensure_texture(tex_id, assets_dir, texture_res, retries=1).items()}
    except Exception:
        pass
    for res in ("1k", "2k", "4k"):
        paths = cached_texture(tex_id, assets_dir, res) if res != texture_res else None
        if paths:
            return {k: str(v) for k, v in paths.items()}
    return None


def surface_stats(photo8: np.ndarray, seg: SegmentationResult) -> dict:
    """Cor média linear de piso/parede/teto e escala de albedo (parede clara ≈ 0,8)."""
    lin = srgb_to_linear(photo8.astype(np.float32) / 255.0)
    if seg.labels.shape != photo8.shape[:2]:
        groups = np.asarray(Image.fromarray(seg.groups).resize((photo8.shape[1], photo8.shape[0]), Image.NEAREST))
    else:
        groups = seg.groups
    from ..analysis.segmentation import GROUP_IDS

    means = {}
    for g in ("floor", "wall", "ceiling"):
        m = groups == GROUP_IDS[g]
        means[g] = np.median(lin[m], axis=0).tolist() if m.sum() > 100 else [0.4, 0.4, 0.4]
    wall = groups == GROUP_IDS["wall"]
    lum_w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    lum = lin @ lum_w
    ref = float(np.percentile(lum[wall], 90)) if wall.sum() > 100 else float(np.percentile(lum, 90))
    k = float(np.clip(0.8 / max(ref, 1e-3), 0.5, 3.0))
    # cor do iluminante (balanço de branco da foto): teto costuma ser branco; senão,
    # as paredes mais claras. Os móveis recebem a mesma "cor de luz" que a sala.
    ceil = groups == GROUP_IDS["ceiling"]
    src = ceil if ceil.sum() > 500 else (wall & (lum >= np.percentile(lum[wall], 70))) if wall.sum() > 100 else lum > 0
    illum = np.median(lin[src], axis=0)
    illum = illum / max(1e-6, float(illum @ lum_w))
    illum = 1.0 + 0.85 * (illum - 1.0)
    return {"albedo_mean": means, "albedo_scale": k, "illuminant": np.clip(illum, 0.6, 1.6).tolist()}


# pixels quase saturados da foto: só limite inferior na calibração (o valor real é ≥ o recortado)
SATURATED_LUM = 0.97
SATURATED_CHANNEL = 250 / 255


def calibration_mask(photo8: np.ndarray, seg: SegmentationResult, out_path: Path) -> Path:
    """Máscara de calibração da luz (PNG de 3 níveis, ver ``render/calibration.py``):

    - 255: piso/parede/teto (com erosão nas bordas), usado no ajuste por mínimos quadrados;
    - 128: o mesmo, mas quase saturado (luminância ≥ 0,97 ou um canal ≥ 250): entra como limite
      inferior (a sala renderizada tem de ser pelo menos tão clara), em vez de ser descartado;
    - 0: fora (outros objetos, bordas, pixels quase pretos)."""
    from ..analysis.segmentation import GROUP_IDS

    groups = seg.groups
    if groups.shape != photo8.shape[:2]:
        groups = np.asarray(Image.fromarray(groups).resize((photo8.shape[1], photo8.shape[0]), Image.NEAREST))
    m = np.isin(groups, [GROUP_IDS["floor"], GROUP_IDS["wall"], GROUP_IDS["ceiling"]])
    m = ndimage.binary_erosion(m, iterations=max(2, photo8.shape[1] // 200))
    rgb = photo8.astype(np.float32) / 255.0
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    m &= lum > 0.02                           # exclui pixels quase pretos (ruído, sem informação)
    sat = (lum >= SATURATED_LUM) | (rgb.max(-1) >= SATURATED_CHANNEL)
    out = np.where(m & ~sat, 255, np.where(m & sat, 128, 0)).astype(np.uint8)
    Image.fromarray(out).save(out_path)
    return out_path


def sun_patch_mask(mask: np.ndarray | None, shape: tuple[int, int], out_path: Path) -> Path | None:
    """Grava as manchas de sol da foto (``sunlight.detect_sun``) no tamanho da placa: a calibração
    ajusta a força do sol só nelas."""
    if mask is None or not np.asarray(mask).any():
        return None
    m = np.asarray(mask, bool)
    if m.shape != tuple(shape):
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) > 127
    Image.fromarray(m.astype(np.uint8) * 255).save(out_path)
    return out_path


def wall_extensions(walls: list[dict], ext: float = 0.35, step: float = 0.01,
                    tol: float = 0.01) -> list[list[float]]:
    """Quanto cada parede-proxy pode ser estendida além de cada ponta (fecha as quinas).

    A extensão só pode passar POR TRÁS das outras paredes. Se avançasse para o lado de dentro
    da sala diante de outra parede visível (quinas que não são de 90°), o proxy taparia os
    móveis encostados nela (ex.: um quadro cortado pela quina).
    """
    out = []
    for w in walls:
        p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
        L = float(np.linalg.norm(p1 - p0))
        d = (p1 - p0) / max(L, 1e-9)
        ends = []
        for base, sign in ((p0, -1.0), (p1, 1.0)):
            allowed = 0.0
            s = step
            while s <= ext + 1e-9:
                q = base + sign * d * s
                blocked = False
                for o in walls:
                    if o is w:
                        continue
                    a, b = np.asarray(o["p0"], float), np.asarray(o["p1"], float)
                    Lo = float(np.linalg.norm(b - a))
                    do = (b - a) / max(Lo, 1e-9)
                    t = float(np.dot(q - a, do))
                    in_front = float(np.dot(q - a, np.asarray(o["normal"], float))) > tol
                    if in_front and -tol <= t <= Lo + tol:
                        blocked = True
                        break
                if blocked:
                    break
                allowed = s
                s += step
            ends.append(round(allowed, 3))
        out.append(ends)
    return out


def item_px_per_m(position, size, yaw_deg: float, cam_from_room: np.ndarray, intr: dict,
                   safety: float = 1.15) -> float:
    """Maior ampliação (px por m de deslocamento) da projeção na caixa do item em repouso.

    Para um ponto da câmera (x, y, z), ‖∂(u, v)/∂p‖ = f/z · √(1 + (x/z)² + (y/z)²). O máximo
    nos cantos e no centro da caixa, com uma folga, é o que ``choreography`` usa para decidir
    quando o resto do assentamento fica abaixo de ``SNAP_PX``."""
    w, d, h = (float(v) for v in (list(size) + [0, 0, 0])[:3])
    c, s_ = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    x0, y0, z0 = (float(v) for v in position)
    pts = [[x0, y0, z0 + h / 2]]
    for lx in (-w / 2, w / 2):
        for ly in (-d / 2, d / 2):
            for z in (z0, z0 + h):
                pts.append([x0 + c * lx - s_ * ly, y0 + s_ * lx + c * ly, z])
    P = np.asarray(pts) @ cam_from_room[:3, :3].T + cam_from_room[:3, 3]
    f = max(float(intr["fx"]), float(intr["fy"]))
    if np.any(P[:, 2] <= 0.05):
        return round(ch.DEFAULT_PX_PER_M * safety, 1)
    z = np.maximum(P[:, 2], 0.3)
    gain = f / z * np.sqrt(1 + (P[:, 0] / z) ** 2 + (P[:, 1] / z) ** 2)
    return round(float(gain.max()) * safety, 1)


def build_scene_spec(room: RoomModel, layout: Layout, plate8: np.ndarray, plate_path: Path,
                     seg: SegmentationResult, work_dir: Path, assets_dir: Path, *, fps: int = 24,
                     samples: int = 64, samples_layers: dict | None = None, device: str = "AUTO",
                     texture_res: str = "1k",
                     hold_start: float = 0.8, hold_end: float = 2.5, motion_blur: bool = True,
                     seed: int = 7, calibrate: bool = True, threads: int = 0, sun: dict | None = None,
                     occluders: tuple | None = None, sun_mask: np.ndarray | None = None,
                     photo_emission: bool = True) -> dict:
    H, W = plate8.shape[:2]
    intr = room.intr().scaled(W, H)
    stats = surface_stats(plate8, seg)
    mask_path = calibration_mask(plate8, seg, work_dir / "calib_mask.png")
    sun_mask_path = sun_patch_mask(sun_mask, (H, W), work_dir / "calib_sun_mask.png") if sun else None
    x0, y0 = room.grid_origin
    ny, nx = room.grid_shape
    extent = [x0, y0, x0 + nx * room.grid_cell, y0 + ny * room.grid_cell]
    for w in room.walls:
        extent = [min(extent[0], w.p0[0], w.p1[0]), min(extent[1], w.p0[1], w.p1[1]),
                  max(extent[2], w.p0[0], w.p1[0]), max(extent[3], w.p0[1], w.p1[1])]

    raw_items = []
    cam_from_room = room.cam_from_room()
    for it in layout.items:
        spec_item = asdict(it)
        spec_item["px_per_m"] = item_px_per_m(it.position, it.size, it.yaw_deg, cam_from_room, intr.to_dict())
        try:
            a = catalog.get(it.asset_id)
            spec_item["yaw_fix"] = a.yaw_fix
        except KeyError:
            spec_item["yaw_fix"] = 0.0
        if it.asset_id.startswith("ph:"):
            gltf = ensure_model(it.asset_id[3:], assets_dir, texture_res)
            spec_item["source"] = {"type": "gltf", "path": str(gltf)}
        elif it.asset_id.startswith("proc:"):
            kind = it.asset_id[5:]
            params = dict(it.params)
            params.setdefault("size", list(it.size))
            if kind == "rug" and params.get("texture_id"):
                params["texture"] = rug_texture_paths(params["texture_id"], assets_dir, texture_res)
            spec_item["source"] = {"type": "procedural", "kind": kind,
                                   "params": normalize_procedural_params(kind, params)}
        else:
            raise ValueError(f"asset desconhecido: {it.asset_id}")
        raw_items.append(spec_item)
    occ_path = None
    if occluders is not None and len(occluders[1]):
        occ_path = work_dir / "occluders.npz"
        np.savez_compressed(occ_path, verts=occluders[0], faces=occluders[1])
    scheduled, end_anim = ch.schedule_items(raw_items, hold_start=hold_start)
    n_frames = int(math.ceil(ch.timeline_length(end_anim, hold_end) * fps))
    spec = {
        "version": 1,
        "photo_path": str(plate_path),
        "width": W, "height": H,
        "intrinsics": intr.to_dict(),
        "cam_to_room": room.cam_to_room,
        "room": {"walls": [dict(asdict(w), ext=e) for w, e in
                           zip(room.walls, wall_extensions([asdict(w) for w in room.walls]))],
                 "openings": [asdict(o) for o in room.openings],
                 "ceiling_height": room.ceiling_height, "extent": extent},
        "occluders": str(occ_path) if occ_path else None,
        "albedo_scale": stats["albedo_scale"],
        "albedo_mean": stats["albedo_mean"],
        "illuminant": stats["illuminant"],
        # ``photo_emission``: a sala-proxy emite a própria foto para os raios que não são da câmera
        # (luz rebatida real nos móveis); a calibração ajusta as luzes com ela como termo fixo
        "lighting": {"calibrate": calibrate, "calib_mask_path": str(mask_path),
                     "photo_emission": bool(photo_emission),
                     "window_color": stats["illuminant"], "world_color": stats["illuminant"],
                     "sun": ({"direction": sun["direction"], "angle_deg": 0.6,
                              "color": [float(c) for c in np.clip(np.asarray(stats["illuminant"]) * [1.04, 1.0, 0.94], 0.5, 1.6)],
                              "mask_path": str(sun_mask_path) if sun_mask_path else None}
                             if sun else None)},
        "items": scheduled,
        # ``samples`` = quadro final; ``samples_layers`` por camada (ver render/layers.py)
        "render": {"fps": fps, "n_frames": n_frames, "samples": int(samples),
                   "samples_layers": clamp_layers(samples_layers, samples),
                   "adaptive_threshold": ADAPTIVE_THRESHOLD, "adaptive_min_samples": dict(ADAPTIVE_MIN),
                   "sample_ramp_frames": RAMP_FRAMES, "device": device, "denoise": True,
                   "motion_blur": motion_blur, "shutter": 0.5, "seed": seed, "threads": threads,
                   "out_dir": str(work_dir / "render"), "border_margin_m": 0.6, "calib_width": 240,
                   # EXR de cada quadro: só o retângulo renderizado, compressão rápida (render/border.py)
                   "crop_to_border": True, "exr_codec": EXR_CODEC},
    }
    (work_dir / "scene.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    return spec
