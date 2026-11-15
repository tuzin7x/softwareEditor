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
from ..assets.polyhaven import ensure_model
from ..layout import catalog
from ..layout.model import Layout
from ..photo import srgb_to_linear
from . import choreography as ch


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


def calibration_mask(photo8: np.ndarray, seg: SegmentationResult, out_path: Path) -> Path:
    from ..analysis.segmentation import GROUP_IDS

    groups = seg.groups
    if groups.shape != photo8.shape[:2]:
        groups = np.asarray(Image.fromarray(groups).resize((photo8.shape[1], photo8.shape[0]), Image.NEAREST))
    m = np.isin(groups, [GROUP_IDS["floor"], GROUP_IDS["wall"], GROUP_IDS["ceiling"]])
    m = ndimage.binary_erosion(m, iterations=max(2, photo8.shape[1] // 200))
    lum = (photo8.astype(np.float32) / 255.0) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    m &= (lum < 0.97) & (lum > 0.02)          # exclui pixels estourados/pretos
    Image.fromarray((m * 255).astype(np.uint8)).save(out_path)
    return out_path


def build_scene_spec(room: RoomModel, layout: Layout, plate8: np.ndarray, plate_path: Path,
                     seg: SegmentationResult, work_dir: Path, assets_dir: Path, *, fps: int = 30,
                     samples: int = 64, device: str = "AUTO", texture_res: str = "1k",
                     hold_start: float = 0.8, hold_end: float = 2.5, motion_blur: bool = True,
                     seed: int = 7, calibrate: bool = True, threads: int = 0) -> dict:
    H, W = plate8.shape[:2]
    intr = room.intr().scaled(W, H)
    stats = surface_stats(plate8, seg)
    mask_path = calibration_mask(plate8, seg, work_dir / "calib_mask.png")
    x0, y0 = room.grid_origin
    ny, nx = room.grid_shape
    extent = [x0, y0, x0 + nx * room.grid_cell, y0 + ny * room.grid_cell]
    for w in room.walls:
        extent = [min(extent[0], w.p0[0], w.p1[0]), min(extent[1], w.p0[1], w.p1[1]),
                  max(extent[2], w.p0[0], w.p1[0]), max(extent[3], w.p0[1], w.p1[1])]

    raw_items = []
    for it in layout.items:
        spec_item = asdict(it)
        try:
            a = catalog.get(it.asset_id)
            spec_item["yaw_fix"] = a.yaw_fix
        except KeyError:
            spec_item["yaw_fix"] = 0.0
        if it.asset_id.startswith("ph:"):
            gltf = ensure_model(it.asset_id[3:], assets_dir, texture_res)
            spec_item["source"] = {"type": "gltf", "path": str(gltf)}
        elif it.asset_id.startswith("proc:"):
            params = dict(it.params)
            params.setdefault("size", list(it.size))
            spec_item["source"] = {"type": "procedural", "kind": it.asset_id[5:], "params": params}
        else:
            raise ValueError(f"asset desconhecido: {it.asset_id}")
        raw_items.append(spec_item)
    scheduled, end_anim = ch.schedule_items(raw_items, hold_start=hold_start)
    n_frames = int(math.ceil(ch.timeline_length(end_anim, hold_end) * fps))
    spec = {
        "version": 1,
        "photo_path": str(plate_path),
        "width": W, "height": H,
        "intrinsics": intr.to_dict(),
        "cam_to_room": room.cam_to_room,
        "room": {"walls": [asdict(w) for w in room.walls], "openings": [asdict(o) for o in room.openings],
                 "ceiling_height": room.ceiling_height, "extent": extent},
        "albedo_scale": stats["albedo_scale"],
        "albedo_mean": stats["albedo_mean"],
        "illuminant": stats["illuminant"],
        "lighting": {"calibrate": calibrate, "calib_mask_path": str(mask_path),
                     "window_color": stats["illuminant"], "world_color": stats["illuminant"]},
        "items": scheduled,
        "render": {"fps": fps, "n_frames": n_frames, "samples": samples, "device": device, "denoise": True,
                   "motion_blur": motion_blur, "shutter": 0.5, "seed": seed, "threads": threads,
                   "out_dir": str(work_dir / "render"), "border_margin_m": 0.6, "calib_width": 240},
    }
    (work_dir / "scene.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    return spec
