"""Detecção de sol direto a partir das manchas de sol no piso.

Fotos de imóveis frequentemente têm "manchas" de sol no piso: a projeção paralela
da parte transparente das janelas na direção do sol. Procuramos a direção d
(azimute, elevação) que faz a união das janelas projetadas no piso coincidir com
a união das manchas detectadas (IoU dentro da região visível do piso).

Só aceitamos a hipótese com sobreposição alta; caso contrário o render não usa
sol, porque uma sombra nítida na direção errada é pior que nenhuma.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .room import RoomModel
from .segmentation import SegmentationResult


@dataclass
class SunEstimate:
    direction: list[float]          # vetor unitário apontando PARA o sol (referencial da sala)
    elevation_deg: float
    azimuth_deg: float
    iou: float                      # coeficiente de Dice janelas projetadas × manchas (região visível)
    patches: int
    windows: list[int]
    excess_ratio: float             # luminância na mancha / luminância do piso sem sol
    patch_area_m2: float
    precision: float = 0.0          # fração da área das manchas explicada pelas janelas projetadas
    recall: float = 0.0             # fração da projeção visível que de fato está iluminada

    def to_dict(self) -> dict:
        return asdict(self)


def _linear_lum(rgb8: np.ndarray) -> np.ndarray:
    lin = rgb8.astype(np.float32) / 255.0
    lin = np.where(lin <= 0.04045, lin / 12.92, ((lin + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def _floor_points(mask: np.ndarray, rays: np.ndarray, cam: np.ndarray, step: int = 1) -> np.ndarray:
    r = rays[mask][::step]
    t = -cam[2] / np.where(np.abs(r[:, 2]) > 1e-6, r[:, 2], -1e-6)
    P = cam + r * t[:, None]
    return P[(t > 0) & (t < 40)][:, :2]


def _clear_window_rects(room: RoomModel, seg: SegmentationResult, rays: np.ndarray, cam: np.ndarray):
    """Retângulo transparente de cada janela (exclui a área coberta por persiana/cortina)."""
    glass = seg.mask("window")
    out = []
    for i, o in enumerate(room.openings):
        if o.kind != "window":
            continue
        w = room.walls[o.wall]
        x0, y0, x1, y1 = o.pixel_bbox
        m = np.zeros_like(glass)
        m[y0:y1 + 1, x0:x1 + 1] = glass[y0:y1 + 1, x0:x1 + 1]
        z_top = o.z1
        if m.sum() > 50:
            n3 = np.array([w.normal[0], w.normal[1], 0.0])
            p0 = np.array([w.p0[0], w.p0[1], 0.0])
            r = rays[m]
            den = r @ n3
            ok = np.abs(den) > 1e-6
            t = ((p0 - cam) @ n3) / np.where(ok, den, 1.0)
            hit = cam + r * t[:, None]
            hit = hit[ok & (t > 0)]
            if len(hit) > 20:
                z_top = float(min(o.z1, np.percentile(hit[:, 2], 98)))
        base = np.asarray(w.p0, float)
        d = w.direction
        corners = []
        for s, z in ((o.s0, o.z0), (o.s1, o.z0), (o.s1, z_top), (o.s0, z_top)):
            xy = base + d * s
            corners.append([xy[0], xy[1], z])
        out.append((i, np.asarray(corners), np.asarray(w.normal, float)))
    return out


def detect_sun(room: RoomModel, seg: SegmentationResult, rgb8: np.ndarray, min_dice: float = 0.5,
               diag: dict | None = None) -> SunEstimate | None:
    """``diag`` (opcional) recebe os números do diagnóstico (manchas, Dice, precisão...) e, em
    ``patch_mask``, a máscara booleana das manchas de sol aceitas (tamanho da foto)."""
    diag = diag if diag is not None else {}
    from shapely.geometry import MultiPoint, Polygon
    from shapely.ops import unary_union

    H, W = rgb8.shape[:2]
    if seg.groups.shape != (H, W):
        return None
    floor = seg.mask("floor")
    if floor.sum() < 0.03 * H * W:
        return None
    lum = _linear_lum(rgb8)
    f = 4
    sl, sm = lum[::f, ::f], floor[::f, ::f]
    med = float(np.median(sl[sm]))
    base_s = ndimage.median_filter(np.where(sm, sl, med), size=max(9, min(sl.shape) // 6))
    base = np.kron(base_s, np.ones((f, f), np.float32))[:H, :W]
    cand = floor & (lum > np.maximum(1.45 * base, base + 0.05))
    cand = ndimage.binary_opening(cand, iterations=2)
    cand = ndimage.binary_closing(cand, iterations=3)
    lab, n = ndimage.label(cand)
    intr = room.intr()
    if intr.width != W:
        intr = intr.scaled(W, H)
    M = room.M
    rays = intr.rays() @ M[:3, :3].T
    cam = M[:3, 3]
    patches, ratios = [], []
    kept = np.zeros((H, W), bool)
    for k in range(1, n + 1):
        comp = lab == k
        if comp.mean() < 0.001:
            continue
        pts = _floor_points(comp, rays, cam, step=max(1, int(comp.sum() // 3000)))
        if len(pts) < 30:
            continue
        hull = MultiPoint([tuple(p) for p in pts]).convex_hull
        if hull.area < 0.02:
            continue
        patches.append(hull)
        kept |= comp
        ratios.append(float(np.median(lum[comp]) / max(1e-4, np.median(base[comp]))))
    diag["patches"] = len(patches)
    # máscara das manchas (pixels da foto): a calibração da luz ajusta o sol só nelas (PT8)
    diag["patch_mask"] = kept
    if not patches:
        return None
    P = unary_union(patches)
    vis_pts = _floor_points(floor, rays, cam, step=max(1, int(floor.sum() // 6000)))
    V = MultiPoint([tuple(p) for p in vis_pts]).convex_hull
    wins = _clear_window_rects(room, seg, rays, cam)
    if not wins:
        return None

    def score(az: float, el: float):
        d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        quads, used = [], []
        for wi, corners, nrm in wins:
            if d[:2] @ nrm >= -0.05:          # o sol precisa estar do lado de fora da janela
                continue
            t = corners[:, 2] / d[2]
            q = Polygon(corners[:, :2] - d[None, :2] * t[:, None])
            if q.is_valid and q.area > 1e-3:
                quads.append(q)
                used.append(wi)
        if not quads:
            return 0.0, d, used
        Q = unary_union(quads).intersection(V)
        inter = Q.intersection(P).area
        # F0,5: prioriza EXPLICAR as manchas (precisão) — robusto a caixilhos/travessas
        # (que dividem a mancha real em faixas) e evita soluções degeneradas de sol muito
        # baixo, cujas projeções longas cobrem tudo
        prec = inter / max(1e-9, P.area)
        rec = inter / max(1e-9, Q.area)
        return ((1.25 * prec * rec / (0.25 * prec + rec)) if inter > 0 else 0.0), d, used

    best = (0.0, None, [], 0.0, 0.0)
    for az in np.radians(np.arange(0, 360, 3)):
        for el in np.radians(np.arange(15, 81, 3)):
            s, d, used = score(az, el)
            if s > best[0]:
                best = (s, d, used, az, el)
    if best[1] is None:
        return None
    _, _, _, az0, el0 = best
    for az in az0 + np.radians(np.arange(-3, 3.01, 0.5)):
        for el in el0 + np.radians(np.arange(-3, 3.01, 0.5)):
            if not (np.radians(14) < el < np.radians(85)):
                continue
            s, d, used = score(az, el)
            if s > best[0]:
                best = (s, d, used, az, el)
    dice, d, used, az, el = best
    # métricas finais para o critério de aceitação
    quads = []
    for wi, corners, nrm in wins:
        if wi in used:
            t = corners[:, 2] / d[2]
            quads.append(Polygon(corners[:, :2] - d[None, :2] * t[:, None]))
    Q = unary_union(quads).intersection(V)
    inter = Q.intersection(P).area
    precision = inter / max(1e-9, P.area)        # quanto das manchas é explicado
    recall = inter / max(1e-9, Q.area)           # quanto da projeção cai em mancha
    diag.update(dice=round(float(dice), 3), precision=round(float(precision), 3), recall=round(float(recall), 3),
                azimuth_deg=round(float(np.degrees(az)) % 360, 1), elevation_deg=round(float(np.degrees(el)), 1),
                windows=sorted(used))
    if dice < min_dice or precision < 0.6 or recall < 0.3:
        return None
    iou = dice
    return SunEstimate(direction=d.tolist(), elevation_deg=float(np.degrees(el)),
                       azimuth_deg=float(np.degrees(az) % 360), iou=float(iou), patches=len(patches),
                       windows=sorted(used), excess_ratio=float(np.median(ratios)), patch_area_m2=float(P.area),
                       precision=float(precision), recall=float(recall))
