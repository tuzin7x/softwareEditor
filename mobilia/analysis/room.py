"""Reconstrução do modelo da sala a partir da geometria métrica + segmentação.

Referenciais:
- câmera (OpenCV): x direita, y baixo, z frente — saída do MoGe;
- sala (igual ao Blender): Z para cima, piso em Z = 0, câmera em (0, 0, h),
  eixo Y alinhado à direção dominante das paredes (mundo de Manhattan).

Saídas: plano do piso, altura da câmera, paredes (segmentos 2D com extensão),
teto, janelas/portas (retângulos nas paredes) e grade de ocupação do piso.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from .geometry import GeometryResult, Intrinsics
from .segmentation import SegmentationResult

# Estados da grade de ocupação (vista de cima, células de CELL m)
UNKNOWN, FREE, BLOCKED, OUTSIDE = 0, 1, 2, 3
CELL = 0.05


@dataclass
class Wall:
    id: int
    p0: list[float]          # extremidades no piso (x, y) em metros, referencial da sala
    p1: list[float]
    normal: list[float]      # normal 2D apontando para dentro da sala
    height: float
    support: int             # nº de pixels que sustentam a parede
    rms: float               # resíduo do ajuste (m)

    @property
    def direction(self) -> np.ndarray:
        d = np.asarray(self.p1) - np.asarray(self.p0)
        return d / (np.linalg.norm(d) + 1e-12)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.asarray(self.p1) - np.asarray(self.p0)))

    def signed_distance(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy) - np.asarray(self.p0)) @ np.asarray(self.normal)

    def along(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy) - np.asarray(self.p0)) @ self.direction


@dataclass
class Opening:
    kind: str                # "window" | "door"
    wall: int
    s0: float                # posição ao longo da parede (m, a partir de p0)
    s1: float
    z0: float                # altura inferior / superior (m)
    z1: float
    pixel_bbox: list[int]    # x0, y0, x1, y1
    mean_rgb: list[float]    # cor média (sRGB 0..1) — usada para a luz da janela
    luminance: float


@dataclass
class RoomModel:
    intrinsics: dict
    cam_to_room: list[list[float]]   # 4x4: p_room = M @ [p_cam, 1]
    camera_height: float
    ceiling_height: float
    ceiling_observed: bool
    walls: list[Wall]
    openings: list[Opening]
    grid_origin: list[float]         # (x, y) do canto da célula [0, 0]
    grid_cell: float
    grid_shape: list[int]            # (ny, nx)
    scale_factor: float = 1.0
    quality: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(s: str) -> "RoomModel":
        d = json.loads(s)
        d["walls"] = [Wall(**w) for w in d["walls"]]
        d["openings"] = [Opening(**o) for o in d["openings"]]
        return RoomModel(**d)

    @property
    def M(self) -> np.ndarray:
        return np.asarray(self.cam_to_room, dtype=np.float64)

    def cam_from_room(self) -> np.ndarray:
        return np.linalg.inv(self.M)

    def intr(self) -> Intrinsics:
        return Intrinsics(**self.intrinsics)

    def project_room_points(self, pts_room: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Projeta pontos da sala (N,3) para pixels (N,2). Retorna também z da câmera."""
        Minv = self.cam_from_room()
        p = pts_room @ Minv[:3, :3].T + Minv[:3, 3]
        z = p[:, 2]
        uv = self.intr().project(p)
        return uv, z


# ----------------------------------------------------------------------------- ajuste de planos

def fit_plane_ransac(pts: np.ndarray, iters: int = 400, thresh: float = 0.02,
                     rng: np.random.Generator | None = None,
                     normal_prior: np.ndarray | None = None, prior_cos: float = 0.8):
    """RANSAC + mínimos quadrados. Retorna (n, d, inliers) com n·x + d = 0."""
    rng = rng or np.random.default_rng(0)
    n_pts = len(pts)
    if n_pts < 3:
        raise ValueError("pontos insuficientes para ajustar plano")
    best = (None, None, np.zeros(n_pts, bool), -1)
    for _ in range(iters):
        i = rng.choice(n_pts, 3, replace=False)
        a, b, c = pts[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if normal_prior is not None and abs(n @ normal_prior) < prior_cos:
            continue
        d = -n @ a
        inl = np.abs(pts @ n + d) < thresh
        score = inl.sum()
        if score > best[3]:
            best = (n, d, inl, score)
    n, d, inl, _ = best
    if n is None:
        raise ValueError("RANSAC não encontrou plano")
    # refino por PCA nos inliers
    P = pts[inl]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n2 = vt[-1]
    if n2 @ n < 0:
        n2 = -n2
    d2 = -n2 @ c
    inl2 = np.abs(pts @ n2 + d2) < thresh
    return n2, float(d2), inl2


def fit_lines_2d(xy: np.ndarray, nxy: np.ndarray, weights_dist: np.ndarray, max_lines: int = 8,
                 min_support: int = 400, iters: int = 600, rng: np.random.Generator | None = None):
    """RANSAC sequencial de retas 2D (paredes vistas de cima) com consistência de normais.

    xy: pontos (N,2); nxy: normais horizontais unitárias (N,2) (podem ser 0 se inválidas);
    weights_dist: distância à câmera (N,) para tolerância proporcional.
    """
    rng = rng or np.random.default_rng(1)
    remaining = np.ones(len(xy), bool)
    lines = []
    for _ in range(max_lines):
        idx = np.flatnonzero(remaining)
        if len(idx) < min_support:
            break
        best = None
        P, N, Dd = xy[idx], nxy[idx], weights_dist[idx]
        tol = 0.03 + 0.012 * Dd
        for _ in range(iters):
            i, j = rng.choice(len(idx), 2, replace=False)
            d = P[j] - P[i]
            L = np.linalg.norm(d)
            if L < 0.3:
                continue
            d /= L
            nrm = np.array([-d[1], d[0]])
            dist = (P - P[i]) @ nrm
            cons = np.abs(N @ nrm) > 0.85            # normal do ponto ~ normal da reta
            inl = (np.abs(dist) < tol) & cons
            s = int(inl.sum())
            if best is None or s > best[0]:
                best = (s, nrm, P[i].copy(), inl)
        if best is None or best[0] < min_support:
            break
        s, nrm, p0, inl = best
        # refino PCA
        Q = P[inl]
        c = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        dvec = vt[0]
        nrm = np.array([-dvec[1], dvec[0]])
        dist = (P - c) @ nrm
        inl = (np.abs(dist) < tol) & (np.abs(N @ nrm) > 0.85)
        Q = P[inl]
        rms = float(np.sqrt(np.mean(((Q - c) @ nrm) ** 2))) if len(Q) else 0.0
        lines.append(dict(center=c, direction=dvec, normal=nrm, inliers=idx[inl], rms=rms))
        remaining[idx[inl]] = False
    return lines


def _circular_mean_mod90(angles: np.ndarray, weights: np.ndarray) -> float:
    a = angles * 4.0
    c = np.sum(weights * np.cos(a)); s = np.sum(weights * np.sin(a))
    return float(np.arctan2(s, c) / 4.0)


# ----------------------------------------------------------------------------- construção do modelo

def build_room_model(geo: GeometryResult, seg: SegmentationResult, photo_rgb8: np.ndarray,
                     max_points: int = 60000, seed: int = 0) -> tuple[RoomModel, dict]:
    """Constrói o modelo da sala. Retorna (modelo, artefatos de depuração)."""
    rng = np.random.default_rng(seed)
    H, W = geo.depth.shape
    intr = geo.intrinsics
    valid = geo.mask & np.isfinite(geo.depth) & (geo.depth > 0.1)
    floor_m = seg.mask("floor") & valid
    wall_m = seg.mask("wall") & valid
    ceil_m = seg.mask("ceiling") & valid
    debug: dict = {}

    # ---- 1. piso -------------------------------------------------------------------------
    n_cam_up_prior = np.array([0.0, -1.0, 0.0])  # "para cima" na câmera OpenCV (sem pitch)
    floor_pts = geo.points[floor_m]
    if len(floor_pts) < 500:
        # fallback: pixels com normal quase vertical na metade inferior da imagem
        lower = np.zeros_like(valid); lower[H // 2:] = True
        cand = valid & lower & ((geo.normals @ n_cam_up_prior) > 0.85)
        floor_pts = geo.points[cand]
        debug["floor_fallback"] = True
    if len(floor_pts) < 200:
        raise RuntimeError("Não foi possível identificar o piso na foto.")
    sub = floor_pts[rng.choice(len(floor_pts), min(len(floor_pts), max_points), replace=False)]
    thr = 0.015 + 0.01 * float(np.median(np.linalg.norm(sub, axis=1)))
    n_f, d_f, inl = fit_plane_ransac(sub, thresh=thr, rng=rng, normal_prior=n_cam_up_prior, prior_cos=0.5)
    if d_f < 0:  # orienta: câmera (origem) do lado positivo -> d = altura da câmera
        n_f, d_f = -n_f, -d_f
    cam_h = float(d_f)
    up = n_f
    floor_inlier_frac = float(inl.mean())

    # ---- 2. referencial da sala ------------------------------------------------------------
    fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    R = np.stack([right, fwd, up])                      # linhas = eixos da sala na câmera
    pts_room = geo.points @ R.T + np.array([0, 0, cam_h])
    nrm_room = geo.normals @ R.T

    # ---- 3. paredes (retas 2D) ---------------------------------------------------------------
    wall_idx = np.flatnonzero(wall_m.ravel())
    if len(wall_idx) > max_points:
        wall_idx = rng.choice(wall_idx, max_points, replace=False)
    WP = pts_room.reshape(-1, 3)[wall_idx]
    WN = nrm_room.reshape(-1, 3)[wall_idx][:, :2]
    nlen = np.linalg.norm(WN, axis=1, keepdims=True)
    WN = np.where(nlen > 0.5, WN / np.maximum(nlen, 1e-9), 0.0)
    keep = (WP[:, 2] > 0.05) & (WP[:, 2] < 4.0)
    WP, WN = WP[keep], WN[keep]
    dist_cam = np.linalg.norm(WP[:, :2], axis=1)
    lines = fit_lines_2d(WP[:, :2], WN, dist_cam, min_support=max(150, int(0.02 * len(WP))), rng=rng)

    # yaw de Manhattan: alinha as paredes principais aos eixos X/Y
    yaw = 0.0
    if lines:
        ang = np.array([np.arctan2(l["direction"][1], l["direction"][0]) for l in lines])
        wts = np.array([len(l["inliers"]) for l in lines], float)
        yaw = _circular_mean_mod90(ang, wts)
    cz, sz = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # escolhe, entre as 4 rotações de 90°, a que deixa a câmera olhando ~ +Y
    best_rot, best_dot = Rz, -9
    for k in range(4):
        c4, s4 = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        R4 = np.array([[c4, -s4, 0], [s4, c4, 0], [0, 0, 1]]) @ Rz
        f_room = R4 @ np.array([0.0, 1.0, 0.0])
        if f_room[1] > best_dot:
            best_dot, best_rot = f_room[1], R4
    Rz = best_rot
    R_full = Rz @ R
    M = np.eye(4); M[:3, :3] = R_full; M[:3, 3] = [0, 0, cam_h]
    pts_room = geo.points @ R_full.T + np.array([0, 0, cam_h])
    nrm_room = geo.normals @ R_full.T
    WP2 = WP @ Rz.T

    walls: list[Wall] = []
    for i, l in enumerate(lines):
        c = Rz[:2, :2] @ l["center"]; dvec = Rz[:2, :2] @ l["direction"]; nrm = Rz[:2, :2] @ l["normal"]
        # snap para o eixo mais próximo se o desvio for < 6°
        for axis in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
            if abs(dvec @ axis) > np.cos(np.radians(6)):
                dvec = axis * np.sign(dvec @ axis); nrm = np.array([-dvec[1], dvec[0]])
                break
        if (np.zeros(2) - c) @ nrm < 0:      # normal aponta para a câmera (interior)
            nrm = -nrm
        inl_xy = WP2[l["inliers"], :2]
        inl_z = WP2[l["inliers"], 2]
        s = (inl_xy - c) @ dvec
        s0, s1 = np.percentile(s, 1), np.percentile(s, 99)
        off = (inl_xy - c) @ nrm
        c = c + nrm * float(np.median(off))
        walls.append(Wall(i, (c + dvec * s0).tolist(), (c + dvec * s1).tolist(), nrm.tolist(),
                          float(np.percentile(inl_z, 99)), int(len(l["inliers"])), l["rms"]))
    walls = _merge_and_trim_walls(walls)

    # ---- 4. teto -----------------------------------------------------------------------------
    ceil_frac = float(ceil_m.mean())
    if ceil_m.sum() > 500:
        cz_vals = pts_room[ceil_m][:, 2]
        ceiling_h = float(np.median(cz_vals))
        ceiling_obs = True
    else:
        ceiling_h = max([w.height for w in walls] + [2.5])
        ceiling_obs = False
    for w in walls:
        w.height = ceiling_h

    # ---- 5. aberturas (janelas e portas) -----------------------------------------------------
    rays_room = intr.rays() @ R_full.T       # direções no referencial da sala, origem na câmera
    cam_pos = np.array([0, 0, cam_h])
    openings: list[Opening] = []
    for kind, mask in (("window", seg.mask("window") | seg.mask("window_covering")), ("door", seg.mask("door"))):
        lab, n = ndimage.label(ndimage.binary_opening(mask, iterations=2))
        for k in range(1, n + 1):
            comp = lab == k
            area = comp.mean()
            if area < 0.004:
                continue
            ring = ndimage.binary_dilation(comp, iterations=12) & ~comp & wall_m
            wall_id = _wall_for_points(pts_room[ring][:, :2], walls) if ring.sum() > 30 else None
            if wall_id is None:
                wall_id = _wall_for_rays(rays_room[comp], cam_pos, walls)
            if wall_id is None:
                continue
            wl = walls[wall_id]
            hit = _intersect_wall(rays_room[comp], cam_pos, wl)
            if hit is None or len(hit) < 20:
                continue
            s = wl.along(hit[:, :2]); z = hit[:, 2]
            ys, xs = np.nonzero(comp)
            rgb = photo_rgb8[comp].astype(np.float32).mean(0) / 255.0
            lum = float(rgb @ np.array([0.2126, 0.7152, 0.0722]))
            o = Opening(kind, wall_id, float(np.percentile(s, 1)), float(np.percentile(s, 99)),
                        float(max(0.0, np.percentile(z, 1))), float(min(ceiling_h, np.percentile(z, 99))),
                        [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], rgb.tolist(), lum)
            if kind == "door":
                o.z0 = 0.0
            openings.append(o)

    # ---- 6. pistas de escala ---------------------------------------------------------------------
    cues = {}
    doors = [o for o in openings if o.kind == "door" and o.z1 < ceiling_h - 0.05]
    if doors:
        cues["door_height"] = float(np.median([o.z1 - o.z0 for o in doors]))
    if ceiling_obs:
        cues["ceiling_height"] = ceiling_h
    cues["camera_height"] = cam_h
    scale = _scale_from_cues(cues)

    # ---- 7. grade de ocupação ----------------------------------------------------------------------
    grid, origin = _occupancy_grid(pts_room, rays_room, cam_pos, floor_m, seg, valid, walls)

    model = RoomModel(
        intrinsics=intr.to_dict(), cam_to_room=M.tolist(), camera_height=cam_h,
        ceiling_height=ceiling_h, ceiling_observed=ceiling_obs, walls=walls, openings=openings,
        grid_origin=origin.tolist(), grid_cell=CELL, grid_shape=list(grid.shape), scale_factor=1.0,
        quality=dict(floor_inlier_frac=floor_inlier_frac, floor_pixels=int(floor_m.sum()),
                     wall_pixels=int(wall_m.sum()), ceiling_frac=ceil_frac, n_walls=len(walls),
                     wall_rms=[w.rms for w in walls], fov_x_deg=intr.fov_x_deg,
                     manhattan_yaw_deg=float(np.degrees(yaw)), scale_cues=cues,
                     suggested_scale=scale),
    )
    if abs(scale - 1.0) > 0.03:
        model = rescale_room(model, scale)
        grid, origin = _occupancy_grid(pts_room * np.array([scale, scale, scale]) + np.array([0, 0, 0]),
                                       rays_room, cam_pos * scale, floor_m, seg, valid, model.walls)
        model.grid_origin = origin.tolist(); model.grid_shape = list(grid.shape)
    debug.update(grid=grid, pts_room=pts_room * model.scale_factor, nrm_room=nrm_room)
    return model, debug


def _merge_and_trim_walls(walls: list[Wall]) -> list[Wall]:
    """Funde paredes quase coincidentes e descarta segmentos muito curtos."""
    out: list[Wall] = []
    for w in sorted(walls, key=lambda w: -w.support):
        dup = False
        for o in out:
            if abs(np.dot(w.normal, o.normal)) > np.cos(np.radians(8)):
                if abs(o.signed_distance(np.asarray(w.p0)[None])[0]) < 0.12:
                    # estende o segmento existente
                    s = [o.along(np.asarray(p)[None])[0] for p in (w.p0, w.p1, o.p0, o.p1)]
                    d = o.direction; base = np.asarray(o.p0)
                    o.p0 = (base + d * min(s)).tolist(); o.p1 = (base + d * max(s)).tolist()
                    o.support += w.support
                    dup = True
                    break
        if not dup and w.length > 0.4:
            out.append(w)
    for i, w in enumerate(out):
        w.id = i
    return out


def _wall_for_points(xy: np.ndarray, walls: list[Wall]) -> int | None:
    if not walls or len(xy) == 0:
        return None
    best, best_score = None, 0
    for w in walls:
        d = np.abs(w.signed_distance(xy))
        s = w.along(xy)
        score = np.sum((d < 0.15) & (s > -0.3) & (s < w.length + 0.3))
        if score > best_score:
            best, best_score = w.id, score
    return best if best_score > 0.3 * len(xy) else None


def _intersect_wall(rays: np.ndarray, origin: np.ndarray, w: Wall) -> np.ndarray | None:
    n3 = np.array([w.normal[0], w.normal[1], 0.0])
    p0 = np.array([w.p0[0], w.p0[1], 0.0])
    denom = rays @ n3
    ok = np.abs(denom) > 1e-6
    t = np.full(len(rays), -1.0)
    t[ok] = ((p0 - origin) @ n3) / denom[ok]
    good = t > 0
    if good.sum() == 0:
        return None
    return origin + rays[good] * t[good, None]


def _wall_for_rays(rays: np.ndarray, origin: np.ndarray, walls: list[Wall]) -> int | None:
    best, best_t = None, np.inf
    center_ray = rays.mean(0)
    for w in walls:
        hit = _intersect_wall(center_ray[None], origin, w)
        if hit is None:
            continue
        s = w.along(hit[:, :2])[0]
        if -0.3 < s < w.length + 0.3:
            t = np.linalg.norm(hit[0] - origin)
            if t < best_t:
                best, best_t = w.id, t
    return best


def _scale_from_cues(cues: dict) -> float:
    """Combina pistas métricas (porta ~2,05 m; pé-direito plausível) num fator de escala.

    Só corrige quando há evidência forte; caso contrário mantém a escala do MoGe-2.
    """
    est = []
    if "door_height" in cues and 1.4 < cues["door_height"] < 2.8:
        est.append(2.05 / cues["door_height"])
    if "ceiling_height" in cues:
        c = cues["ceiling_height"]
        if c < 2.25:
            est.append(2.45 / c)
        elif c > 3.6:
            est.append(3.2 / c)
    if not est:
        return 1.0
    return float(np.clip(np.median(est), 0.75, 1.33))


def rescale_room(m: RoomModel, s: float) -> RoomModel:
    """Aplica um fator de escala métrica uniforme (intrínsecos não mudam)."""
    M = m.M.copy(); M[:3, 3] *= s
    walls = [Wall(w.id, (np.asarray(w.p0) * s).tolist(), (np.asarray(w.p1) * s).tolist(), w.normal,
                  w.height * s, w.support, w.rms * s) for w in m.walls]
    ops = [Opening(o.kind, o.wall, o.s0 * s, o.s1 * s, o.z0 * s, o.z1 * s, o.pixel_bbox, o.mean_rgb,
                   o.luminance) for o in m.openings]
    return RoomModel(m.intrinsics, M.tolist(), m.camera_height * s, m.ceiling_height * s,
                     m.ceiling_observed, walls, ops, m.grid_origin, m.grid_cell, m.grid_shape,
                     m.scale_factor * s, dict(m.quality, scale_applied=s))


def _occupancy_grid(pts_room, rays_room, cam_pos, floor_m, seg, valid, walls):
    """Grade 2D do piso: FREE (piso visível), BLOCKED (obstáculos), OUTSIDE (além das paredes)."""
    H, W = floor_m.shape
    # piso: interseção exata raio x plano Z=0 (mais estável que os pontos do MoGe)
    r = rays_room[floor_m]
    t = -cam_pos[2] / np.where(np.abs(r[:, 2]) > 1e-6, r[:, 2], -1e-6)
    fp = cam_pos + r * t[:, None]
    fp = fp[(t > 0) & (t < 30)]
    obst_m = (seg.mask("structure") | seg.mask("fixture") | seg.mask("furniture")) & valid
    op = pts_room[obst_m]
    op = op[(op[:, 2] < 2.2) & (op[:, 2] > -0.1)]
    allxy = [fp[:, :2]] + ([op[:, :2]] if len(op) else []) + [np.array([[0.0, 0.0]])]
    for w in walls:
        allxy.append(np.array([w.p0, w.p1]))
    allxy = np.concatenate(allxy)
    lo = np.percentile(allxy, 0.5, axis=0) - 0.5
    hi = np.percentile(allxy, 99.5, axis=0) + 0.5
    lo = np.maximum(lo, -15); hi = np.minimum(hi, 15)
    nx = int(np.ceil((hi[0] - lo[0]) / CELL)); ny = int(np.ceil((hi[1] - lo[1]) / CELL))
    grid = np.full((ny, nx), UNKNOWN, np.uint8)

    def cells(xy):
        ij = np.floor((xy - lo) / CELL).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
        return ij[ok]

    c = cells(fp[:, :2]); grid[c[:, 1], c[:, 0]] = FREE
    # fecha pequenos buracos do piso visível
    free = ndimage.binary_closing(grid == FREE, iterations=2)
    grid[free & (grid == UNKNOWN)] = FREE
    if len(op):
        c = cells(op[:, :2])
        blk = np.zeros_like(grid, bool); blk[c[:, 1], c[:, 0]] = True
        blk = ndimage.binary_dilation(blk, iterations=2)
        grid[blk] = BLOCKED
    # fora das paredes
    yy, xx = np.mgrid[0:ny, 0:nx]
    centers = np.stack([lo[0] + (xx + 0.5) * CELL, lo[1] + (yy + 0.5) * CELL], -1).reshape(-1, 2)
    for w in walls:
        sd = w.signed_distance(centers)
        s = w.along(centers)
        out = (sd < -0.02) & (s > -0.2) & (s < w.length + 0.2)
        grid.reshape(-1)[out] = OUTSIDE
    return grid, lo
