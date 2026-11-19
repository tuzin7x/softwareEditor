"""Profundidade estrutural: a sala sem os móveis, para guiar a geração quando a disposição deve MUDAR.

A "Reforma" usava a profundidade da foto mobiliada: o ControlNet redesenhava o mesmo sofá no mesmo lugar (só com
outra cor), e uma reforma que mantém a disposição é só uma troca de estilo. Aqui os pixels de móveis (grupos
``furniture``/``fixture``/luminárias e o que a segmentação não reconheceu) saem do mapa e o buraco é preenchido com
o que está ATRÁS deles: as superfícies da caixa da sala.

Em disparidade (1/z) um plano é uma função LINEAR da posição na imagem (câmera pinhole: 1/z = a·u + b·v + c), então:

* piso e teto ganham um plano ajustado (mínimos quadrados robustos) nos pixels visíveis deles e estendido para
  baixo dos móveis;
* o resto (paredes, cantos) vem de uma interpolação harmônica a partir da borda do buraco — ela reproduz exatamente
  um plano quando a borda é um plano, e só arredonda os cantos (o ControlNet de profundidade é suave mesmo);
* dentro da sala (um volume convexo visto de dentro) a superfície visível é a MAIS PERTO entre as candidatas, ou
  seja, a de maior disparidade; e nada pode ficar mais perto que o próprio móvel que estava ali (a parede atrás do
  sofá está atrás do sofá): ``min(foto, max(planos, harmônica))``.

Tudo em resolução reduzida (o controle é de baixa frequência) e numpy/OpenCV puro (testável sem modelo).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..analysis.segmentation import GROUP_IDS, UNKNOWN

# o que sai do mapa estrutural (o resto — piso, parede, teto, janela, porta, cortina, estrutura — fica)
MOVABLE_GROUPS = ("furniture", "fixture", "ceiling_fixture")
WORK_SIDE = 192


def movable_mask(groups: np.ndarray, size: tuple[int, int] | None = None, unknown: bool = True,
                 grow_frac: float = 0.01, fixtures: bool = True) -> np.ndarray:
    """Máscara dos móveis/bancadas/luminárias (e, com ``unknown``, do que a segmentação não classificou),
    crescida ``grow_frac`` da diagonal (a sombra e a borda do móvel também saem). ``size`` = (largura, altura).
    ``fixtures=False``: armários, bancadas e eletrodomésticos ficam (a cozinha integrada de uma sala não some)."""
    g = np.asarray(groups)
    m = np.isin(g, [GROUP_IDS[k] for k in MOVABLE_GROUPS if fixtures or k != "fixture"])
    if unknown:
        m |= g == UNKNOWN
    if size is not None and (m.shape[1], m.shape[0]) != tuple(size):
        m = cv2.resize(m.astype(np.uint8), tuple(size), interpolation=cv2.INTER_NEAREST).astype(bool)
    r = int(round(grow_frac * np.hypot(*m.shape)))
    if r > 0 and m.any():
        m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    return m


def _harmonic_fill(v: np.ndarray, known: np.ndarray, iters: int = 400) -> np.ndarray:
    """Preenche ``v`` fora de ``known`` resolvendo Laplace (Jacobi com início piramidal). Plano na borda -> plano."""
    v = np.where(known, v, 0.0).astype(np.float32)
    if known.all() or not known.any():
        return v if known.any() else np.full(v.shape, float(np.nanmean(v)) if v.size else 0.0, np.float32)
    # início: média ponderada por pirâmide (push-pull), depois relaxação mantendo os conhecidos
    w = known.astype(np.float32)
    levels = []
    cv, cw = v * w, w
    while min(cv.shape) > 4:
        levels.append((cv, cw))
        cv = cv2.pyrDown(cv)
        cw = cv2.pyrDown(cw)
    est = np.where(cw > 1e-6, cv / np.maximum(cw, 1e-6), 0.0).astype(np.float32)
    fill_val = float((v * w).sum() / max(w.sum(), 1e-6))
    est[cw <= 1e-6] = fill_val
    for lv, lw in reversed(levels):
        up = cv2.resize(est, (lv.shape[1], lv.shape[0]), interpolation=cv2.INTER_LINEAR)
        cur = np.where(lw > 1e-6, lv / np.maximum(lw, 1e-6), up)
        a = np.clip(lw * 4.0, 0, 1)
        est = (a * cur + (1 - a) * up).astype(np.float32)
    out = np.where(known, v, est).astype(np.float32)
    k = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], np.float32)
    for _ in range(iters):
        nb = cv2.filter2D(out, -1, k, borderType=cv2.BORDER_REPLICATE)
        out = np.where(known, v, nb)
    return out


def fit_plane(disp: np.ndarray, sel: np.ndarray, min_px: int = 60, iters: int = 3) -> np.ndarray | None:
    """Plano em disparidade ``d = a·x + b·y + c`` (x, y normalizados em [0, 1]) ajustado nos pixels ``sel`` com
    descarte iterativo dos resíduos grandes. ``None`` se há poucos pixels ou o ajuste é ruim."""
    ys, xs = np.nonzero(sel)
    if len(xs) < min_px:
        return None
    H, W = sel.shape
    X = np.c_[xs / W, ys / H, np.ones(len(xs))]
    d = disp[ys, xs]
    keep = np.isfinite(d)
    coef = None
    for _ in range(iters):
        if keep.sum() < min_px:
            return None
        coef, *_ = np.linalg.lstsq(X[keep], d[keep], rcond=None)
        res = np.abs(X @ coef - d)
        thr = max(2.5 * float(np.median(res[keep])), 1e-6)
        keep = keep & (res <= thr)
    # ajuste ruim (a "superfície" não é um plano): não usa
    rel = float(np.median(np.abs(X[keep] @ coef - d[keep])) / max(float(np.median(np.abs(d[keep]))), 1e-6))
    return coef if rel < 0.08 else None


def structural_depth(depth: np.ndarray, hole: np.ndarray, groups: np.ndarray | None = None,
                     work_side: int = WORK_SIDE) -> np.ndarray:
    """Profundidade (z) com o buraco ``hole`` (móveis) trocado pelas superfícies da sala atrás dele.

    ``groups`` (opcional, qualquer resolução): habilita o plano do piso e do teto. Pixels inválidos da foto
    (z <= 0) continuam inválidos fora do buraco. Mesma resolução de ``depth``."""
    z = np.asarray(depth, np.float32)
    H, W = z.shape
    hole = np.asarray(hole, bool)
    if hole.shape != z.shape:
        hole = cv2.resize(hole.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    valid = np.isfinite(z) & (z > 1e-4)
    if not hole.any() or valid.sum() < 16:
        return z.copy()
    s = min(1.0, work_side / max(H, W))
    w, h = max(8, int(round(W * s))), max(8, int(round(H * s)))
    disp = np.zeros_like(z)
    disp[valid] = 1.0 / z[valid]
    ds = cv2.resize(disp, (w, h), interpolation=cv2.INTER_AREA)
    vs = cv2.resize(valid.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.99
    hs = cv2.resize(hole.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.01
    known = vs & ~hs
    if known.sum() < 16:
        return z.copy()
    planes = []
    planar = np.zeros_like(known)
    if groups is not None:
        g = cv2.resize(np.asarray(groups, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.mgrid[0:h, 0:w]
        X = np.stack([xs / w, ys / h, np.ones((h, w))], -1)
        for name in ("floor", "ceiling"):
            sel = known & (g == GROUP_IDS[name])
            coef = fit_plane(ds, sel)
            if coef is not None:
                planes.append((X @ coef).astype(np.float32))
                planar |= sel
    # paredes (e o que não tem plano): harmônica só a partir delas — o piso perto da câmera puxaria a parede
    # para a frente perto do canto; piso e teto entram pelos próprios planos (a superfície visível é a mais perto)
    kh = known & ~planar
    cand = _harmonic_fill(ds, kh if kh.sum() >= 16 else known)
    for pl in planes:
        cand = np.where(pl > 0, np.maximum(cand, pl), cand)
    # nada fica mais perto que o móvel que estava no lugar
    occl = np.where(vs, ds, np.inf)
    cand = np.minimum(cand, occl)
    lo = float(np.percentile(ds[known], 1)) if known.any() else 1e-3
    cand = np.clip(cand, max(lo * 0.8, 1e-4), None)
    up = cv2.resize(cand, (W, H), interpolation=cv2.INTER_LINEAR)
    up = cv2.GaussianBlur(up, (0, 0), max(1.0, 0.004 * np.hypot(H, W)))
    out = z.copy()
    out[hole] = 1.0 / np.maximum(up[hole], 1e-6)
    return out


__all__ = ["MOVABLE_GROUPS", "movable_mask", "structural_depth", "fit_plane"]
