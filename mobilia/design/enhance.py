"""'Melhorar foto': edição de foto de anúncio (sem redesenho, a arquitetura nunca muda).

Etapas (cada uma pode ser desligada; todas ligadas por padrão):

* **Remover objetos** (``mask``): o usuário pinta o que deve sumir (fio, lixo, objeto pessoal) e o removedor de
  objetos (big-lama, já usado no modo B) refaz o fundo, num recorte em volta de cada área pintada;
* **Redução de ruído** leve (sobretudo do ruído colorido, o que mais aparece em foto de celular);
* **Verticais** (``verticals``): as linhas quase verticais (batentes, cantos de parede, armários) são encontradas
  (LSD), o ponto de fuga vertical sai por RANSAC e a câmera é "girada" (homografia K·R·K⁻¹) até elas ficarem
  verticais; a imagem é recortada na maior área válida com a mesma proporção. Pouca confiança = não mexe;
* **Balanço de branco** (``white_balance``): o tom do teto (quase sempre branco) e das paredes claras — achados
  pela segmentação quando ela roda, senão pelos pixels claros quase neutros — vira neutro, com limite (parede bege
  continua bege);
* **Exposição + "HDR" natural** (``auto_tone``): mapeamento de tom local base/detalhe — a luminância em log é
  separada em base (filtro guiado, que respeita bordas: sem halo) e detalhe; a base é exposta, as sombras sobem e
  os realces (janelas) são comprimidos; o detalhe volta inteiro. Um CLAHE leve em Lab dá o contraste local;
* **Céu** (``sky``): o céu (rótulo "sky" do OneFormer ADE20K) vira um céu azul limpo gerado aqui mesmo
  (``sky_image``: degradê + nuvens suaves de ruído fractal, sem nenhuma imagem de terceiros), com o brilho casado
  com a cena e borda refinada por filtro guiado; fora do céu nada muda;
* **Nitidez** (``sharpen``): máscara de nitidez só na luminância, com limiar (o ruído não é realçado).

Tudo menos o céu e a remoção é processamento de imagem clássico (OpenCV/numpy, rápido em CPU). A saída fica na
resolução da foto (ou 4K com ``upscale_4k``)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

Progress = Callable[[dict], None]


# ====================================================================================== utilidades
def _f(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(rgb, np.float32) / 255.0


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def luminance(lin: np.ndarray) -> np.ndarray:
    return (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]).astype(np.float32)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float, subsample: int = 1) -> np.ndarray:
    """Filtro guiado (He et al.) com guia em tons de cinza; ``subsample`` > 1 = versão rápida (coeficientes numa
    resolução menor, aplicados na cheia)."""
    I = np.asarray(guide, np.float32)
    p = np.asarray(src, np.float32)
    H, W = I.shape[:2]
    s = max(1, int(subsample))
    if s > 1:
        size = (max(1, W // s), max(1, H // s))
        Is, ps = cv2.resize(I, size, interpolation=cv2.INTER_AREA), cv2.resize(p, size, interpolation=cv2.INTER_AREA)
        r = max(1, radius // s)
    else:
        Is, ps, r = I, p, max(1, radius)
    k = (2 * r + 1, 2 * r + 1)
    mI, mp = cv2.blur(Is, k), cv2.blur(ps, k)
    cov = cv2.blur(Is * ps, k) - mI * mp
    var = cv2.blur(Is * Is, k) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    ma, mb = cv2.blur(a, k), cv2.blur(b, k)
    if s > 1:
        ma = cv2.resize(ma, (W, H), interpolation=cv2.INTER_LINEAR)
        mb = cv2.resize(mb, (W, H), interpolation=cv2.INTER_LINEAR)
    return (ma * I + mb).astype(np.float32)


# ====================================================================================== exposição / tom
@dataclass
class ToneInfo:
    exposicao_ev: float
    compressao_realces: float
    compressao_sombras: float
    sombras_antes: float
    realces_antes: float


TARGET_KEY = -2.1          # log2 da luminância (linear) "média" de uma foto de imóvel bem exposta (~0,23: clara)


def auto_tone(rgb: np.ndarray, strength: float = 1.0, local_contrast: float = 0.25) -> tuple[np.ndarray, dict]:
    """Exposição automática + mapeamento de tom local (sombras sobem, janelas estouradas descem), sem halos.

    ``strength`` 0..1 escala tudo (0 = devolve a foto). A correção já é proporcional ao problema: uma foto bem
    exposta quase não muda (nada de "filtro de Instagram")."""
    if strength <= 0:
        return np.asarray(rgb).copy(), {}
    x = _f(rgb)
    lin = srgb_to_linear(x)
    Y = luminance(lin)
    eps = 1.0 / 1024
    lY = np.log2(Y + eps)
    H, W = Y.shape
    diag = math.hypot(H, W)
    sub = max(1, int(round(max(H, W) / 640)))
    base = guided_filter(lY, lY, max(4, int(0.025 * diag)), 0.12, subsample=sub)
    detail = lY - base
    ys = cv2.resize(Y, (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
    shadows = float((linear_to_srgb(ys) < 0.12).mean())
    clipped = float((np.asarray(rgb).max(-1) >= 250).mean())
    lo, med, hi = np.percentile(base[::2, ::2], [1.0, 50.0, 99.5])
    # exposição: só parte da distância até o alvo (uma foto escura de propósito não vira dia); foto de imóvel é
    # clara de propósito: escurecer quase nunca
    # (uma sala branca bem exposta saía acinzentada): só quando há realce estourado de verdade, até -0,2 EV
    neg = -0.2 * min(1.0, max(0.0, clipped - 0.03) / 0.07)
    ev = float(np.clip((TARGET_KEY - med) * 0.7, neg, 1.6)) * strength
    b2 = base + ev
    pivot = med + ev
    # realces (janelas): o que passa do branco é comprimido para caber
    hi2 = hi + ev
    white = math.log2(0.80)
    c_hi = 1.0 if hi2 <= white else float(np.clip((white - pivot) / max(hi2 - pivot, 1e-3), 0.45, 1.0))
    c_hi = 1.0 - (1.0 - c_hi) * strength
    # sombras: o fundo fica perto de um piso (preto de verdade continua escuro, só abre)
    lo2 = lo + ev
    floor = math.log2(0.012)
    c_lo = 1.0 if lo2 >= floor else float(np.clip((pivot - floor) / max(pivot - lo2, 1e-3), 0.6, 1.0))
    c_lo = 1.0 - (1.0 - c_lo) * strength
    b3 = np.where(b2 > pivot, pivot + (b2 - pivot) * c_hi, pivot + (b2 - pivot) * c_lo)
    Y2 = np.power(2.0, b3 + detail * (1.0 + 0.08 * strength)) - eps
    Y2 = np.maximum(Y2, 0.0)
    ratio = Y2 / np.maximum(Y, 1e-5)
    lin2 = lin * ratio[..., None]
    # cor que estoura (um canal > 1): desatura em direção ao cinza da mesma luminância, sem mudar o matiz
    mx = lin2.max(-1)
    over = mx > 1.0
    if over.any():
        Yc = np.minimum(Y2, 1.0)[..., None]
        t = np.clip((1.0 - Yc) / np.maximum(mx[..., None] - Yc, 1e-5), 0, 1)
        lin2 = np.where(over[..., None], Yc + (lin2 - Yc) * t, lin2)
    out = linear_to_srgb(lin2)
    # o que já estava estourado na foto (janela branca) continua branco, não vira cinza
    blown = cv2.GaussianBlur((np.asarray(rgb).min(-1) >= 248).astype(np.float32), (0, 0), max(1.0, 0.002 * diag))
    if blown.any():
        out = out + (np.maximum(out, x) - out) * blown[..., None]
    out8 = _u8(out)
    if local_contrast > 0:
        out8 = clahe_lab(out8, local_contrast * strength)
    info = {"exposicao_ev": round(ev, 3), "compressao_realces": round(c_hi, 3),
            "compressao_sombras": round(c_lo, 3), "sombras_antes": round(shadows, 4),
            "estourado_antes": round(clipped, 4)}
    return out8, info


def clahe_lab(rgb: np.ndarray, amount: float = 0.25, clip: float = 1.6) -> np.ndarray:
    """CLAHE em duas escalas no L (Lab), misturado em ``amount``: contraste local sem mexer na cor."""
    if amount <= 0:
        return np.asarray(rgb).copy()
    lab = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    H, W = L.shape
    acc = np.zeros(L.shape, np.float32)
    for tiles in (4, 12):
        tg = (max(1, round(tiles * W / max(H, W))), max(1, round(tiles * H / max(H, W))))
        acc += cv2.createCLAHE(clipLimit=clip, tileGridSize=tg).apply(L).astype(np.float32)
    acc /= 2.0
    lab[..., 0] = np.clip(L + (acc - L) * amount, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ====================================================================================== balanço de branco
WB_TARGET = (0.0, 3.0)     # (a, b) do branco de referência: neutro com um toque quente (foto de imóvel acolhedora)


def white_balance(rgb: np.ndarray, ref_mask: np.ndarray | None = None, amount: float = 0.85,
                  max_shift: float = 9.0, min_cast: float = 2.0) -> tuple[np.ndarray, dict]:
    """Tira a dominante de cor (amarela de lâmpada, azul de sombra) usando superfícies que deveriam ser brancas.

    ``ref_mask``: teto/paredes (segmentação); só os pixels claros e pouco saturados dele contam (uma parede verde
    não vira referência). Sem máscara (ou poucos pixels nela), os pixels claros quase neutros da foto inteira, com
    metade da força. A correção é um ganho por canal em luz linear (von Kries), limitada a ``max_shift`` em Lab."""
    img = np.asarray(rgb)
    lab = cv2.cvtColor(_f(img), cv2.COLOR_RGB2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    cand = (L > 50) & (L < 97) & (chroma < 24)
    src = "superficies"
    k = amount
    if ref_mask is not None:
        rm = np.asarray(ref_mask, bool)
        if rm.shape != L.shape:
            rm = cv2.resize(rm.astype(np.uint8), (L.shape[1], L.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        sel = cand & rm
        if sel.sum() < 0.01 * sel.size:
            sel, src, k = cand, "neutros", amount * 0.75
    else:
        sel, src, k = cand, "neutros", amount * 0.75
    if sel.sum() < 0.005 * sel.size:
        return img.copy(), {"aplicado": False, "motivo": "sem superfície neutra"}
    # os mais claros pesam mais (white-patch): a referência é a metade mais clara dos candidatos
    Ls = L[sel]
    top = sel & (L >= np.percentile(Ls, 50))
    a0, b0 = float(np.median(a[top])), float(np.median(b[top]))
    da, db = a0 - WB_TARGET[0], b0 - WB_TARGET[1]
    cast = math.hypot(da, db)
    if cast < min_cast or math.hypot(a0, b0) < 3.5:      # já neutro (ou quase): não esquenta um cinza
        return img.copy(), {"aplicado": False, "dominante": round(cast, 2), "referencia": src}
    s = min(1.0, max_shift / cast) * k
    # alvo da referência em Lab -> ganhos em luz linear que levam a cor média dela até lá
    ref_lab = np.array([[[float(np.median(L[top])), a0, b0]]], np.float32)
    tgt_lab = np.array([[[ref_lab[0, 0, 0], a0 - da * s, b0 - db * s]]], np.float32)
    ref_rgb = srgb_to_linear(cv2.cvtColor(ref_lab, cv2.COLOR_Lab2RGB))[0, 0]
    tgt_rgb = srgb_to_linear(cv2.cvtColor(tgt_lab, cv2.COLOR_Lab2RGB))[0, 0]
    gains = tgt_rgb / np.maximum(ref_rgb, 1e-4)
    gains = gains / max(1e-6, float(0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2]))
    lin = srgb_to_linear(_f(img)) * gains[None, None]
    out = _u8(linear_to_srgb(lin))
    return out, {"aplicado": True, "dominante": round(cast, 2), "corrigido": round(cast * s, 2),
                 "referencia": src, "ganhos": [round(float(g), 4) for g in gains]}


# ====================================================================================== verticais
@dataclass
class VerticalInfo:
    aplicado: bool
    motivo: str = ""
    linhas: int = 0
    inclinacao_graus: float = 0.0      # pitch (câmera para cima/baixo)
    giro_graus: float = 0.0            # roll
    fracao: float = 0.0                # parte da correção aplicada (1 = toda)
    escala_recorte: float = 1.0
    H: list = field(default_factory=list)


MAX_PITCH = 14.0
MAX_ROLL = 6.0
MIN_LINES = 5
MIN_CROP_SCALE = 0.84


def _segments(gray: np.ndarray) -> np.ndarray:
    """Segmentos de reta (x1, y1, x2, y2) pelo LSD do OpenCV (com Hough como reserva)."""
    try:
        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(gray)[0]
        if lines is not None:
            return lines.reshape(-1, 4).astype(np.float64)
    except Exception:  # noqa: BLE001 - OpenCV sem LSD
        pass
    edges = cv2.Canny(gray, 60, 160)
    h = cv2.HoughLinesP(edges, 1, np.pi / 360, 60, minLineLength=max(20, gray.shape[0] // 12), maxLineGap=4)
    return np.zeros((0, 4)) if h is None else h.reshape(-1, 4).astype(np.float64)


def vertical_lines(rgb: np.ndarray, max_dev_deg: float = 25.0, min_len_frac: float = 0.06) -> np.ndarray:
    """Segmentos quase verticais e compridos, nas coordenadas da imagem dada."""
    img = np.asarray(rgb)
    H, W = img.shape[:2]
    s = min(1.0, 1024 / max(H, W))
    small = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    seg = _segments(gray) / s
    if not len(seg):
        return seg
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    ln = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    keep = (ang <= max_dev_deg) & (ln >= min_len_frac * H)
    return seg[keep]


def _line(seg: np.ndarray) -> np.ndarray:
    p1 = np.c_[seg[:, 0], seg[:, 1], np.ones(len(seg))]
    p2 = np.c_[seg[:, 2], seg[:, 3], np.ones(len(seg))]
    lines = np.cross(p1, p2)
    return lines / np.maximum(np.linalg.norm(lines[:, :2], axis=1, keepdims=True), 1e-9)


def _vp_error_deg(seg: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Ângulo entre cada segmento e a reta do ponto médio dele até o ponto de fuga ``v`` (homogêneo)."""
    mid = np.c_[(seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2]
    d = np.c_[seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]]
    if abs(v[2]) < 1e-9:
        t = np.tile(v[:2], (len(seg), 1))
    else:
        t = v[:2] / v[2] - mid
    cosang = np.abs((d * t).sum(1)) / np.maximum(np.linalg.norm(d, axis=1) * np.linalg.norm(t, axis=1), 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, 0, 1)))


def vertical_vp(seg: np.ndarray, iters: int = 400, tol_deg: float = 1.2, seed: int = 0
                ) -> tuple[np.ndarray | None, np.ndarray]:
    """Ponto de fuga vertical (homogêneo) por RANSAC sobre os segmentos, refinado pelos inliers (autovetor de
    Σ w·l·lᵀ, pesos = comprimento). Devolve (v, máscara de inliers)."""
    n = len(seg)
    if n < 2:
        return None, np.zeros(n, bool)
    L = _line(seg)
    w = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    rng = np.random.default_rng(seed)
    best, best_score = None, -1.0
    p = w / w.sum()
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False, p=p)
        v = np.cross(L[i], L[j])
        if np.linalg.norm(v) < 1e-12:
            continue
        inl = _vp_error_deg(seg, v) < tol_deg
        sc = float(w[inl].sum())
        if sc > best_score:
            best, best_score = v, sc
    if best is None:
        return None, np.zeros(n, bool)
    inl = _vp_error_deg(seg, best) < tol_deg
    if inl.sum() >= 2:
        M = (L[inl] * w[inl, None]).T @ L[inl]
        _, vecs = np.linalg.eigh(M)
        v = vecs[:, 0]
        inl2 = _vp_error_deg(seg, v) < tol_deg
        if inl2.sum() >= inl.sum():
            best, inl = v, inl2
    return best / np.linalg.norm(best), inl


def _rot_to_y(d: np.ndarray) -> np.ndarray:
    """Rotação mínima (Rodrigues) que leva a direção ``d`` ao eixo y (para baixo, convenção de imagem)."""
    d = d / np.linalg.norm(d)
    if d[1] < 0:
        d = -d
    y = np.array([0.0, 1.0, 0.0])
    axis = np.cross(d, y)
    s = np.linalg.norm(axis)
    c = float(np.dot(d, y))
    if s < 1e-12:
        return np.eye(3)
    axis /= s
    ang = math.atan2(s, c)
    R, _ = cv2.Rodrigues(axis * ang)
    return R


def _scaled_rotation(R: np.ndarray, t: float) -> np.ndarray:
    rv, _ = cv2.Rodrigues(R)
    return cv2.Rodrigues(rv * t)[0]


def _crop_scale(Hm: np.ndarray, W: int, H: int) -> tuple[float, np.ndarray]:
    """Maior retângulo com a proporção da foto, centrado no centro da imagem corrigida, dentro da área válida
    (o quadrilátero da foto depois da homografia). Devolve (escala do lado, centro)."""
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
    q = cv2.perspectiveTransform(corners[None], Hm)[0]
    c = cv2.perspectiveTransform(np.array([[[W / 2, H / 2]]], np.float64), Hm)[0, 0]
    poly = q.astype(np.float32).reshape(-1, 1, 2)

    def fits(s):
        hw, hh = W * s / 2, H * s / 2
        pts = [(c[0] - hw, c[1] - hh), (c[0] + hw, c[1] - hh), (c[0] + hw, c[1] + hh), (c[0] - hw, c[1] + hh)]
        return all(cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0 for px, py in pts)

    lo, hi = 0.0, 1.2
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo, c


def correct_verticals(rgb: np.ndarray, focal: float | None = None, max_pitch: float = MAX_PITCH,
                      max_roll: float = MAX_ROLL, min_lines: int = MIN_LINES,
                      min_scale: float = MIN_CROP_SCALE) -> tuple[np.ndarray, VerticalInfo]:
    """Paredes e batentes verticais: gira a câmera virtual (homografia K·R·K⁻¹) até o ponto de fuga vertical ir ao
    infinito e recorta na maior área válida com a mesma proporção (saída no mesmo tamanho). ``focal`` em px
    (``None`` = 0,7 × lado maior, uma grande-angular típica de foto de imóvel; as verticais ficam retas com
    qualquer focal, ela só muda a distorção no resto). Recusa (devolve a foto) com poucas linhas, ângulo grande
    demais (foto de propósito para cima, ou linhas erradas) ou recorte grande demais (aplica parte da correção)."""
    img = np.asarray(rgb)
    H, W = img.shape[:2]
    seg = vertical_lines(img)
    if len(seg) < min_lines:
        return img.copy(), VerticalInfo(False, "poucas linhas verticais", len(seg))
    v, inl = vertical_vp(seg)
    ninl = int(inl.sum()) if v is not None else 0
    if v is None or ninl < min_lines:
        return img.copy(), VerticalInfo(False, "linhas sem ponto de fuga comum", ninl)
    span = np.r_[seg[inl, 0], seg[inl, 2]]
    if span.max() - span.min() < 0.3 * W:
        return img.copy(), VerticalInfo(False, "linhas concentradas num lado só", ninl)
    f = float(focal or 0.7 * max(W, H))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)
    d = np.linalg.inv(K) @ v
    d = d / np.linalg.norm(d)
    if d[1] < 0:
        d = -d
    pitch = math.degrees(math.atan2(d[2], d[1]))
    roll = math.degrees(math.atan2(d[0], d[1]))
    info = VerticalInfo(False, "", ninl, round(pitch, 2), round(roll, 2))
    if abs(pitch) < 0.25 and abs(roll) < 0.2:
        info.motivo = "já está reto"
        return img.copy(), info
    if abs(pitch) > max_pitch or abs(roll) > max_roll:
        info.motivo = "inclinação grande demais (pouca confiança)"
        return img.copy(), info
    R = _rot_to_y(d)
    for t in (1.0, 0.85, 0.7, 0.55, 0.4):
        Rt = _scaled_rotation(R, t)
        Hm = K @ Rt @ np.linalg.inv(K)
        s, c = _crop_scale(Hm, W, H)
        if s >= min_scale:
            break
    else:
        info.motivo = "o recorte cortaria demais"
        return img.copy(), info
    # recorte: o retângulo (W·s × H·s) em volta de c vira a imagem inteira (W × H)
    A = np.array([[1 / s, 0, -(c[0] - W * s / 2) / s], [0, 1 / s, -(c[1] - H * s / 2) / s], [0, 0, 1]])
    M = A @ Hm
    out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    info.aplicado, info.fracao, info.escala_recorte = True, t, round(float(s), 4)
    info.H = [[round(float(x), 8) for x in row] for row in M]
    return out, info


def warp_like(arr: np.ndarray, info: VerticalInfo, size: tuple[int, int] | None = None) -> np.ndarray:
    """Aplica a mesma correção de verticais a uma máscara/rótulos (vizinho mais próximo)."""
    if not info.aplicado or not info.H:
        return arr
    a = np.asarray(arr)
    W, H = size or (a.shape[1], a.shape[0])
    return cv2.warpPerspective(a, np.asarray(info.H, np.float64), (W, H), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REPLICATE)


# ====================================================================================== céu
SKY_PRESETS: tuple[dict, ...] = (
    # céus próprios, gerados aqui (degradê + nuvens de ruído fractal): sem direitos de terceiros
    {"nome": "azul_limpo", "zenite": (52, 112, 196), "horizonte": (178, 210, 238), "nuvens": 0.10, "seed": 11},
    {"nome": "azul_nuvens_leves", "zenite": (60, 122, 204), "horizonte": (188, 216, 240), "nuvens": 0.32, "seed": 23},
    {"nome": "azul_profundo", "zenite": (38, 92, 178), "horizonte": (160, 198, 236), "nuvens": 0.18, "seed": 37},
    {"nome": "fim_de_tarde_suave", "zenite": (74, 126, 196), "horizonte": (236, 214, 190), "nuvens": 0.22,
     "seed": 41},
)


def _fractal_noise(h: int, w: int, seed: int, octaves: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh, gw = max(2, int(3 * 2 ** o * h / max(h, w)) + 2), max(2, int(3 * 2 ** o * w / max(h, w)) + 2)
        g = rng.random((gh, gw)).astype(np.float32)
        acc += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.55
    acc /= tot
    return (acc - acc.min()) / max(1e-6, float(acc.max() - acc.min()))


def sky_image(width: int, height: int, preset: int | dict = 0) -> np.ndarray:
    """Céu azul (RGB uint8) com degradê zênite -> horizonte e nuvens suaves. Determinístico por preset."""
    p = SKY_PRESETS[int(preset) % len(SKY_PRESETS)] if not isinstance(preset, dict) else preset
    w, h = max(8, int(width)), max(8, int(height))
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None] ** 0.8
    z, hz = np.array(p["zenite"], np.float32) / 255, np.array(p["horizonte"], np.float32) / 255
    zl, hl = srgb_to_linear(z), srgb_to_linear(hz)
    sky = zl[None, None] * (1 - t[..., None]) + hl[None, None] * t[..., None]
    sky = np.broadcast_to(sky, (h, w, 3)).copy()
    k = float(p.get("nuvens", 0.0))
    if k > 0:
        sm = max(h, w)
        small = (max(8, int(w * 256 / sm)), max(8, int(h * 256 / sm)))
        n = _fractal_noise(small[1], small[0], int(p.get("seed", 0)))
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        dens = np.clip((n - (1.0 - k) * 0.85) / 0.35, 0, 1) ** 1.5
        dens = cv2.GaussianBlur(dens, (0, 0), max(1.0, 0.004 * sm)) * (0.35 + 0.65 * (1 - t))
        cloud = srgb_to_linear(np.array([0.96, 0.965, 0.975], np.float32))
        sky = sky * (1 - 0.85 * dens[..., None]) + cloud[None, None] * 0.85 * dens[..., None]
    return _u8(linear_to_srgb(sky))


SKY_LABELS = ("sky",)
SKY_MIN_FRAC = 0.004
SKY_MAX_FRAC = 0.75


def sky_mask_from_labels(labels: np.ndarray, id2label: dict, size: tuple[int, int]) -> np.ndarray:
    ids = [int(k) for k, v in id2label.items() if str(v).split(",")[0].strip().lower() in SKY_LABELS]
    m = np.isin(np.asarray(labels), ids)
    W, H = size
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return m


def sky_alpha(rgb: np.ndarray, sky: np.ndarray) -> np.ndarray:
    """Máscara suave do céu: a da segmentação (grosseira, em baixa resolução) refinada por filtro guiado na
    luminância da foto — a borda segue o telhado, os galhos e o caixilho; zero longe do céu segmentado."""
    img = np.asarray(rgb)
    H, W = img.shape[:2]
    diag = math.hypot(H, W)
    m = np.asarray(sky, bool)
    if not m.any():
        return np.zeros((H, W), np.float32)
    # só os pixels claros/azulados do céu segmentado (céu não é escuro): tira caixilho e fio que vieram junto
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32)
    vs = v[m]
    thr = max(60.0, float(np.percentile(vs, 5)) * 0.85)
    core = m & (v >= thr)
    core = cv2.erode(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    r = max(2, int(0.004 * diag))
    a = guided_filter(gray, core.astype(np.float32), r, 1e-3)
    a = np.clip((a - 0.15) / 0.7, 0, 1)
    allowed = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    a[~allowed] = 0.0
    return a.astype(np.float32)


def replace_sky(rgb: np.ndarray, sky: np.ndarray, preset: int = 0) -> tuple[np.ndarray, dict]:
    """Troca o céu (máscara ``sky``) por um céu azul com brilho casado: a luminância média do céu novo fica perto
    da do céu antigo (céu estourado -> céu claro; céu já escurecido pela exposição -> mais profundo), sem nunca
    passar do branco da cena. Fora do céu os pixels ficam idênticos."""
    img = np.asarray(rgb)
    H, W = img.shape[:2]
    m = np.asarray(sky, bool)
    frac = float(m.mean())
    if frac < SKY_MIN_FRAC:
        return img.copy(), {"aplicado": False, "motivo": "sem céu", "fracao": round(frac, 4)}
    if frac > SKY_MAX_FRAC:
        return img.copy(), {"aplicado": False, "motivo": "céu demais (segmentação suspeita)", "fracao": round(frac, 4)}
    alpha = sky_alpha(img, m)
    if float(alpha.mean()) < SKY_MIN_FRAC * 0.5:
        return img.copy(), {"aplicado": False, "motivo": "céu escuro/ocupado", "fracao": round(frac, 4)}
    ys = np.nonzero(m.any(1))[0]
    top, bottom = int(ys.min()), int(ys.max()) + 1
    # o degradê vai do alto da foto até a linha mais baixa de céu (o horizonte visível)
    hsky = max(8, bottom)
    s = sky_image(W, hsky, preset)
    new = np.empty_like(img)
    new[:hsky] = s
    if hsky < H:
        new[hsky:] = s[-1:]
    lin_old = srgb_to_linear(_f(img))
    lin_new = srgb_to_linear(_f(new))
    Y_old = float(luminance(lin_old)[alpha > 0.5].mean()) if (alpha > 0.5).any() else 0.6
    Y_new = float(luminance(lin_new)[alpha > 0.5].mean()) if (alpha > 0.5).any() else 0.5
    target = float(np.clip(Y_old * 0.92, 0.18, 0.62))
    gain = target / max(Y_new, 1e-4)
    lin_new = np.clip(lin_new * gain, 0, 1)
    # composição "descontaminada": na borda (telhado, galhos) o pixel é uma mistura de céu antigo e primeiro plano
    # (I = a·C_antigo + (1-a)·F); trocar só a parte do céu (I + a·(C_novo - C_antigo)) tira a franja clara que a
    # mistura simples deixava. C_antigo = cor local do céu antigo (média gaussiana normalizada só sobre o céu)
    core = (alpha > 0.95).astype(np.float32)
    sig = max(2.0, 0.01 * math.hypot(H, W))
    den = cv2.GaussianBlur(core, (0, 0), sig)
    sky_old = cv2.GaussianBlur(lin_old * core[..., None], (0, 0), sig) / np.maximum(den, 1e-4)[..., None]
    sky_old = np.where((den > 1e-3)[..., None], sky_old, lin_old)
    out_lin = np.clip(lin_old + alpha[..., None] * (lin_new - sky_old), 0, 1)
    out = _u8(linear_to_srgb(out_lin))
    out[alpha <= 0] = img[alpha <= 0]
    return out, {"aplicado": True, "fracao": round(frac, 4), "preset": SKY_PRESETS[int(preset) % len(SKY_PRESETS)]["nome"],
                 "ganho": round(gain, 3), "topo": top}


# ====================================================================================== ruído / nitidez
def noise_sigma(rgb: np.ndarray) -> float:
    """Desvio do ruído (0..255) pelo método de Immerkær na luminância."""
    g = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    c = cv2.filter2D(g, -1, k)
    H, W = g.shape
    return float(math.sqrt(math.pi / 2) * np.abs(c[1:-1, 1:-1]).sum() / (6 * (W - 2) * (H - 2)))


def denoise(rgb: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    """Leve: ruído de cor (crominância) sempre suavizado com filtro guiado pela luminância (não borra bordas de
    cor); ruído de luminância só quando ele é visível (σ alto), com NL-means fraco."""
    img = np.asarray(rgb)
    if strength <= 0:
        return img.copy(), {}
    sig = noise_sigma(img)
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y = ycc[..., 0] / 255.0
    H, W = Y.shape
    r = max(2, int(0.003 * math.hypot(H, W)))
    for c in (1, 2):
        ch = ycc[..., c] / 255.0
        f = guided_filter(Y, ch, r, 4e-3)
        ycc[..., c] = (ch + (f - ch) * min(1.0, strength)) * 255.0
    out = cv2.cvtColor(np.clip(ycc + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    lum = False
    if sig > 3.0:
        h = float(np.clip(sig * 0.9 * strength, 2.0, 7.0))
        y = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
        y[..., 0] = cv2.fastNlMeansDenoising(y[..., 0], None, h, 5, 15)
        out = cv2.cvtColor(y, cv2.COLOR_YCrCb2RGB)
        lum = True
    return out, {"ruido_sigma": round(sig, 2), "luminancia": lum}


def sharpen(rgb: np.ndarray, amount: float = 0.6, radius: float | None = None, threshold: float = 2.0
            ) -> np.ndarray:
    """Máscara de nitidez na luminância (Lab L): raio proporcional ao tamanho da foto, com limiar (o ruído e as
    áreas lisas não ganham granulado) e sem estourar (o realce é limitado nas bordas fortes: sem halo branco)."""
    img = np.asarray(rgb)
    if amount <= 0:
        return img.copy()
    H, W = img.shape[:2]
    rad = radius or max(0.8, 0.0007 * math.hypot(H, W))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32)
    blur = cv2.GaussianBlur(L, (0, 0), rad)
    d = L - blur
    d = np.sign(d) * np.maximum(np.abs(d) - threshold, 0)
    d = np.clip(d, -12, 12)
    lab[..., 0] = np.clip(L + amount * d + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ====================================================================================== remover objetos
REMOVE_GROW = 0.006        # a pincelada cresce um pouco (× diagonal): pega a sombra e a borda do objeto
REMOVE_MAX_SIDE = 1280     # o removedor trabalha num recorte com lado maior até isto


def remove_objects(rgb: np.ndarray, mask: np.ndarray, inpainter=None, device: str = "cpu"
                   ) -> tuple[np.ndarray, dict]:
    """Remove o que o usuário pintou: cada área (crescida um pouco) é refeita pelo removedor de objetos num
    recorte com contexto (3× o tamanho dela), na resolução da foto quando cabe. Só a área pintada muda (borda
    suavizada de 1–2 px). ``inpainter(rgb, máscara) -> rgb`` (padrão: big-lama; sem os pesos: Telea)."""
    from .archlock import _disk, feather_alpha

    img = np.asarray(rgb)
    H, W = img.shape[:2]
    m = np.asarray(mask, bool)
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    if not m.any():
        return img.copy(), {"aplicado": False}
    diag = math.hypot(H, W)
    m = cv2.dilate(m.astype(np.uint8), _disk(max(2, int(REMOVE_GROW * diag)))) > 0
    kind = "remocao"
    own = None
    if inpainter is None:
        try:
            from ..removal.plate import load_lama

            own = load_lama(device if device.startswith("cuda") else "cpu", offline=True)
            inpainter = own
        except Exception:  # noqa: BLE001 - sem pesos: Telea (pior, mas não trava)
            from ..removal.plate import telea_inpaint

            inpainter, kind = telea_inpaint, "telea"
    out = img.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    try:
        for k in range(1, n):
            x, y, w, h = (int(v) for v in st[k, :4])
            cx, cy = x + w / 2, y + h / 2
            side_w, side_h = max(w * 3, 192), max(h * 3, 192)
            X0, X1 = int(max(0, cx - side_w / 2)), int(min(W, cx + side_w / 2))
            Y0, Y1 = int(max(0, cy - side_h / 2)), int(min(H, cy + side_h / 2))
            crop = out[Y0:Y1, X0:X1]
            cm = lab[Y0:Y1, X0:X1] == k
            ch, cw = crop.shape[:2]
            s = min(1.0, REMOVE_MAX_SIDE / max(ch, cw))
            if s < 1.0:
                sz = (max(8, int(cw * s)), max(8, int(ch * s)))
                small = cv2.resize(crop, sz, interpolation=cv2.INTER_AREA)
                sm = cv2.resize(cm.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
                sm = cv2.dilate(sm.astype(np.uint8), _disk(1)) > 0
                filled = cv2.resize(inpainter(small, sm), (cw, ch), interpolation=cv2.INTER_CUBIC)
            else:
                filled = inpainter(crop, cm)
            a = feather_alpha(cm, max(1.5, 0.0015 * diag), inside=1.0)
            out[Y0:Y1, X0:X1] = np.clip(crop * (1 - a[..., None]) + filled * a[..., None] + 0.5, 0, 255).astype(np.uint8)
    finally:
        if own is not None:
            own.release()
    return out, {"aplicado": True, "areas": int(n - 1), "fracao": round(float(m.mean()), 4), "metodo": kind}


# ====================================================================================== função completa
@dataclass
class EnhanceOptions:
    exposicao: bool = True        # a) exposição automática + "HDR" natural
    branco: bool = True           # b) balanço de branco
    verticais: bool = True        # c) correção de verticais
    ceu: bool = True              # d) troca do céu (fachadas, varandas: onde a segmentação vê céu)
    nitidez: bool = True          # f) nitidez + redução de ruído leve
    ampliar_4k: bool = False      # f) saída em 4K (Ampliar 4K encadeado)
    ceu_preset: int = 0
    intensidade: float = 1.0      # 0..1: força geral do tom


WALL_REFS = ("ceiling", "wall")


def _seg_default(device: str):
    from .quality import Segmenter

    seg = Segmenter(device)

    def fn(rgb):
        return seg.labels(rgb), seg.id2label

    fn.close = seg.close       # type: ignore[attr-defined]
    return fn


def enhance_image(rgb: np.ndarray, opts: EnhanceOptions | None = None, mask: np.ndarray | None = None,
                  seg_fn=None, inpainter=None, device: str = "cpu", progress=None) -> tuple[np.ndarray, dict]:
    """Aplica as etapas ligadas em ``opts`` à foto (RGB uint8) e devolve (imagem, relatório).

    ``seg_fn(rgb) -> (rótulos ADE20K, id2label)``: segmentação (só roda se o céu está ligado; o balanço de branco
    usa teto/paredes dela quando há). ``mask``: o que remover (branco = remover)."""
    o = opts or EnhanceOptions()
    img = np.asarray(rgb)[..., :3].astype(np.uint8)
    rep: dict = {"opcoes": asdict(o)}
    t: dict = {}

    def step(name, frac, msg):
        if progress:
            progress(name, frac, msg)

    t0 = time.perf_counter()
    if mask is not None and np.asarray(mask).any():
        step("remover", 0.05, "Removendo os objetos pintados")
        img, rep["remocao"] = remove_objects(img, mask, inpainter, device)
        t["remocao_s"] = round(time.perf_counter() - t0, 2)
    if o.nitidez:
        t0 = time.perf_counter()
        step("ruido", 0.2, "Reduzindo o ruído")
        img, rep["ruido"] = denoise(img, 1.0)
        t["ruido_s"] = round(time.perf_counter() - t0, 2)
    vinfo = VerticalInfo(False, "desligado")
    if o.verticais:
        t0 = time.perf_counter()
        step("verticais", 0.3, "Endireitando as verticais")
        img, vinfo = correct_verticals(img)
        rep["verticais"] = {k: v for k, v in asdict(vinfo).items() if k != "H"}
        t["verticais_s"] = round(time.perf_counter() - t0, 2)
    labels = id2label = None
    if o.ceu and seg_fn is not None:
        t0 = time.perf_counter()
        step("analise", 0.4, "Encontrando o céu, as paredes e o teto")
        try:
            labels, id2label = seg_fn(img)
        except Exception as e:  # noqa: BLE001 - sem segmentação o resto continua
            rep["segmentacao_erro"] = str(e)[:200]
    sky_m = None
    if labels is not None:
        # só o que a segmentação chama de céu (fachadas, varandas, vistas abertas). A vista pela janela é rotulada
        # "janela" inteira, e um teste de cor ali trocou uma parede de concreto cinza-azulada por céu (foto real):
        # janela fica como está
        Hh, Ww = img.shape[:2]
        sky_m = sky_mask_from_labels(labels, id2label, (Ww, Hh))
        t["segmentacao_s"] = round(time.perf_counter() - t0, 2)
    H, W = img.shape[:2]
    if o.branco:
        t0 = time.perf_counter()
        step("branco", 0.6, "Corrigindo o balanço de branco")
        ref = None
        if labels is not None:
            ids = [int(k) for k, v in id2label.items() if str(v).split(",")[0].strip().lower() in WALL_REFS]
            ref = np.isin(labels, ids)
        img, rep["branco"] = white_balance(img, ref)
        t["branco_s"] = round(time.perf_counter() - t0, 2)
    if o.exposicao:
        t0 = time.perf_counter()
        step("tom", 0.7, "Ajustando a exposição e as janelas")
        img, rep["tom"] = auto_tone(img, float(np.clip(o.intensidade, 0, 1)))
        t["tom_s"] = round(time.perf_counter() - t0, 2)
    if o.ceu:
        t0 = time.perf_counter()
        if sky_m is None:
            rep["ceu"] = {"aplicado": False, "motivo": "sem segmentação"}
        else:
            step("ceu", 0.8, "Trocando o céu")
            img, rep["ceu"] = replace_sky(img, sky_m, o.ceu_preset)
        t["ceu_s"] = round(time.perf_counter() - t0, 2)
    if o.nitidez:
        step("nitidez", 0.9, "Aumentando a nitidez")
        img = sharpen(img, 0.55)
    rep["tempos"] = t
    return img, rep


def enhance(photo, out_dir=None, opts: EnhanceOptions | None = None, mask=None, progress: Progress | None = None,
            device: str = "auto", seg_fn=None, inpainter=None, upscaler=None):
    """Função "Melhorar foto" completa: lê a foto, aplica :func:`enhance_image`, grava ``melhorar_1.jpg`` (resolução
    da foto, ou 4K com ``opts.ampliar_4k``), ``antes.jpg`` (a foto no mesmo tamanho, para o comparador) e
    ``design.json``. Devolve um ``api.DesignResult`` (mesmo evento ``done`` das outras funções)."""
    from . import api

    o = opts or EnhanceOptions()
    stages = (("preparo", "Preparando a foto", 0.05), ("analise", "Melhorando a foto", 0.75),
              ("final", "Finalizando", 0.2))
    prog = api._Prog(progress, stages)
    out = api._out_dir(out_dir)
    t_all = time.perf_counter()
    prog("preparo")
    rgb = api._load_rgb(photo)
    m = None
    if mask is not None:
        from .archlock import user_mask

        m = user_mask(api._load_mask(mask), (rgb.shape[1], rgb.shape[0]))
        if not m.any():
            m = None
    dev = device
    if dev == "auto":
        try:
            from .engine import pick_device

            dev = pick_device("auto")
        except Exception:  # noqa: BLE001
            dev = "cpu"
    own_seg = None
    if o.ceu and seg_fn is None:
        try:
            own_seg = _seg_default(dev)
            seg_fn = own_seg
        except Exception as e:  # noqa: BLE001 - modelo ausente: segue sem céu
            prog.emit({"event": "log", "level": "warning",
                       "message": f"Troca do céu indisponível (modelo de segmentação não carregou): {e}"})
    try:
        img, rep = enhance_image(rgb, o, m, seg_fn, inpainter, dev,
                                 progress=lambda st, f, msg: prog("analise", f, msg))
    finally:
        if own_seg is not None and hasattr(own_seg, "close"):
            own_seg.close()
    prog("final", 0.0, "Gravando a foto")
    before_img = rgb
    if o.ampliar_4k and max(img.shape[:2]) < 3840:
        prog("final", 0.1, "Ampliando para 4K")
        try:
            if upscaler is None:
                from .upscale import upscale as _up

                img, info = _up(img, None, long_side=3840, device=dev)
            else:
                img, info = upscaler(img, 3840)
            rep["ampliar_4k"] = {k: v for k, v in (info or {}).items() if k in ("de", "para", "s")}
        except Exception as e:  # noqa: BLE001 - sem o modelo: Lanczos (avisado)
            s = 3840 / max(img.shape[:2])
            img = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))),
                             interpolation=cv2.INTER_LANCZOS4)
            rep["ampliar_4k"] = {"metodo": "lanczos", "erro": str(e)[:200]}
            prog.emit({"event": "log", "level": "warning",
                       "message": "Ampliação por IA indisponível: a foto foi ampliada do jeito simples."})
    before_img = api._resize(before_img, (img.shape[1], img.shape[0]))
    path = api._save(img, out / "melhorar_1.jpg")
    prog.emit({"event": "preview", "path": path, "index": 0})
    before = api._save(before_img, out / "antes.jpg")
    rep.update({"funcao": "melhorar", "tamanho": [img.shape[1], img.shape[0]], "saidas": [path], "antes": before,
                "dispositivo": dev})
    rep.setdefault("tempos", {})["total_s"] = round(time.perf_counter() - t_all, 2)
    report = out / "design.json"
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=api._json_default), encoding="utf-8")
    return api.DesignResult([path], before, str(report), "melhorar", None, "sala", [], (img.shape[1], img.shape[0]),
                            rep["tempos"], [img])


__all__ = ["EnhanceOptions", "enhance", "enhance_image", "auto_tone", "clahe_lab", "white_balance",
           "correct_verticals", "vertical_lines", "vertical_vp", "warp_like", "sky_image", "SKY_PRESETS",
           "replace_sky", "sky_alpha", "sky_mask_from_labels", "denoise", "sharpen", "noise_sigma",
           "remove_objects", "guided_filter", "VerticalInfo"]

