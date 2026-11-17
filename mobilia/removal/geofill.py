"""Sala vazia reconstruída pela geometria da sala (estratégia "geometria" da placa do modo B).

Nada é imaginado: a área escondida pelos móveis é refeita só com o material **real** que a foto mostra, plano a plano,
nas vistas retificadas da caixa da sala (:class:`mobilia.removal.plate.RoomBox`, câmera e planos do MoGe-2).

1. **Rótulos.** Os mesmos de :func:`plate.assign_labels` (piso, teto, parede *i*, "outro", protegido), sem a regra
   "parede vira piso por vizinhança" (ela deixa a divisa torta). A **linha parede/piso** de cada parede é corrigida
   pela divisa parede/piso que a segmentação mostra fora da máscara (ajuste robusto de deslocamento e inclinação
   sobre a linha projetada da caixa): a divisa no buraco é uma reta, onde a foto diz que ela está.
2. **Piso** (vista de cima): campo de luz suave (polinômio de 2º grau do log da luminância, ajuste robusto) medido no
   piso visível; albedo = foto / luz; o buraco é preenchido por **síntese por exemplos** (retalhos do piso visível
   escolhidos pela diferença na sobreposição, emendas suavizadas em poucos pixels). Revestimento com grade
   (porcelanato, azulejo) ou tábuas: o período de cada eixo é medido por autocorrelação e os retalhos só andam
   múltiplos do período, então o rejunte continua alinhado com a grade real. Depois: albedo × luz.
3. **Paredes e teto** (vista de frente / de baixo): mesmo campo de luz. Parede **lisa** (pouca energia de textura):
   perfil de cor por altura acima da linha do piso (traz o rodapé, reto) × luz + grão medido; nada de névoa.
   Parede **com textura** (papel, madeira, pedra): síntese por exemplos com os retalhos presos à mesma altura (o
   rodapé e as faixas horizontais continuam). Parede sem nada à vista: perfil das outras paredes.
4. **Acabamento:** membrana de tom na borda (Poisson) e grão casado com o anel; a costura fica em poucos pixels.
5. **Protegidos** (janela, porta, cortina) nunca são repintados fora da máscara; o pedaço escondido continua os
   pixels reais da abertura, decidido por componente (cortina até a linha do piso; vidro até o piso ou até o
   peitoril, ver :func:`continue_openings`; o resto é parede). **"Outro"**
   (sem plano confiável, vista por uma porta): cor neutra do entorno, lisa, sem conteúdo inventado. Sobras sem
   vista: continuação dos vizinhos já refeitos. Nenhum inpainter é chamado.

Puro numpy/OpenCV: segundos por foto na CPU.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from . import plate as rp

# --- parâmetros ------------------------------------------------------------------------------------------------
LINE_MIN_POINTS = 12          # colunas com divisa parede/piso visível para corrigir a linha da caixa
LINE_MAX_SHIFT = 0.18         # correção máxima da linha (fração da altura da foto)
LINE_SLOPE_SPAN = 0.30        # vão mínimo (fração das colunas da parede) para ajustar também a inclinação
MIN_KNOWN_PX = 600            # material visível mínimo (vista retificada) para sintetizar por exemplos
PLAIN_BAND_STD = 2.6          # desvio robusto da banda média do albedo abaixo disto -> parede/teto lisos
N_CAND = 128                  # candidatos por bloco na síntese por exemplos
PERIOD_MIN_ACF = 0.35
SILL_MIN_M = 0.85             # vidro visto abaixo disto (m sobre o piso): porta/janela de piso a teto, vai até o chão
#                               (peitoril de janela comum fica em ~0,9-1,1 m)
SILL_TYPICAL_M = 0.9          # sem medida do peitoril: o vidro escondido para nesta altura (m)
MIN_PERIODS = 3               # período horizontal de parede só com >= 3 repetições visíveis
STRIPE_INV_MIN = 0.6          # invariância vertical do detalhe a partir da qual a parede é "listras" (veio, ripas)
STRIPE_TEX_MIN = 0.8          # ... desde que tenha alguma textura (parede lisa com ruído não tem invariância)
WALL_RES_DECAY = 0.06         # o resíduo de cor/luz da borda da parede se apaga em ~6% da diagonal dentro do buraco
WALL_AO_RING = 0.035          # faixa (fração da diagonal) em volta do móvel onde a parede mais escura é oclusão
WALL_AO_DARK = 0.06           # ... escurecer mais que 6% em relação ao entorno largo não é luz da parede
FLOOR_DARK_MAX = 0.03         # a correção de luz do piso na borda escurece no máximo 3% (sombra/reflexo não entra)
FINE_DETAIL_MIN = 4.0         # desvio robusto do passa-alta do piso visível a partir do qual copia o detalhe fino


# ============================================================================================================
# utilidades
# ============================================================================================================
def _rstd(a: np.ndarray) -> float:
    a = np.asarray(a, np.float32).ravel()
    if a.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def _lum(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _poly_terms(u: np.ndarray, v: np.ndarray, deg: int) -> np.ndarray:
    cols = [np.ones_like(u)]
    if deg >= 1:
        cols += [u, v]
    if deg >= 2:
        cols += [u * u, v * v, u * v]
    return np.stack(cols, -1)


def illumination_field(img: np.ndarray, known: np.ndarray, rng: np.random.Generator, deg: int = 2,
                       max_ratio: float = 2.5) -> np.ndarray:
    """Campo de luz suave (escala da luminância) ajustado no material visível: polinômio de grau ``deg`` do log da
    luminância borrada, com ajuste robusto (tira manchas de sol, sombras e objetos). Média geométrica 1 no visível;
    limitado a [1/max_ratio, max_ratio] para não explodir longe do visível."""
    h, w = known.shape
    ys, xs = np.nonzero(known)
    if ys.size < 50:
        return np.ones((h, w), np.float32)
    lum = cv2.GaussianBlur(_lum(img), (0, 0), 2.0)
    if ys.size > 30000:
        k = rng.choice(ys.size, 30000, replace=False)
        ys, xs = ys[k], xs[k]
    s = float(max(h, w))
    u, v = xs / s - 0.5, ys / s - 0.5
    span = max(np.ptp(u), np.ptp(v))
    # pouco material visível (ou só uma faixa): grau menor para não extrapolar uma curva
    d = deg if (ys.size >= 2000 and span > 0.25) else (1 if ys.size >= 300 else 0)
    A = _poly_terms(u, v, d)
    y = np.log(np.maximum(lum[ys, xs], 1.0))
    keep = np.ones(len(y), bool)
    coef = np.zeros(A.shape[1])
    coef[0] = float(np.median(y))
    for _ in range(4):
        if keep.sum() < A.shape[1] * 10:
            break
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        res = y - A @ coef
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-3
        keep = np.abs(res) < 2.5 * mad
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    F = _poly_terms(uu / s - 0.5, vv / s - 0.5, d) @ coef
    F -= float(np.median(A @ coef))
    return np.exp(np.clip(F, -np.log(max_ratio), np.log(max_ratio))).astype(np.float32)


def _peak(vals: np.ndarray, p: int, r: int = 2) -> float:
    """Posição subpixel do máximo perto de ``p`` (parábola nos 3 pontos em volta do máximo local)."""
    lo, hi = max(1, p - r), min(len(vals) - 2, p + r)
    j = lo + int(np.argmax(vals[lo:hi + 1]))
    a, b, c = vals[j - 1], vals[j], vals[j + 1]
    den = a - 2 * b + c
    return float(j + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0))


def periodic_copy(img: np.ndarray, known: np.ndarray, hole: np.ndarray, period, reach: int = 14):
    """Como :func:`plate.periodic_fill`, com período subpixel: cada pixel do buraco copia o pixel conhecido mais
    próximo a um deslocamento ``round(i·py), round(j·px)`` (rejunte alinhado mesmo longe da parte visível)."""
    py, px = period
    out = img.copy()
    todo = hole.copy()
    src_ok = rp._ero(known, 2)
    h, w = hole.shape
    offs = [(round(i * py), round(j * px)) for i in range(-reach, reach + 1)
            for j in range(-reach, reach + 1) if i or j]
    offs.sort(key=lambda o: o[0] ** 2 + o[1] ** 2)
    for dy, dx in offs:
        ys, xs = np.nonzero(todo)
        if ys.size == 0:
            break
        sy, sx = ys + dy, xs + dx
        ok = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
        ys, xs, sy, sx = ys[ok], xs[ok], sy[ok], sx[ok]
        ok = src_ok[sy, sx]
        out[ys[ok], xs[ok]] = img[sy[ok], sx[ok]]
        todo[ys[ok], xs[ok]] = False
    return out, todo


def axis_invariance(gray: np.ndarray, known: np.ndarray, axis: int, lag: int) -> float:
    """Correlação (detalhe sem a luz) entre cada pixel conhecido e o vizinho a ``lag`` px no eixo ``axis``: perto
    de 1 = listras que correm nesse eixo (tábuas, painéis, frisos verticais)."""
    g = gray.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 8.0)
    if axis == 0:
        a, b, ka, kb = g[:-lag], g[lag:], known[:-lag], known[lag:]
    else:
        a, b, ka, kb = g[:, :-lag], g[:, lag:], known[:, :-lag], known[:, lag:]
    sel = ka & kb
    if sel.sum() < 300:
        return 0.0
    x, y = a[sel], b[sel]
    x, y = x - x.mean(), y - y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 1e-6 else 0.0


def stripe_continue(det: np.ndarray, known: np.ndarray, hole: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Listras (tábuas, painéis): cada pixel do buraco copia o pixel conhecido mais próximo **na mesma listra**
    (mesma coluna se ``axis`` = 0, mesma linha se 1). Devolve (detalhe, sobras sem listra conhecida)."""
    out = det.copy()
    left = hole.copy()
    d = det if axis == 0 else np.swapaxes(det, 0, 1)
    k = known if axis == 0 else known.T
    hl = hole if axis == 0 else hole.T
    o = out if axis == 0 else np.swapaxes(out, 0, 1)
    lf = left if axis == 0 else left.T
    for x in np.nonzero(hl.any(0))[0]:
        kx = np.nonzero(k[:, x])[0]
        if kx.size < 3:
            continue
        ys = np.nonzero(hl[:, x])[0]
        j = np.clip(np.searchsorted(kx, ys), 0, kx.size - 1)
        j2 = np.clip(j - 1, 0, kx.size - 1)
        near = np.where(np.abs(kx[j2] - ys) < np.abs(kx[j] - ys), kx[j2], kx[j])
        o[ys, x] = d[near, x]
        lf[ys, x] = False
    return out, left


def detect_axis_periods(gray: np.ndarray, known: np.ndarray, min_p: int = 6,
                        min_acf: float = PERIOD_MIN_ACF) -> tuple[int | None, int | None]:
    """Período (py, px) de cada eixo, independente (tábuas: só um eixo; porcelanato: os dois). Autocorrelação
    normalizada só nos pixels conhecidos, depois de tirar o gradiente de luz."""
    if known.sum() < 400:
        return None, None
    g = gray.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 8.0)
    m = known.astype(np.float32)
    g = (g - g[known].mean()) * m
    var = float((g[known] ** 2).mean())
    if var < 1.0:
        return None, None
    h, w = g.shape
    s = (2 * h, 2 * w)
    F = np.fft.rfft2(g, s)
    Mf = np.fft.rfft2(m, s)
    num = np.fft.irfft2(np.abs(F) ** 2, s)
    den = np.fft.irfft2(np.abs(Mf) ** 2, s)
    acf = num / np.maximum(den, 1.0) / var
    overlap = den / max(den[0, 0], 1.0)

    def axis_period(vals, ov):
        n = len(vals)
        for p in range(min_p, n - 1):
            if ov[p] < 0.2:
                break
            if vals[p] >= min_acf and vals[p] >= vals[p - 1] and vals[p] >= vals[p + 1] \
                    and vals[1:p].min() < vals[p] - 0.25:
                # confirma o 2º harmônico quando cabe (rejeita picos isolados)
                p2 = 2 * p
                if p2 + 1 < n and ov[p2] >= 0.4 and vals[p2 - 1:p2 + 2].max() < 0.3 * vals[p]:
                    continue
                # subpixel: o pico do k-ésimo harmônico mais longe que ainda tem sobreposição / k
                best = _peak(vals, p)
                k = 2
                while k * p + 2 < n and ov[k * p] >= 0.3:
                    q = _peak(vals, round(k * best))
                    if vals[round(q)] < 0.5 * vals[p]:
                        break
                    best = q / k
                    k += 1
                return float(best)
        return None

    px = axis_period(acf[0, : w // 2], overlap[0, : w // 2])
    py = axis_period(acf[: h // 2, 0], overlap[: h // 2, 0])
    return py, px


def _nblur(img: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Convolução normalizada (como :func:`plate._nblur`); sigma grande roda numa versão reduzida (rápido)."""
    if sigma <= 6.0:
        return rp._nblur(img, w, sigma)
    f = sigma / 3.0
    h, wd = w.shape
    sh, sw = max(4, round(h / f)), max(4, round(wd / f))
    wf = w.astype(np.float32)
    num = cv2.resize(img.astype(np.float32) * (wf[..., None] if img.ndim == 3 else wf), (sw, sh),
                     interpolation=cv2.INTER_AREA)
    den = cv2.resize(wf, (sw, sh), interpolation=cv2.INTER_AREA)
    s2 = sigma * sw / wd
    num = cv2.GaussianBlur(num, (0, 0), s2)
    den = cv2.GaussianBlur(den, (0, 0), s2)
    out = num / (np.maximum(den, 1e-6)[..., None] if num.ndim == 3 else np.maximum(den, 1e-6))
    return cv2.resize(out, (wd, h), interpolation=cv2.INTER_LINEAR)


# ============================================================================================================
# síntese por exemplos (retalhos com sobreposição)
# ============================================================================================================
def quilt(img: np.ndarray, known: np.ndarray, hole: np.ndarray, rng: np.random.Generator,
          period: tuple[int | None, int | None] = (None, None), row_lock: np.ndarray | None = None,
          block: int | None = None, n_cand: int = N_CAND) -> tuple[np.ndarray, np.ndarray, dict]:
    """Preenche ``hole`` copiando retalhos do material conhecido (``known``) de ``img`` (float32, HxWxC).

    Blocos de lado ``B`` em grade (passo ``B - B/4``), na ordem da borda para dentro; para cada bloco, ``n_cand``
    origens aleatórias (retalho todo conhecido) e fica a de menor diferença quadrática na parte já válida do bloco
    (escolha aleatória entre as quase iguais, para não repetir sempre o mesmo retalho). ``period`` (py, px): o
    deslocamento origem→destino é múltiplo do período naquele eixo (grade alinhada). ``row_lock`` (HxW, float):
    "altura" de cada pixel (paredes); a origem fica na mesma altura do destino (rodapé e faixas continuam).
    Emendas: rampa linear na sobreposição. Só pixels de ``hole`` mudam. Devolve (imagem, sobras, info)."""
    h, w = hole.shape
    cur = img.astype(np.float32).copy()
    src = img.astype(np.float32)
    info: dict = {}
    if not hole.any():
        return cur, np.zeros_like(hole), info
    if block is None:
        block = int(np.clip(round(min(h, w) / 18), 16, 40))
    B = block
    ok_src = None
    while B >= 8:
        ii = cv2.integral(known.astype(np.uint8))
        tot = ii[B:, B:] - ii[:-B, B:] - ii[B:, :-B] + ii[:-B, :-B]
        ok_src = tot[: h - B + 1, : w - B + 1] >= B * B
        if ok_src.sum() >= 30:
            break
        B //= 2
    if ok_src is None or ok_src.sum() < 30:
        return cur, hole.copy(), {"sem_fonte": True}
    sy_all, sx_all = np.nonzero(ok_src)
    O = max(2, B // 4)
    S = B - O
    info.update(bloco=B, fontes=int(sy_all.size))
    valid = known & ~hole
    ys, xs = np.nonzero(hole)
    y_lo, y_hi, x_lo, x_hi = ys.min(), ys.max(), xs.min(), xs.max()
    gy = list(range(max(0, y_lo - O), max(1, y_hi + 1), S))
    gx = list(range(max(0, x_lo - O), max(1, x_hi + 1), S))
    dt = ndimage.distance_transform_edt(hole)
    blocks = []
    for y0 in gy:
        y0 = min(y0, h - B) if h >= B else 0
        for x0 in gx:
            x0 = min(x0, w - B) if w >= B else 0
            sub = hole[y0:y0 + B, x0:x0 + B]
            if sub.any():
                blocks.append((float(dt[y0:y0 + B, x0:x0 + B][sub].min()), y0, x0))
    blocks = sorted(set(blocks))
    dy_g, dx_g = np.mgrid[0:B, 0:B]
    src_l = src.mean(-1)
    cur_l = cur.mean(-1)
    py, px = period
    n_relax = 0
    for _, y0, x0 in blocks:
        bh, bw = min(B, h - y0), min(B, w - x0)
        tgt = cur[y0:y0 + bh, x0:x0 + bw]
        wv = valid[y0:y0 + bh, x0:x0 + bw].astype(np.float32)
        for tries in (1, 4, 16):                  # mais candidatos com as travas antes de largar as travas
            k = rng.integers(0, sy_all.size, n_cand * tries)
            sy, sx = sy_all[k].astype(np.int64), sx_all[k].astype(np.int64)
            if row_lock is not None:
                # mesma altura: origem na linha cuja altura (na coluna de origem) é a do destino
                hd = float(row_lock[y0 + bh // 2, x0 + bw // 2])
                col = row_lock[:, np.clip(sx + bw // 2, 0, w - 1)]               # (h, n)
                sy = np.argmin(np.abs(col - hd), axis=0) - bh // 2
            if py:
                sy = y0 + np.round(np.round((sy - y0) / py) * py).astype(np.int64)
            if px:
                sx = x0 + np.round(np.round((sx - x0) / px) * px).astype(np.int64)
            inb = (sy >= 0) & (sx >= 0) & (sy <= h - B) & (sx <= w - B)
            good = np.zeros_like(inb)
            good[inb] = ok_src[sy[inb], sx[inb]]
            if good.sum() >= 4 or (row_lock is None and not py and not px):
                break
        if good.sum() < 4:
            n_relax += 1
            k = rng.integers(0, sy_all.size, n_cand)
            sy, sx = sy_all[k].astype(np.int64), sx_all[k].astype(np.int64)
            if row_lock is not None:              # pelo menos a altura mais próxima possível
                hd = float(row_lock[y0 + bh // 2, x0 + bw // 2])
                d = np.abs(row_lock[np.clip(sy + bh // 2, 0, h - 1), np.clip(sx + bw // 2, 0, w - 1)] - hd)
                order = np.argsort(d)[: max(8, n_cand // 6)]
                sy, sx = sy[order], sx[order]
            good = np.ones(sy.shape, bool)
        sy, sx = sy[good], sx[good]
        if wv.sum() > 0:
            # custo na luminância (3× mais barato), só nos pixels já válidos do bloco
            iy, ix = np.nonzero(wv)
            Pl = src_l[sy[:, None] + iy[None], sx[:, None] + ix[None]]
            cost = ((Pl - cur_l[y0 + iy, x0 + ix][None]) ** 2).mean(1)
            near = np.nonzero(cost <= cost.min() * 1.08 + 1e-6)[0]
            j = int(near[rng.integers(near.size)])
        else:
            j = int(rng.integers(len(sy)))
        patch = src[sy[j]:sy[j] + bh, sx[j]:sx[j] + bw]
        hl = hole[y0:y0 + bh, x0:x0 + bw]
        prev = wv > 0
        a = np.ones((bh, bw), np.float32)
        if prev.any() and (~prev).any():
            # rampa: perto da parte nova, valor novo; dentro da parte já preenchida, cai para o antigo em O px
            dnew = cv2.distanceTransform((prev).astype(np.uint8), cv2.DIST_L2, 3)
            a = np.where(prev, np.clip(1.0 - dnew / O, 0.0, 1.0), 1.0).astype(np.float32)
        a = a * hl
        tgt[:] = a[..., None] * patch + (1 - a[..., None]) * tgt
        cur_l[y0:y0 + bh, x0:x0 + bw] = tgt.mean(-1)
        valid[y0:y0 + bh, x0:x0 + bw] |= hl
    info["blocos"] = len(blocks)
    info["relaxados"] = n_relax
    return cur, hole & ~valid, info


# ============================================================================================================
# linha parede/piso corrigida
# ============================================================================================================
def _box_floor_rows(box) -> tuple[np.ndarray, np.ndarray]:
    """Por coluna: 1ª linha de piso da caixa (y) e a parede logo acima dela (-1 = nenhuma)."""
    lab = box.label
    H, W = lab.shape
    fl = lab == rp.FLOOR
    has = fl.any(0)
    yf = np.where(has, fl.argmax(0), H).astype(np.int64)
    above = np.full(W, -1, np.int64)
    cols = np.nonzero(has & (yf > 0))[0]
    above[cols] = lab[yf[cols] - 1, cols]
    above[above < 0] = -1
    return yf, above


def floor_lines(mask: np.ndarray, groups: np.ndarray, box) -> dict:
    """Linha parede/piso de cada parede, corrigida pela divisa visível da segmentação.

    Para cada coluna fora da máscara, a divisa é a 1ª transição parede→piso perto da linha projetada da caixa. O
    deslocamento (divisa − linha da caixa) é ajustado por uma reta robusta em x (ou só a mediana, se a divisa
    visível cobre pouco da parede). Devolve {id_parede: (c0, c1, n_pontos)}: y_corrigida(x) = y_caixa(x) + c0 + c1·x."""
    H, _W = mask.shape
    yf, above = _box_floor_rows(box)
    near_mask = rp._dil(mask, 3)
    wall_g = (groups == rp.G_WALL) & ~near_mask
    floor_g = (groups == rp.G_FLOOR) & ~near_mask
    # piso a até GAP px abaixo do pixel de parede (rodapé com rótulo incerto entre os dois)
    GAP = 4
    fb = np.zeros_like(floor_g)
    for k in range(1, GAP + 1):
        fb[:-k] |= floor_g[k:]
    out: dict = {}
    reach = int(LINE_MAX_SHIFT * H)
    for wl in box.room.walls:
        cols = np.nonzero(above == wl.id)[0]
        if cols.size < 4:
            continue
        xs_pts, off = [], []
        for x in cols[:: max(1, cols.size // 400)]:
            y0, y1 = max(1, yf[x] - reach), min(H - 2, yf[x] + reach)
            if y1 <= y0:
                continue
            tr = np.nonzero(wall_g[y0:y1, x] & fb[y0:y1, x] & ~wall_g[y0 + 1:y1 + 1, x])[0]
            if tr.size == 0:
                continue
            yy = y0 + tr + 0.5
            j = int(np.argmin(np.abs(yy - yf[x])))
            xs_pts.append(float(x))
            off.append(float(yy[j] - yf[x]))
        n = len(off)
        if n < LINE_MIN_POINTS:
            continue
        xs_pts, off = np.array(xs_pts), np.array(off)
        span = (xs_pts.max() - xs_pts.min()) / max(1.0, float(cols.max() - cols.min()))
        c1 = 0.0
        c0 = float(np.median(off))
        if span >= LINE_SLOPE_SPAN:
            keep = np.abs(off - c0) < max(6.0, 3 * _rstd(off))
            for _ in range(3):
                if keep.sum() < LINE_MIN_POINTS:
                    break
                A = np.c_[np.ones(keep.sum()), xs_pts[keep]]
                (c0, c1), *_ = np.linalg.lstsq(A, off[keep], rcond=None)
                res = off - (c0 + c1 * xs_pts)
                keep = np.abs(res) < max(3.0, 2.5 * _rstd(res[keep]))
        # descarta o ajuste que não explica a maioria (divisa ruidosa: objetos rentes, tapete)
        res = off - (c0 + c1 * xs_pts)
        if np.mean(np.abs(res) < 6.0) < 0.5:
            continue
        # o deslocamento é relativo à coluna: c0 + c1*x vale para as colunas desta parede
        out[wl.id] = (float(c0), float(c1), int(n))
    return out


def far_floor(mask: np.ndarray, groups: np.ndarray, box, labels: np.ndarray, margin: float = 1.08) -> np.ndarray:
    """Pixels mascarados que a caixa chama de piso mas que ficam bem além (distância no plano do piso) de todo o
    piso visível: a caixa sem a parede do fundo leva o "piso" até o horizonte, atrás de cortinas e janelas. Esses
    pixels não têm plano confiável."""
    vis = (labels == rp.FLOOR) & ~rp._dil(mask, 2) & (groups == rp.G_FLOOR)
    cand = mask & (labels == rp.FLOOR)
    if vis.sum() < 200 or not cand.any():
        return np.zeros_like(mask)
    t = box.t
    lim = float(np.percentile(t[vis], 97)) * margin
    return cand & (t > lim)


def other_to_opening(labels: np.ndarray, mask: np.ndarray, groups: np.ndarray) -> int:
    """Pixel mascarado "outro" (sem plano confiável) cujo 1º pixel visível acima, na mesma coluna, é
    janela/cortina/porta (mesinha na frente da cortina, com o "piso" da caixa indo além do piso visível): atrás
    dele está a abertura e ele vira "protegido" (:func:`continue_openings`). Nunca uma cor neutra lisa no meio da
    cortina. (Os de parede só são candidatos: ver :func:`fill_geometry`.)"""
    sel = opening_below(labels, mask, groups, (rp.OTHER,))
    labels[sel] = rp.PROTECTED
    return int(sel.sum())


def opening_below(labels: np.ndarray, mask: np.ndarray, groups: np.ndarray, which) -> np.ndarray:
    """Pixels mascarados com rótulo em ``which`` (ou de parede, se ``which`` é None) cujo 1º pixel visível acima,
    na mesma coluna, é janela/cortina/porta."""
    other = mask & (np.isin(labels, which) if which is not None else labels >= 0)
    if not other.any():
        return other
    H, W = mask.shape
    vis = ~rp._dil(mask, 2)
    rows = np.where(vis, np.arange(H)[:, None], -1)
    last = np.maximum.accumulate(rows, axis=0)              # última linha visível acima (ou -1)
    ok = last >= 0
    g_above = np.full((H, W), -1, np.int64)
    cols = np.broadcast_to(np.arange(W)[None], (H, W))
    g_above[ok] = groups[last[ok], cols[ok]]
    return other & np.isin(g_above, rp.G_PROTECTED)


def apply_floor_lines(labels: np.ndarray, mask: np.ndarray, box, lines: dict) -> int:
    """Rótulos dentro da máscara entre a linha da caixa e a corrigida: abaixo da corrigida piso, acima parede."""
    H, _W = labels.shape
    yf, above = _box_floor_rows(box)
    rows = np.arange(H)[:, None]
    changed = 0
    for wid, (c0, c1, _) in lines.items():
        cols = np.nonzero(above == wid)[0]
        if cols.size == 0:
            continue
        yc = yf[cols] + c0 + c1 * cols
        sub = labels[:, cols]
        msk = mask[:, cols]
        cand = msk & ((sub == wid) | (sub == rp.FLOOR))
        new = np.where(rows >= yc[None], rp.FLOOR, wid)
        ch = cand & (new != sub)
        sub[ch] = new[ch]
        labels[:, cols] = sub
        changed += int(ch.sum())
    return changed


# ============================================================================================================
# preenchimento de um plano
# ============================================================================================================
def _warp_in(img, Hm, size, interp=cv2.INTER_LINEAR, border=cv2.BORDER_REFLECT):
    return cv2.warpPerspective(img, Hm, size, flags=interp | cv2.WARP_INVERSE_MAP, borderMode=border)


def _warp_mask(m, Hm, size, thr=127):
    return _warp_in(m.astype(np.uint8) * 255, Hm, size, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT) > thr


def _grain(img: np.ndarray, sel: np.ndarray) -> float:
    g = _lum(img)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    return _rstd(hp[sel])


def _band_std(alb: np.ndarray, known: np.ndarray, lo: float = 1.0, hi: float = 6.0) -> float:
    """Energia de textura: desvio robusto da banda média (DoG) do albedo no material visível (longe da borda)."""
    g = _lum(alb)
    kk = known.astype(np.float32)
    b1 = _nblur(g, known, lo)
    b2 = _nblur(g, known, hi)
    inner = rp._ero(known, int(2 * hi))
    if inner.sum() < 200:
        inner = known
    return _rstd((b1 - b2)[inner]) if kk.sum() else 0.0


def _add_grain(vals: np.ndarray, sel: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0.2 or not sel.any():
        return vals
    n = np.zeros(sel.shape, np.float32)
    n[sel] = rng.normal(0, 1, int(sel.sum()))
    n = cv2.GaussianBlur(n, (0, 0), 0.6)
    s = _rstd(n[sel]) + 1e-6
    out = vals.copy()
    out[sel] += (n[sel] * (sigma / s))[:, None]
    return out


def _height_map(Hm: np.ndarray, size, ppm: float, b1: float, line_pts: np.ndarray | None) -> np.ndarray:
    """Altura (m) acima da linha do piso em cada pixel da vista retificada de uma parede. Sem correção: b
    da caixa; com correção: a linha corrigida (pontos da imagem) levada para a vista e ajustada por uma reta."""
    tw, th = size
    vv, _uu = np.mgrid[0:th, 0:tw].astype(np.float32)
    v_line = np.full(tw, b1 * ppm, np.float32)
    if line_pts is not None and len(line_pts) >= 2:
        Hi = np.linalg.inv(Hm)
        p = np.c_[line_pts, np.ones(len(line_pts))] @ Hi.T
        ok = np.abs(p[:, 2]) > 1e-9
        uv = p[ok, :2] / p[ok, 2:3]
        good = np.isfinite(uv).all(1) & (uv[:, 0] > -tw) & (uv[:, 0] < 2 * tw)
        if good.sum() >= 2:
            c = np.polyfit(uv[good, 0], uv[good, 1], 1)
            v_line = (c[0] * np.arange(tw) + c[1]).astype(np.float32)
    return ((v_line[None, :] - vv) / ppm).astype(np.float32)


def wall_base(img: np.ndarray, known: np.ndarray, hmap: np.ndarray, step: float = 0.01,
              fallback: dict | None = None) -> dict | None:
    """Cor de base de uma parede: cor lisa (mediana acima do rodapé) e, se houver, o **rodapé**: faixa contígua a
    partir do piso com cor bem diferente da parede (até 0,30 m), com a cor mediana de cada altura dentro dela.
    ``fallback``: base das outras paredes (parede sem rodapé à vista usa o rodapé delas)."""
    hk = hmap[known]
    px = img[known]
    ok = (hk >= -0.02) & (hk <= 4.0)
    hk, px = hk[ok], px[ok]
    if hk.size < 80:
        return fallback
    above = hk >= 0.40
    flat = np.median(px[above], axis=0) if above.sum() >= 50 else np.median(px, axis=0)
    nb = int(0.32 / step) + 1
    idx = (hk / step).astype(np.int64)
    sel = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[sel], minlength=nb)
    prof = np.tile(flat, (nb, 1)).astype(np.float32)
    for k in np.nonzero(cnt >= 8)[0]:
        prof[k] = np.median(px[sel][idx[sel] == k], axis=0)
    good = cnt >= 8
    ref = rp._lab(flat[None, None])[0, 0]
    de = np.linalg.norm(rp._lab(prof[None])[0] - ref, axis=1)
    top, miss = 0.0, 0
    if good[: int(0.12 / step)].sum() >= 3:
        for k in range(nb):
            if not good[k]:
                miss += 1
                if miss > 3:
                    break
                continue
            if de[k] > 7.0:
                top = (k + 1) * step
            else:
                break
    base = {"flat": flat.astype(np.float32), "bb_h": 0.0, "bb": None, "visivel_baixo": bool(good[:5].sum() >= 2)}
    if 0.03 <= top <= 0.30:
        kk = round(top / step)
        bbp = prof[:kk].copy()
        gi = np.nonzero(good[:kk])[0]
        for k in range(kk):
            if not good[k]:
                bbp[k] = prof[gi[np.argmin(np.abs(gi - k))]]
        # o rodapé como razão sobre a cor da parede (vale em outra parede de outra cor/luz)
        base.update(bb_h=float(top), bb=bbp / np.maximum(flat, 1.0))
    elif fallback is not None and fallback.get("bb") is not None and not base["visivel_baixo"]:
        base.update(bb_h=fallback["bb_h"], bb=fallback["bb"])
    return base


def base_color(base: dict, h: np.ndarray, step: float = 0.01) -> np.ndarray:
    col = np.tile(base["flat"], (h.size, 1)).astype(np.float32)
    if base.get("bb") is not None:
        inb = (h < base["bb_h"]) & (h > -0.05)
        k = np.clip((h[inb] / step).astype(np.int64), 0, len(base["bb"]) - 1)
        col[inb] = base["bb"][k] * base["flat"][None]
    return col


class _Ctx:
    def __init__(self, photo, mask, groups, box, labels, rng, max_side, rug):
        self.photo, self.mask, self.groups, self.box, self.labels = photo, mask, groups, box, labels
        self.rng, self.max_side, self.rug = rng, max_side, rug
        self.H, self.W = mask.shape
        self.lab = rp._lab(photo)
        self.known_cache: dict = {}


def _plane_known(ctx: _Ctx, key: int, any_wall: bool = False) -> np.ndarray:
    ck = (key, any_wall)
    if ck not in ctx.known_cache:
        ctx.known_cache[ck] = _plane_known_raw(ctx, key, any_wall)
    return ctx.known_cache[ck]


def _plane_known_raw(ctx: _Ctx, key: int, any_wall: bool = False) -> np.ndarray:
    """Material visível do plano: mesmo rótulo da caixa, grupo da segmentação certo, longe da máscara, sem tapete
    (piso) nem objetinhos (tomada, quadro pequeno) e **com o croma do material dominante**: a caixa erra a divisa
    e um pedaço de outro material (painel de madeira ao lado da parede bege, porta) não pode tingir a parede.
    ``any_wall``: material de todas as paredes (parede com pouco à vista usa a luz/cor das vizinhas)."""
    grp = rp.G_FLOOR if key == rp.FLOOR else (rp.G_CEIL if key == rp.CEIL else rp.G_WALL)
    same = ~rp._dil(ctx.mask, 2) & (ctx.groups == grp)
    same &= (ctx.labels >= 0) if any_wall else (ctx.labels == key)
    if key == rp.FLOOR and ctx.rug is not None:
        same &= ~rp._dil(ctx.rug, 3)
    if key != rp.FLOOR:
        same &= ~rp.clutter_mask(ctx.photo, same)
    if key < 0:
        same &= _chroma_consistent(ctx.lab, same, ctx.mask, k=4.0, floor_de=12.0)
    return same


def _chroma_consistent(lab: np.ndarray, sel: np.ndarray, mask: np.ndarray, k: float = 3.0,
                       floor_de: float = 7.0) -> np.ndarray:
    """Pixels de ``sel`` cujo croma (a, b do Lab) fica perto do croma dominante do anel em volta da máscara
    (luz muda o L; outro material muda o croma)."""
    if sel.sum() < 200:
        return sel
    ring = sel & rp._dil(mask, max(12, int(0.04 * np.hypot(*mask.shape))))
    ref_sel = ring if ring.sum() >= 200 else sel
    ab = lab[..., 1:]
    med = np.median(ab[ref_sel], axis=0)
    d = np.linalg.norm(ab - med, axis=-1)
    mad = 1.4826 * float(np.median(d[ref_sel]))
    return sel & (d <= max(floor_de, k * mad))


def fill_plane(ctx: _Ctx, key: int, sel: np.ndarray, line_pts: np.ndarray | None = None,
               wall_fallback: dict | None = None, wall_low: np.ndarray | None = None,
               force_bb: dict | None = None) -> tuple[np.ndarray | None, np.ndarray, dict]:
    """Preenche os pixels ``sel`` (imagem) do plano ``key`` na vista retificada. Devolve (imagem de volta,
    pixels cobertos, info) ou (None, vazio, info) quando não há material visível do plano.

    Modelo: foto = base × luz × detalhe. Base: cor mediana do material (paredes: cor lisa + rodapé por altura).
    Luz: passa-baixa (convolução normalizada) da razão foto/base no visível, levada para dentro do buraco por
    membrana (push-pull): contínua com a borda, sem névoa. Detalhe (≈ 1): textura real, copiada por exemplos
    (grade/tábuas alinhadas ao período; paredes com a altura presa) ou, em superfície lisa, só o grão medido."""
    box = ctx.box
    plane = box.planes[key]
    is_wall = key >= 0
    pr = box.rect_params(plane, sel, ctx.max_side if is_wall else int(ctx.max_side * 1.5),
                         pad_px=60 if is_wall else 40, pct=0.5 if is_wall else 0.0, near_pct=50.0 if is_wall else 25.0)
    empty = np.zeros_like(sel)
    if pr is None:
        return None, empty, {"metodo": "sem_vista"}
    a0, b1, ppm, size = pr
    Hm = box.homography(plane, a0, b1, ppm)
    tw, th = size
    orig = _warp_in(ctx.photo, Hm, size).astype(np.float32)
    inside = _warp_in(np.full((ctx.H, ctx.W), 255, np.uint8), Hm, size, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT) > 0
    inside = rp._ero(inside, 1)
    known_img = _plane_known(ctx, key)
    known = _warp_mask(known_img, Hm, size, 250) & inside
    hole = rp._dil(_warp_mask(sel, Hm, size, 0), 2) & inside
    known &= ~hole
    nk = int(known.sum())
    info: dict = {"vista": [int(tw), int(th)], "visivel_px": nk}
    if is_wall and nk < MIN_KNOWN_PX and wall_low is None:
        # parede com pouco à vista (pedaço estreito da caixa): luz e cor das paredes visíveis em volta
        shared = _warp_mask(_plane_known(ctx, key, any_wall=True), Hm, size, 250) & inside & ~hole
        if shared.sum() > nk:
            known, nk = shared, int(shared.sum())
            info["luz_das_paredes_vizinhas"] = True
    if nk < MIN_KNOWN_PX and not (is_wall and (wall_fallback is not None or wall_low is not None)):
        info["metodo"] = "sem_contexto"
        return None, empty, info
    rng = ctx.rng
    hmap = _height_map(Hm, size, ppm, b1, line_pts) if is_wall else None
    # ---- base
    if is_wall:
        base = wall_base(orig, known, hmap, fallback=wall_fallback) if nk >= MIN_KNOWN_PX else wall_fallback
        if base is None and wall_low is not None:
            base = {"flat": np.full(3, 128.0, np.float32), "bb_h": 0.0, "bb": None}
        if base is None:
            info["metodo"] = "sem_contexto"
            return None, empty, info
        if force_bb is not None and base.get("bb") is None:
            base = {**base, "bb_h": force_bb["bb_h"], "bb": force_bb["bb"]}
        info["base"] = base
        info["rodape_m"] = round(base["bb_h"], 3)
        info["rodape_visivel"] = bool(base.get("visivel_baixo", True))
        B = base_color(base, hmap.ravel()).reshape(th, tw, 3)
    else:
        B = np.broadcast_to(np.median(orig[known], axis=0), orig.shape).astype(np.float32)
    ratio = orig / np.maximum(B, 1.0)
    flat = base["flat"][None, None] if is_wall else B
    # ---- textura e período (paredes: só na horizontal; a altura fica presa)
    per = detect_axis_periods(_lum(orig), known) if nk >= MIN_KNOWN_PX else (None, None)
    if is_wall:
        per = (None, per[1])
        if per[1]:
            # período horizontal só vale com pelo menos 3 repetições na largura visível (senão é uma quina/borda)
            kx = np.nonzero(known.any(0))[0]
            if kx.size == 0 or (kx.max() - kx.min()) < MIN_PERIODS * per[1]:
                info["periodo_descartado_px"] = round(float(per[1]), 1)
                per = (None, None)
    pmax = float(max([p for p in per if p] or [0]))
    sig0 = float(max(6.0, 0.025 * max(tw, th)))
    sig = float(max(sig0, 1.5 * pmax))
    # ---- luz: passa-baixa robusta da razão no visível, levada para dentro do buraco
    wl_rect = None if (wall_low is None or not is_wall) else _warp_in(wall_low, Hm, size) / np.maximum(flat, 1.0)
    if is_wall and wl_rect is not None and nk < MIN_KNOWN_PX:
        low = wl_rect
        detail = np.ones_like(orig)
    elif nk >= MIN_KNOWN_PX:
        if is_wall and base.get("bb") is not None:
            # a luz da parede vem de cima do rodapé (o rodapé entra pela base)
            up = known & (hmap >= base["bb_h"] + 0.03)
            lk = known if up.sum() < MIN_KNOWN_PX // 2 else up
        else:
            lk = known
        low_k, inl = robust_lowpass(ratio, lk, sig)
        if is_wall and wl_rect is not None and not pmax:
            # luz/cor da parede lisa: campo único no espaço da imagem (todas as paredes, a cor de cima continua
            # para baixo; atravessa as divisas falsas da caixa), relativo à cor lisa (o rodapé entra pela base)
            low = wl_rect
        elif is_wall:
            low = column_continue(low_k, inl, hole)       # parede com textura: a cor de cima continua para baixo
        else:
            low = floor_light(ratio, low_k, inl, hole, rng)
        lowd = low.copy()
        lowd[known] = _nblur(ratio, known, sig)[known]
        detail = ratio / np.maximum(lowd, 1e-3)
        low[~hole] = lowd[~hole]
    else:                                            # parede sem nada à vista: base das outras paredes, luz plana
        low = np.ones_like(orig)
        detail = np.ones_like(orig)
    grain = _grain(orig, rp._ero(known, 2) if rp._ero(known, 2).sum() > 200 else known) if nk else 1.0
    tex = _band_std(orig / np.maximum(low, 1e-3), known) if nk >= MIN_KNOWN_PX else 0.0
    info.update(grao=round(grain, 2), textura=round(tex, 2), sigma_luz=round(sig, 1))
    plain = nk < MIN_KNOWN_PX or (key != rp.FLOOR and tex < PLAIN_BAND_STD and not pmax)
    stripes_v = False
    if is_wall and nk >= MIN_KNOWN_PX and not pmax:
        # veio de madeira, painel ripado, frisos verticais: textura de escala média que corre na vertical, mesmo
        # sem período (a banda fina sozinha a chamava de lisa e o painel virava uma placa laranja chapada)
        tex2 = _band_std(orig / np.maximum(low, 1e-3), known, 3.0, 12.0)
        inv = axis_invariance(_lum(detail), known, 0, max(6, int(0.03 * th)))
        info.update(textura_larga=round(tex2, 2), inv_v=round(inv, 2))
        if inv > STRIPE_INV_MIN and max(tex, tex2) > STRIPE_TEX_MIN:
            plain, stripes_v = False, True
    det = detail.copy()
    info["periodo_px"] = [per[0], per[1]]
    if plain:
        det[hole] = 1.0
        info["metodo"] = "liso"
        noise_sel = hole
    else:
        # padrão periódico: rejunte/frisos são o material (não manchas); sem período, sai o que destoa
        src = known if pmax else clean_sources(detail, known)
        left = hole.copy()
        inv_ax = 0 if stripes_v else None
        if inv_ax is None and bool(per[0]) != bool(per[1]):
            ax = 0 if per[1] else 1                  # período em x -> listras correm em y (e vice-versa)
            lag = int(max(6, round(0.6 * (per[1] or per[0]))))
            if axis_invariance(_lum(detail), src, ax, lag) > 0.6:
                inv_ax = ax
        if inv_ax is not None:
            det, left = stripe_continue(detail, src, hole, inv_ax)
            info["metodo"] = "listras"
            info["listras_eixo"] = "vertical" if inv_ax == 0 else "horizontal"
        elif per[0] and per[1]:
            # grade nos dois eixos (porcelanato, azulejo): cada pixel copia o pixel real a um múltiplo exato da
            # grade (rejunte alinhado com o visível); o que não acha fonte vai para os retalhos
            det, left = periodic_copy(detail, src, hole, per, reach=max(14, int(max(th, tw) / min(per)) + 2))
            info["metodo"] = "grade"
        if left.any():
            lock = hmap * ppm if is_wall else None
            known_q = src | (hole & ~left)
            det, left, qinfo = quilt(det, known_q, left, rng, period=per, row_lock=lock)
            info.update(**qinfo)
            info.setdefault("metodo", "exemplos")
        det[left] = 1.0
        noise_sel = left
        if left.any():
            info["sobra_px"] = int(left.sum())
    filled = B * low * det
    if noise_sel.any():
        filled = _add_grain(filled, noise_sel, grain, rng)
    f8 = np.clip(filled, 0, 255).round().astype(np.uint8)
    o8 = np.clip(orig, 0, 255).round().astype(np.uint8)
    f8[~hole] = o8[~hole]
    if nk >= MIN_KNOWN_PX:
        f8 = harmonize_tone(f8, known, hole, sigma=max(3.0, 0.6 * pmax), band=8)
    back = cv2.warpPerspective(f8, Hm, (ctx.W, ctx.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cover = cv2.warpPerspective(rp._ero(inside, 2).astype(np.uint8) * 255, Hm, (ctx.W, ctx.H),
                                flags=cv2.INTER_NEAREST) > 0
    return back, cover & sel, info


def harmonize_tone(img: np.ndarray, known: np.ndarray, hole: np.ndarray, sigma: float = 3.0, band: int = 8,
                   max_corr: float = 30.0) -> np.ndarray:
    """Membrana de tom na borda (como :func:`plate.harmonize`), com o passa-baixa na escala da textura (o rejunte
    e os frisos não viram correção) e a correção limitada a ``max_corr`` níveis (nunca um borrão claro/escuro)."""
    if not hole.any() or known.sum() < 30:
        return img
    f = img.astype(np.float32)
    lk = _nblur(f, known, sigma)
    lf = _nblur(f, hole, sigma)
    inner = hole & rp._dil(known, band)
    if inner.sum() < 10:
        return img
    d = np.zeros_like(f)
    d[inner] = np.clip((lk - lf)[inner], -max_corr, max_corr)
    corr = rp.push_pull(d, inner.astype(np.float32))
    corr = _nblur(corr, hole, 2.0)
    f[hole] += corr[hole]
    out = img.copy()
    out[hole] = f[hole].round().clip(0, 255).astype(np.uint8)
    return out


def floor_light(ratio: np.ndarray, low_k: np.ndarray, inl: np.ndarray, hole: np.ndarray,
                rng: np.random.Generator, decay: float = 0.08) -> np.ndarray:
    """Luz do piso/teto dentro do buraco: campo global suave (polinômio robusto) e a diferença medida na borda
    levada para dentro por membrana **que se apaga com a distância** (``decay`` × lado da vista). Reflexo escuro de
    móvel ou sombra que tocam a borda não viram uma mancha no meio do piso."""
    h, w = inl.shape
    L = illumination_field(ratio * 100.0, inl, rng)
    col = np.median(ratio[inl] / L[inl][:, None], axis=0)
    glob = L[..., None] * col[None, None]
    ring = inl & rp._dil(hole, max(6, int(0.02 * max(h, w))))
    if ring.sum() < 30:
        return glob
    # só um lado: o escuro da borda (reflexo/sombra de contato do móvel) não entra no buraco além de um pouco
    d = np.where(ring[..., None], np.maximum(low_k - glob, -FLOOR_DARK_MAX * glob), 0.0).astype(np.float32)
    corr = rp.push_pull(d, ring.astype(np.float32))
    corr = _nblur(corr, np.ones_like(inl), 3.0)
    dist = ndimage.distance_transform_edt(hole)
    fade = np.exp(-dist / max(8.0, decay * max(h, w))).astype(np.float32)
    out = glob + corr * fade[..., None]
    # nunca mais claro/escuro que a luz vista no próprio material (o polinômio extrapola nas bordas da vista)
    lo = np.percentile(low_k[inl], 1, axis=0)
    hi = np.percentile(low_k[inl], 99, axis=0)
    return np.clip(out, lo[None, None], hi[None, None])


def robust_lowpass(img: np.ndarray, known: np.ndarray, sig: float, lo: float = 0.8, hi: float = 1.25):
    """Passa-baixa (convolução normalizada) só sobre os pixels "típicos" do material: sai o que foge muito da
    luminância local larga (reflexo escuro de móvel no piso polido, sombra de contato, mancha de sol pequena).
    Devolve (passa-baixa, pixels usados)."""
    g = _lum(img * 100.0) + 1e-3
    coarse = _nblur(g, known, 3.0 * sig)
    r = g / np.maximum(coarse, 1e-3)
    inl = known & (r > lo) & (r < hi)
    if inl.sum() < 0.4 * known.sum():
        inl = known
    return _nblur(img, inl, sig), inl


def column_continue(low: np.ndarray, known: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Luz/cor de parede para dentro do buraco: em cada coluna, interpolação na vertical entre o visível acima e
    abaixo (só um lado: constante); colunas sem nada visível: interpolação na horizontal, linha a linha."""
    h, w = known.shape
    out = low.copy()
    rows = np.arange(h, dtype=np.float32)
    have = np.zeros(w, bool)
    for x in range(w):
        kx = np.nonzero(known[:, x])[0]
        hx = hole[:, x]
        if kx.size < 3 or not hx.any():
            have[x] = kx.size >= 3
            continue
        have[x] = True
        for c in range(low.shape[2]):
            out[hx, x, c] = np.interp(rows[hx], kx.astype(np.float32), low[kx, x, c])
    cols = np.nonzero(have)[0]
    miss = np.nonzero(~have)[0]
    if cols.size >= 2 and miss.size:
        for y in np.nonzero(hole[:, miss].any(1))[0]:
            sel = miss[hole[y, miss]]
            for c in range(low.shape[2]):
                out[y, sel, c] = np.interp(sel.astype(np.float32), cols.astype(np.float32), out[y, cols, c])
    elif cols.size < 2:
        out = rp.push_pull(np.where(known[..., None], low, 0.0), known.astype(np.float32))
    return out


def wall_light_field(ctx: _Ctx) -> np.ndarray | None:
    """Cor/luz de baixa frequência das paredes no espaço da imagem: passa-baixa robusta de todas as paredes
    visíveis (acima da faixa do rodapé), levada para dentro da máscara coluna a coluna (o que se vê acima do móvel
    continua para baixo; entre visível acima e abaixo, interpolação). Atravessa as divisas falsas da caixa (uma
    parede real partida em duas) e segue as quinas reais vistas acima do móvel."""
    m, g, lab = ctx.mask, ctx.groups, ctx.labels
    hole = m & (lab >= 0)
    if not hole.any():
        return None
    wk = ~rp._dil(m, 2) & (g == rp.G_WALL) & (lab >= 0)
    wk &= ~rp.clutter_mask(ctx.photo, wk)
    Z = ctx.box.Z
    up = wk & ~(np.isfinite(Z) & (Z < 0.30))
    if up.sum() >= 0.5 * wk.sum():
        wk = up
    if wk.sum() < 300:
        return None
    diag = float(np.hypot(ctx.H, ctx.W))
    sig = max(4.0, 0.012 * diag)
    # faixa larga: a parede que escurece de verdade (canto, oclusão ao lado do móvel) é luz que a borda lateral do
    # preenchimento precisa casar; só manchas fortes (tomada, sombra dura) saem
    low_k, inl = robust_lowpass(ctx.photo.astype(np.float32), wk, sig, lo=0.6, hi=1.4)
    # assimétrico perto do móvel: a sombra/oclusão que o móvel faz na parede (fora da máscara) não é luz da
    # parede; escurecer além de WALL_AO_DARK em relação ao entorno largo sai, clarear pode ficar
    # só acima do móvel (é o que desce coluna a coluna para dentro do buraco); ao lado fica: é a luz que a borda
    # lateral do preenchimento precisa casar
    rr = max(6, int(WALL_AO_RING * diag))
    m_below = np.zeros_like(m)
    for k in range(1, rr + 1, max(1, rr // 8)):
        m_below[:-k] |= m[k:]
    ring = rp._dil(m, rr) & inl & m_below
    if ring.any():
        far = inl & ~ring
        if far.sum() >= 500:
            ref = _lum(_nblur(ctx.photo.astype(np.float32), far, 3.0 * sig))
            dark = ring & (_lum(low_k) < (1.0 - WALL_AO_DARK) * ref)
            if dark.any() and (inl & ~dark).sum() >= 300:
                inl = inl & ~rp._dil(dark, 2)
                low_k = _nblur(ctx.photo.astype(np.float32), inl, sig)
    # gradiente de luz suave (polinômio robusto em toda a parede visível) × resíduo continuado coluna a coluna: a
    # parede clareando para cima não deixa o contorno do sofá desenhado (cada coluna parte de uma altura)
    P = np.ones((ctx.H, ctx.W), np.float32)
    glob = illumination_field(ctx.photo, inl, ctx.rng, deg=1)
    P[:] = glob
    for wl in ctx.box.room.walls:          # gradiente linear por parede (o quadrático global extrapolava demais)
        sel_l = lab == wl.id
        if (inl & sel_l).sum() >= 2000:
            Pi = illumination_field(ctx.photo, inl & sel_l, ctx.rng, deg=1)
            # mesma escala do global no visível da parede (a média geométrica 1 é por ajuste)
            k = np.median(glob[inl & sel_l] / Pi[inl & sel_l])
            P[sel_l] = (Pi * k)[sel_l]
    # a divisa entre paredes da caixa pode ser falsa (uma parede real partida em fatias): o gradiente por parede
    # passa suave de uma para outra, nunca em degrau dentro do buraco
    P = cv2.GaussianBlur(P, (0, 0), 0.03 * diag)
    P = P[..., None]
    res = column_continue(low_k / P, inl, hole).astype(np.float32)
    # o resíduo da borda (luz rebatida do móvel claro, mancha) se apaga com a distância dentro do buraco: no fundo
    # fica o gradiente da parede ajustado em todo o visível (sem a silhueta do sofá desenhada em tom)
    # longe da borda de cima: membrana com TODA a borda (de cima e dos lados): a parede que escurece para baixo ao
    # lado do sofá continua escurecendo atrás dele, sem degrau no contorno
    r0 = low_k / P
    band = inl & rp._dil(hole, max(6, int(0.02 * diag)))
    mem = rp.push_pull(np.where(band[..., None], r0, 0.0).astype(np.float32), band.astype(np.float32)) \
        if band.sum() >= 50 else np.broadcast_to(np.median(r0[inl], axis=0), r0.shape)
    dist = ndimage.distance_transform_edt(hole)
    fade = np.exp(-dist / max(8.0, WALL_RES_DECAY * diag)).astype(np.float32)[..., None]
    res = np.where(hole[..., None], mem + (res - mem) * fade, res).astype(np.float32)
    res = smooth_inside(res, hole, 3.0 * sig, 0.5 * sig)
    return res * P


def smooth_inside(img: np.ndarray, region: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Borra só dentro de ``region`` (convolução normalizada, núcleo anisotrópico): colunas vizinhas que
    continuaram de alturas diferentes não deixam degrau vertical."""
    w = region.astype(np.float32)
    num = cv2.GaussianBlur(img * w[..., None], (0, 0), sigmaX=sx, sigmaY=sy)
    den = cv2.GaussianBlur(w, (0, 0), sigmaX=sx, sigmaY=sy)[..., None]
    out = img.copy()
    sm = num / np.maximum(den, 1e-6)
    out[region] = sm[region]
    return out


def clean_sources(detail: np.ndarray, known: np.ndarray, k: float = 2.5) -> np.ndarray:
    """Material visível bom para servir de retalho: sem manchas de média frequência (reflexo de móvel no piso
    polido, mancha, sombra de pé de cadeira) que, copiadas, virariam borrões no meio do piso."""
    g = _lum(detail * 100.0)
    mid = _nblur(g, known, 4.0) - _nblur(g, known, 20.0)
    v = mid[known]
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med))) + 1e-3
    bad = known & (np.abs(mid - med) > k * mad)
    good = known & ~rp._dil(bad, 4)
    return good if good.sum() >= 0.5 * known.sum() else known


def extend_fill(out: np.ndarray, photo: np.ndarray, region: np.ndarray, source: np.ndarray,
                rng: np.random.Generator) -> None:
    """Sobra sem vista (canto da foto fora da vista retificada, fiapo entre planos): continua as cores já
    reconstruídas/visíveis em volta (``source``) por membrana, com grão. Nada novo aparece."""
    if not region.any():
        return
    H, W = region.shape
    n, cc, st, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x, y, w, h, _ = st[k]
        r = max(8, int(0.25 * max(w, h)))
        x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(W, x + w + r), min(H, y + h + r)
        comp = cc[y0:y1, x0:x1] == k
        src = source[y0:y1, x0:x1] & ~comp
        if src.sum() < 5:
            continue
        img = out[y0:y1, x0:x1].astype(np.float32)
        val = rp.push_pull(_nblur(img, src, 2.0), src.astype(np.float32))
        g = _grain(img, src) if src.sum() > 50 else 1.0
        val = _add_grain(val, comp, min(g, 4.0), rng)
        sub = out[y0:y1, x0:x1]
        sub[comp] = np.clip(val[comp], 0, 255).round().astype(np.uint8)


def fine_detail(out: np.ndarray, photo: np.ndarray, sel: np.ndarray, known: np.ndarray, dist: np.ndarray,
                rng: np.random.Generator, sigma: float = 2.5) -> tuple[np.ndarray, dict]:
    """Repõe o detalhe fino (passa-alta, ``sigma``) que falta em ``sel``, copiado por retalhos do visível
    ``known`` à mesma distância da câmera (``dist``: travada como a altura nas paredes)."""
    f = out.astype(np.float32)
    p = photo.astype(np.float32)
    hp_src = p - cv2.GaussianBlur(p, (0, 0), sigma)
    hp_cur = f - cv2.GaussianBlur(f, (0, 0), sigma)
    k_in = rp._ero(known, 2)
    # só o grão típico: fora estampas, bordas de tapete, pés de cadeira (energia local muito acima da mediana)
    en = cv2.GaussianBlur(np.abs(_lum(hp_src)), (0, 0), 4.0)
    if k_in.sum() >= 2000:
        med_en = float(np.median(en[k_in]))
        k_in &= ~rp._dil((en > 1.6 * med_en + 0.5) | (en < 0.5 * med_en), 6)
    if k_in.sum() < 2000:
        return out, {}
    want = _rstd(_lum(hp_src)[k_in])
    have = _rstd(_lum(hp_cur)[rp._ero(sel, 2)] if rp._ero(sel, 2).any() else _lum(hp_cur)[sel])
    # só para piso de textura fina de verdade (granilite, carpete, madeira com veio) e com material visível de sobra;
    # piso liso (porcelanato, cimento queimado) fica só com o grão: retalho repetido viraria "carpete"
    if want <= 1.2 * have + 0.3 or want < FINE_DETAIL_MIN or k_in.sum() < 0.25 * sel.sum():
        return out, {}
    lock = np.log(np.clip(np.nan_to_num(dist, nan=50.0, posinf=50.0), 0.05, 50.0)).astype(np.float32) * 200.0
    ys, xs = np.nonzero(sel | k_in)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = (slice(y0, y1), slice(x0, x1))
    src = np.where(k_in[..., None], hp_src, 0.0)[sub]
    q, left, _ = quilt(src, k_in[sub], sel[sub], rng, row_lock=lock[sub], block=24, n_cand=96)
    hq = np.zeros_like(f)
    hq[sub] = q
    hs = sel & ~np.pad(left, ((y0, f.shape[0] - y1), (x0, f.shape[1] - x1)))
    got = _rstd(_lum(hq)[hs]) + 1e-3
    gain = float(min(1.5, np.sqrt(max(want ** 2 - have ** 2, 0.0)) / got))
    f[hs] += gain * hq[hs]
    res = out.copy()
    res[sel] = np.clip(f[sel], 0, 255).round().astype(np.uint8)
    return res, {"antes": round(have, 2), "alvo": round(want, 2), "ganho": round(gain, 2)}


def corrected_floor_rows(box, lines: dict) -> np.ndarray:
    """Por coluna: linha (y, float) da divisa parede/piso da caixa já corrigida pela segmentação (H = sem piso)."""
    yf, above = _box_floor_rows(box)
    yc = yf.astype(np.float32)
    for wid, (c0, c1, _) in lines.items():
        cols = np.nonzero(above == wid)[0]
        yc[cols] = yf[cols] + c0 + c1 * cols
    return yc


def _opening_bottom_gap(vis_kind: np.ndarray, mask: np.ndarray, floor_rows: np.ndarray, x0: int, x1: int,
                        top: int, bot: int) -> float | None:
    """Altura (px) do peitoril da abertura acima da linha do piso, medida nas colunas vizinhas ao buraco onde a
    abertura aparece inteira (o fim de baixo dela não encosta na máscara). ``0`` = vai até o chão; ``None`` =
    nada medido."""
    H, W = mask.shape
    R = max(10, int(0.6 * (x1 - x0 + 1)))
    near_m = rp._dil(mask, 3)
    gaps = []
    for x in list(range(max(0, x0 - R), x0)) + list(range(x1 + 1, min(W, x1 + 1 + R))):
        k = np.nonzero(vis_kind[:, x])[0]
        if k.size < 5 or k.min() > bot or k.max() < top:
            continue
        b = int(k.max())
        if near_m[min(H - 1, b + 1):min(H, b + 4), x].any() or b >= H - 2:
            continue                                   # o fim dela está escondido (ou é a borda da foto)
        fr = floor_rows[x]
        if fr >= H:
            continue
        gaps.append(float(fr - b - 1))
    if len(gaps) < 3:
        return None
    g = float(np.median(gaps))
    return 0.0 if g < 0.02 * H else g


def continue_openings(out: np.ndarray, photo: np.ndarray, region: np.ndarray, mask: np.ndarray,
                      groups: np.ndarray, rng: np.random.Generator, filled: np.ndarray | None = None,
                      floor_rows: np.ndarray | None = None, zmap: np.ndarray | None = None,
                      optional: np.ndarray | None = None) -> tuple[int, np.ndarray]:
    """Pedaço escondido de janela, porta ou cortina: continua os pixels **reais** da abertura vistos acima/abaixo.

    A decisão é **por componente** do buraco (não por coluna): todas as colunas de uma componente param na mesma
    reta, então a barra de baixo não fica serrilhada. Cortina: cai até a linha do piso (as dobras seguem). Vidro e
    porta: se as colunas vizinhas mostram o fim da abertura, param à mesma altura sobre o piso (peitoril, ou o
    chão); sem essa medida, vão até o chão quando o vidro visto já desce abaixo de :data:`SILL_MIN_M` (porta de
    correr, janela de piso a teto) e, senão, até a altura típica de peitoril :data:`SILL_TYPICAL_M` (``zmap``:
    altura de cada pixel sobre o piso, da caixa). O vidro continua refletindo, na vertical, a última faixa vista (as barras
    do guarda-corpo e os caixilhos seguem retos), a cortina continua a última linha. Visível acima e abaixo:
    ponte. Nada é inventado. O resto (abaixo do peitoril) é parede: continuação da superfície vizinha, exceto em
    ``optional`` (parede já refeita pelo plano, candidata a abertura): ali só entra o que a abertura cobre.
    Devolve (pixels do resto, pixels refeitos como abertura)."""
    H, W = region.shape
    vis = ~rp._dil(mask, 2) & np.isin(groups, rp.G_PROTECTED)
    curtain = groups == rp.GROUP_IDS["window_covering"]
    opt = np.zeros_like(region) if optional is None else (optional & region)
    if vis.sum() < 20:
        extend_fill(out, photo, region & ~opt, ~mask, rng)
        return int((region & ~opt).sum()), np.zeros_like(region)
    if floor_rows is None:
        floor_rows = np.full(W, float(H), np.float32)
    src = cv2.GaussianBlur(photo.astype(np.float32), (0, 0), 0.7)
    rows = np.arange(H, dtype=np.float32)
    done = np.zeros_like(region)
    res = out.astype(np.float32)
    band_max = max(6, int(0.06 * H))
    # linha do piso por coluna: a da caixa, a menos que ela fique acima da abertura (caixa sem parede do fundo, o
    # "piso" sobe até o horizonte): aí o 1º piso visível abaixo do buraco (ou a borda da foto)
    vis_floor = ~rp._dil(mask, 2) & (groups == rp.G_FLOOR)
    floor_rows = floor_rows.astype(np.float32).copy()
    ys_r, xs_r = np.nonzero(region)
    top_r = np.full(W, H, np.int64)
    np.minimum.at(top_r, xs_r, ys_r)
    bot_r = np.full(W, -1, np.int64)
    np.maximum.at(bot_r, xs_r, ys_r)
    for x in np.nonzero(bot_r >= 0)[0]:
        if floor_rows[x] <= top_r[x]:
            fv = np.nonzero(vis_floor[bot_r[x]:, x])[0]
            floor_rows[x] = float(bot_r[x] + fv[0]) if fv.size else float(H)
    n, cc, st, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
    for ci in range(1, n):
        x0, y0c, wc, _hc, _ = st[ci]
        x1 = x0 + wc - 1
        comp = cc == ci
        cols = np.nonzero(comp.any(0))[0]
        # tipo da abertura: o pixel visível logo acima de cada coluna
        n_cur = n_any = 0
        tops = []
        for x in cols:
            k = np.nonzero(vis[:, x])[0]
            ys = np.nonzero(comp[:, x])[0]
            ab = k[k < ys.min()]
            if ab.size:
                n_any += 1
                n_cur += bool(curtain[ab.max(), x])
                tops.append(int(ab.min()))
        if n_any == 0:
            continue
        is_curtain = n_cur > 0.5 * n_any
        gap, z_stop = 0.0, None
        if not is_curtain:
            kind = vis & ~curtain
            g = _opening_bottom_gap(kind, mask, floor_rows, x0, x1, int(np.median(tops)), y0c)
            if g is not None:
                gap = g
            elif zmap is not None:
                zb = []
                for x in cols:
                    k = np.nonzero(vis[:, x])[0]
                    ys = np.nonzero(comp[:, x])[0]
                    ab = k[k < ys.min()]
                    if ab.size and np.isfinite(zmap[ab.max(), x]):
                        zb.append(float(zmap[ab.max(), x]))
                if zb and float(np.median(zb)) >= SILL_MIN_M:
                    z_stop = max(SILL_TYPICAL_M, float(np.median(zb)) - 0.15)
        for x in cols:
            k = np.nonzero(vis[:, x])[0]
            ys = np.nonzero(comp[:, x])[0]
            if k.size < 3:
                continue
            above = k[k < ys.min()]
            below = k[k > ys.max()]
            if below.size and above.size:                 # visível acima e abaixo: ponte
                sel = ys
                for c in range(3):
                    res[sel, x, c] = np.interp(rows[sel], k.astype(np.float32), src[k, x, c])
                done[sel, x] = True
                continue
            if not above.size:                            # só abaixo: continua pouco (raro: soleira vista)
                sel = ys[ys >= below.min() - band_max] if below.size else ys[:0]
                for c in range(3):
                    res[sel, x, c] = np.interp(rows[sel], k.astype(np.float32), src[k, x, c])
                done[sel, x] = True
                continue
            stop = floor_rows[x] - gap if floor_rows[x] < H else H
            sel = ys[ys < stop]
            if z_stop is not None:
                zz = zmap[sel, x]
                sel = sel[~(np.isfinite(zz) & (zz < z_stop))]
            if sel.size == 0:
                continue
            ya = int(above.max())
            if is_curtain:
                res[sel, x] = src[ya, x]
            else:
                # última faixa contígua vista acima, refletida para baixo (onda triangular): sem emenda
                run = 1
                while run < band_max and ya - run >= 0 and vis[ya - run, x]:
                    run += 1
                d = sel - ya
                t = np.mod(d, 2 * run)
                srow = np.where(t < run, ya - t, ya - (2 * run - t))
                srow = np.clip(srow, ya - run + 1, ya)
                res[sel, x] = src[srow, x]
            done[sel, x] = True
    if done.any():
        # emenda horizontal entre colunas (fiapo de coluna sem fonte) e grão da abertura
        g = _grain(photo, vis)
        res = _add_grain(res, done, min(g, 3.0), rng)
        out[done] = np.clip(res[done], 0, 255).round().astype(np.uint8)
    rest = region & ~done & ~opt
    if rest.any():
        # o resto é parede/superfície (abaixo do peitoril, coluna sem nada à vista): continua a superfície vizinha
        # (visível ou já refeita), nunca o vidro
        ok = (~mask & np.isin(groups, (rp.G_WALL, rp.G_FLOOR))) | (filled if filled is not None else False)
        ok = ok & ~(done & opt) if isinstance(ok, np.ndarray) else ok
        extend_fill(out, photo, rest, ok, rng)
    return int(rest.sum()), done


def neutral_fill(out: np.ndarray, photo: np.ndarray, region: np.ndarray, mask: np.ndarray,
                 rng: np.random.Generator, ok: np.ndarray | None = None) -> None:
    """Cor neutra do entorno, lisa (membrana: interpolação suave das cores do anel visível) + grão do anel. Para o
    que não tem plano confiável: nada de conteúdo inventado."""
    if not region.any():
        return
    H, W = region.shape
    n, cc, st, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x, y, w, h, _ = st[k]
        r = max(6, int(0.15 * max(w, h)))
        x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(W, x + w + r), min(H, y + h + r)
        comp = cc[y0:y1, x0:x1] == k
        ring = rp._dil(comp, r) & ~mask[y0:y1, x0:x1]
        if ok is not None and (ring & ok[y0:y1, x0:x1]).sum() >= 10:
            ring &= ok[y0:y1, x0:x1]
        if ring.sum() < 10:
            ring = ~comp
        img = out[y0:y1, x0:x1].astype(np.float32)
        low = _nblur(img, ring, max(3.0, 0.05 * max(w, h)))
        val = rp.push_pull(low, ring.astype(np.float32))
        g = _grain(photo[y0:y1, x0:x1], ring) if ring.sum() > 50 else 1.0
        val = _add_grain(val, comp, min(g, 4.0), rng)
        sub = out[y0:y1, x0:x1]
        sub[comp] = np.clip(val[comp], 0, 255).round().astype(np.uint8)


# ============================================================================================================
# ponto de entrada
# ============================================================================================================
def fill_geometry(photo: np.ndarray, mask: np.ndarray, groups: np.ndarray, box, depth: np.ndarray | None = None,
                  rng: np.random.Generator | None = None, inpaint=None, max_side: int = rp.MAX_SIDE,
                  rug: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sala vazia pela geometria (ver o docstring do módulo). Mesma assinatura de saída de
    :func:`plate.fill_planes`: (placa, rótulos, info por plano). ``placa[~mask] == foto[~mask]``."""
    rng = rng or np.random.default_rng(0)
    H, W = mask.shape
    labels, info = rp.assign_labels(mask, groups, box, depth, wall_to_floor=False)
    far = far_floor(mask, groups, box, labels)
    if far.any():
        labels[far] = rp.OTHER
        info["piso_alem_do_visivel_px"] = int(far.sum())
    n_op = other_to_opening(labels, mask, groups)
    if n_op:
        info["outro_virou_abertura_px"] = n_op
    lines = floor_lines(mask, groups, box)
    if lines:
        n = apply_floor_lines(labels, mask, box, lines)
        info["linha_piso_corrigida"] = {f"parede{k}": {"desloc_px": round(v[0], 1), "incl": round(v[1], 4),
                                                       "pontos": v[2]} for k, v in lines.items()}
        info["linha_piso_px_trocados"] = n
        # protegidos continuam protegidos (a troca só mexe em piso/parede)
    out = photo.copy()
    done_all = np.zeros((H, W), bool)
    ctx = _Ctx(photo, mask, groups, box, labels, rng, max_side, rug)

    # ---- piso e teto
    for key in (rp.FLOOR, rp.CEIL):
        sel = mask & (labels == key)
        if sel.sum() < 20:
            continue
        back, cover, pinfo = fill_plane(ctx, key, sel)
        if back is not None:
            out[cover] = back[cover]
            done_all |= cover
        info[rp.LABEL_NAMES[key]] = {**pinfo, "area_px": int(cover.sum())}

    wall_low = wall_light_field(ctx)
    # ---- paredes: 1ª passada mede os perfis (para a parede sem nada à vista usar o das outras)
    yf, above = _box_floor_rows(box)
    wall_sel = {w.id: mask & (labels == w.id) for w in box.room.walls}
    line_pts = {}
    for w in box.room.walls:
        cols = np.nonzero(above == w.id)[0]
        if cols.size >= 2:
            c0, c1, _ = lines.get(w.id, (0.0, 0.0, 0))
            xs = np.linspace(cols.min(), cols.max(), 16)
            ys = np.interp(xs, cols, yf[cols]) + c0 + c1 * xs
            line_pts[w.id] = np.c_[xs + 0.5, ys - 0.5]
    results = {}
    bases = []
    order = sorted(box.room.walls, key=lambda w: -int(_plane_known(ctx, w.id).sum()))
    for w in order:
        sel = wall_sel[w.id]
        if sel.sum() < 20:
            continue
        fb = bases[0] if bases else None
        back, cover, pinfo = fill_plane(ctx, w.id, sel, line_pts.get(w.id), fb, wall_low)
        base = pinfo.pop("base", None)
        if base is not None and pinfo.get("visivel_px", 0) >= MIN_KNOWN_PX:
            bases.append(base)
        results[w.id] = (back, cover, pinfo)
    # parede sem rodapé à vista (escondido pelo sofá de ponta a ponta) usa o rodapé das outras paredes
    with_bb = [b for b in bases if b.get("bb") is not None]
    if with_bb:
        for w in order:
            if w.id not in results:
                continue
            back, cover, pinfo = results[w.id]
            if back is None or pinfo.get("rodape_m", 0) > 0 or pinfo.get("rodape_visivel", True):
                continue
            back2, cover2, pinfo2 = fill_plane(ctx, w.id, wall_sel[w.id], line_pts.get(w.id), with_bb[0], wall_low,
                                               force_bb=with_bb[0])
            pinfo2.pop("base", None)
            pinfo2["rodape_das_outras_paredes"] = True
            results[w.id] = (back2, cover2, pinfo2)
    for wid, (back, cover, pinfo) in results.items():
        if back is not None:
            out[cover] = back[cover]
            done_all |= cover
        info[f"parede{wid}"] = {**pinfo, "area_px": int(cover.sum()), "buraco_px": int(wall_sel[wid].sum())}

    surf = np.isin(groups, (rp.G_FLOOR, rp.G_WALL, rp.G_CEIL) + rp.G_PROTECTED)
    # ---- protegidos: continuação local da abertura (inpainter com contexto local), nunca fora da máscara
    prot = mask & (labels == rp.PROTECTED) & ~done_all
    # parede (já refeita) logo abaixo de janela/cortina/porta visível: candidata; a abertura decide se desce
    cand = opening_below(labels, mask, groups, None) & done_all
    if prot.any() or cand.any():
        n, got = continue_openings(out, photo, prot | cand, mask, groups, rng, done_all & ~cand,
                                   floor_rows=corrected_floor_rows(box, lines),
                                   zmap=np.where(box.label >= 0, box.Z, np.nan), optional=cand)
        moved = got & cand
        labels[moved] = rp.PROTECTED
        info["protegido"] = {"metodo": "continuacao_vertical", "area_px": int((prot | moved).sum()),
                             "sem_coluna_px": n, "parede_virou_abertura_px": int(moved.sum())}
        done_all |= prot
    # ---- "outro" (sem plano confiável): cor neutra do entorno; sobras sem vista: continuação dos vizinhos
    rest = mask & ~done_all
    if rest.any():
        other = rest & (labels == rp.OTHER)
        leftover = rest & ~other
        if other.any():
            neutral_fill(out, photo, other, mask & ~done_all, rng, ok=surf)
            info["outro"] = {"metodo": "neutro", "area_px": int(other.sum())}
            done_all |= other
        if leftover.any():
            extend_fill(out, photo, leftover, (~mask & surf) | done_all, rng)
            info["sobras"] = {"metodo": "continuacao", "area_px": int(leftover.sum())}
    # ---- detalhe fino do piso no espaço da imagem (a vista de cima tem menos pixels que a foto perto da câmera):
    # passa-alta real do piso visível, por retalhos tirados da MESMA distância (mesma escala de perspectiva)
    fsel = mask & (labels == rp.FLOOR) & done_all
    fper = (info.get(rp.LABEL_NAMES[rp.FLOOR]) or {}).get("periodo_px") or [None, None]
    if fsel.sum() >= 500 and not any(fper):          # padrão periódico: o detalhe solto desalinharia a grade
        fk = _plane_known(ctx, rp.FLOOR)
        out, fd = fine_detail(out, photo, fsel, fk, box.t, rng)
        if fd:
            info.setdefault("piso", {})["detalhe_fino"] = fd
    # ---- grão casado com a foto, por plano (a ida e volta da vista retificada suaviza um pouco)
    for key in [rp.FLOOR, rp.CEIL] + [w.id for w in box.room.walls]:
        sel = mask & (labels == key) & done_all
        if sel.sum() < 200:
            continue
        ring = _plane_known(ctx, key) & rp._dil(sel, max(8, int(0.03 * np.hypot(H, W))))
        if ring.sum() < 500:
            continue
        want, have = _grain(photo, ring), _grain(out, rp._ero(sel, 2) if rp._ero(sel, 2).sum() > 100 else sel)
        if want > 1.15 * have + 0.3:
            add = float(np.sqrt(max(want ** 2 - have ** 2, 0.0)))
            f = _add_grain(out.astype(np.float32), sel, add, rng)
            out[sel] = np.clip(f[sel], 0, 255).round().astype(np.uint8)
            info.setdefault("grao", {})[rp.LABEL_NAMES.get(key, f"parede{key}")] = [round(have, 2), round(want, 2)]
    out[~mask] = photo[~mask]
    return out, labels, info
