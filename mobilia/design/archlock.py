"""Trava da arquitetura: janelas, portas (e, no "Mobiliar", o teto) da foto original voltam por cima da imagem
gerada, com borda suave e casamento leve de cor/brilho.

O SDXL redesenha o ambiente inteiro; mesmo com o ControlNet de profundidade ele às vezes desloca, apaga ou duplica
uma janela. Aqui a abertura é segmentada (OneFormer, grupos ``window``/``door`` de ``analysis/segmentation.py``) e,
depois da geração, os pixels ORIGINAIS dela são compostos de volta:

* a máscara é limpa (componentes minúsculos fora, buracos fechados) e a composição usa uma rampa pela distância à
  borda (``feather_alpha``): alpha = 1 dentro, 0 fora da faixa, nada de halo além dela;
* antes de colar, a abertura original recebe um casamento de cor de baixa frequência com o que o gerador pôs no
  mesmo lugar (``match_color``, limitado), para a vista da janela não "brigar" com a luz nova do ambiente.

As mesmas rotinas servem ao "Editar área": a máscara do usuário vira uma rampa suave e tudo fora dela é
pixel-idêntico à foto.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..analysis.segmentation import GROUP_IDS

LOCK_GROUPS = ("window", "door")


def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def clean_mask(mask: np.ndarray, min_frac: float = 0.0008, close_frac: float = 0.006) -> np.ndarray:
    """Tira componentes menores que ``min_frac`` da imagem e fecha frestas (``close_frac`` da diagonal)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return m
    H, W = m.shape
    r = max(1, int(close_frac * np.hypot(H, W)))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, _disk(r)).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_frac * H * W
    return keep[lab]


def lock_mask(groups: np.ndarray, which: tuple[str, ...] = LOCK_GROUPS, size: tuple[int, int] | None = None
              ) -> np.ndarray:
    """Máscara (bool) dos grupos travados, opcionalmente redimensionada para ``size`` = (largura, altura)."""
    g = np.asarray(groups)
    m = np.zeros(g.shape, bool)
    for name in which:
        m |= g == GROUP_IDS[name]
    if size is not None and (m.shape[1], m.shape[0]) != tuple(size):
        m = cv2.resize(m.astype(np.uint8), tuple(size), interpolation=cv2.INTER_NEAREST).astype(bool)
    return clean_mask(m)


def feather_alpha(mask: np.ndarray, radius: float, inside: float = 0.5) -> np.ndarray:
    """Alpha float32 em [0, 1] com rampa suave (smoothstep) de largura ``radius`` px ao redor da borda da máscara.

    ``inside`` = fração da rampa que fica dentro da máscara (0 = rampa toda por fora, a máscara inteira com alpha
    1; 1 = rampa toda por dentro, nada fora da máscara muda). Fora da faixa o alpha é exatamente 0."""
    m = np.asarray(mask, bool)
    if not m.any():
        return np.zeros(m.shape, np.float32)
    if radius <= 0:
        return m.astype(np.float32)
    d_out = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)     # distância até a máscara (fora)
    d_in = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)         # distância até a borda (dentro)
    sd = np.where(m, -d_in + 0.5, d_out - 0.5)                               # negativo dentro
    lo, hi = -radius * inside, radius * (1 - inside)
    t = np.clip((hi - sd) / max(hi - lo, 1e-6), 0, 1)
    a = t * t * (3 - 2 * t)
    a[sd >= hi] = 0.0
    return a.astype(np.float32)


def match_color(src: np.ndarray, ref: np.ndarray, mask: np.ndarray, sigma: float, max_shift: float = 8.0,
                amount: float = 0.6) -> np.ndarray:
    """Desloca a cor de baixa frequência de ``src`` (Lab) na direção de ``ref`` dentro de ``mask``.

    O deslocamento é a média (na máscara) de ``passa-baixa(ref) - passa-baixa(src)`` × ``amount``, limitado a
    ``max_shift`` unidades Lab por canal — é uma correção de exposição/tom global da abertura, não uma troca de
    conteúdo."""
    m = np.asarray(mask, bool)
    if not m.any():
        return src
    s = cv2.cvtColor(src.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    r = cv2.cvtColor(ref.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    if sigma > 0:
        s_lp = cv2.GaussianBlur(s, (0, 0), sigma)
        r_lp = cv2.GaussianBlur(r, (0, 0), sigma)
    else:
        s_lp, r_lp = s, r
    shift = np.clip((r_lp[m] - s_lp[m]).mean(axis=0) * amount, -max_shift, max_shift)
    out = s + shift[None, None, :].astype(np.float32)
    rgb = cv2.cvtColor(out, cv2.COLOR_Lab2RGB)
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def composite(base: np.ndarray, top: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """``base·(1-α) + top·α`` em uint8; onde α = 0 o resultado é exatamente ``base`` e onde α = 1, ``top``."""
    a = np.clip(alpha, 0, 1).astype(np.float32)[..., None]
    out = base.astype(np.float32) * (1 - a) + top.astype(np.float32) * a
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    z = a[..., 0] <= 0
    o = a[..., 0] >= 1
    out[z] = base[z]
    out[o] = top[o]
    return out


def lock_openings(generated: np.ndarray, original: np.ndarray, mask: np.ndarray, feather_frac: float = 0.006,
                  color_match: bool = True) -> tuple[np.ndarray, dict]:
    """Compõe as aberturas originais (``mask``, mesma resolução de ``generated``) sobre a imagem gerada.

    Cada componente da máscara recebe o próprio casamento de cor. Devolve a imagem e um resumo (pixels travados,
    componentes, deslocamento médio de cor aplicado)."""
    H, W = mask.shape
    assert generated.shape[:2] == (H, W) and original.shape[:2] == (H, W)
    m = np.asarray(mask, bool)
    if not m.any():
        return generated, {"componentes": 0, "fracao": 0.0}
    diag = float(np.hypot(H, W))
    radius = max(2.0, feather_frac * diag)
    src = original
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    if color_match:
        src = original.copy()
        for k in range(1, n):
            comp = lab == k
            src_k = match_color(original, generated, comp, sigma=max(2.0, 0.01 * diag))
            src[comp] = src_k[comp]
            # a faixa da rampa fora do componente também usa a versão casada (sem degrau de cor na borda)
            ring = cv2.dilate(comp.astype(np.uint8), _disk(int(radius) + 1)).astype(bool) & ~comp
            src[ring] = src_k[ring]
    alpha = feather_alpha(m, radius, inside=0.5)
    out = composite(generated, src, alpha)
    return out, {"componentes": int(n - 1), "fracao": round(float(m.mean()), 4),
                 "raio_px": round(radius, 1)}


def generation_mask(lock: np.ndarray, erode_frac: float = 0.004) -> np.ndarray:
    """Máscara do que o gerador pode refazer quando há trava: tudo menos o miolo das aberturas (o miolo é erodido
    para o gerador ainda poder redesenhar o contorno e casar a borda)."""
    lk = np.asarray(lock, bool)
    if not lk.any():
        return np.ones(lk.shape, bool)
    r = max(1, int(erode_frac * np.hypot(*lk.shape)))
    core = cv2.erode(lk.astype(np.uint8), _disk(r)).astype(bool)
    return ~core


def user_mask(mask: np.ndarray, size: tuple[int, int] | None = None, threshold: int = 128) -> np.ndarray:
    """Máscara do pincel do usuário (PNG cinza/RGBA: branco = editar) em bool, opcionalmente redimensionada."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 3] if m.shape[2] == 4 and m[..., 3].min() < 255 and m[..., :3].max() == m[..., :3].min() \
            else m[..., :3].max(axis=2)
    m = m >= threshold if m.dtype != bool else m
    if size is not None and (m.shape[1], m.shape[0]) != tuple(size):
        m = cv2.resize(m.astype(np.uint8), tuple(size), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def mask_bbox(mask: np.ndarray, margin_frac: float = 0.25, min_side: int = 0, multiple: int = 8
              ) -> tuple[int, int, int, int]:
    """Caixa (x0, y0, x1, y1) da máscara com margem de contexto, quadrada o bastante para o SDXL e dentro da
    imagem."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, W, H
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    side = max(bw, bh) * (1 + 2 * margin_frac)
    side = max(side, min_side, 0.35 * max(H, W))
    cw, ch = min(W, max(bw * (1 + 2 * margin_frac), side * 0.75)), min(H, max(bh * (1 + 2 * margin_frac), side * 0.75))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0 = int(np.clip(cx - cw / 2, 0, W - cw))
    Y0 = int(np.clip(cy - ch / 2, 0, H - ch))
    X1, Y1 = int(min(W, X0 + round(cw))), int(min(H, Y0 + round(ch)))
    return X0, Y0, X1, Y1


# ====================================================================================== aberturas: ajustes gerais
def label_mask(labels: np.ndarray, id2label: dict, names: tuple[str, ...], size: tuple[int, int] | None = None
               ) -> np.ndarray:
    """Máscara dos rótulos ADE20K cujo nome base (antes da vírgula) está em ``names`` (ex.: eletrodomésticos)."""
    lab = np.asarray(labels)
    ids = [int(i) for i, n in (id2label or {}).items() if str(n).split(",")[0].strip().lower() in names]
    m = np.isin(lab, ids) if ids else np.zeros(lab.shape, bool)
    if size is not None and (m.shape[1], m.shape[0]) != tuple(size):
        m = cv2.resize(m.astype(np.uint8), tuple(size), interpolation=cv2.INTER_NEAREST).astype(bool)
    return clean_mask(m, min_frac=0.002)


def bright_coverings(covering: np.ndarray, rgb: np.ndarray, pct: float = 85.0, min_frac: float = 0.003
                     ) -> np.ndarray:
    """Cortinas/persianas claras demais (luz atravessando: cortina transparente na frente de uma janela) — a
    janela atrás delas quase não aparece na segmentação, então a própria cortina vira a abertura a travar."""
    cov = np.asarray(covering, bool)
    out = np.zeros(cov.shape, bool)
    if not cov.any():
        return out
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    thr = np.percentile(L, pct)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cov.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_frac * cov.size:
            continue
        comp = lab == k
        if float(np.median(L[comp])) >= thr:
            out |= comp
    return out


def doorway_views(door: np.ndarray, depth: np.ndarray | None, ratio: float = 1.15, grow: float = 0.6
                  ) -> np.ndarray:
    """O vão de uma porta aberta: a segmentação marca só a folha da porta, e a vista para o cômodo ao lado virava
    uma janela para o jardim. Perto de cada porta (caixa alargada ``grow`` × a largura dela, ou 0,4 × a altura,
    para os lados), os
    pixels claramente além do plano da parede em volta (profundidade > ``ratio`` × a da parede) e encostados na
    porta entram na trava."""
    dm = np.asarray(door, bool)
    out = np.zeros(dm.shape, bool)
    if depth is None or not dm.any():
        return out
    z = np.asarray(depth, np.float32)
    if z.shape != dm.shape:
        z = cv2.resize(z, (dm.shape[1], dm.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(z) & (z > 1e-4)
    H, W = dm.shape
    r = max(2, int(0.03 * np.hypot(H, W)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dm.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x, y, w, h = stats[k, :4]
        if stats[k, cv2.CC_STAT_AREA] < 0.002 * H * W:
            continue
        comp = lab == k
        ring = cv2.dilate(comp.astype(np.uint8), _disk(r)).astype(bool) & ~comp & valid
        if ring.sum() < 16:
            continue
        wall_z = float(np.percentile(z[ring], 30))
        gx = max(grow * w, 0.4 * h)      # a folha aberta aparece estreita; o vão chega a ~metade da altura
        X0, X1 = max(0, int(x - gx)), min(W, int(x + w + gx))
        Y0, Y1 = max(0, int(y - 0.1 * h)), min(H, int(y + h))
        region = np.zeros_like(dm)
        region[Y0:Y1, X0:X1] = True
        deep = region & valid & (z > ratio * wall_z) & ~dm
        if not deep.any():
            continue
        touch = cv2.dilate(comp.astype(np.uint8), _disk(3)).astype(bool)
        m2, lab2 = cv2.connectedComponents(deep.astype(np.uint8), connectivity=8)
        hit = np.unique(lab2[touch & deep])
        for j in hit[hit > 0]:
            cj = lab2 == j
            if cj.sum() >= 0.001 * H * W:
                out |= cj
    return clean_mask(out)


def frame_only(mask: np.ndarray, width_frac: float = 0.012) -> np.ndarray:
    """Só o contorno (moldura) de uma máscara: o miolo fica livre para o gerador."""
    m = np.asarray(mask, bool)
    if not m.any():
        return m
    r = max(2, int(width_frac * np.hypot(*m.shape)))
    return m & ~cv2.erode(m.astype(np.uint8), _disk(r)).astype(bool)


def cap_doors(lock: np.ndarray, door: np.ndarray, max_frac: float = 0.12, all_frames: bool = False
              ) -> tuple[np.ndarray, int]:
    """Componentes de porta grandes demais (> ``max_frac`` da imagem: box de vidro do chuveiro, guarda-roupa
    espelhado, painel de vidro) — ou todos, com ``all_frames`` (banheiro) — ficam travados só pela moldura. Colar
    um box bege inteiro numa sala de mármore dava o aspecto de recorte e até sumia com o vaso sanitário."""
    lk = np.asarray(lock, bool).copy()
    dm = np.asarray(door, bool) & lk
    if not dm.any():
        return lk, 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dm.astype(np.uint8), connectivity=8)
    changed = 0
    for k in range(1, n):
        if all_frames or stats[k, cv2.CC_STAT_AREA] > max_frac * dm.size:
            comp = lab == k
            lk[comp] = False
            lk |= frame_only(comp)
            changed += 1
    return lk, changed


def new_openings(gen_groups: np.ndarray, original_openings: np.ndarray, size: tuple[int, int],
                 min_frac: float = 0.004, source_groups: np.ndarray | None = None, wall_frac: float = 0.5
                 ) -> np.ndarray:
    """Janela/porta/cortina na imagem gerada onde a foto não tinha abertura (a abertura original, dilatada, fica de
    fora: cortinas emoldurando a janela existente são bem-vindas). Resolução ``size`` = (largura, altura).

    ``source_groups`` (grupos da foto): só conta o que nasceu numa PAREDE (ou teto) lisa da foto — ao menos
    ``wall_frac`` do componente. Um box de vidro ou um armário de portas onde a foto tinha um armário/espelho é
    móvel novo, não abertura inventada; refazê-lo como "parede lisa" deixava um borrão bege no lugar."""
    g = np.asarray(gen_groups)
    if (g.shape[1], g.shape[0]) != tuple(size):
        g = cv2.resize(g, tuple(size), interpolation=cv2.INTER_NEAREST)
    orig = np.asarray(original_openings, bool)
    if orig.shape != g.shape:
        orig = cv2.resize(orig.astype(np.uint8), tuple(size), interpolation=cv2.INTER_NEAREST).astype(bool)
    H, W = g.shape
    diag = np.hypot(H, W)
    near_open = cv2.dilate(orig.astype(np.uint8), _disk(int(0.02 * diag))).astype(bool)
    near_cover = cv2.dilate(orig.astype(np.uint8), _disk(int(0.08 * diag))).astype(bool)
    hard = np.isin(g, [GROUP_IDS["window"], GROUP_IDS["door"]]) & ~near_open
    soft = (g == GROUP_IDS["window_covering"]) & ~near_cover
    out = clean_mask(hard | soft, min_frac=min_frac)
    if source_groups is not None and out.any():
        sg = np.asarray(source_groups)
        if sg.shape != g.shape:
            sg = cv2.resize(sg, tuple(size), interpolation=cv2.INTER_NEAREST)
        plain = np.isin(sg, [GROUP_IDS["wall"], GROUP_IDS["ceiling"]])
        n, lab = cv2.connectedComponents(out.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            comp = lab == k
            if float(plain[comp].mean()) < wall_frac:
                out[comp] = False
    return out


def missing_openings(gen_groups: np.ndarray, source_openings: np.ndarray, size: tuple[int, int],
                     min_frac: float = 0.002, keep_frac: float = 0.35) -> np.ndarray:
    """Aberturas da foto (janela, cortina/persiana) que SUMIRAM na imagem gerada: componentes de
    ``source_openings`` em que menos de ``keep_frac`` dos pixels continuam janela/porta/cortina na segmentação da
    imagem gerada (viraram parede, tijolo, armário...). Resolução ``size`` = (largura, altura)."""
    g = np.asarray(gen_groups)
    if (g.shape[1], g.shape[0]) != tuple(size):
        g = cv2.resize(g, tuple(size), interpolation=cv2.INTER_NEAREST)
    src = np.asarray(source_openings, bool)
    if src.shape != g.shape:
        src = cv2.resize(src.astype(np.uint8), tuple(size), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = np.zeros(g.shape, bool)
    if not src.any():
        return out
    still = np.isin(g, [GROUP_IDS["window"], GROUP_IDS["door"], GROUP_IDS["window_covering"]])
    n, lab, stats, _ = cv2.connectedComponentsWithStats(src.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_frac * src.size:
            continue
        comp = lab == k
        if float(still[comp].mean()) < keep_frac:
            out |= comp
    return out


def paste_ceiling(generated: np.ndarray, original: np.ndarray, mask: np.ndarray, feather_frac: float = 0.025
                  ) -> tuple[np.ndarray, dict]:
    """Teto original por cima da imagem gerada (``mask``: teto da foto, já sem as luminárias/móveis que o gerador
    pôs ali). Rampa toda por DENTRO do teto (a junção teto/parede da imagem gerada fica intacta: nada da faixa
    borrada que manter o teto nos latentes deixava) e cor de baixa frequência casada com a do gerador."""
    m = np.asarray(mask, bool)
    if not m.any():
        return generated, {"fracao": 0.0}
    diag = float(np.hypot(*m.shape))
    src = match_color(original, generated, m, sigma=max(2.0, 0.01 * diag), max_shift=10.0, amount=0.7)
    alpha = feather_alpha(m, max(2.0, feather_frac * diag), inside=1.0)
    return composite(generated, src, alpha), {"fracao": round(float(m.mean()), 4)}


def ceiling_keep_mask(src_groups: np.ndarray, gen_groups: np.ndarray, size: tuple[int, int],
                      lock: np.ndarray | None = None, edge_frac: float = 0.006, clear_frac: float = 0.03
                      ) -> np.ndarray:
    """Onde o teto da foto volta: teto na foto (sem a faixa junto da parede), menos onde a imagem gerada tem
    luminária, móvel ou armário (um pendente novo não perde o fio) e menos a trava das aberturas."""
    W, H = size
    ceil = lock_mask(src_groups, ("ceiling",), (W, H))
    if not ceil.any():
        return ceil
    diag = float(np.hypot(W, H))
    ceil = cv2.erode(ceil.astype(np.uint8), _disk(max(1, int(edge_frac * diag)))).astype(bool)
    g = np.asarray(gen_groups)
    if (g.shape[1], g.shape[0]) != (W, H):
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
    busy = np.isin(g, [GROUP_IDS["ceiling_fixture"], GROUP_IDS["furniture"], GROUP_IDS["fixture"]])
    busy = cv2.dilate(busy.astype(np.uint8), _disk(max(1, int(clear_frac * diag)))).astype(bool)
    out = ceil & ~busy
    if lock is not None:
        out &= ~np.asarray(lock, bool)
    return clean_mask(out, min_frac=0.002)


def anchor_tone(img: np.ndarray, ref: np.ndarray, amount_l: float = 0.35, amount_ab: float = 0.35,
                max_ab_shift: float = 6.0, chroma_cap: float = 1.15, chroma_floor: float = 18.0) -> np.ndarray:
    """Ancora a exposição e o balanço de branco da imagem gerada nos da foto (o gerador às vezes devolve tudo
    amarelado, cinzento ou saturado demais, com cara de CG): puxa ``amount_l`` da média/desvio da luminância e
    ``amount_ab`` da média de a/b (Lab) na direção da foto, e limita a saturação média a ``chroma_cap`` × a da foto
    (nunca abaixo de ``chroma_floor``, para um estilo colorido numa sala branca continuar colorido)."""
    s = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    r = cv2.cvtColor(ref.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    L, a, b = s[..., 0], s[..., 1], s[..., 2]
    mi, si = float(L.mean()), float(L.std()) + 1e-3
    mr, sr = float(r[..., 0].mean()), float(r[..., 0].std()) + 1e-3
    tm, ts = mi + amount_l * (mr - mi), si + amount_l * (sr - si)
    L2 = (L - mi) * (ts / si) + tm
    da = float(np.clip(amount_ab * (r[..., 1].mean() - a.mean()), -max_ab_shift, max_ab_shift))
    db = float(np.clip(amount_ab * (r[..., 2].mean() - b.mean()), -max_ab_shift, max_ab_shift))
    a2, b2 = a + da, b + db
    ci = float(np.hypot(a2, b2).mean())
    cr = float(np.hypot(r[..., 1], r[..., 2]).mean())
    cap = max(chroma_cap * cr, chroma_floor)
    if ci > cap:
        k = cap / ci
        a2, b2 = a2 * k, b2 * k
    out = cv2.cvtColor(np.stack([np.clip(L2, 0, 100), a2, b2], -1).astype(np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


__all__ = ["LOCK_GROUPS", "clean_mask", "lock_mask", "feather_alpha", "match_color", "composite", "lock_openings",
           "generation_mask", "user_mask", "mask_bbox", "label_mask", "bright_coverings", "doorway_views",
           "frame_only", "cap_doors", "new_openings", "missing_openings", "paste_ceiling", "ceiling_keep_mask",
           "anchor_tone"]
