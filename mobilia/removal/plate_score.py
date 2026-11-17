"""Medidas de realismo da sala vazia: névoa, fantasma, placas soltas e aberturas novas (PT-V3-4).

A nota antiga do portão (:func:`mobilia.removal.plate.assess`) só media cor e grão na costura e a linha
parede/piso. Uma placa com névoa branca, o fantasma do sofá e um pedaço de parede "solto" no piso tirava 90
(avaliação do PT-V3-8: a placa que o crítico deu 2/10). Estas medidas olham para **dentro** do preenchimento e
comparam com o material visível do mesmo plano, sem nenhum limiar ajustado a uma foto específica:

- **névoa** (:func:`fog_metric`): contraste local dentro do preenchimento muito abaixo do contraste do material
  visível do mesmo plano (borrão) e/ou o preenchimento mais claro e menos saturado que a cor esperada no lugar
  (véu leitoso);
- **fantasma** (:func:`ghost_metric`): a estrutura da peça removida continua na placa. Correlação, dentro do
  núcleo das peças, entre o desvio de tom da placa e o desvio de tom da foto (a silhueta clara do sofá branco
  fica como mancha clara) e entre as bordas da placa e as bordas da foto (contorno do quadro, pés da cadeira);
- **placas soltas** (:func:`slab_metric`): manchas grandes cuja cor destoa da cor esperada **do plano** naquele
  lugar (pedaço de parede branca no meio do piso, bloco cinza de cadeira, piso de outro material);
- **aberturas novas** (:func:`new_openings`): porta ou janela que o OneFormer acha na placa dentro da máscara,
  longe de qualquer janela/porta da foto (inventar uma porta é risco jurídico: CDC, art. 37).

A "cor esperada no lugar" é a mesma em todas: o passa-baixa do material visível do **mesmo plano** (piso, cada
parede, teto; rótulos do ray-cast na caixa vazia) estendido para dentro do buraco (push-pull). Sem plano medido,
o anel inteiro em volta da máscara.

Todas as medidas usam frações da diagonal da imagem (independem da resolução) e devolvem severidades em 0..1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from ..analysis.segmentation import GROUP_IDS

# mesmos rótulos de plano de ``plate.py`` (cópia para evitar import circular)
FLOOR, OTHER, CEIL, PROTECTED = -1, -2, -3, -4
G_FLOOR, G_WALL, G_CEIL = GROUP_IDS["floor"], GROUP_IDS["wall"], GROUP_IDS["ceiling"]
G_PROTECTED = (GROUP_IDS["window"], GROUP_IDS["window_covering"], GROUP_IDS["door"])

SEAM_FRAC = 0.015            # faixa da costura (fração da diagonal): fora das medidas de "dentro"
RING_FRAC = 0.05             # anel do material visível usado como referência de contraste
LOWPASS_FRAC = 0.012         # passa-baixa da cor (fração da diagonal)
SLAB_DE = 12.0               # ΔE76 contra a cor esperada do plano para virar "mancha"
SLAB_MIN_AREA = 0.0015       # área mínima de uma mancha (fração da foto)
OPENING_MIN_AREA = 0.002     # porta/janela nova mínima (fração da foto)
OPENING_REACH = 0.02         # longe disto (diagonal) de janela/porta da foto = nova


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)


def _nblur(img: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    wf = w.astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * (wf[..., None] if img.ndim == 3 else wf), (0, 0), sigma)
    den = np.maximum(cv2.GaussianBlur(wf, (0, 0), sigma), 1e-6)
    return num / (den[..., None] if img.ndim == 3 else den)


def _dil(m: np.ndarray, r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


def _push_pull(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    from .plate import push_pull

    return push_pull(values, weights)


def label_blur(img: np.ndarray, labels: np.ndarray | None, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Passa-baixa que não mistura planos: dentro da máscara, cada pixel só faz média com pixels do mesmo rótulo
    de plano (a quina parede/piso não vira "mancha" nem "fantasma"). Fora da máscara e sem rótulos: gaussiana."""
    out = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    if labels is None:
        return out
    for L in np.unique(labels[mask]):
        sel = mask & (labels == L)
        out[sel] = _nblur(img, labels == L, sigma)[sel]
    return out


def plane_group(L: int) -> int:
    return G_FLOOR if L == FLOOR else (G_CEIL if L == CEIL else G_WALL)


def _pearson(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> float:
    """Correlação de Pearson (ponderada por ``w``)."""
    if a.size < 50:
        return 0.0
    w = np.ones_like(a) if w is None else w
    sw = float(w.sum())
    if sw <= 1e-6:
        return 0.0
    a = a - (w * a).sum() / sw
    b = b - (w * b).sum() / sw
    den = float(np.sqrt((w * a * a).sum() * (w * b * b).sum()))
    return float((w * a * b).sum() / den) if den > 1e-9 else 0.0


@dataclass
class PlaneRef:
    """Cor esperada (Lab, passa-baixa) dentro da máscara e de onde ela veio, por pixel."""

    lab: np.ndarray              # (H, W, 3) float32, válido dentro da máscara
    planar: np.ndarray           # (H, W) bool: referência do próprio plano (senão, anel inteiro)
    known: dict = field(default_factory=dict)   # rótulo -> máscara do material visível do plano


def plane_reference(img: np.ndarray, mask: np.ndarray, labels: np.ndarray | None, groups: np.ndarray | None,
                    sigma_frac: float = LOWPASS_FRAC, lab: np.ndarray | None = None,
                    reach_frac: float | None = None) -> PlaneRef:
    """Cor esperada dentro de ``mask``: por plano, o passa-baixa do material visível do mesmo plano (rótulo da caixa
    **e** grupo da segmentação) estendido para dentro (push-pull); o que sobra usa o anel inteiro em volta.
    ``reach_frac``: só o material visível até essa distância da máscara (membrana presa à borda: sem degrau na
    costura); None = o plano visível inteiro."""
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    sigma = max(2.0, sigma_frac * diag)
    lab = _lab(img) if lab is None else lab
    ref = np.zeros((H, W, 3), np.float32)
    planar = np.zeros((H, W), bool)
    known_by = {}
    near_all = _dil(mask, max(3, int(reach_frac * diag))) if reach_frac else np.ones_like(mask)
    if labels is not None and groups is not None:
        for L in np.unique(labels[mask]):
            L = int(L)
            if L in (OTHER, PROTECTED):
                continue
            sel = mask & (labels == L)
            known = ~mask & (labels == L) & (groups == plane_group(L)) & near_all
            if known.sum() < max(50, 0.0005 * H * W):
                continue
            r = _push_pull(_nblur(lab, known, sigma), known.astype(np.float32))
            ref[sel] = r[sel]
            planar |= sel
            known_by[L] = known
    rest = mask & ~planar
    if rest.any():
        ring = ~mask & _dil(mask, max(4, int((reach_frac or RING_FRAC) * diag)))
        if groups is not None:
            # sem plano medido: a cor esperada vem das SUPERFÍCIES em volta (piso, parede, teto), não de móvel que
            # fica, janela ou eletrodoméstico escuro (a membrana espalhava o preto da geladeira pela parede crua)
            surf = ring & np.isin(groups, (G_FLOOR, G_WALL, G_CEIL))
            if surf.sum() >= max(50, 0.15 * ring.sum()):
                ring = surf
        if ring.sum() < 20:
            ring = ~mask
        if ring.sum() >= 1:
            r = _push_pull(_nblur(lab, ring, sigma), ring.astype(np.float32))
            ref[rest] = r[rest]
    return PlaneRef(ref, planar, known_by)


def _interior(mask: np.ndarray, width: int) -> np.ndarray:
    return mask & (ndimage.distance_transform_edt(mask) > width)


def local_contrast(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Desvio padrão local (janela gaussiana ``sigma``) da luminância."""
    m = cv2.GaussianBlur(gray, (0, 0), sigma)
    m2 = cv2.GaussianBlur(gray * gray, (0, 0), sigma)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def fog_metric(plate: np.ndarray, mask: np.ndarray, ref: PlaneRef, labels: np.ndarray | None,
               lab: np.ndarray | None = None) -> dict:
    """Névoa: (a) contraste local do preenchimento / contraste do material visível do mesmo plano; (b) véu
    leitoso: preenchimento mais claro e menos saturado que a cor esperada. Severidade 0..1 ponderada pela área."""
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    seam = max(4, int(SEAM_FRAC * diag))
    ring_w = max(8, int(RING_FRAC * diag))
    s = max(1.5, 0.004 * diag)
    gray = _gray(plate)
    lc = local_contrast(gray, s)
    lab = _lab(plate) if lab is None else lab
    low = label_blur(lab, labels, mask, max(2.0, LOWPASS_FRAC * diag))
    inner = _interior(mask, seam)
    if inner.sum() < 100:
        inner = mask
    near = ~mask & _dil(mask, ring_w)
    parts = []
    keys = list(ref.known.keys()) if ref.known else []
    regions = [(L, inner & (labels == L), ref.known[L] & near) for L in keys] if labels is not None else []
    covered = np.zeros_like(mask)
    for _, sel, _ in regions:
        covered |= sel
    regions.append((None, inner & ~covered, near))
    for L, sel, known in regions:
        n = int(sel.sum())
        if n < 100 or known.sum() < 50:
            continue
        c_in = float(np.median(lc[sel]))
        c_out = float(np.median(lc[known]))
        # contraste perdido: só conta quando o material visível tem textura (parede lisa tem contraste ~ruído)
        ratio = (c_in + 0.6) / (c_out + 0.6)
        # sem plano medido o anel mistura materiais (bordas entre eles): o contraste não é comparável
        tex = 0.0 if L is None else float(np.clip((c_out - 1.0) / 3.0, 0.0, 1.0))
        blur = tex * float(np.clip((0.8 - ratio) / 0.5, 0.0, 1.0))
        d = low[sel] - ref.lab[sel]
        c_ref = np.hypot(ref.lab[sel][:, 1], ref.lab[sel][:, 2])
        c_fill = np.hypot(low[sel][:, 1], low[sel][:, 2])
        milk_de = float(np.mean(np.clip(d[:, 0], 0, None) + 0.7 * np.clip(c_ref - c_fill, 0, None)))
        milk = float(np.clip((milk_de - 2.0) / 8.0, 0.0, 1.0))
        sev = float(np.clip(blur + milk - blur * milk, 0.0, 1.0))
        parts.append((n, sev, ratio, milk_de, "sem_plano" if L is None else L))
    if not parts:
        return {"severidade": 0.0, "razao_contraste": 1.0, "veu_de": 0.0, "por_regiao": {}}
    n = np.array([p[0] for p in parts], float)
    sev = float((n * np.array([p[1] for p in parts])).sum() / n.sum())
    return {"severidade": round(sev, 3),
            "razao_contraste": round(float((n * np.array([p[2] for p in parts])).sum() / n.sum()), 3),
            "veu_de": round(float((n * np.array([p[3] for p in parts])).sum() / n.sum()), 2),
            "por_regiao": {str(p[4]): {"severidade": round(p[1], 3), "razao_contraste": round(p[2], 3),
                                       "veu_de": round(p[3], 2)} for p in parts}}


HAZE_MIN_AREA = 0.004       # mancha de névoa mínima (fração da foto)
HAZE_BAND_FRAC = 0.06        # escala larga do passa-banda da mancha (fração da diagonal)
HAZE_BAND_WIDE = 0.15        # ... e a escala mais larga (mancha de "fumaça" do tamanho de uma parede)
HAZE_BIG_AREA = 0.02         # mancha de névoa deste tamanho (fração da foto, além do mínimo) = severidade 1
HAZE_FLAT_ABS = 2.2          # contraste local (níveis) abaixo do qual a mancha é "lisa" mesmo em material liso
HAZE_BLOB_L = 5.0            # mancha mínima (níveis de L*, passa-banda) dentro do preenchimento


def _detrend(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``v`` menos o plano a + b·x + c·y ajustado (mínimos quadrados) nos pixels ``w``."""
    ys, xs = np.nonzero(w)
    if ys.size < 30:
        return v.astype(np.float32)
    step = max(1, ys.size // 20000)
    ys, xs = ys[::step], xs[::step]
    A = np.stack([np.ones_like(xs, np.float64), xs, ys], 1)
    coef, *_ = np.linalg.lstsq(A, v[ys, xs].astype(np.float64), rcond=None)
    H, W = v.shape
    yy, xx = np.mgrid[0:H, 0:W]
    return (v - (coef[0] + coef[1] * xx + coef[2] * yy)).astype(np.float32)


def _ref_contrast(lc: np.ndarray, ring: np.ndarray, edge_zone: np.ndarray) -> float:
    """Contraste local típico do material visível: mediana no anel, sem a vizinhança das bordas fortes (encontro de
    materiais, quina, batente). Sem isso o anel de uma região sem plano mede bordas, não textura."""
    r = ring & ~edge_zone
    if r.sum() < 100:
        r = ring
    return float(np.median(lc[r])) if r.any() else 0.0


def haze_metric(plate: np.ndarray, mask: np.ndarray, labels: np.ndarray | None, groups: np.ndarray | None,
                ref: PlaneRef, lab: np.ndarray | None = None, gray: np.ndarray | None = None) -> dict:
    """Névoa e fumaça **por mancha** (revisão rodada 2): ``fog_metric`` faz a média por plano e ignora o que não tem
    plano medido (a mancha escura "de fumaça" onde estava a cozinha e a névoa sobre a estante passavam com severidade
    0). Aqui, por superfície (cada plano medido e o "sem plano"), o tom L* sem a rampa de luz (plano ajustado) passa
    por um passa-banda (passa-baixa curto menos passa-baixa largo, duas escalas): um pixel é "véu" quando a mancha
    passa de :data:`HAZE_BLOB_L` (e da variação do mesmo material visível, nos planos medidos), longe de bordas fortes
    reais (rodapé, quina) **e** é lisa: clara e sem o grão do material (o reflexo da janela num piso encerado
    mantém a textura das tábuas e não conta) ou escura sem mais textura que o material em volta. A cor esperada do
    plano não entra: a pintura lisa copia a própria referência, e quando ela vem poluída do anel a fumaça passava.

    Severidade 0..1: a maior entre (fração do preenchimento coberta por manchas acima de :data:`HAZE_MIN_AREA`) / 0,25
    e (maior mancha − mínimo) / :data:`HAZE_BIG_AREA` (uma mancha grande sozinha já é visível de longe).
    Calibração (controle negativo: placa = foto VAZIA real com máscaras de móveis emprestadas, 4 fotos x 7 máscaras):
    1 de 28 passa de 0,6 (reflexo grande e liso de janela num piso encerado)."""
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    n_mask = int(mask.sum())
    empty = {"severidade": 0.0, "fracao": 0.0, "manchas": 0, "por_regiao": {}}
    if n_mask < 100:
        return empty
    lab = _lab(plate) if lab is None else lab
    gray = _gray(plate) if gray is None else gray
    seam = max(4, int(SEAM_FRAC * diag))
    s = max(1.5, 0.004 * diag)
    lc = local_contrast(gray, s)
    mag, _ = _gradients(gray, diag)
    near = ~mask & _dil(mask, max(8, int(RING_FRAC * diag)))
    thr_e = max(float(np.percentile(mag[near], 90)) if near.sum() > 100 else EDGE_FLOOR, 2.0 * EDGE_FLOOR)
    edge_zone = _dil(mag > thr_e, max(2, int(0.006 * diag)))
    inner = _interior(mask, seam)
    s_low, s_big = max(2.0, LOWPASS_FRAC * diag), max(6.0, HAZE_BAND_FRAC * diag)
    # o passa-banda de uma borda forte de verdade (rodapé, quina, batente) se espalha ~2,5 sigmas para cada lado:
    # ali a "mancha" é a borda, não véu
    near_edge = _dil(mag > thr_e, int(2.5 * s_low))
    veil = np.zeros((H, W), bool)
    regions = []
    covered = np.zeros_like(mask)
    if labels is not None:
        for L, known in ref.known.items():
            sel = inner & (labels == L)
            regions.append((str(L), sel, known & near))
            covered |= sel
    surf = near & np.isin(groups, (G_FLOOR, G_WALL, G_CEIL)) if groups is not None else near
    regions.append(("sem_plano", inner & ~covered, surf if surf.sum() >= 150 else near))
    info = {}
    for name, sel, ring in regions:
        if sel.sum() < 100 or ring.sum() < 50:
            continue
        cr = _ref_contrast(lc, ring, edge_zone)
        # passa-banda só com os pixels desta superfície (preenchimento + material visível): sem misturar móvel,
        # janela ou outro plano, dentro e no anel; a variação do mesmo material visível (luz de arandela que
        # continua de fora para dentro) sobe o limite
        w = sel | ring
        # tira a rampa de luz (plano a + b·x + c·y ajustado na superfície): perto da borda do suporte o passa-baixa
        # normalizado de uma rampa fica deslocado e virava "mancha" numa parede lisa com gradiente de luz
        Lr = _detrend(lab[..., 0], w)
        lo = _nblur(Lr, w, s_low)
        d1 = lo - _nblur(Lr, w, s_big)
        d2 = lo - _nblur(Lr, w, HAZE_BAND_WIDE / HAZE_BAND_FRAC * s_big)
        blob = np.maximum(np.abs(d1), np.abs(d2))
        dark = np.where(np.abs(d1) >= np.abs(d2), d1, d2) < 0
        # referência: pixels LISOS do anel, longe de bordas (encontro de materiais no anel tem passa-banda alto
        # mas não é liso; a fumaça é lisa e manchada ao mesmo tempo)
        rf = ring & ~edge_zone & (lc < max(3.0, 1.2 * cr))
        b_ring = float(np.percentile(blob[rf if rf.sum() >= 50 else ring], 90))
        # sem plano medido o anel mistura superfícies (vidro, quina, parede e piso): a variação dele não é do
        # material que deveria estar ali, então vale o limite absoluto
        b_thr = max(HAZE_BLOB_L, 1.5 * b_ring + 1.0) if name != "sem_plano" else HAZE_BLOB_L
        # a mancha precisa ser LISA (sem o grão do material): o reflexo da janela num piso de tábuas encerado é
        # uma mancha clara de verdade, mas mantém a textura das tábuas (controle com fotos vazias)
        # fumaça ESCURA não vem de reflexo: para ela basta não ter mais textura que o material em volta
        smooth = np.where(dark, lc < max(3.0, 1.2 * cr), lc < max(HAZE_FLAT_ABS, 0.6 * cr))
        v = sel & (blob > b_thr) & smooth & ~near_edge
        veil |= v
        info[name] = {"contraste_ref": round(cr, 2), "contraste_dentro": round(float(np.median(lc[sel])), 2),
                      "mancha_anel_p90": round(b_ring, 2), "mancha_dentro_p90": round(float(np.percentile(blob[sel],
                                                                                                         90)), 2),
                      "veu_px": int(v.sum()), "area_px": int(sel.sum())}
    veil = cv2.morphologyEx(veil.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    lab_c, n = ndimage.label(veil)
    if n == 0:
        return {**empty, "por_regiao": info}
    sizes = ndimage.sum(veil, lab_c, range(1, n + 1))
    big = sizes >= HAZE_MIN_AREA * H * W
    area = float(sizes[big].sum())
    frac = area / n_mask
    biggest = float(sizes.max()) / (H * W)
    # uma mancha grande sozinha já é visível de longe (a "fumaça" de 2% da foto numa máscara de 45% era só 6% do
    # preenchimento): a severidade é a maior entre a fração do preenchimento e o tamanho da maior mancha
    sev = max(frac / 0.25, (biggest - HAZE_MIN_AREA) / HAZE_BIG_AREA) if big.any() else 0.0
    return {"severidade": round(float(np.clip(sev, 0.0, 1.0)), 3), "fracao": round(frac, 4),
            "manchas": int(big.sum()), "maior_pct": round(100.0 * float(sizes.max()) / (H * W), 2),
            "por_regiao": info}


def ghost_metric(photo: np.ndarray, plate: np.ndarray, mask: np.ndarray, core: np.ndarray | None, ref: PlaneRef,
                 lab_plate: np.ndarray | None = None, labels: np.ndarray | None = None) -> dict:
    """Fantasma: a estrutura da peça removida que sobrou na placa. Dentro do núcleo das peças (longe da costura):
    correlação entre o desvio de tom (passa-baixa − cor esperada) da placa e o da foto, e entre as bordas da placa
    e as bordas da foto, cada uma vezes a intensidade do que sobrou. Severidade 0..1."""
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    seam = max(4, int(SEAM_FRAC * diag))
    reg = _interior(mask, seam)
    if core is not None:
        reg &= core
    if reg.sum() < 200:
        return {"severidade": 0.0, "tom_corr": 0.0, "borda_corr": 0.0, "area_px": int(reg.sum())}
    lab_p = _lab(plate) if lab_plate is None else lab_plate
    lab_f = _lab(photo)
    # só conta onde a foto mostra de fato um móvel (cor longe da cor esperada do fundo): onde a foto já mostra o
    # próprio fundo (vão entre pés, vidro, máscara folgada), a placa certa É parecida com a foto
    # e onde a placa de fato mudou em relação à foto (se placa ≈ foto, nada foi tirado ali; móvel inteiro que
    # sobrou é caso do OneFormer, não de fantasma)
    s_ref = max(2.0, LOWPASS_FRAC * diag)
    f_low = label_blur(lab_f, labels, mask, s_ref)
    p_low = label_blur(lab_p, labels, mask, s_ref)
    furn = np.clip((np.linalg.norm(f_low - ref.lab, axis=2) - 6.0) / 10.0, 0, 1) * \
        np.clip((np.linalg.norm(f_low - p_low, axis=2) - 3.0) / 6.0, 0, 1)
    wgt = furn[reg].astype(np.float64)
    if wgt.sum() < 200:
        return {"severidade": 0.0, "tom_corr": 0.0, "borda_corr": 0.0, "area_px": int(reg.sum()),
                "movel_px": round(float(wgt.sum()), 1)}
    # passa-banda (forma da peça, sem o gradiente de luz da sala, que a foto e a placa compartilham por natureza)
    s_low, s_big = max(2.0, 0.006 * diag), max(6.0, 0.04 * diag)

    def band(L):
        big = _nblur(L, reg | ~mask, s_big)
        if labels is not None:          # o fundo de cada pixel é o do próprio plano (a quina não é forma de móvel)
            for k in np.unique(labels[reg]):
                sel = reg & (labels == k)
                big[sel] = _nblur(L, (reg | ~mask) & (labels == k), s_big)[sel]
        return label_blur(L, labels, mask, s_low) - big

    dp, df = band(lab_p[..., 0]), band(lab_f[..., 0])
    s_e = max(1.0, 0.0015 * diag)
    s_b = max(1.5, 0.003 * diag)

    def edges(rgb):
        g = cv2.GaussianBlur(_gray(rgb), (0, 0), s_e)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), s_b)

    ep, ef = edges(plate), edges(photo)
    near = ~mask & _dil(mask, max(8, int(RING_FRAC * diag)))
    e_ring = float(np.median(ep[near])) if near.sum() > 50 else float(np.median(ep))
    fw = furn.astype(np.float64)

    def judge(sel: np.ndarray) -> tuple[float, float, float, float]:
        w = fw[sel]
        a, b = dp[sel], df[sel]
        tc = _pearson(a, b, w)
        mu = (w * a).sum() / max(w.sum(), 1e-6)
        amp = float(np.sqrt((w * (a - mu) ** 2).sum() / max(w.sum(), 1e-6)))
        # correlação fraca (< 0,15) é coincidência com a estrutura do próprio plano (rodapé, luz), não fantasma
        tone = max(0.0, tc - 0.15) / 0.85 * float(np.clip((amp - 1.5) / 6.0, 0.0, 1.0))
        ec = _pearson(ep[sel], ef[sel], w)
        strong = w > 0.5
        e_in = float(np.percentile(ep[sel][strong], 90)) if strong.sum() > 50 else float(np.percentile(ep[sel], 90))
        edge = max(0.0, ec - 0.15) / 0.85 * float(np.clip(e_in / (2.0 * e_ring + 4.0), 0.0, 1.0))
        # evidência: fração da região em que a foto mostra um móvel que a placa tirou (pouca = pouco a julgar)
        ev = float(np.clip(w.sum() / max(sel.sum(), 1) / 0.25, 0.0, 1.0))
        return ev * float(np.clip((max(tone, edge) - 0.02) / 0.25, 0.0, 1.0)), tc, amp, ec

    # por peça (componente do núcleo): um fantasma local (o quadro na parede) não some na média do sofá grande
    sev_all, tc_all, amp_all, ec_all = judge(reg)
    lab_c, n = ndimage.label(reg)
    worst, pieces = sev_all, []
    if n > 1:
        sizes = ndimage.sum(reg, lab_c, range(1, n + 1))
        for k in np.argsort(-sizes)[:12]:
            sel = lab_c == (k + 1)
            if sizes[k] < 0.002 * H * W or fw[sel].sum() < 200:
                continue
            sev_k = judge(sel)[0] * float(np.clip(sizes[k] / (0.01 * H * W), 0.3, 1.0))
            pieces.append(round(sev_k, 3))
            worst = max(worst, sev_k)
    return {"severidade": round(worst, 3), "severidade_global": round(sev_all, 3), "por_peca": pieces,
            "tom_corr": round(tc_all, 3), "tom_amp": round(amp_all, 2), "borda_corr": round(ec_all, 3),
            "area_px": int(reg.sum()), "movel_px": round(float(wgt.sum()), 1)}


def slab_metric(plate: np.ndarray, mask: np.ndarray, ref: PlaneRef, lab: np.ndarray | None = None,
                de: float = SLAB_DE, labels: np.ndarray | None = None) -> dict:
    """Placas soltas: manchas conexas (> :data:`SLAB_MIN_AREA` da foto) cuja cor passa-baixa destoa mais de
    ``de`` ΔE da cor esperada do plano. Devolve a fração da máscara coberta por manchas e a maior delas."""
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    lab = _lab(plate) if lab is None else lab
    low = label_blur(lab, labels, mask, max(2.0, LOWPASS_FRAC * diag))
    d = np.linalg.norm(low - ref.lab, axis=2)
    seam = max(3, int(0.5 * SEAM_FRAC * diag))
    cand = _interior(mask, seam) & (d > de)
    n_mask = max(1, int(mask.sum()))
    if not cand.any():
        return {"fracao": 0.0, "maior_pct": 0.0, "manchas": 0, "de_medio": round(float(d[mask].mean()), 2)
                if mask.any() else 0.0}
    lab_c, n = ndimage.label(cand)
    sizes = ndimage.sum(cand, lab_c, range(1, n + 1))
    big = sizes >= SLAB_MIN_AREA * H * W
    area = float(sizes[big].sum())
    return {"fracao": round(area / n_mask, 4), "maior_pct": round(100.0 * float(sizes.max()) / (H * W), 2),
            "manchas": int(big.sum()), "de_medio": round(float(d[mask].mean()), 2),
            "area_pct": round(100.0 * area / (H * W), 2)}


def new_openings(seg2, mask: np.ndarray, groups: np.ndarray, labels: np.ndarray | None = None,
                 min_area: float = OPENING_MIN_AREA) -> list:
    """Portas/janelas que o OneFormer acha na placa dentro da máscara e longe das aberturas da foto."""
    if seg2 is None or getattr(seg2, "groups", None) is None:
        return []
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    g2 = seg2.groups
    if g2.shape != mask.shape:
        g2 = cv2.resize(g2, (W, H), interpolation=cv2.INTER_NEAREST)
    orig = np.isin(groups, G_PROTECTED) & ~mask
    if labels is not None:
        orig |= mask & (labels == PROTECTED)
    near = _dil(orig, max(3, int(OPENING_REACH * diag))) if orig.any() else np.zeros_like(mask)
    out = []
    names = {GROUP_IDS["door"]: "porta", GROUP_IDS["window"]: "janela", GROUP_IDS["window_covering"]: "cortina"}
    for gid, name in names.items():
        sel = mask & (g2 == gid) & ~near
        if sel.sum() < min_area * H * W:
            continue
        lab_c, n = ndimage.label(sel)
        sizes = ndimage.sum(sel, lab_c, range(1, n + 1))
        a = float(sizes.max())
        if a >= min_area * H * W:
            out.append({"rotulo": name, "area_pct": round(100.0 * a / (H * W), 2)})
    return out


N_ORIENT = 12                # bins do histograma de orientação das bordas (0..180°)
EDGE_FLOOR = 6.0             # magnitude mínima (níveis/px, Sobel) de uma "borda" (abaixo é grão/ruído de JPEG)


def _gradients(gray: np.ndarray, diag: float) -> tuple[np.ndarray, np.ndarray]:
    s = max(1.0, 0.0012 * diag)
    g = cv2.GaussianBlur(gray, (0, 0), s)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    return np.hypot(gx, gy), np.mod(np.arctan2(gy, gx), np.pi)


LINE_MIN_FRAC = 0.025        # segmento de reta "longo" (fração da diagonal): rejunte, tábua, aresta de bloco
_LSD = None


def line_segments(gray: np.ndarray) -> np.ndarray:
    """Segmentos de reta (N, 4: x0, y0, x1, y1) pelo LSD do OpenCV. Sem LSD na versão instalada: nenhum."""
    global _LSD
    try:
        if _LSD is None:
            _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
        lines = _LSD.detect(np.clip(gray, 0, 255).astype(np.uint8))[0]
    except Exception:  # noqa: BLE001
        return np.zeros((0, 4), np.float32)
    return np.zeros((0, 4), np.float32) if lines is None else lines.reshape(-1, 4)


def _continues(segs: np.ndarray, cand: np.ndarray, ref: np.ndarray, diag: float) -> np.ndarray:
    """Segmentos ``cand`` que continuam (mesma reta: ângulo < 5° e distância < 0,6% da diagonal) algum segmento
    ``ref``: rodapé, quina e junta reais que atravessam o preenchimento não são estrutura inventada."""
    out = np.zeros(len(cand), bool)
    if not cand.any() or not ref.any():
        return out
    r = segs[ref]
    ra = np.arctan2(r[:, 3] - r[:, 1], r[:, 2] - r[:, 0])
    rn = np.stack([-np.sin(ra), np.cos(ra)], 1)                 # normal de cada reta de referência
    rc = rn[:, 0] * r[:, 0] + rn[:, 1] * r[:, 1]
    for k in np.nonzero(cand)[0]:
        x0, y0, x1, y1 = segs[k]
        a = np.arctan2(y1 - y0, x1 - x0)
        dang = np.abs(np.mod(a - ra + np.pi / 2, np.pi) - np.pi / 2)
        d = np.maximum(np.abs(rn[:, 0] * x0 + rn[:, 1] * y0 - rc), np.abs(rn[:, 0] * x1 + rn[:, 1] * y1 - rc))
        out[k] = bool(((dang < np.radians(5)) & (d < 0.006 * diag)).any())
    return out


def _line_stats(segs: np.ndarray, reg: np.ndarray, diag: float, ring: np.ndarray | None = None,
                mask: np.ndarray | None = None) -> tuple[float, np.ndarray, int]:
    """Densidade de retas longas (comprimento total × diagonal / área / 10: independe da resolução), histograma de
    orientação ponderado pelo comprimento e nº de segmentos com o meio dentro de ``reg``. Com ``ring``, as retas
    que só continuam uma reta do anel (rodapé, quina) não contam."""
    H, W = reg.shape
    n = int(reg.sum())
    if segs.size == 0 or n == 0:
        return 0.0, np.full(N_ORIENT, 1.0 / N_ORIENT), 0
    ln = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    mx = np.clip(((segs[:, 0] + segs[:, 2]) / 2).astype(int), 0, W - 1)
    my = np.clip(((segs[:, 1] + segs[:, 3]) / 2).astype(int), 0, H - 1)
    keep = reg[my, mx] & (ln > LINE_MIN_FRAC * diag)
    if ring is not None:
        keep &= ~_continues(segs, keep, ring[my, mx] & (ln > 0.5 * LINE_MIN_FRAC * diag), diag)
        if mask is not None and keep.any():
            # reta que sai da máscara e segue na foto (rodapé, quina atravessando o buraco) é estrutura real
            t = np.linspace(0.0, 1.0, 9)[None]
            idx = np.nonzero(keep)[0]
            xs = np.clip((segs[idx, 0:1] + t * (segs[idx, 2:3] - segs[idx, 0:1])).astype(int), 0, W - 1)
            ys = np.clip((segs[idx, 1:2] + t * (segs[idx, 3:4] - segs[idx, 1:2])).astype(int), 0, H - 1)
            keep[idx[(~mask[ys, xs]).mean(1) > 0.25]] = False
    dens = float(ln[keep].sum()) * diag / n / 10.0
    ang = np.mod(np.arctan2(segs[keep, 3] - segs[keep, 1], segs[keep, 2] - segs[keep, 0]), np.pi)
    h = np.bincount(np.minimum((ang / np.pi * N_ORIENT).astype(int), N_ORIENT - 1), weights=ln[keep],
                    minlength=N_ORIENT).astype(np.float64)
    h = h + 0.5 * (np.roll(h, 1) + np.roll(h, -1))
    return dens, h / max(h.sum(), 1e-9), int(keep.sum())


def texture_metric(plate: np.ndarray, mask: np.ndarray, labels: np.ndarray | None, ref: PlaneRef,
                   gray: np.ndarray | None = None) -> dict:
    """Material e estrutura do preenchimento contra o material visível **do mesmo plano** (anel em volta da
    máscara). Mede o que a costura e a cor média não veem (avaliação SDXL rodada 1: porcelanato liso virou placas
    com rejunte, madeira virou ardósia, parede lisa ganhou bloco de "tijolinho", cano e placas soltas):

    - **estrutura nova**: (a) densidade de bordas fortes dentro do preenchimento contra a do anel (limiar = p90 da
      magnitude no anel: o material real define o que é forte) e (b) densidade de **retas longas** (LSD) contra a
      do anel: rejunte, aresta de bloco, cano e borda de placa são retas longas mesmo quando fracas. Medido na
      avaliação: o piso refinado pelo SDXL teve 2–8× a densidade de retas da LaMa, que fica abaixo do anel;
    - **material diferente**: distância (variação total, 0..1) entre os histogramas de orientação das retas longas
      dentro e no anel, quando há retas suficientes (tábuas na direção da profundidade x grade de placas de
      ardósia; reflexo vertical do porcelanato x rejunte diagonal);
    - **névoa por frequência alta**: energia de passa-alta fino (mediana) dentro / anel. Vale também para o que
      não tem plano (a mediana não é inflada pelas bordas entre materiais do anel, ao contrário do contraste
      local usado em :func:`fog_metric`).

    Severidades 0..1 por região (piso, cada parede, teto e "sem plano") e a pior, ponderada pela área."""
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    seam = max(4, int(SEAM_FRAC * diag))
    ring_w = max(10, int(0.08 * diag))
    gray = _gray(plate) if gray is None else gray
    mag, _ = _gradients(gray, diag)
    segs = line_segments(gray)
    hp = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), max(1.0, 0.0015 * diag)))
    inner = _interior(mask, seam)
    if inner.sum() < 100:
        inner = mask
    near = ~mask & _dil(mask, ring_w)
    regions = []
    covered = np.zeros_like(mask)
    if labels is not None:
        for L, known in ref.known.items():
            sel = inner & (labels == L)
            regions.append((str(L) if L >= 0 else {FLOOR: "piso", CEIL: "teto"}.get(L, str(L)), sel, known & near,
                            True))
            covered |= sel
    regions.append(("sem_plano", inner & ~covered, near, False))
    out: dict = {}
    for name, sel, ring, planar in regions:
        n, nr = int(sel.sum()), int(ring.sum())
        if n < max(150, 0.0015 * H * W) or nr < 150:
            continue
        thr = max(float(np.percentile(mag[ring], 90)), EDGE_FLOOR)
        e_in, e_out = mag[sel] > thr, mag[ring] > thr
        d_in, d_out = float(e_in.mean()), float(e_out.mean())
        dens = (d_in + 0.01) / (d_out + 0.01)
        # densidade de bordas só decide no que não tem plano (excesso grande); nos planos decidem as retas longas
        # (a densidade de bordas acusava o rodapé que atravessa um buraco pequeno numa sala limpa)
        s_struct = 0.0 if planar else float(np.clip((dens - 2.5) / 3.0, 0.0, 1.0))
        s_orient, tv = 0.0, 0.0
        l_in, h_in, k_in = _line_stats(segs, sel, diag, ring, _ero_(mask, max(2, seam // 2)))
        l_out, h_out, k_out = _line_stats(segs, ring, diag)
        if planar:
            # retas longas além das do material visível (folga: 30% ou 0,8 de densidade)
            s_struct = max(s_struct, float(np.clip((l_in - max(1.3 * l_out, l_out + 0.8)) / 1.5, 0.0, 1.0)))
            if k_in >= 6 and k_out >= 3:
                tv = 0.5 * float(np.abs(h_in - h_out).sum())
                s_orient = float(np.clip((tv - 0.40) / 0.25, 0.0, 1.0)) * float(np.clip((l_in - 0.5) / 1.0, 0.0,
                                                                                         1.0))
        hf_in, hf_out = float(np.median(hp[sel])), float(np.median(hp[ring]))
        hf = (hf_in + 0.25) / (hf_out + 0.25)
        s_haze = float(np.clip((0.55 - hf) / 0.35, 0.0, 1.0)) * float(np.clip((hf_out - 0.6) / 1.0, 0.0, 1.0))
        out[name] = {"area_px": n, "densidade_bordas": round(dens, 2), "orientacao_tv": round(tv, 3),
                     "retas_dentro": round(l_in, 2), "retas_anel": round(l_out, 2),
                     "alta_freq": round(hf, 2), "estrutura": round(s_struct, 3), "material": round(s_orient, 3),
                     "nevoa_af": round(s_haze, 3)}
    if not out:
        return {"estrutura_nova": 0.0, "material": 0.0, "nevoa_af": 0.0, "por_regiao": {}}
    tot = float(sum(v["area_px"] for v in out.values()))

    def worst(key: str) -> float:
        # pior região, com peso pela área (uma região minúscula não reprova a placa inteira sozinha)
        return max(v[key] * float(np.clip(v["area_px"] / (0.25 * tot), 0.3, 1.0)) for v in out.values())

    return {"estrutura_nova": round(worst("estrutura"), 3), "material": round(worst("material"), 3),
            "nevoa_af": round(worst("nevoa_af"), 3), "por_regiao": out}


FLOOR_RING_FRAC = 0.10       # anel do piso visível usado como referência do material (fração da diagonal)
FLOOR_DE_MAX = 18.0          # ΔE76 das medianas a partir do qual a cor do piso é outra (controle: até 16)
FLOOR_BLUR_LOG2 = 1.4        # piso preenchido com 2^1,4 = 2,6× menos bordas que o visível: borrão/material liso
FLOOR_BUSY_LOG2 = 2.0        # ... ou 4× mais: listras/tábuas inventadas num piso liso


def _orient_hist(mag: np.ndarray, ang: np.ndarray, sel: np.ndarray, floor: float) -> tuple[np.ndarray, float]:
    """Histograma de orientação (12 bins, ponderado pela magnitude, só bordas acima do ruído) e a fração de pixels
    com borda."""
    m = sel & (mag > floor)
    if m.sum() < 30:
        return np.full(N_ORIENT, 1.0 / N_ORIENT), float(m.mean()) if sel.any() else 0.0
    h = np.bincount(np.minimum((ang[m] / np.pi * N_ORIENT).astype(int), N_ORIENT - 1), weights=mag[m],
                    minlength=N_ORIENT).astype(np.float64)
    h = h + 0.5 * (np.roll(h, 1) + np.roll(h, -1))
    return h / max(h.sum(), 1e-9), float(m.sum()) / max(1, int(sel.sum()))


def floor_material_metric(plate: np.ndarray, mask: np.ndarray, labels: np.ndarray | None,
                          groups: np.ndarray | None, rug: np.ndarray | None = None,
                          gray: np.ndarray | None = None, lab: np.ndarray | None = None) -> dict:
    """Material do piso preenchido contra o piso **visível** do mesmo plano (tapete fora: sob um tapete grande o
    piso de referência é o que aparece em volta dele, não o tapete). Revisão rodada 2: porcelanato liso virou
    "tábuas" listradas e madeira avermelhada virou quase preta sem que a nota percebesse (ela media lisura e a cor
    média relaxada). Três distâncias, cada uma 0..1, a severidade é a maior:

    - **cor**: ΔE76 entre as medianas Lab (robustas a reflexo) do preenchimento e do piso visível, além de
      :data:`FLOOR_DE_MAX`;
    - **energia de textura**: log2 da razão das densidades de bordas acima do ruído: piso de tábuas que virou
      borrão (abaixo de -:data:`FLOOR_BLUR_LOG2`) ou listras num piso liso (acima de :data:`FLOOR_BUSY_LOG2`);
    - **orientação**: variação total entre os histogramas de orientação das bordas, só quando o preenchimento tem
      mais bordas que o piso visível e a distância é extrema (> 0,65).

    Sem piso visível suficiente (> 2000 px no anel), não mede (0)."""
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    out0 = {"severidade": 0.0}
    if labels is None or groups is None:
        return out0
    seam = max(4, int(SEAM_FRAC * diag))
    sel = _interior(mask, seam) & (labels == FLOOR)
    if sel.sum() < max(500, 0.005 * H * W):
        return out0
    vis = ~mask & (labels == FLOOR) & (groups == G_FLOOR)
    if rug is not None:
        vis &= ~_dil(rug, 2)
    ring = vis & _dil(mask, max(10, int(FLOOR_RING_FRAC * diag)))
    if ring.sum() < 2000:
        return {**out0, "motivo": "pouco piso visível (sem tapete) para comparar"}
    gray = _gray(plate) if gray is None else gray
    lab = _lab(plate) if lab is None else lab
    mag, ang = _gradients(gray, diag)
    noise = max(EDGE_FLOOR * 0.5, float(np.percentile(mag[ring], 50)))
    med_in, med_out = np.median(lab[sel], axis=0), np.median(lab[ring], axis=0)
    de = float(np.linalg.norm(med_in - med_out))
    h_in, e_in = _orient_hist(mag, ang, sel, 2 * noise)
    h_out, e_out = _orient_hist(mag, ang, ring, 2 * noise)
    energy = float(np.log2((e_in + 0.01) / (e_out + 0.01)))
    tv = 0.5 * float(np.abs(h_in - h_out).sum()) if min(e_in, e_out) >= 0.03 else 0.0
    # calibrado com o controle negativo (placa = foto VAZIA de verdade, máscaras de móveis emprestadas, 4 fotos x 7
    # máscaras): no piso real a mediana de cor varia até ΔE 16 entre o fundo e a frente da sala e a orientação das
    # bordas até 0,56 (perspectiva, reflexos), então cor e orientação só contam em casos extremos; a energia de bordas
    # ficou entre -1,3 e +1,8 (log2) e as placas ruins da avaliação (piso borrado, carpete no lugar de porcelanato)
    # ficaram abaixo de -1,6
    s_col = float(np.clip((de - FLOOR_DE_MAX) / 8.0, 0.0, 1.0))
    s_en = float(np.clip((-energy - FLOOR_BLUR_LOG2) / 0.6, 0.0, 1.0)) if energy < 0 else \
        float(np.clip((energy - FLOOR_BUSY_LOG2) / 1.0, 0.0, 1.0))
    s_or = float(np.clip((tv - 0.65) / 0.25, 0.0, 1.0)) * float(np.clip(energy / 1.0, 0.0, 1.0))
    return {"severidade": round(max(s_col, s_en, s_or), 3), "de_mediana": round(de, 2),
            "energia_log2": round(energy, 2), "orientacao_tv": round(tv, 3), "cor": round(s_col, 3),
            "energia": round(s_en, 3), "orientacao": round(s_or, 3), "piso_visivel_px": int(ring.sum()),
            "area_px": int(sel.sum())}


CONTOUR_STEP_DE = 7.0        # degrau de cor (ΔE76, passa-baixa curto) através do contorno que o marca


def contour_metric(plate: np.ndarray, mask: np.ndarray, core: np.ndarray | None, labels: np.ndarray | None = None,
                   groups: np.ndarray | None = None, gray: np.ndarray | None = None) -> dict:
    """Contorno da peça removida que ficou na placa (o retângulo claro do quadro, a silhueta do sofá). As outras
    medidas excluem a faixa da costura; esta olha só para ela, por peça: em cada ponto do contorno em que **os dois
    lados são a mesma superfície** (o pixel de fora é piso/parede/teto do mesmo grupo do plano de dentro; contorno
    encostado num móvel que fica, numa janela ou noutro plano não conta: ali a borda é real), mede

    - o degrau de cor através do contorno (passa-baixa de cada lado, só com pixels do próprio lado) e
    - uma linha fina **ao longo** do contorno (gradiente na direção da normal do contorno mais forte que o p90 do
      material do anel; linha real da sala, como rodapé ou junta, **cruza** o contorno e não conta).

    A fração marcada do contorno da pior peça (com contorno válido suficiente) vira a severidade 0..1."""
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    if not mask.any() or mask.all() or labels is None or groups is None:
        return {"severidade": 0.0, "fracao": 0.0}
    gray = _gray(plate) if gray is None else gray
    lab = _lab(plate)
    w = max(2, int(0.004 * diag))
    s = max(1.0, 0.0012 * diag)
    g = cv2.GaussianBlur(gray, (0, 0), s)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 4.0, cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    mag = np.hypot(gx, gy)
    sd = (ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)).astype(np.float32)
    sd = cv2.GaussianBlur(sd, (0, 0), 1.5)
    nx, ny = cv2.Sobel(sd, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sd, cv2.CV_32F, 0, 1, ksize=3)
    nn = np.hypot(nx, ny) + 1e-6
    nx, ny = nx / nn, ny / nn                                   # normal apontando para dentro da máscara
    edge = mask & ~_ero_(mask, 1)
    ys, xs = np.nonzero(edge)
    if ys.size < 30:
        return {"severidade": 0.0, "fracao": 0.0}
    d = 2.5 * w
    yo = np.clip(np.round(ys - d * ny[ys, xs]).astype(int), 0, H - 1)
    xo = np.clip(np.round(xs - d * nx[ys, xs]).astype(int), 0, W - 1)
    yi = np.clip(np.round(ys + d * ny[ys, xs]).astype(int), 0, H - 1)
    xi = np.clip(np.round(xs + d * nx[ys, xs]).astype(int), 0, W - 1)
    lin = labels[yi, xi]
    grp_in = np.array([plane_group(int(L)) if L not in (OTHER, PROTECTED) else -1 for L in lin])
    valid = (~mask[yo, xo]) & mask[yi, xi] & (groups[yo, xo] == grp_in)
    if valid.sum() < 30:
        return {"severidade": 0.0, "fracao": 0.0, "contorno_valido_px": int(valid.sum())}
    sig = max(1.5, 0.5 * w)
    l_in, l_out = _nblur(lab, mask, sig), _nblur(lab, ~mask, sig)
    step = np.linalg.norm(l_in[yi, xi] - l_out[yo, xo], axis=1)
    ring = ~mask & _dil(mask, max(10, int(RING_FRAC * diag))) & ~_dil(edge, w)
    thr = max(float(np.percentile(mag[ring], 90)), EDGE_FLOOR) if ring.sum() > 100 else EDGE_FLOOR
    normal = np.abs(gx * nx + gy * ny)
    k = np.ones((2 * w + 1, 2 * w + 1), np.uint8)
    line_hit = (cv2.dilate(np.where(normal > 0.7 * mag, normal, 0).astype(np.float32), k) > 1.5 * thr)[ys, xs]
    marked = valid & ((step > CONTOUR_STEP_DE) | line_hit)
    frac = float(marked.sum()) / float(valid.sum())
    worst, pieces = frac, []
    if core is not None:
        lab_c, n = ndimage.label(_dil(core, 3 * w))
        cid = lab_c[ys, xs]
        for c in range(1, n + 1):
            v = valid & (cid == c)
            if v.sum() >= max(40, 0.04 * diag):
                f = float((marked & v).sum()) / float(v.sum())
                pieces.append(round(f, 3))
                worst = max(worst, f)
    sev = float(np.clip((worst - 0.25) / 0.35, 0.0, 1.0))
    return {"severidade": round(sev, 3), "fracao": round(frac, 3), "pior_peca": round(worst, 3),
            "contorno_valido_px": int(valid.sum()), "por_peca": sorted(pieces, reverse=True)[:12]}


def _ero_(m: np.ndarray, r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


def realism(photo: np.ndarray, plate: np.ndarray, mask: np.ndarray, labels: np.ndarray | None,
            groups: np.ndarray | None, core: np.ndarray | None = None, rug: np.ndarray | None = None) -> dict:
    """As medidas visuais de uma placa (sem modelo de IA): névoa, fantasma, placas soltas, textura (estrutura
    nova, material diferente, névoa por alta frequência) e contorno da peça removida."""
    if not mask.any():
        return {"nevoa": {"severidade": 0.0}, "fantasma": {"severidade": 0.0}, "placas_soltas": {"fracao": 0.0},
                "veu": {"severidade": 0.0}, "material_piso": {"severidade": 0.0},
                "textura": {"estrutura_nova": 0.0, "material": 0.0, "nevoa_af": 0.0},
                "contorno": {"severidade": 0.0}}
    lab = _lab(plate)
    gray = _gray(plate)
    ref = plane_reference(plate, mask, labels, groups, lab=lab)
    return {"nevoa": fog_metric(plate, mask, ref, labels, lab),
            "fantasma": ghost_metric(photo, plate, mask, core, ref, lab, labels),
            "placas_soltas": slab_metric(plate, mask, ref, lab, labels=labels),
            "textura": texture_metric(plate, mask, labels, ref, gray),
            "veu": haze_metric(plate, mask, labels, groups, ref, lab, gray),
            "material_piso": floor_material_metric(plate, mask, labels, groups, rug, gray, lab),
            "contorno": contour_metric(plate, mask, core, labels, groups, gray)}


__all__ = ["plane_reference", "fog_metric", "haze_metric", "floor_material_metric", "ghost_metric", "slab_metric", "new_openings", "realism", "PlaneRef"]
