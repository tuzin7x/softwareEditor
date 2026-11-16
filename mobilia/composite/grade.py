"""Ajuste fino da cor do CG à foto (PT8), aplicado SÓ na camada CG (C/A), nunca na placa.

Três passos, todos calculados uma vez na calibração da luz (worker do Blender, sala vazia) e
gravados em ``frames.json["calibration"]["ajuste_cg"]`` (o ``recompor`` usa os mesmos):

1. **Mapa de ganho residual de baixa frequência, por canal**: ``G = blur(m·foto/pred)/blur(m)``
   (σ = 0,1 do lado maior), limitado a [0,85; 1,2]. ``pred`` é a sala vazia renderizada com a luz
   já ajustada; ``m`` são os pixels de piso/parede/teto não saturados. Onde há pouca máscara por
   perto o mapa tende ao ganho global; com pouca máscara no quadro (sala branca, quase tudo acima
   de 0,97) o mapa vira só o ganho global.
2. **Curva de tom monotônica** (regressão isotônica na luminância em log₂) da predição linear para
   a luminância da foto. Fica identidade quando o ajuste é ruim (resíduo alto, poucos pixels ou
   nenhum ganho sobre a identidade). Aplicada na cor sem pré-multiplicação (C/A) e
   pré-multiplicada de novo.
3. **Joelho nas altas luzes** (0,8 linear): compressão suave por canal, para o CG não estourar
   onde a foto "rola" as altas luzes.

A placa nunca passa por aqui: tudo multiplica C onde A > 0, então fora do CG o compositor continua
copiando os bytes da foto.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)

GAIN_SIGMA_FRAC = 0.10          # σ do desfoque do mapa (fração do lado maior)
GAIN_LIMITS = (0.85, 1.20)
GAIN_MIN_MASK_FRAC = 0.05       # abaixo disso (fração do quadro com máscara) o mapa vira ganho global
GAIN_MIN_SURFACE_FRAC = 0.25    # idem para a fração das superfícies da sala que NÃO está estourada
GAIN_MIN_PX = 50                # abaixo disso nem o ganho global é confiável: identidade
GAIN_RATIO_LIMITS = (0.5, 2.0)  # razão foto/pred por pixel antes do desfoque (outliers)
GAIN_TAU = 0.05                 # peso do ganho global onde a máscara local é rala
GAIN_GRID_STEP = 8              # o mapa é gravado numa grade 8× menor que o render da calibração

TONE_BINS = 24
TONE_MIN_PX = 500
TONE_MAX_SPREAD = 0.45          # stops: dispersão robusta por pixel (textura + erro) acima disso, identidade
TONE_MAX_RMS = 0.10             # stops: resíduo das medianas por faixa que a curva pode deixar
TONE_MIN_GAIN = 0.03            # stops: a curva precisa melhorar as medianas pelo menos isso
TONE_MAX_DEV = 0.35             # stops: desvio máximo da identidade
TONE_MIN_SLOPE = 0.3            # inclinação mínima (o CG nunca perde todo o contraste numa faixa)

KNEE = 0.8                      # joelho das altas luzes (linear)


# ------------------------------------------------------------------------------- mapa de ganho
def gain_map(photo: np.ndarray, pred: np.ndarray, normal: np.ndarray, lower: np.ndarray | None = None,
             sigma_frac: float = GAIN_SIGMA_FRAC, limits: tuple[float, float] = GAIN_LIMITS,
             min_frac: float = GAIN_MIN_MASK_FRAC) -> tuple[np.ndarray, dict]:
    """Mapa de ganho (h, w, 3) e o diagnóstico (modo, fração da máscara, faixa).

    ``photo``/``pred``: lineares (h, w, 3) no mesmo tamanho; ``normal``: pixels confiáveis;
    ``lower``: pixels das superfícies que estão estourados (só contam para a fração: numa sala
    branca o que sobra não estourado são cantos e rodapés, que não representam a sala)."""
    h, w = normal.shape
    ok = np.asarray(normal, bool) & (pred.min(-1) > 1e-4)
    frac = float(np.asarray(normal, bool).mean())
    n_lo = int(np.asarray(lower, bool).sum()) if lower is not None else 0
    n_ok = int(np.asarray(normal, bool).sum())
    surf = n_ok / max(1, n_ok + n_lo)
    info = {"fracao_mascara": round(frac, 4), "fracao_superficies": round(surf, 4), "pixels": int(ok.sum()),
            "sigma_px": round(sigma_frac * max(h, w), 2), "limites": list(limits)}
    if ok.sum() < GAIN_MIN_PX:
        info.update(modo="identidade", motivo="poucos pixels de calibração", global_=[1.0, 1.0, 1.0])
        G = np.ones((h, w, 3), np.float32)
        return G, _finish_info(info, G)
    g0 = photo[ok].sum(0, dtype=np.float64) / np.maximum(pred[ok].sum(0, dtype=np.float64), 1e-9)
    g0 = np.clip(g0, *limits).astype(np.float32)
    info["global_"] = [round(float(v), 4) for v in g0]
    if frac < min_frac or surf < GAIN_MIN_SURFACE_FRAC:
        info.update(modo="global", motivo=(f"máscara cobre só {frac:.1%} do quadro" if frac < min_frac else
                                           f"só {surf:.0%} das superfícies não estão estouradas"))
        G = np.broadcast_to(g0, (h, w, 3)).astype(np.float32)
        return G, _finish_info(info, G)
    sig = sigma_frac * max(h, w)
    wgt = ok.astype(np.float32)
    ratio = np.clip(photo / np.maximum(pred, 1e-4), *GAIN_RATIO_LIMITS).astype(np.float32)
    bw = ndimage.gaussian_filter(wgt, sig)
    G = np.empty((h, w, 3), np.float32)
    for c in range(3):
        num = ndimage.gaussian_filter(ratio[..., c] * wgt, sig)
        G[..., c] = (num + GAIN_TAU * g0[c]) / (bw + GAIN_TAU)
    G = np.clip(G, *limits)
    info["modo"] = "mapa"
    return G, _finish_info(info, G)


def _finish_info(info: dict, G: np.ndarray) -> dict:
    info["global"] = info.pop("global_")
    info.update(min=round(float(G.min()), 4), max=round(float(G.max()), 4),
                mediana=[round(float(v), 4) for v in np.median(G.reshape(-1, 3), 0)])
    return info


def gain_grid(G: np.ndarray, step: int = GAIN_GRID_STEP) -> np.ndarray:
    """O mapa (de baixa frequência) numa grade ``step``× menor, para caber no ``frames.json``."""
    from PIL import Image

    h, w = G.shape[:2]
    gw, gh = max(1, math.ceil(w / step)), max(1, math.ceil(h / step))
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(G[..., c], np.float32), "F")
                                .resize((gw, gh), Image.BILINEAR)) for c in range(3)], -1)


def upsample_grid(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Grade (gh, gw, 3) → mapa (H, W, 3) bilinear com centros alinhados (o mesmo enquadramento)."""
    from PIL import Image

    H, W = int(shape[0]), int(shape[1])
    grid = np.asarray(grid, np.float32)
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(grid[..., c]), "F")
                                .resize((W, H), Image.BILINEAR)) for c in range(3)], -1)


# ------------------------------------------------------------------------------- curva de tom
def _pav(y: np.ndarray, w: np.ndarray) -> list[tuple[int, int, float, float]]:
    """Pool-adjacent-violators: blocos (início, fim, valor, peso) não decrescentes."""
    blocks: list[list] = []
    for i, (yi, wi) in enumerate(zip(y, w)):
        blocks.append([i, i, float(yi), float(wi)])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            b = blocks.pop()
            a = blocks[-1]
            tw = a[3] + b[3]
            blocks[-1] = [a[0], b[1], (a[2] * a[3] + b[2] * b[3]) / tw, tw]
    return [tuple(b) for b in blocks]


def identity_curve(motivo: str, **extra) -> dict:
    return {"ok": False, "motivo": motivo, **extra}


def fit_tone_curve(pred_lum: np.ndarray, photo_lum: np.ndarray, n_bins: int = TONE_BINS) -> dict:
    """Curva monotônica log₂(luminância prevista) → log₂(luminância da foto) por regressão
    isotônica sobre as medianas de faixas de quantis da predição (2% a 98%: as pontas são rodapés,
    frestas e reflexos, não a resposta da câmera). A textura do albedo é ruído em y e some nas
    medianas; o que sobra é a resposta de tom da foto na faixa de luminância da sala.

    Devolve ``{"ok": False, ...}`` (identidade) quando o ajuste não é confiável: poucos pixels,
    faixa estreita, dispersão alta (a foto não acompanha a predição), medianas fora de ordem (a
    curva não as explica) ou correção pequena demais para valer a pena."""
    pl, yl = np.asarray(pred_lum, np.float64).ravel(), np.asarray(photo_lum, np.float64).ravel()
    ok = (pl > 1e-4) & (yl > 1e-4) & np.isfinite(pl) & np.isfinite(yl)
    if ok.sum() < TONE_MIN_PX:
        return identity_curve("poucos pixels", pixels=int(ok.sum()))
    x, y = np.log2(pl[ok]), np.log2(yl[ok])
    lo, hi = np.quantile(x, [0.02, 0.98])
    keep = (x >= lo) & (x <= hi)
    x, y = x[keep], y[keep]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 4 or edges[-1] - edges[0] < 0.25:
        return identity_curve("faixa de luminância estreita", pixels=int(ok.sum()))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    xm, ym, cnt = [], [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        if sel.sum() >= 5:
            xm.append(float(np.median(x[sel])))
            ym.append(float(np.median(y[sel])))
            cnt.append(float(sel.sum()))
    if len(xm) < 3:
        return identity_curve("faixa de luminância estreita", pixels=int(ok.sum()))
    xm, ym, cnt = np.asarray(xm), np.asarray(ym), np.asarray(cnt)
    kx, ky = [], []
    for a, b, val, _ in _pav(ym, cnt):
        kx.append(float(np.average(xm[a:b + 1], weights=cnt[a:b + 1])))
        ky.append(val)
    kx, ky = np.asarray(kx), np.asarray(ky)
    if len(kx) < 2:
        return identity_curve("curva degenerada", pixels=int(ok.sum()))
    # inclinação mínima (sem faixa achatada) e desvio limitado da identidade
    for i in range(1, len(ky)):
        ky[i] = max(ky[i], ky[i - 1] + TONE_MIN_SLOPE * (kx[i] - kx[i - 1]))
    ky = np.clip(ky, kx - TONE_MAX_DEV, kx + TONE_MAX_DEV)
    curve = {"x_log2": kx, "y_log2": ky}
    w = cnt / cnt.sum()
    rms_c = float(np.sqrt(np.sum(w * (ym - eval_log2(xm, curve)) ** 2)))
    rms_i = float(np.sqrt(np.sum(w * (ym - xm) ** 2)))
    r = y - eval_log2(x, curve)
    spread = float(1.4826 * np.median(np.abs(r - np.median(r))))
    out = {"pixels": int(ok.sum()), "rms_faixas_curva": round(rms_c, 4), "rms_faixas_identidade": round(rms_i, 4),
           "dispersao": round(spread, 4), "x_log2": [round(float(v), 4) for v in kx],
           "y_log2": [round(float(v), 4) for v in ky]}
    if spread > TONE_MAX_SPREAD:
        return identity_curve("dispersão alta: a foto não acompanha a predição", **out)
    if rms_c > TONE_MAX_RMS:
        return identity_curve("resíduo alto", **out)
    if rms_i - rms_c < TONE_MIN_GAIN:
        return identity_curve("correção pequena: identidade", **out)
    return {"ok": True, "motivo": "", **out}


def eval_log2(x: np.ndarray, curve: dict) -> np.ndarray:
    """Avalia a curva (em log₂). Dentro dos nós: interpolação linear; fora deles o desvio da
    identidade cai linearmente a zero em 1 stop (sempre monotônica, inclinação ≥ 1 − desvio)."""
    kx = np.asarray(curve["x_log2"], np.float64)
    ky = np.asarray(curve["y_log2"], np.float64)
    x = np.asarray(x, np.float64)
    off = np.interp(x, kx, ky - kx)
    lo, hi = x < kx[0], x > kx[-1]
    off = np.where(lo, (ky[0] - kx[0]) * np.clip(1.0 - (kx[0] - x), 0.0, 1.0), off)
    off = np.where(hi, (ky[-1] - kx[-1]) * np.clip(1.0 - (x - kx[-1]), 0.0, 1.0), off)
    return x + off


def knee(x: np.ndarray, k: float = KNEE) -> np.ndarray:
    """Compressão suave acima de ``k`` (contínua com derivada 1 em ``k``, assíntota em 1)."""
    x = np.asarray(x, np.float32)
    hi = x > k
    if not hi.any():
        return x
    s = np.float32(1.0 - k)
    y = x.copy()
    y[hi] = np.float32(k) + s * np.tanh((x[hi] - np.float32(k)) / s)
    return y


# ------------------------------------------------------------------------------- calibração → dados
def grade_data(photo: np.ndarray, pred: np.ndarray, normal: np.ndarray, lower: np.ndarray | None = None,
               knee_at: float | None = KNEE, return_map: bool = False):
    """O que vai para ``frames.json["calibration"]["ajuste_cg"]``: grade do mapa de ganho, curva de
    tom e joelho. ``photo``/``pred`` lineares no tamanho do render da calibração; ``lower``: os
    pixels estourados das superfícies. Com ``return_map``, devolve também o mapa de ganho nesse
    tamanho."""
    G, ginfo = gain_map(photo, pred, normal, lower)
    gain = dict(ginfo)
    if ginfo["modo"] == "mapa":
        grid = gain_grid(G)
        gain["grade"] = np.round(grid, 4).tolist()
        gain["forma"] = list(grid.shape)
    curve = fit_tone_curve(((G * pred) @ LUM)[normal], (photo @ LUM)[normal])
    data = {"versao": 1, "ganho": gain, "curva_tom": curve, "joelho": knee_at}
    return (data, G) if return_map else data


# ------------------------------------------------------------------------------- aplicação no CG
CURVE_LUT_STEP = 256            # passos por stop da tabela da curva de tom no compositor


def curve_lut(curve: dict) -> tuple[float, np.ndarray]:
    """Tabela do fator multiplicativo 2^(f(x) − x) da curva, de 1 stop antes do primeiro nó até
    1 stop depois do último (fora disso o fator é 1), em passos de 1/256 stop."""
    kx = np.asarray(curve["x_log2"], np.float64)
    x0, x1 = float(kx[0]) - 1.0, float(kx[-1]) + 1.0
    xs = x0 + np.arange(int(math.ceil((x1 - x0) * CURVE_LUT_STEP)) + 1) / CURVE_LUT_STEP
    return x0, np.exp2(eval_log2(xs, curve) - xs).astype(np.float32)


def _lum(c: np.ndarray) -> np.ndarray:
    """Luminância por pixel, conta elemento a elemento (nenhuma redução do BLAS: o resultado de um
    pixel não depende do tamanho do recorte)."""
    return c[..., 0] * LUM[0] + c[..., 1] * LUM[1] + c[..., 2] * LUM[2]


@dataclass
class CGGrade:
    """Ajuste do CG pronto para o compositor (somente leitura; seguro entre threads)."""

    gain: np.ndarray | None = None          # (H, W, 3) — mapa no tamanho do quadro
    gain_const: np.ndarray | None = None    # (3,) — ganho global (sem mapa)
    curve: dict | None = None               # nós em log₂ (só quando aceita)
    knee_at: float | None = KNEE

    def __post_init__(self):
        self._lut = curve_lut(self.curve) if self.curve is not None else None

    @classmethod
    def from_calibration(cls, calib: dict | None, shape: tuple[int, int]) -> "CGGrade | None":
        """Do ``frames.json["calibration"]``. Sem ``ajuste_cg`` (trabalho antigo): None (nada muda)."""
        data = (calib or {}).get("ajuste_cg")
        if not data:
            return None
        g = data.get("ganho") or {}
        gain = gain_const = None
        if g.get("modo") == "mapa" and g.get("grade"):
            gain = upsample_grid(np.asarray(g["grade"], np.float32), shape)
            gain = np.clip(gain, *g.get("limites", GAIN_LIMITS)).astype(np.float32)
            gain.flags.writeable = False
        elif g.get("modo") == "global" and g.get("global"):
            gain_const = np.asarray(g["global"], np.float32)
        c = data.get("curva_tom") or {}
        curve = {"x_log2": c["x_log2"], "y_log2": c["y_log2"]} if c.get("ok") else None
        kn = data.get("joelho")
        return cls(gain, gain_const, curve, float(kn) if kn else None)

    @property
    def active(self) -> bool:
        return self.gain is not None or self.gain_const is not None or self.curve is not None \
            or self.knee_at is not None

    def describe(self) -> dict:
        return {"mapa_ganho": "mapa" if self.gain is not None else ("global" if self.gain_const is not None
                                                                    else "identidade"),
                "curva_tom": self.curve is not None, "joelho": self.knee_at}

    def apply(self, C: np.ndarray, A: np.ndarray, box: list[int]) -> np.ndarray:
        """C (pré-multiplicada, h×w×3) do retângulo ``box`` do quadro, com A (h×w). Só pixels com
        A > 0 mudam; o resto (C = 0) fica como está. Contas elemento a elemento na caixa dos pixels
        com CG: o valor de um pixel não depende do recorte."""
        if not self.active:
            return C
        m = A > 0
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        if not rows.size:
            return C
        r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        a = A[r0:r1, c0:c1]
        ms = m[r0:r1, c0:c1]
        inv = np.where(ms, np.float32(1.0) / np.where(ms, a, np.float32(1.0)), np.float32(0.0))
        c = C[r0:r1, c0:c1] * inv[..., None]                       # cor sem pré-multiplicação
        if self.gain is not None:
            Y0, X0 = int(box[1]), int(box[0])
            c = c * self.gain[Y0 + r0:Y0 + r1, X0 + c0:X0 + c1]
        elif self.gain_const is not None:
            c = c * self.gain_const
        if self._lut is not None:
            x0, lut = self._lut
            lx = np.log2(np.maximum(_lum(c), np.float32(1e-12)))
            idx = np.clip(np.floor((lx - np.float32(x0)) * np.float32(CURVE_LUT_STEP) + np.float32(0.5)),
                          0, len(lut) - 1).astype(np.intp)
            c = c * lut[idx][..., None]
        if self.knee_at is not None:
            c = knee(c, self.knee_at)
        out = C.copy()
        out[r0:r1, c0:c1] = np.where(ms[..., None], c * a[..., None], C[r0:r1, c0:c1])
        return out
