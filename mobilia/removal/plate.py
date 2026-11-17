"""Sala vazia do modo B ("placa"), PLANO_V3 §3.2, §3.6 e PT-V3-3.

A placa é a foto sem os móveis. Ela é o 1º quadro do vídeo e, fora da máscara de remoção, é **a foto byte a byte**
(cópia de volta no fim, conferida com ``np.array_equal``). Etapas:

1. **Máscara de efeitos (2 passadas).** Núcleo = peças (máscara dura ∪ alpha > 0,02) dilatadas 0,4% da diagonal.
   Candidatos de sombra/reflexo/luz = anel de 1,2% da diagonal + faixa espelhada abaixo dos pés (até 45% da altura
   da peça; 60% e também na parede para os planejados: luz de LED sob o aéreo, brilho no backsplash). Uma 1ª
   passada barata (inpainting no espaço da imagem) dá uma placa provisória; só ficam na máscara os candidatos em
   que a foto difere dela **e** que estão ligados à peça. Os efeitos são separados em "sombra" (foto mais escura)
   e "luz" (foto mais clara; reflexo no piso polido e luz emitida por planejados): os dois saem da placa.
2. **Ray-cast na caixa vazia** (``RoomModel``): cada pixel mascarado vira piso, teto, parede *i*, "outro" (visto
   por uma porta, pela profundidade do MoGe) ou **protegido** (janela, porta, cortina: dentro de uma abertura do
   modelo da sala ou colado a pixels dessas classes).
3. **Preenchimento por plano** em vistas retificadas (piso e teto de cima, paredes de frente), **sempre com a
   máscara inteira como buraco** (senão o LaMa reconstrói a peça a partir dos pedaços dela que caem no plano
   vizinho). Contexto do piso/teto: só o próprio material. Em cada vista:
   - revestimento com padrão (azulejo, porcelanato): a grade é detectada por autocorrelação e as placas visíveis
     são repetidas (síntese periódica, rejunte alinhado);
   - senão LaMa + reinjeção do grão (o LaMa borra buracos grandes);
   - **harmonização**: membrana tipo Poisson (a diferença de tom na borda é interpolada para dentro do buraco) e
     corte de "ilhas" de tom fora da faixa do anel original.
4. **Parede quase toda escondida** (cozinha planejada de ponta a ponta): pintura lisa com o perfil de cor por
   altura das paredes visíveis (inclui o rodapé) e o gradiente de luz medido na parte visível, mais grão leve.
   Nas paredes preenchidas pelo LaMa, o **rodapé** detectado no perfil é redesenhado reto na altura certa.
5. **Protegidos e "outro"**: inpainting no espaço da imagem, com contexto local (recorte em volta). Nunca
   recebem textura de parede retificada; nada de porta, tomada ou vista nova.
6. **Cópia de volta** byte a byte fora da máscara.

A **nota de qualidade** (:class:`PlateQuality`) mede, por plano: ΔE de costura (faixa interna x anel) e ΔE do
preenchimento todo, razão de grão/textura, distância e continuidade da linha parede/piso projetada, e (opcional)
o OneFormer rodado de novo na placa (nenhum móvel com área > 0,5% pode reaparecer dentro da máscara). Ela
alimenta o portão de qualidade (:func:`build_plate_gated`): placa ruim refaz mantendo os planejados e, se ainda
ruim, avisa; nunca entrega uma placa ruim em silêncio.

O inpainter é injetável (``inpaint(rgb8, mask) -> rgb8``): no produto é o big-lama (:class:`Lama`, CUDA se houver,
senão CPU); nos testes rápidos, :func:`telea_inpaint` (OpenCV, sem modelo).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np
from scipy import ndimage

from ..analysis.segmentation import GROUP_IDS

InpaintFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

# rótulos do ray-cast (paredes usam o id >= 0)
FLOOR, OTHER, CEIL, PROTECTED = -1, -2, -3, -4
LABEL_NAMES = {FLOOR: "piso", OTHER: "outro", CEIL: "teto", PROTECTED: "protegido"}

G_FLOOR, G_WALL, G_CEIL = GROUP_IDS["floor"], GROUP_IDS["wall"], GROUP_IDS["ceiling"]
G_PROTECTED = (GROUP_IDS["window"], GROUP_IDS["window_covering"], GROUP_IDS["door"])

# --- máscara de efeitos ----------------------------------------------------------------------------------------
CORE_DILATE = 0.004          # fração da diagonal
EFFECT_RING = 0.012
EFFECT_RING_BUILTIN = 0.025
BAND_DOWN = 0.45             # faixa espelhada abaixo dos pés (fração da altura da peça)
BAND_DOWN_BUILTIN = 0.6
EFFECT_REL = 0.07            # diferença relativa (borrada) foto x placa provisória
EFFECT_ABS = 12.0            # ou diferença absoluta em níveis

# --- preenchimento ---------------------------------------------------------------------------------------------
PLAIN_WALL_VISIBLE = 0.25    # parede com menos que isto visível -> pintura lisa
PROTECT_REACH = 0.012        # pixel mascarado a menos disto (diag) de janela/porta/cortina -> protegido
PERIOD_MIN_ACF = 0.40
MAX_SIDE = 1024              # lado máximo de uma vista retificada
MIN_CONTEXT_PX = 1500        # material visível mínimo do plano na vista retificada para preencher por plano
MIN_CONTEXT_FRAC = 0.01      # ... e em fração do buraco

# --- limites da nota (portão) ----------------------------------------------------------------------------------
# estratégias do portão (:func:`build_plate_gated`): "planos" = LaMa por plano retificado (padrão);
# "imagem" = LaMa direto no espaço da imagem (melhor quando o modelo da sala erra: linhas tortas por plano errado)
STRATEGIES = ("planos", "imagem")
GATE_WIDE_DILATE = 2.5       # máscara ampla do portão: núcleo dilatado 2,5× (sobra de borda/sombra do móvel)

LIMITS = {
    "costura_de_max": 6.0,           # ΔE76 entre faixa interna e anel, por plano (aceite PT-V3-3: <= 3)
    "grao_log2_max": 1.0,            # |log2(grão dentro / grão no anel)|
    "linha_px_max": 4.0,             # distância mediana da borda à linha parede/piso projetada (aceite: <= 3)
    "linha_continuidade_min": 0.5,   # fração dos pontos da linha (dentro da máscara) com borda achada
    "preenchimento_de_max": 6.0,     # ΔE76 preenchimento x anel (média global ou referência local), por plano
    "preenchimento_de_max_relaxado": 9.0,   # ... quando a costura é ótima (<= 2) e o grão bom: gradiente de luz
    "costura_relaxa_de": 4.5,        # costura a partir da qual o limite do preenchimento começa a subir
    "grao_relaxa_log2": 0.6,         # |log2 grão| acima disto o limite do preenchimento não sobe
    "costura_sem_plano_max": 16.0,   # ΔE76 da costura do que não tem plano (vãos, protegidos, sobre bancadas)
    "area_gerada_max": 0.70,         # fração da foto reconstruída
    # geometria da sala não achada (HDR, retrato, piso todo coberto): quase todo o preenchimento fica "sem plano",
    # que só tem o limite frouxo de costura. Área grande e maioria sem plano = não medido -> reprova
    "sem_plano_area_min": 0.30,      # área gerada (fração da foto) a partir da qual a regra vale
    "sem_plano_frac_max": 0.50,      # fração da área gerada que fica "sem plano" (vãos, protegidos, plano perdido)
    "redetectado_area_min": 0.005,   # móvel reencontrado pelo OneFormer na placa (> 0,5% da foto)
}


def disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _dil(m: np.ndarray, r: int) -> np.ndarray:
    return cv2.dilate(m.astype(np.uint8), disk(r)) > 0


def _ero(m: np.ndarray, r: int) -> np.ndarray:
    return cv2.erode(m.astype(np.uint8), disk(r)) > 0


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def _nblur(img: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Média gaussiana normalizada só sobre os pixels com peso ``w`` (convolução normalizada)."""
    wf = w.astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * (wf[..., None] if img.ndim == 3 else wf), (0, 0), sigma)
    den = cv2.GaussianBlur(wf, (0, 0), sigma)
    den = np.maximum(den, 1e-6)
    return num / (den[..., None] if img.ndim == 3 else den)


# ============================================================================================================
# Inpainters
# ============================================================================================================
class Lama:
    """big-lama TorchScript (Apache-2.0). CUDA se disponível, senão CPU (mesmo resultado, mais lento)."""

    def __init__(self, path, device: str = "auto"):
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = torch.jit.load(str(path), map_location=device).eval()
        self.calls: list[tuple[tuple[int, int], float]] = []

    def __call__(self, rgb8: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch

        t0 = time.perf_counter()
        h, w = mask.shape
        if not mask.any():
            return rgb8.copy()
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        img = np.pad(rgb8, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        m = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
        ti = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div(255.0).to(self.device)
        tm = torch.from_numpy(np.ascontiguousarray(m))[None, None].float().to(self.device)
        with torch.inference_mode():
            out = self.model(ti, tm)
        out = (out[0].permute(1, 2, 0).float().cpu().numpy()[:h, :w] * 255.0).round().clip(0, 255).astype(np.uint8)
        res = rgb8.copy()
        res[mask] = out[mask]
        self.calls.append(((w, h), time.perf_counter() - t0))
        return res

    def release(self) -> None:
        import gc

        del self.model
        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


def load_lama(device: str = "auto", offline: bool = False) -> Lama:
    """Carrega o big-lama do diretório de pesos (SHA-256 conferido); ``WeightsError`` se ausente/corrompido."""
    from ..analysis.weights import big_lama_path

    return Lama(big_lama_path(offline=offline), device)


def telea_inpaint(rgb8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpainting clássico do OpenCV (Telea). Sem modelo: usado nos testes rápidos e como último recurso."""
    if not mask.any():
        return rgb8.copy()
    res = cv2.inpaint(np.ascontiguousarray(rgb8), mask.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    out = rgb8.copy()
    out[mask] = res[mask]
    return out


# ============================================================================================================
# Caixa vazia: ray-cast e vistas retificadas
# ============================================================================================================
@dataclass
class Plane:
    key: int                    # FLOOR, CEIL ou id da parede
    A: np.ndarray               # (4, 3): (a, b, 1) -> ponto homogêneo da sala
    point: np.ndarray           # ponto do plano (3,)
    normal: np.ndarray          # normal (3,)
    wall: object | None = None


class RoomBox:
    """Projeções da caixa da sala (piso, teto, paredes do ``RoomModel``) na foto de tamanho (W, H)."""

    def __init__(self, room, W: int, H: int):
        self.room = room
        intr = room.intr()
        if (intr.width, intr.height) != (W, H):
            intr = intr.scaled(W, H)
        self.intr = intr
        self.K = intr.matrix().copy()
        self.K[0, 2] -= 0.5        # convenção de borda -> centro do pixel (OpenCV)
        self.K[1, 2] -= 0.5
        self.M = room.M
        self.Minv = np.linalg.inv(self.M)
        self.W, self.H = W, H
        self.rays = intr.rays() @ self.M[:3, :3].T          # direções no referencial da sala (H, W, 3)
        self.o = self.M[:3, 3].copy()
        ch = float(room.ceiling_height)
        self.planes: dict[int, Plane] = {
            FLOOR: Plane(FLOOR, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1.0]]), np.zeros(3),
                         np.array([0, 0, 1.0])),
            CEIL: Plane(CEIL, np.array([[1, 0, 0], [0, 1, 0], [0, 0, ch], [0, 0, 1.0]]), np.array([0, 0, ch]),
                        np.array([0, 0, 1.0])),
        }
        for w in room.walls:
            d, p0 = w.direction, np.asarray(w.p0, float)
            self.planes[w.id] = Plane(w.id, np.array([[d[0], 0, p0[0]], [d[1], 0, p0[1]], [0, 1, 0], [0, 0, 1.0]]),
                                      np.array([p0[0], p0[1], 0.0]),
                                      np.array([w.normal[0], w.normal[1], 0.0]), w)
        self.label, self.t, self.S, self.Z = self._raycast()

    def _raycast(self, margin: float = 0.35):
        d = self.rays
        H, W = d.shape[:2]
        best = np.full((H, W), np.inf)
        label = np.full((H, W), OTHER, np.int32)
        S = np.full((H, W), np.nan, np.float32)
        Z = np.full((H, W), np.nan, np.float32)
        ch = float(self.room.ceiling_height)
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = -self.o[2] / d[..., 2]
            ok = (d[..., 2] < 0) & (tf > 0)
            best[ok], label[ok] = tf[ok], FLOOR
            tc = (ch - self.o[2]) / d[..., 2]
            ok = (d[..., 2] > 0) & (tc > 0)
            best[ok], label[ok] = tc[ok], CEIL
            for w in self.room.walls:
                n = np.asarray(w.normal, float)
                p0 = np.asarray(w.p0, float)
                tw = ((p0 - self.o[:2]) @ n) / (d[..., 0] * n[0] + d[..., 1] * n[1])
                hit = self.o + tw[..., None] * d
                s = (hit[..., :2] - p0) @ w.direction
                okw = ((tw > 0) & (s > -margin) & (s < w.length + margin) & (hit[..., 2] > -0.05)
                       & (hit[..., 2] < ch + 0.05) & (tw < best))
                best[okw], label[okw] = tw[okw], w.id
                S[okw], Z[okw] = s[okw], hit[..., 2][okw]
        return label, best, S, Z

    def coords(self, plane: Plane, us: np.ndarray, vs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coordenadas (a, b) no plano e distância ao longo do raio para os pixels (us, vs)."""
        r = self.rays[vs, us]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((plane.point - self.o) @ plane.normal) / (r @ plane.normal)
        hit = self.o + t[:, None] * r
        if plane.wall is None:
            a, b = hit[:, 0], hit[:, 1]
        else:
            a = (hit[:, :2] - np.asarray(plane.wall.p0)) @ plane.wall.direction
            b = hit[:, 2]
        return a, b, t * np.linalg.norm(r, axis=1)

    def homography(self, plane: Plane, a0: float, b1: float, ppm: float) -> np.ndarray:
        """Pixel (u, v) da vista retificada -> pixel da foto. a = u/ppm + a0 ; b = b1 - v/ppm."""
        G = np.array([[1 / ppm, 0, a0], [0, -1 / ppm, b1], [0, 0, 1.0]])
        return self.K @ self.Minv[:3, :] @ plane.A @ G

    def rect_for(self, plane: Plane, sel: np.ndarray, max_side: int = MAX_SIDE, pad_px: int = 40):
        """Homografia e tamanho da vista retificada que cobre ``sel`` (dilatado) com densidade ~ a da foto."""
        vs, us = np.nonzero(_dil(sel, pad_px))
        a, b, dist = self.coords(plane, us, vs)
        ok = np.isfinite(a) & np.isfinite(b) & (dist > 0) & (dist < 15.0)
        if ok.sum() < 20:
            return None
        a, b, dist = a[ok], b[ok], dist[ok]
        a0, a1 = np.percentile(a, [0.5, 99.5])
        b0, b1 = np.percentile(b, [0.5, 99.5])
        if plane.wall is not None:
            a0, a1 = max(a0, -0.5), min(a1, plane.wall.length + 0.5)
            b0, b1 = max(b0, -0.3), min(b1, float(self.room.ceiling_height) + 0.2)
        ext = max(a1 - a0, b1 - b0, 1e-3)
        sv, su = np.nonzero(sel)
        _, _, dsel = self.coords(plane, su, sv)
        dsel = dsel[np.isfinite(dsel) & (dsel > 0)]
        med = float(np.median(dsel)) if dsel.size else float(np.median(dist))
        ppm = min(self.K[0, 0] / max(med, 0.3), max_side / ext)
        size = (max(16, int((a1 - a0) * ppm) + 1), max(16, int((b1 - b0) * ppm) + 1))
        return self.homography(plane, a0, b1, ppm), size

    def floor_line(self, wall, step_px: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
        """Pontos (N, 2) da linha parede/piso projetada (centro de pixel) e a normal da linha na imagem (N, 2)."""
        L = wall.length
        n = max(8, int(L * self.K[0, 0] / step_px))
        s = np.linspace(0, L, n)
        p = np.asarray(wall.p0)[None] + s[:, None] * wall.direction[None]
        pts = np.c_[p, np.zeros(n), np.ones(n)]
        cam = pts @ self.Minv.T
        z = cam[:, 2]
        uv = (cam[:, :3] @ self.K.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uv[:, :2] / uv[:, 2:3]
        tang = np.gradient(uv, axis=0)
        nrm = np.c_[-tang[:, 1], tang[:, 0]]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        ok = (z > 0.1) & np.isfinite(uv).all(1)
        return uv[ok], nrm[ok]


def _warp_in(img, Hm, size, interp=cv2.INTER_LINEAR, border=cv2.BORDER_REFLECT):
    return cv2.warpPerspective(img, Hm, size, flags=interp | cv2.WARP_INVERSE_MAP, borderMode=border)


def _warp_mask_in(m, Hm, size, thr=0):
    return _warp_in(m.astype(np.uint8) * 255, Hm, size, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT) > thr


# ============================================================================================================
# Textura: grade periódica, grão, harmonização
# ============================================================================================================
def detect_period(gray: np.ndarray, known: np.ndarray, min_p: int = 6) -> tuple[int, int] | None:
    """Período (py, px) de um revestimento com padrão, por autocorrelação normalizada só nos pixels conhecidos.
    ``None`` se não houver grade nos dois eixos (piso liso, madeira corrida, parede pintada)."""
    if known.sum() < 400:
        return None
    g = gray.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 8.0)          # tira o gradiente de luz: fica o rejunte
    m = known.astype(np.float32)
    g = (g - g[known].mean()) * m
    var = float((g[known] ** 2).mean())
    if var < 1.0:
        return None
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
        best = None
        for p in range(min_p, n):
            if ov[p] < 0.25:
                break
            if vals[p] >= PERIOD_MIN_ACF and vals[p] >= vals[p - 1] and (p + 1 >= n or vals[p] >= vals[p + 1]):
                if vals[1:p].min() < vals[p] - 0.25:
                    best = p
                    break
        return best

    px = axis_period(acf[0, : w // 2], overlap[0, : w // 2])
    py = axis_period(acf[: h // 2, 0], overlap[: h // 2, 0])
    if px is None or py is None:
        return None
    return int(py), int(px)


def periodic_fill(img: np.ndarray, known: np.ndarray, hole: np.ndarray, period: tuple[int, int],
                  reach: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Copia, para cada pixel do buraco, o pixel conhecido mais próximo deslocado de um múltiplo da grade
    (rejunte alinhado). Retorna (imagem, pixels que ficaram sem fonte)."""
    py, px = period
    out = img.copy()
    todo = hole.copy()
    src_ok = _ero(known, 2)
    h, w = hole.shape
    offs = [(i * py, j * px) for i in range(-reach, reach + 1) for j in range(-reach, reach + 1) if i or j]
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


def grain_transfer(filled: np.ndarray, known: np.ndarray, hole: np.ndarray, rng: np.random.Generator,
                   patch: int = 48, sigma: float = 2.0) -> np.ndarray:
    """Reinjeta no buraco o grão (passa-alta) de retalhos aleatórios do material visível."""
    f = filled.astype(np.float32)
    low = cv2.GaussianBlur(f, (0, 0), sigma)
    hp = f - low
    th, tw = hole.shape
    ok = _ero(known, patch // 2 + 1)
    ys, xs = np.nonzero(ok)
    if len(ys) < 50:
        return filled
    acc = np.zeros_like(f)
    wsum = np.zeros((th, tw, 1), np.float32)
    win = np.outer(np.hanning(patch + 2)[1:-1], np.hanning(patch + 2)[1:-1]).astype(np.float32)[..., None]
    hb = _dil(hole, patch // 2)
    step = patch // 2
    for y in range(0, th, step):
        for x in range(0, tw, step):
            if not hb[y:y + patch, x:x + patch].any():
                continue
            k = rng.integers(len(ys))
            sy, sx = ys[k] - patch // 2, xs[k] - patch // 2
            src = hp[max(sy, 0):sy + patch, max(sx, 0):sx + patch]
            h2, w2 = min(patch, th - y), min(patch, tw - x)
            if sy < 0 or sx < 0 or src.shape[0] < h2 or src.shape[1] < w2:
                continue
            acc[y:y + h2, x:x + w2] += src[:h2, :w2] * win[:h2, :w2]
            wsum[y:y + h2, x:x + w2] += win[:h2, :w2]
    g = acc / np.maximum(wsum, 1e-3)
    # o LaMa já deixa algum grão: soma só o que falta (mantém o desvio do anel)
    lowh = cv2.GaussianBlur(f, (0, 0), sigma)
    rstd = lambda a: float(1.4826 * np.median(np.abs(a - np.median(a)))) if a.size else 0.0   # robusto ao rejunte
    have = rstd((f - lowh)[hole])
    want = rstd(hp[ok])
    gain = float(np.sqrt(max(want ** 2 - have ** 2, 0.0)) / max(rstd(g[hole]) if hole.any() else 1.0, 1e-3))
    out = f.copy()
    out[hole] = (f + min(gain, 1.0) * g)[hole]
    return out.round().clip(0, 255).astype(np.uint8)


def push_pull(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Interpolação "push-pull" (pirâmide) dos valores com peso > 0 para o resto da imagem (suave)."""
    v = values.astype(np.float32)
    w = np.clip(weights.astype(np.float32), 0, 1)
    h, wd = w.shape
    if w.max() <= 0:
        return v
    if min(h, wd) <= 4 or w.min() >= 1:
        if w.min() >= 1:
            return v
        mean = (v * w[..., None]).sum((0, 1)) / w.sum() if v.ndim == 3 else (v * w).sum() / w.sum()
        return w[..., None] * v + (1 - w[..., None]) * mean if v.ndim == 3 else w * v + (1 - w) * mean
    size = (max(1, (wd + 1) // 2), max(1, (h + 1) // 2))
    vw = cv2.resize(v * (w[..., None] if v.ndim == 3 else w), size, interpolation=cv2.INTER_AREA)
    ws = cv2.resize(w, size, interpolation=cv2.INTER_AREA)
    vc = vw / (np.maximum(ws, 1e-6)[..., None] if v.ndim == 3 else np.maximum(ws, 1e-6))
    coarse = push_pull(vc, np.clip(ws * 4, 0, 1))
    up = cv2.resize(coarse, (wd, h), interpolation=cv2.INTER_LINEAR)
    if v.ndim == 3 and up.ndim == 2:
        up = up[..., None]
    return (w[..., None] * v + (1 - w[..., None]) * up) if v.ndim == 3 else (w * v + (1 - w) * up)


def harmonize(img: np.ndarray, known: np.ndarray, hole: np.ndarray, band: int = 4, island_sigma: float = 0.0,
              tol: float = 3.0) -> np.ndarray:
    """Harmonização de um preenchimento: (1) membrana tipo Poisson: a diferença de tom (passa-baixa) entre o anel
    original e a borda interna do buraco é interpolada para dentro do buraco; (2) corte de ilhas: o passa-baixa
    largo do preenchimento é limitado à faixa (p2..p98) do passa-baixa do anel original."""
    if not hole.any() or known.sum() < 30:
        return img
    f = img.astype(np.float32)
    lk = _nblur(f, known, 3.0)
    lf = _nblur(f, hole, 3.0)
    inner = hole & _dil(known, band)
    if inner.sum() >= 10:
        d = np.zeros_like(f)
        d[inner] = (lk - lf)[inner]
        corr = push_pull(d, inner.astype(np.float32))
        corr = _nblur(corr, hole, 2.0)
        f[hole] += corr[hole]
    if island_sigma > 0:
        low = _nblur(f, hole, island_sigma)
        ring = known & _dil(hole, int(3 * island_sigma))
        if ring.sum() >= 50:
            lowk = _nblur(f, known, island_sigma)[ring]
            lo = np.percentile(lowk, 2, axis=0) - tol
            hi = np.percentile(lowk, 98, axis=0) + tol
            f[hole] += (np.clip(low, lo, hi) - low)[hole]
    out = img.copy()
    out[hole] = f[hole].round().clip(0, 255).astype(np.uint8)
    return out


def _finish_surface(orig: np.ndarray, filled: np.ndarray, known: np.ndarray, hole: np.ndarray,
                    rng: np.random.Generator, periodic: bool = True, grain: bool = True) -> tuple[np.ndarray, dict]:
    """Pós-processo numa vista retificada: grade periódica OU grão, e harmonização."""
    info: dict = {"metodo": "lama"}
    if periodic:
        per = detect_period(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), known)
        if per is not None:
            filled, left = periodic_fill(filled, known, hole, per)
            info.update(metodo="periodico", periodo_px=list(per), sem_fonte_px=int(left.sum()))
    if grain and info["metodo"] == "lama":
        filled = grain_transfer(filled, known, hole, rng)
    diag = float(np.hypot(*hole.shape))
    # casamento de estatística por plano: mediana (Lab) do preenchimento -> mediana do anel do mesmo material
    ring = known & _dil(hole, max(8, int(0.08 * diag)))
    if ring.sum() >= 200 and hole.sum() >= 50:
        lab = _lab(filled)
        shift = np.median(lab[ring], axis=0) - np.median(lab[hole], axis=0)
        info["ajuste_tom_de"] = round(float(np.linalg.norm(shift)), 2)
        if np.linalg.norm(shift) > 1.0:
            lab[hole] += shift
            rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB) * 255.0
            filled = filled.copy()
            filled[hole] = np.clip(rgb[hole], 0, 255).round().astype(np.uint8)
    sig = max(4.0, 0.02 * diag)
    filled = harmonize(filled, known, hole, island_sigma=sig)
    return filled, info


# ============================================================================================================
# Máscara de efeitos
# ============================================================================================================
def core_mask(pieces: Sequence, shape: tuple[int, int], dilate: float = CORE_DILATE) -> np.ndarray:
    H, W = shape
    m = np.zeros(shape, bool)
    for p in pieces:
        m |= p.mask | (p.alpha > 0.02)
    return _dil(m, max(3, int(dilate * np.hypot(H, W))))


def effect_candidates(pieces: Sequence, groups: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Região candidata a sombra/reflexo/luz: anel em volta + faixa espelhada abaixo dos pés (sobre o piso; para
    planejados também sobre a parede, onde cai a luz do LED sob o aéreo)."""
    H, W = core.shape
    diag = np.hypot(H, W)
    floorish = _dil(groups == G_FLOOR, 6)
    wallish = _dil(np.isin(groups, (G_FLOOR, G_WALL)), 6)
    cand = np.zeros((H, W), bool)
    for p in pieces:
        builtin = getattr(p, "kind", "") == "planejado"
        m = p.mask
        if not m.any():
            continue
        cand |= _dil(m, int((EFFECT_RING_BUILTIN if builtin else EFFECT_RING) * diag))
        ys, xs = np.nonzero(m)
        bottom = np.full(W, -1)
        top = np.full(W, H)
        np.maximum.at(bottom, xs, ys)
        np.minimum.at(top, xs, ys)
        frac = BAND_DOWN_BUILTIN if builtin else BAND_DOWN
        band = np.zeros((H, W), bool)
        for c in np.nonzero(bottom >= 0)[0]:
            hgt = bottom[c] - top[c]
            band[bottom[c] + 1:min(H, bottom[c] + 1 + int(frac * hgt)), c] = True
        band = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_CLOSE, disk(6)) > 0
        cand |= band & (wallish if builtin else floorish)
    prot = np.isin(groups, G_PROTECTED)
    return cand & ~core & ~prot


def verify_effects(photo: np.ndarray, plate1: np.ndarray, cand: np.ndarray, core: np.ndarray):
    """Só ficam os candidatos em que a foto difere da placa provisória e que estão ligados à peça.
    Retorna (efeito, sombra, luz)."""
    lum = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), 3.0)
    lp, l1 = lum(photo), lum(plate1)
    d = np.abs(lp - l1).max(axis=2)
    rel = d / (l1.mean(axis=2) + 10.0)
    eff = cand & ((rel > EFFECT_REL) | (d > EFFECT_ABS))
    eff = cv2.morphologyEx(eff.astype(np.uint8), cv2.MORPH_OPEN, disk(2)) > 0
    n, lab = cv2.connectedComponents(eff.astype(np.uint8))
    touch = set(np.unique(lab[_dil(core, 3)]).tolist()) - {0}
    eff = np.isin(lab, list(touch)) if touch else np.zeros_like(eff)
    eff = cv2.morphologyEx(eff.astype(np.uint8), cv2.MORPH_CLOSE, disk(4)) > 0
    eff &= (cand | _dil(cand, 2)) & ~core
    darker = lp.mean(axis=2) < l1.mean(axis=2)
    return eff, eff & darker, eff & ~darker


# ============================================================================================================
# Parede lisa (quase toda escondida) e rodapé
# ============================================================================================================
@dataclass
class WallProfile:
    """Cor mediana das paredes visíveis por altura (bins de ``step`` m) + rodapé detectado."""
    z: np.ndarray
    rgb: np.ndarray             # (n, 3)
    count: np.ndarray
    baseboard_h: float = 0.0    # 0 = sem rodapé detectado
    grain: float = 1.0
    flat: np.ndarray | None = None   # cor lisa (mediana) acima do rodapé

    def color_at(self, z: np.ndarray) -> np.ndarray:
        """Rodapé: perfil por altura; acima dele: a cor lisa (§3.6: "pintura lisa na cor média das paredes
        visíveis"; o perfil por altura acima do rodapé copiaria a faixa iluminada do backsplash e portas de
        armário mal segmentadas como parede)."""
        idx = np.clip(np.round((z - self.z[0]) / (self.z[1] - self.z[0])).astype(int), 0, len(self.z) - 1)
        col = self.rgb[idx]
        if self.flat is not None:
            col = np.where((z >= self.baseboard_h)[:, None], self.flat[None], col)
        return col


def wall_profile(photo: np.ndarray, box: RoomBox, visible_wall: np.ndarray, step: float = 0.01) -> WallProfile | None:
    ys, xs = np.nonzero(visible_wall & np.isfinite(box.Z))
    if ys.size < 300:
        return None
    z = box.Z[ys, xs]
    ch = float(box.room.ceiling_height)
    bins = np.arange(0.0, ch + step, step)
    idx = np.clip(((z - bins[0]) / step).astype(int), 0, len(bins) - 1)
    rgb = np.zeros((len(bins), 3), np.float32)
    cnt = np.bincount(idx, minlength=len(bins))
    px = photo[ys, xs].astype(np.float32)
    order = np.argsort(idx, kind="stable")
    sidx, spx = idx[order], px[order]
    starts = np.searchsorted(sidx, np.arange(len(bins)))
    ends = np.searchsorted(sidx, np.arange(len(bins)), side="right")
    for k in range(len(bins)):
        if ends[k] > starts[k]:
            rgb[k] = np.median(spx[starts[k]:ends[k]], axis=0)
    good = cnt >= 15
    if good.sum() < 5:
        return None
    gi = np.nonzero(good)[0]
    for k in range(len(bins)):                         # bins vazios: vizinho válido mais próximo
        if not good[k]:
            rgb[k] = rgb[gi[np.argmin(np.abs(gi - k))]]
    # rodapé: faixa contígua a partir do chão com cor bem diferente da parede (z 0,35..1,2 m)
    ref_sel = good & (bins >= 0.35) & (bins <= 1.2)
    bb = 0.0
    if ref_sel.sum() >= 5 and good[: int(0.3 / step)].sum() >= 3:
        ref = _lab(np.median(rgb[ref_sel], axis=0)[None, None])[0, 0]
        lab = _lab(rgb[None])[0]
        de = np.linalg.norm(lab - ref, axis=1)
        top = 0.0
        miss = 0
        for k in range(int(0.32 / step)):
            if not good[k]:
                miss += 1
                if miss > 3:
                    break
                continue
            if de[k] > 8.0:
                top = bins[k] + step
            else:
                break
        if 0.03 <= top <= 0.30:
            bb = float(top)
    # grão (desvio robusto do passa-alta da luminância das paredes visíveis)
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = (gray - cv2.GaussianBlur(gray, (0, 0), 1.5))[visible_wall]
    grain = float(min(6.0, 1.4826 * np.median(np.abs(hp - np.median(hp))))) if hp.size else 1.0
    above = z >= bb + 0.05
    flat = np.median(px[above], axis=0) if above.sum() >= 50 else np.median(px, axis=0)
    return WallProfile(bins, rgb, cnt, bb, grain, flat.astype(np.float32))


def _light_model(photo, box, prof: WallProfile, sel: np.ndarray, rng) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Gradiente de luz medido: razão luminância/perfil ajustada por um polinômio de 2º grau em (u, v)."""
    ys, xs = np.nonzero(sel & np.isfinite(box.Z))
    H, W = sel.shape
    if ys.size > 20000:
        k = rng.choice(ys.size, 20000, replace=False)
        ys, xs = ys[k], xs[k]
    if ys.size < 200:
        return lambda u, v: np.ones(u.shape, np.float32)
    base = prof.color_at(box.Z[ys, xs]).mean(axis=1)
    ratio = photo[ys, xs].astype(np.float32).mean(axis=1) / np.maximum(base, 1.0)
    u, v = xs / W - 0.5, ys / H - 0.5
    A = np.c_[np.ones_like(u), u, v, u * u, v * v, u * v]
    keep = np.ones(len(ratio), bool)
    for _ in range(3):                    # ajuste robusto: tira faixas iluminadas, quadros, sombras fortes
        coef, *_ = np.linalg.lstsq(A[keep], ratio[keep], rcond=None)
        res = ratio - A @ coef
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-3
        keep = np.abs(res) < 2.0 * mad
        if keep.sum() < 100:
            break

    def f(uu, vv):
        uu, vv = uu / W - 0.5, vv / H - 0.5
        r = coef[0] + coef[1] * uu + coef[2] * vv + coef[3] * uu * uu + coef[4] * vv * vv + coef[5] * uu * vv
        return np.clip(r, 0.6, 1.4).astype(np.float32)
    return f


def paint_plain(out: np.ndarray, photo: np.ndarray, box: RoomBox, prof: WallProfile, region: np.ndarray,
                light_sel: np.ndarray, rng: np.random.Generator) -> None:
    """Pinta ``region`` (pixels de parede) como parede lisa: perfil por altura × gradiente de luz + grão leve."""
    ys, xs = np.nonzero(region & np.isfinite(box.Z))
    if ys.size == 0:
        return
    light = _light_model(photo, box, prof, light_sel, rng)
    col = prof.color_at(box.Z[ys, xs]) * light(xs.astype(np.float32), ys.astype(np.float32))[:, None]
    noise = np.zeros(region.shape, np.float32)
    noise[ys, xs] = rng.normal(0, prof.grain, ys.size)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.7)[ys, xs]
    out[ys, xs] = np.clip(col + noise[:, None], 0, 255).round().astype(np.uint8)


# ============================================================================================================
# Resultado e nota
# ============================================================================================================
@dataclass
class PlateQuality:
    score: float                         # 0..100
    ok: bool
    per_plane: dict                      # nome -> {area_px, costura_de, preenchimento_de, grao_log2}
    line: dict                           # distância/continuidade da linha parede/piso
    area_generated: float
    redetected: list = field(default_factory=list)   # [{rotulo, area_pct}] reencontrados pelo OneFormer
    reasons: list = field(default_factory=list)       # motivos de reprovação (português)
    raw_score: float | None = None                    # nota antes do teto de 59 (compara duas placas reprovadas)
    unplanned: float = 0.0                            # fração da área gerada sem plano medido

    def to_dict(self) -> dict:
        return {"nota": round(self.score, 1), "aprovada": self.ok, "por_plano": self.per_plane,
                "linha_parede_piso": self.line, "area_gerada_pct": round(100 * self.area_generated, 2),
                "sem_plano_pct": round(100 * self.unplanned, 1),
                "moveis_reencontrados": self.redetected, "motivos": list(self.reasons)}


@dataclass
class PlateResult:
    plate: np.ndarray
    mask: np.ndarray                     # máscara de remoção final (placa == foto fora dela)
    core: np.ndarray                     # peças dilatadas
    effect: np.ndarray                   # sombra ∪ luz removidas
    shadow: np.ndarray
    light: np.ndarray
    labels: np.ndarray                   # rótulo de plano por pixel (válido dentro da máscara)
    quality: PlateQuality | None
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    kept_builtins: bool = False

    def report(self) -> dict:
        return {"placa": {"mascara_pct": round(100 * float(self.mask.mean()), 2),
                          "efeitos_pct": round(100 * float(self.effect.mean()), 2),
                          "sombra_px": int(self.shadow.sum()), "luz_px": int(self.light.sum()),
                          "planejados_mantidos": self.kept_builtins, "planos": self.info.get("planos", {}),
                          "tempos_s": {k: round(v, 2) for k, v in self.info.get("tempos", {}).items()},
                          "qualidade": None if self.quality is None else self.quality.to_dict(),
                          "estrategia": self.info.get("estrategia_portao", self.info.get("estrategia", "planos")),
                          "tentativas": list(self.info.get("tentativas", [])),
                          "avisos": list(self.warnings)}}


def _bands(mask: np.ndarray, width: int):
    """Faixa interna (dentro da máscara, até ``width`` px da borda) e anel externo."""
    dist_in = ndimage.distance_transform_edt(mask)
    dist_out = ndimage.distance_transform_edt(~mask)
    return mask & (dist_in <= width), ~mask & (dist_out <= width)


def _fill_local_de(lab: np.ndarray, sel: np.ndarray, ring: np.ndarray, diag: float) -> float:
    """ΔE do preenchimento contra a cor **esperada no lugar**: o passa-baixa do anel do mesmo material estendido
    para dentro do buraco (push-pull), e não a média do anel inteiro. Uma faixa de sol, a queda de luz para o fundo
    ou um material escuro vizinho em um lado do anel puxam a média global, mas não a referência local de cada pixel.
    Média de ‖passa-baixa(preenchimento) − referência‖ sobre o preenchimento: uma mancha no meio continua contando."""
    s = max(4.0, 0.02 * diag)
    ref = push_pull(_nblur(lab, ring, s), ring.astype(np.float32))
    fl = _nblur(lab, sel, s)
    return float(np.linalg.norm((fl - ref)[sel], axis=1).mean())


def fill_de_limit(seam: float, grain_log2: float, relax: bool = True) -> float:
    """Limite do ΔE do preenchimento conforme a costura: com a costura e o grão bons, a cor já casa com o anel na
    borda e a diferença no meio é quase sempre gradiente de luz (sol, fundo da sala, sombra sob a bancada), então o
    limite sobe de ``preenchimento_de_max`` (costura >= ``costura_relaxa_de``) até ``preenchimento_de_max_relaxado``
    (costura <= 2). Grão ruim, ou ``relax=False`` (a placa tem mancha forte em outro lugar), não relaxa nada."""
    lo, hi = LIMITS["preenchimento_de_max"], LIMITS["preenchimento_de_max_relaxado"]
    if not relax or abs(grain_log2) > LIMITS["grao_relaxa_log2"]:
        return lo
    t = float(np.clip((LIMITS["costura_relaxa_de"] - seam) / (LIMITS["costura_relaxa_de"] - 2.0), 0.0, 1.0))
    return lo + t * (hi - lo)


def plane_stats(photo: np.ndarray, plate: np.ndarray, mask: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                min_area: float = 0.002) -> dict:
    """ΔE de costura e do preenchimento e razão de grão por plano (piso, teto, cada parede)."""
    H, W = mask.shape
    diag = np.hypot(H, W)
    w_seam = max(6, int(0.015 * diag))
    w_ring = max(10, int(0.03 * diag))
    lab_p = _lab(plate)
    gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.5))
    inner, ring = _bands(mask, w_seam)
    _, ring_w = _bands(mask, w_ring)
    out = {}
    for L in np.unique(labels[mask]):
        if L in (OTHER, PROTECTED):
            continue
        sel = mask & (labels == L)
        if sel.sum() < min_area * H * W:
            continue
        grp = G_FLOOR if L == FLOOR else (G_CEIL if L == CEIL else G_WALL)
        same = (labels == L) & (groups == grp)
        rin, rout = inner & sel, ring & same
        if rin.sum() < 30 or rout.sum() < 30:
            continue
        # costura local: passa-baixa do anel original x passa-baixa do preenchimento, pixel a pixel na faixa
        # interna (robusto a parede de 2 materiais dentro do mesmo plano)
        sig = max(3.0, w_seam / 2.0)
        lk = _nblur(lab_p, rout | (ring & (labels == L)), sig)
        lf = _nblur(lab_p, sel, sig)
        valid = rin & (cv2.GaussianBlur(rout.astype(np.float32), (0, 0), sig) > 0.05)
        if valid.sum() < 30:
            continue
        seam = float(np.linalg.norm((lk - lf)[valid], axis=1).mean())
        rw = ring_w & same
        fill_g = fill = float(np.linalg.norm(lab_p[sel].mean(0) - lab_p[rw].mean(0))) if rw.sum() >= 30 else seam
        if rw.sum() >= 30:      # média global ou referência local, a menor (anel poluído só de um lado)
            fill = min(fill_g, _fill_local_de(lab_p, sel, rw, diag))
        g_in = float(np.median(hp[sel])) + 0.3
        g_out = float(np.median(hp[rw if rw.sum() >= 30 else rout])) + 0.3
        name = LABEL_NAMES.get(int(L), f"parede{int(L)}")
        out[name] = {"area_px": int(sel.sum()), "costura_de": round(seam, 2), "preenchimento_de": round(fill, 2),
                     "preenchimento_global_de": round(fill_g, 2),
                     "grao_log2": round(float(np.log2(g_in / g_out)), 2)}
    # o que não tem plano (vãos, protegidos, peças apoiadas em algo que fica): costura local contra o anel inteiro
    # (qualquer material), para que uma mancha ali também conte na nota
    sel = mask & np.isin(labels, (OTHER, PROTECTED))
    if sel.sum() >= min_area * H * W:
        sig = max(3.0, w_seam / 2.0)
        lk = _nblur(lab_p, ring, sig)
        lf = _nblur(lab_p, sel, sig)
        valid = inner & sel & (cv2.GaussianBlur(ring.astype(np.float32), (0, 0), sig) > 0.05)
        if valid.sum() >= 30:
            de = np.linalg.norm((lk - lf)[valid], axis=1)
            out["sem_plano"] = {"area_px": int(sel.sum()), "costura_de": round(float(de.mean()), 2),
                                "costura_p90_de": round(float(np.percentile(de, 90)), 2)}
    return out


def line_stats(plate: np.ndarray, photo: np.ndarray, mask: np.ndarray, box: RoomBox, radius: int = 8,
               labels: np.ndarray | None = None) -> dict:
    """Linha parede/piso projetada: na placa, dentro da máscara, procura a borda mais forte na normal da linha.
    Retorna a distância mediana (px), a continuidade (fração de pontos com borda) e, para calibrar, a mesma
    distância na parte visível da foto."""
    H, W = mask.shape
    g_plate = cv2.GaussianBlur(cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    g_photo = cv2.GaussianBlur(cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    offs = np.arange(-radius, radius + 1, dtype=np.float32)
    res_in, res_vis = [], []
    for w in box.room.walls:
        uv, nrm = box.floor_line(w)
        if len(uv) == 0:
            continue
        m = (uv[:, 0] > radius + 2) & (uv[:, 0] < W - radius - 3) & (uv[:, 1] > radius + 2) & (uv[:, 1] < H - radius - 3)
        uv, nrm = uv[m], nrm[m]
        if len(uv) == 0:
            continue
        # o ponto só conta se a caixa diz "parede w" logo acima e "piso" logo abaixo (linha visível na caixa)
        up = np.round(uv - 4 * nrm * np.sign(nrm[:, 1:2] + 1e-9)).astype(int)
        dn = np.round(uv + 4 * nrm * np.sign(nrm[:, 1:2] + 1e-9)).astype(int)
        lab = box.label if labels is None else np.where(mask, labels, box.label)
        valid = (lab[np.clip(up[:, 1], 0, H - 1), np.clip(up[:, 0], 0, W - 1)] == w.id) & \
                (lab[np.clip(dn[:, 1], 0, H - 1), np.clip(dn[:, 0], 0, W - 1)] == FLOOR)
        uv, nrm = uv[valid], nrm[valid]
        if len(uv) == 0:
            continue
        xs = uv[:, 0:1] + offs[None] * nrm[:, 0:1]
        ys = uv[:, 1:2] + offs[None] * nrm[:, 1:2]
        ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inm = mask[vi, ui] & _ero(mask, 3)[vi, ui]
        for g, sel, acc in ((g_plate, inm, res_in), (g_photo, ~mask[vi, ui], res_vis)):
            if not sel.any():
                continue
            prof = ndimage.map_coordinates(g, [ys[sel].ravel(), xs[sel].ravel()], order=1).reshape(-1, len(offs))
            der = np.abs(np.diff(prof, axis=1))
            # a borda forte mais perto da linha (o topo do rodapé costuma ser mais forte que a borda do piso)
            strong = der >= 0.5 * der.max(1, keepdims=True)
            k = np.where(strong, np.abs(offs[:-1] + 0.5)[None], np.inf).argmin(1)
            acc.extend(zip(np.abs(offs[k] + 0.5).tolist(), der[np.arange(len(k)), k].tolist()))
    out: dict = {"pontos_mascara": len(res_in), "pontos_visiveis": len(res_vis)}
    if res_vis:
        v = np.array(res_vis)
        strong_vis = float(np.median(v[:, 1]))
        out["distancia_visivel_px"] = round(float(np.median(v[:, 0])), 2)
    else:
        strong_vis = 6.0
    if len(res_in) >= 8:
        a = np.array(res_in)
        found = a[:, 1] >= max(2.0, 0.35 * strong_vis)
        out["continuidade"] = round(float(found.mean()), 3)
        out["distancia_px"] = round(float(np.median(a[found, 0])) if found.any() else float(radius), 2)
    return out


def redetected_furniture(seg, pan, mask: np.ndarray, min_area: float = LIMITS["redetectado_area_min"]) -> list:
    """Móveis removíveis que o OneFormer encontra na placa, com área > ``min_area`` e metade dentro da máscara."""
    from . import instances as ri

    H, W = mask.shape
    found = []
    for c in ri.gather_candidates(seg, pan):
        a = c.mask.sum()
        if a < min_area * H * W:
            continue
        if (c.mask & mask).sum() >= 0.5 * a:
            found.append({"rotulo": c.label, "area_pct": round(100.0 * a / (H * W), 2)})
    return found


def assess(photo: np.ndarray, plate: np.ndarray, mask: np.ndarray, labels: np.ndarray, groups: np.ndarray,
           box: RoomBox | None, redetect: Callable | None = None, fill_info: dict | None = None,
           min_area: float = 0.002) -> PlateQuality:
    """Nota automática da placa (0..100) e aprovação pelo :data:`LIMITS`. Plano preenchido que não pôde ser
    medido (nenhum pixel visível do próprio material em volta) reprova: o que não se mede não é aprovado."""
    per = plane_stats(photo, plate, mask, labels, groups, min_area=min_area)
    line = line_stats(plate, photo, mask, box, labels=labels) if box is not None else {}
    area = float(mask.mean())
    reasons = []
    terms = []
    H, W = mask.shape
    n_mask = int(mask.sum())
    unplanned = float((mask & np.isin(labels, (OTHER, PROTECTED))).sum()) / n_mask if n_mask else 0.0
    if area >= LIMITS["sem_plano_area_min"]:
        # cor média certa e estrutura errada (névoa, fantasma translúcido, parede inventada) passam pela costura;
        # sem plano medido não há como aprovar uma sala tão grande inventada
        if unplanned >= LIMITS["sem_plano_frac_max"]:
            reasons.append(f"geometria da sala não encontrada: {100 * unplanned:.0f}% do preenchimento sem plano "
                           f"medido ({100 * area:.0f}% da foto)")
        terms.append(np.clip(1 - (unplanned - 0.2) / 0.3, 0, 1))
    for L in np.unique(labels[mask]):
        if L in (OTHER, PROTECTED):
            continue
        name = LABEL_NAMES.get(int(L), f"parede{int(L)}")
        n = int((mask & (labels == L)).sum())
        if name in per or n < min_area * H * W or (fill_info or {}).get(name, {}).get("metodo") == "liso":
            continue
        reasons.append(f"{name} preenchido sem referência visível do material ({100 * n / (H * W):.1f}% da foto)")
        terms.append(0.0)
    # o limite da cor do preenchimento só relaxa se o que não tem plano não tem mancha forte (resto de móvel que a
    # reconferência não pegou): senão uma tentativa com fantasma passaria antes da tentativa limpa
    relax_fill = per.get("sem_plano", {}).get("costura_p90_de", 0.0) <= LIMITS["costura_sem_plano_max"]
    for name, s in per.items():
        if name == "sem_plano":
            if s["costura_de"] > LIMITS["costura_sem_plano_max"]:
                reasons.append(f"mancha no preenchimento sem plano (ΔE {s['costura_de']:.1f})")
            terms.append(np.clip(1 - (s["costura_de"] - 4.0) / 12.0, 0, 1))
            continue
        if s["preenchimento_de"] > fill_de_limit(s["costura_de"], s["grao_log2"], relax_fill):
            reasons.append(f"cor do preenchimento destoa no(a) {name} (ΔE {s['preenchimento_de']:.1f})")
        if s["costura_de"] > LIMITS["costura_de_max"]:
            reasons.append(f"costura visível no(a) {name} (ΔE {s['costura_de']:.1f})")
        if abs(s["grao_log2"]) > LIMITS["grao_log2_max"]:
            reasons.append(f"textura diferente no(a) {name} (grão {2 ** s['grao_log2']:.2f}×)")
        terms.append(np.clip(1 - (s["costura_de"] - 2.0) / 8.0, 0, 1))
        terms.append(np.clip(1 - (abs(s["grao_log2"]) - 0.3) / 1.2, 0, 1))
    if "distancia_px" in line:
        if line["distancia_px"] > LIMITS["linha_px_max"]:
            reasons.append(f"linha parede/piso fora do lugar ({line['distancia_px']:.1f} px)")
        if line["continuidade"] < LIMITS["linha_continuidade_min"]:
            reasons.append(f"rodapé/linha parede-piso interrompido ({100 * line['continuidade']:.0f}% contínuo)")
        terms.append(np.clip(1 - (line["distancia_px"] - 2.0) / 6.0, 0, 1))
        terms.append(np.clip(line["continuidade"] / 0.8, 0, 1))
    if area > LIMITS["area_gerada_max"]:
        reasons.append(f"área reconstruída muito grande ({100 * area:.0f}% da foto)")
    terms.append(np.clip(1 - (area - 0.5) / 0.4, 0, 1))
    red = []
    if redetect is not None:
        seg2, pan2 = redetect(plate)
        red = redetected_furniture(seg2, pan2, mask)
        if red:
            reasons.append("móvel reaparece na sala vazia: " + ", ".join(r["rotulo"] for r in red))
        terms.append(0.0 if red else 1.0)
    raw = 100.0 * float(np.mean(terms)) if terms else 50.0
    score = min(raw, 59.0) if reasons else raw
    return PlateQuality(score, not reasons, per, line, area, red, reasons, raw, unplanned)


# ============================================================================================================
# Preenchimento por plano
# ============================================================================================================
def _protected_labels(mask, labels, groups, box: RoomBox | None, diag) -> np.ndarray:
    """Pixels mascarados que pertencem a janela/porta/cortina: dentro de uma abertura do modelo da sala ou a
    menos de PROTECT_REACH de pixels visíveis dessas classes."""
    prot_vis = np.isin(groups, G_PROTECTED) & ~mask
    near = mask & _dil(prot_vis, max(2, int(PROTECT_REACH * diag)))
    if box is not None:
        for op in box.room.openings:
            sel = mask & (labels == op.wall) & np.isfinite(box.S)
            ins = (box.S >= op.s0) & (box.S <= op.s1)
            z0 = op.z0
            # a borda de baixo da abertura foi medida só nos pixels visíveis: se um móvel a esconde, a janela
            # (ou porta de sacada) pode continuar até o chão atrás dele
            edge = (labels == op.wall) & ins & (box.Z >= op.z0 - 0.05) & (box.Z <= op.z0 + 0.10)
            if edge.sum() >= 20 and (edge & mask).sum() >= 0.3 * edge.sum():
                z0 = 0.0
            near |= sel & ins & (box.Z >= z0) & (box.Z <= op.z1)
    return near


def _resting_on_kept(mask, labels, groups, diag, ring_frac: float = 0.015) -> np.ndarray:
    """Pixels mascarados cujo plano da caixa (piso/teto/parede) está escondido atrás de algo que FICA na foto
    (copo sobre a bancada mantida, vaso sobre a ilha): o anel visível em volta quase não tem o material do plano
    e é dominado por objetos. Preencher na vista do plano aí inventa uma mancha de piso/parede no meio da bancada;
    esses pixels vão para o espaço da imagem com contexto local."""
    out = np.zeros(mask.shape, bool)
    plane = mask & ((labels >= 0) | (labels == FLOOR) | (labels == CEIL))
    if not plane.any():
        return out
    surfaces = np.isin(groups, (G_FLOOR, G_WALL, G_CEIL) + G_PROTECTED)
    r = max(4, int(ring_frac * diag))
    H, W = mask.shape
    for L in np.unique(labels[plane]):
        grp = G_FLOOR if L == FLOOR else (G_CEIL if L == CEIL else G_WALL)
        n, cc, st, _ = cv2.connectedComponentsWithStats((plane & (labels == L)).astype(np.uint8), connectivity=8)
        for k in range(1, n):
            x, y, w, h, _ = st[k]
            x0, y0, x1, y1 = max(0, x - r - 1), max(0, y - r - 1), min(W, x + w + r + 1), min(H, y + h + r + 1)
            comp = cc[y0:y1, x0:x1] == k
            ring = _dil(comp, r) & ~mask[y0:y1, x0:x1]
            nr = int(ring.sum())
            if nr < 30:
                continue
            g = groups[y0:y1, x0:x1][ring]
            if (g == grp).mean() < 0.15 and surfaces[y0:y1, x0:x1][ring].mean() < 0.5:
                out[y0:y1, x0:x1] |= comp
    return out


def clutter_mask(photo: np.ndarray, surface: np.ndarray, max_area: float = 0.0015, de: float = 14.0) -> np.ndarray:
    """Objetos pequenos e contrastantes sobre uma superfície (tomada, interruptor, quadro pequeno): não servem de
    contexto para o preenchimento, senão o LaMa os repete na parede (objeto inventado)."""
    H, W = surface.shape
    lab = _lab(photo)
    med = _nblur(lab, surface, max(6.0, 0.02 * np.hypot(H, W)))
    dev = np.linalg.norm(lab - med, axis=2) > de
    blobs = _dil(dev & surface, 2)
    n, cc, st, _ = cv2.connectedComponentsWithStats(blobs.astype(np.uint8), connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] <= max_area * H * W
    return small[cc] & surface


def _inpaint_local(inpaint: InpaintFn, img: np.ndarray, hole: np.ndarray, margin_frac: float = 0.25) -> np.ndarray:
    """Inpainting no espaço da imagem, um recorte por componente (contexto local)."""
    out = img.copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    H, W = hole.shape
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        mg = max(24, int(margin_frac * max(w, h)))
        x0, y0, x1, y1 = max(0, x - mg), max(0, y - mg), min(W, x + w + mg), min(H, y + h + mg)
        sub = lab[y0:y1, x0:x1] == k
        res = inpaint(out[y0:y1, x0:x1], sub)
        out[y0:y1, x0:x1][sub] = res[sub]
    return out


def fill_planes(inpaint: InpaintFn, photo: np.ndarray, mask: np.ndarray, groups: np.ndarray, box: RoomBox | None,
                depth: np.ndarray | None = None, rng: np.random.Generator | None = None,
                max_side: int = MAX_SIDE) -> tuple[np.ndarray, np.ndarray, dict]:
    """Preenche ``mask`` por plano. Retorna (placa, rótulos, informações por plano)."""
    rng = rng or np.random.default_rng(0)
    H, W = mask.shape
    diag = float(np.hypot(H, W))
    info: dict = {}
    if box is None:                                   # sem modelo da sala: só espaço da imagem
        out = inpaint(photo, mask)
        out[~mask] = photo[~mask]
        info["imagem"] = {"metodo": "lama-imagem", "area_px": int(mask.sum())}
        return out, np.where(mask, OTHER, OTHER).astype(np.int32), info
    labels = box.label.copy()
    if depth is not None:                             # visto por uma porta: bem mais longe que a caixa
        dep = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST) \
            if depth.shape != (H, W) else depth
        beyond = (dep > 1.2 * box.t) & ~np.isin(groups, G_PROTECTED) & np.isfinite(dep)
        agree = np.abs(dep - box.t) / np.maximum(box.t, 1e-3) < 0.12
        known = ~_dil(mask, 3)
        for L in np.unique(labels[mask]):
            bad = known & (labels == L) & beyond
            if L == OTHER or bad.sum() == 0:
                continue
            d_ok = cv2.distanceTransform((~(known & (labels == L) & agree)).astype(np.uint8), cv2.DIST_L2, 5)
            d_bad = cv2.distanceTransform((~bad).astype(np.uint8), cv2.DIST_L2, 5)
            labels[mask & (labels == L) & (d_bad < d_ok)] = OTHER
    # parede da caixa onde a foto mostra piso (MoGe/RoomModel erram muito em cozinhas planejadas): o plano do piso
    # é a parte confiável; os pixels mascarados dessa parede cujo vizinho visível é piso viram piso
    vis_surf = ~_dil(mask, 2) & np.isin(groups, (G_FLOOR, G_WALL, G_CEIL))
    if vis_surf.sum() >= 100:
        idx = ndimage.distance_transform_edt(~vis_surf, return_distances=False, return_indices=True)
        near = groups[idx[0], idx[1]]
        down = box.rays[..., 2] < 0
        for w in box.room.walls:
            visw = vis_surf & (labels == w.id)
            if visw.sum() < 50:
                continue
            ffrac = float((groups[visw] == G_FLOOR).mean())
            if ffrac > 0.3:
                sel = mask & (labels == w.id) & (near == G_FLOOR) & down
                labels[sel] = FLOOR
                info.setdefault("parede_virou_piso", {})[f"parede{w.id}"] = {"piso_visivel": round(ffrac, 2),
                                                                             "px": int(sel.sum())}
    prot = _protected_labels(mask, labels, groups, box, diag)
    labels[prot] = PROTECTED
    on_obj = _resting_on_kept(mask, labels, groups, diag)
    if on_obj.any():
        labels[on_obj] = OTHER
        info["apoiado_em_objeto_px"] = int(on_obj.sum())
    out = photo.copy()
    remaining = mask.copy()

    # ---- piso e teto (vista de cima / de baixo), contexto só do próprio material ------------------------------
    for key, grp in ((FLOOR, G_FLOOR), (CEIL, G_CEIL)):
        sel = mask & (labels == key)
        if sel.sum() < 20:
            continue
        r = box.rect_for(box.planes[key], sel, max_side)
        if r is None:
            labels[sel] = OTHER
            continue
        Hm, size = r
        same = (labels == key) & ~mask & (groups == grp)
        back, cover, pinfo = _fill_rect(inpaint, out, photo, mask, same, Hm, size, rng, ctx_restrict=True)
        done = sel & cover
        out[done] = back[done]
        remaining &= ~done
        labels[sel & ~cover] = OTHER
        info[LABEL_NAMES[key]] = {**pinfo, "area_px": int(done.sum()), "vista": list(size)}

    # ---- paredes ------------------------------------------------------------------------------------------------
    visible_wall = ~mask & (groups == G_WALL) & (labels >= 0)
    prof = wall_profile(photo, box, visible_wall)
    for w in box.room.walls:
        sel = mask & (labels == w.id)
        if sel.sum() < 20:
            continue
        tot = int((labels == w.id).sum())
        vis = visible_wall & (labels == w.id)
        frac = float(vis.sum()) / max(tot, 1)
        own = wall_profile(photo, box, vis) if vis.sum() >= 1500 else None
        if own is not None:               # cor da própria parede (paredes vizinhas podem ser de outro material)
            if own.baseboard_h == 0 and prof is not None:
                own.baseboard_h = prof.baseboard_h
            wprof = own
        else:
            wprof = prof
        if wprof is not None and (frac < PLAIN_WALL_VISIBLE or vis.sum() < 0.003 * H * W):
            light_sel = vis if vis.sum() >= 2000 else visible_wall
            paint_plain(out, photo, box, wprof, sel, light_sel, rng)
            ring_known = ~mask & (labels == w.id) & (groups == G_WALL)
            out = harmonize(out, ring_known, sel)
            remaining &= ~sel
            info[f"parede{w.id}"] = {"metodo": "liso", "visivel": round(frac, 3), "area_px": int(sel.sum()),
                                     "rodape_m": round(wprof.baseboard_h, 3)}
            continue
        r = box.rect_for(box.planes[w.id], sel, max_side, pad_px=60)
        if r is None:
            labels[sel] = OTHER
            continue
        Hm, size = r
        same = (labels == w.id) & ~mask & (groups == G_WALL)
        same &= ~clutter_mask(photo, same)
        back, cover, pinfo = _fill_rect(inpaint, out, photo, remaining | sel, same, Hm, size, rng,
                                        ctx_restrict=True)
        done = sel & cover
        out[done] = back[done]
        # rodapé reto: redesenha a faixa do rodapé (perfil das paredes visíveis) na altura certa
        if wprof is not None and wprof.baseboard_h > 0:
            bb = done & (box.Z < wprof.baseboard_h) & np.isfinite(box.Z)
            if bb.any():
                paint_plain(out, photo, box, WallProfile(wprof.z, wprof.rgb, wprof.count, wprof.baseboard_h,
                                                         wprof.grain * 0.5, None), bb, vis if vis.sum() >= 2000
                            else visible_wall, rng)
                pinfo["rodape_redesenhado_px"] = int(bb.sum())
        remaining &= ~done
        labels[sel & ~cover] = OTHER
        info[f"parede{w.id}"] = {**pinfo, "visivel": round(frac, 3), "area_px": int(done.sum()),
                                 "vista": list(size)}

    # ---- "outro" (vãos de porta, sem plano) e protegidos: espaço da imagem, contexto local --------------------
    for key in (OTHER, PROTECTED):
        sel = remaining & ((labels == key) if key == PROTECTED else (labels != PROTECTED))
        if sel.sum() == 0:
            continue
        out = _inpaint_local(inpaint, out, sel)
        remaining &= ~sel
        info[LABEL_NAMES[key]] = {"metodo": "lama-local", "area_px": int(sel.sum())}
    out[~mask] = photo[~mask]
    return out, labels, info


def _fill_rect(inpaint, cur, photo, hole_img, same_img, Hm, size, rng, ctx_restrict: bool):
    """Vista retificada: buraco = máscara inteira (+ fora da foto [+ outro material]); LaMa; textura; volta."""
    tw, th = size
    H, W = hole_img.shape
    rect = _warp_in(cur, Hm, size)
    orig = _warp_in(photo, Hm, size)
    rh = _dil(_warp_mask_in(hole_img, Hm, size), 2)
    inside = _warp_in(np.full((H, W), 255, np.uint8), Hm, size, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT) > 0
    same = _warp_mask_in(same_img, Hm, size, 200) & ~rh & inside
    n_hole = int((rh & inside).sum())
    if int(same.sum()) < max(MIN_CONTEXT_PX, MIN_CONTEXT_FRAC * n_hole):
        # quase nada do material do plano à vista: o LaMa com o buraco do tamanho da vista devolve uma cor
        # inventada (cinza/verde-azulado). Esses pixels vão para o espaço da imagem com contexto local.
        return cur, np.zeros((H, W), bool), {"metodo": "sem_contexto", "contexto_px": int(same.sum())}
    rmask = rh | ~inside
    if ctx_restrict:
        rmask |= ~same
    filled = inpaint(rect, rmask)
    filled, pinfo = _finish_surface(orig, filled, same, rmask & inside, rng)
    back = cv2.warpPerspective(filled, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    full = np.full((th, tw), 255, np.uint8)
    cover = cv2.warpPerspective(cv2.erode(full, disk(2)), Hm, (W, H), flags=cv2.INTER_NEAREST) > 0
    return back, cover, pinfo


# ============================================================================================================
# Pontos de entrada
# ============================================================================================================
def build_plate(photo: np.ndarray, pieces: Sequence, seg, room=None, depth: np.ndarray | None = None,
                inpaint: InpaintFn | None = None, redetect: Callable | None = None, seed: int = 0,
                max_side: int = MAX_SIDE, assess_quality: bool = True, strategy: str = "planos",
                core_dilate: float = CORE_DILATE) -> PlateResult:
    """Sala vazia a partir das peças a remover (``FurnitureInstance`` ou qualquer objeto com ``mask``, ``alpha``
    e ``kind``). ``seg``: segmentação semântica da foto. ``room``: ``RoomModel`` (sem ele, só espaço da
    imagem). ``depth``: profundidade do MoGe (detecta o que é visto por portas). ``redetect(placa) ->
    (seg, pan)``: OneFormer para a nota. ``strategy``: "planos" (LaMa por plano) ou "imagem" (LaMa no espaço da
    imagem; a nota usa os mesmos planos da caixa, para medir as duas do mesmo jeito). ``core_dilate``: dilatação
    do núcleo (fração da diagonal). Garante ``placa[~mask] == foto[~mask]`` byte a byte."""
    if strategy not in STRATEGIES:
        raise ValueError(f"estratégia de placa desconhecida: {strategy}")
    if inpaint is None:
        inpaint = load_lama()
    t: dict = {}
    t0 = time.perf_counter()
    H, W = photo.shape[:2]
    groups = seg.groups
    warnings: list[str] = []
    rng = np.random.default_rng(seed)
    if not pieces:
        return PlateResult(photo.copy(), np.zeros((H, W), bool), np.zeros((H, W), bool), np.zeros((H, W), bool),
                           np.zeros((H, W), bool), np.zeros((H, W), bool), np.full((H, W), OTHER, np.int32),
                           None, {"planos": {}, "tempos": {}}, ["nenhuma peça para remover"])
    box = None
    if room is not None and room.walls is not None:
        try:
            box = RoomBox(room, W, H)
        except Exception as e:  # modelo da sala degenerado: segue no espaço da imagem
            warnings.append(f"modelo da sala inválido ({e}); sala vazia só no espaço da imagem")
    else:
        warnings.append("sem modelo da sala: sala vazia só no espaço da imagem (linhas podem entortar)")
    core = core_mask(pieces, (H, W), core_dilate)
    cand = effect_candidates(pieces, groups, core)
    t["mascaras"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    plate1 = inpaint(photo, core | cand)                # passada 1 (barata): só para medir os efeitos
    t["passada1"] = time.perf_counter() - t0
    effect, shadow, light = verify_effects(photo, plate1, cand, core)
    del plate1
    mask = core | _dil(effect, 3)
    t0 = time.perf_counter()
    if strategy == "imagem":
        plate = np.array(inpaint(photo, mask), np.uint8, copy=True)
        plate[~mask] = photo[~mask]
        labels = box.label.copy() if box is not None else np.full((H, W), OTHER, np.int32)
        pinfo = {"imagem": {"metodo": "lama-imagem", "area_px": int(mask.sum())}}
    else:
        plate, labels, pinfo = fill_planes(inpaint, photo, mask, groups, box, depth, rng, max_side)
    t["passada2_planos"] = time.perf_counter() - t0
    if not np.array_equal(plate[~mask], photo[~mask]):     # regra absoluta: nunca deveria acontecer
        raise AssertionError("placa difere da foto fora da máscara")
    q = None
    if assess_quality:
        t0 = time.perf_counter()
        q = assess(photo, plate, mask, labels, groups, box, redetect, pinfo)
        t["nota"] = time.perf_counter() - t0
    return PlateResult(plate, mask, core, effect, shadow, light, labels, q,
                       {"planos": pinfo, "tempos": t, "estrategia": strategy, "dilatacao_nucleo": core_dilate},
                       warnings)


def build_plate_gated(photo: np.ndarray, inst_set, seg, room=None, depth=None, inpaint: InpaintFn | None = None,
                      redetect: Callable | None = None, seed: int = 0, max_side: int = MAX_SIDE,
                      alternatives: bool = True, on_attempt: Callable | None = None) -> PlateResult:
    """Portão de qualidade (§3.6 e PT-V3-8). Tentativas, parando na primeira aprovada:

    1. ``crua_por_plano``: soltos + planejados, LaMa por plano;
    2. ``planejados_mantidos`` (só se há planejados): só os soltos saem;
    3. ``lama_imagem`` (``alternatives``): LaMa direto na imagem, com o conjunto da melhor tentativa até aqui;
    4. ``mascara_ampla`` (``alternatives``): por plano com o núcleo dilatado :data:`GATE_WIDE_DILATE` vezes.

    Fica a melhor (aprovada > menos motivos > nota sem teto). Se nenhuma passa, devolve a melhor com aviso (a
    interface oferece gerar mesmo assim, o modo A ou outra foto); nunca entrega uma placa ruim em silêncio.
    ``info["tentativas"]`` registra cada tentativa (estratégia, nota, motivos, tempo). ``on_attempt(nome, placa)``
    é chamado depois de cada tentativa (progresso na interface)."""
    if inpaint is None:
        inpaint = load_lama()
    attempts: list[dict] = []

    def run(name: str, keep: bool, strategy: str, dil: float) -> PlateResult:
        t0 = time.perf_counter()
        r = build_plate(photo, inst_set.removal(keep), seg, room, depth, inpaint, redetect, seed, max_side,
                        strategy=strategy, core_dilate=dil)
        r.kept_builtins = keep
        r.info["estrategia_portao"] = name
        q = r.quality
        attempts.append({"estrategia": name, "planejados_mantidos": keep, "metodo": strategy,
                         "dilatacao_nucleo": round(dil, 4),
                         "nota": None if q is None else round(q.score, 1),
                         "nota_sem_teto": None if q is None or q.raw_score is None else round(q.raw_score, 1),
                         "aprovada": None if q is None else q.ok, "motivos": [] if q is None else list(q.reasons),
                         "area_gerada_pct": round(100 * float(r.mask.mean()), 2),
                         "tempo_s": round(time.perf_counter() - t0, 2)})
        if on_attempt is not None:
            on_attempt(name, r)
        return r

    # as reprovadas têm a nota limitada a 59: compara pela aprovação, pela área gerada dentro do limite (a máscara
    # larga não pode "ganhar" trocando uma costura por uma sala quase toda inventada), nº de motivos e nota sem teto
    rank = lambda r: (r.quality.ok, r.quality.area_generated <= LIMITS["area_gerada_max"],  # noqa: E731
                      -len(r.quality.reasons),
                      r.quality.raw_score if r.quality.raw_score is not None else r.quality.score)
    best = run("crua_por_plano", False, "planos", CORE_DILATE)
    if best.quality is not None and not best.quality.ok:
        first_reasons = list(best.quality.reasons)
        if any(i.kind == "planejado" for i in inst_set.instances):
            r2 = run("planejados_mantidos", True, "planos", CORE_DILATE)
            r2.warnings.append("sala crua reprovada (" + "; ".join(first_reasons) + "): planejados mantidos na foto")
            if rank(r2) >= rank(best):
                best = r2
        # área acima do limite: nenhuma alternativa passa (LaMa na imagem não muda a área e a máscara larga só a
        # aumenta), mesmo com outros motivos junto; não gasta ~2 preenchimentos + OneFormer à toa (foto lotada)
        area_blocked = any(r.startswith("área reconstruída") for r in best.quality.reasons)
        if alternatives and not best.quality.ok and not area_blocked:
            keep = best.kept_builtins
            for name, strategy, dil in (("lama_imagem", "imagem", CORE_DILATE),
                                        ("mascara_ampla", "planos", CORE_DILATE * GATE_WIDE_DILATE)):
                r = run(name, keep, strategy, dil)
                if rank(r) > rank(best):
                    r.warnings.extend(w for w in best.warnings if w not in r.warnings and "planejados" in w)
                    best = r
                if best.quality.ok:
                    break
        if best.quality.ok and best.info["estrategia_portao"] != "crua_por_plano":
            best.warnings.append(f"sala vazia aprovada na tentativa \"{best.info['estrategia_portao']}\" "
                                 f"(a primeira reprovou: {'; '.join(first_reasons)})")
    best.info["tentativas"] = attempts
    if best.quality is not None and not best.quality.ok:
        best.warnings.append("a sala vazia ficou abaixo da qualidade mínima (" + "; ".join(best.quality.reasons)
                             + "): use o modo A ou envie outra foto")
    return best
