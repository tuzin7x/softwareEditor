"""Refino SDXL da sala vazia do modo B (PLANO_V3 §3.2 item 4, PT-V3-4).

O big-lama deixa a sala vazia com névoa, fantasma dos móveis, piso inventado sob tapetes grandes e pedaços de parede
"soltos" (avaliação final do PT-V3-8: 2/10). Este estágio roda **depois** da placa LaMa e só **dentro** da máscara
de remoção:

1. **Estrutura da sala vazia, não da foto mobiliada.** O ControlNet de profundidade recebe a profundidade da caixa
   vazia (ray-cast do ``RoomModel``: piso, paredes e teto, sem os móveis), casada em escala com o MoGe fora da
   máscara. Sem modelo da sala, a profundidade do MoGe com o buraco preenchido por interpolação suave (push-pull).
2. **Início "cru".** Por padrão (``init="plano"``) o SDXL não parte da LaMa (que tem a silhueta dos móveis): dentro
   da máscara o ponto de partida é a cor esperada de cada plano (passa-baixa do material visível do mesmo plano,
   estendido para dentro) com ruído leve. O SDXL desenha textura de piso e parede lisa por cima.
3. **Prompt da sala vazia com os materiais da própria foto** (cor e tipo do piso visível, cor das paredes) e
   prompt negativo contra móveis, pessoas, texto, portas/janelas/tomadas novas, névoa e borrão.
4. **Composição segura.** Janelas, portas e cortinas (visíveis ou rotuladas "protegido" dentro da máscara) e todo
   pixel fora da máscara ficam ``array_equal`` à placa LaMa de entrada; a borda é uma rampa **por dentro** da
   máscara (sem halo fora); a cor de baixa frequência é casada, por plano, com o material visível em volta.
5. **2–3 sementes e escolha automática** pela própria nota do portão (:func:`mobilia.removal.plate.assess`, com
   névoa, fantasma, placas soltas, OneFormer de novo na placa: móvel ou porta/janela nova reprova). A semente só
   substitui a LaMa se for melhor que ela; senão fica a LaMa (fallback), e o motivo vai para o ``report.json``.

O gerador é injetável (``generate(imagem, máscara, controle, prompt, semente, força) -> imagem``, tudo na
resolução de trabalho): no produto é :class:`SdxlInpainter` (RealVisXL V5.0 Lightning + ControlNet depth small +
VAE fp16-fix, pesos com SHA-256 em ``analysis/weights.py``); nos testes rápidos, um stub sem modelo.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import cv2
import numpy as np
from scipy import ndimage

from ..removal import plate_score as ps

# --------------------------------------------------------------------------------------------------- constantes
ENV_SDXL = "MOBILIA_SDXL"   # "0" desliga; "1" força (até na CPU, para testes); "auto" (padrão) segue o perfil

ROOM_EN = {"sala": "living room", "quarto": "bedroom", "cozinha": "kitchen", "banheiro": "bathroom",
           "escritorio": "home office", "sala_jantar": "dining room"}

NEGATIVE = ("furniture, sofa, chair, table, bed, cabinet, shelf, rug, lamp, plant, painting, picture frame, "
            "mirror, curtain, appliance, person, text, watermark, new door, new window, power outlet, pipe, "
            "tile grout, bricks, fog, haze, blur, smudge, ghost, stain, cartoon, render")


@dataclass
class RefineConfig:
    """Parâmetros do refino. :meth:`for_device` dá os padrões da GPU e da CPU (mais leve)."""

    seeds: int = 3
    steps: int = 8                    # passos do agendador (Lightning); efetivos = steps × strength
    strength: float = 0.75
    guidance: float = 2.0
    control_scale: float = 0.7
    work_side: int = 1024             # lado maior da resolução de trabalho do SDXL
    init: str = "misto"               # "misto" (tom do plano + grão da LaMa) | "plano" (tom + ruído) | "lama"
    feather_frac: float = 0.012       # rampa da composição, por dentro da máscara (fração da diagonal)
    margin_frac: float = 0.02         # a máscara do SDXL passa da de remoção (contexto da costura)
    color_sigma_frac: float = 0.04    # passa-baixa do casamento de cor
    min_crop_frac: float = 0.5        # recorte mínimo (fração do lado maior da foto)
    min_gain: float = 1.0             # a semente só troca a LaMa se a nota (sem teto) subir pelo menos isto
    refine_protected_mask: bool = False  # True: refina também os pixels mascarados rotulados "protegido" (móvel na
                                         # frente da janela). Desligado: no teste o SDXL trocou o balcão visto
                                         # pela janela por parede com radiador. Janela/porta VISÍVEIS: nunca
    scope: str = "piso"               # "piso": SDXL só no piso; paredes lisas, teto e vãos viram pintura lisa
                                      # (membrana da cor da borda + grão); parede com textura fica na LaMa.
                                      # "tudo": SDXL em toda a máscara (inventou radiador e painéis no teste)
    grain: bool = True                # reinjeta o grão do material visível na pintura lisa
    floor_detail: str = "lama"        # "lama": do SDXL no piso só fica a baixa frequência (tom e luz: tira névoa,
                                      # sombra e mancha da silhueta); a textura (tábua, porcelanato, granilite) vem
                                      # da LaMa por plano, que copia o piso visível. "sdxl": textura do SDXL (na
                                      # avaliação ele desenhou placas com rejunte sobre porcelanato liso e ardósia
                                      # sobre madeira)
    floor_detail_sigma_frac: float = 0.01   # corte entre "tom" (SDXL) e "textura" (LaMa), fração da diagonal
    floor_detail_clip: float = 12.0   # amplitude máxima da textura da LaMa (níveis): contorno forte não passa

    @staticmethod
    def for_device(device: str) -> "RefineConfig":
        if str(device).startswith("cuda"):
            return RefineConfig()
        return RefineConfig(seeds=2, work_side=768)


@dataclass(frozen=True)
class Prompt:
    text: str
    negative: str
    floor: str = ""
    walls: str = ""

    def key(self) -> str:
        return hashlib.sha1((self.text + "\n" + self.negative).encode("utf-8")).hexdigest()[:16]


GenerateFn = Callable[[np.ndarray, np.ndarray, np.ndarray, Prompt, int, float], np.ndarray]


# --------------------------------------------------------------------------------------------------- utilidades
def _dil(m: np.ndarray, r: int) -> np.ndarray:
    return ps._dil(m, r)


def _lab(rgb: np.ndarray) -> np.ndarray:
    return ps._lab(rgb)


def _lab2rgb(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB)
    return np.clip(rgb * 255.0, 0, 255)


def _resize(img: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    if nearest:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


# --------------------------------------------------------------------------------------------------- prompt
def _color_word(lab_mean: np.ndarray) -> str:
    L, a, b = (float(x) for x in lab_mean)
    C = float(np.hypot(a, b))
    if C < 9:
        if L > 82:
            return "white"
        if L > 64:
            return "light gray"
        if L > 40:
            return "gray"
        return "dark gray"
    hue = float(np.degrees(np.arctan2(b, a))) % 360
    if 40 <= hue <= 100:          # tons quentes (bege, madeira)
        if L > 72:
            return "beige"
        if L > 50:
            return "light brown"
        return "dark brown"
    if 100 < hue <= 200:
        return "greenish gray" if C < 20 else "green"
    if 200 < hue <= 300:
        return "bluish gray" if C < 20 else "blue"
    return "warm gray" if C < 20 else "terracotta"


def describe_surfaces(photo: np.ndarray, mask: np.ndarray, groups: np.ndarray | None) -> dict:
    """Cor e tipo do piso visível e cor das paredes visíveis (heurística simples, vocabulário fixo em inglês para o
    prompt). Sem material visível suficiente, termos neutros."""
    H, W = mask.shape
    out = {"floor": "floor", "walls": "plain painted walls", "floor_color": None, "wall_color": None,
           "floor_kind": None}
    if groups is None:
        return out
    lab = _lab(photo)
    gray = cv2.cvtColor(np.ascontiguousarray(photo), cv2.COLOR_RGB2GRAY).astype(np.float32)
    keep = ~_dil(mask, max(2, int(0.006 * np.hypot(H, W))))
    near = _dil(mask, max(8, int(0.08 * np.hypot(H, W))))     # o material em volta do que vai ser preenchido
    floor = keep & (groups == ps.G_FLOOR)
    wall = keep & (groups == ps.G_WALL) & near
    if wall.sum() < 0.005 * H * W:
        wall = keep & (groups == ps.G_WALL)
    if wall.sum() >= 0.005 * H * W:
        cw = _color_word(np.median(lab[wall], axis=0))
        out.update(wall_color=cw, walls=f"plain {cw} painted walls")
    if floor.sum() >= 0.004 * H * W:
        cf = _color_word(np.median(lab[floor], axis=0))
        # tipo: tábuas de madeira = textura orientada (tensor de estrutura coerente) em tom quente; grão alto sem
        # orientação = revestimento texturizado (carpete, granilite); grão baixo = piso liso (porcelanato, cimento)
        g = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        s = 3.0
        jxx, jyy, jxy = (cv2.GaussianBlur(v, (0, 0), s) for v in (gx * gx, gy * gy, gx * gy))
        coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-3)
        hp = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.5))
        coherence = float(np.median(coh[floor]))
        grain = float(np.median(hp[floor]))
        warm = cf in ("beige", "light brown", "dark brown")
        if coherence > 0.45 and warm:
            kind = "wooden plank floor"
        elif grain > 2.5:
            kind = "textured floor"
        else:
            kind = "smooth tiled floor"
        # o tipo vai só para o relatório: no prompt ele virava ordem ("smooth tiled floor" fez o SDXL desenhar
        # placas com rejunte sobre porcelanato liso e ardósia sobre madeira, avaliação SDXL rodada 1). O prompt só
        # leva a cor; a textura vem do piso visível (início pela LaMa por plano, que copia o material em volta)
        out.update(floor_color=cf, floor_kind=kind, floor=f"{cf} floor", floor_coherence=round(coherence, 2),
                   floor_grain=round(grain, 2))
    return out


def build_prompt(desc: dict, room_type: str = "sala") -> Prompt:
    room = ROOM_EN.get(room_type, "room")
    text = (f"photo of an empty unfurnished {room}, bare {desc.get('floor', 'floor')} continuing the existing floor, "
            f"{desc.get('walls', 'plain painted walls')}, nothing on the walls, clean empty space, "
            "natural light, real estate photography, sharp focus, realistic")
    return Prompt(text, NEGATIVE, desc.get("floor", ""), desc.get("walls", ""))


# --------------------------------------------------------------------------------------------------- estrutura
def empty_room_depth(shape: tuple[int, int], mask: np.ndarray, room=None, depth: np.ndarray | None = None,
                     box=None) -> np.ndarray:
    """Mapa de controle (H, W) em [0, 1], perto = claro (convenção MiDaS do ControlNet depth). Dentro da máscara:
    profundidade da **caixa vazia** (sem móveis), casada em escala com o MoGe nos planos visíveis; fora: o MoGe.
    Sem caixa: o MoGe com o buraco preenchido por interpolação suave da disparidade (push-pull)."""
    from ..removal.plate import push_pull

    H, W = shape
    moge = None
    if depth is not None:
        moge = cv2.resize(np.asarray(depth, np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        moge = np.where(np.isfinite(moge) & (moge > 1e-3), moge, np.nan)
    if box is None and room is not None and getattr(room, "walls", None) is not None:
        try:
            from ..removal.plate import RoomBox

            box = RoomBox(room, W, H)
        except Exception:
            box = None
    z = np.full((H, W), np.nan, np.float32)
    if box is not None:
        t = np.where(np.isfinite(box.t) & (box.t > 0) & (box.label != ps.OTHER), box.t, np.nan).astype(np.float32)
        if moge is not None:
            vis = ~_dil(mask, 4) & np.isfinite(t) & np.isfinite(moge)
            if vis.sum() > 100:
                k = float(np.median(moge[vis] / t[vis]))
                if np.isfinite(k) and 0.2 < k < 5:
                    t = t * k
            # dentro da máscara nunca o MoGe (ele tem os móveis): onde a caixa não diz nada, interpolação suave
            z = np.where(mask, t, np.where(np.isfinite(moge), moge, t))
        else:
            z = t
    elif moge is not None:
        z = np.where(mask, np.nan, moge)
    disp = np.where(np.isfinite(z) & (z > 1e-3), 1.0 / np.maximum(z, 1e-3), np.nan).astype(np.float32)
    known = np.isfinite(disp)
    if not known.any():
        return np.full((H, W), 0.5, np.float32)
    filled = push_pull(np.where(known, disp, 0.0).astype(np.float32), known.astype(np.float32))
    lo, hi = np.percentile(filled, [1, 99])
    out = np.clip((filled - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.GaussianBlur(out.astype(np.float32), (0, 0), 1.0)


def protected_pixels(mask: np.ndarray, labels: np.ndarray | None, groups: np.ndarray | None,
                     include_masked: bool = True) -> np.ndarray:
    """Janelas, portas e cortinas **visíveis** (fora da máscara): nunca passam pelo SDXL (o protótipo transformou a
    sacada em ladrilhos e inventou porta e tomada). ``include_masked``: também os pixels mascarados que o portão
    rotulou "protegido" (móvel na frente da janela); por padrão o refino os trata (senão o fantasma da mesa na
    frente da cortina fica), e uma janela/porta nova longe das da foto reprova a semente."""
    prot = np.zeros(mask.shape, bool)
    if groups is not None:
        prot |= np.isin(groups, ps.G_PROTECTED) & ~mask
    if labels is not None and include_masked:
        prot |= mask & (labels == ps.PROTECTED)
    return prot


def crop_box(gen_mask: np.ndarray, min_frac: float = 0.5, pad_frac: float = 0.06) -> tuple[int, int, int, int]:
    """Recorte (y0, y1, x0, x1) que cobre a máscara do SDXL com folga, com lado mínimo ``min_frac`` do lado maior
    da foto (o SDXL precisa de contexto e de ~768 px para não inventar detalhe em escala errada)."""
    H, W = gen_mask.shape
    ys, xs = np.nonzero(gen_mask)
    if ys.size == 0:
        return 0, H, 0, W
    pad = int(pad_frac * np.hypot(H, W))
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
    m = int(min_frac * max(H, W))

    def grow(a0, a1, n):
        need = min(n, m) - (a1 - a0)
        if need > 0:
            a0 = max(0, a0 - need // 2)
            a1 = min(n, a0 + max(a1 - a0, min(n, m)))
            a0 = max(0, a1 - max(a1 - a0, min(n, m)))
        return a0, a1

    y0, y1 = grow(y0, y1, H)
    x0, x1 = grow(x0, x1, W)
    return int(y0), int(y1), int(x0), int(x1)


def work_size(h: int, w: int, side: int, max_up: float = 2.5) -> tuple[int, int]:
    """(largura, altura) da resolução de trabalho: lado maior ``side`` (sem ampliar mais que ``max_up``), múltiplos
    de 8."""
    s = min(side / max(h, w), max_up)
    return max(64, int(round(w * s / 8)) * 8), max(64, int(round(h * s / 8)) * 8)


def prepare_init(plate: np.ndarray, region: np.ndarray, ref: ps.PlaneRef, mode: str = "plano",
                 seed: int = 0) -> np.ndarray:
    """Ponto de partida do SDXL. ``lama``: a placa LaMa. ``plano``: dentro da região, a cor esperada de cada plano
    (sem a silhueta dos móveis) + metade do grão da LaMa + ruído leve; fora, a placa."""
    if mode == "lama":
        return plate.copy()
    if mode not in ("plano", "misto"):
        raise ValueError(f"início do refino desconhecido: {mode}")
    H, W = region.shape
    diag = float(np.hypot(H, W))
    base = _lab2rgb(ref.lab)
    if mode == "misto":
        # tom do plano (sem névoa nem mancha da silhueta) + textura da LaMa (grão do material, não a forma)
        f = plate.astype(np.float32)
        hp = f - cv2.GaussianBlur(f, (0, 0), max(2.0, 0.006 * diag))
        hp = np.clip(hp, -5, 5)                         # só o grão: contornos fortes (quadro, sofá) não passam
        base = base + 0.8 * hp
    rng = np.random.default_rng(seed)
    # sem o grão da LaMa: a borda do quadro e o contorno do sofá que ela deixa viram "objeto" para o SDXL
    noise = cv2.GaussianBlur(rng.normal(0, 4.0, (H, W, 1)).astype(np.float32), (0, 0), max(0.8, 0.0015 * diag))
    if noise.ndim == 2:
        noise = noise[..., None]
    flat = np.clip(base + noise, 0, 255)
    out = plate.copy()
    out[region] = flat[region].round().astype(np.uint8)
    return out


def match_color(gen: np.ndarray, region: np.ndarray, labels: np.ndarray | None, ref: ps.PlaneRef,
                sigma: float) -> np.ndarray:
    """Casa a cor do gerado com a do material visível, como uma membrana: a diferença de cor (passa-baixa) é medida
    só na faixa junto à borda da região, onde a cor esperada vem do anel real, e interpolada suavemente para dentro
    (push-pull), região por região de plano. Mantém a textura e a estrutura do SDXL (rodapé, juntas) e tira a
    deriva de tom dele; no meio de um buraco grande, a correção é a da borda, não uma cor inventada."""
    from ..removal.plate import push_pull

    H, W = region.shape
    lab = _lab(gen)
    out = lab.copy()
    band_w = max(3.0, sigma)
    din = ndimage.distance_transform_edt(region)
    band = region & (din <= band_w)
    s_small = max(2.0, 0.25 * sigma)
    parts = [region & (labels == L) for L in np.unique(labels[region])] if labels is not None else [region]
    for sel in parts:
        b = band & sel
        if sel.sum() < 20 or b.sum() < 20:
            continue
        low = ps._nblur(lab, sel, s_small)
        diff = np.zeros_like(lab)
        diff[b] = (ref.lab - low)[b]
        d_s = ps._nblur(diff, b, s_small)
        field = push_pull(np.where(b[..., None], d_s, 0).astype(np.float32), b.astype(np.float32))
        field = ps._nblur(field, sel, sigma)
        out[sel] = lab[sel] + field[sel]
    return _lab2rgb(out).round().astype(np.uint8)


PLAIN_CONTRAST_MAX = 3.0   # contraste local (níveis) do material visível até o qual a superfície é "lisa"


def split_scope(plate: np.ndarray, region: np.ndarray, labels: np.ndarray | None, ref: ps.PlaneRef,
                scope: str = "piso") -> tuple[np.ndarray, np.ndarray]:
    """(onde o SDXL gera, onde vira pintura lisa) dentro da região. ``piso``: o SDXL só no piso (o que mais
    aparece e o que a LaMa mais erra: material inventado sob o tapete, névoa, sombra da peça); parede e teto
    **lisos** (pouca textura visível) e vãos sem plano viram pintura lisa (a cor da borda, interpolada, com o
    grão do material): "cru, mas bem feito"; parede com textura (azulejo, ripado) fica com a LaMa por plano, que
    repete o padrão. Sem rótulos de plano (sem modelo da sala): SDXL em tudo."""
    if scope == "tudo" or labels is None or not ((labels[region] >= 0) | (labels[region] == ps.FLOOR)
                                                 | (labels[region] == ps.CEIL)).any():
        return region.copy(), np.zeros_like(region)
    if scope != "piso":
        raise ValueError(f"escopo do refino desconhecido: {scope}")
    sdxl = region & (labels == ps.FLOOR)
    plain = np.zeros_like(region)
    H, W = region.shape
    gray = cv2.cvtColor(np.ascontiguousarray(plate), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lc = ps.local_contrast(gray, max(1.5, 0.004 * np.hypot(H, W)))
    for L in np.unique(labels[region]):
        L = int(L)
        sel = region & (labels == L)
        if L == ps.FLOOR:
            continue
        known = ref.known.get(L)
        smooth = known is None or known.sum() < 50 or float(np.median(lc[known])) <= PLAIN_CONTRAST_MAX
        if L in (ps.OTHER, ps.PROTECTED) or smooth:
            plain |= sel
    return sdxl, plain


def reinject_grain(img: np.ndarray, region: np.ndarray, labels: np.ndarray | None, ref: ps.PlaneRef,
                   rng: np.random.Generator) -> np.ndarray:
    """Grão (passa-alta) de retalhos do material visível do mesmo plano, só onde o gerado ficou mais liso que ele
    (granilite, carpete e madeira saem "plásticos" do SDXL em resolução de trabalho)."""
    from ..removal.plate import grain_transfer

    if labels is None or not ref.known:
        return img
    H, W = region.shape
    patch = max(16, int(0.04 * np.hypot(H, W)) // 2 * 2)
    out = img
    for L, known in ref.known.items():
        sel = region & (labels == L)
        if sel.sum() < 200 or known.sum() < 4 * patch * patch:
            continue
        out = grain_transfer(out, known, sel, rng, patch=patch)
    return out


def floor_texture_from_lama(gen: np.ndarray, lama: np.ndarray, sel: np.ndarray, sigma: float,
                            clip: float = 12.0) -> np.ndarray:
    """Piso = baixa frequência do SDXL + alta frequência da LaMa por plano, só em ``sel``. O SDXL fica com o que
    acerta (tom, luz, sem névoa nem sombra do móvel) e a textura vem da LaMa, que continua o piso visível na vista
    retificada (sem material inventado). A textura é limitada a ±``clip`` níveis (contorno de móvel não passa)."""
    if not sel.any():
        return gen
    g, l_ = gen.astype(np.float32), lama.astype(np.float32)
    low_g = ps._nblur(g, sel, sigma)
    hp_l = np.clip(l_ - ps._nblur(l_, sel, sigma), -clip, clip)
    out = gen.copy()
    out[sel] = np.clip(low_g + hp_l, 0, 255).round().astype(np.uint8)[sel]
    return out


def feather_alpha(region: np.ndarray, px: float) -> np.ndarray:
    """Peso da composição: 0 na borda da região (e fora), 1 a ``px`` para dentro. Rampa só por dentro: nada muda
    fora da máscara."""
    d = ndimage.distance_transform_edt(region).astype(np.float32)
    a = np.clip(d / max(px, 1.0), 0.0, 1.0)
    return a * a * (3 - 2 * a)                     # smoothstep


def composite(plate: np.ndarray, gen: np.ndarray, alpha: np.ndarray, region: np.ndarray) -> np.ndarray:
    out = plate.astype(np.float32) * (1 - alpha[..., None]) + gen.astype(np.float32) * alpha[..., None]
    out = np.clip(out.round(), 0, 255).astype(np.uint8)
    out[~region] = plate[~region]
    return out


# --------------------------------------------------------------------------------------------------- refino
@dataclass
class SeedResult:
    seed: int
    plate: np.ndarray
    quality: object
    seconds: float
    info: dict = field(default_factory=dict)

    def summary(self) -> dict:
        q = self.quality
        return {"semente": self.seed, "nota": None if q is None else round(q.score, 1),
                "nota_sem_teto": None if q is None or q.raw_score is None else round(q.raw_score, 1),
                "aprovada": None if q is None else q.ok, "motivos": [] if q is None else list(q.reasons),
                "componentes": None if q is None else q.components, "tempo_s": round(self.seconds, 2), **self.info}


def _labels_of(items) -> set:
    return {str(x.get("rotulo")) for x in (items or [])}


def new_objects(q, base_q) -> list:
    """Móveis/aberturas que o OneFormer acha na semente e **não** achava na placa LaMa de entrada. O que já
    estava na LaMa (a reconferência é ruidosa: "toalha", "janela" na névoa da LaMa) não é culpa do SDXL; antes
    toda semente era descartada por isso (14 de 22 tentativas na avaliação) e o SDXL rodava à toa."""
    base = _labels_of(getattr(base_q, "redetected", [])) | _labels_of(getattr(base_q, "openings", []))
    return sorted((_labels_of(q.redetected) | _labels_of(getattr(q, "openings", []))) - base)


def _rank(q, base_q=None) -> tuple:
    clean = not new_objects(q, base_q) if base_q is not None else \
        not q.redetected and not getattr(q, "openings", [])
    raw = q.raw_score if q.raw_score is not None else q.score
    # 1º sem objeto novo; 2º aprovada; 3º a PIOR falha dura (névoa, fantasma, estrutura inventada, material,
    # contorno), arredondada a 0,1 (uma semente com rejunte inventado não ganha pela nota média); 4º a nota contínua
    return (clean, q.ok, -round(float(getattr(q, "hard", 0.0)), 1), raw)


HARD_SLACK = 0.15   # uma falha dura da semente pode passar a da LaMa até isto (ruído da medida) ...
HARD_FLOOR = 0.30   # ... e abaixo disto nunca é "piora"


def hard_regressions(q, base_q) -> dict:
    """Falhas duras (névoa, fantasma, estrutura inventada, material, contorno) em que a semente ficou pior que a
    placa LaMa: ``{nome: (semente, lama)}``. Uma só piora descarta a semente, por melhor que seja a nota média
    (a nota subia com o SDXL enquanto o piso virava placas com rejunte: avaliação SDXL rodada 1)."""
    hs, hb = getattr(q, "hard_detail", {}) or {}, getattr(base_q, "hard_detail", {}) or {}
    return {k: (round(v, 2), round(hb.get(k, 0.0), 2)) for k, v in hs.items()
            if v > max(HARD_FLOOR, hb.get(k, 0.0) + HARD_SLACK)}


def refine_plate(photo: np.ndarray, res, seg, room=None, depth: np.ndarray | None = None,
                 generate: GenerateFn | None = None, redetect: Callable | None = None, seed: int = 0,
                 cfg: RefineConfig | None = None, room_type: str = "sala",
                 on_seed: Callable | None = None):
    """Refina a placa ``res`` (``PlateResult`` da LaMa) com o SDXL. Devolve um ``PlateResult`` com a mesma máscara
    (só a placa e a nota mudam) e ``info["refino_sdxl"]`` (sementes, notas, escolhida, tempos, motivo). Sempre
    garante ``placa[~região] == placa_lama[~região]`` (fora da máscara e em janelas/portas). **Qualquer** erro
    (geração, nota com o OneFormer sem VRAM, casamento de cor...) devolve a placa LaMa com o motivo."""
    try:
        return _refine_plate(photo, res, seg, room, depth, generate, redetect, seed, cfg, room_type, on_seed)
    except Exception as e:  # noqa: BLE001
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        res.info["refino_sdxl"] = {"usado": False, "motivo": f"erro no refino SDXL ({type(e).__name__}: "
                                                              f"{str(e)[:200]}); ficou a placa LaMa"}
        return res


def _refine_plate(photo, res, seg, room, depth, generate, redetect, seed, cfg, room_type, on_seed):
    from ..removal import plate as rp

    cfg = cfg or RefineConfig()
    t_all = time.perf_counter()
    H, W = res.mask.shape
    diag = float(np.hypot(H, W))
    groups = getattr(seg, "groups", None)
    labels = res.labels
    info: dict = {"usado": False, "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}}
    prot = protected_pixels(res.mask, labels, groups, include_masked=not cfg.refine_protected_mask)
    region = res.mask & ~prot
    if generate is None or region.sum() < 0.0005 * H * W:
        info["motivo"] = "sem gerador" if generate is None else "nada para refinar"
        res.info["refino_sdxl"] = info
        return res
    box = None
    if room is not None and getattr(room, "walls", None) is not None:
        try:
            box = rp.RoomBox(room, W, H)
        except Exception:
            box = None
    rug = rp.rug_mask(seg, (H, W))
    base_q = res.quality
    if base_q is None:
        base_q = rp.assess(photo, res.plate, res.mask, labels, groups, box, redetect, res.info.get("planos"),
                           core=res.core, rug=rug)
    # estrutura, início e prompt (iguais para todas as sementes)
    t0 = time.perf_counter()
    ref_init = ps.plane_reference(res.plate, res.mask, labels, groups, sigma_frac=0.004, reach_frac=0.02)
    ref_color = ps.plane_reference(res.plate, res.mask, labels, groups, sigma_frac=0.25 * cfg.color_sigma_frac,
                                   reach_frac=0.03)
    control = empty_room_depth((H, W), res.mask, room, depth, box)
    sdxl_sel, plain_sel = split_scope(res.plate, region, labels, ref_init, cfg.scope)
    gen_mask = _dil(sdxl_sel, max(2, int(cfg.margin_frac * diag))) & ~(prot & ~res.mask)
    if not gen_mask.any():
        gen_mask = _dil(region, max(2, int(cfg.margin_frac * diag))) & ~(prot & ~res.mask)
    y0, y1, x0, x1 = crop_box(gen_mask, cfg.min_crop_frac)
    ww, wh = work_size(y1 - y0, x1 - x0, cfg.work_side)
    desc = describe_surfaces(photo, res.mask, groups)
    prompt = build_prompt(desc, room_type)
    alpha = feather_alpha(region, cfg.feather_frac * diag)
    m_w = _resize(gen_mask[y0:y1, x0:x1].astype(np.uint8), (ww, wh), nearest=True) > 0
    c_w = _resize(control[y0:y1, x0:x1], (ww, wh))
    info.update(prompt=prompt.text, materiais=desc, escopo=cfg.scope,
                area_sdxl_pct=round(100 * float(sdxl_sel.sum()) / max(1, int(region.sum())), 1),
                area_lisa_pct=round(100 * float(plain_sel.sum()) / max(1, int(region.sum())), 1), recorte=[y0, y1, x0, x1], resolucao_trabalho=[ww, wh],
                preparo_s=round(time.perf_counter() - t0, 2), nota_lama=round(base_q.score, 1),
                nota_lama_sem_teto=None if base_q.raw_score is None else round(base_q.raw_score, 1))
    seeds = [int(seed) * 1000 + 17 * k + 1 for k in range(max(1, cfg.seeds))]
    results: list[SeedResult] = []
    for s in seeds:
        t0 = time.perf_counter()
        init = prepare_init(res.plate, region, ref_init, cfg.init, s)
        plain = prepare_init(res.plate, region, ref_init, "plano", s + 1)
        init[plain_sel] = plain[plain_sel]          # o SDXL vê as paredes já lisas (contexto coerente)
        i_w = _resize(init[y0:y1, x0:x1], (ww, wh))
        try:
            out_w = np.asarray(generate(i_w, m_w, c_w, prompt, s, cfg.strength), np.uint8)
        except Exception as e:  # falta de memória, driver... fica a LaMa
            info.setdefault("erros", []).append(f"semente {s}: {type(e).__name__}: {str(e)[:200]}")
            break
        t_gen = time.perf_counter() - t0
        full = res.plate.copy()
        full[y0:y1, x0:x1] = _resize(out_w, (x1 - x0, y1 - y0))
        keep = region & ~sdxl_sel & ~plain_sel        # parede com textura fora do escopo: fica a LaMa
        full[plain_sel] = plain[plain_sel]
        full[keep] = res.plate[keep]
        if cfg.floor_detail == "lama":
            full = floor_texture_from_lama(full, res.plate, sdxl_sel, max(2.0, cfg.floor_detail_sigma_frac * diag),
                                           cfg.floor_detail_clip)
        elif cfg.floor_detail != "sdxl":
            raise ValueError(f"textura do piso desconhecida: {cfg.floor_detail}")
        full = match_color(full, region, labels, ref_color, max(3.0, cfg.color_sigma_frac * diag))
        if cfg.grain and plain_sel.any():   # só na pintura lisa: no piso o grão vem do próprio SDXL (os retalhos
            # do anel trazem sombra de pé de móvel e viram manchas)
            full = reinject_grain(full, plain_sel, labels, ref_init, np.random.default_rng(s))
        cand = composite(res.plate, full, alpha, region)
        if not np.array_equal(cand[~region], res.plate[~region]):     # regra absoluta: nunca deveria acontecer
            raise AssertionError("refino SDXL mudou pixels fora da região permitida")
        t1 = time.perf_counter()
        q = rp.assess(photo, cand, res.mask, labels, groups, box, redetect, res.info.get("planos"), core=res.core,
                      rug=rug)
        sr = SeedResult(s, cand, q, time.perf_counter() - t0, {"geracao_s": round(t_gen, 2),
                                                              "nota_s": round(time.perf_counter() - t1, 2)})
        results.append(sr)
        if on_seed is not None:
            on_seed(sr)
    info["sementes"] = [r.summary() for r in results]
    info["tempo_s"] = round(time.perf_counter() - t_all, 2)
    if not results:
        info["motivo"] = "o SDXL falhou; ficou a placa LaMa"
        res.info["refino_sdxl"] = info
        return res
    best = max(results, key=lambda r: _rank(r.quality, base_q))
    bq = best.quality
    base_raw = base_q.raw_score if base_q.raw_score is not None else base_q.score
    best_raw = bq.raw_score if bq.raw_score is not None else bq.score
    hard_b = float(getattr(base_q, "hard", 0.0))
    info["falha_dura_lama"] = round(hard_b, 3)
    # entre as sementes sem piora dura, a de melhor posição; se todas pioram algo, a melhor (para o motivo)
    ok_seeds = [r for r in results if not hard_regressions(r.quality, base_q) and not new_objects(r.quality, base_q)]
    if ok_seeds:
        best = max(ok_seeds, key=lambda r: _rank(r.quality, base_q))
        bq = best.quality
        best_raw = bq.raw_score if bq.raw_score is not None else bq.score
    worse = hard_regressions(bq, base_q)
    if new_objects(bq, base_q):
        info["motivo"] = ("todas as sementes têm objeto novo (" + ", ".join(new_objects(bq, base_q))
                          + "); ficou a placa LaMa")
    elif worse:
        info["motivo"] = ("todas as sementes pioram uma falha dura da LaMa ("
                          + ", ".join(f"{k} {a:.2f} contra {b:.2f}" for k, (a, b) in worse.items())
                          + "); ficou a placa LaMa")
    elif (bq.ok, best_raw) < (base_q.ok, base_raw) or best_raw < base_raw + cfg.min_gain:
        info["motivo"] = (f"o SDXL não melhorou a placa (nota {best_raw:.1f} contra {base_raw:.1f} da LaMa); "
                          "ficou a placa LaMa")
    else:
        info.update(usado=True, semente=best.seed, nota_final=round(bq.score, 1),
                    motivo=f"semente {best.seed} escolhida (nota {best_raw:.1f} contra {base_raw:.1f} da LaMa)")
        new_info = dict(res.info)
        new_info["refino_sdxl"] = info
        return replace(res, plate=best.plate, quality=bq, info=new_info)
    res.info["refino_sdxl"] = info
    if res.quality is None:
        res.quality = base_q
    return res


# --------------------------------------------------------------------------------------------------- modelo real
class SdxlInpainter:
    """RealVisXL V5.0 Lightning (UNet único, OpenRAIL++) + ControlNet depth small + VAE fp16-fix, pelo diffusers.

    GPU: fp16, com ``enable_model_cpu_offload`` quando pedido (8 GB: o OneFormer da nota divide a VRAM).
    CPU: bfloat16 (só testes e o perfil forçado). Os *text encoders* só são carregados para calcular os embeddings
    de um prompt novo, que ficam em cache no disco (``<dados>/models/sdxl_prompts``) e são liberados em seguida."""

    def __init__(self, device: str = "auto", offload: bool = False, steps: int = 8, guidance: float = 1.5,
                 control_scale: float = 0.6, offline: bool = True, dtype=None):
        import torch
        from diffusers import (AutoencoderKL, ControlNetModel, EulerDiscreteScheduler,
                               StableDiffusionXLControlNetInpaintPipeline, UNet2DConditionModel)

        from ..analysis import weights as W

        t0 = time.perf_counter()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype or (torch.float16 if device.startswith("cuda") else torch.bfloat16)
        self.steps, self.guidance, self.control_scale = steps, guidance, control_scale
        self.offline = offline
        self.dirs = W.sdxl_dirs(offline=offline)
        gen = self.dirs["gerador"]
        unet = UNet2DConditionModel.from_pretrained(gen, subfolder="unet", variant="fp16", torch_dtype=self.dtype)
        ctl = ControlNetModel.from_pretrained(self.dirs["controlnet"], variant="fp16", torch_dtype=self.dtype)
        vae = AutoencoderKL.from_pretrained(self.dirs["vae"], torch_dtype=self.dtype)
        sch = EulerDiscreteScheduler.from_pretrained(gen, subfolder="scheduler", timestep_spacing="trailing")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline(
            vae=vae, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, unet=unet,
            controlnet=ctl, scheduler=sch, add_watermarker=False)
        self.pipe.set_progress_bar_config(disable=True)
        self.offload = bool(offload and device.startswith("cuda"))
        if device.startswith("cuda"):
            # decodificação do VAE em blocos: o pico da decodificação a ~1024 px não soma ao do UNet na GPU de 8 GB
            try:
                self.pipe.enable_vae_tiling()
            except Exception:  # noqa: BLE001 - versão do diffusers sem o recurso: segue sem
                pass
        if self.offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)
        self._embeds: dict = {}
        self.load_s = time.perf_counter() - t0
        self.calls: list[dict] = []

    # ---- embeddings dos prompts (cache em disco: o cliente não precisa dos text encoders para prompts conhecidos)
    def _cache_path(self, prompt: Prompt):
        from ..analysis.weights import models_dir

        return models_dir() / "sdxl_prompts" / f"{prompt.key()}.pt"

    def embeddings(self, prompt: Prompt):
        import torch

        if prompt.key() in self._embeds:
            return self._embeds[prompt.key()]
        p = self._cache_path(prompt)
        emb = None
        if p.exists():
            try:
                emb = torch.load(p, map_location="cpu", weights_only=True)
            except Exception:
                emb = None
        if emb is None:
            emb = self._encode(prompt)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(emb, p)
            except OSError:
                pass
        self._embeds[prompt.key()] = emb
        return emb

    def _encode(self, prompt: Prompt) -> dict:
        import gc

        import torch
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        from ..analysis import weights as W

        tdir = W.sdxl_text_dir(offline=self.offline)
        te1 = CLIPTextModel.from_pretrained(tdir, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32)
        te2 = CLIPTextModelWithProjection.from_pretrained(tdir, subfolder="text_encoder_2", variant="fp16",
                                                          torch_dtype=torch.float32)
        tk1 = CLIPTokenizer.from_pretrained(tdir, subfolder="tokenizer")
        tk2 = CLIPTokenizer.from_pretrained(tdir, subfolder="tokenizer_2")
        p = self.pipe
        p.text_encoder, p.text_encoder_2, p.tokenizer, p.tokenizer_2 = te1, te2, tk1, tk2
        try:
            with torch.inference_mode():
                pe, ne, pp, npool = p.encode_prompt(prompt.text, device="cpu", num_images_per_prompt=1,
                                                    do_classifier_free_guidance=True,
                                                    negative_prompt=prompt.negative)
        finally:
            p.text_encoder = p.text_encoder_2 = p.tokenizer = p.tokenizer_2 = None
            del te1, te2
            gc.collect()
        return {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in
                {"prompt": pe, "negative": ne, "pooled": pp, "negative_pooled": npool}.items()}

    def __call__(self, image: np.ndarray, mask: np.ndarray, control: np.ndarray, prompt: Prompt, seed: int,
                 strength: float) -> np.ndarray:
        import torch
        from PIL import Image

        e = self.embeddings(prompt)
        dev = "cuda" if self.offload else self.device
        dt = self.dtype
        h, w = mask.shape
        steps = max(self.steps, int(np.ceil(1.0 / max(strength, 1e-3))))
        ctrl = np.repeat((np.clip(control, 0, 1) * 255).round().astype(np.uint8)[..., None], 3, axis=2)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.pipe(
                prompt_embeds=e["prompt"].to(dev, dt), negative_prompt_embeds=e["negative"].to(dev, dt),
                pooled_prompt_embeds=e["pooled"].to(dev, dt),
                negative_pooled_prompt_embeds=e["negative_pooled"].to(dev, dt),
                image=Image.fromarray(image), mask_image=Image.fromarray(mask.astype(np.uint8) * 255),
                control_image=Image.fromarray(ctrl), height=h, width=w, strength=float(strength),
                num_inference_steps=steps, guidance_scale=self.guidance,
                controlnet_conditioning_scale=self.control_scale,
                generator=torch.Generator(device="cpu").manual_seed(int(seed)), output_type="np").images[0]
        dt_s = time.perf_counter() - t0
        n_eff = max(1, int(steps * strength))
        self.calls.append({"tamanho": [w, h], "passos_efetivos": n_eff, "s": round(dt_s, 2),
                           "s_por_passo": round(dt_s / n_eff, 3)})
        return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)

    def release(self) -> None:
        import gc

        del self.pipe
        gc.collect()
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


def load_sdxl(device: str = "auto", offload: bool = False, cfg: RefineConfig | None = None,
              offline: bool = True) -> SdxlInpainter:
    """Carrega o refinador (``WeightsError`` se os pesos faltam ou não conferem o SHA-256)."""
    cfg = cfg or RefineConfig.for_device(device)
    return SdxlInpainter(device, offload, cfg.steps, cfg.guidance, cfg.control_scale, offline=offline)


__all__ = ["RefineConfig", "Prompt", "describe_surfaces", "build_prompt", "empty_room_depth", "protected_pixels",
           "crop_box", "work_size", "prepare_init", "match_color", "feather_alpha", "composite", "refine_plate",
           "SdxlInpainter", "load_sdxl", "ENV_SDXL"]
