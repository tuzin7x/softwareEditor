"""Esvaziar ambiente: a foto mobiliada vira o MESMO cômodo vazio (função "esvaziar" e sala vazia "ia" do modo B).

Fluxo de :func:`empty_room` (a função de design) e de :class:`PlateFiller` (o modo B):

1. **Máscara do que sai** (:func:`empty_mask`): móveis soltos e decoração (grupo ``furniture`` da segmentação, o
   tapete e as peças soltas que a segmentação põe junto dos planejados: cômoda, aparador, estante), mais o que não
   foi classificado e encosta neles. Ficam: armários e bancadas da cozinha, louças, portas, janelas, cortinas
   (opcional), luminárias de teto e arandelas, radiadores, escadas. A máscara cresce um pouco (borda e sombra) e
   bem mais para baixo nos móveis apoiados no piso: a sombra de contato sai junto. No modo B a máscara é a da
   remoção (peças da foto, ``removal/plate.py``) e só o preenchimento vem daqui.
2. **Profundidade estrutural** (``structure.py``): a caixa da sala sem os móveis guia o ControlNet; paredes e piso
   continuam retos por trás de onde estavam os móveis.
3. **Ponto de partida** dentro da máscara: a sala refeita pela geometria (``removal/geofill.py``, só o material
   visível) e o gerador apenas refina (força ~0,6), ou um preenchimento suave do entorno com força ~0,95. Os
   candidatos alternam as duas partidas e a escolha automática fica com a melhor (ver "Escolha").
4. **Geração** só dentro da máscara (o resto da foto vem dos latentes originais a cada passo), com prompt de cômodo
   vazio e negativos fortes (móvel, tapete, cama, sofá, planta, luminária, decoração, pessoas, texto, janela/porta
   nova).
5. **Acabamento**: mistura de Poisson na borda (cor e luz casadas com a foto), rampa curta por dentro; janelas e
   portas da foto voltam por cima (``archlock``). Fora da máscara a imagem é a foto, byte a byte.
6. **Escolha** (:func:`empty_score`): segmentação de cada candidato; perde pontos móvel que sobrou na máscara,
   porta/janela nova e degrau de luz na borda.

Tudo é injetável (gerador, profundidade, segmentação, preenchimento pela geometria, segmentador da escolha) para
testes sem modelo.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import archlock as AL
from . import presets as P
from . import structure as S
from ..analysis.segmentation import GROUP_IDS, UNKNOWN

# ---------------------------------------------------------------------------------------------- prompt
EMPTY_POSITIVE = ("empty room, bare floor, plain walls, same flooring and wall color, unfurnished interior, "
                  "nothing on the floor, clean empty space, photorealistic, natural light, sharp focus, "
                  "realistic materials, real estate photograph")
EMPTY_NEGATIVE = ("furniture, rug, carpet, bed, sofa, couch, table, chair, armchair, plant, lamp, decor, "
                  "decoration, cushion, pillow, blanket, painting, picture frame, shelf, cabinet, box, clutter, "
                  "people, person, text, watermark, new window, new door, opening, doorway, fireplace, "
                  "ghost, shadow of furniture, smudge, blur, fog, haze, blob, stain, distorted walls, bent lines, "
                  "cgi, 3d render, cartoon, painting style, lowres, oversaturated")

# ---------------------------------------------------------------------------------------------- máscara
# rótulos ADE20K que saem além do grupo ``furniture`` (o tapete está no grupo do piso; cômoda/aparador/estante
# são móveis soltos que a segmentação agrupa com os planejados)
EXTRA_REMOVE = ("rug", "chest of drawers", "buffet", "bookcase", "case", "shelf", "plaything", "bulletin board",
                "crt screen", "bottle", "glass", "tray", "plate", "food", "sculpture", "cradle", "ball",
                "fan")
# luminária de teto e arandela ficam (são da casa); "fan" acima só sai se estiver no chão (ver ``_ceiling_band``)
KEEP_GROUPS = ("window", "door", "structure", "fixture", "ceiling_fixture")
GROW = 0.01                  # crescimento isotrópico (× diagonal): borda, moldura e halo do móvel
CLOSE = 0.012                # fechamento da máscara (× diagonal): sem "dedos" entre pés de móveis
SHADOW_DOWN = 0.03           # sombra de contato: a máscara desce até 3% da diagonal sobre o piso
UNKNOWN_MAX = 0.02           # "não classificado" encostado num móvel e menor que 2% da foto sai junto
HOLE_FILL = 0.004            # buraco fechado da máscara até 0,4% da foto entra (vão entre pés de cadeira)
MIN_EMPTY = 0.004            # abaixo disto não há o que esvaziar
HANGING_BOTTOM = 0.45        # luminária que termina acima disto (fração da altura) está pendurada: fica

# ---------------------------------------------------------------------------------------------- geração
# Teste na CPU (sala com sofás e tapete, 768 px): com 0,95 a partida suave deixava a "mancha" do entorno virar um
# bloco bege no lugar do sofá; com ruído puro (1,0) e a profundidade estrutural a 0,6 sai um piso de verdade. A
# partida pela geometria a 0,62 mantinha os defeitos dela; 0,75 refaz textura e luz e mantém o material visível.
# Profundidade forte (0,9 até o fim) redesenhava a silhueta do sofá nas bordas da máscara.
STRENGTH_GEO = 0.75          # partida pela geometria: o gerador refina (textura, luz, sombras suaves)
STRENGTH_SOFT = 1.0          # partida suave: o gerador refaz tudo dentro da máscara (a profundidade guia)
# modo B: a partida é a sala pela geometria já conferida; com 0,75 o gerador inventou um "bloco" branco no lugar do
# sofá (sala com sofás brancos, CPU). 0,55 refaz textura, luz e sombras suaves e mantém a geometria
STRENGTH_PLATE = 0.55
CONTROL_SCALE = 0.6
CONTROL_END = 0.8
GUIDANCE = 2.2
FEATHER = 0.008              # rampa por dentro da máscara (× diagonal): borda recortada não aparece

# ---------------------------------------------------------------------------------------------- escolha
RESIDUAL_WEIGHT = 250.0      # nota perdida por fração da máscara com móvel reencontrado
OPENING_WEIGHT = 400.0       # ... por fração da imagem com porta/janela nova
SEAM_WEIGHT = 2.0            # ... por nível de luz (0–255) de degrau na borda
FLOOR_LIKE = 0.3             # "móvel" com 30% da máscara ou mais, encostado na base da foto, é piso mal rotulado


def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _resize_nn(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    a = np.asarray(a)
    if (a.shape[1], a.shape[0]) == tuple(size):
        return a
    return cv2.resize(a.astype(np.uint8), tuple(size), interpolation=cv2.INTER_NEAREST)


def _fill_holes(m: np.ndarray, max_area: float) -> np.ndarray:
    inv = (~m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = m.copy()
    H, W = m.shape
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a <= max_area and x > 0 and y > 0 and x + w < W and y + h < H:
            out[lab == k] = True
    return out


def empty_mask(groups: np.ndarray, labels: np.ndarray | None = None, id2label: dict | None = None,
               size: tuple[int, int] | None = None, keep_curtains: bool = True, grow: float = GROW,
               shadow_down: float = SHADOW_DOWN, pieces: np.ndarray | None = None) -> np.ndarray:
    """Máscara (bool, ``size`` = (largura, altura)) dos móveis soltos e da decoração a tirar da foto.

    ``groups``: grupos de ``analysis/segmentation.GROUPS``; ``labels``/``id2label`` (rótulos ADE20K) acrescentam o
    tapete e os móveis soltos agrupados com os planejados. ``keep_curtains=False`` tira também cortinas e
    persianas. Janelas, portas, planejados, luminárias de teto e a estrutura nunca entram. ``pieces``: máscara das
    peças soltas do modo B (instâncias do panóptico: o tapete inteiro, não só a parte que a segmentação semântica
    chama de tapete), somada antes do crescimento."""
    g = np.asarray(groups)
    if size is None:
        size = (g.shape[1], g.shape[0])
    g = _resize_nn(g, size)
    W, H = size
    diag = float(np.hypot(W, H))
    m = g == GROUP_IDS["furniture"]
    if pieces is not None:
        m |= _resize_nn(np.asarray(pieces, np.uint8), size).astype(bool)
    if labels is not None and id2label and m.any():
        # luminária pendente (rótulo "lamp" no alto, sem tocar o piso) é da casa: fica
        lab0 = _resize_nn(labels, size)
        lamp_ids = [int(k) for k, v in id2label.items() if str(v).split(",")[0].strip().lower() == "lamp"]
        lamps = np.isin(lab0, lamp_ids) & m
        if lamps.any():
            n, lab_l, st, _ = cv2.connectedComponentsWithStats(lamps.astype(np.uint8), connectivity=8)
            for k in range(1, n):
                if st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] < HANGING_BOTTOM * H:
                    m[lab_l == k] = False
    if not keep_curtains:
        m |= g == GROUP_IDS["window_covering"]
    if labels is not None and id2label:
        lab = _resize_nn(labels, size)
        ids = [int(k) for k, v in id2label.items() if str(v).split(",")[0].strip().lower() in EXTRA_REMOVE]
        extra = np.isin(lab, ids)
        # ventilador de teto e o que estiver no alto da parede (prateleira presa) ficam: só o terço de baixo
        extra[: int(H * 0.33)] &= ~np.isin(lab[: int(H * 0.33)], [k for k in ids if id2label[k] in ("fan", "shelf")])
        m |= extra
    # "não classificado" pequeno encostado num móvel: objeto de decoração que a segmentação não conhece
    unk = g == UNKNOWN
    if unk.any() and m.any():
        near = cv2.dilate(m.astype(np.uint8), _disk(int(0.006 * diag))).astype(bool)
        n, lab_u, st, _ = cv2.connectedComponentsWithStats(unk.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            comp = lab_u == k
            if st[k, cv2.CC_STAT_AREA] <= UNKNOWN_MAX * H * W and (comp & near).any():
                m |= comp
    if not m.any():
        return m
    keep = np.isin(g, [GROUP_IDS[k] for k in KEEP_GROUPS])
    if keep_curtains:
        keep |= g == GROUP_IDS["window_covering"]
    r = int(round(grow * diag))
    if r > 0:
        m = cv2.dilate(m.astype(np.uint8), _disk(r)).astype(bool)
    # sombra de contato: abaixo do móvel, sobre o piso (não sobe em parede nem em planejado)
    d = int(round(shadow_down * diag))
    if d > 0:
        k = np.zeros((2 * d + 1, 3), np.uint8)
        k[d:, 1] = 1                              # dilatação só para baixo
        down = cv2.dilate(m.astype(np.uint8), k, anchor=(1, 0)).astype(bool)
        floorish = (g == GROUP_IDS["floor"]) | (g == UNKNOWN)
        m |= down & floorish
    c = int(round(CLOSE * diag))
    if c > 0:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, _disk(c)).astype(bool)
    m = _fill_holes(m, HOLE_FILL * H * W)
    # o que é da casa fica (núcleo erodido: a borda do móvel sobre a janela pode ser refeita)
    core_keep = cv2.erode(keep.astype(np.uint8), _disk(max(1, int(0.002 * diag)))).astype(bool)
    return m & ~core_keep


def _color_words(lab: np.ndarray) -> str:
    """Palavras simples para uma cor média Lab (OpenCV, L 0–255): "light beige", "dark brown", "white"..."""
    L, a, b = float(lab[0]) * 100 / 255, float(lab[1]) - 128, float(lab[2]) - 128
    chroma = float(np.hypot(a, b))
    tone = "light" if L > 68 else "medium" if L > 42 else "dark"
    if chroma < 7:
        return "white" if L > 85 else f"{tone} gray"
    if b > 0 and a > 6 and L < 60:
        return f"{tone} brown"
    if b > 0:
        return f"{tone} beige" if L > 55 else f"{tone} brown"
    return f"{tone} gray blue" if b < -6 else f"{tone} gray"


def surface_hint(rgb: np.ndarray, groups: np.ndarray, hole: np.ndarray, labels: np.ndarray | None = None,
                 id2label: dict | None = None) -> str:
    """Pista de material para o prompt: a cor do piso e da parede VISÍVEIS (fora da máscara, sem tapete), para o
    gerador continuar o mesmo piso e a mesma parede ("light beige floor, white walls")."""
    H, W = hole.shape
    g = _resize_nn(groups, (W, H))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    out = []
    floor = (g == GROUP_IDS["floor"]) & ~hole
    if labels is not None and id2label:
        rug = [int(k) for k, v in id2label.items() if str(v).split(",")[0].strip().lower() == "rug"]
        floor &= ~np.isin(_resize_nn(labels, (W, H)), rug)
    for name, sel in (("floor", floor), ("walls", (g == GROUP_IDS["wall"]) & ~hole)):
        if sel.sum() >= 0.003 * H * W:
            out.append(f"{_color_words(np.median(lab[sel], axis=0))} {name}")
    return ", ".join(out)


def openings_mask(groups: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return AL.lock_mask(groups, ("window", "door"), size)


def empty_prompt(extra: str = "") -> P.DesignPrompt:
    pos = EMPTY_POSITIVE + (", " + extra.strip() if extra and extra.strip() else "")
    return P.DesignPrompt(pos, EMPTY_NEGATIVE)


def soft_prefill(img: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Ponto de partida suave: o entorno puxado para dentro (Telea) e desfocado. Fora da máscara = ``img``."""
    if not hole.any():
        return img.copy()
    dw = float(np.hypot(*hole.shape))
    pre = cv2.inpaint(np.ascontiguousarray(img), hole.astype(np.uint8) * 255, max(3, int(0.01 * dw)),
                      cv2.INPAINT_TELEA)
    pre = cv2.GaussianBlur(pre, (0, 0), 0.008 * dw)
    out = img.copy()
    out[hole] = pre[hole]
    return out


def membrane_blend(gen: np.ndarray, photo: np.ndarray, hole: np.ndarray, ring_frac: float = 0.006,
                   work_side: int = 1024) -> np.ndarray:
    """Casamento de cor/luz na borda (Poisson por membrana): a diferença foto − gerado medida num anel logo fora
    da máscara é interpolada (Laplace) para dentro e somada ao gerado. Sem degrau na borda, a textura gerada fica.
    A conta é feita em resolução reduzida (a correção é de baixa frequência)."""
    H, W = hole.shape
    diag = float(np.hypot(H, W))
    ring = cv2.dilate(hole.astype(np.uint8), _disk(max(2, int(ring_frac * diag)))).astype(bool) & ~hole
    if not ring.any():
        return gen.copy()
    s = min(1.0, work_side / max(H, W))
    sz = (max(8, int(round(W * s))), max(8, int(round(H * s))))
    d = photo.astype(np.float32) - gen.astype(np.float32)
    # o anel é amostrado (média por área) e o resto fora da máscara não conta (só a borda define a correção)
    w_full = ring.astype(np.float32)
    d_s = cv2.resize(d * w_full[..., None], sz, interpolation=cv2.INTER_AREA)
    w_s = cv2.resize(w_full, sz, interpolation=cv2.INTER_AREA)
    known = w_s > 0.05
    corr = np.zeros((sz[1], sz[0], 3), np.float32)
    for c in range(3):
        v = np.where(known, d_s[..., c] / np.maximum(w_s, 1e-6), 0.0).astype(np.float32)
        corr[..., c] = S._harmonic_fill(v, known, iters=250)
    corr = cv2.resize(corr, (W, H), interpolation=cv2.INTER_LINEAR)
    out = gen.astype(np.float32)
    out[hole] += corr[hole]
    res = photo.copy()
    res[hole] = np.clip(out[hole] + 0.5, 0, 255).astype(np.uint8)
    return res


def finish(gen: np.ndarray, photo: np.ndarray, hole: np.ndarray, lock: np.ndarray | None = None,
           feather: float = FEATHER) -> np.ndarray:
    """Acabamento do preenchimento: Poisson na borda (cor/luz da foto), rampa curta por dentro, aberturas da foto
    por cima. Fora de ``hole`` o resultado é ``photo`` byte a byte."""
    hole = np.asarray(hole, bool)
    if not hole.any():
        return photo.copy()
    blended = membrane_blend(gen, photo, hole)
    a = AL.feather_alpha(hole, max(1.5, feather * float(np.hypot(*hole.shape))), inside=1.0)
    out = AL.composite(photo, blended, a)
    if lock is not None and (lock & hole).any():
        lk = lock & hole
        out = AL.composite(out, photo, AL.feather_alpha(lk, 2.0, inside=0.0) * hole)
    out[~hole] = photo[~hole]
    return out


def generate_empty(eng, img: np.ndarray, hole: np.ndarray, init: np.ndarray, control: np.ndarray | None,
                   seed: int, strength: float, steps: int, lock: np.ndarray | None = None,
                   prompt: P.DesignPrompt | None = None, guidance: float = GUIDANCE,
                   control_scale: float = CONTROL_SCALE, control_end: float = CONTROL_END,
                   callback=None) -> np.ndarray:
    """Uma geração "cômodo vazio" restrita a ``hole`` (tudo na mesma resolução de ``img``). ``init``: ponto de
    partida (``img`` fora da máscara, preenchimento dentro). Devolve a imagem acabada (:func:`finish`)."""
    gen_mask = hole & ~(lock if lock is not None else np.zeros_like(hole))
    if not gen_mask.any():
        return img.copy()
    out = eng.generate(init, gen_mask, control, prompt or empty_prompt(), int(seed), float(strength), int(steps),
                       float(guidance), float(control_scale if control is not None else 0.0), float(control_end),
                       callback)
    if out.shape[:2] != img.shape[:2]:
        out = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return finish(out, img, hole, lock)


def empty_score(gen_groups: np.ndarray, hole: np.ndarray, src_openings: np.ndarray, img: np.ndarray,
                src_groups: np.ndarray | None = None, gen_labels: np.ndarray | None = None,
                id2label: dict | None = None) -> dict:
    """Nota de um candidato (maior = melhor): móvel reencontrado na máscara, porta/janela nova, degrau na borda."""
    H, W = hole.shape
    g = _resize_nn(gen_groups, (W, H))
    resid_m = (g == GROUP_IDS["furniture"]) & hole
    if gen_labels is not None and id2label:
        lab = _resize_nn(gen_labels, (W, H))
        rug = [int(k) for k, v in id2label.items() if str(v).split(",")[0].strip().lower() in EXTRA_REMOVE]
        resid_m |= np.isin(lab, rug) & hole
    # restos pequenos e espalhados (ruído da segmentação) não contam: só componentes acima de 0,1% da imagem
    n, lab_r, st, _ = cv2.connectedComponentsWithStats(resid_m.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(resid_m)
    n_hole = max(1, int(hole.sum()))
    floor_like = 0
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA]
        if a < 0.001 * H * W:
            continue
        # piso claro/brilhante que a segmentação chama de "mesa": um "móvel" do tamanho do piso, encostado na
        # base da foto, é o piso (teste na CPU: o piso limpo saiu 56% "table" e a escolha ficou com a pior versão)
        if a >= FLOOR_LIKE * n_hole and st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] >= H - 2:
            floor_like += int(a)
            continue
        keep |= lab_r == k
    resid = float(keep.sum()) / max(1, int(hole.sum()))
    src = None
    if src_groups is not None:
        # atrás dos móveis a foto "era parede": porta/janela que nasce ali é inventada
        src = _resize_nn(src_groups, (W, H)).copy()
        src[hole] = GROUP_IDS["wall"]
    new = AL.new_openings(g, src_openings, (W, H), source_groups=src) & hole
    new_frac = float(new.mean())
    seam = AL.seam_step(img, hole)
    nota = 100.0 - RESIDUAL_WEIGHT * resid - OPENING_WEIGHT * new_frac - SEAM_WEIGHT * seam
    return {"nota": round(nota, 2), "movel_restante": round(resid, 4), "aberturas_novas": round(new_frac, 4),
            "degrau_borda": round(seam, 2), "piso_tomado_por_movel": round(floor_like / n_hole, 4)}


# ============================================================================================ função de design
@dataclass
class EmptyAnalysis:
    """O que a função precisa da foto (na resolução de trabalho): profundidade, segmentação e, se houver, o
    preenchimento pela geometria da área a esvaziar."""

    depth: np.ndarray | None
    groups: np.ndarray
    labels: np.ndarray | None = None
    id2label: dict | None = None
    geofill: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    info: dict = field(default_factory=dict)
    pieces: np.ndarray | None = None          # peças soltas do modo B (panóptico), na resolução de trabalho


def analyse(rgb: np.ndarray, device: str, with_geometry: bool = True) -> EmptyAnalysis:
    """MoGe-2 + OneFormer na foto de trabalho e, com ``with_geometry``, o modelo da sala para o preenchimento pela
    geometria (sem o modelo, ou se ele falhar, só a partida suave)."""
    from ..analysis.geometry import estimate_geometry
    from ..analysis.segmentation import SegmentationResult, labels_to_groups, segment
    from ..config import GEOMETRY_MODEL_ID, GEOMETRY_MODEL_ID_SMALL, SEGMENTATION_MODEL_ID

    gpu = device.startswith("cuda")
    H, W = rgb.shape[:2]
    info: dict = {}
    t0 = time.perf_counter()
    geo = estimate_geometry(rgb, GEOMETRY_MODEL_ID if gpu else GEOMETRY_MODEL_ID_SMALL, device=device, use_fp16=gpu)
    info["profundidade_s"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    s = min(1.0, 768 / max(H, W))
    small = cv2.resize(rgb, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    pan = None
    try:
        from ..removal import instances as ri

        seg, pan = ri.run_oneformer(small, device=device, use_fp16=gpu)     # semântico + panóptico, uma carga
    except Exception as e:  # noqa: BLE001 - sem panóptico: só a segmentação semântica
        info["panoptico_erro"] = f"{type(e).__name__}: {str(e)[:160]}"
        seg = segment(small, SEGMENTATION_MODEL_ID, device=device, use_fp16=gpu)
    info["segmentacao_s"] = round(time.perf_counter() - t0, 2)
    labels = _resize_nn(seg.labels, (W, H))
    groups = labels_to_groups(labels, seg.id2label)
    depth = geo.depth.copy()
    depth[~geo.mask] = 0
    fill = None
    box = None
    if with_geometry:
        try:
            from ..analysis.room import build_room_model
            from ..removal import plate as rp
            from ..removal.geofill import fill_geometry

            seg_w = SegmentationResult(labels, groups, seg.id2label, seg.model_id)
            room, _ = build_room_model(geo, seg_w, rgb)
            if room is not None and getattr(room, "walls", None) is not None:
                box = rp.RoomBox(room, W, H)

                def fill(img, hole, _box=box, _g=groups, _d=geo.depth, _seg=seg_w):
                    return fill_geometry(img, hole, _g, _box, _d, np.random.default_rng(0), rp.telea_inpaint,
                                         rug=rp.rug_mask(_seg, hole.shape))[0]
        except Exception as e:  # noqa: BLE001 - modelo da sala degenerado: só a partida suave
            info["geometria_erro"] = f"{type(e).__name__}: {str(e)[:160]}"
            fill = None
    info["geometria"] = fill is not None
    pieces = None
    if pan is not None:
        try:
            from ..removal import instances as ri
            from ..removal.instances import PanopticResult

            ids = _resize_nn(pan.ids, (W, H))
            rows = np.array(pan.segments, np.float64).reshape(-1, 4).copy()
            for r in rows:
                r[3] = float((ids == int(r[0])).sum())
            seg_w2 = SegmentationResult(labels, groups, seg.id2label, seg.model_id)
            inst = ri.detect_instances(rgb, seg_w2, PanopticResult(ids, rows), sam=None, device=device)
            # só os soltos (e o que está dentro de um planejado fica com ele): armários e bancadas são da casa
            sel = [i for i in inst.removal(keep_builtins=True)]
            if box is not None and sel:
                # peça vista por uma porta (a cama do quarto ao lado) fica: não há parede desta sala atrás dela
                from ..removal import plate as rp_

                beyond = set(rp_.pieces_beyond_openings(sel, box, geo.depth, groups))
                if beyond:
                    info["pecas_alem_da_abertura"] = sorted(i.label for i in sel if i.id in beyond)
                    sel = [i for i in sel if i.id not in beyond]
            if sel:
                pieces = np.logical_or.reduce([i.mask for i in sel])
            info["pecas"] = sorted({i.label for i in sel})
        except Exception as e:  # noqa: BLE001
            info["pecas_erro"] = f"{type(e).__name__}: {str(e)[:160]}"
    return EmptyAnalysis(depth, groups, labels, seg.id2label, fill, info, pieces)


def empty_room(photo, n: int = 2, seed: int | None = None, out_dir=None, quality: str = "final",
               engine=None, analysis: EmptyAnalysis | None = None, progress=None, device: str = "auto",
               long_side: int | None = None, candidates: int | None = None, rank_fn=None,
               prefill: str = "auto", keep_curtains: bool = True, prompt: str = ""):
    """Função "Esvaziar ambiente": ``n`` versões do mesmo cômodo sem os móveis soltos e a decoração.

    ``prefill``: "auto" (candidatos alternam a partida pela geometria e a suave; sem geometria, só a suave),
    "geometria" ou "suave". ``analysis`` (:class:`EmptyAnalysis`) evita rodar MoGe/OneFormer (testes e reuso);
    ``rank_fn(rgb) -> (grupos, rótulos | None)`` troca o OneFormer da escolha (testes). ``candidates``: quantas
    gerar antes de escolher as ``n`` melhores (padrão: n + 1 na GPU, n na CPU)."""
    from . import api as A
    from . import memplan as MP
    from .engine import free_cuda_cache, get_engine, pick_device

    t_all = time.perf_counter()
    prog = A._Prog(progress)
    out = A._out_dir(out_dir)
    timings: dict = {}
    prog("preparo")
    rgb = A._load_rgb(photo)
    eng = engine or get_engine(device)
    dev = getattr(eng, "device", pick_device(device))
    gpu = dev.startswith("cuda")
    if engine is None:
        A._announce_downloads(prog, dev)
    qp = P.quality_profile(quality, dev)
    mplan = getattr(eng, "plan", None) or MP.current_plan(dev)
    W, H = P.work_size(rgb.shape[1], rgb.shape[0], long_side or qp.long_side)
    base = A._resize(rgb, (W, H))

    t0 = time.perf_counter()
    prog("analise", 0.0, "Medindo a profundidade e encontrando os móveis")
    an = analysis or analyse(base, dev, with_geometry=prefill in ("auto", "geometria"))
    gr = _resize_nn(an.groups, (W, H))
    lock = openings_mask(gr, (W, H))
    hole = empty_mask(gr, an.labels, an.id2label, (W, H), keep_curtains=keep_curtains, pieces=an.pieces)
    # só a borda da janela/porta que a máscara cresceu por cima volta da foto: "janela" no MIOLO de uma peça é
    # quadro ou cúpula que a segmentação confundiu (devolver traria o fantasma do quadro)
    core = cv2.erode(hole.astype(np.uint8), _disk(int(GROW * np.hypot(W, H)) + 2)).astype(bool)
    lock = lock & ~core
    area = float(hole.mean())
    if area < MIN_EMPTY:
        raise ValueError("Nenhum móvel encontrado na foto para tirar: o ambiente já parece vazio.")
    control = None
    if an.depth is not None:
        dz = cv2.resize(np.asarray(an.depth, np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        sd = S.structural_depth(dz, hole, gr)
        control = A.depth_control(sd, (W, H), ref=dz)
    prog("analise", 0.6, "Preparando o cômodo vazio")
    soft = soft_prefill(base, hole)
    geo_init = None
    if prefill in ("auto", "geometria") and an.geofill is not None:
        try:
            geo_init = an.geofill(base, hole)
            geo_init[~hole] = base[~hole]
        except Exception as e:  # noqa: BLE001
            an.info["geometria_erro"] = f"{type(e).__name__}: {str(e)[:160]}"
            geo_init = None
    timings["analise_s"] = round(time.perf_counter() - t0, 2)

    n = max(1, int(n))
    m = max(n, int(candidates if candidates is not None else (n + 1 if gpu else n)))
    seeds = A._seeds(seed, m)
    starts = []
    for i in range(m):
        use_geo = geo_init is not None and (prefill == "geometria" or (prefill == "auto" and i % 2 == 0))
        starts.append("geometria" if use_geo else "suave")
    steps = qp.steps
    hint = surface_hint(base, gr, hole, an.labels, an.id2label)
    pr = empty_prompt(", ".join(x for x in (hint, prompt.strip()) if x))
    if hasattr(eng, "embeddings"):
        prog("geracao", 0.0, "Preparando o texto do cômodo vazio")
        eng.embeddings([pr])
    free_cuda_cache(dev)
    cand_dir = out / "candidatos" if m > n else out
    images, paths = [], []
    t_gen = time.perf_counter()
    for i, (sd, st) in enumerate(zip(seeds, starts)):
        def cb(k, tot, i=i):
            prog("geracao", min(1.0, (i + k / max(tot, 1)) / m) * 0.85, f"Esvaziando: versão {i + 1} de {m}")

        cb(0, 1)
        init, strength = (geo_init, STRENGTH_GEO) if st == "geometria" else (soft, STRENGTH_SOFT)
        img = generate_empty(eng, base, hole, init, control, sd, strength, steps, lock=lock, prompt=pr, callback=cb)
        images.append(img)
        p = A._save(img, cand_dir / f"esvaziar_{i + 1}.jpg")
        paths.append(p)
        prog.emit({"event": "preview", "path": p, "index": i})
    timings["geracao_s"] = round(time.perf_counter() - t_gen, 2)
    timings["por_imagem_s"] = round(timings["geracao_s"] / m, 2)

    # ---- escolha: o mesmo segmentador confere todos (móvel que sobrou, porta/janela nova, borda)
    prog("geracao", 0.88, "Conferindo se sobrou algum móvel")
    t0 = time.perf_counter()
    seg = None
    fn = rank_fn
    if fn is None:
        from .quality import Segmenter

        seg = Segmenter(dev)

        def fn(im):
            lab = seg.labels(im)
            from ..analysis.segmentation import labels_to_groups

            return labels_to_groups(lab, seg.id2label), lab

    try:
        notas = []
        for k, im in enumerate(images):
            res = fn(im)
            gg, ll = res if isinstance(res, tuple) else (res, None)
            ids = getattr(seg, "id2label", None) if seg is not None else an.id2label
            sc = empty_score(gg, hole, lock, im, src_groups=gr, gen_labels=ll, id2label=ids if ll is not None else None)
            sc["partida"] = starts[k]
            notas.append(sc)
    finally:
        if seg is not None:
            seg.close()
    order = sorted(range(m), key=lambda k: -notas[k]["nota"])[:n]
    timings["escolha_s"] = round(time.perf_counter() - t0, 2)
    outputs = [A._save(images[k], out / f"esvaziar_{j + 1}.jpg") if m > n else paths[k] for j, k in enumerate(order)]
    prog("final")
    before = A._save(base, out / "antes.jpg")
    mask_path = A._save((hole.astype(np.uint8) * 255), out / "mascara_esvaziar.png")
    timings["total_s"] = round(time.perf_counter() - t_all, 2)
    rep = {"funcao": "esvaziar", "tamanho": [W, H], "qualidade": qp.key, "dispositivo": dev,
           "area_esvaziada_pct": round(100 * area, 2), "partida": prefill, "geometria": geo_init is not None,
           "cortinas_mantidas": keep_curtains, "sementes": [seeds[k] for k in order], "passos": steps,
           "pista_material": hint,
           "prompt": {"positivo": pr.positive, "negativo": pr.negative},
           "forcas": {"geometria": STRENGTH_GEO, "suave": STRENGTH_SOFT},
           "candidatos": {"sementes": seeds, "partidas": starts, "arquivos": paths, "notas": notas,
                          "escolhidos": order},
           "analise": an.info, "saidas": outputs, "antes": before, "mascara": mask_path, "tempos": timings,
           "motor": {"carga_s": round(getattr(eng, "load_s", 0.0), 2), "chamadas": getattr(eng, "calls", [])[-m:]}}
    report = out / "design.json"
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=A._json_default), encoding="utf-8")
    return A.DesignResult(outputs, before, str(report), "esvaziar", None, "sala", [seeds[k] for k in order],
                          (W, H), timings, [images[k] for k in order])


# ============================================================================================ modo B
PLATE_LONG_SIDE_GPU = 1024
PLATE_LONG_SIDE_CPU = 768
PLATE_CONTEXT = 0.25         # margem do recorte em volta da máscara (× o lado dela)
PLATE_MIN_CROP = 0.5         # recorte cobre pelo menos metade do lado maior da foto


def plate_crop(mask: np.ndarray, aspect: float) -> tuple[int, int, int, int]:
    H, W = mask.shape
    from .api import edit_crop_box

    return edit_crop_box(mask, aspect, PLATE_CONTEXT, int(PLATE_MIN_CROP * max(H, W)))


class PlateFiller:
    """Preenchimento "ia" da sala vazia do modo B: o gerador de difusão refina, só dentro da máscara de remoção, a
    sala refeita pela geometria (a partida). Chamado por ``removal.plate.build_plate_gated`` como
    ``filler(foto, placa_geometria, mascara, grupos, profundidade, semente) -> (placa, info)``.

    A geração roda num recorte em volta da máscara (resolução perto da nativa da foto do vídeo) e volta só para
    dentro dela: ``placa[~mascara] == foto[~mascara]`` byte a byte. ``engine``: gerador já carregado ou uma função
    que o carrega (só na 1ª chamada). ``given``: imagem do cômodo vazio já escolhida pelo usuário (resultado do
    "Esvaziar ambiente"): usada no lugar de uma geração nova."""

    def __init__(self, engine, device: str = "cpu", steps: int | None = None, long_side: int | None = None,
                 strength: float = STRENGTH_PLATE, given: np.ndarray | None = None, wrap=None):
        self._engine = engine
        self.device = device
        gpu = str(device).startswith("cuda")
        self.long_side = long_side or (PLATE_LONG_SIDE_GPU if gpu else PLATE_LONG_SIDE_CPU)
        self.steps = steps or (8 if gpu else 6)
        self.strength = strength
        self.given = given
        self.wrap = wrap
        self.calls: list[dict] = []
        self.last_init: np.ndarray | None = None      # partida pela geometria da última chamada (conferência)

    def engine(self):
        if callable(self._engine) and not hasattr(self._engine, "generate"):
            self._engine = self._engine()
        return self._engine

    @property
    def variants(self) -> tuple[str, ...]:
        """Tentativas do portão: "refino" (a sala pela geometria refinada, força :data:`STRENGTH_PLATE`) e
        "livre" (partida suave, ruído puro dentro da máscara, a profundidade estrutural guia): quando a geometria
        sai ruim (cortina esticada, parede em retalhos), refinar não conserta; a livre refaz. Com o cômodo vazio
        escolhido pelo usuário, só ele."""
        return ("escolhida",) if self.given is not None else ("refino", "livre")

    def __call__(self, photo: np.ndarray, init: np.ndarray, mask: np.ndarray, groups: np.ndarray,
                 depth: np.ndarray | None, seed: int = 0, variant: str = "refino") -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        H, W = mask.shape
        mask = np.asarray(mask, bool)
        info: dict = {"area_pct": round(100 * float(mask.mean()), 2)}
        if not mask.any():
            return photo.copy(), info
        self.last_init = init
        if self.given is not None:
            g = self.given
            if g.shape[:2] != (H, W):
                g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LANCZOS4)
            plate = finish(g, photo, mask)
            info.update(origem="escolhida pelo usuário", tempo_s=round(time.perf_counter() - t0, 2))
            self.calls.append(info)
            return plate, info
        from .api import depth_control, edit_gen_size

        x0, y0, x1, y1 = plate_crop(mask, W / H)
        cw, ch = x1 - x0, y1 - y0
        gw, gh = edit_gen_size(cw, ch, self.long_side)
        sl = (slice(y0, y1), slice(x0, x1))
        rs = lambda a, interp=cv2.INTER_AREA: cv2.resize(a, (gw, gh), interpolation=interp)  # noqa: E731
        ph_c, in_c = rs(photo[sl]), rs(init[sl])
        hole_c = rs(mask[sl].astype(np.uint8), cv2.INTER_NEAREST).astype(bool)
        g_c = _resize_nn(_resize_nn(groups, (W, H))[sl], (gw, gh))
        lock_c = openings_mask(g_c, (gw, gh)) & ~cv2.dilate(hole_c.astype(np.uint8), _disk(2)).astype(bool)
        control = None
        if depth is not None:
            dz = cv2.resize(np.asarray(depth, np.float32), (W, H), interpolation=cv2.INTER_NEAREST)[sl]
            dz = cv2.resize(dz, (gw, gh), interpolation=cv2.INTER_NEAREST)
            sd = S.structural_depth(dz, hole_c, g_c)
            control = depth_control(sd, (gw, gh), ref=dz)
        strength = self.strength
        if variant == "livre":
            init_c = soft_prefill(ph_c, hole_c)
            strength = STRENGTH_SOFT
        else:
            init_c = in_c.copy()
            init_c[~hole_c] = ph_c[~hole_c]
        eng = self.engine()
        gen = eng.generate
        if self.wrap is not None:
            gen = self.wrap(gen)

        class _E:                                  # gerador com a troca de VRAM do modo B em volta
            generate = staticmethod(gen)

        hint = surface_hint(ph_c, g_c, hole_c)
        img = generate_empty(_E, ph_c, hole_c, init_c, control, seed, strength, self.steps, lock=None,
                             prompt=empty_prompt(hint))
        up = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        full = photo.copy()
        full[sl] = up
        # alta frequência da partida pela geometria (grão real do piso) sob a baixa frequência gerada: a
        # geração a ~768–1024 px ampliada para a foto do vídeo ficaria mais macia que o entorno
        if variant != "livre":
            full = _detail_transfer(full, init, mask, scale=max(cw / gw, ch / gh))
        # sem trava de aberturas aqui: dentro da máscara do modo B a abertura escondida pela peça já vem
        # protegida da geometria, e devolver os pixels "janela" da foto trazia de volta quadro e luminária que a
        # segmentação confundiu com janela (fantasmas na foto do usuário)
        plate = finish(full, photo, mask)
        info.update(origem="gerada", pista_material=hint, recorte=[x0, y0, x1, y1], tamanho_geracao=[gw, gh], semente=int(seed),
                    variante=variant, forca=strength, passos=self.steps, tempo_s=round(time.perf_counter() - t0, 2))
        self.calls.append(info)
        return plate, info


def _detail_transfer(gen: np.ndarray, init: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    """Devolve à imagem gerada (ampliada ``scale`` vezes) o detalhe fino da partida (grão do material visível),
    só dentro de ``mask`` e só quando a ampliação passa de 1,3×."""
    if scale < 1.3:
        return gen
    sig = 0.6 * scale
    g = gen.astype(np.float32)
    i = init.astype(np.float32)
    hf = i - cv2.GaussianBlur(i, (0, 0), sig)
    out = g - (g - cv2.GaussianBlur(g, (0, 0), sig)) + hf
    res = gen.copy()
    res[mask] = np.clip(out[mask] + 0.5, 0, 255).astype(np.uint8)
    return res


def design_engine_available(device: str = "auto") -> tuple[bool, str]:
    """O motor de difusão do redesenho está instalado (bibliotecas e pesos no disco)? (sim, motivo)."""
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"bibliotecas do gerador ausentes ({type(e).__name__})"
    try:
        from ..analysis import weights as W

        if not W.sdxl_installed():
            return False, "pesos do gerador não instalados"
        W.sdxl_text_dir(offline=True)
    except Exception as e:  # noqa: BLE001
        return False, f"pesos do gerador não instalados ({str(e)[:120]})"
    return True, "gerador de difusão disponível"


__all__ = ["empty_mask", "empty_room", "empty_score", "generate_empty", "finish", "soft_prefill", "empty_prompt",
           "PlateFiller", "EmptyAnalysis", "analyse", "design_engine_available", "EMPTY_POSITIVE",
           "EMPTY_NEGATIVE"]
