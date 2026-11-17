"""Camadas 2D dos móveis reais da foto (modo B, PLANO_V3 §3.3 e PT-V3-5).

Entrada: a foto mobiliada, a sala vazia (``PlateResult`` do ``removal/plate.py``) e as mesmas peças usadas para
fazer a placa. Saída: uma :class:`LayerStack` com

* **placa corrigida**: a placa do LaMa, com o resíduo que não pertence a nenhuma peça corrigido *nela* (vira o
  pixel da foto). Assim o piso e as paredes longe dos móveis não mudam durante o vídeo (anti-morph): a crítica
  mediu 7,1% da imagem "se curando" no protótipo, onde esse resíduo entrava em fade a cada pouso;
* **uma camada por peça**, na ordem de pintura (de trás para frente):
  - RGBA **pré-multiplicado** ``C = foto − (1−A)·fundo_abaixo``, calculado na ordem de pintura e limitado ao
    intervalo válido ``0 <= C <= 255·A`` (a peça em movimento nunca carrega cor impossível);
  - **sombreamento** (sombra/reflexo escuro) = razão ``foto / placa`` suavizada e limitada a ``<= 1``, só na
    sombra que a placa detectou perto da peça (mais escura que a placa). Multiplica o fundo e pode acompanhar
    o deslocamento da peça (é uma mancha suave, sem a textura do piso);
  - **delta de contato**: o que ainda falta para a pilha completa reproduzir a foto, só no alpha da peça (corte
    do intervalo válido), no anel estreito em volta dela (a dilatação do núcleo da máscara de remoção) e na
    região de efeito dela (textura da sombra, reflexo claro no piso polido). Fica parado na posição de repouso
    e entra com o pouso (a coreografia controla a intensidade: sombra de contato crescente).

Garantias (testadas):

1. ``composite(stack, nenhuma)`` é ``array_equal`` à placa corrigida, e a placa corrigida é ``array_equal`` à
   foto fora da máscara de remoção;
2. ``composite(stack, todas em repouso)`` difere da foto em no máximo 1–2 níveis (arredondamento);
3. fora da região dinâmica (alpha ∪ anel ∪ efeito de alguma peça) o primeiro e o último quadro são iguais
   byte a byte: 0% de morph.

Tudo em numpy (CPU). A composição por quadro em GPU (torch) é o PT-V3-7; aqui não há caminho de GPU.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from ..removal import plate as rp

# --- rótulos ADE20K (nome base) com regra própria de ordem de pintura ------------------------------------------
RUG_LABELS = frozenset({"rug"})
WALL_HUNG_LABELS = frozenset({"painting", "poster", "clock", "mirror"})
SMALL_LABELS = frozenset({
    "vase", "flower", "plant", "pot", "pillow", "cushion", "lamp", "book", "blanket", "tray", "bottle", "glass",
    "plate", "food", "towel", "clothes", "box", "basket", "sculpture", "bag", "plaything", "tv", "computer",
    "monitor", "crt screen",
})

ALPHA_MIN = 1.0 / 512.0      # alpha abaixo disto não muda nem meio nível: vira 0 (o resíduo vai para o delta)
CONTACT_MIN_PX = 12          # pixels de borda comum para decidir quem está na frente
DEPTH_FRONT = 0.97           # i na frente de j se z_i < 0,97·z_j na borda comum
SHADE_MIN = 0.2              # razão mínima do sombreamento (o resto do escurecimento vai para o delta)
SHADE_SIGMA = 0.002          # suavização da razão (fração da diagonal)
EFFECT_SLACK_PX = 4
GHOST_LEVELS = 30           # pixel corrigido na placa que difere disto da placa do LaMa = "fantasma" de móvel
GHOST_WARN_FRAC = 0.001     # aviso se os fantasmas passam de 0,1% da imagem
RECON_TOL = 2                # aceite PT-V3-5: pilha completa x foto <= 2 níveis


@dataclass
class Placement:
    """Como uma peça aparece num quadro. ``dx``/``dy`` (px inteiros) deslocam o RGBA e o sombreamento (o
    reflexo acompanha a peça); ``shade`` (0..1) e ``contact`` (0..1) são as intensidades do sombreamento e do
    delta de contato (que fica na posição de repouso); ``alpha_mul`` (recorte do tamanho da caixa, 0..1)
    revela só parte da peça no lugar (tapete desenrolando). ``scale`` (<= 1) encolhe RGBA e sombreamento em torno
    de ``anchor`` (x, y em pixels absolutos da posição de repouso; padrão: centro da base da caixa), antes do
    deslocamento: "escala no ponto de contato" da coreografia (PT-V3-6)."""
    dx: int = 0
    dy: int = 0
    shade: float = 1.0
    contact: float = 1.0
    alpha_mul: np.ndarray | None = None
    scale: float = 1.0
    anchor: tuple[float, float] | None = None


@dataclass
class Layer:
    id: int
    label: str
    kind: str
    box: tuple[int, int, int, int]      # (x0, y0, x1, y1) exclusivo, na posição de repouso
    alpha: np.ndarray                   # (h, w) float32
    color: np.ndarray                   # (h, w, 3) float32, pré-multiplicado (0 <= C <= 255·A)
    shade: np.ndarray | None            # (h, w, 3) float32 em (0, 1], ou None
    delta: np.ndarray | None            # (h, w, 3) float32 (com sinal), ou None
    ring: np.ndarray                    # (h, w) bool: anel de contato
    effect: np.ndarray                  # (h, w) bool: sombra/reflexo
    touches: tuple[str, ...] = ()
    hard: np.ndarray | None = None      # (h, w) bool: máscara dura da peça (oclusão, coreografia)

    @property
    def support(self) -> np.ndarray:
        return self.alpha > 0

    def summary(self, image_area: int) -> dict:
        d = {"id": self.id, "rotulo": self.label, "tipo": self.kind, "caixa": list(self.box),
             "alpha_pct": round(100.0 * float(self.support.sum()) / image_area, 3),
             "anel_pct": round(100.0 * float(self.ring.sum()) / image_area, 3),
             "efeito_pct": round(100.0 * float(self.effect.sum()) / image_area, 3)}
        if self.delta is not None:
            a = np.abs(self.delta).max(axis=2)
            sel = a[self.support | self.ring]
            d["delta_p99"] = round(float(np.percentile(sel, 99)), 1) if sel.size else 0.0
        return d


@dataclass
class LayerStack:
    photo_shape: tuple[int, int]
    plate: np.ndarray                   # (H, W, 3) uint8: placa corrigida (quadro 0 do vídeo)
    layers: list[Layer]                 # na ordem de pintura (de trás para frente)
    front: np.ndarray                   # (n, n) bool: front[i, j] = camada i na frente da j (índices de ``layers``)
    supports: dict = field(default_factory=dict)   # índice da peça apoiada -> índice do apoio
    dynamic: np.ndarray | None = None   # (H, W) bool: alpha ∪ anel ∪ efeito de alguma peça
    corrected: np.ndarray | None = None  # (H, W) bool: pixels da placa trocados pelos da foto
    stats: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def index(self, piece_id: int) -> int:
        for i, l in enumerate(self.layers):
            if l.id == piece_id:
                return i
        raise KeyError(piece_id)

    def report(self) -> dict:
        area = int(self.photo_shape[0] * self.photo_shape[1])
        return {"camadas": {"n": len(self.layers), "ordem_pintura": [l.label for l in self.layers],
                            "apoios": {self.layers[i].label + f"#{self.layers[i].id}":
                                       self.layers[j].label + f"#{self.layers[j].id}"
                                       for i, j in self.supports.items()},
                            "pecas": [l.summary(area) for l in self.layers],
                            **self.stats, "avisos": list(self.warnings)}}


# ============================================================================================================
# Utilitários
# ============================================================================================================
def _bbox(m: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _dil(m: np.ndarray, r: int) -> np.ndarray:
    return cv2.dilate(m.astype(np.uint8), rp.disk(r)) > 0 if r > 0 else m.copy()


def _resize_depth(depth: np.ndarray | None, H: int, W: int) -> np.ndarray | None:
    if depth is None:
        return None
    d = np.asarray(depth, np.float32)
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    return d


def _bottoms(m: np.ndarray) -> np.ndarray:
    """Para cada coluna, a última linha da máscara (−1 se a coluna está vazia)."""
    H = m.shape[0]
    has = m.any(axis=0)
    last = H - 1 - np.argmax(m[::-1], axis=0)
    return np.where(has, last, -1)


# ============================================================================================================
# Ordem de pintura
# ============================================================================================================
def occlusion_graph(labels: Sequence[str], hard: Sequence[np.ndarray], depth: np.ndarray | None = None
                    ) -> tuple[np.ndarray, dict]:
    """``front[i, j]`` = peça i na frente da j, e ``supports`` {apoiada: apoio}.

    1. Borda comum (>= 12 px de cada lado): com profundidade (MoGe), a de menor z mediano na borda está na frente;
       sem profundidade, a que termina mais embaixo na foto (mais perto da câmera, no piso).
    2. Tapete atrás de tudo o que toca nele; quadro/espelho/relógio atrás de tudo o que toca nele.
    3. Peça pequena (vaso, almofada, luminária...) cuja base está apoiada noutra peça fica na frente do apoio.
    """
    n = len(hard)
    front = np.zeros((n, n), bool)
    if n == 0:
        return front, {}
    H, W = hard[0].shape
    depth = _resize_depth(depth, H, W)
    dil = [_dil(h, 3) for h in hard]
    bottom = [(_bottoms(h).max() if h.any() else -1) for h in hard]
    touch = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            bj = dil[i] & hard[j]          # pixels de j junto de i
            bi = dil[j] & hard[i]
            if bj.sum() < CONTACT_MIN_PX or bi.sum() < CONTACT_MIN_PX:
                continue
            touch[i, j] = touch[j, i] = True
            zi = zj = np.nan
            if depth is not None:
                vi, vj = depth[bi], depth[bj]
                vi, vj = vi[np.isfinite(vi) & (vi > 0)], vj[np.isfinite(vj) & (vj > 0)]
                if vi.size >= CONTACT_MIN_PX // 2 and vj.size >= CONTACT_MIN_PX // 2:
                    zi, zj = float(np.median(vi)), float(np.median(vj))
            if np.isfinite(zi) and np.isfinite(zj):
                if zi < zj * DEPTH_FRONT:
                    front[i, j] = True
                elif zj < zi * DEPTH_FRONT:
                    front[j, i] = True
            else:
                if bottom[i] > bottom[j] + 0.01 * H:
                    front[i, j] = True
                elif bottom[j] > bottom[i] + 0.01 * H:
                    front[j, i] = True
    for i, lab in enumerate(labels):
        if lab in RUG_LABELS or lab in WALL_HUNG_LABELS:
            for j in range(n):
                if j != i and touch[i, j] and not (labels[j] in RUG_LABELS and lab in WALL_HUNG_LABELS):
                    front[i, j] = False
                    front[j, i] = True
    # Apoio direto: a peça embaixo da base. Peça pequena sobre outra pequena (almofada sobre manta/almofada no
    # sofá, garrafa entre almofadas) herda o apoio grande dela, senão a regra "termina mais embaixo" punha o sofá
    # NA FRENTE da almofada que está sentada nele.
    direct: dict = {}
    for i, lab in enumerate(labels):
        if lab not in SMALL_LABELS or not hard[i].any():
            continue
        bt = _bottoms(hard[i])
        cols = np.nonzero(bt >= 0)[0]
        votes = np.zeros(n, np.int64)
        for dy in range(1, 7):
            ys = np.minimum(bt[cols] + dy, H - 1)
            for j in range(n):
                if j != i and labels[j] not in RUG_LABELS:
                    votes[j] += int(hard[j][ys, cols].sum())
        need = max(CONTACT_MIN_PX, 0.25 * 6 * cols.size)
        j = int(np.argmax(votes))
        if votes[j] < need:
            continue
        if labels[j] in SMALL_LABELS and not (lab in ("flower", "plant") and labels[j] in ("vase", "pot")):
            big = [q for q in range(n) if labels[q] not in SMALL_LABELS and votes[q] >= need / 2]
            if big:                                   # apoio grande também embaixo (ex.: manta + sofá)
                j = max(big, key=lambda q: votes[q])
                direct[i] = (j, True)
            else:
                direct[i] = (j, False)                # só sobre outra pequena: resolve pela cadeia abaixo
        else:
            direct[i] = (j, True)
    supports: dict = {}
    for i in direct:
        j, ok, seen = direct[i][0], direct[i][1], {i}
        while not ok and j in direct and j not in seen:    # sobe a cadeia até um apoio de verdade
            seen.add(j)
            j, ok = direct[j]
        if not ok or j == i:
            continue
        supports[i] = j
        front[i, j] = True
        front[j, i] = False
    return front, supports


def toposort(before: np.ndarray, priority: Sequence[float]) -> list[int]:
    """Kahn: ``before[a, b]`` = a vem antes de b. Empate (ou ciclo): maior ``priority`` primeiro."""
    n = before.shape[0]
    indeg = before.sum(axis=0).astype(int)
    done = np.zeros(n, bool)
    order: list[int] = []
    while len(order) < n:
        cand = [k for k in range(n) if not done[k] and indeg[k] == 0]
        if not cand:
            cand = [k for k in range(n) if not done[k]]
        k = max(cand, key=lambda q: (priority[q], -q))
        order.append(k)
        done[k] = True
        for b in np.nonzero(before[k])[0]:
            indeg[b] -= 1
    return order


def paint_order(labels: Sequence[str], hard: Sequence[np.ndarray], depth: np.ndarray | None = None
                ) -> tuple[list[int], np.ndarray, dict]:
    """Ordem de pintura de trás para frente (índices das peças), mais ``front`` e ``supports``."""
    front, supports = occlusion_graph(labels, hard, depth)
    prio = [(2.0 if labels[k] in RUG_LABELS else 1.0 if labels[k] in WALL_HUNG_LABELS else 0.0)
            * 1e9 + float(hard[k].sum()) for k in range(len(hard))]
    return toposort(front.T, prio), front, supports


# ============================================================================================================
# Construção
# ============================================================================================================
def _reach(piece, diag: float) -> float:
    builtin = getattr(piece, "kind", "") == "planejado"
    ys = np.nonzero(piece.mask.any(axis=1))[0]
    hgt = float(ys[-1] - ys[0] + 1) if ys.size else 0.0
    ring = (rp.EFFECT_RING_BUILTIN if builtin else rp.EFFECT_RING) * diag
    band = (rp.BAND_DOWN_BUILTIN if builtin else rp.BAND_DOWN) * hgt
    return max(ring, band) + EFFECT_SLACK_PX


def build_layers(photo: np.ndarray, plate_res, pieces: Sequence, depth: np.ndarray | None = None,
                 ring_px: int | None = None) -> LayerStack:
    """Camadas das ``pieces`` (as mesmas que geraram ``plate_res``: ``inst_set.removal(plate_res.kept_builtins)``).

    ``plate_res`` precisa de ``plate``, ``mask`` e ``shadow`` (``PlateResult``). ``depth``: profundidade do
    MoGe (qualquer resolução) para a ordem de pintura. ``ring_px``: raio do anel de contato (padrão: a dilatação
    do núcleo da máscara de remoção + 2 px)."""
    t0 = time.perf_counter()
    H, W = photo.shape[:2]
    diag = float(np.hypot(H, W))
    area = H * W
    P = photo.astype(np.float32)
    plate8 = np.asarray(plate_res.plate)
    mask = np.asarray(plate_res.mask, bool)
    warnings: list[str] = []
    if plate8.shape != photo.shape or mask.shape != (H, W):
        raise ValueError("placa/máscara com tamanho diferente da foto")
    if not np.array_equal(plate8[~mask], photo[~mask]):
        raise ValueError("placa difere da foto fora da máscara de remoção")
    n = len(pieces)
    if ring_px is None:
        ring_px = max(3, int(rp.CORE_DILATE * diag)) + 2

    # --- alpha limitado à máscara de remoção (fora dela a placa já é a foto) ---------------------------------
    A, hard, outside = [], [], 0.0
    for p in pieces:
        a = np.clip(np.asarray(p.alpha, np.float32), 0.0, 1.0)
        outside += float(a[~mask].sum())
        a = np.where(mask & (a >= ALPHA_MIN), a, 0.0).astype(np.float32)
        A.append(a)
        hard.append(np.asarray(p.mask, bool) & mask)
    total_alpha = sum(float(a.sum()) for a in A) + outside
    if total_alpha > 0 and outside / total_alpha > 0.01:
        warnings.append(f"{100 * outside / total_alpha:.1f}% do alpha das peças fica fora da máscara da placa "
                        "(peças diferentes das usadas na placa?)")
    supp = [a > 0 for a in A]
    labels = [str(getattr(p, "label", "")) for p in pieces]
    order, front, supports = paint_order(labels, [h | s for h, s in zip(hard, supp)], depth)

    support = np.zeros((H, W), bool)
    for s in supp:
        support |= s
    # --- dono de cada pixel fora do alpha: a peça mais próxima ------------------------------------------------
    # O dono de um pixel de efeito é a peça mais próxima cujo alcance (anel/faixa do ``plate.py``) o cobre: a luz
    # de LED sob o armário aéreo é do armário, não do vaso pequeno que está mais perto dela.
    reach = np.array([_reach(p, diag) for p in pieces] + [0.0], np.float32)
    dmin = np.full((H, W), np.inf, np.float32)
    owner = np.full((H, W), -1, np.int32)
    dmin_r = np.full((H, W), np.inf, np.float32)
    owner_r = np.full((H, W), -1, np.int32)
    for k in range(n):
        if not supp[k].any():
            continue
        dt = cv2.distanceTransform((~supp[k]).astype(np.uint8), cv2.DIST_L2, 5)
        upd = dt < dmin
        dmin[upd] = dt[upd]
        owner[upd] = k
        upd = (dt <= reach[k]) & (dt < dmin_r)
        dmin_r[upd] = dt[upd]
        owner_r[upd] = k
    ring = mask & ~support & (dmin <= ring_px)

    # --- sombra/reflexo: razão foto/placa <= 1, só perto da peça e só onde a foto é mais escura ---------------
    # A razão é suave (média gaussiana normalizada dentro da região), então o sombreamento que acompanha a peça
    # em movimento é uma mancha de sombra e não carrega a textura do piso; a textura que sobra vai para o delta.
    plate = plate8.astype(np.float32)
    shadow = np.asarray(getattr(plate_res, "shadow", np.zeros((H, W), bool)), bool)
    effect = np.asarray(getattr(plate_res, "effect", shadow), bool) | shadow
    near = owner_r >= 0
    owner_e = np.where(near, owner_r, owner)
    free = mask & ~support & ~ring & (owner >= 0)
    # região de efeito da peça: sombra ∪ luz da placa (com a mesma folga de 3 px da máscara), só ao alcance de
    # ALGUMA peça (não só da mais próxima: devolver à foto a luz do armário aéreo porque um vaso está mais perto
    # punha um pedaço da foto mobiliada flutuando na sala vazia do quadro 0). A luz (reflexo claro) não vira
    # razão > 1: entra só pelo delta, junto com a peça.
    eff = _dil(effect, 3) & free & near
    rejected = int((effect & mask & ~support & ~ring & ~eff).sum())
    S = np.ones((H, W, 3), np.float32)
    sh = shadow & eff
    if sh.any():
        sig = max(1.5, SHADE_SIGMA * diag)
        ratio = rp._nblur(P, sh, sig) / np.maximum(rp._nblur(plate, sh, sig), 1.0)
        S[sh] = np.clip(ratio[sh], SHADE_MIN, 1.0)
    shaded = (S < 1.0).any(axis=2)

    # --- correção do resíduo NA PLACA: o que não é de nenhuma peça (nem alpha, nem anel, nem efeito) vira o
    # pixel da foto. Fica igual em todos os quadros: o piso e as paredes não "se curam" durante o vídeo.
    plate_fix = plate8.copy()
    leftover = mask & ~support & ~ring & ~eff
    plate_fix[leftover] = photo[leftover]
    corrected = leftover
    # "fantasma": pixel corrigido muito diferente da placa = pedaço da foto mobiliada voltando para a sala vazia
    ghost = leftover & (np.abs(photo.astype(np.int16) - plate8.astype(np.int16)).max(axis=2) > GHOST_LEVELS)

    # --- RGBA pré-multiplicado na ordem de pintura -------------------------------------------------------------
    below = plate_fix.astype(np.float32) * S
    C = [None] * n
    topmost = np.full((H, W), -1, np.int32)
    for k in order:
        s = supp[k]
        if not s.any():
            C[k] = np.zeros((0, 3), np.float32)
            continue
        a = A[k][s][:, None]
        c = np.clip(P[s] - (1.0 - a) * below[s], 0.0, 255.0 * a)
        C[k] = c
        below[s] = c + (1.0 - a) * below[s]
        topmost[s] = k
    R = P - below
    # delta de contato: no alpha (dono = a de cima), no anel e no efeito (dono = a mais próxima)
    downer = np.where(support, topmost, np.where(ring, owner, np.where(eff, owner_e, -1)))
    stray = np.abs(R[downer < 0]).max() if (downer < 0).any() else 0.0
    if stray > 0.05:      # nunca deveria acontecer: placa corrigida, foto fora da máscara
        raise AssertionError(f"resíduo fora das camadas ({stray:.3f} níveis)")

    layers: list[Layer] = []
    idx_of = {}
    for k in order:
        p = pieces[k]
        own = downer == k
        e = eff & (owner_e == k)
        region = supp[k] | own | e
        es = e & shaded
        if not region.any():
            warnings.append(f"peça {getattr(p, 'id', k)} ({labels[k]}) sem pixels dentro da máscara: ignorada")
            continue
        x0, y0, x1, y1 = _bbox(region)
        sl = (slice(y0, y1), slice(x0, x1))
        al = A[k][sl].copy()
        col = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        sc = supp[k][sl]
        col[sc] = C[k]
        sh = None
        if es.any():
            sh = np.ones((y1 - y0, x1 - x0, 3), np.float32)
            sh[es[sl]] = S[sl][es[sl]]
        dl = None
        if own.any():
            dl = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
            dl[own[sl]] = R[sl][own[sl]]
        idx_of[k] = len(layers)
        layers.append(Layer(int(getattr(p, "id", k)), labels[k], str(getattr(p, "kind", "")), (x0, y0, x1, y1),
                            al, col, sh, dl, (ring & (owner == k))[sl], e[sl],
                            tuple(getattr(p, "touches", ()) or ()), hard[k][sl]))
    kept = list(idx_of)
    fr = front[np.ix_(kept, kept)] if kept else np.zeros((0, 0), bool)
    sup = {idx_of[i]: idx_of[j] for i, j in supports.items() if i in idx_of and j in idx_of}
    dynamic = support | ring | eff
    stack = LayerStack((H, W), plate_fix, layers, fr, sup, dynamic, corrected, {}, warnings)

    full = composite(stack, range(len(layers)))
    diff = np.abs(full.astype(np.int16) - photo.astype(np.int16)).max(axis=2)
    stack.stats = {
        "reconstrucao_max": int(diff.max()), "reconstrucao_p99": float(np.percentile(diff, 99)),
        "morph_fora_pct": round(100.0 * morph_fraction(stack, stack.plate, full), 4),
        "placa_corrigida_pct": round(100.0 * float(corrected.sum()) / area, 3),
        "placa_corrigida_da_mascara_pct": round(100.0 * float(corrected.sum()) / max(1, int(mask.sum())), 2),
        "anel_px": ring_px, "anel_pct": round(100.0 * float(ring.sum()) / area, 3),
        "efeito_pct": round(100.0 * float(eff.sum()) / area, 3),
        "sombreamento_pct": round(100.0 * float(shaded.sum()) / area, 3),
        "efeito_longe_px": rejected,
        "placa_corrigida_fantasma_pct": round(100.0 * float(ghost.sum()) / area, 3),
        "tempo_s": round(time.perf_counter() - t0, 2),
    }
    if diff.max() > RECON_TOL:
        warnings.append(f"pilha completa difere da foto em até {int(diff.max())} níveis (esperado <= {RECON_TOL})")
    if ghost.sum() > GHOST_WARN_FRAC * area:
        warnings.append(f"{100.0 * ghost.sum() / area:.2f}% da imagem voltou a ser a foto mobiliada na sala vazia "
                        "(pedaço de móvel sem camada): conferir a placa")
    if corrected.sum() > 0.25 * max(1, int(mask.sum())):
        warnings.append("mais de 25% da máscara de remoção voltou a ser a foto na placa (sombras/luz fortes ou "
                        "peças que não cobrem a máscara)")
    return stack


# ============================================================================================================
# Composição
# ============================================================================================================
def _paste_slices(box, dx, dy, H, W):
    x0, y0, x1, y1 = box
    X0, Y0, X1, Y1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
    cx0, cy0, cx1, cy1 = max(0, X0), max(0, Y0), min(W, X1), min(H, Y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return None
    return (slice(cy0, cy1), slice(cx0, cx1)), (slice(cy0 - Y0, cy1 - Y0), slice(cx0 - X0, cx1 - X0))


def _scaled(L: Layer, q: Placement):
    """(alpha, cor, sombreamento, caixa) da camada com ``alpha_mul`` e ``scale`` aplicados. Com escala, a caixa
    é a da peça encolhida em torno de ``anchor`` (coordenadas absolutas, pode ficar fora da caixa)."""
    a, c, sh = L.alpha, L.color, L.shade
    if q.alpha_mul is not None:
        m = np.asarray(q.alpha_mul, np.float32)
        a, c = a * m, c * m[..., None]
    if q.scale >= 1.0 - 1e-6:
        return a, c, sh, L.box
    x0, y0, x1, y1 = L.box
    ax, ay = q.anchor if q.anchor is not None else ((x0 + x1) / 2.0, float(y1 - 1))
    s = float(q.scale)
    X0, Y0 = int(math.floor(ax + s * (x0 - ax))), int(math.floor(ay + s * (y0 - ay)))
    X1, Y1 = int(math.ceil(ax + s * (x1 - ax))) + 1, int(math.ceil(ay + s * (y1 - ay))) + 1
    # destino (X, Y) <- origem absoluta a + s·(p − a); em coordenadas locais das duas caixas
    M = np.float32([[s, 0, ax + s * (x0 - ax) - X0], [0, s, ay + s * (y0 - ay) - Y0]])
    size = (X1 - X0, Y1 - Y0)
    warp = lambda img, border: cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=border)
    a2 = warp(np.ascontiguousarray(a, np.float32), 0)
    c2 = warp(np.ascontiguousarray(c, np.float32), (0, 0, 0))
    c2 = np.minimum(np.maximum(c2, 0.0), 255.0 * a2[..., None])     # interpolação linear mantém C <= 255·A
    sh2 = None if sh is None else warp(np.ascontiguousarray(sh, np.float32), (1, 1, 1))
    return a2, c2, sh2, (X0, Y0, X1, Y1)


def composite(stack: LayerStack, present: Iterable[int] | Mapping[int, Placement], as_float: bool = False
              ) -> np.ndarray:
    """Quadro com as camadas ``present`` (índices de ``stack.layers``; um dict dá a :class:`Placement` de cada
    uma). Ordem: placa → sombreamento (desloca com a peça) → RGBA na ordem de pintura → delta de contato (fixo
    na posição de repouso). Sem nenhuma camada devolve a placa corrigida byte a byte."""
    H, W = stack.photo_shape
    pl = dict(present) if isinstance(present, Mapping) else {int(i): Placement() for i in present}
    if not pl:
        return stack.plate.astype(np.float32) if as_float else stack.plate.copy()
    out = stack.plate.astype(np.float32)
    views = {i: _scaled(stack.layers[i], pl[i]) for i in pl}
    for i in sorted(pl):
        L, q = stack.layers[i], pl[i]
        shade, box = views[i][2], views[i][3]
        if shade is None or q.shade <= 0:
            continue
        s = _paste_slices(box, q.dx, q.dy, H, W)
        if s is None:
            continue
        dst, src = s
        sh = shade[src]
        out[dst] *= sh if q.shade >= 1 else 1.0 - q.shade * (1.0 - sh)
    for i in sorted(pl):                       # layers já estão na ordem de pintura
        L, q = stack.layers[i], pl[i]
        s = _paste_slices(views[i][3], q.dx, q.dy, H, W)
        if s is None:
            continue
        dst, src = s
        a, c = views[i][0][src], views[i][1][src]
        out[dst] = c + (1.0 - a[..., None]) * out[dst]
    for i in sorted(pl):
        L, q = stack.layers[i], pl[i]
        if L.delta is None or q.contact <= 0:
            continue
        x0, y0, x1, y1 = L.box
        d = L.delta * min(1.0, q.contact)
        if q.alpha_mul is not None:          # parte ainda não revelada (tapete enrolado) não recebe o delta
            d = d * np.asarray(q.alpha_mul, np.float32)[..., None]
        out[y0:y1, x0:x1] += d
    if as_float:
        return out
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def morph_fraction(stack: LayerStack, first: np.ndarray, last: np.ndarray, tol: int = 0) -> float:
    """Fração da imagem FORA da região dinâmica (alpha ∪ anel ∪ efeito) que muda entre dois quadros (aceite
    PT-V3-5: 0 entre o 1º e o último)."""
    d = np.abs(first.astype(np.int16) - last.astype(np.int16)).max(axis=2) > tol
    return float((d & ~stack.dynamic).sum()) / float(d.size)
