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
DEPTH_STRONG = 0.95          # folga da profundidade "certa" (verificação independente)
HULL_ZONE = 0.03             # pistas da envoltória só perto da borda comum (fração da diagonal)
HULL_RATIO = 2.0             # dono do contorno: a outra peça entra 2× mais na envoltória
HULL_STRONG = 4.0            # ... 4× para a evidência "certa" da verificação independente
HULL_MIN_FRAC = 0.5          # ... e com pelo menos metade dos pixels da borda comum
SHADE_MIN = 0.2              # razão mínima do sombreamento (o resto do escurecimento vai para o delta)
SHADE_SIGMA = 0.002          # suavização da razão (fração da diagonal)
EFFECT_SLACK_PX = 4
GHOST_LEVELS = 30           # pixel corrigido na placa que difere disto da placa do LaMa = "fantasma" de móvel
GHOST_WARN_FRAC = 0.001     # aviso se os fantasmas passam de 0,1% da imagem
RECON_TOL = 2                # aceite PT-V3-5: pilha completa x foto <= 2 níveis
RUG_CORE_ALPHA = 0.98        # tapete "puro" (fonte do preenchimento); a franja abaixo disto junto do móvel é refeita
RUG_FILL_MIN_PX = 16         # completar o tapete só se a parte escondida tem pelo menos isso


@dataclass
class Placement:
    """Como uma peça aparece num quadro. ``dx``/``dy`` (px inteiros) deslocam o RGBA e o sombreamento (o
    reflexo acompanha a peça); ``shade`` (0..1) e ``contact`` (0..1) são as intensidades do sombreamento e do
    delta de contato (que fica na posição de repouso); ``alpha_mul`` (recorte do tamanho da caixa, 0..1)
    revela só parte da peça no lugar (tapete desenrolando). ``scale`` (<= 1) encolhe RGBA e sombreamento em torno
    de ``anchor`` (x, y em pixels absolutos da posição de repouso; padrão: centro da base da caixa), antes do
    deslocamento: "escala no ponto de contato" da coreografia (PT-V3-6). ``warp`` (h, w, 2) float32, do tamanho
    da caixa: deslocamento (dx, dy) em px de cada pixel da peça dentro da própria caixa (tecido assentando quando
    a peça pousa, ``choreo.fabric_warp``); aplicado antes da escala, nunca sai da caixa. ``None`` = sem."""
    dx: int = 0
    dy: int = 0
    shade: float = 1.0
    contact: float = 1.0
    alpha_mul: np.ndarray | None = None
    scale: float = 1.0
    anchor: tuple[float, float] | None = None
    warp: np.ndarray | None = None


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
    amodal: np.ndarray | None = None    # (h, w) bool: tapete completado por baixo dos móveis (envoltória): dentro
    #                                     dela a borda da camada é a silhueta (não é corte); ver ``_complete_rugs``

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
    # (n, n) bool: evidência clara de oclusão tirada só das máscaras/profundidade/rótulos, SEM o grafo (desempates,
    # ciclos, blocos). Base da verificação independente do "pixel oculto visível" (``choreo.verify_independent``)
    evidence: np.ndarray | None = None
    # tapete completado -> peças da frente que ele não precisa esperar: o tapete desenrola ANTES delas
    rug_first: dict = field(default_factory=dict)

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
def _hull(m: np.ndarray) -> np.ndarray:
    """Envoltória convexa (máscara cheia) dos pixels de ``m``."""
    out = np.zeros(m.shape, np.uint8)
    if m.any():
        pts = np.argwhere(m)[:, ::-1].astype(np.int32)
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out > 0


@dataclass
class _Pair:
    """Pistas de quem está na frente na borda comum de duas peças (sinal > 0: a primeira, ``i``, na frente)."""
    depth: int = 0          # profundidade mediana na borda (MoGe), tolerância DEPTH_FRONT
    depth_strong: int = 0   # idem com DEPTH_STRONG (pista "certa" para a verificação independente)
    hull: int = 0           # a borda comum fica dentro da envoltória convexa de quem está atrás
    hull_strong: int = 0    # idem com folga HULL_STRONG (pista "certa")
    bottom: int = 0         # nas colunas da borda comum, quem desce mais (fraca: só desempate)
    prior: int = 0          # tapete / quadro de parede atrás de quem toca nele (física, não imagem)


def _pair_cues(i: int, j: int, labels, hard, dil, depth, zone_k) -> _Pair | None:
    bj = dil[i] & hard[j]          # pixels de j junto de i
    bi = dil[j] & hard[i]
    if bj.sum() < CONTACT_MIN_PX or bi.sum() < CONTACT_MIN_PX:
        return None
    c = _Pair()
    if depth is not None:
        vi, vj = depth[bi], depth[bj]
        vi, vj = vi[np.isfinite(vi) & (vi > 0)], vj[np.isfinite(vj) & (vj > 0)]
        if vi.size >= CONTACT_MIN_PX // 2 and vj.size >= CONTACT_MIN_PX // 2:
            zi, zj = float(np.median(vi)), float(np.median(vj))
            c.depth = 1 if zi < zj * DEPTH_FRONT else (-1 if zj < zi * DEPTH_FRONT else 0)
            c.depth_strong = 1 if zi < zj * DEPTH_STRONG else (-1 if zj < zi * DEPTH_STRONG else 0)
    # Dono do contorno: quem está na frente "morde" a silhueta de quem está atrás, então a borda comum (e os pixels
    # da peça da frente junto dela) caem DENTRO da envoltória convexa da peça de trás; a borda da peça da frente
    # é a silhueta dela (fica na envoltória dela). Mesa x cadeira na frente: a cadeira entra no retângulo da mesa.
    # A envoltória é LOCAL (só a parte de cada peça perto da borda comum): a global do sofá inteiro engolia a mesa
    # lateral atrás do braço dele.
    r = zone_k.shape[0] // 2
    ys, xs = np.nonzero(bi | bj)
    Hh, Ww = hard[i].shape
    y0, y1 = max(0, int(ys.min()) - r), min(Hh, int(ys.max()) + r + 1)
    x0, x1 = max(0, int(xs.min()) - r), min(Ww, int(xs.max()) + r + 1)
    sl = (slice(y0, y1), slice(x0, x1))                # tudo no recorte da zona (rápido em foto grande)
    zone = cv2.dilate((bi[sl] | bj[sl]).astype(np.uint8), zone_k) > 0
    hi, hj = hard[i][sl], hard[j][sl]
    a_j = int((hj & _hull(hi & zone) & ~hi).sum())     # j dentro da envoltória de i: j na frente
    a_i = int((hi & _hull(hj & zone) & ~hj).sum())
    need = max(4 * CONTACT_MIN_PX, int(HULL_MIN_FRAC * min(int(bi.sum()), int(bj.sum()))))
    if a_j >= need and a_j >= HULL_RATIO * max(a_i, 1):
        c.hull = -1
        c.hull_strong = -1 if a_j >= HULL_STRONG * max(a_i, 1) else 0
    elif a_i >= need and a_i >= HULL_RATIO * max(a_j, 1):
        c.hull = 1
        c.hull_strong = 1 if a_i >= HULL_STRONG * max(a_j, 1) else 0
    # base local: só as colunas da borda comum (a base global engana: pé de mesa visível entre as cadeiras)
    # (sem colunas em comum, lado a lado: a base de cada uma perto da borda e, por fim, a base global)
    tol = max(1.0, 0.01 * hard[i].shape[0])
    cols = np.nonzero(zone.any(axis=0))[0] + x0
    d = 0.0
    bti, btj = _bottoms(hard[i][:, cols]), _bottoms(hard[j][:, cols])
    both = (bti >= 0) & (btj >= 0)
    if both.sum() >= 3:
        d = float(np.median(bti[both] - btj[both]))
    if abs(d) <= tol:
        yi, yj = np.nonzero((hi & zone).any(axis=1))[0], np.nonzero((hj & zone).any(axis=1))[0]
        if yi.size and yj.size:
            d = float(yi[-1] - yj[-1])
    if abs(d) <= tol:
        d = float(_bottoms(hard[i]).max() - _bottoms(hard[j]).max())
    c.bottom = 1 if d > tol else (-1 if d < -tol else 0)
    li, lj = labels[i], labels[j]
    back_i = li in RUG_LABELS or (li in WALL_HUNG_LABELS and lj not in RUG_LABELS)
    back_j = lj in RUG_LABELS or (lj in WALL_HUNG_LABELS and li not in RUG_LABELS)
    if back_i and not back_j:
        c.prior = -1
    elif back_j and not back_i:
        c.prior = 1
    return c


def occlusion_cues(labels: Sequence[str], hard: Sequence[np.ndarray], depth: np.ndarray | None = None
                   ) -> dict:
    """Pistas de oclusão de cada par de peças com borda comum: {(i, j): _Pair} com i < j."""
    n = len(hard)
    if n == 0:
        return {}
    H, W = hard[0].shape
    depth = _resize_depth(depth, H, W)
    dil = [_dil(h, 3) for h in hard]
    r = max(6, int(HULL_ZONE * math.hypot(H, W)))
    zone_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            c = _pair_cues(i, j, labels, hard, dil, depth, zone_k)
            if c is not None:
                out[(i, j)] = c
    return out


def confident_front(cues: dict, n: int) -> np.ndarray:
    """``ev[i, j]`` = há evidência clara (sem contradição) de que i está na frente de j na borda comum: tapete e
    quadro atrás de quem toca neles, profundidade com folga de 5% ou dono do contorno pela envoltória convexa com
    folga de 4×, sem nenhuma pista (nem fraca) dizendo o contrário. Não usa o grafo de oclusão (sem desempates,
    ciclos quebrados, blocos, apoios): é a base da verificação
    independente do "pixel oculto visível" (:func:`choreo.verify_independent`)."""
    ev = np.zeros((n, n), bool)
    for (i, j), c in cues.items():
        # qualquer pista contrária (mesmo a fraca) tira a certeza
        votes = {v for v in (c.depth_strong, c.hull_strong) if v} | ({c.depth} if c.depth else set()) \
            | ({c.hull} if c.hull else set())
        if not (c.depth_strong or c.hull_strong):
            votes = set()
        if c.prior:
            votes = {c.prior}                       # física: o tapete está no chão, o quadro na parede
        if len(votes) == 1:
            v = votes.pop()
            if v > 0:
                ev[i, j] = True
            else:
                ev[j, i] = True
    return ev


def occlusion_graph(labels: Sequence[str], hard: Sequence[np.ndarray], depth: np.ndarray | None = None,
                    cues: dict | None = None) -> tuple[np.ndarray, dict]:
    """``front[i, j]`` = peça i na frente da j, e ``supports`` {apoiada: apoio}.

    1. Borda comum (>= 12 px de cada lado): com profundidade (MoGe), a de menor z mediano na borda está na frente.
       Empate (diferença < 3%, comum entre peças encostadas: mesa e cadeira na mesma borda), pistas da imagem,
       nesta ordem: dono do contorno (a borda comum cai dentro da envoltória convexa de quem está atrás); peça
       pequena na frente da grande; nas colunas da borda comum, a que desce mais (mais perto da câmera); por
       fim a menor na frente. Toda borda comum ganha uma ordem: sem ela, a coreografia não via o corte e a peça
       de trás entrava com o buraco da da frente.
    2. Tapete atrás de tudo o que toca nele; quadro/espelho/relógio atrás de tudo o que toca nele.
    3. Peça pequena (vaso, almofada, luminária...) cuja base está apoiada noutra peça fica na frente do apoio.
    """
    n = len(hard)
    front = np.zeros((n, n), bool)
    if n == 0:
        return front, {}
    H, W = hard[0].shape
    cues = occlusion_cues(labels, hard, depth) if cues is None else cues
    touch = np.zeros((n, n), bool)
    area = [int(h.sum()) for h in hard]
    for (i, j), c in cues.items():
        touch[i, j] = touch[j, i] = True
        v = c.depth or c.hull
        if not v:
            si, sj = labels[i] in SMALL_LABELS, labels[j] in SMALL_LABELS
            # manta jogada no encosto que desce até a borda da foto junto com a poltrona: a pequena na frente
            v = 1 if (si and not sj) else (-1 if (sj and not si) else 0)
        if not v:
            v = c.bottom
        if not v and area[i] != area[j]:
            v = 1 if area[i] < area[j] else -1
        if v > 0:
            front[i, j] = True
        elif v < 0:
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
def rug_mirror_fill(rgb8: np.ndarray, unknown: np.ndarray, rounds: int = 8) -> np.ndarray:
    """Preenchimento sem modelo da parte escondida do tapete: cada pixel desconhecido copia o pixel conhecido
    espelhado na borda mais próxima (``q' = 2p − q``), o que continua a estampa e a textura do tapete sem borrão
    (o Telea vira uma mancha lisa em área grande). Em rodadas: o que foi preenchido vira conhecido e o espelho
    avança para o meio do buraco. O que sobra no fim (raro), Telea."""
    img = np.ascontiguousarray(rgb8).copy()
    unk = np.asarray(unknown, bool).copy()
    h, w = unk.shape
    for _ in range(rounds):
        known = ~unk
        if not known.any() or not unk.any():
            break
        _, lab = cv2.distanceTransformWithLabels(unk.astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        kc = np.argwhere(known)                  # ordem de varredura = rótulos 1..N
        if lab[known].max() != len(kc):
            break
        qy, qx = np.nonzero(unk)
        near = kc[np.clip(lab[qy, qx] - 1, 0, len(kc) - 1)]
        my, mx = 2 * near[:, 0] - qy, 2 * near[:, 1] - qx
        ok = (my >= 0) & (mx >= 0) & (my < h) & (mx < w)
        ok[ok] &= known[my[ok], mx[ok]]
        if not ok.any():
            break
        img[qy[ok], qx[ok]] = img[my[ok], mx[ok]]
        unk[qy[ok], qx[ok]] = False
    return rp.telea_inpaint(img, unk) if unk.any() else img


def _complete_rugs(labels, hard, supp, A, front, mask, photo, fill_fn) -> dict:
    """Completa cada tapete por baixo dos móveis que estão na frente dele (PLANO_V3 §3.4: o tapete desenrola
    ANTES de os móveis pousarem nele). A parte escondida estimada é a envoltória convexa do tapete visível, só
    onde um móvel opaco o cobre na foto (nunca no piso à vista); é preenchida a partir dos pixels do próprio tapete
    (``fill_fn``: :func:`rug_mirror_fill` por padrão). Como o móvel opaco pousa por cima, a pilha completa
    continua igual à foto.
    Devolve {k: (região completada, envoltória, cores)}."""
    out = {}
    n = len(labels)
    for k in range(n):
        if labels[k] not in RUG_LABELS or not supp[k].any():
            continue
        hull = _hull(hard[k] | supp[k])
        # tudo o que não é tapete nem está atrás dele (inclusive a caixa sob a mesa de vidro, que está na frente
        # da mesa e não toca no tapete), com a franja suave: sob o alpha parcial a cor da peça é calculada sobre
        # o tapete completado, então a pilha completa continua igual à foto
        solid = np.zeros_like(mask)
        for j in range(n):
            if j != k and not front[k, j] and labels[j] not in RUG_LABELS:
                solid |= hard[j] | supp[j]
        core = supp[k] & (A[k] >= RUG_CORE_ALPHA)
        # a franja suave do tapete junto do móvel também é refeita: ela carrega um pouco da cor do móvel (contorno
        # claro do tampo de vidro "desenhado" no tapete antes de a mesa pousar)
        ext = hull & solid & ~core & mask
        if int(ext.sum()) < RUG_FILL_MIN_PX:
            continue
        x0, y0, x1, y1 = _bbox(hull)
        pad = 8
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(mask.shape[1], x1 + pad), min(mask.shape[0], y1 + pad)
        sl = (slice(y0, y1), slice(x0, x1))
        known = cv2.erode(core[sl].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if known.sum() < RUG_FILL_MIN_PX:
            continue
        filled = fill_fn(np.ascontiguousarray(photo[sl]), ~known)
        col = np.zeros(mask.shape + (3,), np.float32)
        col[sl] = np.asarray(filled, np.float32)
        out[k] = (ext, hull, col[ext])
    return out


def _reach(piece, diag: float) -> float:
    builtin = getattr(piece, "kind", "") == "planejado"
    ys = np.nonzero(piece.mask.any(axis=1))[0]
    hgt = float(ys[-1] - ys[0] + 1) if ys.size else 0.0
    ring = (rp.EFFECT_RING_BUILTIN if builtin else rp.EFFECT_RING) * diag
    band = (rp.BAND_DOWN_BUILTIN if builtin else rp.BAND_DOWN) * hgt
    return max(ring, band) + EFFECT_SLACK_PX


def build_layers(photo: np.ndarray, plate_res, pieces: Sequence, depth: np.ndarray | None = None,
                 ring_px: int | None = None, rug_fill=None) -> LayerStack:
    """Camadas das ``pieces`` (as mesmas que geraram ``plate_res``: ``inst_set.removal(plate_res.kept_builtins)``).

    ``plate_res`` precisa de ``plate``, ``mask`` e ``shadow`` (``PlateResult``). ``depth``: profundidade do
    MoGe (qualquer resolução) para a ordem de pintura. ``ring_px``: raio do anel de contato (padrão: a dilatação
    do núcleo da máscara de remoção + 2 px). ``rug_fill(rgb8, máscara) -> rgb8``: preenche a parte do tapete
    escondida pelos móveis (padrão: :func:`rug_mirror_fill`, sem modelo)."""
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
        # a dilatação que a placa usou de fato (a tentativa "mascara_ampla" do portão usa 2,5×): com a constante,
        # a faixa extra da máscara larga voltava a ser a foto na sala vazia (contorno de pés de cadeira no piso)
        info = getattr(plate_res, "info", None) or {}
        dil = float(info.get("dilatacao_nucleo") or rp.CORE_DILATE)
        ring_px = max(3, int(dil * diag)) + 2

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
    hs = [h | s for h, s in zip(hard, supp)]
    cues = occlusion_cues(labels, hs, depth)
    front, supports = occlusion_graph(labels, hs, depth, cues)
    prio = [(2.0 if labels[k] in RUG_LABELS else 1.0 if labels[k] in WALL_HUNG_LABELS else 0.0)
            * 1e9 + float(hs[k].sum()) for k in range(n)]
    order = toposort(front.T, prio)
    evidence = confident_front(cues, n)

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

    # --- tapete completado por baixo dos móveis (cor própria, alpha 1, sob peças opacas) -----------------------
    rugs = _complete_rugs(labels, hard, supp, A, front, mask, photo,
                          rug_fill if rug_fill is not None else rug_mirror_fill)
    for k, (ext, _, _) in rugs.items():
        A[k] = np.where(ext, 1.0, A[k]).astype(np.float32)
        supp[k] = supp[k] | ext
        hard[k] = hard[k] | ext

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
        if k in rugs:
            ext, _, fill = rugs[k]
            full = np.zeros((H, W, 3), np.float32)
            full[ext] = np.clip(fill, 0.0, 255.0)
            c[ext[s]] = full[s][ext[s]]
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
                            tuple(getattr(p, "touches", ()) or ()), hard[k][sl],
                            rugs[k][1][sl] if k in rugs else None))
    kept = list(idx_of)
    fr = front[np.ix_(kept, kept)] if kept else np.zeros((0, 0), bool)
    sup = {idx_of[i]: idx_of[j] for i, j in supports.items() if i in idx_of and j in idx_of}
    dynamic = support | ring | eff
    # peças da frente que o tapete completado esconde por inteiro dentro da envoltória dele: sem corte do tapete
    # nelas, então o tapete pode (e deve) desenrolar antes de elas pousarem
    rug_first = {}
    for k, (ext, hull, _) in rugs.items():
        if k not in idx_of:
            continue
        clear = set()
        for j in idx_of:
            if j != k and front[j, k]:
                left = int((hard[j] & hull & ~hard[k]).sum())
                if left <= max(CONTACT_MIN_PX, 0.002 * int(hard[j].sum())):
                    clear.add(idx_of[j])
        if clear:
            rug_first[idx_of[k]] = clear
    stack = LayerStack((H, W), plate_fix, layers, fr, sup, dynamic, corrected, {}, warnings,
                       evidence[np.ix_(kept, kept)] if kept else np.zeros((0, 0), bool), rug_first)

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
        "tapete_completado_pct": round(100.0 * sum(float(r[0].sum()) for r in rugs.values()) / area, 3),
        "oclusao_pares": len(cues),
        "oclusao_pistas": {"profundidade": sum(bool(c.depth) for c in cues.values()),
                           "contorno": sum(bool(c.hull) and not c.depth for c in cues.values()),
                           "evidencia_clara": int(evidence.sum())},
        "tempo_s": round(time.perf_counter() - t0, 2),
    }
    # cobertura da verificação independente: quantas bordas de oclusão do grafo têm evidência clara própria. As
    # bordas sem evidência só são conferidas pelo próprio grafo (a evidência usa as mesmas pistas de profundidade e
    # envoltória: um erro sistemático do MoGe passa pelos dois). Vai para o relatório para não superestimar a
    # verificação.
    n_front = int(np.asarray(fr, bool).sum())
    n_ev = int((np.asarray(stack.evidence, bool) & np.asarray(fr, bool)).sum()) if n_front else 0
    stack.stats["oclusao_cobertura_evidencia"] = {"bordas_grafo": n_front, "com_evidencia": n_ev,
                                                  "fracao": round(n_ev / n_front, 3) if n_front else None}
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


def warp_layer(a: np.ndarray, c: np.ndarray, sh: np.ndarray | None, disp: np.ndarray):
    """Alpha, cor pré-multiplicada e sombreamento com cada pixel deslocado ``disp`` (h, w, 2) dentro da caixa
    (amostragem inversa: destino p <- origem p − d). Fora da caixa entra transparente; C <= 255·A continua valendo."""
    h, w = a.shape
    d = np.asarray(disp, np.float32)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = xs - d[..., 0], ys - d[..., 1]
    rm = lambda img, border: cv2.remap(np.ascontiguousarray(img, np.float32), mx, my, cv2.INTER_LINEAR,  # noqa: E731
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    a2 = rm(a, 0)
    c2 = np.minimum(np.maximum(rm(c, (0, 0, 0)), 0.0), 255.0 * a2[..., None])
    sh2 = None if sh is None else rm(sh, (1, 1, 1))
    return a2, c2, sh2


def _scaled(L: Layer, q: Placement):
    """(alpha, cor, sombreamento, caixa) da camada com ``alpha_mul`` e ``scale`` aplicados. Com escala, a caixa
    é a da peça encolhida em torno de ``anchor`` (coordenadas absolutas, pode ficar fora da caixa)."""
    a, c, sh = L.alpha, L.color, L.shade
    if q.alpha_mul is not None:
        m = np.asarray(q.alpha_mul, np.float32)
        a, c = a * m, c * m[..., None]
    if q.warp is not None:
        a, c, sh = warp_layer(a, c, sh, q.warp)
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
