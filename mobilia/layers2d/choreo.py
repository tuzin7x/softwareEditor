"""Coreografia 2D do modo B sem revelar partes que não existem (PLANO_V3 §3.4, §3.6 e PT-V3-6).

Entrada: a :class:`~mobilia.layers2d.layers.LayerStack` (placa corrigida + uma camada por peça real da foto, na
ordem de pintura). Saída: uma :class:`Choreography` que diz, para cada quadro, quais camadas estão na tela e como
(:class:`~mobilia.layers2d.layers.Placement`), e um gerador de quadros que começa na sala vazia e termina
``array_equal`` à foto.

**O problema.** A camada de uma peça só tem os pixels que a foto mostra. O pedaço escondido atrás de outra peça
(ou fora do quadro) não existe: se a peça aparece ou se mexe sem quem a cobre na frente, o corte reto fica à vista
("pixel oculto visível"). As regras abaixo garantem que isso nunca acontece, e :func:`verify` mede em todos os
quadros:

* **Unidades.** A peça pequena apoiada noutra (almofada no sofá, vaso na mesa, abajur no criado-mudo) cobre parte
  do apoio, então entra *junto* com ele, como um bloco rígido. O mesmo vale para a bancada (e a cuba) sobre o
  armário planejado quando ela o cobre; se não cobre, ela pousa depois.
* **Ordem de entrada.** De frente para trás: uma unidade só entra depois que todas as que estão na frente dela já
  pousaram. Exceção: o tapete completado por baixo dos móveis (``layers.LayerStack.rug_first``) desenrola ANTES de
  quem está em cima dele, e quem pousa nele só termina o movimento com o tapete aberto. Entre as livres, a
  prioridade é: tapete (desenrola primeiro), planejados (base, ilha, eletros e louças → bancada → aéreos/coifa),
  móveis soltos (os mais perto da câmera primeiro), quadros e decoração.
* **Movimento** (escolhido por candidatos, cada um testado ponto a ponto contra o corte da peça):
  - tapete: **desenrola no lugar** (revelação com borda suave de poucos pixels, nunca fade);
  - peça encostada na borda da foto: **desliza a partir dessa borda**, saindo de fora do quadro (nunca cai);
  - peça coberta por outra: **sobe por trás** dela, com amplitude limitada para que o corte fique sempre atrás do
    oclusor; se a amplitude possível for < 5% da altura da peça, **escala 0,94→1 no ponto de contato**;
  - peça isolada e inteira: **queda curta (<= 25% da altura do quadro) com quique**;
  - planejados (§3.6): base, ilha, eletros e louças **sobem do chão** (recorte na linha do piso: nada aparece
    abaixo dele); bancada **pousa** sobre a base; aéreos e coifa **descem** (do teto, se encostam no topo);
  - quadro de parede cortado só pelo topo: cresce no lugar preso ao topo, ou desliza rente à parede pelo lado
    (descer do topo parecia "cair do teto");
  - peça que só poderia surgir: tenta entrar em bloco com quem a cobre (todas, ou só parte delas, as menores
    primeiro); último recurso: surge no lugar, de uma vez (opacidade 100%, nunca fade).
* **Verificação independente** (:func:`verify_independent`): o "pixel oculto visível" medido sem o grafo de
  oclusão (só a evidência clara das camadas e a borda da foto). O :func:`plan` replaneja com a relação que faltava;
  o que sobrar vai para os avisos e o vídeo é recusado (§3.5, regra 5b).
* **Sem fade.** A peça tem opacidade 100% desde o primeiro quadro em que aparece. A sombra de contato cresce
  conforme ela chega ao chão (proporcional à altura que falta) e o delta de contato (textura da sombra, reflexo)
  entra no pouso; o sombreamento acompanha o deslocamento.
* **Tempo.** Sala vazia na tela por <= 0,5 s, total de 7–10 s (até 12 s com muitos planejados), dissolução final de
  no máximo 4 quadros (só o arredondamento) e hold >= 1 s com a foto exata.

Tudo em numpy (CPU). A composição por quadro em GPU é o PT-V3-7.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import cv2
import numpy as np

from ..render.choreography import clamp01, damped_spring, ease_out_cubic, smoothstep
from . import layers as L

# --- tempos (s) ------------------------------------------------------------------------------------------------
FPS = 24
HOLD_START = 0.4            # sala vazia na tela antes da 1ª entrada (aceite: <= 0,5 s)
SETTLE_END = 0.35           # depois do último pouso, antes da dissolução
DISSOLVE_FRAMES = 4         # só o arredondamento (pilha completa x foto <= 2 níveis)
HOLD_END = 1.2              # foto exata no fim (>= 1 s)
TOTAL_MIN, TOTAL_MAX, TOTAL_MAX_BUILTIN = 7.0, 10.0, 12.0
GAP_MIN, GAP_MAX = 0.14, 1.4
DUR_SCALE_MIN, DUR_SCALE_MAX = 0.7, 1.25
MOTION_END = 0.85           # fração da duração em que o deslocamento/escala já é o final (só o contato segue)

DURATION = {"desenrolar": 1.3, "deslizar": 1.0, "subir_atras": 0.9, "escala": 0.6, "queda": 1.0, "descer": 0.9,
            "emergir": 1.1, "pousar": 0.8, "surgir": 0.45}
SMALL_DUR = 0.75            # decoração sozinha entra mais rápido

# --- amplitudes ------------------------------------------------------------------------------------------------
DROP_MAX = 0.25             # queda <= 25% da altura do quadro
DROP_MIN = 0.08
DROP_SMALL = (0.05, 0.12)
DESCEND = 0.08              # quadro/aéreo: desce rente à parede
LAND = 0.05                 # bancada pousando
RISE_MIN = 0.05             # sobe por trás só com amplitude >= 5% da altura da peça
RISE_CAP = 0.15             # e no máximo 15% do quadro
RISE_SIZE = 0.4             # ou 40% do tamanho da peça (o salto inicial fica pequeno)
SCALES = (0.94, 0.96, 0.98)
WALL_SCALES = (0.85, 0.9, 0.94)   # quadro preso ao topo da foto: cresce no lugar
WALL_SLIDE_MAX = 0.4        # ... ou desliza pelo lado se a borda lateral está a <= 40% da largura
EXIT_MARGIN = 6             # px além da borda no início do deslize
BOUNCE = 0.06               # quique: fração da altura da queda
MERGE_ROUNDS = 4            # tentativas de juntar peça "surgir" com quem a cobre
MERGE_MAX_AREA = 0.2        # bloco opcional no máximo com 20% da imagem
MERGE_MAX_PIECES = 8
CYCLE_BREAK_MAX = 1.5       # aresta de ciclo com até 1,5 × diagonal pontos de corte pode ser descartada
REPAIR_ROUNDS = 2
INDEPENDENT_REPAIRS = 2     # replanejamentos pedidos pela verificação independente
SOFT_FRAC = 0.008           # borda suave do tapete (fração da diagonal)

# --- papéis dos planejados (§3.6) ------------------------------------------------------------------------------
BASE_LABELS = frozenset({"cabinet", "kitchen island", "chest of drawers", "buffet", "case", "wardrobe", "bookcase",
                         "shelf", "bar"})
TOP_LABELS = frozenset({"counter", "countertop"})
APPLIANCE_LABELS = frozenset({"refrigerator", "stove", "oven", "dishwasher", "washer", "microwave"})
SANITARY_LABELS = frozenset({"toilet", "tub", "shower", "sink"})
UPPER_ONLY = frozenset({"hood"})
BUILTIN_RANK = {"base": 4, "eletro": 4, "louca": 4, "bancada": 3, "aereo": 2, "parede": 1}

DIRS = {"baixo": (0, 1), "esquerda": (-1, 0), "direita": (1, 0), "cima": (0, -1)}
EDGE_DIR = {"esquerda": (-1, 0), "direita": (1, 0), "base": (0, 1), "topo": (0, -1)}   # de onde vem (fora do quadro)


@dataclass
class Entrance:
    members: list[int]                  # índices de ``stack.layers`` (movem juntos)
    style: str
    direction: tuple[int, int] = (0, 0)  # deslocamento no início = amp · direction
    amp: float = 0.0                    # px
    scale0: float = 1.0
    anchor: tuple[float, float] | None = None
    start: float = 0.0
    dur: float = 1.0
    role: str = ""
    reason: str = ""
    axis: int = 0                       # desenrolar: 0 = ao longo de x, 1 = ao longo de y
    from_end: bool = False              # desenrolar: começa pelo fim do eixo (direita/baixo)
    lo: float = 0.0
    hi: float = 0.0
    after: dict = field(default_factory=dict)   # membro -> apoio (de outra unidade) que tem de pousar antes
    clip: str | None = None             # "piso": nada aparece abaixo da linha do piso; "teto": nada acima do teto
    profile: np.ndarray | None = field(default=None, repr=False)

    @property
    def motion_end(self) -> float:
        return self.start + self.dur * MOTION_END

    @property
    def end(self) -> float:
        return self.start + self.dur


@dataclass
class Choreography:
    fps: int
    shape: tuple[int, int]
    entrances: list[Entrance]
    anim_end: float
    n_anim: int                         # quadros compostos (antes da dissolução)
    n_frames: int
    stats: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    _soft: float = 8.0
    _hard: list | None = field(default=None, repr=False)
    front: np.ndarray | None = field(default=None, repr=False)   # relação de oclusão usada (ciclos quebrados)
    _bands: list | None = field(default=None, repr=False)
    # tecido assentando no pouso (camada -> :class:`FabricSpec`); vazio = sem efeito
    fabric: dict = field(default_factory=dict, repr=False)

    @property
    def duration(self) -> float:
        return self.n_frames / float(self.fps)

    def report(self, stack: L.LayerStack | None = None) -> dict:
        def name(i):
            return f"{stack.layers[i].label}#{stack.layers[i].id}" if stack is not None else str(i)
        ent = [{"pecas": [name(i) for i in e.members], "estilo": e.style, "papel": e.role,
                "inicio_s": round(e.start, 3), "duracao_s": round(e.dur, 3), "amplitude_px": round(e.amp, 1),
                "escala0": e.scale0, "motivo": e.reason} for e in self.entrances]
        tecido = [{"peca": name(i), "tipo": f.kind, "amplitude_px": round(f.amp, 2), "inicio_s": round(f.t0, 3),
                   "duracao_s": round(f.dur, 3)} for i, f in sorted(self.fabric.items())]
        return {"coreografia": {"duracao_s": round(self.duration, 3), "quadros": self.n_frames, "fps": self.fps,
                                "vazio_inicial_s": round(self.entrances[0].start, 3) if self.entrances else 0.0,
                                "entradas": ent, "tecido": tecido, **self.stats, "avisos": list(self.warnings)}}


# ============================================================================================================
# Geometria: máscaras absolutas, cortes (pontos "além da borda" da peça) e cobertura
# ============================================================================================================
def _abs_hard(stack: L.LayerStack) -> list[np.ndarray]:
    H, W = stack.photo_shape
    out = []
    for lay in stack.layers:
        m = np.zeros((H, W), bool)
        x0, y0, x1, y1 = lay.box
        hard = lay.hard if lay.hard is not None else lay.alpha > 0.5
        m[y0:y1, x0:x1] = hard
        out.append(m)
    return out


def band_radius(shape) -> int:
    return max(2, int(round(0.003 * math.hypot(*shape))))


def cut_bands(stack: L.LayerStack, hard: Sequence[np.ndarray] | None = None, r: int | None = None
              ) -> list[np.ndarray]:
    """Para cada camada, os pontos (x, y) logo além da borda da peça que pertencem a quem está na frente dela
    (``front[j, i]``) ou ficam fora do quadro. É onde a peça "continuaria": se um desses pontos cair dentro do
    quadro sem um oclusor por cima, o corte está à vista."""
    H, W = stack.photo_shape
    hard = _abs_hard(stack) if hard is None else hard
    r = band_radius((H, W)) if r is None else r
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inside = np.zeros((H + 2 * r, W + 2 * r), bool)
    inside[r:r + H, r:r + W] = True
    n = len(stack.layers)
    out = []
    for i in range(n):
        occ = np.zeros((H + 2 * r, W + 2 * r), bool)
        for j in range(n):
            if j != i and stack.front[j, i]:
                occ[r:r + H, r:r + W] |= hard[j]
        hp = np.pad(hard[i], r)
        ring = (cv2.dilate(hp.astype(np.uint8), k) > 0) & ~hp
        am = stack.layers[i].amodal
        if am is not None:
            # tapete completado por baixo dos móveis: fora da envoltória a borda é a silhueta estimada dele
            hull = np.zeros((H + 2 * r, W + 2 * r), bool)
            x0, y0, x1, y1 = stack.layers[i].box
            hull[r + y0:r + y1, r + x0:r + x1] = am
            occ &= hull
        sel = ring & (occ | ~inside)
        ys, xs = np.nonzero(sel)
        out.append(np.stack([xs - r, ys - r], axis=1).astype(np.float32))
    return out


def _anchor_of(q: L.Placement, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    ax, ay = q.anchor if q.anchor is not None else ((x0 + x1) / 2.0, float(y1 - 1))
    return np.float32([ax, ay])


def _xform(p: np.ndarray, q: L.Placement, box) -> np.ndarray:
    """Ponto de repouso -> posição no quadro (mesma convenção de ``layers.composite``)."""
    if q.scale >= 1.0 - 1e-6:
        return p + np.float32([q.dx, q.dy])
    a = _anchor_of(q, box)
    return a + q.scale * (p - a) + np.float32([q.dx, q.dy])


def _inv(qpt: np.ndarray, q: L.Placement, box) -> np.ndarray:
    d = qpt - np.float32([q.dx, q.dy])
    if q.scale >= 1.0 - 1e-6:
        return d
    a = _anchor_of(q, box)
    return a + (d - a) / q.scale


def _mul_at(q: L.Placement, box, p: np.ndarray) -> np.ndarray:
    """alpha_mul da camada nos pontos ``p`` (coordenadas absolutas de repouso); 1 fora da caixa."""
    if q.alpha_mul is None:
        return np.ones(len(p), np.float32)
    x0, y0, x1, y1 = box
    xi, yi = np.rint(p[:, 0]).astype(int) - x0, np.rint(p[:, 1]).astype(int) - y0
    ok = (xi >= 0) & (yi >= 0) & (xi < x1 - x0) & (yi < y1 - y0)
    out = np.ones(len(p), np.float32)
    out[ok] = np.asarray(q.alpha_mul)[yi[ok], xi[ok]]
    return out


def uncovered(stack: L.LayerStack, hard: Sequence[np.ndarray], bands: Sequence[np.ndarray], i: int,
              placements: dict, clip: tuple | None = None) -> int:
    """Quantos pontos do corte da camada ``i`` aparecem no quadro sem ninguém por cima (pixel oculto visível).
    ``clip`` = (tipo, linha) da entrada recortada no piso/teto: o que fica além da linha não aparece."""
    pts = bands[i]
    if not len(pts) or i not in placements:
        return 0
    H, W = stack.photo_shape
    qi = placements[i]
    lay = stack.layers[i]
    live = _mul_at(qi, lay.box, pts) >= 0.5           # tapete enrolado: o corte ainda não existe
    if clip is not None and clip[0]:
        live &= _clip_live(clip[0], clip[1], pts, qi.dy)
    q = _xform(pts[live], qi, lay.box)
    xi, yi = np.rint(q[:, 0]).astype(int), np.rint(q[:, 1]).astype(int)
    vis = (xi >= 0) & (yi >= 0) & (xi < W) & (yi < H)
    if not vis.any():
        return 0
    q = q[vis]
    cov = np.zeros(len(q), bool)
    for j, qj in placements.items():
        if j == i:          # a própria peça não tem alpha no corte: qualquer outra peça ali esconde o buraco
            continue
        lj = stack.layers[j]
        p = _inv(q[~cov], qj, lj.box)
        px, py = np.rint(p[:, 0]).astype(int), np.rint(p[:, 1]).astype(int)
        ok = (px >= 0) & (py >= 0) & (px < W) & (py < H)
        hit = np.zeros(len(p), bool)
        hit[ok] = hard[j][py[ok], px[ok]]
        if qj.alpha_mul is not None and hit.any():
            hit &= _mul_at(qj, lj.box, p) >= 0.5
        idx = np.nonzero(~cov)[0]
        cov[idx[hit]] = True
        if cov.all():
            break
    return int((~cov).sum())


# ============================================================================================================
# Unidades, papéis e ordem de entrada
# ============================================================================================================
def _bottoms(m: np.ndarray) -> np.ndarray:
    has = m.any(axis=0)
    last = m.shape[0] - 1 - np.argmax(m[::-1], axis=0)
    return np.where(has, last, -1)


def clip_profile(union: np.ndarray, kind: str, r: int) -> np.ndarray:
    """Linha do piso (``kind="piso"``: último y da peça por coluna) ou do teto (``"teto"``: primeiro y), alargada
    ``r + 1`` px para os lados (o corte vizinho também some atrás da linha). Coluna sem peça: tudo escondido."""
    H = union.shape[0]
    has = union.any(axis=0)
    k = np.ones((1, 2 * r + 3), np.uint8)
    if kind == "piso":
        prof = np.where(has, H - 1 - np.argmax(union[::-1], axis=0), -1).astype(np.int32)
        return cv2.dilate(prof[None, :].astype(np.float32), k)[0].astype(np.int32)
    prof = np.where(has, np.argmax(union, axis=0), H).astype(np.int32)
    return cv2.erode(prof[None, :].astype(np.float32), k)[0].astype(np.int32)


def _clip_live(kind: str, prof: np.ndarray, pts: np.ndarray, dy: int) -> np.ndarray:
    x = np.clip(np.rint(pts[:, 0]).astype(int), 0, len(prof) - 1)
    y = pts[:, 1] + dy
    return y <= prof[x] if kind == "piso" else y >= prof[x]


def _clip_mul(kind: str, prof: np.ndarray, box, dy: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    ys = np.arange(y0, y1)[:, None] + dy
    p = prof[x0:x1][None, :]
    return ((ys <= p) if kind == "piso" else (ys >= p)).astype(np.float32)


def builtin_role(label: str, kind: str, hard: np.ndarray, H: int) -> str:
    """Papel do planejado na coreografia (§3.6): base | bancada | aereo | eletro | louca | parede | '' (solto)."""
    if label in L.WALL_HUNG_LABELS:
        return "parede"
    if kind != "planejado":
        return ""
    if label in TOP_LABELS:
        return "bancada"
    if label in UPPER_ONLY:
        return "aereo"
    ys = np.nonzero(hard.any(axis=1))[0]
    y0, y1 = (int(ys[0]), int(ys[-1])) if ys.size else (0, H - 1)
    upper = y1 < 0.55 * H and y0 < 0.4 * H
    if label in SANITARY_LABELS:
        return "louca"
    if label in APPLIANCE_LABELS:
        return "aereo" if (label == "microwave" and upper) else "eletro"
    return "aereo" if upper else "base"


def _support_below(i: int, hard: Sequence[np.ndarray], cand: Sequence[int]) -> int | None:
    bt = _bottoms(hard[i])
    cols = np.nonzero(bt >= 0)[0]
    if not cols.size:
        return None
    H = hard[i].shape[0]
    votes = {}
    for j in cand:
        v = 0
        for dy in range(1, 7):
            ys = np.minimum(bt[cols] + dy, H - 1)
            v += int(hard[j][ys, cols].sum())
        votes[j] = v
    if not votes:
        return None
    j = max(votes, key=votes.get)
    return j if votes[j] >= max(L.CONTACT_MIN_PX, 0.25 * 6 * cols.size) else None


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def build_units(stack: L.LayerStack, hard: Sequence[np.ndarray], front: np.ndarray | None = None
                ) -> tuple[list[list[int]], dict, list[str]]:
    """Agrupa apoiado + apoio (quando o apoiado cobre o apoio) e devolve (unidades, ordem extra, papéis).

    Bancada (ou cuba) assentada num planejado: a regra "termina mais embaixo = na frente" põe o armário na frente
    dela, mas a borda de baixo da bancada é a silhueta real. Em ``front`` (cópia local, se dada) essa aresta sai,
    e a bancada pousa depois da base; se é a bancada que cobre a base, entram juntas."""
    front = stack.front if front is None else front
    n = len(stack.layers)
    H = stack.photo_shape[0]
    roles = [builtin_role(l.label, l.kind, hard[i], H) for i, l in enumerate(stack.layers)]
    dsu = _DSU(n)
    after: dict = {}                  # dependente -> apoio que tem de pousar antes (sem cobrir)
    sup = dict(stack.supports)
    bases = [j for j in range(n) if roles[j] in ("base", "eletro", "louca")]
    for i in range(n):
        if roles[i] == "bancada" or (stack.layers[i].label == "sink" and stack.layers[i].kind == "planejado"):
            cand = [j for j in range(n) if j != i and (roles[j] in ("base", "eletro") or
                                                      (stack.layers[i].label == "sink" and roles[j] == "bancada"))]
            j = _support_below(i, hard, cand if cand else bases)
            if j is not None:
                sup.setdefault(i, j)
    for i, j in sup.items():
        if i not in stack.supports and stack.layers[i].label == "sink" and front[j, i]:
            # cuba embutida: a borda da bancada (ou do tampo da ilha) cobre a cuba de verdade. Tirar a aresta
            # deixava a cuba escalar ou subir dentro do furo com a placa à vista. Entram juntas, como bloco rígido.
            dsu.union(i, j)
            continue
        if i not in stack.supports and front[j, i] and front is not stack.front:
            front[j, i] = False                      # planejado assentado: a base não "cobre" a bancada
        if front[i, j]:
            dsu.union(i, j)
        else:
            after[i] = j
    groups: dict = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)
    units = [sorted(g) for g in groups.values()]
    units.sort(key=lambda u: u[0])
    return units, after, roles


def _unit_priority(stack, hard, unit, roles, occluded: bool) -> float:
    H = stack.photo_shape[0]
    labels = [stack.layers[i].label for i in unit]
    union = np.zeros_like(hard[0])
    for i in unit:
        union |= hard[i]
    ys = np.nonzero(union.any(axis=1))[0]
    bottom = float(ys[-1]) / H if ys.size else 0.0
    area = float(union.sum()) / union.size
    rs = [BUILTIN_RANK.get(roles[i], 0) for i in unit]
    if all(lab in L.RUG_LABELS for lab in labels):
        return 1000.0 if not occluded else 300.0
    if max(rs) >= 2:
        return 500.0 + 10.0 * max(rs) + bottom
    if all(lab in L.WALL_HUNG_LABELS for lab in labels):
        return 20.0 + bottom
    if all(lab in L.SMALL_LABELS for lab in labels):
        return 50.0 + bottom
    return 100.0 + 10.0 * bottom + 5.0 * area


def _unit_graph(stack, units, after) -> tuple[np.ndarray, np.ndarray]:
    of = {i: u for u, mem in enumerate(units) for i in mem}
    n_u = len(units)
    before = np.zeros((n_u, n_u), bool)              # before[a, b]: unidade a entra antes da b
    occluded = np.zeros(n_u, bool)
    n = len(stack.layers)
    rf = getattr(stack, "rug_first", None) or {}
    for a in range(n):
        for b in range(n):
            if stack.front[a, b] and of[a] != of[b]:
                if a in rf.get(b, ()):
                    before[of[b], of[a]] = True          # tapete completado: desenrola antes de quem pousa nele
                    continue
                before[of[a], of[b]] = True
                occluded[of[b]] = True
    for i, j in after.items():
        if of[i] != of[j]:
            before[of[j], of[i]] = True
    return before, occluded


def _closure(before: np.ndarray) -> np.ndarray:
    reach = before.copy()
    while True:
        nxt = reach | ((reach.astype(np.uint8) @ reach.astype(np.uint8)) > 0)
        if np.array_equal(nxt, reach):
            return reach
        reach = nxt


def break_cycles(stack, hard, bands, units, after) -> tuple[np.ndarray, int]:
    """Quebra ciclos de oclusão entre unidades na aresta mais fraca (menos pontos de corte), numa cópia de
    ``front``. Um ciclo vem de uma relação errada (profundidade ruidosa, base de peça escondida); a mais fraca é a
    mais provável de estar errada e, sem ela, a peça "de trás" vira a da frente naquele trecho (o corte some da
    conta porque a borda passa a ser a silhueta real). Ciclo só de arestas fortes fica para virar bloco rígido."""
    front = stack.front.copy()
    n = len(stack.layers)
    H, W = stack.photo_shape
    w = np.zeros((n, n), np.int64)
    for b in range(n):
        p = bands[b]
        if not len(p):
            continue
        xi, yi = np.rint(p[:, 0]).astype(int), np.rint(p[:, 1]).astype(int)
        ok = (xi >= 0) & (yi >= 0) & (xi < W) & (yi < H)
        for a in range(n):
            if front[a, b]:
                w[a, b] = int(hard[a][yi[ok], xi[ok]].sum())
    of = {i: u for u, mem in enumerate(units) for i in mem}
    hard_after = {(of[j], of[i]) for i, j in after.items() if of[i] != of[j]}
    hard_after |= {(of[r_], of[a]) for r_, clear in (getattr(stack, "rug_first", None) or {}).items()
                   for a in clear if of[r_] != of[a]}           # tapete antes de quem pousa nele: não é oclusão
    limit = CYCLE_BREAK_MAX * math.hypot(H, W)
    broken = 0
    for _ in range(4 * n):
        before, _ = _unit_graph(dataclasses.replace(stack, front=front), units, after)
        reach = _closure(before)
        mutual = reach & reach.T
        if not mutual.any():
            break
        cand = []
        for u, v in zip(*np.nonzero(before & mutual)):
            if (u, v) in hard_after:
                continue
            cand.append((int(sum(w[a, b] for a in units[u] for b in units[v] if front[a, b])), int(u), int(v)))
        if not cand:
            break
        wt, u, v = min(cand)
        if wt > limit:
            break
        for a in units[u]:
            for b in units[v]:
                front[a, b] = False
        broken += 1
    return front, broken


def acyclic_units(stack, units, after) -> tuple[list[list[int]], int]:
    """Junta em bloco rígido as unidades em ciclo de oclusão (A cobre B que cobre C que cobre A): não há ordem de
    entrada que esconda os cortes de todas, então entram juntas. Devolve (unidades, quantos ciclos)."""
    before, _ = _unit_graph(stack, units, after)
    reach = _closure(before)
    mutual = reach & reach.T
    seen, out, cycles = set(), [], 0
    for u in range(len(units)):
        if u in seen:
            continue
        comp = [u] + [v for v in range(len(units)) if v != u and mutual[u, v]]
        seen.update(comp)
        cycles += len(comp) > 1
        out.append(sorted(i for v in comp for i in units[v]))
    out.sort(key=lambda m: m[0])
    return out, cycles


def entrance_order(stack, hard, units, after, roles) -> list[int]:
    n_u = len(units)
    before, occluded = _unit_graph(stack, units, after)
    prio = [_unit_priority(stack, hard, units[u], roles, bool(occluded[u])) for u in range(n_u)]
    return L.toposort(before, prio)


# ============================================================================================================
# Movimentos
# ============================================================================================================
def _progress(e: Entrance, t: float):
    """(k, escala, sombreamento, contato, revelação): k = fração do deslocamento inicial que ainda falta."""
    x = clamp01((t - e.start) / e.dur)
    contact = smoothstep(0.62, 1.0, x)
    s = e.style
    if s in ("queda", "pousar"):
        tf = 0.6
        if x < tf:
            k = 1.0 - (x / tf) ** 2
        else:
            tl = (x - tf) * e.dur
            k = BOUNCE * (1.6 if s == "queda" else 0.6) * max(0.0, damped_spring(tl, 2.4, 0.55))
            k *= 1.0 - smoothstep(0.72, MOTION_END, x)
        return k, 1.0, 1.0 - min(1.0, k), smoothstep(tf, 0.95, x), 1.0
    if s in ("descer", "subir_atras", "emergir"):
        k = 1.0 - ease_out_cubic(x / MOTION_END)
        return k, 1.0, 1.0 - k, contact, 1.0
    if s == "deslizar":
        k = 1.0 - ease_out_cubic(x / MOTION_END)
        return k, 1.0, 1.0, contact, 1.0
    if s == "escala":
        u = ease_out_cubic(x / 0.7)
        return 0.0, e.scale0 + (1.0 - e.scale0) * u, u, smoothstep(0.6, 1.0, x), 1.0
    if s == "desenrolar":
        w = ease_out_cubic(x / MOTION_END)
        return 0.0, 1.0, smoothstep(0.5, 1.0, x), 1.0, w
    return 0.0, 1.0, 1.0, contact, 1.0                 # surgir


def _placements_for(ch: Choreography, stack: L.LayerStack, e: Entrance, t: float) -> dict:
    if t < e.start:
        return {}
    k, sc, shade, contact, w = _progress(e, t) if t < e.end else (0.0, 1.0, 1.0, 1.0, 1.0)
    dx = int(round(k * e.amp * e.direction[0]))
    dy = int(round(k * e.amp * e.direction[1]))
    out = {}
    for i in e.members:
        lay = stack.layers[i]
        x0, y0, x1, y1 = lay.box
        mul = None
        if e.style == "desenrolar" and w < 1.0:
            coord = (np.arange(x0, x1, dtype=np.float32)[None, :] if e.axis == 0
                     else np.arange(y0, y1, dtype=np.float32)[:, None])
            span = e.hi - e.lo + ch._soft
            if e.from_end:
                front = e.hi - w * span
                m = np.clip((coord - front) / ch._soft, 0.0, 1.0)
            else:
                front = e.lo + w * span
                m = np.clip((front - coord) / ch._soft, 0.0, 1.0)
            mul = np.broadcast_to(m, (y1 - y0, x1 - x0)).astype(np.float32)
        elif e.clip and dy != 0:
            mul = _clip_mul(e.clip, e.profile, lay.box, dy)
        fs = ch.fabric.get(i)
        warp = fabric_warp(fs, t) if fs is not None and t < e.end else None
        out[i] = L.Placement(dx=dx, dy=dy, shade=shade, contact=contact if (dx == 0 and dy == 0 and sc >= 1) else 0.0,
                             alpha_mul=mul, scale=sc, anchor=e.anchor, warp=warp)
    return out


# ============================================================================================================
# Tecido assentando no pouso (efeito 2D, sem física): roupa de cama, almofadas, cortinas
# ============================================================================================================
# A peça pousa e o tecido "atrasa": a parte de cima da roupa de cama desce alguns pixels, volta com 1–2
# ultrapassagens pequenas amortecidas e assenta, com uma onda que atravessa a peça; almofadas e travesseiros dão um
# quique curto (achatam e voltam); cortina balança uma vez. Amplitude de poucos px a 1080p, tudo suave, e o
# deslocamento é ZERO a partir do fim da entrada: o quadro em repouso, a dissolução e o último quadro não mudam
# (último quadro == foto, byte a byte). Perto dos cortes da peça (onde quem está na frente a cobre) o deslocamento
# some, para nunca mostrar o corte.
FABRIC_DRAPE = frozenset({"bed", "blanket", "quilt", "duvet", "bedspread"})
FABRIC_SOFT = frozenset({"sofa", "armchair", "couch"})          # assento estofado: o mesmo, mais discreto
FABRIC_SQUASH = frozenset({"pillow", "cushion"})
FABRIC_SWAY = frozenset({"curtain", "drape", "drapery"})
FABRIC_AMP = 0.0036          # amplitude máxima (× altura do quadro): ~3,9 px a 1080p
FABRIC_SOFT_AMP = 0.45       # sofá/poltrona: fração da amplitude da cama
FABRIC_SQUASH_AMP = 0.07     # almofada: até 7% da altura dela (limitado à amplitude máxima)
FABRIC_DUR = (0.4, 0.7)      # janela do efeito no fim da entrada (s)
FABRIC_LAG = 0.18            # atraso da onda de um lado ao outro da peça (fração da janela)
FABRIC_TOP = (0.45, 0.72)    # cama/sofá: o tecido é a parte de cima (até 45% inteiro, some até 72% da altura)
FABRIC_STYLES = frozenset({"queda", "pousar", "descer", "deslizar", "subir_atras", "emergir", "escala"})


@dataclass
class FabricSpec:
    kind: str                        # "drape" | "squash" | "sway"
    weight: np.ndarray               # (h, w) float32 em [0, 1], na caixa da camada
    amp: float                       # px
    t0: float                        # início do efeito (s)
    dur: float                       # duração (termina exatamente no fim da entrada)
    lag: np.ndarray                  # (w,) atraso da onda por coluna (fração da janela)
    rows: np.ndarray                 # (h,) posição relativa da linha na peça (0 = topo, 1 = base)
    cols: np.ndarray                 # (w,) posição relativa da coluna em relação ao centro (−1..1)


def _fabric_profile(kind: str, u: np.ndarray) -> np.ndarray:
    """Curva no tempo (u = 0..1 na janela): começa e termina em 0, com velocidade final 0."""
    u = np.clip(u, 0.0, 1.0)
    env = 1.0 - np.asarray([smoothstep(0.72, 1.0, float(x)) for x in np.ravel(u)]).reshape(u.shape)
    if kind == "squash":
        return np.exp(-3.5 * u) * np.sin(4.0 * np.pi * u) * env
    if kind == "sway":
        return np.exp(-2.0 * u) * np.sin(2.0 * np.pi * u) * env
    return np.exp(-2.6 * u) * np.sin(3.0 * np.pi * u) * env      # drape: desce, volta um pouco, assenta


def fabric_warp(fs: FabricSpec, t: float) -> np.ndarray | None:
    """Deslocamento (h, w, 2) do tecido no instante ``t`` (None fora da janela do efeito)."""
    if t < fs.t0 or t >= fs.t0 + fs.dur:
        return None
    u = (t - fs.t0) / fs.dur
    h, w = fs.weight.shape
    d = np.zeros((h, w, 2), np.float32)
    if fs.kind == "squash":
        s = float(_fabric_profile("squash", np.array([u]))[0]) * fs.amp
        d[..., 1] = s * fs.rows[:, None]                         # o topo desce s, a base fica
        d[..., 0] = 0.35 * s * fs.cols[None, :]                  # e alarga um pouco (achata)
    elif fs.kind == "sway":
        f = _fabric_profile("sway", u - fs.lag)                  # (w,)
        d[..., 0] = fs.amp * f[None, :] * (fs.rows[:, None] ** 1.5)
    else:
        f = _fabric_profile("drape", u - fs.lag)                 # onda atravessando a peça
        d[..., 1] = fs.amp * f[None, :]
    d *= fs.weight[..., None]
    if not np.any(np.abs(d) > 1e-3):
        return None
    return d


def _fabric_kind(label: str) -> str | None:
    b = label.split(",")[0].strip().lower()
    if b in FABRIC_DRAPE or b in FABRIC_SOFT:
        return "drape"
    if b in FABRIC_SQUASH:
        return "squash"
    if b in FABRIC_SWAY:
        return "sway"
    return None


def fabric_regions(stack: L.LayerStack, ch: Choreography, bands: Sequence[np.ndarray] | None = None) -> dict:
    """Peças de tecido da coreografia e o peso (onde o tecido mexe) de cada uma. Só nas entradas com movimento
    (pouso, queda, descida, deslize, subida por trás, escala); nunca no tapete desenrolando nem no "surgir"."""
    H, W = stack.photo_shape
    bands = ch._bands if bands is None else bands
    out: dict = {}
    for e in ch.entrances:
        if e.style not in FABRIC_STYLES:
            continue
        dur = float(np.clip(0.6 * e.dur, *FABRIC_DUR))
        dur = min(dur, e.dur)
        for i in e.members:
            lay = stack.layers[i]
            kind = _fabric_kind(lay.label)
            if kind is None:
                continue
            sup = lay.alpha > 0.5
            if sup.sum() < 64:
                continue
            ys, xs = np.nonzero(sup)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            h, w = sup.shape
            rows = np.clip((np.arange(h, dtype=np.float32) - y0) / max(1.0, y1 - y0 - 1), 0, 1)
            cols = np.clip((np.arange(w, dtype=np.float32) - (x0 + x1) / 2.0) / max(1.0, (x1 - x0) / 2.0), -1, 1)
            if kind == "drape":
                lo, hi = FABRIC_TOP
                wy = 1.0 - np.array([smoothstep(lo, hi, float(r)) for r in rows], np.float32)
                weight = np.repeat(wy[:, None], w, axis=1)
                amp = FABRIC_AMP * H * (FABRIC_SOFT_AMP if lay.label.split(",")[0].strip().lower() in FABRIC_SOFT
                                        else 1.0)
            elif kind == "squash":
                weight = np.ones((h, w), np.float32)
                amp = min(FABRIC_AMP * H * 1.2, FABRIC_SQUASH_AMP * (y1 - y0))
            else:
                weight = np.ones((h, w), np.float32)
                amp = FABRIC_AMP * H * 0.8
            # some perto dos cortes (onde a peça continua atrás de outra ou fora do quadro)
            bx0, by0, bx1, by1 = lay.box
            cut = np.zeros((h, w), np.uint8)
            if bands is not None and i < len(bands) and len(bands[i]):
                pts = np.round(bands[i]).astype(int)
                px, py = pts[:, 0] - bx0, pts[:, 1] - by0
                ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                cut[py[ok], px[ok]] = 1
            r = max(3, int(round(2.5 * amp)) + band_radius((H, W)))
            if cut.any():
                near = cv2.dilate(cut, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
                att = 1.0 - cv2.GaussianBlur(near.astype(np.float32), (0, 0), r / 2.0)
                weight = weight * np.clip(att, 0.0, 1.0)
            # bordas da caixa: rampa curta para dentro (nada é amostrado de fora da caixa de uma vez)
            ex = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
            ey = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
            weight = weight * np.clip(np.minimum(ey[:, None], ex[None, :]) / 3.0, 0.0, 1.0)
            if float(weight.max()) <= 0.05 or amp < 0.5:
                continue
            lag = FABRIC_LAG * (cols + 1.0) / 2.0
            out[i] = FabricSpec(kind, weight.astype(np.float32), float(amp), e.end - dur, dur,
                                lag.astype(np.float32), rows, cols)
    return out


def placements_at(ch: Choreography, stack: L.LayerStack, f: int) -> dict:
    """Camadas na tela no quadro ``f`` (índice -> Placement). Depois da animação: todas em repouso."""
    t = f / float(ch.fps)
    out: dict = {}
    for e in ch.entrances:
        out.update(_placements_for(ch, stack, e, t))
    return out


# ============================================================================================================
# Planejamento
# ============================================================================================================
def _unit_union(hard, unit):
    u = np.zeros_like(hard[0])
    for i in unit:
        u |= hard[i]
    return u


def _bbox(m):
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _edge_contact(union: np.ndarray) -> dict:
    return {"esquerda": int(union[:, :2].sum()), "direita": int(union[:, -2:].sum()),
            "topo": int(union[:2].sum()), "base": int(union[-2:].sum())}


class _Planner:
    def __init__(self, stack: L.LayerStack, hard, bands):
        self.stack, self.hard, self.bands = stack, hard, bands
        self.H, self.W = stack.photo_shape
        self.rest: dict = {}                         # camadas já pousadas -> Placement() de repouso

    def hidden(self, unit, q: L.Placement, clip: tuple | None = None) -> int:
        pl = dict(self.rest)
        for i in unit:
            mul = _clip_mul(clip[0], clip[1], self.stack.layers[i].box, q.dy) if clip and q.dy else None
            pl[i] = L.Placement(dx=q.dx, dy=q.dy, scale=q.scale, anchor=q.anchor, alpha_mul=mul)
        return sum(uncovered(self.stack, self.hard, self.bands, i, pl, clip) for i in unit)

    def path_ok(self, unit, direction, amp, clip: tuple | None = None) -> bool:
        a = int(math.ceil(amp))
        step = 1 if a <= 120 else 2
        for d in list(range(a, -1, -step)) + [0]:
            q = L.Placement(dx=d * direction[0], dy=d * direction[1])
            if self.hidden(unit, q, clip):
                return False
        return True

    def max_safe(self, unit, direction, cap) -> int:
        best = 0
        step = 2
        for d in range(step, int(cap) + 1, step):
            if self.hidden(unit, L.Placement(dx=d * direction[0], dy=d * direction[1])):
                break
            best = d
        return best

    def scale_ok(self, unit, s0, anchor) -> bool:
        for s in np.linspace(s0, 1.0, 7):
            if self.hidden(unit, L.Placement(scale=float(s), anchor=anchor)):
                return False
        return True

    def choose(self, unit, role, ch: Choreography) -> Entrance:
        st, hard = self.stack, self.hard
        H, W = self.H, self.W
        union = _unit_union(hard, unit)
        x0, y0, x1, y1 = _bbox(union)
        hU, wU = y1 - y0, x1 - x0
        labels = [st.layers[i].label for i in unit]
        occ = bool(len(self._external_cut(unit)))
        edges = {k: v for k, v in _edge_contact(union).items() if v > 0}
        small = all(lab in L.SMALL_LABELS for lab in labels)
        cands: list[tuple] = []

        def slide_from(edge):
            d = EDGE_DIR[edge]
            amp = {"esquerda": x1, "direita": W - x0, "base": H - y0, "topo": y1}[edge] + EXIT_MARGIN
            return ("deslizar", d, float(amp), f"encosta na borda ({edge}): desliza de fora do quadro", None)

        if all(lab in L.RUG_LABELS for lab in labels):
            axis = 0 if wU >= hU else 1
            from_end = ("direita" in edges) if axis == 0 else ("base" in edges and "topo" not in edges)
            return Entrance(list(unit), "desenrolar", role=role or "tapete", reason="tapete desenrola no lugar",
                            axis=axis, from_end=from_end, lo=float(x0 if axis == 0 else y0),
                            hi=float(x1 if axis == 0 else y1))
        # bordas laterais (e a de baixo) antes do topo: peça de borda lateral nunca vem de cima
        side = sorted([k for k in edges if k in ("esquerda", "direita", "base")], key=lambda k: -edges[k])
        wall = role == "parede" or all(lab in L.WALL_HUNG_LABELS for lab in labels)
        if wall and set(edges) == {"topo"}:
            # quadro cortado pelo topo da foto: deslizar de cima parecia "cair do teto". Cresce no lugar a partir
            # do ponto do topo (o corte fica fora do quadro) ou, se não der, desliza rente à parede pelo lado mais
            # perto; descer do topo é o último recurso.
            anchor = ((x0 + x1) / 2.0, 0.0)
            for s0 in WALL_SCALES:
                if self.scale_ok(unit, s0, anchor):
                    return Entrance(list(unit), "escala", scale0=s0, anchor=anchor, role=role,
                                    reason="quadro no topo da foto: cresce no lugar, preso ao topo")
            near = "esquerda" if x0 <= W - x1 else "direita"
            dist = x1 if near == "esquerda" else W - x0
            if dist <= WALL_SLIDE_MAX * W:
                d = EDGE_DIR[near]
                amp = float(dist + EXIT_MARGIN)
                if self.path_ok(unit, d, amp):
                    return Entrance(list(unit), "deslizar", d, amp, role=role,
                                    reason=f"quadro no topo da foto: desliza rente à parede ({near})")
        floor_role = role in ("base", "eletro", "louca", "bancada")
        for k in side:
            if not (floor_role and k == "base"):
                cands.append(slide_from(k))
        if "topo" in edges and not side:
            cands.append(slide_from("topo"))
        if role in ("base", "eletro", "louca") and "topo" not in edges:
            cands.append(("emergir", (0, 1), float(hU + EXIT_MARGIN), "planejado sobe do chão", "piso"))
        if role == "aereo" and "base" not in edges:
            cands.append(("descer", (0, -1), float(hU + EXIT_MARGIN), "aéreo desce do teto", "teto"))
        if not occ:
            if role in ("aereo", "parede") and not edges:
                cands.append(("descer", (0, -1), DESCEND * H, "desce rente à parede", None))
            elif role == "bancada" and not edges:
                cands.append(("pousar", (0, -1), LAND * H, "bancada pousa sobre a base", None))
            elif not edges and not role:
                lo, hi = DROP_SMALL if small else (DROP_MIN, DROP_MAX)
                amp = min(max(0.6 * hU, lo * H), hi * H)
                cands.append(("queda", (0, -1), float(amp), "peça isolada: queda curta com quique", None))
        r = band_radius((H, W))
        for style, d, amp, why, ck in cands:
            clip = (ck, clip_profile(union, ck, r)) if ck else None
            if self.path_ok(unit, d, amp, clip):
                return Entrance(list(unit), style, d, amp, role=role, reason=why, clip=ck,
                                profile=None if clip is None else clip[1])
        # coberta por outra (ou nada acima foi seguro): entra por trás, escala no contato, surge
        best = None
        for name in ("baixo", "esquerda", "direita", "cima"):
            if name == "cima" and role not in ("aereo", "parede"):
                continue
            d = DIRS[name]
            size = hU if name in ("baixo", "cima") else wU
            frame = H if name in ("baixo", "cima") else W
            cap = max(4, int(min(RISE_SIZE * size, RISE_CAP * frame)))
            m = self.max_safe(unit, d, cap)
            if m >= RISE_MIN * size:
                best = (name, d, m)
                break
        if best is not None:
            return Entrance(list(unit), "subir_atras", best[1], float(best[2]), role=role,
                            reason=f"coberta por outra: entra por trás ({best[0]}, {best[2]} px)")
        for anchor in self._anchors(unit, union, edges):
            for s0 in SCALES:
                if self.scale_ok(unit, s0, anchor):
                    return Entrance(list(unit), "escala", scale0=s0, anchor=anchor, role=role,
                                    reason="sem espaço para entrar por trás: escala no ponto de contato")
        if not small and role != "parede" and "topo" not in edges and not all(
                lab in L.WALL_HUNG_LABELS for lab in labels):
            # como as estantes/cômodas do v2: sobe do chão, recortada na linha do piso (sem salto, sem fade)
            clip = ("piso", clip_profile(union, "piso", band_radius((H, W))))
            if self.path_ok(unit, (0, 1), float(hU + EXIT_MARGIN), clip):
                return Entrance(list(unit), "emergir", (0, 1), float(hU + EXIT_MARGIN), role=role, clip="piso",
                                profile=clip[1], reason="sem espaço para entrar por trás: sobe do chão")
        return Entrance(list(unit), "surgir", role=role, reason="nenhum movimento seguro: surge no lugar")

    def _external_cut(self, unit) -> np.ndarray:
        """Pontos de corte da unidade cobertos por peças de FORA dela (dentro do quadro)."""
        H, W = self.H, self.W
        mem = set(unit)
        out = []
        for i in unit:
            pts = self.bands[i]
            if not len(pts):
                continue
            xi, yi = np.rint(pts[:, 0]).astype(int), np.rint(pts[:, 1]).astype(int)
            ok = (xi >= 0) & (yi >= 0) & (xi < W) & (yi < H)
            ext = np.zeros(len(pts), bool)
            for j in range(len(self.stack.layers)):
                if j not in mem and self.stack.front[j, i]:
                    ext[ok] |= self.hard[j][yi[ok], xi[ok]]
            out.append(pts[ext])
        return np.concatenate(out) if out else np.zeros((0, 2), np.float32)

    def _anchors(self, unit, union, edges):
        H, W = self.H, self.W
        x0, y0, x1, y1 = _bbox(union)
        pts = self._external_cut(unit)
        c = pts.mean(axis=0) if len(pts) else np.float32([(x0 + x1) / 2.0, y1 - 1])
        cx, cy = float(np.clip(c[0], 0, W - 1)), float(np.clip(c[1], 0, H - 1))
        out = [(cx, cy)]
        for k in edges:                               # na borda: o corte da borda não entra no quadro
            out.append({"esquerda": (0.0, cy), "direita": (W - 1.0, cy), "topo": (cx, 0.0),
                        "base": (cx, H - 1.0)}[k])
        if len(edges) >= 2:
            ex = 0.0 if "esquerda" in edges else (W - 1.0 if "direita" in edges else cx)
            ey = 0.0 if "topo" in edges else (H - 1.0 if "base" in edges else cy)
            out.append((ex, ey))
        out.append(((x0 + x1) / 2.0, float(y1 - 1)))
        return out


def _schedule(ents: list[Entrance], units_of: dict, stack, gap: float, dscale: float, small: dict):
    """Início de cada entrada: ``gap`` depois da anterior e nunca antes de quem a cobre (ou apoia) pousar."""
    t = HOLD_START
    done: dict = {}
    of = units_of
    for k, e in enumerate(ents):
        e.dur = DURATION[e.style] * dscale * (SMALL_DUR if small.get(k) else 1.0)
        start = t
        rf = getattr(stack, "rug_first", None) or {}
        for a in range(len(stack.layers)):
            for b in e.members:
                if not (stack.front[a, b] and of[a] in done and of[a] != of[b]):
                    continue
                if a in rf.get(b, ()):
                    continue
                start = max(start, done[of[a]].motion_end + 1e-3)
        for r_, clear in rf.items():
            # quem pousa no tapete só termina o movimento depois de o tapete estar todo desenrolado
            if of[r_] in done and of[r_] != of[e.members[0]] and clear & set(e.members):
                start = max(start, done[of[r_]].motion_end - e.dur * MOTION_END + 1e-3)
        for dep in e.members:
            j = e.after.get(dep)
            if j is not None and of[j] in done:
                start = max(start, done[of[j]].motion_end + 1e-3)
        e.start = round(start, 4)
        done[of[e.members[0]]] = e
        t = e.start + gap * (0.7 if small.get(k) else 1.0)
    anim_end = max((e.end for e in ents), default=HOLD_START)
    return anim_end


def _total(anim_end: float, fps: int) -> float:
    n_anim = int(math.ceil((anim_end + SETTLE_END) * fps))
    return (n_anim + DISSOLVE_FRAMES + int(math.ceil(HOLD_END * fps))) / fps


def _plan_units(stack, hard, bands, units, after, roles, ch) -> tuple[list[Entrance], int]:
    order = entrance_order(stack, hard, units, after, roles)
    pl = _Planner(stack, hard, bands)
    ents: list[Entrance] = []
    n_builtin = 0
    for u in order:
        unit = units[u]
        rs = [roles[i] for i in unit]
        role = max(rs, key=lambda r: BUILTIN_RANK.get(r, 0)) if any(rs) else ""
        n_builtin += sum(stack.layers[i].kind == "planejado" for i in unit)
        e = pl.choose(unit, role, ch)
        e.after = {i: after[i] for i in unit if i in after}
        ents.append(e)
        for i in unit:
            pl.rest[i] = L.Placement()
    return ents, n_builtin


def _plan_variant(stack, hard, bands, units, after, roles, ch):
    """Unidades acíclicas + estilos + fusão de "surge" com quem cobre, para uma relação de oclusão."""
    H, W = stack.photo_shape
    n = len(stack.layers)
    units, cycles = acyclic_units(stack, units, after)
    ents, n_builtin = _plan_units(stack, hard, bands, units, after, roles, ch)
    # Peça que só poderia surgir de uma vez: tenta entrar como bloco rígido junto com quem a cobre (manta + poltrona
    # deslizando da borda, cadeira de trás + cadeira da frente). Aceita se o bloco ganha um movimento de verdade.
    merged = 0
    for _ in range(MERGE_ROUNDS):
        pops = [e for e in ents if e.style == "surgir"]
        improved = False
        for e in sorted(pops, key=lambda e: -float(_unit_union(hard, e.members).sum())):   # as grandes primeiro
            mem = set(e.members)
            occ = {j for i in e.members for j in range(n) if j not in mem and stack.front[j, i]}
            if not occ:
                continue
            # Grupos candidatos: todas as que a cobrem e, se o bloco fica grande demais, só parte delas (as
            # menores primeiro, uma a uma e acumulando). O sofá coberto pela mesa lateral e pela poltrona da frente
            # entra com a mesa lateral e sobe do chão; a poltrona (grande) continua com a entrada dela.
            occ_units = sorted({tuple(u) for u in units if set(u) & occ},
                               key=lambda u: float(_unit_union(hard, list(u)).sum()))
            tries = [set(i for u in occ_units for i in u)]
            tries += [set(u) for u in occ_units]
            acc: set = set()
            for u in occ_units[:-1]:
                acc |= set(u)
                tries.append(set(acc))
            seen_g = []
            for extra in tries:
                grp = set(mem) | extra
                if grp in seen_g:
                    continue
                seen_g.append(grp)
                if len(grp) > MERGE_MAX_PIECES or \
                        float(_unit_union(hard, sorted(grp)).sum()) > MERGE_MAX_AREA * H * W:
                    continue
                new_units = [u for u in units if not (set(u) & grp)] + [sorted(grp)]
                new_units, _ = acyclic_units(stack, new_units, after)
                if not any(grp <= set(u) and len(u) <= MERGE_MAX_PIECES for u in new_units):
                    continue
                g2 = next(set(u) for u in new_units if grp <= set(u))
                cand, nb = _plan_units(stack, hard, bands, new_units, after, roles, ch)
                ce = next(c for c in cand if set(c.members) == g2)
                if ce.style != "surgir" and sum(c.style == "surgir" for c in cand) < len(pops):
                    units, ents, n_builtin = new_units, cand, nb
                    ce.reason += " (em bloco com quem a cobre)"
                    merged += 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return stack, bands, units, ents, n_builtin, merged, cycles


def plan(stack: L.LayerStack, fps: int = FPS, target: float | None = None, fabric: bool = True) -> Choreography:
    """Planeja a coreografia da pilha (ordem, estilo, amplitude, tempos) e mede o "pixel oculto visível".

    Depois do plano, a verificação independente (:func:`verify_independent`, que não usa o grafo) confere todos os
    quadros. Se ela acha um corte à vista (a peça entrou antes de quem a cobria na foto, porque o grafo não tinha
    a relação ou a tirou num ciclo), a relação que falta é posta no grafo e o plano é refeito (até
    :data:`INDEPENDENT_REPAIRS` vezes; um ciclo vira bloco rígido). O que ainda sobrar fica em ``ch.warnings`` e a
    verificação do vídeo (§3.5) recusa o MP4."""
    ch = _plan_once(stack, fps, target)
    if not ch.entrances:
        return ch
    # ``fabric``: tecido (roupa de cama, almofadas, cortina) assenta com um movimento pequeno no pouso
    # (:func:`fabric_regions`); o último quadro continua sendo a foto
    ind = verify_independent(stack, ch)
    best, best_px = ch, ind["oculto_independente_px"]
    front = stack.front.copy()
    fixes = 0
    for _ in range(INDEPENDENT_REPAIRS):
        if not best_px:
            break
        changed = False
        for d in ind["oculto_independente_pares"]:
            i, j = d["_i"], d["_j"]
            if not front[j, i]:
                front[j, i], front[i, j] = True, False
                changed = True
        if not changed:
            break
        fixes += 1
        cand = _plan_once(dataclasses.replace(stack, front=front.copy()), fps, target)
        ind = verify_independent(stack, cand)
        if ind["oculto_independente_px"] < best_px:
            best, best_px = cand, ind["oculto_independente_px"]
    best.stats["reparos_independentes"] = fixes
    if fabric:
        best.fabric = fabric_regions(stack, best)
    return best


def _plan_once(stack: L.LayerStack, fps: int = FPS, target: float | None = None) -> Choreography:
    H, W = stack.photo_shape
    ch = Choreography(fps, (H, W), [], HOLD_START, 0, 0)
    ch._soft = max(4.0, SOFT_FRAC * math.hypot(H, W))
    n = len(stack.layers)
    if n == 0:
        ch.n_anim = int(math.ceil(HOLD_START * fps))
        ch.n_frames = ch.n_anim + int(math.ceil(HOLD_END * fps))
        ch.warnings.append("nenhuma peça para entrar (foto vazia: usar o modo A)")
        return ch
    hard = _abs_hard(stack)
    bands = cut_bands(stack, hard)
    front0 = stack.front.copy()
    units, after, roles = build_units(stack, hard, front0)
    units0 = [list(u) for u in units]
    seated = not np.array_equal(front0, stack.front)
    if seated:
        stack = dataclasses.replace(stack, front=front0)      # cópia rasa: só a relação de oclusão muda
        bands = cut_bands(stack, hard)
    front, broken = break_cycles(stack, hard, bands, units, after)
    stack_b, bands_b = stack, bands
    if broken:
        stack_b = dataclasses.replace(stack, front=front)
        bands_b = cut_bands(stack_b, hard)
    stack, bands, units, ents, n_builtin, merged, cycles = _plan_variant(stack_b, hard, bands_b, units, after,
                                                                         roles, ch)
    if broken:
        # A aresta "mais fraca" do ciclo pode ser verdadeira (vaso do aparador na frente do braço do sofá que cobre
        # o aparador): tirá-la mostra o recorte do vaso no sofá. O bloco rígido é seguro nas duas leituras; fica
        # com ele quando não piora (não surge mais peça de uma vez).
        alt = _plan_variant(dataclasses.replace(stack, front=front0), hard, cut_bands(
            dataclasses.replace(stack, front=front0), hard), units0, after, roles, ch)
        pops = lambda es: sum(e.style == "surgir" for e in es)          # noqa: E731
        if pops(alt[3]) <= pops(ents):
            stack, bands, units, ents, n_builtin, merged, cycles = alt
            broken = 0
    units_of = {i: u for u, mem in enumerate(units) for i in mem}
    small = {k: all(stack.layers[i].label in L.SMALL_LABELS for i in e.members) for k, e in enumerate(ents)}
    # --- encaixa o total em 7–10 s (até 12 s com muitos planejados)
    n_u = len(ents)
    tmax = TOTAL_MAX_BUILTIN if (n_builtin >= 4 or n_u > 16) else TOTAL_MAX
    want = target if target is not None else min(tmax, max(TOTAL_MIN, 4.0 + 0.45 * n_u))
    best = None
    for dscale in (1.0, 0.9, 0.8, DUR_SCALE_MIN, 1.1, DUR_SCALE_MAX):
        for gap in np.arange(GAP_MIN, GAP_MAX + 1e-9, 0.02):
            end = _schedule(ents, units_of, stack, float(gap), dscale, small)
            tot = _total(end, fps)
            score = abs(tot - want) + (0.0 if TOTAL_MIN <= tot <= tmax else 100.0) + 0.3 * abs(dscale - 1.0)
            if best is None or score < best[0]:
                best = (score, float(gap), dscale)
    _, gap, dscale = best
    anim_end = _schedule(ents, units_of, stack, gap, dscale, small)
    # reparo: entrada que ainda mostra corte em algum quadro (amostragem do caminho, peças vizinhas em movimento)
    # vira "surge no lugar", que é seguro quando quem a cobre já pousou
    repaired = 0
    for _ in range(REPAIR_ROUNDS):
        ch.entrances, ch.anim_end = ents, anim_end
        ch.n_anim = int(math.ceil((anim_end + SETTLE_END) * fps))
        bad = _leaking(stack, ch, hard, bands)
        if not bad:
            break
        for k in bad:
            e = ents[k]
            ents[k] = Entrance(e.members, "surgir", role=e.role, after=e.after,
                               reason=f"{e.style} mostrava corte: surge no lugar")
            repaired += 1
        anim_end = _schedule(ents, units_of, stack, gap, dscale, small)
    ch.entrances = ents
    ch.anim_end = anim_end
    ch.n_anim = int(math.ceil((anim_end + SETTLE_END) * fps))
    hold = HOLD_END
    tot = _total(anim_end, fps)
    if tot < TOTAL_MIN:                     # poucas peças: estica o hold final com a foto exata
        hold += TOTAL_MIN - tot
    ch.n_frames = ch.n_anim + DISSOLVE_FRAMES + int(math.ceil(hold * fps))
    ch.stats = {"unidades": n_u, "blocos_fundidos": merged, "ciclos_quebrados": broken,
                "ciclos_em_bloco": cycles, "reparos": repaired, "planejados": n_builtin, "intervalo_s": round(gap, 3), "escala_duracao": dscale,
                "estilos": {s: sum(e.style == s for e in ents) for s in sorted({e.style for e in ents})}}
    if ch.duration > tmax + 1e-6:
        ch.warnings.append(f"coreografia com {ch.duration:.1f} s (acima de {tmax:.0f} s): muitas peças em sequência")
    ch._hard, ch._bands = hard, bands
    ch.front = stack.front
    return ch


def _leaking(stack, ch, hard, bands) -> list[int]:
    """Índices das entradas que mostram corte durante o próprio movimento."""
    bad = set()
    for f in range(ch.n_anim):
        t = f / float(ch.fps)
        moving = [k for k, e in enumerate(ch.entrances) if e.start <= t < e.motion_end and k not in bad]
        if not moving:
            continue
        pl = placements_at(ch, stack, f)
        for k in moving:
            e = ch.entrances[k]
            clip = (e.clip, e.profile) if e.clip else None
            if any(uncovered(stack, hard, bands, i, pl, clip) for i in e.members):
                bad.add(k)
    return sorted(bad)


# ============================================================================================================
# Verificação e quadros
# ============================================================================================================
def verify(stack: L.LayerStack, ch: Choreography, step: int = 1) -> dict:
    """Mede em todos os quadros (``step`` = 1): pixels ocultos visíveis, opacidade parcial de peça, peça de borda
    lateral/de baixo vindo de cima, e os tempos. Grava em ``ch.stats`` e devolve o dicionário."""
    hard = ch._hard if ch._hard is not None else _abs_hard(stack)
    bands = ch._bands if ch._bands is not None else cut_bands(stack, hard)
    edge_units = []
    for e in ch.entrances:
        u = _unit_union(hard, e.members)
        c = _edge_contact(u)
        if c["esquerda"] or c["direita"] or c["base"]:
            edge_units.append(e)
    owner = {i: e for e in ch.entrances for i in e.members}
    hidden_total, hidden_frames, worst = 0, 0, 0
    partial = 0
    from_above = 0
    cache: dict = {}
    for f in range(0, ch.n_anim, step):
        pl = placements_at(ch, stack, f)
        frame_hidden = 0
        for i in pl:
            key = (i,) + tuple((j, pl[j].dx, pl[j].dy, round(pl[j].scale, 5),
                                None if pl[j].alpha_mul is None else float(np.asarray(pl[j].alpha_mul).sum()))
                               for j in sorted(pl) if j >= i)
            if key not in cache:
                ei = owner[i]
                cache[key] = uncovered(stack, hard, bands, i, pl, (ei.clip, ei.profile) if ei.clip else None)
            frame_hidden += cache[key]
            q = pl[i]
            if q.alpha_mul is not None:
                m = np.asarray(q.alpha_mul)
                soft = (m > 0.05) & (m < 0.95)
                e = next(e for e in ch.entrances if i in e.members)
                if e.style != "desenrolar" and soft.any():
                    partial += 1
                elif soft.any():                    # só a frente da revelação, com largura <= borda suave
                    width = soft.any(axis=0).sum() if e.axis == 0 else soft.any(axis=1).sum()
                    if width > ch._soft + 2:
                        partial += 1
        for e in edge_units:
            if f / ch.fps >= e.start and any(pl[i].dy < 0 for i in e.members if i in pl):
                from_above += 1
        hidden_total += frame_hidden
        hidden_frames += frame_hidden > 0
        worst = max(worst, frame_hidden)
    res = {"oculto_visivel_px": int(hidden_total), "quadros_com_oculto": int(hidden_frames),
           "oculto_pior_quadro_px": int(worst), "opacidade_parcial": int(partial),
           "borda_vindo_de_cima": int(from_above), "duracao_s": round(ch.duration, 3),
           "vazio_inicial_s": round(ch.entrances[0].start, 3) if ch.entrances else round(ch.duration, 3),
           "hold_final_s": round((ch.n_frames - ch.n_anim - DISSOLVE_FRAMES) / ch.fps, 3)}
    ind = verify_independent(stack, ch, step)
    res["oculto_independente_px"] = ind["oculto_independente_px"]
    res["quadros_com_oculto_independente"] = ind["quadros_com_oculto_independente"]
    res["oculto_independente_pares"] = [{k: v for k, v in d.items() if not k.startswith("_")}
                                        for d in ind["oculto_independente_pares"][:10]]
    ch.stats.update(res)
    if hidden_total:
        ch.warnings.append(f"{hidden_total} pixels de parte oculta ficaram visíveis em {hidden_frames} quadros")
    if ind["oculto_independente_px"]:
        ch.warnings.append(f"verificação independente: {ind['oculto_independente_px']} pixels de corte à vista em "
                           f"{ind['quadros_com_oculto_independente']} quadros (peça entrando antes de quem a cobre)")
    return res


def _at_rest(q: L.Placement) -> bool:
    return q.dx == 0 and q.dy == 0 and q.scale >= 1.0 - 1e-6 and q.alpha_mul is None


def independent_front(stack: L.LayerStack, hard: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """Quem cobre quem para a verificação independente: a evidência clara gravada nas camadas
    (``stack.evidence``: tapete/quadro atrás, profundidade com folga de 5%, dono do contorno pela envoltória
    convexa), nunca o grafo ``stack.front`` que a coreografia usou. Pilha sem evidência (montada à mão): as
    mesmas pistas recalculadas das máscaras, sem profundidade."""
    ev = getattr(stack, "evidence", None)
    if ev is not None and ev.shape == (len(stack.layers),) * 2:
        return np.asarray(ev, bool)
    hard = _abs_hard(stack) if hard is None else hard
    labels = [l.label for l in stack.layers]
    return L.confident_front(L.occlusion_cues(labels, hard, None), len(stack.layers))


def verify_independent(stack: L.LayerStack, ch: Choreography, step: int = 1) -> dict:
    """Verificação INDEPENDENTE do "pixel oculto visível" (não usa o grafo de oclusão da coreografia).

    O corte de uma camada é a borda em que falta alpha porque outra peça a cobria na foto (evidência clara, ver
    :func:`independent_front`) ou porque a foto acaba ali. Em todo quadro, cada ponto de corte de uma peça na tela
    tem de estar coberto por uma peça JÁ POUSADA (em repouso) ou do mesmo bloco rígido; senão é pixel oculto
    visível. Assim um grafo sem a relação mesa × cadeira (a mesa entrava antes, com o buraco das cadeiras) é
    pego aqui mesmo com ``verify`` dizendo 0. Devolve as contagens e os pares (peça, quem a cobria)."""
    n = len(stack.layers)
    res = {"oculto_independente_px": 0, "quadros_com_oculto_independente": 0, "oculto_independente_pares": []}
    if n == 0 or not ch.entrances:
        return res
    hard = ch._hard if ch._hard is not None else _abs_hard(stack)
    ev = independent_front(stack, hard)
    st_ev = dataclasses.replace(stack, front=ev)
    bands = cut_bands(st_ev, hard)
    H, W = stack.photo_shape
    owner = {i: k for k, e in enumerate(ch.entrances) for i in e.members}
    total, frames = 0, 0
    pairs: dict = {}
    cache: dict = {}
    for f in range(0, ch.n_anim, step):
        pl = placements_at(ch, stack, f)
        rest = {j for j, q in pl.items() if _at_rest(q)}
        fh = 0
        for i in pl:
            if not len(bands[i]):
                continue
            ei = ch.entrances[owner[i]]
            cov = {j: pl[j] for j in pl if j == i or j in rest or owner.get(j) == owner[i]}
            key = (i,) + tuple((j, cov[j].dx, cov[j].dy, round(cov[j].scale, 5),
                                None if cov[j].alpha_mul is None else float(np.asarray(cov[j].alpha_mul).sum()))
                               for j in sorted(cov))
            if key not in cache:
                clip = (ei.clip, ei.profile) if ei.clip else None
                k = uncovered(st_ev, hard, bands, i, cov, clip)
                who = set()
                if k:
                    # quem cobria o corte na foto (para o reparo): donos dos pontos de corte em repouso
                    p = bands[i]
                    xi, yi = np.rint(p[:, 0]).astype(int), np.rint(p[:, 1]).astype(int)
                    ok = (xi >= 0) & (yi >= 0) & (xi < W) & (yi < H)
                    for j in range(n):
                        if ev[j, i] and hard[j][yi[ok], xi[ok]].any():
                            who.add(j)
                cache[key] = (k, who)
            k, who = cache[key]
            if k:
                fh += k
                for j in who:
                    pairs[(i, j)] = pairs.get((i, j), 0) + k
        total += fh
        frames += fh > 0
    res["oculto_independente_px"] = int(total)
    res["quadros_com_oculto_independente"] = int(frames)
    res["oculto_independente_pares"] = [
        {"peca": f"{stack.layers[i].label}#{stack.layers[i].id}", "coberta_por":
         f"{stack.layers[j].label}#{stack.layers[j].id}", "px": int(v), "_i": int(i), "_j": int(j)}
        for (i, j), v in sorted(pairs.items(), key=lambda kv: -kv[1])]
    return res


def frame_at(stack: L.LayerStack, ch: Choreography, photo: np.ndarray, f: int) -> np.ndarray:
    """Um quadro mestre isolado (mesmo resultado de :func:`render_frames` no índice ``f``)."""
    if f < ch.n_anim:
        return L.composite(stack, placements_at(ch, stack, f))
    k = f - ch.n_anim
    if k >= DISSOLVE_FRAMES:
        return photo.copy()
    landed = L.composite(stack, placements_at(ch, stack, ch.n_anim - 1)).astype(np.float32)
    u = (k + 1) / (DISSOLVE_FRAMES + 1)
    return np.clip(np.rint((1 - u) * landed + u * photo.astype(np.float32)), 0, 255).astype(np.uint8)


def render_frames(stack: L.LayerStack, ch: Choreography, photo: np.ndarray) -> Iterator[np.ndarray]:
    """Quadros mestres (uint8): sala vazia → entradas → pilha completa → dissolução de <= 4 quadros → foto exata.
    O quadro 0 é a placa corrigida; o último é ``photo`` byte a byte."""
    landed = None
    for f in range(ch.n_anim):
        fr = L.composite(stack, placements_at(ch, stack, f))
        if f == ch.n_anim - 1:
            landed = fr
        yield fr
    if landed is None:
        landed = L.composite(stack, range(len(stack.layers)))
    lf, pf = landed.astype(np.float32), photo.astype(np.float32)
    for k in range(DISSOLVE_FRAMES):
        u = (k + 1) / (DISSOLVE_FRAMES + 1)
        yield np.clip(np.rint((1 - u) * lf + u * pf), 0, 255).astype(np.uint8)
    for _ in range(ch.n_frames - ch.n_anim - DISSOLVE_FRAMES):
        yield photo.copy()
