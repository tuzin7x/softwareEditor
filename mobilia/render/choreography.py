"""Coreografia da montagem: tempos, curvas de easing e física de mola (Python puro).

Um único estilo, refinado: a câmera fica travada e cada peça surge e se monta em
sequência — tapete primeiro, depois as peças grandes de trás para frente, depois
luminárias e plantas, quadros e por fim os objetos pequenos sobre as superfícies.

Cada movimento é uma função ``pose(t)`` (t em segundos desde o início da peça)
que devolve deslocamentos relativos à pose final. O worker do Blender amostra
essas funções em todos os quadros (keyframes lineares por quadro), então o que
se vê no vídeo é exatamente o que está definido aqui.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ------------------------------------------------------------------ curvas


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out_cubic(x: float) -> float:
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in_out_cubic(x: float) -> float:
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_quint(x: float) -> float:
    x = clamp01(x)
    return 1 - (1 - x) ** 5


def smoothstep(a: float, b: float, x: float) -> float:
    t = clamp01((x - a) / (b - a)) if b != a else float(x >= b)
    return t * t * (3 - 2 * t)


def damped_spring(t: float, freq_hz: float = 3.2, zeta: float = 0.42) -> float:
    """Resposta de mola amortecida a um impulso: começa em 0, oscila e volta a 0."""
    if t <= 0:
        return 0.0
    w = 2 * math.pi * freq_hz
    wd = w * math.sqrt(max(1e-6, 1 - zeta * zeta))
    return math.exp(-zeta * w * t) * math.sin(wd * t)


# ------------------------------------------------------------------ pose relativa


@dataclass
class Pose:
    """Deslocamento relativo à pose final (no referencial da sala / do pai)."""

    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    rx: float = 0.0          # radianos
    ry: float = 0.0
    rz: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    sz: float = 1.0
    fade: float = 1.0        # 0 = invisível, 1 = opaco
    visible: bool = True

    def key(self, nd: int = 5) -> tuple:
        return tuple(round(v, nd) for v in (self.dx, self.dy, self.dz, self.rx, self.ry, self.rz,
                                            self.sx, self.sy, self.sz, self.fade)) + (self.visible,)


HIDDEN = Pose(fade=0.0, visible=False)
REST = Pose()


# ------------------------------------------------------------------ movimentos


def motion_drop(t: float, dur: float, height: float, size: float = 1.0, wobble: float = 1.0,
                fall: float | None = None) -> Pose:
    """Queda com aceleração (gravidade), toque no chão com leve squash e acomodação por mola.

    ``fall`` (s) fixa a duração da queda; sem ele a queda ocupa 55% de ``dur``."""
    if t < 0:
        return HIDDEN
    t_fall = min(dur, fall) if fall else dur * 0.55
    # fade curto: a peça fica opaca no primeiro terço da queda (evita o aspecto "fantasma")
    fade = smoothstep(0.0, t_fall * 0.3, t)
    if t < t_fall:
        x = t / t_fall
        dz = height * (1 - x * x)                    # queda acelerada (t^2)
        tilt = 0.035 * wobble * (1 - x)              # leve inclinação durante a queda
        return Pose(dz=dz, rx=tilt, ry=-0.6 * tilt, fade=fade)
    tl = t - t_fall
    # squash & stretch muito sutil (móvel real não é borracha) + acomodação
    sq = 0.022 * damped_spring(tl, 2.6, 0.55) / max(0.6, size)
    bounce = 0.012 * height * max(0.0, damped_spring(tl, 2.2, 0.6))
    rock = 0.012 * wobble * damped_spring(tl, 3.0, 0.5)
    return Pose(dz=bounce, rx=rock, ry=-0.5 * rock, sx=1 + sq * 0.5, sy=1 + sq * 0.5, sz=1 - sq, fade=1.0)


def motion_slide_down(t: float, dur: float, dist: float = 0.22) -> Pose:
    """Quadros: descem rente à parede até o prego e assentam."""
    if t < 0:
        return HIDDEN
    x = ease_out_cubic(t / dur)
    fade = smoothstep(0.0, dur * 0.35, t)
    settle = 0.006 * damped_spring(t - dur * 0.8, 2.5, 0.5)
    return Pose(dz=dist * (1 - x) + settle, fade=fade)


def motion_grow(t: float, dur: float, start_scale: float = 0.15) -> Pose:
    """Plantas: crescem a partir do vaso com leve ultrapassagem (ease-out-back)."""
    if t < 0:
        return HIDDEN
    x = clamp01(t / dur)
    c1 = 1.25
    back = 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2   # easeOutBack
    s = start_scale + (1 - start_scale) * back
    fade = smoothstep(0.0, dur * 0.3, t)
    sway = 0.02 * damped_spring(t - dur * 0.7, 1.4, 0.35)
    return Pose(sx=s, sy=s, sz=s, rx=sway, fade=fade)


def motion_rise(t: float, dur: float, depth: float = 0.06) -> Pose:
    """Peças internas (terra do vaso, colchão): sobem suavemente para o lugar."""
    if t < 0:
        return HIDDEN
    x = ease_out_quint(t / dur)
    return Pose(dz=-depth * (1 - x), fade=smoothstep(0.0, dur * 0.4, t))


def motion_spread(t: float, dur: float) -> Pose:
    """Fallback do tapete (quando o desenrolar procedural não está disponível)."""
    if t < 0:
        return HIDDEN
    x = ease_out_cubic(t / dur)
    return Pose(sy=0.02 + 0.98 * x, fade=smoothstep(0.0, dur * 0.25, t))


def motion_unfold(t: float, dur: float) -> Pose:
    """Cortinas: o painel "desce" a partir do varão (escala Z de 0 a 1 no pivô do topo)."""
    if t < 0:
        return HIDDEN
    x = ease_out_cubic(t / dur)
    sway = 0.012 * damped_spring(t - dur * 0.85, 1.6, 0.4)
    return Pose(sz=max(0.02, x), rx=sway, fade=smoothstep(0.0, dur * 0.2, t))


def motion_settle(t: float, dur: float, height: float = 0.06) -> Pose:
    """Tecidos (edredom, manta, almofadas): pousam de pouca altura e "assentam" com squash leve."""
    if t < 0:
        return HIDDEN
    t_fall = dur * 0.45
    fade = smoothstep(0.0, t_fall * 0.5, t)
    if t < t_fall:
        x = t / t_fall
        return Pose(dz=height * (1 - x * x), fade=fade)
    sq = 0.035 * damped_spring(t - t_fall, 2.0, 0.6)
    return Pose(sx=1 + sq * 0.4, sy=1 + sq * 0.4, sz=1 - sq, fade=1.0)


MOTIONS = {"drop": motion_drop, "slide": motion_slide_down, "grow": motion_grow, "rise": motion_rise,
           "spread": motion_spread, "unfold": motion_unfold, "settle": motion_settle}


# ------------------------------------------------------------------ linha do tempo


# Fim do "assentamento invisível": quando o deslocamento projetado de qualquer ponto da peça
# fica abaixo de SNAP_PX pixels (e o fade já está completo), a pose passa a ser EXATAMENTE a de
# repouso. Os quadros dessa cauda ficam iguais ao quadro final e são reaproveitados
# (``frame_signature``) em vez de renderizados de novo por uma diferença que ninguém vê.
SNAP_PX = 0.3
FADE_SNAP = 1.0 / 512          # 1 − fade abaixo de meio nível de 8 bits
SNAP_DT = 1.0 / 480            # passo da busca do instante de assentamento (s)
DEFAULT_PX_PER_M = 2000.0      # sem a câmera: escala conservadora (perto e de frente, 2560 px)
SETTLE_FACTOR = 2.2            # fim nominal do movimento = início + 2,2 × duração


def pose_residual_px(p: Pose, extent=(1.0, 1.0, 1.0), px_per_m: float = DEFAULT_PX_PER_M) -> float:
    """Limite superior do deslocamento (px) de qualquer ponto da peça entre ``p`` e o repouso.

    ``extent`` = (largura, profundidade, altura) em m do item (limita o braço de alavanca de
    giros e escalas, com o pivô em qualquer lugar da caixa); ``px_per_m`` = maior ampliação
    da projeção perto do item. Peça invisível ou ainda em fade nunca é "repouso" (inf)."""
    if not p.visible or 1.0 - p.fade > FADE_SNAP:
        return math.inf
    w, d, h = (abs(float(v)) for v in extent)
    diag = math.sqrt(w * w + d * d + h * h)
    m = (math.sqrt(p.dx * p.dx + p.dy * p.dy + p.dz * p.dz)
         + (abs(p.rx) + abs(p.ry) + abs(p.rz)) * diag
         + abs(p.sx - 1) * w + abs(p.sy - 1) * d + abs(p.sz - 1) * h)
    return m * px_per_m


@dataclass
class PartPlan:
    name: str
    motion: str
    start: float           # s, relativo ao início do vídeo
    duration: float
    params: dict = field(default_factory=dict)
    px_per_m: float | None = None              # ampliação da projeção perto do item (scene_spec)
    extent: tuple | None = None                # tamanho do item (m), para o resíduo de giro/escala
    _settle: float | None = field(default=None, init=False, repr=False, compare=False)

    def raw_pose(self, t: float) -> Pose:
        """Pose do movimento em t s desde o início da peça, sem o corte do assentamento."""
        if t < 0:
            return HIDDEN
        fn = MOTIONS[self.motion]
        if self.motion == "drop":
            return fn(t, self.duration, self.params.get("height", 0.45), self.params.get("size", 1.0),
                      self.params.get("wobble", 1.0), self.params.get("fall"))
        if self.motion == "slide":
            return fn(t, self.duration, self.params.get("dist", 0.22))
        if self.motion == "grow":
            return fn(t, self.duration, self.params.get("start_scale", 0.15))
        if self.motion == "rise":
            return fn(t, self.duration, self.params.get("depth", 0.06))
        if self.motion == "settle":
            return fn(t, self.duration, self.params.get("settle_height", 0.06))
        return fn(t, self.duration)

    @property
    def settle_time(self) -> float:
        """Instante (s desde o início da peça) a partir do qual a pose é exatamente o repouso.

        É o fim do último trecho com resíduo projetado ≥ ``SNAP_PX``; nunca passa de
        ``SETTLE_FACTOR`` × duração (o fim nominal, que continua definindo a linha do tempo)."""
        if self._settle is None:
            T = self.duration * SETTLE_FACTOR
            ext = tuple(self.extent) if self.extent else (1.0, 1.0, 1.0)
            ppm = float(self.px_per_m or DEFAULT_PX_PER_M)
            n = int(math.ceil(T / SNAP_DT))
            settle = 0.0
            for k in range(n, -1, -1):          # de trás para a frente: acha o último pico visível
                t = min(T, k * SNAP_DT)
                if pose_residual_px(self.raw_pose(t), ext, ppm) >= SNAP_PX:
                    settle = T if k == n else t + SNAP_DT
                    break
            self._settle = settle
        return self._settle

    def pose(self, time_s: float) -> Pose:
        t = time_s - self.start
        if t < 0:
            return HIDDEN
        T = self.duration * SETTLE_FACTOR
        if t > T or (self.settle_time < T and t >= self.settle_time):
            return REST                      # já acomodado: pose final exata
        return self.raw_pose(t)

    @property
    def end(self) -> float:
        return self.start + self.duration * SETTLE_FACTOR

    @property
    def settle_end(self) -> float:
        """Instante absoluto (s) em que a peça para de mudar na imagem."""
        return self.start + self.settle_time


# Duração (s) e movimento padrão por papel
ROLE_TIMING = {
    "rug": (1.5, "unroll"), "sofa": (1.15, "drop"), "bed": (1.2, "drop"), "armchair": (1.0, "drop"),
    "coffee_table": (0.95, "drop"), "dining_table": (1.0, "drop"), "desk": (1.0, "drop"),
    "tv_console": (1.0, "drop"), "dresser": (1.0, "drop"), "shelf": (1.0, "drop"),
    "side_table": (0.85, "drop"), "nightstand": (0.85, "drop"), "ottoman": (0.85, "drop"),
    "dining_chair": (0.8, "drop"), "bar_stool": (0.8, "drop"), "floor_lamp": (0.9, "drop"),
    "table_lamp": (0.7, "drop"), "plant_large": (1.1, "grow"), "plant_medium": (1.0, "grow"),
    "plant_small": (0.8, "grow"), "wall_art": (0.9, "slide"), "wall_mirror": (0.9, "slide"),
    "curtain": (1.0, "slide"), "pillows": (0.7, "drop"), "throw": (0.7, "drop"), "vase": (0.6, "drop"),
    "bowl": (0.6, "drop"), "books": (0.7, "drop"), "basket": (0.65, "drop"), "kitchen_decor": (0.6, "drop"),
}


# Decoração sobre tampo/assento: cai de pouco (12 cm) e depressa. Uma pilha ou um conjunto de
# livros entra em no máximo ON_TOP_WAVES levas, ON_TOP_STEP s uma da outra; assim nunca fica
# "meio no ar" por muito tempo (soma do tempo com alguma peça > 1 cm acima do repouso < 0,35 s).
ON_TOP_DROP = 0.12
ON_TOP_FALL = 0.13        # s de queda (um pouco mais rápida que a queda livre de 12 cm, ~0,16 s)
ON_TOP_STEP = 0.10        # s entre levas
ON_TOP_WAVES = 3


def drop_height(role: str, size_z: float, mount: str) -> float:
    if mount == "on_top":
        return ON_TOP_DROP
    if role in ("sofa", "bed", "tv_console", "dresser", "shelf", "desk", "dining_table"):
        return 0.48
    return max(0.3, min(0.55, 0.35 + 0.2 * size_z))


# No máximo MAX_MOVERS itens mudando na imagem ao mesmo tempo (do início até o assentamento
# invisível, ver ``PartPlan.settle_time``). Menos peças simultâneas = recorte dinâmico menor e
# uma montagem mais legível; o item só espera quando o limite seria passado.
MAX_MOVERS = 3
SMALL_ROLES = ("vase", "bowl", "books", "pillows", "throw", "kitchen_decor", "basket")


def item_extent(item: dict) -> tuple | None:
    sz = item.get("size")
    if not sz or len(sz) < 3 or max(abs(float(v)) for v in sz[:3]) <= 0:
        return None
    return tuple(float(v) for v in sz[:3])


def item_motion_window(item: dict) -> float:
    """Duração (s) em que o item muda na imagem: do início ao assentamento invisível da peça
    principal (mais as levas, para livros). Aproximação por item, sem conhecer as peças do
    modelo; o desenrolar do tapete dura exatamente ``duration``."""
    dur, role = float(item["duration"]), item["role"]
    motion = item.get("motion", "drop")
    if motion == "unroll":
        return dur
    size_z = (item.get("size") or [0, 0, 1])[2] or 1.0
    quick = item.get("mount", "floor") == "on_top" or role == "books"
    params = {"height": ON_TOP_DROP if quick else drop_height(role, size_z, item.get("mount", "floor")),
              "size": size_z, "dist": 0.22}
    if quick:
        params["fall"] = ON_TOP_FALL
    m = "spread" if motion == "spread" else motion if motion in MOTIONS else "drop"
    plan = PartPlan("_item", m, 0.0, dur, params, item.get("px_per_m"), item_extent(item))
    extra = (ON_TOP_WAVES - 1) * ON_TOP_STEP if role == "books" else 0.0
    return plan.settle_time + extra


def schedule_items(items: list[dict], hold_start: float = 0.8, gap_factor: float = 0.55,
                   min_gap: float = 0.32, max_movers: int | None = MAX_MOVERS) -> tuple[list[dict], float]:
    """Define o instante de início de cada item (ordenados por ``order``).

    O próximo item começa quando o anterior completou ``gap_factor`` da sua duração,
    gerando sobreposição fluida (sem "fila" mecânica). Decoração pequena entra mais rápido.
    Se já houver ``max_movers`` itens em movimento nesse instante, o item espera o primeiro
    deles assentar (só então a linha do tempo cresce, e só o necessário).
    Retorna (itens com 'start' e 'duration', fim da última animação).
    """
    t = hold_start
    out = []
    windows: list[tuple[float, float]] = []          # (início, fim do movimento visível)
    for it in sorted(items, key=lambda i: i["order"]):
        dur, motion = ROLE_TIMING.get(it["role"], (0.9, "drop"))
        it = dict(it, duration=dur, motion=it.get("motion", motion))
        if max_movers:
            while True:
                active = sorted(e for s0, e in windows if s0 <= t + 1e-9 and t + 1e-9 < e)
                if len(active) < max_movers:
                    break
                # espera o assentamento (arredondado para cima, para não sobrepor por 0,1 ms)
                t = math.ceil(active[len(active) - max_movers] * 1e4 - 1e-6) / 1e4
        it["start"] = round(t, 4)
        windows.append((it["start"], it["start"] + item_motion_window(it)))
        out.append(it)
        small = it["role"] in SMALL_ROLES
        t += max(min_gap * (0.75 if small else 1.0), dur * gap_factor * (0.7 if small else 1.0))
    end = max((i["start"] + i["duration"] * SETTLE_FACTOR for i in out), default=hold_start)
    return out, end


def max_concurrent(windows: list[tuple[float, float]]) -> int:
    """Maior número de janelas (início, fim) abertas ao mesmo tempo."""
    eps = 1e-9                                       # janelas que só se tocam não se sobrepõem
    ev = sorted([(s0, 1) for s0, _ in windows] + [(e - eps, -1) for _, e in windows],
                key=lambda x: (x[0], x[1]))
    n = best = 0
    for _, d in ev:
        n += d
        best = max(best, n)
    return best


# Ordem de montagem de peças de um mesmo asset (por palavra-chave no nome da malha)
PART_RANK = [
    (("base", "body", "frame", "pot", "legs", "leg", "stand", "pole", "bed_frame", "headboard"), 0),
    (("seat", "cushion", "mattress", "dirt", "pebbles", "soil", "top", "shade"), 1),
    (("door", "lid", "drawer", "duvet", "blanket"), 2),
    (("stem", "trunk"), 3),
    (("leaves", "leaf", "pillow", "book", "flower"), 4),
]


def part_rank(name: str) -> int:
    n = name.lower()
    for keys, rank in PART_RANK:
        if any(k in n for k in keys):
            return rank
    return 1


def plan_parts(item: dict, part_names: list[str], part_tags: dict | None = None) -> list[PartPlan]:
    """Planeja a montagem peça a peça de um item já agendado.

    ``part_tags`` (opcional) vem dos assets procedurais: {nome: (assembly_order, assembly_motion)}.
    Decoração "on_top" (e qualquer conjunto de livros) cai de ``ON_TOP_DROP`` em ``ON_TOP_FALL`` s,
    em no máximo ``ON_TOP_WAVES`` levas separadas por ``ON_TOP_STEP`` s.
    """
    start, dur, role = item["start"], item["duration"], item["role"]
    motion = item.get("motion", "drop")
    size_z = item.get("size", [0, 0, 1])[2] or 1.0
    on_top = item.get("mount", "floor") == "on_top"
    h = drop_height(role, size_z, item.get("mount", "floor"))
    quick = on_top or role == "books"
    if quick:
        h = ON_TOP_DROP
    ppm, ext = item.get("px_per_m"), item_extent(item)
    # assets procedurais trazem o movimento de cada peça (ex.: cortina = varão desliza,
    # painéis desdobram); eles têm prioridade sobre o movimento genérico do papel
    tagged = bool(part_tags) and motion != "unroll" and any(
        part_tags.get(n, (0, ""))[1] in MOTIONS for n in part_names)
    if not tagged and (motion in ("slide", "unroll", "spread") or len(part_names) <= 1):
        m = "spread" if motion == "unroll" else motion
        params = {"height": h, "size": size_z, "dist": 0.22}
        if quick:
            params["fall"] = ON_TOP_FALL
        return [PartPlan(n, m, start, dur, params, ppm, ext) for n in part_names]
    if part_tags:
        ranked = sorted(part_names, key=lambda n: (part_tags.get(n, (1, ""))[0], n))
        ranks = {n: part_tags.get(n, (1, ""))[0] for n in part_names}
    else:
        ranked = sorted(part_names, key=lambda n: (part_rank(n), n))
        ranks = {n: part_rank(n) for n in part_names}
    plans = []
    distinct = sorted(set(ranks.values()))
    step = ON_TOP_STEP if quick else min(0.32, dur * 0.42)
    is_plant = role.startswith("plant")
    seq_books = [n for n in ranked if "book" in n.lower()]
    for n in ranked:
        level = distinct.index(ranks[n])
        if quick:
            # conjuntos de livros (ex.: 20 enciclopédias em pé) entram em levas contíguas
            if n in seq_books and len(seq_books) > ON_TOP_WAVES:
                level = seq_books.index(n) * ON_TOP_WAVES // len(seq_books)
            level = min(level, ON_TOP_WAVES - 1)
        s = start + level * step
        if not quick and n in seq_books and len(seq_books) > 3:      # livros entram um a um, rápidos
            s = start + seq_books.index(n) * min(0.05, dur / max(1, len(seq_books)))
        tag_motion = part_tags.get(n, (0, ""))[1] if part_tags else ""
        if tag_motion in MOTIONS:
            m = tag_motion
        elif is_plant and ranks[n] >= 3:
            m = "grow"
        elif is_plant and ranks[n] == 1:
            m = "rise"
        else:
            m = "drop"
        d = dur if level == 0 else dur * 0.8
        params = {"height": h if level == 0 or quick else max(0.18, h * 0.6), "size": size_z,
                  "wobble": 1.0 if level == 0 else 0.6, "start_scale": 0.12, "depth": 0.05}
        if quick:
            params["fall"] = ON_TOP_FALL
        plans.append(PartPlan(n, m, round(s, 4), d, params, ppm, ext))
    return plans


def timeline_length(end_anim: float, hold_end: float = 2.5) -> float:
    return end_anim + hold_end
