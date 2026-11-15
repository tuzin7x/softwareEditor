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


def motion_drop(t: float, dur: float, height: float, size: float = 1.0, wobble: float = 1.0) -> Pose:
    """Queda com aceleração (gravidade), toque no chão com leve squash e acomodação por mola."""
    if t < 0:
        return HIDDEN
    t_fall = dur * 0.55
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


MOTIONS = {"drop": motion_drop, "slide": motion_slide_down, "grow": motion_grow, "rise": motion_rise,
           "spread": motion_spread}


# ------------------------------------------------------------------ linha do tempo


@dataclass
class PartPlan:
    name: str
    motion: str
    start: float           # s, relativo ao início do vídeo
    duration: float
    params: dict = field(default_factory=dict)

    def pose(self, time_s: float) -> Pose:
        t = time_s - self.start
        if t < 0:
            return HIDDEN
        if t > self.duration * 2.2:          # já acomodado: pose final exata
            return REST
        fn = MOTIONS[self.motion]
        if self.motion == "drop":
            return fn(t, self.duration, self.params.get("height", 0.45), self.params.get("size", 1.0),
                      self.params.get("wobble", 1.0))
        if self.motion == "slide":
            return fn(t, self.duration, self.params.get("dist", 0.22))
        if self.motion == "grow":
            return fn(t, self.duration, self.params.get("start_scale", 0.15))
        if self.motion == "rise":
            return fn(t, self.duration, self.params.get("depth", 0.06))
        return fn(t, self.duration)

    @property
    def end(self) -> float:
        return self.start + self.duration * 2.2


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


def drop_height(role: str, size_z: float, mount: str) -> float:
    if mount == "on_top":
        return 0.22
    if role in ("sofa", "bed", "tv_console", "dresser", "shelf", "desk", "dining_table"):
        return 0.48
    return max(0.3, min(0.55, 0.35 + 0.2 * size_z))


def schedule_items(items: list[dict], hold_start: float = 0.8, gap_factor: float = 0.55,
                   min_gap: float = 0.32) -> tuple[list[dict], float]:
    """Define o instante de início de cada item (ordenados por ``order``).

    O próximo item começa quando o anterior completou ``gap_factor`` da sua duração,
    gerando sobreposição fluida (sem "fila" mecânica). Decoração pequena entra mais rápido.
    Retorna (itens com 'start' e 'duration', fim da última animação).
    """
    t = hold_start
    out = []
    for it in sorted(items, key=lambda i: i["order"]):
        dur, motion = ROLE_TIMING.get(it["role"], (0.9, "drop"))
        it = dict(it, start=round(t, 4), duration=dur, motion=it.get("motion", motion))
        out.append(it)
        small = it["role"] in ("vase", "bowl", "books", "pillows", "throw", "kitchen_decor", "basket")
        t += max(min_gap * (0.75 if small else 1.0), dur * gap_factor * (0.7 if small else 1.0))
    end = max((i["start"] + i["duration"] * 2.2 for i in out), default=hold_start)
    return out, end


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
    """
    start, dur, role = item["start"], item["duration"], item["role"]
    motion = item.get("motion", "drop")
    size_z = item.get("size", [0, 0, 1])[2] or 1.0
    h = drop_height(role, size_z, item.get("mount", "floor"))
    if motion in ("slide", "unroll", "spread") or len(part_names) <= 1:
        m = "spread" if motion == "unroll" else motion
        params = {"height": h, "size": size_z, "dist": 0.22}
        return [PartPlan(n, m, start, dur, params) for n in part_names]
    if part_tags:
        ranked = sorted(part_names, key=lambda n: (part_tags.get(n, (1, ""))[0], n))
        ranks = {n: part_tags.get(n, (1, ""))[0] for n in part_names}
    else:
        ranked = sorted(part_names, key=lambda n: (part_rank(n), n))
        ranks = {n: part_rank(n) for n in part_names}
    plans = []
    distinct = sorted(set(ranks.values()))
    step = min(0.32, dur * 0.42)
    is_plant = role.startswith("plant")
    seq_books = [n for n in ranked if "book" in n.lower()]
    for n in ranked:
        level = distinct.index(ranks[n])
        s = start + level * step
        if n in seq_books and len(seq_books) > 3:      # livros entram um a um, rápidos
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
        params = {"height": h if level == 0 else max(0.18, h * 0.6), "size": size_z,
                  "wobble": 1.0 if level == 0 else 0.6, "start_scale": 0.12, "depth": 0.05}
        plans.append(PartPlan(n, m, round(s, 4), d, params))
    return plans


def timeline_length(end_anim: float, hold_end: float = 2.5) -> float:
    return end_anim + hold_end
