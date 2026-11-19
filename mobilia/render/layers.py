"""Amostras do Cycles por camada de quadro (Python puro: usado pelo pipeline e pelo worker).

Camadas (as mesmas da telemetria do worker):
- ``final``: a cena completa em repouso (o *hold* do fim, o quadro que o espectador mais olha);
- ``marco``: quadro parado com só parte dos itens (pausa entre entradas);
- ``dinamico``: alguma peça em movimento (o desfoque de movimento esconde o ruído residual).

O quadro final nunca perde amostras. Os dinâmicos usam bem menos; a amostragem adaptativa
tem mínimo explícito (8 nos dinâmicos, 32 nos estáticos) e limiar 0,02. No modo de camada única
(``render.camadas = false``) cada quadro é a cena inteira: para o salto de ruído entre o último
quadro dinâmico e o quadro parado seguinte não "pular", os ``RAMP_FRAMES`` quadros dinâmicos que
vêm logo antes de um quadro parado sobem as amostras em progressão geométrica até as dele.

RENDER EM CAMADAS (PT6, ``plan_layers``): os quadros parados são **marcos** (render estático de
todas as peças já assentadas, com muitas amostras; o último é o quadro final). Um quadro com
movimento renderiza só a **camada dinâmica** — as peças ainda não incluídas no marco vigente —
com as peças do marco na cena como *shadow catcher* (escondem a peça nova atrás delas e recebem a
sombra dela). A composição põe a camada dinâmica sobre o marco já composto (ver
``compositor.LayerAccumulator``). Na troca de marco, ``CROSSFADE_FRAMES`` quadros misturam a
composição antiga (marco anterior + camada dele) com a nova, para o ruído não "pular".
"""
from __future__ import annotations

from dataclasses import dataclass, field

LAYERS = ("final", "marco", "dinamico")
ADAPTIVE_THRESHOLD = 0.02
ADAPTIVE_MIN = {"final": 32, "marco": 32, "dinamico": 8}
RAMP_FRAMES = 3


def samples_by_layer(final: int, preview: bool = False) -> dict:
    """Amostras por camada a partir das do quadro final (perfil de hardware/qualidade)."""
    final = max(1, int(final))
    if preview:
        return {"final": final, "marco": final, "dinamico": max(min(final, 12), final // 2)}
    return {"final": final, "marco": max(min(final, 32), final // 2),
            "dinamico": max(min(final, 16), final // 4)}


def clamp_layers(layers: dict | None, final: int) -> dict:
    """Normaliza o dicionário de camadas: nenhuma camada passa do quadro final."""
    final = max(1, int(final))
    base = samples_by_layer(final)
    out = {k: int(min(final, max(1, int((layers or {}).get(k, base[k]))))) for k in LAYERS}
    out["final"] = final
    return out


def adaptive_min(layer: str, samples: int) -> int:
    """``adaptive_min_samples`` explícito da camada (nunca acima das amostras do quadro)."""
    return int(min(int(samples), ADAPTIVE_MIN.get(layer, ADAPTIVE_MIN["dinamico"])))


def frame_samples(layers_seq: list[str], spl: dict, ramp: int = RAMP_FRAMES) -> list[int]:
    """Amostras de cada quadro RENDERIZADO, na ordem, a partir da camada de cada um.

    Os ``ramp`` quadros dinâmicos imediatamente antes de um quadro parado (final/marco) com
    mais amostras sobem em progressão geométrica: o j-ésimo antes dele recebe
    ``dyn · (alvo/dyn)^((ramp + 1 − j)/(ramp + 1))``."""
    out = [int(spl[layer]) for layer in layers_seq]
    if ramp <= 0:
        return out
    for i, layer in enumerate(layers_seq):
        if layer == "dinamico" or i == 0 or layers_seq[i - 1] != "dinamico":
            continue
        target = int(spl[layer])
        for j in range(1, ramp + 1):
            k = i - j
            if k < 0 or layers_seq[k] != "dinamico":
                break
            dyn = int(spl["dinamico"])
            if target <= dyn:
                break
            s = int(round(dyn * (target / dyn) ** ((ramp + 1 - j) / (ramp + 1))))
            out[k] = max(out[k], min(target, s))
    return out


# ============================================================================ render em camadas (PT6)
CROSSFADE_FRAMES = 3          # quadros de transição na troca de marco
DYNAMIC_MARGIN_M = 0.35       # margem (m) da camada dinâmica: sombra de contato em volta das peças


@dataclass(frozen=True)
class LayerPart:
    """Uma peça (objeto) da cena para o plano de camadas: ``plan`` tem ``pose(t)`` (``PartPlan``);
    ``unroll`` = (quadro inicial, final) do desenrolar do tapete, quando houver."""

    key: str
    item: str
    plan: object
    unroll: tuple | None = None


@dataclass
class Milestone:
    """Marco estático: render de ``parts`` (todas em repouso exato a partir de ``frame``)."""

    frame: int
    parts: frozenset
    final: bool = False
    itens: list = field(default_factory=list)
    implicit: bool = False        # pausa sem movimento fora das classes da coreografia

    @property
    def id(self) -> str:
        return f"m_{self.frame:05d}"

    @property
    def source(self) -> str:
        return self.id + ".exr"

    @property
    def layer(self) -> str:
        return "final" if self.final else "marco"


@dataclass
class LayerFrame:
    """Plano de um quadro do vídeo no render em camadas.

    ``kind``: "foto" (nada visível), "estatico" (o quadro é o marco ``base``) ou "dinamico" (marco
    ``base`` — None = a foto — mais a camada ``layer``). ``sig`` identifica o render da camada
    (base + poses das peças em f−1, f, f+1): quadros com a mesma ``sig`` reaproveitam o EXR.
    ``trans`` = (marco antigo ou None, camada antiga, assinatura antiga, peso da composição antiga)
    nos quadros de transição depois de uma troca de marco."""

    frame: int
    kind: str
    base: str | None = None
    layer: frozenset = frozenset()
    sig: tuple | None = None
    trans: tuple | None = None


@dataclass
class LayerPlan:
    milestones: list
    frames: list
    settle: dict = field(default_factory=dict)     # peça -> quadro em que assenta de vez

    def milestone(self, mid: str | None) -> Milestone | None:
        for m in self.milestones:
            if m.id == mid:
                return m
        return None


def _rest_key():
    from .choreography import REST
    return REST.key(4)


def _poses(part: LayerPart, f: int, fps: int) -> list:
    return [part.plan.pose((f - 1 + df) / fps) for df in (-1, 0, 1)]


def _changing(part: LayerPart, poses: list, f: int, rest_key) -> bool:
    """Mesma regra do ``dynamic_parts`` do worker: alguma pose de f−1..f+1 fora do repouso
    exato (movimento, fade, visibilidade) ou o tapete desenrolando."""
    ur = part.unroll
    if ur is not None and ur[0] - 1 <= f <= ur[1] + 1:
        return True
    return any(p.key(4) != rest_key for p in poses)


def part_sig(part: LayerPart, f: int, fps: int) -> tuple:
    """Assinatura da peça no quadro f (poses em f−1, f, f+1 e estado do desenrolar)."""
    sig = [part.key] + [p.key(4) for p in _poses(part, f, fps)]
    if part.unroll:
        a, b = part.unroll
        sig.append(("unroll", min(max(f, a - 1), b + 1)))
    return tuple(sig)


def settle_frames(parts: list[LayerPart], fps: int, n_frames: int) -> dict:
    """Primeiro quadro a partir do qual cada peça fica visível e em repouso exato até o fim
    (n_frames + 1 se ela nunca assenta dentro do vídeo)."""
    rk = _rest_key()
    out = {}
    for p in parts:
        s = 1
        for f in range(n_frames, 0, -1):
            poses = _poses(p, f, fps)
            if not poses[1].visible or _changing(p, poses, f, rk):
                s = f + 1
                break
        out[p.key] = s
    return out


def plan_layers(parts: list[LayerPart], fps: int, n_frames: int, groups: list[list[str]] | None = None,
                crossfade: int = CROSSFADE_FRAMES) -> LayerPlan:
    """Plano do render em camadas (Python puro; o worker só o executa).

    ``groups``: itens de cada marco da coreografia (``choreography.milestones``); cada um vira o
    primeiro quadro em que todas as peças desses itens estão em repouso, com TODAS as peças
    assentadas até ali. O último marco (todas as peças assentadas) é o quadro final. Toda pausa
    sem movimento cujo conjunto de peças paradas não é um marco vira um marco implícito (nada
    parado é renderizado com as amostras baixas da camada dinâmica por mais de um instante).

    Em cada quadro com movimento, a base é o marco mais recente já assentado e a camada dinâmica
    tem as peças visíveis (ou que aparecem/somem em f±1) fora da base."""
    by_key = {p.key: p for p in parts}
    settle = settle_frames(parts, fps, n_frames)
    item_parts: dict[str, list[str]] = {}
    for p in parts:
        item_parts.setdefault(p.item, []).append(p.key)

    def settled_by(F: int) -> frozenset:
        return frozenset(k for k, s in settle.items() if s <= F)

    ms: dict[frozenset, Milestone] = {}

    def add(F: int, final: bool = False, implicit: bool = False) -> Milestone | None:
        B = settled_by(F)
        if not B:
            return None
        m = ms.get(B)
        if m is None or F < m.frame:
            items = sorted({by_key[k].item for k in B})
            m2 = Milestone(F, B, final, items, implicit)
            if m is not None:
                m2.final = m2.final or m.final
            ms[B] = m2
            return m2
        m.final = m.final or final
        return m

    for g in groups or []:
        ks = [k for it in g for k in item_parts.get(str(it), [])]
        if ks:
            F = max(settle[k] for k in ks)
            if F <= n_frames:
                add(F)
    finite = [s for s in settle.values() if s <= n_frames]
    if finite:
        add(max(finite), final=len(finite) == len(settle))

    frames: list[LayerFrame] = []
    prev_base: str | None = None
    prev_kind = "foto"
    trans_left = 0
    old_base: Milestone | None = None
    for f in range(1, n_frames + 1):
        poses = {k: _poses(p, f, fps) for k, p in by_key.items()}
        vis = {k for k, ps in poses.items() if ps[1].visible}
        if not vis:
            frames.append(LayerFrame(f, "foto"))
            prev_kind, trans_left = "foto", 0
            continue
        pres = {k for k, ps in poses.items() if any(q.visible for q in ps)}
        moving = {k for k in pres if f < settle[k]}
        settled = {k for k in vis if settle[k] <= f}
        cands = [m for m in ms.values() if m.frame <= f]
        base = max(cands, key=lambda m: m.frame) if cands else None
        layer = (moving | settled) - (base.parts if base else frozenset())
        if not moving and layer:
            # pausa com peças paradas fora do marco vigente: marco implícito neste quadro
            base = add(f, implicit=True)
            layer = frozenset()
        layer = frozenset(layer)
        bid = base.id if base else None
        if not layer:
            fr = LayerFrame(f, "estatico", bid, layer, ("estatico", bid))
        else:
            sig = (bid,) + tuple(part_sig(by_key[k], f, fps) for k in sorted(layer))
            fr = LayerFrame(f, "dinamico", bid, layer, sig)
        # transição: a composição antiga (marco anterior + a camada dele) some em ``crossfade`` quadros
        if prev_kind != "foto" and bid != prev_base:
            old_base = ms_by_id(ms, prev_base)
            trans_left = max(0, int(crossfade))
        if trans_left > 0 and prev_kind != "foto":
            j = crossfade - trans_left
            w_old = 1.0 - (j + 1) / (crossfade + 1)
            ob = old_base.parts if old_base is not None else frozenset()
            old_layer = frozenset((moving | settled) - ob)
            if old_layer and w_old > 0:
                osig = (old_base.id if old_base else None,) + tuple(
                    part_sig(by_key[k], f, fps) for k in sorted(old_layer))
                fr.trans = (old_base.id if old_base else None, old_layer, osig, round(w_old, 6))
            trans_left -= 1
        frames.append(fr)
        prev_base, prev_kind = bid, fr.kind
    used = {fr.base for fr in frames} | {fr.trans[0] for fr in frames if fr.trans}
    mlist = sorted((m for m in ms.values() if m.id in used or m.final), key=lambda m: m.frame)
    return LayerPlan(mlist, frames, settle)


def dynamic_jobs(plan: LayerPlan, only: set | None = None) -> list[tuple]:
    """Renders da camada dinâmica que o worker vai fazer, na ordem: (quadro, peças, assinatura,
    marco base). Mesma regra de reaproveitamento do ``render_layered``: uma assinatura igual a um
    render do quadro anterior (ou já feito neste quadro) não renderiza de novo; um quadro só-foto
    zera o histórico. Inclui as camadas antigas das transições. Usado na previsão do PT7."""
    out: list[tuple] = []
    prev: set = set()
    for fr in plan.frames:
        if only and fr.frame not in only:
            continue
        if fr.kind == "foto":
            prev = set()
            continue
        cur: set = set()
        reqs = []
        if fr.kind == "dinamico":
            reqs.append((fr.layer, fr.sig, fr.base))
        if fr.trans is not None:
            reqs.append((fr.trans[1], fr.trans[2], fr.trans[0]))
        for keys, sig, base in reqs:
            if sig not in prev and sig not in cur:
                out.append((fr.frame, keys, sig, base))
            cur.add(sig)
        prev = cur
    return out


def ms_by_id(ms: dict, mid: str | None) -> Milestone | None:
    for m in ms.values():
        if m.id == mid:
            return m
    return None


def frame_counts(frames: list[dict], rendered: int) -> dict:
    """Contagem dos quadros do vídeo para a telemetria. ``renderizados`` = chamadas de render (no
    render em camadas: marcos + camadas dinâmicas + camadas da transição); ``quadros_novos`` =
    quadros com composição própria, então ``total = quadros_novos + reutilizados + so_foto``."""
    reused = sum(1 for fr in frames if fr.get("reused"))
    photo = sum(1 for fr in frames if fr.get("source") is None)
    return {"total": len(frames), "renderizados": int(rendered), "reutilizados": reused, "so_foto": photo,
            "quadros_novos": len(frames) - reused - photo}
