"""Compositor de quadros do modo B (PT-V3-7): as camadas 2D (:mod:`layers2d.layers`) sobre a placa, quadro a quadro.

Faz a MESMA conta de :func:`layers2d.layers.composite` (placa → sombreamento → RGBA pré-multiplicado na ordem de
pintura → delta de contato, em float32, arredondado com ``rint``), mas só dentro das caixas das camadas presentes:

- as caixas de destino (com deslocamento/escala) e as caixas de repouso do delta são agrupadas em **regiões
  disjuntas** (caixas que se tocam viram uma região); cada região é convertida para float, composta e devolvida
  para uint8. Fora delas o quadro é a placa byte a byte, sem conta nenhuma (``rint(float(x)) == x``);
- backend ``numpy`` (padrão sem GPU): resultado idêntico, bit a bit, ao de ``layers.composite``;
- backend ``torch`` (CUDA quando há; também roda em CPU para teste): as camadas sobem UMA vez para a GPU, cada
  quadro faz as mesmas operações e só o quadro uint8 desce. Diferença para o numpy ≤ 1 nível (aceite PT-V3-7);
  na prática 0, porque as operações são as mesmas em float32.

``master_frames`` reproduz :func:`layers2d.choreo.render_frames` (sala vazia → entradas → dissolução de ≤ 4 quadros
→ foto exata) com o compositor escolhido, reaproveitando o quadro anterior quando nada mudou.
"""
from __future__ import annotations

import math
import os
from typing import Iterator, Mapping

import numpy as np

from . import layers as L

# "numpy" | "torch" | "cuda" | "auto" (padrão): força o backend do compositor
ENV_BACKEND = "MOBILIA_COMPOSITOR_2D"


def _torch():
    try:
        import torch

        return torch
    except Exception:          # torch ausente ou quebrado: fica no numpy
        return None


def choose_backend(pref: str | None = None) -> tuple[str, str]:
    """(backend, device): ``("torch", "cuda")`` quando há CUDA, senão ``("numpy", "cpu")``. ``pref`` (ou a
    variável ``MOBILIA_COMPOSITOR_2D``) força: ``numpy``, ``torch`` (torch na CPU, para teste) ou ``cuda``."""
    p = (pref or os.environ.get(ENV_BACKEND, "") or "auto").strip().lower()
    t = _torch()
    if p == "numpy" or t is None:
        return "numpy", "cpu"
    cuda = bool(t.cuda.is_available())
    if p == "torch":
        return "torch", "cuda" if cuda else "cpu"
    if p == "cuda":
        if not cuda:
            raise RuntimeError("CUDA pedida para o compositor 2D, mas não há GPU CUDA disponível")
        return "torch", "cuda"
    return ("torch", "cuda") if cuda else ("numpy", "cpu")


# ============================================================================================================
# Caixas
# ============================================================================================================
def _clip_box(b, H, W):
    x0, y0, x1, y1 = b
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def merge_boxes(boxes) -> list[tuple[int, int, int, int]]:
    """Caixas (x0, y0, x1, y1) → regiões disjuntas que as cobrem (caixas que se sobrepõem viram a caixa da união,
    até nenhuma região tocar outra)."""
    out = [tuple(b) for b in boxes if b is not None]
    changed = True
    while changed and len(out) > 1:
        changed = False
        res: list = []
        for b in out:
            for k, r in enumerate(res):
                if _overlap(b, r):
                    res[k] = (min(b[0], r[0]), min(b[1], r[1]), max(b[2], r[2]), max(b[3], r[3]))
                    changed = True
                    break
            else:
                res.append(b)
        out = res
    return out


def _sub(box, reg):
    """Fatias de ``box`` (absoluta) recortada a ``reg``: (fatias na região, fatias dentro da caixa) ou None."""
    x0, y0, x1, y1 = box
    cx0, cy0, cx1, cy1 = max(x0, reg[0]), max(y0, reg[1]), min(x1, reg[2]), min(y1, reg[3])
    if cx0 >= cx1 or cy0 >= cy1:
        return None
    return ((slice(cy0 - reg[1], cy1 - reg[1]), slice(cx0 - reg[0], cx1 - reg[0])),
            (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0)))


# ============================================================================================================
# Compositor
# ============================================================================================================
class FrameCompositor:
    """Compõe quadros de um :class:`layers.LayerStack` a partir das ``Placement`` de cada camada presente."""

    def __init__(self, stack: L.LayerStack, backend: str | None = None):
        self.stack = stack
        self.H, self.W = stack.photo_shape
        self.backend, self.device = choose_backend(backend)
        self.regions_last: list = []
        if self.backend == "torch":
            t = _torch()
            self._t = t
            dev = t.device(self.device)
            self._dev = dev
            self._plate = t.from_numpy(np.ascontiguousarray(stack.plate)).to(dev)
            self._lay = []
            for lay in stack.layers:
                up = lambda x: None if x is None else t.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev)
                self._lay.append({"a": up(lay.alpha), "c": up(lay.color), "sh": up(lay.shade), "d": up(lay.delta)})
            self._scaled_cache: dict = {}

    # ---------------------------------------------------------------------------------------------- comum
    def _view_np(self, i: int, q: L.Placement):
        return L._scaled(self.stack.layers[i], q)

    def _regions(self, pl: Mapping[int, L.Placement], boxes: Mapping[int, tuple]) -> list:
        H, W = self.H, self.W
        bx = []
        for i, q in pl.items():
            b = boxes[i]
            bx.append(_clip_box((b[0] + q.dx, b[1] + q.dy, b[2] + q.dx, b[3] + q.dy), H, W))
            if self.stack.layers[i].delta is not None and q.contact > 0:
                bx.append(_clip_box(self.stack.layers[i].box, H, W))
        return merge_boxes(bx)

    def compose(self, present: Mapping[int, L.Placement]) -> np.ndarray:
        """Quadro uint8 (H, W, 3). Sem camadas: cópia da placa corrigida."""
        pl = dict(present)
        if not pl:
            self.regions_last = []
            return self.stack.plate.copy()
        if self.backend == "torch":
            return self._compose_torch(pl)
        return self._compose_np(pl)

    # ---------------------------------------------------------------------------------------------- numpy
    def _compose_np(self, pl) -> np.ndarray:
        st = self.stack
        views = {i: self._view_np(i, pl[i]) for i in pl}
        regs = self._regions(pl, {i: views[i][3] for i in pl})
        self.regions_last = regs
        out = st.plate.copy()
        order = sorted(pl)
        for reg in regs:
            rx0, ry0, rx1, ry1 = reg
            buf = out[ry0:ry1, rx0:rx1].astype(np.float32)
            for i in order:
                q = pl[i]
                shade, box = views[i][2], views[i][3]
                if shade is None or q.shade <= 0:
                    continue
                s = _sub((box[0] + q.dx, box[1] + q.dy, box[2] + q.dx, box[3] + q.dy), reg)
                if s is None:
                    continue
                dst, src = s
                sh = shade[src]
                buf[dst] *= sh if q.shade >= 1 else 1.0 - q.shade * (1.0 - sh)
            for i in order:
                q = pl[i]
                box = views[i][3]
                s = _sub((box[0] + q.dx, box[1] + q.dy, box[2] + q.dx, box[3] + q.dy), reg)
                if s is None:
                    continue
                dst, src = s
                a, c = views[i][0][src], views[i][1][src]
                buf[dst] = c + (1.0 - a[..., None]) * buf[dst]
            for i in order:
                lay, q = st.layers[i], pl[i]
                if lay.delta is None or q.contact <= 0:
                    continue
                s = _sub(lay.box, reg)
                if s is None:
                    continue
                dst, src = s
                d = lay.delta[src] * min(1.0, q.contact)
                if q.alpha_mul is not None:
                    d = d * np.asarray(q.alpha_mul, np.float32)[src][..., None]
                buf[dst] += d
            out[ry0:ry1, rx0:rx1] = np.clip(np.rint(buf), 0, 255).astype(np.uint8)
        return out

    # ---------------------------------------------------------------------------------------------- torch
    def _view_t(self, i: int, q: L.Placement):
        """(alpha, cor, sombreamento, caixa) no device. Escala < 1 (raro: só a entrada "escala") passa pelo
        ``warpAffine`` da CPU, igual ao numpy, e sobe pronta (cache por escala)."""
        t = self._t
        lay, g = self.stack.layers[i], self._lay[i]
        if q.warp is not None:            # tecido assentando: deformação na CPU (igual ao numpy), sobe pronta
            a, c, sh, box = L._scaled(lay, q)
            up = lambda x: None if x is None else t.from_numpy(np.ascontiguousarray(x, np.float32)).to(self._dev)
            return up(a), up(c), up(sh), box
        if q.scale < 1.0 - 1e-6:
            key = (i, round(float(q.scale), 6), q.anchor,
                   None if q.alpha_mul is None else hash(np.asarray(q.alpha_mul, np.float32).tobytes()))
            v = self._scaled_cache.get(key)
            if v is None:
                a, c, sh, box = L._scaled(lay, q)
                up = lambda x: None if x is None else t.from_numpy(np.ascontiguousarray(x, np.float32)).to(self._dev)
                v = (up(a), up(c), up(sh), box)
                if len(self._scaled_cache) > 64:
                    self._scaled_cache.clear()
                self._scaled_cache[key] = v
            return v
        a, c = g["a"], g["c"]
        if q.alpha_mul is not None:
            m = t.from_numpy(np.ascontiguousarray(q.alpha_mul, np.float32)).to(self._dev)
            a, c = a * m, c * m[..., None]
        return a, c, g["sh"], lay.box

    def _compose_torch(self, pl) -> np.ndarray:
        t = self._t
        st = self.stack
        views = {i: self._view_t(i, pl[i]) for i in pl}
        regs = self._regions(pl, {i: views[i][3] for i in pl})
        self.regions_last = regs
        out = self._plate.clone()
        order = sorted(pl)
        mul_dev: dict = {}
        for reg in regs:
            rx0, ry0, rx1, ry1 = reg
            buf = out[ry0:ry1, rx0:rx1].to(t.float32)
            for i in order:
                q = pl[i]
                shade, box = views[i][2], views[i][3]
                if shade is None or q.shade <= 0:
                    continue
                s = _sub((box[0] + q.dx, box[1] + q.dy, box[2] + q.dx, box[3] + q.dy), reg)
                if s is None:
                    continue
                dst, src = s
                sh = shade[src]
                buf[dst] *= sh if q.shade >= 1 else 1.0 - float(np.float32(q.shade)) * (1.0 - sh)
            for i in order:
                q = pl[i]
                box = views[i][3]
                s = _sub((box[0] + q.dx, box[1] + q.dy, box[2] + q.dx, box[3] + q.dy), reg)
                if s is None:
                    continue
                dst, src = s
                a, c = views[i][0][src], views[i][1][src]
                buf[dst] = c + (1.0 - a[..., None]) * buf[dst]
            for i in order:
                lay, q = st.layers[i], pl[i]
                if lay.delta is None or q.contact <= 0:
                    continue
                s = _sub(lay.box, reg)
                if s is None:
                    continue
                dst, src = s
                d = self._lay[i]["d"][src] * float(np.float32(min(1.0, q.contact)))
                if q.alpha_mul is not None:
                    if i not in mul_dev:
                        mul_dev[i] = t.from_numpy(np.ascontiguousarray(q.alpha_mul, np.float32)).to(self._dev)
                    d = d * mul_dev[i][src][..., None]
                buf[dst] += d
            out[ry0:ry1, rx0:rx1] = t.clamp(t.round(buf), 0, 255).to(t.uint8)
        return out.cpu().numpy()


# ============================================================================================================
# Quadros mestres
# ============================================================================================================
def _pl_key(pl: Mapping[int, L.Placement]):
    """Chave exata das placements (None quando há recorte ``alpha_mul``: aí o quadro é sempre recomposto)."""
    k = []
    for i in sorted(pl):
        q = pl[i]
        if q.alpha_mul is not None or q.warp is not None:
            return None
        k.append((i, q.dx, q.dy, float(q.shade), float(q.contact), float(q.scale), q.anchor))
    return tuple(k)


def dissolve(landed: np.ndarray, photo: np.ndarray, k: int, n: int) -> np.ndarray:
    """Quadro ``k`` (0..n−1) da dissolução pilha completa → foto (mesma conta de ``choreo.render_frames``)."""
    u = (k + 1) / (n + 1)
    return np.clip(np.rint((1 - u) * landed.astype(np.float32) + u * photo.astype(np.float32)), 0, 255
                   ).astype(np.uint8)


def master_frames(stack: L.LayerStack, ch, photo: np.ndarray, comp: FrameCompositor | None = None,
                  with_placements: bool = False) -> Iterator:
    """Quadros mestres uint8 na ordem do vídeo, iguais aos de ``choreo.render_frames`` (backend numpy: bit a
    bit). Com ``with_placements`` devolve ``(quadro, placements)`` (``None`` na dissolução e no hold final)."""
    from . import choreo as C

    comp = comp or FrameCompositor(stack)
    prev_key, prev = None, None
    landed = None
    for f in range(ch.n_anim):
        pl = C.placements_at(ch, stack, f)
        key = _pl_key(pl)
        if key is not None and key == prev_key:
            fr = prev
        else:
            fr = comp.compose(pl)
        prev_key, prev = key, fr
        if f == ch.n_anim - 1:
            landed = fr
        yield (fr, pl) if with_placements else fr
    if landed is None:
        landed = comp.compose({i: L.Placement() for i in range(len(stack.layers))})
    nd = C.DISSOLVE_FRAMES
    for k in range(nd):
        fr = dissolve(landed, photo, k, nd)
        yield (fr, None) if with_placements else fr
    for _ in range(ch.n_frames - ch.n_anim - nd):
        yield (photo, None) if with_placements else photo


def moving(q: L.Placement) -> bool:
    return bool(q.dx or q.dy or q.scale < 1.0 - 1e-6 or q.warp is not None)


def scaled_box(box, q: L.Placement) -> tuple[int, int, int, int]:
    """Caixa (com folga de 1 px) que contém a camada escalada por ``q.scale`` em torno de ``q.anchor``, sem
    deslocamento. Calculada aqui de forma independente (só geometria) para a verificação."""
    x0, y0, x1, y1 = box
    if q.scale >= 1.0 - 1e-6:
        return tuple(box)
    ax, ay = q.anchor if q.anchor is not None else ((x0 + x1) / 2.0, float(y1 - 1))
    s = float(q.scale)
    return (int(math.floor(ax + s * (x0 - ax))) - 1, int(math.floor(ay + s * (y0 - ay))) - 1,
            int(math.ceil(ax + s * (x1 - ax))) + 2, int(math.ceil(ay + s * (y1 - ay))) + 2)
