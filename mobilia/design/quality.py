"""Escolha automática entre candidatos: gera mais versões do que o pedido e devolve as melhores.

No "Mobiliar ambiente vazio" o gerador às vezes devolve uma sala quase vazia (um tapete e uma cadeira) ou inventa
uma janela numa parede lisa. Cada candidato é segmentado (OneFormer, carregado uma vez para todos, em resolução
reduzida) e recebe uma nota:

* ``mobiliado``: fração da área que pode mudar (fora da trava) coberta por móveis (grupos ``furniture`` e
  ``fixture`` de ``analysis/segmentation.py``) — quanto mais perto de um ambiente completo, melhor (com teto: uma
  sala entulhada não ganha mais nota);
* ``aberturas_novas``: fração de janela/porta na imagem gerada fora das aberturas originais (dilatadas) — pesa
  forte contra.
"""
from __future__ import annotations

import gc

import cv2
import numpy as np

from ..analysis.segmentation import GROUP_IDS, labels_to_groups

FURNISHED_TARGET = 0.30      # fração mobiliada a partir da qual a nota não sobe mais
NEW_OPENING_WEIGHT = 4.0
MISSING_OPENING_WEIGHT = 3.0
OVERSIZE_FRAC = 0.22         # um único móvel cobrindo mais que isto da imagem: fora de escala (sofá gigante)
OVERSIZE_WEIGHT = 2.0
# móvel novo cortado pela borda da foto (pedaço sem sentido no canto). 1,5 não bastou: na GPU (0.9.0) ainda saíram
# uma poltrona cortada embaixo à direita e cadeiras cortadas nas laterais — pesa o dobro e vale até 10% da imagem
# (uma poltrona cortada perto da câmera passa de 6%)
CUT_WEIGHT = 3.0
CUT_MAX_FRAC = 0.10          # só pedaços: um móvel grande que sai do quadro (sofá em primeiro plano) é composição normal
CUT_RETRY = 0.008            # "Mobiliar": escolhidos com mais que isto de móvel cortado -> gera mais candidatos
TREE_WEIGHT = 3.0            # Natal: a árvore é o que faz a decoração ser reconhecida


def score(groups: np.ndarray, lock: np.ndarray, furnish: bool = True, source_openings: np.ndarray | None = None,
          source_furniture: np.ndarray | None = None) -> dict:
    """Nota de um candidato a partir dos grupos semânticos da imagem gerada (``lock`` na mesma resolução).

    ``source_openings``: janelas/cortinas da foto fora da trava; as que sumiram (viraram parede) pesam contra
    (``aberturas_perdidas``). ``maior_movel``: fração da imagem do maior móvel contínuo; acima de
    ``OVERSIZE_FRAC`` pesa contra (móvel fora de escala). ``cortados``: fração da imagem em PEDAÇOS de móvel
    (até ``CUT_MAX_FRAC``) encostados na borda esquerda/direita/de baixo que não estavam na foto
    (``source_furniture``) — no "Mobiliar" do teste real saiu um móvel cortado e sem sentido no canto."""
    g = np.asarray(groups)
    lk = np.asarray(lock, bool)
    if lk.shape != g.shape:
        lk = cv2.resize(lk.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    free = ~lk
    furn = np.isin(g, [GROUP_IDS["furniture"], GROUP_IDS["fixture"]]) & free
    f = float(furn.sum() / max(free.sum(), 1))
    r = max(2, int(0.02 * np.hypot(*g.shape)))
    known = lk
    if source_openings is not None:       # abertura da foto que continua lá não é "nova"
        so = np.asarray(source_openings, bool)
        if so.shape != g.shape:
            so = cv2.resize(so.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        known = lk | so
    near = cv2.dilate(known.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)).astype(bool)
    openings = np.isin(g, [GROUP_IDS["window"], GROUP_IDS["door"]]) & ~near
    o = float(openings.mean())
    s = (min(f, FURNISHED_TARGET) / FURNISHED_TARGET if furnish else 0.0) - NEW_OPENING_WEIGHT * o / 0.02
    miss = 0.0
    if source_openings is not None:
        from .archlock import missing_openings

        src = np.asarray(source_openings, bool)
        if src.shape != g.shape:
            src = cv2.resize(src.astype(np.uint8), (g.shape[1], g.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        if src.any():
            lost = missing_openings(g, src, (g.shape[1], g.shape[0]))
            miss = float(lost.sum() / max(float(src.sum()), 1.0))
            s -= MISSING_OPENING_WEIGHT * miss
    big = 0.0
    fm = np.isin(g, [GROUP_IDS["furniture"]]).astype(np.uint8)
    if fm.any():
        n, _, stats, _ = cv2.connectedComponentsWithStats(fm, connectivity=8)
        big = float(stats[1:, cv2.CC_STAT_AREA].max() / fm.size) if n > 1 else 0.0
        if furnish and big > OVERSIZE_FRAC:
            s -= OVERSIZE_WEIGHT * (big - OVERSIZE_FRAC) / 0.1
    cut = cut_pieces(g, source_furniture) if furnish else 0.0
    if cut > 0:
        s -= CUT_WEIGHT * min(1.0, cut / 0.02)
    return {"nota": round(s, 4), "mobiliado": round(f, 4), "aberturas_novas": round(o, 4),
            "aberturas_perdidas": round(miss, 4), "maior_movel": round(big, 4), "cortados": round(cut, 4)}


def cut_pieces(groups: np.ndarray, source_furniture: np.ndarray | None = None, max_frac: float = CUT_MAX_FRAC,
               band_frac: float = 0.01) -> float:
    """Fração da imagem em pedaços de móvel (componentes até ``max_frac`` da imagem) encostados na borda
    esquerda, direita ou de baixo, fora dos móveis que já estavam na foto (``source_furniture``)."""
    g = np.asarray(groups)
    fm = np.isin(g, [GROUP_IDS["furniture"]])
    if source_furniture is not None:
        sf = np.asarray(source_furniture, bool)
        if sf.shape != g.shape:
            sf = cv2.resize(sf.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        fm &= ~sf
    if not fm.any():
        return 0.0
    H, W = g.shape
    b = max(1, int(band_frac * max(H, W)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fm.astype(np.uint8), connectivity=8)
    edge = np.zeros(g.shape, bool)
    edge[-b:] = True
    edge[:, :b] = True
    edge[:, -b:] = True
    touching = set(np.unique(lab[edge & (lab > 0)]).tolist())
    tot = 0
    for k in touching:
        a = int(st[k, cv2.CC_STAT_AREA])
        if a <= max_frac * fm.size:
            tot += a
    return float(tot / fm.size)


def tree_score(rgb: np.ndarray, exclude: np.ndarray | None = None, long_side: int = 384) -> dict:
    """Há uma árvore de Natal na imagem? Heurística de cor (sem modelo): a maior mancha contínua de verde de
    pinheiro (fora de ``exclude`` — o verde que já estava na foto, a vista da janela), com a altura e a área que uma
    árvore tem, e brilhos quentes (pisca-pisca) dentro dela. ``arvore`` em [0, 1.25]. No teste real nenhuma das
    duas versões de Natal tinha árvore: os candidatos sem ela perdem a vez."""
    img = np.asarray(rgb)
    H0, W0 = img.shape[:2]
    k = min(1.0, long_side / max(H0, W0))
    if k < 1.0:
        img = cv2.resize(img, (max(8, int(W0 * k)), max(8, int(H0 * k))), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s_, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    green = (h >= 33) & (h <= 95) & (s_ >= 50) & (v >= 20) & (v <= 215)
    if exclude is not None:
        ex = np.asarray(exclude, bool)
        if ex.shape != (H, W):
            ex = cv2.resize(ex.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        green &= ~ex
    r = max(1, int(0.012 * np.hypot(H, W)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    blob = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, ker)
    n, lab, st, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    if n <= 1:
        return {"arvore": 0.0, "altura": 0.0, "area": 0.0, "luzes": 0.0}
    best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, hh, area = (int(st[best, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                     cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
    hf, af = hh / H, area / (H * W)
    tall = hh >= 0.9 * w                                   # árvore em pé, não um tapete/sofá verde deitado
    sc = min(1.0, hf / 0.35) * min(1.0, af / 0.03) * (1.0 if tall else 0.4)
    box = (slice(y, y + hh), slice(x, x + w))
    warm = ((h[box] <= 35) | (h[box] >= 160)) & (v[box] >= 225)
    lights = float(warm.mean())
    if lights >= 0.004:
        sc += 0.25 * min(1.0, lights / 0.02)
    return {"arvore": round(float(sc), 4), "altura": round(float(hf), 4), "area": round(float(af), 4),
            "luzes": round(lights, 4)}


class Segmenter:
    """OneFormer ADE20K carregado uma vez para segmentar vários candidatos."""

    def __init__(self, device: str = "cpu", long_side: int = 512):
        import torch
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        from ..config import SEGMENTATION_MODEL_ID

        from ..analysis.geometry import _ensure_weights
        from ..analysis.weights import ensure_local, hub_offline

        _ensure_weights(ensure_local, SEGMENTATION_MODEL_ID)
        self.device, self.long_side = device, long_side
        with hub_offline(SEGMENTATION_MODEL_ID):     # já no disco: sem consultar o Hub (IPv6 lento: ~42 s cada)
            self.processor = OneFormerProcessor.from_pretrained(SEGMENTATION_MODEL_ID)
            # .bin do repositório: sem ``use_safetensors=False`` o transformers busca na rede uma conversão
            self.model = OneFormerForUniversalSegmentation.from_pretrained(SEGMENTATION_MODEL_ID,
                                                                           use_safetensors=False).eval()
        self.fp16 = device.startswith("cuda")
        self.model = self.model.to(device, torch.float16 if self.fp16 else torch.float32)
        self.parked = False
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self._torch = torch

    def park(self) -> None:
        """Tira o OneFormer da GPU (fica na RAM, fp16) enquanto o SDXL gera: o pico do UNet não soma ao dele."""
        if self.device.startswith("cuda") and not self.parked:
            self.model = self.model.to("cpu")
            self.parked = True
            self._torch.cuda.empty_cache()

    def _unpark(self) -> None:
        if self.parked:
            self.model = self.model.to(self.device)
            self.parked = False

    def groups(self, rgb: np.ndarray) -> np.ndarray:
        return labels_to_groups(self.labels(rgb), self.id2label)

    def labels(self, rgb: np.ndarray) -> np.ndarray:
        """Rótulos ADE20K (ids) na resolução reduzida (``long_side``)."""
        from PIL import Image

        torch = self._torch
        self._unpark()
        s = min(1.0, self.long_side / max(rgb.shape[:2]))
        small = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
        inputs = self.processor(images=Image.fromarray(small), task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if self.fp16:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            out = self.model(**inputs)
        seg = self.processor.post_process_semantic_segmentation(out, target_sizes=[small.shape[:2]])[0]
        return seg.cpu().numpy().astype(np.uint8)

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()


def rank(images: list[np.ndarray], lock: np.ndarray, device: str = "cpu", furnish: bool = True,
         group_fn=None) -> list[dict]:
    """Nota de cada imagem (mesma ordem). ``group_fn(rgb) -> grupos`` substitui o OneFormer (testes)."""
    seg = None
    if group_fn is None:
        seg = Segmenter(device)
        group_fn = seg.groups
    try:
        return [score(group_fn(im), lock, furnish) for im in images]
    finally:
        if seg is not None:
            seg.close()


__all__ = ["score", "rank", "Segmenter", "tree_score", "cut_pieces", "CUT_RETRY", "FURNISHED_TARGET", "NEW_OPENING_WEIGHT",
           "MISSING_OPENING_WEIGHT", "OVERSIZE_FRAC", "CUT_WEIGHT", "TREE_WEIGHT"]
