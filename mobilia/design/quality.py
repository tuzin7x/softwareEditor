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


def score(groups: np.ndarray, lock: np.ndarray, furnish: bool = True, source_openings: np.ndarray | None = None
          ) -> dict:
    """Nota de um candidato a partir dos grupos semânticos da imagem gerada (``lock`` na mesma resolução).

    ``source_openings``: janelas/cortinas da foto fora da trava; as que sumiram (viraram parede) pesam contra
    (``aberturas_perdidas``). ``maior_movel``: fração da imagem do maior móvel contínuo; acima de
    ``OVERSIZE_FRAC`` pesa contra (móvel fora de escala)."""
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
    return {"nota": round(s, 4), "mobiliado": round(f, 4), "aberturas_novas": round(o, 4),
            "aberturas_perdidas": round(miss, 4), "maior_movel": round(big, 4)}


class Segmenter:
    """OneFormer ADE20K carregado uma vez para segmentar vários candidatos."""

    def __init__(self, device: str = "cpu", long_side: int = 512):
        import torch
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        from ..config import SEGMENTATION_MODEL_ID

        from ..analysis.geometry import _ensure_weights
        from ..analysis.weights import ensure_local

        _ensure_weights(ensure_local, SEGMENTATION_MODEL_ID)
        self.device, self.long_side = device, long_side
        self.processor = OneFormerProcessor.from_pretrained(SEGMENTATION_MODEL_ID)
        # .bin do repositório: sem ``use_safetensors=False`` o transformers busca na rede uma conversão .safetensors
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
        return labels_to_groups(seg.cpu().numpy().astype(np.uint8), self.id2label)

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


__all__ = ["score", "rank", "Segmenter", "FURNISHED_TARGET", "NEW_OPENING_WEIGHT", "MISSING_OPENING_WEIGHT",
           "OVERSIZE_FRAC"]
