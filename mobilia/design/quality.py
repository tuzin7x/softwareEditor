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


def score(groups: np.ndarray, lock: np.ndarray, furnish: bool = True) -> dict:
    """Nota de um candidato a partir dos grupos semânticos da imagem gerada (``lock`` na mesma resolução)."""
    g = np.asarray(groups)
    lk = np.asarray(lock, bool)
    if lk.shape != g.shape:
        lk = cv2.resize(lk.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    free = ~lk
    furn = np.isin(g, [GROUP_IDS["furniture"], GROUP_IDS["fixture"]]) & free
    f = float(furn.sum() / max(free.sum(), 1))
    r = max(2, int(0.02 * np.hypot(*g.shape)))
    near = cv2.dilate(lk.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)).astype(bool)
    openings = np.isin(g, [GROUP_IDS["window"], GROUP_IDS["door"]]) & ~near
    o = float(openings.mean())
    s = (min(f, FURNISHED_TARGET) / FURNISHED_TARGET if furnish else 0.0) - NEW_OPENING_WEIGHT * o / 0.02
    return {"nota": round(s, 4), "mobiliado": round(f, 4), "aberturas_novas": round(o, 4)}


class Segmenter:
    """OneFormer ADE20K carregado uma vez para segmentar vários candidatos."""

    def __init__(self, device: str = "cpu", long_side: int = 512):
        import torch
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        from ..config import SEGMENTATION_MODEL_ID

        self.device, self.long_side = device, long_side
        self.processor = OneFormerProcessor.from_pretrained(SEGMENTATION_MODEL_ID)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(SEGMENTATION_MODEL_ID).to(device).eval()
        self.fp16 = device.startswith("cuda")
        if self.fp16:
            self.model = self.model.half()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self._torch = torch

    def groups(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        torch = self._torch
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


__all__ = ["score", "rank", "Segmenter", "FURNISHED_TARGET", "NEW_OPENING_WEIGHT"]
