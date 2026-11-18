"""Segmentação semântica do ambiente com OneFormer ADE20K (SHI Labs, licença MIT).

As 150 classes do ADE20K são agrupadas em categorias úteis para a decoração:
piso, parede, teto, janela, porta, obstáculos estruturais e itens já existentes.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field

import numpy as np

# Grupos por nome de classe ADE20K (o id real vem de model.config.id2label).
GROUPS: dict[str, tuple[str, ...]] = {
    "floor": ("floor", "rug"),
    "wall": ("wall",),
    "ceiling": ("ceiling",),
    "window": ("window", "windowpane", "sky", "building", "tree", "house", "skyscraper", "mountain",
               "awning", "palm"),
    "window_covering": ("curtain", "blind"),
    "door": ("door", "screen door"),
    "structure": ("column", "stairs", "stairway", "step", "fireplace", "radiator", "railing", "rail",
                  "bannister", "base"),
    "fixture": ("cabinet", "counter", "countertop", "sink", "stove", "oven", "refrigerator",
                "dishwasher", "microwave", "hood", "kitchen island", "toilet", "bathtub", "tub",
                "shower", "mirror", "wardrobe", "bookcase", "shelf", "washer", "chest of drawers",
                "buffet"),
    "furniture": ("bed", "table", "chair", "sofa", "armchair", "seat", "desk", "lamp", "cushion",
                  "pillow", "coffee table", "swivel chair", "bench", "ottoman", "stool", "pool table",
                  "television receiver", "tv", "computer", "monitor", "box", "plant", "flower", "vase",
                  "pot", "basket", "painting", "poster", "clock", "person", "animal", "bag", "ashcan",
                  "trash can", "clothes", "blanket", "book", "towel", "bicycle", "plaything"),
    "ceiling_fixture": ("light", "chandelier", "sconce", "fan"),
}
GROUP_IDS = {name: i for i, name in enumerate(GROUPS)}   # floor=0, wall=1, ...
UNKNOWN = 255


@dataclass
class SegmentationResult:
    labels: np.ndarray                 # (H, W) uint8 — ids ADE20K (0..149)
    groups: np.ndarray                 # (H, W) uint8 — ids de GROUPS (255 = outro)
    id2label: dict[int, str] = field(default_factory=dict)
    model_id: str = ""

    def mask(self, group: str) -> np.ndarray:
        return self.groups == GROUP_IDS[group]

    def fraction(self, group: str) -> float:
        return float(self.mask(group).mean())

    def save(self, path) -> None:
        np.savez_compressed(path, labels=self.labels, groups=self.groups,
                            id2label=np.array([self.id2label.get(i, "") for i in range(256)]),
                            model_id=np.array(self.model_id))

    @staticmethod
    def load(path) -> "SegmentationResult":
        d = np.load(path, allow_pickle=False)
        names = d["id2label"].tolist()
        return SegmentationResult(d["labels"], d["groups"],
                                  {i: n for i, n in enumerate(names) if n}, str(d["model_id"]))


def labels_to_groups(labels: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    lut = np.full(256, UNKNOWN, dtype=np.uint8)
    for gname, classes in GROUPS.items():
        for cid, cname in id2label.items():
            base = cname.split(",")[0].strip().lower()
            if base in classes or cname.lower() in classes:
                lut[int(cid)] = GROUP_IDS[gname]
    return lut[labels]


def segment(rgb8: np.ndarray, model_id: str = "shi-labs/oneformer_ade20k_swin_large",
            device: str = "cpu", use_fp16: bool = False, timings: dict | None = None) -> SegmentationResult:
    """``timings`` (opcional) recebe ``carga`` (import + pesos + envio ao dispositivo) e ``download``
    (rede, só se os pesos ainda não estavam no cache) em segundos."""
    import time

    from .geometry import _ensure_weights
    from .weights import ensure_local

    t0 = time.perf_counter()
    dl = _ensure_weights(ensure_local, model_id)
    import torch
    from PIL import Image
    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

    processor = OneFormerProcessor.from_pretrained(model_id)
    model = OneFormerForUniversalSegmentation.from_pretrained(model_id, use_safetensors=False).to(device).eval()
    if use_fp16 and device != "cpu":
        model = model.half()
    if timings is not None:
        timings["download"] = dl
        timings["carga"] = time.perf_counter() - t0 - dl
    try:
        image = Image.fromarray(rgb8)
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if use_fp16 and device != "cpu":
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            outputs = model(**inputs)
        seg = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(rgb8.shape[0], rgb8.shape[1])])[0]
        labels = seg.cpu().numpy().astype(np.uint8)
        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    finally:
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return SegmentationResult(labels, labels_to_groups(labels, id2label), id2label, model_id)
