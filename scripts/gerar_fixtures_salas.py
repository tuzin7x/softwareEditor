"""Gera as fixtures cacheadas de análise de uma foto de teste (``tests/fixtures/rooms/<nome>/``).

Uso (precisa dos modelos MoGe-2 e OneFormer no cache; roda em CPU, ~2–6 min por foto)::

    python scripts/gerar_fixtures_salas.py tests/fixtures/rooms/<nome>

A pasta já deve conter ``photo.jpg`` (<= 1280 px, licença comercial registrada em ``SOURCE.md``).
Arquivos gerados (todos pequenos, para os testes rápidos rodarem sem modelos em < 1 s):

- ``seg.npz``: segmentação semântica ADE20K (``SegmentationResult.save``), na resolução da foto;
- ``panoptic.npz``: OneFormer panóptico (mapa de segmentos ``uint8`` + ``label_id``/``score``/``area``
  por segmento), base para as instâncias de móveis do modo B;
- ``geometry_small.npz``: geometria MoGe-2 reduzida (pontos e normais em float16 a 1/4 da resolução,
  máscara, intrínsecos na resolução da foto);
- ``room.json`` e ``grid.npy``: modelo da sala (``build_room_model``) e grade de ocupação;
- ``meta.json``: modelos usados, resolução, tempos de CPU medidos.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GEOMETRY_MODEL = "Ruicheng/moge-2-vitl-normal"
SEG_MODEL = "shi-labs/oneformer_ade20k_swin_large"
GEOMETRY_STRIDE = 4


def panoptic(rgb8: np.ndarray, model_id: str = SEG_MODEL, device: str = "cpu") -> dict:
    """OneFormer panóptico (sem fundir rótulos). Retorna o mapa de segmentos e a tabela por segmento."""
    import torch
    from PIL import Image
    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

    proc = OneFormerProcessor.from_pretrained(model_id)
    model = OneFormerForUniversalSegmentation.from_pretrained(model_id).to(device).eval()
    img = Image.fromarray(rgb8)
    with torch.inference_mode():
        inp = proc(images=img, task_inputs=["panoptic"], return_tensors="pt")
        out = model(**{k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()})
        res = proc.post_process_panoptic_segmentation(
            out, target_sizes=[rgb8.shape[:2]], threshold=0.5, label_ids_to_fuse=set())[0]
    seg = res["segmentation"].cpu().numpy().astype(np.int32)
    infos = res["segments_info"]
    if len(infos) > 254:
        raise RuntimeError("segmentos demais para uint8")
    ids = np.zeros(seg.shape, np.uint8)  # 0 = nenhum segmento
    rows = []
    for k, s in enumerate(infos, start=1):
        m = seg == int(s["id"])
        ids[m] = k
        rows.append((k, int(s["label_id"]), float(s["score"]), int(m.sum())))
    table = np.array(rows, dtype=np.float64).reshape(-1, 4)
    return {"ids": ids, "segments": table}


def main(folder: str) -> None:
    from mobilia.analysis.geometry import estimate_geometry
    from mobilia.analysis.room import build_room_model
    from mobilia.analysis.segmentation import segment
    from mobilia.photo import load_photo

    d = Path(folder)
    rgb = load_photo(d / "photo.jpg").rgb8
    h, w = rgb.shape[:2]
    if max(h, w) > 1280:
        raise SystemExit("photo.jpg deve ter no máximo 1280 px no lado maior")
    tm = {}
    t = time.perf_counter()
    geo = estimate_geometry(rgb, model_id=GEOMETRY_MODEL, device="cpu")
    tm["geometria_cpu_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    seg = segment(rgb, model_id=SEG_MODEL, device="cpu")
    tm["segmentacao_cpu_s"] = round(time.perf_counter() - t, 1)
    seg.save(d / "seg.npz")
    t = time.perf_counter()
    pan = panoptic(rgb)
    tm["panoptico_cpu_s"] = round(time.perf_counter() - t, 1)
    np.savez_compressed(d / "panoptic.npz", ids=pan["ids"], segments=pan["segments"],
                        model_id=np.array(SEG_MODEL))
    room, dbg = build_room_model(geo, seg, rgb)
    (d / "room.json").write_text(room.to_json(), encoding="utf-8")
    np.save(d / "grid.npy", dbg["grid"])
    s = GEOMETRY_STRIDE
    np.savez_compressed(
        d / "geometry_small.npz",
        points=geo.points[::s, ::s].astype(np.float16), normals=geo.normals[::s, ::s].astype(np.float16),
        mask=geo.mask[::s, ::s], intrinsics=np.array(list(geo.intrinsics.to_dict().values())),
        stride=np.array(s), model_id=np.array(geo.model_id))
    meta = {"largura": w, "altura": h, "modelo_geometria": GEOMETRY_MODEL, "modelo_segmentacao": SEG_MODEL,
            "passo_geometria": s, "tempos_cpu_medidos": tm}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
