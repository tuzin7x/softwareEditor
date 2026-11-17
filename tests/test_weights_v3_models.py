"""PT-V3-1 (lentos): os pesos baixados carregam e rodam na CPU. Precisam de SAM 2.1 e big-lama no disco."""
import numpy as np
import pytest

from mobilia.analysis import weights

pytestmark = [pytest.mark.slow, pytest.mark.models]


def _or_skip(fn):
    try:
        return fn(offline=True)
    except weights.WeightsError as e:
        pytest.skip(str(e))


def test_big_lama_loads_and_inpaints_on_cpu():
    import torch

    path = _or_skip(weights.big_lama_path)
    model = torch.jit.load(str(path), map_location="cpu").eval()
    h = w = 128
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 2, yy * 2, np.full_like(xx, 120)], -1).astype(np.uint8)
    mask = np.zeros((h, w), bool)
    mask[40:88, 40:88] = True
    hole = img.copy()
    hole[mask] = 0
    ti = torch.from_numpy(hole).permute(2, 0, 1)[None].float() / 255.0
    tm = torch.from_numpy(mask)[None, None].float()
    with torch.inference_mode():
        out = model(ti, tm)
    out = (out[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    assert out.shape == (h, w, 3)
    # o buraco preto foi preenchido com algo próximo do gradiente ao redor
    err = np.abs(out[mask].astype(int) - img[mask].astype(int)).mean()
    assert err < 40, err


def test_sam2_loads_and_segments_box():
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor

    folder = _or_skip(weights.sam2_dir)
    proc = Sam2Processor.from_pretrained(str(folder))
    model = Sam2Model.from_pretrained(str(folder)).eval()
    h, w = 256, 320
    img = np.full((h, w, 3), 200, np.uint8)
    img[80:180, 100:220] = (40, 60, 160)               # "peça" azul sobre fundo claro
    with torch.inference_mode():
        inp = proc(images=Image.fromarray(img), input_boxes=[[[90.0, 70.0, 230.0, 190.0]]], return_tensors="pt")
        out = model(**inp, multimask_output=False)
        m = proc.post_process_masks(out.pred_masks, inp["original_sizes"])[0][0, 0].numpy()
    gt = np.zeros((h, w), bool)
    gt[80:180, 100:220] = True
    iou = (m & gt).sum() / max(1, (m | gt).sum())
    assert iou > 0.8, iou
