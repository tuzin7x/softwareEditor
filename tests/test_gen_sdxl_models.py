"""PT-V3-4 (lento): refino SDXL de verdade (RealVisXL Lightning + ControlNet depth, CPU bf16) em baixa resolução.

Pulado se os pesos do SDXL não estão no disco (``analysis/weights.py``: SDXL_WEIGHTS). Medido nesta máquina de
desenvolvimento (CPU 4 núcleos, sem GPU): ~1–2 min com 1 semente a 512 px, mais a carga do modelo.
"""
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mobilia.analysis import weights

pytestmark = [pytest.mark.slow, pytest.mark.models]


@pytest.fixture(scope="module")
def refiner():
    if not weights.sdxl_installed():
        pytest.skip("pesos do SDXL não instalados")
    from mobilia.gen import sdxl as G

    return G.load_sdxl("cpu", cfg=G.RefineConfig(seeds=1, steps=6, work_side=512))


def test_real_sdxl_refine_keeps_invariants(refiner, room_fixture):
    from mobilia import pipeline_b as pb
    from mobilia.gen import sdxl as G
    from mobilia.removal import instances as ri
    from mobilia.removal import plate as rp

    f = room_fixture("sala_sofas_brancos")
    H0, W0 = f.photo.shape[:2]
    W, H = 480, int(round(H0 * 480 / W0)) // 2 * 2
    photo = np.asarray(Image.fromarray(f.photo).resize((W, H), Image.LANCZOS))
    pts = f.geometry["points"].astype(np.float32)
    geo = SimpleNamespace(points=pts, mask=f.geometry["mask"], depth=pts[..., 2])
    seg, pan, geometry = pb.to_video_resolution(f.seg, ri.PanopticResult(f.panoptic_ids, f.panoptic_segments),
                                                geo, W, H)
    inst = ri.detect_instances(photo, seg, pan, sam=None, geometry=geometry, room=f.room)
    res = rp.build_plate(photo, inst.removal(False), seg, f.room, geo.depth, rp.telea_inpaint)
    cfg = G.RefineConfig(seeds=1, steps=6, work_side=512)
    seen = []
    t0 = time.perf_counter()
    out = G.refine_plate(photo, res, seg, f.room, geo.depth, refiner, None, seed=7, cfg=cfg, on_seed=seen.append)
    dt = time.perf_counter() - t0
    info = out.info["refino_sdxl"]
    assert len(seen) == 1 and info["sementes"][0]["nota"] is not None, info
    prot = G.protected_pixels(res.mask, res.labels, seg.groups, include_masked=not cfg.refine_protected_mask)
    for plate in (seen[0].plate, out.plate):
        assert np.array_equal(plate[~res.mask], res.plate[~res.mask])
        assert np.array_equal(plate[prot], res.plate[prot])
        assert np.array_equal(plate[~res.mask], photo[~res.mask])          # regra absoluta: fora = foto
    assert not np.array_equal(seen[0].plate[res.mask], res.plate[res.mask])  # o SDXL mudou o preenchimento
    assert refiner.calls[-1]["tamanho"][0] <= 520 and dt < 900
    assert "empty" in info["prompt"]
