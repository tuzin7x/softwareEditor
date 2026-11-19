"""PT-V3-5 (lento): camadas sobre a sala vazia do big-lama de verdade (CPU)."""
import time

import numpy as np
import pytest

from mobilia.analysis import weights
from mobilia.layers2d import layers as L
from mobilia.removal import instances as ri
from mobilia.removal import plate as rp

pytestmark = [pytest.mark.slow, pytest.mark.models]


@pytest.mark.parametrize("name", ["sala_sofas_brancos"])
def test_layers_on_lama_plate_acceptance(room_fixture, name):
    try:
        lama = rp.load_lama("cpu", offline=True)
    except weights.WeightsError as e:
        pytest.skip(str(e))
    f = room_fixture(name)
    s = ri.detect_instances(f.photo, f.seg, ri.PanopticResult(f.panoptic_ids, f.panoptic_segments), sam=None,
                            geometry=f.geometry, room=f.room)
    depth = f.geometry["points"][..., 2].astype(np.float32)
    r = rp.build_plate(f.photo, s.removal(), f.seg, f.room, depth, lama, assess_quality=False)
    t0 = time.perf_counter()
    st = L.build_layers(f.photo, r, s.removal(), depth)
    dt = time.perf_counter() - t0
    full = L.composite(st, range(len(st.layers)))
    empty = L.composite(st, [])
    assert np.abs(full.astype(int) - f.photo).max() <= L.RECON_TOL
    assert np.array_equal(empty, st.plate)
    assert L.morph_fraction(st, empty, full) == 0.0
    assert st.stats["placa_corrigida_da_mascara_pct"] < 5.0
    assert dt < 10.0, dt          # CPU de 4 núcleos; a meta [GPU] < 1 s é do PT-V3-7
