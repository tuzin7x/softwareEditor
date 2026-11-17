"""PT-V3-6 (lento): coreografia sobre a sala vazia do big-lama de verdade (CPU), com todos os quadros."""
import time

import numpy as np
import pytest

from mobilia.analysis import weights
from mobilia.layers2d import choreo as C
from mobilia.layers2d import layers as L
from mobilia.removal import instances as ri
from mobilia.removal import plate as rp

pytestmark = [pytest.mark.slow, pytest.mark.models]


@pytest.mark.parametrize("name", ["nice_terrazzo"])
def test_choreography_on_lama_plate_all_frames(room_fixture, name):
    try:
        lama = rp.load_lama("cpu", offline=True)
    except weights.WeightsError as e:
        pytest.skip(str(e))
    f = room_fixture(name)
    s = ri.detect_instances(f.photo, f.seg, ri.PanopticResult(f.panoptic_ids, f.panoptic_segments), sam=None,
                            geometry=f.geometry, room=f.room)
    depth = f.geometry["points"][..., 2].astype(np.float32)
    r = rp.build_plate(f.photo, s.removal(), f.seg, f.room, depth, lama, assess_quality=False)
    st = L.build_layers(f.photo, r, s.removal(), depth)
    t0 = time.perf_counter()
    ch = C.plan(st)
    v = C.verify(st, ch)
    dt = time.perf_counter() - t0
    assert v["oculto_visivel_px"] == 0 and v["opacidade_parcial"] == 0 and v["borda_vindo_de_cima"] == 0
    assert C.TOTAL_MIN - 1e-6 <= v["duracao_s"] <= C.TOTAL_MAX + 1e-6
    assert dt < 10.0, dt
    frames = C.render_frames(st, ch, f.photo)
    first = next(frames)
    assert np.array_equal(first, st.plate)
    n, last = 1, first
    for fr in frames:
        # fora da máscara de remoção, todo quadro é a foto (só as peças em voo podem passar por fora dela)
        if n < ch.n_anim and not any(q.dx or q.dy for q in C.placements_at(ch, st, n).values()):
            assert np.array_equal(fr[~r.mask], f.photo[~r.mask]), n
        last, n = fr, n + 1
    assert n == ch.n_frames
    assert np.array_equal(last, f.photo)
