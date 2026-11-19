"""PT-V3-7 (lento): vídeo do modo B sobre a sala vazia do big-lama de verdade (CPU): composição, verificação §3.5
em 100% dos quadros, MP4 (x264) e MKV sem perdas decodificado byte a byte, e numpy x torch."""
import numpy as np
import pytest

from mobilia.analysis import weights
from mobilia.layers2d import choreo as C
from mobilia.layers2d import compose as K
from mobilia.layers2d import layers as L
from mobilia.layers2d import video as VB
from mobilia.removal import instances as ri
from mobilia.removal import plate as rp
from mobilia.video import encoder as E

pytestmark = [pytest.mark.slow, pytest.mark.models]


@pytest.mark.parametrize("name", ["nice_terrazzo", "sala_sofas_brancos"])
def test_mode_b_video_on_lama_plate(room_fixture, name, tmp_path, monkeypatch):
    try:
        lama = rp.load_lama("cpu", offline=True)
    except weights.WeightsError as e:
        pytest.skip(str(e))
    monkeypatch.setenv(E.ENV_CODEC, "libx264")
    f = room_fixture(name)
    s = ri.detect_instances(f.photo, f.seg, ri.PanopticResult(f.panoptic_ids, f.panoptic_segments), sam=None,
                            geometry=f.geometry, room=f.room)
    depth = f.geometry["points"][..., 2].astype(np.float32)
    r = rp.build_plate(f.photo, s.removal(), f.seg, f.room, depth, lama, assess_quality=False)
    pieces = s.removal(r.kept_builtins)
    st = L.build_layers(f.photo, r, pieces, depth)
    ch = C.plan(st)
    v = C.verify(st, ch)
    rep = VB.render_mode_b(st, ch, f.photo, r.mask, tmp_path / "b.mp4", audit_mkv=tmp_path / "b.mkv",
                           backend="numpy", choreo_stats=v)
    vb, ver = rep["video_modo_b"], rep["verificacao_modo_b"]
    assert ver["ok"], ver["falhas"]
    assert ver["quadros"] == ch.n_frames and vb["auditoria_mkv"]["quadros_identicos"] == ch.n_frames
    assert vb["psnr_mp4"]["ultimo_vs_foto"] > 35
    got = dict(E.decode_lossless(tmp_path / "b.mkv", f.photo.shape[1], f.photo.shape[0]))
    assert np.array_equal(got[1], st.plate) and np.array_equal(got[ch.n_frames], f.photo)
    pytest.importorskip("torch")
    cn, ct = K.FrameCompositor(st, "numpy"), K.FrameCompositor(st, "torch")
    worst = 0
    for k in range(0, ch.n_anim, 5):
        pl = C.placements_at(ch, st, k)
        worst = max(worst, int(np.abs(cn.compose(pl).astype(np.int16) - ct.compose(pl).astype(np.int16)).max()))
    assert worst <= 1, worst
