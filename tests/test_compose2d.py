"""PT-V3-7: compositor 2D (numpy/torch), verificação §3.5 e codificação do modo B (rápidos, sem modelos)."""
import numpy as np
import pytest

from mobilia.composite import verify as V
from mobilia.layers2d import choreo as C
from mobilia.layers2d import compose as K
from mobilia.layers2d import layers as L
from mobilia.layers2d import video as VB
from mobilia.video import encoder as E

from test_choreo import ROOM, _build


@pytest.fixture(scope="module")
def room():
    photo, res, st = _build(ROOM)
    ch = C.plan(st)
    v = C.verify(st, ch)
    return photo, res, st, ch, v


def test_merge_boxes_disjoint():
    regs = K.merge_boxes([(0, 0, 10, 10), (5, 5, 20, 20), (30, 30, 40, 40), (19, 0, 31, 2)])
    for a in regs:
        for b in regs:
            assert a is b or not K._overlap(a, b)
    assert (0, 0, 31, 20) in regs and (30, 30, 40, 40) in regs


def test_numpy_compositor_bit_identical_to_layers_composite(room):
    photo, res, st, ch, _ = room
    comp = K.FrameCompositor(st, "numpy")
    for f in range(0, ch.n_anim, 3):
        pl = C.placements_at(ch, st, f)
        assert np.array_equal(comp.compose(pl), L.composite(st, pl)), f


def test_master_frames_equal_render_frames(room):
    photo, res, st, ch, _ = room
    a = list(K.master_frames(st, ch, photo, K.FrameCompositor(st, "numpy")))
    b = list(C.render_frames(st, ch, photo))
    assert len(a) == len(b) == ch.n_frames
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.array_equal(a[0], st.plate) and np.array_equal(a[-1], photo)


def test_torch_matches_numpy_within_one_level(room):
    pytest.importorskip("torch")
    photo, res, st, ch, _ = room
    cn, ct = K.FrameCompositor(st, "numpy"), K.FrameCompositor(st, "torch")
    assert ct.backend == "torch"
    worst = 0
    for f in range(0, ch.n_anim, 2):
        pl = C.placements_at(ch, st, f)
        d = np.abs(cn.compose(pl).astype(np.int16) - ct.compose(pl).astype(np.int16)).max()
        worst = max(worst, int(d))
    assert worst <= 1, worst


def test_choose_backend_env(monkeypatch):
    monkeypatch.setenv(K.ENV_BACKEND, "numpy")
    assert K.choose_backend() == ("numpy", "cpu")
    monkeypatch.setenv(K.ENV_BACKEND, "auto")
    b, d = K.choose_backend()
    assert (b, d) in {("numpy", "cpu"), ("torch", "cuda")}


def test_verifier_passes_all_frames(room):
    photo, res, st, ch, v = room
    ver = V.ModeBVerifier(photo, st, res.mask, ch.n_frames)
    for fr, pl in K.master_frames(st, ch, photo, with_placements=True):
        ver.check(fr, pl)
    rep = ver.finish(v)["verificacao_modo_b"]
    assert rep["ok"] and rep["quadros"] == ch.n_frames and rep["morph_px"] == 0
    assert rep["quadro0_igual_placa"] and rep["ultimo_igual_foto"]


def _frames(room):
    photo, res, st, ch, _ = room
    return [(fr.copy(), pl) for fr, pl in K.master_frames(st, ch, photo, with_placements=True)]


@pytest.mark.parametrize("bug", ["frame0", "last", "outside", "morph", "plate"])
def test_verifier_catches_violations(room, bug):
    photo, res, st, ch, _ = room
    frames = _frames(room)
    mask = res.mask
    if bug == "plate":
        st2 = L.LayerStack(st.photo_shape, st.plate.copy(), st.layers, st.front, dynamic=st.dynamic)
        ys, xs = np.nonzero(~mask)
        st2.plate[ys[0], xs[0]] ^= 1
        with pytest.raises(V.ModeBVerificationError, match="regra 4"):
            V.ModeBVerifier(photo, st2, mask, ch.n_frames)
        return
    if bug == "frame0":
        y, x = np.argwhere(mask)[0]
        frames[0][0][y, x] ^= 4
        rule = "regra 1"
    elif bug == "last":
        y, x = np.argwhere(mask)[0]
        frames[-1][0][y, x] ^= 4
        rule = "regra 2"
    elif bug == "outside":
        k = ch.n_anim // 2
        movb = V.moving_boxes(st, frames[k][1], st.photo_shape)
        y, x = np.argwhere(~(mask | movb))[0]
        frames[k][0][y, x] ^= 1                     # 1 nível fora da máscara: tem de acusar
        rule = "regra 3"
    else:                                            # pixel fora da região dinâmica muda num quadro só
        k = ch.n_anim // 2
        allow = st.dynamic | V.moving_boxes(st, frames[k][1], st.photo_shape) | \
            V.moving_boxes(st, frames[k - 1][1], st.photo_shape) | \
            V.moving_boxes(st, frames[k + 1][1], st.photo_shape)
        y, x = np.argwhere(~allow)[-1]
        frames[k][0][y, x] ^= 8
        rule = "regra 6"
    ver = V.ModeBVerifier(photo, st, mask, ch.n_frames, strict=False)
    for fr, pl in frames:
        ver.check(fr, pl)
    rep = ver.finish()["verificacao_modo_b"]
    assert not rep["ok"] and any(s.startswith(rule) for s in rep["falhas"]), rep["falhas"]


def test_moving_boxes_cover_scaled_and_shifted(room):
    photo, res, st, ch, _ = room
    q = L.Placement(dx=7, dy=-5, scale=0.94)
    i = 0
    m = V.moving_boxes(st, {i: q}, st.photo_shape)
    a, c, sh, box = L._scaled(st.layers[i], q)
    ys, xs = np.nonzero(a > 0)
    H, W = st.photo_shape
    yy, xx = ys + box[1] + q.dy, xs + box[0] + q.dx
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    assert m[yy[ok], xx[ok]].all()
    assert not V.moving_boxes(st, {i: L.Placement()}, st.photo_shape).any()


def test_render_mode_b_mp4_and_lossless_audit(room, tmp_path, monkeypatch):
    monkeypatch.setenv(E.ENV_CODEC, "libx264")
    photo, res, st, ch, v = room
    mp4, mkv = tmp_path / "b.mp4", tmp_path / "b.mkv"
    rep = VB.render_mode_b(st, ch, photo, res.mask, mp4, audit_mkv=mkv, backend="numpy", choreo_stats=v)
    assert mp4.exists() and mkv.exists()
    r = rep["video_modo_b"]
    assert r["quadros"] == ch.n_frames and r["codificador"] == "libx264"
    assert r["auditoria_mkv"]["quadros_identicos"] == ch.n_frames
    assert r["psnr_mp4"]["ultimo_vs_foto"] > 35 and r["psnr_mp4"]["quadro0_vs_placa"] > 35
    assert rep["verificacao_modo_b"]["ok"]
    got = dict(E.decode_lossless(mkv, photo.shape[1], photo.shape[0]))
    assert np.array_equal(got[1], st.plate) and np.array_equal(got[ch.n_frames], photo)


def test_render_mode_b_aborts_without_mp4(room, tmp_path, monkeypatch):
    monkeypatch.setenv(E.ENV_CODEC, "libx264")
    photo, res, st, ch, v = room
    bad = photo.copy()
    ys, xs = np.nonzero(~res.mask)
    bad[ys[0], xs[0]] ^= 2                           # foto "diferente" fora da máscara: regra 4 falha
    mp4, mkv = tmp_path / "x.mp4", tmp_path / "x.mkv"
    with pytest.raises(V.ModeBVerificationError):
        VB.render_mode_b(st, ch, bad, res.mask, mp4, audit_mkv=mkv, backend="numpy")
    assert not list(tmp_path.iterdir())


def test_render_mode_b_aborts_mid_stream(room, tmp_path, monkeypatch):
    monkeypatch.setenv(E.ENV_CODEC, "libx264")
    photo, res, st, ch, v = room
    orig = K.FrameCompositor.compose

    def broken(self, pl):
        fr = orig(self, pl)
        if len(pl) >= 2:
            fr = fr.copy()
            fr[0, 0] ^= 3                            # canto (fora da máscara) corrompido no meio do vídeo
        return fr

    monkeypatch.setattr(K.FrameCompositor, "compose", broken)
    mp4 = tmp_path / "y.mp4"
    with pytest.raises(V.ModeBVerificationError, match="regra 3"):
        VB.render_mode_b(st, ch, photo, res.mask, mp4, backend="numpy")
    assert not list(tmp_path.iterdir())


def test_render_mode_b_rejects_odd_size(room, tmp_path):
    photo, res, st, ch, v = room
    st2 = L.LayerStack((st.photo_shape[0] - 1, st.photo_shape[1]), st.plate[:-1], st.layers, st.front)
    with pytest.raises(ValueError, match="pares"):
        VB.render_mode_b(st2, ch, photo[:-1], res.mask[:-1], tmp_path / "z.mp4")


def test_render_mode_b_audit_close_failure_leaves_no_mp4(room, tmp_path, monkeypatch):
    # o MP4 fecha (e é renomeado) antes do MKV: se o MKV falha no close, o MP4 já pronto também tem de sair
    monkeypatch.setenv(E.ENV_CODEC, "libx264")
    photo, res, st, ch, v = room

    def boom(self):
        raise RuntimeError("FFmpeg falhou (simulado)")

    monkeypatch.setattr(E.LosslessWriter, "close", boom)
    with pytest.raises(RuntimeError, match="simulado"):
        VB.render_mode_b(st, ch, photo, res.mask, tmp_path / "w.mp4", audit_mkv=tmp_path / "w.mkv",
                         backend="numpy", choreo_stats=v)
    assert not list(tmp_path.iterdir())


def test_render_mode_b_rejects_broken_mp4(room, tmp_path, monkeypatch):
    # MP4 decodificado muito longe da placa/foto (codificador com defeito) não é entregue
    monkeypatch.setenv(E.ENV_CODEC, "libx264")
    photo, res, st, ch, v = room
    monkeypatch.setattr(E, "psnr", lambda a, b, mask=None: 12.0)
    with pytest.raises(RuntimeError, match="PSNR"):
        VB.render_mode_b(st, ch, photo, res.mask, tmp_path / "v.mp4", backend="numpy", choreo_stats=v)
    assert not list(tmp_path.iterdir())
