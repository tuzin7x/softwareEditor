"""Encode MP4: cores preservadas (BT.709 explícito) e alta fidelidade."""
import numpy as np

from mobilia.video.encoder import EncodeSettings, VideoWriter, decode_frames, psnr


def test_encode_roundtrip_color_and_quality(tmp_path):
    H, W = 144, 192
    rng = np.random.default_rng(1)
    base = np.zeros((H, W, 3), np.uint8)
    base[:, : W // 3] = (200, 160, 120)          # bege (parede/piso)
    base[:, W // 3: 2 * W // 3] = (60, 90, 140)  # azul
    base[:, 2 * W // 3:] = (230, 230, 225)       # branco
    base = np.clip(base.astype(int) + rng.integers(-3, 4, size=base.shape), 0, 255).astype(np.uint8)
    out = tmp_path / "t.mp4"
    w = VideoWriter(out, W, H, 24, EncodeSettings(crf=14, preset="medium"))
    for i in range(12):
        f = base.copy()
        f[40:80, 20 + i * 4: 60 + i * 4] = (30, 30, 30)    # objeto em movimento
        w.write(f)
    w.close()
    assert out.exists() and out.stat().st_size > 0
    frames = dict(decode_frames(out, W, H))
    assert len(frames) == 12
    dec = frames[12]
    ref = base.copy()
    ref[40:80, 64:104] = (30, 30, 30)
    assert psnr(dec, ref) > 38
    # sem deslocamento de cor (matriz BT.601 x BT.709 erraria vários níveis)
    for sl in (np.s_[100:140, 5:55], np.s_[100:140, 70:120], np.s_[100:140, 140:185]):
        assert np.abs(dec[sl].mean((0, 1)) - ref[sl].mean((0, 1))).max() < 2.0
