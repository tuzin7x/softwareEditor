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


def test_seek_verify(tmp_path):
    """``seek_frames`` (um FFmpeg com -ss por quadro, sem decodificar o vídeo inteiro) devolve
    exatamente os mesmos quadros da decodificação completa: início, meio, fim e quadros em volta
    das fronteiras de GOP (quadro-chave a cada 4 s)."""
    from mobilia.video.encoder import seek_frames

    H, W = 64, 96
    rng = np.random.default_rng(4)
    base = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    for fps, n in ((12, 61), (24, 121)):
        out = tmp_path / f"v{fps}.mp4"
        w = VideoWriter(out, W, H, fps, EncodeSettings())
        for i in range(n):
            f = base.copy()
            f[:8] = (i * 5) % 256                        # carimbo: cada quadro é diferente
            f[20 + (i % 30):30 + (i % 30), 10:50] = (i * 37) % 256
            w.write(f)
        w.close()
        full = dict(decode_frames(out, W, H))
        g = 4 * fps
        want = sorted({1, n // 2, n, g - 1, g, g + 1, 2})
        got = dict(seek_frames(out, W, H, want, fps))
        assert sorted(got) == want
        for k in want:
            assert np.array_equal(got[k], full[k]), (fps, k)


def test_codec_choice(monkeypatch):
    """NVENC só com GPU que codifica de verdade; sem GPU o teste do NVENC nem roda; a variável
    ``MOBILIA_CODIFICADOR`` força a escolha. O preset padrão do x264 é ``medium``."""
    from mobilia.video import encoder

    calls = []
    monkeypatch.delenv(encoder.ENV_CODEC, raising=False)
    monkeypatch.setattr(encoder, "nvenc_available", lambda: calls.append(1) or False)
    assert encoder.choose_codec(gpu=False) == "libx264" and calls == []
    assert encoder.choose_codec(gpu=True) == "libx264" and calls == [1]
    monkeypatch.setattr(encoder, "nvenc_available", lambda: True)
    assert encoder.choose_codec(gpu=True) == "h264_nvenc"
    monkeypatch.setenv(encoder.ENV_CODEC, "libx264")
    assert encoder.choose_codec(gpu=True) == "libx264"
    assert EncodeSettings().preset == "medium"
    assert "-preset" in (a := encoder._codec_args(EncodeSettings(codec="h264_nvenc"), 24)) and "p7" in a


def test_nvenc_probe_uses_real_encode_options(monkeypatch):
    """O teste do NVENC roda com as opções do encode do vídeo (preset p7, AQ temporal...): uma GPU
    que abre o NVENC "puro" mas não tem um desses recursos não pode passar no teste e derrubar o
    FFmpeg no primeiro quadro, já com o render em andamento. O teste pode correr numa thread no
    início do trabalho (o resultado é guardado uma vez por processo)."""
    import subprocess
    import threading

    from mobilia import hardware
    from mobilia.video import encoder

    cmds = []

    class R:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: cmds.append(cmd) or R())
    monkeypatch.setattr(encoder, "_NVENC", {})
    monkeypatch.delenv(encoder.ENV_CODEC, raising=False)
    assert encoder.nvenc_available() is True and len(cmds) == 1
    want = encoder._codec_args(EncodeSettings(codec="h264_nvenc"), 24)
    cmd = cmds[0]
    i = cmd.index("-c:v")
    assert cmd[i:i + len(want)] == want and "-temporal-aq" in cmd and "p7" in cmd
    assert encoder.nvenc_available() is True and len(cmds) == 1          # uma vez por processo
    # sem opções extras, o teste antigo continua disponível
    assert hardware.nvenc_available("ffmpeg") is True and "-temporal-aq" not in cmds[-1]

    monkeypatch.setattr(encoder, "_NVENC", {})
    done = threading.Event()
    monkeypatch.setattr(encoder, "nvenc_available", lambda: done.set())
    monkeypatch.setenv(encoder.ENV_CODEC, "libx264")
    encoder.probe_nvenc_background()                     # codec forçado: nem testa
    assert not done.wait(0.2)
    monkeypatch.delenv(encoder.ENV_CODEC)
    encoder.probe_nvenc_background()
    assert done.wait(5)
