"""Vídeo antes e depois (mobilia/video/showcase.py) e contrato das CLIs ``showcase`` e ``upscale``."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from PIL import Image

from mobilia.video import showcase as S

COLORS = [(200, 60, 50), (40, 150, 70), (50, 80, 200), (210, 180, 40)]


def _psnr(a, b) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    return 10 * np.log10(255.0 ** 2 / max(d.mean(), 1e-9))


def _stripes(w=768, h=512, base=(128, 128, 128)) -> np.ndarray:
    """Imagem que só varia na horizontal: qualquer recorte vertical dela é igual (teste do 1º quadro)."""
    x = np.arange(w, dtype=np.float32)
    v = 0.5 + 0.5 * np.sin(x / 9.0) * np.cos(x / 41.0)
    img = np.empty((h, w, 3), np.float32)
    for c in range(3):
        img[..., c] = base[c] * (0.6 + 0.4 * v)[None]
    return img.astype(np.uint8)


def _flat(color, w=768, h=512) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = color
    img[::16] = 255                       # um pouco de textura
    return img


@pytest.fixture
def imgs():
    return _stripes(), [_flat(c) for c in COLORS[:3]]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_timeline_length_and_frames(n):
    tl = S.plan_timeline(n, fps=30)
    assert S.MIN_S <= tl.total <= S.MAX_S
    assert tl.frames == round(tl.total * 30)
    assert len(tl.shots) == n and tl.shots[-1][1] == pytest.approx(tl.total)
    assert tl.intro[1] == tl.wipe[0] and tl.wipe[1] == tl.shots[0][0]
    for i in range(1, n):
        assert tl.fades[i][0] == pytest.approx(tl.shots[i - 1][1])
        assert tl.fades[i][1] == pytest.approx(tl.shots[i][0])


def test_timeline_errors_and_explicit_duration():
    with pytest.raises(ValueError):
        S.plan_timeline(0)
    assert S.plan_timeline(3, duration=12.0).frames == 360
    with pytest.raises(ValueError):
        S.plan_timeline(6, duration=8.0)       # não cabe


def test_norm_format():
    assert S.norm_format("16:9") == "16x9" and S.norm_format("9:16") == "9x16" and S.norm_format("reels") == "9x16"
    with pytest.raises(ValueError):
        S.norm_format("4x3")


def test_first_frame_is_the_before_photo(imgs):
    before, afters = imgs
    sc = S.Showcase(before, afters, "16x9", show_labels=False, brand=False)
    f0 = sc.frame(0)
    assert f0.shape == (1080, 1920, 3) and sc.n_frames == sc.tl.frames
    # 3:2 -> 16:9 em zoom 1: largura inteira; como a foto só varia na horizontal, o recorte vertical não importa
    ref = cv2.resize(before[:432], (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
    assert _psnr(f0, ref) > 30
    # nenhuma versão aparece no começo
    assert all(abs(f0.mean((0, 1)) - np.array(c)).max() > 25 for c in COLORS[:3])


def test_first_frame_with_labels_keeps_photo(imgs):
    before, afters = imgs
    sc = S.Showcase(before, afters, "16x9")
    f0 = sc.frame(0)
    ref = S.Showcase(before, afters, "16x9", show_labels=False, brand=False).frame(0)
    assert np.array_equal(f0, ref)              # rótulos e marca entram com fade (alfa 0 no 1º quadro)


def test_last_frame_is_last_design(imgs):
    before, afters = imgs
    sc = S.Showcase(before, afters, "16x9", show_labels=False, brand=False)
    last = sc.frame(sc.n_frames - 1)
    assert sc.state((sc.n_frames - 1) / sc.fps) == {"kind": "design", "index": 3}
    # a imagem é a última versão (cor média dela) e não a foto ou outra versão
    assert np.abs(last[200:880].mean((0, 1)) - (np.array(COLORS[2]) * 15 + 255) / 16).max() < 6


def test_wipe_splits_before_and_after(imgs):
    before, afters = imgs
    sc = S.Showcase(before, afters, "16x9", show_labels=False, brand=False)
    t = (sc.tl.wipe_hold[0] + sc.tl.wipe_hold[1]) / 2
    st = sc.state(t)
    assert st == {"kind": "wipe", "pos": 0.5}
    f = sc.frame(int(round(t * sc.fps)))
    right = f[100:-100, 1100:].reshape(-1, 3).mean(0)
    assert np.abs(right - (np.array(COLORS[0]) * 15 + 255) / 16).max() < 6     # depois à direita
    left = f[100:-100, :800].reshape(-1, 3).mean(0)
    assert np.abs(left - (np.array(COLORS[0]) * 15 + 255) / 16).max() > 30     # antes à esquerda
    # linha clara do divisor no meio
    assert f[100:-100, 958:962].mean() > f[100:-100, 900:940].mean()


def test_crossfade_mixes_aligned_images(imgs):
    before, afters = imgs
    sc = S.Showcase(before, afters, "16x9", show_labels=False, brand=False)
    f0, f1 = sc.tl.fades[1]
    i = int(round((f0 + f1) / 2 * sc.fps))
    st = sc.state(i / sc.fps)
    assert st["kind"] == "fade" and 0.3 < st["mix"] < 0.7
    m = sc.frame(i)[200:880].reshape(-1, 3).mean(0)
    a, b = (np.array(COLORS[0]) * 15 + 255) / 16, (np.array(COLORS[1]) * 15 + 255) / 16
    assert np.abs(m - (a * (1 - st["mix"]) + b * st["mix"])).max() < 6


def test_vertical_format_and_camera_motion(imgs):
    before, afters = imgs
    sc = S.Showcase(before, afters, "9x16", show_labels=False, brand=False)
    assert sc.frame(0).shape == (1920, 1080, 3)
    r = [sc.cam.rect(t) for t in np.linspace(0, sc.tl.total, 60)]
    for x0, y0, w, h in r:      # o recorte nunca sai da imagem
        assert x0 >= -1e-6 and y0 >= -1e-6 and x0 + w <= sc.cam.src_w + 1e-6 and y0 + h <= sc.cam.src_h + 1e-6
        assert w / h == pytest.approx(1080 / 1920, rel=1e-3)
    zooms = [1080 / w for _, _, w, _ in r]
    assert zooms[0] == pytest.approx(1.0) and max(zooms) <= sc.cam.zoom_max + 1e-6 and max(zooms) > sc.cam.zoom_max - 0.02
    xs = np.array([x0 for x0, *_ in r])
    assert np.abs(np.diff(xs)).max() < 0.05 * sc.cam.src_w      # movimento lento e contínuo
    assert xs.max() - xs.min() > 0.05 * sc.cam.src_w             # mas atravessa a sala (passeio)


def test_vertical_crop_follows_the_change():
    before = np.full((512, 768, 3), 128, np.uint8)
    after = before.copy()
    after[200:400, 520:700] = (200, 40, 40)      # "móveis novos" à direita
    sc = S.Showcase(before, [after], "9x16", show_labels=False, brand=False)
    assert sc.cam.cx > sc.cam.src_w * 0.55


def test_labels_and_brand_are_drawn(imgs):
    before, afters = imgs
    plain = S.Showcase(before, afters, "16x9", show_labels=False, brand=False)
    deco = S.Showcase(before, afters, "16x9", labels=["Escandinavo", "Luxo", "Boho"])
    i = int(round(plain.tl.shots[0][0] * plain.fps)) + 5
    a, b = plain.frame(i), deco.frame(i)
    assert np.abs(a[-200:, -700:].astype(int) - b[-200:, -700:]).mean() > 5      # rótulo "Depois" embaixo à direita
    assert np.abs(a[:120, -300:].astype(int) - b[:120, -300:]).mean() > 1         # marca 7xAi no canto
    assert np.array_equal(a[300:700, 400:1400], b[300:700, 400:1400])             # o centro da imagem fica limpo
    assert deco.labels == ["Escandinavo", "Luxo", "Boho"]


def test_infer_label(tmp_path):
    d = tmp_path / "job"
    d.mkdir()
    (d / "design.json").write_text(json.dumps({"funcao": "reforma", "estilo": "luxo"}), encoding="utf-8")
    assert S.infer_label(d / "reforma_luxo_1.jpg") == "Luxo"
    assert S.infer_label(tmp_path / "mudar_estilo_escandinavo_2.jpg") == "Escandinavo"
    assert S.infer_label(tmp_path / "foto.jpg") == ""
    (d / "design.json").write_text(json.dumps({"funcao": "evento", "estilo": None, "evento": "natal"}))
    assert S.infer_label(d / "evento_natal_1.jpg") == "Natal"


def test_many_designs_are_capped(imgs):
    before, _ = imgs
    sc = S.Showcase(before, [_flat(COLORS[i % 4]) for i in range(8)], "16x9", show_labels=False, brand=False)
    assert sc.dropped == 2 and len(sc.tl.shots) == S.MAX_DESIGNS and sc.tl.total <= S.MAX_S


def test_cli_showcase_contract(tmp_path, capsys, imgs):
    from mobilia import cli
    from mobilia.video.encoder import decode_frames

    before, afters = imgs
    paths = []
    for k, im in enumerate([before] + afters[:2]):
        p = tmp_path / f"{k}.png"
        Image.fromarray(im).save(p)
        paths.append(str(p))
    out = tmp_path / "video.mp4"
    code = cli.main(["showcase", "--before", paths[0], "--after", *paths[1:], "--out", str(out),
                     "--format", "16x9", "--duracao", "8", "--fps", "10", "--sem-rotulos", "--sem-marca",
                     "--codec", "libx264", "--json"])
    ev = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert code == 0 and out.is_file()
    assert ev[0]["event"] == "progress" and ev[0]["stage"] == "video"
    assert all(e["event"] == "progress" for e in ev[:-1]) and ev[-2]["fraction"] == 1.0
    done = ev[-1]
    assert done["event"] == "done" and done["output"] == str(out) and done["size"] == [1920, 1080]
    assert done["frames"] == 80 and done["fps"] == 10 and done["duration_s"] == 8.0 and done["codec"] == "libx264"
    frames = dict(decode_frames(out, 1920, 1080))
    assert len(frames) == 80
    ref = cv2.resize(before[:432], (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
    assert _psnr(frames[1], ref) > 28
    assert np.abs(frames[80][200:880].reshape(-1, 3).mean(0) - (np.array(COLORS[1]) * 15 + 255) / 16).max() < 8


def test_cli_showcase_errors(tmp_path, capsys):
    from mobilia import cli

    code = cli.main(["showcase", "--before", str(tmp_path / "nao.jpg"), "--after", str(tmp_path / "a.jpg"),
                     "--out", str(tmp_path / "v.mp4"), "--json"])
    ev = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert code == 1 and ev["event"] == "error" and "não encontrada" in ev["message"]
    img = tmp_path / "b.png"
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(img)
    code = cli.main(["showcase", "--before", str(img), "--after", str(img), "--out", str(tmp_path / "v.mp4"),
                     "--format", "4x3", "--json"])
    assert code == 1 and json.loads(capsys.readouterr().out.splitlines()[-1])["event"] == "error"


def test_cli_upscale_alias(monkeypatch, capsys, tmp_path):
    from mobilia import cli
    from mobilia.design import upscale as U

    img = tmp_path / "v.jpg"
    Image.fromarray(np.full((64, 96, 3), 120, np.uint8)).save(img)
    seen = {}

    def fake(image, out_path=None, long_side=3840, device="auto", progress=None, **k):
        seen["lado"] = long_side
        progress(0.5)
        progress(1.0)
        Image.fromarray(np.zeros((256, 384, 3), np.uint8)).save(out_path)
        return None, {"para": [384, 256], "s": 0.1}

    monkeypatch.setattr(U, "upscale", fake)
    code = cli.main(["upscale", "--image", str(img), "--out", str(tmp_path / "o.jpg"), "--json"])
    ev = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert code == 0 and seen["lado"] == 3840
    assert [e["event"] for e in ev] == ["progress", "progress", "progress", "done"]
    assert ev[0]["fraction"] == 0.0 and all(e["stage"] == "ampliar" for e in ev[:-1])
    assert ev[-1] == {"event": "done", "output": str(tmp_path / "o.jpg"), "size": [384, 256], "s": 0.1}


def test_wipe_labels_never_cut_by_divider(imgs):
    before, afters = imgs
    sc = S.Showcase(before, afters, "16x9", labels=["Escandinavo", "Luxo", "Boho"], brand=False)
    plain = S.Showcase(before, afters, "16x9", labels=["Escandinavo", "Luxo", "Boho"], brand=False)
    orig = plain._label
    plain._label = lambda frame, idx, alpha: None if idx == 0 else orig(frame, idx, alpha)
    bx1 = sc.pos_before[0] + sc.lab_before.shape[1]
    checked = False
    for i in range(int(sc.tl.wipe[0] * sc.fps), int(sc.tl.wipe[1] * sc.fps)):
        st = sc.state(i / sc.fps)
        x = int(round(st["pos"] * sc.W))
        if sc.pos_before[0] < x <= bx1 + 5:     # a linha passando sobre o rótulo "Antes": ele já sumiu
            f, p = sc.frame(i), plain.frame(i)
            y0 = sc.pos_before[1]
            assert np.abs(f[y0:, :bx1].astype(int) - p[y0:, :bx1].astype(int)).max() <= 2
            checked = True
    assert checked


def test_duplicate_labels_get_version_suffix():
    assert S.unique_labels(["Contemporâneo", "Contemporâneo", "Luxo", ""]) == [
        "Contemporâneo", "Contemporâneo · versão 2", "Luxo", ""]


def test_4x3_source_is_padded_not_cropped_hard():
    before = _stripes()
    before = cv2.resize(before, (640, 480))
    sc = S.Showcase(before, [_flat(COLORS[0], 640, 480)], "16x9", show_labels=False, brand=False)
    assert sc.cam.src_w / sc.cam.src_h == pytest.approx(S.PAD_TO_ASPECT, rel=0.01)
