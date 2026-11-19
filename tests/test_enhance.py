"""'Melhorar foto' (``mobilia/design/enhance.py``), 'Exportar para anúncio' e 'Lote' (``mobilia/design/listing.py``).

Tudo com imagens sintéticas e com as fotos CC0 de ``tests/fixtures``; segmentação e removedor de objetos são dublês
(o CI não tem PyTorch)."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from mobilia.design import enhance as E
from mobilia.design import listing as L

FX = Path(__file__).parent / "fixtures"
PHOTOS = sorted((FX / "photos").glob("*.jpg")) + sorted((FX / "rooms").glob("*/photo.jpg"))


def _rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def _lum(img):
    return float(cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0].mean())


def _tilt(img, pitch_deg=5.0, roll_deg=1.5):
    H, W = img.shape[:2]
    f = 0.7 * max(W, H)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    R = cv2.Rodrigues(np.array([math.radians(pitch_deg), 0, math.radians(roll_deg)]))[0]
    return cv2.warpPerspective(img, K @ R @ np.linalg.inv(K), (W, H), borderMode=cv2.BORDER_REFLECT)


def _lines_room(W=960, H=720):
    """Parede com batentes e cantos verticais (linhas fortes) e piso: o caso típico das verticais."""
    img = np.full((H, W, 3), 225, np.uint8)
    img[int(H * 0.7):] = (150, 120, 90)
    for x in (80, 230, 260, 470, 640, 670, 880):
        cv2.line(img, (x, 40), (x, int(H * 0.7)), (60, 60, 60), 4)
    cv2.rectangle(img, (300, 120), (430, 300), (90, 90, 95), 3)
    return img


# ================================================================================== exposição / tom
def test_auto_tone_brightens_dark_photo_and_keeps_good_one():
    img = _rgb(PHOTOS[0])
    dark = E._u8(E.linear_to_srgb(E.srgb_to_linear(img / 255.0) * 0.35))
    out, info = E.auto_tone(dark)
    assert info["exposicao_ev"] > 0.4 and _lum(out) > _lum(dark) + 8
    good, ginfo = E.auto_tone(img)
    assert abs(_lum(good) - _lum(img)) < 10        # foto boa: quase nada muda
    assert ginfo["exposicao_ev"] >= -0.2


def test_auto_tone_never_darkens_bright_airy_photo():
    img = np.full((300, 400, 3), 215, np.uint8)
    img[100:200, 100:300] = (180, 170, 160)
    _, info = E.auto_tone(img)
    assert info["exposicao_ev"] >= 0.0


def test_auto_tone_compresses_window_without_halo():
    """Janela clara (não estourada) numa sala escura: a janela desce, a sala sobe, e não aparece halo escuro na
    parede logo ao lado da janela."""
    H, W = 400, 600
    lin = np.full((H, W, 3), 0.03, np.float32)
    lin[100:300, 350:500] = 0.85                  # janela
    rng = np.random.default_rng(0)
    lin *= 1 + 0.05 * rng.standard_normal((H, W, 1)).astype(np.float32)
    img = E._u8(E.linear_to_srgb(lin))
    out, info = E.auto_tone(img)
    room = (slice(150, 250), slice(50, 250))
    assert _lum(out[room]) > _lum(img[room]) + 10
    assert info["exposicao_ev"] > 0.3
    assert out[150:250, 380:470].mean() < 250            # a janela clareou junto, mas não estourou
    # sem halo: a faixa da parede colada na janela não fica mais escura que a parede longe dela
    near = out[150:250, 330:345].astype(np.float32).mean()
    far = out[150:250, 150:250].astype(np.float32).mean()
    assert near >= far - 6


def test_auto_tone_keeps_blown_white_white():
    img = np.full((200, 300, 3), 60, np.uint8)
    img[50:150, 100:200] = 255
    out, _ = E.auto_tone(img)
    assert out[80:120, 130:170].min() >= 245


def test_auto_tone_zero_strength_is_identity():
    img = _rgb(PHOTOS[0])
    out, _ = E.auto_tone(img, 0.0)
    assert np.array_equal(out, img)


# ================================================================================== branco
def test_white_balance_removes_yellow_cast_on_walls():
    img = np.full((240, 320, 3), 200, np.uint8)
    img[160:] = (120, 90, 60)                      # piso de madeira
    cast = E._u8(E.linear_to_srgb(E.srgb_to_linear(img / 255.0) * np.array([1.0, 0.92, 0.7], np.float32)))
    walls = np.zeros(img.shape[:2], bool); walls[:160] = True
    out, info = E.white_balance(cast, walls)
    lab_in = cv2.cvtColor(cast, cv2.COLOR_RGB2LAB).astype(np.float32)[:160]
    lab_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)[:160]
    assert info["aplicado"] and info["referencia"] == "superficies"
    assert abs(lab_out[..., 2].mean() - 128) < abs(lab_in[..., 2].mean() - 128) - 5
    assert abs(_lum(out) - _lum(cast)) < 4           # o brilho não muda


def test_white_balance_leaves_neutral_photo_and_limits_strong_color():
    neutral = np.full((100, 100, 3), 200, np.uint8)
    out, info = E.white_balance(neutral)
    assert not info["aplicado"] and np.array_equal(out, neutral)
    beige = np.zeros((100, 100, 3), np.uint8); beige[:] = (225, 205, 170)   # parede bege de verdade
    out, info = E.white_balance(beige)
    if info.get("aplicado"):
        assert info["corrigido"] <= 9.0 + 1e-6
        b_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[..., 2].astype(float).mean() - 128
        assert b_out > 3                                  # continua bege, não vira cinza


# ================================================================================== verticais
def test_correct_verticals_straightens_tilted_room():
    img = _tilt(_lines_room(), 6.0, 1.5)
    out, info = E.correct_verticals(img)
    assert info.aplicado and abs(info.inclinacao_graus) > 3 and info.escala_recorte >= E.MIN_CROP_SCALE
    assert out.shape == img.shape
    seg = E.vertical_lines(out)
    dev = np.degrees(np.arctan2(np.abs(seg[:, 2] - seg[:, 0]), np.abs(seg[:, 3] - seg[:, 1])))
    w = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    assert np.average(dev, weights=w) < 0.8


def test_correct_verticals_keeps_straight_photo_and_refuses_low_confidence():
    straight = _lines_room()
    out, info = E.correct_verticals(straight)
    assert not info.aplicado and np.array_equal(out, straight)
    noise = np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    out, info = E.correct_verticals(noise)
    assert not info.aplicado
    steep = _tilt(_lines_room(), 25.0, 0.0)
    out, info = E.correct_verticals(steep)
    assert not info.aplicado and "grande" in info.motivo


@pytest.mark.parametrize("photo", PHOTOS[:2], ids=lambda p: p.stem)
def test_correct_verticals_on_real_photos_after_synthetic_tilt(photo):
    img = _rgb(photo)
    _, base = E.correct_verticals(img)
    tilted = _tilt(img, 4.0, 1.0)
    _, info = E.correct_verticals(tilted)
    assert info.aplicado
    # a correção medida acompanha a inclinação aplicada (mesma convenção, ±1,5°)
    assert abs((info.inclinacao_graus - base.inclinacao_graus) - 4.0) < 1.5


def test_warp_like_follows_correction():
    img = _tilt(_lines_room(), 6.0, 0.0)
    _, info = E.correct_verticals(img)
    m = np.zeros(img.shape[:2], np.uint8); m[300:400, 400:500] = 1
    w = E.warp_like(m, info)
    assert w.shape == m.shape and 0 < w.sum() and set(np.unique(w)) <= {0, 1}


# ================================================================================== céu
def test_sky_image_is_blue_gradient_and_deterministic():
    a = E.sky_image(320, 200, 0)
    b = E.sky_image(320, 200, 0)
    assert np.array_equal(a, b) and a.shape == (200, 320, 3)
    top, bottom = a[:20].reshape(-1, 3).mean(0), a[-20:].reshape(-1, 3).mean(0)
    assert top[2] > top[0] + 40                      # azul no zênite
    assert bottom.mean() > top.mean()                # horizonte mais claro
    assert len({E.sky_image(64, 64, k).tobytes() for k in range(len(E.SKY_PRESETS))}) == len(E.SKY_PRESETS)


def _facade():
    H, W = 360, 480
    img = np.zeros((H, W, 3), np.uint8)
    img[:150] = (232, 234, 236)                      # céu nublado claro
    img[150:] = (150, 110, 90)                       # casa
    cv2.rectangle(img, (180, 190), (300, 300), (60, 50, 45), -1)
    sky = np.zeros((H, W), bool); sky[:148] = True
    return img, sky


def test_replace_sky_only_touches_sky():
    img, sky = _facade()
    out, info = E.replace_sky(img, sky, 1)
    assert info["aplicado"]
    assert np.array_equal(out[160:], img[160:])      # nada fora do céu (nem perto da borda) muda
    s = out[10:120].reshape(-1, 3).astype(float).mean(0)
    assert s[2] > s[0] + 25                          # ficou azul
    assert np.abs(out.astype(int) - img.astype(int))[:140].mean() > 10


def test_replace_sky_skips_without_sky_or_with_suspicious_mask():
    img, sky = _facade()
    out, info = E.replace_sky(img, np.zeros_like(sky), 0)
    assert not info["aplicado"] and np.array_equal(out, img)
    out, info = E.replace_sky(img, np.ones_like(sky), 0)
    assert not info["aplicado"]


def test_sky_mask_from_labels():
    labels = np.zeros((10, 20), np.uint8); labels[:4] = 2
    m = E.sky_mask_from_labels(labels, {0: "wall", 2: "sky"}, (40, 20))
    assert m.shape == (20, 40) and m[:8].all() and not m[8:].any()


# ================================================================================== ruído / nitidez
def test_denoise_reduces_color_noise_and_sharpen_respects_threshold():
    rng = np.random.default_rng(3)
    base = np.full((200, 300, 3), 128, np.float32)
    base[:, 150:] = 60
    noisy = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    out, info = E.denoise(noisy)
    ycc_in = cv2.cvtColor(noisy, cv2.COLOR_RGB2YCrCb).astype(float)
    ycc_out = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb).astype(float)
    assert ycc_out[..., 1].std() < 0.6 * ycc_in[..., 1].std()
    assert info["ruido_sigma"] > 2
    flat = np.full((100, 100, 3), 128, np.uint8)
    assert np.array_equal(E.sharpen(flat), flat)     # liso: nada (limiar)
    edge = np.zeros((100, 100, 3), np.uint8); edge[:, 50:] = 200
    edge = cv2.GaussianBlur(edge, (0, 0), 2)
    sh = E.sharpen(edge, 1.0)
    def g(im):
        return np.abs(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F)).sum()
    assert g(sh) > 1.03 * g(edge)


# ================================================================================== remover objetos
def test_remove_objects_changes_only_the_painted_area():
    img = _rgb(PHOTOS[0])
    img = cv2.resize(img, (640, 480))
    mask = np.zeros(img.shape[:2], bool); mask[300:340, 200:260] = True
    calls = []

    def fake(rgb, m):
        calls.append(rgb.shape)
        out = rgb.copy(); out[m] = (0, 255, 0)
        return out

    out, info = E.remove_objects(img, mask, inpainter=fake)
    assert info["aplicado"] and info["areas"] == 1 and calls
    diff = np.abs(out.astype(int) - img.astype(int)).sum(-1) > 0
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    assert diff[mask].all() and not diff[~grown].any()
    same, info = E.remove_objects(img, np.zeros_like(mask), inpainter=fake)
    assert not info["aplicado"] and np.array_equal(same, img)


def test_remove_objects_falls_back_to_classic_fill_without_weights(monkeypatch):
    import mobilia.removal.plate as plate

    def boom(*a, **k):
        raise RuntimeError("sem pesos")

    monkeypatch.setattr(plate, "load_lama", boom)
    img = np.full((120, 160, 3), 180, np.uint8); img[50:70, 70:90] = 0
    m = np.zeros(img.shape[:2], bool); m[50:70, 70:90] = True
    out, info = E.remove_objects(img, m)
    assert info["metodo"] == "telea" and out[55:65, 75:85].mean() > 150


# ================================================================================== função completa
def _fake_seg(sky_rows=0):
    def fn(rgb):
        H, W = rgb.shape[:2]
        lab = np.full((H // 2, W // 2), 1, np.uint8)       # parede
        lab[: (H // 2) // 6] = 3                           # teto
        if sky_rows:
            lab[:sky_rows] = 2
        return lab, {1: "wall", 2: "sky", 3: "ceiling"}
    return fn


def test_enhance_end_to_end_writes_outputs(tmp_path):
    photo = PHOTOS[0]
    mask = np.zeros(_rgb(photo).shape[:2], np.uint8); mask[500:560, 600:700] = 255
    cv2.imwrite(str(tmp_path / "m.png"), mask)
    events = []
    res = E.enhance(str(photo), tmp_path / "out", mask=str(tmp_path / "m.png"), progress=events.append,
                    seg_fn=_fake_seg(), inpainter=lambda rgb, m: cv2.inpaint(rgb, m.astype(np.uint8) * 255, 3,
                                                                              cv2.INPAINT_TELEA), device="cpu")
    ev = res.to_event()
    assert ev["funcao"] == "melhorar" and Path(ev["output"]).is_file() and Path(ev["before"]).is_file()
    out = cv2.imread(ev["output"]); src = cv2.imread(str(photo))
    assert out.shape == src.shape                         # resolução da foto
    assert cv2.imread(ev["before"]).shape == out.shape
    rep = json.loads(Path(ev["report"]).read_text(encoding="utf-8"))
    assert rep["remocao"]["aplicado"] and "tom" in rep and "verticais" in rep and rep["branco"]["referencia"]
    assert any(e.get("event") == "preview" for e in events)
    fr = [e["fraction"] for e in events if e.get("event") == "progress"]
    assert fr == sorted(fr) and fr[-1] <= 1.0


def test_enhance_options_switch_steps_off(tmp_path):
    img = _rgb(PHOTOS[0])
    off = E.EnhanceOptions(exposicao=False, branco=False, verticais=False, ceu=False, nitidez=False)
    out, rep = E.enhance_image(img, off)
    assert np.array_equal(out, img) and "tom" not in rep and "verticais" not in rep


def test_enhance_4k_uses_upscaler_and_falls_back(tmp_path):
    img = cv2.resize(_rgb(PHOTOS[0]), (400, 300))
    cv2.imwrite(str(tmp_path / "p.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    o = E.EnhanceOptions(ceu=False, ampliar_4k=True)
    up = lambda rgb, side: (cv2.resize(rgb, (side, int(side * rgb.shape[0] / rgb.shape[1]))), {"s": 0})  # noqa: E731
    res = E.enhance(str(tmp_path / "p.jpg"), tmp_path / "a", o, upscaler=up, device="cpu")
    assert res.size == (3840, 2880)

    def broken(rgb, side):
        raise RuntimeError("sem modelo")

    events = []
    res = E.enhance(str(tmp_path / "p.jpg"), tmp_path / "b", o, upscaler=broken, device="cpu",
                    progress=events.append)
    assert res.size == (3840, 2880) and any(e.get("level") == "warning" for e in events)


def test_enhance_sky_from_segmentation(tmp_path):
    img, _ = _facade()
    big = cv2.resize(img, (960, 720), interpolation=cv2.INTER_NEAREST)
    out, rep = E.enhance_image(big, E.EnhanceOptions(verticais=False), seg_fn=_fake_seg(sky_rows=100))
    assert rep["ceu"]["aplicado"]
    s = out[20:150].reshape(-1, 3).astype(float).mean(0)
    assert s[2] > s[0] + 20


def test_enhance_runs_on_all_cc0_photos_quickly():
    import time
    for p in PHOTOS:
        img = _rgb(p)
        t = time.perf_counter()
        out, rep = E.enhance_image(img, E.EnhanceOptions(ceu=False))
        dt = time.perf_counter() - t
        assert out.shape == img.shape and dt < 20
        # natural: o brilho médio não pula mais que ~15 níveis numa foto já boa
        assert abs(_lum(out) - _lum(img)) < 15, p


# ================================================================================== presets / api / cli
def test_preset_and_catalog():
    from mobilia.design import api, presets
    f = presets.function("melhorar_foto")
    assert f.key == "melhorar" and not f.generative and presets.catalog()["funcoes"]["melhorar"] == "Melhorar foto"
    assert "melhorar" not in presets.ILLUSTRATIVE and "mobiliar" in presets.ILLUSTRATIVE
    with pytest.raises(ValueError, match="enhance"):
        api.redesign(np.zeros((64, 64, 3), np.uint8), "melhorar", engine=object())


def test_cli_melhorar_exportar_and_lote(tmp_path, capsys):
    from mobilia import cli
    photo = str(PHOTOS[0])
    assert cli.main(["design", "--funcao", "melhorar", "--photo", photo, "--sem-ceu", "--out-dir",
                     str(tmp_path / "m"), "--json", "--device", "cpu"]) == 0
    done = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")][-1]
    assert done["event"] == "done" and done["funcao"] == "melhorar" and Path(done["output"]).is_file()
    assert cli.main(["exportar", "--images", done["output"], "--out-dir", str(tmp_path / "an"), "--funcao",
                     "mobiliar", "--json"]) == 0
    ev = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")][-1]
    assert ev["rotulo"] is True and ev["lado"] == 2048
    assert max(cv2.imread(ev["output"]).shape[:2]) == 2048
    bad = str(tmp_path / "nao_existe.jpg")
    assert cli.main(["design-lote", "--fotos", photo, bad, "--imovel", "Casa", "--json"]) == 1
    ok = cli.main(["design-lote", "--fotos", *map(str, PHOTOS[:2]), "--imovel", "Casa 1", "--sem-ceu",
                   "--out-dir", str(tmp_path / "Casa 1"), "--exportar", "2048", "--json", "--device", "cpu"])
    evs = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert ok == 0
    items = [e for e in evs if e["event"] == "item" and e["status"] == "pronto"]
    done = evs[-1]
    assert len(items) == 2 and done["event"] == "done" and len(done["anuncio"]) == 2
    assert all(Path(p).parent == tmp_path / "Casa 1" for p in done["outputs"])


# ================================================================================== exportar / lote
def test_add_label_is_discreet_and_in_the_corner():
    img = np.full((1000, 1500, 3), 200, np.uint8)
    out = L.add_label(img)
    diff = np.abs(out.astype(int) - img.astype(int)).sum(-1) > 0
    ys, xs = np.nonzero(diff)
    assert ys.min() > 900 and xs.min() > 1100       # só no canto inferior direito
    assert diff.mean() < 0.01


def test_export_sizes_and_label_default(tmp_path):
    src = tmp_path / "v.jpg"
    cv2.imwrite(str(src), np.full((900, 1200, 3), 120, np.uint8))
    p = L.export_for_listing([str(src)], tmp_path / "a", 2048, label=False)
    assert cv2.imread(p[0]).shape[:2] == (1536, 2048)
    calls = []
    up = lambda rgb, side: calls.append(side) or cv2.resize(rgb, (side, side * 3 // 4))  # noqa: E731
    p = L.export_for_listing([str(src)], tmp_path / "a", 3840, label=True, upscaler=up)
    assert calls == [3840] and cv2.imread(p[0]).shape[:2] == (2880, 3840) and p[0].endswith("_anuncio_2.jpg")
    big = tmp_path / "big.jpg"
    cv2.imwrite(str(big), np.full((3000, 4000, 3), 120, np.uint8))
    assert cv2.imread(L.export_for_listing([str(big)], tmp_path / "b", 2048)[0]).shape[:2] == (1536, 2048)
    assert L.default_label("mudar_estilo") and not L.default_label("melhorar")
    assert L.norm_long_side("4K") == 3840 and L.norm_long_side("2048") == 2048
    with pytest.raises(ValueError):
        L.norm_long_side("8k")


def test_property_folder_is_safe():
    assert L.property_folder('Apto 302 / Rua "X": bloco B?') == "Apto 302 Rua X bloco B"
    assert L.property_folder("  ") == "imovel"
    assert L.property_folder("CON") == "_CON"
    assert L.property_folder("Casa à venda – Praia") == "Casa à venda – Praia"


def test_run_batch_keeps_going_after_a_failure(tmp_path):
    from mobilia.design.api import DesignResult
    photos = []
    for k in range(3):
        p = tmp_path / f"f{k}.jpg"; cv2.imwrite(str(p), np.full((60, 80, 3), 50 * k, np.uint8)); photos.append(str(p))
    events = []

    def proc(photo, out, progress):
        if photo.endswith("f1.jpg"):
            raise RuntimeError("foto ruim")
        progress({"event": "progress", "fraction": 0.5, "message": "meio"})
        Path(out).mkdir(parents=True, exist_ok=True)
        o = Path(out) / "melhorar_1.jpg"
        cv2.imwrite(str(o), cv2.imread(photo))
        return DesignResult([str(o)], None, None, "melhorar", None, "sala", [], (80, 60))

    rep = L.run_batch(photos, tmp_path / "Casa", "melhorar", progress=events.append, process=proc,
                      export={"lado": 2048, "rotulo": None})
    st = [i["status"] for i in rep["itens"]]
    assert st == ["pronto", "erro", "pronto"]
    assert sorted(Path(p).name for p in rep["entregas"]) == ["f0_melhorar.jpg", "f2_melhorar.jpg"]
    assert len(rep["anuncio"]) == 2 and (tmp_path / "Casa" / "lote.json").is_file()
    fr = [e["fraction"] for e in events if e["event"] == "progress"]
    assert fr == sorted(fr)
    with pytest.raises(ValueError):
        L.run_batch([], tmp_path / "x", process=proc)
