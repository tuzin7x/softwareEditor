"""Composição em fluxo (PT5, ``mobilia/composite/stream.py``).

Referência: o laço antigo do ``compose_video`` (quadro a quadro, ``LayerAccumulator`` + composição
do quadro inteiro). O fluxo tem de entregar ao codificador exatamente os mesmos bytes, na mesma
ordem, compondo cada EXR uma vez só (reaproveitados vêm do cache).
"""
import hashlib
import time

import numpy as np
import pytest

H, W = 72, 96


def _write_exr(path, border, shift=0, second=False):
    """EXR recortado no border, como o nó File Output do worker (Combined + Shadow Catcher, RLE)."""
    import OpenEXR

    x0, y0, x1, y1 = border
    rgba = np.zeros((H, W, 4), np.float16)
    sh = np.ones((H, W, 3), np.float16)
    rgba[30:50, 20 + shift:44 + shift] = [0.3, 0.2, 0.1, 1.0]
    rgba[29, 20 + shift:44 + shift] = [0.1, 0.05, 0.02, 0.5]
    sh[50:56, 18 + shift:48 + shift] = 0.6
    if second:
        rgba[10:22, 70:84] = [0.1, 0.4, 0.2, 1.0]
        sh[22:25, 68:86] = 0.7
    ch = {f"Combined.{c}": np.ascontiguousarray(rgba[y0:y1, x0:x1, i]) for i, c in enumerate("RGBA")}
    ch.update({f"Shadow Catcher.{c}": np.ascontiguousarray(sh[y0:y1, x0:x1, i]) for i, c in enumerate("RGB")})
    with OpenEXR.File({"compression": OpenEXR.RLE_COMPRESSION, "type": OpenEXR.scanlineimage}, ch) as f:
        f.write(str(path))


def _job(tmp_path, mode="full"):
    plate8 = np.random.default_rng(1).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    frames_dir = tmp_path / "render" / "frames"
    frames_dir.mkdir(parents=True)
    b1, b2, b3 = [10, 20, 60, 64], [14, 20, 70, 64], [12, 4, 92, 64]
    _write_exr(frames_dir / "f_00002.exr", b1)
    _write_exr(frames_dir / "f_00004.exr", b2, shift=6)
    _write_exr(frames_dir / "f_00007.exr", b3, shift=10, second=True)
    m = {"mode": mode, "crop": True}
    frames = [{"frame": 1, "source": None, "border": None},
              {"frame": 2, "source": "f_00002.exr", "border": b1, **m},
              {"frame": 3, "source": "f_00002.exr", "border": b1, "reused": True},
              {"frame": 4, "source": "f_00004.exr", "border": b2, **m},
              {"frame": 5, "source": "f_00004.exr", "border": b2, "reused": True},
              {"frame": 6, "source": "f_00004.exr", "border": b2, "reused": True},
              {"frame": 7, "source": "f_00007.exr", "border": b3, **m},
              {"frame": 8, "source": "f_00007.exr", "border": b3, "reused": True}]
    return plate8, {"exr": {"recorte": True, "codec": "RLE"}, "frames": frames}


def _md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _reference(tmp_path, plate8, meta):
    """O laço antigo, quadro a quadro, no quadro inteiro."""
    from mobilia.composite.compositor import Compositor, FrameStats, LayerAccumulator, load_render_passes

    comp = Compositor(plate8, seed=7)
    acc = LayerAccumulator(H, W, feather_px=max(6, W // 120),
                           shadow_feather_px=max(16, int(0.04 * float(np.hypot(W, H)))))
    out, loaded = [], None
    for fr in meta["frames"]:
        src = fr.get("source")
        if src is None:
            acc.reset()
            out.append(_md5(plate8))
            continue
        if src != loaded and not fr.get("reused"):
            rgba, sh = load_render_passes(tmp_path / "render" / "frames" / src, fr["border"], (H, W))
            acc.update(rgba, sh, fr["border"], fr.get("mode", "full"))
            loaded = src
        o = comp._composite_region(FrameStats(fr["frame"]), acc.rgba, acc.shadow, None, [0, 0, W, H])
        out.append(_md5(o))
    return out


@pytest.fixture()
def capture(monkeypatch):
    """Hashes dos quadros na ordem em que chegam ao FFmpeg + contagem de composições."""
    from mobilia.composite.compositor import Compositor
    from mobilia.video.encoder import VideoWriter

    got = {"frames": [], "composite": 0}
    orig_w, orig_c = VideoWriter.write, Compositor.composite

    def write(self, fr):
        got["frames"].append(_md5(fr))
        return orig_w(self, fr)

    def composite(self, *a, **kw):
        got["composite"] += 1
        return orig_c(self, *a, **kw)

    monkeypatch.setattr(VideoWriter, "write", write)
    monkeypatch.setattr(Compositor, "composite", composite)
    return got


def test_stream_matches_reference_and_reuses_cache(tmp_path, monkeypatch, capture):
    """Mesmos bytes do laço antigo, em ordem, mesmo com as tarefas terminando fora de ordem; cada
    EXR é composto uma vez só (os reaproveitados vêm do cache)."""
    from mobilia import pipeline
    from mobilia.composite import stream
    from mobilia.jobs import ProgressEmitter

    plate8, meta = _job(tmp_path)
    ref = _reference(tmp_path, plate8, meta)
    orig = stream.load_layers
    delays = {"f_00002.exr": 0.25, "f_00004.exr": 0.1, "f_00007.exr": 0.0}   # o primeiro termina por último

    def slow_load(path, *a, **kw):
        time.sleep(delays[path.name])
        return orig(path, *a, **kw)

    monkeypatch.setattr(stream, "load_layers", slow_load)
    res = pipeline.compose_video(tmp_path, plate8, meta, tmp_path / "v.mp4", 12, 7, ProgressEmitter(), workers=3)
    assert capture["frames"] == ref
    assert capture["composite"] == 3                       # 3 EXR, 8 quadros
    c = res["composicao"]
    assert (c["quadros_compostos"], c["quadros_cache"], c["quadros_so_foto"]) == (3, 4, 1)
    assert c["trabalhadores"] == 3 and c["paralela"] is False and c["exr_recortado"] is True
    assert [p["cache"] for p in c["por_quadro"]] == [False, False, True, False, True, True, False, True]
    assert 0 < c["recorte_composicao_medio_px"] < W * H
    assert res["preservacao"]["preservado_exato_todos"] and res["preservacao"]["quadros_verificados"] == 8
    assert [v["posicao"] for v in res["encode"]["verificacao"]] == [1, 4, 8]


def test_incremental_mode_runs_in_order(tmp_path, capture):
    """O render incremental (experimental) acumula camadas em ordem: o fluxo usa uma thread só e
    o acumulador compartilhado, com o mesmo resultado do laço antigo."""
    from mobilia import pipeline
    from mobilia.jobs import ProgressEmitter

    plate8, meta = _job(tmp_path, mode="incremental")
    ref = _reference(tmp_path, plate8, meta)
    res = pipeline.compose_video(tmp_path, plate8, meta, tmp_path / "v.mp4", 12, 7, ProgressEmitter())
    assert res["composicao"]["sequencial"] is True and res["composicao"]["trabalhadores"] == 1
    assert capture["frames"] == ref


def test_reused_after_photo_frame_comes_from_cache(tmp_path, capture):
    """Quadro reaproveitado depois de um quadro só-foto: é o quadro composto do EXR (do cache), não
    a foto (o laço antigo zerava as camadas no quadro só-foto e não relia o EXR)."""
    from mobilia import pipeline
    from mobilia.jobs import ProgressEmitter

    plate8, meta = _job(tmp_path)
    b1 = meta["frames"][1]["border"]
    meta["frames"] = [meta["frames"][1], {"frame": 2, "source": None, "border": None},
                      {"frame": 3, "source": "f_00002.exr", "border": b1, "reused": True}]
    pipeline.compose_video(tmp_path, plate8, meta, tmp_path / "v.mp4", 12, 7, ProgressEmitter())
    f = capture["frames"]
    assert f[0] == f[2] != f[1] == _md5(plate8) and capture["composite"] == 1


def test_stream_push_during_render_and_reconcile(tmp_path):
    """No fluxo os quadros são compostos e escritos enquanto chegam; no fim o ``frames.json`` tem
    de bater com o que foi anunciado (o que faltar é entregue; divergência é erro)."""
    from mobilia.composite.stream import StreamComposer
    from mobilia.jobs import ProgressEmitter

    plate8, meta = _job(tmp_path)
    frames = meta["frames"]
    sc = StreamComposer(tmp_path, plate8, tmp_path / "v.mp4", 12, 7, ProgressEmitter(), streaming=True,
                        n_frames=len(frames), workers=2)
    for fr in frames[:5]:
        sc.push(dict(fr))
    time.sleep(0.5)                                     # "o Blender ainda renderiza"
    res = sc.finish(meta)                               # entrega os 3 que faltam
    c = res["composicao"]
    assert c["paralela"] is True and 1 <= c["compostos_durante_render"] <= 5
    assert res["preservacao"]["quadros"] == len(frames) and (tmp_path / "v.mp4").exists()

    sc = StreamComposer(tmp_path, plate8, tmp_path / "w.mp4", 12, 7, ProgressEmitter(), streaming=True)
    for fr in frames[:3]:
        sc.push(dict(fr))
    bad = {"frames": [dict(f) for f in frames]}
    bad["frames"][1]["border"] = [0, 0, W, H]
    with pytest.raises(RuntimeError, match="não batem"):
        sc.finish(bad)
    sc.abort()
    assert not (tmp_path / "w.mp4").exists()


def test_stream_failure_stops_and_keeps_exr(tmp_path, monkeypatch):
    """Um quadro que muda 1 byte da foto longe dos móveis derruba o trabalho naquele quadro: nada de
    MP4, e os EXR daquele quadro em diante ficam no disco para diagnóstico."""
    from mobilia import pipeline
    from mobilia.composite.compositor import Compositor
    from mobilia.composite.stream import PreservationError
    from mobilia.jobs import ProgressEmitter

    plate8, meta = _job(tmp_path)
    orig = Compositor.composite

    def buggy(self, frame, *a, **kw):
        out, st = orig(self, frame, *a, **kw)
        if frame == 4:
            out = out.copy()
            out[H - 1, W - 1, 0] ^= 4                    # canto oposto aos móveis
        return out, st

    monkeypatch.setattr(Compositor, "composite", buggy)
    with pytest.raises(PreservationError, match="quadro 4"):
        pipeline.compose_video(tmp_path, plate8, meta, tmp_path / "v.mp4", 12, 7, ProgressEmitter(),
                               delete_exr=True, workers=3)
    d = tmp_path / "render" / "frames"
    assert not (d / "f_00002.exr").exists()             # já escrito e conferido antes da falha
    assert (d / "f_00004.exr").exists() and (d / "f_00007.exr").exists()
    assert not (tmp_path / "v.mp4").exists() and not list(tmp_path.glob("v.partial*"))


def test_plate_is_read_only_for_the_compositor(tmp_path, monkeypatch):
    """A referência da verificação é o mesmo buffer que o compositor lê (e os quadros só-foto são
    ele): uma escrita nele em memória mudaria tudo junto sem nenhuma conferência perceber. A placa
    do fluxo é uma cópia só de leitura: o trabalho para, sem MP4, e a foto de quem chamou fica
    intacta."""
    from mobilia import pipeline
    from mobilia.composite.compositor import Compositor
    from mobilia.jobs import ProgressEmitter

    plate8, meta = _job(tmp_path)
    before = plate8.copy()
    orig = Compositor.composite

    def buggy(self, frame, *a, **kw):
        out, st = orig(self, frame, *a, **kw)
        self.photo8[H - 1, W - 1] = 0                    # "otimização" que escreve na placa
        return out, st

    monkeypatch.setattr(Compositor, "composite", buggy)
    with pytest.raises(RuntimeError, match="read-only|somente leitura"):
        pipeline.compose_video(tmp_path, plate8, meta, tmp_path / "v.mp4", 12, 7, ProgressEmitter())
    assert np.array_equal(plate8, before) and not (tmp_path / "v.mp4").exists()


def _calibration():
    """Calibração com o ajuste do CG (PT8): mapa de ganho, curva de tom e joelho."""
    yy, xx = np.mgrid[0:4, 0:6]
    grid = np.stack([0.86 + 0.3 * xx / 5, 1.15 - 0.25 * yy / 3, np.full((4, 6), 0.95)], -1)
    return {"rel_rms": 0.12, "ajuste_cg": {
        "versao": 1, "joelho": 0.8,
        "ganho": {"modo": "mapa", "limites": [0.85, 1.2], "grade": grid.tolist(), "forma": [4, 6, 3]},
        "curva_tom": {"ok": True, "x_log2": [-4.0, -2.0, -1.0], "y_log2": [-4.1, -2.2, -0.9]}}}


def test_stream_applies_calibration_like_recompose(tmp_path, capture):
    """PT8: o ajuste do CG chega pelo evento da calibração (``set_calibration``, antes dos quadros)
    no fluxo e pelo ``frames.json`` no ``recompor``: os quadros saem byte a byte iguais. Só os
    pixels do CG mudam em relação ao quadro sem ajuste; a placa continua exata."""
    from mobilia.composite.compositor import Compositor, FrameStats, load_layers
    from mobilia.composite.grade import CGGrade
    from mobilia.composite.stream import StreamComposer
    from mobilia.jobs import ProgressEmitter

    plate8, meta = _job(tmp_path)
    frames = meta["frames"]
    calib = _calibration()
    # fluxo: a calibração chega antes do primeiro quadro, como o evento do worker
    sc = StreamComposer(tmp_path, plate8, tmp_path / "a.mp4", 12, 7, ProgressEmitter(), streaming=True,
                        n_frames=len(frames), workers=2)
    sc.set_calibration(calib)
    for fr in frames:
        sc.push(dict(fr))
    res = sc.finish({**meta, "calibration": calib})
    streamed = list(capture["frames"])
    assert res["composicao"]["ajuste_cg"] == {"aplicado": True, "mapa_ganho": "mapa", "curva_tom": True,
                                              "joelho": 0.8}
    assert res["preservacao"]["preservado_exato_todos"] is True
    # recompor: a mesma calibração lida do frames.json
    capture["frames"].clear()
    StreamComposer(tmp_path, plate8, tmp_path / "b.mp4", 12, 7, ProgressEmitter(), n_frames=len(frames),
                   calibration=calib).finish({**meta, "calibration": calib})
    assert capture["frames"] == streamed
    # sem calibração (trabalho antigo): o quadro de antes; com ela, só o CG muda
    capture["frames"].clear()
    sc = StreamComposer(tmp_path, plate8, tmp_path / "c.mp4", 12, 7, ProgressEmitter(), n_frames=len(frames))
    res0 = sc.finish(meta)
    assert res0["composicao"]["ajuste_cg"] == {"aplicado": False}
    assert capture["frames"][0] == streamed[0]                     # quadro só-foto
    assert capture["frames"][1] != streamed[1]
    comp = Compositor(plate8, seed=7)
    rgba, sh, origin = load_layers(tmp_path / "render" / "frames" / "f_00002.exr", frames[1]["border"], (H, W),
                                   max(16, int(0.04 * float(np.hypot(W, H)))))
    a, _ = comp.composite(2, rgba, sh, None, origin=origin)
    b, st = comp.composite(2, rgba, sh, None, origin=origin, grade=CGGrade.from_calibration(calib, (H, W)))
    assert _md5(b) == streamed[1] and st.preserved_exact
    full = np.zeros((H, W), bool)
    full[origin[1]:origin[1] + rgba.shape[0], origin[0]:origin[0] + rgba.shape[1]] = rgba[..., 3] > 1 / 1024
    from scipy import ndimage

    far = ~ndimage.binary_dilation(full, iterations=4)
    assert np.array_equal(a[far], b[far]) and not np.array_equal(a[full], b[full])
    assert FrameStats(1).preserved_exact


def test_stream_calibration_from_frames_json_or_mismatch(tmp_path, capture):
    """Worker sem o evento ``calibration``: vale a do ``frames.json`` (nenhum quadro de móvel foi
    composto antes). Se quadros já foram compostos com outra calibração, é erro (nunca um MP4 com
    dois ajustes diferentes)."""
    from mobilia.composite.stream import StreamComposer
    from mobilia.jobs import ProgressEmitter

    plate8, meta = _job(tmp_path)
    frames = meta["frames"]
    calib = _calibration()
    StreamComposer(tmp_path, plate8, tmp_path / "a.mp4", 12, 7, ProgressEmitter(), n_frames=len(frames),
                   calibration=calib).finish({**meta, "calibration": calib})
    ref = list(capture["frames"])
    capture["frames"].clear()
    sc = StreamComposer(tmp_path, plate8, tmp_path / "b.mp4", 12, 7, ProgressEmitter(), streaming=True,
                        n_frames=len(frames))
    sc.push(dict(frames[0]))                                       # só a foto antes do fim
    sc.finish({**meta, "calibration": calib})
    assert capture["frames"] == ref
    sc = StreamComposer(tmp_path, plate8, tmp_path / "c.mp4", 12, 7, ProgressEmitter(), streaming=True)
    for fr in frames[:3]:
        sc.push(dict(fr))
    with pytest.raises(RuntimeError, match="calibração"):
        sc.finish({**meta, "calibration": calib})
    sc.abort()


def test_stream_contact_shadow_like_recompose(tmp_path, capture):
    """PT10: o modelo da sombra de contato chega pelo evento ``contact`` (``set_contact``) no fluxo e
    pelo ``frames.json`` no ``recompor``; as poses vêm em cada quadro (``contato``). Os quadros saem
    iguais nos dois caminhos, mudam em relação ao quadro sem sombra de contato e a placa continua
    exata fora da influência (a verificação de cada quadro passa)."""
    from test_compositor import _contact_meta, _pose

    from mobilia.composite.stream import StreamComposer
    from mobilia.jobs import ProgressEmitter

    plate8, meta = _job(tmp_path)
    contact = _contact_meta(footprint=(3.0, 3.0), center=(0.0, 3.0), cell=0.03)
    frames = [dict(fr, contato=_pose(0.02 * k)) if fr.get("source") and not fr.get("reused") else dict(fr)
              for k, fr in enumerate(meta["frames"])]
    meta = {**meta, "frames": frames, "contact": contact}
    sc = StreamComposer(tmp_path, plate8, tmp_path / "a.mp4", 12, 7, ProgressEmitter(), streaming=True,
                        n_frames=len(frames), workers=2)
    sc.set_contact(contact)
    for fr in frames:
        sc.push(dict(fr))
    res = sc.finish(meta)
    streamed = list(capture["frames"])
    sumc = res["composicao"]["sombra_contato"]
    assert sumc["aplicada"] is True and sumc["pecas"] == 1 and sumc["pixels_total"] > 0
    assert res["preservacao"]["preservado_exato_todos"] is True
    capture["frames"].clear()
    StreamComposer(tmp_path, plate8, tmp_path / "b.mp4", 12, 7, ProgressEmitter(), n_frames=len(frames),
                   contact=contact).finish(meta)
    assert capture["frames"] == streamed
    # worker sem o evento: vale o frames.json (nenhum quadro de móvel composto antes)
    capture["frames"].clear()
    sc = StreamComposer(tmp_path, plate8, tmp_path / "c.mp4", 12, 7, ProgressEmitter(), streaming=True,
                        n_frames=len(frames))
    sc.push(dict(frames[0]))
    sc.finish(meta)
    assert capture["frames"] == streamed
    # sem modelo (trabalho antigo): outro quadro, sem sombra de contato
    capture["frames"].clear()
    res0 = StreamComposer(tmp_path, plate8, tmp_path / "d.mp4", 12, 7, ProgressEmitter(),
                          n_frames=len(frames)).finish({k: v for k, v in meta.items() if k != "contact"})
    assert res0["composicao"]["sombra_contato"] == {"aplicada": False}
    assert capture["frames"][0] == streamed[0] and capture["frames"][1] != streamed[1]
    # quadros de móvel já compostos com outro modelo: erro
    sc = StreamComposer(tmp_path, plate8, tmp_path / "e.mp4", 12, 7, ProgressEmitter(), streaming=True)
    for fr in frames[:3]:
        sc.push(dict(fr))
    with pytest.raises(RuntimeError, match="sombra de contato"):
        sc.finish(meta)
    sc.abort()
