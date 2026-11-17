"""PT-V3-8: modo B integrado ao pipeline (``run_job``), ponta a ponta na CPU.

Rápidos: a análise (MoGe-2/OneFormer/sala) vem das fixtures já analisadas (``tests/fixtures/rooms``), o inpainter é
o Telea do OpenCV e o SAM/OneFormer da nota ficam desligados; o resto (detecção do modo, instâncias, portão de
qualidade com as tentativas, camadas, coreografia, composição, verificação §3.5 e MP4) é o código real.
Lento (``slow``/``models``): o mesmo com o big-lama de verdade.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import load_room_fixture

ROOMS = Path(__file__).parent / "fixtures" / "rooms"
REAL_HOME = os.environ.get("MOBILIA_HOME")      # pesos já baixados (os testes usam uma pasta de dados temporária)


def _tiny_quality(monkeypatch, pipeline, long_edge):
    orig = pipeline._quality

    def q(req, prof):
        d = orig(req, prof)
        d.update(long_edge=long_edge, fps=12)
        return d

    monkeypatch.setattr(pipeline, "_quality", q)


def _fixture_analysis(monkeypatch, name):
    """Troca MoGe-2, OneFormer e ``build_room_model`` pela análise gravada da fixture (na resolução pedida)."""
    from mobilia import pipeline
    from mobilia.analysis import geometry, room as room_mod
    from mobilia.analysis.geometry import GeometryResult, Intrinsics
    from mobilia.analysis.segmentation import SegmentationResult
    from mobilia.removal.instances import PanopticResult

    f = load_room_fixture(name)

    def up(a, w, h):
        import cv2
        return cv2.resize(np.asarray(a, np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

    def fake_geometry(rgb8, timings=None, **kw):
        h, w = rgb8.shape[:2]
        pts = up(f.geometry["points"], w, h)
        nrm = up(f.geometry["normals"], w, h)
        msk = up(f.geometry["mask"].astype(np.float32), w, h) > 0.5
        i = f.geometry["intrinsics"]
        s = w / i["width"]
        return GeometryResult(pts, pts[..., 2].copy(), nrm, msk,
                              Intrinsics(w, h, i["fx"] * s, i["fy"] * s, i["cx"] * s, i["cy"] * s), "fixture")

    def nearest(a, w, h):
        return np.asarray(Image.fromarray(a).resize((w, h), Image.NEAREST))

    def fake_seg_pan(rgb8, device, use_fp16, timings):
        h, w = rgb8.shape[:2]
        seg = SegmentationResult(nearest(f.seg.labels, w, h), nearest(f.seg.groups, w, h), f.seg.id2label,
                                 f.seg.model_id)
        return seg, PanopticResult(nearest(f.panoptic_ids, w, h), f.panoptic_segments)

    monkeypatch.setattr(geometry, "estimate_geometry", fake_geometry)
    monkeypatch.setattr(pipeline, "_segment_panoptic", fake_seg_pan)
    monkeypatch.setattr(room_mod, "build_room_model", lambda geo, seg, rgb: (f.room, {"grid": f.grid}))
    return f


def _fast_models(monkeypatch, gate_ok: bool | None = None):
    from mobilia import pipeline_b
    from mobilia.removal import plate as rp

    monkeypatch.setattr(pipeline_b, "load_inpainter", lambda dev: rp.telea_inpaint)
    monkeypatch.setattr(pipeline_b, "load_redetector", lambda dev, fp16: (None, None))
    monkeypatch.setattr(pipeline_b, "sam_choice", lambda dev: None)
    monkeypatch.setenv(pipeline_b.ENV_REDETECTAR, "0")
    if gate_ok is not None:
        orig = rp.assess

        def assess(*a, **k):
            q = orig(*a, **k)
            if gate_ok:
                q.ok, q.reasons = True, []
            else:
                q.ok, q.reasons, q.score = False, ["costura visível no(a) piso (ΔE 9.0) [teste]"], 40.0
            return q

        monkeypatch.setattr(rp, "assess", assess)


def _job(tmp_path, photo, **kw):
    from mobilia.jobs import JobRequest

    return JobRequest(str(photo), "sala", str(tmp_path / "video.mp4"), quality="previa",
                      work_dir=str(tmp_path / "job"), **kw)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from mobilia import pipeline
    from mobilia.video import encoder as E

    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(E.ENV_CODEC, "libx264")
    monkeypatch.setenv("MOBILIA_MODO_B_MKV", "1")
    _tiny_quality(monkeypatch, pipeline, 480)
    return pipeline


@pytest.mark.parametrize("name", ["sala_sofas_brancos", "nice_terrazzo"])
def test_mode_b_end_to_end_cpu(env, tmp_path, monkeypatch, name):
    """Foto mobiliada -> modo B automático -> MP4 + MKV de auditoria; todas as verificações §3.5 passam e o
    report.json traz modo, área gerada, peças, semente, nota/tentativas do portão e tempos."""
    from mobilia.jobs import ProgressEmitter
    from mobilia.telemetry import ETAPAS_B, stages_sum
    from mobilia.video import encoder as E

    _fixture_analysis(monkeypatch, name)
    _fast_models(monkeypatch, gate_ok=True)
    events = []
    res = env.run_job(_job(tmp_path, ROOMS / name / "photo.jpg"), ProgressEmitter(callback=events.append))
    assert res.mode == "B" and res.warning is None and res.gate["aprovada"]
    out = Path(res.output_path)
    assert out.exists() and out.stat().st_size > 1000
    rep = json.loads(Path(res.report_path).read_text(encoding="utf-8"))
    assert rep["modo"]["modo"] == "B" and not rep["modo"]["forcado"]
    ver = rep["verificacao_modo_b"]
    assert ver["ok"], ver["falhas"]
    assert rep["video_modo_b"]["auditoria_mkv"]["quadros_identicos"] == ver["quadros"]
    # último quadro do MKV sem perdas == foto na resolução do vídeo; 1º == sala vazia gravada
    W, H = rep["foto"]["placa_video"]
    frames = dict(E.decode_lossless(Path(res.work_dir) / "auditoria.mkv", W, H))
    photo_v = np.asarray(Image.open(Path(res.work_dir) / "plate.png").convert("RGB"))
    assert np.array_equal(frames[len(frames)], photo_v)
    empty = np.asarray(Image.open(Path(res.work_dir) / "sala_vazia.png").convert("RGB"))
    assert frames[1].shape == empty.shape
    # relatório
    assert 0 < rep["pixels_gerados"]["area_pct"] < 75
    assert rep["pecas"]["n_pecas"] >= 2 and rep["pecas_no_video"]
    assert rep["semente"] == 7 and rep["placa"]["qualidade"]["nota"] is not None
    assert rep["portao_qualidade"]["aprovada"] and rep["portao_qualidade"]["tentativas"] == 1
    assert rep["narrativa"] and rep["titulo_narrativa"] == "O QUE O 7xAi FEZ"
    assert rep["video_entregue"] is True and rep["aviso"] is None
    t = rep["tempos_s"]
    for k in ("pecas", "sala_vazia", "camadas", "video"):
        assert t[k] > 0, k
    assert abs(stages_sum(t, ETAPAS_B) - t["total"]) <= 0.1 * t["total"] + 0.5
    # eventos: modo anunciado com as etapas do modo B; prévia da sala vazia
    ev = [e for e in events if e["event"] == "mode"]
    assert ev and ev[0]["mode"] == "B" and [s[0] for s in ev[0]["stages"]][2:4] == ["pecas", "sala_vazia"]
    assert any(e["event"] == "preview" and e["path"].endswith("sala_vazia.jpg") for e in events)
    assert any(e["event"] == "progress" and e["stage"] == "video" for e in events)


def test_mode_b_gate_rejects_without_mp4_then_user_accepts(env, tmp_path, monkeypatch):
    """Portão: placa reprovada em todas as tentativas -> sem MP4, aviso em português no resultado e no relatório;
    com ``accept_weak_plate`` o vídeo sai, com o aviso registrado."""
    from mobilia.jobs import ProgressEmitter

    _fixture_analysis(monkeypatch, "sala_sofas_brancos")
    _fast_models(monkeypatch, gate_ok=False)
    out = tmp_path / "video.mp4"
    out.write_bytes(b"antigo")                     # um MP4 velho com o mesmo nome nunca passa por entregue
    events = []
    em = ProgressEmitter(callback=events.append)
    res = env.run_job(_job(tmp_path, ROOMS / "sala_sofas_brancos" / "photo.jpg"), em)
    em.done(res)
    assert res.output_path == "" and not out.exists()
    assert "nenhum MP4 foi gerado" in res.warning and "modo A" in res.warning
    rep = json.loads(Path(res.report_path).read_text(encoding="utf-8"))
    g = rep["portao_qualidade"]
    assert not g["aprovada"] and rep["video_entregue"] is False and rep["aviso"] == res.warning
    names = [t["estrategia"] for t in g["tentativas_detalhe"]]
    # sala refeita pela geometria primeiro; sem as tentativas do LaMa (nada imaginado)
    assert names[0] == "crua_geometria" and "mascara_ampla" in names and "lama_imagem" not in names
    assert all(t["metodo"] == "geometria" for t in g["tentativas_detalhe"])
    assert "verificacao_modo_b" not in rep
    done = [e for e in events if e["event"] == "done"][-1]
    assert done["output"] is None and done["warning"] == res.warning and done["mode"] == "B"
    # o usuário escolhe "gerar mesmo assim"
    res2 = env.run_job(_job(tmp_path, ROOMS / "sala_sofas_brancos" / "photo.jpg", mode="B",
                            accept_weak_plate=True), ProgressEmitter())
    assert Path(res2.output_path).exists() and "a seu pedido" in res2.warning
    rep2 = json.loads(Path(res2.report_path).read_text(encoding="utf-8"))
    assert rep2["portao_qualidade"]["aceita_pelo_usuario"] and rep2["verificacao_modo_b"]["ok"]
    assert rep2["modo"]["forcado"]


def test_empty_photo_routes_to_mode_a_and_forced_mode_b_needs_furniture(env, tmp_path, monkeypatch):
    """Foto sem móveis: a detecção escolhe o modo A (o motor v2). Forçar o modo B numa foto sem nada para tirar
    termina com uma mensagem clara (sem MP4)."""
    from mobilia.jobs import ProgressEmitter
    from mobilia.removal import instances as ri

    f = _fixture_analysis(monkeypatch, "sala_sofas_brancos")
    from mobilia import pipeline
    from mobilia.analysis.segmentation import SegmentationResult

    wall = [k for k, v in f.seg.id2label.items() if v == "wall"][0]
    empty_seg = SegmentationResult(np.full_like(f.seg.labels, wall), np.zeros_like(f.seg.groups), f.seg.id2label)

    def no_furniture(rgb8, device, use_fp16, timings):
        h, w = rgb8.shape[:2]
        seg = SegmentationResult(np.full((h, w), wall, np.uint8), np.zeros((h, w), np.uint8), f.seg.id2label)
        return seg, ri.PanopticResult(np.zeros((h, w), np.uint8), np.zeros((0, 4)))

    monkeypatch.setattr(pipeline, "_segment_panoptic", no_furniture)
    assert ri.detect_mode(empty_seg, ri.PanopticResult(np.zeros(f.seg.labels.shape, np.uint8),
                                                       np.zeros((0, 4)))).mode == "A"
    events = []

    class StopA(Exception):
        pass

    def stop_layout(*a, **k):                      # modo A confirmado: o resto é o motor v2 (testado à parte)
        raise StopA()

    from mobilia.layout import solver
    monkeypatch.setattr(solver, "plan_layout", stop_layout)
    with pytest.raises(StopA):
        env.run_job(_job(tmp_path, ROOMS / "sala_sofas_brancos" / "photo.jpg"),
                    ProgressEmitter(callback=events.append))
    assert [e["mode"] for e in events if e["event"] == "mode"] == ["A"]
    _fast_models(monkeypatch)
    with pytest.raises(ValueError, match="Nenhum móvel"):
        env.run_job(_job(tmp_path, ROOMS / "sala_sofas_brancos" / "photo.jpg", mode="B"), ProgressEmitter())
    assert not (tmp_path / "video.mp4").exists()


def test_to_video_resolution_keeps_labels_and_areas():
    from mobilia.analysis.geometry import GeometryResult, Intrinsics
    from mobilia.analysis.segmentation import SegmentationResult
    from mobilia.pipeline_b import to_video_resolution
    from mobilia.removal.instances import PanopticResult

    ids = np.zeros((40, 60), np.uint8)
    ids[10:30, 20:40] = 1
    seg = SegmentationResult(ids.copy(), ids.copy(), {0: "wall", 1: "sofa"})
    pan = PanopticResult(ids, np.array([[1, 19, 0.9, 400.0]]))
    pts = np.random.default_rng(0).normal(size=(40, 60, 3)).astype(np.float32)
    geo = GeometryResult(pts, pts[..., 2], pts, np.ones((40, 60), bool), Intrinsics(60, 40, 50, 50, 30, 20), "x")
    s, p, g = to_video_resolution(seg, pan, geo, 30, 20)
    assert s.labels.shape == (20, 30) and p.ids.shape == (20, 30)
    assert p.segments[0, 3] == (p.ids == 1).sum() == 100
    assert g["stride"] == 4 and g["points"].shape == (5, 8, 3) and g["mask"].shape == (5, 8)


@pytest.mark.slow
@pytest.mark.models
@pytest.mark.parametrize("name", ["sala_sofas_brancos", "nice_terrazzo"])
def test_mode_b_end_to_end_big_lama_cpu(env, tmp_path, monkeypatch, name):
    """Com o big-lama de verdade (CPU) e o portão real (sem o OneFormer da nota, caro na CPU): o trabalho termina
    ou com o MP4 verificado ou, reprovado, sem MP4 e com aviso; com ``accept_weak_plate`` sempre sai o MP4
    verificado."""
    from mobilia import pipeline_b
    from mobilia.analysis import weights
    from mobilia.jobs import ProgressEmitter
    from mobilia.removal import plate as rp

    tmp_home = os.environ["MOBILIA_HOME"]
    if REAL_HOME:
        monkeypatch.setenv("MOBILIA_HOME", REAL_HOME)
    else:
        monkeypatch.delenv("MOBILIA_HOME")
    try:
        lama = rp.load_lama("cpu", offline=True)
    except weights.WeightsError as e:
        pytest.skip(str(e))
    finally:
        monkeypatch.setenv("MOBILIA_HOME", tmp_home)
    _fixture_analysis(monkeypatch, name)
    monkeypatch.setattr(pipeline_b, "load_inpainter", lambda dev: lama)
    monkeypatch.setattr(pipeline_b, "load_redetector", lambda dev, fp16: (None, None))
    monkeypatch.setattr(pipeline_b, "sam_choice", lambda dev: None)
    monkeypatch.setenv(pipeline_b.ENV_REDETECTAR, "0")
    res = env.run_job(_job(tmp_path, ROOMS / name / "photo.jpg", accept_weak_plate=True), ProgressEmitter())
    rep = json.loads(Path(res.report_path).read_text(encoding="utf-8"))
    assert res.mode == "B" and Path(res.output_path).exists()
    assert rep["verificacao_modo_b"]["ok"], rep["verificacao_modo_b"]["falhas"]
    assert rep["portao_qualidade"]["tentativas"] >= 1
    print(name, json.dumps({k: rep["portao_qualidade"][k] for k in ("aprovada", "nota", "estrategia", "motivos")},
                           ensure_ascii=False), rep["tempos_s"]["sala_vazia"])
