"""Pipeline ponta a ponta.

- ``test_fast_pipeline_telemetry`` (rápido): pipeline real com os modelos de IA e o Blender
  trocados por dublês (sala de teste já analisada + ``tests/fake_worker.py blender``); composição,
  codificação e verificação são as reais. Confere a telemetria do ``report.json`` (PT1).
- ``test_full_pipeline_preserves_photo`` (lento, precisa dos pesos): modelos de IA + Blender +
  FFmpeg reais em resolução mínima. Executar com:
  ``python -m pytest -m "slow and models" tests/test_pipeline_e2e.py``
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

FX = Path(__file__).parent / "fixtures"
FAKE = Path(__file__).parent / "fake_worker.py"


def _tiny_quality(monkeypatch, pipeline, **kw):
    orig_quality = pipeline._quality

    def tiny_quality(req, prof):
        q = orig_quality(req, prof)
        q.update(kw)
        return q

    monkeypatch.setattr(pipeline, "_quality", tiny_quality)


def _fake_analysis(monkeypatch, tmp_path):
    """Troca só o que precisa dos pesos (MoGe-2/OneFormer) e a reconstrução da sala pela sala de
    teste já analisada; o resto da análise (oclusores, sol, gravações) roda de verdade."""
    from PIL import Image

    from mobilia.analysis import geometry, room as room_mod, segmentation
    from mobilia.analysis.geometry import GeometryResult, Intrinsics
    from mobilia.analysis.room import RoomModel
    from mobilia.analysis.segmentation import SegmentationResult
    from mobilia.assets import polyhaven
    from mobilia.render import scene_spec

    seg_full = SegmentationResult.load(FX / "rooms/stafford/seg.npz")
    room = RoomModel.from_json((FX / "rooms/stafford/room.json").read_text(encoding="utf-8"))
    grid = np.load(FX / "rooms/stafford/grid.npy")

    def fake_geometry(rgb8, timings=None, **kw):
        t = time.perf_counter()
        time.sleep(0.02)                                   # "rede": pesos que faltavam no cache
        dl = time.perf_counter() - t
        time.sleep(0.03)                                   # "carga dos pesos"
        if timings is not None:
            timings["download"] = dl
            timings["carga"] = time.perf_counter() - t - dl
        time.sleep(0.02)                                   # "inferência"
        h, w = rgb8.shape[:2]
        z = np.zeros((h, w), np.float32)
        return GeometryResult(np.zeros((h, w, 3), np.float32), z, np.zeros((h, w, 3), np.float32),
                              np.zeros((h, w), bool), Intrinsics(w, h, w, w, w / 2, h / 2), "dublê")

    def fake_segment(rgb8, timings=None, **kw):
        if timings is not None:
            timings["carga"] = 0.01
        time.sleep(0.02)
        h, w = rgb8.shape[:2]
        rs = lambda a: np.asarray(Image.fromarray(a).resize((w, h), Image.NEAREST))  # noqa: E731
        return SegmentationResult(rs(seg_full.labels), rs(seg_full.groups), seg_full.id2label, "dublê")

    def fake_ensure_model(asset_id, cache_dir, res="1k"):
        polyhaven.NET_STATS["s"] += 0.01                     # simula um download curto
        p = tmp_path / "assets" / asset_id / f"{asset_id}.gltf"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
        return p

    monkeypatch.setattr(geometry, "estimate_geometry", fake_geometry)
    monkeypatch.setattr(segmentation, "segment", fake_segment)
    monkeypatch.setattr(room_mod, "build_room_model", lambda geo, seg, rgb: (room, {"grid": grid}))
    monkeypatch.setattr(scene_spec, "ensure_model", fake_ensure_model)


def test_fast_pipeline_telemetry(tmp_path, monkeypatch):
    from mobilia import bootstrap, pipeline, runtime
    from mobilia.jobs import JobRequest, ProgressEmitter
    from mobilia.telemetry import ETAPAS, ETAPAS_ANALISE, stages_sum

    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(pipeline.ENV_BLENDER_WORKER, json.dumps([sys.executable, str(FAKE), "blender"]))
    _tiny_quality(monkeypatch, pipeline, long_edge=160, samples=8, fps=12)
    _fake_analysis(monkeypatch, tmp_path)
    events = []
    out = tmp_path / "video.mp4"
    req = JobRequest(str(FX / "photos/sala_vazia_stafford.jpg"), "sala", str(out), quality="previa",
                     work_dir=str(tmp_path / "job"))
    res = pipeline.run_job(req, ProgressEmitter(callback=events.append))
    assert out.exists()
    rep = json.loads(Path(res.report_path).read_text(encoding="utf-8"))
    t = rep["tempos_s"]
    # todas as etapas da seção 4 do PLANO_V2, com as subetapas da análise
    for k in ETAPAS + ("total", "total_percebido"):
        assert k in t, k
    for k in ETAPAS_ANALISE:
        assert k in t["analise"], k
    assert t["analise"]["geometria_carga"] > 0 and t["analise"]["geometria_inferencia"] > 0
    assert t["download"] > 0 and t["blender_inicio"] > 0 and t["render_dinamico"] > 0
    # as etapas particionam o total
    assert abs(stages_sum(t) - t["total"]) <= 0.05 * t["total"], (stages_sum(t), t["total"])
    assert t["total_percebido"] >= t["total"]
    # primeira execução: registrada no estado da instalação para a próxima ser "segunda"
    assert rep["primeira_execucao"] is True
    st = bootstrap.load_state(runtime.runtime_paths().state_file)
    assert st["execucoes"] == 1 and not bootstrap.is_first_run(st)
    assert set(rep["primeira_execucao_custos"]) >= {"download_s", "blender_kernels_s", "modelos_carga_s",
                                                   "kernels_cycles_total_s", "aquecimento_instalador"}
    # sem instalador (desenvolvimento): o aquecimento não foi feito e isso fica registrado
    assert rep["primeira_execucao_custos"]["aquecimento_instalador"]["feito"] is False
    # download = rede dos móveis (Poly Haven) + dos pesos dos modelos, fora de "ativos" e "analise"
    d = rep["download"]
    assert d["modelos_s"] >= 0.02 and d["ativos_s"] > 0 and abs(d["s"] - d["modelos_s"] - d["ativos_s"]) < 0.02
    assert abs(t["download"] - d["s"]) < 0.02
    # render: dispositivo efetivo, denoiser, camadas, por quadro, quadros
    r = rep["render"]
    assert r["dispositivo"] == "CPU" and r["denoiser_dispositivo"] == "CPU" and r["optix"] is False
    assert set(r["amostras_camadas"]) == {"final", "marco", "dinamico"}
    q = r["quadros"]
    assert q["total"] == q["renderizados"] + q["reutilizados"] + q["so_foto"] and q["reutilizados"] > 0
    assert len(r["por_quadro"]) == q["renderizados"]
    assert {p["camada"] for p in r["por_quadro"]} == {"dinamico", "final"}
    for p in r["por_quadro"]:
        assert {"quadro", "camada", "area_px", "spp", "amostragem_s", "denoise_s", "gravacao_s",
                "parede_s"} <= set(p)
    assert r["desempenho"]["amostras_por_s"] > 0 and r["desempenho"]["custo_fixo_quadro_s"] >= 0
    assert r["camadas"]["final"]["quadros"] >= 1 and r["camadas"]["dinamico"]["quadros"] >= 1
    assert 0 < r["cobertura_recorte_media"] < 1
    # composição / encode / preservação / GPU
    c = rep["composicao"]
    assert len(c["por_quadro"]) == q["total"] and c["recorte_medio_px"] > 0 and c["paralela"] is False
    assert {"leitura_s", "composicao_s", "codificacao_s"} <= set(c["por_quadro"][0])
    assert rep["encode"]["codificador"] == "libx264" and rep["encode"]["bytes"] > 0 and rep["encode"]["s"] > 0
    assert rep["preservacao"]["preservado_exato_todos"] is True
    assert rep["preservacao"]["quadros_verificados"] == q["total"]
    assert set(rep["gpu"]["vram_pico_mb"]) >= {"analise", "render"}
    assert rep["foto"]["megapixels"] == pytest.approx(1920 * 1440 / 1e6, abs=1e-3)
    assert set(rep["calibracao_luz"]) >= {"rel_rms", "ganhos_canal", "fracao_mascara", "sol_ajustado"}
    assert any(e["event"] == "log" and "Blender" in e["message"] for e in events)
    # PT4: EXR recortados no border, colados no quadro inteiro e apagados depois da composição
    assert r["exr"] == {"recorte": True, "codec": "RLE"} and c["exr_recortado"] is True
    assert c["exr_apagados"] == q["renderizados"] and c["exr_bytes_apagados"] > 0
    assert not list((tmp_path / "job" / "render" / "frames").glob("*.exr"))
    # sem os EXR, recompor falha com uma mensagem clara (e não com um quadro vazio)
    with pytest.raises(FileNotFoundError, match=pipeline.ENV_MANTER_EXR):
        pipeline.recompose(tmp_path / "job", tmp_path / "de_novo.mp4", ProgressEmitter(callback=events.append))


def test_fast_pipeline_keeps_exr_on_request(tmp_path, monkeypatch):
    """Com ``MOBILIA_MANTER_EXR=1`` os EXR ficam e ``recompor`` refaz o MP4 com os mesmos quadros."""
    from mobilia import pipeline
    from mobilia.jobs import JobRequest, ProgressEmitter
    from mobilia.video.encoder import decode_frames

    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(pipeline.ENV_BLENDER_WORKER, json.dumps([sys.executable, str(FAKE), "blender"]))
    monkeypatch.setenv(pipeline.ENV_MANTER_EXR, "1")
    _tiny_quality(monkeypatch, pipeline, long_edge=160, samples=8, fps=12)
    _fake_analysis(monkeypatch, tmp_path)
    out = tmp_path / "video.mp4"
    req = JobRequest(str(FX / "photos/sala_vazia_stafford.jpg"), "sala", str(out), quality="previa",
                     work_dir=str(tmp_path / "job"))
    res = pipeline.run_job(req, ProgressEmitter())
    rep = json.loads(Path(res.report_path).read_text(encoding="utf-8"))
    assert rep["composicao"]["exr_apagados"] == 0
    n = rep["render"]["quadros"]["renderizados"]
    assert len(list((tmp_path / "job" / "render" / "frames").glob("*.exr"))) == n
    again = tmp_path / "de_novo.mp4"
    rep2 = pipeline.recompose(tmp_path / "job", again, ProgressEmitter())
    assert rep2["preservacao"]["preservado_exato_todos"] is True
    from PIL import Image

    h, w = np.asarray(Image.open(tmp_path / "job" / "plate.png")).shape[:2]
    total = rep["render"]["quadros"]["total"]
    idx = [1, total // 2, total]
    a = dict(decode_frames(out, w, h, idx))
    b = dict(decode_frames(again, w, h, idx))
    assert all(np.array_equal(a[i], b[i]) for i in idx)


def test_compose_deletes_exr_after_last_use_and_survives_locked_file(tmp_path, monkeypatch):
    """Cada EXR sai do disco só depois do último quadro que o usa (o reaproveitado também); um
    arquivo preso (antivírus no Windows) vira aviso, o vídeo sai igual e a telemetria só conta
    o que foi apagado de fato."""
    import OpenEXR

    from mobilia import pipeline
    from mobilia.jobs import ProgressEmitter
    from mobilia.video.encoder import VideoWriter

    H, W = 48, 64
    plate8 = np.random.default_rng(1).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    frames_dir = tmp_path / "render" / "frames"
    frames_dir.mkdir(parents=True)

    def write(name, border):
        x0, y0, x1, y1 = border
        h, w = y1 - y0, x1 - x0
        rgba = np.zeros((h, w, 4), np.float16)
        rgba[h // 4:3 * h // 4, w // 4:3 * w // 4] = [0.3, 0.2, 0.1, 1.0]
        sh = np.ones((h, w, 3), np.float16)
        ch = {f"Combined.{c}": np.ascontiguousarray(rgba[..., i]) for i, c in enumerate("RGBA")}
        ch.update({f"Shadow Catcher.{c}": np.ascontiguousarray(sh[..., i]) for i, c in enumerate("RGB")})
        with OpenEXR.File({"compression": OpenEXR.RLE_COMPRESSION, "type": OpenEXR.scanlineimage}, ch) as f:
            f.write(str(frames_dir / name))

    write("f_00001.exr", [8, 10, 40, 38])
    write("f_00003.exr", [4, 6, 50, 44])
    size1 = (frames_dir / "f_00001.exr").stat().st_size
    meta = {"exr": {"recorte": True, "codec": "RLE"}, "frames": [
        {"frame": 1, "source": "f_00001.exr", "border": [8, 10, 40, 38], "mode": "full", "crop": True},
        {"frame": 2, "source": "f_00001.exr", "border": [8, 10, 40, 38], "reused": True},
        {"frame": 3, "source": "f_00003.exr", "border": [4, 6, 50, 44], "mode": "full", "crop": True},
        {"frame": 4, "source": None, "border": None}]}

    written = {"n": 0}
    orig_write, orig_unlink = VideoWriter.write, Path.unlink
    deleted: list[tuple[str, int]] = []

    def counting_write(self, frame):
        written["n"] += 1
        return orig_write(self, frame)

    def flaky_unlink(self, *a, **kw):
        if self.suffix == ".exr" and self.parent == frames_dir:
            deleted.append((self.name, written["n"]))
            if self.name == "f_00003.exr":
                raise PermissionError(13, "arquivo em uso por outro processo", str(self))
        return orig_unlink(self, *a, **kw)

    monkeypatch.setattr(VideoWriter, "write", counting_write)
    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    events: list[dict] = []
    comp = pipeline.compose_video(tmp_path, plate8, meta, tmp_path / "v.mp4", 12, 7,
                                  ProgressEmitter(callback=events.append), delete_exr=True)
    # f_00001 só depois do quadro 2 (reaproveitado: 1 quadro já enviado ao codificador), não no 1
    assert deleted == [("f_00001.exr", 1), ("f_00003.exr", 2)]
    assert not (frames_dir / "f_00001.exr").exists() and (frames_dir / "f_00003.exr").exists()
    c = comp["composicao"]
    assert c["exr_apagados"] == 1 and c["exr_bytes_apagados"] == size1
    assert any(e["event"] == "log" and e["level"] == "warning" and "f_00003.exr" in e["message"] for e in events)
    assert comp["preservacao"]["preservado_exato_todos"] is True and written["n"] == 4


@pytest.mark.slow
@pytest.mark.models
def test_full_pipeline_preserves_photo(tmp_path, monkeypatch):
    from mobilia import pipeline
    from mobilia.jobs import JobRequest, ProgressEmitter

    # resolução mínima para o teste caber em CPU
    _tiny_quality(monkeypatch, pipeline, long_edge=320, samples=8, fps=12)
    events = []
    em = ProgressEmitter(callback=events.append)
    out = tmp_path / "video.mp4"
    req = JobRequest(str(FX / "photos/sala_vazia_stafford.jpg"), "sala", str(out), quality="previa",
                     work_dir=str(tmp_path / "job"))
    res = pipeline.run_job(req, em)
    assert out.exists() and out.stat().st_size > 10_000
    rep = json.loads(Path(res.report_path).read_text(encoding="utf-8"))
    assert rep["preservacao"]["todos_bit_exatos_fora_dos_objetos"] is True
    assert rep["preservacao"]["quadros_identicos_a_foto"] >= 1       # o início do vídeo é a foto pura
    assert len(rep["layout"]["items"]) >= 4
    fracs = [e["fraction"] for e in events if e["event"] == "progress"]
    assert fracs == sorted(fracs) or max(np.diff(fracs)) > -0.05        # progresso essencialmente monotônico
    assert rep["encode"]["verificacao"][0]["psnr_fundo_db"] > 35
    from mobilia.telemetry import stages_sum
    t = rep["tempos_s"]
    assert abs(stages_sum(t) - t["total"]) <= 0.05 * t["total"]
    assert rep["render"]["por_quadro"] and rep["render"]["denoiser_dispositivo"] == "CPU"
