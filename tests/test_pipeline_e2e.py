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
from PIL import Image

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
    def fake_ensure_texture(tex_id, cache_dir, res="1k", retries=4):     # textura CC0 do tapete: sem rede no teste
        base = tmp_path / "assets" / "textures" / tex_id
        base.mkdir(parents=True, exist_ok=True)
        return {k: base / f"{k}.jpg" for k in ("diff", "nor", "rough")}

    monkeypatch.setattr(scene_spec, "ensure_model", fake_ensure_model)
    monkeypatch.setattr(scene_spec, "ensure_texture", fake_ensure_texture)


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
    assert q["total"] == q["quadros_novos"] + q["reutilizados"] + q["so_foto"] and q["reutilizados"] > 0
    assert len(r["por_quadro"]) == q["renderizados"]
    # PT6: render em camadas (padrão): marcos (o final primeiro), camada dinâmica e transição
    assert r["modo_camadas"] == "camadas" and len(r["plano_camadas"]["marcos"]) == 2
    assert {p["camada"] for p in r["por_quadro"]} == {"dinamico", "final", "marco"}
    assert r["por_quadro"][0]["camada"] == "final"
    for p in r["por_quadro"]:
        assert {"quadro", "camada", "area_px", "spp", "amostragem_s", "denoise_s", "gravacao_s",
                "parede_s"} <= set(p)
    assert r["desempenho"]["amostras_por_s"] > 0 and r["desempenho"]["custo_fixo_quadro_s"] >= 0
    assert r["camadas"]["final"]["quadros"] >= 1 and r["camadas"]["dinamico"]["quadros"] >= 1
    assert 0 < r["cobertura_recorte_media"] < 1
    # composição / encode / preservação / GPU
    c = rep["composicao"]
    assert len(c["por_quadro"]) == q["total"] and c["recorte_medio_px"] > 0
    assert {"leitura_s", "composicao_s", "verificacao_s", "codificacao_s"} <= set(c["por_quadro"][0])
    # PT5: composição em fluxo, durante o render; reaproveitados vêm do cache (não são recompostos)
    assert c["paralela"] is True and c["trabalhadores"] >= 1
    assert 0 < c["compostos_durante_render"] <= q["total"]
    assert c["quadros_compostos"] == q["quadros_novos"] and c["quadros_cache"] == q["reutilizados"]
    assert c["quadros_so_foto"] == q["so_foto"]
    assert 0 < c["recorte_composicao_medio_px"] < 160 * 120
    assert sum(p["cache"] for p in c["por_quadro"]) == q["reutilizados"]
    assert all(p["composicao_s"] == 0 for p in c["por_quadro"] if p["cache"])
    e = rep["encode"]
    assert e["codificador"] == "libx264" and e["preset"] == "medium" and e["bytes"] > 0 and e["s"] > 0
    # MP4 conferido por seek em 3 quadros: início, meio e fim (o dublê renderiza menos quadros que
    # o previsto pela cena: o "meio" é o quadro guardado mais perto dele)
    pos = [v["posicao"] for v in e["verificacao"]]
    assert e["verificacao_metodo"] == "seek" and len(pos) == 3 and pos[0] == 1 and pos[-1] == q["total"]
    assert q["total"] / 4 <= pos[1] <= 3 * q["total"] / 4
    pres = rep["preservacao"]
    assert pres["preservado_exato_todos"] is True
    assert pres["quadros_verificados"] == q["total"]
    assert pres["verificacao"]["compostos"] == q["quadros_novos"] and pres["verificacao"]["raio_px"] > 0
    assert set(rep["gpu"]["vram_pico_mb"]) >= {"analise", "render"}
    assert rep["foto"]["megapixels"] == pytest.approx(1920 * 1440 / 1e6, abs=1e-3)
    assert set(rep["calibracao_luz"]) >= {"rel_rms", "ganhos_canal", "fracao_mascara", "sol_ajustado"}
    # PT8: o ajuste do CG chegou no evento da calibração (antes dos quadros) e foi aplicado
    cl = rep["calibracao_luz"]
    assert cl["curva_tom_ok"] is False and cl["mapa_ganho"]["modo"] == "mapa" and "grade" not in cl["mapa_ganho"]
    assert cl["fracao_limite_inferior"] == pytest.approx(0.025)
    assert c["ajuste_cg"] == {"aplicado": True, "mapa_ganho": "mapa", "curva_tom": False, "joelho": 0.8}
    assert any(e["event"] == "log" and "ganho do CG: mapa" in e["message"] for e in events)
    # PT10: o modelo da sombra de contato chegou no evento ``contact`` e as poses em cada quadro
    assert c["sombra_contato"]["aplicada"] is True and c["sombra_contato"]["pecas"] == 1
    # a cena manda a máscara de calibração de 3 níveis e as manchas de sol da foto ao worker
    spec = json.loads((tmp_path / "job" / "scene.json").read_text(encoding="utf-8"))
    assert spec["lighting"]["photo_emission"] is True
    m = np.asarray(Image.open(spec["lighting"]["calib_mask_path"]))
    assert set(np.unique(m)) <= {0, 128, 255} and (m == 255).any()
    if spec["lighting"].get("sun"):
        assert Path(spec["lighting"]["sun"]["mask_path"]).exists()
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


def test_fast_pipeline_worker_without_frame_events(tmp_path, monkeypatch):
    """Worker sem os eventos ``frame_ready`` (versão antiga): tudo é composto no fim, a partir do
    ``frames.json``, com o mesmo resultado garantido."""
    from mobilia import pipeline
    from mobilia.jobs import JobRequest, ProgressEmitter

    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(pipeline.ENV_BLENDER_WORKER, json.dumps([sys.executable, str(FAKE), "blender"]))
    monkeypatch.setenv("MOBILIA_DUBLE_SEM_FLUXO", "1")
    _tiny_quality(monkeypatch, pipeline, long_edge=160, samples=8, fps=12)
    _fake_analysis(monkeypatch, tmp_path)
    req = JobRequest(str(FX / "photos/sala_vazia_stafford.jpg"), "sala", str(tmp_path / "v.mp4"),
                     quality="previa", work_dir=str(tmp_path / "job"))
    rep = json.loads(Path(pipeline.run_job(req, ProgressEmitter()).report_path).read_text(encoding="utf-8"))
    c, q = rep["composicao"], rep["render"]["quadros"]
    assert c["compostos_durante_render"] == 0 and c["quadros_compostos"] == q["quadros_novos"]
    assert c["sombra_contato"]["aplicada"] is True          # do frames.json (sem eventos de quadro)
    assert rep["preservacao"]["preservado_exato_todos"] is True
    assert rep["preservacao"]["quadros_verificados"] == q["total"]


def test_fast_pipeline_single_layer_fallback(tmp_path, monkeypatch):
    """Chave de retorno do PT6: ``MOBILIA_CAMADAS=0`` grava ``render.camadas = false`` na cena e o
    worker volta ao quadro único (um render por quadro novo, sem marcos), com a placa intacta."""
    from mobilia import pipeline
    from mobilia.jobs import JobRequest, ProgressEmitter

    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(pipeline.ENV_BLENDER_WORKER, json.dumps([sys.executable, str(FAKE), "blender"]))
    monkeypatch.setenv(pipeline.ENV_CAMADAS, "0")
    _tiny_quality(monkeypatch, pipeline, long_edge=160, samples=8, fps=12)
    _fake_analysis(monkeypatch, tmp_path)
    req = JobRequest(str(FX / "photos/sala_vazia_stafford.jpg"), "sala", str(tmp_path / "v.mp4"),
                     quality="previa", work_dir=str(tmp_path / "job"))
    rep = json.loads(Path(pipeline.run_job(req, ProgressEmitter()).report_path).read_text(encoding="utf-8"))
    spec = json.loads((tmp_path / "job" / "scene.json").read_text(encoding="utf-8"))
    assert spec["render"]["camadas"] is False
    r, q = rep["render"], rep["render"]["quadros"]
    assert r["modo_camadas"] == "unica" and {p["camada"] for p in r["por_quadro"]} == {"dinamico", "final"}
    assert q["renderizados"] == q["quadros_novos"] == rep["composicao"]["quadros_compostos"]
    assert rep["preservacao"]["preservado_exato_todos"] is True


def test_stream_failure_stops_the_job(tmp_path, monkeypatch):
    """Um quadro que muda bytes da foto fora da máscara de influência para o trabalho durante o
    render (o Blender é encerrado), sem MP4, com os EXR ainda não conferidos no disco."""
    from mobilia import pipeline
    from mobilia.composite.compositor import Compositor
    from mobilia.composite.stream import PreservationError
    from mobilia.jobs import JobRequest, ProgressEmitter

    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(pipeline.ENV_BLENDER_WORKER, json.dumps([sys.executable, str(FAKE), "blender"]))
    _tiny_quality(monkeypatch, pipeline, long_edge=160, samples=8, fps=12)
    _fake_analysis(monkeypatch, tmp_path)
    orig = Compositor.composite

    def buggy(self, frame, *a, **kw):
        out, st = orig(self, frame, *a, **kw)
        if frame >= 8:
            out = out.copy()
            out[-1, 0] = 255 - out[-1, 0]               # canto longe do quadrado que entra em cena
        return out, st

    monkeypatch.setattr(Compositor, "composite", buggy)
    out = tmp_path / "v.mp4"
    req = JobRequest(str(FX / "photos/sala_vazia_stafford.jpg"), "sala", str(out), quality="previa",
                     work_dir=str(tmp_path / "job"))
    with pytest.raises(PreservationError, match="quadro 8"):
        pipeline.run_job(req, ProgressEmitter())
    assert not out.exists() and not list(tmp_path.glob("v.partial*"))
    assert list((tmp_path / "job" / "render" / "frames").glob("*.exr"))


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
    # f_00001 só depois de o quadro 2 (reaproveitado, vem do cache) ir para o codificador, não no 1
    assert deleted == [("f_00001.exr", 2), ("f_00003.exr", 3)]
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
