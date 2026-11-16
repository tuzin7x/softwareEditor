"""Telemetria (PT1): cronômetro, leitura das mensagens do Cycles, resumo do render,
estado de primeira execução e pacote de diagnóstico."""
import json
import zipfile

import pytest

from mobilia import bootstrap, diagnostics
from mobilia.telemetry import (ETAPAS, RenderStatsCollector, Stopwatch, VramMonitor, env_time, stages_sum,
                               summarize_render)

# Mensagens reais do handler ``render_stats`` (Blender 5.3, CPU, 1600 × 1200, 4 amostras, denoiser
# OIDN ligado), com o instante relativo em que chegaram.
CYCLES_DENOISE = [
    (0.0041, "Mem: 1M | Updating Procedurals"),
    (0.0045, "Mem: 1M | Updating Scene BVH | Building"),
    (0.0055, "Mem: 1M | Loading denoising kernels (may take a few minutes the first time)"),
    (0.0055 + 1.5, "Mem: 1M | Sample 0/4"),
    (1.7555, "Remaining: 00:00.75 | Mem: 191M | Sample 1/4"),
    (10.5588, "Mem: 191M | Sample 4/4"),
    (10.5588, "Mem: 191M | Finished"),
    (10.742, "Time: 00:10.74 (Saving: 00:00.11)"),
]


def _collect(msgs, denoise=True, t_end=None):
    c = RenderStatsCollector(clock=lambda: 0.0)
    c.start()
    for t, m in msgs:
        c.feed(m, t)
    return c.finish(denoise=denoise, t_end=t_end if t_end is not None else msgs[-1][0] + 0.001)


def test_stopwatch_nested_and_sum():
    sw = Stopwatch()
    sw.add("analise.geometria_carga", 1.0)
    sw.add("analise.geometria_carga", 0.5)
    sw.set("layout", 2.0)
    with sw.measure("analise.sala"):
        pass
    assert sw.get("analise.geometria_carga") == 1.5
    assert sw.get("nao.existe", None) is None
    t = sw.rounded()
    assert t["analise"]["geometria_carga"] == 1.5 and "sala" in t["analise"]
    t["total"] = 99.0                                  # chaves fora de ETAPAS não entram na soma
    assert stages_sum(t) == pytest.approx(3.5, abs=0.01)
    assert ETAPAS[0] == "inicio_motor" and "render_dinamico" in ETAPAS and "total" not in ETAPAS


def test_render_stats_split_sampling_denoise_saving():
    st = _collect(CYCLES_DENOISE)
    assert st["kernels_s"] == pytest.approx(1.5, abs=1e-3)          # carga dos kernels do denoiser
    assert st["sincronizacao_s"] == pytest.approx(1.5055, abs=1e-3)
    assert st["amostras"] == 4 and st["amostras_max"] == 4
    # ritmo da 1ª amostra (0,25 s) × 4 = 1,0 s de amostragem; o resto até "Finished" é denoise
    assert st["amostragem_s"] == pytest.approx(1.0, abs=0.01)
    assert st["denoise_s"] == pytest.approx(8.05, abs=0.02) and st["denoise_estimado"]
    assert st["gravacao_s"] == pytest.approx(0.11)
    assert st["mem_mb"] == 191.0


def test_render_stats_without_denoise_or_intermediate_samples():
    msgs = [(0.01, "Mem: 2M | Sample 0/16"), (0.81, "Mem: 1.5G | Sample 16/16"), (0.81, "Mem: 1.5G | Finished")]
    st = _collect(msgs, denoise=False, t_end=0.86)
    assert st["amostragem_s"] == pytest.approx(0.8) and st["denoise_s"] == 0.0
    assert st["gravacao_s"] == pytest.approx(0.05, abs=1e-3)          # sem "Saving": até o fim da chamada
    assert st["mem_mb"] == 1536.0
    # com denoiser mas sem amostra intermediária: não dá para separar
    st = _collect(msgs, denoise=True, t_end=0.86)
    assert st["amostragem_s"] == pytest.approx(0.8) and st["denoise_s"] is None
    # chamada sem nenhuma mensagem (ex.: dublê): só o tempo de parede
    st = _collect([(0.0, "Mem: 1M")], t_end=0.2)
    assert st["parede_s"] == pytest.approx(0.2) and st["amostragem_s"] is None


def test_summarize_render_layers_and_throughput():
    rows = [
        {"quadro": 10, "camada": "dinamico", "area_px": 10_000, "spp": 16, "amostras": 16,
         "amostragem_s": 0.2, "parede_s": 0.5, "denoise_s": 0.1, "gravacao_s": 0.05, "mem_mb": 300},
        {"quadro": 11, "camada": "dinamico", "area_px": 30_000, "spp": 16, "amostras": 16,
         "amostragem_s": 0.4, "parede_s": 0.7},
        {"quadro": 20, "camada": "marco", "itens": 3, "area_px": 50_000, "spp": 64, "amostras": 64,
         "amostragem_s": 1.0, "parede_s": 1.3},
        {"quadro": 30, "camada": "final", "area_px": 100_000, "spp": 128, "amostras": 128,
         "amostragem_s": 2.0, "parede_s": 2.4, "mem_mb": 900},
    ]
    s = summarize_render(rows, 400, 500)
    c = s["camadas"]
    assert c["dinamico"]["quadros"] == 2 and c["dinamico"]["area_media_mp"] == 0.02
    assert c["dinamico"]["s"] == pytest.approx(1.2)
    assert c["marcos"] == [{"quadro": 20, "itens": 3, "area_mp": 0.05, "spp": 64, "s": 1.3}]
    assert c["final"]["spp"] == 128 and c["final"]["cobertura_media"] == 0.5
    d = s["desempenho"]
    assert d["medido_em"] == "final" and d["amostras_por_s"] == 100_000 * 128 / 2.0
    assert d["custo_fixo_quadro_s"] == pytest.approx(0.3)               # mediana de 0,3 0,3 0,3 0,4
    assert d["mem_cycles_pico_mb"] == 900
    assert s["cobertura_recorte_media"] == pytest.approx(190_000 / 4 / 200_000)
    assert summarize_render([], 10, 10)["desempenho"]["amostras_por_s"] is None


def test_vram_monitor_inactive_without_nvml():
    m = VramMonitor(interval_s=0.01).start()
    with m.track("analise"):
        pass
    if not m.active:                                   # máquina sem GPU NVIDIA (CI, este ambiente)
        assert m.peak("analise") is None
    m.stop()


def test_env_time_rejects_bad_values():
    import time
    now = time.time()
    assert env_time("X", {"X": repr(now - 5)}) == pytest.approx(now - 5)
    assert env_time("X", {"X": "abc"}) is None
    assert env_time("X", {"X": repr(now + 3600)}) is None
    assert env_time("X", {"X": repr(now - 3 * 86400)}) is None
    assert env_time("X", {}) is None


def test_first_run_state(tmp_path):
    p = tmp_path / "setup_state.json"
    assert bootstrap.is_first_run(bootstrap.load_state(p))
    st = bootstrap.record_run(p)
    assert st["execucoes"] == 1 and not bootstrap.is_first_run(bootstrap.load_state(p))
    # dependências reinstaladas (Blender/PyTorch novos): volta a ser primeira execução
    st = bootstrap.load_state(p)
    st.setdefault("hashes", {})["dependencias"] = "novo"
    bootstrap.save_state(p, st)
    assert bootstrap.is_first_run(bootstrap.load_state(p))
    bootstrap.record_run(p)
    assert not bootstrap.is_first_run(bootstrap.load_state(p))


def _job_dir(tmp_path):
    from PIL import Image

    photo = tmp_path / "cliente.jpg"
    Image.new("RGB", (32, 24), (200, 180, 160)).save(photo)
    work = tmp_path / "job"
    (work / "render").mkdir(parents=True)
    (work / "analise").mkdir()
    (work / "quadros").mkdir()
    (work / "request.json").write_text(json.dumps({"photo_path": str(photo)}), encoding="utf-8")
    (work / "report.json").write_text(json.dumps({"tempos_s": {"layout": 1.0}}), encoding="utf-8")
    (work / "render" / "frames.json").write_text("{}", encoding="utf-8")
    (work / "analise" / "room.json").write_text("{}", encoding="utf-8")
    (work / "encode.log").write_text("x", encoding="utf-8")
    Image.new("RGB", (32, 24)).save(work / "plate.png")
    Image.new("RGB", (32, 24)).save(work / "quadros" / "quadro_00001.jpg")
    Image.new("RGB", (32, 24)).save(work / "analise" / "analise_overlay.jpg")
    (work / "analise" / "geometry.npz").write_bytes(b"0" * 100)
    (work / "render" / "frames").mkdir()
    (work / "render" / "frames" / "f_00001.exr").write_bytes(b"0" * 100)
    return work


IMAGE_EXT = (".jpg", ".jpeg", ".png", ".exr", ".npz", ".mp4")


def test_diagnostic_zip_without_photo_by_default(tmp_path):
    work = _job_dir(tmp_path)
    z = diagnostics.make_diagnostic_zip(work, tmp_path / "out" / "d.zip", extra_log="saida do motor")
    names = zipfile.ZipFile(z).namelist()
    assert {"report.json", "render/frames.json", "analise/room.json", "encode.log", "interface.log",
            "manifesto.json", "request.json"} <= set(names)
    assert not [n for n in names if n.lower().endswith(IMAGE_EXT)]
    man = json.loads(zipfile.ZipFile(z).read("manifesto.json"))
    assert man["inclui_foto"] is False
    z2 = diagnostics.make_diagnostic_zip(work, tmp_path / "out" / "d2.zip", include_photo=True)
    names = zipfile.ZipFile(z2).namelist()
    assert "foto/cliente.jpg" in names and "plate.png" in names and "quadros/quadro_00001.jpg" in names
    assert not [n for n in names if n.endswith((".exr", ".npz"))]        # nunca: pesados e inúteis


def test_stage_summary_text():
    rep = {"primeira_execucao": True,
           "tempos_s": {"inicio_motor": 1.0, "carga_foto": 0.5, "analise": {"a": 10.0, "b": 2.0},
                        "layout": 3.0, "ativos": 0.2, "download": 0.0, "blender_inicio": 2.0,
                        "render_dinamico": 70.0, "composicao": 5.0, "total": 95.0, "total_percebido": 96.0}}
    rows = dict(diagnostics.stage_summary(rep))
    assert rows["Análise da foto"] == 12.0 and "Download de modelos" not in rows
    assert rows["Render dos quadros"] == 70.0
    txt = diagnostics.summary_text(rep)
    assert "Total: 1 min 36 s" in txt and "Primeira execução" in txt
    assert diagnostics.format_seconds(4.25) == "4.2 s" and diagnostics.format_seconds(42) == "42 s"
