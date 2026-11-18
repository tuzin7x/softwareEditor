"""PT2: pré-download dos pesos/móveis e aquecimento do instalador (sem rede e, nos rápidos, sem GPU)."""
import json
import sys
from pathlib import Path

import pytest

FAKE = Path(__file__).parent / "fake_worker.py"


def test_ensure_local_separates_network_time(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")   # não está nas dependências do CI

    from mobilia.analysis import weights

    cache: set = set()
    calls = []

    def snapshot_download(repo, local_files_only=False, allow_patterns=None, **kw):
        if local_files_only:
            if repo not in cache:
                raise FileNotFoundError(repo)
            return "/cache/" + repo
        calls.append(("snapshot", repo))
        cache.add(repo)
        return "/cache/" + repo

    def hf_hub_download(repo, fname, repo_type=None, **kw):
        calls.append(("arquivo", repo, fname, repo_type))
        cache.add((repo, fname))
        return "/cache/x"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, fname, repo_type=None: "/cache/x" if (repo, fname) in cache else None)
    weights.reset_net_stats()
    seg = weights.SEGMENTATION_MODEL_ID
    assert weights.ensure_local(seg) > 0.0
    # pesos + metadados das classes do ADE20K (repositório de dados à parte)
    assert calls == [("snapshot", seg), ("arquivo", *weights.ONEFORMER_METADATA, "dataset")]
    assert weights.NET_STATS["repos"] == 1 and weights.NET_STATS["s"] > 0
    calls.clear()
    assert weights.ensure_local(seg) == 0.0 and calls == []          # já no cache: sem rede
    weights.prefetch_models(v3=False, design=False)   # V3 e redesenho têm testes próprios
    assert [c[1] for c in calls] == weights.model_repos() + [weights.ONEFORMER_METADATA[0]]


def test_model_loaders_report_download_apart(monkeypatch):
    """A carga dos modelos não conta o tempo de rede: ele vai para ``tempos_s.download``."""
    from mobilia import pipeline
    from mobilia.telemetry import Stopwatch

    sw = Stopwatch()
    assert pipeline._minus_download(sw, 5.0, {"carga": 4.0, "download": 3.0}) == 2.0
    assert sw.get("download") == 3.0
    assert pipeline._minus_download(sw, 1.0, {"carga": 0.5}) == 1.0 and sw.get("download") == 3.0


def test_prefetch_cli_parts(monkeypatch, capsys):
    from mobilia import cli
    from mobilia.analysis import weights
    from mobilia.assets import polyhaven

    got = {}
    monkeypatch.setattr(weights, "prefetch_models", lambda progress=None: got.setdefault("modelos", True))
    monkeypatch.setattr(polyhaven, "prefetch_catalog",
                        lambda d, res, progress=None: got.setdefault("res", res) and [])
    assert cli.main(["prefetch", "--json", "--ativos", "--res", "1k,2k"]) == 0
    assert got == {"res": ("1k", "2k")}
    got.clear()
    assert cli.main(["prefetch", "--json", "--modelos"]) == 0
    assert got == {"modelos": True}
    got.clear()
    assert cli.main(["prefetch", "--json"]) == 0                     # sem opção: tudo
    assert set(got) == {"modelos", "res"}
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last == {"event": "done", "failed": []}


def test_warm_blender_parses_worker_events(monkeypatch):
    from mobilia import pipeline, warmup

    monkeypatch.setenv(pipeline.ENV_BLENDER_WORKER, json.dumps([sys.executable, str(FAKE), "blender"]))
    evs = []
    res = warmup.warm_blender(evs.append, span=(0.0, 0.5))
    assert res["ok"] and res["dispositivos"][0]["dispositivo"] == "CPU" and res["processo_s"] > 0
    fr = [e["fraction"] for e in evs if e["event"] == "progress"]
    assert fr and max(fr) == 0.5                                     # escala para a faixa pedida
    assert any(e["event"] == "log" and "CPU" in e["message"] for e in evs)
    # worker que não sobe: resultado com erro, sem exceção (o instalador registra e segue)
    monkeypatch.setenv(pipeline.ENV_BLENDER_WORKER, json.dumps([sys.executable, "-c", "raise SystemExit(3)"]))
    bad = warmup.warm_blender(evs.append)
    assert bad["ok"] is False and "3" in bad["erro"]


def test_warm_blender_survives_locked_temp_files(monkeypatch):
    """No Windows o antivírus pode segurar um EXR recém-gravado: a limpeza da pasta temporária falha,
    mas o aquecimento que deu certo continua dando certo (não vira falha no setup_state.json)."""
    import os
    import shutil
    import tempfile

    from mobilia import pipeline, warmup

    real = shutil.rmtree

    def locked_rmtree(name, onexc=None, **kw):
        real(name, ignore_errors=True)
        onexc(os.unlink, os.path.join(name, "optix_0.exr"),
              OSError(32, "arquivo em uso por outro processo"))

    monkeypatch.setattr(tempfile._shutil, "rmtree", locked_rmtree)
    monkeypatch.setenv(pipeline.ENV_BLENDER_WORKER, json.dumps([sys.executable, str(FAKE), "blender"]))
    res = warmup.warm_blender(lambda ev: None)
    assert res["ok"] and res["dispositivos"][0]["ok"]


def test_warm_models_records_failures(monkeypatch):
    from mobilia import warmup
    from mobilia.analysis import geometry, segmentation
    from mobilia.hardware import Profile

    prof = Profile("cpu", "cpu", False, "x/moge", 1500, "CPU", 32, 1280, "teste")

    def fake_geo(rgb, timings=None, **kw):
        assert rgb.shape == (*warmup.WARM_IMAGE, 3)
        timings.update(carga=0.0, download=0.0)

    def bad_seg(rgb, timings=None, **kw):
        raise RuntimeError("sem pesos")

    monkeypatch.setattr(geometry, "estimate_geometry", fake_geo)
    monkeypatch.setattr(segmentation, "segment", bad_seg)
    evs = []
    res = warmup.warm_models(evs.append, prof=prof)
    assert res["geometria"]["ok"] and not res["segmentacao"]["ok"] and res["ok"] is False
    assert "sem pesos" in res["segmentacao"]["erro"]
    assert any(e["event"] == "log" and e["level"] == "warning" for e in evs)


def test_warmup_cli_always_reports(monkeypatch, capsys):
    from mobilia import cli, pipeline

    monkeypatch.setenv(pipeline.ENV_BLENDER_WORKER, json.dumps([sys.executable, str(FAKE), "blender"]))
    assert cli.main(["aquecer", "--json", "--sem-modelos"]) == 0
    done = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert done["event"] == "done" and done["aquecimento"]["ok"] is True
    assert set(done["aquecimento"]) >= {"blender", "ok", "parede_s"} and "modelos" not in done["aquecimento"]


def test_gpu_cache_env(monkeypatch):
    from mobilia import pipeline, runtime

    env = runtime.worker_environment({"PATH": "x"}, frozen=False)
    assert env["CUDA_CACHE_MAXSIZE"] == "4294967296" and env["OPTIX_CACHE_MAXSIZE"] == "4294967296"
    # escolha do usuário é respeitada
    assert runtime.worker_environment({"CUDA_CACHE_MAXSIZE": "123"}, frozen=False)["CUDA_CACHE_MAXSIZE"] == "123"
    monkeypatch.delenv("CUDA_CACHE_MAXSIZE", raising=False)
    assert pipeline.blender_worker_env()["CUDA_CACHE_MAXSIZE"] == "4294967296"


@pytest.mark.slow
@pytest.mark.models
def test_real_model_warmup_cpu():
    """Carga real dos dois modelos na CPU com a imagem sintética (precisa dos pesos no cache)."""
    from mobilia import warmup
    from mobilia.hardware import choose_profile, detect_hardware

    res = warmup.warm_models(lambda ev: None, prof=choose_profile(detect_hardware()))
    assert res["ok"], res
    assert res["geometria"]["carga_s"] > 0 and res["segmentacao"]["inferencia_s"] > 0
