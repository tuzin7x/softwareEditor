"""Lógica testável do instalador de primeiro uso (sem baixar nada)."""
from mobilia import bootstrap


def test_patch_pth_enables_site_and_paths():
    src = "python313.zip\r\n.\r\n\r\n# Uncomment to run site.main() automatically\r\n#import site\r\n"
    out = bootstrap.patch_pth(src, ["Lib\\site-packages", "..\\app"])
    lines = out.split("\r\n")
    assert "import site" in lines and "#import site" not in lines
    assert lines.index("Lib\\site-packages") < lines.index("import site")
    assert bootstrap.patch_pth(out, ["Lib\\site-packages", "..\\app"]) == out     # idempotente


def test_parse_requirements_separates_options():
    opts, reqs = bootstrap.parse_requirements("--extra-index-url https://x/cu132\n# c\ntorch==2.15.0  # gpu\nbpy\n")
    assert opts == ["--extra-index-url", "https://x/cu132"] and reqs == ["torch==2.15.0", "bpy"]


def test_state_resume(tmp_path):
    p = tmp_path / "s.json"
    st = bootstrap.load_state(p)
    assert bootstrap.pending_steps(st) == bootstrap.STEPS
    st["done"] = ["python", "pip"]; bootstrap.save_state(p, st)
    assert bootstrap.pending_steps(bootstrap.load_state(p)) == ["dependencias", "app", "modelos", "ativos",
                                                                "aquecimento"]


def test_requirements_files_exist_and_pin_gpu_wheels():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "requirements"
    opts, reqs = bootstrap.parse_requirements((root / "runtime-cuda.txt").read_text())
    assert any("cu132" in o for o in opts) and any(r.startswith("torch==") for r in reqs)
    assert any(r.startswith("bpy") for r in reqs)


def test_update_redoes_only_changed_steps(tmp_path):
    from pathlib import Path
    payload = tmp_path / "payload"
    (payload / "mobilia" / "ui").mkdir(parents=True)
    (payload / "mobilia" / "pipeline.py").write_text("v = 1\n")
    (payload / "mobilia" / "ui" / "app.py").write_text("ui = 1\n")
    (payload / "requirements").mkdir()
    (payload / "requirements" / "runtime-cuda.txt").write_text("torch==2.15.0\n")
    fps = bootstrap.payload_fingerprints(payload, gpu="")
    state = {"done": list(bootstrap.STEPS), "hashes": dict(fps)}
    # mesmo .exe: nada a refazer
    assert bootstrap.pending_steps(bootstrap.sync_with_payload(dict(state, done=list(state["done"])), fps)) == []
    # mudança só na interface (não vai para o motor): nada a refazer
    (payload / "mobilia" / "ui" / "app.py").write_text("ui = 2\n")
    assert bootstrap.payload_fingerprints(payload, gpu="") == fps
    # código do motor mudou: só a cópia do app
    (payload / "mobilia" / "pipeline.py").write_text("v = 2\n")
    fps2 = bootstrap.payload_fingerprints(payload, gpu="")
    s2 = bootstrap.sync_with_payload({"done": list(bootstrap.STEPS), "hashes": dict(fps)}, fps2)
    assert bootstrap.pending_steps(s2) == ["app"]
    # dependências mudaram também: pip + app + aquecimento (kernels do Blender novo); Python, pip,
    # modelos e móveis não
    (payload / "requirements" / "runtime-cuda.txt").write_text("torch==2.16.0\n")
    fps3 = bootstrap.payload_fingerprints(payload, gpu="")
    s3 = bootstrap.sync_with_payload({"done": list(bootstrap.STEPS), "hashes": dict(fps)}, fps3)
    assert bootstrap.pending_steps(s3) == ["dependencias", "app", "aquecimento"]


def test_old_state_without_hashes_recopies_app_and_rechecks_deps(tmp_path):
    payload = tmp_path / "payload"
    (payload / "mobilia").mkdir(parents=True)
    (payload / "mobilia" / "x.py").write_text("a\n")
    (payload / "requirements").mkdir()
    (payload / "requirements" / "r.txt").write_text("b\n")
    s = bootstrap.sync_with_payload({"done": list(bootstrap.STEPS)}, bootstrap.payload_fingerprints(payload, gpu=""))
    assert bootstrap.pending_steps(s) == ["dependencias", "app", "aquecimento"]


def test_runtime_requirements_never_need_git():
    """O cliente não tem Git: nada de git+ nos requisitos; MoGe vem de ZIP com commit fixo."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "requirements"
    for f in ("runtime-cuda.txt", "runtime-cpu.txt", "runtime-nodeps.txt"):
        opts, reqs = bootstrap.parse_requirements((root / f).read_text())
        assert not any("git+" in r for r in opts + reqs), f
    _, reqs = bootstrap.parse_requirements((root / "runtime-nodeps.txt").read_text())
    assert {r.split()[0] for r in reqs} == {"moge", "utils3d_moge"}
    assert all("/archive/" in r and r.endswith(".zip") for r in reqs)



# ----------------------------------------------------------------------------- PT2: pré-download e aquecimento

def _payload(tmp_path, catalog="CATALOG = []\n", model='GEOMETRY_MODEL_ID = "a/b"\n'):
    payload = tmp_path / "payload"
    (payload / "mobilia" / "layout").mkdir(parents=True, exist_ok=True)
    (payload / "mobilia" / "analysis").mkdir(parents=True, exist_ok=True)
    (payload / "mobilia" / "config.py").write_text(model + "OUTRA = 1\n")
    (payload / "mobilia" / "layout" / "catalog.py").write_text(catalog)
    (payload / "mobilia" / "analysis" / "weights.py").write_text("PATTERNS = []\n")
    (payload / "requirements").mkdir(exist_ok=True)
    (payload / "requirements" / "runtime-cuda.txt").write_text("torch==2.15.0\n")
    return payload


def test_new_steps_follow_app_and_warmup_is_last_and_optional():
    st = bootstrap.STEPS
    assert st.index("app") < st.index("modelos") < st.index("ativos") < st.index("aquecimento") == len(st) - 1
    assert bootstrap.OPTIONAL_STEPS == ("aquecimento",)
    assert bootstrap.TEXTURE_RES["cuda"].split(",") == ["1k", "2k"] and bootstrap.TEXTURE_RES["cpu"] == "1k"


def test_fingerprints_redo_only_the_step_that_changed(tmp_path):
    payload = _payload(tmp_path)
    fps = bootstrap.payload_fingerprints(payload, gpu="RTX 5070 Laptop GPU, 580.97")
    assert all(fps.values())
    full = lambda: {"done": list(bootstrap.STEPS), "hashes": dict(fps)}  # noqa: E731
    pend = lambda f: bootstrap.pending_steps(bootstrap.sync_with_payload(full(), f))  # noqa: E731
    # driver NVIDIA novo: só o aquecimento (caches de kernels do driver são por versão)
    assert pend(bootstrap.payload_fingerprints(payload, gpu="RTX 5070 Laptop GPU, 591.10")) == ["aquecimento"]
    # mudança de código que não toca catálogo nem modelos: só a cópia do motor
    (payload / "mobilia" / "config.py").write_text('GEOMETRY_MODEL_ID = "a/b"\nOUTRA = 2\n')
    assert pend(bootstrap.payload_fingerprints(payload, gpu="RTX 5070 Laptop GPU, 580.97")) == ["app"]
    # móvel novo no catálogo: cópia do motor + conferência dos móveis (baixa só o que falta)
    (payload / "mobilia" / "layout" / "catalog.py").write_text("CATALOG = ['ph:sofa_04']\n")
    assert pend(bootstrap.payload_fingerprints(payload, gpu="RTX 5070 Laptop GPU, 580.97")) == ["app", "ativos"]
    # outro modelo de IA: cópia do motor + modelos (e o catálogo novo de antes)
    (payload / "mobilia" / "config.py").write_text('GEOMETRY_MODEL_ID = "a/c"\nOUTRA = 2\n')
    assert pend(bootstrap.payload_fingerprints(payload, gpu="RTX 5070 Laptop GPU, 580.97")) == [
        "app", "modelos", "ativos"]


class _Steps:
    """Etapas simuladas: contam execuções e podem falhar nas primeiras tentativas."""

    def __init__(self, fail=None):
        self.calls: list[str] = []
        self.fail = dict(fail or {})

    def runners(self, warm_result=None):
        def make(step):
            def run():
                self.calls.append(step)
                if self.fail.get(step, 0) > 0:
                    self.fail[step] -= 1
                    raise bootstrap.SetupError(f"falha simulada em {step}")
                if step == "aquecimento":
                    return {"event": "done", "aquecimento": warm_result or {
                        "ok": True, "blender": {"ok": True, "dispositivos": [
                            {"dispositivo": "OPTIX", "ok": True, "kernels_s": 95.0},
                            {"dispositivo": "CUDA", "ok": True, "kernels_s": 3.0}]},
                        "modelos": {"ok": True}}}
                return None
            return run
        return {s: make(s) for s in bootstrap.STEPS}


def test_run_steps_resumes_after_failure_and_is_idempotent(tmp_path):
    sf = tmp_path / "runtime" / "setup_state.json"
    fps = bootstrap.payload_fingerprints(_payload(tmp_path), gpu="")
    sim = _Steps(fail={"ativos": 1})
    msgs = []
    progress = lambda s, f, m: msgs.append((s, f, m))  # noqa: E731
    # 1ª tentativa: a internet cai no download dos móveis
    try:
        bootstrap.run_steps(progress, bootstrap.load_state(sf), fps, sim.runners(), sf)
        raise AssertionError("deveria falhar")
    except bootstrap.SetupError:
        pass
    st = bootstrap.load_state(sf)
    assert st["done"] == ["python", "pip", "dependencias", "app", "modelos"]
    assert bootstrap.pending_steps(st) == ["ativos", "aquecimento"]
    # 2ª tentativa: continua da etapa que falhou, sem refazer as anteriores
    sim.calls.clear()
    st = bootstrap.run_steps(progress, bootstrap.sync_with_payload(bootstrap.load_state(sf), fps), fps,
                             sim.runners(), sf)
    assert sim.calls == ["ativos", "aquecimento"]
    assert bootstrap.pending_steps(bootstrap.load_state(sf)) == []
    assert bootstrap.load_state(sf)["hashes"] == {k: v for k, v in fps.items() if v}
    # 3ª: nada a fazer (idempotente), e o estado não muda
    before = sf.read_text()
    sim.calls.clear()
    bootstrap.run_steps(progress, bootstrap.sync_with_payload(bootstrap.load_state(sf), fps), fps,
                        sim.runners(), sf)
    assert sim.calls == [] and sf.read_text() == before
    assert any(s == "aquecimento" and f == 1.0 for s, f, _ in msgs)


def test_warmup_is_recorded_and_marks_first_run(tmp_path):
    sf = tmp_path / "setup_state.json"
    fps = bootstrap.payload_fingerprints(_payload(tmp_path), gpu="")
    st = {"done": [s for s in bootstrap.STEPS if s != "aquecimento"], "hashes": dict(fps),
          "execucoes": 4, "execucoes_dependencias": fps["dependencias"]}
    assert not bootstrap.is_first_run(st)
    st = bootstrap.run_steps(lambda *a: None, st, fps, _Steps().runners(), sf)
    st = bootstrap.load_state(sf)
    w = bootstrap.warmup_status(st)
    assert w["feito"] and w["ok"] and w["dispositivos"] == ["OPTIX", "CUDA"] and w["blender_kernels_s"] == 98.0
    assert w["modelos_ok"] is True and w["parede_s"] is not None
    # a próxima geração é "primeira execução" (medida à parte), com o aviso brando
    assert bootstrap.is_first_run(st)
    assert "um pouco mais lenta" in bootstrap.first_run_notice(st)
    bootstrap.record_run(sf)
    st = bootstrap.load_state(sf)
    assert not bootstrap.is_first_run(st) and bootstrap.first_run_notice(st) is None
    # dependências trocadas depois do aquecimento: o aquecimento antigo não vale mais
    st["hashes"]["dependencias"] = "outra"
    assert bootstrap.warmup_status(st)["feito"] is False


def test_warmup_failure_never_blocks_installation(tmp_path):
    sf = tmp_path / "setup_state.json"
    fps = bootstrap.payload_fingerprints(_payload(tmp_path), gpu="")
    st = {"done": [s for s in bootstrap.STEPS if s != "aquecimento"], "hashes": dict(fps)}
    msgs = []
    st = bootstrap.run_steps(lambda s, f, m: msgs.append(m), st, fps, _Steps(fail={"aquecimento": 1}).runners(), sf)
    st = bootstrap.load_state(sf)
    assert bootstrap.pending_steps(st) == []                     # o programa abre normalmente
    w = bootstrap.warmup_status(st)
    assert w["feito"] and not w["ok"] and "falha simulada" in w["erro"]
    assert "mais lenta" in bootstrap.first_run_notice(st) and "kernels" in bootstrap.first_run_notice(st)
    assert any("primeiro vídeo vai ser mais lento" in m for m in msgs)
    # o mesmo vale para um aquecimento que roda mas não consegue usar o Blender
    st2 = {"done": [s for s in bootstrap.STEPS if s != "aquecimento"], "hashes": dict(fps)}
    st2 = bootstrap.run_steps(lambda *a: None, st2, fps,
                              _Steps().runners({"ok": False, "blender": {"ok": False, "erro": "x"}}), sf)
    assert not bootstrap.warmup_status(st2)["ok"] and bootstrap.pending_steps(st2) == []


def test_other_step_failures_still_raise(tmp_path):
    sf = tmp_path / "setup_state.json"
    fps = bootstrap.payload_fingerprints(_payload(tmp_path), gpu="")
    import pytest
    with pytest.raises(bootstrap.SetupError):
        bootstrap.run_steps(lambda *a: None, {"done": []}, fps, _Steps(fail={"modelos": 1}).runners(), sf)
    assert bootstrap.pending_steps(bootstrap.load_state(sf))[0] == "modelos"


def test_step_runners_call_the_engine_cli(tmp_path, monkeypatch):
    from mobilia import runtime
    calls = []
    monkeypatch.setattr(bootstrap, "_run", lambda cmd, progress, step, env=None, log=None:
                        calls.append((step, cmd, env)) or {"event": "done", "aquecimento": {"ok": True}})
    paths = runtime.runtime_paths(tmp_path / "home")
    for variant, res in (("cuda", "1k,2k"), ("cpu", "1k")):
        calls.clear()
        r = bootstrap.step_runners(lambda *a: None, paths, tmp_path, {"variant": variant})
        r["modelos"](); r["ativos"](); out = r["aquecimento"]()
        (s1, c1, env), (s2, c2, _), (s3, c3, _) = calls
        assert c1[0] == str(paths.python_exe) and c1[-3:] == ["prefetch", "--json", "--modelos"]
        assert c2[-5:] == ["prefetch", "--json", "--ativos", "--res", res]
        assert c3[-2:] == ["aquecer", "--json"] and out["aquecimento"]["ok"] is True
        # caches de kernels da NVIDIA grandes no processo do motor (e no Blender, filho dele)
        assert env["CUDA_CACHE_MAXSIZE"] == str(4 << 30) and env["OPTIX_CACHE_MAXSIZE"] == str(4 << 30)


def test_run_returns_done_event_and_forwards_logs(tmp_path):
    import sys
    script = tmp_path / "w.py"
    script.write_text("import json\n"
                      "print(json.dumps({'event': 'progress', 'fraction': 0.5, 'message': 'meio'}))\n"
                      "print(json.dumps({'event': 'log', 'level': 'warning', 'message': 'aviso'}))\n"
                      "print(json.dumps({'event': 'done', 'aquecimento': {'ok': True}}))\n")
    msgs = []
    done = bootstrap._run([sys.executable, str(script)], lambda s, f, m: msgs.append((f, m)), "aquecimento")
    assert done == {"event": "done", "aquecimento": {"ok": True}}
    assert (0.5, "meio") in msgs and (-1.0, "aviso") in msgs


def test_cpu_installer_warmup_runs_real_blender_and_saves_state(tmp_path, monkeypatch):
    """Validação em CPU (PT2): a etapa de aquecimento, pela mesma função do instalador, roda a CLI
    real do motor (``aquecer``) com o Blender (bpy) de verdade em 64 × 64 na CPU, e o estado é salvo."""
    import importlib.util
    import pytest
    if importlib.util.find_spec("bpy") is None:
        pytest.skip("bpy não instalado")
    from mobilia import runtime
    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MOBILIA_BLENDER_WORKER", raising=False)
    monkeypatch.delenv("MOBILIA_WORKER", raising=False)
    paths = runtime.runtime_paths(tmp_path / "home")
    sf = paths.state_file
    fps = bootstrap.payload_fingerprints(runtime.payload_root(), gpu="")
    env = runtime.worker_environment(frozen=False, paths=paths)
    cmd = runtime.worker_command(["aquecer", "--json", "--sem-modelos"], frozen=False)
    st = {"done": [s for s in bootstrap.STEPS if s != "aquecimento"], "hashes": dict(fps)}
    msgs = []
    runners = {"aquecimento": lambda: bootstrap._run(cmd, lambda s, f, m: msgs.append(m), "aquecimento", env=env)}
    bootstrap.run_steps(lambda s, f, m: msgs.append(m), st, fps, runners, sf)
    st = bootstrap.load_state(sf)
    w = bootstrap.warmup_status(st)
    assert w["feito"] and w["ok"], st["aquecimento"]
    assert w["dispositivos"] == ["CPU"]
    devs = st["aquecimento"]["resultado"]["blender"]["dispositivos"]
    assert len(devs[0]["renders"]) == 3 and all(r["amostras"] for r in devs[0]["renders"])
    assert st["primeira_execucao_pendente"] is True
    assert any("Kernels de render prontos" in m for m in msgs)
