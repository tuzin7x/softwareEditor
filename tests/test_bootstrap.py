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
    assert bootstrap.pending_steps(bootstrap.load_state(p)) == ["dependencias", "app", "modelos"]


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
    fps = bootstrap.payload_fingerprints(payload)
    state = {"done": list(bootstrap.STEPS), "hashes": dict(fps)}
    # mesmo .exe: nada a refazer
    assert bootstrap.pending_steps(bootstrap.sync_with_payload(dict(state, done=list(state["done"])), fps)) == []
    # mudança só na interface (não vai para o motor): nada a refazer
    (payload / "mobilia" / "ui" / "app.py").write_text("ui = 2\n")
    assert bootstrap.payload_fingerprints(payload) == fps
    # código do motor mudou: só a cópia do app
    (payload / "mobilia" / "pipeline.py").write_text("v = 2\n")
    fps2 = bootstrap.payload_fingerprints(payload)
    s2 = bootstrap.sync_with_payload({"done": list(bootstrap.STEPS), "hashes": dict(fps)}, fps2)
    assert bootstrap.pending_steps(s2) == ["app"]
    # dependências mudaram também: pip + app (Python, pip e modelos não)
    (payload / "requirements" / "runtime-cuda.txt").write_text("torch==2.16.0\n")
    fps3 = bootstrap.payload_fingerprints(payload)
    s3 = bootstrap.sync_with_payload({"done": list(bootstrap.STEPS), "hashes": dict(fps)}, fps3)
    assert bootstrap.pending_steps(s3) == ["dependencias", "app"]


def test_old_state_without_hashes_recopies_app_and_rechecks_deps(tmp_path):
    payload = tmp_path / "payload"
    (payload / "mobilia").mkdir(parents=True)
    (payload / "mobilia" / "x.py").write_text("a\n")
    (payload / "requirements").mkdir()
    (payload / "requirements" / "r.txt").write_text("b\n")
    s = bootstrap.sync_with_payload({"done": list(bootstrap.STEPS)}, bootstrap.payload_fingerprints(payload))
    assert bootstrap.pending_steps(s) == ["dependencias", "app"]


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
