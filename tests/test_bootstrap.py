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
