"""Sistema de Releases oficiais: script das notas (scripts/release_notes.py) e regras dos workflows.

Os workflows não rodam aqui; estes testes conferem de forma estática o que não pode regredir (só a main publica,
permissões mínimas, ações fixadas por hash, builds de desenvolvimento sem Release, fonte única da versão).
"""
from __future__ import annotations

import hashlib
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"
sys.path.insert(0, str(ROOT / "scripts"))
import release_notes as rn  # noqa: E402

sys.path.remove(str(ROOT / "scripts"))

CHANGELOG_OK = """# Changelog

## [Não publicado]

### Novidades

- algo futuro

## [1.2.0] - 2026-12-01

### Novidades

- Função nova A
- Função nova B

### Correções

- Corrige o vídeo 9:16

### Segurança

- Atualiza dependência X

## [1.1.0] - 2026-11-01

### Correções

- antigo

[1.2.0]: https://github.com/o/r/releases/tag/v1.2.0
"""


def _fake_repo(tmp_path: Path, version: str, changelog: str = CHANGELOG_OK) -> Path:
    (tmp_path / "mobilia").mkdir()
    (tmp_path / "mobilia" / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    return tmp_path


def _assets(tmp_path: Path, version: str) -> Path:
    d = tmp_path / "release"
    d.mkdir()
    (d / f"7xAI-Setup-v{version}.exe").write_bytes(b"setup" * 1000)
    (d / f"7xAI-Portable-v{version}.exe").write_bytes(b"portable" * 1000)
    return d


# ------------------------------------------------------------------------------------- versão (fonte única)

def test_version_has_a_single_source():
    sys.path.insert(0, str(ROOT / "packaging"))
    try:
        import win_version
    finally:
        sys.path.remove(str(ROOT / "packaging"))
    from mobilia import __version__
    assert rn.code_version() == __version__ == win_version.app_version(ROOT)
    rn.check_pyproject()                                        # pyproject sem versão escrita à mão
    iss = (ROOT / "packaging" / "installer.iss").read_bytes()[3:].decode("utf-8")
    assert not re.search(r'#define AppVersion\s+"', iss), "o instalador não pode ter a versão fixa"
    assert "#error" in iss


@pytest.mark.parametrize("bad", ["", "1.0", "v1.0.0", "1.0.0-beta", "1.0.0+5", "01.0.0", "1.0.0 ", "1.0.0;rm"])
def test_check_rejects_non_semver(tmp_path, bad):
    root = _fake_repo(tmp_path, "1.2.0")
    with pytest.raises(rn.ReleaseError, match="inválida"):
        rn.check(bad, root)


def test_check_requires_code_version(tmp_path):
    root = _fake_repo(tmp_path, "1.1.0")
    with pytest.raises(rn.ReleaseError, match="não é a do código"):
        rn.check("1.2.0", root)


def test_check_rejects_fixed_version_in_pyproject(tmp_path):
    root = _fake_repo(tmp_path, "1.2.0")
    p = root / "pyproject.toml"
    p.write_text(p.read_text(encoding="utf-8").replace('dynamic = ["version"]', 'version = "1.2.0"'),
                 encoding="utf-8")
    with pytest.raises(rn.ReleaseError, match="pyproject"):
        rn.check("1.2.0", root)


# ------------------------------------------------------------------------------------------------ CHANGELOG

def test_check_accepts_published_section(tmp_path):
    sec = rn.check("1.2.0", _fake_repo(tmp_path, "1.2.0"))
    assert sec.date == "2026-12-01" and not sec.is_draft
    assert list(sec.subsections) == ["Novidades", "Correções", "Segurança"]
    assert "antigo" not in sec.body and "releases/tag" not in sec.body   # não vaza para a versão seguinte


def test_check_requires_section(tmp_path):
    root = _fake_repo(tmp_path, "1.3.0")
    with pytest.raises(rn.ReleaseError, match=r"não tem a seção '## \[1.3.0\]'"):
        rn.check("1.3.0", root)


def test_check_refuses_draft_and_empty_sections(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    draft = CHANGELOG_OK.replace("## [1.2.0] - 2026-12-01", "## [1.2.0] - RASCUNHO")
    with pytest.raises(rn.ReleaseError, match="RASCUNHO"):
        rn.check("1.2.0", _fake_repo(tmp_path / "a", "1.2.0", draft))
    empty = "# Changelog\n\n## [1.2.0] - 2026-12-01\n\n### Novidades\n\n"
    with pytest.raises(rn.ReleaseError, match="vazia"):
        rn.check("1.2.0", _fake_repo(tmp_path / "b", "1.2.0", empty))


def test_real_changelog_has_unreleased_and_published_first_version(tmp_path):
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Não publicado]" in text
    assert rn.code_version() == "0.9.1", "versão em desenvolvimento: 0.9.1 (a 0.9.0 já foi publicada)"
    cur = rn.find_section(text, "0.9.1")
    assert cur is not None and "Melhorar foto" in cur.body   # rascunho até a data ser fechada na publicação
    sec = rn.find_section(text, "0.9.0")
    assert sec is not None and not sec.is_draft, "a 0.9.0 foi aprovada pelo dono e sai como primeira versão oficial"
    for item in ("Mudar estilo", "Reforma completa", "Redesign simples", "Mobiliar ambiente vazio", "Editar área",
                 "Fachada", "Decoração de evento", "Texto para ambiente", "Ampliar 4K", "Vídeo antes e depois",
                 "Lead Finder"):
        assert item in sec.body, item
    root = tmp_path / "r"
    root.mkdir()
    shutil.copy(ROOT / "CHANGELOG.md", _fake_repo(root, "0.9.0") / "CHANGELOG.md")
    assert rn.check("0.9.0", root).heading.startswith("## [0.9.0] - 20")   # pronta para o workflow publicar
    with pytest.raises(rn.ReleaseError):                          # e não existe uma 1.0.0 para publicar
        rn.check("1.0.0", root)


# ----------------------------------------------------------------------------------------------- notas/SHA

def test_notes_have_version_date_categories_and_sha_table(tmp_path):
    root = _fake_repo(tmp_path, "1.2.0")
    sec = rn.check("1.2.0", root)
    assets = _assets(tmp_path, "1.2.0")
    sha = "a" * 40
    notes = rn.render_notes(sec, "1.2.0", sorted(assets.iterdir()), sha, "2026-12-02", "o/r")
    assert notes.startswith("# 7xAi v1.2.0\n")
    assert "**Data:** 2026-12-02" in notes and "**Versão:** 1.2.0" in notes
    assert f"https://github.com/o/r/commit/{sha}" in notes
    for title in ("## Novidades", "## Correções", "## Melhorias de desempenho", "## Notas importantes de atualização",
                  "## Segurança", "## Verificação de integridade (SHA-256)"):
        assert title in notes, title
    assert "- Função nova A" in notes and "- Corrige o vídeo 9:16" in notes and "- Atualiza dependência X" in notes
    perf = notes.split("## Melhorias de desempenho", 1)[1].split("##", 1)[0]
    assert "Nada nesta versão" in perf
    for p in assets.iterdir():
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert f"| `{p.name}` | `{digest}` |" in notes
    assert "algo futuro" not in notes and "antigo" not in notes


def test_notes_refuse_missing_asset(tmp_path):
    root = _fake_repo(tmp_path, "1.2.0")
    assets = _assets(tmp_path, "1.2.0")
    (assets / "7xAI-Portable-v1.2.0.exe").unlink()
    with pytest.raises(rn.ReleaseError, match="Portable"):
        rn.render_notes(rn.check("1.2.0", root), "1.2.0", list(assets.iterdir()), "b" * 40, "2026-12-02")


def test_sums_file_matches_sha256sum_format(tmp_path):
    assets = _assets(tmp_path, "1.2.0")
    assert rn.main(["sums", "--assets", str(assets)]) == 0
    lines = (assets / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        digest, name = line.split("  ")
        assert hashlib.sha256((assets / name).read_bytes()).hexdigest() == digest
    assert rn.asset_names("1.2.0") == {"setup": "7xAI-Setup-v1.2.0.exe", "portable": "7xAI-Portable-v1.2.0.exe",
                                       "sums": "SHA256SUMS.txt"}


def test_cli_check_reports_error(capsys):
    assert rn.main(["check", "--version", "9.9.9"]) == 1
    assert "::error::" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------- workflows

yaml = pytest.importorskip("yaml")


def _wf(name: str) -> dict:
    d = yaml.safe_load((WF / name).read_text(encoding="utf-8"))
    d["on"] = d.pop(True, d.get("on"))                        # o YAML 1.1 lê "on" como True
    return d


def test_all_actions_pinned_by_commit_hash():
    for f in WF.glob("*.yml"):
        for m in re.finditer(r"uses:\s*(\S+)", f.read_text(encoding="utf-8")):
            ref = m.group(1)
            if ref.startswith("./"):
                continue
            assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", ref), f"{f.name}: {ref} sem hash fixo"


def test_release_workflow_is_manual_main_only_and_least_privilege():
    wf = _wf("release.yml")
    assert list(wf["on"]) == ["workflow_dispatch"]
    assert wf["on"]["workflow_dispatch"]["inputs"]["versao"]["required"] is True
    assert wf["permissions"] == {}
    assert wf["concurrency"]["group"] and wf["concurrency"]["cancel-in-progress"] is False
    jobs = wf["jobs"]
    assert list(jobs) == ["validar", "testes", "build", "publicar"]
    writers = [n for n, j in jobs.items() if (j.get("permissions") or {}).get("contents") == "write"]
    assert writers == ["publicar"]
    assert set(jobs["publicar"]["needs"]) == {"validar", "testes", "build"}
    assert set(jobs["build"]["needs"]) == {"validar", "testes"}
    assert jobs["testes"]["uses"] == "./.github/workflows/tests.yml"
    text = (WF / "release.yml").read_text(encoding="utf-8")
    assert '"$GITHUB_REF" != "refs/heads/main"' in text
    assert "release_notes.py check" in text and "sha256sum -c SHA256SUMS.txt" in text
    assert "--target \"$GITHUB_SHA\"" in text and "--draft" in text
    assert '.object.sha' in text                                # a tag aponta para o commit testado
    assert "${{ inputs.versao }}" not in text.split("env:", 1)[1].split("jobs:", 1)[1], \
        "a versão digitada só entra nos scripts por variável de ambiente (sem injeção)"
    for name, job in jobs.items():
        for step in job.get("steps", []):
            if "actions/checkout" in step.get("uses", ""):
                assert step["with"]["ref"] == "${{ github.sha }}", name
                assert step["with"]["persist-credentials"] is False, name


def test_release_latest_only_for_stable_and_optional_prerelease():
    wf = _wf("release.yml")
    pre = wf["on"]["workflow_dispatch"]["inputs"]["prerelease"]
    assert pre["type"] == "boolean" and pre["default"] is False and pre["required"] is False
    assert "PRÉ-LANÇAMENTO" in pre["description"] and "Latest" in pre["description"]
    text = (WF / "release.yml").read_text(encoding="utf-8")
    jobs_text = text.split("\njobs:", 1)[1]
    assert "${{ inputs.prerelease" not in jobs_text, "o valor entra nos scripts só por variável de ambiente"
    step = next(s for s in wf["jobs"]["publicar"]["steps"] if s.get("name", "").startswith("Publicar"))
    run = step["run"]
    pre_branch, stable_branch = run.split("else", 1)
    assert '"$PRERELEASE" = "true"' in pre_branch
    assert "--prerelease --latest=false" in pre_branch and "--latest\n" not in pre_branch
    assert "--prerelease=false --latest" in stable_branch
    assert "--latest" not in text.split("Publicar (cria a tag")[0], "só o passo de publicar decide o Latest"


def test_dev_builds_never_publish_releases():
    wf = _wf("build-windows.yml")
    assert wf["permissions"] == {"contents": "read"}
    assert "inputs" not in (wf["on"].get("workflow_dispatch") or {})
    text = (WF / "build-windows.yml").read_text(encoding="utf-8")
    for forbidden in ("action-gh-release", "gh release", "[release]", "prerelease", "contents: write"):
        assert forbidden not in text, forbidden
    assert set(wf["on"]["push"]["branches"]) >= {"claude/**", "feature/**"}
    assert "/DAppVersion=" in text


def test_tests_workflow_is_reusable_and_runs_every_test_dir():
    wf = _wf("tests.yml")
    assert "workflow_call" in wf["on"] and wf["permissions"] == {"contents": "read"}
    text = (WF / "tests.yml").read_text(encoding="utf-8")
    run = next(s["run"] for s in wf["jobs"]["tests"]["steps"] if "pytest" in s.get("run", ""))
    assert run.strip().endswith(" tests"), "o CI roda a pasta tests/ inteira (sem lista fixa de arquivos)"
    assert '-m "not slow and not models and not gpu"' in run
    assert "pip check" in text and "packaging/ci/requirements-tests.txt" in text


def test_ci_requirements_are_pinned():
    for name in ("requirements-tests.txt", "requirements-build.txt"):
        lines = [ln.strip() for ln in (ROOT / "packaging" / "ci" / name).read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.startswith("#")]
        assert lines and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[\w.]+", ln) for ln in lines), name


# ------------------------------------------------------------------------------------- conteúdo do .exe

def _check_bundle():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_bundle", ROOT / "scripts" / "check_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bundle_check_requires_every_module_file_and_refuses_engine_libs():
    cb = _check_bundle()
    expected = cb.expected_modules()
    for name in ("mobilia.modules.lead_finder", "mobilia.modules.lead_finder.ui.page",
                 "mobilia.modules.lead_finder.storage", "mobilia.modules.lead_finder.sources.osm"):
        assert name in expected, name
    good = "[" + ",\n".join(f"('{m}', '/x/{m}.py', 'PYMODULE')" for m in (*cb.REQUIRED_CORE, *expected)) + "]"
    assert cb.check(good) == ([], [], len(expected))
    missing_page = good.replace("('mobilia.modules.lead_finder.ui.page',", "('outro',")
    assert cb.check(missing_page)[0] == ["mobilia.modules.lead_finder.ui.page"]
    heavy = good[:-1] + ",\n('torch.nn', '/x.py', 'PYMODULE'), ('diffusers', '/y.py', 'PYMODULE')]"
    assert cb.check(heavy)[1] == ["diffusers", "torch.nn"]


def test_spec_excludes_engine_libraries_and_workflows_check_the_bundle():
    spec = (ROOT / "packaging" / "7xai.spec").read_text(encoding="utf-8")
    excludes = spec.split("excludes = [", 1)[1].split("]", 1)[0]
    for lib in _check_bundle().FORBIDDEN_TOP:
        assert f'"{lib}"' in excludes, lib
    for wf in ("build-windows.yml", "release.yml"):
        text = (WF / wf).read_text(encoding="utf-8")
        assert "scripts/check_bundle.py build/7xai/PYZ-00.toc build-portable/7xai/PYZ-00.toc" in text, wf
    bw = (WF / "build-windows.yml").read_text(encoding="utf-8")
    assert 'SEVENXAI_ONEFILE = "1"' in bw and "dist-portable" in bw     # o portátil também é gerado e testado
