"""Empacotamento Windows do 7xAi, conferido de forma estática (o PowerShell não roda aqui).

Garante o que o dono pediu: o programa se chama 7xAi (dist\\7xAi\\7xAi.exe), é instalado para o
usuário e ganha o ícone "7xAi" na Área de Trabalho, com ou sem o Inno Setup.
"""
from __future__ import annotations

import os
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PS1 = ROOT / "packaging" / "build_windows.ps1"
BAT = ROOT / "GERAR_EXE.bat"
ISS = ROOT / "packaging" / "installer.iss"
SPEC = ROOT / "packaging" / "7xai.spec"
ICO = ROOT / "packaging" / "assets" / "icon.ico"
APP_ID = "{{6E1E7C1A-6A0C-4E58-9B7E-4D5F2C0B9A11}"


def _code_lines(text: str) -> list[str]:
    """Linhas de código do .ps1 sem comentários e sem o conteúdo das strings."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = re.sub(r'"(?:`.|[^"`])*"', '""', s)      # "..." (escape do PowerShell: crase)
        s = re.sub(r"'[^']*'", "''", s)
        s = s.split(" #", 1)[0]
        out.append(s)
    return out


# ------------------------------------------------------------------------------------- build_windows.ps1

def test_ps1_encoding_bom_and_crlf():
    raw = PS1.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "o Windows PowerShell 5.1 precisa do BOM para ler os acentos"
    raw.decode("utf-8")
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n"), "todas as linhas devem terminar em CRLF"


def test_ps1_has_no_powershell7_only_syntax():
    code = _code_lines(PS1.read_bytes()[3:].decode("utf-8"))
    for line in code:
        assert "??" not in line, f"operador ?? (PowerShell 7): {line}"
        assert "?." not in line and "?[" not in line, f"operador ?. (PowerShell 7): {line}"
        assert "&&" not in line and "||" not in line, f"cadeia && / || (PowerShell 7): {line}"
        assert not re.search(r"\s\?\s.+\s:\s", line), f"operador ternário (PowerShell 7): {line}"
        assert "-Parallel" not in line and "$PSStyle" not in line, line
        assert "-AsHashtable" not in line, line


def test_ps1_builds_tests_installs_and_creates_desktop_icon_in_order():
    text = PS1.read_bytes()[3:].decode("utf-8")
    steps = [
        "packaging\\7xai.spec",                                  # PyInstaller -> dist\7xAi\7xAi.exe
        '-ArgumentList "--selftest"',                            # autoteste
        'Join-Path $env:LOCALAPPDATA "Programs\\7xAi"',          # instala para o usuário
        "& robocopy $distDir $installDir /MIR",
        "if ($rc -ge 8)",                                        # robocopy < 8 = sucesso
        "New-Object -ComObject WScript.Shell",
        '[Environment]::GetFolderPath("Desktop")',               # funciona com OneDrive
        'Join-Path $desktop "7xAi.lnk"',
    ]
    pos = [text.find(s) for s in steps]
    assert all(p >= 0 for p in pos), [s for s, p in zip(steps, pos) if p < 0]
    assert pos == sorted(pos), "ordem: build, autoteste, cópia, atalho"
    for s in ('$ExeName = "7xAi.exe"', '[Environment]::GetFolderPath("Programs")', "$s.TargetPath = $installedExe",
              "$s.WorkingDirectory = $installDir", '$s.IconLocation = "$installedExe,0"',
              '"\\MobiliaStudio.exe"', "Feche o 7xAi", "packaging\\installer.iss"):
        assert s in text, s
    # nada de instalar se o autoteste falhar
    assert re.search(r'if \(\$p\.ExitCode -ne 0\) \{ throw', text)


def test_bat_success_message_and_crlf():
    raw = BAT.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "BOM quebraria o @echo off"
    assert raw.count(b"\n") == raw.count(b"\r\n")
    text = raw.decode("utf-8")
    assert "chcp 65001" in text and "packaging\\build_windows.ps1" in text
    assert "Pronto! O 7xAi foi instalado e o ícone está na sua Área de Trabalho." in text


# --------------------------------------------------------------------------------------- Inno Setup

def test_inno_setup_installer_names_and_desktop_shortcut():
    raw = ISS.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "Inno Setup só lê UTF-8 (acentos) com BOM"
    text = raw[3:].decode("utf-8")
    assert f"AppId={APP_ID}" in text                       # mesmo AppId: atualiza a instalação antiga
    assert '#define AppName "7xAi"' in text and "AppName={#AppName}" in text
    assert "DefaultDirName={localappdata}\\Programs\\7xAi" in text
    assert "OutputBaseFilename=7xAi-Setup-{#AppVersion}" in text
    assert 'Source: "..\\dist\\7xAi\\*"' in text and '#define ExeName "7xAi.exe"' in text
    task = next(line for line in text.splitlines() if line.startswith('Name: "desktopicon"'))
    assert "unchecked" not in task.lower(), "o ícone na Área de Trabalho deve vir marcado"
    desk = next(line for line in text.splitlines() if line.startswith('Name: "{userdesktop}'))
    assert "Tasks: desktopicon" in desk and "{app}\\{#ExeName}" in desk
    assert "{localappdata}\\MobiliaStudio" in text         # pasta de dados mantida


# ------------------------------------------------------------------------------- PyInstaller / versão

def _stub_pyinstaller_win32(monkeypatch):
    """versioninfo do PyInstaller importa pefile/win32api, que só existem no Windows."""
    pytest.importorskip("PyInstaller")
    if sys.platform != "win32":
        monkeypatch.setitem(sys.modules, "pefile", sys.modules.get("pefile") or types.ModuleType("pefile"))
        import PyInstaller.compat as compat
        if not hasattr(compat, "win32api"):
            monkeypatch.setattr(compat, "win32api", None, raising=False)


def test_spec_outputs_7xai_exe_with_version_resource(monkeypatch):
    spec = SPEC.read_text(encoding="utf-8")
    assert "name=EXE_NAME" in spec and "version=build_version_info(ROOT)" in spec
    assert not (ROOT / "packaging" / "mobilia_studio.spec").exists()
    _stub_pyinstaller_win32(monkeypatch)
    monkeypatch.syspath_prepend(str(ROOT / "packaging"))
    import win_version

    from mobilia import __version__
    assert win_version.EXE_NAME == "7xAi" and win_version.app_version(ROOT) == __version__
    vi = win_version.build_version_info(ROOT)
    text = str(vi)
    for key in ("CompanyName", "FileDescription", "ProductName"):
        assert f"StringStruct('{key}', '7xAi')" in text, key
    assert f"StringStruct('ProductVersion', '{__version__}')" in text
    assert "StringStruct('OriginalFilename', '7xAi.exe')" in text
    assert len(vi.toRaw()) > 200


def test_workflow_uses_new_names():
    wf = (ROOT / ".github" / "workflows" / "build-windows.yml").read_text(encoding="utf-8")
    assert "packaging/7xai.spec" in wf and "dist\\7xAi\\7xAi.exe" in wf and "MobiliaStudio" not in wf


# ------------------------------------------------------------------------------------------------ ícone

def test_icon_has_all_sizes_from_vector():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QGuiApplication

    _app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import make_icon
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    sizes = make_icon.read_ico(ICO)
    assert [w for w, h, _ in sizes] == [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
    assert all(w == h and bits == 32 for w, h, bits in sizes)
    # os SVGs versionados batem com o gerador (texto determinístico)
    from mobilia.ui import brand
    assets = ROOT / "packaging" / "assets"
    assert (assets / "logo.svg").read_text(encoding="utf-8") == brand.icon_svg("L")
    assert (assets / "wordmark.svg").read_text(encoding="utf-8") == brand.wordmark_svg()
    assert "7xAi" in (assets / "logo.svg").read_text(encoding="utf-8")
