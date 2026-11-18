"""Atualização automática (``mobilia/ui/updater.py`` e ``updater_dialog.py``), sem internet.

A rede é um dublê (``FakeOpener``) com as respostas da API do GitHub em ``tests/fixtures/updater``; a instalação, a
saída do app e o download na interface também são dublês. O que só dá para conferir no Windows (o instalador
rodando em silêncio e o 7xAi reabrindo) está no plano de teste manual de ``docs/RELEASES.md``.
"""
from __future__ import annotations

import email.message
import hashlib
import importlib.util
import io
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from mobilia.ui import updater as up

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "updater"
ISS = ROOT / "packaging" / "installer.iss"
BASE = "https://github.com/tuzin7x/softwareEditor/releases/download/v9.9.1/"


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("MOBILIA_HOME", str(tmp_path / "home"))     # nada vai para a pasta de dados de verdade


def latest_json() -> dict:
    return json.loads((FIX / "release_latest.json").read_text(encoding="utf-8"))


def release(**kw) -> up.Release:
    return up.parse_release({**latest_json(), **kw})


# ------------------------------------------------------------------------------------------- dublê da rede

class FakeResp:
    def __init__(self, body: bytes, url: str, headers: dict | None = None):
        self._buf, self.url = io.BytesIO(body), url
        self.headers = {"Content-Length": str(len(body)), **(headers or {})}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def geturl(self) -> str:
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    """``routes``: url -> bytes | Exception | (bytes, url_final)."""

    def __init__(self, routes: dict):
        self.routes, self.requests = routes, []

    def open(self, req, timeout=None):
        self.requests.append(req)
        r = self.routes[req.full_url]
        if isinstance(r, BaseException):
            raise r
        body, final = r if isinstance(r, tuple) else (r, req.full_url)
        return FakeResp(body, final)


def http_error(code: int, **headers) -> urllib.error.HTTPError:
    msg = email.message.Message()
    for k, v in headers.items():
        msg[k.replace("_", "-")] = v
    return urllib.error.HTTPError(up.API_URL, code, "erro", msg, None)


def files_for(content: bytes, name="7xAI-Setup-v9.9.1.exe", digest: str | None = None) -> dict:
    digest = digest or hashlib.sha256(content).hexdigest()
    sums = f"{digest}  {name}\n{'0' * 64}  outro.exe\n".encode()
    return {BASE + "SHA256SUMS.txt": sums, BASE + name: content}


def rel_sized(content: bytes, name="7xAI-Setup-v9.9.1.exe") -> up.Release:
    r = release()
    assets = tuple(up.Asset(a.name, a.url, len(content) if a.name == name else 0) for a in r.assets)
    return up.Release(r.version, r.tag, r.notes, r.html_url, r.published_at, assets)


# ------------------------------------------------------------------------------------------------ versões

@pytest.mark.parametrize("remote,local,newer", [
    ("0.9.1", "0.9.0", True), ("v0.10.0", "0.9.9", True), ("1.0.0", "0.9.0", True), ("0.9.0", "0.9.0", False),
    ("0.8.9", "0.9.0", False), ("1.0.0-beta", "0.9.0", False), ("1.0", "0.9.0", False), ("", "0.9.0", False),
    ("v2.0.0", "garbage", False), ("01.0.0", "0.9.0", False)])
def test_version_compare(remote, local, newer):
    assert up.is_newer(remote, local) is newer


def test_parse_version_strict():
    assert up.parse_version("v1.2.3") == (1, 2, 3) and up.parse_version(" 10.0.1 ") == (10, 0, 1)
    for bad in ("1.2.3.4", "1.2.3+build", "x1.2.3", None, "v"):
        assert up.parse_version(bad) is None


def test_asset_names_match_release_script():
    spec = importlib.util.spec_from_file_location("release_notes", ROOT / "scripts" / "release_notes.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["release_notes"] = mod; spec.loader.exec_module(mod)
    assert up.asset_names("1.2.3") == mod.asset_names("1.2.3")
    wf = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "7xAI-Setup-v$env:VERSAO.exe" in wf and "SHA256SUMS.txt" in wf
    assert up.REPO in (ROOT / "docs" / "RELEASES.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------- Release JSON

def test_parse_release_fixture():
    r = up.parse_release(latest_json())
    assert r.version == "9.9.1" and r.tag == "v9.9.1" and r.published_at == "2026-12-01"
    assert [a.name for a in r.assets] == ["7xAI-Setup-v9.9.1.exe", "7xAI-Portable-v9.9.1.exe", "SHA256SUMS.txt"]
    assert r.asset("evil.exe") is None                      # host fora da lista: descartado
    assert r.html_url.startswith("https://github.com/")


def test_parse_release_rejects_prerelease_draft_and_garbage():
    assert up.parse_release(json.loads((FIX / "release_prerelease.json").read_text(encoding="utf-8"))) is None
    assert up.parse_release({**latest_json(), "draft": True}) is None
    assert up.parse_release({**latest_json(), "tag_name": "v9.9.1-rc1"}) is None
    assert up.parse_release({**latest_json(), "tag_name": 991}) is None
    assert up.parse_release([]) is None and up.parse_release(None) is None
    r = up.parse_release({**latest_json(), "assets": "x", "body": None, "html_url": "http://evil.example/"})
    assert r.assets == () and r.notes == "" and r.html_url == up.RELEASES_URL


def test_notes_for_display_hides_downloads_and_hashes():
    text = up.notes_for_display(latest_json()["body"])
    assert "Atualização automática dentro do app" in text and "## Correções" in text
    assert "Downloads" not in text and "SHA-256" not in text and "# 7xAi" not in text
    assert "example.com" not in text                         # imagens viram só o texto alternativo
    assert up.notes_for_display("") == "Sem notas para esta versão."


def test_select_asset():
    r = release()
    a, s = up.select_asset(r, "setup")
    assert a.name == "7xAI-Setup-v9.9.1.exe" and s.name == "SHA256SUMS.txt"
    assert up.select_asset(r, "portable")[0].name == "7xAI-Portable-v9.9.1.exe"
    no_sums = up.Release(r.version, r.tag, assets=tuple(x for x in r.assets if x.name != "SHA256SUMS.txt"))
    with pytest.raises(up.UpdateError, match="SHA256SUMS"):
        up.select_asset(no_sums, "setup")
    no_setup = up.Release(r.version, r.tag, assets=tuple(x for x in r.assets if "Setup" not in x.name))
    with pytest.raises(up.UpdateError, match="7xAI-Setup-v9.9.1.exe"):
        up.select_asset(no_setup, "setup")
    with pytest.raises(ValueError):
        up.select_asset(r, "zip")


# ------------------------------------------------------------------------------------------------ segurança

@pytest.mark.parametrize("url,ok", [
    ("https://github.com/tuzin7x/softwareEditor/releases/download/v1.0.0/x.exe", True),
    ("https://objects.githubusercontent.com/github-production-release-asset/1", True),
    ("https://release-assets.githubusercontent.com/github-production-release-asset/1", True),
    ("https://api.github.com/repos/x/y/releases/latest", True),
    ("http://github.com/tuzin7x/softwareEditor/releases/download/v1.0.0/x.exe", False),
    ("https://github.com.evil.example/x.exe", False), ("https://evil.example/github.com", False),
    ("https://user:pw@github.com/x", False), ("https://github.com:8443/x", False), ("file:///C:/x.exe", False),
    ("ftp://github.com/x", False), ("https://[::1/x", False)])
def test_host_allow_list(url, ok):
    assert up.url_allowed(url) is ok


def test_asset_url_must_be_this_repo():
    assert up.asset_url_allowed(BASE + "x.exe")
    assert not up.asset_url_allowed("https://github.com/outro/repo/releases/download/v1.0.0/x.exe")
    assert not up.asset_url_allowed("https://objects.githubusercontent.com/x.exe")
    assert not up.asset_url_allowed(BASE + "../../../x.exe")


def test_redirect_only_to_allowed_hosts():
    h = up._SafeRedirect()
    req = urllib.request.Request(BASE + "x.exe")
    ok = h.redirect_request(req, None, 302, "Found", {}, "https://objects.githubusercontent.com/a?sig=1")
    assert ok.full_url.startswith("https://objects.githubusercontent.com/")
    for bad in ("https://evil.example/x.exe", "http://objects.githubusercontent.com/a"):
        with pytest.raises(urllib.error.HTTPError):
            h.redirect_request(req, None, 302, "Found", {}, bad)


def test_checksum_parse_and_verify(tmp_path):
    f = tmp_path / "a.exe"; f.write_bytes(b"conteudo")
    good = hashlib.sha256(b"conteudo").hexdigest()
    sums = up.parse_sums(f"\ufeff{good.upper()}  a.exe\n{'1' * 64} *b.exe\nlixo\n")
    assert sums == {"a.exe": good, "b.exe": "1" * 64}
    up.verify_file(f, sums["a.exe"])                             # bom: não levanta
    with pytest.raises(up.ChecksumError):
        up.verify_file(f, "0" * 64)
    assert not f.exists()                                        # ruim: apagado


# --------------------------------------------------------------------------------------------------- download

def test_download_verified_good(tmp_path):
    content = b"MZ instalador de teste" * 1000
    op = FakeOpener(files_for(content))
    seen = []
    p = up.download_verified(rel_sized(content), "setup", tmp_path, opener=op,
                             progress=lambda d, t: seen.append((d, t)))
    assert p == tmp_path / "7xAI-Setup-v9.9.1.exe" and p.read_bytes() == content
    assert seen[-1] == (len(content), len(content))
    assert not (tmp_path / "SHA256SUMS.txt").exists() and not list(tmp_path.glob("*.part"))
    assert all(r.get_header("User-agent") == "7xAi-updater" for r in op.requests)
    assert all(not r.has_header("Authorization") for r in op.requests)
    # de novo: arquivo já baixado e válido é reaproveitado (só o SHA256SUMS é consultado)
    op2 = FakeOpener(files_for(content))
    assert up.download_verified(rel_sized(content), "setup", tmp_path, opener=op2) == p
    assert [r.full_url for r in op2.requests] == [BASE + "SHA256SUMS.txt"]


def test_download_verified_bad_checksum(tmp_path):
    content = b"MZ alterado"
    op = FakeOpener(files_for(content, digest="f" * 64))
    with pytest.raises(up.ChecksumError, match="não confere"):
        up.download_verified(rel_sized(content), "setup", tmp_path, opener=op)
    assert not list(tmp_path.iterdir())


def test_download_verified_sums_without_entry(tmp_path):
    routes = files_for(b"x"); routes[BASE + "SHA256SUMS.txt"] = b"0" * 64 + b"  outro.exe\n"
    with pytest.raises(up.ChecksumError, match="não lista"):
        up.download_verified(release(), "setup", tmp_path, opener=FakeOpener(routes))
    assert not list(tmp_path.iterdir())


def test_download_rejects_bad_final_host_size_and_cancel(tmp_path):
    url = BASE + "x.exe"
    with pytest.raises(up.UpdateError, match="não permitido"):
        up.download(url, tmp_path / "x.exe", FakeOpener({url: (b"x", "https://evil.example/x.exe")}))
    with pytest.raises(up.UpdateError, match="não permitido"):
        up.download("http://github.com/x", tmp_path / "x.exe", FakeOpener({}))
    with pytest.raises(up.UpdateError, match="tamanho"):
        up.download(url, tmp_path / "x.exe", FakeOpener({url: b"abc"}), expected_size=10)
    with pytest.raises(up.UpdateError):
        up.download(url, tmp_path / "x.exe", FakeOpener({url: b"a" * 100}), max_bytes=10)
    with pytest.raises(up.DownloadCancelled):
        up.download(url, tmp_path / "x.exe", FakeOpener({url: b"abc"}), cancelled=lambda: True)
    with pytest.raises(up.UpdateError, match="sem conexão"):
        up.download(url, tmp_path / "x.exe", FakeOpener({url: urllib.error.URLError("offline")}))
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------------------------- consulta

def test_fetch_latest_update_and_current():
    body = json.dumps(latest_json()).encode()
    op = FakeOpener({up.API_URL: body})
    res = up.fetch_latest(op, current="0.9.0")
    assert res.status == "update" and res.release.version == "9.9.1"
    req = op.requests[0]
    assert req.get_header("User-agent") == "7xAi-updater" and "github" in req.get_header("Accept")
    assert up.fetch_latest(FakeOpener({up.API_URL: body}), current="9.9.1").status == "current"
    assert up.fetch_latest(FakeOpener({up.API_URL: body}), current="10.0.0").status == "current"
    pre = (FIX / "release_prerelease.json").read_bytes()
    assert up.fetch_latest(FakeOpener({up.API_URL: pre}), current="0.9.0").status == "current"


@pytest.mark.parametrize("err,status", [
    (http_error(403, X_RateLimit_Remaining="0"), "rate_limited"), (http_error(429), "rate_limited"),
    (http_error(403), "error"), (http_error(500), "error"), (http_error(404), "current"),
    (urllib.error.URLError("sem rede"), "offline"), (socket.timeout("tempo"), "offline"),
    (ConnectionResetError(), "offline"), (OSError("rede"), "offline")])
def test_fetch_latest_offline_and_rate_limit_are_silent(err, status):
    assert up.fetch_latest(FakeOpener({up.API_URL: err}), current="0.9.0").status == status


def test_fetch_latest_bad_json():
    assert up.fetch_latest(FakeOpener({up.API_URL: b"<html>"}), current="0.9.0").status == "error"
    big = b"[" + b" " * (up.MAX_API_BYTES + 10) + b"]"
    assert up.fetch_latest(FakeOpener({up.API_URL: big}), current="0.9.0").status == "error"


# ------------------------------------------------------------------------------------- preferências / modo

def test_settings_persist_skip_and_auto(tmp_path):
    p = tmp_path / "s" / "updater.json"
    s = up.UpdaterSettings.load(p)
    assert s.auto_check and s.skipped_version == "" and s.last_check == 0
    s.skipped_version, s.auto_check, s.last_check = "9.9.1", False, 123.0
    s.save(p)
    s2 = up.UpdaterSettings.load(p)
    assert (s2.skipped_version, s2.auto_check, s2.last_check) == ("9.9.1", False, 123.0)
    p.write_text("{lixo", encoding="utf-8")
    assert up.UpdaterSettings.load(p).auto_check
    p.write_text(json.dumps({"skipped_version": "../x", "last_check": "x"}), encoding="utf-8")
    s3 = up.UpdaterSettings.load(p)
    assert s3.skipped_version == "" and s3.last_check == 0
    assert up.UpdaterSettings.path() == Path(os.environ["MOBILIA_HOME"]) / "settings" / "updater.json"


def test_throttling_once_per_day():
    s = up.UpdaterSettings(last_check=1_000_000.0)
    assert not s.due(1_000_000.0 + 3600)
    assert s.due(1_000_000.0 + 24 * 3600)
    assert s.due(1_000_000.0 - 3 * 24 * 3600)                     # relógio voltou: consulta
    s.auto_check = False
    assert not s.due(1_000_000.0 + 10 * 24 * 3600)


def test_record_check_and_should_offer():
    s = up.UpdaterSettings()
    s.record_check(up.CheckResult("offline"), now=50.0)
    assert s.last_check == 0                                      # sem internet: tenta na próxima abertura
    for st in ("current", "rate_limited", "error"):
        s.last_check = 0; s.record_check(up.CheckResult(st), now=50.0); assert s.last_check == 50.0
    res = up.CheckResult("update", release())
    assert s.should_offer(res)
    s.skipped_version = "9.9.1"
    assert not s.should_offer(res) and s.should_offer(res, manual=True)
    assert not s.should_offer(up.CheckResult("current"), manual=True)


def test_updater_enabled_only_frozen_or_test_env():
    assert not up.updater_enabled({}, frozen=False)
    assert up.updater_enabled({"MOBILIA_UPDATER_TEST": "1"}, frozen=False)
    assert not up.updater_enabled({"MOBILIA_UPDATER_TEST": "0"}, frozen=False)
    assert up.updater_enabled({}, frozen=True)


def test_install_kind(tmp_path):
    exe = tmp_path / "Programs" / "7xAi" / "7xAi.exe"
    assert up.install_kind(frozen=False) == "dev"
    assert up.install_kind(True, str(exe), str(exe.parent / "_internal")) == "installed"
    assert up.install_kind(True, str(exe), str(tmp_path / "Temp" / "_MEI12345")) == "portable"
    assert up.asset_kind("portable") == "portable" and up.asset_kind("installed") == "setup"


def test_installer_command_silent_update():
    cmd = up.installer_command(Path("C:/x/7xAI-Setup-v9.9.1.exe"), Path("C:/Programs/7xAi"), Path("C:/x/s.log"))
    assert cmd[0].endswith("7xAI-Setup-v9.9.1.exe")
    for flag in ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/NORESTARTAPPLICATIONS",
                 "/AUTOUPDATE"):
        assert flag in cmd
    assert any(c.startswith("/DIR=") and c.endswith("7xAi") for c in cmd)
    assert any(c.startswith("/LOG=") for c in cmd)


def test_cleanup_updates(tmp_path):
    for n in ("7xAI-Setup-v0.8.0.exe", "7xAI-Setup-v0.8.0.log", "7xAI-Setup-v0.9.0.exe", "7xAI-Setup-v0.9.0.log",
              "7xAI-Setup-v1.0.0.exe", "7xAI-Setup-v1.0.0.exe.part", "outro.txt"):
        (tmp_path / n).write_bytes(b"x")
    up.cleanup_updates(tmp_path, "0.9.0")
    # o instalador já usado sai; o registro da instalação da versão atual fica (diagnóstico)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["7xAI-Setup-v0.9.0.log", "7xAI-Setup-v1.0.0.exe",
                                                          "outro.txt"]
    up.cleanup_updates(tmp_path / "nao-existe")


# ------------------------------------------------------------------------------------------- installer.iss

def _iss_sections() -> dict[str, list[str]]:
    raw = ISS.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    secs: dict[str, list[str]] = {}
    cur = ""
    for line in raw[3:].decode("utf-8").splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            cur = s[1:-1]; secs.setdefault(cur, [])
        elif s and not s.startswith(";"):
            secs.setdefault(cur, []).append(s)
    return secs


def test_installer_relaunches_only_on_autoupdate():
    secs = _iss_sections()
    runs = secs["Run"]
    normal = [r for r in runs if "postinstall" in r]
    auto = [r for r in runs if "IsAutoUpdate" in r]
    assert len(normal) == 1 and "skipifsilent" in normal[0]       # instalação normal: igual a antes
    assert len(auto) == 1
    a = auto[0]
    assert 'Filename: "{app}\\{#ExeName}"' in a and "Check: IsAutoUpdate" in a
    flags = a.split("Flags:", 1)[1].split(";", 1)[0].split()
    assert "nowait" in flags and "skipifsilent" not in flags and "postinstall" not in flags
    code = "\n".join(secs["Code"])
    assert "function IsAutoUpdate: Boolean;" in code
    assert f"'{up.AUTOUPDATE_FLAG}'" in code and "ParamStr(I)" in code
    assert "function InitializeSetup: Boolean;" in code and "Result := True;" in code
    # o instalador só escreve na pasta do programa: os dados do usuário não são apagados numa atualização
    assert not any("MobiliaStudio" in ln for ln in secs["InstallDelete"])
    assert all("{app}" in ln for ln in secs["Files"])


# ------------------------------------------------------------------------------------- interface (offscreen)

qt = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    from mobilia.ui import theme
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(theme.QSS)
    yield a


def wait_until(cond, timeout_s: float = 5.0) -> bool:
    """Processa eventos até ``cond()`` (os timers do Qt atrasam quando a máquina está ocupada)."""
    import time

    from PySide6.QtTest import QTest
    end = time.monotonic() + timeout_s
    while not cond() and time.monotonic() < end:
        QTest.qWait(20)
    return bool(cond())


class Harness:
    """Controller com tudo dublado; ``spawn`` síncrono (a thread vira chamada direta)."""

    def __init__(self, tmp_path, kind="installed", fetch_result=None, download=None, busy=False, confirm=False):
        from mobilia.ui.updater_dialog import UpdateController
        self.fetches, self.downloads, self.launched, self.opened, self.quits, self.confirms = 0, [], [], [], 0, []
        self.result = fetch_result or up.CheckResult("update", release())
        self.download_fn = download or self._ok_download
        self.tmp = tmp_path
        self.ctl = UpdateController(
            None, settings_path=tmp_path / "settings.json", enabled=True, kind=kind, fetch=self._fetch,
            downloader=self._download, launcher=self.launched.append, open_file=self.opened.append,
            quit_app=self._quit, busy=lambda: busy, confirm=self._confirm, spawn=lambda fn: fn(),
            download_dir=tmp_path / "dl", current="0.9.0")

    def _fetch(self):
        self.fetches += 1
        return self.result

    def _ok_download(self, rel, kind, folder, progress=None, cancelled=None):
        progress(50, 100); progress(100, 100)
        p = Path(folder) / up.asset_names(rel.version)[kind]
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"ok")
        return p

    def _download(self, *a, **kw):
        self.downloads.append(a[1])
        return self.download_fn(*a, **kw)

    def _quit(self):
        self.quits += 1

    def _confirm(self, title, text):
        self.confirms.append(text)
        return self.confirm_answer

    confirm_answer = False


def test_dialog_renders_notes_without_html(app):
    from mobilia.ui.updater_dialog import UpdateDialog
    d = UpdateDialog(release(), "installed", "0.9.0")
    assert d.title.text() == "Nova versão 9.9.1 disponível" and "0.9.0" in d.subtitle.text()
    plain = d.notes.toPlainText()
    assert "Atualização automática dentro do app" in plain and "SHA-256" not in plain
    assert "<script" not in d.notes.toHtml()                     # HTML das notas vira texto, nunca código
    assert d.btn_update.text() == "Atualizar agora" and d.btn_later.text() == "Depois"
    assert d.btn_skip.text() == "Pular esta versão" and not d.isModal()
    assert UpdateDialog(release(), "portable").btn_update.text() == "Baixar nova versão"
    d.close()


def test_flow_installed_update_now(app, tmp_path):
    h = Harness(tmp_path)
    assert h.ctl.start_auto_check(now=10 ** 9)
    d = h.ctl.dialog
    assert d is not None and d.isVisible() and h.fetches == 1
    assert up.UpdaterSettings.load(tmp_path / "settings.json").last_check > 0
    d.btn_update.click()
    assert h.downloads == ["setup"]
    assert d.state == "installing" and "fechar e abrir de novo" in d.status.text()
    assert h.launched == [tmp_path / "dl" / "7xAI-Setup-v9.9.1.exe"]
    assert wait_until(lambda: h.quits == 1)
    d.close()


def test_flow_throttle_and_disabled(app, tmp_path):
    h = Harness(tmp_path, fetch_result=up.CheckResult("current"))
    (tmp_path / "dl").mkdir()
    (tmp_path / "dl" / "7xAI-Setup-v0.9.0.exe").write_bytes(b"usado")
    assert h.ctl.start_auto_check()
    assert not (tmp_path / "dl" / "7xAI-Setup-v0.9.0.exe").exists()   # instalador já usado: apagado
    assert not h.ctl.start_auto_check()                          # já consultou hoje
    assert h.fetches == 1 and h.ctl.dialog is None and h.ctl.message is None   # automática: silenciosa
    h.ctl.enabled = False
    h.ctl.check_now()
    assert h.fetches == 1


def test_flow_skip_version_persists(app, tmp_path):
    h = Harness(tmp_path)
    h.ctl.start_auto_check(now=10 ** 9)
    h.ctl.dialog.btn_skip.click()
    assert up.UpdaterSettings.load(tmp_path / "settings.json").skipped_version == "9.9.1"
    assert not h.ctl.dialog.isVisible()
    h2 = Harness(tmp_path)                                       # próxima abertura, dias depois
    assert h2.ctl.start_auto_check(now=10 ** 10)
    assert h2.ctl.dialog is None
    h2.ctl.check_now()                                           # consulta manual mostra mesmo pulada
    assert h2.ctl.dialog is not None and h2.ctl.dialog.isVisible()
    h2.ctl.dialog.btn_later.click()
    assert not h2.ctl.dialog.isVisible() and h2.downloads == []


def test_flow_manual_check_messages(app, tmp_path):
    h = Harness(tmp_path, fetch_result=up.CheckResult("offline"))
    h.ctl.check_now()
    assert h.ctl.dialog is None and "sem internet" in h.ctl.message.text()
    assert up.UpdaterSettings.load(tmp_path / "settings.json").last_check == 0
    h.result = up.CheckResult("rate_limited"); h.ctl.check_now()
    assert "limitou" in h.ctl.message.text()
    h.result = up.CheckResult("current"); h.ctl.check_now()
    assert "mais recente" in h.ctl.message.text() and "0.9.0" in h.ctl.message.text()
    h.ctl.message.close()


def test_flow_fetch_exception_never_crashes(app, tmp_path):
    h = Harness(tmp_path)
    h._fetch = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    h.ctl.fetch = h._fetch
    h.ctl.check_now()
    assert h.ctl.dialog is None and "boom" in h.ctl.message.text() and not h.ctl.checking
    h.ctl.message.close()


def test_flow_checksum_error_and_cancel(app, tmp_path):
    def bad(*a, **kw):
        raise up.ChecksumError("O arquivo baixado não confere com o SHA-256 publicado.")
    h = Harness(tmp_path, download=bad)
    h.ctl.check_now()
    d = h.ctl.dialog
    d.btn_update.click()
    assert h.launched == [] and h.quits == 0
    assert d.state == "offer" and "SHA-256" in d.status.text() and d.btn_update.text() == "Tentar de novo"

    def cancelled(*a, **kw):
        raise up.DownloadCancelled("cancelado")
    h.download_fn = cancelled
    d.btn_update.click()
    assert d.state == "offer" and "cancelado" in d.status.text() and h.launched == []
    d.close()


def test_flow_cancel_button_sets_flag(app, tmp_path):
    seen = {}

    def slow(rel, kind, folder, progress=None, cancelled=None):
        seen["before"] = cancelled()
        h.ctl.dialog.btn_cancel.click()                         # o usuário clica em Cancelar durante o download
        seen["after"] = cancelled()
        raise up.DownloadCancelled("x")
    h = Harness(tmp_path, download=slow)
    h.ctl.check_now()
    h.ctl.dialog.btn_update.click()
    assert seen == {"before": False, "after": True} and h.ctl.dialog.state == "offer"
    h.ctl.dialog.close()


def test_flow_busy_job_asks_first(app, tmp_path):
    h = Harness(tmp_path, busy=True)
    h.ctl.check_now()
    h.ctl.dialog.btn_update.click()
    assert h.confirms and h.downloads == []                     # respondeu "não": nada baixado
    h.confirm_answer = True
    h.ctl.dialog.btn_update.click()
    assert h.downloads == ["setup"] and h.launched
    h.ctl.dialog.close()


def test_flow_portable_download_then_open(app, tmp_path):
    h = Harness(tmp_path, kind="portable")
    h.ctl.check_now()
    d = h.ctl.dialog
    assert d.btn_update.text() == "Baixar nova versão"
    d.btn_update.click()
    assert h.downloads == ["portable"] and h.launched == []      # nunca roda o instalador no portátil
    assert d.state == "ready" and d.btn_update.text() == "Abrir a nova versão"
    assert "7xAI-Portable-v9.9.1.exe" in d.status.text() and d.btn_folder.isVisible()
    d.btn_update.click()
    assert h.opened == [tmp_path / "dl" / "7xAI-Portable-v9.9.1.exe"]
    assert wait_until(lambda: h.quits == 1)
    d.close()


def test_auto_check_toggle_persists(app, tmp_path):
    h = Harness(tmp_path)
    h.ctl.check_now()
    h.ctl.dialog.chk_auto.setChecked(False)
    assert up.UpdaterSettings.load(tmp_path / "settings.json").auto_check is False
    assert not h.ctl.start_auto_check(now=10 ** 11)
    h.ctl.dialog.close()


def test_attach_updater_status_bar(app, tmp_path):
    from PySide6.QtWidgets import QMainWindow, QPushButton, QStatusBar

    from mobilia.ui.updater_dialog import attach_updater
    win = QMainWindow(); sb = QStatusBar(); win.setStatusBar(sb)
    off = attach_updater(win, sb, enabled=False, settings_path=tmp_path / "s.json")
    assert not off.enabled and not sb.findChildren(QPushButton)  # pelo código-fonte: nada aparece
    on = attach_updater(win, sb, enabled=True, settings_path=tmp_path / "s.json",
                        fetch=lambda: up.CheckResult("current"), spawn=lambda fn: fn(), download_dir=tmp_path / "dl")
    assert on.button.text() == "Atualizações"
    assert on.auto_timer.isActive() and on.auto_timer.interval() == 5000     # consulta automática agendada
    on.auto_timer.stop()
    assert [a.text() for a in on.menu.actions()] == ["Procurar atualizações", "Procurar automaticamente ao abrir",
                                                     "Ver todas as versões no GitHub"]
    on.action_auto.setChecked(False)
    assert up.UpdaterSettings.load(tmp_path / "s.json").auto_check is False
    on.action_check.trigger()
    assert on.message is not None and "mais recente" in on.message.text()
    on.message.close(); win.close()


def test_main_window_updater_off_from_source(app, monkeypatch):
    monkeypatch.delenv("MOBILIA_UPDATER_TEST", raising=False)
    from mobilia.ui.app import MainWindow
    w = MainWindow()
    assert w.updater.enabled is False and w.updater.kind == "dev"
    w.close()
