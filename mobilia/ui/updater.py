"""Atualização automática do 7xAi pelo GitHub Releases (gratuito, sem servidor próprio).

Fluxo (a interface fica em ``updater_dialog.py``; aqui só a lógica, sem Qt, testável):

1. ``fetch_latest`` consulta ``api.github.com/repos/<repo>/releases/latest`` (sem login, sem dados pessoais; o
   User-Agent é genérico). Essa rota já ignora rascunhos e pré-lançamentos; ``parse_release`` confere de novo e só
   aceita a tag ``vX.Y.Z`` de uma versão estável. Sem internet, tempo esgotado ou limite de consultas do GitHub
   (403/429): nada acontece (``CheckResult.status``).
2. ``is_newer`` compara com ``mobilia.__version__``. ``UpdaterSettings`` guarda (em ``<dados>/settings``) a versão
   pulada, a última consulta (no máximo uma consulta automática por dia) e se a consulta automática está ligada.
3. ``download_verified`` baixa o ``SHA256SUMS.txt`` e o arquivo da versão (instalador ou portátil), só por HTTPS e
   só de hosts do GitHub (``ALLOWED_HOSTS``, conferido também em cada redirecionamento), e confere o SHA-256.
   Sem soma ou com soma diferente, o arquivo é apagado e nada é executado.
4. Instalado (Inno Setup): ``installer_command`` roda o instalador em modo silencioso com ``/AUTOUPDATE`` (o
   ``installer.iss`` reabre o 7xAi no fim só com esse parâmetro) e o app fecha para os arquivos serem trocados.
   Portátil (arquivo único): o novo ``.exe`` é salvo ao lado do atual (um ``.exe`` aberto não pode ser trocado).

Os dados do usuário (motor em ``runtime``, modelos, trabalhos, Lead Finder) ficam em ``%LOCALAPPDATA%\\MobiliaStudio``
e o instalador só escreve na pasta do programa. Ao reabrir, ``bootstrap`` compara as impressões digitais e só refaz
as etapas do motor cujo conteúdo mudou (em geral, apenas a cópia do código, em segundos). Este módulo fica em
``mobilia/ui``, fora do motor: mudá-lo não força nem a cópia do motor.

Desligado ao rodar pelo código-fonte (``updater_enabled``), a não ser com ``MOBILIA_UPDATER_TEST=1``. Só biblioteca
padrão.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import __version__, runtime

REPO = "tuzin7x/softwareEditor"
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_URL = f"https://github.com/{REPO}/releases"
USER_AGENT = "7xAi-updater"
ALLOWED_HOSTS = frozenset({"api.github.com", "github.com", "objects.githubusercontent.com",
                           "release-assets.githubusercontent.com"})
# o primeiro endereço de um arquivo (``browser_download_url``) tem de ser deste repositório
DOWNLOAD_PREFIX = f"/{REPO}/releases/download/"
CHECK_INTERVAL_S = 24 * 3600
TIMEOUT_S = 15
MAX_API_BYTES = 2 << 20            # resposta JSON da API
MAX_SUMS_BYTES = 64 << 10
MAX_ASSET_BYTES = 1 << 30          # o instalador tem ~70 MB; 1 GB é um teto de segurança
MAX_NOTES_CHARS = 60_000
ENV_TEST = "MOBILIA_UPDATER_TEST"
SUMS_NAME = "SHA256SUMS.txt"
AUTOUPDATE_FLAG = "/AUTOUPDATE"
_SEMVER = re.compile(r"^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_SUM_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")


class UpdateError(Exception):
    """Falha que a interface mostra ao usuário (a mensagem já está em português)."""


class ChecksumError(UpdateError):
    pass


class DownloadCancelled(UpdateError):
    pass


# ------------------------------------------------------------------------------------------------- versões

def parse_version(text: str | None) -> tuple[int, int, int] | None:
    """``"v1.2.3"``/``"1.2.3"`` -> ``(1, 2, 3)``; qualquer outra coisa (``1.2``, ``1.2.3-beta``) -> None."""
    m = _SEMVER.match((text or "").strip())
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def is_newer(remote: str, local: str = __version__) -> bool:
    r, cur = parse_version(remote), parse_version(local)
    return bool(r and cur and r > cur)


def asset_names(version: str) -> dict[str, str]:
    """Nomes oficiais (iguais a ``scripts/release_notes.asset_names``)."""
    return {"setup": f"7xAI-Setup-v{version}.exe", "portable": f"7xAI-Portable-v{version}.exe", "sums": SUMS_NAME}


# --------------------------------------------------------------------------------------------- segurança

def url_allowed(url: str) -> bool:
    """Só HTTPS, porta padrão, sem usuário/senha e só nos hosts do GitHub (``ALLOWED_HOSTS``)."""
    try:
        u = urllib.parse.urlsplit(url)
        port = u.port
    except ValueError:
        return False
    return (u.scheme == "https" and (u.hostname or "").lower() in ALLOWED_HOSTS and port in (None, 443)
            and not u.username and not u.password)


def asset_url_allowed(url: str) -> bool:
    """Endereço inicial de um arquivo da Release: ``https://github.com/<repo>/releases/download/...``."""
    if not url_allowed(url):
        return False
    u = urllib.parse.urlsplit(url)
    return u.hostname == "github.com" and u.path.startswith(DOWNLOAD_PREFIX) and ".." not in u.path


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Segue redirecionamentos (o GitHub manda o download para outro host) só para hosts permitidos."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802 (API do urllib)
        newurl = urllib.parse.urljoin(req.full_url, newurl)
        if not url_allowed(newurl):
            raise urllib.error.HTTPError(newurl, code, "redirecionamento para host não permitido", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()      # confere o certificado e o nome do host
    return urllib.request.build_opener(_SafeRedirect(), urllib.request.HTTPSHandler(context=ctx))


def _request(url: str, accept: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})


# --------------------------------------------------------------------------------------------- Release

@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int = 0


@dataclass(frozen=True)
class Release:
    version: str
    tag: str
    notes: str = ""
    html_url: str = RELEASES_URL
    published_at: str = ""
    assets: tuple[Asset, ...] = ()

    def asset(self, name: str) -> Asset | None:
        return next((a for a in self.assets if a.name == name), None)


def parse_release(data: object) -> Release | None:
    """JSON de ``/releases/latest`` -> ``Release``; None para rascunho, pré-lançamento ou tag fora de ``vX.Y.Z``."""
    if not isinstance(data, dict) or data.get("draft") or data.get("prerelease"):
        return None
    tag = data.get("tag_name")
    ver = parse_version(tag if isinstance(tag, str) else None)
    if not ver:
        return None
    assets = []
    for a in data.get("assets") or []:
        if not isinstance(a, dict):
            continue
        name, url = a.get("name"), a.get("browser_download_url")
        if isinstance(name, str) and isinstance(url, str) and asset_url_allowed(url):
            size = a.get("size")
            assets.append(Asset(name, url, size if isinstance(size, int) and size >= 0 else 0))
    html = data.get("html_url")
    body = data.get("body")
    return Release(version="%d.%d.%d" % ver, tag=tag,
                   notes=(body if isinstance(body, str) else "")[:MAX_NOTES_CHARS],
                   html_url=html if isinstance(html, str) and url_allowed(html) else RELEASES_URL,
                   published_at=str(data.get("published_at") or "")[:10], assets=tuple(assets))


_HIDDEN_SECTIONS = ("downloads", "verificação de integridade")


def notes_for_display(markdown: str) -> str:
    """Notas da Release sem o título, a tabela de downloads e a seção de SHA-256 (o app cuida disso)."""
    out: list[str] = []
    skip = False
    for line in (markdown or "").replace("\r\n", "\n").split("\n"):
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            skip = line[3:].strip().casefold().startswith(_HIDDEN_SECTIONS)
        if not skip:
            out.append(line)
    text = "\n".join(out).strip()
    # imagens e HTML nunca entram (a interface também liga o modo Markdown sem HTML)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text or "Sem notas para esta versão."


def select_asset(release: Release, kind: str) -> tuple[Asset, Asset]:
    """(arquivo, SHA256SUMS.txt) da versão. ``kind``: ``"setup"`` (instalado) ou ``"portable"``."""
    names = asset_names(release.version)
    if kind not in ("setup", "portable"):
        raise ValueError(kind)
    asset, sums = release.asset(names[kind]), release.asset(names["sums"])
    if asset is None:
        raise UpdateError(f"A versão {release.version} não tem o arquivo {names[kind]}.")
    if sums is None:
        raise UpdateError(f"A versão {release.version} não tem o {SUMS_NAME}; por segurança, nada foi baixado.")
    return asset, sums


def parse_sums(text: str) -> dict[str, str]:
    """``SHA256SUMS.txt`` (formato do ``sha256sum``) -> {nome: sha256 em minúsculas}."""
    out = {}
    for line in text.splitlines():
        m = _SUM_LINE.match(line.strip().lstrip("﻿"))
        if m:
            out[m.group(2)] = m.group(1).lower()
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    """Levanta ``ChecksumError`` (e apaga o arquivo) quando o SHA-256 não confere."""
    got = sha256_file(path)
    if not expected or got != expected.lower():
        try:
            Path(path).unlink()
        except OSError:
            pass
        raise ChecksumError(f"O arquivo baixado ({Path(path).name}) não confere com o SHA-256 publicado. "
                            "Ele foi apagado e nada foi instalado. Tente de novo mais tarde.")


# ----------------------------------------------------------------------------------------- consulta

@dataclass(frozen=True)
class CheckResult:
    status: str                      # "update" | "current" | "offline" | "rate_limited" | "error"
    release: Release | None = None
    message: str = ""


Opener = Callable[..., object]       # ``opener.open(request, timeout=...)``


def _read_limited(resp, limit: int) -> bytes:
    data = resp.read(limit + 1)
    if len(data) > limit:
        raise UpdateError("Resposta grande demais.")
    return data


def fetch_latest(opener=None, current: str = __version__, timeout: float = TIMEOUT_S) -> CheckResult:
    """Consulta a última versão estável. Nunca levanta exceção: os problemas viram ``CheckResult.status``."""
    opener = opener or build_opener()
    try:
        with opener.open(_request(API_URL, "application/vnd.github+json"), timeout=timeout) as resp:
            data = json.loads(_read_limited(resp, MAX_API_BYTES).decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429 or (e.code == 403 and (e.headers or {}).get("X-RateLimit-Remaining") == "0"):
            return CheckResult("rate_limited", message="O GitHub limitou as consultas por agora. Tente mais tarde.")
        if e.code == 404:          # nenhuma versão estável publicada ainda
            return CheckResult("current")
        return CheckResult("error", message=f"O GitHub respondeu com o código {e.code}.")
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, ssl.SSLError, OSError):
        return CheckResult("offline", message="Sem conexão com o GitHub.")
    except (ValueError, UnicodeDecodeError, UpdateError):
        return CheckResult("error", message="Resposta inválida do GitHub.")
    rel = parse_release(data)
    if rel is None:
        return CheckResult("current")
    return CheckResult("update" if is_newer(rel.version, current) else "current", rel)


# --------------------------------------------------------------------------------------------- download

Progress = Callable[[int, int], None]


def download(url: str, dest: Path, opener=None, progress: Progress | None = None,
             cancelled: Callable[[], bool] | None = None, max_bytes: int = MAX_ASSET_BYTES,
             expected_size: int = 0, timeout: float = 60) -> Path:
    """Baixa ``url`` para ``dest`` (via ``dest.part``). Só hosts permitidos, inclusive depois dos redirecionamentos."""
    if not url_allowed(url):
        raise UpdateError("Endereço de download não permitido.")
    opener = opener or build_opener()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with opener.open(_request(url, "application/octet-stream"), timeout=timeout) as resp:
            final = resp.geturl() if hasattr(resp, "geturl") else url
            if not url_allowed(final or url):
                raise UpdateError("O download foi redirecionado para um endereço não permitido.")
            total = int(resp.headers.get("Content-Length") or 0) if getattr(resp, "headers", None) else 0
            total = total or expected_size
            if total > max_bytes or (expected_size and total and total != expected_size):
                raise UpdateError("O tamanho do arquivo não é o esperado.")
            done = 0
            with open(part, "wb") as f:
                while True:
                    if cancelled and cancelled():
                        raise DownloadCancelled("Download cancelado.")
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    done += len(chunk)
                    if done > max_bytes:
                        raise UpdateError("O arquivo é maior que o esperado.")
                    f.write(chunk)
                    if progress:
                        progress(done, total)
        if expected_size and done != expected_size:
            raise UpdateError("O download veio incompleto. Tente de novo.")
        os.replace(part, dest)
        return dest
    except UpdateError:
        _unlink(part)
        raise
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, ssl.SSLError, OSError) as e:
        _unlink(part)
        raise UpdateError(f"Não foi possível baixar a atualização (sem conexão?). Detalhe: {e}") from e


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


def download_verified(release: Release, kind: str, folder: Path, opener=None, progress: Progress | None = None,
                      cancelled: Callable[[], bool] | None = None) -> Path:
    """Baixa ``SHA256SUMS.txt`` e o arquivo (``kind``) para ``folder`` e confere o SHA-256. Devolve o caminho."""
    asset, sums_asset = select_asset(release, kind)
    opener = opener or build_opener()
    folder = Path(folder)
    sums_path = download(sums_asset.url, folder / SUMS_NAME, opener, cancelled=cancelled,
                         max_bytes=MAX_SUMS_BYTES)
    try:
        expected = parse_sums(sums_path.read_text(encoding="utf-8", errors="replace")).get(asset.name)
    finally:
        _unlink(sums_path)
    if not expected:
        raise ChecksumError(f"O {SUMS_NAME} não lista {asset.name}; por segurança, nada foi baixado.")
    dest = folder / asset.name
    if dest.exists():               # um download anterior completo e válido é reaproveitado
        if sha256_file(dest) == expected:
            if progress:
                progress(1, 1)
            return dest
        _unlink(dest)
    download(asset.url, dest, opener, progress=progress, cancelled=cancelled, expected_size=asset.size)
    verify_file(dest, expected)
    return dest


# ------------------------------------------------------------------------------------- instalação / modo

def updater_enabled(env=None, frozen: bool | None = None) -> bool:
    """Só no executável (ou com ``MOBILIA_UPDATER_TEST=1``): pelo código-fonte, o ``git pull`` é a atualização."""
    env = os.environ if env is None else env
    frozen = runtime.is_frozen() if frozen is None else frozen
    return bool(frozen) or env.get(ENV_TEST) == "1"


def install_kind(frozen: bool | None = None, executable: str | None = None, meipass: str | None = None) -> str:
    """``"dev"`` (código-fonte), ``"portable"`` (arquivo único, extraído numa pasta temporária) ou ``"installed"``
    (pasta do programa: pelo instalador ou pelo ``GERAR_EXE.bat``)."""
    frozen = runtime.is_frozen() if frozen is None else frozen
    if not frozen:
        return "dev"
    exe_dir = Path(executable or sys.executable).resolve().parent
    mei = meipass if meipass is not None else getattr(sys, "_MEIPASS", None)
    if mei:
        try:
            Path(mei).resolve().relative_to(exe_dir)
        except ValueError:
            return "portable"
    return "installed"


def asset_kind(kind: str) -> str:
    return "portable" if kind == "portable" else "setup"


def updates_dir(home: Path | None = None) -> Path:
    return Path(home or runtime.app_home()) / "updates"


def portable_target_dir(executable: str | None = None) -> Path:
    """Onde salvar o novo portátil: ao lado do atual; sem permissão de escrita, na pasta Downloads."""
    exe_dir = Path(executable or sys.executable).resolve().parent
    if os.access(exe_dir, os.W_OK):
        return exe_dir
    dl = Path.home() / "Downloads"
    return dl if dl.is_dir() else updates_dir()


def installer_command(setup: Path, app_dir: Path, log: Path | None = None) -> list[str]:
    """Linha de comando do instalador Inno Setup em modo de atualização silenciosa (reabre o 7xAi no fim)."""
    cmd = [str(setup), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS",
           "/NORESTARTAPPLICATIONS", f"/DIR={app_dir}", AUTOUPDATE_FLAG]
    if log:
        cmd.append(f"/LOG={log}")
    return cmd


def _detached_flags() -> int:
    return (getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if os.name == "nt" else 0)


def launch_detached(cmd: list[str]) -> None:
    """Inicia um processo que continua vivo depois que o 7xAi fecha."""
    subprocess.Popen(cmd, close_fds=True, creationflags=_detached_flags(),
                     start_new_session=os.name != "nt")


def launch_installer(setup: Path, app_dir: Path | None = None) -> list[str]:
    app_dir = app_dir or Path(sys.executable).resolve().parent
    cmd = installer_command(setup, app_dir, Path(setup).with_suffix(".log"))
    launch_detached(cmd)
    return cmd


def cleanup_updates(folder: Path | None = None, current: str = __version__) -> None:
    """Limpa ``<dados>/updates``: instaladores já usados (versão <= atual), downloads interrompidos e registros
    de versões anteriores (o registro da instalação da versão atual fica, para diagnóstico)."""
    folder = folder or updates_dir()
    if not folder.is_dir():
        return
    cur = parse_version(current)
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = re.search(r"-v(\d+\.\d+\.\d+)\.(exe|log)$", p.name)
        ver = parse_version(m.group(1)) if m else None
        old = bool(ver and cur and (ver <= cur if m.group(2) == "exe" else ver < cur))
        if old or p.name.endswith(".part"):
            _unlink(p)


# --------------------------------------------------------------------------------------------- preferências

@dataclass
class UpdaterSettings:
    auto_check: bool = True
    last_check: float = 0.0
    skipped_version: str = ""
    extra: dict = field(default_factory=dict)

    @staticmethod
    def path(home: Path | None = None) -> Path:
        return Path(home or runtime.app_home()) / "settings" / "updater.json"

    @classmethod
    def load(cls, path: Path | None = None) -> "UpdaterSettings":
        path = path or cls.path()
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        s = cls()
        s.auto_check = bool(raw.get("auto_check", True))
        try:
            s.last_check = float(raw.get("last_check") or 0.0)
        except (TypeError, ValueError):
            s.last_check = 0.0
        sk = raw.get("skipped_version")
        s.skipped_version = sk if isinstance(sk, str) and parse_version(sk) else ""
        return s

    def save(self, path: Path | None = None) -> None:
        path = Path(path or self.path())
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            data = asdict(self)
            data.pop("extra", None)
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass                    # preferência não gravada não pode atrapalhar o uso do app

    def due(self, now: float | None = None, interval: float = CHECK_INTERVAL_S) -> bool:
        """Consulta automática: ligada e a última foi há mais de ``interval`` (ou o relógio voltou)."""
        now = time.time() if now is None else now
        return self.auto_check and (now - self.last_check >= interval or self.last_check > now + 60)

    def should_offer(self, result: CheckResult, manual: bool = False) -> bool:
        """Mostrar a oferta? A versão pulada só volta a aparecer numa consulta manual."""
        if result.status != "update" or result.release is None:
            return False
        return manual or result.release.version != self.skipped_version

    def record_check(self, result: CheckResult, now: float | None = None) -> None:
        """Conta como consulta do dia só quando o GitHub respondeu (sem internet, tenta na próxima abertura)."""
        if result.status in ("update", "current", "rate_limited", "error"):
            self.last_check = time.time() if now is None else now
