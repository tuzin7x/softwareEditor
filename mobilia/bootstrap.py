"""Instalação do motor no primeiro uso (Windows): Python embutido + PyTorch + Blender (bpy) + modelos.

O executável (.exe) é leve — só a interface. Na primeira execução este módulo monta, em
``%LOCALAPPDATA%\\MobiliaStudio\\runtime``, o motor pesado:

1. baixa o Python embutido oficial (python.org) e ativa ``site-packages`` no ``._pth``;
2. instala o pip;
3. instala as dependências fixadas em ``requirements/runtime-cuda.txt`` (ou ``runtime-cpu.txt``
   quando não há GPU NVIDIA);
4. copia o pacote ``mobilia`` para ``runtime/app``;
5. baixa os pesos dos modelos e os assets 3D (``mobilia.cli prefetch``) — depois disso tudo
   funciona offline.

Cada etapa concluída é gravada em ``setup_state.json``: se a internet cair, a próxima
tentativa continua de onde parou.

Atualizações: o estado guarda uma impressão digital (SHA-256) do código ``mobilia`` e dos
``requirements`` que vieram no .exe. Ao instalar um .exe novo, só as etapas cujo conteúdo mudou são
refeitas — em geral, apenas a cópia do motor (segundos); o pip só roda de novo se as dependências
mudaram, e mesmo assim baixa apenas o que é diferente. PyTorch, Blender e os modelos não são
baixados de novo. Usa só a biblioteca padrão (roda dentro do .exe).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from . import runtime

PY_VERSION = "3.13.17"
PY_EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
STEPS = ["python", "pip", "dependencias", "app", "modelos"]
STEP_LABELS = {
    "python": "Baixando o Python embutido",
    "pip": "Preparando o instalador de pacotes",
    "dependencias": "Instalando PyTorch (CUDA), Blender e bibliotecas de IA (~4 GB)",
    "app": "Copiando o motor do Mobília Studio",
    "modelos": "Baixando os modelos de IA e os móveis 3D (~2,5 GB)",
}

Progress = Callable[[str, float, str], None]


class SetupError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- funções puras (testáveis)

def patch_pth(text: str, extra_paths: list[str]) -> str:
    """Ajusta ``pythonXY._pth`` do Python embutido: ativa ``import site`` e acrescenta caminhos."""
    out = [ln.rstrip("\r") for ln in text.splitlines()
           if ln.strip() not in ("import site", "#import site", "# import site")]
    for p in extra_paths:
        if p not in out:
            out.append(p)
    out.append("import site")
    return "\r\n".join(out) + "\r\n"


def parse_requirements(text: str) -> tuple[list[str], list[str]]:
    """Separa opções do pip (``--extra-index-url ...``) dos requisitos."""
    opts, reqs = [], []
    for ln in text.splitlines():
        ln = ln.split(" #", 1)[0].strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("-"):
            opts += ln.split(None, 1)
        else:
            reqs.append(ln)
    return opts, reqs


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"done": []}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def is_first_run(state: dict) -> bool:
    """Primeira geração depois de instalar o motor ou de trocar as dependências (Blender/PyTorch):
    kernels da GPU, caches de disco e modelos ainda frios. Usado só na telemetria e no aviso
    "primeira execução — mais lenta"."""
    return (int(state.get("execucoes", 0) or 0) == 0
            or state.get("execucoes_dependencias") != state.get("hashes", {}).get("dependencias"))


def record_run(path: Path) -> dict:
    """Registra no estado da instalação que uma geração foi concluída."""
    state = load_state(path)
    state["execucoes"] = int(state.get("execucoes", 0) or 0) + 1
    state["execucoes_dependencias"] = state.get("hashes", {}).get("dependencias")
    save_state(path, state)
    return state


def pending_steps(state: dict) -> list[str]:
    return [s for s in STEPS if s not in state.get("done", [])]


def fingerprint(root: Path, skip: tuple[str, ...] = ("__pycache__",)) -> str:
    """SHA-256 dos caminhos relativos e do conteúdo dos arquivos sob ``root`` (ordem estável)."""
    h = hashlib.sha256()
    if not root.exists():
        return ""
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = f.relative_to(root).as_posix()
        if any(part in skip for part in rel.split("/")) or f.suffix in (".pyc", ".pyo"):
            continue
        h.update(rel.encode("utf-8") + b"\0")
        h.update(f.read_bytes() + b"\0")
    return h.hexdigest()


def payload_fingerprints(payload: Path) -> dict[str, str]:
    """Impressões digitais do que vem no .exe: o código do motor e as dependências fixadas."""
    return {"app": fingerprint(payload / "mobilia", skip=("__pycache__", "ui")),
            "dependencias": fingerprint(payload / "requirements")}


def sync_with_payload(state: dict, fps: dict[str, str]) -> dict:
    """Marca como pendentes as etapas cujo conteúdo mudou desde a última instalação (atualização)."""
    done = state.get("done", [])
    for step, fp in fps.items():
        if fp and step in done and state.get("hashes", {}).get(step) != fp:
            done.remove(step)
    state["done"] = done
    return state


def has_nvidia_driver() -> bool:
    """Verifica o driver NVIDIA sem PyTorch (nvidia-smi vem com o driver)."""
    exe = shutil.which("nvidia-smi") or (r"C:\Windows\System32\nvidia-smi.exe" if os.name == "nt" else None)
    if not exe or not Path(exe).exists():
        return False
    try:
        r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=20,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:
        return False


# ----------------------------------------------------------------------------- instalação

def _download(url: str, dest: Path, progress: Progress, step: str) -> None:
    """Download retomável (cabeçalho Range) para ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "MobiliaStudio-setup",
                                               **({"Range": f"bytes={have}-"} if have else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have and r.status == 206 else "wb"
        if mode == "wb":
            have = 0
        with open(part, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                have += len(chunk)
                if total:
                    progress(step, have / total, f"{STEP_LABELS[step]} ({have / 1e6:.0f}/{total / 1e6:.0f} MB)")
    os.replace(part, dest)


def _run(cmd: list[str], progress: Progress, step: str, env: dict | None = None, log=None) -> None:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    tail: list[str] = []
    for line in p.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-80:]
        if log:
            log.write(line + "\n")
        if line.startswith("{"):
            try:
                ev = json.loads(line)
                if ev.get("event") == "progress":
                    progress(step, float(ev.get("fraction", 0)), ev.get("message", ""))
                continue
            except ValueError:
                pass
        if line.startswith(("Collecting", "Downloading", "Installing", "Successfully")):
            progress(step, -1.0, line[:120])
    if p.wait() != 0:
        raise SetupError(f"Falha na etapa '{STEP_LABELS[step]}'.\n" + "\n".join(tail[-25:]))


def install(progress: Progress, paths: runtime.RuntimePaths | None = None, force_cpu: bool = False) -> None:
    """Executa (ou retoma) a instalação do motor. Lança SetupError com detalhes em caso de falha."""
    if os.name != "nt":
        raise SetupError("A instalação automática do motor é para Windows. Em Linux/macOS use requirements/dev.txt.")
    paths = paths or runtime.runtime_paths()
    paths.runtime.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    payload = runtime.payload_root()
    fps = payload_fingerprints(payload)
    state = sync_with_payload(load_state(paths.state_file), fps)
    log = open(paths.logs_dir / "instalacao.log", "a", encoding="utf-8")
    try:
        for step in pending_steps(state):
            progress(step, 0.0, STEP_LABELS[step])
            if step == "python":
                z = paths.downloads / Path(PY_EMBED_URL).name
                if not z.exists():
                    _download(PY_EMBED_URL, z, progress, step)
                if paths.python_dir.exists():
                    shutil.rmtree(paths.python_dir)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(paths.python_dir)
                pth = next(paths.python_dir.glob("python3*._pth"))
                pth.write_text(patch_pth(pth.read_text(encoding="utf-8"), ["Lib\\site-packages", "..\\app"]),
                               encoding="utf-8")
            elif step == "pip":
                gp = paths.downloads / "get-pip.py"
                if not gp.exists():
                    _download(GET_PIP_URL, gp, progress, step)
                _run([str(paths.python_exe), str(gp), "--no-warn-script-location"], progress, step, log=log)
            elif step == "dependencias":
                gpu = has_nvidia_driver() and not force_cpu
                req = payload / "requirements" / ("runtime-cuda.txt" if gpu else "runtime-cpu.txt")
                state["variant"] = "cuda" if gpu else "cpu"
                pip = [str(paths.python_exe), "-m", "pip", "install", "--no-warn-script-location",
                       "--disable-pip-version-check"]
                _run(pip + ["-r", str(req)], progress, step, log=log)
                _run(pip + ["--no-deps", "-r", str(payload / "requirements" / "runtime-nodeps.txt")],
                     progress, step, log=log)
            elif step == "app":
                if paths.app_dir.exists():
                    shutil.rmtree(paths.app_dir)
                shutil.copytree(payload / "mobilia", paths.app_dir / "mobilia",
                                ignore=shutil.ignore_patterns("__pycache__", "ui"))
            elif step == "modelos":
                env = runtime.worker_environment(frozen=True, paths=paths)
                _run(runtime.worker_command(["prefetch", "--json"], frozen=True, paths=paths), progress, step,
                     env=env, log=log)
            state.setdefault("done", []).append(step)
            if step in fps:
                state.setdefault("hashes", {})[step] = fps[step]
            save_state(paths.state_file, state)
            progress(step, 1.0, STEP_LABELS[step] + " — concluído")
        shutil.rmtree(paths.downloads, ignore_errors=True)
    finally:
        log.close()


def runtime_ready(paths: runtime.RuntimePaths | None = None) -> bool:
    paths = paths or runtime.runtime_paths()
    state = sync_with_payload(load_state(paths.state_file), payload_fingerprints(runtime.payload_root()))
    return paths.python_exe.exists() and not pending_steps(state)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":  # instalação pela linha de comando
    install(lambda s, f, m: print(f"[{s}] {m}", flush=True), force_cpu="--cpu" in sys.argv)
