"""Instalação do motor no primeiro uso (Windows): Python embutido + PyTorch + Blender (bpy) + modelos.

O executável (.exe) é leve — só a interface. Na primeira execução este módulo monta, em
``%LOCALAPPDATA%\\MobiliaStudio\\runtime``, o motor pesado:

1. baixa o Python embutido oficial (python.org) e ativa ``site-packages`` no ``._pth``;
2. instala o pip;
3. instala as dependências fixadas em ``requirements/runtime-cuda.txt`` (ou ``runtime-cpu.txt``
   quando não há GPU NVIDIA);
4. copia o pacote ``mobilia`` para ``runtime/app``;
5. baixa os pesos dos modelos de IA (``mobilia.cli prefetch --modelos``);
6. baixa todos os móveis 3D do catálogo com as texturas 1k e 2k (``prefetch --ativos``) —
   depois disso tudo funciona offline;
7. aquece a GPU (``mobilia.cli aquecer``): renders de 64 × 64 com OptiX e com CUDA no Blender
   instalado (kernels de render e do denoiser nos caches do driver, com 4 GiB) e uma carga dos
   modelos de análise. Marca a próxima geração como "primeira execução" (medida à parte).

Nenhum custo único (download, compilação de kernels, carga fria) cai no tempo do vídeo. Cada
etapa concluída é gravada em ``setup_state.json``: se a internet cair, a próxima tentativa
continua de onde parou; repetir uma etapa concluída não refaz o trabalho (os caches são
conferidos). O aquecimento nunca bloqueia: uma falha fica registrada e o primeiro vídeo paga o
custo.

Atualizações: o estado guarda uma impressão digital (SHA-256) do código ``mobilia`` e dos
``requirements`` que vieram no .exe. Ao instalar um .exe novo, só as etapas cujo conteúdo mudou são
refeitas — em geral, apenas a cópia do motor (segundos); o pip só roda de novo se as dependências
mudaram, e mesmo assim baixa apenas o que é diferente. PyTorch, Blender e os modelos não são
baixados de novo. Os móveis voltam a ser conferidos quando o catálogo muda, os modelos quando os
ids mudam, e o aquecimento é refeito quando mudam as dependências (Blender/PyTorch) ou a GPU/o
driver NVIDIA (os caches de kernels do driver são por versão). Usa só a biblioteca padrão (roda
dentro do .exe).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from . import runtime

PY_VERSION = "3.13.17"
PY_EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
STEPS = ["python", "pip", "dependencias", "app", "modelos", "ativos", "aquecimento"]
STEP_LABELS = {
    "python": "Baixando o Python embutido",
    "pip": "Preparando o instalador de pacotes",
    "dependencias": "Instalando PyTorch (CUDA), Blender e bibliotecas de IA (~4 GB)",
    "app": "Copiando o motor do 7xAi",
    "modelos": "Baixando os modelos de IA (~2 GB)",
    "ativos": "Baixando os móveis 3D do catálogo (texturas 2k)",
    "aquecimento": "Preparando a GPU: kernels de render e modelos de IA (só desta vez)",
}
# Etapas que nunca bloqueiam a instalação: a falha fica em ``state[<etapa>]`` e o primeiro
# trabalho paga o custo que ela pouparia.
OPTIONAL_STEPS = ("aquecimento",)
# Resoluções das texturas que os trabalhos usam em cada variante (ver ``pipeline._quality``:
# "final" com GPU usa 2k; "previa" e a CPU usam 1k).
TEXTURE_RES = {"cuda": "1k,2k", "cpu": "1k"}

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
    """Primeira geração depois de instalar o motor, de trocar as dependências (Blender/PyTorch) ou
    de refazer o aquecimento (GPU/driver novos): caches de disco e de kernels possivelmente ainda
    frios. Usado na telemetria (custos medidos à parte) e no aviso "primeira execução"."""
    return (int(state.get("execucoes", 0) or 0) == 0
            or bool(state.get("primeira_execucao_pendente"))
            or state.get("execucoes_dependencias") != state.get("hashes", {}).get("dependencias"))


def record_run(path: Path) -> dict:
    """Registra no estado da instalação que uma geração foi concluída."""
    state = load_state(path)
    state["execucoes"] = int(state.get("execucoes", 0) or 0) + 1
    state["execucoes_dependencias"] = state.get("hashes", {}).get("dependencias")
    state.pop("primeira_execucao_pendente", None)
    save_state(path, state)
    return state


def warmup_status(state: dict) -> dict:
    """Situação do aquecimento do instalador para as dependências instaladas agora.

    ``feito``: o aquecimento rodou depois da última troca de dependências; ``ok``: Blender e
    modelos aqueceram sem erro. Vai para ``report.json`` (``primeira_execucao_custos``)."""
    aq = state.get("aquecimento") or {}
    deps = state.get("hashes", {}).get("dependencias")
    feito = bool(aq) and "aquecimento" in state.get("done", []) and aq.get("dependencias") == deps
    res = aq.get("resultado") or {}
    devs = (res.get("blender") or {}).get("dispositivos") or []
    return {"feito": feito, "ok": bool(feito and aq.get("ok")), "data": aq.get("data"),
            "parede_s": aq.get("parede_s"),
            "blender_kernels_s": round(sum(float(d.get("kernels_s") or 0.0) for d in devs), 3) if devs else None,
            "dispositivos": [d.get("dispositivo") for d in devs if d.get("ok")],
            "modelos_ok": (res.get("modelos") or {}).get("ok"),
            "erro": aq.get("erro")}


def first_run_notice(state: dict) -> str | None:
    """Aviso da interface ao gerar: só na primeira execução (ver ``is_first_run``)."""
    if not is_first_run(state):
        return None
    if warmup_status(state)["ok"]:
        return ("Primeira execução depois da instalação: pode ser um pouco mais lenta "
                "(caches de disco ainda frios). As próximas ficam mais rápidas.")
    return ("Primeira execução — mais lenta: a GPU ainda vai preparar os kernels de render e carregar "
            "os modelos pela primeira vez (pode levar alguns minutos). As próximas ficam mais rápidas.")


def mark_warmup(state: dict, result: dict | None, fps: dict[str, str], erro: str | None = None,
                seconds: float | None = None) -> dict:
    """Grava o resultado do aquecimento e marca a próxima geração como primeira execução."""
    result = result if isinstance(result, dict) else {}
    state["aquecimento"] = {
        "ok": bool(result.get("ok")) and not erro,
        "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dependencias": fps.get("dependencias") or state.get("hashes", {}).get("dependencias"),
        "gpu": gpu_signature(),
        "parede_s": round(float(seconds), 2) if seconds is not None else result.get("parede_s"),
        "resultado": result,
        "erro": erro,
    }
    state["primeira_execucao_pendente"] = True
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


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def _file_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def models_fingerprint(payload: Path) -> str:
    """Os ids dos modelos (``config.py``) e a lista de downloads (``analysis/weights.py``)."""
    ids = re.findall(r'^(\w*MODEL_ID\w*)\s*=\s*"([^"]+)"', _file_text(payload / "mobilia" / "config.py"), re.M)
    weights = _file_text(payload / "mobilia" / "analysis" / "weights.py")
    return _text_sha("\n".join(f"{k}={v}" for k, v in sorted(ids)) + ("\0" + weights if ids else ""))


def _nvidia_smi() -> str | None:
    """Caminho do ``nvidia-smi`` (vem com o driver NVIDIA) ou None."""
    exe = shutil.which("nvidia-smi") or (r"C:\Windows\System32\nvidia-smi.exe" if os.name == "nt" else None)
    return exe if exe and Path(exe).exists() else None


@functools.lru_cache(maxsize=1)
def gpu_signature() -> str:
    """Nome das GPUs NVIDIA + versão do driver (``nvidia-smi``); vazio sem driver. Os caches de
    kernels (OptiX/CUDA) valem por GPU e por versão de driver: a troca refaz o aquecimento."""
    exe = _nvidia_smi()
    if not exe:
        return ""
    try:
        r = subprocess.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return ";".join(sorted(ln.strip() for ln in r.stdout.splitlines() if ln.strip())) if r.returncode == 0 else ""
    except Exception:
        return ""


def payload_fingerprints(payload: Path, gpu: str | None = None) -> dict[str, str]:
    """Impressões digitais do que vem no .exe (e da GPU), uma por etapa que depende delas:
    código do motor, dependências fixadas, ids dos modelos, catálogo de móveis e, para o
    aquecimento, dependências + GPU/driver (``gpu``; ``None`` consulta o ``nvidia-smi``)."""
    deps = fingerprint(payload / "requirements")
    gpu = gpu_signature() if gpu is None else gpu
    app = fingerprint(payload / "mobilia", skip=("__pycache__", "ui"))
    fonts = fingerprint(payload / "mobilia" / "ui" / "fonts")   # vão para o motor (rótulos do vídeo)
    return {"app": _text_sha(app + "|" + fonts) if app and fonts else app,
            "dependencias": deps,
            "modelos": models_fingerprint(payload),
            "ativos": _text_sha(_file_text(payload / "mobilia" / "layout" / "catalog.py")),
            "aquecimento": _text_sha(deps + "|" + gpu) if deps else ""}


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
    exe = _nvidia_smi()
    if not exe:
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


def _run(cmd: list[str], progress: Progress, step: str, env: dict | None = None, log=None) -> dict | None:
    """Roda um comando mostrando o progresso; devolve o último evento JSON ``done`` (se houver)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    tail: list[str] = []
    done = None
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
                elif ev.get("event") == "log" and ev.get("message"):
                    progress(step, -1.0, str(ev["message"])[:160])
                elif ev.get("event") == "done":
                    done = ev
                continue
            except ValueError:
                pass
        if line.startswith(("Collecting", "Downloading", "Installing", "Successfully")):
            progress(step, -1.0, line[:120])
    if p.wait() != 0:
        raise SetupError(f"Falha na etapa '{STEP_LABELS[step]}'.\n" + "\n".join(tail[-25:]))
    return done


StepRunner = Callable[[], "dict | None"]


def run_steps(progress: Progress, state: dict, fps: dict[str, str], runners: dict[str, StepRunner],
              state_file: Path) -> dict:
    """Executa as etapas pendentes na ordem de ``STEPS``, gravando o estado depois de CADA uma.

    Retomada: uma falha (``SetupError`` ou outra exceção) interrompe aqui com o estado das etapas
    já concluídas salvo; a próxima chamada recomeça da etapa que falhou. Idempotência: etapas
    concluídas (e com a mesma impressão digital) não rodam de novo. Etapas de ``OPTIONAL_STEPS``
    que falham são registradas e dadas como concluídas (não bloqueiam o uso do programa).
    """
    for step in pending_steps(state):
        progress(step, 0.0, STEP_LABELS[step])
        t0 = time.perf_counter()
        result, erro = None, None
        try:
            result = runners[step]()
        except Exception as e:
            if step not in OPTIONAL_STEPS:
                raise
            erro = (str(e) or e.__class__.__name__)[:2000]
            progress(step, -1.0, f"{STEP_LABELS[step]}: não foi possível concluir ({erro.splitlines()[0][:120]}). "
                                 "O primeiro vídeo vai ser mais lento.")
        if step == "aquecimento":
            mark_warmup(state, (result or {}).get("aquecimento") if isinstance(result, dict) else None, fps,
                        erro=erro, seconds=time.perf_counter() - t0)
        done = state.setdefault("done", [])
        if step not in done:
            done.append(step)
        if fps.get(step):
            state.setdefault("hashes", {})[step] = fps[step]
        save_state(state_file, state)
        progress(step, 1.0, STEP_LABELS[step] + " — concluído")
    return state


def step_runners(progress: Progress, paths: runtime.RuntimePaths, payload: Path, state: dict, log=None,
                 force_cpu: bool = False) -> dict[str, StepRunner]:
    """O que cada etapa da instalação executa (o motor embutido roda as etapas de download e de
    aquecimento pela própria CLI, ``mobilia.cli``)."""
    def worker(step: str, args: list[str]) -> dict | None:
        env = runtime.worker_environment(frozen=True, paths=paths)
        return _run(runtime.worker_command(args, frozen=True, paths=paths), progress, step, env=env, log=log)

    def python_step():
        z = paths.downloads / Path(PY_EMBED_URL).name
        if not z.exists():
            _download(PY_EMBED_URL, z, progress, "python")
        if paths.python_dir.exists():
            shutil.rmtree(paths.python_dir)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(paths.python_dir)
        pth = next(paths.python_dir.glob("python3*._pth"))
        pth.write_text(patch_pth(pth.read_text(encoding="utf-8"), ["Lib\\site-packages", "..\\app"]),
                       encoding="utf-8")

    def pip_step():
        gp = paths.downloads / "get-pip.py"
        if not gp.exists():
            _download(GET_PIP_URL, gp, progress, "pip")
        _run([str(paths.python_exe), str(gp), "--no-warn-script-location"], progress, "pip", log=log)

    def deps_step():
        gpu = has_nvidia_driver() and not force_cpu
        req = payload / "requirements" / ("runtime-cuda.txt" if gpu else "runtime-cpu.txt")
        state["variant"] = "cuda" if gpu else "cpu"
        pip = [str(paths.python_exe), "-m", "pip", "install", "--no-warn-script-location",
               "--disable-pip-version-check"]
        _run(pip + ["-r", str(req)], progress, "dependencias", log=log)
        _run(pip + ["--no-deps", "-r", str(payload / "requirements" / "runtime-nodeps.txt")],
             progress, "dependencias", log=log)

    def app_step():
        if paths.app_dir.exists():
            shutil.rmtree(paths.app_dir)
        shutil.copytree(payload / "mobilia", paths.app_dir / "mobilia",
                        ignore=shutil.ignore_patterns("__pycache__", "ui"))
        # a interface não vai para o motor, mas as fontes dela sim: o vídeo antes e depois (``video/showcase``)
        # escreve os rótulos com a Inter (sem elas cairia na fonte bitmap padrão do PIL)
        fonts = payload / "mobilia" / "ui" / "fonts"
        if fonts.is_dir():
            shutil.copytree(fonts, paths.app_dir / "mobilia" / "ui" / "fonts")

    return {
        "python": python_step,
        "pip": pip_step,
        "dependencias": deps_step,
        "app": app_step,
        "modelos": lambda: worker("modelos", ["prefetch", "--json", "--modelos"]),
        "ativos": lambda: worker("ativos", ["prefetch", "--json", "--ativos", "--res",
                                            TEXTURE_RES.get(state.get("variant", "cuda"), "1k,2k")]),
        "aquecimento": lambda: worker("aquecimento", ["aquecer", "--json"]),
    }


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
    runners = step_runners(progress, paths, payload, state, log, force_cpu)
    try:
        run_steps(progress, state, fps, runners, paths.state_file)
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
