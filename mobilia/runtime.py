"""Onde ficam os dados do aplicativo e como o processo de trabalho (worker) é iniciado.

A interface gráfica nunca roda o pipeline no próprio processo (ver ``mobilia/jobs.py``).
Este módulo resolve QUAL interpretador executa ``mobilia.cli``:

- desenvolvimento (código-fonte): o próprio interpretador atual,
  ``[sys.executable, "-X", "utf8", "-m", "mobilia.cli", ...]``;
- executável congelado (PyInstaller): a interface é leve e o motor pesado (Python
  embutido com PyTorch CUDA, Blender ``bpy`` e modelos de IA) fica em
  ``%LOCALAPPDATA%\\MobiliaStudio\\runtime``; o worker é
  ``<runtime>\\python\\python.exe -X utf8 -m mobilia.cli ...``.

``MOBILIA_HOME`` substitui a pasta base (mesma regra de ``mobilia/config.py``).
``MOBILIA_WORKER`` (lista JSON) substitui o prefixo do comando do worker — útil para
depuração e para os testes automatizados.

``-X utf8`` é obrigatório: o Python embutido do Windows ignora variáveis ``PYTHON*``
(o arquivo ``._pth`` ativa o modo isolado) e, sem o modo UTF-8, a saída padrão usaria
a página de código ANSI, quebrando mensagens como "Foto 1920×1080 → vídeo".

Este módulo usa apenas a biblioteca padrão: é importado pela interface congelada.
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

ENV_HOME = "MOBILIA_HOME"
# Caches de kernels do driver NVIDIA: o padrão (256 MB a 1 GB) é pequeno para PyTorch + Cycles
# (OptiX e CUDA) + OIDN, e um cache cheio faz a compilação de kernels voltar a cada execução.
# 4 GiB é o máximo aceito pelo driver. Valem para o worker e para o Blender (processo filho).
GPU_CACHE_ENV = {"CUDA_CACHE_MAXSIZE": str(4 << 30), "OPTIX_CACHE_MAXSIZE": str(4 << 30)}
ENV_WORKER = "MOBILIA_WORKER"
WORKER_MODULE = "mobilia.cli"
CREATE_NO_WINDOW = 0x08000000

# Raiz do código-fonte (pasta que contém o pacote ``mobilia``) quando não congelado.
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def is_frozen() -> bool:
    """True quando roda dentro do executável gerado pelo PyInstaller."""
    return bool(getattr(sys, "frozen", False))


def app_home(env: Mapping[str, str] | None = None, platform: str | None = None) -> Path:
    """Pasta base dos dados do usuário (motor, modelos, trabalhos, registros).

    Windows: ``%LOCALAPPDATA%\\MobiliaStudio``; Linux/macOS: ``~/.local/share/mobilia-studio``.
    ``MOBILIA_HOME`` tem prioridade. Mesma regra de ``mobilia.config``.

    Não renomear para "7xAi": a pasta guarda o motor já instalado (vários GB). Com outro nome, quem
    atualiza o app perderia a instalação e baixaria tudo de novo.
    """
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if env.get(ENV_HOME):
        return Path(env[ENV_HOME])
    if platform == "win32":
        base = env.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "MobiliaStudio"
    base = env.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "mobilia-studio"


@dataclass(frozen=True)
class RuntimePaths:
    """Todas as pastas e arquivos do motor instalado no primeiro uso."""

    home: Path

    @property
    def runtime(self) -> Path:
        return self.home / "runtime"

    @property
    def python_dir(self) -> Path:
        return self.runtime / "python"

    @property
    def python_exe(self) -> Path:
        return self.python_dir / "python.exe"

    @property
    def site_packages(self) -> Path:
        return self.python_dir / "Lib" / "site-packages"

    @property
    def app_dir(self) -> Path:
        """Cópia do pacote ``mobilia`` usada pelo worker (entra no ``sys.path`` via ``._pth``)."""
        return self.runtime / "app"

    @property
    def downloads(self) -> Path:
        """Arquivos baixados (retomáveis); apagados ao fim de cada etapa."""
        return self.runtime / "downloads"

    @property
    def wheelhouse(self) -> Path:
        return self.runtime / "wheelhouse"

    @property
    def state_file(self) -> Path:
        return self.runtime / "setup_state.json"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def hf_home(self) -> Path:
        """Cache do Hugging Face com os pesos MoGe-2 e OneFormer."""
        return self.models_dir / "huggingface"

    @property
    def jobs_dir(self) -> Path:
        return self.home / "jobs"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"


def runtime_paths(home: Path | None = None) -> RuntimePaths:
    return RuntimePaths(Path(home) if home is not None else app_home())


def payload_root() -> Path:
    """Onde estão o pacote ``mobilia`` e ``requirements/`` que o instalador copia para o motor.

    No executável congelado: ``<app>/_internal/payload`` (incluído pelo .spec).
    Em desenvolvimento: a raiz do repositório.
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "payload"
    return SOURCE_ROOT


def worker_prefix(env: Mapping[str, str] | None = None, frozen: bool | None = None,
                  paths: RuntimePaths | None = None) -> list[str]:
    """Prefixo do comando do worker (sem os argumentos da CLI)."""
    env = os.environ if env is None else env
    frozen = is_frozen() if frozen is None else frozen
    override = env.get(ENV_WORKER)
    if override:
        try:
            prefix = json.loads(override)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{ENV_WORKER} deve ser uma lista JSON de strings: {exc}") from exc
        if not isinstance(prefix, list) or not prefix or not all(isinstance(p, str) for p in prefix):
            raise ValueError(f"{ENV_WORKER} deve ser uma lista JSON não vazia de strings.")
        return list(prefix)
    if frozen:
        paths = paths or runtime_paths(app_home(env))
        return [str(paths.python_exe), "-X", "utf8", "-m", WORKER_MODULE]
    return [sys.executable, "-X", "utf8", "-m", WORKER_MODULE]


def worker_command(args: Sequence[str], env: Mapping[str, str] | None = None, frozen: bool | None = None,
                   paths: RuntimePaths | None = None) -> list[str]:
    """Comando completo do worker, por exemplo ``worker_command(["hardware", "--json"])``."""
    return worker_prefix(env, frozen, paths) + [str(a) for a in args]


def uses_embedded_runtime(env: Mapping[str, str] | None = None, frozen: bool | None = None) -> bool:
    """True quando o worker é o Python embutido instalado no primeiro uso."""
    env = os.environ if env is None else env
    frozen = is_frozen() if frozen is None else frozen
    return frozen and not env.get(ENV_WORKER)


def worker_environment(base: Mapping[str, str] | None = None, frozen: bool | None = None,
                       paths: RuntimePaths | None = None) -> dict[str, str]:
    """Variáveis de ambiente do worker.

    - UTF-8 sempre (além do ``-X utf8``);
    - ``MOBILIA_HOME`` explícito: interface e worker usam a mesma pasta de dados;
    - caches de kernels da NVIDIA com 4 GiB (``GPU_CACHE_ENV``);
    - no motor embutido, ``HF_HOME`` aponta para os modelos baixados no primeiro uso;
    - em desenvolvimento, a raiz do código entra no ``PYTHONPATH``.
    """
    env = dict(os.environ if base is None else base)
    frozen = is_frozen() if frozen is None else frozen
    paths = paths or runtime_paths(app_home(env))
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env[ENV_HOME] = str(paths.home)
    # O transformers tenta baixar em segundo plano uma conversão .safetensors do OneFormer
    # (~900 MB) a cada execução online; os pesos .bin já baixados bastam.
    env["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    apply_gpu_cache_env(env)
    if uses_embedded_runtime(env, frozen):
        env["HF_HOME"] = str(paths.hf_home)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
    else:
        old = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(SOURCE_ROOT) + (os.pathsep + old if old else "")
    return env


def apply_gpu_cache_env(env: dict) -> dict:
    """Define o tamanho dos caches de kernels da NVIDIA (``GPU_CACHE_ENV``) sem sobrescrever um
    valor já escolhido pelo usuário."""
    for k, v in GPU_CACHE_ENV.items():
        env.setdefault(k, v)
    return env


def worker_cwd(frozen: bool | None = None, paths: RuntimePaths | None = None) -> Path:
    frozen = is_frozen() if frozen is None else frozen
    if frozen:
        return (paths or runtime_paths()).runtime
    return SOURCE_ROOT


# ----------------------------------------------------------------------------- árvore de processos

def background_popen_kwargs() -> dict:
    """Argumentos de ``subprocess.Popen`` para um processo filho que pode ser encerrado com
    todos os seus descendentes (novo grupo de processos) e sem abrir janela de console."""
    if os.name == "nt":
        return {"creationflags": CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_process_tree(pid: int, grace_s: float = 3.0) -> None:
    """Encerra o processo ``pid`` e todos os descendentes.

    Windows: ``taskkill /T /F`` (o worker inicia o Blender em outro processo).
    POSIX: o worker é líder de sessão (pgid = pid): SIGTERM no grupo e, depois do
    prazo, SIGKILL.
    """
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                       creationflags=CREATE_NO_WINDOW, check=False)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        os.kill(pid, signal.SIGTERM)
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class _IoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


class ProcessJob:
    """Agrupa um processo e todos os seus descendentes (Windows: Job Object).

    Com ``KILL_ON_JOB_CLOSE``, se a interface fechar ou travar, o Windows encerra o worker
    e o Blender junto. Em outros sistemas é um objeto vazio (usa-se o grupo de processos).
    """

    def __init__(self) -> None:
        self._handle = None
        self._k32 = None
        if os.name != "nt":
            return
        try:
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                    wintypes.DWORD]
            k32.SetInformationJobObject.restype = wintypes.BOOL
            k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k32.AssignProcessToJobObject.restype = wintypes.BOOL
            k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k32.TerminateJobObject.restype = wintypes.BOOL
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            k32.CloseHandle.restype = wintypes.BOOL
            handle = k32.CreateJobObjectW(None, None)
            if not handle:
                return
            info = _ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                               ctypes.byref(info), ctypes.sizeof(info)):
                k32.CloseHandle(handle)
                return
            self._k32, self._handle = k32, handle
        except (OSError, AttributeError):
            self._k32 = self._handle = None

    @property
    def active(self) -> bool:
        return self._handle is not None

    def add(self, pid: int) -> bool:
        if not self._handle or pid <= 0:
            return False
        proc = self._k32.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid)
        if not proc:
            return False
        try:
            return bool(self._k32.AssignProcessToJobObject(self._handle, proc))
        finally:
            self._k32.CloseHandle(proc)

    def terminate(self, exit_code: int = 1) -> bool:
        if not self._handle:
            return False
        return bool(self._k32.TerminateJobObject(self._handle, exit_code))

    def close(self) -> None:
        if self._handle:
            self._k32.CloseHandle(self._handle)
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - coletor de lixo
        try:
            self.close()
        except Exception:
            pass
