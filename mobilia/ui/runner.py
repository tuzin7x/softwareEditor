"""Execução do trabalho em processo separado e leitura dos eventos JSON (protocolo em mobilia/jobs.py).

O pipeline nunca roda dentro da interface: o worker (``mobilia.cli run --json``) é um
processo filho; aqui só lemos as linhas JSON e emitimos sinais Qt. Cancelar encerra a
árvore inteira de processos (o worker inicia o Blender em outro processo).
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from .. import runtime
from ..jobs import JobRequest
from ..telemetry import ENV_T_CLIQUE, ENV_T_INICIO


@dataclass
class QueuedJob:
    request: JobRequest
    status: str = "na fila"          # na fila | executando | pronto | erro | cancelado
    output: str | None = None
    error: str | None = None
    detail: str = ""
    log: list[str] = field(default_factory=list)
    report: str | None = None        # report.json (telemetria), quando o motor conclui
    work_dir: str | None = None
    submitted_at: float = field(default_factory=time.time)   # clique em "Gerar" (tempo percebido)


class JobRunner(QObject):
    """Fila de trabalhos (pronta para lote); executa um de cada vez."""

    progress = Signal(str, float, str)       # etapa, fração 0..1, mensagem
    preview = Signal(str)                    # caminho de imagem
    log = Signal(str)
    finished = Signal(object)                # QueuedJob
    queue_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue: deque[QueuedJob] = deque()
        self.history: list[QueuedJob] = []
        self.current: QueuedJob | None = None
        self._proc: subprocess.Popen | None = None
        self._cancel = threading.Event()

    # ------------------------------------------------------------------ fila
    def submit(self, req: JobRequest) -> QueuedJob:
        job = QueuedJob(req)
        self.queue.append(job)
        self.queue_changed.emit()
        if self.current is None:
            self._start_next()
        return job

    def busy(self) -> bool:
        return self.current is not None

    def cancel(self) -> None:
        self._cancel.set()
        if self._proc is not None and self._proc.poll() is None:
            runtime.kill_process_tree(self._proc.pid)

    def _start_next(self) -> None:
        if not self.queue:
            return
        job = self.queue.popleft()
        job.status = "executando"
        self.current = job
        self._cancel.clear()
        self.queue_changed.emit()
        threading.Thread(target=self._run, args=(job,), daemon=True).start()

    # ------------------------------------------------------------------ execução
    def command_for(self, req: JobRequest) -> list[str]:
        args = ["run", "--photo", req.photo_path, "--room", req.room_type, "--out", req.output_path,
                "--quality", req.quality, "--seed", str(req.seed), "--json"]
        if req.work_dir:
            args += ["--work-dir", req.work_dir]
        return runtime.worker_command(args)

    def _run(self, job: QueuedJob) -> None:
        job.work_dir = job.work_dir or job.request.work_dir
        env = runtime.worker_environment()
        env[ENV_T_CLIQUE] = repr(job.submitted_at)
        env[ENV_T_INICIO] = repr(time.time())
        try:
            self._proc = subprocess.Popen(
                self.command_for(job.request), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, cwd=str(runtime.worker_cwd()), text=True,
                encoding="utf-8", errors="replace", **runtime.background_popen_kwargs())
        except OSError as e:
            job.status, job.error = "erro", f"Não foi possível iniciar o motor: {e}"
            self._finish(job)
            return
        for raw in self._proc.stdout:
            line = raw.strip()
            if not line.startswith("{"):
                if line:
                    job.log.append(line)
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                job.log.append(line)
                continue
            self._handle(job, ev)
        rc = self._proc.wait()
        if self._cancel.is_set():
            job.status, job.error = "cancelado", "Cancelado pelo usuário."
        elif job.status == "executando":
            job.status = "erro"
            job.error = job.error or f"O motor terminou sem concluir (código {rc})."
            job.detail = job.detail or "\n".join(job.log[-60:])
        self._finish(job)

    def _handle(self, job: QueuedJob, ev: dict) -> None:
        kind = ev.get("event")
        if kind == "progress":
            self.progress.emit(ev.get("stage", ""), float(ev.get("fraction", 0.0)), ev.get("message", ""))
        elif kind == "preview":
            self.preview.emit(ev.get("path", ""))
        elif kind == "log":
            job.log.append(ev.get("message", ""))
            self.log.emit(ev.get("message", ""))
        elif kind == "done":
            job.status, job.output = "pronto", ev.get("output")
            job.report = ev.get("report") or job.report
            job.work_dir = ev.get("work_dir") or job.work_dir
        elif kind == "error":
            job.status, job.error, job.detail = "erro", ev.get("message", "Erro"), ev.get("detail", "")

    def _save_log(self, job: QueuedJob) -> None:
        """Grava a saída do motor na pasta do trabalho (entra no zip de diagnóstico).

        Cria a pasta se o motor falhou antes de criá-la (erro na partida): o diagnóstico da tela
        de erro continua disponível, com este registro."""
        if not job.work_dir:
            return
        lines = [f"status: {job.status}"] + ([f"erro: {job.error}"] if job.error else []) + job.log
        if job.detail:
            lines += ["", "detalhe:", job.detail]
        try:
            Path(job.work_dir).mkdir(parents=True, exist_ok=True)
            (Path(job.work_dir) / "interface.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass

    def _finish(self, job: QueuedJob) -> None:
        self._save_log(job)
        self._proc = None
        self.current = None
        self.history.append(job)
        self.finished.emit(job)
        self.queue_changed.emit()
        self._start_next()


class HardwareProbe(QObject):
    """Consulta GPU/VRAM/perfil em segundo plano (``mobilia.cli hardware --json``)."""

    done = Signal(object)            # dict ou None

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            r = subprocess.run(runtime.worker_command(["hardware", "--json"]), capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=runtime.worker_environment(),
                               cwd=str(runtime.worker_cwd()), timeout=180,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            line = next((ln for ln in r.stdout.splitlines()[::-1] if ln.startswith("{")), None)
            self.done.emit(json.loads(line) if line else None)
        except Exception:
            self.done.emit(None)
