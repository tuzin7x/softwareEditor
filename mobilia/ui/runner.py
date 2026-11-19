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
    status: str = "na fila"          # na fila | executando | pronto | reprovado | erro | cancelado
    output: str | None = None
    error: str | None = None
    detail: str = ""
    log: list[str] = field(default_factory=list)
    report: str | None = None        # report.json (telemetria), quando o motor conclui
    work_dir: str | None = None
    submitted_at: float = field(default_factory=time.time)   # clique em "Gerar" (tempo percebido)
    forecast_s: float | None = None  # previsão do tempo total (PT7), quando o render a calcula
    mode: str | None = None          # "A" (sala vazia) | "B" (foto mobiliada), anunciado pelo motor
    warning: str | None = None       # aviso do portão de qualidade do modo B (tela de resultado)


class JobRunner(QObject):
    """Fila de trabalhos (pronta para lote); executa um de cada vez."""

    progress = Signal(str, float, str)       # etapa, fração 0..1, mensagem
    preview = Signal(str)                    # caminho de imagem
    log = Signal(str)
    forecast = Signal(float, str)            # previsão do tempo total (s, desde o clique), texto
    mode = Signal(str, list)                 # modo escolhido pelo motor e a lista de etapas [(chave, rótulo)]
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
        if req.mode != "auto":
            args += ["--mode", req.mode]
        if req.accept_weak_plate:
            args.append("--aceitar-placa-fraca")
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
        elif kind == "forecast":
            # PT7: previsão do controle de orçamento, feita depois do quadro final
            job.forecast_s = float(ev.get("total_s") or 0.0)
            self.forecast.emit(job.forecast_s, ev.get("message", ""))
        elif kind == "log":
            job.log.append(ev.get("message", ""))
            self.log.emit(ev.get("message", ""))
        elif kind == "mode":
            job.mode = ev.get("mode")
            self.mode.emit(ev.get("mode", ""), [tuple(x) for x in ev.get("stages") or []])
        elif kind == "done":
            # modo B com a sala vazia reprovada: sem MP4, com aviso (a tela oferece as alternativas)
            job.status = "pronto" if ev.get("output") else "reprovado"
            job.output = ev.get("output")
            job.warning = ev.get("warning")
            job.mode = ev.get("mode") or job.mode
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


# ============================================================================ redesenho por IA (fluxo principal)
@dataclass
class CliResult:
    """Resultado de um subcomando do motor (``design``, ``upscale``, ``showcase``)."""
    kind: str                         # design | upscale | showcase
    args: list[str]
    status: str = "executando"        # executando | pronto | erro | cancelado
    done: dict = field(default_factory=dict)   # evento "done" do motor
    error: str | None = None
    detail: str = ""
    log: list[str] = field(default_factory=list)
    tag: object = None                # dado livre de quem pediu (ex.: a versão ampliada)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def output(self) -> str | None:
        return self.done.get("output")

    @property
    def outputs(self) -> list[str]:
        outs = self.done.get("outputs")
        if outs:
            return [str(o) for o in outs]
        return [self.output] if self.output else []


class CliTask(QObject):
    """Executa UM subcomando do motor em processo separado (``mobilia.cli <kind> ... --json``).

    A interface nunca espera o motor: o processo roda numa thread e os eventos JSON viram sinais Qt (entregues na
    thread da interface). ``cancel()`` encerra a árvore de processos. Um ``CliTask`` por tipo de trabalho: o
    redesenho e as ações da tela de resultados (ampliar, vídeo) não disputam o mesmo objeto."""

    progress = Signal(str, float, str)       # etapa, fração 0..1, mensagem
    preview = Signal(str, int)               # caminho da versão pronta, índice
    log = Signal(str)
    item = Signal(object)                    # lote: evento "item" (index, total, photo, status, outputs, message)
    finished = Signal(object)                # CliResult
    _ended = Signal(object)                  # interno: fim do processo (da thread do worker para a da interface)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current: CliResult | None = None
        self._proc: subprocess.Popen | None = None
        self._cancel = threading.Event()
        # ``current`` só é limpo na thread da interface, logo antes de ``finished``: quem vê ``busy() == False``
        # já recebeu o resultado (sem janela em que a tela acha que acabou e ainda não sabe como)
        self._ended.connect(self._on_ended)

    def _on_ended(self, res: "CliResult") -> None:
        if self.current is res:
            self.current = None
        self.finished.emit(res)

    def busy(self) -> bool:
        return self.current is not None

    def start(self, kind: str, args: list[str], tag: object = None) -> CliResult:
        """Inicia ``mobilia.cli <kind> <args> --json``. Levanta ``RuntimeError`` se já houver um em andamento."""
        if self.current is not None:
            raise RuntimeError("Já existe um trabalho em andamento.")
        args = [str(a) for a in args]
        if "--json" not in args:
            args.append("--json")
        res = CliResult(kind=str(kind), args=[str(kind)] + args, tag=tag)
        self.current = res
        self._cancel.clear()
        threading.Thread(target=self._run, args=(res,), daemon=True).start()
        return res

    def cancel(self) -> None:
        self._cancel.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            runtime.kill_process_tree(proc.pid)

    def command(self, res: CliResult) -> list[str]:
        return runtime.worker_command(res.args)

    def _run(self, res: CliResult) -> None:
        try:
            self._proc = subprocess.Popen(
                self.command(res), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=runtime.worker_environment(),
                cwd=str(runtime.worker_cwd()), text=True, encoding="utf-8", errors="replace",
                **runtime.background_popen_kwargs())
        except OSError as e:
            res.status, res.error = "erro", f"Não foi possível iniciar o motor: {e}"
            self._finish(res)
            return
        if self._cancel.is_set():          # cancelado antes do processo existir
            runtime.kill_process_tree(self._proc.pid)
        for raw in self._proc.stdout:
            line = raw.strip()
            if not line.startswith("{"):
                if line:
                    res.log.append(line)
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                res.log.append(line)
                continue
            self._handle(res, ev)
        rc = self._proc.wait()
        if self._cancel.is_set():
            res.status, res.error = "cancelado", "Cancelado pelo usuário."
        elif res.status == "executando":
            res.status = "erro"
            res.error = res.error or f"O motor terminou sem concluir (código {rc})."
            res.detail = res.detail or "\n".join(res.log[-60:])
        self._finish(res)

    def _handle(self, res: CliResult, ev: dict) -> None:
        kind = ev.get("event")
        if kind == "progress":
            try:
                frac = float(ev.get("fraction") or 0.0)
            except (TypeError, ValueError):
                frac = 0.0
            self.progress.emit(str(ev.get("stage") or ""), max(0.0, min(1.0, frac)), str(ev.get("message") or ""))
        elif kind == "preview":
            try:
                idx = int(ev.get("index") or 0)
            except (TypeError, ValueError):
                idx = 0
            self.preview.emit(str(ev.get("path") or ""), idx)
        elif kind == "log":
            msg = str(ev.get("message") or "")
            res.log.append(msg)
            self.log.emit(msg)
        elif kind == "item":
            self.item.emit(dict(ev))
        elif kind == "done":
            res.done = dict(ev)
            res.status = "pronto"
        elif kind == "error":
            res.status, res.error, res.detail = "erro", str(ev.get("message") or "Erro"), str(ev.get("detail") or "")

    def _finish(self, res: CliResult) -> None:
        res.finished_at = time.time()
        self._save_log(res)
        self._proc = None
        self._ended.emit(res)

    @staticmethod
    def _save_log(res: CliResult) -> None:
        """Grava a saída do motor ao lado do resultado (pasta de saída do pedido), para diagnóstico."""
        out = None
        a = res.args
        for flag in ("--out-dir", "--out"):
            if flag in a and a.index(flag) + 1 < len(a):
                out = Path(a[a.index(flag) + 1])
                out = out if flag == "--out-dir" else out.parent
                break
        if out is None:
            return
        lines = [f"comando: {' '.join(a)}", f"status: {res.status}"] + ([f"erro: {res.error}"] if res.error else [])
        lines += res.log + (["", "detalhe:", res.detail] if res.detail else [])
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / f"interface_{res.kind}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass
