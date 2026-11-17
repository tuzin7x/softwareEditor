"""Trabalho fora da thread da interface: banco (uma thread dedicada) e tarefas de rede canceláveis.

Mesmo padrão do ``CliTask`` do 7xAi: ``threading.Thread`` + um ``Signal`` de um ``QObject`` criado na thread da
interface. Emitido de outra thread, o sinal é entregue na fila de eventos da interface (conexão enfileirada), e
os callbacks rodam com segurança na thread do Qt.
"""
from __future__ import annotations

import logging
import queue
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import Future

from PySide6.QtCore import QObject, Signal

from .net import Cancelled, CancelToken, NetError, redact
from .sources import SourceUnavailable

log = logging.getLogger("mobilia.lead_finder")


class _Bridge(QObject):
    call = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.call.connect(self._run)

    @staticmethod
    def _run(fn, args):
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 - um callback com erro não derruba a janela
            log.exception("erro num callback do Lead Finder")


class Dispatcher:
    """Executa funções na thread da interface a partir de qualquer thread."""

    def __init__(self):
        self._bridge = _Bridge()     # precisa ser criado na thread da interface

    def post(self, fn: Callable | None, *args) -> None:
        if fn is not None:
            self._bridge.call.emit(fn, args)


def friendly_error(e: BaseException) -> str:
    if isinstance(e, (NetError, SourceUnavailable, ValueError, PermissionError)):
        return str(e)
    return f"Erro inesperado: {redact(type(e).__name__ + ': ' + str(e))[:300]}"


class DbService:
    """Thread única dona do banco SQLite. ``call(fn, ...)`` enfileira ``fn(store, ...)``."""

    def __init__(self, factory: Callable[[], object], dispatcher: Dispatcher):
        self._factory = factory
        self._dispatcher = dispatcher
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="lead-finder-db", daemon=True)
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.store = None
        self._thread.start()

    def _loop(self) -> None:
        try:
            self.store = self._factory()
        except BaseException as e:  # noqa: BLE001
            self._error = e
            log.exception("não foi possível abrir o banco do Lead Finder")
        self._ready.set()
        while True:
            item = self._q.get()
            if item is None:
                break
            fn, args, kwargs, fut, on_done, on_error = item
            if fut.set_running_or_notify_cancel():
                try:
                    if self._error is not None:
                        raise RuntimeError(f"Banco indisponível: {self._error}")
                    res = fn(self.store, *args, **kwargs)
                except BaseException as e:  # noqa: BLE001
                    if not isinstance(e, (ValueError, PermissionError, KeyError)):
                        log.error("erro no banco: %s", redact(traceback.format_exc()))
                    fut.set_exception(e)
                    self._dispatcher.post(on_error, e)
                else:
                    fut.set_result(res)
                    self._dispatcher.post(on_done, res)
        if self.store is not None:
            try:
                self.store.close()
            except Exception:  # noqa: BLE001
                pass

    def call(self, fn: Callable, *args, on_done: Callable | None = None, on_error: Callable | None = None,
             **kwargs) -> Future:
        fut: Future = Future()
        self._q.put((fn, args, kwargs, fut, on_done, on_error))
        return fut

    def stop(self, timeout: float = 3.0) -> None:
        self._q.put(None)
        self._thread.join(timeout)


class Task:
    """Tarefa cancelável numa thread: ``fn(ctx)`` usa ``ctx.progress``, ``ctx.emit`` e ``ctx.check()``."""

    def __init__(self, fn: Callable, dispatcher: Dispatcher, on_progress=None, on_item=None, on_done=None,
                 on_error=None, on_cancel=None, name: str = "lead-finder-task"):
        self.token = CancelToken()
        self._fn = fn
        self._d = dispatcher
        self._cb = (on_progress, on_item, on_done, on_error, on_cancel)
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.finished = threading.Event()

    # interface usada pela função de trabalho
    def progress(self, fraction: float | None, message: str = "") -> None:
        self._d.post(self._cb[0], fraction, message)

    def emit(self, item) -> None:
        self._d.post(self._cb[1], item)

    def check(self) -> None:
        self.token.check()

    def sleep(self, s: float) -> None:
        self.token.sleep(s)

    @property
    def cancelled(self) -> bool:
        return self.token.cancelled

    # controle
    def start(self) -> Task:
        self.thread.start()
        return self

    def cancel(self) -> None:
        self.token.cancel()

    def running(self) -> bool:
        return self.thread.is_alive()

    def _run(self) -> None:
        try:
            res = self._fn(self)
        except Cancelled:
            self._d.post(self._cb[4])
        except BaseException as e:  # noqa: BLE001
            log.warning("tarefa falhou: %s", redact(traceback.format_exc()))
            self._d.post(self._cb[3], e)
        else:
            if self.token.cancelled:
                self._d.post(self._cb[4])
            else:
                self._d.post(self._cb[2], res)
        finally:
            self.finished.set()
