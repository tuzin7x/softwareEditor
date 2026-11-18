"""HTTP do Lead Finder: só biblioteca padrão (``urllib``), sempre fora da thread da interface.

* **Cancelável:** a requisição roda numa thread auxiliar e o chamador verifica o cancelamento a cada 0,2 s;
  cancelar devolve o controle na hora (a resposta atrasada é descartada).
* **Educado com os provedores:** intervalo mínimo por host, novas tentativas com espera crescente em 429/5xx e
  respeito ao ``Retry-After``.
* **Logs seguros:** chaves, tokens e parâmetros sensíveis nunca aparecem nas mensagens (``redact``).
* **Testável sem internet:** ``transport`` pode ser trocado por respostas gravadas.
"""
from __future__ import annotations

import json
import logging
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger("mobilia.lead_finder.net")

USER_AGENT = "7xAi-LeadFinder/0.1 (aplicativo desktop; consultas iniciadas pelo usuario)"
_SECRET_PARAM = re.compile(r"(?i)((?:key|token|secret|password|apikey|api_key|access_token)=)[^&\s]+")
_SECRET_HEADER = re.compile(r"(?i)\b(sk-ant-[\w-]{6})[\w-]+|\b(AIza[\w-]{4})[\w-]{10,}")


class Cancelled(Exception):
    """A operação foi cancelada pelo usuário."""


class NetError(Exception):
    """Falha de rede ou do provedor, com mensagem pronta para a interface (sem segredos)."""

    def __init__(self, message: str, status: int = 0, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def redact(text: str) -> str:
    """Remove chaves e tokens de URLs e mensagens antes de registrar."""
    t = _SECRET_PARAM.sub(r"\1***", str(text))
    return _SECRET_HEADER.sub(lambda m: (m.group(1) or m.group(2)) + "***", t)


class CancelToken:
    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def check(self) -> None:
        if self._ev.is_set():
            raise Cancelled()

    def sleep(self, seconds: float) -> None:
        """Espera ``seconds`` mas acorda (e levanta :class:`Cancelled`) se cancelarem."""
        if self._ev.wait(max(0.0, seconds)):
            raise Cancelled()


@dataclass
class Response:
    status: int
    headers: dict = field(default_factory=dict)
    body: bytes = b""

    def json(self):
        return json.loads(self.body.decode("utf-8"))

    def text(self, limit: int | None = None) -> str:
        data = self.body if limit is None else self.body[:limit]
        ctype = self.headers.get("content-type", "")
        m = re.search(r"charset=([\w-]+)", ctype, re.I)
        for enc in ([m.group(1)] if m else []) + ["utf-8", "latin-1"]:
            try:
                return data.decode(enc)
            except (LookupError, UnicodeDecodeError):
                continue
        return data.decode("utf-8", "replace")


Transport = Callable[[urllib.request.Request, float, int], Response]


def urllib_transport(req: urllib.request.Request, timeout: float, max_bytes: int) -> Response:
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:   # noqa: S310 - URLs fixas https
            body = r.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise NetError("Resposta grande demais; a consulta foi interrompida.")
            return Response(r.status, {k.lower(): v for k, v in r.headers.items()}, body)
    except urllib.error.HTTPError as e:
        try:
            body = e.read(min(max_bytes, 1 << 20))
        except Exception:  # noqa: BLE001
            body = b""
        return Response(e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, body)


class HttpClient:
    """Cliente HTTP com cancelamento, limite de frequência por host e novas tentativas."""

    def __init__(self, transport: Transport | None = None, min_interval: dict[str, float] | None = None,
                 retries: int = 3, timeout: float = 30.0, max_bytes: int = 8 << 20, sleeper=None):
        self.transport = transport or urllib_transport
        self.min_interval = dict(min_interval or {})
        self.retries = retries
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._sleep = sleeper            # testes: função de espera sem atraso real
        self.calls = 0

    # ----------------------------------------------------------------------------------- público
    def get_json(self, url: str, ctx: CancelToken | None = None, headers: dict | None = None, **kw):
        return self._json(self.request("GET", url, ctx=ctx, headers=headers, **kw))

    def post_json(self, url: str, payload, ctx: CancelToken | None = None, headers: dict | None = None, **kw):
        h = {"Content-Type": "application/json", **(headers or {})}
        return self._json(self.request("POST", url, ctx=ctx, headers=h,
                                       data=json.dumps(payload).encode("utf-8"), **kw))

    def post_form(self, url: str, form: str, ctx: CancelToken | None = None, **kw):
        h = {"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}
        return self._json(self.request("POST", url, ctx=ctx, headers=h, data=form.encode("utf-8"), **kw))

    def request(self, method: str, url: str, ctx: CancelToken | None = None, headers: dict | None = None,
                data: bytes | None = None, timeout: float | None = None, ok=(200,),
                retries: int | None = None) -> Response:
        """``retries=0`` para requisições pagas: um timeout tem resultado desconhecido (o provedor pode ter
        processado e cobrado), então não se repete sozinho — cada tentativa precisa passar pelo limite de custo."""
        max_retries = self.retries if retries is None else max(0, int(retries))
        ctx = ctx or CancelToken()
        host = urllib.request.urlparse(url).hostname or ""
        hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json, text/html;q=0.8, */*;q=0.5",
                **(headers or {})}
        attempt = 0
        while True:
            ctx.check()
            self._throttle(host, ctx)
            req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
            try:
                resp = self._run(req, timeout or self.timeout, ctx)
            except (Cancelled, NetError):
                raise
            except (TimeoutError, OSError, urllib.error.URLError) as e:
                err = NetError(_network_message(e), retryable=True)
                resp = None
            else:
                err = None
            if resp is not None:
                if resp.status in ok:
                    return resp
                retry = resp.status in (429, 500, 502, 503, 504)
                err = NetError(_status_message(resp, host), resp.status, retry)
                wait = _retry_after(resp)
            else:
                wait = None
            attempt += 1
            log.warning("HTTP %s %s falhou (%s), tentativa %d", method, redact(url), redact(str(err)), attempt)
            if not err.retryable or attempt > max_retries:
                raise err
            self._wait(ctx, wait if wait is not None else min(30.0, 1.5 * 2 ** (attempt - 1)))

    # ----------------------------------------------------------------------------------- interno
    def _json(self, resp: Response):
        try:
            return resp.json()
        except ValueError as e:
            raise NetError("O provedor respondeu num formato inesperado.") from e

    def _wait(self, ctx: CancelToken, seconds: float) -> None:
        if self._sleep is not None:
            ctx.check(); self._sleep(seconds); ctx.check()
        else:
            ctx.sleep(seconds)

    def _throttle(self, host: str, ctx: CancelToken) -> None:
        gap = self.min_interval.get(host, 0.0)
        if gap <= 0:
            return
        with self._lock:
            wait = self._last.get(host, 0.0) + gap - time.monotonic()
        if wait > 0:
            self._wait(ctx, wait)
        with self._lock:
            self._last[host] = time.monotonic()

    def _run(self, req, timeout: float, ctx: CancelToken) -> Response:
        self.calls += 1
        box: dict = {}

        def work():
            try:
                box["r"] = self.transport(req, timeout, self.max_bytes)
            except BaseException as e:  # noqa: BLE001 - repassado ao chamador
                box["e"] = e
        t = threading.Thread(target=work, name="lead-finder-http", daemon=True)
        t.start()
        while t.is_alive():
            t.join(0.2)
            if ctx.cancelled and t.is_alive():
                raise Cancelled()
        if "e" in box:
            raise box["e"]
        return box["r"]


def _retry_after(resp: Response) -> float | None:
    v = resp.headers.get("retry-after", "")
    try:
        return min(120.0, max(0.0, float(v)))
    except ValueError:
        return None


def _network_message(e: Exception) -> str:
    reason = getattr(e, "reason", e)
    if isinstance(reason, TimeoutError) or "timed out" in str(reason):
        return "O provedor demorou demais para responder."
    if isinstance(reason, ssl.SSLError):
        return "Falha na conexão segura (certificado)."
    return "Sem conexão com a internet ou o provedor está fora do ar."


def _status_message(resp: Response, host: str) -> str:
    detail = ""
    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error")
            detail = (err.get("message") if isinstance(err, dict) else err) or data.get("message") or ""
    except ValueError:
        pass
    base = {400: "Pedido recusado pelo provedor", 401: "Chave de API inválida ou ausente",
            403: "Acesso negado pelo provedor (confira a chave e as permissões)",
            404: "Não encontrado", 429: "Limite de consultas do provedor atingido; tente mais tarde"}.get(
        resp.status, f"O provedor respondeu com erro {resp.status}")
    detail = redact(str(detail))[:200]
    return f"{base} ({host}){': ' + detail if detail else ''}."
