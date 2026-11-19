"""Conexões HTTP(S) que não travam em IPv6 lento (só biblioteca padrão).

Medido no notebook do dono: cada conexão HTTPS aberta pelo Python levava ~42 s (o ``curl`` 0,6 s). O
``socket.create_connection`` tenta os endereços na ordem do ``getaddrinfo`` — IPv6 primeiro — e só passa ao próximo
depois do tempo de conexão do sistema (~21 s por endereço no Windows): com dois endereços IPv6 sem rota de verdade,
42 s antes do IPv4 que funciona. Aqui (no estilo *happy eyeballs*, sem mexer no módulo ``socket`` global):

* IPv4 primeiro, depois IPv6, alternando as famílias;
* cada tentativa com um tempo de conexão curto (``CONNECT_TIMEOUT``); a resposta continua com o ``timeout``
  normal da requisição;
* o proxy do sistema (``HTTPS_PROXY``) continua funcionando: só a abertura do soquete muda.

Uso: :func:`build_opener` (no lugar do ``urllib.request.build_opener``) ou :func:`urlopen`.
"""
from __future__ import annotations

import http.client
import socket
import ssl
import urllib.request

CONNECT_TIMEOUT = 4.0       # s por endereço; uma conexão saudável abre em < 1 s


def order_addresses(infos: list) -> list:
    """Endereços do ``getaddrinfo`` com IPv4 primeiro e as famílias alternadas (v4, v6, v4, v6...)."""
    v4 = [i for i in infos if i[0] == socket.AF_INET]
    v6 = [i for i in infos if i[0] != socket.AF_INET]
    out = []
    for k in range(max(len(v4), len(v6))):
        out += [x[k] for x in (v4, v6) if k < len(x)]
    return out


def create_connection(address: tuple, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None,
                      connect_timeout: float = CONNECT_TIMEOUT, resolver=socket.getaddrinfo,
                      factory=socket.socket) -> socket.socket:
    """Como ``socket.create_connection``, mas IPv4 primeiro e no máximo ``connect_timeout`` s por endereço."""
    host, port = address[0], address[1]
    infos = resolver(host, port, 0, socket.SOCK_STREAM)
    if not infos:
        raise OSError(f"sem endereço para {host}")
    t_read = None if timeout is socket._GLOBAL_DEFAULT_TIMEOUT else timeout
    t_conn = connect_timeout if t_read is None else min(connect_timeout, t_read)
    last: BaseException | None = None
    for family, stype, proto, _canon, sa in order_addresses(list(infos)):
        sock = None
        try:
            sock = factory(family, stype, proto)
            sock.settimeout(t_conn)
            if source_address:
                sock.bind(source_address)
            sock.connect(sa)
            sock.settimeout(t_read)
            return sock
        except OSError as e:           # inclui TimeoutError: tenta o próximo endereço
            last = e
            if sock is not None:
                sock.close()
    raise last if last is not None else OSError(f"não foi possível conectar a {host}")


class FastHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._create_connection = create_connection


class FastHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._create_connection = create_connection


class FastHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(FastHTTPConnection, req)


class FastHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, context: ssl.SSLContext | None = None):
        super().__init__(context=context or ssl.create_default_context())
        self._ctx = context or ssl.create_default_context()

    def https_open(self, req):
        return self.do_open(FastHTTPSConnection, req, context=self._ctx)


def build_opener(*handlers, context: ssl.SSLContext | None = None) -> urllib.request.OpenerDirector:
    """``urllib.request.build_opener`` com as conexões rápidas (proxy e redirecionamentos como no padrão)."""
    return urllib.request.build_opener(FastHTTPHandler(), FastHTTPSHandler(context), *handlers)


def urlopen(req, timeout: float = 30.0, context: ssl.SSLContext | None = None):
    """``urllib.request.urlopen`` com as conexões rápidas."""
    return build_opener(context=context).open(req, timeout=timeout)


__all__ = ["create_connection", "order_addresses", "build_opener", "urlopen", "FastHTTPSHandler",
           "FastHTTPHandler", "CONNECT_TIMEOUT"]
