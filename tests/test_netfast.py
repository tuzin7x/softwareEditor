"""Rede no teste real (notebook do dono): cada conexão HTTPS do Python levava ~42 s (IPv6 lento; o curl 0,6 s) e
todo trabalho consultava o Hugging Face mesmo com os modelos baixados. IPv4 primeiro com tempo curto por endereço
(``mobilia/netfast``) e carga só do disco quando os pesos já estão no cache (``weights.hub_offline``)."""
from __future__ import annotations

import os
import socket
import urllib.request

import pytest

from mobilia import netfast


def _info(fam, ip):
    return (fam, socket.SOCK_STREAM, 6, "", (ip, 443) if fam == socket.AF_INET else (ip, 443, 0, 0))


def test_ipv4_first_and_interleaved():
    infos = [_info(socket.AF_INET6, "::1"), _info(socket.AF_INET6, "::2"), _info(socket.AF_INET, "1.1.1.1"),
             _info(socket.AF_INET, "1.1.1.2")]
    assert [i[4][0] for i in netfast.order_addresses(infos)] == ["1.1.1.1", "::1", "1.1.1.2", "::2"]


class FakeSock:
    log: list = []

    def __init__(self, fam, typ, proto):
        self.fam, self.timeouts = fam, []

    def settimeout(self, t):
        self.timeouts.append(t)

    def connect(self, sa):
        FakeSock.log.append((sa[0], self.timeouts[-1]))
        if sa[0].startswith("1.") and sa[0] != "1.1.1.9":
            raise TimeoutError("sem rota")            # IPv4 que não responde: passa para o próximo logo

    def bind(self, a):
        pass

    def close(self):
        pass


def test_create_connection_short_connect_timeout_then_fallback():
    FakeSock.log = []
    infos = [_info(socket.AF_INET6, "::1"), _info(socket.AF_INET, "1.1.1.1"), _info(socket.AF_INET, "1.1.1.9")]
    s = netfast.create_connection(("h", 443), timeout=60, resolver=lambda *a: infos, factory=FakeSock,
                                  connect_timeout=4.0)
    # IPv4 primeiro (1.1.1.1 falha), depois o IPv6 conecta; cada tentativa com no máximo 4 s, leitura com 60 s
    assert [ip for ip, _ in FakeSock.log] == ["1.1.1.1", "::1"]
    assert all(t == 4.0 for _, t in FakeSock.log) and s.timeouts[-1] == 60
    with pytest.raises(OSError):
        netfast.create_connection(("h", 443), timeout=5, resolver=lambda *a: [infos[1]], factory=FakeSock)


def test_opener_uses_fast_connections_everywhere():
    op = netfast.build_opener()
    kinds = {type(h).__name__ for h in op.handlers}
    assert "FastHTTPSHandler" in kinds and "FastHTTPHandler" in kinds
    assert "HTTPSHandler" not in kinds and "HTTPHandler" not in kinds
    conn = netfast.FastHTTPSConnection("example.com")
    assert conn._create_connection is netfast.create_connection
    from mobilia.ui import updater
    assert any(type(h).__name__ == "FastHTTPSHandler" for h in updater.build_opener().handlers)
    from mobilia.modules.lead_finder import net
    assert net.fast_urlopen is netfast.urlopen


def test_hub_offline_only_when_everything_is_cached(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import constants

    from mobilia.analysis import weights as W

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    before = constants.HF_HUB_OFFLINE
    monkeypatch.setattr(W, "all_local", lambda r: True)
    with W.hub_offline("a/b") as off:
        assert off and constants.HF_HUB_OFFLINE and os.environ["HF_HUB_OFFLINE"] == "1"
    assert constants.HF_HUB_OFFLINE == before and "HF_HUB_OFFLINE" not in os.environ
    monkeypatch.setattr(W, "all_local", lambda r: r != "falta/um")
    with W.hub_offline("a/b", "falta/um") as off:              # algo faltando: online como antes (vai baixar)
        assert not off and constants.HF_HUB_OFFLINE == before


def test_model_loaders_use_hub_offline():
    """MoGe e OneFormer (análise, conferência e instâncias) carregam dentro de ``hub_offline``."""
    import inspect

    from mobilia.analysis import geometry, segmentation
    from mobilia.design import quality
    from mobilia.removal import instances

    for mod, fn in ((geometry, "estimate_geometry"), (segmentation, None), (quality.Segmenter, "__init__"),
                    (instances, "load_oneformer")):
        src = inspect.getsource(getattr(mod, fn) if fn else mod)
        assert "hub_offline(" in src, mod
