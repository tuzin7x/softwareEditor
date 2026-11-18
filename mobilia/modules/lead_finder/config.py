"""Configurações e credenciais do Lead Finder (na pasta de dados do usuário, nunca no repositório).

* ``settings.json``: preferências e limites de custo (não secretos).
* ``secrets.json``: chaves de API digitadas pelo usuário. Permissão 600 onde o sistema permite. Variáveis de
  ambiente têm prioridade (``LEADFINDER_GOOGLE_API_KEY``, ``ANTHROPIC_API_KEY``) e nunca são gravadas.

As chaves não entram no executável, nos logs (``net.redact``) nem nas exportações.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ENV_GOOGLE = "LEADFINDER_GOOGLE_API_KEY"
ENV_ANTHROPIC = "ANTHROPIC_API_KEY"
SECRET_NAMES = {"google_places": ENV_GOOGLE, "anthropic": ENV_ANTHROPIC}


def data_dir() -> Path:
    """``<pasta de dados do 7xAi>/lead_finder`` (respeita ``MOBILIA_HOME``)."""
    from mobilia import runtime
    return runtime.app_home() / "lead_finder"


@dataclass
class Settings:
    sender_name: str = ""
    sender_company: str = "7x Intelligence"
    include_optout: bool = True
    qualified_threshold: int = 60
    page_size: int = 100
    # Google Places (custos: confira a tabela atual do Google Maps Platform)
    google_enabled: bool = False
    google_price_per_1000_usd: float = 35.0
    google_max_requests_per_search: int = 20
    google_monthly_request_cap: int = 500
    google_max_pages: int = 1
    # OpenStreetMap (Overpass)
    osm_enabled: bool = True
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    # base aberta de CNPJ (índice local)
    cnpj_enabled: bool = True
    # mensagens por IA (opcional)
    ai_enabled: bool = False
    ai_model: str = "claude-opus-5-5"
    ai_daily_cap: int = 30
    # análise de sites
    site_check_max_per_batch: int = 25
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, base: Path | None = None) -> Settings:
        f = (base or data_dir()) / "settings.json"
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        defaults = cls()
        known = {fl.name for fl in fields(cls)}
        clean = {}
        for k, v in (raw.items() if isinstance(raw, dict) else ()):
            if k not in known:
                continue
            d = getattr(defaults, k)
            if isinstance(d, bool):
                ok = isinstance(v, bool)
            elif isinstance(d, float):
                ok = isinstance(v, (int, float)) and not isinstance(v, bool)
                v = float(v) if ok else v
            elif isinstance(d, int):
                ok = isinstance(v, int) and not isinstance(v, bool)
            else:
                ok = isinstance(v, type(d))
            if ok:
                clean[k] = v
        return cls(**clean)

    def save(self, base: Path | None = None) -> None:
        d = base or data_dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "settings.json.tmp"
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(d / "settings.json")


class Secrets:
    """Chaves de API. Ordem: variável de ambiente > arquivo local do usuário."""

    def __init__(self, base: Path | None = None, env: dict | None = None):
        self.base = base
        self.env = os.environ if env is None else env

    @property
    def path(self) -> Path:
        return (self.base or data_dir()) / "secrets.json"

    def _read(self) -> dict:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def get(self, name: str) -> str:
        env = SECRET_NAMES.get(name)
        if env and self.env.get(env):
            return self.env[env].strip()
        return str(self._read().get(name, "")).strip()

    def origin(self, name: str) -> str:
        """``"ambiente"``, ``"arquivo"`` ou ``""`` (para mostrar na interface sem revelar a chave)."""
        env = SECRET_NAMES.get(name)
        if env and self.env.get(env):
            return "ambiente"
        return "arquivo" if self._read().get(name) else ""

    def set(self, name: str, value: str) -> None:
        data = self._read()
        value = (value or "").strip()
        if value:
            data[name] = value
        else:
            data.pop(name, None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(self.path)


def mask(secret: str) -> str:
    """"AIza…9xQ" — para mostrar que existe uma chave sem expô-la."""
    s = secret or ""
    return "" if not s else (s[:4] + "…" + s[-3:] if len(s) > 10 else "•" * len(s))
