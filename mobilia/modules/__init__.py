"""Módulos opcionais da janela principal do 7xAi (abas de primeiro nível).

Cada subpacote daqui (ex.: ``mobilia/modules/lead_finder/``) registra a si mesmo ao ser importado, chamando
``mobilia.ui.modules.register(ModuleSpec(...))``. :func:`discover` importa todos os subpacotes; um subpacote que
falhar ao importar é registrado em ``mobilia.ui.modules.failed`` e ignorado, sem derrubar a janela.

Hoje não há nenhum módulo. Exemplo e regras em ``docs/PARALLEL_DEVELOPMENT.md``.
No executável, o PyInstaller inclui os subpacotes por ``collect_submodules("mobilia.modules")`` (packaging/7xai.spec).
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import traceback

log = logging.getLogger(__name__)

__all__ = ["discover"]


def discover() -> list[str]:
    """Importa cada subpacote/submódulo de ``mobilia.modules`` (nomes começando com ``_`` são ignorados).

    Devolve os nomes importados com sucesso. Erros ficam em ``mobilia.ui.modules.failed``.
    """
    from mobilia.ui import modules as registry

    ok: list[str] = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        name = f"{__name__}.{info.name}"
        try:
            importlib.import_module(name)
            ok.append(name)
        except Exception:  # noqa: BLE001 - módulo quebrado é ignorado, a janela abre mesmo assim
            registry.failed.append((name, traceback.format_exc()))
            log.exception("módulo %s não carregou; ignorado", name)
    return ok
