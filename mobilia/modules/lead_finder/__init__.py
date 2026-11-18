"""7xAi Lead Finder: pesquisa de imobiliárias em fontes autorizadas, contatos comerciais e CRM local.

Este ``__init__`` só registra o módulo; nada pesado é importado aqui. A página (PySide6) é criada na primeira
abertura da aba, pela ``factory`` preguiçosa. Documentação: ``docs/LEAD_FINDER.md``.
"""
from mobilia.ui.modules import ModuleSpec, register

MODULE_ID = "lead_finder"
VERSION = "0.1.0"


def _page(window):
    from .ui.page import LeadFinderPage   # import adiado: só carrega ao abrir a aba
    return LeadFinderPage(window)


register(ModuleSpec(id=MODULE_ID, title="Lead Finder", factory=_page, order=10,
                    tooltip="Encontrar imobiliárias e organizar contatos comerciais (usa a internet)",
                    local=False))   # consulta fontes on-line: o selo "Processado no seu computador" some nesta aba
