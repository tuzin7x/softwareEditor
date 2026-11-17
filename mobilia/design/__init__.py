"""Redesenho de ambientes por IA (fluxo principal do 7xAi): a mesma sala, com móveis, materiais e estilo novos.

``presets`` (funções, estilos, ambientes, eventos e prompts), ``engine`` (pipeline SDXL + ControlNet em cache),
``archlock`` (janelas e portas originais por cima) e ``api`` (``redesign``, ``edit_area``, ``text_to_design``).
"""
from .api import DesignResult, edit_area, redesign, text_to_design
from .presets import catalog

__all__ = ["redesign", "edit_area", "text_to_design", "DesignResult", "catalog"]
