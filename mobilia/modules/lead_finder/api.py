"""Interface pública mínima para a integração com o AI Studio (usada pela sessão principal, quando quiser).

Exemplo — depois de gerar um vídeo, oferecer "Vincular a uma empresa do Lead Finder":

    from mobilia.modules.lead_finder import api
    escolhas = api.search_leads("alfa")              # [(id, "Alfa Imóveis — Campinas/SP"), ...]
    api.link_output(escolhas[0][0], "/caminho/antes_depois_9x16.mp4", kind="video")

As funções abrem e fecham o banco a cada chamada (SQLite). São rápidas, mas **não** devem rodar na thread da
interface: chame-as numa thread de trabalho, como o restante do Lead Finder. Nenhuma delas toca o motor de IA.
"""
from __future__ import annotations

from pathlib import Path

from .config import data_dir
from .storage import DB_NAME, Store


def _store() -> Store:
    return Store(data_dir() / DB_NAME)


def search_leads(text: str = "", limit: int = 20) -> list[tuple[int, str]]:
    """Empresas do CRM (exceto "Não contatar") que casam com ``text``."""
    s = _store()
    try:
        rows, _ = s.page({"text": text, "hide_dnc": True}, sort="name", desc=False, limit=limit)
        return [(r["id"], f"{r['name']} — {r['city']}/{r['uf']}".rstrip(" —/")) for r in rows]
    finally:
        s.close()


def link_output(lead_id: int, path: str | Path, kind: str = "projeto", title: str = "") -> None:
    """Vincula uma pasta de projeto ou um arquivo gerado (imagem/vídeo) a uma empresa. Registra no histórico."""
    if kind not in ("projeto", "video", "imagem"):
        raise ValueError("kind deve ser 'projeto', 'video' ou 'imagem'")
    s = _store()
    try:
        s.link_media(int(lead_id), str(path), kind, title)
    finally:
        s.close()


def links_for(lead_id: int) -> list[dict]:
    s = _store()
    try:
        d = s.get(int(lead_id))
        return d["media"] if d else []
    finally:
        s.close()
