"""Ponte com o AI Studio (o fluxo de redesenho do 7xAi), só leitura.

O AI Studio grava cada geração numa pasta ``<dados do app>/jobs/design_<data>_<função>/`` com as versões
(``*.jpg``), a foto original (``antes.jpg``), o relatório ``design.json`` e, quando gerado, o vídeo
``antes_depois_<formato>.mp4``. Este módulo apenas **lista** essas pastas para que a pessoa vincule um projeto ou
vídeo a uma empresa do CRM; nada é alterado, movido ou enviado.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

FUNCTION_TITLES = {
    "mudar_estilo": "Mudar estilo", "reforma": "Reforma completa", "redesign_simples": "Redesign simples",
    "mobiliar": "Mobiliar vazio", "editar_area": "Editar área", "fachada": "Fachada", "evento": "Evento",
    "texto": "Texto para ambiente",
}


@dataclass
class StudioProject:
    path: str
    title: str
    created: str
    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    before: str = ""

    @property
    def thumb(self) -> str:
        return self.images[0] if self.images else self.before


def jobs_dir() -> Path:
    from mobilia import runtime
    return runtime.runtime_paths().jobs_dir


def read_project(folder: Path) -> StudioProject | None:
    folder = Path(folder)
    report = folder / "design.json"
    if not report.is_file():
        return None
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    outs = [str(Path(o)) for o in (data.get("saidas") or []) if isinstance(o, str)]
    images = [o for o in outs if Path(o).is_file()] or \
        [str(p) for p in sorted(folder.glob("*.jpg")) if p.name != "antes.jpg" and "_4k" not in p.stem]
    videos = [str(p) for p in sorted(folder.glob("*.mp4"))]
    func = FUNCTION_TITLES.get(str(data.get("funcao", "")), str(data.get("funcao", "")) or "Projeto")
    style = str(data.get("estilo") or "").replace("_", " ")
    room = str(data.get("ambiente") or "")
    title = " · ".join(x for x in (func, style.title() if style else "", room.title() if room else "") if x)
    created = datetime.fromtimestamp(report.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
    before = str(folder / "antes.jpg") if (folder / "antes.jpg").is_file() else ""
    return StudioProject(str(folder), title, created, images, videos, before)


def list_projects(base: Path | None = None, limit: int = 200) -> list[StudioProject]:
    """Projetos do AI Studio, do mais recente para o mais antigo."""
    base = Path(base) if base is not None else jobs_dir()
    if not base.is_dir():
        return []
    dirs = sorted((d for d in base.glob("design_*") if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    out = []
    for d in dirs[:limit]:
        p = read_project(d)
        if p is not None:
            out.append(p)
    return out
