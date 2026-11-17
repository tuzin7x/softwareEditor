"""Exportação CSV respeitando licenças: só campos de fontes que permitem reutilização, com atribuição."""
from __future__ import annotations

import csv
from pathlib import Path

from .models import now_iso


def write_csv(path: Path | str, rows: list[dict], columns: tuple[str, ...], attributions: list[str]) -> Path:
    """CSV em UTF-8 com BOM (abre direto no Excel) e separador ``;`` (padrão brasileiro)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), delimiter=";", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _safe(r.get(k, "")) for k in columns})
    tmp.replace(path)
    if attributions:
        note = path.with_name(path.stem + "_fontes.txt")
        note.write_text("Exportado pelo 7xAi Lead Finder em " + now_iso() + "\n\nFontes e atribuições:\n- "
                        + "\n- ".join(attributions)
                        + "\n\nDados obtidos do Google Maps não são exportados, conforme os termos do Google.\n"
                        "Use estes contatos comerciais apenas para fins legítimos e respeite pedidos de não contato "
                        "(LGPD).\n", encoding="utf-8")
    return path


def _safe(v) -> str:
    """Evita injeção de fórmulas em planilhas (=, +, -, @ no início)."""
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@") and not s[1:2].isdigit() else s
