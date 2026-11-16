"""Pacote de diagnóstico ("Copiar diagnóstico") e resumo dos tempos para a tela de conclusão.

O zip leva o que explica UMA execução — ``report.json`` (telemetria completa), ``frames.json``,
``scene.json``/``layout.json``, registros (``*.log``) e um manifesto — e, por padrão, NENHUMA
imagem: a foto do cliente e tudo o que é derivado dela (placa, prévias, sobreposições, EXRs,
MP4, geometria) só entram se o usuário marcar "incluir a foto".

Só biblioteca padrão: é importado pela interface congelada.
"""
from __future__ import annotations

import json
import platform
import zipfile
from datetime import datetime
from pathlib import Path

# Arquivos de texto (relativos à pasta do trabalho) que sempre entram, se existirem.
TEXT_FILES: tuple[str, ...] = (
    "report.json", "request.json", "layout.json", "scene.json", "render/frames.json",
    "analise/room.json", "analise/sol.json",
)
# Imagens derivadas da foto: só com ``include_photo``.
PHOTO_FILES: tuple[str, ...] = (
    "plate.png", "analise/analise_overlay.jpg", "layout_overlay.jpg", "calib_mask.png",
)
PHOTO_GLOBS: tuple[str, ...] = ("quadros/*.jpg",)
MAX_LOG_BYTES = 4 * 2**20          # registros muito longos: guarda só o final


def _add_text(zf: zipfile.ZipFile, arcname: str, path: Path) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if len(data) > MAX_LOG_BYTES and path.suffix == ".log":
        data = b"[...inicio cortado...]\n" + data[-MAX_LOG_BYTES:]
    zf.writestr(arcname, data)
    return True


def make_diagnostic_zip(work_dir: str | Path, dest: str | Path, include_photo: bool = False,
                        extra_log: str | None = None, app_version: str | None = None) -> Path:
    """Cria o zip de diagnóstico de um trabalho e devolve o caminho.

    ``extra_log``: texto do registro da interface (saída do motor), gravado como ``interface.log``.
    """
    work = Path(work_dir)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    included: list[str] = []
    tmp = dest.with_suffix(".partial.zip")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in TEXT_FILES:
            if _add_text(zf, rel, work / rel):
                included.append(rel)
        for p in sorted(work.rglob("*.log")):
            rel = p.relative_to(work).as_posix()
            if _add_text(zf, rel, p):
                included.append(rel)
        if extra_log:
            zf.writestr("interface.log", extra_log)
            included.append("interface.log")
        if include_photo:
            photo = None
            try:
                req = json.loads((work / "request.json").read_text(encoding="utf-8"))
                photo = Path(req.get("photo_path", ""))
            except (OSError, ValueError):
                pass
            if photo is not None and photo.is_file():
                zf.write(photo, "foto/" + photo.name)
                included.append("foto/" + photo.name)
            for rel in PHOTO_FILES:
                if (work / rel).is_file():
                    zf.write(work / rel, rel)
                    included.append(rel)
            for pattern in PHOTO_GLOBS:
                for p in sorted(work.glob(pattern)):
                    rel = p.relative_to(work).as_posix()
                    zf.write(p, rel)
                    included.append(rel)
        manifest = {
            "criado": datetime.now().isoformat(timespec="seconds"),
            "versao": app_version,
            "plataforma": platform.platform(),
            "trabalho": work.name,
            "inclui_foto": bool(include_photo),
            "arquivos": included,
        }
        zf.writestr("manifesto.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------- resumo para a interface

# (rótulo, etapas de ``tempos_s`` somadas)
SUMMARY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Início e foto", ("inicio_motor", "carga_foto")),
    ("Análise da foto", ("analise",)),
    ("Layout e móveis", ("layout", "ativos")),
    ("Download de modelos", ("download",)),
    ("Preparação do 3D", ("blender_inicio", "blender_kernels", "cena_montagem", "calibracao")),
    ("Render dos quadros", ("render_final", "render_marcos", "render_dinamico", "blender_fim")),
    ("Composição e MP4", ("composicao", "codificacao", "verificacao", "relatorio")),
)


def _total(v) -> float:
    if isinstance(v, dict):
        return float(sum(_total(x) for x in v.values()))
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def stage_summary(report: dict) -> list[tuple[str, float]]:
    """Linhas (rótulo, segundos) para a tela de conclusão; omite grupos abaixo de 0,05 s."""
    t = report.get("tempos_s") or {}
    rows = []
    for label, keys in SUMMARY_GROUPS:
        s = sum(_total(t.get(k)) for k in keys)
        if s >= 0.05:
            rows.append((label, s))
    return rows


def format_seconds(s: float) -> str:
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"
    m, r = divmod(int(round(s)), 60)
    return f"{m} min {r:02d} s"


def summary_text(report: dict) -> str:
    rows = stage_summary(report)
    if not rows:
        return ""
    t = report.get("tempos_s") or {}
    lines = [f"{label}: {format_seconds(s)}" for label, s in rows]
    total = t.get("total_percebido") or t.get("total")
    if total:
        lines.append(f"Total: {format_seconds(_total(total))}")
    if report.get("primeira_execucao"):
        lines.append("Primeira execução depois da instalação — as próximas são mais rápidas.")
    return "\n".join(lines)
