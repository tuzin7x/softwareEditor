"""Validação da versão e notas da Release oficial do 7xAi (usado por .github/workflows/release.yml).

Fonte única da versão: ``__version__`` em ``mobilia/__init__.py`` (o pyproject, o .spec do PyInstaller e o
instalador leem dali). O CHANGELOG.md precisa ter a seção ``## [X.Y.Z]`` dessa versão, sem a marca RASCUNHO.

Uso::

    python scripts/release_notes.py check --version 1.0.0
    python scripts/release_notes.py notes --version 1.0.0 --assets dist/release --sha <commit> --out notas.md

``check`` termina com código 1 e uma mensagem clara quando algo não bate. Só biblioteca padrão.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")   # oficial: só X.Y.Z (sem -beta, sem +build)
DRAFT = re.compile(r"rascunho", re.I)
_HEADING = re.compile(r"^## \[(?P<name>[^\]]+)\](?P<rest>.*)$", re.M)
_SUB = re.compile(r"^### +(?P<title>.+?)\s*$", re.M)
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# categorias que toda nota de versão mostra (mesmo vazias), na ordem; cada uma junta estes títulos do CHANGELOG
CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Novidades", ("novidades", "adicionado", "alterado")),
    ("Correções", ("correções", "corrigido")),
    ("Melhorias de desempenho", ("desempenho", "melhorias de desempenho")),
    ("Notas importantes de atualização", ("notas importantes de atualização", "notas de atualização")),
)

# nomes oficiais dos arquivos publicados (o workflow renomeia o que o PyInstaller/Inno Setup geram)
def asset_names(version: str) -> dict[str, str]:
    return {"setup": f"7xAI-Setup-v{version}.exe", "portable": f"7xAI-Portable-v{version}.exe",
            "sums": "SHA256SUMS.txt"}


class ReleaseError(Exception):
    """Algo impede a publicação; a mensagem diz o que corrigir."""


# ------------------------------------------------------------------------------------------------- versão

def code_version(root: Path = ROOT) -> str:
    text = (root / "mobilia" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not m:
        raise ReleaseError("mobilia/__init__.py não declara __version__.")
    return m.group(1)


def check_pyproject(root: Path = ROOT) -> None:
    """O pyproject não pode ter uma segunda versão escrita à mão: ela vem de mobilia.__version__."""
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    project = text.split("[project]", 1)[1].split("\n[", 1)[0] if "[project]" in text else ""
    if re.search(r"^version\s*=", project, re.M):
        raise ReleaseError("pyproject.toml tem 'version = ...' fixo; use dynamic = [\"version\"] "
                           "(a versão vem de mobilia/__init__.py).")
    if not re.search(r'^dynamic\s*=\s*\[[^\]]*"version"', project, re.M) or \
            not re.search(r'version\s*=\s*\{\s*attr\s*=\s*"mobilia\.__version__"\s*\}', text):
        raise ReleaseError("pyproject.toml precisa de dynamic = [\"version\"] e "
                           "[tool.setuptools.dynamic] version = {attr = \"mobilia.__version__\"}.")


# ---------------------------------------------------------------------------------------------- changelog

@dataclass
class Section:
    version: str
    heading: str
    body: str
    date: str = ""
    subsections: dict[str, list[str]] = field(default_factory=dict)   # título -> linhas (sem linhas vazias nas pontas)

    @property
    def is_draft(self) -> bool:
        return bool(DRAFT.search(self.heading) or DRAFT.search(self.body))


def find_section(changelog: str, version: str) -> Section | None:
    heads = list(_HEADING.finditer(changelog))
    for i, h in enumerate(heads):
        if h.group("name").strip().lstrip("v") != version:
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(changelog)
        body = changelog[h.end():end]
        # links de referência do Keep a Changelog no fim do arquivo ("[1.0.0]: https://...") não são conteúdo
        body = re.sub(r"^\[[^\]]+\]:\s*\S+\s*$", "", body, flags=re.M).strip("\n")
        date = _DATE.search(h.group("rest"))
        sec = Section(version, h.group(0).strip(), body, date.group(1) if date else "")
        subs = list(_SUB.finditer(body))
        for j, s in enumerate(subs):
            s_end = subs[j + 1].start() if j + 1 < len(subs) else len(body)
            lines = body[s.end():s_end].strip("\n").splitlines()
            while lines and not lines[-1].strip():
                lines.pop()
            sec.subsections[s.group("title").strip()] = lines
        return sec
    return None


def check(version: str, root: Path = ROOT, changelog: Path | None = None) -> Section:
    """Tudo que precisa estar certo ANTES de gastar tempo com o build. Levanta ReleaseError."""
    if not SEMVER.match(version or ""):
        raise ReleaseError(f"Versão '{version}' inválida: use X.Y.Z (ex.: 1.0.0), sem 'v' e sem sufixos.")
    code = code_version(root)
    if code != version:
        raise ReleaseError(f"A versão pedida ({version}) não é a do código ({code}). Atualize __version__ em "
                           "mobilia/__init__.py num PR para a main antes de publicar.")
    check_pyproject(root)
    path = changelog or (root / "CHANGELOG.md")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ReleaseError(f"Não encontrei o {path.name}: {e}") from e
    sec = find_section(text, version)
    if sec is None:
        raise ReleaseError(f"O CHANGELOG.md não tem a seção '## [{version}]'.")
    if sec.is_draft:
        raise ReleaseError(f"A seção {version} do CHANGELOG.md ainda está marcada como RASCUNHO. Revise o texto, "
                           "tire a marca e faça o merge na main antes de publicar.")
    if not re.search(r"^\s*[-*] +\S", sec.body, re.M):
        raise ReleaseError(f"A seção {version} do CHANGELOG.md está vazia (nenhum item '- ...').")
    return sec


# ------------------------------------------------------------------------------------------------- notas

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sums_text(files: list[Path]) -> str:
    """Formato do ``sha256sum`` (``<hash>  <nome>``): confere com ``sha256sum -c SHA256SUMS.txt``."""
    return "".join(f"{sha256_file(p)}  {p.name}\n" for p in sorted(files, key=lambda p: p.name))


def _size(n: int) -> str:
    return f"{n / (1 << 20):.1f} MB"


def render_notes(sec: Section, version: str, assets: list[Path], sha: str, date: str, repo: str = "") -> str:
    names = asset_names(version)
    by_name = {p.name: p for p in assets}
    missing = [n for k, n in names.items() if k != "sums" and n not in by_name]
    if missing:
        raise ReleaseError(f"Arquivos ausentes para a Release: {', '.join(missing)}")
    commit = f"[`{sha[:12]}`](https://github.com/{repo}/commit/{sha})" if repo else f"`{sha[:12]}`"
    out = [f"# 7xAi v{version}", "", f"**Data:** {date} · **Versão:** {version} · **Commit:** {commit}", ""]

    out += ["## Downloads", "", "| Arquivo | Para quem | Tamanho |", "|---|---|---|"]
    desc = {names["setup"]: "Instalador (recomendado): instala, cria o atalho na Área de Trabalho e atualiza "
                             "versões anteriores",
            names["portable"]: "Portátil: um único arquivo, roda sem instalar (abre um pouco mais devagar)"}
    for n in (names["setup"], names["portable"]):
        out.append(f"| `{n}` | {desc[n]} | {_size(by_name[n].stat().st_size)} |")
    out += ["", "**Como instalar:** baixe o instalador, abra e siga as telas (não precisa de Python nem de "
                "administrador). No primeiro uso o 7xAi baixa o motor de IA (alguns GB, uma única vez). Requisitos: "
                "Windows 10/11 64 bits; placa NVIDIA com 8 GB recomendada.",
            "", "Se o Windows mostrar \"O Windows protegeu o computador\" (o programa ainda não tem assinatura "
                "digital paga), clique em **Mais informações → Executar assim mesmo**.", ""]

    used: set[str] = set()
    for title, keys in CATEGORIES:
        lines: list[str] = []
        for sub, sub_lines in sec.subsections.items():
            if sub.casefold() in keys:
                used.add(sub)
                lines += [ln for ln in sub_lines if ln.strip()]
        out += [f"## {title}", ""] + (lines or ["- Nada nesta versão."]) + [""]
    for sub, sub_lines in sec.subsections.items():           # outras seções (Removido, Segurança...) também saem
        if sub not in used and any(ln.strip() for ln in sub_lines):
            out += [f"## {sub}", ""] + [ln for ln in sub_lines if ln.strip()] + [""]

    out += ["## Verificação de integridade (SHA-256)", "", "| Arquivo | SHA-256 |", "|---|---|"]
    for p in sorted((by_name[n] for n in (names["setup"], names["portable"])), key=lambda p: p.name):
        out.append(f"| `{p.name}` | `{sha256_file(p)}` |")
    out += ["", f"A mesma lista está em `{names['sums']}`. Para conferir no Windows (PowerShell):",
            "", "```powershell", f"Get-FileHash .\\{names['setup']} -Algorithm SHA256", "```", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="valida versão, pyproject e CHANGELOG")
    c.add_argument("--version", required=True)
    n = sub.add_parser("notes", help="gera o texto da Release")
    n.add_argument("--version", required=True)
    n.add_argument("--assets", required=True, type=Path, help="pasta com os arquivos da Release")
    n.add_argument("--sha", required=True)
    n.add_argument("--repo", default="")
    n.add_argument("--date", default="", help="AAAA-MM-DD (padrão: hoje, UTC)")
    n.add_argument("--out", type=Path)
    s = sub.add_parser("sums", help="escreve SHA256SUMS.txt dos arquivos da pasta")
    s.add_argument("--assets", required=True, type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            sec = check(a.version)
            print(f"ok: versão {a.version} = código; CHANGELOG '{sec.heading}'")
        elif a.cmd == "notes":
            sec = check(a.version)
            files = sorted(p for p in a.assets.iterdir() if p.is_file())
            date = a.date or _dt.datetime.now(_dt.timezone.utc).date().isoformat()
            text = render_notes(sec, a.version, files, a.sha, date, a.repo)
            if a.out:
                a.out.write_text(text, encoding="utf-8")
            else:
                sys.stdout.write(text)
        else:
            files = sorted(p for p in a.assets.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
            if not files:
                raise ReleaseError(f"Nenhum arquivo em {a.assets}")
            (a.assets / "SHA256SUMS.txt").write_text(sums_text(files), encoding="utf-8", newline="\n")
            print((a.assets / "SHA256SUMS.txt").read_text(encoding="utf-8"), end="")
    except ReleaseError as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
