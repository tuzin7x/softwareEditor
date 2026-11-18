"""Confere o conteúdo do executável gerado pelo PyInstaller (usado pelos workflows de build e de Release).

Lê o ``PYZ-00.toc`` da pasta de trabalho do PyInstaller (``--workpath``) e confirma que:

* todos os módulos de ``mobilia/modules/`` (ex.: o Lead Finder, com ``ui`` e ``sources``) estão no executável;
* o registro de módulos e a interface estão lá;
* nenhuma biblioteca pesada do motor entrou (o ``.exe`` leva só a interface).

Uso::

    python scripts/check_bundle.py build/7xai/PYZ-00.toc [build-portable/7xai/PYZ-00.toc ...]

Termina com código 1 e mensagens ``::error::`` quando algo falta ou sobra. Só biblioteca padrão.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CORE = ("mobilia.ui.app", "mobilia.ui.modules", "mobilia.modules")
FORBIDDEN_TOP = ("torch", "torchvision", "numpy", "scipy", "cv2", "PIL", "bpy", "transformers", "diffusers",
                 "accelerate", "huggingface_hub", "safetensors", "kornia", "imageio_ffmpeg")
_NAME = re.compile(r"\(\s*'([A-Za-z_][\w.]*)'\s*,")


def expected_modules(root: Path = ROOT) -> list[str]:
    """Nomes pontuados de todos os arquivos .py de ``mobilia/modules`` (o que ``collect_submodules`` deve levar)."""
    pkg = root / "mobilia" / "modules"
    names = []
    for f in sorted(pkg.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        rel = f.relative_to(root).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts.pop()
        names.append(".".join(parts))
    return names


def toc_modules(text: str) -> set[str]:
    return set(_NAME.findall(text))


def check(toc_text: str, root: Path = ROOT) -> tuple[list[str], list[str], int]:
    """Devolve (faltando, proibidos, quantos módulos de mobilia.modules estão no executável)."""
    mods = toc_modules(toc_text)
    expected = expected_modules(root)
    missing = [m for m in (*REQUIRED_CORE, *expected) if m not in mods]
    forbidden = sorted({m for m in mods if m.split(".")[0] in FORBIDDEN_TOP})
    present = sum(1 for m in expected if m in mods)
    return missing, forbidden, present


def main(argv: list[str] | None = None) -> int:
    paths = argv if argv is not None else sys.argv[1:]
    if not paths:
        print("uso: python scripts/check_bundle.py <PYZ-00.toc> [...]", file=sys.stderr)
        return 2
    ok = True
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"::error::Não consegui ler {p}: {e}", file=sys.stderr)
            ok = False
            continue
        missing, forbidden, present = check(text)
        for m in missing:
            print(f"::error::{p}: módulo ausente do executável: {m}", file=sys.stderr)
        for m in forbidden[:20]:
            print(f"::error::{p}: biblioteca pesada dentro do executável: {m}", file=sys.stderr)
        ok = ok and not missing and not forbidden
        print(f"{p}: {present} de {len(expected_modules())} arquivos de mobilia/modules no executável"
              f"{'' if not missing and not forbidden else ' (COM PROBLEMAS)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
