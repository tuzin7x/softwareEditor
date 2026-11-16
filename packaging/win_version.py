"""Recurso de versão do Windows para o 7xAi.exe (o Explorer mostra o nome em Propriedades > Detalhes).

Usado por packaging/7xai.spec; os testes também o importam (tests/test_packaging.py).
"""
from __future__ import annotations

import re
from pathlib import Path

APP_NAME = "7xAi"
EXE_NAME = "7xAi"          # dist\7xAi\7xAi.exe


def app_version(root: Path) -> str:
    """``mobilia.__version__`` lido do código-fonte (sem importar o pacote)."""
    text = (Path(root) / "mobilia" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    return m.group(1) if m else "0.0.0"


def version_tuple(version: str) -> tuple[int, int, int, int]:
    nums = [int(n) for n in re.findall(r"\d+", version)][:4]
    return tuple(nums + [0] * (4 - len(nums)))  # type: ignore[return-value]


def build_version_info(root: Path):
    """VSVersionInfo com CompanyName / FileDescription / ProductName = 7xAi (pt-BR, Unicode)."""
    from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                     VarFileInfo, VarStruct, VSVersionInfo)

    ver = app_version(root)
    nums = version_tuple(ver)
    strings = [
        StringStruct("CompanyName", APP_NAME),
        StringStruct("FileDescription", APP_NAME),
        StringStruct("FileVersion", ver),
        StringStruct("InternalName", EXE_NAME),
        StringStruct("LegalCopyright", APP_NAME),
        StringStruct("OriginalFilename", f"{EXE_NAME}.exe"),
        StringStruct("ProductName", APP_NAME),
        StringStruct("ProductVersion", ver),
    ]
    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=nums, prodvers=nums, mask=0x3F, flags=0x0, OS=0x40004, fileType=0x1,
                          subtype=0x0, date=(0, 0)),
        kids=[StringFileInfo([StringTable("041604B0", strings)]),     # 0x0416 = português (Brasil)
              VarFileInfo([VarStruct("Translation", [0x0416, 1200])])])
