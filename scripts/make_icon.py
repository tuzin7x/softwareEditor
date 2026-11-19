"""Gera o ícone do 7xAi (packaging/assets/icon.ico + icon.png + SVGs da marca) a partir dos vetores.

Uso (na raiz do repositório):
    python scripts/make_icon.py                # grava em packaging/assets
    python scripts/make_icon.py --check        # só confere se os arquivos versionados estão atualizados

Cada tamanho do .ico é desenhado do vetor no seu próprio tamanho óptico (mobilia/ui/brand.py): o emblema
hexagonal "7x" da identidade 7x Intelligence, completo (marcas de mira e faceta) a partir de 40 px e simplificado,
com traços mais grossos, abaixo disso. Nada é reduzido de uma imagem grande.
Também regrava a marca d'água do vídeo antes e depois (mobilia/video/res/marca_7xai.png).
O resultado é determinístico (mesmos vetores + mesma versão do Qt = mesmos bytes).
A janela usa a mesma arte em tempo de execução (theme.window_icon -> brand.app_icon_pixmap).
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QGuiApplication, QImage  # noqa: E402

from mobilia.ui import brand  # noqa: E402

ASSETS = ROOT / "packaging" / "assets"


def decode_ico_image(blob: bytes) -> QImage | None:
    """Uma imagem do .ico: PNG, ou BMP 32 bits (BITMAPINFOHEADER, altura dobrada, de baixo para cima)."""
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return QImage.fromData(blob)
    hsize, w, h2, planes, bits, comp = struct.unpack_from("<IiiHHI", blob, 0)
    h = h2 // 2
    if hsize != 40 or planes != 1 or bits != 32 or comp != 0 or w <= 0 or h2 != 2 * w:
        return None
    stride = w * 4
    xor = blob[hsize:hsize + stride * h]
    if len(xor) != stride * h or len(blob) < hsize + stride * h + ((w + 31) // 32) * 4 * h:
        return None
    top_down = b"".join(xor[y * stride:(y + 1) * stride] for y in range(h - 1, -1, -1))
    return QImage(top_down, w, h, stride, QImage.Format_ARGB32).copy()


def read_ico(path: Path) -> list[tuple[int, int, int]]:
    """Lê o diretório do .ico: [(largura, altura, bits)] — e confere que cada imagem decodifica."""
    data = path.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    if reserved != 0 or kind != 1 or count == 0:
        raise ValueError(f"{path}: não é um .ico válido")
    out = []
    for i in range(count):
        w, h, _, _, _, bits, size, off = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
        w, h = w or 256, h or 256
        img = decode_ico_image(data[off:off + size])
        if img is None or img.isNull() or (img.width(), img.height()) != (w, h):
            raise ValueError(f"{path}: imagem {w}x{h} corrompida")
        out.append((w, h, bits))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ASSETS)
    ap.add_argument("--check", action="store_true", help="falha se os arquivos versionados estiverem desatualizados")
    a = ap.parse_args(argv)
    _app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841
    if a.check:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = brand.export_assets(tmp)
            stale = [f.name for f in fresh if (a.out / f.name).read_bytes() != f.read_bytes()]
        if stale:
            print("desatualizados:", ", ".join(stale), "-> rode: python scripts/make_icon.py")
            return 1
        print("ícones atualizados")
        return 0
    for f in brand.export_assets(a.out):
        print(f.relative_to(ROOT) if f.is_relative_to(ROOT) else f)
    if a.out == ASSETS:
        f = brand.video_mark_png(ROOT / "mobilia" / "video" / "res" / "marca_7xai.png")
        print(f.relative_to(ROOT))
    sizes = read_ico(a.out / "icon.ico")
    print("icon.ico:", ", ".join(f"{w}x{h}" for w, h, _ in sizes))
    expected = [(s, s) for s in brand.ICO_SIZES]
    if [(w, h) for w, h, _ in sizes] != expected:
        print("ERRO: tamanhos inesperados no .ico")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
