# -*- mode: python ; coding: utf-8 -*-
# PyInstaller (onedir, janela): gera dist\7xAi\7xAi.exe. O .exe contém SÓ a interface (PySide6, ~150 MB).
# O motor pesado (PyTorch CUDA, Blender bpy, modelos) é instalado no primeiro uso a partir do
# "payload" abaixo (código-fonte do pacote mobilia + requirements). Gerar com:
#   pyinstaller --noconfirm packaging/7xai.spec
# Obs.: o pacote Python continua "mobilia" e os dados continuam em %LOCALAPPDATA%\MobiliaStudio
# (o motor de vários GB já instalado é reaproveitado); só o nome visível e o .exe viraram 7xAi.
import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT / "packaging"))
from win_version import EXE_NAME, build_version_info  # noqa: E402

datas = [(str(ROOT / "mobilia"), "payload/mobilia"), (str(ROOT / "requirements"), "payload/requirements"),
         # interface 7xAi: fontes Inter (SIL OFL, com OFL.txt) e imagens da folha de estilo
         (str(ROOT / "mobilia" / "ui" / "fonts"), "mobilia/ui/fonts"),
         (str(ROOT / "mobilia" / "ui" / "res"), "mobilia/ui/res")]
excludes = ["torch", "torchvision", "numpy", "scipy", "cv2", "bpy", "transformers", "PIL", "moge",
            "OpenEXR", "shapely", "matplotlib", "tkinter",
            "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick3D",
            "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf"]

a = Analysis([str(ROOT / "packaging" / "launcher.py")], pathex=[str(ROOT)], datas=datas,
             hiddenimports=["mobilia.ui.app", "mobilia.ui.legacy", "mobilia.ui.design_widgets", "mobilia.ui.widgets",
                            "mobilia.ui.setup_dialog", "mobilia.bootstrap",
                            "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets"],
             excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=EXE_NAME, console=False,
          icon=str(ROOT / "packaging" / "assets" / "icon.ico"), version=build_version_info(ROOT), upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=EXE_NAME)
