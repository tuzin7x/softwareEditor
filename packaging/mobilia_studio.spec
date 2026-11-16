# -*- mode: python ; coding: utf-8 -*-
# PyInstaller (onedir, janela). O .exe contém SÓ a interface (PySide6, ~150 MB).
# O motor pesado (PyTorch CUDA, Blender bpy, modelos) é instalado no primeiro uso a partir do
# "payload" abaixo (código-fonte do pacote mobilia + requirements). Gerar com:
#   pyinstaller --noconfirm packaging/mobilia_studio.spec
from pathlib import Path

ROOT = Path(SPECPATH).parent
datas = [(str(ROOT / "mobilia"), "payload/mobilia"), (str(ROOT / "requirements"), "payload/requirements")]
excludes = ["torch", "torchvision", "numpy", "scipy", "cv2", "bpy", "transformers", "PIL", "moge",
            "OpenEXR", "shapely", "matplotlib", "tkinter",
            "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick3D",
            "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf"]

a = Analysis([str(ROOT / "packaging" / "launcher.py")], pathex=[str(ROOT)], datas=datas,
             hiddenimports=["mobilia.ui.app", "mobilia.ui.setup_dialog", "mobilia.bootstrap",
                            "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets"],
             excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MobiliaStudio", console=False,
          icon=str(ROOT / "packaging" / "assets" / "icon.ico"), upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="MobiliaStudio")
