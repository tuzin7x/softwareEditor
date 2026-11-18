# -*- mode: python ; coding: utf-8 -*-
# PyInstaller (janela). Padrão onedir: gera dist\7xAi\7xAi.exe (é o que o instalador empacota).
# Com a variável SEVENXAI_ONEFILE=1 gera o PORTÁTIL de arquivo único dist\7xAi.exe (mesmo conteúdo, extraído numa
# pasta temporária a cada abertura); use outra --distpath/--workpath para não misturar com o onedir.
# O .exe contém SÓ a interface (PySide6, ~150 MB).
# O motor pesado (PyTorch CUDA, Blender bpy, modelos) é instalado no primeiro uso a partir do
# "payload" abaixo (código-fonte do pacote mobilia + requirements). Gerar com:
#   pyinstaller --noconfirm packaging/7xai.spec
# Obs.: o pacote Python continua "mobilia" e os dados continuam em %LOCALAPPDATA%\MobiliaStudio
# (o motor de vários GB já instalado é reaproveitado); só o nome visível e o .exe viraram 7xAi.
import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT / "packaging"))
sys.path.insert(0, str(ROOT))  # collect_submodules("mobilia.modules") importa o pacote
from win_version import EXE_NAME, build_version_info  # noqa: E402
from PyInstaller.utils.hooks import collect_submodules  # noqa: E402

datas = [(str(ROOT / "mobilia"), "payload/mobilia"), (str(ROOT / "requirements"), "payload/requirements"),
         # interface 7xAi: fontes Exo 2, IBM Plex Sans e JetBrains Mono (SIL OFL, com OFL-*.txt) e imagens da folha de estilo
         (str(ROOT / "mobilia" / "ui" / "fonts"), "mobilia/ui/fonts"),
         (str(ROOT / "mobilia" / "ui" / "res"), "mobilia/ui/res")]
# Bibliotecas do motor nunca entram no .exe (o motor roda no Python embutido instalado no primeiro uso). Num ambiente
# de build limpo (CI, .venv-build) elas nem existem; a lista protege o build feito num venv de desenvolvimento.
# scripts/check_bundle.py confere isso no CI.
excludes = ["torch", "torchvision", "numpy", "scipy", "cv2", "bpy", "transformers", "PIL", "moge",
            "OpenEXR", "shapely", "matplotlib", "tkinter",
            "diffusers", "accelerate", "huggingface_hub", "safetensors", "tokenizers", "kornia", "sympy", "imageio_ffmpeg",
            "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick3D",
            "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf"]

a = Analysis([str(ROOT / "packaging" / "launcher.py")], pathex=[str(ROOT)], datas=datas,
             hiddenimports=["mobilia.ui.app", "mobilia.ui.legacy", "mobilia.ui.design_widgets", "mobilia.ui.widgets",
                            "mobilia.ui.setup_dialog", "mobilia.bootstrap", "mobilia.ui.modules", "mobilia.modules",
                            "mobilia.ui.updater", "mobilia.ui.updater_dialog",  # atualização automática
                            *collect_submodules("mobilia.modules"),  # módulos opcionais (abas)
                            "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets"],
             excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
ICON = str(ROOT / "packaging" / "assets" / "icon.ico")
if os.environ.get("SEVENXAI_ONEFILE") == "1":
    # portátil: um único .exe com tudo dentro (binários e dados entram no próprio EXE, sem COLLECT)
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name=EXE_NAME, console=False, icon=ICON,
              version=build_version_info(ROOT), upx=False, strip=False, runtime_tmpdir=None)
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=EXE_NAME, console=False, icon=ICON,
              version=build_version_info(ROOT), upx=False)
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=EXE_NAME)
