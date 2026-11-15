#!/usr/bin/env bash
# Recria o ambiente de desenvolvimento Linux (CPU) usado neste projeto.
# Uso: bash scripts/preparar_ambiente_dev.sh
set -euo pipefail
VENV="${VENV:-$HOME/.venvs/staging}"
python3.13 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip wheel setuptools
"$VENV/bin/python" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/python" -m pip install -q bpy opencv-python-headless transformers huggingface_hub scipy scikit-image \
  imageio imageio-ffmpeg shapely pytest OpenEXR pyflakes "git+https://github.com/microsoft/MoGe.git"
export MOBILIA_HOME="${MOBILIA_HOME:-$HOME/mobilia_data}"
cd "$(dirname "$0")/.."
"$VENV/bin/python" -m mobilia.cli prefetch || true      # modelos de IA + assets 3D (uso offline)
"$VENV/bin/python" -m pytest -q tests/test_compositor.py tests/test_choreography.py tests/test_room.py \
  tests/test_encoder.py tests/test_sunlight.py tests/test_depth_mesh.py tests/test_scene_spec.py
echo "Ambiente pronto em $VENV (MOBILIA_HOME=$MOBILIA_HOME)"
