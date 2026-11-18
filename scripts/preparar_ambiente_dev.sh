#!/usr/bin/env bash
# Recria o ambiente de desenvolvimento Linux (CPU) usado neste projeto.
# Uso: bash scripts/preparar_ambiente_dev.sh
set -euo pipefail
VENV="${VENV:-$HOME/.venvs/staging}"
python3.13 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip wheel setuptools
cd "$(dirname "$0")/.."
# Mesmas versões fixas do motor (requirements/runtime-cpu.txt, com PyTorch CPU) + interface e pytest.
"$VENV/bin/python" -m pip install -q -r requirements/dev.txt
"$VENV/bin/python" -m pip install -q --no-deps -r requirements/runtime-nodeps.txt   # MoGe-2 (commit fixo)
export MOBILIA_HOME="${MOBILIA_HOME:-$HOME/mobilia_data}"
"$VENV/bin/python" -m mobilia.cli prefetch || true      # modelos de IA + assets 3D (uso offline)
QT_QPA_PLATFORM=offscreen "$VENV/bin/python" -m pytest -q -m "not slow and not models and not gpu" tests
echo "Ambiente pronto em $VENV (MOBILIA_HOME=$MOBILIA_HOME)"
