#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
LEROBOT_EXTRA="${LEROBOT_EXTRA:-libero}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
python -m pip install "lerobot[${LEROBOT_EXTRA}]"

if [[ -n "${LIBERO_ROOT:-}" && -d "${LIBERO_ROOT}/libero" ]]; then
  python -m pip install -e "${LIBERO_ROOT}/libero"
elif [[ -d ../LIBERO/libero ]]; then
  python -m pip install -e ../LIBERO/libero
fi

python -m pip install -e .

echo "Bootstrap complete."
echo "Activate with: source ${VENV_DIR}/bin/activate"
