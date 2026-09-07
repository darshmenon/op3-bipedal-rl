#!/usr/bin/env bash
# Stage 1: verify the Isaac Lab + Isaac Sim environment set up by setup_env.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
ISAACLAB_PATH="${PROJECT_ROOT}/IsaacLab"

if [[ ! -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
  echo "ERROR: ${ISAACLAB_PATH}/isaaclab.sh not found. Run scripts/setup_env.sh first." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Python: $(python --version)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
else
  echo "ERROR: nvidia-smi not found. This must run on a GPU machine." >&2
  exit 2
fi

echo "Checking torch + CUDA..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda.is_available() is false")
PY

cd "${ISAACLAB_PATH}"

echo "Checking isaacsim / isaaclab / pxr imports..."
./isaaclab.sh -p "${PROJECT_ROOT}/scripts/check_imports.py"

echo "Listing Isaac Lab built-in environments..."
./isaaclab.sh -p scripts/environments/list_envs.py

echo "Running headless empty-scene smoke test..."
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

echo "Stage 1 verification completed successfully."
