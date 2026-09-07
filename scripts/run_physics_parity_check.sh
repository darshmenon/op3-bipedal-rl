#!/usr/bin/env bash
# Run the full Isaac Lab <-> MuJoCo physics parity check: dump both sides'
# live config, then diff them. See scripts/compare_physics_parity.py for
# what's actually checked and why it exists.
#
# Usage:
#   scripts/run_physics_parity_check.sh [stage]
#
# stage defaults to walk_easy_rebalanced (the current main training stage).
# Exits nonzero if compare_physics_parity.py finds an unexpected mismatch.
#
# Do not run this while an Isaac Lab training job is active on the same
# GPU: a second concurrent Isaac Sim (Kit) process can crash on a
# circular-import race against the running one (seen firsthand -- not a
# GPU-memory limit, a shared extension-cache/import-order collision). Stop
# any active `train_ppo_resilient.sh`/`isaaclab.sh` process first.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-walk_easy_rebalanced}"

if pgrep -f "isaaclab.sh -p ../scripts/train_ppo.py" >/dev/null 2>&1; then
  echo "[run_physics_parity_check] ERROR: an Isaac Lab training job is already running." >&2
  echo "Stop it first -- a second concurrent Isaac Sim process can crash both on a shared-cache import race." >&2
  exit 3
fi

cd "${PROJECT_ROOT}"
echo "[run_physics_parity_check] dumping MuJoCo params..."
.venv/bin/python3 scripts/dump_mujoco_physics_params.py

echo "[run_physics_parity_check] dumping Isaac Lab params (stage=${STAGE}, ~15-20s Isaac Sim boot)..."
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM="xterm-256color"
fi
(cd IsaacLab && ./isaaclab.sh -p ../scripts/dump_isaac_physics_params.py --stage "${STAGE}")

echo "[run_physics_parity_check] comparing..."
.venv/bin/python3 scripts/compare_physics_parity.py
