#!/usr/bin/env bash
# Run named replay benchmarks from benchmarks/op3_checkpoint_benchmarks.json.
#
# Usage:
#   scripts/run_checkpoint_benchmarks.sh [case_name ...]
#
# With no case names, runs every case in the config. Each case writes a JSON
# report under outputs/benchmarks/.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/benchmarks/op3_checkpoint_benchmarks.json"
OUT_DIR="${PROJECT_ROOT}/outputs/benchmarks"

mkdir -p "${OUT_DIR}"

cd "${PROJECT_ROOT}"
python3 - "$CONFIG" "$OUT_DIR" "$@" <<'PY'
import json
import shlex
import subprocess
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
selected = set(sys.argv[3:])

config = json.loads(config_path.read_text())
defaults = config.get("defaults", {})
cases = config.get("cases", [])
if selected:
    cases = [case for case in cases if case["name"] in selected]
    missing = selected - {case["name"] for case in cases}
    if missing:
        raise SystemExit(f"Unknown benchmark case(s): {', '.join(sorted(missing))}")

if not cases:
    raise SystemExit("No benchmark cases selected.")

project_root = config_path.parent.parent
isaaclab = project_root / "IsaacLab" / "isaaclab.sh"
evaluator = project_root / "scripts" / "evaluate_checkpoint_performance.py"

for case in cases:
    merged = {**defaults, **case}
    thresholds = merged.get("thresholds", {})
    out_file = out_dir / f"{merged['name']}.json"
    cmd = [
        str(isaaclab),
        "-p",
        str(evaluator),
        "--checkpoint",
        str(project_root / merged["checkpoint"]),
        "--stage",
        merged["stage"],
        "--num_envs",
        str(merged["num_envs"]),
        "--max_steps",
        str(merged["max_steps"]),
        "--out",
        str(out_file),
    ]
    if merged.get("zero_command"):
        cmd.append("--zero_command")
    if merged.get("stochastic"):
        cmd.append("--stochastic")
    for key, value in thresholds.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    print(f"[benchmark] running {merged['name']}: {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

print("[benchmark] all selected benchmark cases passed.")
PY
