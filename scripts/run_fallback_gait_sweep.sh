#!/usr/bin/env bash
set -euo pipefail

# Run a small, bounded sweep of the non-RL feedback CPG fallback controller.
# Each trial writes JSON metrics from fallback_gait_controller.py so the best
# parameter set can be picked by fall-step/alive-at-end numbers, not eyeballing.

OUT_DIR="${1:-outputs/diagnostics/fallback_gait_sweep}"
NUM_ENVS="${NUM_ENVS:-32}"
MAX_STEPS="${MAX_STEPS:-250}"

mkdir -p "${OUT_DIR}"

run_trial() {
  local name="$1"
  shift
  local out_json="${OUT_DIR}/${name}.json"
  echo "[fallback-sweep] trial=${name} out=${out_json} args=$*"
  IsaacLab/isaaclab.sh -p scripts/fallback_gait_controller.py \
    --num_envs "${NUM_ENVS}" \
    --max_steps "${MAX_STEPS}" \
    --out_json "${out_json}" \
    "$@"
}

# Baseline and lateral-sign/amount checks.
run_trial "baseline_f06_lat008_pos" \
  --step_freq_hz 0.6 --hip_amp 0.30 --knee_amp 0.35 \
  --roll_kp 1.5 --pitch_kp 1.0 --lateral_shift_amp 0.08 --lateral_shift_sign 1
run_trial "baseline_f06_lat008_neg" \
  --step_freq_hz 0.6 --hip_amp 0.30 --knee_amp 0.35 \
  --roll_kp 1.5 --pitch_kp 1.0 --lateral_shift_amp 0.08 --lateral_shift_sign -1
run_trial "f06_lat004_pos" \
  --step_freq_hz 0.6 --hip_amp 0.30 --knee_amp 0.35 \
  --roll_kp 1.5 --pitch_kp 1.0 --lateral_shift_amp 0.04 --lateral_shift_sign 1
run_trial "f06_lat012_pos" \
  --step_freq_hz 0.6 --hip_amp 0.30 --knee_amp 0.35 \
  --roll_kp 1.5 --pitch_kp 1.0 --lateral_shift_amp 0.12 --lateral_shift_sign 1

# Slower and gentler gait checks, since forced residual CPG variants failed
# at both 0.6 Hz and 0.3 Hz under PPO but the open-loop controller may prefer
# a different speed/amplitude tradeoff.
run_trial "f045_lat008_pos" \
  --step_freq_hz 0.45 --hip_amp 0.30 --knee_amp 0.35 \
  --roll_kp 1.5 --pitch_kp 1.0 --lateral_shift_amp 0.08 --lateral_shift_sign 1
run_trial "f03_lat008_pos" \
  --step_freq_hz 0.3 --hip_amp 0.30 --knee_amp 0.35 \
  --roll_kp 1.5 --pitch_kp 1.0 --lateral_shift_amp 0.08 --lateral_shift_sign 1
run_trial "f045_gentle_lat008_pos" \
  --step_freq_hz 0.45 --hip_amp 0.20 --knee_amp 0.25 \
  --roll_kp 1.5 --pitch_kp 1.0 --lateral_shift_amp 0.08 --lateral_shift_sign 1
run_trial "f03_gentle_lat008_pos" \
  --step_freq_hz 0.3 --hip_amp 0.20 --knee_amp 0.25 \
  --roll_kp 1.5 --pitch_kp 1.0 --lateral_shift_amp 0.08 --lateral_shift_sign 1

# Feedback-gain sensitivity.
run_trial "f045_gentle_stronger_feedback" \
  --step_freq_hz 0.45 --hip_amp 0.20 --knee_amp 0.25 \
  --roll_kp 2.0 --pitch_kp 1.5 --lateral_shift_amp 0.08 --lateral_shift_sign 1
run_trial "f045_gentle_weaker_feedback" \
  --step_freq_hz 0.45 --hip_amp 0.20 --knee_amp 0.25 \
  --roll_kp 1.0 --pitch_kp 0.7 --lateral_shift_amp 0.08 --lateral_shift_sign 1

echo "[fallback-sweep] complete. JSON files in ${OUT_DIR}"
