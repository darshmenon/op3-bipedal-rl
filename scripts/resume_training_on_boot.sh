#!/usr/bin/env bash
# Run via a crontab @reboot entry so a full system crash/reboot doesn't
# strand the active Isaac Lab training run -- the scripts/train_ppo_resilient.sh
# watchdog only recovers from a process crash while the machine stays up; it
# does not itself survive a reboot. This just relaunches the same watchdog,
# which then resumes from whatever checkpoint was last saved for that
# experiment.
#
# The active experiment used to be hardcoded here, which silently stranded a
# deliberately-isolated fresh run (op3_walk_easy_rebalanced_contact_fix) at
# iteration 50 on 2026-09-03: it crashed (Xid 154), the machine got
# rebooted, and this script relaunched the *previous*, already-plateaued
# experiment instead -- losing the actual experiment for over 2 hours
# across two more crash/reboot cycles before anyone noticed. See
# STAGE4_PPO_TRAINING.md.
#
# Then a second hardcoded-launch bug appeared: the state file named
# op3_walk_tiny_forward_contactauthored, but this script still relaunched it
# as walk_easy_rebalanced with no observation normalization. The state file now
# stores the complete train_ppo_resilient.sh argument set, not just a name.
PROJECT_ROOT="/home/asimov/robotis_op3_bipedal_candidate_task"
ACTIVE_EXPERIMENT_FILE="${PROJECT_ROOT}/logs/.active_isaac_experiment"
if [[ ! -s "${ACTIVE_EXPERIMENT_FILE}" ]]; then
  echo "[resume_training_on_boot] ${ACTIVE_EXPERIMENT_FILE} missing or empty, nothing to resume." >&2
  exit 1
fi
set -uo pipefail

if grep -q "=" "${ACTIVE_EXPERIMENT_FILE}"; then
  # shellcheck disable=SC1090
  source "${ACTIVE_EXPERIMENT_FILE}"
else
  # Backward-compatible fallback for the old one-line file format. This keeps
  # old state files from failing mysteriously, but new active experiments should
  # always use the key/value format below.
  EXPERIMENT_NAME="$(cat "${ACTIVE_EXPERIMENT_FILE}")"
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
NUM_ENVS="${NUM_ENVS:-512}"
MAX_ITERATIONS="${MAX_ITERATIONS:-71000}"
MAX_RETRIES="${MAX_RETRIES:-50}"
STAGE="${STAGE:-walk_easy_rebalanced}"
SEED_CHECKPOINT="${SEED_CHECKPOINT:-}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
DESIRED_KL="${DESIRED_KL:-}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-}"
DOF_TORQUES_WEIGHT="${DOF_TORQUES_WEIGHT:--1.0e-4}"
RESET_CRITIC="${RESET_CRITIC:-}"
RESET_OPTIMIZER="${RESET_OPTIMIZER:-}"
OBS_NORMALIZATION="${OBS_NORMALIZATION:-}"

if [[ -z "${EXPERIMENT_NAME}" ]]; then
  echo "[resume_training_on_boot] EXPERIMENT_NAME missing from ${ACTIVE_EXPERIMENT_FILE}." >&2
  exit 1
fi

if pgrep -f "train_ppo_resilient.sh ${EXPERIMENT_NAME}" >/dev/null; then
  echo "[resume_training_on_boot] ${EXPERIMENT_NAME} watchdog already running, nothing to do."
  exit 0
fi

cd "${PROJECT_ROOT}"
# A cron @reboot shell has no venv activated, so isaaclab.sh's python lookup
# (CONDA_PREFIX / VIRTUAL_ENV / _isaac_sim/python.sh) falls through to a
# fallback path that doesn't exist here -- every attempt then fails in <1s
# with "Unable to find any Python executable", which train_ppo_resilient.sh
# can't tell apart from a real crash and just retries forever.
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate"
nohup ./scripts/train_ppo_resilient.sh \
  "${EXPERIMENT_NAME}" "${NUM_ENVS}" "${MAX_ITERATIONS}" "${MAX_RETRIES}" \
  "${STAGE}" "${SEED_CHECKPOINT}" "${LEARNING_RATE}" "${DESIRED_KL}" \
  "${NUM_STEPS_PER_ENV}" "${DOF_TORQUES_WEIGHT}" "${RESET_CRITIC}" \
  "${RESET_OPTIMIZER}" "${OBS_NORMALIZATION}" \
  >> "logs/${EXPERIMENT_NAME}_resilient.log" 2>&1 &
disown
echo "[resume_training_on_boot] relaunched ${EXPERIMENT_NAME} (${STAGE}) watchdog after reboot (pid $!)."
