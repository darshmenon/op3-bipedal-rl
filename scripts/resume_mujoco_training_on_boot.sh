#!/usr/bin/env bash
# Run via a crontab @reboot entry so a reboot doesn't strand the MuJoCo
# training run the way it stranded op3_mujoco_walk_easy_rebalanced on
# 2026-09-03 (no auto-resume existed; had to be relaunched by hand).
# Unlike the Isaac Sim watchdog, this run is plain CPU Python (no PhysX/CUDA),
# so it doesn't crash-loop on its own -- the only thing a reboot breaks is
# that nothing relaunches it, hence this being a plain one-shot relaunch
# rather than a retry loop like train_ppo_resilient.sh.
#
# EXPERIMENT_NAME/args used to be hardcoded here, which would have silently
# stranded op3_mujoco_walk_symmetric_fresh (obs-normalization + asymmetric
# velocity-range run) at iteration ~8000 on the next reboot and resumed the
# older op3_mujoco_walk_easy_rebalanced config instead -- the same class of
# bug documented for Isaac Lab in STAGE4_PPO_TRAINING.md. Now reads the
# active experiment from a state file that must be updated by hand every
# time a new MuJoCo experiment is launched.
#
# 2026-09-07: this used to source exactly one file
# (logs/.active_mujoco_experiment) and relaunch exactly one experiment. Once
# a second MuJoCo experiment (op3_mujoco_walk_entropy_unstick) started
# running concurrently with the first, that one-file assumption silently
# stranded it on the very next reboot -- the file simply wasn't updated in
# time, and there was nowhere for a second experiment to be listed even if it
# had been. Now loops over every logs/.active_mujoco_experiment* file, so any
# number of concurrent experiments can each get their own state file and all
# resume after a reboot.
PROJECT_ROOT="/home/asimov/robotis_op3_bipedal_candidate_task"
shopt -s nullglob
ACTIVE_EXPERIMENT_FILES=("${PROJECT_ROOT}"/logs/.active_mujoco_experiment*)
shopt -u nullglob
if [[ ${#ACTIVE_EXPERIMENT_FILES[@]} -eq 0 ]]; then
  echo "[resume_mujoco_training_on_boot] no logs/.active_mujoco_experiment* files, nothing to resume." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate"

for ACTIVE_EXPERIMENT_FILE in "${ACTIVE_EXPERIMENT_FILES[@]}"; do
  # Each file sourced in its own subshell so EXPERIMENT_NAME/MAX_ITERATIONS/
  # MUJOCO_EXTRA_ARGS from one file can never leak into the next.
  (
    # shellcheck disable=SC1090
    source "${ACTIVE_EXPERIMENT_FILE}"
    set -uo pipefail

    if pgrep -f "train_ppo_mujoco.py.*${EXPERIMENT_NAME}" >/dev/null; then
      echo "[resume_mujoco_training_on_boot] ${EXPERIMENT_NAME} already running, nothing to do."
      exit 0
    fi

    LOG_DIR="${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}"
    CKPT="$(ls -v "${LOG_DIR}"/model_*.pt 2>/dev/null | tail -n 1)"
    CKPT_ARGS=()
    [[ -n "${CKPT}" ]] && CKPT_ARGS=(--checkpoint "${CKPT}")

    nohup python3 scripts/train_ppo_mujoco.py --num_envs 64 --max_iterations "${MAX_ITERATIONS}" \
      --experiment_name "${EXPERIMENT_NAME}" "${MUJOCO_EXTRA_ARGS[@]}" "${CKPT_ARGS[@]}" \
      >> "logs/${EXPERIMENT_NAME}_train.log" 2>&1 &
    disown
    echo "[resume_mujoco_training_on_boot] relaunched ${EXPERIMENT_NAME} after reboot (pid $!, checkpoint: ${CKPT:-none})."
  )
done
