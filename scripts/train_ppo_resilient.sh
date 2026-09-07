#!/usr/bin/env bash
# Wraps scripts/train_ppo.py with auto-resume: if the underlying process
# dies (e.g. the sporadic "CUDA error: unspecified launch failure" crash
# documented in STAGE4_PPO_TRAINING.md -- a PhysX/driver-level GPU fault,
# not a bug in our training code), automatically resume from the latest
# checkpoint rather than losing progress or requiring a manual restart.
#
# Usage: scripts/train_ppo_resilient.sh <experiment_name> <num_envs> <max_iterations> [max_retries] [stage] [seed_checkpoint] [learning_rate] [desired_kl] [num_steps_per_env] [dof_torques_weight] [reset_critic] [reset_optimizer] [obs_normalization]
#
# seed_checkpoint: only used when <experiment_name> has no checkpoints of its
# own yet (e.g. warm-starting a new walk-stage run from a stand-stage
# checkpoint) -- once the experiment has its own checkpoints, those always
# take priority so retries resume the in-progress run, not the seed.
#
# learning_rate, desired_kl, num_steps_per_env, dof_torques_weight,
# reset_critic, reset_optimizer, obs_normalization: optional PPO/reward/transfer overrides,
# forwarded to train_ppo.py's matching flags on every attempt (including
# retries). Pass reset_critic/reset_optimizer/obs_normalization as 1/true/yes
# to enable.
set -uo pipefail

# isaaclab.sh runs `tabs 4` under `set -e` before it does anything else. If
# TERM is unset OR "dumb" (both true for a cron-launched shell -- cron's own
# default environment sets TERM=dumb explicitly, it does not leave it unset,
# so an earlier `${TERM:-xterm-256color}` fallback here still let `tabs` fail
# with `'dumb': cannot reset tabs.` and kill isaaclab.sh before Python ever
# starts) -- indistinguishable from a real crash to the retry loop below, so
# it burns through all retries in seconds with zero progress. Force a valid
# TERM whenever the inherited one is missing or known-bad.
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM="xterm-256color"
fi

EXPERIMENT_NAME="${1:?experiment_name required}"
NUM_ENVS="${2:?num_envs required}"
MAX_ITERATIONS="${3:?max_iterations required}"
MAX_RETRIES="${4:-10}"
STAGE="${5:-walk}"
SEED_CHECKPOINT="${6:-}"
LEARNING_RATE="${7:-}"
DESIRED_KL="${8:-}"
NUM_STEPS_PER_ENV="${9:-}"
DOF_TORQUES_WEIGHT="${10:-}"
RESET_CRITIC="${11:-}"
RESET_OPTIMIZER="${12:-}"
OBS_NORMALIZATION="${13:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}"
KERN_LOG="/var/log/kern.log"

# The training command below runs from IsaacLab/, so a relative
# seed_checkpoint (as typed on the CLI, relative to PROJECT_ROOT) must be
# resolved to an absolute path here -- otherwise it silently 404s every
# attempt (torch.load: FileNotFoundError) since it's never found relative
# to IsaacLab/.
if [[ -n "${SEED_CHECKPOINT}" && "${SEED_CHECKPOINT}" != /* ]]; then
  SEED_CHECKPOINT="${PROJECT_ROOT}/${SEED_CHECKPOINT}"
fi

latest_checkpoint() {
  ls -v "${LOG_DIR}"/model_*.pt 2>/dev/null | tail -n 1
}

# rsl_rl saves the final checkpoint as model_<max_iterations - 1>.pt (0-indexed
# iteration count), so a checkpoint at or past that index means the run already
# reached the target -- regardless of whether the completion marker printed.
checkpoint_reached_target() {
  local ckpt="$1" iter
  [[ -n "${ckpt}" ]] || return 1
  iter="$(basename "${ckpt}")"
  iter="${iter#model_}"
  iter="${iter%.pt}"
  [[ "${iter}" =~ ^[0-9]+$ ]] || return 1
  (( iter >= MAX_ITERATIONS - 1 ))
}

# Catch the "max_iterations <= seed checkpoint's own iteration" mistake here,
# before burning ~20-30s on an Isaac Sim boot that would just immediately
# hit train_ppo.py's "checkpoint already at/past max_iterations; nothing to
# do" and exit -- this exact class of launch mistake (forgetting the target
# must be seed_iteration + desired_more_iterations, not just
# desired_more_iterations) has happened twice in this project's history; see
# STAGE4_PPO_TRAINING.md.
if [[ -z "$(latest_checkpoint)" && -n "${SEED_CHECKPOINT}" ]] && checkpoint_reached_target "${SEED_CHECKPOINT}"; then
  seed_iter="$(basename "${SEED_CHECKPOINT}")"; seed_iter="${seed_iter#model_}"; seed_iter="${seed_iter%.pt}"
  echo "[train_ppo_resilient] ERROR: seed checkpoint ${SEED_CHECKPOINT} is already at iteration" \
       "${seed_iter}, at/past --max_iterations ${MAX_ITERATIONS}. This would train 0-1 iterations" \
       "and immediately exit. Did you mean --max_iterations $((seed_iter + 3000)) (or similar" \
       "seed_iteration + desired_more_iterations)?" >&2
  exit 3
fi

attempt=0
while [[ "${attempt}" -le "${MAX_RETRIES}" ]]; do
  CKPT="$(latest_checkpoint)"
  if checkpoint_reached_target "${CKPT}"; then
    echo "[train_ppo_resilient] ${CKPT} already at/past target iteration ${MAX_ITERATIONS} -- nothing to do."
    exit 0
  fi
  if [[ -z "${CKPT}" && -n "${SEED_CHECKPOINT}" ]]; then
    CKPT="${SEED_CHECKPOINT}"
    echo "[train_ppo_resilient] attempt ${attempt}: no checkpoint for ${EXPERIMENT_NAME} yet, seeding from ${CKPT}"
    CKPT_ARGS=(--checkpoint "${CKPT}")
  elif [[ -n "${CKPT}" ]]; then
    echo "[train_ppo_resilient] attempt ${attempt}: resuming from ${CKPT}"
    CKPT_ARGS=(--checkpoint "${CKPT}")
  else
    echo "[train_ppo_resilient] attempt ${attempt}: starting fresh (no checkpoint found)"
    CKPT_ARGS=()
  fi

  # Watermark kern.log before the attempt so a post-attempt check only looks
  # at lines this attempt could plausibly have caused.
  KERN_LOG_START_LINES=0
  [[ -r "${KERN_LOG}" ]] && KERN_LOG_START_LINES="$(wc -l < "${KERN_LOG}")"

  LR_ARGS=()
  [[ -n "${LEARNING_RATE}" ]] && LR_ARGS=(--learning_rate "${LEARNING_RATE}")
  KL_ARGS=()
  [[ -n "${DESIRED_KL}" ]] && KL_ARGS=(--desired_kl "${DESIRED_KL}")
  ROLLOUT_ARGS=()
  [[ -n "${NUM_STEPS_PER_ENV}" ]] && ROLLOUT_ARGS=(--num_steps_per_env "${NUM_STEPS_PER_ENV}")
  TORQUE_ARGS=()
  # `=`-joined, not a separate argv token -- a bare negative value like
  # -1.0e-4 doesn't match argparse's negative-number regex (no exponent
  # support), so it gets misparsed as an unrecognized option flag otherwise.
  [[ -n "${DOF_TORQUES_WEIGHT}" ]] && TORQUE_ARGS=("--dof_torques_weight=${DOF_TORQUES_WEIGHT}")
  RESET_ARGS=()
  case "${RESET_CRITIC,,}" in
    1|true|yes) RESET_ARGS+=(--reset_critic) ;;
  esac
  case "${RESET_OPTIMIZER,,}" in
    1|true|yes) RESET_ARGS+=(--reset_optimizer) ;;
  esac
  OBS_NORM_ARGS=()
  case "${OBS_NORMALIZATION,,}" in
    1|true|yes) OBS_NORM_ARGS+=(--obs_normalization) ;;
  esac

  cd "${PROJECT_ROOT}/IsaacLab"
  ATTEMPT_LOG="$(mktemp)"
  ./isaaclab.sh -p ../scripts/train_ppo.py \
    --stage "${STAGE}" \
    --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERATIONS}" \
    --experiment_name "${EXPERIMENT_NAME}" "${CKPT_ARGS[@]}" "${LR_ARGS[@]}" "${KL_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${TORQUE_ARGS[@]}" "${RESET_ARGS[@]}" "${OBS_NORM_ARGS[@]}" 2>&1 | tee "${ATTEMPT_LOG}"
  status=$?

  # Isaac Sim's Kit wrapper can swallow an unhandled Python exception (e.g.
  # a CUDA init failure) and still exit 0, so a plain exit-code check alone
  # is unreliable -- confirm train_ppo.py's own completion marker printed.
  # That marker can also go missing on a genuinely successful run: Kit's
  # shutdown can drop buffered stdout before it flushes (same class of issue
  # as the buffered-stdout-on-shutdown gotcha in SETUP.md), so also accept a
  # checkpoint that already reached the target iteration as proof of success.
  if [[ "${status}" -eq 0 ]] && { grep -q "Stage 4 training run completed" "${ATTEMPT_LOG}" || checkpoint_reached_target "$(latest_checkpoint)"; }; then
    echo "[train_ppo_resilient] completed successfully after ${attempt} retries."
    rm -f "${ATTEMPT_LOG}"
    exit 0
  fi

  echo "[train_ppo_resilient] run exited with status ${status} (attempt ${attempt}/${MAX_RETRIES}, completion marker $(grep -q "Stage 4 training run completed" "${ATTEMPT_LOG}" && echo found || echo missing))."
  rm -f "${ATTEMPT_LOG}"

  # A PhysX GPU segfault can leave the NVIDIA driver reporting Xid 154 ("GPU
  # recovery action: Node Reboot Required") -- see STAGE4_PPO_TRAINING.md.
  # That fences the GPU until the machine is rebooted; no amount of process
  # retry/cooldown recovers it (every retry fails identically and fast), so
  # detect it and stop immediately instead of burning through max_retries.
  if [[ -r "${KERN_LOG}" ]] && tail -n "+$((KERN_LOG_START_LINES + 1))" "${KERN_LOG}" 2>/dev/null \
       | grep -q "Xid.*154.*Node Reboot Required"; then
    echo "[train_ppo_resilient] GPU fenced by driver (Xid 154: Node Reboot Required)." >&2
    echo "[train_ppo_resilient] This needs a full machine reboot, not a retry -- giving up now. Latest checkpoint: $(latest_checkpoint)" >&2
    exit 2
  fi
  # After a crash (e.g. the sporadic PhysX/CUDA "unspecified launch failure"),
  # relaunching immediately can hit a transient "CUDA devices not available"
  # failure -- the driver hasn't released the previous process's CUDA context
  # yet. A short cooldown avoids that race.
  echo "[train_ppo_resilient] cooling down 20s before retry..."
  sleep 20
  attempt=$((attempt + 1))
done

echo "[train_ppo_resilient] gave up after ${MAX_RETRIES} retries." >&2
exit 1
