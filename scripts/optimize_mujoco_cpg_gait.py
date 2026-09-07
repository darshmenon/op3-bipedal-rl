"""Random-search a simple CPG gait in MuJoCo and save the best parameters.

This is a fast trajectory/gait-optimization diagnostic: it does not train a
network. It searches the same CPG parameter family used by
scripts/fallback_gait_controller.py, scores survival plus forward progress in
the MuJoCo OP3 env, and writes a JSON file that Isaac can replay.

Usage:
    python3 scripts/optimize_mujoco_cpg_gait.py --trials 200 --num_envs 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from op3_rl.mujoco_env import OP3MujocoVecEnv  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=160)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out_json", type=str, default="logs/mujoco_cpg_gait_search_best.json")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def sample_params() -> dict[str, float]:
    return {
        "step_freq_hz": random.uniform(0.15, 0.75),
        "hip_amp": random.uniform(0.04, 0.34),
        "knee_amp": random.uniform(0.00, 0.42),
        "ankle_comp": random.uniform(0.15, 0.85),
        "lateral_shift_amp": random.uniform(0.00, 0.12),
        "lateral_shift_sign": random.choice([-1.0, 1.0]),
        "roll_kp": random.uniform(0.0, 2.5),
        "pitch_kp": random.uniform(0.0, 2.0),
        "action_clip": random.uniform(1.0, 4.0),
    }


def euler_from_quat_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def make_actions(env: OP3MujocoVecEnv, params: dict[str, float], step: int) -> torch.Tensor:
    actions = np.zeros((env.num_envs, env.num_actions), dtype=np.float32)
    t = step * env.dt
    phase = 2.0 * math.pi * params["step_freq_hz"] * t
    lateral_shift = params["lateral_shift_sign"] * params["lateral_shift_amp"] * math.sin(phase)

    joint_index = {name: i for i, name in enumerate(env._ALL_JOINTS_FOR_DIAGNOSTICS)}
    default = env._DEFAULT_POS_FOR_DIAGNOSTICS

    for env_i, data in enumerate(env.datas):
        roll, pitch, _ = euler_from_quat_wxyz(data.qpos[3:7])
        targets = default.copy()
        for side, sign, phase_offset in (("l", 1.0, 0.0), ("r", -1.0, math.pi)):
            leg_phase = phase + phase_offset
            hip_swing = params["hip_amp"] * math.sin(leg_phase)
            knee_swing = params["knee_amp"] * max(0.0, math.sin(leg_phase))
            ankle_pitch_comp = -params["ankle_comp"] * hip_swing

            # Feedback sign was inverted (amplifying tilt instead of opposing
            # it) -- see the matching fix/comment in
            # scripts/fallback_gait_controller.py and STAGE4_PPO_TRAINING.md.
            targets[joint_index[f"{side}_hip_pitch"]] += sign * hip_swing - params["pitch_kp"] * pitch
            targets[joint_index[f"{side}_knee"]] += sign * knee_swing
            targets[joint_index[f"{side}_ank_pitch"]] += sign * ankle_pitch_comp - 0.5 * params["pitch_kp"] * pitch

            side_sign = 1.0 if side == "l" else -1.0
            targets[joint_index[f"{side}_hip_roll"]] += side_sign * lateral_shift + 0.5 * params["roll_kp"] * roll
            targets[joint_index[f"{side}_ank_roll"]] += -0.5 * side_sign * lateral_shift + params["roll_kp"] * roll

        actions[env_i] = np.clip((targets - default) / env._ACTION_SCALE_FOR_DIAGNOSTICS, -params["action_clip"], params["action_clip"])

    return torch.from_numpy(actions)


def evaluate(params: dict[str, float]) -> dict:
    env = OP3MujocoVecEnv(
        num_envs=args.num_envs,
        lin_vel_x_range=(0.08, 0.16),
        ang_vel_z_range=(0.0, 0.0),
        reset_noise_scale=0.0,
    )
    start_x = np.array([data.qpos[0] for data in env.datas])
    alive = np.ones(args.num_envs, dtype=bool)
    fall_steps: list[int | None] = [None] * args.num_envs
    max_roll = np.zeros(args.num_envs)
    max_pitch = np.zeros(args.num_envs)

    for step in range(args.max_steps):
        actions = make_actions(env, params, step)
        _, _, dones, _ = env.step(actions)
        done_np = dones.cpu().numpy().astype(bool)
        for i, data in enumerate(env.datas):
            roll, pitch, _ = euler_from_quat_wxyz(data.qpos[3:7])
            max_roll[i] = max(max_roll[i], abs(roll))
            max_pitch[i] = max(max_pitch[i], abs(pitch))
        newly_dead = done_np & alive
        for i in np.where(newly_dead)[0]:
            fall_steps[i] = step
        alive &= ~done_np
        if not alive.any():
            break

    end_x = np.array([data.qpos[0] for data in env.datas])
    progress = end_x - start_x
    alive_steps = np.array([args.max_steps if v is None else v for v in fall_steps], dtype=float)
    score = (
        alive_steps.mean()
        + 250.0 * np.clip(progress, -0.5, 2.0).mean()
        - 25.0 * max_roll.mean()
        - 25.0 * max_pitch.mean()
        + 100.0 * alive.mean()
    )
    close = getattr(env, "close", None)
    if callable(close):
        close()
    return {
        "score": float(score),
        "alive_at_end": int(alive.sum()),
        "mean_alive_steps": float(alive_steps.mean()),
        "mean_progress_x": float(progress.mean()),
        "fall_steps": fall_steps,
        "max_abs_roll_mean": float(max_roll.mean()),
        "max_abs_pitch_mean": float(max_pitch.mean()),
        "params": params,
    }


# Attach diagnostic constants without widening the public env API.
OP3MujocoVecEnv._ALL_JOINTS_FOR_DIAGNOSTICS = OP3MujocoVecEnv.__init__.__globals__["_ALL_JOINTS"]
OP3MujocoVecEnv._DEFAULT_POS_FOR_DIAGNOSTICS = np.array(
    [OP3MujocoVecEnv.__init__.__globals__["_DEFAULT_JOINT_POS"].get(name, 0.0) for name in OP3MujocoVecEnv._ALL_JOINTS_FOR_DIAGNOSTICS]
)
OP3MujocoVecEnv._ACTION_SCALE_FOR_DIAGNOSTICS = OP3MujocoVecEnv.__init__.__globals__["_ACTION_SCALE"]

best: dict | None = None
for trial in range(args.trials):
    result = evaluate(sample_params())
    if best is None or result["score"] > best["score"]:
        best = result
        print(
            f"[mujoco-cpg-search] trial={trial:04d} score={best['score']:.2f} "
            f"alive={best['alive_at_end']}/{args.num_envs} steps={best['mean_alive_steps']:.1f} "
            f"dx={best['mean_progress_x']:.3f} params={best['params']}",
            flush=True,
        )

out_path = Path(args.out_json)
out_path.parent.mkdir(parents=True, exist_ok=True)
assert best is not None
payload = {"seed": args.seed, "trials": args.trials, "num_envs": args.num_envs, "max_steps": args.max_steps, "best": best}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"[mujoco-cpg-search] wrote {out_path}", flush=True)
