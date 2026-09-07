"""Push-recovery test for the non-RL CPG fallback gait (MuJoCo track, CPU
only -- safe to run alongside a live Isaac Lab GPU training session).

Runs the same CPG+feedback law scripts/fallback_gait_controller.py runs in
Isaac (gait law copied from scripts/optimize_mujoco_cpg_gait.py's
`make_actions`, not imported -- that module runs its full random-search
sweep at import time, no `__main__` guard) against OP3MujocoVecEnv, applies
a torso shove at --push_at_step across every env, and reports alive counts
immediately before/after and at the end -- the same before/after framing
used for the PPO checkpoints' push-recovery demo in the README.

Usage:
    python3 scripts/test_fallback_gait_push_recovery.py --num_envs 16 \
        --params_json logs/mujoco_cpg_gait_search_best.json --push_at_step 150
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from op3_rl import mujoco_env  # noqa: E402
from op3_rl.mujoco_env import OP3MujocoVecEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--params_json", type=str, default="benchmarks/mujoco_cpg_fallback_params.json")
parser.add_argument("--push_at_step", type=int, default=150)
parser.add_argument("--push_force", type=float, default=40.0, help="Shove magnitude, N, applied for 5 substeps.")
parser.add_argument("--push_axis", choices=("x", "y"), default="x", help="x=forward shove, y=lateral shove.")
args = parser.parse_args()


def euler_from_quat_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def make_actions(env: OP3MujocoVecEnv, params: dict, step: int) -> torch.Tensor:
    actions = np.zeros((env.num_envs, env.num_actions), dtype=np.float32)
    t = step * env.dt
    phase = 2.0 * math.pi * params["step_freq_hz"] * t + params.get("phase_offset", 0.0)
    lateral_shift = params["lateral_shift_sign"] * params["lateral_shift_amp"] * math.sin(phase)

    all_joints = mujoco_env._ALL_JOINTS
    default = np.array([mujoco_env._DEFAULT_JOINT_POS.get(name, 0.0) for name in all_joints])
    joint_index = {name: i for i, name in enumerate(all_joints)}

    for env_i, data in enumerate(env.datas):
        roll, pitch, _ = euler_from_quat_wxyz(data.qpos[3:7])
        targets = default.copy()
        for side, sign, phase_offset in (
            ("l", params.get("hip_sign_l", 1.0), params.get("left_phase", 0.0)),
            ("r", params.get("hip_sign_r", -1.0), params.get("right_phase", math.pi)),
        ):
            leg_phase = phase + phase_offset
            hip_swing = params["hip_amp"] * math.sin(leg_phase)
            knee_sin = math.sin(leg_phase)
            knee_mode = params.get("knee_rect", "pos")
            if knee_mode == "abs":
                knee_swing = params["knee_amp"] * abs(knee_sin)
            elif knee_mode == "neg":
                knee_swing = params["knee_amp"] * max(0.0, -knee_sin)
            else:
                knee_swing = params["knee_amp"] * max(0.0, knee_sin)
            ankle_pitch_comp = -params["ankle_comp"] * hip_swing

            targets[joint_index[f"{side}_hip_pitch"]] += sign * hip_swing - params["pitch_kp"] * pitch
            targets[joint_index[f"{side}_knee"]] += sign * knee_swing
            targets[joint_index[f"{side}_ank_pitch"]] += sign * ankle_pitch_comp - 0.5 * params["pitch_kp"] * pitch

            side_sign = 1.0 if side == "l" else -1.0
            targets[joint_index[f"{side}_hip_roll"]] += side_sign * lateral_shift + 0.5 * params["roll_kp"] * roll
            targets[joint_index[f"{side}_ank_roll"]] += -0.5 * side_sign * lateral_shift + params["roll_kp"] * roll

        actions[env_i] = np.clip((targets - default) / mujoco_env._ACTION_SCALE, -params["action_clip"], params["action_clip"])

    return torch.from_numpy(actions)


payload = json.loads(Path(args.params_json).read_text())
params = payload["best"]["params"] if "best" in payload else payload

env = OP3MujocoVecEnv(num_envs=args.num_envs, device="cpu")
env.reset()

alive = [True] * args.num_envs
fall_step = [None] * args.num_envs
pre_push_alive = None

for step in range(args.max_steps):
    actions = make_actions(env, params, step)
    _, _, dones, _ = env.step(actions)
    for i, done in enumerate(dones.tolist()):
        if done and alive[i]:
            alive[i] = False
            fall_step[i] = step

    if step == args.push_at_step:
        pre_push_alive = sum(alive)
        axis_index = 0 if args.push_axis == "x" else 1
        for data in env.datas:
            data.xfrc_applied[env._torso_body_id, axis_index] = args.push_force
        for _ in range(5):
            for data in env.datas:
                mujoco.mj_step(env.model, data)
        for data in env.datas:
            data.xfrc_applied[env._torso_body_id, :] = 0.0
        label = "forward" if args.push_axis == "x" else "lateral"
        print(f"[push-test] step {step}: applied {args.push_force}N {label} shove to all "
              f"{args.num_envs} envs ({pre_push_alive} alive going in)")

alive_at_end = sum(alive)
fallen_from_push = [s for s in fall_step if s is not None and s >= args.push_at_step]
print(f"[push-test] alive at push (step {args.push_at_step}): {pre_push_alive}/{args.num_envs}")
print(f"[push-test] alive at end (step {args.max_steps - 1}): {alive_at_end}/{args.num_envs}")
print(f"[push-test] fell after/at the push: {len(fallen_from_push)}/{pre_push_alive}")
if fallen_from_push:
    print(f"[push-test] fall steps after push: {sorted(fallen_from_push)}")
