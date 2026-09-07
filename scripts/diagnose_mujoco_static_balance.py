#!/usr/bin/env python3
"""MuJoCo zero-action static-balance diagnostic for OP3."""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from op3_rl import mujoco_env


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=150)
parser.add_argument("--reset_noise_scale", type=float, default=0.0)
parser.add_argument("--out_json", type=Path, default=Path("logs/diagnose_mujoco_static_balance.json"))
args = parser.parse_args()

env = mujoco_env.OP3MujocoVecEnv(
    num_envs=args.num_envs,
    reset_noise_scale=args.reset_noise_scale,
    lin_vel_x_range=(0.0, 0.0),
    ang_vel_z_range=(0.0, 0.0),
)

alive = np.ones(args.num_envs, dtype=bool)
fall_steps = np.full(args.num_envs, -1, dtype=np.int64)
env0_samples = []

for step in range(args.max_steps):
    actions = np.zeros((args.num_envs, env.num_actions), dtype=np.float32)
    obs, reward, dones, extras = env.step(torch.from_numpy(actions))
    del obs, reward, extras

    d0 = env.datas[0]
    torso_id = env.model.body(mujoco_env._TORSO_BODY).id
    torso_contact = env._body_in_contact_with_floor(d0, torso_id)
    quat = d0.qpos[3:7].copy()
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    rot = mat.reshape(3, 3)
    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    roll = np.arctan2(rot[2, 1], rot[2, 2])

    newly_dead = dones.cpu().numpy().astype(bool) & alive
    fall_steps[newly_dead] = step
    alive &= ~newly_dead

    if step % 5 == 0 or newly_dead[0]:
        env0_samples.append(
            {
                "step": step,
                "time_s": step * env.dt,
                "height": float(d0.qpos[2]),
                "roll": float(roll),
                "pitch": float(pitch),
                "torso_contact": bool(torso_contact),
                "alive_count": int(alive.sum()),
                "env0_fell": bool(newly_dead[0]),
            }
        )

    if not alive.any():
        break

fallen = fall_steps[fall_steps >= 0]
summary = {
    "num_envs": args.num_envs,
    "max_steps": args.max_steps,
    "reset_noise_scale": args.reset_noise_scale,
    "alive_at_end": int(alive.sum()),
    "first_fall_step": int(fallen.min()) if fallen.size else None,
    "median_fall_step": float(np.median(fallen)) if fallen.size else None,
    "mean_fall_step": float(np.mean(fallen)) if fallen.size else None,
    "fall_steps": [None if int(x) < 0 else int(x) for x in fall_steps.tolist()],
    "env0_samples": env0_samples,
}
args.out_json.parent.mkdir(parents=True, exist_ok=True)
args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
