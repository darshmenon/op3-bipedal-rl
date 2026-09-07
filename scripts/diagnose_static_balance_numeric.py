"""Diagnostic (not part of the main training pipeline): same zero-action,
zero-perturbation static-balance test as diagnose_static_balance.py, but
logs numeric fall-mode signals (base roll/pitch, base height, horizontal
drift, per-foot contact force) instead of camera frames.

Superseded diagnose_static_balance_capture.py after that script timed out
at 900s with zero output on this machine -- same PCIe-link/render-slowness
issue noted in SETUP.md (rendering, not physics stepping, is the slow
path). This script never calls sim.render()/camera.update(), only
env.step(), which every other diagnostic (and real training) has shown to
be fast.

Distinguishes the three fall-mode candidates from STAGE4_PPO_TRAINING.md's
"Root-cause investigation" section by which signal moves first/most:
  - COM/support-polygon mismatch -> roll or pitch grows ~monotonically in
    one consistent sign from step 0.
  - Underdamped PD -> roll/pitch oscillates (sign-changing) with growing
    amplitude before crossing the termination threshold.
  - Foot slip/friction -> base x/y drifts horizontally while roll/pitch
    stay small, and/or foot contact force drops toward 0 before the robot
    is actually tilted much.

Usage:
    isaaclab.sh -p ../scripts/diagnose_static_balance_numeric.py --num_envs 8
"""

import argparse
import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=None, help="Cap steps; default = full episode length.")
parser.add_argument("--out_json", type=str, default=None, help="Optional structured JSON diagnostic output.")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

from op3_rl.op3_env_cfg import OP3FlatEnvCfg  # noqa: E402

env_cfg = OP3FlatEnvCfg()
env_cfg.scene.num_envs = args.num_envs

env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
env_cfg.events.reset_base.params["velocity_range"] = {
    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
}
env_cfg.events.push_robot = None

print(f"[diagnose-numeric] num_envs={env_cfg.scene.num_envs}, zero action, zero perturbation")

env = ManagerBasedRLEnv(cfg=env_cfg)
obs, _ = env.reset()

robot = env.scene["robot"]
contact_sensor = env.scene["contact_forces"]
foot_ids, foot_names = contact_sensor.find_bodies(".*ank_roll_link")
print(f"[diagnose-numeric] tracking feet: {foot_names}")

num_envs = env_cfg.scene.num_envs
num_actions = env.action_manager.total_action_dim
zero_actions = torch.zeros((num_envs, num_actions), device=env.device)

start_xy = robot.data.root_pos_w[:, :2].clone()

max_steps = args.max_steps or int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
print(f"[diagnose-numeric] max_steps per episode = {max_steps}")

alive = torch.ones(num_envs, dtype=torch.bool, device=env.device)
fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
env0_samples = []
max_abs_roll = torch.zeros(num_envs, device=env.device)
max_abs_pitch = torch.zeros(num_envs, device=env.device)

for step in range(max_steps):
    obs, rew, terminated, truncated, info = env.step(zero_actions)

    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    height = robot.data.root_pos_w[:, 2]
    max_abs_roll = torch.maximum(max_abs_roll, torch.abs(roll))
    max_abs_pitch = torch.maximum(max_abs_pitch, torch.abs(pitch))
    drift = torch.norm(robot.data.root_pos_w[:, :2] - start_xy, dim=-1)
    foot_force = torch.norm(contact_sensor.data.net_forces_w[:, foot_ids, :], dim=-1)  # (num_envs, 2)

    newly_dead = terminated & alive
    fall_step[newly_dead] = step
    alive = alive & ~terminated

    # Env 0 only, every few steps -- one representative trajectory is enough
    # to read off the fall mode; averaging across envs would blur exactly
    # the signal we need (sign of tilt, timing of slip vs tilt).
    if step % 5 == 0 or newly_dead[0]:
        env0_samples.append(
            {
                "step": step,
                "time_s": step * env_cfg.sim.dt * env_cfg.decimation,
                "roll": float(roll[0].item()),
                "pitch": float(pitch[0].item()),
                "height": float(height[0].item()),
                "drift": float(drift[0].item()),
                "foot_force": [float(x) for x in foot_force[0].tolist()],
                "alive_count": int(alive.sum().item()),
                "env0_fell": bool(newly_dead[0].item()),
            }
        )
        print(
            f"[diagnose-numeric] step {step:4d} ({step * env_cfg.sim.dt * env_cfg.decimation:5.2f}s) env0: "
            f"roll={roll[0].item():+.3f} pitch={pitch[0].item():+.3f} height={height[0].item():.3f} "
            f"drift={drift[0].item():.3f} foot_force={foot_force[0].tolist()} "
            f"alive_count={int(alive.sum().item())}/{num_envs}"
            + (" <-- FELL" if newly_dead[0] else "")
        )

    if not alive.any():
        print(f"[diagnose-numeric] ALL envs fell by step {step}")
        break

fall_steps = [None if int(value) < 0 else int(value) for value in fall_step.cpu().tolist()]
fallen_steps = [value for value in fall_steps if value is not None]
summary = {
    "num_envs": num_envs,
    "max_steps": max_steps,
    "alive_at_end": int(alive.sum().item()),
    "first_fall_step": min(fallen_steps) if fallen_steps else None,
    "median_fall_step": float(torch.median(torch.tensor(fallen_steps, dtype=torch.float32)).item()) if fallen_steps else None,
    "mean_fall_step": float(torch.mean(torch.tensor(fallen_steps, dtype=torch.float32)).item()) if fallen_steps else None,
    "fall_steps": fall_steps,
    "env0_samples": env0_samples,
    "max_abs_roll_mean": float(max_abs_roll.mean().item()),
    "max_abs_pitch_mean": float(max_abs_pitch.mean().item()),
}
if args.out_json:
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[diagnose-numeric] wrote JSON diagnostic: {out_path}")

print("[diagnose-numeric] done.")
env.close()
simulation_app.close()
