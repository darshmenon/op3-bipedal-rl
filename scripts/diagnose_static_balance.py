"""Diagnostic (not part of the main training pipeline): checks whether OP3
can hold its own configured standing pose with ZERO policy action and ZERO
reset/push perturbation -- i.e. can the PD controller alone, targeting the
crouch pose already baked into init_state, keep the robot upright at all.

No PPO, no learning -- just resets once and steps the env with an all-zero
action vector (== "hold init_state joint targets") for a full episode,
tracking how many of the parallel envs are still alive (not terminated by
base_contact) at each 100-step checkpoint.

If envs fall within the first ~30-50 steps even here, the ~20-40 step
episode-length ceiling seen in every PPO run (and in the
reset-perturbation-disabled diagnostic) is NOT a training/policy problem at
all -- it's the pose/PD-gains/robot-import combination being unable to
statically balance, full stop.

Usage:
    isaaclab.sh -p ../scripts/diagnose_static_balance.py --num_envs 64
"""

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from op3_rl.op3_env_cfg import OP3FlatEnvCfg  # noqa: E402

env_cfg = OP3FlatEnvCfg()
env_cfg.scene.num_envs = args.num_envs

# Zero every source of perturbation, same as diagnose_reset_perturbation.py.
env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
env_cfg.events.reset_base.params["velocity_range"] = {
    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
}
env_cfg.events.push_robot = None
# reset_robot_joints already uses position_range=(1.0, 1.0) -- no scale randomization.

print(f"[diagnose-static] num_envs={env_cfg.scene.num_envs}, zero action, zero perturbation")

env = ManagerBasedRLEnv(cfg=env_cfg)
obs, _ = env.reset()

num_envs = env_cfg.scene.num_envs
num_actions = env.action_manager.total_action_dim
zero_actions = torch.zeros((num_envs, num_actions), device=env.device)

alive = torch.ones(num_envs, dtype=torch.bool, device=env.device)
max_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
print(f"[diagnose-static] max_steps per episode = {max_steps}")

checkpoints = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, max_steps - 1]
checkpoints = sorted(set(c for c in checkpoints if c < max_steps))

first_fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)

for step in range(max_steps):
    obs, rew, terminated, truncated, info = env.step(zero_actions)
    newly_dead = terminated & alive
    first_fall_step[newly_dead] = step
    alive = alive & ~terminated
    if step in checkpoints:
        n_alive = int(alive.sum().item())
        print(f"[diagnose-static] step {step:4d} ({step * env_cfg.sim.dt * env_cfg.decimation:5.2f}s): "
              f"{n_alive}/{num_envs} envs still standing")
    if not alive.any():
        print(f"[diagnose-static] ALL envs fell by step {step}")
        break

fallen_mask = first_fall_step >= 0
if fallen_mask.any():
    fall_steps = first_fall_step[fallen_mask].float()
    print(f"[diagnose-static] fall step stats: mean={fall_steps.mean().item():.1f} "
          f"min={fall_steps.min().item():.0f} max={fall_steps.max().item():.0f} "
          f"n_fallen={int(fallen_mask.sum().item())}/{num_envs}")
else:
    print(f"[diagnose-static] ALL {num_envs} envs survived the full episode standing still.")

print("[diagnose-static] done.")
env.close()
simulation_app.close()
