"""Launch a visible Isaac Sim static-balance diagnostic for OP3.

This is the visual counterpart to diagnose_static_balance_numeric.py. It does
not create a Camera sensor or save rendered frames, because that path timed out
on this machine. Instead it opens the Isaac viewport, resets OP3 into the
zero-perturbation standing case, then steps slowly with zero policy action so
the fall can be inspected directly.

Usage:
    cd IsaacLab
    ./isaaclab.sh -p ../scripts/view_static_balance.py --num_envs 1 --max_steps 160
"""

import argparse
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=160)
parser.add_argument("--sleep_s", type=float, default=0.05, help="Wall-clock delay after each sim step for visual inspection.")
parser.add_argument("--hold_s", type=float, default=30.0, help="Keep the Isaac window open this long after stepping.")
parser.add_argument("--stage", choices=("walk", "stand"), default="stand", help="Use the stand or walk environment config.")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": False})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

from op3_rl.op3_env_cfg import OP3FlatEnvCfg, OP3StandEnvCfg  # noqa: E402

env_cfg = OP3StandEnvCfg() if args.stage == "stand" else OP3FlatEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.scene.env_spacing = 2.5
env_cfg.observations.policy.enable_corruption = False

print(f"[view-static] launching GUI stage={args.stage} num_envs={args.num_envs}")
env = ManagerBasedRLEnv(cfg=env_cfg)
obs, _ = env.reset()

robot = env.scene["robot"]
num_actions = env.action_manager.total_action_dim
zero_actions = torch.zeros((args.num_envs, num_actions), device=env.device)
alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)

env.sim.render()

for step in range(args.max_steps):
    obs, rew, terminated, truncated, info = env.step(zero_actions)
    env.sim.render()

    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    height = robot.data.root_pos_w[:, 2]
    newly_dead = terminated & alive
    alive = alive & ~terminated

    if step % 10 == 0 or newly_dead[0]:
        print(
            f"[view-static] step {step:4d}: roll={roll[0].item():+.3f} "
            f"pitch={pitch[0].item():+.3f} height={height[0].item():.3f} "
            f"alive={int(alive.sum().item())}/{args.num_envs}"
            + (" <-- FELL" if newly_dead[0] else "")
        )

    if not alive.any():
        print(f"[view-static] all envs fell by step {step}; holding window for {args.hold_s:.1f}s")
        break

    if args.sleep_s > 0.0:
        time.sleep(args.sleep_s)

end_time = time.time() + args.hold_s
while time.time() < end_time:
    env.sim.render()
    simulation_app.update()
    time.sleep(0.05)

print("[view-static] done.")
env.close()
simulation_app.close()
