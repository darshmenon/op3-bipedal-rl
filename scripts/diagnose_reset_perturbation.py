"""Diagnostic (not part of the main training pipeline): isolates whether
OP3's ~20-30 step episode-length ceiling (seen across every PD-gain
revision in Stage 4) is caused by the reset-time perturbation (inherited
unchanged from Cassie's default EventCfg: +/-0.5 m/s linear, +/-0.5 rad/s
angular velocity kick, +/-0.5m/180deg pose randomization on every reset,
plus periodic push_robot) rather than the PD gains/pose themselves.

Disables reset_base's pose/velocity randomization (down to zero range,
spawning exactly at the configured init_state every time) and disables
push_robot entirely. Everything else (PD gains, crouch pose, rewards,
observations) is untouched. If mean episode length jumps far above the
~20-30 baseline, the reset perturbation is the bottleneck, not the
actuators. If it's still tiny, the problem is elsewhere (pose/gains).

Usage:
    isaaclab.sh -p ../scripts/diagnose_reset_perturbation.py --num_envs 256 --max_iterations 150
"""

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=150)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from op3_rl.op3_env_cfg import OP3FlatEnvCfg  # noqa: E402
from op3_rl.ppo_cfg import OP3FlatPPORunnerCfg  # noqa: E402

env_cfg = OP3FlatEnvCfg()
env_cfg.scene.num_envs = args.num_envs

# Zero out reset-time perturbation: spawn exactly at init_state every time.
env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
env_cfg.events.reset_base.params["velocity_range"] = {
    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
}
# Disable the periodic push during the episode too.
env_cfg.events.push_robot = None

agent_cfg = OP3FlatPPORunnerCfg()
agent_cfg.max_iterations = args.max_iterations
agent_cfg.experiment_name = "op3_flat_diag_noreset"
agent_cfg.save_interval = 1000  # not interested in checkpoints, just the reward/ep-len curve

print(f"[diagnose] num_envs={env_cfg.scene.num_envs} max_iterations={agent_cfg.max_iterations} "
      "reset perturbation DISABLED (zero pose/velocity range, push_robot off)")

env = ManagerBasedRLEnv(cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

log_dir = str(Path(__file__).resolve().parent.parent / "logs" / agent_cfg.experiment_name)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

print(f"[diagnose] done. Logs in: {log_dir}")
env.close()
simulation_app.close()
