"""Stage 3: basic PPO smoke test for OP3.

Proves the full loop -- env resets, steps, PD-controlled OP3 articulation,
PPO (rsl_rl) collects rollouts and updates a policy -- runs end-to-end
without crashing. This is deliberately NOT a real training run: few
parallel envs, few iterations, untuned reward/PD-gain placeholders (see
op3_rl/op3_robot_cfg.py and op3_rl/op3_env_cfg.py). Run via:

    isaaclab.sh -p scripts/train_ppo_smoke.py --num_envs 16 --max_iterations 5

Run via `isaaclab.sh -p` (needs SimulationApp before importing isaaclab.*).
"""

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel envs (kept small for the smoke test).")
parser.add_argument("--max_iterations", type=int, default=5, help="PPO learning iterations (kept small for the smoke test).")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from op3_rl.op3_env_cfg import OP3FlatEnvCfg  # noqa: E402
from op3_rl.ppo_cfg import OP3FlatPPORunnerCfg  # noqa: E402

env_cfg = OP3FlatEnvCfg()
env_cfg.scene.num_envs = args.num_envs

agent_cfg = OP3FlatPPORunnerCfg()
agent_cfg.max_iterations = args.max_iterations

print(f"[smoke test] num_envs={env_cfg.scene.num_envs} max_iterations={agent_cfg.max_iterations}")

env = ManagerBasedRLEnv(cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

log_dir = str(Path(__file__).resolve().parent.parent / "logs" / "op3_flat_smoke")
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

print("Stage 3 PPO smoke test completed successfully.")
env.close()
simulation_app.close()
