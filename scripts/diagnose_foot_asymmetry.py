"""Quick standalone check: measure real foot-load asymmetry for a checkpoint.

Not a training/logging change -- a one-off diagnostic to get a concrete
before/after number for the foot_load_asymmetry_weight experiment, since
the training loop doesn't log this term separately yet.

Usage:
    python3 scripts/diagnose_foot_asymmetry.py --checkpoint checkpoints/op3_walk_mujoco_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from op3_rl.mujoco_env import OP3MujocoVecEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--obs_normalization", action="store_true", default=True)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

env = OP3MujocoVecEnv(num_envs=1, device="cpu", lin_vel_x_range=(0.15, 0.15), ang_vel_z_range=(0.0, 0.0))

train_cfg = {
    "num_steps_per_env": 24,
    "max_iterations": 1,
    "save_interval": 50,
    "experiment_name": "diagnose_asymmetry_scratch",
    "empirical_normalization": None,
    "obs_groups": {"policy": ["policy"]},
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "actor_obs_normalization": args.obs_normalization,
        "critic_obs_normalization": args.obs_normalization,
        "actor_hidden_dims": [128, 128, 128],
        "critic_hidden_dims": [128, 128, 128],
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 3e-4,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
}

runner = OnPolicyRunner(env, train_cfg, log_dir="/tmp/diagnose_asymmetry_scratch", device="cpu")
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device="cpu")

obs, _ = env.reset()
d = env.datas[0]

left_forces = []
right_forces = []
asymmetry_ratios = []

with torch.inference_mode():
    for step in range(args.steps):
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        f_left = env._foot_contact_force(d, env._feet_body_ids[0])
        f_right = env._foot_contact_force(d, env._feet_body_ids[1])
        left_forces.append(f_left)
        right_forces.append(f_right)
        total = max(f_left + f_right, 1e-6)
        asymmetry_ratios.append(abs(f_left - f_right) / total)
        if bool(dones[0]):
            break

left_forces = np.array(left_forces)
right_forces = np.array(right_forces)
asymmetry_ratios = np.array(asymmetry_ratios)

print(f"[diagnose_foot_asymmetry] checkpoint={args.checkpoint} steps_run={len(left_forces)}")
print(f"  mean left foot force:  {left_forces.mean():.3f}")
print(f"  mean right foot force: {right_forces.mean():.3f}")
print(f"  mean |L-R|/(L+R) asymmetry ratio: {asymmetry_ratios.mean():.4f}  (0=perfectly symmetric, 1=one foot never loaded)")
print(f"  median asymmetry ratio: {np.median(asymmetry_ratios):.4f}")
print(f"  fraction of steps with one foot load > 2x the other: {(np.maximum(left_forces, right_forces) > 2 * np.minimum(left_forces, right_forces) + 1e-6).mean():.3f}")
