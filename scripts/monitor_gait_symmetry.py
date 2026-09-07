"""Left/right foot symmetry check for a MuJoCo-trained checkpoint.

Companion to `replay_validate_mujoco.py` (which measures forward/backward
and yaw tracking) -- this measures the OTHER open issue: per-foot contact
time and per-foot load, to see whether one leg is doing more of the work
than the other ("one-leg-forward/one-leg-back" gait asymmetry).

Usage:
    python3 scripts/monitor_gait_symmetry.py \
        --checkpoint logs/op3_mujoco_walk_entropy_unstick/model_63000.pt \
        --num_envs 8 --steps 1000 --obs_normalization
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
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lin_vel_x_range", type=float, nargs=2, default=(-0.2, 0.2), metavar=("MIN", "MAX"))
parser.add_argument("--ang_vel_z_range", type=float, nargs=2, default=(-0.3, 0.3), metavar=("MIN", "MAX"))
parser.add_argument("--obs_normalization", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

env = OP3MujocoVecEnv(
    num_envs=args.num_envs,
    device="cpu",
    lin_vel_x_range=tuple(args.lin_vel_x_range),
    ang_vel_z_range=tuple(args.ang_vel_z_range),
)

train_cfg = {
    "num_steps_per_env": 24,
    "max_iterations": 1,
    "save_interval": 50,
    "experiment_name": "monitor_gait_symmetry_scratch",
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

runner = OnPolicyRunner(env, train_cfg, log_dir="/tmp/monitor_gait_symmetry_scratch", device="cpu")
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device="cpu")

obs, _ = env.reset()
fell = np.zeros(env.num_envs, dtype=bool)
commanded_vel = np.array([env.commands[i, 0].item() for i in range(env.num_envs)])

# Per-env, per-foot accumulators (foot 0 = left, foot 1 = right, matching
# op3_rl.mujoco_env._FEET_BODIES).
contact_steps = np.zeros((env.num_envs, 2), dtype=np.int64)
force_sum = np.zeros((env.num_envs, 2), dtype=np.float64)
alive_steps = np.zeros(env.num_envs, dtype=np.int64)

with torch.inference_mode():
    for t in range(args.steps):
        actions = policy(obs)
        obs, rewards, dones, extras = env.step(actions)
        dones_np = dones.numpy()
        for i in range(env.num_envs):
            if fell[i]:
                continue
            d = env.datas[i]
            for k, bid in enumerate(env._feet_body_ids):
                if env._body_in_contact_with_floor(d, bid):
                    contact_steps[i, k] += 1
                force_sum[i, k] += env._foot_contact_force(d, bid)
            alive_steps[i] += 1
            if dones_np[i] and not extras["time_outs"][i].item():
                fell[i] = True

contact_frac = contact_steps / np.maximum(alive_steps[:, None], 1)
mean_force = force_sum / np.maximum(alive_steps[:, None], 1)
force_total = mean_force.sum(axis=1)
force_asymmetry = np.abs(mean_force[:, 0] - mean_force[:, 1]) / np.maximum(force_total, 1e-6)
contact_asymmetry = np.abs(contact_frac[:, 0] - contact_frac[:, 1])

print(f"[monitor_gait_symmetry] checkpoint={args.checkpoint} num_envs={env.num_envs} steps={args.steps}")
print(f"  commanded forward vel (m/s):     {np.round(commanded_vel, 4).tolist()}")
print(f"  fell during rollout: {fell.sum()}/{env.num_envs}, alive_steps: {alive_steps.tolist()}")
print(f"  left  foot contact-time frac:    {np.round(contact_frac[:, 0], 3).tolist()}")
print(f"  right foot contact-time frac:    {np.round(contact_frac[:, 1], 3).tolist()}")
print(f"  |L-R| contact-time asymmetry:    {np.round(contact_asymmetry, 3).tolist()}  (0 = perfectly alternating, 1 = one leg never touches down)")
print(f"  mean left  foot force (N):       {np.round(mean_force[:, 0], 3).tolist()}")
print(f"  mean right foot force (N):       {np.round(mean_force[:, 1], 3).tolist()}")
print(f"  |L-R| force asymmetry (0-1):     {np.round(force_asymmetry, 3).tolist()}  (0 = balanced load, 1 = all load on one foot)")
print(f"  mean over envs -- contact asym: {float(np.mean(contact_asymmetry)):.4f}, force asym: {float(np.mean(force_asymmetry)):.4f}")
