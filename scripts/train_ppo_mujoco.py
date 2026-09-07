"""Train OP3 walking with PPO against raw MuJoCo (no Isaac Sim/Isaac Lab
involved), reusing the exact same `rsl_rl` PPO library/hyperparameters
Isaac Lab training uses. See `op3_rl/mujoco_env.py` for why this exists:
fourteen Isaac Lab experiments converged to the same collapse tonight,
and training the same task spec against a different physics engine is a
direct test of whether that's task-spec-general or Isaac-Lab-specific.

Usage:
    python3 scripts/train_ppo_mujoco.py --num_envs 64 --max_iterations 5000
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from op3_rl.mujoco_env import OP3MujocoVecEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--experiment_name", type=str, default="op3_mujoco_walk_easy_rebalanced")
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--dof_torques_weight", type=float, default=-1.0e-4)
parser.add_argument("--feet_air_time_weight", type=float, default=0.125)
parser.add_argument("--foot_load_asymmetry_weight", type=float, default=0.0)
parser.add_argument(
    "--entropy_coef", type=float, default=0.0,
    help="PPO entropy bonus. Default 0.0 matches this project's own fix for Cassie's harmful "
    "entropy_coef=0.01 runaway-exploration problem (see STAGE4_PPO_TRAINING.md). Exposed as a CLI "
    "flag to test a separate, later hypothesis: that zero entropy lets PPO permanently lock onto "
    "one mode on symmetric axes (yaw-turn direction, left/right foot loading) early in training with "
    "no pressure ever pulling it back, since std can only shrink. A small nonzero value re-admits "
    "just enough exploration pressure to test that without reintroducing the original runaway-std "
    "problem a full 0.01 caused.",
)
parser.add_argument(
    "--lin_vel_x_range", type=float, nargs=2, default=(0.02, 0.08),
    metavar=("MIN", "MAX"),
    help="Commanded forward velocity range (m/s). Default matches Isaac's tiny-forward stage, but "
    "replay-validating op3_mujoco_walk_easy_rebalanced's converged checkpoint found this range lets "
    "the policy reward-hack by standing still (net x-displacement 0.0m over a full episode) -- see "
    "op3_rl/mujoco_env.py's own comment and STAGE4_PPO_TRAINING.md. Raise this well above the "
    "tracking tolerance to force genuine stepping.",
)
parser.add_argument(
    "--ang_vel_z_range", type=float, nargs=2, default=(0.0, 0.0),
    metavar=("MIN", "MAX"),
    help="Commanded yaw rate range (rad/s). Default 0 (no turning task at all). Set nonzero to add "
    "heading tracking, matching Isaac's walk_easy_rebalanced stage.",
)
parser.add_argument(
    "--obs_normalization", action="store_true",
    help="Enable rsl_rl's empirical (running mean/std) observation normalization. Default off, "
    "matching op3_rl/ppo_cfg.py's Isaac Lab config exactly -- but replay-validating "
    "op3_mujoco_walk_hard_curriculum found the policy's behavior doesn't discriminate on the "
    "command at all (walks forward or stands regardless of commanded sign/magnitude), which is "
    "consistent with the command's small raw values (0.02-0.2ish) being swamped by other much "
    "larger-scale observation entries (joint velocities, etc.) with no normalization to rescale "
    "them onto comparable footing. See STAGE4_PPO_TRAINING.md.",
)
args = parser.parse_args()

env = OP3MujocoVecEnv(
    num_envs=args.num_envs,
    device="cpu",
    dof_torques_weight=args.dof_torques_weight,
    feet_air_time_weight=args.feet_air_time_weight,
    foot_load_asymmetry_weight=args.foot_load_asymmetry_weight,
    lin_vel_x_range=tuple(args.lin_vel_x_range),
    ang_vel_z_range=tuple(args.ang_vel_z_range),
)

# Plain-dict rsl_rl config -- deliberately NOT importing op3_rl/ppo_cfg.py's
# OP3FlatPPORunnerCfg, since that pulls in isaaclab.utils.configclass ->
# isaaclab.utils.mesh -> `pxr` (USD), which only exists inside an active
# Isaac Sim process. This script must run standalone against plain MuJoCo,
# so the config is rebuilt here as a raw dict matching rsl_rl's own native
# schema (see rsl_rl/runners/on_policy_runner.py's `_construct_algorithm`
# for the exact keys/`class_name` string convention). Values match
# op3_rl/ppo_cfg.py's `OP3FlatPPORunnerCfg` exactly -- same hyperparameters,
# just not expressed through Isaac Lab's wrapper classes.
max_iterations = args.max_iterations
experiment_name = args.experiment_name
train_cfg = {
    "num_steps_per_env": 24,
    "max_iterations": max_iterations,
    "save_interval": 50,
    "experiment_name": experiment_name,
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
        "entropy_coef": args.entropy_coef,  # default 0.0 matches Isaac Lab's ppo_cfg.py fix; see --entropy_coef help
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": args.learning_rate,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
}

print(
    f"[train_ppo_mujoco] num_envs={env.num_envs} num_obs={env.num_obs} num_actions={env.num_actions} "
    f"dt={env.dt} max_episode_length={env.max_episode_length} max_iterations={max_iterations} "
    f"experiment={experiment_name} dof_torques_weight={args.dof_torques_weight} "
    f"lin_vel_x_range={tuple(args.lin_vel_x_range)} ang_vel_z_range={tuple(args.ang_vel_z_range)} "
    f"obs_normalization={args.obs_normalization} entropy_coef={args.entropy_coef}"
)

log_dir = str(Path(__file__).resolve().parent.parent / "logs" / experiment_name)
runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device="cpu")

start_iteration = 0
if args.checkpoint:
    runner.load(args.checkpoint)
    start_iteration = runner.current_learning_iteration
    print(f"[train_ppo_mujoco] resumed from {args.checkpoint} at iteration {start_iteration}")

remaining_iterations = max(max_iterations - start_iteration, 0)
if remaining_iterations == 0:
    print(f"[train_ppo_mujoco] checkpoint already at/past max_iterations={max_iterations}; nothing to do.")
else:
    runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=True)

print(f"MuJoCo training run completed. Logs/checkpoints in: {log_dir}")
