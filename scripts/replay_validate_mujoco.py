"""Replay-validate a MuJoCo-trained checkpoint (op3_rl/mujoco_env.py's env)
against real x-displacement/foot-alternation, not just the training reward
curve. Reward alone can't tell "genuinely walking" apart from "standing
nearly still and collecting the reward anyway" -- especially relevant here
because op3_mujoco_walk_easy_rebalanced's commanded forward velocity range
is tiny (0.02-0.08 m/s), so a policy that just stands could plausibly score
well on track_lin_vel_xy_exp without ever taking a step.

Per this project's own rule (see MEMORY/STAGE4_PPO_TRAINING.md): never call
a checkpoint "best" from a training curve alone -- always replay it first.

Usage:
    python3 scripts/replay_validate_mujoco.py \
        --checkpoint logs/op3_mujoco_walk_easy_rebalanced/model_7999.pt --num_envs 8 --steps 1000
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
parser.add_argument("--lin_vel_x_range", type=float, nargs=2, default=(0.02, 0.08), metavar=("MIN", "MAX"))
parser.add_argument("--ang_vel_z_range", type=float, nargs=2, default=(0.0, 0.0), metavar=("MIN", "MAX"))
parser.add_argument(
    "--obs_normalization", action="store_true",
    help="Must match whatever the checkpoint was trained with -- enabling rsl_rl's obs "
    "normalization adds actor/critic normalizer buffers to the policy's state_dict, so loading a "
    "checkpoint trained with a different setting fails with a state_dict key mismatch.",
)
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
    "experiment_name": "replay_validate_scratch",
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

runner = OnPolicyRunner(env, train_cfg, log_dir="/tmp/replay_validate_scratch", device="cpu")
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device="cpu")

obs, _ = env.reset()
start_x = np.array([d.qpos[0] for d in env.datas])
start_y = np.array([d.qpos[1] for d in env.datas])
last_x = start_x.copy()
last_y = start_y.copy()
fell = np.zeros(env.num_envs, dtype=bool)
fell_at_step = np.full(env.num_envs, -1)
foot_contact_switches = np.zeros(env.num_envs, dtype=int)
prev_contact = np.array(
    [env._body_in_contact_with_floor(env.datas[i], env._feet_body_ids[0]) for i in range(env.num_envs)]
)
achieved_vel_sum = np.zeros(env.num_envs)
achieved_ang_vel_sum = np.zeros(env.num_envs)
commanded_vel = np.array([env.commands[i, 0].item() for i in range(env.num_envs)])
commanded_ang_vel = np.array([env.commands[i, 2].item() for i in range(env.num_envs)])

with torch.inference_mode():
    for t in range(args.steps):
        pre_x = np.array([d.qpos[0] for d in env.datas])
        pre_y = np.array([d.qpos[1] for d in env.datas])
        actions = policy(obs)
        obs, rewards, dones, extras = env.step(actions)
        dones_np = dones.numpy()
        timeouts_np = extras["time_outs"].cpu().numpy().astype(bool)

        # OP3MujocoVecEnv auto-resets immediately when an episode ends. If we
        # read qpos after a final timeout, x/y is back at the reset pose. Keep
        # the last pre-reset pose as the rollout endpoint for displacement.
        post_x = np.array([d.qpos[0] for d in env.datas])
        post_y = np.array([d.qpos[1] for d in env.datas])
        last_x = np.where(dones_np & timeouts_np, pre_x, post_x)
        last_y = np.where(dones_np & timeouts_np, pre_y, post_y)

        for i in range(env.num_envs):
            d = env.datas[i]
            cur_contact = env._body_in_contact_with_floor(d, env._feet_body_ids[0])
            if cur_contact != prev_contact[i]:
                foot_contact_switches[i] += 1
            prev_contact[i] = cur_contact
            quat = d.qpos[3:7]
            lin_vel_w = d.qvel[0:3]
            ang_vel_w = d.qvel[3:6]
            base_lin_vel = env._quat_rotate_inverse(quat, lin_vel_w)
            base_ang_vel = env._quat_rotate_inverse(quat, ang_vel_w)
            achieved_vel_sum[i] += base_lin_vel[0]
            achieved_ang_vel_sum[i] += base_ang_vel[2]

        for i in range(env.num_envs):
            if dones_np[i] and fell_at_step[i] == -1 and not extras["time_outs"][i].item():
                fell[i] = True
                fell_at_step[i] = t

end_x = last_x
end_y = last_y
displacement_x = end_x - start_x
displacement_xy = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
mean_achieved_vel = achieved_vel_sum / args.steps
mean_achieved_ang_vel = achieved_ang_vel_sum / args.steps

print(f"[replay_validate_mujoco] checkpoint={args.checkpoint} num_envs={env.num_envs} steps={args.steps}")
print(f"  fell during rollout: {fell.sum()}/{env.num_envs} (step of fall: {fell_at_step.tolist()})")
print(f"  commanded forward vel (m/s):  {np.round(commanded_vel, 4).tolist()}")
print(f"  mean achieved fwd vel (m/s):  {np.round(mean_achieved_vel, 4).tolist()}")
print(f"  commanded yaw vel (rad/s):    {np.round(commanded_ang_vel, 4).tolist()}")
print(f"  mean achieved yaw vel (rad/s): {np.round(mean_achieved_ang_vel, 4).tolist()}")
print(f"  net x displacement (m):       {np.round(displacement_x, 4).tolist()}")
print(f"  net straight-line disp. (m):  {np.round(displacement_xy, 4).tolist()}")
print(f"  foot contact-state switches:  {foot_contact_switches.tolist()} (0 or near-0 = standing still, not stepping)")
print(f"  expected displacement if tracking perfectly: {np.round(commanded_vel * args.steps * env.dt, 4).tolist()}")
