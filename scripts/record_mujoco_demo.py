"""Render a demo video of a MuJoCo-trained OP3 checkpoint: forward walking,
command changes (backward, turning), and push recovery -- the task brief's
required demo content ("walking, recovering from minor perturbations, ...
teleoperated" bonus covered by the command-change segments here).

Runs entirely on the MuJoCo/CPU track so it doesn't contend with a live
Isaac Lab GPU training run. Uses MuJoCo's own `Renderer` (software OSMesa
backend, no display needed) plus imageio/ffmpeg to write an MP4 directly --
no external video tool required.

Usage:
    MUJOCO_GL=osmesa python3 scripts/record_mujoco_demo.py \
        --checkpoint logs/op3_mujoco_walk_symmetric_fresh_finetune/model_33100.pt \
        --out demo/op3_walk_demo.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from op3_rl.mujoco_env import OP3MujocoVecEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, default="demo/op3_walk_demo.mp4")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps", type=int, default=50, help="MuJoCo control rate (dt=0.02s) -- real-time playback.")
parser.add_argument(
    "--obs_normalization", action="store_true", default=True,
    help="Must match training. This project's MuJoCo checkpoints are all trained with it on.",
)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

env = OP3MujocoVecEnv(num_envs=1, device="cpu", lin_vel_x_range=(0.15, 0.15), ang_vel_z_range=(0.0, 0.0))

train_cfg = {
    "num_steps_per_env": 24,
    "max_iterations": 1,
    "save_interval": 50,
    "experiment_name": "record_demo_scratch",
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

runner = OnPolicyRunner(env, train_cfg, log_dir="/tmp/record_demo_scratch", device="cpu")
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device="cpu")

model = env.model
data = env.datas[0]
renderer = mujoco.Renderer(model, height=args.height, width=args.width)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = env._torso_body_id
cam.distance = 1.1
cam.azimuth = 120
cam.elevation = -15

try:
    _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except OSError:
    _FONT = ImageFont.load_default()


def render_frame(label: str) -> np.ndarray:
    renderer.update_scene(data, camera=cam)
    frame = renderer.render()
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width, 34)], fill=(0, 0, 0))
    draw.text((10, 5), label, fill=(255, 255, 255), font=_FONT)
    return np.asarray(img)


def push_torso(force_xy: tuple[float, float], steps: int = 5) -> None:
    """Apply a brief external force impulse to the torso (push-recovery test)."""
    data.xfrc_applied[env._torso_body_id, 0] = force_xy[0]
    data.xfrc_applied[env._torso_body_id, 1] = force_xy[1]
    for _ in range(steps):
        mujoco.mj_step(model, data)
    data.xfrc_applied[env._torso_body_id, :] = 0.0


frames: list[np.ndarray] = []
obs, _ = env.reset()

SEG_STEPS = int(args.fps * 3)  # ~3s of sim time per label-change granularity


def run_segment(cmd_x: float, cmd_yaw: float, seconds: float, label: str) -> None:
    global obs
    env.commands[0, 0] = cmd_x
    env.commands[0, 2] = cmd_yaw
    n = int(args.fps * seconds)
    with torch.inference_mode():
        for _ in range(n):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            frames.append(render_frame(label))
            if dones[0].item() and not extras["time_outs"][0].item():
                frames.append(render_frame(f"{label} -- FELL, resetting"))


print("[record_mujoco_demo] segment 1/6: forward walking")
run_segment(cmd_x=0.15, cmd_yaw=0.0, seconds=10, label="Forward walking (0.15 m/s command)")

print("[record_mujoco_demo] segment 2/6: push recovery (forward shove)")
push_torso((25.0, 0.0), steps=5)
run_segment(cmd_x=0.15, cmd_yaw=0.0, seconds=8, label="Push recovery: forward shove")

print("[record_mujoco_demo] segment 3/6: push recovery (lateral shove)")
push_torso((0.0, 20.0), steps=5)
run_segment(cmd_x=0.15, cmd_yaw=0.0, seconds=8, label="Push recovery: lateral shove")

print("[record_mujoco_demo] segment 4/6: command change -- backward")
run_segment(cmd_x=-0.15, cmd_yaw=0.0, seconds=10, label="Command change: walking backward")

print("[record_mujoco_demo] segment 5/6: command change -- turning")
run_segment(cmd_x=0.08, cmd_yaw=0.3, seconds=10, label="Command change: turning (yaw command)")

print("[record_mujoco_demo] segment 6/6: forward again + push")
push_torso((-20.0, 0.0), steps=5)
run_segment(cmd_x=0.15, cmd_yaw=0.0, seconds=8, label="Forward + push recovery (backward shove)")

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
print(f"[record_mujoco_demo] writing {len(frames)} frames ({len(frames) / args.fps:.1f}s) to {out_path}")
imageio.mimwrite(str(out_path), frames, fps=args.fps, quality=8, macro_block_size=1)
print("[record_mujoco_demo] done.")
