"""Watch a trained MuJoCo PPO checkpoint live in an interactive GLFW window.

Companion to `record_mujoco_demo.py` (which renders headlessly to an mp4):
this one opens a real window on $DISPLAY instead, for watching the policy
step in real time rather than reviewing a saved file. Runs entirely on the
MuJoCo/CPU track, so it's safe to run alongside a live Isaac Lab GPU
training run (unlike a second Isaac Sim process, which is not).

Two modes, matching `teleop_play_ppo.py`'s Isaac-side pattern:
- `--mode script` (default): alternates forward walking with push-recovery
  forward shoves, ending on forward walking (never stops).
- `--mode keyboard`: live control from the MuJoCo viewer window --
  w=forward, s=stop, x=backward, a=turn, d=request opposite turn,
  p=push (forward shove), q=quit.

Usage (from a machine with a real display -- check `echo $DISPLAY`):
    python3 scripts/view_mujoco_policy.py \
        --checkpoint checkpoints/op3_walk_mujoco_best.pt

    python3 scripts/view_mujoco_policy.py \
        --checkpoint checkpoints/op3_walk_mujoco_best.pt --mode keyboard
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from op3_rl.mujoco_env import OP3MujocoVecEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--mode", choices=("script", "keyboard"), default="script")
parser.add_argument(
    "--seconds",
    type=float,
    default=None,
    help="Optional time limit. Script mode defaults to one full scripted demo; keyboard mode defaults to run until q.",
)
parser.add_argument("--forward_vel", type=float, default=0.15)
parser.add_argument("--turn_vel", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=0)
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
    "experiment_name": "view_mujoco_scratch",
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

runner = OnPolicyRunner(env, train_cfg, log_dir="/tmp/view_mujoco_scratch", device="cpu")
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device="cpu")

model = env.model
data = env.datas[0]

obs, _ = env.reset()

FORWARD_SECONDS = 8.0
PUSH_SECONDS = 3.0
DEFAULT_SCRIPT_SECONDS = 96.0  # ~90s+; override with --seconds


def _script_segments():
    """Cycle forward walking with push-recovery shoves forever; the
    outer time limit (below) is what actually bounds the run, so
    `--seconds` can extend the demo past one cycle."""
    while True:
        yield (0.15, 0.0, FORWARD_SECONDS, "Forward walking")
        yield (0.15, 0.0, PUSH_SECONDS, "Push recovery: forward shove")


time_limit = args.seconds
if time_limit is None and args.mode == "script":
    time_limit = DEFAULT_SCRIPT_SECONDS


def _time_expired(start_time: float) -> bool:
    return time_limit is not None and time.time() - start_time > time_limit


def push_torso(force_xy: tuple[float, float], steps: int = 5) -> None:
    data.xfrc_applied[env._torso_body_id, 0] = force_xy[0]
    data.xfrc_applied[env._torso_body_id, 1] = force_xy[1]
    for _ in range(steps):
        mujoco.mj_step(model, data)
    data.xfrc_applied[env._torso_body_id, :] = 0.0


command = [0.0, 0.0]
push_requested = False
should_quit = False


def _key_callback(keycode: int) -> None:
    """MuJoCo viewer key callback. GLFW letter keys arrive as ASCII codes."""
    global push_requested, should_quit
    key = chr(keycode).lower() if 0 <= keycode < 128 else ""
    if key == "q":
        should_quit = True
    elif key == "w":
        command[:] = [args.forward_vel, 0.0]
        print("[view_mujoco_policy] command: forward")
    elif key == "s":
        command[:] = [0.0, 0.0]
        print("[view_mujoco_policy] command: stop")
    elif key == "x":
        command[:] = [-args.forward_vel, 0.0]
        print("[view_mujoco_policy] command: backward")
    elif key == "a":
        command[:] = [args.forward_vel, args.turn_vel]
        print("[view_mujoco_policy] command: turn")
    elif key == "d":
        command[:] = [args.forward_vel, -args.turn_vel]
        print("[view_mujoco_policy] command: opposite turn requested")
    elif key == "p":
        push_requested = True


print(f"[view_mujoco_policy] opening viewer window on DISPLAY={os.environ.get('DISPLAY')}")
callback = _key_callback if args.mode == "keyboard" else None
with mujoco.viewer.launch_passive(model, data, key_callback=callback) as viewer:
    viewer.cam.trackbodyid = env._torso_body_id
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.distance = 1.3
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -15

    start = time.time()
    try:
        if args.mode == "script":
            with torch.inference_mode():
                for cmd_x, cmd_yaw, seconds, label in _script_segments():
                    if _time_expired(start):
                        break
                    print(f"[view_mujoco_policy] {label} (cmd_x={cmd_x}, cmd_yaw={cmd_yaw})")
                    env.commands[0, 0] = cmd_x
                    env.commands[0, 2] = cmd_yaw
                    if "shove" in label:
                        force = (25.0, 0.0) if "forward" in label else (0.0, 20.0)
                        push_torso(force)
                    n_steps = int(seconds / env.dt)
                    for _ in range(n_steps):
                        if _time_expired(start) or not viewer.is_running():
                            break
                        step_start = time.time()
                        actions = policy(obs)
                        obs, _, dones, _ = env.step(actions)
                        viewer.sync()
                        elapsed = time.time() - step_start
                        if elapsed < env.dt:
                            time.sleep(env.dt - elapsed)
        else:
            print(
                "[view_mujoco_policy] keyboard mode: focus the MuJoCo window, "
                "then use w=forward s=stop x=backward a=turn "
                "d=opposite-turn-request p=push q=quit"
            )
            with torch.inference_mode():
                while not _time_expired(start) and viewer.is_running():
                    if should_quit:
                        break
                    env.commands[0, 0] = command[0]
                    env.commands[0, 2] = command[1]
                    if push_requested:
                        push_requested = False
                        push_torso((25.0, 0.0))
                        print("[view_mujoco_policy] push applied (forward shove)")
                    step_start = time.time()
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    viewer.sync()
                    elapsed = time.time() - step_start
                    if elapsed < env.dt:
                        time.sleep(env.dt - elapsed)
    except KeyboardInterrupt:
        print("\n[view_mujoco_policy] interrupted, closing.")
    print("[view_mujoco_policy] done, closing.")

    # mujoco.viewer.launch_passive runs its GLFW render loop on a background
    # thread; tearing it down through the normal `with`-exit / interpreter
    # shutdown path segfaults on some driver setups (see
    # google-deepmind/mujoco#790, #1265). All our work is done by this
    # point, so exit here rather than let the with-block's __exit__ (and
    # the crash) run -- the OS reclaims the GL context and threads anyway.
    # os._exit() skips the normal stdio flush on interpreter shutdown, so
    # flush explicitly or buffered print() output (stdout isn't a tty when
    # piped/logged) gets silently dropped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
