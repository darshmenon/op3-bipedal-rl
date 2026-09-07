"""Watch the non-RL CPG fallback gait live in an interactive GLFW window
(MuJoCo track, CPU only -- safe to run alongside a live Isaac Lab GPU
training run, same as view_mujoco_policy.py).

Every joint target here comes directly from a hand-designed phase
oscillator plus roll/pitch tilt feedback (see
scripts/fallback_gait_controller.py's docstring for the design) -- no
policy network, no checkpoint. Gait law copied from
scripts/optimize_mujoco_cpg_gait.py's `make_actions` (not imported: that
module runs its full random-search sweep at import time, no `__main__`
guard).

Unlike the PPO viewer, the CPG doesn't take a velocity command -- it's an
open-loop forward gait with balance feedback only, so there's no
forward/backward/turn switching here, just walking + push recovery.

Two modes, matching view_mujoco_policy.py:
- `--mode script` (default): forward walking, then a forward shove, then
  a lateral shove, then more forward walking.
- `--mode keyboard`: p=push (forward shove), q=quit.

Usage (from a machine with a real display -- check `echo $DISPLAY`):
    python3 scripts/view_mujoco_fallback_gait.py

    python3 scripts/view_mujoco_fallback_gait.py --mode keyboard
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from op3_rl import mujoco_env  # noqa: E402
from op3_rl.mujoco_env import OP3MujocoVecEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--params_json", type=str, default="benchmarks/mujoco_cpg_fallback_params.json")
parser.add_argument("--mode", choices=("script", "keyboard"), default="script")
parser.add_argument(
    "--seconds",
    type=float,
    default=None,
    help="Optional time limit. Script mode defaults to one full scripted demo; keyboard mode defaults to run until q.",
)
args = parser.parse_args()

payload = json.loads(Path(args.params_json).read_text())
params = payload["best"]["params"] if "best" in payload else payload


def euler_from_quat_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def make_action(env: OP3MujocoVecEnv, data, step: int) -> torch.Tensor:
    t = step * env.dt
    phase = 2.0 * math.pi * params["step_freq_hz"] * t + params.get("phase_offset", 0.0)
    lateral_shift = params["lateral_shift_sign"] * params["lateral_shift_amp"] * math.sin(phase)

    joint_index = {name: i for i, name in enumerate(mujoco_env._ALL_JOINTS)}
    default = np.array([mujoco_env._DEFAULT_JOINT_POS.get(name, 0.0) for name in mujoco_env._ALL_JOINTS])
    roll, pitch, _ = euler_from_quat_wxyz(data.qpos[3:7])
    targets = default.copy()
    for side, sign, phase_offset in (
        ("l", params.get("hip_sign_l", 1.0), params.get("left_phase", 0.0)),
        ("r", params.get("hip_sign_r", -1.0), params.get("right_phase", math.pi)),
    ):
        leg_phase = phase + phase_offset
        hip_swing = params["hip_amp"] * math.sin(leg_phase)
        knee_sin = math.sin(leg_phase)
        knee_mode = params.get("knee_rect", "pos")
        if knee_mode == "abs":
            knee_swing = params["knee_amp"] * abs(knee_sin)
        elif knee_mode == "neg":
            knee_swing = params["knee_amp"] * max(0.0, -knee_sin)
        else:
            knee_swing = params["knee_amp"] * max(0.0, knee_sin)
        ankle_pitch_comp = -params["ankle_comp"] * hip_swing

        targets[joint_index[f"{side}_hip_pitch"]] += sign * hip_swing - params["pitch_kp"] * pitch
        targets[joint_index[f"{side}_knee"]] += sign * knee_swing
        targets[joint_index[f"{side}_ank_pitch"]] += sign * ankle_pitch_comp - 0.5 * params["pitch_kp"] * pitch

        side_sign = 1.0 if side == "l" else -1.0
        targets[joint_index[f"{side}_hip_roll"]] += side_sign * lateral_shift + 0.5 * params["roll_kp"] * roll
        targets[joint_index[f"{side}_ank_roll"]] += -0.5 * side_sign * lateral_shift + params["roll_kp"] * roll

    action = np.clip((targets - default) / mujoco_env._ACTION_SCALE, -params["action_clip"], params["action_clip"])
    return torch.from_numpy(action[None, :].astype(np.float32))


env = OP3MujocoVecEnv(num_envs=1, device="cpu")
env.reset()

model = env.model
data = env.datas[0]

SEGMENTS = [
    (12.0, "Forward walking"),
    (3.0, "Push recovery: forward shove"),
    (3.0, "Push recovery: lateral shove"),
    (12.0, "Forward again"),
]

script_duration = sum(segment[0] for segment in SEGMENTS)
time_limit = args.seconds
if time_limit is None and args.mode == "script":
    time_limit = script_duration


def _time_expired(start_time: float) -> bool:
    return time_limit is not None and time.time() - start_time > time_limit


def push_torso(force_xy: tuple[float, float], steps: int = 5) -> None:
    data.xfrc_applied[env._torso_body_id, 0] = force_xy[0]
    data.xfrc_applied[env._torso_body_id, 1] = force_xy[1]
    for _ in range(steps):
        mujoco.mj_step(model, data)
    data.xfrc_applied[env._torso_body_id, :] = 0.0


push_requested = False
should_quit = False


def _key_callback(keycode: int) -> None:
    global push_requested, should_quit
    key = chr(keycode).lower() if 0 <= keycode < 128 else ""
    if key == "q":
        should_quit = True
    elif key == "p":
        push_requested = True


print(f"[view_mujoco_fallback_gait] opening viewer window on DISPLAY={__import__('os').environ.get('DISPLAY')}")
callback = _key_callback if args.mode == "keyboard" else None
with mujoco.viewer.launch_passive(model, data, key_callback=callback) as viewer:
    viewer.cam.trackbodyid = env._torso_body_id
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.distance = 1.3
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -15

    start = time.time()
    step = 0
    if args.mode == "script":
        for seconds, label in SEGMENTS:
            if _time_expired(start):
                break
            print(f"[view_mujoco_fallback_gait] {label}")
            if "shove" in label:
                force = (25.0, 0.0) if "forward" in label else (0.0, 20.0)
                push_torso(force)
            n_steps = int(seconds / env.dt)
            for _ in range(n_steps):
                if _time_expired(start) or not viewer.is_running():
                    break
                step_start = time.time()
                action = make_action(env, data, step)
                env.step(action)
                step += 1
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < env.dt:
                    time.sleep(env.dt - elapsed)
    else:
        print("[view_mujoco_fallback_gait] keyboard mode: focus the MuJoCo window, then p=push q=quit")
        while not _time_expired(start) and viewer.is_running():
            if should_quit:
                break
            if push_requested:
                push_requested = False
                push_torso((25.0, 0.0))
                print("[view_mujoco_fallback_gait] push applied (forward shove)")
            step_start = time.time()
            action = make_action(env, data, step)
            env.step(action)
            step += 1
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < env.dt:
                time.sleep(env.dt - elapsed)
    print("[view_mujoco_fallback_gait] done, closing.")
