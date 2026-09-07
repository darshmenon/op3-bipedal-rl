"""Diagnostic (not part of the main training pipeline): same zero-action,
zero-perturbation static-balance test as diagnose_static_balance.py, but
with a fixed camera capturing RGB frames at chosen steps so the fall
pattern (tips one way? oscillates? feet slide?) can be inspected visually
instead of only from the alive/dead step counts.

Usage:
    isaaclab.sh -p ../scripts/diagnose_static_balance_capture.py --out_dir /path/to/save
"""

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

from op3_rl.op3_env_cfg import OP3FlatEnvCfg  # noqa: E402

env_cfg = OP3FlatEnvCfg()
env_cfg.scene.num_envs = 1

env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
env_cfg.events.reset_base.params["velocity_range"] = {
    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
}
env_cfg.events.push_robot = None

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

env = ManagerBasedRLEnv(cfg=env_cfg)

camera_cfg = CameraCfg(
    prim_path="/World/DiagCamera",
    update_period=0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 1.0e5)
    ),
)
camera = Camera(cfg=camera_cfg)

obs, _ = env.reset()

camera_positions = torch.tensor([[1.1, -1.1, 0.55]], device=env.device)
camera_targets = torch.tensor([[0.0, 0.0, 0.32]], device=env.device)
camera.set_world_poses_from_view(camera_positions, camera_targets)

# Let the camera render at least once before reading data.
env.sim.render()
camera.update(dt=0.0)

num_actions = env.action_manager.total_action_dim
zero_actions = torch.zeros((1, num_actions), device=env.device)

max_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
capture_steps = sorted(set([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, min(120, max_steps - 1)]))

def save_frame(step: int):
    camera.update(dt=0.0)
    rgb = camera.data.output["rgb"][0].cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgb[..., :3])
    path = out_dir / f"step_{step:04d}.png"
    img.save(path)
    print(f"[capture] saved {path}")

save_frame(0)

for step in range(1, max_steps):
    obs, rew, terminated, truncated, info = env.step(zero_actions)
    if step in capture_steps:
        save_frame(step)
    if step >= max(capture_steps):
        break

print("[capture] done.")
env.close()
simulation_app.close()
