"""Stage 4 follow-up: play back a trained PPO checkpoint and capture frames.

Fills the "no evaluation/playback script yet" gap noted in
STAGE4_PPO_TRAINING.md. Follows this repo's direct-instantiation pattern
(ManagerBasedRLEnv + RslRlVecEnvWrapper, same as scripts/train_ppo.py)
rather than Isaac Lab's gym-registry-based play.py, since this project
never registered a gym task id.

Revision 2: the first version rendered a frame on every physics step (via
a continuous cv2.VideoWriter), which never completed -- SETUP.md's
"PCIe-link render slowness" gotcha (rendering, not physics stepping, is
what's slow on this machine; env.step() itself never renders unless asked
to, which is why the actual PPO training runs -- thousands of steps,
zero rendering -- were fast). This version renders only a handful of
sparse PNG snapshots, same pattern as diagnose_static_balance_capture.py,
plus a snapshot at the exact termination step. Each capture prints wall
time so a render stall shows up immediately instead of silently eating
20+ minutes.

Usage:
    isaaclab.sh -p scripts/play_ppo.py --checkpoint logs/op3_flat_v5/model_1900.pt \
        --out_dir logs/op3_flat_v5/frames
"""

import argparse
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument(
    "--stage", choices=("walk", "walk_easy_rebalanced"), default="walk",
    help="Which _PLAY env config to use -- must match what the checkpoint was actually trained "
    "under (STAGE4_PPO_TRAINING.md's checkpoint-selection protocol: replay under the real training "
    "config, not a different stage's command range/disturbance settings).",
)
parser.add_argument("--max_steps", type=int, default=None, help="Cap steps; default = full episode length.")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from op3_rl.op3_env_cfg import OP3FlatEnvCfg_PLAY, OP3WalkEasyRebalancedEnvCfg_PLAY  # noqa: E402
from op3_rl.ppo_cfg import OP3FlatPPORunnerCfg  # noqa: E402

_STAGE_ENV_CFGS = {"walk": OP3FlatEnvCfg_PLAY, "walk_easy_rebalanced": OP3WalkEasyRebalancedEnvCfg_PLAY}
env_cfg = _STAGE_ENV_CFGS[args.stage]()
env_cfg.scene.num_envs = 1

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

raw_env = ManagerBasedRLEnv(cfg=env_cfg)

# The camera must exist before the environment's first reset (below, inside
# RslRlVecEnvWrapper.__init__, which resets since RSL-RL's runner doesn't)
# so its sensor-initialize callback fires and populates internal buffers
# like `_ALL_INDICES` -- creating the camera after that reset leaves it
# uninitialized and set_world_poses_from_view() below raises AttributeError.
camera_cfg = CameraCfg(
    prim_path="/World/PlayCamera",
    update_period=0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 1.0e5)
    ),
)
camera = Camera(cfg=camera_cfg)

env = RslRlVecEnvWrapper(raw_env)

agent_cfg = OP3FlatPPORunnerCfg()
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
print(f"[play_ppo] loading checkpoint: {args.checkpoint}")
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

camera_positions = torch.tensor([[1.1, -1.1, 0.55]], device=raw_env.device)
camera_targets = torch.tensor([[0.0, 0.0, 0.32]], device=raw_env.device)
camera.set_world_poses_from_view(camera_positions, camera_targets)

max_steps = args.max_steps or int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

# Dense early on since the Stage 4 static-balance diagnostic found falls
# happen within ~106 steps (~2.1s); sparser afterward as a fallback in case
# this policy does better than zero-action holding.
capture_steps = sorted(set([0, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100, 130, 160, 200, max_steps - 1]))


def save_frame(step: int, tag: str = ""):
    t0 = time.time()
    camera.update(dt=0.0)
    rgb = camera.data.output["rgb"][0].cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgb[..., :3])
    suffix = f"_{tag}" if tag else ""
    path = out_dir / f"step_{step:04d}{suffix}.png"
    img.save(path)
    print(f"[play_ppo] saved {path} (render took {time.time() - t0:.1f}s)")


obs = env.get_observations()
save_frame(0)

terminated_step = None
for step in range(1, max_steps + 1):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
    fell = terminated_step is None and bool(dones[0])
    if step in capture_steps or fell:
        save_frame(step, tag="fell" if fell else "")
    if fell:
        terminated_step = step
        print(f"[play_ppo] robot fell/terminated at step {step} ({step * env_cfg.sim.dt * env_cfg.decimation:.2f}s)")
        break

if terminated_step is None:
    print(f"[play_ppo] robot survived the full {max_steps}-step episode without falling.")

print("[play_ppo] done.")
env.close()
simulation_app.close()
