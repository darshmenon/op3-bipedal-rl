"""Diagnostic (not part of the main training pipeline): static check of
whether OP3's nominal standing pose has its center of mass over the foot
support polygon at all -- no policy, no training, just geometry right
after reset (before any physics settling/action).

Motivated by every walk-stage replay this session showing the same
monotonic, one-directional (non-oscillating) tip from step 0 -- the
"COM/support-polygon mismatch" fall mode named in STAGE4_PPO_TRAINING.md's
root-cause taxonomy. If the nominal pose's COM already sits outside or at
the edge of the polygon spanned by the two feet, no amount of policy
training fixes that; the nominal pose/robot config needs adjusting instead.

Usage:
    isaaclab.sh -p ../scripts/diagnose_com_support_polygon.py --num_envs 4
"""

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from op3_rl.op3_env_cfg import OP3StandEnvCfg  # noqa: E402

env_cfg = OP3StandEnvCfg()
env_cfg.scene.num_envs = args.num_envs

import traceback  # noqa: E402


def flush_print(*a, **kw):
    print(*a, **kw, flush=True)


flush_print("[diagnose-com] constructing env...")
env = ManagerBasedRLEnv(cfg=env_cfg)
flush_print("[diagnose-com] env constructed, resetting...")
obs, _ = env.reset()
flush_print("[diagnose-com] reset done.")

robot = env.scene["robot"]
foot_ids, foot_names = robot.find_bodies(".*ank_roll_link")
flush_print(f"[diagnose-com] feet: {foot_names} (ids {foot_ids})")

try:
    masses = robot.root_physx_view.get_masses().to(robot.device)  # (num_envs, num_bodies)
except Exception:
    flush_print("[diagnose-com] get_masses() failed:")
    flush_print(traceback.format_exc())
    raise
flush_print(f"[diagnose-com] masses shape: {tuple(masses.shape)}")

# The foot *joint* position is a point, not the sole's actual footprint --
# get the real collision-mesh bounding box in world space so the support
# polygon reflects the physical foot extent, not just the ankle point.
try:
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    for name in foot_names:
        prim_path = f"/World/envs/env_0/Robot/{name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            flush_print(f"[diagnose-com] WARNING: no prim at {prim_path}")
            continue
        bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.ComputeAlignedRange()
        flush_print(f"[diagnose-com] {name} world AABB: min={list(rng.GetMin())} max={list(rng.GetMax())} "
                    f"size={[rng.GetMax()[i] - rng.GetMin()[i] for i in range(3)]}")
except Exception:
    flush_print("[diagnose-com] foot bbox query failed:")
    flush_print(traceback.format_exc())

body_com_pos_w = robot.data.body_com_pos_w  # (num_envs, num_bodies, 3)
total_mass = masses.sum(dim=1)  # (num_envs,)
robot_com_w = (body_com_pos_w * masses.unsqueeze(-1)).sum(dim=1) / total_mass.unsqueeze(-1)  # (num_envs, 3)

root_pos_w = robot.data.root_pos_w  # (num_envs, 3)
foot_pos_w = robot.data.body_pos_w[:, foot_ids, :]  # (num_envs, 2, 3)

for e in range(args.num_envs):
    com_xy = robot_com_w[e, :2]
    feet_xy = foot_pos_w[e, :, :2]
    foot_min = feet_xy.min(dim=0).values
    foot_max = feet_xy.max(dim=0).values
    # margin: how far inside [foot_min, foot_max] the COM sits per axis (negative = outside)
    margin_x = min((com_xy[0] - foot_min[0]).item(), (foot_max[0] - com_xy[0]).item())
    margin_y = min((com_xy[1] - foot_min[1]).item(), (foot_max[1] - com_xy[1]).item())
    flush_print(
        f"[diagnose-com] env{e}: total_mass={total_mass[e].item():.3f}kg "
        f"root_xy=({root_pos_w[e,0].item():+.4f},{root_pos_w[e,1].item():+.4f}) "
        f"com_xy=({com_xy[0].item():+.4f},{com_xy[1].item():+.4f}) "
        f"foot0_xy=({feet_xy[0,0].item():+.4f},{feet_xy[0,1].item():+.4f}) "
        f"foot1_xy=({feet_xy[1,0].item():+.4f},{feet_xy[1,1].item():+.4f}) "
        f"support_bbox_x=[{foot_min[0].item():+.4f},{foot_max[0].item():+.4f}] "
        f"support_bbox_y=[{foot_min[1].item():+.4f},{foot_max[1].item():+.4f}] "
        f"margin_x={margin_x:+.4f} margin_y={margin_y:+.4f}"
        + (" <-- COM OUTSIDE support bbox on at least one axis!" if margin_x < 0 or margin_y < 0 else "")
    )

print("[diagnose-com] done.")
env.close()
simulation_app.close()
