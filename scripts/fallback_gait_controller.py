"""Track 2 fallback: a hand-engineered central-pattern-generator (CPG) gait
plus a simple tilt-feedback stabilizer, for OP3 -- explicitly permitted by
the task brief ("gait-generation or trajectory-optimization fallback...
must include balance feedback or recovery behavior") as a contingency if
PPO training doesn't converge to reliable walking in time. Not RL-trained;
every joint target is directly computed each step from a phase variable
and live roll/pitch feedback.

Reuses the exact same env/action/actuator pipeline every PPO checkpoint in
this project runs through (`OP3FlatEnvCfg`, PD-tracked joint-position
actions, `scale=0.3` residual on the default pose) -- the "action" fed to
`env.step()` is just computed by the CPG+feedback law below instead of a
policy network, so this is directly comparable via the same numeric
diagnostics (`diagnose_walk_numeric.py`) and playback tooling
(`play_ppo.py`) used for the learned policies.

Gait design (kept deliberately simple, per the task doc's "prioritize a
working, understandable system" guidance):
- Alternating leg phase (left/right 180 degrees apart). Each leg's
  hip_pitch swings forward/back around the verified nominal stance pose
  (`op3_rl/op3_robot_cfg.py`); knee flexes only during the forward-swing
  half of its own leg's cycle (rectified sine) for foot clearance;
  ankle_pitch partially compensates hip_pitch so the foot stays closer to
  parallel with the ground.
- Balance feedback ("ankle strategy" + hip correction, the standard
  simple biped stabilization scheme): proportional correction on
  ankle_roll/hip_roll from measured base roll, and on hip_pitch/ankle_pitch
  from measured base pitch, added on top of the nominal gait.

Usage:
    isaaclab.sh -p ../scripts/fallback_gait_controller.py --num_envs 8 --max_steps 300
"""

import argparse
import json
import math
import sys
from pathlib import Path

from isaacsim import SimulationApp

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--step_freq_hz", type=float, default=0.6, help="Gait cycle frequency.")
parser.add_argument("--hip_amp", type=float, default=0.30, help="Hip-pitch swing amplitude, rad.")
parser.add_argument("--knee_amp", type=float, default=0.35, help="Knee-flexion (swing clearance) amplitude, rad.")
parser.add_argument("--ankle_comp", type=float, default=0.5, help="Fraction of hip-pitch swing compensated at the ankle.")
parser.add_argument("--roll_kp", type=float, default=1.5, help="Ankle/hip roll feedback gain on measured base roll.")
parser.add_argument("--pitch_kp", type=float, default=1.0, help="Hip/ankle pitch feedback gain on measured base pitch.")
parser.add_argument(
    "--lateral_shift_amp",
    type=float,
    default=0.08,
    help="Nominal hip/ankle-roll amplitude for shifting weight over the stance foot. Set 0 to disable.",
)
parser.add_argument(
    "--lateral_shift_sign",
    type=float,
    choices=(-1.0, 1.0),
    default=1.0,
    help="Flip if roll diverges immediately; OP3 roll joint signs are easy to get wrong.",
)
parser.add_argument(
    "--action_clip",
    type=float,
    default=4.0,
    help="Clamp raw residual action before env.step(), preventing impossible CPG targets from dominating diagnostics.",
)
parser.add_argument("--out_json", type=str, default=None, help="Optional structured JSON diagnostic output.")
parser.add_argument(
    "--params_json",
    type=str,
    default=None,
    help="Load CPG parameters from scripts/optimize_mujoco_cpg_gait.py output.",
)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument(
    "--fine_physics", action="store_true", default=False,
    help="Match MuJoCo's 0.002s physics substep timestep (default Isaac stages use "
    "0.005s) while holding the control period at the same 0.02s (decimation 10 vs "
    "4). Isolates the substep-timestep hypothesis for the Isaac/MuJoCo CPG-gait "
    "behavioral gap documented in STAGE4_PPO_TRAINING.md -- everything else in the "
    "env/actuator/reward pipeline is unchanged.",
)
args = parser.parse_args()

if args.params_json:
    payload = json.loads(Path(args.params_json).read_text())
    params = payload["best"]["params"] if "best" in payload else payload
    for key in (
        "step_freq_hz",
        "hip_amp",
        "knee_amp",
        "ankle_comp",
        "roll_kp",
        "pitch_kp",
        "lateral_shift_amp",
        "lateral_shift_sign",
        "action_clip",
    ):
        if key in params:
            setattr(args, key, params[key])

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

from op3_rl.op3_env_cfg import OP3WalkTinyForwardEnvCfg  # noqa: E402

env_cfg = OP3WalkTinyForwardEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.observations.policy.enable_corruption = False
if args.fine_physics:
    env_cfg.sim.dt = 0.002
    env_cfg.decimation = 10
    env_cfg.sim.render_interval = env_cfg.decimation

print(f"[fallback-gait] num_envs={args.num_envs} step_freq={args.step_freq_hz}Hz "
      f"hip_amp={args.hip_amp} knee_amp={args.knee_amp} roll_kp={args.roll_kp} pitch_kp={args.pitch_kp} "
      f"lateral_shift_amp={args.lateral_shift_amp} lateral_shift_sign={args.lateral_shift_sign}")

import traceback  # noqa: E402


def fp(*a, **kw):
    print(*a, **kw, flush=True)


fp("[fallback-gait] constructing env...")
env = ManagerBasedRLEnv(cfg=env_cfg)
fp("[fallback-gait] env constructed, resetting...")
obs, _ = env.reset()
fp("[fallback-gait] reset done.")

try:
    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("joint_pos")
    joint_names = list(action_term._joint_names)  # order actions must be provided in
    fp(f"[fallback-gait] joint_names ({len(joint_names)}): {joint_names}")
    default_pos = action_term._asset.data.default_joint_pos[:, action_term._joint_ids]
    fp(f"[fallback-gait] default_pos shape: {tuple(default_pos.shape)}")
    action_scale = action_term.cfg.scale  # 0.3, matches every PPO checkpoint's action term
    fp(f"[fallback-gait] action_scale: {action_scale}")
except Exception:
    fp("[fallback-gait] setup failed:")
    fp(traceback.format_exc())
    raise

# Nominal stance values for the joints the gait actually drives (everything
# else -- arms, head, hip_yaw/roll's *nominal* component -- stays at the
# default pose, i.e. its computed action is 0 unless feedback adjusts it).
_NOMINAL = {n: default_pos[0, i].item() for i, n in enumerate(joint_names)}


def cpg_targets(t: float, roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Per-env joint position targets (num_envs, num_joints), in the same
    joint order as `joint_names` / the action term expects."""
    num_envs = roll.shape[0]
    targets = torch.zeros((num_envs, len(joint_names)), device=env.device)
    phase = 2.0 * math.pi * args.step_freq_hz * t
    # Positive during the left-leg swing half, negative during right-leg swing.
    # The sign is intentionally CLI-flippable because OP3 roll-axis convention
    # mistakes show up as immediate monotonic roll divergence.
    lateral_shift = args.lateral_shift_sign * args.lateral_shift_amp * math.sin(phase)

    for side, sign, leg_phase in (("l", +1.0, phase), ("r", -1.0, phase + math.pi)):
        hip_pitch_swing = args.hip_amp * math.sin(leg_phase)
        knee_swing = args.knee_amp * max(0.0, math.sin(leg_phase))  # only during forward swing
        ankle_pitch_comp = -args.ankle_comp * hip_pitch_swing

        for name, base_val in (
            (f"{side}_hip_pitch", _NOMINAL[f"{side}_hip_pitch"] + sign * hip_pitch_swing),
            (f"{side}_knee", _NOMINAL[f"{side}_knee"] + sign * knee_swing),
            (f"{side}_ank_pitch", _NOMINAL[f"{side}_ank_pitch"] + sign * ankle_pitch_comp),
            (f"{side}_hip_yaw", _NOMINAL[f"{side}_hip_yaw"]),
            (f"{side}_hip_roll", _NOMINAL[f"{side}_hip_roll"]),
            (f"{side}_ank_roll", _NOMINAL[f"{side}_ank_roll"]),
        ):
            idx = joint_names.index(name)
            targets[:, idx] = base_val

    # Balance feedback ("ankle strategy"): counteract measured tilt by
    # rolling the ankles and hips against it, and pitching hip/ankle against
    # fore-aft lean. Same correction sign/magnitude for both legs.
    for side in ("l", "r"):
        ank_roll_idx = joint_names.index(f"{side}_ank_roll")
        hip_roll_idx = joint_names.index(f"{side}_hip_roll")
        hip_pitch_idx = joint_names.index(f"{side}_hip_pitch")
        ank_pitch_idx = joint_names.index(f"{side}_ank_pitch")
        side_sign = +1.0 if side == "l" else -1.0
        targets[:, hip_roll_idx] += side_sign * lateral_shift
        targets[:, ank_roll_idx] += -side_sign * 0.5 * lateral_shift
        # Sign was inverted here (amplifying tilt instead of opposing it,
        # confirmed by direct step-by-step trace + a MuJoCo A/B test showing
        # a 28->76-step survival improvement from flipping it -- see
        # STAGE4_PPO_TRAINING.md). Flipped to actually counteract tilt.
        targets[:, ank_roll_idx] += args.roll_kp * roll
        targets[:, hip_roll_idx] += 0.5 * args.roll_kp * roll
        targets[:, hip_pitch_idx] += -args.pitch_kp * pitch
        targets[:, ank_pitch_idx] += -0.5 * args.pitch_kp * pitch

    return targets


robot_data = robot.data
num_envs = env_cfg.scene.num_envs
alive = torch.ones(num_envs, dtype=torch.bool, device=env.device)
fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
start_xy = robot_data.root_pos_w[:, :2].clone()
dt = env_cfg.sim.dt * env_cfg.decimation
env0_samples = []
max_abs_roll = torch.zeros(num_envs, device=env.device)
max_abs_pitch = torch.zeros(num_envs, device=env.device)

for step in range(args.max_steps):
    roll, pitch, _ = euler_xyz_from_quat(robot_data.root_quat_w)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))

    targets = cpg_targets(step * dt, roll, pitch)
    # env.step() expects the RAW action (pre-scale, pre-offset) that the
    # action term will turn into `default_pos + scale * action` -- invert
    # that here so `targets` above is what's actually commanded.
    action = (targets - default_pos) / action_scale
    action = torch.clamp(action, -args.action_clip, args.action_clip)
    obs, rew, terminated, truncated, info = env.step(action)

    height = robot_data.root_pos_w[:, 2]
    drift = torch.norm(robot_data.root_pos_w[:, :2] - start_xy, dim=-1)
    newly_dead = terminated & alive
    fall_step[newly_dead] = step
    alive = alive & ~terminated
    max_abs_roll = torch.maximum(max_abs_roll, torch.abs(roll))
    max_abs_pitch = torch.maximum(max_abs_pitch, torch.abs(pitch))

    if step % 10 == 0 or newly_dead[0]:
        env0_samples.append(
            {
                "step": step,
                "time_s": step * dt,
                "roll": float(roll[0].item()),
                "pitch": float(pitch[0].item()),
                "height": float(height[0].item()),
                "drift": float(drift[0].item()),
                "action_abs_max": float(torch.max(torch.abs(action[0])).item()),
                "alive_count": int(alive.sum().item()),
                "env0_fell": bool(newly_dead[0].item()),
            }
        )
        fp(
            f"[fallback-gait] step {step:4d} ({step * dt:5.2f}s) env0: "
            f"roll={roll[0].item():+.3f} pitch={pitch[0].item():+.3f} height={height[0].item():.3f} "
            f"drift={drift[0].item():.3f} action_abs_max={torch.max(torch.abs(action[0])).item():.2f} "
            f"alive_count={int(alive.sum().item())}/{num_envs}"
            + (" <-- FELL" if newly_dead[0] else "")
        )

    if not alive.any():
        print(f"[fallback-gait] ALL envs fell by step {step}")
        break

if alive.any():
    print(f"[fallback-gait] {int(alive.sum().item())}/{num_envs} envs survived the full {args.max_steps}-step run.")

fall_steps = [None if int(value) < 0 else int(value) for value in fall_step.cpu().tolist()]
fallen_steps = [value for value in fall_steps if value is not None]
summary = {
    "mode": "fallback_cpg",
    "num_envs": num_envs,
    "max_steps": args.max_steps,
    "step_freq_hz": args.step_freq_hz,
    "hip_amp": args.hip_amp,
    "knee_amp": args.knee_amp,
    "ankle_comp": args.ankle_comp,
    "roll_kp": args.roll_kp,
    "pitch_kp": args.pitch_kp,
    "lateral_shift_amp": args.lateral_shift_amp,
    "lateral_shift_sign": args.lateral_shift_sign,
    "action_clip": args.action_clip,
    "fallen_by_step40": sum(1 for value in fallen_steps if value <= 40),
    "fallen_by_step50": sum(1 for value in fallen_steps if value <= 50),
    "fallen_by_step75": sum(1 for value in fallen_steps if value <= 75),
    "fallen_by_step100": sum(1 for value in fallen_steps if value <= 100),
    "alive_at_end": int(alive.sum().item()),
    "first_fall_step": min(fallen_steps) if fallen_steps else None,
    "median_fall_step": float(torch.median(torch.tensor(fallen_steps, dtype=torch.float32)).item()) if fallen_steps else None,
    "mean_fall_step": float(torch.mean(torch.tensor(fallen_steps, dtype=torch.float32)).item()) if fallen_steps else None,
    "max_abs_roll_mean": float(max_abs_roll.mean().item()),
    "max_abs_pitch_mean": float(max_abs_pitch.mean().item()),
    "fall_steps": fall_steps,
    "env0_samples": env0_samples,
}
if args.out_json:
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fp(f"[fallback-gait] wrote JSON diagnostic: {out_path}")

print("[fallback-gait] done.")
env.close()
simulation_app.close()
