"""Diagnostic (not part of the main training pipeline): run a trained walk
checkpoint's own policy under a real velocity command and log per-step
numeric fall-mode signals -- same idea as diagnose_static_balance_numeric.py
(base roll/pitch, height, drift, per-foot contact force), but driven by the
policy's own actions instead of zero actions, plus commanded vs. actual base
velocity, so an actual walk failure mode can be read off instead of guessed
at from reward-curve shape alone.

Context (see STAGE4_PPO_TRAINING.md): every walk-stage run so far --
op3_walk_from_stand_v2 (full difficulty), op3_walk_easy (reduced
disturbances), op3_walk_easy_alive (+ alive bonus) -- shows the same
pattern: episode length and track_lin_vel_xy_exp climb briefly right after
warm-starting from the stand checkpoint, then decline back down over a few
hundred iterations, with Episode_Termination/base_contact pinned at ~1.0
throughout (falls every single episode). Reward-curve shape alone can't
distinguish "never learns to step at all" from "steps but the gait is
unstable" from "the warm-started stand policy actively resists moving its
legs" -- this script is meant to.

No camera/rendering calls (SETUP.md's PCIe-link render-slowness gotcha) --
only env.step(), matching every diagnostic/training run that's been fast on
this machine.

Usage:
    isaaclab.sh -p ../scripts/diagnose_walk_numeric.py \
        --checkpoint logs/op3_walk_easy_alive/model_3400.pt --num_envs 8
"""

import argparse
import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to a .pt checkpoint. Omit to run completely untrained/random policy weights as a "
    "control -- if random weights survive about as long as a trained checkpoint, that's evidence "
    "the checkpoint isn't actually influencing behavior here (a loading/setup bug), not evidence "
    "the checkpoint is genuinely no better than random.",
)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument(
    "--stage",
    choices=(
        "walk",
        "walk_easy",
        "walk_easy_rebalanced",
        "walk_tiny_forward",
        "walk_tiny_forward_low_undesired_contact",
        "walk_easy_rebalanced_low_undesired_contact",
        "walk_tiny_forward_recovery",
        "walk_tiny_forward_recovery_low_term",
        "walk_tiny_forward_danger_zone",
        "walk_tiny_forward_cpg_imitation",
        "walk_tiny_forward_cpg_imitation_slow",
        "walk_tiny_forward_foot_load_asymmetry",
        "walk_tiny_forward_cpg_imitation_foot_load",
        "walk_tiny_forward_cpg_residual",
        "walk_tiny_forward_cpg_residual_slow",
        "walk_tiny_forward_cpg_residual_feedback",
        "walk_tiny_forward_cpg_residual_phase_obs",
        "walk_tiny_forward_cpg_residual_phase_obs_high_rate",
        "walk_recovery",
        "walk_structured",
        "walk_gait_phase",
        "stand",
        "stand_legs_only",
    ),
    default="walk_easy",
)
parser.add_argument(
    "--zero_command", action="store_true",
    help="Force the velocity command to (0,0,0) -- isolates whether a checkpoint trained "
    "under nonzero commands is unstable regardless of command, vs. only once commanded to move.",
)
parser.add_argument(
    "--stochastic", action="store_true",
    help="Sample actions from the policy's own training-time distribution (rsl_rl's .act(), "
    "same as training rollout collection) instead of the deterministic mean (.act_inference(), "
    "same as play_ppo.py) -- checks whether training-time episode-length stats (measured under "
    "stochastic sampling) reflect deterministic-policy behavior or not.",
)
parser.add_argument(
    "--obs_normalization",
    action="store_true",
    help="Enable rsl_rl actor/critic observation normalizers. Required for checkpoints trained with "
    "scripts/train_ppo_mujoco.py --obs_normalization.",
)
parser.add_argument("--max_steps", type=int, default=None, help="Cap steps; default = full episode length.")
parser.add_argument("--out_json", type=str, default=None, help="Optional structured JSON diagnostic output.")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from op3_rl.op3_env_cfg import (  # noqa: E402
    OP3FlatEnvCfg,
    OP3StandEnvCfg,
    OP3StandLegsOnlyEnvCfg,
    OP3WalkEasyEnvCfg,
    OP3WalkEasyRebalancedEnvCfg,
    OP3WalkGaitPhaseEnvCfg,
    OP3WalkRecoveryEnvCfg,
    OP3WalkStructuredEnvCfg,
    OP3WalkTinyForwardEnvCfg,
    OP3WalkTinyForwardLowUndesiredContactEnvCfg,
    OP3WalkEasyRebalancedLowUndesiredContactEnvCfg,
    OP3WalkTinyForwardRecoveryEnvCfg,
    OP3WalkTinyForwardRecoveryLowTermEnvCfg,
    OP3WalkTinyForwardDangerZoneEnvCfg,
    OP3WalkTinyForwardCpgImitationEnvCfg,
    OP3WalkTinyForwardCpgImitationSlowEnvCfg,
    OP3WalkTinyForwardFootLoadAsymmetryEnvCfg,
    OP3WalkTinyForwardCpgImitationFootLoadEnvCfg,
    OP3WalkTinyForwardCpgResidualEnvCfg,
    OP3WalkTinyForwardCpgResidualSlowEnvCfg,
    OP3WalkTinyForwardCpgResidualFeedbackEnvCfg,
    OP3WalkTinyForwardCpgResidualPhaseObsEnvCfg,
    OP3WalkTinyForwardCpgResidualPhaseObsHighRateEnvCfg,
)
from op3_rl.ppo_cfg import OP3FlatPPORunnerCfg  # noqa: E402

_STAGE_ENV_CFGS = {
    "stand": OP3StandEnvCfg,
    "stand_legs_only": OP3StandLegsOnlyEnvCfg,
    "walk_easy": OP3WalkEasyEnvCfg,
    "walk_easy_rebalanced": OP3WalkEasyRebalancedEnvCfg,
    "walk_tiny_forward": OP3WalkTinyForwardEnvCfg,
    "walk_tiny_forward_low_undesired_contact": OP3WalkTinyForwardLowUndesiredContactEnvCfg,
    "walk_easy_rebalanced_low_undesired_contact": OP3WalkEasyRebalancedLowUndesiredContactEnvCfg,
    "walk_tiny_forward_recovery": OP3WalkTinyForwardRecoveryEnvCfg,
    "walk_tiny_forward_recovery_low_term": OP3WalkTinyForwardRecoveryLowTermEnvCfg,
    "walk_tiny_forward_danger_zone": OP3WalkTinyForwardDangerZoneEnvCfg,
    "walk_tiny_forward_cpg_imitation": OP3WalkTinyForwardCpgImitationEnvCfg,
    "walk_tiny_forward_cpg_imitation_slow": OP3WalkTinyForwardCpgImitationSlowEnvCfg,
    "walk_tiny_forward_foot_load_asymmetry": OP3WalkTinyForwardFootLoadAsymmetryEnvCfg,
    "walk_tiny_forward_cpg_imitation_foot_load": OP3WalkTinyForwardCpgImitationFootLoadEnvCfg,
    "walk_tiny_forward_cpg_residual": OP3WalkTinyForwardCpgResidualEnvCfg,
    "walk_tiny_forward_cpg_residual_slow": OP3WalkTinyForwardCpgResidualSlowEnvCfg,
    "walk_tiny_forward_cpg_residual_feedback": OP3WalkTinyForwardCpgResidualFeedbackEnvCfg,
    "walk_tiny_forward_cpg_residual_phase_obs": OP3WalkTinyForwardCpgResidualPhaseObsEnvCfg,
    "walk_tiny_forward_cpg_residual_phase_obs_high_rate": OP3WalkTinyForwardCpgResidualPhaseObsHighRateEnvCfg,
    "walk_recovery": OP3WalkRecoveryEnvCfg,
    "walk_structured": OP3WalkStructuredEnvCfg,
    "walk_gait_phase": OP3WalkGaitPhaseEnvCfg,
    "walk": OP3FlatEnvCfg,
}
env_cfg = _STAGE_ENV_CFGS[args.stage]()
env_cfg.scene.num_envs = args.num_envs
if args.zero_command:
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

print(f"[diagnose-walk] stage={args.stage} num_envs={args.num_envs} checkpoint={args.checkpoint}", flush=True)

raw_env = ManagerBasedRLEnv(cfg=env_cfg)
env = RslRlVecEnvWrapper(raw_env)

agent_cfg = OP3FlatPPORunnerCfg()
agent_cfg.policy.actor_obs_normalization = args.obs_normalization
agent_cfg.policy.critic_obs_normalization = args.obs_normalization
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
if args.checkpoint:
    runner.load(args.checkpoint)
else:
    print("[diagnose-walk] WARNING: no --checkpoint given -- running UNTRAINED/RANDOM policy weights as a control.")
if args.stochastic:
    runner.eval_mode()
    policy = runner.alg.policy.act
else:
    policy = runner.get_inference_policy(device=env.unwrapped.device)

robot = raw_env.scene["robot"]
contact_sensor = raw_env.scene["contact_forces"]
foot_ids, foot_names = contact_sensor.find_bodies(".*ank_roll_link")
print(f"[diagnose-walk] tracking feet: {foot_names}", flush=True)

num_envs = env_cfg.scene.num_envs
start_xy = robot.data.root_pos_w[:, :2].clone()

max_steps = args.max_steps or int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
print(f"[diagnose-walk] max_steps per episode = {max_steps}", flush=True)

obs = env.get_observations()
alive = torch.ones(num_envs, dtype=torch.bool, device=raw_env.device)
fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=raw_env.device)
env0_samples = []
max_abs_roll = torch.zeros(num_envs, device=raw_env.device)
max_abs_pitch = torch.zeros(num_envs, device=raw_env.device)
max_torque_clip_fraction = torch.zeros(num_envs, device=raw_env.device)
# Per-env mean achieved yaw rate while alive, with ang_vel_z always commanded
# 0.0 (see env_cfg override above) -- a multi-env, seed-diverse check for a
# systematic yaw-drift bias, the way replay_validate_mujoco.py's yaw check
# was extended for the MuJoCo track. A single-episode teleop_play_ppo.py
# check found opposite-sign drift on two different launches of the same
# checkpoint (see STAGE4_PPO_TRAINING.md's correction) -- this aggregates
# over many parallel envs/initial conditions in one rollout instead of
# relying on one episode at a time.
yaw_vel_sum = torch.zeros(num_envs, device=raw_env.device)
alive_step_count = torch.zeros(num_envs, device=raw_env.device)


def print_distribution(step: int) -> None:
    fallen = fall_step[fall_step >= 0]
    n_fallen = fallen.numel()
    if n_fallen == 0:
        print(f"[diagnose-walk] step {step:4d}: 0/{num_envs} fallen yet", flush=True)
        return
    sorted_falls = torch.sort(fallen).values.float()
    median = sorted_falls[n_fallen // 2].item()
    print(
        f"[diagnose-walk] step {step:4d}: fall-step distribution over {n_fallen}/{num_envs} fallen so far -- "
        f"min={fallen.min().item()} median={median:.0f} mean={fallen.float().mean().item():.1f} "
        f"max={fallen.max().item()} alive_now={int(alive.sum().item())}"
        ,
        flush=True,
    )


for step in range(max_steps):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
    terminated = dones.bool()

    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    height = robot.data.root_pos_w[:, 2]
    max_abs_roll = torch.maximum(max_abs_roll, torch.abs(roll))
    max_abs_pitch = torch.maximum(max_abs_pitch, torch.abs(pitch))
    drift = torch.norm(robot.data.root_pos_w[:, :2] - start_xy, dim=-1)
    foot_force = torch.norm(contact_sensor.data.net_forces_w[:, foot_ids, :], dim=-1)  # (num_envs, 2)
    cmd = raw_env.command_manager.get_command("base_velocity")  # (num_envs, 3): lin_vel_x, lin_vel_y, ang_vel_z
    actual_lin_vel = robot.data.root_lin_vel_b[:, :2]
    actual_ang_vel_z = robot.data.root_ang_vel_b[:, 2]
    yaw_vel_sum += actual_ang_vel_z * alive.float()
    alive_step_count += alive.float()
    action_mag = torch.norm(actions, dim=-1)
    joint_pos_abs_max = torch.max(torch.abs(robot.data.joint_pos), dim=-1).values
    joint_vel_abs_max = torch.max(torch.abs(robot.data.joint_vel), dim=-1).values
    applied_torque_abs_max = torch.max(torch.abs(robot.data.applied_torque), dim=-1).values
    computed_torque_abs_max = torch.max(torch.abs(robot.data.computed_torque), dim=-1).values
    effort_limits = robot.data.joint_effort_limits
    torque_clip_fraction = torch.zeros(num_envs, device=raw_env.device)
    if effort_limits is not None:
        clipped = torch.abs(robot.data.applied_torque) >= 0.98 * torch.abs(effort_limits)
        torque_clip_fraction = clipped.float().mean(dim=-1)
    max_torque_clip_fraction = torch.maximum(max_torque_clip_fraction, torque_clip_fraction)

    newly_dead = terminated & alive
    fall_step[newly_dead] = step
    alive = alive & ~terminated

    # Env 0 detail, every few steps -- one representative trajectory is enough
    # to read off the fall mode signature (sign of tilt, timing of slip vs
    # tilt); the fall-step distribution across all envs (printed separately,
    # periodically -- not just at the end, in case the process gets cut off
    # by the known buffered-stdout-on-shutdown issue) is what tells us
    # whether env0 is typical or an outlier.
    if step % 5 == 0 or newly_dead[0]:
        env0_samples.append(
            {
                "step": step,
                "time_s": step * env_cfg.sim.dt * env_cfg.decimation,
                "roll": float(roll[0].item()),
                "pitch": float(pitch[0].item()),
                "height": float(height[0].item()),
                "drift": float(drift[0].item()),
                "cmd_vel": [float(cmd[0, 0].item()), float(cmd[0, 1].item()), float(cmd[0, 2].item())],
                "actual_vel": [float(actual_lin_vel[0, 0].item()), float(actual_lin_vel[0, 1].item())],
                "action_mag": float(action_mag[0].item()),
                "joint_abs_max": float(joint_pos_abs_max[0].item()),
                "joint_vel_abs_max": float(joint_vel_abs_max[0].item()),
                "applied_torque_abs_max": float(applied_torque_abs_max[0].item()),
                "computed_torque_abs_max": float(computed_torque_abs_max[0].item()),
                "torque_clip_frac": float(torque_clip_fraction[0].item()),
                "alive_count": int(alive.sum().item()),
                "env0_fell": bool(newly_dead[0].item()),
            }
        )
        print(
            f"[diagnose-walk] step {step:4d} ({step * env_cfg.sim.dt * env_cfg.decimation:5.2f}s) env0: "
            f"roll={roll[0].item():+.3f} pitch={pitch[0].item():+.3f} height={height[0].item():.3f} "
            f"drift={drift[0].item():.3f} foot_force={foot_force[0].tolist()} "
            f"cmd_vel=({cmd[0, 0].item():+.3f},{cmd[0, 1].item():+.3f},{cmd[0, 2].item():+.3f}) "
            f"actual_vel=({actual_lin_vel[0, 0].item():+.3f},{actual_lin_vel[0, 1].item():+.3f}) "
            f"action_mag={action_mag[0].item():.3f} "
            f"joint_abs_max={joint_pos_abs_max[0].item():.3f} "
            f"joint_vel_abs_max={joint_vel_abs_max[0].item():.3f} "
            f"applied_torque_abs_max={applied_torque_abs_max[0].item():.3f} "
            f"computed_torque_abs_max={computed_torque_abs_max[0].item():.3f} "
            f"torque_clip_frac={torque_clip_fraction[0].item():.2f} "
            f"alive_count={int(alive.sum().item())}/{num_envs}"
            + (" <-- FELL" if newly_dead[0] else ""),
            flush=True,
        )
    if step % 25 == 0 and step > 0:
        print_distribution(step)

    if not alive.any():
        print(f"[diagnose-walk] ALL envs fell by step {step}", flush=True)
        break

print_distribution(step)
if alive.any():
    print(f"[diagnose-walk] {int(alive.sum().item())}/{num_envs} envs survived the full {max_steps}-step episode.", flush=True)

mean_yaw_vel_per_env = yaw_vel_sum / torch.clamp(alive_step_count, min=1.0)
valid_yaw = alive_step_count > 0
n_valid = int(valid_yaw.sum().item())
n_positive = int((mean_yaw_vel_per_env[valid_yaw] > 0.005).sum().item())
n_negative = int((mean_yaw_vel_per_env[valid_yaw] < -0.005).sum().item())
n_near_zero = n_valid - n_positive - n_negative
print(
    f"[diagnose-walk] YAW DRIFT (commanded 0.0 throughout): mean over {n_valid} envs = "
    f"{float(mean_yaw_vel_per_env[valid_yaw].mean().item()):+.4f} rad/s -- "
    f"{n_positive}/{n_valid} envs net-positive, {n_negative}/{n_valid} net-negative, "
    f"{n_near_zero}/{n_valid} near-zero (|mean|<=0.005 rad/s)",
    flush=True,
)

fall_steps = [None if int(value) < 0 else int(value) for value in fall_step.cpu().tolist()]
fallen_steps = [value for value in fall_steps if value is not None]
summary = {
    "checkpoint": args.checkpoint,
    "stage": args.stage,
    "num_envs": num_envs,
    "max_steps": max_steps,
    "zero_command": args.zero_command,
    "stochastic": args.stochastic,
    "fallen_by_step40": sum(1 for value in fallen_steps if value <= 40),
    "fallen_by_step50": sum(1 for value in fallen_steps if value <= 50),
    "fallen_by_step75": sum(1 for value in fallen_steps if value <= 75),
    "fallen_by_step100": sum(1 for value in fallen_steps if value <= 100),
    "alive_at_end": int(alive.sum().item()),
    "first_fall_step": min(fallen_steps) if fallen_steps else None,
    "median_fall_step": float(torch.median(torch.tensor(fallen_steps, dtype=torch.float32)).item()) if fallen_steps else None,
    "mean_fall_step": float(torch.mean(torch.tensor(fallen_steps, dtype=torch.float32)).item()) if fallen_steps else None,
    "fall_steps": fall_steps,
    "env0_samples": env0_samples,
    "max_abs_roll_mean": float(max_abs_roll.mean().item()),
    "max_abs_pitch_mean": float(max_abs_pitch.mean().item()),
    "max_torque_clip_fraction_mean": float(max_torque_clip_fraction.mean().item()),
    "yaw_drift_mean_rad_s": float(mean_yaw_vel_per_env[valid_yaw].mean().item()) if n_valid else None,
    "yaw_drift_n_positive": n_positive,
    "yaw_drift_n_negative": n_negative,
    "yaw_drift_n_near_zero": n_near_zero,
    "yaw_drift_n_valid": n_valid,
}
if args.out_json:
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[diagnose-walk] wrote JSON diagnostic: {out_path}", flush=True)

print("[diagnose-walk] done.", flush=True)
env.close()
simulation_app.close()
