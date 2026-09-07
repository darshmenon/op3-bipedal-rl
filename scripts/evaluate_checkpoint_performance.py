"""Replay-evaluate PPO checkpoints and fail fast on regression.

This is a performance gate, not a training script. It runs one or more saved
checkpoints through the same policy/env path as playback diagnostics and emits
machine-readable JSON metrics. Use it to decide whether a checkpoint is moving
the project in the right direction before continuing a training lineage.

Usage:
    cd IsaacLab
    ./isaaclab.sh -p ../scripts/evaluate_checkpoint_performance.py \
        --stage walk_easy --num_envs 128 --max_steps 150 \
        --checkpoint ../logs/op3_walk_easy_lr3e4/model_8150.pt \
        --checkpoint ../logs/op3_walk_easy_kl005/model_9150.pt \
        --max_fallen_by_step40 4 --max_fallen_by_step50 16 \
        --min_alive_at_end 96 --out ../outputs/checkpoint_eval.json

Exit code is 0 only if every checkpoint passes the requested thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PerformanceThresholds:
    max_fallen_by_step40: int | None = None
    max_fallen_by_step50: int | None = None
    max_fallen_by_step100: int | None = None
    min_alive_at_end: int | None = None
    max_abs_roll_env0: float | None = None
    max_abs_pitch_env0: float | None = None


@dataclass(frozen=True)
class PerformanceSummary:
    checkpoint: str
    stage: str
    num_envs: int
    max_steps: int
    fallen_by_step40: int
    fallen_by_step50: int
    fallen_by_step100: int
    alive_at_end: int
    first_fall_step: int | None
    median_fall_step: float | None
    mean_fall_step: float | None
    env0_max_abs_roll: float
    env0_max_abs_pitch: float
    env0_final_height: float
    env0_final_actual_vel_x: float
    env0_final_actual_vel_y: float
    env0_command_vel_x: float
    env0_command_vel_y: float
    env0_command_yaw: float
    max_abs_joint_pos: float
    max_abs_joint_vel: float
    max_abs_applied_torque: float
    max_abs_computed_torque: float
    torque_clip_fraction: float
    passed: bool
    failures: list[str]


def _count_fallen_by(fall_steps: Iterable[int | None], step: int) -> int:
    return sum(1 for fall_step in fall_steps if fall_step is not None and fall_step <= step)


def _fall_stats(fall_steps: list[int | None]) -> tuple[int | None, float | None, float | None]:
    fallen = sorted(step for step in fall_steps if step is not None)
    if not fallen:
        return None, None, None
    mid = len(fallen) // 2
    if len(fallen) % 2:
        median = float(fallen[mid])
    else:
        median = 0.5 * (fallen[mid - 1] + fallen[mid])
    mean = sum(fallen) / len(fallen)
    return fallen[0], median, mean


def threshold_failures(summary: PerformanceSummary, thresholds: PerformanceThresholds) -> list[str]:
    failures: list[str] = []
    if thresholds.max_fallen_by_step40 is not None and summary.fallen_by_step40 > thresholds.max_fallen_by_step40:
        failures.append(
            f"fallen_by_step40={summary.fallen_by_step40} > {thresholds.max_fallen_by_step40}"
        )
    if thresholds.max_fallen_by_step50 is not None and summary.fallen_by_step50 > thresholds.max_fallen_by_step50:
        failures.append(
            f"fallen_by_step50={summary.fallen_by_step50} > {thresholds.max_fallen_by_step50}"
        )
    if thresholds.max_fallen_by_step100 is not None and summary.fallen_by_step100 > thresholds.max_fallen_by_step100:
        failures.append(
            f"fallen_by_step100={summary.fallen_by_step100} > {thresholds.max_fallen_by_step100}"
        )
    if thresholds.min_alive_at_end is not None and summary.alive_at_end < thresholds.min_alive_at_end:
        failures.append(f"alive_at_end={summary.alive_at_end} < {thresholds.min_alive_at_end}")
    if thresholds.max_abs_roll_env0 is not None and summary.env0_max_abs_roll > thresholds.max_abs_roll_env0:
        failures.append(
            f"env0_max_abs_roll={summary.env0_max_abs_roll:.3f} > {thresholds.max_abs_roll_env0:.3f}"
        )
    if thresholds.max_abs_pitch_env0 is not None and summary.env0_max_abs_pitch > thresholds.max_abs_pitch_env0:
        failures.append(
            f"env0_max_abs_pitch={summary.env0_max_abs_pitch:.3f} > {thresholds.max_abs_pitch_env0:.3f}"
        )
    return failures


def make_summary(
    *,
    checkpoint: str,
    stage: str,
    num_envs: int,
    max_steps: int,
    fall_steps: list[int | None],
    env0_max_abs_roll: float,
    env0_max_abs_pitch: float,
    env0_final_height: float,
    env0_final_actual_vel_x: float,
    env0_final_actual_vel_y: float,
    env0_command_vel_x: float,
    env0_command_vel_y: float,
    env0_command_yaw: float,
    thresholds: PerformanceThresholds,
    max_abs_joint_pos: float = 0.0,
    max_abs_joint_vel: float = 0.0,
    max_abs_applied_torque: float = 0.0,
    max_abs_computed_torque: float = 0.0,
    torque_clip_fraction: float = 0.0,
) -> PerformanceSummary:
    first, median, mean = _fall_stats(fall_steps)
    provisional = PerformanceSummary(
        checkpoint=checkpoint,
        stage=stage,
        num_envs=num_envs,
        max_steps=max_steps,
        fallen_by_step40=_count_fallen_by(fall_steps, 40),
        fallen_by_step50=_count_fallen_by(fall_steps, 50),
        fallen_by_step100=_count_fallen_by(fall_steps, 100),
        alive_at_end=sum(1 for step in fall_steps if step is None),
        first_fall_step=first,
        median_fall_step=median,
        mean_fall_step=mean,
        env0_max_abs_roll=env0_max_abs_roll,
        env0_max_abs_pitch=env0_max_abs_pitch,
        env0_final_height=env0_final_height,
        env0_final_actual_vel_x=env0_final_actual_vel_x,
        env0_final_actual_vel_y=env0_final_actual_vel_y,
        env0_command_vel_x=env0_command_vel_x,
        env0_command_vel_y=env0_command_vel_y,
        env0_command_yaw=env0_command_yaw,
        max_abs_joint_pos=max_abs_joint_pos,
        max_abs_joint_vel=max_abs_joint_vel,
        max_abs_applied_torque=max_abs_applied_torque,
        max_abs_computed_torque=max_abs_computed_torque,
        torque_clip_fraction=torque_clip_fraction,
        passed=True,
        failures=[],
    )
    failures = threshold_failures(provisional, thresholds)
    return PerformanceSummary(**{**asdict(provisional), "passed": not failures, "failures": failures})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint path. Repeatable.")
    parser.add_argument(
        "--stage",
        choices=(
            "walk",
            "walk_easy",
            "walk_easy_rebalanced",
            "walk_tiny_forward",
            "walk_tiny_forward_recovery",
            "walk_tiny_forward_recovery_low_term",
            "walk_tiny_forward_danger_zone",
            "walk_tiny_forward_cpg_imitation",
            "walk_tiny_forward_cpg_imitation_slow",
            "walk_tiny_forward_foot_load_asymmetry",
            "walk_tiny_forward_cpg_imitation_foot_load",
            "walk_tiny_forward_cpg_residual",
            "walk_recovery",
            "walk_structured",
            "walk_gait_phase",
            "stand",
            "stand_legs_only",
        ),
        default="walk_easy",
    )
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--zero_command", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--max_fallen_by_step40", type=int, default=None)
    parser.add_argument("--max_fallen_by_step50", type=int, default=None)
    parser.add_argument("--max_fallen_by_step100", type=int, default=None)
    parser.add_argument("--min_alive_at_end", type=int, default=None)
    parser.add_argument("--max_abs_roll_env0", type=float, default=None)
    parser.add_argument("--max_abs_pitch_env0", type=float, default=None)
    return parser.parse_args()


def _evaluate_checkpoint(args: argparse.Namespace, checkpoint: str, thresholds: PerformanceThresholds):
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import euler_xyz_from_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from op3_rl.op3_env_cfg import (
        OP3FlatEnvCfg,
        OP3StandEnvCfg,
        OP3StandLegsOnlyEnvCfg,
        OP3WalkEasyEnvCfg,
        OP3WalkEasyRebalancedEnvCfg,
        OP3WalkGaitPhaseEnvCfg,
        OP3WalkRecoveryEnvCfg,
        OP3WalkStructuredEnvCfg,
        OP3WalkTinyForwardCpgImitationFootLoadEnvCfg,
        OP3WalkTinyForwardCpgImitationEnvCfg,
        OP3WalkTinyForwardCpgImitationSlowEnvCfg,
        OP3WalkTinyForwardCpgResidualEnvCfg,
        OP3WalkTinyForwardDangerZoneEnvCfg,
        OP3WalkTinyForwardEnvCfg,
        OP3WalkTinyForwardFootLoadAsymmetryEnvCfg,
        OP3WalkTinyForwardRecoveryEnvCfg,
        OP3WalkTinyForwardRecoveryLowTermEnvCfg,
    )
    from op3_rl.ppo_cfg import OP3FlatPPORunnerCfg

    stage_env_cfgs = {
        "stand": OP3StandEnvCfg,
        "stand_legs_only": OP3StandLegsOnlyEnvCfg,
        "walk_easy": OP3WalkEasyEnvCfg,
        "walk_easy_rebalanced": OP3WalkEasyRebalancedEnvCfg,
        "walk_tiny_forward": OP3WalkTinyForwardEnvCfg,
        "walk_tiny_forward_recovery": OP3WalkTinyForwardRecoveryEnvCfg,
        "walk_tiny_forward_recovery_low_term": OP3WalkTinyForwardRecoveryLowTermEnvCfg,
        "walk_tiny_forward_danger_zone": OP3WalkTinyForwardDangerZoneEnvCfg,
        "walk_tiny_forward_cpg_imitation": OP3WalkTinyForwardCpgImitationEnvCfg,
        "walk_tiny_forward_cpg_imitation_slow": OP3WalkTinyForwardCpgImitationSlowEnvCfg,
        "walk_tiny_forward_foot_load_asymmetry": OP3WalkTinyForwardFootLoadAsymmetryEnvCfg,
        "walk_tiny_forward_cpg_imitation_foot_load": OP3WalkTinyForwardCpgImitationFootLoadEnvCfg,
        "walk_tiny_forward_cpg_residual": OP3WalkTinyForwardCpgResidualEnvCfg,
        "walk_recovery": OP3WalkRecoveryEnvCfg,
        "walk_structured": OP3WalkStructuredEnvCfg,
        "walk_gait_phase": OP3WalkGaitPhaseEnvCfg,
        "walk": OP3FlatEnvCfg,
    }
    env_cfg = stage_env_cfgs[args.stage]()
    env_cfg.scene.num_envs = args.num_envs
    if args.zero_command:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    raw_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env)
    try:
        agent_cfg = OP3FlatPPORunnerCfg()
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        if args.stochastic:
            runner.eval_mode()
            policy = runner.alg.policy.act
        else:
            policy = runner.get_inference_policy(device=env.unwrapped.device)

        robot = raw_env.scene["robot"]
        obs = env.get_observations()
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=raw_env.device)
        fall_step = torch.full((args.num_envs,), -1, dtype=torch.long, device=raw_env.device)
        env0_max_abs_roll = 0.0
        env0_max_abs_pitch = 0.0
        env0_final_height = float(robot.data.root_pos_w[0, 2].item())
        env0_final_actual_vel_x = 0.0
        env0_final_actual_vel_y = 0.0
        env0_command_vel_x = 0.0
        env0_command_vel_y = 0.0
        env0_command_yaw = 0.0
        max_abs_joint_pos = 0.0
        max_abs_joint_vel = 0.0
        max_abs_applied_torque = 0.0
        max_abs_computed_torque = 0.0
        torque_clip_count = 0
        torque_sample_count = 0

        for step in range(args.max_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            terminated = dones.bool()
            roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
            roll = torch.atan2(torch.sin(roll), torch.cos(roll))
            pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
            actual_lin_vel = robot.data.root_lin_vel_b[:, :2]
            cmd = raw_env.command_manager.get_command("base_velocity")
            joint_pos = robot.data.joint_pos
            joint_vel = robot.data.joint_vel
            applied_torque = robot.data.applied_torque
            computed_torque = robot.data.computed_torque
            effort_limits = robot.data.joint_effort_limits

            env0_max_abs_roll = max(env0_max_abs_roll, abs(float(roll[0].item())))
            env0_max_abs_pitch = max(env0_max_abs_pitch, abs(float(pitch[0].item())))
            env0_final_height = float(robot.data.root_pos_w[0, 2].item())
            env0_final_actual_vel_x = float(actual_lin_vel[0, 0].item())
            env0_final_actual_vel_y = float(actual_lin_vel[0, 1].item())
            env0_command_vel_x = float(cmd[0, 0].item())
            env0_command_vel_y = float(cmd[0, 1].item())
            env0_command_yaw = float(cmd[0, 2].item())
            max_abs_joint_pos = max(max_abs_joint_pos, float(torch.max(torch.abs(joint_pos)).item()))
            max_abs_joint_vel = max(max_abs_joint_vel, float(torch.max(torch.abs(joint_vel)).item()))
            max_abs_applied_torque = max(
                max_abs_applied_torque, float(torch.max(torch.abs(applied_torque)).item())
            )
            max_abs_computed_torque = max(
                max_abs_computed_torque, float(torch.max(torch.abs(computed_torque)).item())
            )
            if effort_limits is not None:
                clipped = torch.abs(applied_torque) >= 0.98 * torch.abs(effort_limits)
                torque_clip_count += int(torch.count_nonzero(clipped).item())
                torque_sample_count += clipped.numel()

            newly_dead = terminated & alive
            fall_step[newly_dead] = step
            alive = alive & ~terminated
            if not alive.any():
                break

        fall_steps = [None if int(step) < 0 else int(step) for step in fall_step.cpu().tolist()]
        return make_summary(
            checkpoint=checkpoint,
            stage=args.stage,
            num_envs=args.num_envs,
            max_steps=args.max_steps,
            fall_steps=fall_steps,
            env0_max_abs_roll=env0_max_abs_roll,
            env0_max_abs_pitch=env0_max_abs_pitch,
            env0_final_height=env0_final_height,
            env0_final_actual_vel_x=env0_final_actual_vel_x,
            env0_final_actual_vel_y=env0_final_actual_vel_y,
            env0_command_vel_x=env0_command_vel_x,
            env0_command_vel_y=env0_command_vel_y,
            env0_command_yaw=env0_command_yaw,
            max_abs_joint_pos=max_abs_joint_pos,
            max_abs_joint_vel=max_abs_joint_vel,
            max_abs_applied_torque=max_abs_applied_torque,
            max_abs_computed_torque=max_abs_computed_torque,
            torque_clip_fraction=torque_clip_count / torque_sample_count if torque_sample_count else 0.0,
            thresholds=thresholds,
        )
    finally:
        env.close()


def main() -> int:
    args = _parse_args()
    thresholds = PerformanceThresholds(
        max_fallen_by_step40=args.max_fallen_by_step40,
        max_fallen_by_step50=args.max_fallen_by_step50,
        max_fallen_by_step100=args.max_fallen_by_step100,
        min_alive_at_end=args.min_alive_at_end,
        max_abs_roll_env0=args.max_abs_roll_env0,
        max_abs_pitch_env0=args.max_abs_pitch_env0,
    )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        summaries = [_evaluate_checkpoint(args, checkpoint, thresholds) for checkpoint in args.checkpoint]
    finally:
        simulation_app.close()

    payload = {
        "thresholds": asdict(thresholds),
        "summaries": [asdict(summary) for summary in summaries],
        "passed": all(summary.passed for summary in summaries),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
