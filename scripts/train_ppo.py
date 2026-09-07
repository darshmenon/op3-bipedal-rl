"""Stage 4: train (or resume training) the OP3 PPO locomotion policy.

This is the project's actual train/resume script (distinct from
scripts/train_ppo_smoke.py, which only proved the loop runs). Logs to
TensorBoard under logs/<experiment_name>/ and checkpoints periodically
(rsl_rl's OnPolicyRunner default: every cfg.save_interval iterations, plus
a final save), so a run can be resumed with --checkpoint.

Usage:
    isaaclab.sh -p scripts/train_ppo.py --num_envs 512 --max_iterations 300
    isaaclab.sh -p scripts/train_ppo.py --stage stand --num_envs 512 --max_iterations 300
    isaaclab.sh -p scripts/train_ppo.py --stage walk_easy --num_envs 512 --max_iterations 300
    isaaclab.sh -p scripts/train_ppo.py --num_envs 512 --max_iterations 300 \
        --checkpoint logs/op3_flat/model_100.pt   # resume
"""

import argparse
import json
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel envs.")
parser.add_argument("--max_iterations", type=int, default=300, help="PPO learning iterations.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint to resume from.")
parser.add_argument("--experiment_name", type=str, default=None, help="Override the run's log/experiment name.")
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
        "walk_tiny_forward_cpg_residual_phase_obs_fine_physics",
        "walk_tiny_forward_cpg_residual_phase_obs_high_rate",
        "walk_recovery",
        "walk_structured",
        "walk_gait_phase",
        "stand",
        "stand_legs_only",
    ),
    default="walk",
    help="Training curriculum stage.",
)
parser.add_argument(
    "--learning_rate", type=float, default=None,
    help="Override PPO learning rate (default: op3_rl/ppo_cfg.py's 1e-3). "
    "See STAGE4_PPO_TRAINING.md/WALKING_FAILURE_RESEARCH.md for why a lower value (e.g. 3e-4) is worth testing.",
)
parser.add_argument(
    "--desired_kl", type=float, default=None,
    help="Override the adaptive learning-rate schedule's target KL divergence (default: "
    "op3_rl/ppo_cfg.py's 0.01). Lower = the schedule shrinks the LR more readily once an update "
    "moves the policy too far -- targets the 'training keeps degrading past its peak' pattern "
    "documented in STAGE4_PPO_TRAINING.md, which a fixed lower --learning_rate alone didn't fully fix.",
)
parser.add_argument(
    "--num_steps_per_env", type=int, default=None,
    help="Override PPO rollout length per env per update (default: op3_rl/ppo_cfg.py's 24). "
    "STAGE4_PPO_TRAINING.md's fall-timescale analysis found the default's 0.48s rollout window is "
    "close to (not much longer than) the ~0.23-0.3s measured fall-onset time constant, so many "
    "batches likely capture a fall starting without its outcome -- try 48 or 64 to test this.",
)
parser.add_argument(
    "--dof_torques_weight", type=float, default=None,
    help="Override the inherited dof_torques_l2 reward weight (default: -1.0e-5, ANYmal-scale, "
    "never overridden by any OP3 stage so far). Targets the torque_clip_frac~0.20-0.30 finding in "
    "STAGE4_PPO_TRAINING.md directly (discourage torque-saturating actions) without touching the "
    "hardware-grounded actuator model. Try e.g. -1.0e-4.",
)
parser.add_argument(
    "--obs_normalization",
    action="store_true",
    help="Enable actor/critic observation normalizers. Use when transferring or continuing from a "
    "checkpoint trained with observation normalization, including MuJoCo checkpoints produced with "
    "scripts/train_ppo_mujoco.py --obs_normalization.",
)
parser.add_argument(
    "--reset_critic", action="store_true",
    help="After loading --checkpoint, reinitialize critic weights while keeping the actor warm-started. "
    "Use with --reset_optimizer for actor-only curriculum transfer experiments.",
)
parser.add_argument(
    "--reset_optimizer", action="store_true",
    help="Do not restore PPO optimizer state from --checkpoint. Useful when changing reward/curriculum scale.",
)
parser.add_argument(
    "--debug_manifest",
    action="store_true",
    default=True,
    help="Write logs/<experiment>/run_manifest.json with resolved stage, command, reward, and PPO settings.",
)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
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
    OP3WalkTinyForwardCpgResidualPhaseObsFinePhysicsEnvCfg,
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
    "walk_tiny_forward_cpg_residual_phase_obs_fine_physics": OP3WalkTinyForwardCpgResidualPhaseObsFinePhysicsEnvCfg,
    "walk_tiny_forward_cpg_residual_phase_obs_high_rate": OP3WalkTinyForwardCpgResidualPhaseObsHighRateEnvCfg,
    "walk_recovery": OP3WalkRecoveryEnvCfg,
    "walk_structured": OP3WalkStructuredEnvCfg,
    "walk_gait_phase": OP3WalkGaitPhaseEnvCfg,
    "walk": OP3FlatEnvCfg,
}
env_cfg = _STAGE_ENV_CFGS[args.stage]()
env_cfg.scene.num_envs = args.num_envs
if args.dof_torques_weight is not None:
    env_cfg.rewards.dof_torques_l2.weight = args.dof_torques_weight

agent_cfg = OP3FlatPPORunnerCfg()
agent_cfg.max_iterations = args.max_iterations
if args.stage != "walk" and args.experiment_name is None:
    agent_cfg.experiment_name = f"op3_{args.stage}"
if args.experiment_name:
    agent_cfg.experiment_name = args.experiment_name
if args.learning_rate is not None:
    agent_cfg.algorithm.learning_rate = args.learning_rate
if args.desired_kl is not None:
    agent_cfg.algorithm.desired_kl = args.desired_kl
if args.num_steps_per_env is not None:
    agent_cfg.num_steps_per_env = args.num_steps_per_env
if args.obs_normalization:
    agent_cfg.policy.actor_obs_normalization = True
    agent_cfg.policy.critic_obs_normalization = True

log_dir = str(Path(__file__).resolve().parent.parent / "logs" / agent_cfg.experiment_name)


def _reward_weights(rewards_cfg) -> dict[str, float]:
    weights = {}
    for name in dir(rewards_cfg):
        if name.startswith("_"):
            continue
        value = getattr(rewards_cfg, name)
        if value is None or not hasattr(value, "weight"):
            continue
        weights[name] = float(value.weight)
    return dict(sorted(weights.items()))


def _stage_debug_payload() -> dict:
    cmd = env_cfg.commands.base_velocity
    reset_base = getattr(env_cfg.events, "reset_base", None)
    action_cfg = env_cfg.actions.joint_pos
    action_payload = {
        "class": getattr(getattr(action_cfg, "class_type", None), "__name__", str(getattr(action_cfg, "class_type", ""))),
        "scale": action_cfg.scale,
        "joint_names": action_cfg.joint_names,
    }
    for attr in (
        "leg_joint_names",
        "step_freq_hz",
        "hip_amp",
        "knee_amp",
        "ankle_comp",
        "lateral_shift_amp",
        "lateral_shift_sign",
    ):
        if hasattr(action_cfg, attr):
            action_payload[attr] = getattr(action_cfg, attr)
    return {
        "created_unix_time": time.time(),
        "argv": sys.argv,
        "stage": args.stage,
        "experiment_name": agent_cfg.experiment_name,
        "checkpoint": args.checkpoint,
        "num_envs": env_cfg.scene.num_envs,
        "max_iterations": agent_cfg.max_iterations,
        "sim": {
            "dt": env_cfg.sim.dt,
            "decimation": env_cfg.decimation,
            "control_dt": env_cfg.sim.dt * env_cfg.decimation,
            "control_hz": 1.0 / (env_cfg.sim.dt * env_cfg.decimation),
            "render_interval": env_cfg.sim.render_interval,
        },
        "ppo": {
            "learning_rate": agent_cfg.algorithm.learning_rate,
            "desired_kl": agent_cfg.algorithm.desired_kl,
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "entropy_coef": agent_cfg.algorithm.entropy_coef,
            "clip_param": agent_cfg.algorithm.clip_param,
            "num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
            "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            "gamma": agent_cfg.algorithm.gamma,
            "lam": agent_cfg.algorithm.lam,
        },
        "obs_normalization": {
            "actor": agent_cfg.policy.actor_obs_normalization,
            "critic": agent_cfg.policy.critic_obs_normalization,
        },
        "transfer": {
            "reset_critic": args.reset_critic,
            "reset_optimizer": args.reset_optimizer,
        },
        "command_ranges": {
            "lin_vel_x": list(cmd.ranges.lin_vel_x),
            "lin_vel_y": list(cmd.ranges.lin_vel_y),
            "ang_vel_z": list(cmd.ranges.ang_vel_z),
            "heading": list(cmd.ranges.heading),
            "heading_command": cmd.heading_command,
            "rel_standing_envs": cmd.rel_standing_envs,
            "rel_heading_envs": cmd.rel_heading_envs,
        },
        "events": {
            "push_robot_enabled": env_cfg.events.push_robot is not None,
            "add_base_mass_enabled": env_cfg.events.add_base_mass is not None,
            "base_com_enabled": env_cfg.events.base_com is not None,
            "reset_base_pose_range": reset_base.params.get("pose_range") if reset_base is not None else None,
            "reset_base_velocity_range": reset_base.params.get("velocity_range") if reset_base is not None else None,
        },
        "action": action_payload,
        "reward_weights": _reward_weights(env_cfg.rewards),
    }


manifest = _stage_debug_payload()
print(
    f"[train_ppo] stage={args.stage} num_envs={env_cfg.scene.num_envs} max_iterations={agent_cfg.max_iterations} "
    f"experiment={agent_cfg.experiment_name} resume={args.checkpoint} "
    f"learning_rate={agent_cfg.algorithm.learning_rate} desired_kl={agent_cfg.algorithm.desired_kl} "
    f"num_steps_per_env={agent_cfg.num_steps_per_env} "
    f"dof_torques_weight={env_cfg.rewards.dof_torques_l2.weight}",
    flush=True,
)
if args.debug_manifest:
    manifest_path = Path(log_dir) / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[train_ppo] wrote debug manifest: {manifest_path}", flush=True)

env = ManagerBasedRLEnv(cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)


def _reset_module_parameters(module) -> None:
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


start_iteration = 0
if args.checkpoint:
    resume_info = runner.load(args.checkpoint, load_optimizer=not args.reset_optimizer)
    start_iteration = runner.current_learning_iteration
    print(f"[train_ppo] resumed from {args.checkpoint} at iteration {start_iteration}", flush=True)
    if args.reset_critic:
        _reset_module_parameters(runner.alg.policy.critic)
        if hasattr(runner.alg.policy, "critic_obs_normalizer"):
            _reset_module_parameters(runner.alg.policy.critic_obs_normalizer)
        print("[train_ppo] reset critic parameters after actor warm-start", flush=True)
    if args.reset_optimizer:
        print("[train_ppo] using freshly initialized PPO optimizer state", flush=True)

remaining_iterations = max(agent_cfg.max_iterations - start_iteration, 0)
if remaining_iterations == 0:
    print(f"[train_ppo] checkpoint already at/past max_iterations={agent_cfg.max_iterations}; nothing to do.", flush=True)
else:
    runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=True)

print(f"Stage 4 training run completed. Logs/checkpoints in: {log_dir}", flush=True)
env.close()
simulation_app.close()
