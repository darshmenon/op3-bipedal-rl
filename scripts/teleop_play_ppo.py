"""Playback a PPO checkpoint with scripted or keyboard velocity commands.

This is a demo/evaluation helper, not a training script. It supports the
bonus requirement for forward/stop/turn command changes by overriding Isaac
Lab's `base_velocity` command buffer during policy playback.

`op3_walk_isaac_best.pt` (stage `walk_tiny_forward_low_undesired_contact`)
was trained with `ang_vel_z.ranges = (0.0, 0.0)` -- it never saw a turn
command, so driving it with 'a'/'d' or the scripted turn segments is
out-of-distribution and not a real turning-behavior test. Use
`--checkpoint checkpoints/op3_walk_isaac_wide_command.pt --stage
walk_easy_rebalanced_low_undesired_contact` for turning validation instead
(trained with `ang_vel_z.ranges = (-0.3, 0.3)`; see `checkpoints/README.md`).

Headless scripted demo:
    IsaacLab/isaaclab.sh -p scripts/teleop_play_ppo.py \
      --checkpoint checkpoints/op3_walk_isaac_best.pt \
      --stage walk_tiny_forward_low_undesired_contact --mode script --max_steps 300

Interactive keyboard demo:
    IsaacLab/isaaclab.sh -p scripts/teleop_play_ppo.py \
      --checkpoint checkpoints/op3_walk_isaac_best.pt \
      --stage walk_tiny_forward_low_undesired_contact --mode keyboard --headless false

Turning validation (wide-command checkpoint):
    IsaacLab/isaaclab.sh -p scripts/teleop_play_ppo.py \
      --checkpoint checkpoints/op3_walk_isaac_wide_command.pt \
      --stage walk_easy_rebalanced_low_undesired_contact --mode script --max_steps 300

Keyboard controls: w=forward, s=stop, a=turn left, d=turn right, q=quit.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import tty
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--stage",
    choices=(
        "walk",
        "walk_easy_rebalanced",
        "walk_easy_rebalanced_low_undesired_contact",
        "walk_tiny_forward_cpg_residual",
        "walk_tiny_forward_low_undesired_contact",
    ),
    default="walk_tiny_forward_low_undesired_contact",
)
parser.add_argument("--mode", choices=("script", "keyboard"), default="script")
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--forward_vel", type=float, default=0.06)
parser.add_argument("--turn_vel", type=float, default=0.25)
parser.add_argument("--lateral_vel", type=float, default=0.08)
parser.add_argument(
    "--test_lateral", action="store_true", default=False,
    help="Add sideways-left/right segments to the scripted sequence, and stop forcing "
    "lin_vel_y to zero. Only meaningful against a checkpoint trained with nonzero "
    "lin_vel_y range (e.g. checkpoints/op3_walk_isaac_wide_command.pt, "
    "--stage walk_easy_rebalanced_low_undesired_contact -- trained with lin_vel_y="
    "(-0.1, 0.1)). checkpoints/op3_walk_isaac_best.pt was trained with lin_vel_y=(0,0) "
    "and cannot be expected to move sideways.",
)
parser.add_argument("--headless", type=lambda value: value.lower() == "true", default=True)
parser.add_argument(
    "--obs_normalization", action="store_true", default=True,
    help="Must match how --checkpoint was trained. checkpoints/op3_walk_isaac_best.pt was "
    "trained with --obs_normalization, so this defaults on.",
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from op3_rl.op3_env_cfg import (  # noqa: E402
    OP3FlatEnvCfg_PLAY,
    OP3WalkEasyRebalancedEnvCfg_PLAY,
    OP3WalkEasyRebalancedLowUndesiredContactEnvCfg,
    OP3WalkTinyForwardCpgResidualEnvCfg,
    OP3WalkTinyForwardLowUndesiredContactEnvCfg,
)
from op3_rl.ppo_cfg import OP3FlatPPORunnerCfg  # noqa: E402


def _scripted_command(step: int) -> tuple[float, float, float]:
    if not args.test_lateral:
        cycle = step % 200
        if cycle < 50:
            return args.forward_vel, 0.0, 0.0
        if cycle < 80:
            return 0.0, 0.0, 0.0
        if cycle < 125:
            return args.forward_vel, 0.0, args.turn_vel
        if cycle < 155:
            return 0.0, 0.0, 0.0
        return args.forward_vel, 0.0, -args.turn_vel
    # Lateral test cycle: forward -> stop -> strafe left -> stop -> strafe right.
    cycle = step % 250
    if cycle < 50:
        return args.forward_vel, 0.0, 0.0
    if cycle < 80:
        return 0.0, 0.0, 0.0
    if cycle < 130:
        return 0.0, args.lateral_vel, 0.0
    if cycle < 160:
        return 0.0, 0.0, 0.0
    return 0.0, -args.lateral_vel, 0.0


def _keyboard_command(previous: tuple[float, float, float]) -> tuple[tuple[float, float, float], bool]:
    if not select.select([sys.stdin], [], [], 0.0)[0]:
        return previous, False
    key = sys.stdin.read(1)
    if key == "q":
        return previous, True
    if key == "w":
        return (args.forward_vel, 0.0, 0.0), False
    if key == "s":
        return (0.0, 0.0, 0.0), False
    if key == "a":
        return (args.forward_vel, 0.0, args.turn_vel), False
    if key == "d":
        return (args.forward_vel, 0.0, -args.turn_vel), False
    return previous, False


stage_cfgs = {
    "walk": OP3FlatEnvCfg_PLAY,
    "walk_easy_rebalanced": OP3WalkEasyRebalancedEnvCfg_PLAY,
    "walk_easy_rebalanced_low_undesired_contact": OP3WalkEasyRebalancedLowUndesiredContactEnvCfg,
    "walk_tiny_forward_cpg_residual": OP3WalkTinyForwardCpgResidualEnvCfg,
    "walk_tiny_forward_low_undesired_contact": OP3WalkTinyForwardLowUndesiredContactEnvCfg,
}
env_cfg = stage_cfgs[args.stage]()
env_cfg.scene.num_envs = 1
env_cfg.observations.policy.enable_corruption = False
env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
env_cfg.commands.base_velocity.ranges.lin_vel_x = (-abs(args.forward_vel), abs(args.forward_vel))
env_cfg.commands.base_velocity.ranges.lin_vel_y = (
    (-abs(args.lateral_vel), abs(args.lateral_vel)) if args.test_lateral else (0.0, 0.0)
)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (-abs(args.turn_vel), abs(args.turn_vel))
env_cfg.commands.base_velocity.rel_standing_envs = 0.0

raw_env = ManagerBasedRLEnv(cfg=env_cfg)
env = RslRlVecEnvWrapper(raw_env)
robot = raw_env.scene["robot"]

agent_cfg = OP3FlatPPORunnerCfg()
if args.obs_normalization:
    agent_cfg.policy.actor_obs_normalization = True
    agent_cfg.policy.critic_obs_normalization = True
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

command_term = raw_env.command_manager.get_term("base_velocity")
obs = env.get_observations()
command = (0.0, 0.0, 0.0)
old_terminal_settings = None
start_xy = robot.data.root_pos_w[:, :2].clone()

if args.mode == "keyboard":
    old_terminal_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

try:
    for step in range(args.max_steps):
        if args.mode == "script":
            command = _scripted_command(step)
            should_quit = False
        else:
            command, should_quit = _keyboard_command(command)
        if should_quit:
            break

        command_term.vel_command_b[:, 0] = command[0]
        command_term.vel_command_b[:, 1] = command[1]
        command_term.vel_command_b[:, 2] = command[2]

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        if step % 10 == 0 or bool(dones[0]):
            achieved_lin_vel_b = robot.data.root_lin_vel_b[0, :2]
            achieved_ang_vel_z = robot.data.root_ang_vel_b[0, 2]
            displacement_xy = torch.norm(robot.data.root_pos_w[0, :2] - start_xy[0])
            height = robot.data.root_pos_w[0, 2]
            print(
                f"[teleop-play] step={step:04d} cmd=({command[0]:+.3f},{command[1]:+.3f},{command[2]:+.3f}) "
                f"achieved_vel=({achieved_lin_vel_b[0].item():+.3f},{achieved_lin_vel_b[1].item():+.3f},"
                f"{achieved_ang_vel_z.item():+.3f}) net_disp={displacement_xy.item():.3f}m "
                f"height={height.item():.3f}m done={bool(dones[0])}",
                flush=True,
            )
        if bool(dones[0]):
            break
finally:
    if old_terminal_settings is not None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_settings)
    env.close()
    simulation_app.close()
