"""Track 3 (demo pipeline): watch a trained PPO checkpoint live in the Isaac
Sim GUI, for recording a demo video by screen-capturing the window.

Deliberately reuses `view_static_balance.py`'s GUI-render pattern
(`env.sim.render()` each step, no Camera sensor) rather than
`play_ppo.py`'s camera-based PNG capture -- the camera path has repeatedly
hung for its full timeout with zero output on this machine (documented in
SETUP.md and reconfirmed this session), while the direct-render GUI path
has already worked (used earlier this session to show the robot model
live). This script is that same proven pattern, driven by the checkpoint's
own policy actions instead of zero actions.

Usage (from a machine with a real display -- check `echo $DISPLAY`):
    cd IsaacLab
    ./isaaclab.sh -p ../scripts/view_ppo_policy.py \
        --checkpoint ../checkpoints/op3_walk_best.pt --stage walk_easy_rebalanced \
        --num_envs 1 --max_steps 300 --hold_s 10
"""

import argparse
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--stage", choices=(
        "walk", "walk_easy", "walk_easy_rebalanced", "stand", "stand_legs_only",
        "walk_tiny_forward_low_undesired_contact", "walk_easy_rebalanced_low_undesired_contact",
    ),
    default="walk_easy_rebalanced",
    help="Must match what the checkpoint was actually trained under -- see the checkpoint-selection "
    "protocol in STAGE4_PPO_TRAINING.md.",
)
parser.add_argument("--max_steps", type=int, default=None, help="Cap steps; default = full episode length.")
parser.add_argument("--sleep_s", type=float, default=0.0, help="Extra wall-clock delay after each step.")
parser.add_argument("--hold_s", type=float, default=15.0, help="Keep the window open this long after stepping ends.")
parser.add_argument("--obs_normalization", action="store_true", default=True)
parser.add_argument(
    "--push_at_step", type=int, default=None,
    help="Apply a forward-shove push-recovery test at this step (sets root velocity directly, same "
    "mechanism as Isaac Lab's own push_robot event term).",
)
parser.add_argument(
    "--push_vel", type=float, default=1.5,
    help="Forward velocity delta (m/s) applied at --push_at_step. Scale this to the checkpoint's own "
    "trained command range (e.g. op3_walk_isaac_best.pt's lin_vel_x range is only 0.02-0.08 m/s) -- "
    "an oversized push is not a fair recovery test.",
)
parser.add_argument(
    "--cmd_vel", type=float, nargs=3, default=None, metavar=("LIN_X", "LIN_Y", "ANG_Z"),
    help="Override the sampled command with a fixed (lin_vel_x, lin_vel_y, ang_vel_z) held for the "
    "whole run -- e.g. '0.15 0 0' for a clearly-visible forward walk, bypassing the stage's own "
    "(often tiny) default command range. Same override mechanism teleop_play_ppo.py uses, but fixed "
    "from the CLI instead of read from the keyboard -- no need to click into the terminal and press "
    "keys just to see forward motion. Default (unset) keeps the stage's own sampled command.",
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": False})

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
    OP3WalkEasyRebalancedLowUndesiredContactEnvCfg,
    OP3WalkTinyForwardLowUndesiredContactEnvCfg,
)
from op3_rl.ppo_cfg import OP3FlatPPORunnerCfg  # noqa: E402

_STAGE_ENV_CFGS = {
    "stand": OP3StandEnvCfg,
    "stand_legs_only": OP3StandLegsOnlyEnvCfg,
    "walk_easy": OP3WalkEasyEnvCfg,
    "walk_easy_rebalanced": OP3WalkEasyRebalancedEnvCfg,
    "walk": OP3FlatEnvCfg,
    "walk_tiny_forward_low_undesired_contact": OP3WalkTinyForwardLowUndesiredContactEnvCfg,
    "walk_easy_rebalanced_low_undesired_contact": OP3WalkEasyRebalancedLowUndesiredContactEnvCfg,
}
env_cfg = _STAGE_ENV_CFGS[args.stage]()
env_cfg.scene.num_envs = args.num_envs
env_cfg.scene.env_spacing = 2.5
env_cfg.observations.policy.enable_corruption = False

print(f"[view-ppo] launching GUI stage={args.stage} num_envs={args.num_envs} checkpoint={args.checkpoint}")

def fp(*a, **kw):
    print(*a, **kw, flush=True)


fp("[view-ppo] constructing env...")
raw_env = ManagerBasedRLEnv(cfg=env_cfg)
fp("[view-ppo] env constructed, wrapping...")
env = RslRlVecEnvWrapper(raw_env)
fp("[view-ppo] wrapped, building runner...")

agent_cfg = OP3FlatPPORunnerCfg()
if args.obs_normalization:
    agent_cfg.policy.actor_obs_normalization = True
    agent_cfg.policy.critic_obs_normalization = True
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
fp("[view-ppo] runner built, loading checkpoint...")
runner.load(args.checkpoint)
fp("[view-ppo] checkpoint loaded, building inference policy...")
policy = runner.get_inference_policy(device=env.unwrapped.device)
fp("[view-ppo] policy ready.")

robot = raw_env.scene["robot"]
max_steps = args.max_steps or int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
print(f"[view-ppo] max_steps = {max_steps}")
if args.cmd_vel is not None:
    print(f"[view-ppo] overriding command with fixed cmd_vel={tuple(args.cmd_vel)} for the whole run")
command_term = raw_env.command_manager.get_term("base_velocity") if args.cmd_vel is not None else None

obs = env.get_observations()
alive = torch.ones(args.num_envs, dtype=torch.bool, device=raw_env.device)
start_xy = robot.data.root_pos_w[:, :2].clone()

raw_env.sim.render()

for step in range(max_steps):
    if command_term is not None:
        command_term.vel_command_b[:, 0] = args.cmd_vel[0]
        command_term.vel_command_b[:, 1] = args.cmd_vel[1]
        command_term.vel_command_b[:, 2] = args.cmd_vel[2]

    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
    raw_env.sim.render()

    if args.push_at_step is not None and step == args.push_at_step:
        # Same mechanism as Isaac Lab's own `push_robot` event term
        # (isaaclab.envs.mdp.events.push_by_setting_velocity): set the root
        # velocity directly to simulate an impulse, instead of sampling a
        # random one.
        env_ids = torch.arange(args.num_envs, device=raw_env.device)
        vel_w = robot.data.root_vel_w[env_ids].clone()
        vel_w[:, 0] += args.push_vel
        robot.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
        print(f"[view-ppo] step {step:4d}: applied push (forward shove, +{args.push_vel} m/s)")

    terminated = dones.bool()
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    height = robot.data.root_pos_w[:, 2]
    newly_dead = terminated & alive
    alive = alive & ~terminated

    if step % 10 == 0 or newly_dead[0]:
        extra = ""
        if command_term is not None:
            achieved_lin_vel_b = robot.data.root_lin_vel_b[0, :2]
            achieved_ang_vel_z = robot.data.root_ang_vel_b[0, 2]
            displacement_xy = torch.norm(robot.data.root_pos_w[0, :2] - start_xy[0])
            extra = (
                f" cmd=({args.cmd_vel[0]:+.3f},{args.cmd_vel[1]:+.3f},{args.cmd_vel[2]:+.3f}) "
                f"achieved_vel=({achieved_lin_vel_b[0].item():+.3f},{achieved_lin_vel_b[1].item():+.3f},"
                f"{achieved_ang_vel_z.item():+.3f}) net_disp={displacement_xy.item():.3f}m"
            )
        print(
            f"[view-ppo] step {step:4d}: roll={roll[0].item():+.3f} pitch={pitch[0].item():+.3f} "
            f"height={height[0].item():.3f} alive={int(alive.sum().item())}/{args.num_envs}{extra}"
            + (" <-- FELL" if newly_dead[0] else "")
        )

    if not alive.any():
        print(f"[view-ppo] all envs fell by step {step}; holding window for {args.hold_s:.1f}s")
        break

    if args.sleep_s > 0.0:
        time.sleep(args.sleep_s)

end_time = time.time() + args.hold_s
while time.time() < end_time:
    raw_env.sim.render()
    simulation_app.update()
    time.sleep(0.05)

print("[view-ppo] done.")
env.close()
simulation_app.close()
