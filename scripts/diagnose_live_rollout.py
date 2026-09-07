"""Diagnostic (not part of the main training pipeline): resolve the
train-vs-replay discrepancy found in STAGE4_PPO_TRAINING.md by running the
*real* training entry point -- `OnPolicyRunner.learn()`, the exact same
call `scripts/train_ppo.py` makes -- for a short burst resumed from a
checkpoint, while monkey-patching the env's `step()` to inject the same
numeric per-step signals `diagnose_walk_numeric.py` uses (roll/pitch,
height, per-env fall-step distribution).

This lets rsl_rl's own console printer report "Mean episode length" for
this burst using its own unmodified code path, directly alongside a
physically-grounded per-step account of the same steps -- if they disagree,
that's a real bug in metric computation/reporting, not a training-vs-replay
setup difference (this *is* the training code path, just resumed briefly).
If they agree (both show fast falls), that means the ~170 figure logged
near this checkpoint's save point was a real, if perhaps short-lived,
result that a resumed burst from the saved checkpoint doesn't reproduce --
pointing at something save/load-related instead.

Usage:
    isaaclab.sh -p ../scripts/diagnose_live_rollout.py \
        --checkpoint logs/op3_stand_v2/model_2999.pt --num_iterations 5
"""

import argparse
import statistics
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_iterations", type=int, default=5)
parser.add_argument("--stage", choices=("walk", "walk_easy", "stand"), default="stand")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from op3_rl.op3_env_cfg import OP3FlatEnvCfg, OP3StandEnvCfg, OP3WalkEasyEnvCfg  # noqa: E402
from op3_rl.ppo_cfg import OP3FlatPPORunnerCfg  # noqa: E402

_STAGE_ENV_CFGS = {"stand": OP3StandEnvCfg, "walk_easy": OP3WalkEasyEnvCfg, "walk": OP3FlatEnvCfg}
env_cfg = _STAGE_ENV_CFGS[args.stage]()
env_cfg.scene.num_envs = args.num_envs

print(f"[diagnose-live] stage={args.stage} num_envs={args.num_envs} checkpoint={args.checkpoint} "
      f"num_iterations={args.num_iterations}")

raw_env = ManagerBasedRLEnv(cfg=env_cfg)
env = RslRlVecEnvWrapper(raw_env)

robot = raw_env.scene["robot"]
num_envs = env_cfg.scene.num_envs
fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=raw_env.device)
step_counter = {"n": 0}
total_termination_events = {"n": 0}  # counts every termination, including repeats after auto-reset

_real_step = env.step


def _instrumented_step(actions):
    obs, rewards, dones, extras = _real_step(actions)
    step = step_counter["n"]
    step_counter["n"] += 1

    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    height = robot.data.root_pos_w[:, 2]

    terminated = dones.bool()
    total_termination_events["n"] += int(terminated.sum().item())
    newly_dead = terminated & (fall_step < 0)
    fall_step[newly_dead] = step

    if step % 10 == 0:
        n_fallen_total = int((fall_step >= 0).sum().item())
        implied_mean_ep_len = (step + 1) * num_envs / max(total_termination_events["n"], 1)
        print(
            f"[diagnose-live] physics-step {step:4d} env0: roll={roll[0].item():+.3f} "
            f"pitch={pitch[0].item():+.3f} height={height[0].item():.3f} -- "
            f"distinct envs fallen at least once: {n_fallen_total}/{num_envs}, "
            f"total termination EVENTS so far (incl. repeats): {total_termination_events['n']}, "
            f"implied mean ep length from raw termination rate: {implied_mean_ep_len:.1f}"
        )
    return obs, rewards, dones, extras


env.step = _instrumented_step

agent_cfg = OP3FlatPPORunnerCfg()
agent_cfg.max_iterations = args.num_iterations
# log_dir must be truthy for rsl_rl's own console printer (self.log()) to run at
# all -- that printer's "Mean episode length" line is exactly what this script
# needs to compare against. Throwaway dir; checkpoints it writes aren't used.
throwaway_log_dir = str(Path(__file__).resolve().parent.parent / "logs" / "_diagnose_live_rollout_throwaway")
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=throwaway_log_dir, device=agent_cfg.device)
runner.load(args.checkpoint)
start_iteration = runner.current_learning_iteration
print(f"[diagnose-live] loaded checkpoint at iteration {start_iteration}; "
      f"running {args.num_iterations} more learning iterations "
      f"({args.num_iterations * 24} physics steps) with rsl_rl's own unmodified learn() loop...")

runner.learn(num_learning_iterations=args.num_iterations, init_at_random_ep_len=False)

fallen = fall_step[fall_step >= 0]
if fallen.numel() > 0:
    sorted_falls = torch.sort(fallen).values.float()
    median = sorted_falls[fallen.numel() // 2].item()
    print(
        f"[diagnose-live] FINAL: {fallen.numel()}/{num_envs} envs fell during this burst -- "
        f"min={fallen.min().item()} median={median:.0f} mean={fallen.float().mean().item():.1f} "
        f"max={fallen.max().item()}"
    )
else:
    print(f"[diagnose-live] FINAL: 0/{num_envs} envs fell during this burst.")

print("[diagnose-live] Compare the line above against rsl_rl's own printed 'Mean episode length' "
      "for the same iterations, above this line.")
print("[diagnose-live] done.")
env.close()
simulation_app.close()
