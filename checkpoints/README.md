# Exported Checkpoints

Not "latest" checkpoints — best *replay-validated* ones. Raw training-curve
peaks are not reliable (this project found real instances of reward/episode
length climbing for thousands of iterations then declining again with
continued training). Every checkpoint here was verified by replaying it
against real behavior (falls, displacement, velocity tracking), not just
picked by iteration number. See `STAGE4_PPO_TRAINING.md` for the full
evidence trail.

## `op3_walk_mujoco_best.pt`

- Source: `logs/op3_mujoco_walk_symmetric_fresh_finetune/model_59999.pt`
  (the fine-tune run's final checkpoint — full 60000/60000 iterations
  completed cleanly, no peak-then-decline: `Train/mean_reward` and
  `Train/mean_episode_length` climbed steadily the entire fine-tune
  segment from iteration 43850 to 59999, episode length reaching 995/1000
  at the end).
- Simulator: MuJoCo (`op3_rl/mujoco_env.py`), not Isaac Lab.
- Validated (3 seeds, 8 envs each, 1000-step rollouts,
  `scripts/replay_validate_mujoco.py`): forward-command tracking ratio
  **91.0%** (13 samples), backward **78.2%** (11 samples), sign-matching
  **24/24**, **23/24 rollouts survived the full episode** (1 fall, at
  step 600, one seed). Supersedes the previous best
  (`model_33100.pt`: forward 64.8%, backward 54.8%, sign 21/22, 24/24
  survived) — meaningfully better tracking on both axes and perfect sign
  accuracy, at the cost of one fall out of 24 rollouts instead of zero.
  Net improvement; promoted 2026-09-06.
- Load with `rsl_rl`'s `OnPolicyRunner.load()` against
  `op3_rl/mujoco_env.py::OP3MujocoVecEnv`, `actor_obs_normalization=True`
  (trained with `--obs_normalization`).
- Demo videos recorded from this checkpoint in keyboard-teleop mode:
  `../demo/op3_push_recovery_demo.mp4` (112s, forward walking + manually
  triggered forward-shove push recovery), `../demo/op3_push_recovery_short_demo.mp4`
  (66s, shorter take of the same), and `../demo/op3_backward_demo.mp4`
  (35s, backward command).

## `op3_walk_isaac_best.pt`

- Source: `logs/op3_walk_tiny_forward_low_undesired_contact/model_24500.pt`
- Simulator: Isaac Lab (`op3_rl/op3_env_cfg.py`,
  `--stage walk_tiny_forward_low_undesired_contact`).
- Validated three separate ways (`scripts/diagnose_walk_numeric.py`):
  128/128 envs survive a full 300-step episode (confirmed twice,
  independent runs), and 127/128 survive to the natural 1000-step episode
  timeout on a third, longer run — only 1 genuine fall (at step 815) out
  of 128 envs over the full 20-second episode. This is the best-validated
  Isaac checkpoint in the project by a wide margin (previous best:
  81/128, 63%).
- Training on this lineage continued past this checkpoint and declined
  again (a third instance of the same peak-then-decline pattern, this
  time from 913 down to a 410-595 band by iteration ~29700) — deliberately
  **not** resumed again after that; this exported checkpoint is the
  intended final answer for this lineage, not a snapshot mid-improvement.
  See `STAGE4_PPO_TRAINING.md`'s "Isaac's third peak-then-decline cycle"
  section for the full validation numbers and reasoning.
- Load with `rsl_rl`'s `OnPolicyRunner.load()` against
  `op3_rl/op3_env_cfg.py::OP3WalkTinyForwardLowUndesiredContactEnvCfg`,
  `actor_obs_normalization=True`.

## `op3_walk_isaac_wide_command.pt`

- Source: `logs/op3_walk_easy_rebalanced_low_undesired_contact/model_99999.pt`
  (the lineage's own natural training-budget end — extended in stages
  from an original 30000-iteration run up through 55000, then 100000,
  replay-validated at each stopping point rather than trusted blindly;
  see `STAGE4_PPO_TRAINING.md` for the full extension history)
- Simulator: Isaac Lab, `--stage walk_easy_rebalanced_low_undesired_contact`
  — the full `lin_vel_x=(-0.2,0.2)`, `lin_vel_y=(-0.1,0.1)`,
  `ang_vel_z=(-0.3,0.3)` command range (matching MuJoCo's range), not
  `op3_walk_isaac_best.pt`'s narrow forward-only `(0.02,0.08)` range.
  Warm-started from `op3_walk_isaac_best.pt` itself.
- Validated (128 envs, 1000-step rollouts, `scripts/diagnose_walk_numeric.py`,
  **two independent runs**): **104/128 (81.3%)** and **102/128 (79.7%)**
  genuinely survive to the natural episode end (the apparent "128/128
  fallen" at step 999 is the episode-timeout boundary itself, not a real
  collapse). Supersedes the previous export from this same lineage
  (`model_29999.pt`: 62% survival) — a real, reproducible improvement
  from continued training, not noise between two consistent runs.
  Promoted 2026-09-07.
- This is a different, complementary result to `op3_walk_isaac_best.pt`,
  not a replacement: still somewhat lower survival than that narrow-
  command checkpoint's 99%, but capable of the full velocity/yaw command
  range instead of one narrow forward speed — and validated for turning
  and lateral movement specifically (`STAGE4_PPO_TRAINING.md`'s
  teleop-script sections).
- Load with `rsl_rl`'s `OnPolicyRunner.load()` against
  `op3_rl/op3_env_cfg.py::OP3WalkEasyRebalancedLowUndesiredContactEnvCfg`,
  `actor_obs_normalization=True`.

## `op3_stand_best.pt`

- Source: `logs/op3_stand_lr3e4/model_7000.pt`
- Stage: `stand` (zero velocity command, survival-focused curriculum stage)
- Validated: 11.7% danger-zone failure rate (15/128 envs fallen by step 50,
  `diagnose_walk_numeric.py --stage stand --num_envs 128 --max_steps 150`)
- Trained with `learning_rate=3e-4` (not this project's default `1e-3`) —
  see `STAGE4_PPO_TRAINING.md` for why this matters: the lower learning
  rate meaningfully reduced checkpoint-to-checkpoint variance versus the
  default, which swung as widely as 9%-37.5% failure rate with no
  curriculum change at all.
- Load with `rsl_rl`'s `OnPolicyRunner.load()`, matching
  `scripts/play_ppo.py`'s pattern. Requires `op3_rl/op3_env_cfg.py::OP3StandEnvCfg`
  and `op3_rl/ppo_cfg.py::OP3FlatPPORunnerCfg` for the matching
  observation/action/network shapes.

## `op3_walk_pre_reward_v2_iter21450.pt`

Preserved backup of a pre-reward-v2 run's state, not a "best" claim — kept
for reference. See `STAGE4_PPO_TRAINING.md` for the context this was
backed up under.

Will be updated as better-validated checkpoints are found — see
`STAGE4_PPO_TRAINING.md` for the current state of walk-stage training.
