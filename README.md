# ROBOTIS OP3 Bipedal Locomotion

PPO-based ROBOTIS OP3 walking in simulation, with a quick MuJoCo replay path
and an Isaac Lab training track. This README is the reviewer entry point.

Read these first:

- `FINAL_REPORT.md` - short final result summary.
- `TECHNICAL_SUMMARY.md` - short implementation/training summary.
- `SUBMISSION_FILES.md` - files included in the submitted package.

The long Stage 3 / Stage 4 logs are local history only and are not needed in
the submitted package.

Contents: [Reviewer Path](#reviewer-path) - [At a Glance](#at-a-glance) -
[How The Robot Walks](#how-the-robot-walks) - [RL Debugging](#rl-debugging) -
[Quick Setup](#quick-setup) - [Run PPO Replay](#run-ppo-replay) -
[Keyboard Teleoperation](#keyboard-teleoperation) -
[Push Recovery](#push-recovery) - [Non-RL Fallback](#non-rl-fallback) -
[MuJoCo Training](#mujoco-training) - [Isaac Lab Launch](#isaac-lab-launch) -
[Isaac Training](#isaac-training) - [Isaac Fallback](#isaac-fallback) -
[Robot Models](#robot-models) - [Checkpoint Results](#checkpoint-results) -
[Reward Summary](#reward-summary) -
[Training Duration & Hardware](#training-duration--hardware) -
[External Code / References](#external-code--references) -
[Known Limitations](#known-limitations) - [Repository Map](#repository-map)

## Reviewer Path

Please test in this order:

1. **MuJoCo first** - fastest path, CPU-friendly, includes the final demo
   checkpoint, keyboard teleop, and push recovery.
2. **Isaac Lab second** - task-preferred simulator path, but requires the full
   Isaac Sim / Isaac Lab GPU setup. The launch commands are included in the
   Isaac section below.

The MuJoCo path is the recommended first check because it is the easiest way
to verify the submitted walking policy.

## At a Glance

- Demos (`demo/`, recorded in keyboard-teleop mode --
  see [Keyboard Teleoperation](#keyboard-teleoperation)):
  - `op3_push_recovery_demo.mp4` - 112s, forward walking with manually
    triggered push-recovery shoves.
  - `op3_push_recovery_short_demo.mp4` - 66s, shorter take of the same.
  - `op3_backward_demo.mp4` - 35s, backward command behavior.
- Main checkpoint: `checkpoints/op3_walk_mujoco_best.pt`.
- PPO: `rsl_rl` `OnPolicyRunner`.
- Quick simulator: MuJoCo 3.12.0.
- Preferred-stack track: Isaac Sim 5.1.0 / Isaac Lab v2.3.2.

The demos show forward walking, push recovery (triggered live via the `p` key
in keyboard-teleop mode -- visible in the terminal log as `push applied
(forward shove)`), and backward command behavior. They do not include a
turning-command segment; see "MuJoCo right/opposite turning is weak" under
[Known Limitations](#known-limitations).

## How The Robot Walks

The PPO policy is not replaying a fixed animation. It observes the robot state
each control step and outputs joint-position targets for all 20 actuated OP3
joints. A PD controller tracks those targets in simulation.

Observation vector, 72 dimensions:

- base linear velocity,
- base angular velocity,
- projected gravity / torso orientation,
- commanded velocity,
- joint positions,
- joint velocities,
- previous action.

Action vector, 20 dimensions:

- target joint-position residuals for all OP3 actuated joints.

The policy learned to move by optimizing velocity tracking, survival, upright
torso posture, foot air time, smooth actions, and low torque/jerk.

## RL Debugging

Training curves alone were not trusted. Several checkpoints looked good in
TensorBoard but behaved worse in replay, so final checkpoints were promoted
only after rollout replay measured survival, displacement, velocity tracking,
and fall counts.

Important debugging findings:

- Loose velocity tracking let policies stand still and still score well, so
  the tracking tolerance was tightened.
- The inherited undesired-contact penalty was too harsh for OP3 and punished
  recoverable knee/hip touches, so it was reduced.
- A residual-CPG architecture was tested but plateaued below the plain
  joint-position PPO result.
- MuJoCo PPO produced usable walking but with a visible asymmetric gait.
- Non-RL CPG tuning produced a stable fallback with small forward progress
  and push recovery, but PPO remains the main locomotion result.

## Quick Setup

MuJoCo replay is the easiest reviewer path and does not need a GPU:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
pip install mujoco==3.12.0 tensordict==0.14.0 numpy==1.26.0 gitpython
pip install --no-deps rsl-rl-lib==3.1.2
```

Every command below assumes this venv is active - check the shell prompt
starts with `(.venv)` before running anything. A new terminal does not
inherit the activation from another one; running a script there without
re-running `source .venv/bin/activate` first will silently fall back to
any system-wide `rsl_rl` install and fail with a confusing
`KeyError: 'actor'` deep in `rsl_rl`'s runner construction, not an
obvious "wrong environment" error.

Full Isaac setup is documented in `SETUP.md`.

A `Dockerfile` is also included, covering the same MuJoCo replay path in a
container instead of a local venv (the Isaac Lab track is not
containerized - see Known Limitations):

```bash
  docker build -t op3-mujoco .
  docker run --rm op3-mujoco \
    python3 scripts/replay_validate_mujoco.py \
      --checkpoint checkpoints/op3_walk_mujoco_best.pt --num_envs 8 --steps 1000 \
      --lin_vel_x_range -0.2 0.2 --ang_vel_z_range -0.3 0.3 --obs_normalization
```

The replay itself takes about a minute (measured: ~52s) after the model
summary prints, simulating 1000 steps x 8 envs
before the fall/tracking results print at the end.

See the Dockerfile's own header comment for the demo-recording variant. The
container covers numeric replay only - it has no display, so the
interactive live viewer and keyboard teleop (see "Keyboard Teleoperation"
below) need the local venv path instead.

## Run PPO Replay

Numeric replay:

```bash
python3 scripts/replay_validate_mujoco.py \
  --checkpoint checkpoints/op3_walk_mujoco_best.pt \
  --num_envs 8 --steps 1000 \
  --lin_vel_x_range -0.2 0.2 \
  --ang_vel_z_range -0.3 0.3 \
  --obs_normalization
```

Live MuJoCo viewer:

```bash
python3 scripts/view_mujoco_policy.py \
  --checkpoint checkpoints/op3_walk_mujoco_best.pt
```

Default (`--mode script`) alternates forward walking with push-recovery
forward shoves, ending on forward walking (never stops).

## Keyboard Teleoperation

```bash
python3 scripts/view_mujoco_policy.py \
  --checkpoint checkpoints/op3_walk_mujoco_best.pt \
  --mode keyboard
```

Focus the MuJoCo viewer window, then use:

```text
w = forward
s = stop
x = backward
a = turn
d = opposite-turn request
p = push disturbance
q = quit
```

Known teleop limitation: this MuJoCo checkpoint does not reliably turn right
or track the opposite yaw command. `a` is the reliable turning demo; `d` sends
the opposite yaw command but may not produce a clean turn.

## Push Recovery

Scripted PPO push recovery:

```bash
python3 scripts/view_mujoco_policy.py \
  --checkpoint checkpoints/op3_walk_mujoco_best.pt \
  --mode script
```

This scripted sequence includes forward and lateral torso pushes.

Keyboard push recovery:

```bash
python3 scripts/view_mujoco_policy.py \
  --checkpoint checkpoints/op3_walk_mujoco_best.pt \
  --mode keyboard
```

Press `w` to walk, then `p` to apply a forward shove. Press `q` to quit.

**Exact push mechanics** (`push_torso()` in `scripts/view_mujoco_policy.py`):
each `p` press applies a fixed **25N force in the +X (forward) direction**
directly to the torso body (`data.xfrc_applied`), held for 5 simulation
substeps (~0.01s at the default timestep), then released back to zero. It's
always a forward push in the direction of travel -- not sideways, not
backward, and not scaled by anything you set (no CLI flag controls the
magnitude in this script, unlike `record_mujoco_demo.py`'s scripted
sequence, which also includes a lateral push). Each `p` press is
independent -- you can press it repeatedly to stack disturbances while
walking.

Isaac push recovery: see [Isaac Lab Launch](#isaac-lab-launch) below.

## Non-RL Fallback

A non-RL fallback controller is included for comparison. It is a
hand-engineered CPG gait with live roll/pitch feedback. PPO is the stronger
final result, but the fallback now also makes small forward progress.

MuJoCo fallback viewer:

```bash
python3 scripts/view_mujoco_fallback_gait.py
```

MuJoCo fallback keyboard push test:

```bash
python3 scripts/view_mujoco_fallback_gait.py --mode keyboard
```

Focus the MuJoCo window, press `p` to push, and press `q` to quit.

MuJoCo fallback numeric push-recovery test:

```bash
python3 scripts/test_fallback_gait_push_recovery.py \
  --num_envs 8 --max_steps 300 \
  --push_at_step 150
```

Latest local fallback checks:

- Forward progress check: 8/8 survived 300 steps with mean forward
  displacement `+0.191 m`.
- Push-recovery check: 8/8 were alive when the push was applied, 8/8 were
  alive at the end, and 0/8 fell after the push.

This is still weaker and slower than the PPO walking policy, but it is a real
non-RL fallback gait with feedback and push recovery.

Isaac fallback check: see [Isaac Fallback](#isaac-fallback) below.

## MuJoCo Training

Train:

```bash
python3 scripts/train_ppo_mujoco.py \
  --num_envs 64 --max_iterations 60000 \
  --experiment_name my_mujoco_run \
  --lin_vel_x_range -0.2 0.2 \
  --ang_vel_z_range -0.3 0.3 \
  --obs_normalization
```

Resume:

```bash
python3 scripts/train_ppo_mujoco.py \
  --num_envs 64 --max_iterations 100000 \
  --experiment_name my_mujoco_run \
  --lin_vel_x_range -0.2 0.2 \
  --ang_vel_z_range -0.3 0.3 \
  --obs_normalization \
  --checkpoint logs/my_mujoco_run/model_60000.pt
```

## Isaac Lab Launch

Use this path after completing the setup in `SETUP.md`:

```bash
bash scripts/setup_env.sh
cd IsaacLab
```

**Every `./isaaclab.sh` command below must run from inside `IsaacLab/`** --
it isn't in the repo root, so `./isaaclab.sh` on its own fails with
`bash: ./isaaclab.sh: No such file or directory`. Each block below starts
with its own `cd IsaacLab` so it works standalone (e.g. pasted into a fresh
terminal), even if you already ran the `cd` above -- if you're already
inside `IsaacLab/`, drop that block's leading `cd IsaacLab` line or it'll
just fail with the same "No such file or directory" (harmless, since you're
already where it wanted to put you).

Isaac numeric replay:

```bash
cd IsaacLab
./isaaclab.sh -p ../scripts/diagnose_walk_numeric.py \
  --checkpoint ../checkpoints/op3_walk_isaac_best.pt \
  --stage walk_tiny_forward_low_undesired_contact \
  --num_envs 128 --max_steps 300 \
  --obs_normalization
```

Isaac live viewer:

```bash
cd IsaacLab
./isaaclab.sh -p ../scripts/view_ppo_policy.py \
  --checkpoint ../checkpoints/op3_walk_isaac_best.pt \
  --stage walk_tiny_forward_low_undesired_contact \
  --num_envs 1 --max_steps 300 \
  --obs_normalization
```

This will look nearly stationary -- that's expected, not a bug.
`walk_tiny_forward_low_undesired_contact` and
`walk_easy_rebalanced_low_undesired_contact` both hardcode a tiny
`lin_vel_x = (0.02, 0.08)` forward-command range in the environment config
itself (`op3_rl/op3_env_cfg.py`), regardless of which checkpoint you load --
so swapping to `op3_walk_isaac_wide_command.pt` under this same viewer
won't look any different; that checkpoint's "wide" range is lateral
(`lin_vel_y = ±0.1`, not forward speed) and randomized survival difficulty,
not a faster walk. To see forward walking without the keyboard-focus
hassle of the teleop script below, pass `--cmd_vel` to override the
sampled command with a fixed value for the whole run -- no keyboard
interaction needed, works headless too:

```bash
cd IsaacLab
./isaaclab.sh -p ../scripts/view_ppo_policy.py \
  --checkpoint ../checkpoints/op3_walk_isaac_wide_command.pt \
  --stage walk_easy_rebalanced_low_undesired_contact \
  --num_envs 1 --max_steps 600 \
  --cmd_vel 0.15 0 0 \
  --obs_normalization
```

`--cmd_vel LIN_X LIN_Y ANG_Z` also prints `achieved_vel`/`net_disp`
telemetry alongside `cmd` each step, so you can confirm real tracking
numerically instead of eyeballing the viewport -- same fields the teleop
script below prints. Use `op3_walk_isaac_wide_command.pt` here (not
`op3_walk_isaac_best.pt`), same reasoning as the teleop section below.
For interactive control instead of a fixed value, use the teleop script,
which overrides the command from the keyboard live instead of a single
fixed CLI value.

Isaac keyboard teleop:

```bash
cd IsaacLab
./isaaclab.sh -p ../scripts/teleop_play_ppo.py \
  --checkpoint ../checkpoints/op3_walk_isaac_wide_command.pt \
  --stage walk_easy_rebalanced_low_undesired_contact \
  --mode keyboard \
  --headless false
```

**How to make it walk forward:**

1. Wait for the Isaac Sim window to finish loading (`[teleop-play] step=0000 ...` lines start printing in the terminal).
2. Click on the **terminal window** you launched this from -- not the Isaac
   Sim 3D viewport. Keyboard input is read from the terminal's raw stdin
   (`termios`/`tty`), not from clicks inside the graphics window, so typing
   into the viewport (or into an open menu there, e.g. the RTX rendering
   dropdown) does nothing.
3. With the terminal focused, press `w`. That overrides the environment's
   tiny default command (`lin_vel_x` capped at 0.02-0.08 m/s, baked into the
   env config, unrelated to which checkpoint you load) and commands real
   forward motion instead. Other keys: `s` stop, `a`/`d` turn, `q` quit.
4. Check it's actually moving from the printed telemetry, not just the
   viewport: `[teleop-play] step=... cmd=(+0.060,...) achieved_vel=(...)
   net_disp=...m`. `achieved_vel`'s first number should track close to
   `cmd`'s, and `net_disp` (total distance moved) should keep climbing. At
   0.06 m/s the robot covers only ~0.3m over 5 real-time seconds, so it
   will look slow even when it's genuinely tracking -- the numbers are the
   reliable signal, not a glance at the viewport. Add `--max_steps 2000` (or
   higher) if the default 300-step episode ends before you're done watching.

Use `op3_walk_isaac_wide_command.pt` here, not `op3_walk_isaac_best.pt` --
it's the checkpoint actually validated on this stage's lateral/randomized
range, so it tracks an overridden command more reliably. Even so, expect
imperfect tracking under a held command: this checkpoint was trained
against periodically-resampled commands, so holding one value fixed for
hundreds of steps (as this script does) is a harder, out-of-distribution
test -- see `STAGE4_PPO_TRAINING.md` for the measured numbers.

Isaac push recovery:

```bash
./isaaclab.sh -p ../scripts/view_ppo_policy.py \
  --checkpoint ../checkpoints/op3_walk_isaac_best.pt \
  --stage walk_tiny_forward_low_undesired_contact \
  --num_envs 1 --max_steps 300 \
  --push_at_step 100 --push_vel 0.3 --hold_s 10 \
  --obs_normalization
```

Wide-command Isaac checkpoint:

```bash
./isaaclab.sh -p ../scripts/diagnose_walk_numeric.py \
  --checkpoint ../checkpoints/op3_walk_isaac_wide_command.pt \
  --stage walk_easy_rebalanced_low_undesired_contact \
  --num_envs 128 --max_steps 1000 \
  --obs_normalization
```

## Isaac Training

```bash
bash scripts/setup_env.sh
cd IsaacLab
./isaaclab.sh -p ../scripts/train_ppo.py \
  --stage walk_easy_rebalanced_low_undesired_contact \
  --num_envs 512 --max_iterations 55000 \
  --experiment_name my_isaac_run \
  --learning_rate 3e-4 \
  --dof_torques_weight=-1.0e-4 \
  --obs_normalization
```

## Isaac Fallback

```bash
cd IsaacLab
./isaaclab.sh -p ../scripts/fallback_gait_controller.py \
  --num_envs 8 --max_steps 300
```

## Robot Models

MuJoCo:

- Official ROBOTIS OP3 MuJoCo model from `mujoco_menagerie`.
- Vendored at `third_party/mujoco_menagerie/robotis_op3/`.

Isaac Lab:

- Provided source model:
  `ROBOTIS-OP3-Common/op3_description/urdf/robotis_op3.urdf.xacro`.
- Conversion scripts: `scripts/expand_urdf.sh`,
  `scripts/convert_urdf_to_usd.py`.

## Checkpoint Results

| Checkpoint | Simulator | Result |
|---|---|---|
| `checkpoints/op3_walk_mujoco_best.pt` | MuJoCo | 23/24 rollouts survived 1000 steps; 91.0% forward tracking; 78.2% backward tracking; 24/24 correct forward/backward sign. |
| `checkpoints/op3_walk_isaac_best.pt` | Isaac Lab | 127/128 envs survived the full 1000-step narrow forward-command episode. |
| `checkpoints/op3_walk_isaac_wide_command.pt` | Isaac Lab | Two wide-command replay runs survived at 104/128 and 102/128 envs. |

## Reward Summary

Final reward terms include linear velocity tracking, yaw-rate tracking,
upright torso orientation, low vertical velocity, low roll/pitch angular
velocity, torque and joint-acceleration penalties, action-rate penalty, foot
air-time reward, undesired-contact penalty, alive reward, termination penalty,
and arm/head joint-deviation penalty.

## Training Duration & Hardware

- Isaac Lab: `num_envs=512`, on a single NVIDIA RTX 3500 Ada (laptop GPU),
  trained across many resumed sessions totaling roughly 2 days of
  wall-clock time.
- MuJoCo: `num_envs=64`, CPU-only, trained in parallel with the Isaac Lab
  run on the same machine.
- The GPU hit a recurring hardware fault (Xid 154, "Node Reboot
  Required") roughly every 1-3 hours of sustained training, requiring a
  full machine reboot each time. `scripts/train_ppo_resilient.sh`
  auto-resumes through ordinary process crashes but not this specific
  fault; a reboot was needed to clear it.
- No multi-GPU or cluster training was used.

## External Code / References

- ROBOTIS's official OP3 MuJoCo model (`robotis_mujoco_menagerie`) -
  source of the actuator PD gains (`stiffness`, `damping`,
  `effort_limit_sim`) used on both simulator tracks.
- Isaac Lab's `LocomotionVelocityRoughEnvCfg` (its ANYmal/Cassie-style
  base config) - source of the reward/observation design, reused per
  the task brief's "reuse, don't reinvent" allowance, then retargeted to
  OP3's joints, command range, and reward weights.
- `rsl_rl`'s `OnPolicyRunner` - the PPO implementation used directly on
  both tracks, not reimplemented.

## Known Limitations

- The MuJoCo PPO gait has visible left/right asymmetry: one leg tends to lead
  while the other trails. This was measured as contact-load asymmetry, not just
  a visual impression. Fixing it properly likely needs retraining with a better
  alternation/contact reward.
- MuJoCo right/opposite turning is weak, and this isn't limited to
  commanded turns: the policy has a systematic positive-yaw bias, so it
  drifts/turns slowly even under a straight-forward (zero yaw-rate)
  command. Measured, not a per-episode fluke -- `replay_validate_mujoco.py`
  found every achieved yaw velocity coming out positive regardless of
  commanded sign, and a 64-env zero-command check on the Isaac Lab wide
  checkpoint found the same one-directional bias (mean +0.0579 rad/s,
  80% of envs net-positive, all 64 nonzero). Root cause unknown (data
  imbalance, reward-shaping artifact, or something else); fixing it
  properly likely needs more training/reward tuning, not just replay.
- Isaac Lab is heavier to reproduce because it requires the full GPU Isaac Sim
  / Isaac Lab stack.
- MuJoCo viewer shutdown may print a GLX/GLFW segfault after the viewer closes.
  This is viewer cleanup instability, not a policy-load failure.

## Repository Map

| Path | Contents |
|---|---|
| `op3_rl/` | Environments, robot config, reward helpers |
| `scripts/` | Training, replay, teleop, push tests, fallback |
| `checkpoints/` | Exported replay-validated checkpoints |
| `demo/` | Final demo video |
| `ROBOTIS-OP3-Common/` | Provided OP3 source model |
| `third_party/mujoco_menagerie/robotis_op3/` | Vendored MuJoCo OP3 model |
| `FINAL_REPORT.md` | Short final result |
| `TECHNICAL_SUMMARY.md` | Short technical/training summary |
| `SUBMISSION_FILES.md` | Submission checklist |
