# Final Report

## Result

This submission implements PPO-based ROBOTIS OP3 bipedal locomotion in
simulation, with two validated tracks:

- **MuJoCo**: best reviewer path for quick reproduction, live viewing, and the
  recorded demo.
- **Isaac Lab / Isaac Sim 5.1.0**: task-preferred stack, trained and
  numerically replay-validated, but heavier to reproduce because it requires
  the Isaac Sim / Isaac Lab GPU setup.

The demos are (recorded in keyboard-teleop mode, with push-recovery shoves
triggered live via the `p` key):

```text
demo/op3_push_recovery_demo.mp4         (112s, forward walking + push recovery)
demo/op3_push_recovery_short_demo.mp4   (66s, shorter take of the same)
demo/op3_backward_demo.mp4              (35s, backward command behavior)
```

## Best Checkpoints

| Checkpoint | Simulator | Summary |
|---|---|---|
| `checkpoints/op3_walk_mujoco_best.pt` | MuJoCo | Main demo checkpoint. Replay-validated over 24 rollouts: 23/24 survived 1000 steps, 91.0% forward tracking, 78.2% backward tracking, 24/24 correct forward/backward sign. |
| `checkpoints/op3_walk_isaac_best.pt` | Isaac Lab | Best narrow forward-command result. 127/128 envs survived the full 1000-step episode in numeric replay. |
| `checkpoints/op3_walk_isaac_wide_command.pt` | Isaac Lab | Wider command-range checkpoint. Two independent 128-env replay runs survived at 104/128 and 102/128. |

## Quick Reproduction

MuJoCo numeric replay, CPU-only:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
pip install mujoco==3.12.0 tensordict==0.14.0 numpy==1.26.0 gitpython
pip install --no-deps rsl-rl-lib==3.1.2

python3 scripts/replay_validate_mujoco.py \
    --checkpoint checkpoints/op3_walk_mujoco_best.pt \
    --num_envs 8 --steps 1000 \
    --lin_vel_x_range -0.2 0.2 \
    --ang_vel_z_range -0.3 0.3 \
    --obs_normalization
```

MuJoCo live playback, with a display:

```bash
python3 scripts/view_mujoco_policy.py \
    --checkpoint checkpoints/op3_walk_mujoco_best.pt
```

Isaac Lab setup and training/evaluation commands are documented in
`SETUP.md` and `README.md`.

## Controller

- PPO implementation: `rsl_rl` `OnPolicyRunner`.
- Action space: 20 target joint positions, one per actuated OP3 joint.
- Observation space: 72 dimensions, including base velocity, projected
  gravity, commanded velocity, joint positions, joint velocities, and previous
  action.
- Control mode: joint-position targets tracked by PD control.
- Recovery: randomized pushes during Isaac training, randomized resets, and
  explicit push-recovery segments in the MuJoCo demo.

## Robot Models

- Isaac Lab track uses the provided ROBOTIS OP3 xacro/URDF source under
  `ROBOTIS-OP3-Common/`, converted through `scripts/expand_urdf.sh` and
  `scripts/convert_urdf_to_usd.py`.
- MuJoCo track uses the official ROBOTIS OP3 MuJoCo model from
  `mujoco_menagerie`, vendored under
  `third_party/mujoco_menagerie/robotis_op3/`.

## Main Limitations

- The MuJoCo checkpoint turns in one direction more reliably than the other;
  the demo's turning segment should not be interpreted as full bidirectional
  yaw tracking.
- The Isaac wide-command checkpoint has good randomized-command numeric replay
  survival, but its older scripted turning/lateral validation became stale
  after the checkpoint was promoted.
- The MuJoCo gait is not perfectly symmetric; measured contact-load asymmetry
  remains.
- Isaac Lab reproduction is heavier than MuJoCo because it requires the full
  Isaac Sim / Isaac Lab GPU environment.

## Where To Read More

- `README.md` - main reviewer entry point and complete checklist.
- `SUBMISSION_FILES.md` - exact files/folders to submit or omit.
- `CONCEPTS.md` - observation, action, reward, actuator assumptions.
- `TECHNICAL_SUMMARY.md` - full training history and negative results.
