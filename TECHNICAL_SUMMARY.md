# Technical Summary

This is the short replacement for the long stage logs. It summarizes the
implementation and training evidence without requiring the reviewer to read
`STAGE3_PPO_SMOKE_TEST.md` or `STAGE4_PPO_TRAINING.md`.

## Pipeline Smoke Test

The Isaac Lab PPO pipeline was verified end-to-end before full training:

- ROBOTIS OP3 articulation loaded into an Isaac Lab physics scene.
- A manager-based locomotion environment was constructed successfully.
- `rsl_rl` PPO built actor/critic networks with 72 observation inputs and 20
  action outputs.
- The reward terms, fall termination, command sampling, and observation/action
  wiring all produced valid logged values.

This confirmed that the environment, robot articulation, PPO runner, and task
configuration were connected correctly before longer training runs.

## Final PPO Setup

- PPO implementation: `rsl_rl` `OnPolicyRunner`.
- Policy action: 20 joint-position targets, covering all OP3 actuated joints.
- Control: PD-tracked joint-position commands.
- Observation: 72-dimensional proprioceptive vector:
  base velocity, angular velocity, projected gravity, commanded velocity,
  joint positions, joint velocities, and previous actions.
- Simulators:
  - Isaac Lab / Isaac Sim 5.1.0 for the task-preferred training track.
  - MuJoCo 3.12.0 for the reproducible CPU-friendly track and final video.

## Reward Design

The final reward uses standard locomotion terms:

- forward/lateral velocity tracking,
- yaw-rate tracking,
- upright torso orientation,
- low vertical velocity,
- low roll/pitch angular velocity,
- torque and joint-acceleration penalties,
- action-rate smoothing,
- foot air-time reward,
- undesired body-contact penalty,
- alive reward,
- termination penalty on falls,
- arm/head joint-deviation penalty.

Two important fixes were found during training:

- Velocity-tracking tolerance had to be tightened because the inherited default
  rewarded near-zero movement too much for OP3's small target speeds.
- The undesired-contact penalty was too harsh for a small humanoid and
  discouraged recoverable knee/hip contacts; reducing it improved survival.

## Best Results

| Checkpoint | Result |
|---|---|
| `checkpoints/op3_walk_mujoco_best.pt` | 23/24 rollouts survived 1000 steps; 91.0% forward tracking; 78.2% backward tracking; 24/24 correct forward/backward sign. |
| `checkpoints/op3_walk_isaac_best.pt` | 127/128 envs survived the full 1000-step episode on the narrow forward-command task. |
| `checkpoints/op3_walk_isaac_wide_command.pt` | 104/128 and 102/128 envs survived in two independent wide-command replay runs. |

The main visual evidence is the `demo/*.mp4` clips (`op3_push_recovery_demo.mp4`,
`op3_push_recovery_short_demo.mp4`, `op3_backward_demo.mp4`), recorded from the
MuJoCo checkpoint in keyboard-teleop mode.

## Recovery Evidence

Recovery is demonstrated in two ways:

- During training, randomized pushes and randomized initial conditions were
  included in the Isaac Lab environment.
- During evaluation, the MuJoCo demo applies push disturbances to the torso;
  the robot recovers and continues walking.

## Negative Results And Limitations

- Several checkpoints looked good on training curves but were worse under
  replay, so final checkpoint selection is based on replay validation rather
  than TensorBoard alone.
- A residual-CPG action architecture was implemented and tested, but it
  plateaued below the plain joint-position PPO checkpoints and was not used as
  the final result.
- Non-RL CPG sign/phase variants were also tested. The best promoted fallback
  variant survived 8/8 envs for 300 steps with mean forward displacement
  `+0.191 m`, and separately survived a push-recovery test with 8/8 alive at
  the push, 8/8 alive at the end, and 0/8 falling after the push. This is
  slower and weaker than PPO, but it is a real feedback fallback rather than
  a fixed animation.
- MuJoCo turning is not fully bidirectional; it tends to turn one direction
  more reliably.
- The MuJoCo gait has measurable left/right load asymmetry.
- Isaac Lab is heavier to reproduce because it requires the Isaac Sim GPU
  stack, while MuJoCo is the simplest reproduction path.
