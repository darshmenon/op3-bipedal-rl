# Concepts

A reference for the RL/simulation concepts behind this project and how each
one is actually implemented here, with file pointers. The `STAGE*.md` files
are the chronological run log (what was tried, what broke, what changed);
this doc is the concept map that ties it together. See
[`RL_BIPEDAL_LOCOMOTION_CANDIDATE_TASK.md`](RL_BIPEDAL_LOCOMOTION_CANDIDATE_TASK.md)
for the original task brief this project is answering.

## The RL problem framing

OP3 locomotion is posed as a single-agent MDP, one instance per parallel
environment (`isaaclab.envs.ManagerBasedRLEnv`, built by
`op3_rl.op3_env_cfg.OP3FlatEnvCfg`):

- **State/observation**: a 72-dim proprioceptive vector (see below) — no
  privileged world-frame info at inference time, per the task brief.
- **Action**: 20 target joint positions, tracked by a PD controller.
- **Reward**: a weighted sum of terms encouraging forward velocity tracking
  while discouraging instability (see below).
- **Termination**: episode ends on falling (torso contact) or a 20s timeout.
- **Reset**: randomized per-episode (pose, joint angles, velocity command),
  so the policy never sees the exact same starting condition twice.

Isaac Lab runs thousands of copies of this MDP in parallel on the GPU
(`scene.num_envs`), which is what makes PPO practical here — one iteration
collects `num_envs x num_steps_per_env` transitions in a single batched
rollout rather than one trajectory at a time.

## PPO (Proximal Policy Optimization)

**What it is**: an on-policy actor-critic RL algorithm. It collects a batch
of trajectories under the current policy, then takes several gradient steps
on that batch using a *clipped* objective that keeps each update close to
the old policy — this is what makes PPO more stable than vanilla policy
gradient without needing a second-order trust-region solve (as in TRPO).

**Where it lives in this repo**: `rsl_rl.runners.OnPolicyRunner`
(third-party, from the `rsl_rl` package), configured by
`op3_rl/ppo_cfg.py::OP3FlatPPORunnerCfg`. This project does not implement
PPO itself — per the task brief's "reuse, don't reinvent" allowance, it
reuses Isaac Lab's standard RSL-RL integration, matching what
Cassie/ANYmal/Unitree examples in `isaaclab_tasks` use.

**Key pieces, mapped to this project's config**:

| PPO concept | What it does | This project's setting |
|---|---|---|
| Actor-critic network | Separate MLPs producing the action distribution and a value estimate | `actor_hidden_dims=[128,128,128]`, `critic_hidden_dims=[128,128,128]`, ELU activation |
| Rollout length | Steps collected per env before an update | `num_steps_per_env=24` |
| GAE (`lam`) | Generalized Advantage Estimation — trades off bias/variance in the advantage signal | `lam=0.95`, `gamma=0.99` |
| Clipped surrogate objective | Caps how far a policy update can move per step | `clip_param=0.2` |
| Value loss clipping | Same idea applied to the critic | `use_clipped_value_loss=True`, `value_loss_coef=1.0` |
| Entropy bonus | Extra reward for keeping the action distribution wide (encourages exploration) | `entropy_coef=0.0` — deliberately disabled; see below |
| Learning rate schedule | Adapts step size to the observed KL divergence | `schedule="adaptive"`, `desired_kl=0.01`, `learning_rate=1e-3` |
| Epochs / minibatches per update | How many passes over each rollout batch | `num_learning_epochs=5`, `num_mini_batches=4` |
| Gradient clipping | Caps gradient norm to avoid destructive updates | `max_grad_norm=1.0` |
| Action noise (`init_noise_std`) | Std-dev of the Gaussian action distribution at init, shrinks as the policy converges | `init_noise_std=1.0` |

**Why `entropy_coef=0.0`, not the Cassie-default `0.01`**: entropy bonus
rewards exploration, but on this task it caused *runaway* exploration —
action noise std climbed monotonically instead of shrinking, and episode
length fell in lockstep (a noisier policy falls faster). See
`STAGE4_PPO_TRAINING.md`'s "Fix 2" for the run-by-run evidence
(`op3_flat_v3`/`op3_flat_v3_long` vs. `op3_flat_v4`).

## Observation space (72-dim, `op3_rl/op3_env_cfg.py` + Isaac Lab's `LocomotionVelocityRoughEnvCfg`)

| Term | Dims | Noise (Uniform) | Notes |
|---|---|---|---|
| `base_lin_vel` | 3 | ±0.1 | base-frame linear velocity |
| `base_ang_vel` | 3 | ±0.2 | base-frame angular velocity |
| `projected_gravity` | 3 | ±0.05 | gravity vector in base frame — a proxy for torso orientation |
| `velocity_commands` | 3 | none | the commanded (vx, vy, yaw-rate) the policy should track |
| `joint_pos` | 20 | ±0.01 | relative to default pose |
| `joint_vel` | 20 | ±1.5 | relative to default pose |
| `last_action` | 20 | none | previous policy output, for smoothness |

`height_scan` (present in the upstream rough-terrain base class) is
explicitly removed (`self.observations.policy.height_scan = None`) since
this task is flat-terrain only. Corruption/noise is enabled during training
(`enable_corruption=True`) and disabled in the playback variant
(`OP3FlatEnvCfg_PLAY`) for a clean evaluation signal.

## Action space

`mdp.JointPositionActionCfg(joint_names=[".*"], scale=0.3, use_default_offset=True)`
— all 20 joints (12 leg + 6 arm + 2 head), controlled simultaneously. Each
action is a *delta* added to the joint's default pose and scaled by 0.3,
then tracked by the actuator's implicit PD controller (not a raw torque or
velocity command). Controlling all 20 rather than just the 12 leg joints
(the task brief's minimum) lets the policy use arm/head motion for balance
if useful, softly discouraged from drifting too far via
`joint_deviation_arms_head` (see reward table).

## Reward design (`op3_rl/op3_env_cfg.py::OP3RewardsCfg`)

Reused from Isaac Lab's `LocomotionVelocityRoughEnvCfg.RewardsCfg`
wholesale, with OP3-specific body/joint retargeting and two additions.
Total reward per step is the weighted sum of all rows:

| Term | Weight | Source | Purpose |
|---|---|---|---|
| `track_lin_vel_xy_exp` | +1.0 | base | reward for matching commanded forward/lateral velocity |
| `track_ang_vel_z_exp` | +0.5 | base | reward for matching commanded yaw rate |
| `lin_vel_z_l2` | -2.0 | base | penalize vertical bobbing |
| `ang_vel_xy_l2` | -0.05 | base | penalize roll/pitch angular velocity |
| `dof_torques_l2` | -1e-5 | base | energy/torque penalty |
| `dof_acc_l2` | -2.5e-7 | base | penalize jerky joint accelerations |
| `action_rate_l2` | -0.01 | base | penalize large frame-to-frame action changes |
| `feet_air_time` | +0.125 | base, retargeted to `.*ank_roll_link` | rewards alternating stepping rhythm over shuffling |
| `undesired_contacts` | -1.0 | base, retargeted to `.*(hip_pitch\|knee)_link` | penalize non-foot links touching the ground |
| `flat_orientation_l2` | **-2.5** (OP3 override, base default is 0.0) | overridden | penalize non-upright torso — turned on and weighted up for OP3 |
| `dof_pos_limits` | 0.0 | base | joint-limit penalty (present, currently inert) |
| `termination_penalty` | **-200.0** (OP3-added) | new | large one-off penalty for falling, on top of the episode simply ending |
| `joint_deviation_arms_head` | **-0.1** (OP3-added) | new | soft pull of arm/head joints toward their neutral pose, since they aren't required for walking |

This matches the task brief's suggested term list closely (velocity
tracking, orientation, smoothness, torque penalty, termination penalty)
without inventing new reward shaping — see `STAGE3_PPO_SMOKE_TEST.md`'s
"Design: reuse, don't reinvent" section for the rationale.

## Update: the winning lineage turned out to be simpler than this section

This section (residual-CPG action architecture) was this project's
"opinionated" bet for the highest-upside path when written. It was
implemented and tried (`walk_tiny_forward_cpg_residual` and several
variants including phase-observation and fine-physics combinations) and
plateaued at episode length 23-33 -- far worse than what actually ended up
working. The checkpoint this project's `checkpoints/op3_walk_isaac_best.pt`
and `checkpoints/op3_walk_mujoco_best.pt` come from instead use the plain
(non-residual, non-CPG) `OP3WalkTinyForwardEnvCfg`/`OP3WalkEasyRebalancedEnvCfg`
reward set (the table below, *without* the extra `base_height_l2`/
`feet_slide` terms this specific historical variant added), warm-started
from a converged stand-stage checkpoint, plus one fix on top: cutting
`undesired_contacts` from Isaac Lab's inherited `-1.0` to `-0.2` (see
`STAGE4_PPO_TRAINING.md`'s "Overnight run" section for why and the result).
Keeping this section for the historical record and because the analysis
in it is still accurate -- just noting the actual winning path was
different from this section's prediction.

## Final recommended reward table (`walk_tiny_forward_cpg_residual`, historical -- see update above)

For the current highest-upside training path, keep the reward deliberately
simple and debuggable. The gait prior lives in the action term, not in an
extra reward. The residual-CPG stage uses `OP3WalkTinyForwardRecoveryEnvCfg`
rewards plus the CLI torque override used in the recommended command:
`--dof_torques_weight=-1.0e-4`.

Formula:

```text
r_total =
  3.0  * track_lin_vel_xy_exp
+ 1.5  * track_ang_vel_z_exp
+ 0.5  * alive
+ 0.125 * feet_air_time
- 50.0 * is_terminated
- 6.0  * base_height_l2
- 6.0  * flat_orientation_l2
- 2.0  * lin_vel_z_l2
- 1.0  * undesired_contacts
- 0.25 * ang_vel_xy_l2
- 0.10 * feet_slide
- 0.10 * joint_deviation_arms_head
- 0.08 * action_rate_l2
- 1.0e-4 * dof_torques_l2
- 2.5e-7 * dof_acc_l2
+ 0.0  * dof_pos_limits
```

| Term | Weight | Debug meaning |
|---|---:|---|
| `track_lin_vel_xy_exp` | +3.0 | Forward velocity tracking; should rise if it actually walks forward |
| `track_ang_vel_z_exp` | +1.5 | Yaw-rate tracking; mostly neutral in tiny-forward runs because yaw command is zero |
| `alive` | +0.5 | Per-step survival signal |
| `feet_air_time` | +0.125 | Alternating stepping signal; if it stays zero, there is no real liftoff |
| `termination_penalty` | -50.0 | One-off fall penalty |
| `base_height_l2` | -6.0 | Penalizes crouch/collapse away from target torso height |
| `flat_orientation_l2` | -6.0 | Penalizes torso roll/pitch tilt |
| `lin_vel_z_l2` | -2.0 | Penalizes vertical hopping/bobbing |
| `undesired_contacts` | -1.0 | Penalizes non-foot contacts such as hip/knee links |
| `ang_vel_xy_l2` | -0.25 | Penalizes roll/pitch angular velocity |
| `feet_slide` | -0.10 | Penalizes foot slip while in contact |
| `joint_deviation_arms_head` | -0.10 | Keeps arms/head near nominal pose unless useful for balance |
| `action_rate_l2` | -0.08 | Penalizes frame-to-frame residual-action changes |
| `dof_torques_l2` | -1.0e-4 | Torque/energy penalty; stronger than the inherited base value |
| `dof_acc_l2` | -2.5e-7 | Joint acceleration smoothness penalty |
| `dof_pos_limits` | 0.0 | Present but disabled |

Do not add CPG-imitation or foot-load-asymmetry rewards to this stage
initially. The CPG is already baked into the action, so those terms would
make it harder to tell whether success came from the residual architecture
or another reward-side lever.

## Recovery requirement for the residual-CPG route

The residual-CPG stage is not acceptable as a pure open-loop gait. The CPG
only supplies nominal stepping motion; the deliverable must show that the
policy or fallback controller closes the loop using proprioceptive feedback.

For PPO, recovery should be demonstrated by at least one of these before
submission:

| Recovery test | How to run/measure | Pass signal |
|---|---|---|
| Small reset tilt | Evaluate with small randomized initial roll/pitch or reset velocity | robot returns to bounded torso roll/pitch and keeps stepping |
| Push replay | Enable/evaluate `push_robot` or an external impulse during playback | robot takes corrective steps instead of immediate fall |
| Command change | Switch forward command between stop and tiny-forward walking | policy settles without collapse |
| Stochastic policy replay | Compare deterministic and stochastic action modes | recovery is not only a deterministic playback artifact |

For fallback/demo mode, a hand-scripted CPG without feedback is not enough.
Use the existing `scripts/fallback_gait_controller.py` balance terms
(`roll_kp`, `pitch_kp`, lateral shift) and document the disturbance test
that it survives. If neither PPO nor fallback can recover from a small
disturbance, the honest submission path is to label recovery as an unmet
limitation rather than presenting open-loop stepping as complete.

## Actuator model: implicit PD

Each joint is driven by Isaac Lab's `ImplicitActuatorCfg` — the action is a
target position, and the simulator's solver computes the torque needed to
track it via `torque = stiffness * (target - pos) - damping * vel`,
clamped to `effort_limit_sim`. This is the standard "PD control" scheme
referenced throughout the task brief, distinct from commanding raw torques
directly.

Current values (`op3_rl/op3_robot_cfg.py`, Revision 5): `stiffness=21.1`,
`damping=1.084`, `effort_limit_sim=5.0`, `armature=0.045`, `friction=0.03`,
uniform across all 20 joints — sourced from ROBOTIS's own official MuJoCo
model (`robotis_mujoco_menagerie/robotis_op3/op3.xml`), not guessed. See
`STAGE4_PPO_TRAINING.md`'s "Sourcing PD gains from real Dynamixel specs"
for the full trail of what was tried before landing here (generic xacro
limits → recalled datasheet numbers → this).

## Domain randomization & recovery (task brief's "recovery" requirement)

All inherited unchanged from Isaac Lab's default `EventCfg` — not added
specially for OP3, since the defaults already satisfy the brief:

| Event | Mode | What it randomizes |
|---|---|---|
| `physics_material` | startup | foot/body friction and restitution |
| `add_base_mass` / `base_com` | startup | torso mass and center-of-mass offset |
| `reset_base` | on reset | initial pose (incl. yaw) and velocity |
| `reset_robot_joints` | on reset | initial joint positions |
| `push_robot` | every 10-15s during an episode | a velocity-kick disturbance |
| `base_velocity` command | per-episode | a freshly sampled target velocity to track |

This covers the brief's recovery requirement via randomized pushes during
training and randomized initial pose/velocity. Not implemented: explicit
actuator-latency or motor-strength randomization (not required given the
above already satisfies the "and/or" list in the brief).

## Current open problem: static balance, not a training-duration issue

A standalone diagnostic (`scripts/diagnose_static_balance.py` — no policy,
zero action, zero disturbance, just PD holding the configured standing
pose) found the robot falls over on its own within ~2.1s, 64/64 envs. This
means the ~20-40 step episode-length ceiling seen across every training run
so far is a **physical/config limit** (candidates: foot contact geometry,
COM/support-polygon mismatch, or underdamped PD at this timestep), not
something more PPO iterations will fix on their own. See
`STAGE4_PPO_TRAINING.md`'s "Root-cause investigation" section for the full
regression analysis and reasoning for ruling out PD gain magnitude as the
cause. This is the most important open item before further training time
is worth investing heavily.

Implementation update: the inherited Isaac locomotion startup randomization
was too aggressive for OP3 (`(-5, 5)` kg mass noise on a ~3.147 kg robot, plus
large COM noise), so `OP3FlatEnvCfg` now disables those terms by default.
`OP3StandEnvCfg` adds a stand-first curriculum stage: zero velocity command,
zero reset velocity/pose perturbation, no pushes, smaller residual joint
actions, survival reward, and stronger upright/low-motion penalties. Use it
before velocity-tracking walking:

```bash
isaaclab.sh -p scripts/train_ppo.py --stage stand --num_envs 512 --max_iterations 300
```

Or with the auto-resume wrapper:

```bash
scripts/train_ppo_resilient.sh op3_stand 512 300 10 stand
```

## Where each file fits

| File | Concept |
|---|---|
| `op3_rl/op3_robot_cfg.py` | Robot asset + actuator (PD) model |
| `op3_rl/op3_env_cfg.py` | MDP definition: observations, actions, rewards, terminations, events |
| `op3_rl/ppo_cfg.py` | PPO hyperparameters (`rsl_rl`) |
| `scripts/train_ppo.py` | Train/resume entry point |
| `scripts/train_ppo_resilient.sh` | Auto-resume wrapper around `train_ppo.py` |
| `scripts/play_ppo.py` | Checkpoint playback / evaluation, frame capture |
| `scripts/teleop_play_ppo.py` | Scripted or keyboard forward/stop/turn command playback for bonus/demo checks |
| `scripts/diagnose_static_balance.py`, `diagnose_reset_perturbation.py` | Standalone diagnostics isolating the balance problem from the RL loop |

## External research concepts

Blogs, GitHub repos, papers, and cross-project concept tables live in
`local_research/RELATED_BLOGS_RESEARCH.md` (gitignored). Quick bullets:
`local_research/CONCEPTS_SUMMARY.md` (gitignored). High-priority ideas from that research
not yet in this codebase: **phase-clock observations**, **observation
stacking**, **4096-env / lower walk LR sweeps**, **sim2sim MuJoCo replay**
before trusting Isaac training curves.
