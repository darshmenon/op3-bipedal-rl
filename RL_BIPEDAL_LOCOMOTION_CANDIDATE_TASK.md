# One-Week Candidate Task: PPO-Based ROBOTIS OP3 Bipedal Locomotion

## Goal

Train the ROBOTIS OP3 humanoid robot to walk forward on flat terrain in simulation.

The preferred solution is a PPO-based reinforcement learning controller. If PPO training does not converge within the one-week window, a gait-generation or trajectory-optimization fallback is acceptable, but it must include balance feedback or recovery behavior after small disturbances.

The final demo should clearly show the OP3 walking, recovering from minor perturbations, and, as a bonus, being teleoperated from the keyboard.

This task is intentionally based on ROBOTIS OP3 rather than the Blackbird model. The candidate may use any approach, reference implementation, simulator example, paper, or open-source baseline that helps them make progress. They should clearly document what they used and what they changed.

## Provided Robot Model

Use the ROBOTIS OP3 model provided in this repository:

- Main xacro: `ROBOTIS-OP3-Common/op3_description/urdf/robotis_op3.urdf.xacro`
- Gazebo tags: `ROBOTIS-OP3-Common/op3_description/urdf/robotis_op3.gazebo.xacro`
- Transmissions: `ROBOTIS-OP3-Common/op3_description/urdf/robotis_op3.transmissions.xacro`
- Meshes: `ROBOTIS-OP3-Common/op3_description/meshes/`
- Existing Gazebo position controller config: `ROBOTIS-OP3-Common/op3_gazebo/config/position_controller.yaml`

The model uses ROS/xacro package paths such as `package://op3_description/...`. If your simulator does not consume xacro directly, convert it to a simulator-native format and document the conversion process.

## Important Joint Set

The full OP3 model has 20 actuated joints:

- Left leg: `l_hip_yaw`, `l_hip_roll`, `l_hip_pitch`, `l_knee`, `l_ank_pitch`, `l_ank_roll`
- Right leg: `r_hip_yaw`, `r_hip_roll`, `r_hip_pitch`, `r_knee`, `r_ank_pitch`, `r_ank_roll`
- Left arm: `l_sho_pitch`, `l_sho_roll`, `l_el`
- Right arm: `r_sho_pitch`, `r_sho_roll`, `r_el`
- Head: `head_pan`, `head_tilt`

For the minimum task, control at least the 12 leg joints. The arm and head joints may be fixed, controlled to a nominal pose, or included in the policy if useful for balance.

## Platform Requirements

Target Ubuntu 22.04 compatibility.

You may choose the simulation stack, but your setup must be reproducible by the reviewer. A Docker container is acceptable and encouraged if it avoids ROS, simulator, CUDA, or Python version conflicts.

Recommended options:

- Isaac Sim / Isaac Lab: use Isaac Sim 5.1.0 if choosing this route.
- MuJoCo: acceptable if the xacro/URDF is converted cleanly and the setup runs on Ubuntu 22.04.
- Genesis: acceptable if the selected version supports Ubuntu 22.04 and the robot imports correctly.
- Gazebo/Gazebo Sim: acceptable, but document the ROS version and container strategy if needed.

Other simulators may be used only if the final repository is easy to run and the physical behavior is credible.

## Core Technical Requirements

Implement a locomotion controller that:

1. Uses the provided ROBOTIS OP3 morphology.
2. Runs on flat terrain.
3. Commands the robot to move forward.
4. Maintains torso stability without falling immediately.
5. Produces alternating or otherwise plausible bipedal stepping behavior.
6. Includes some form of recovery after small disturbances, such as:
   - randomized pushes during training,
   - external impulse tests during evaluation,
   - randomized initial base orientation,
   - randomized velocity commands,
   - domain randomization over mass, friction, latency, or motor strength.

The existing ROBOTIS OP3 walking/gazebo packages may be used for understanding the model, checking joint names, comparing behavior, or bootstrapping a controller. If any external code, pretrained behavior, or existing gait logic is used, mention it in the README.

## Preferred RL Approach

Use PPO or a PPO-compatible implementation. Examples include:

- Isaac Lab PPO / RSL-RL
- Stable-Baselines3 PPO
- CleanRL PPO
- RL-Games PPO
- Custom PPO, if implemented clearly

The action space may be:

- target joint positions with PD control,
- joint velocity targets,
- joint torque commands,
- residual actions on top of a nominal gait.

For a one-week task, target joint positions with PD control or residual actions on top of a simple gait are usually the most practical.

## Suggested Observation Space

The policy should have access to a compact proprioceptive observation vector. Suggested observations:

- base angular velocity,
- projected gravity or base orientation,
- commanded forward velocity,
- joint positions,
- joint velocities,
- previous actions,
- foot contact states if available,
- gait phase if using a phase-conditioned or residual policy.

Do not rely on privileged world-frame information at inference time unless clearly separated as critic-only information.

## Suggested Reward Terms

Design a reward that encourages walking and discourages unstable motion. A reasonable reward may include:

- forward velocity tracking,
- upright torso orientation,
- low roll/pitch error,
- reasonable base height,
- alternating foot contact or step rhythm,
- foot clearance,
- low foot slip,
- action smoothness,
- torque or energy penalty,
- joint limit penalty,
- survival bonus,
- termination penalty for falling.

Keep the reward simple enough to debug. Include the final reward formula or table of reward terms in your submission.

## Recovery Requirement

The robot must demonstrate some form of recovery, not just open-loop stepping.

At least one of the following is required:

- push recovery during evaluation,
- randomized pushes during training,
- recovery from small random initial body tilt,
- command changes during walking,
- stepping recovery after small lateral/forward disturbances.

Show this in the demo video or provide a separate short recovery clip.

## Minimum Acceptance Criteria

The task is considered successful if the candidate submits:

1. A reproducible repository or compressed package.
2. Clear setup instructions for Ubuntu 22.04 or Docker.
3. A script to train or resume training.
4. A script to run evaluation/playback. (BONUS)
5. A short demo video, 30-90 seconds, showing the OP3 walking on flat terrain.
6. A short README explaining:
   - simulator used,
   - versions used,
   - robot import/conversion steps,
   - actuator assumptions and PD gains,
   - observation/action spaces,
   - reward design,
   - training duration and hardware used,
   - external code or papers referenced,
   - known limitations.

The robot does not need to be perfect, but it should demonstrate real progress toward stable locomotion rather than only falling forward, sliding, or replaying a fixed animation.

## Bonus Criteria

Additional credit for:

- keyboard teleoperation for forward/stop/turn commands,
- command-conditioned policy for different forward speeds,
- turning behavior,
- push recovery demo,
- TensorBoard, Weights & Biases, or equivalent training logs,
- Dockerfile or devcontainer setup,
- exported trained policy checkpoint,
- clean conversion pipeline from xacro/URDF to the simulator format,
- short explanation of sim-to-real considerations.

## Fallback Option

If PPO-based training does not converge in the one-week period, submit the best RL attempt plus a working non-RL fallback.

Acceptable fallback approaches include:

- central pattern generator with balance feedback,
- hand-designed gait with PD stabilization,
- trajectory optimization followed by tracking control,
- model-predictive or heuristic stepping controller,
- residual controller built on top of a nominal gait.

The fallback must still show:

- forward walking or stepping,
- recovery from a small push or initial disturbance,
- clear documentation of the controller.

A purely open-loop gait with no recovery behavior is not sufficient.

## Expected Deliverables

Submit either:

- a GitHub repository link, or
- a compressed archive shared by email.

The submission should include:

- source code,
- setup instructions,
- training/evaluation commands,
- trained policy checkpoint if applicable,
- demo video or video link,
- final README/report.

Do not submit only a video. The reviewer must be able to run or inspect the implementation.


## Notes For The Candidate

Prioritize a working, understandable system over an overly complex one. A simple PPO setup with clean observations, sensible PD control, and a clear reward is better than a large framework that cannot be reproduced.

If you make assumptions about actuator strength, motor velocity limits, friction, control frequency, contact parameters, or ROS/simulator conversions, document them. If you modify the xacro/URDF, keep the modified version in your submission and explain the change.

## Current Implementation Note

This repository now includes the requested PPO route plus a structurally
different residual-gait route for the one-week deadline:

- `scripts/train_ppo.py --stage walk_tiny_forward_cpg_residual` trains a
  20-action PPO policy whose output is a small residual around a live CPG
  joint-position reference, rather than the full joint target around a static
  default pose.
- The residual stage keeps the existing 72-dim observation / 20-dim action
  checkpoint shape, so it can be actor-warm-started from existing walk
  checkpoints while changing the controller interface.
- This stage directly matches the task brief's allowed action space:
  residual actions on top of a nominal gait. It **was** trained (several
  variants: `walk_tiny_forward_cpg_residual_feedback`, `..._phase_obs`,
  etc. — see `STAGE4_PPO_TRAINING.md`), and the approach was superseded by
  the plain residual-on-static-pose action space with the
  `low_undesired_contact` reward fix, which produced every checkpoint
  currently exported in `checkpoints/`. `CONCEPTS.md` documents this
  supersession explicitly rather than deleting the original residual-CPG
  analysis. Not a live recommendation for further work.

## Current Deliverables Status

This section is a snapshot; `README.md`'s "Submission checklist" and
"Bonus checklist" tables are the authoritative, actively-maintained
status trackers — check those for the current state, not this file.

| Deliverable | Current status |
|---|---|
| Reproducible repository or package | Present in this repository; setup scripts reconstruct the Python/Isaac Lab environment |
| Ubuntu 22.04 or Docker setup | Ubuntu 22.04 path documented in `SETUP.md`; `Dockerfile` present for the MuJoCo track (Isaac Lab track not containerized, documented reason in the Dockerfile itself) |
| Train/resume script | Present: `scripts/train_ppo.py` and `scripts/train_ppo_resilient.sh` |
| Evaluation/playback script | Present: `scripts/play_ppo.py`, `scripts/diagnose_walk_numeric.py`, `scripts/evaluate_checkpoint_performance.py` |
| Demo video | Present: `demo/op3_push_recovery_demo.mp4` + `demo/op3_push_recovery_short_demo.mp4` (MuJoCo track, keyboard-teleop) — forward walking with manually triggered push-recovery; `demo/op3_backward_demo.mp4` — backward-command behavior |
| README/report contents | Present across `README.md`, `CONCEPTS.md`, and `STAGE4_PPO_TRAINING.md` |

## Current Bonus Status

| Bonus criterion | Current status |
|---|---|
| Keyboard teleoperation for forward/stop/turn | Helper present: `scripts/teleop_play_ppo.py --mode keyboard`; the command-override code path is validated via `--mode script` (no falls through forward/stop/turn-left/turn-right), but a live interactive keyboard session hasn't been run |
| Command-conditioned policy for different speeds | Validated: forward AND backward command tracking on `checkpoints/op3_walk_mujoco_best.pt` (91.0%/78.2%, 24/24 correct direction); forward demoed in `demo/op3_push_recovery_demo.mp4`, backward demoed in `demo/op3_backward_demo.mp4` |
| Turning behavior | Validated: `checkpoints/op3_walk_isaac_wide_command.pt` survives scripted forward/turn-left/turn-right sequence with no falls (`scripts/teleop_play_ppo.py --mode script`) |
| Push recovery demo | Present: `demo/op3_push_recovery_demo.mp4` and `demo/op3_push_recovery_short_demo.mp4`, forward-shove push recovery triggered live via keyboard teleop (`p` key), policy recovers and continues walking |
| TensorBoard/logging | Present: TensorBoard event files and per-run `run_manifest.json` |
| Dockerfile/devcontainer | Present for the MuJoCo track: `Dockerfile`; Isaac Lab track uses `SETUP.md`'s native path instead (documented reason in the Dockerfile) |
| Exported trained checkpoint | Present, both tracks: `checkpoints/op3_walk_mujoco_best.pt`, `checkpoints/op3_walk_isaac_best.pt`, `checkpoints/op3_walk_isaac_wide_command.pt` |
| Clean xacro/URDF conversion | Present: `scripts/expand_urdf.sh`, `scripts/convert_urdf_to_usd.py` |
| Sim-to-real explanation | Present in actuator/PD assumptions and limitations docs |
