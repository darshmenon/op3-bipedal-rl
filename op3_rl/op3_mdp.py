"""OP3-specific MDP terms not covered by Isaac Lab's stock locomotion `mdp`
module: a gait-phase clock observation, two bipedal gait-shaping rewards
(swing-foot clearance, lateral COM shift over the stance foot), and a
time-windowed "danger zone" upright bonus.

Isaac Lab's built-in locomotion rewards (`isaaclab_tasks...velocity.mdp`)
cover velocity tracking and generic timing (`feet_air_time`,
`feet_air_time_positive_biped`, `feet_slide`) but nothing about *how* a foot
swings (does it actually clear the ground?) or *where* the body's weight
sits relative to the load-bearing foot -- both real components of bipedal
gait that pure velocity-tracking + timing rewards don't ask for directly.
See STAGE4_PPO_TRAINING.md's "gait-phase structure" section for why these
were added and what problem they're meant to address.

Every reward term in this file (and in `op3_env_cfg.py`'s `OP3*RewardsCfg`
classes) up to `danger_zone_upright_bonus` uses a *constant* per-step
weight -- the same incentive at step 2 as at step 200. `danger_zone_upright_
bonus` is different: it varies by elapsed time within the episode, targeting
the specific ~0.7-1.2s (~step 35-60) window this project has repeatedly
found failures cluster around, rather than shaping the whole episode
uniformly. See STAGE4_PPO_TRAINING.md's "five levers converged" section for
why a time-windowed mechanism was tried after five differently-weighted
but uniformly-applied reward configurations all converged to the same
collapse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase_obs(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """Sin/cos gait-clock observation, cycling every `period` seconds.

    A fixed-frequency clock (not derived from contact or velocity) gives the
    policy a consistent per-episode timing reference to phase-lock leg swing
    against -- the same role a central pattern generator's clock plays,
    without hand-designing the gait itself (the phase-dependent shaping is
    left to the reward terms below, not this observation). Resets to phase 0
    at each episode reset because `episode_length_buf` resets there too.
    """
    phase = 2.0 * torch.pi * (env.episode_length_buf.float() * env.step_dt / period)
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)


def swing_foot_clearance_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
) -> torch.Tensor:
    """Reward a foot currently in the air (per the contact sensor) for
    reaching `target_height` above the ground, exp-kernel shaped like
    `track_lin_vel_xy_exp`.

    Isaac Lab's stock `feet_air_time*` rewards only care about *timing* --
    how long a foot stays airborne -- not how high it lifts. A foot that
    barely leaves the ground for the whole air-time window scores the same
    as one that clears a real step height under those rewards; this adds
    the clearance signal they're missing. `sensor_cfg` and `asset_cfg` are
    expected to select the same feet (e.g. both via the `.*ank_roll_link`
    regex) so their body-id orderings line up, the same assumption Isaac
    Lab's own `feet_slide` makes.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_swing = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.0
    asset = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    error = torch.square(foot_height - target_height)
    return torch.sum(torch.exp(-error / std**2) * is_swing, dim=1)


def stance_lateral_com_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward the base sitting laterally over the single stance foot (in the
    base's own yaw-aligned frame), exp-kernel shaped.

    Real bipedal walking shifts the whole-body COM laterally over whichever
    foot is currently load-bearing so the other leg can swing without
    toppling sideways -- nothing in the existing reward set asks for this;
    `track_lin_vel_xy_exp` only cares about forward/turning velocity, not
    lateral weight placement. Zero reward outside single-stance (both feet
    down, or both airborne), since "lateral offset from the stance foot"
    isn't defined there -- mirrors `feet_air_time`'s zero-command gating
    pattern. Computed in the base's yaw-only frame (not raw world y) so a
    turning robot doesn't get penalized for a lateral offset that's really
    just its own heading change.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    asset = env.scene[asset_cfg.name]
    base_pos = asset.data.root_pos_w
    base_quat = asset.data.root_quat_w
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    rel = feet_pos - base_pos.unsqueeze(1)
    num_feet = rel.shape[1]
    yaw_q = yaw_quat(base_quat).unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    rel_body = quat_apply_inverse(yaw_q, rel.reshape(-1, 3)).reshape(rel.shape)
    lateral = rel_body[..., 1]

    stance_lateral_offset = torch.sum(lateral * in_contact.float(), dim=1)
    return torch.exp(-torch.square(stance_lateral_offset) / std**2) * single_stance.float()


def cpg_reference_imitation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_names: list[str],
    step_freq_hz: float,
    hip_amp: float,
    knee_amp: float,
    ankle_comp: float,
    std: float,
) -> torch.Tensor:
    """Reward the policy's actual leg joint positions for tracking a simple
    kinematic reference trajectory -- the same nominal alternating-leg
    swing pattern as `scripts/fallback_gait_controller.py`'s CPG (its
    feedforward component only; no live roll/pitch feedback baked into the
    target, that's left to the policy's own balance behavior via the other
    reward terms).

    Every reward term tried so far (this file's other functions and every
    `OP3*RewardsCfg` in `op3_env_cfg.py`) is *outcome*-shaped: it rewards
    forward velocity, staying upright, or contact timing, but says nothing
    about what the swing/stance motion itself should look like -- PPO has
    to discover "what walking is" purely from those sparse outcome
    signals. This term is fundamentally different: it rewards matching a
    known-reasonable gait *shape* directly, using the existing hand-
    engineered CPG (built for the fallback track, per the task brief) as a
    kinematic prior instead of writing one from scratch. See
    STAGE4_PPO_TRAINING.md's torque-clipping analysis (`torque_clip_frac`
    climbing 0.15->0.35 as roll/pitch grow, desired torque hitting
    3-7x the 5N*m limit) for why a gait that needs *less* large corrective
    torque in the first place -- by tracking a stable reference shape
    instead of improvising one under pressure -- was tried next, after six
    outcome-only reward configurations converged to the same collapse.

    `joint_names` must list exactly the joints `asset_cfg.joint_ids`
    resolves to, in that same order (pass the same explicit list to both,
    e.g. `_LEG_JOINTS`) -- needed to place each reference value at the
    right column of the (num_envs, num_joints) tensors below.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    t = env.episode_length_buf.float() * env.step_dt
    phase = 2.0 * torch.pi * step_freq_hz * t

    default_pos = asset.data.default_joint_pos[:, joint_ids]
    actual_pos = asset.data.joint_pos[:, joint_ids]
    ref = default_pos.clone()

    for side, sign, phase_offset in (("l", 1.0, 0.0), ("r", -1.0, torch.pi)):
        leg_phase = phase + phase_offset
        hip_swing = hip_amp * torch.sin(leg_phase)
        knee_swing = knee_amp * torch.clamp(torch.sin(leg_phase), min=0.0)
        ankle_pitch_comp = -ankle_comp * hip_swing
        for suffix, delta in (
            (f"{side}_hip_pitch", sign * hip_swing),
            (f"{side}_knee", sign * knee_swing),
            (f"{side}_ank_pitch", sign * ankle_pitch_comp),
        ):
            idx = joint_names.index(suffix)
            ref[:, idx] = ref[:, idx] + delta

    # Mean, not sum, across joints -- `std` is meant as a per-joint radian
    # tolerance. Summing over all 12 leg joints made the exp() collapse to
    # ~0 for any merely-imperfect gait (a plausible ~0.2 rad average
    # per-joint deviation already pushes the *summed* error past std**2),
    # giving almost no usable gradient anywhere except near-perfect
    # tracking. Caught live: the first launch's `cpg_imitation` reward sat
    # pinned at ~0.003 for its entire run, never moving -- see
    # STAGE4_PPO_TRAINING.md.
    error = torch.mean(torch.square(actual_pos - ref), dim=1)
    return torch.exp(-error / std**2)


def danger_zone_upright_bonus(
    env: ManagerBasedRLEnv,
    window_start_s: float,
    window_end_s: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Extra upright reward active only during `[window_start_s,
    window_end_s)` of *each episode's own elapsed time*, not the whole
    episode uniformly.

    `flat_orientation_l2` and friends apply the same tilt penalty at every
    step regardless of how far into the episode it is. This project has
    repeatedly found (see STAGE4_PPO_TRAINING.md) that failures cluster in
    a specific early window -- roughly steps 35-60 (~0.7-1.2s at this env's
    0.02s step) -- rather than being uniformly likely across an episode's
    length. A constant-weight term can't express "this moment matters more
    than that one"; this one can. Exp-kernel on projected-gravity tilt, same
    shape as `flat_orientation_l2`'s penalty but reward- (not penalty-)
    signed and gated to the window, so surviving upright specifically
    through the historically dangerous stretch carries extra incentive
    beyond what the constant per-step terms already provide.
    """
    asset = env.scene[asset_cfg.name]
    elapsed_s = env.episode_length_buf.float() * env.step_dt
    in_window = (elapsed_s >= window_start_s) & (elapsed_s < window_end_s)
    tilt = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-tilt / std**2) * in_window.float()


def foot_load_asymmetry_bonus(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Reward asymmetric loading between the two feet -- weight shifting
    toward one foot -- as a continuous, ungated signal toward eventual
    single-support.

    Diagnostic evidence (see STAGE4_PPO_TRAINING.md, "checked for similar
    bugs elsewhere") found `feet_air_time_positive_biped`,
    `swing_foot_clearance`, and `stance_lateral_com` all logged exactly
    `0.0000` for the *entire* `op3_walk_gait_phase` run -- every one of
    them requires a foot to actually leave the ground
    (`current_air_time > 0` / a `current_contact_time`-derived
    single-stance flag) before it can contribute anything, and foot-force
    traces from multiple diagnostics suggest the policy never produces a
    real liftoff at all. None of those terms can ever bootstrap a first
    step, because none of them can express "getting closer to lifting a
    foot" -- only "a foot is airborne" (binary, never true) or "not
    airborne" (always true, reward always 0). This term can: it reads
    live per-foot contact *force* (continuous, always defined, no
    liftoff required) and rewards the *difference* growing, so shifting
    weight toward one side is worth something even while both feet stay
    planted, well before any actual liftoff. Zero reward for
    near-zero commanded velocity (mirrors `feet_air_time`'s zero-command
    gating) -- asymmetric loading while asked to stand still isn't
    something to encourage.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    total = torch.sum(forces, dim=1).clamp_min(1e-6)
    asymmetry = torch.abs(forces[:, 0] - forces[:, 1]) / total

    cmd_moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return asymmetry * cmd_moving.float()
