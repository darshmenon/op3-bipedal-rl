"""Custom action term: residual joint-position control around a live CPG
reference trajectory, instead of the RL policy authoring 100% of joint
motion around a fixed default pose.

Every walk-stage run so far (see STAGE4_PPO_TRAINING.md) puts the CPG
reference only in the *reward* (`op3_mdp.cpg_reference_imitation_l2`) --
the policy is free to satisfy that reward however it likes, and the
"nine levers converged" investigation found it does so via small static
wiggling that never produces a real step (a known imitation-reward
failure mode; also independently flagged by a parallel session's notes,
item 6: "prevents the imitation-reward failure where the policy wiggles
toward reference angles without generating contact transitions").

This term removes that failure mode structurally by moving the reference
into the *action* itself: the joint target is
`cpg_reference(phase) + scale * policy_output`, so the legs cycle through
the reference gait regardless of what the policy outputs -- genuine
stepping is guaranteed by construction, and RL's job becomes learning
residual corrections for balance and disturbance rejection on top of an
already-moving baseline, not discovering that a gait should exist at all.
This is explicitly one of the task brief's suggested action spaces
("residual actions on top of a nominal gait") and suggested fallback
approaches ("residual controller built on top of a nominal gait").
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CpgResidualJointPositionAction(JointPositionAction):
    """`JointPositionAction` whose offset is a live, phase-varying CPG
    reference trajectory (recomputed every `process_actions` call, i.e.
    every `env.step()`) instead of the fixed default pose.

    `self._offset` is normally set once in `JointPositionAction.__init__`
    (the default joint pose) and never touched again. Here it's
    overwritten every step with `default_pose + cpg_delta(phase)` before
    delegating to the base class's own `process_actions`, which then
    computes `target = offset + scale * raw_action` as usual -- so
    `raw_action` becomes a residual around the moving reference rather
    than around a static pose.
    """

    cfg: "CpgResidualJointPositionActionCfg"

    def __init__(self, cfg: "CpgResidualJointPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # The base class's __init__ already set self._offset to the
        # default joint pose (since use_default_offset=True) -- preserve
        # that as the reference's own baseline before process_actions()
        # starts overwriting self._offset every step.
        self._default_offset = self._offset.clone()
        # Resolve each named leg joint's column index within this action
        # term's own joint ordering (self._joint_names, set by the base
        # JointAction.__init__ via find_joints) -- looked up dynamically
        # per name, so this works regardless of preserve_order.
        self._leg_joint_indices = {
            name: self._joint_names.index(name) for name in self.cfg.leg_joint_names if name in self._joint_names
        }

    def process_actions(self, actions: torch.Tensor):
        t = self._env.episode_length_buf.float() * self._env.step_dt
        phase = 2.0 * math.pi * self.cfg.step_freq_hz * t

        ref = self._default_offset.clone()
        lateral_shift = self.cfg.lateral_shift_sign * self.cfg.lateral_shift_amp * torch.sin(phase)

        # Live tilt feedback, added on top of the pure phase-driven swing --
        # same "ankle strategy" law as fallback_gait_controller.py's
        # cpg_targets(), same magnitude/sign for both legs (not side-
        # dependent, unlike the swing/lateral-shift terms above). Both
        # earlier residual-CPG variants (0.6 Hz and 0.3 Hz -- see
        # STAGE4_PPO_TRAINING.md) used the feedforward-only reference and
        # converged to the same worse-than-free-form plateau regardless of
        # frequency; this is the one deliberate omission from
        # fallback_gait_controller.py's own design being tested now. When
        # `roll_kp`/`pitch_kp` are both 0.0 (the default), this reduces to
        # exactly the earlier feedforward-only behavior.
        if self.cfg.roll_kp != 0.0 or self.cfg.pitch_kp != 0.0:
            roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_quat_w)
            roll = torch.atan2(torch.sin(roll), torch.cos(roll))
            pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        else:
            roll = torch.zeros(ref.shape[0], device=ref.device)
            pitch = torch.zeros(ref.shape[0], device=ref.device)
        roll_fb_ank = -self.cfg.roll_kp * roll
        roll_fb_hip = -0.5 * self.cfg.roll_kp * roll
        pitch_fb_hip = self.cfg.pitch_kp * pitch
        pitch_fb_ank = 0.5 * self.cfg.pitch_kp * pitch

        for side, sign, phase_offset in (("l", 1.0, 0.0), ("r", -1.0, math.pi)):
            leg_phase = phase + phase_offset
            hip_swing = self.cfg.hip_amp * torch.sin(leg_phase)
            knee_swing = self.cfg.knee_amp * torch.clamp(torch.sin(leg_phase), min=0.0)
            ankle_pitch_comp = -self.cfg.ankle_comp * hip_swing
            for suffix, delta in (
                (f"{side}_hip_roll", sign * lateral_shift + roll_fb_hip),
                (f"{side}_hip_pitch", sign * hip_swing + pitch_fb_hip),
                (f"{side}_knee", sign * knee_swing),
                (f"{side}_ank_pitch", sign * ankle_pitch_comp + pitch_fb_ank),
                (f"{side}_ank_roll", -0.5 * sign * lateral_shift + roll_fb_ank),
            ):
                idx = self._leg_joint_indices.get(suffix)
                if idx is not None:
                    ref[:, idx] = ref[:, idx] + delta

        self._offset = ref
        super().process_actions(actions)


@configclass
class CpgResidualJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for `CpgResidualJointPositionAction`."""

    class_type: type = CpgResidualJointPositionAction

    leg_joint_names: list[str] = MISSING
    """Which of this action term's joints (by exact name) get a CPG swing
    delta added to their offset each step. Joints in the action term's
    `joint_names` but not in this list keep the fixed default-pose offset
    (e.g. hip_yaw/hip_roll/ank_roll, arms, head)."""

    step_freq_hz: float = 0.6
    hip_amp: float = 0.30
    knee_amp: float = 0.35
    ankle_comp: float = 0.5
    lateral_shift_amp: float = 0.04
    lateral_shift_sign: float = 1.0
    roll_kp: float = 0.0
    """Ankle/hip roll feedback gain on measured base roll (0.0 = feedforward
    only, matching every residual-CPG variant tried before this one).
    `fallback_gait_controller.py`'s own default is 1.5."""
    pitch_kp: float = 0.0
    """Hip/ankle pitch feedback gain on measured base pitch (0.0 =
    feedforward only). `fallback_gait_controller.py`'s own default is 1.0."""
