"""Flat-terrain velocity-tracking environment for OP3.

Reuses Isaac Lab's built-in `LocomotionVelocityRoughEnvCfg` (the same base
class Cassie/ANYmal/Unitree configs use) rather than writing a bespoke MDP.
Its default observations already match the task doc's suggested observation
space (base ang vel, projected gravity, commanded velocity, joint pos/vel,
previous actions) and its default rewards already cover the task doc's
suggested terms (forward velocity tracking, torque penalty, action
smoothness, termination penalty, etc.). What's changed here is retargeting
body/joint names to OP3 and switching to flat terrain -- not new reward
design. See ../STAGE3_PPO_SMOKE_TEST.md.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from . import op3_actions, op3_mdp
from .op3_robot_cfg import OP3_CFG

# A procedurally-generated flat terrain (no external USD asset needed) --
# terrain_type="plane" always spawns a Nucleus-hosted ground plane USD
# (ISAAC_NUCLEUS_DIR/Environments/Grid/default_environment.usd), which isn't
# reachable/configured for this pip-only install. This sidesteps that.
_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(20.0, 20.0),
    num_rows=1,
    num_cols=1,
    sub_terrains={"flat": MeshPlaneTerrainCfg()},
)

# OP3's foot link is the last link in each leg chain (ankle roll).
_FEET_REGEX = ".*ank_roll_link"
# Approximate "shouldn't touch the ground" bodies (upper/lower leg segments);
# OP3 has no link literally named "thigh"/"shin".
_UNDESIRED_CONTACT_REGEX = ".*(hip_pitch|knee)_link"
_TORSO_LINK = "body_link"
_ARM_HEAD_JOINTS = ["l_sho_pitch", "l_sho_roll", "l_el", "r_sho_pitch", "r_sho_roll", "r_el", "head_pan", "head_tilt"]
_LEG_JOINTS = [
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll",
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll",
]


def _zero_base_reset(env_cfg) -> None:
    """Reset at the configured pose with no injected root velocity."""
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }


def _zero_velocity_command(env_cfg) -> None:
    """Make the inherited velocity-tracking task a stationary command task."""
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)


def _gentle_base_reset(env_cfg) -> None:
    """Scale reset-time pose/velocity noise down from ANYmal/Cassie scale to OP3 scale.

    The inherited defaults (+/-0.5 m pose offset, +/-0.5 m/s and +/-0.5 rad/s
    velocity noise) come from LocomotionVelocityRoughEnvCfg, sized for a
    ~40 kg quadruped. For OP3 (3.147 kg, ~45 cm tall) that's a large fraction
    of its own body size/speed injected at every reset. Scale to roughly a
    tenth rather than zeroing outright (OP3StandEnvCfg zeroes these
    entirely, appropriate for a zero-velocity-command stand task; the walk
    task still wants some reset diversity to avoid overfitting to one exact
    starting state, just not ANYmal-scale diversity).
    """
    env_cfg.events.reset_base.params["pose_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.2, 0.2)}
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05),
        "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
    }


@configclass
class OP3RewardsCfg(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # Keep arms/head near their neutral pose since they aren't needed for
    # walking (task doc: arm/head joints "may be fixed, controlled to a
    # nominal pose, or included in the policy if useful for balance" --
    # here they're included but softly discouraged from drifting).
    joint_deviation_arms_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_HEAD_JOINTS)},
    )
    # `RewardsCfg`'s inherited `std=math.sqrt(0.25)` (0.5 m/s) is Isaac
    # Lab's own stock default, unchanged even in its G1/H1 humanoid
    # reference configs -- sized for robots tracking ~0.5-1.5 m/s commands.
    # OP3's own command ranges here are 0.02-0.2 m/s, 5-25x smaller, so
    # that tolerance makes near-zero velocity score well almost everywhere
    # in-range. Confirmed as a real exploit, not a theoretical concern, on
    # the MuJoCo side of this project (op3_rl/mujoco_env.py uses this same
    # reward spec): a converged checkpoint replay-validated to exactly
    # 0.0m net displacement over a full episode -- standing still, not
    # walking -- at both the original tiny command range and a widened one.
    # Tightened to std=0.05 (matching mujoco_env.py's `_VEL_TRACKING_VARIANCE
    # = 0.0025` exactly) so both engines keep training against the
    # identical task spec. See STAGE4_PPO_TRAINING.md.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.05}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": 0.05}
    )


@configclass
class OP3StandRewardsCfg(OP3RewardsCfg):
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class OP3WalkEasyRewardsCfg(OP3RewardsCfg):
    # Unlike OP3StandRewardsCfg, the walk stage had no per-step survival
    # incentive at all -- only a one-time -200 termination_penalty plus
    # small (~0.01-0.03 magnitude) tracking/shaping terms in between. Add a
    # modest alive bonus (half the stand stage's weight, since walking also
    # has to balance a real tracking objective against it, not just
    # standing still) so staying upright carries real per-step weight.
    alive = RewTerm(func=mdp.is_alive, weight=0.5)


@configclass
class OP3WalkRecoveryRewardsCfg(OP3WalkEasyRewardsCfg):
    """Walk reward variant aimed at the observed replay failure mode.

    This keeps the rebalanced walking objective, then adds explicit upright,
    height, angular-velocity, action-rate, and foot-slide pressure. It is a
    candidate stage, not a proven improvement; validate by replay before
    promoting any checkpoint trained with it.
    """

    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-6.0, params={"target_height": 0.32})
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FEET_REGEX),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FEET_REGEX),
        },
    )


@configclass
class OP3WalkDangerZoneRewardsCfg(OP3WalkRecoveryRewardsCfg):
    """Recovery rewards plus `danger_zone_upright_bonus`: extra upright
    incentive during the ~0.7-1.2s (~step 35-60) window this project has
    repeatedly found failures cluster around, rather than another
    uniformly-weighted per-step term. See `op3_mdp.py` and
    STAGE4_PPO_TRAINING.md's "five levers converged" section for why a
    time-windowed mechanism was tried after five uniformly-applied reward
    configurations all converged to the same collapse.
    """

    danger_zone_upright = RewTerm(
        func=op3_mdp.danger_zone_upright_bonus,
        weight=1.0,
        params={"window_start_s": 0.6, "window_end_s": 1.3, "std": 0.3},
    )


@configclass
class OP3WalkFootLoadAsymmetryRewardsCfg(OP3WalkRecoveryRewardsCfg):
    """Recovery rewards plus `foot_load_asymmetry_bonus`: a continuous,
    ungated reward for weight shifting between the two feet, active from
    the very first step regardless of whether a foot ever fully lifts off.

    Motivated directly by the "checked for similar bugs elsewhere" finding
    in STAGE4_PPO_TRAINING.md: `feet_air_time_positive_biped`,
    `swing_foot_clearance`, and `stance_lateral_com` all logged exactly
    0.0000 for the entire `op3_walk_gait_phase` run, because every one of
    them requires a real liftoff (`current_air_time > 0`) to ever activate
    -- and foot-force diagnostics suggest that liftoff basically never
    happens. None of those terms can bootstrap a *first* step; this one
    can, because it reads continuous per-foot contact force instead of a
    liftoff-gated binary. Tried after eight prior levers (reward
    rebalancing, gait-phase shaping, plain actor-reset, +recovery shaping,
    +reduced termination penalty, +danger-zone reward, +CPG-imitation at
    two frequencies) all converged to the same collapse -- see "Eight
    levers converged" in STAGE4_PPO_TRAINING.md.
    """

    foot_load_asymmetry = RewTerm(
        func=op3_mdp.foot_load_asymmetry_bonus,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FEET_REGEX),
            "command_name": "base_velocity",
        },
    )


@configclass
class OP3WalkCpgImitationRewardsCfg(OP3WalkRecoveryRewardsCfg):
    """Recovery rewards plus `cpg_reference_imitation_l2`: reward the
    policy's leg joints for tracking the fallback CPG's nominal gait shape
    (see `op3_mdp.py`), instead of another outcome-only term.

    Motivated by this session's torque-clipping diagnostic (see
    STAGE4_PPO_TRAINING.md): `torque_clip_frac` climbs 0.15->0.35 as
    roll/pitch grow, with desired torque hitting 3-7x the 5N*m limit --
    once tilt starts, the policy structurally can't apply enough corrective
    torque to arrest it. A gait shaped to need less large corrective torque
    in the first place (by tracking a known-stable reference instead of
    improvising one under pressure) may reduce how often that torque
    ceiling gets hit at all. `step_freq_hz`/`hip_amp`/`knee_amp`/
    `ankle_comp` match `fallback_gait_controller.py`'s own defaults exactly
    -- not retuned for this purpose, so this is testing the existing
    hand-engineered gait shape as-is, not a newly optimized one.
    """

    cpg_imitation = RewTerm(
        func=op3_mdp.cpg_reference_imitation_l2,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS, preserve_order=True),
            "joint_names": _LEG_JOINTS,
            "step_freq_hz": 0.6,
            "hip_amp": 0.30,
            "knee_amp": 0.35,
            "ankle_comp": 0.5,
            "std": 0.3,
        },
    )


@configclass
class OP3WalkCpgImitationFootLoadRewardsCfg(OP3WalkCpgImitationRewardsCfg):
    """`OP3WalkCpgImitationRewardsCfg` plus `foot_load_asymmetry_bonus`:
    stacks the two structurally distinct levers tried individually above,
    neither of which broke the collapse alone.

    `cpg_imitation` gives the policy a specific target gait *shape* but no
    pressure to actually commit to single-support (its equilibrium value
    implied ~30+ degrees average per-joint deviation, i.e. it never
    converged close to the reference either way). `foot_load_asymmetry`
    gives continuous, liftoff-independent pressure toward weight-shifting
    but no information about *what* to do with that shift (its own
    logged value stayed flat at its untrained baseline the entire run,
    never increasing). Untested until now whether a specific target shape
    *combined with* an ungated incentive to actually move toward it
    changes the outcome, versus either alone. See STAGE4_PPO_TRAINING.md.
    """

    foot_load_asymmetry = RewTerm(
        func=op3_mdp.foot_load_asymmetry_bonus,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FEET_REGEX),
            "command_name": "base_velocity",
        },
    )


@configclass
class OP3WalkCpgImitationSlowRewardsCfg(OP3WalkCpgImitationRewardsCfg):
    """`OP3WalkCpgImitationRewardsCfg` with the reference gait's cycle
    frequency halved (0.6 -> 0.3 Hz), everything else identical.

    The 0.6 Hz version converged to the same collapse as every other
    lever: `base_contact` pinned at ~1.0, `cpg_imitation` itself
    plateaued at a low value implying ~30+ degrees average per-joint
    deviation from the reference -- the policy never closely tracked it.
    Motivated by the torque-clipping diagnostic: for a fixed swing
    amplitude, required joint angular acceleration (and therefore torque)
    scales roughly with frequency squared, so this should cut peak torque
    demand by ~4x, directly targeting the `torque_clip_frac` 0.15->0.35
    climb documented earlier. See STAGE4_PPO_TRAINING.md's decision-tree
    plan for why this was tried after the 0.6 Hz version plateaued.
    """

    cpg_imitation = RewTerm(
        func=op3_mdp.cpg_reference_imitation_l2,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS, preserve_order=True),
            "joint_names": _LEG_JOINTS,
            "step_freq_hz": 0.3,
            "hip_amp": 0.30,
            "knee_amp": 0.35,
            "ankle_comp": 0.5,
            "std": 0.3,
        },
    )


@configclass
class OP3WalkStructuredRewardsCfg(OP3WalkRecoveryRewardsCfg):
    """Recovery rewards plus biped-specific foot-air-time structure."""

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FEET_REGEX),
            "command_name": "base_velocity",
            "threshold": 0.25,
        },
    )


@configclass
class OP3WalkGaitPhaseRewardsCfg(OP3WalkStructuredRewardsCfg):
    """Structured rewards plus the swing-clearance and stance-lateral-COM
    terms `feet_air_time_positive_biped` doesn't cover on its own.

    `feet_air_time_positive_biped` only rewards *timing* -- one foot down at
    a time, for roughly the right duration. It says nothing about how high
    the swing foot actually lifts, or whether the body's weight shifts onto
    the stance foot the way real bipedal walking requires. These two custom
    terms (see `op3_mdp.py`) add that. Completes item 3 ("gait-phase
    structure") of STAGE4_PPO_TRAINING.md's recommended-next-experiments
    list -- `OP3WalkStructuredRewardsCfg` above only covered the
    alternating-support half of that item.
    """

    swing_foot_clearance = RewTerm(
        func=op3_mdp.swing_foot_clearance_l2,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FEET_REGEX),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FEET_REGEX),
            # ~4 cm foot lift: a small fraction of OP3's ~0.32m standing
            # height, not sourced from a measured real step height (no
            # verified gait reference exists for this robot yet -- see
            # op3_robot_cfg.py's PD-gain disclaimers for the same caveat
            # pattern). Revisit if replay shows the policy either can't
            # reach it or clears it trivially.
            "target_height": 0.04,
            "std": 0.03,
        },
    )
    stance_lateral_com = RewTerm(
        func=op3_mdp.stance_lateral_com_l2,
        weight=0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FEET_REGEX),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FEET_REGEX),
            # OP3's hip width is small (~9cm between leg roots), so the
            # lateral COM shift a real single-support phase needs is
            # correspondingly small; std picked as a fraction of that, not
            # measured.
            "std": 0.03,
        },
    )


@configclass
class OP3FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """OP3 flat-terrain environment configuration."""

    rewards: OP3RewardsCfg = OP3RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # scene / robot
        self.scene.robot = OP3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # velocity-command debug-vis markers (arrow_x.usd) are Nucleus-hosted
        # assets, unreachable from this pip-only install; not needed for a
        # headless smoke test.
        self.commands.base_velocity.debug_vis = False

        # flat terrain, no height scan (matches the task's "flat terrain" requirement)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _FLAT_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = None
        # the default visual_material MDL is Nucleus-hosted; drop it (cosmetic
        # only -- headless smoke test doesn't need a textured ground)
        self.scene.terrain.visual_material = None
        # Ground plane also silently inherited IsaacLab's 0.5/0.5 default
        # friction (see OP3_CFG's own physics_material comment in
        # op3_robot_cfg.py) -- set explicitly to match MuJoCo's op3.xml
        # foot-geom sliding friction (1.0) so the effective foot-ground
        # friction is 1.0 regardless of combine mode.
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        )
        # The robot's own USD spawner (`UsdFileCfg` in op3_robot_cfg.py) has
        # no per-asset `physics_material` field, so OP3's feet fall back to
        # this *scene-wide* default material for every collision prim that
        # doesn't set its own -- `LocomotionVelocityRoughEnvCfg.__post_init__`
        # (called via `super().__post_init__()` above) already copies
        # `self.scene.terrain.physics_material` into `self.sim.physics_material`,
        # but it does so *before* the terrain override two lines up runs, so
        # without this second assignment the scene-wide default -- and
        # therefore the robot's own feet -- would stay on IsaacLab's stale
        # 0.5/0.5 default even though the terrain's own material got fixed.
        # Re-set it here, after the terrain override, so both the ground and
        # the robot's feet resolve to the same matched 1.0/1.0 friction.
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.height_scanner = None
        # same for the default sky light's HDR texture
        self.scene.sky_light.spawn.texture_file = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # actions: all 20 joints under PD position control (task doc allows
        # freezing arms/head, but the smoke test controls everything for
        # simplicity -- can be restricted to leg joints later)
        self.actions.joint_pos.scale = 0.3

        # events: retarget from the generic "base" link name to OP3's torso.
        # The inherited locomotion task adds (-5, 5) kg and +/-5 cm COM noise,
        # which is larger than OP3's entire 3.147 kg body. Keep OP3 training
        # deterministic until clean standing/walking works, then reintroduce
        # domain randomization deliberately.
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = _TORSO_LINK
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # terminations: torso touching the ground = fallen
        self.terminations.base_contact.params["sensor_cfg"].body_names = _TORSO_LINK

        # rewards: retarget body-name params to OP3's real link names
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = _FEET_REGEX
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = _UNDESIRED_CONTACT_REGEX
        self.rewards.flat_orientation_l2.weight = -2.5


@configclass
class OP3StandEnvCfg(OP3FlatEnvCfg):
    """Standing curriculum stage before velocity-tracking walking.

    This keeps the same observation/action surfaces as the walking task but
    removes velocity commands, pushes, reset perturbations, and walking-specific
    foot-air-time reward. PPO then has one job: stay upright near the configured
    pose. If this stage cannot time out consistently, the blocker is still the
    robot/contact/actuator model rather than locomotion reward shaping.
    """

    rewards: OP3StandRewardsCfg = OP3StandRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        _zero_velocity_command(self)
        _zero_base_reset(self)
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Keep the same action scale as the walk stage (0.3, from OP3FlatEnvCfg)
        # rather than a stand-specific smaller value: a policy fine-tuned at a
        # different scale sees the same raw network output mapped to a very
        # different physical joint movement the moment it's warm-started into
        # the walk stage, which is a likely cause of op3_walk_from_stand's
        # regression (episode length ~40 -> ~14 over 1000 iterations) despite
        # inheriting a stand-capable policy. See STAGE4_PPO_TRAINING.md.

        self.rewards.alive.weight = 1.0
        self.rewards.termination_penalty.weight = -500.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -8.0
        self.rewards.ang_vel_xy_l2.weight = -0.5
        self.rewards.lin_vel_z_l2.weight = -4.0
        self.rewards.action_rate_l2.weight = -0.05


@configclass
class OP3StandLegsOnlyEnvCfg(OP3StandEnvCfg):
    """Stand stage restricted to the 12 leg joints (the task brief's stated
    minimum action set), arms/head left uncontrolled.

    Every `op3_stand_v2` replay check this session (see
    STAGE4_PPO_TRAINING.md) showed the same ~35-50 step "danger zone" that
    training only ever got ~83-90% of the population past, never fully
    eliminated, with noisy/non-monotonic checkpoint quality even in plain
    resumption (no curriculum change at all). The ruled-out COM/support-
    polygon check showed the nominal pose itself is geometrically fine, so
    the instability is dynamic. One untested lever: all 20 joints are
    currently actuated, including 8 arm/head joints that aren't needed for
    balance -- their exploration noise could be actively destabilizing
    rather than helping. This restricts the action term (and therefore
    `last_action` in the observation) to just the 12 leg joints; arms/head
    stay locked at their default pose via the implicit PD actuator (never
    commanded, so nothing pulls them away from init). Changes the
    action/observation dimensions, so this is a fresh run, not a resume of
    any existing checkpoint.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.joint_names = _LEG_JOINTS


@configclass
class OP3WalkEasyEnvCfg(OP3FlatEnvCfg):
    """Gentle-walk curriculum stage between standing and full velocity tracking.

    `op3_walk_from_stand_v2` warm-started from a strong stand checkpoint
    (episode length 170) straight into `OP3FlatEnvCfg`'s full walk
    difficulty and immediately plateaued flat at ~40-45 steps for 900+
    iterations with near-zero track_lin_vel_xy_exp reward -- not learning to
    walk, just surviving a fixed number of steps before falling every time
    (Episode_Termination/base_contact pinned at 1.0). Root cause: unlike the
    stand stage, `OP3FlatEnvCfg` never scaled down the inherited
    ANYmal/Cassie-scale `push_robot` interval kicks (+/-0.5 m/s every
    10-15s) or `reset_base` pose/velocity noise (+/-0.5 m, +/-0.5 m/s) for
    OP3's much smaller 3.147 kg / ~45 cm frame -- the same class of bug
    already found and fixed for `add_base_mass`/`base_com`. This stage
    fixes that: small velocity commands (a real but modest walking target
    rather than 0), `push_robot` disabled (reintroduce once this works,
    same as the stand stage's approach to reset perturbations), and
    `_gentle_base_reset` in place of the ANYmal-scale defaults. See
    STAGE4_PPO_TRAINING.md.

    That disturbance-scale fix alone did not hold up: episode length and
    track_lin_vel_xy_exp both improved briefly right after the stage switch,
    then declined steadily back toward the same base_contact=1.0-every-episode
    collapse over ~500 iterations. Added `OP3WalkEasyRewardsCfg`'s `alive`
    bonus on top as the next hypothesis: the walk reward set had no per-step
    survival incentive at all (unlike the stand stage), so this gives staying
    upright real per-step weight instead of relying solely on a one-time
    termination penalty. See STAGE4_PPO_TRAINING.md.
    """

    rewards: OP3WalkEasyRewardsCfg = OP3WalkEasyRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.events.push_robot = None
        _gentle_base_reset(self)


@configclass
class OP3WalkEasyRebalancedEnvCfg(OP3WalkEasyEnvCfg):
    """`OP3WalkEasyEnvCfg` with the reward balance shifted toward actually
    tracking velocity, not just surviving.

    Three PPO-optimization-mechanics levers (lower learning rate, tighter
    adaptive-KL target, longer rollout window) were each tried against the
    walk stage's late-training collapse and none prevented it -- see
    STAGE4_PPO_TRAINING.md. That a *lower* learning rate, a *tighter* KL
    constraint, and a *longer* rollout all failed the same way suggests the
    issue isn't how carefully PPO is optimizing, it's what the reward
    function is telling it to optimize *for*. Evidence: `track_lin_vel_xy_exp`
    stayed at ~0.02-0.06 throughout every walk-stage run, including the best
    checkpoint -- the policy was never actually tracking the commanded
    velocity well, even while surviving longest. Meanwhile
    `termination_penalty=-200` dwarfs the ~2.1 max possible per-step reward
    (a single avoided fall is worth 100+ steps of perfect reward), which
    gives a strong incentive toward risk-averse near-stillness over the
    dynamic weight-shifting real walking requires -- exactly the kind of
    asymmetry that would produce "delay the fall, don't actually walk" as
    the easiest local optimum, matching what's been observed. Rebalances
    by cutting the termination penalty and raising the tracking reward's
    weight so genuine velocity tracking carries real incentive relative to
    pure survival.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.termination_penalty.weight = -50.0
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5


@configclass
class OP3WalkEasyRebalancedLowUndesiredContactEnvCfg(OP3WalkEasyRebalancedEnvCfg):
    """`OP3WalkEasyRebalancedEnvCfg` (full -0.2/0.2 m/s command range, not
    the tiny-forward curriculum's narrow one) plus the `undesired_contacts`
    fix from `OP3WalkTinyForwardLowUndesiredContactEnvCfg` (-1.0 -> -0.2).

    Natural next curriculum step once the narrow tiny-forward stage was
    essentially solved (127/128 survival at
    `checkpoints/op3_walk_isaac_best.pt`, see STAGE4_PPO_TRAINING.md): widen
    the command range while keeping the one reward fix that's actually
    proven to help, and warm-start from that checkpoint instead of starting
    the wider task from scratch.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.undesired_contacts.weight = -0.2


@configclass
class OP3WalkTinyForwardEnvCfg(OP3WalkEasyRebalancedEnvCfg):
    """Very small forward-only curriculum before the broader `walk_easy` task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.02, 0.08)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02),
            "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (-0.02, 0.02),
        }


@configclass
class OP3WalkTinyForwardLowUndesiredContactEnvCfg(OP3WalkTinyForwardEnvCfg):
    """`OP3WalkTinyForwardEnvCfg` with `undesired_contacts` cut from -1.0 to
    -0.2.

    This weight was never touched by this project's own reward rebalancing
    work (`termination_penalty` -200->-50, `track_lin_vel_xy_exp` 1.0->3.0)
    -- it's still Isaac Lab's raw ANYmal-quadruped default (`RewTerm(...,
    weight=-1.0, ...)` in `velocity_env_cfg.py`, originally penalizing THIGH
    contact), the exact same class of unexamined-inheritance bug already
    found and fixed twice on this project for `contact_offset` and
    `solver_position_iteration_count` (both ANYmal-scale defaults silently
    applied to a much smaller/different biped). Live reward-breakdown
    logging on the `op3_walk_tiny_forward_contactauthored` run
    (STAGE4_PPO_TRAINING.md) found `undesired_contacts` at -0.67 to -1.07 --
    by a wide margin the single largest-magnitude per-step reward term,
    dwarfing every other shaping term including the already-rebalanced
    `track_lin_vel_xy_exp`. `undesired_contacts` watches the knee/hip links
    (`_UNDESIRED_CONTACT_REGEX`), separate from the torso-contact
    termination condition -- so a knee/hip touch used to catch a stumble and
    keep going is punished at the same severity as a quadruped brushing
    terrain with its thigh, which may be actively discouraging exactly the
    stumble-recovery behavior this task needs rather than teaching cleaner
    stepping. Untested until this run.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.undesired_contacts.weight = -0.2


@configclass
class OP3WalkRecoveryEnvCfg(OP3WalkEasyRebalancedEnvCfg):
    """Walk-easy task with explicit recovery and contact-quality rewards."""

    rewards: OP3WalkRecoveryRewardsCfg = OP3WalkRecoveryRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.flat_orientation_l2.weight = -6.0
        self.rewards.ang_vel_xy_l2.weight = -0.25
        self.rewards.action_rate_l2.weight = -0.08


@configclass
class OP3WalkStructuredEnvCfg(OP3WalkRecoveryEnvCfg):
    """Recovery task with biped stepping/contact structure enabled."""

    rewards: OP3WalkStructuredRewardsCfg = OP3WalkStructuredRewardsCfg()


@configclass
class OP3WalkTinyForwardRecoveryEnvCfg(OP3WalkRecoveryEnvCfg):
    """`OP3WalkTinyForwardEnvCfg`'s narrow forward-only commands/reset noise
    combined with `OP3WalkRecoveryEnvCfg`'s stronger action-rate/height/
    orientation pressure -- two candidate fixes developed independently
    that had never actually been combined.

    Motivated by `diagnose_walk_numeric.py` run against
    `op3_walk_tiny_forward_actor_reset/model_19350.pt` (see
    STAGE4_PPO_TRAINING.md): that run's `action_mag` and `joint_abs_max`
    escalated rapidly and didn't stabilize (2.65->7.96 and 0.32->1.29 rad
    over just 0.7s) while `actual_vel` stayed *negative* against a small
    positive forward command -- a growing, self-reinforcing instability,
    not simple torque starvation (`torque_clip_frac` was only 0.05-0.15).
    `OP3WalkTinyForwardEnvCfg` never got Recovery's 8x-stronger
    `action_rate_l2` (-0.08 vs the inherited -0.01) or its `base_height_l2`/
    `feet_slide` terms; this stage tests whether that pressure, on the
    easier task, actually damps the instability the diagnostic found rather
    than just being active alongside a task too hard to benefit from it
    (as in `OP3WalkStructuredEnvCfg`, which has the same reward terms but
    on the full `walk_easy` command range).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.02, 0.08)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02),
            "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (-0.02, 0.02),
        }


@configclass
class OP3WalkTinyForwardRecoveryLowTermEnvCfg(OP3WalkTinyForwardRecoveryEnvCfg):
    """`OP3WalkTinyForwardRecoveryEnvCfg` with `termination_penalty` cut from
    -50 to -10, to isolate whether the termination penalty itself is
    dominating the small per-step shaping terms and driving the risk-averse
    "stop moving, minimize fall probability" collapse seen in every actor-
    reset variant so far.

    Arithmetic motivating this (see STAGE4_PPO_TRAINING.md): episodes in
    both prior actor-reset runs settled around ~52-56 steps before falling.
    Over 52 steps, the `alive=0.5` bonus contributes +26 total -- less than
    a single -50 termination penalty, so a typical episode's alive-vs-
    termination balance is already net negative *before* any of the other
    shaping terms are counted, which keeps "fall a little later" a weakly
    losing proposition relative to genuinely risky forward progress. A
    smaller termination penalty changes that balance without touching any
    of the per-step terms already shown (across four different reward
    configurations) not to be the actual lever.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.termination_penalty.weight = -10.0


@configclass
class OP3WalkTinyForwardDangerZoneEnvCfg(OP3WalkTinyForwardRecoveryEnvCfg):
    """`OP3WalkTinyForwardRecoveryEnvCfg` with `OP3WalkDangerZoneRewardsCfg`
    (the time-windowed danger-zone upright bonus) in place of the plain
    recovery reward set. Same narrow forward-only commands/reset noise,
    `termination_penalty` back at the default -50 (the low-termination-
    penalty variant tested independently and didn't change the outcome,
    so no reason to keep combining it here)."""

    rewards: OP3WalkDangerZoneRewardsCfg = OP3WalkDangerZoneRewardsCfg()


@configclass
class OP3WalkTinyForwardCpgImitationEnvCfg(OP3WalkTinyForwardRecoveryEnvCfg):
    """`OP3WalkTinyForwardRecoveryEnvCfg` with `OP3WalkCpgImitationRewardsCfg`
    (reward tracking the fallback CPG's nominal gait shape) in place of the
    plain recovery reward set. Same narrow forward-only commands/reset
    noise as every other tiny-forward variant."""

    rewards: OP3WalkCpgImitationRewardsCfg = OP3WalkCpgImitationRewardsCfg()


@configclass
class OP3WalkTinyForwardFootLoadAsymmetryEnvCfg(OP3WalkTinyForwardRecoveryEnvCfg):
    """`OP3WalkTinyForwardRecoveryEnvCfg` with
    `OP3WalkFootLoadAsymmetryRewardsCfg` (continuous, liftoff-independent
    weight-shift reward) in place of the plain recovery reward set. Same
    narrow forward-only commands/reset noise as every other tiny-forward
    variant."""

    rewards: OP3WalkFootLoadAsymmetryRewardsCfg = OP3WalkFootLoadAsymmetryRewardsCfg()


@configclass
class OP3WalkTinyForwardCpgImitationFootLoadEnvCfg(OP3WalkTinyForwardRecoveryEnvCfg):
    """`OP3WalkTinyForwardRecoveryEnvCfg` with
    `OP3WalkCpgImitationFootLoadRewardsCfg` (CPG-imitation + foot-load-
    asymmetry stacked) in place of the plain recovery reward set. Same
    narrow forward-only commands/reset noise as every other tiny-forward
    variant."""

    rewards: OP3WalkCpgImitationFootLoadRewardsCfg = OP3WalkCpgImitationFootLoadRewardsCfg()


@configclass
class OP3WalkTinyForwardCpgResidualEnvCfg(OP3WalkTinyForwardRecoveryEnvCfg):
    """`OP3WalkTinyForwardRecoveryEnvCfg` with the `joint_pos` action term
    replaced by `op3_actions.CpgResidualJointPositionActionCfg`: the raw
    policy output becomes a residual correction around a live CPG
    reference trajectory baked into the *action* itself, instead of the
    policy authoring 100% of joint motion around a fixed default pose.

    A structurally different fix from every reward-side lever tried so
    far (nine of them, all converged to the same collapse -- see
    STAGE4_PPO_TRAINING.md's "Nine levers" / "one last combined attempt"
    sections). `cpg_reference_imitation_l2` only *rewarded* matching this
    same reference and let the policy satisfy that via small static
    wiggling that never produced real stepping. Baking the reference into
    the action instead removes that failure mode by construction: the
    legs cycle through the reference gait regardless of what the policy
    outputs, so genuine stepping happens whether or not the policy has
    "discovered" it should. Explicitly one of the task brief's suggested
    action spaces ("residual actions on top of a nominal gait").

    Keeps the plain 72-dim observation space (no phase-clock observation
    added) and `OP3WalkRecoveryRewardsCfg`-family rewards (not
    `cpg_imitation`/`foot_load_asymmetry`, both now redundant or
    uninformative once the reference is baked into the action) --
    isolates the action-structure change as the one new variable, rather
    than changing multiple things at once.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        joint_names = self.actions.joint_pos.joint_names
        self.actions.joint_pos = op3_actions.CpgResidualJointPositionActionCfg(
            asset_name="robot",
            joint_names=joint_names,
            scale=0.10,
            leg_joint_names=_LEG_JOINTS,
            step_freq_hz=0.6,
            hip_amp=0.30,
            knee_amp=0.35,
            ankle_comp=0.5,
            lateral_shift_amp=0.04,
            lateral_shift_sign=1.0,
        )


@configclass
class OP3WalkTinyForwardCpgResidualSlowEnvCfg(OP3WalkTinyForwardCpgResidualEnvCfg):
    """`OP3WalkTinyForwardCpgResidualEnvCfg` with the reference gait's
    cycle frequency halved (0.6 -> 0.3 Hz), everything else identical.

    The 0.6 Hz residual run converged to a *worse* plateau than every
    free-form actor-reset run -- episode length ~33 steps vs. ~50-65,
    `base_contact` pinned at 1.0 within 300 iterations (faster than any
    prior collapse). Plausible cause: with a small residual scale and a
    near-zero policy output early in training, the robot is mostly just
    running the raw 0.6 Hz feedforward reference (no live tilt feedback
    baked into it) -- and the torque-clipping bottleneck documented
    earlier means the actuators structurally can't track that cycling
    rate plus any needed correction. Same fix logic already validated for
    the reward-only `cpg_imitation` experiment: halving the frequency
    should cut peak torque demand roughly 4x (angular acceleration for a
    fixed swing amplitude scales with frequency squared) -- directly
    testing whether the forced-motion rate, not the residual-action
    architecture itself, was the problem. See STAGE4_PPO_TRAINING.md.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.step_freq_hz = 0.3


@configclass
class OP3WalkTinyForwardCpgResidualFeedbackEnvCfg(OP3WalkTinyForwardCpgResidualEnvCfg):
    """`OP3WalkTinyForwardCpgResidualEnvCfg` with live roll/pitch tilt
    feedback added to the reference (`roll_kp=1.5`, `pitch_kp=1.0`,
    `fallback_gait_controller.py`'s own defaults), at the original 0.6 Hz.

    Halving the frequency (`OP3WalkTinyForwardCpgResidualSlowEnvCfg`)
    made no difference -- both variants converged to the identical
    worse-than-free-form plateau, ruling out cycling rate as the cause.
    The one deliberate difference between the residual-CPG reference and
    `fallback_gait_controller.py`'s actual gait is that the reference used
    here so far has been feedforward-only (no live tilt correction) --
    `fallback_gait_controller.py` always includes the "ankle strategy"
    feedback law. Testing whether that's the missing piece, at the
    original 0.6 Hz (the frequency that feedback law was actually paired
    with in the source it's copied from, not the untested 0.3 Hz
    combination). See STAGE4_PPO_TRAINING.md.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.roll_kp = 1.5
        self.actions.joint_pos.pitch_kp = 1.0


@configclass
class OP3WalkTinyForwardCpgResidualPhaseObsEnvCfg(OP3WalkTinyForwardCpgResidualFeedbackEnvCfg):
    """`OP3WalkTinyForwardCpgResidualFeedbackEnvCfg` plus a gait-phase
    sin/cos observation, phase-locked to the same reference driving the
    action term's offset.

    `op3_walk_tiny_forward_cpg_residual_feedback` reached `model_650.pt`
    still flat (~34-36 step episodes, `feet_air_time`-family terms still
    zero, `base_contact` still near-constant) -- the tilt feedback alone
    didn't change the picture either. The policy has no way to know
    *where* in the reference cycle it currently is; it can only react to
    the physical consequences (its own joint positions/velocities) after
    the fact, not condition a correction on "swing is about to end" or
    "this leg is mid-stance." Adding the same `gait_phase_obs` used by the
    original (non-residual) `OP3WalkGaitPhaseEnvCfg` gives it that
    directly, phase-locked via `period = 1/step_freq_hz` so the
    observation and the action-term reference are the same clock, not two
    independently-ticking ones. Changes the observation shape (72->74),
    so this is a fresh run, not a checkpoint-compatible tweak. See
    STAGE4_PPO_TRAINING.md.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.gait_phase = ObsTerm(
            func=op3_mdp.gait_phase_obs,
            params={"period": 1.0 / self.actions.joint_pos.step_freq_hz},
        )


@configclass
class OP3WalkTinyForwardCpgResidualPhaseObsHighRateEnvCfg(OP3WalkTinyForwardCpgResidualPhaseObsEnvCfg):
    """125 Hz-control variant of the phase-aware residual-CPG stage.

    Current Isaac Lab runs used `sim.dt=0.005` and `decimation=4`, i.e.
    50 Hz policy/control updates. The first MuJoCo comparison run used
    MuJoCo's default `0.002s` physics timestep with decimation 4, i.e.
    125 Hz control, and looked much more stable before that confound was
    noticed. This stage tests that exact frequency hypothesis inside
    Isaac: keep the same residual-CPG action/reward/curriculum, but use
    `sim.dt=0.002` and `decimation=4` so `env.step_dt = 0.008s`.

    Fresh run only: the observation/action shapes match the non-high-rate
    phase-observation stage, but the dynamics and rollout real-time horizon
    differ, so keep logs/checkpoints separate. Use `--num_steps_per_env 64`
    for training to preserve roughly the old 0.48s PPO rollout horizon
    (`64 * 0.008 ~= 0.512s`) instead of shrinking it to 0.192s.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.dt = 0.002
        self.decimation = 4
        self.sim.render_interval = self.decimation


@configclass
class OP3WalkTinyForwardCpgResidualPhaseObsFinePhysicsEnvCfg(OP3WalkTinyForwardCpgResidualPhaseObsEnvCfg):
    """50 Hz-control variant with MuJoCo-matched 0.002s physics steps.

    The matched-frequency MuJoCo result ruled out policy/control rate as
    the main explanation for the PhysX-vs-MuJoCo gap, but it did not rule
    out solver timestep: raw MuJoCo still stepped physics at 0.002s and
    held each action for 10 substeps, while the Isaac baseline stepped
    PhysX at 0.005s and held each action for 4 substeps. This stage keeps
    Isaac's policy/control period at exactly 0.02s (50 Hz) but changes the
    PhysX integration step to 0.002s, matching MuJoCo's internal physics
    cadence as closely as this config surface allows.

    Fresh run only. Observation/action shapes match
    `OP3WalkTinyForwardCpgResidualPhaseObsEnvCfg`, but the dynamics differ.
    Keep checkpoints/logs separate and compare against both the normal
    50 Hz stage and the high-rate stage:
      - normal:      0.005 * 4  = 0.020s control, 0.005s physics
      - finePhysics: 0.002 * 10 = 0.020s control, 0.002s physics
      - highRate:    0.002 * 4  = 0.008s control, 0.002s physics
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.dt = 0.002
        self.decimation = 10
        self.sim.render_interval = self.decimation


@configclass
class OP3WalkTinyForwardCpgImitationSlowEnvCfg(OP3WalkTinyForwardCpgImitationEnvCfg):
    """`OP3WalkTinyForwardCpgImitationEnvCfg` with the halved-frequency
    (0.3 Hz) CPG reference. Same narrow forward-only commands/reset noise."""

    rewards: OP3WalkCpgImitationSlowRewardsCfg = OP3WalkCpgImitationSlowRewardsCfg()


@configclass
class OP3WalkGaitPhaseEnvCfg(OP3WalkStructuredEnvCfg):
    """`OP3WalkStructuredEnvCfg` plus a gait-phase-clock observation and the
    swing-clearance/stance-lateral-COM rewards that reference it, i.e. the
    full version of STAGE4_PPO_TRAINING.md's item 3 recommendation (not just
    its alternating-support half).

    Adds two new dims to the policy observation (`ObsTerm` order is
    preserved, so this appends after `actions`), so a checkpoint from any
    other stage cannot be resumed here directly with `--checkpoint`
    (shape mismatch on the actor's/critic's input layer) -- warm-start with
    `--reset_critic --reset_optimizer` from an actor trained on a *matching*
    obs space only, or start fresh. This is a real tradeoff against the
    ~21k+ iterations already sunk into `op3_walk_easy_rebalanced_torque`;
    see STAGE4_PPO_TRAINING.md for why it's still worth trying.
    """

    rewards: OP3WalkGaitPhaseRewardsCfg = OP3WalkGaitPhaseRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # 1.0s period is an unverified starting guess (roughly "a plausible
        # human-scale step cadence halved for OP3's small size"), not
        # measured from any real or simulated OP3 gait -- there's no gait
        # reference to tune it against yet. The swing_foot_clearance/
        # stance_lateral_com rewards above don't actually read this
        # observation directly (they're contact-driven, not phase-driven);
        # the clock's job is purely to give the policy itself a timing
        # signal to phase-lock its own leg swing against, the same role a
        # CPG's internal clock plays.
        self.observations.policy.gait_phase = ObsTerm(func=op3_mdp.gait_phase_obs, params={"period": 1.0})


@configclass
class OP3WalkGaitPhaseEnvCfg_PLAY(OP3WalkGaitPhaseEnvCfg):
    """Small/no-randomization variant of `OP3WalkGaitPhaseEnvCfg` for playback."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


class OP3FlatEnvCfg_PLAY(OP3FlatEnvCfg):
    """Small/no-randomization variant for interactive playback."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class OP3WalkEasyRebalancedEnvCfg_PLAY(OP3WalkEasyRebalancedEnvCfg):
    """Small/no-randomization variant of `OP3WalkEasyRebalancedEnvCfg` for
    playback -- same pattern as `OP3FlatEnvCfg_PLAY`, but matching the
    actual stage/reward config the current best walk checkpoints
    (`checkpoints/op3_walk_best.pt`) were trained under, so playback
    reflects real trained behavior rather than a different stage's
    command range/disturbance settings."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
