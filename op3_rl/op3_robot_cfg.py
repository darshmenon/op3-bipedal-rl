"""ArticulationCfg for the converted ROBOTIS OP3 (Stage 2 output).

PD gains, effort limits, and the initial standing pose below are
PLACEHOLDERS, not sourced from a Dynamixel datasheet or measured on this
robot -- documented explicitly per the task doc's requirement to state
actuator assumptions. They are picked to be a plausible order of magnitude
for a ~3kg desktop humanoid using small Dynamixel-class servos (XM430
family), not tuned to real hardware. Revisit before trusting any trained
policy.

Revision 2 (see STAGE4_PPO_TRAINING.md): the first training run (legs
stiffness=20, damping=1, effort_limit=4) never learned to balance --
~100% of episodes ended by falling even at iteration 300, with episode
length barely above the untrained baseline. That points to the PD gains
being too weak to hold static balance at all, independent of policy
quality. Bumped leg stiffness/damping/effort_limit up (45/3/6) as the
first, least-speculative fix to try before touching anything else (e.g.
the initial pose).

Revision 4 (see STAGE4_PPO_TRAINING.md, "Sourcing PD gains from real
Dynamixel specs"): leg effort_limit_sim lowered from 6.0 to 4.5 N*m,
grounded in OP3's actual servos -- commonly documented as Dynamixel
XM430-W350-series (stall torque ~4.1 N*m @ 11.1V / ~4.8 N*m @ 12V). This
was recalled datasheet knowledge, not fetched from a real source --
**superseded by Revision 5 below**, kept here only as the intermediate
step in the revision history.

Revision 5 (see STAGE4_PPO_TRAINING.md): found ROBOTIS's own official
MuJoCo model for OP3 -- github.com/ROBOTIS-GIT/robotis_mujoco_menagerie,
`robotis_op3/op3.xml` -- which gives real, uniform-across-all-20-joints
actuator parameters via a MuJoCo `<position>` actuator (the same
"position-controlled with a gain" scheme as Isaac's implicit PD, so these
map far more directly than the Revision 4 Dynamixel-datasheet guess or the
Gazebo PID reference): `kp=21.1` (-> `stiffness`), joint `damping=1.084`
(-> `damping`), `forcerange=[-5, 5]` (-> `effort_limit_sim=5.0`),
`armature=0.045`, `frictionloss=0.03` (-> `friction`, an imperfect analog:
MuJoCo's frictionloss is a dry-friction force in N*m, Isaac's `friction`
field is described as a unitless static-friction coefficient -- same
physical effect, not a verified exact unit match). Applied uniformly to
both actuator groups below, replacing the previous legs/arms_head-split,
empirically-tuned Revision 2 values (45/3 vs 15/1) -- for the next fresh
run (`op3_flat_v5`), not retrofitted onto `op3_flat_v4_long`.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_USD_PATH = _PROJECT_ROOT / "robot" / "op3" / "usd" / "robotis_op3.usd"

_LEG_JOINTS = [
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll",
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll",
]
_ARM_HEAD_JOINTS = [
    "l_sho_pitch", "l_sho_roll", "l_el", "r_sho_pitch", "r_sho_roll", "r_el",
    "head_pan", "head_tilt",
]

OP3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # position=4/velocity=0 (this repo's original values) is Isaac
            # Lab's own quadruped-reference setting (ANYmal, early Unitree
            # quadrupeds) -- but its own bipedal humanoid references (G1/H1
            # in isaaclab_assets/robots/unitree.py) use 8/4, not 4/0. A
            # biped's support polygon is far smaller and far more sensitive
            # to contact-resolution error than a quadruped's, so copying
            # the quadruped default here was an unexamined mismatch, not a
            # deliberate choice -- raised to match the bipedal-humanoid
            # reference instead. This is a solver-accuracy fix, not a
            # torque/effort-limit change -- doesn't touch the
            # hardware-grounded actuator model this project has repeatedly
            # declined to shortcut. See STAGE4_PPO_TRAINING.md.
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # Contact/rest offsets are authored directly into the generated USD by
        # scripts/fix_foot_collision.py. Keeping this unset avoids Isaac Lab's
        # spawn-time `modify_collision_properties` pass, which cannot edit the
        # converter's instanceable collision/visual child prims and was only
        # producing a warning/no-op for the 512-env training asset.
        # No per-asset `physics_material` field exists here: unlike
        # `TerrainImporterCfg`/`GroundPlaneCfg`, IsaacLab's `UsdFileCfg` (this
        # spawner) has no `physics_material` attribute at all -- an earlier
        # attempt to set one here (`UsdFileCfg(physics_material=...)`) raised
        # `TypeError: unexpected keyword argument 'physics_material'` at
        # import time, breaking every training/dump script that imports
        # `OP3_CFG`. Caught immediately by
        # `scripts/dump_isaac_physics_params.py` (see
        # scripts/compare_physics_parity.py), which is exactly the class of
        # regression that tool exists to catch.
        #
        # With no explicit material, OP3's feet (and every other robot
        # collision prim) fall back to the *scene-wide* default material at
        # `SimulationCfg.physics_material` -- see the fix actually applied in
        # `op3_rl/op3_env_cfg.py`'s `OP3FlatEnvCfg.__post_init__`, which sets
        # both the terrain's own material and that scene-wide default to
        # match MuJoCo's `op3.xml` sliding friction of 1.0 (`friction="1
        # 0.005 0.0001"` on its foot geoms) instead of IsaacLab's silently-
        # inherited 0.5/0.5 default. Lower friction lets the stance foot slip
        # out from under the robot, which would produce exactly the "falls
        # almost instantly" symptom already seen. Untested until the next
        # fresh Isaac Lab run confirms it -- see STAGE4_PPO_TRAINING.md.
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Placeholder standing height -- real OP3 stands ~0.45m tall; 0.32m
        # for the pelvis/torso origin is an unverified guess pending an
        # actual "does it stand without falling" check (blocked by the
        # PCIe-link render slowness noted in SETUP.md / STAGE2_ROBOT_IMPORT.md).
        pos=(0.0, 0.0, 0.32),
        joint_pos={
            # Slight symmetric knee-bend crouch instead of dead-straight legs
            # (revision 3, see STAGE4_PPO_TRAINING.md): training runs 1-2
            # (all leg joints at 0.0) never learned to balance even after
            # bumping PD gains -- straight-legged is a near-singular,
            # dynamically unstable starting configuration. These specific
            # values were verified with a small forward-kinematics script
            # against robot/op3/robotis_op3.urdf's actual joint axes/origins
            # (not guessed): they keep each foot within ~1mm of directly
            # under its hip in x while lowering the pelvis slightly. Left
            # and right use opposite-sign angles because l_*/r_* joints in
            # this URDF have mirrored rotation axes for the same physical
            # motion.
            # (Isaac Lab's regex matcher errors on a joint matching more than
            # one key, so this enumerates every joint explicitly rather than
            # mixing specific overrides with a ".*" catch-all.)
            "l_hip_pitch": -0.15, "l_knee": 0.3, "l_ank_pitch": -0.15,
            "r_hip_pitch": 0.15, "r_knee": -0.3, "r_ank_pitch": 0.15,
            "l_hip_yaw": 0.0, "l_hip_roll": 0.0, "l_ank_roll": 0.0,
            "r_hip_yaw": 0.0, "r_hip_roll": 0.0, "r_ank_roll": 0.0,
            "l_sho_pitch": 0.0, "l_sho_roll": 0.0, "l_el": 0.0,
            "r_sho_pitch": 0.0, "r_sho_roll": 0.0, "r_el": 0.0,
            "head_pan": 0.0, "head_tilt": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[f"{j}$" for j in _LEG_JOINTS],
            effort_limit_sim=5.0,
            stiffness=21.1,
            damping=1.084,
            armature=0.045,
            friction=0.03,
        ),
        "arms_head": ImplicitActuatorCfg(
            joint_names_expr=[f"{j}$" for j in _ARM_HEAD_JOINTS],
            effort_limit_sim=5.0,
            stiffness=21.1,
            damping=1.084,
            armature=0.045,
            friction=0.03,
        ),
    },
)
