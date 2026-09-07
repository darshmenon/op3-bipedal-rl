"""Fast config sanity tests for the OP3 RL setup (op3_rl/).

These don't validate walking behavior (that needs a real training/playback
run, see scripts/play_ppo.py and diagnose_static_balance.py) -- they catch
the class of mistake that would otherwise only surface minutes into a real
sim run: a typo'd joint name, a config that fails to construct, a
manager wired to the wrong number of terms, an actuator param that
silently reverted.

Isaac Lab requires the Kit/omniverse runtime (`omni.*`) to be initialized
before its modules can even be imported, so this still needs to run
through `isaaclab.sh -p` rather than plain pytest -- but unlike a training
or playback run it does no physics stepping or rendering, so it finishes
in seconds, not minutes.

Usage:
    isaaclab.sh -p -m pytest ../tests/test_op3_env_cfg.py -v
"""

import sys
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from op3_rl.op3_actions import CpgResidualJointPositionActionCfg  # noqa: E402
from op3_rl.op3_env_cfg import (  # noqa: E402
    OP3FlatEnvCfg,
    OP3FlatEnvCfg_PLAY,
    OP3StandEnvCfg,
    OP3WalkTinyForwardCpgResidualEnvCfg,
    OP3WalkTinyForwardCpgResidualFeedbackEnvCfg,
    OP3WalkTinyForwardCpgResidualPhaseObsEnvCfg,
    OP3WalkTinyForwardCpgResidualPhaseObsFinePhysicsEnvCfg,
    OP3WalkTinyForwardCpgResidualPhaseObsHighRateEnvCfg,
)
from op3_rl.op3_robot_cfg import OP3_CFG, _ARM_HEAD_JOINTS, _LEG_JOINTS  # noqa: E402

_URDF_PATH = Path(__file__).resolve().parent.parent / "robot" / "op3" / "robotis_op3.urdf"


def _urdf_joint_names() -> set[str]:
    import re

    text = _URDF_PATH.read_text()
    return set(re.findall(r'<joint name="([^"]+)"', text))


@pytest.fixture(scope="session", autouse=True)
def _close_simulation_app():
    yield
    simulation_app.close()


@pytest.fixture(scope="module")
def env():
    cfg = OP3FlatEnvCfg()
    cfg.scene.num_envs = 1
    e = ManagerBasedRLEnv(cfg=cfg)
    yield e
    e.close()


def test_flat_env_cfg_constructs():
    OP3FlatEnvCfg()


def test_play_env_cfg_constructs_with_reduced_envs():
    cfg = OP3FlatEnvCfg_PLAY()
    assert cfg.scene.num_envs == 16
    assert cfg.scene.env_spacing == 2.5
    assert cfg.observations.policy.enable_corruption is False


def test_flat_env_disables_unvalidated_startup_randomization():
    cfg = OP3FlatEnvCfg()
    assert cfg.events.add_base_mass is None
    assert cfg.events.base_com is None


def test_stand_env_is_zero_command_survival_stage():
    cfg = OP3StandEnvCfg()
    cmd = cfg.commands.base_velocity
    assert cmd.heading_command is False
    assert cmd.rel_standing_envs == pytest.approx(1.0)
    assert cmd.rel_heading_envs == pytest.approx(0.0)
    assert cmd.ranges.lin_vel_x == (0.0, 0.0)
    assert cmd.ranges.lin_vel_y == (0.0, 0.0)
    assert cmd.ranges.ang_vel_z == (0.0, 0.0)
    assert cmd.ranges.heading == (0.0, 0.0)
    assert cfg.events.base_external_force_torque is None
    assert cfg.events.push_robot is None
    # Deliberately kept at the walk stage's 0.3, not a stand-specific smaller
    # value -- see the comment in OP3StandEnvCfg.__post_init__ (op3_env_cfg.py):
    # a stand-trained policy warm-started into the walk stage at a different
    # action scale sees the same raw network output mapped to a different
    # physical joint movement, a likely cause of an earlier regression.
    assert cfg.actions.joint_pos.scale == pytest.approx(0.3)
    assert cfg.rewards.alive.weight == pytest.approx(1.0)
    assert cfg.rewards.termination_penalty.weight == pytest.approx(-500.0)
    assert cfg.rewards.track_lin_vel_xy_exp.weight == pytest.approx(0.0)
    assert cfg.rewards.track_ang_vel_z_exp.weight == pytest.approx(0.0)
    assert cfg.rewards.feet_air_time.weight == pytest.approx(0.0)
    assert cfg.rewards.flat_orientation_l2.weight == pytest.approx(-8.0)


def test_actuator_joint_names_exist_in_urdf():
    urdf_joints = _urdf_joint_names()
    controlled = set(_LEG_JOINTS) | set(_ARM_HEAD_JOINTS)
    assert len(controlled) == 20, "expected all 20 OP3 joints to be controlled"
    missing = controlled - urdf_joints
    assert not missing, f"joints referenced in op3_robot_cfg.py but absent from the URDF: {missing}"


def test_actuator_params_match_revision_5():
    # Revision 5 (STAGE4_PPO_TRAINING.md): sourced from ROBOTIS's own
    # MuJoCo model (robotis_op3/op3.xml), applied uniformly to both groups.
    for group in ("legs", "arms_head"):
        actuator = OP3_CFG.actuators[group]
        assert actuator.stiffness == pytest.approx(21.1)
        assert actuator.damping == pytest.approx(1.084)
        assert actuator.effort_limit_sim == pytest.approx(5.0)
        assert actuator.armature == pytest.approx(0.045)
        assert actuator.friction == pytest.approx(0.03)


def test_env_action_and_observation_dims(env):
    assert env.action_manager.total_action_dim == 20
    obs, _ = env.reset()
    assert obs["policy"].shape[-1] == 72


def test_residual_cpg_stage_keeps_checkpoint_compatible_action_shape():
    cfg = OP3WalkTinyForwardCpgResidualEnvCfg()
    assert isinstance(cfg.actions.joint_pos, CpgResidualJointPositionActionCfg)
    assert cfg.actions.joint_pos.joint_names == [".*"]
    assert cfg.actions.joint_pos.scale == pytest.approx(0.10)
    assert cfg.actions.joint_pos.leg_joint_names == _LEG_JOINTS
    assert cfg.actions.joint_pos.step_freq_hz == pytest.approx(0.6)
    assert cfg.actions.joint_pos.lateral_shift_amp == pytest.approx(0.04)


def test_residual_cpg_phase_obs_stage_adds_phase_without_changing_action_shape():
    feedback_cfg = OP3WalkTinyForwardCpgResidualFeedbackEnvCfg()
    cfg = OP3WalkTinyForwardCpgResidualPhaseObsEnvCfg()
    assert isinstance(cfg.actions.joint_pos, CpgResidualJointPositionActionCfg)
    assert cfg.actions.joint_pos.joint_names == feedback_cfg.actions.joint_pos.joint_names
    assert cfg.actions.joint_pos.scale == pytest.approx(feedback_cfg.actions.joint_pos.scale)
    assert cfg.actions.joint_pos.roll_kp == pytest.approx(1.5)
    assert cfg.actions.joint_pos.pitch_kp == pytest.approx(1.0)
    assert cfg.observations.policy.gait_phase.func is not None
    assert cfg.observations.policy.gait_phase.params["period"] == pytest.approx(
        1.0 / cfg.actions.joint_pos.step_freq_hz
    )


def test_residual_cpg_high_rate_stage_uses_mujoco_physics_dt_and_125hz_control():
    cfg = OP3WalkTinyForwardCpgResidualPhaseObsHighRateEnvCfg()
    assert cfg.sim.dt == pytest.approx(0.002)
    assert cfg.decimation == 4
    assert cfg.sim.render_interval == cfg.decimation
    assert cfg.sim.dt * cfg.decimation == pytest.approx(0.008)
    assert cfg.observations.policy.gait_phase.params["period"] == pytest.approx(
        1.0 / cfg.actions.joint_pos.step_freq_hz
    )


def test_residual_cpg_fine_physics_stage_keeps_50hz_control_with_mujoco_physics_dt():
    cfg = OP3WalkTinyForwardCpgResidualPhaseObsFinePhysicsEnvCfg()
    assert cfg.sim.dt == pytest.approx(0.002)
    assert cfg.decimation == 10
    assert cfg.sim.render_interval == cfg.decimation
    assert cfg.sim.dt * cfg.decimation == pytest.approx(0.02)
    assert cfg.observations.policy.gait_phase.params["period"] == pytest.approx(
        1.0 / cfg.actions.joint_pos.step_freq_hz
    )


def test_reward_manager_has_expected_term_count(env):
    assert len(env.reward_manager.active_terms) == 13


def test_termination_manager_has_time_out_and_base_contact(env):
    assert set(env.termination_manager.active_terms) == {"time_out", "base_contact"}
