import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_checkpoint_performance import PerformanceThresholds, make_summary


def test_checkpoint_gate_passes_when_metrics_are_within_thresholds():
    summary = make_summary(
        checkpoint="model_good.pt",
        stage="walk_easy",
        num_envs=8,
        max_steps=150,
        fall_steps=[None, None, None, None, None, None, 120, 140],
        env0_max_abs_roll=0.11,
        env0_max_abs_pitch=0.09,
        env0_final_height=0.31,
        env0_final_actual_vel_x=0.04,
        env0_final_actual_vel_y=0.00,
        env0_command_vel_x=0.05,
        env0_command_vel_y=0.00,
        env0_command_yaw=0.00,
        thresholds=PerformanceThresholds(
            max_fallen_by_step40=0,
            max_fallen_by_step50=0,
            max_fallen_by_step100=0,
            min_alive_at_end=6,
            max_abs_roll_env0=0.2,
            max_abs_pitch_env0=0.2,
        ),
    )

    assert summary.passed
    assert summary.failures == []
    assert summary.fallen_by_step40 == 0
    assert summary.fallen_by_step50 == 0
    assert summary.fallen_by_step100 == 0
    assert summary.alive_at_end == 6
    assert summary.first_fall_step == 120


def test_checkpoint_gate_fails_on_early_falls_and_large_tilt():
    summary = make_summary(
        checkpoint="model_bad.pt",
        stage="walk_easy",
        num_envs=8,
        max_steps=150,
        fall_steps=[37, 40, 45, 48, None, None, None, None],
        env0_max_abs_roll=0.95,
        env0_max_abs_pitch=0.62,
        env0_final_height=0.09,
        env0_final_actual_vel_x=-1.0,
        env0_final_actual_vel_y=0.4,
        env0_command_vel_x=0.05,
        env0_command_vel_y=0.00,
        env0_command_yaw=0.00,
        thresholds=PerformanceThresholds(
            max_fallen_by_step40=1,
            max_fallen_by_step50=2,
            min_alive_at_end=6,
            max_abs_roll_env0=0.2,
            max_abs_pitch_env0=0.2,
        ),
    )

    assert not summary.passed
    assert summary.fallen_by_step40 == 2
    assert summary.fallen_by_step50 == 4
    assert summary.alive_at_end == 4
    assert any("fallen_by_step40=2 > 1" == failure for failure in summary.failures)
    assert any("fallen_by_step50=4 > 2" == failure for failure in summary.failures)
    assert any("alive_at_end=4 < 6" == failure for failure in summary.failures)
    assert any(failure.startswith("env0_max_abs_roll=0.950") for failure in summary.failures)
    assert any(failure.startswith("env0_max_abs_pitch=0.620") for failure in summary.failures)
