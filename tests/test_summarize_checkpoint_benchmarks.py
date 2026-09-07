import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.summarize_checkpoint_benchmarks import load_rows, rank_rows, render_markdown


def _write_result(path: Path, checkpoint: str, *, passed: bool, fall50: int, alive: int) -> None:
    payload = {
        "passed": passed,
        "summaries": [
            {
                "checkpoint": checkpoint,
                "stage": "walk_easy_rebalanced",
                "num_envs": 128,
                "max_steps": 200,
                "fallen_by_step40": max(0, fall50 - 2),
                "fallen_by_step50": fall50,
                "fallen_by_step100": fall50 + 10,
                "alive_at_end": alive,
                "first_fall_step": 41,
                "median_fall_step": 58.0,
                "mean_fall_step": 61.5,
                "env0_max_abs_roll": 0.2,
                "env0_max_abs_pitch": 0.3,
                "env0_final_height": 0.31,
                "env0_final_actual_vel_x": 0.04,
                "env0_final_actual_vel_y": 0.00,
                "env0_command_vel_x": 0.05,
                "env0_command_vel_y": 0.00,
                "env0_command_yaw": 0.00,
                "max_abs_joint_pos": 1.0,
                "max_abs_joint_vel": 2.0,
                "max_abs_applied_torque": 4.0,
                "max_abs_computed_torque": 5.5,
                "torque_clip_fraction": 0.12,
                "passed": passed,
                "failures": [] if passed else ["fallen_by_step50 too high"],
            }
        ],
    }
    path.write_text(json.dumps(payload))


def test_load_rows_uses_file_stem_as_case_name(tmp_path):
    result = tmp_path / "candidate.json"
    _write_result(result, "/repo/logs/run/model_100.pt", passed=True, fall50=4, alive=100)

    rows = load_rows([result])

    assert len(rows) == 1
    assert rows[0].case == "candidate"
    assert rows[0].checkpoint == "/repo/logs/run/model_100.pt"
    assert rows[0].fallen_by_step50 == 4


def test_rank_rows_prefers_passing_lower_early_fall_count(tmp_path):
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    _write_result(good, "/repo/logs/good/model_100.pt", passed=True, fall50=8, alive=90)
    _write_result(bad, "/repo/logs/bad/model_100.pt", passed=False, fall50=2, alive=120)

    ranked = rank_rows(load_rows([bad, good]))

    assert ranked[0].case == "good"
    assert ranked[1].case == "bad"


def test_render_markdown_includes_table_and_failures(tmp_path):
    result = tmp_path / "candidate.json"
    _write_result(result, "/repo/logs/run/model_100.pt", passed=False, fall50=40, alive=50)

    text = render_markdown(load_rows([result]))

    assert "| rank | case | pass | checkpoint |" in text
    assert "logs/run/model_100.pt" in text
    assert "Failures:" in text
    assert "fallen_by_step50 too high" in text
