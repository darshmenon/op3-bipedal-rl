import json
from pathlib import Path


_CONFIG = Path(__file__).resolve().parent.parent / "benchmarks" / "op3_checkpoint_benchmarks.json"


def test_benchmark_case_names_are_unique():
    data = json.loads(_CONFIG.read_text())
    names = [case["name"] for case in data["cases"]]
    assert len(names) == len(set(names))


def test_benchmark_cases_have_required_fields_and_known_stages():
    data = json.loads(_CONFIG.read_text())
    known_stages = {
        "walk",
        "walk_easy",
        "walk_easy_rebalanced",
        "walk_tiny_forward",
        "walk_tiny_forward_recovery",
        "walk_tiny_forward_recovery_low_term",
        "walk_tiny_forward_danger_zone",
        "walk_tiny_forward_cpg_imitation",
        "walk_tiny_forward_cpg_imitation_slow",
        "walk_tiny_forward_foot_load_asymmetry",
        "walk_tiny_forward_cpg_imitation_foot_load",
        "walk_tiny_forward_cpg_residual",
        "walk_recovery",
        "walk_structured",
        "walk_gait_phase",
        "stand",
        "stand_legs_only",
    }
    required_thresholds = {"max_fallen_by_step40", "max_fallen_by_step50"}

    for case in data["cases"]:
        assert case["name"]
        assert case["checkpoint"]
        assert case["stage"] in known_stages
        assert case["num_envs"] > 0
        assert case["max_steps"] >= 40
        assert required_thresholds <= set(case["thresholds"])


def test_benchmark_paths_are_relative_to_repo_root():
    data = json.loads(_CONFIG.read_text())

    for case in data["cases"]:
        checkpoint = Path(case["checkpoint"])
        assert not checkpoint.is_absolute()
        assert ".." not in checkpoint.parts
