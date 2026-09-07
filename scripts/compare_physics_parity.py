#!/usr/bin/env python3
"""Compare dumped Isaac Lab vs MuJoCo OP3 physics parameters and report
mismatches -- the "test all the known Isaac/MuJoCo bugs together" tool.

This project found three real Isaac-vs-MuJoCo mismatches one at a time, by
hand, over several sessions (foot collider shape, foot-ground friction,
and a still-open physics-substep-timestep difference) -- see
STAGE4_PPO_TRAINING.md. This script re-checks all of them (and the
properties already confirmed to match) in one pass, against the live
config on both sides, so the next drift gets caught immediately instead of
needing another manual investigation.

Plain Python, no Isaac/MuJoCo runtime deps -- just reads the two JSON dumps.

Usage (from repo root):
    # 1. Dump MuJoCo params (project .venv):
    .venv/bin/python3 scripts/dump_mujoco_physics_params.py
    # 2. Dump Isaac params (needs isaaclab.sh, ~15s Isaac Sim boot):
    (cd IsaacLab && TERM=xterm-256color ./isaaclab.sh -p ../scripts/dump_isaac_physics_params.py --stage <stage>)
    # 3. Compare:
    .venv/bin/python3 scripts/compare_physics_parity.py

Or just run scripts/run_physics_parity_check.sh, which does all three.

Exit code is nonzero if any check marked "must match" fails -- suitable for
wiring into a pre-training sanity gate later, not just interactive use.
"""
import json
import sys
from pathlib import Path

_OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "physics_parity"
_TOL = 1e-6

_ALL_JOINTS = [
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll",
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll",
    "l_sho_pitch", "l_sho_roll", "l_el", "r_sho_pitch", "r_sho_roll", "r_el", "head_pan", "head_tilt",
]


def _close(a, b, tol=_TOL) -> bool:
    return abs(a - b) <= tol


class Reporter:
    def __init__(self):
        self.rows = []  # (property, isaac_val, mujoco_val, verdict, note)
        self.failures = 0

    def check(self, name, isaac_val, mujoco_val, must_match: bool, note: str = "") -> None:
        if isinstance(isaac_val, (int, float)) and isinstance(mujoco_val, (int, float)):
            matched = _close(float(isaac_val), float(mujoco_val))
        else:
            matched = isaac_val == mujoco_val
        if matched:
            verdict = "MATCH"
        elif must_match:
            verdict = "MISMATCH"
            self.failures += 1
        else:
            verdict = "DIFFERS (known/allowed)"
        self.rows.append((name, isaac_val, mujoco_val, verdict, note))

    def info(self, name, isaac_val, mujoco_val, note: str = "") -> None:
        self.rows.append((name, isaac_val, mujoco_val, "INFO", note))

    def print_table(self) -> None:
        widths = [34, 22, 22, 24]
        header = ["property", "isaac", "mujoco", "verdict"]
        print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
        print("-+-".join("-" * w for w in widths))
        for name, iv, mv, verdict, note in self.rows:
            line = " | ".join([
                str(name).ljust(widths[0]),
                str(iv).ljust(widths[1]),
                str(mv).ljust(widths[2]),
                verdict.ljust(widths[3]),
            ])
            print(line)
            if note:
                print(f"    -> {note}")


def main() -> int:
    isaac_path = _OUT_DIR / "isaac_params.json"
    mujoco_path = _OUT_DIR / "mujoco_params.json"
    for p in (isaac_path, mujoco_path):
        if not p.exists():
            print(f"Missing {p} -- run the matching dump script first (see this script's docstring).", file=sys.stderr)
            return 2

    isaac = json.loads(isaac_path.read_text())
    mujoco = json.loads(mujoco_path.read_text())

    r = Reporter()

    # --- Per-joint actuator params: Isaac's are uniform across all joints
    # (one "legs" + one "arms_head" ImplicitActuatorCfg group), so check
    # each MuJoCo joint against whichever Isaac group covers it.
    isaac_groups = isaac["actuators"]
    joint_to_group = {}
    for group_name, group in isaac_groups.items():
        for expr in group["joint_expr"]:
            joint_to_group[expr.rstrip("$")] = group_name

    for joint in _ALL_JOINTS:
        group_name = joint_to_group.get(joint)
        if group_name is None:
            r.check(f"joint[{joint}] present in Isaac actuator groups", False, True, must_match=True,
                     note="joint missing from every Isaac actuator group's joint_names_expr")
            continue
        g = isaac_groups[group_name]
        m = mujoco["joints"][joint]
        r.check(f"joint[{joint}].stiffness_vs_kp", g["stiffness"], m["actuator_kp"], must_match=True)
        r.check(f"joint[{joint}].damping", g["damping"], m["damping"], must_match=True)
        r.check(f"joint[{joint}].armature", g["armature"], m["armature"], must_match=True)
        r.check(f"joint[{joint}].effort_limit_vs_forcerange", g["effort_limit_sim"], m["forcerange"][1], must_match=True)
        # Isaac's `friction` is a unitless static-friction coefficient;
        # MuJoCo's `frictionloss` is a dry-friction force (N*m). Same
        # nominal number by convention in this project (see
        # op3_robot_cfg.py Revision 5), not a verified unit match -- report,
        # don't fail the gate on it.
        r.check(f"joint[{joint}].friction_vs_frictionloss (unit mismatch, informational)",
                 g["friction"], m["frictionloss"], must_match=False,
                 note="same nominal value by project convention; units differ (coefficient vs N*m) and are not verified equivalent")

    # --- Foot-ground friction (the second real bug found this session)
    isaac_foot_mat = isaac["foot_physics_material"]
    mujoco_foot_friction = mujoco["feet"]["l_ank_roll_link"]["boxes"][0]["friction"][0]  # sliding component
    r.check("foot_ground_static_friction", isaac_foot_mat["static_friction"], mujoco_foot_friction, must_match=True,
            note="governed by Isaac's scene-wide sim.physics_material default (UsdFileCfg has no per-asset field) -- see op3_env_cfg.py")
    r.check("foot_ground_dynamic_friction", isaac_foot_mat["dynamic_friction"], mujoco_foot_friction, must_match=True)

    isaac_terrain_mat = isaac["terrain_physics_material"]
    r.check("terrain_static_friction", isaac_terrain_mat["static_friction"], mujoco["floor_friction"][0], must_match=True)

    # --- Foot collider geometry (the first real bug found this session)
    # fix_foot_collision.py hand-copied MuJoCo's box pos/size into the USD;
    # re-derive the same numbers from the live MuJoCo model here so a future
    # upstream op3.xml change is caught, rather than re-checking the USD
    # itself (would need usd-core, a non-project-venv dependency).
    for body in ("l_ank_roll_link", "r_ank_roll_link"):
        boxes = mujoco["feet"][body]["boxes"]
        r.check(f"foot_collider[{body}].box_count", 2, len(boxes), must_match=True,
                note="fix_foot_collision.py assumes exactly 2 box geoms per foot")
    r.info("foot_collider_geometry_source", "robot/op3/usd/... (hand-patched by fix_foot_collision.py, not live-introspected here)",
           "read live from scene.xml above",
           note="Isaac's actual USD collider geometry isn't dumped (would need usd-core); "
                "cross-check fix_foot_collision.py's hardcoded FEET constants against mujoco_params.json's feet.*.boxes by eye if op3.xml ever changes")

    # --- Physics timestep: control-rate matches, physics substep does not
    # for this stage (walk_easy_rebalanced never adopted the 0.002s substep
    # a different experiment branch tested and found no improvement from --
    # see OP3WalkTinyForwardCpgResidualPhaseObsFinePhysicsEnvCfg). Not a bug
    # to fix, but worth surfacing every time so it's never mistaken for
    # "matched" again.
    r.check("control_dt (policy rate)", isaac["control_dt"], mujoco["timestep"] * 10, must_match=True,
            note="MuJoCo's own control_dt = timestep * decimation(10), hardcoded to match op3_rl/mujoco_env.py's _DECIMATION")
    r.check(f"physics_substep_dt (stage={isaac['stage']})", isaac["sim_dt"], mujoco["timestep"], must_match=False,
            note="known, already-tested difference (0.005s vs 0.002s) for this stage specifically -- "
                 "a separate experiment branch (FinePhysics variant) matched this and found no improvement, "
                 "so not currently treated as a live bug, but flag if that stage's own result ever gets reused here")

    # --- Solver iterations: different mechanisms, not a 1:1 comparable number
    r.info("solver_position_iteration_count", isaac["solver_position_iteration_count"], mujoco["solver_iterations"],
           note="different solvers (PhysX TGS vs MuJoCo's own); not directly comparable, no fix implied")

    # --- Gravity: trivial sanity check
    r.check("gravity_z", isaac["gravity"][2], mujoco["gravity"][2], must_match=True)

    r.print_table()
    print()
    if r.failures:
        print(f"{r.failures} unexpected mismatch(es) -- see MISMATCH rows above.")
    else:
        print("No unexpected mismatches (DIFFERS rows are known/already-investigated, not failures).")
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())
