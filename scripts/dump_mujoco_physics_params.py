#!/usr/bin/env python3
"""Dump MuJoCo OP3 physics parameters to JSON, for cross-checking against
Isaac Lab's config (see scripts/dump_isaac_physics_params.py and
scripts/compare_physics_parity.py).

This exists because this project has repeatedly found real Isaac-vs-MuJoCo
mismatches by hand, one at a time (foot collider shape, foot-ground
friction, physics timestep) after weeks of confusing training results --
see STAGE4_PPO_TRAINING.md. A repeatable dump-and-diff pair catches the
next one automatically instead of needing another manual investigation.

Run with the project's own .venv (has the `mujoco` pip package; Isaac Sim's
python does not):
    python3 scripts/dump_mujoco_physics_params.py
"""
import json
from pathlib import Path

import mujoco
import numpy as np

_MODEL_PATH = Path("/home/asimov/mujoco_menagerie/robotis_op3/scene.xml")

_ALL_JOINTS = [
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll",
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll",
    "l_sho_pitch", "l_sho_roll", "l_el", "r_sho_pitch", "r_sho_roll", "r_el", "head_pan", "head_tilt",
]
_FEET_BODIES = ["l_ank_roll_link", "r_ank_roll_link"]

# The two `class="foot"` box geoms per ankle-roll link (op3.xml:182-183,
# 225-226) are unnamed in the source MJCF, so they're identified by
# (body, geom type) below rather than by name -- unlike a named geom,
# `model.geom(<name>)` isn't usable here. These are the exact boxes
# scripts/fix_foot_collision.py hand-copied into the Isaac Lab USD; dumping
# them straight from the live model (not retyped) means a future upstream
# op3.xml change that moves/resizes them gets caught by the comparison
# instead of silently going stale in the USD-patch script.
_MJGEOM_BOX = 6  # mjtGeom.mjGEOM_BOX


def _box_geoms_for_body(model: "mujoco.MjModel", body_name: str) -> list[int]:
    body_id = model.body(body_name).id
    return [
        g for g in range(model.ngeom)
        if model.geom_bodyid[g] == body_id and model.geom_type[g] == _MJGEOM_BOX
    ]


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(_MODEL_PATH))

    joints = {}
    for name in _ALL_JOINTS:
        j = model.joint(name)
        act = model.actuator(f"{name}_act")
        joints[name] = {
            "damping": float(j.damping[0]),
            "armature": float(j.armature[0]),
            "frictionloss": float(j.frictionloss[0]),
            "actuator_kp": float(act.gainprm[0]),
            "forcerange": [float(x) for x in act.forcerange],
        }

    feet = {}
    for body_name in _FEET_BODIES:
        boxes = []
        for gid in _box_geoms_for_body(model, body_name):
            boxes.append({
                "geom_id": gid,
                "pos": [float(x) for x in model.geom_pos[gid]],
                "half_extent": [float(x) for x in model.geom_size[gid]],
                "friction": [float(x) for x in model.geom_friction[gid]],
            })
        feet[body_name] = {"boxes": boxes}

    # scene.xml's ground plane geom -- found by type rather than assuming
    # index 0, since world-body geom ordering isn't a documented guarantee.
    floor_candidates = [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0]
    assert floor_candidates, "no world-body (floor) geom found in the model"
    floor_friction = [float(x) for x in model.geom_friction[floor_candidates[0]]]

    params = {
        "source": str(_MODEL_PATH),
        "timestep": float(model.opt.timestep),
        "gravity": [float(x) for x in model.opt.gravity],
        "solver_iterations": int(model.opt.iterations),
        "total_mass_kg": float(np.sum(model.body_mass)),
        "joints": joints,
        "feet": feet,
        "floor_friction": floor_friction,
    }

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "physics_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mujoco_params.json"
    out_path.write_text(json.dumps(params, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
