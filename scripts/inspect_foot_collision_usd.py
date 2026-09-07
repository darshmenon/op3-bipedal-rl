#!/usr/bin/env python3
"""Inspect OP3 foot collision prims in the generated USD.

This is a narrow verification tool for scripts/fix_foot_collision.py. It
prints whether the old ankle-roll mesh colliders are disabled and whether the
MuJoCo-style sole box colliders are present with PhysicsCollisionAPI.
"""

import argparse
import json
from pathlib import Path

try:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
    simulation_app = None
except ModuleNotFoundError:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHYSICS_USD = PROJECT_ROOT / "robot" / "op3" / "usd" / "configuration" / "robotis_op3_physics.usd"
ROOT_USD = PROJECT_ROOT / "robot" / "op3" / "usd" / "robotis_op3.usd"

EXPECTED = {
    "l_ank_roll_link": ("ll6", ((0.024, 0.013, -0.0265), (0.024, 0.0125, -0.0265))),
    "r_ank_roll_link": ("rl6", ((0.024, -0.013, -0.0265), (0.024, -0.0125, -0.0265))),
}


def _get_attr(prim, name):
    attr = prim.GetAttribute(name)
    return attr.Get() if attr and attr.HasAuthoredValueOpinion() else None


def _has_collision_api(prim):
    return prim.HasAPI(UsdPhysics.CollisionAPI)


def _physx_offsets(prim):
    if not prim.IsValid() or not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        return None, None
    api = PhysxSchema.PhysxCollisionAPI(prim)
    return api.GetContactOffsetAttr().Get(), api.GetRestOffsetAttr().Get()


def _near(value, expected, eps=1.0e-7):
    return value is not None and abs(float(value) - expected) <= eps


def _inspect_physics_layer():
    stage = Usd.Stage.Open(str(PHYSICS_USD))
    print(f"physics_layer={PHYSICS_USD}")
    ok = True
    rows = []

    for link, (mesh, centers) in EXPECTED.items():
        mesh_path = f"/colliders/{link}/{mesh}/node_STL_BINARY_"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        mesh_enabled = _get_attr(mesh_prim, "physics:collisionEnabled") if mesh_prim.IsValid() else None
        mesh_approx = _get_attr(mesh_prim, "physics:approximation") if mesh_prim.IsValid() else None
        contact_offset, rest_offset = _physx_offsets(mesh_prim)
        print(
            f"physics mesh {mesh_path}: valid={mesh_prim.IsValid()} "
            f"enabled={mesh_enabled} approximation={mesh_approx} "
            f"contact_offset={contact_offset} rest_offset={rest_offset}"
        )
        rows.append(
            {
                "stage": "physics",
                "kind": "mesh",
                "path": mesh_path,
                "valid": mesh_prim.IsValid(),
                "collision_enabled": mesh_enabled,
                "approximation": mesh_approx,
                "contact_offset": contact_offset,
                "rest_offset": rest_offset,
            }
        )
        ok = ok and mesh_prim.IsValid() and mesh_enabled is False and _near(contact_offset, 0.005) and _near(rest_offset, 0.0)

        for i, center in enumerate(centers):
            box_path = f"/colliders/{link}/sole_collider_{i}"
            box_prim = stage.GetPrimAtPath(box_path)
            xform = UsdGeom.Xformable(box_prim) if box_prim.IsValid() else None
            ops = xform.GetOrderedXformOps() if xform else []
            values = [op.Get() for op in ops]
            enabled = _get_attr(box_prim, "physics:collisionEnabled") if box_prim.IsValid() else None
            contact_offset, rest_offset = _physx_offsets(box_prim)
            print(
                f"physics sole {box_path}: valid={box_prim.IsValid()} "
                f"collision_api={_has_collision_api(box_prim) if box_prim.IsValid() else False} "
                f"enabled={enabled} contact_offset={contact_offset} rest_offset={rest_offset} ops={values}"
            )
            rows.append(
                {
                    "stage": "physics",
                    "kind": "sole_box",
                    "path": box_path,
                    "valid": box_prim.IsValid(),
                    "collision_api": _has_collision_api(box_prim) if box_prim.IsValid() else False,
                    "collision_enabled": enabled,
                    "contact_offset": contact_offset,
                    "rest_offset": rest_offset,
                    "xform_values": [tuple(float(x) for x in value) for value in values],
                }
            )
            ok = (
                ok
                and box_prim.IsValid()
                and _has_collision_api(box_prim)
                and enabled is True
                and _near(contact_offset, 0.005)
                and _near(rest_offset, 0.0)
                and values
                and tuple(round(float(v), 7) for v in values[0]) == center
            )

    print(f"physics_layer_ok={ok}")
    return ok, rows


def _inspect_composed_stage():
    stage = Usd.Stage.Open(str(ROOT_USD))
    print(f"root_stage={ROOT_USD}")
    ok = True
    rows = []

    for link, (mesh, _centers) in EXPECTED.items():
        paths = [
            f"/robotis_op3/{link}/collisions/{mesh}/node_STL_BINARY_",
            f"/robotis_op3/{link}/collisions/sole_collider_0",
            f"/robotis_op3/{link}/collisions/sole_collider_1",
        ]
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            enabled = _get_attr(prim, "physics:collisionEnabled") if prim.IsValid() else None
            contact_offset, rest_offset = _physx_offsets(prim)
            print(
                f"root prim {path}: valid={prim.IsValid()} "
                f"collision_api={_has_collision_api(prim) if prim.IsValid() else False} enabled={enabled} "
                f"contact_offset={contact_offset} rest_offset={rest_offset}"
            )
            rows.append(
                {
                    "stage": "root",
                    "path": path,
                    "valid": prim.IsValid(),
                    "collision_api": _has_collision_api(prim) if prim.IsValid() else False,
                    "collision_enabled": enabled,
                    "contact_offset": contact_offset,
                    "rest_offset": rest_offset,
                }
            )
            ok = ok and prim.IsValid()

    print(f"root_stage_ok={ok}")
    return ok, rows


def _inspect_instance_flags():
    rows = []
    for label, usd_path in (("physics", PHYSICS_USD), ("root", ROOT_USD)):
        stage = Usd.Stage.Open(str(usd_path))
        for prim in stage.Traverse():
            if prim.IsInstanceable() or prim.IsInstance() or prim.IsInstanceProxy():
                rows.append(
                    {
                        "stage": label,
                        "path": str(prim.GetPath()),
                        "is_instanceable": prim.IsInstanceable(),
                        "is_instance": prim.IsInstance(),
                        "is_instance_proxy": prim.IsInstanceProxy(),
                    }
                )
    print(f"instance_flag_count={len(rows)}")
    for row in rows[:80]:
        print(
            "instance_flag "
            f"stage={row['stage']} path={row['path']} "
            f"instanceable={row['is_instanceable']} instance={row['is_instance']} proxy={row['is_instance_proxy']}"
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=Path, default=PROJECT_ROOT / "logs" / "foot_collision_usd_inspection.json")
    args = parser.parse_args()

    physics_ok, physics_rows = _inspect_physics_layer()
    root_ok, root_rows = _inspect_composed_stage()
    instance_rows = _inspect_instance_flags()
    report = {
        "physics_layer": str(PHYSICS_USD),
        "root_stage": str(ROOT_USD),
        "physics_layer_ok": physics_ok,
        "root_stage_ok": root_ok,
        "ok": physics_ok and root_ok,
        "instance_flag_count": len(instance_rows),
        "instance_flags": instance_rows,
        "rows": physics_rows + root_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote_json={args.out_json}")
    if simulation_app is not None:
        simulation_app.close()
    raise SystemExit(0 if physics_ok and root_ok else 1)
