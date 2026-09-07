"""Dump Isaac Lab OP3 physics parameters to JSON, for cross-checking against
MuJoCo's config (see scripts/dump_mujoco_physics_params.py and
scripts/compare_physics_parity.py).

Diagnostic, not part of the main training pipeline -- does no physics
stepping, just constructs the config objects and reads their fields, same
class of "fast config sanity check" as tests/test_op3_env_cfg.py. Reads
values live off the actual config classes (not retyped constants) so a
future edit to op3_robot_cfg.py/op3_env_cfg.py that silently drifts from
MuJoCo gets caught by the comparison automatically.

Usage:
    isaaclab.sh -p ../scripts/dump_isaac_physics_params.py --stage walk_easy_rebalanced
"""
import argparse
import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stage", default="walk_easy_rebalanced",
    help="Which env cfg class to inspect for stage-level settings (sim.dt/decimation) -- "
    "these can differ per-stage (see OP3WalkTinyForwardCpgResidualPhaseObsFinePhysicsEnvCfg). "
    "PD gains/friction/collider come from OP3_CFG directly and don't vary by stage.",
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# isaaclab.envs must be imported before op3_rl's own configs -- op3_env_cfg
# pulls in op3_actions, which imports ActionTerm from
# isaaclab.managers.action_manager; importing op3_rl first hits that module
# mid-initialization (a real circular-import sensitivity in this IsaacLab
# version, not a GPU-contention issue -- reproduces with no other Isaac Sim
# process running). tests/test_op3_env_cfg.py already imports in this same
# order for the same reason.
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402, F401

from op3_rl.op3_robot_cfg import OP3_CFG  # noqa: E402
from op3_rl import op3_env_cfg  # noqa: E402

_STAGE_CLASS_MAP = {
    "walk": op3_env_cfg.OP3FlatEnvCfg,
    "walk_easy": op3_env_cfg.OP3WalkEasyEnvCfg,
    "walk_easy_rebalanced": op3_env_cfg.OP3WalkEasyRebalancedEnvCfg,
    "walk_tiny_forward": op3_env_cfg.OP3WalkTinyForwardEnvCfg,
    "walk_recovery": op3_env_cfg.OP3WalkRecoveryEnvCfg,
    "walk_structured": op3_env_cfg.OP3WalkStructuredEnvCfg,
    "stand": op3_env_cfg.OP3StandEnvCfg,
}


def main() -> None:
    env_cls = _STAGE_CLASS_MAP.get(args.stage)
    if env_cls is None:
        print(f"Unknown --stage {args.stage!r}; known: {sorted(_STAGE_CLASS_MAP)}", file=sys.stderr)
        sys.exit(2)
    env_cfg = env_cls()

    # OP3_CFG.actuators is a dict of ImplicitActuatorCfg keyed by actuator
    # group (e.g. "legs", "arms_head") -- both groups were set uniformly in
    # Revision 5 (see op3_robot_cfg.py), but dump per-group rather than
    # assume, so a future split shows up here instead of silently passing.
    actuators = {}
    for group_name, act_cfg in OP3_CFG.actuators.items():
        actuators[group_name] = {
            "joint_expr": act_cfg.joint_names_expr,
            "stiffness": act_cfg.stiffness,
            "damping": act_cfg.damping,
            "armature": act_cfg.armature,
            "friction": act_cfg.friction,
            "effort_limit_sim": act_cfg.effort_limit_sim,
        }

    spawn_cfg = OP3_CFG.spawn
    collision_props = spawn_cfg.collision_props
    articulation_props = spawn_cfg.articulation_props
    terrain_material = env_cfg.scene.terrain.physics_material
    # `UsdFileCfg` (the robot's own spawner) has no per-asset `physics_material`
    # field -- OP3's feet inherit whatever `sim.physics_material` (the
    # scene-wide default) resolves to instead. Report that value under
    # `foot_physics_material` since it's what actually governs foot-ground
    # friction, not a robot-specific setting.
    scene_default_material = env_cfg.sim.physics_material

    params = {
        "stage": args.stage,
        "sim_dt": env_cfg.sim.dt,
        "decimation": env_cfg.decimation,
        "control_dt": env_cfg.sim.dt * env_cfg.decimation,
        "gravity": list(env_cfg.sim.gravity),
        "solver_position_iteration_count": articulation_props.solver_position_iteration_count if articulation_props else None,
        "solver_velocity_iteration_count": articulation_props.solver_velocity_iteration_count if articulation_props else None,
        "contact_offset": collision_props.contact_offset if collision_props else None,
        "rest_offset": collision_props.rest_offset if collision_props else None,
        "foot_physics_material": {
            "static_friction": scene_default_material.static_friction,
            "dynamic_friction": scene_default_material.dynamic_friction,
            "restitution": scene_default_material.restitution,
        } if scene_default_material else None,
        "terrain_physics_material": {
            "static_friction": terrain_material.static_friction,
            "dynamic_friction": terrain_material.dynamic_friction,
            "restitution": terrain_material.restitution,
        } if terrain_material else None,
        "actuators": actuators,
        "init_pos": list(OP3_CFG.init_state.pos),
        "init_joint_pos": OP3_CFG.init_state.joint_pos,
    }

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "physics_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "isaac_params.json"
    out_path.write_text(json.dumps(params, indent=2))
    print(f"Wrote {out_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
