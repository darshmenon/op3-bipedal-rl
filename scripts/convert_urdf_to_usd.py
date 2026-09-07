"""Stage 2, step 2: convert the flat OP3 URDF (from expand_urdf.sh) to USD.

Run via `isaaclab.sh -p scripts/convert_urdf_to_usd.py` (needs SimulationApp,
since the URDF importer is an Isaac Sim/Kit extension, not a pure-Python lib).

Conversion choices (documented here since they affect the robot's physical
behavior later):
  - fix_base=False: OP3 is a free-standing biped, not a fixed-base arm.
  - self_collision=True: legs/arms can contact each other; matters for a
    biped that will bend at the hips/knees. Off by default in Isaac Lab.
  - collider_type="convex_hull": default; the provided meshes are simple
    enough (per-limb STL segments) that convex hulls are a reasonable
    approximation for Stage 2. Revisit if collision behavior looks wrong.
  - joint_drive: stiffness/damping are NOT auto-parsed from the URDF by the
    converter (its "gains" field has no default and must be supplied
    explicitly). Set to a placeholder 0.0/0.0 here -- Stage 2 is only about
    getting a correct, simulatable articulation, not tuned control. Real PD
    gains need real OP3 actuator research (it uses Dynamixel servos) in a
    later stage. The URDF's own effort/velocity limits (1000 / 100,
    identical across every joint) are placeholders too, not real specs.
"""

import argparse
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", type=str, default=None, help="Path to the flat URDF (default: robot/op3/robotis_op3.urdf)")
parser.add_argument("--usd-dir", type=str, default=None, help="Output directory for the USD (default: robot/op3/usd)")
parser.add_argument(
    "--no-instanceable",
    action="store_true",
    help="Generate a non-instanceable USD so downstream collision/contact property edits can be applied to "
    "collision prims. This is preferable for OP3 debugging even if it is less memory-efficient.",
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True})

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

project_root = Path(__file__).resolve().parent.parent
urdf_path = Path(args.urdf) if args.urdf else project_root / "robot" / "op3" / "robotis_op3.urdf"
usd_dir = Path(args.usd_dir) if args.usd_dir else project_root / "robot" / "op3" / "usd"

if not urdf_path.is_file():
    raise SystemExit(f"URDF not found: {urdf_path}. Run scripts/expand_urdf.sh first.")

placeholder_gains = UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
cfg = UrdfConverterCfg(
    asset_path=str(urdf_path),
    usd_dir=str(usd_dir),
    usd_file_name="robotis_op3.usd",
    fix_base=False,
    merge_fixed_joints=True,
    self_collision=True,
    collider_type="convex_hull",
    force_usd_conversion=True,
    make_instanceable=not args.no_instanceable,
    joint_drive=UrdfConverterCfg.JointDriveCfg(target_type="position", gains=placeholder_gains),
)
converter = UrdfConverter(cfg)

print(f"USD written to: {converter.usd_path}")

simulation_app.close()
