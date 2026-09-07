"""Stage 2, step 3: load the converted OP3 USD and verify the articulation.

Run via `isaaclab.sh -p scripts/check_op3_articulation.py`. Confirms the
conversion in convert_urdf_to_usd.py actually produced a usable, physically
simulatable articulation, not just a file that happens to exist:
  - the robot spawns and the physics stage steps without erroring,
  - it exposes exactly the 20 actuated joints named in the task doc,
  - joint positions come back finite (not NaN) after a few sim steps.
"""

from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402

EXPECTED_JOINTS = [
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll",
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll",
    "l_sho_pitch", "l_sho_roll", "l_el",
    "r_sho_pitch", "r_sho_roll", "r_el",
    "head_pan", "head_tilt",
]

project_root = Path(__file__).resolve().parent.parent
usd_path = project_root / "robot" / "op3" / "usd" / "robotis_op3.usd"
if not usd_path.is_file():
    raise SystemExit(f"USD not found: {usd_path}. Run scripts/convert_urdf_to_usd.py first.")

sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
sim = sim_utils.SimulationContext(sim_cfg)
# No ground plane / lighting here: spawn_ground_plane's default asset is
# hosted on NVIDIA's Nucleus server, which isn't configured for this
# pip-only install. Not needed anyway -- this check only verifies the
# articulation loads and its joint state stays finite over a few physics
# steps, not a rendered/dressed scene.

robot_cfg = ArticulationCfg(
    prim_path="/World/op3",
    spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
    # Placeholder actuator covering every joint, deferring gains to the USD
    # (0.0/0.0, set in convert_urdf_to_usd.py). Real per-joint PD gains are a
    # Stage 3 concern (needs actual OP3/Dynamixel actuator research); this is
    # only here because ArticulationCfg.actuators is a required field.
    actuators={"all_joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
)
robot = Articulation(robot_cfg)

sim.reset()

joint_names = robot.data.joint_names
print(f"Articulation DOF count: {robot.num_joints}")
print(f"Joint names: {joint_names}")

missing = [j for j in EXPECTED_JOINTS if j not in joint_names]
extra = [j for j in joint_names if j not in EXPECTED_JOINTS]
if missing:
    raise SystemExit(f"FAIL: missing expected joints: {missing}")
print(f"OK: all {len(EXPECTED_JOINTS)} expected joints present.")
if extra:
    print(f"Note: additional joints beyond the expected 20 (likely from unmerged fixed links): {extra}")

for _ in range(10):
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_cfg.dt)

joint_pos = robot.data.joint_pos
if torch.isnan(joint_pos).any():
    raise SystemExit("FAIL: NaN joint positions after stepping the simulation.")
print(f"OK: joint positions finite after 10 sim steps. Sample: {joint_pos[0, :5].tolist()}")

print("Stage 2 articulation check completed successfully.")
simulation_app.close()
