"""Stage 1 sanity check: Isaac Sim, Isaac Lab, and pxr (USD) all import cleanly.

`pxr` is injected by Kit's extension/plugin system once `SimulationApp` boots,
so it is only importable after that point -- importing it beforehand raises
`ModuleNotFoundError: No module named 'pxr'` even though the install is fine.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import isaaclab  # noqa: E402
from pxr import Usd  # noqa: E402, F401

print("Isaac Sim, Isaac Lab, and pxr imports OK")
print("isaaclab:", isaaclab.__file__)

simulation_app.close()
