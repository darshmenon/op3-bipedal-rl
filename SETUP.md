# Stage 1 — Isaac Lab Environment Setup

This is the first of several stages toward a PPO-trained ROBOTIS OP3 bipedal
walking policy (see `RL_BIPEDAL_LOCOMOTION_CANDIDATE_TASK.md`). This stage
only sets up and verifies the simulator — no OP3 robot work happens yet.

## Why this stack

The task doc recommends Isaac Sim 5.1.0 / Isaac Lab. This machine already has
a proven recipe for that exact combination, used successfully to train a
Unitree Go2 quadruped in a sibling project
(`isaacsim-quadruped-terrain/reference_repos/isaaclab-robust-go2`). Stage 1
reuses that recipe rather than inventing a new one:

| Component | Version |
|---|---|
| Isaac Sim | 5.1.0 (installed via pip, not the GUI installer) |
| Isaac Lab | v2.3.2 |
| Python | 3.11 (required by Isaac Lab v2.3.2; system default is 3.10) |
| PyTorch | 2.7.0 / cu128 |
| PPO implementation | `rsl_rl` (Isaac Lab's standard RL library) |

GPU on this machine: NVIDIA RTX 3500 Ada Laptop GPU, 12GB VRAM.

## Layout

- `.venv/` — project-local Python 3.11 virtualenv (gitignored).
- `IsaacLab/` — Isaac Lab checkout, pinned to `v2.3.2` (gitignored; cloned
  fresh by `scripts/setup_env.sh`, not committed — it's a multi-GB external
  dependency).
- `scripts/setup_env.sh` — reproduces the environment from scratch.
- `scripts/verify_env.sh` — runs the verification checks below.

## Reproducing

```bash
cd robotis_op3_bipedal_candidate_task
bash scripts/setup_env.sh
bash scripts/verify_env.sh
```

## Verification checks (Stage 1 "done" criteria)

1. `torch.cuda.is_available()` returns `True`.
2. `import isaacsim; import isaaclab; from pxr import Usd` succeeds, in that
   order, via `scripts/check_imports.py` (see gotcha below on why order
   matters) run inside `isaaclab.sh -p`.
3. `scripts/environments/list_envs.py` lists Isaac Lab's built-in tasks
   (confirms the task registry works).
4. `scripts/tutorials/00_sim/create_empty.py --headless` runs a headless
   empty-scene smoke test (confirms the simulator boots on this GPU). Treated
   as best-effort on this machine — see gotcha below.

## Run log

Ran on 2026-09-01. Total wall time roughly 20 minutes across the package
install (~11 min) and verification (~9 min including one hung smoke test).

**`scripts/setup_env.sh`** — completed successfully after two fixes (below).
Final disk usage: ~30GB free remaining out of the ~51GB available before the
install (so the install itself used ~20GB, comfortably under the ~25-30GB
historical estimate).

**`scripts/verify_env.sh`** results:

1. Torch/CUDA check:
   ```
   torch: 2.7.0+cu128
   cuda: 12.8
   cuda available: True
   ```
2. Isaac Sim / Isaac Lab / pxr import check (`scripts/check_imports.py`):
   passed — the script ran to completion (Kit extension stack loaded,
   `SimulationApp` started and shut down cleanly, `isaaclab.sh -p` returned
   exit 0). The script's own `print()` confirmation lines did not show up in
   the captured log; see the buffered-stdout gotcha below for why that's
   expected and not a failure signal.
3. `list_envs.py`: printed Isaac Lab's full built-in task table (100+
   registered tasks), including bipedal-relevant references for later stages
   such as `Isaac-Velocity-Flat-Cassie-v0` / `Isaac-Velocity-Rough-Cassie-v0`.
   No errors.
4. `create_empty.py --headless`: **inconclusive on this machine.** It loaded
   the Kit extension stack and reached the PhysX simulation-context warning,
   then made no further log progress for ~18 minutes while still
   legitimately consuming CPU (confirmed via `/proc/<pid>/stat` utime
   increasing across checks, not a deadlock) and ~2.2GB GPU memory. It was
   killed rather than waited out further. Given checks 1-3 already
   demonstrate a working install (Isaac Sim boots, extensions load, Isaac
   Lab's task registry and core package all import cleanly), this is treated
   as a known limitation, not a Stage 1 blocker. See the PCIe-link gotcha
   below for the likely cause.

### Gotchas hit (and fixed in the scripts)

- **`flatdict==4.0.1` build failure.** The core `isaaclab` package hard-depends
  on `flatdict==4.0.1`, whose legacy `setup.py` needs `pkg_resources`. pip
  builds it in an isolated build environment that bootstraps a fresh
  `setuptools` without `pkg_resources` (newer setuptools dropped it), so the
  build fails there even though the venv's own `setuptools==59.6.0` still has
  it. Fixed in `setup_env.sh` by pre-installing `flatdict` with
  `--no-build-isolation` (forcing it to use the venv's setuptools), after
  first installing `wheel` (needed for the build) and then re-pinning
  `packaging==23.0` (installing `wheel` had bumped `packaging` past what
  `isaacsim-core`/`isaaclab_rl` pin).
- **`pxr` import order.** `pxr` (USD bindings) is injected by Kit's
  extension/plugin system once `SimulationApp` is instantiated — it is not a
  normal top-level import beforehand. The reused go2-project check
  (`import isaacsim; import isaaclab; from pxr import Usd`) was written for a
  different context and raised `ModuleNotFoundError: No module named 'pxr'`
  here. Fixed by adding `scripts/check_imports.py`, which instantiates
  `SimulationApp({"headless": True})` first, then imports `isaaclab` and
  `pxr`, matching the ordering Isaac Sim's own error message describes.
- **Buffered stdout lost on `SimulationApp.close()`.** Prints issued between
  `SimulationApp` startup and `simulation_app.close()` didn't appear in the
  captured (non-tty, hence block-buffered) log, even though the run
  succeeded — Kit's shutdown path appears to terminate the process before
  Python's normal exit-time stdout flush happens. Not fixed (cosmetic only);
  noted here so it isn't mistaken for a silent failure in later stages. The
  exit code and absence of any traceback are the reliable signal, not the
  print output.
- **`create_empty.py --headless` slow/hung first run.** Boot logs show the
  GPU's PCIe link running at Gen1 x8 instead of its Gen4 x16 max (a laptop
  power-saving state: `PCIe link generation current (1) and maximum (4)...
  don't match`), which likely made first-run shader/asset-cache compilation
  far slower than normal. Not resolved in Stage 1; worth revisiting in a
  later stage (e.g. checking/forcing the PCIe power state, or trying a
  lower-fidelity rendering flag) before relying on GUI/RTX rendering for
  training or video capture.

## Known constraints going in

- Disk: only ~51GB free before this install. `isaacsim[all,extscache]` +
  IsaacLab + PyTorch have historically totaled ~25-30GB on this machine. This
  is being watched, not assumed safe.
- VRAM: 12GB. Isaac Lab's default training configs often use thousands of
  parallel environments; `num_envs` will likely need to be reduced for OP3
  training in a later stage. Not a Stage 1 concern (Stage 1 has no training).
