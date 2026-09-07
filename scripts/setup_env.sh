#!/usr/bin/env bash
# Stage 1: Isaac Lab + Isaac Sim 5.1.0 environment setup for the OP3 bipedal task.
#
# Recipe reused from a previously-validated setup on this machine
# (isaacsim-quadruped-terrain/reference_repos/isaaclab-robust-go2), which
# trained Go2 quadruped locomotion successfully on the same GPU.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_VERSION="${ISAACLAB_VERSION:-v2.3.2}"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.1.0}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu128}"
ISAACLAB_REPO_URL="${ISAACLAB_REPO_URL:-https://github.com/isaac-sim/IsaacLab.git}"
VENV_DIR="${PROJECT_ROOT}/.venv"
ISAACLAB_PATH="${PROJECT_ROOT}/IsaacLab"

echo "Project root: ${PROJECT_ROOT}"
echo "Isaac Lab target: ${ISAACLAB_VERSION}"
echo "Isaac Sim target: ${ISAACSIM_VERSION}"
echo "PyTorch target: ${PYTORCH_VERSION} (cu128)"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating Python 3.11 venv at ${VENV_DIR}..."
  python3.11 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"ERROR: Python 3.11 is required; got {sys.version.split()[0]}")
print("Python:", sys.version.split()[0])
PY

python -m pip install --upgrade pip

echo "Installing isaacsim[all,extscache]==${ISAACSIM_VERSION} (large download)..."
python -m pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
  --extra-index-url https://pypi.nvidia.com

echo "Installing torch==${PYTORCH_VERSION} (cu128)..."
python -m pip install "torch==${PYTORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${CUDA_WHEEL_INDEX}"

if [[ ! -d "${ISAACLAB_PATH}/.git" ]]; then
  echo "Cloning IsaacLab ${ISAACLAB_VERSION} into ${ISAACLAB_PATH}..."
  git clone "${ISAACLAB_REPO_URL}" "${ISAACLAB_PATH}"
fi
cd "${ISAACLAB_PATH}"
git fetch --tags
git checkout "${ISAACLAB_VERSION}"

if [[ ! -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
  echo "ERROR: ${ISAACLAB_PATH}/isaaclab.sh is missing or not executable after checkout." >&2
  exit 2
fi

# Isaac Lab v2.3.2's UrdfConverter calls ImportConfig.set_merge_fixed_ignore_inertia(),
# which does not exist on the isaacsim.asset.importer.urdf extension shipped
# with isaacsim==5.1.0 (confirmed via `strings` on its compiled _urdf .so:
# only set_merge_fixed_joints exists, no _ignore_inertia variant) -- an
# Isaac-Lab-release vs. Isaac-Sim-release API mismatch, not a config bug.
# Drop the extra call; set_merge_fixed_joints (called right before it) already
# covers the core "merge fixed joints" setting.
URDF_CONVERTER="${ISAACLAB_PATH}/source/isaaclab/isaaclab/sim/converters/urdf_converter.py"
if grep -q "set_merge_fixed_ignore_inertia" "${URDF_CONVERTER}"; then
  echo "Patching urdf_converter.py: dropping set_merge_fixed_ignore_inertia() (not present in isaacsim==${ISAACSIM_VERSION})..."
  sed -i '/set_merge_fixed_ignore_inertia/d' "${URDF_CONVERTER}"
fi

# Isaac Lab's core package (source/isaaclab) hard-depends on flatdict==4.0.1,
# whose legacy setup.py needs pkg_resources. pip's isolated build env for it
# bootstraps a fresh setuptools that no longer ships pkg_resources, so the
# build fails there even though the venv's own setuptools (with pkg_resources)
# is fine. Pre-building flatdict with --no-build-isolation uses the venv's
# setuptools instead and sidesteps the issue. `wheel` is required for that
# build; installing it can bump `packaging` past what isaacsim-core/isaaclab_rl
# pin, so it's pinned back immediately after.
echo "Pre-installing flatdict (works around a pkg_resources build-isolation issue)..."
python -m pip install wheel
python -m pip install --no-build-isolation "flatdict==4.0.1"
python -m pip install "packaging==23.0"

echo "Installing rsl_rl (PPO implementation used by Isaac Lab)..."
./isaaclab.sh --install rsl_rl

echo "Stage 1 setup completed. Run scripts/verify_env.sh next."
