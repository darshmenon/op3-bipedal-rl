#!/usr/bin/env bash
# Stage 2, step 1: expand the provided OP3 xacro into a flat URDF.
#
# The xacro files use ROS "$(find op3_description)/..." package-path
# substitution, which on ROS2 (this machine has Humble, not the ROS1 catkin
# build op3_description was written for) requires op3_description to be
# discoverable through ament_index. Rather than pull in a full catkin-in-ROS2
# build, this script registers op3_description with ament_index directly, by
# creating the marker file + share symlink a real `colcon build` would
# produce, and pointing AMENT_PREFIX_PATH at it.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OP3_DESCRIPTION_DIR="${PROJECT_ROOT}/ROBOTIS-OP3-Common/op3_description"
XACRO_FILE="${OP3_DESCRIPTION_DIR}/urdf/robotis_op3.urdf.xacro"
OUT_DIR="${PROJECT_ROOT}/robot/op3"
OUT_URDF="${OUT_DIR}/robotis_op3.urdf"
AMENT_SHIM="${PROJECT_ROOT}/robot/.ament_shim"

if ! command -v xacro >/dev/null 2>&1; then
  echo "ERROR: 'xacro' not found. Install ROS 2 (this was tested against ROS 2 Humble's xacro)." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
mkdir -p "${AMENT_SHIM}/share/ament_index/resource_index/packages"
touch "${AMENT_SHIM}/share/ament_index/resource_index/packages/op3_description"
ln -sfn "${OP3_DESCRIPTION_DIR}" "${AMENT_SHIM}/share/op3_description"

echo "Expanding ${XACRO_FILE} -> ${OUT_URDF}"
AMENT_PREFIX_PATH="${AMENT_SHIM}:${AMENT_PREFIX_PATH:-}" xacro "${XACRO_FILE}" -o "${OUT_URDF}"

# The expanded URDF still references meshes via ROS "package://op3_description/..."
# URIs, which Isaac Sim's URDF importer does not resolve. Rewrite them to a
# path relative to the output URDF's own directory, so the result stays
# portable across machines/clones (no absolute paths baked in) as long as the
# repo's directory layout (robot/op3/ next to ROBOTIS-OP3-Common/) is intact.
python3 - "${OUT_URDF}" "${OP3_DESCRIPTION_DIR}/meshes" "${OUT_DIR}" <<'PY'
import sys
from pathlib import Path

urdf_path, meshes_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
rel_meshes = f"{__import__('os').path.relpath(meshes_dir, out_dir)}/"

text = Path(urdf_path).read_text()
old = "package://op3_description/meshes/"
count = text.count(old)
text = text.replace(old, rel_meshes)
Path(urdf_path).write_text(text)
print(f"Rewrote {count} mesh reference(s): '{old}' -> '{rel_meshes}'")
PY

echo "Stage 2 step 1 (xacro expansion) completed: ${OUT_URDF}"
