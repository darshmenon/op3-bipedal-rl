# OP3 bipedal locomotion -- MuJoCo track container.
#
# Scope: this Dockerfile covers the MuJoCo training/eval/demo-recording
# pipeline (op3_rl/mujoco_env.py, scripts/train_ppo_mujoco.py,
# scripts/replay_validate_mujoco.py, scripts/record_mujoco_demo.py) --
# the pure-Python, CPU-friendly half of this project, and the source of
# the best-validated checkpoint (checkpoints/op3_walk_mujoco_best.pt) and
# the demo video (demo/op3_walk_demo.mp4).
#
# The Isaac Lab track (op3_rl/op3_env_cfg.py, scripts/train_ppo.py) is
# NOT containerized here: Isaac Sim/Isaac Lab is a multi-GB Omniverse
# install requiring GPU passthrough and NVIDIA's own container runtime,
# out of scope for a lightweight Dockerfile. Use SETUP.md for that track
# instead (native Ubuntu 22.04 setup, documented and reproducible).
#
# Build:
#   docker build -t op3-mujoco .
# Run (replay-validate the best checkpoint):
#   docker run --rm op3-mujoco \
#     python3 scripts/replay_validate_mujoco.py \
#       --checkpoint checkpoints/op3_walk_mujoco_best.pt --num_envs 8 --steps 1000 \
#       --lin_vel_x_range -0.2 0.2 --ang_vel_z_range -0.3 0.3 --obs_normalization
# Run (record the demo video -- OSMesa software rendering, ~1 fps, works in
# this plain image with no GPU; the actual demo/op3_walk_demo.mp4 was
# recorded via MUJOCO_GL=egl hardware rendering on the host instead, ~250x
# faster, which this base image does not have the NVIDIA/EGL userspace
# libraries for -- rebuild FROM an nvidia/cuda base image and add
# nvidia-container-toolkit + --gpus all for that path):
#   docker run --rm -v "$PWD/demo:/app/demo" op3-mujoco \
#     python3 scripts/record_mujoco_demo.py \
#       --checkpoint checkpoints/op3_walk_mujoco_best.pt --out demo/op3_walk_demo.mp4

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa

# Ubuntu 22.04's default python3 is 3.10 (not this project's dev-venv 3.11,
# which was pinned only because Isaac Lab v2.3.2 requires it -- the MuJoCo
# track has no such constraint, so this container deliberately uses the
# stock interpreter rather than pulling in a PPA). OSMesa for CPU-only
# software rendering, ffmpeg for scripts/record_mujoco_demo.py's encoding,
# git because rsl_rl imports GitPython at module load time regardless of
# whether it's actually used (fails hard with "Bad git executable" if the
# binary is missing, found by actually running the built image).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libosmesa6 libosmesa6-dev libgl1 libglx-mesa0 \
        ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin exact versions this project trained/validated against (see README.md
# "Simulator & versions") -- not "latest", to keep results reproducible.
# CPU-only torch build: every MuJoCo script in this project passes
# device="cpu" explicitly (training ran on CPU alongside the GPU-bound
# Isaac Lab track), so no CUDA runtime is needed here.
#
# rsl-rl-lib installed with --no-deps deliberately: installing it normally
# resolves its own unpinned `torch>=2.6.0`/`torchvision>=0.5.0` against
# PyPI's default (CUDA) index, silently upgrading past the CPU-only torch
# just installed and pulling ~2GB of NVIDIA cuDNN/NCCL/triton wheels never
# used by this project (found by actually building this image, not
# assumed -- see STAGE4_PPO_TRAINING.md). Installing its remaining runtime
# deps (torchvision, GitPython, onnx -- tensordict/numpy below already
# satisfy the rest) separately, also from the CPU index, avoids that.
RUN python3 -m pip install --no-cache-dir \
        torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --no-cache-dir \
        mujoco==3.12.0 \
        tensordict==0.14.0 \
        numpy==1.26.0 \
        imageio==2.37.0 \
        imageio-ffmpeg==0.6.0 \
        pillow \
        gitpython \
        onnx \
    && python3 -m pip install --no-cache-dir --no-deps rsl-rl-lib==3.1.2

# ROBOTIS's official OP3 MuJoCo model (Apache-2.0, from Google Deepmind's
# mujoco_menagerie), vendored in-repo at third_party/ -- see that
# directory's LICENSE. op3_rl/mujoco_env.py resolves this path relative to
# the repo root, so no absolute host path is baked into the image.
COPY third_party/ third_party/
COPY op3_rl/ op3_rl/
COPY scripts/ scripts/
COPY checkpoints/ checkpoints/

CMD ["python3", "scripts/replay_validate_mujoco.py", "--help"]
