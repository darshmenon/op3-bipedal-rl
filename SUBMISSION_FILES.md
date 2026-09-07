# Submission Files Checklist

Use this as the final packing checklist before submitting the repository or
zip archive.

## Submit these

- `README.md` - main reviewer entry point and task checklist.
- `FINAL_REPORT.md` - short reviewer-friendly summary of the final result.
- `TECHNICAL_SUMMARY.md` - brief implementation/training summary replacing
  the long stage logs.
- `SUBMISSION_FILES.md` - this quick checklist.
- `RL_BIPEDAL_LOCOMOTION_CANDIDATE_TASK.md` - original task brief.
- `SETUP.md` - Ubuntu 22.04 setup instructions.
- `Dockerfile` - reproducible MuJoCo path.
- `CONCEPTS.md` - observation/action/reward and RL design notes.
- `op3_rl/` - environment configs, PPO config, robot config, MuJoCo env.
- `scripts/` - setup, train, resume, playback, diagnostics, demo recording.
- `tests/` - fast sanity tests.
- `benchmarks/` - replay benchmark config and benchmark README.
- `checkpoints/` - exported replay-validated policy checkpoints.
- `demo/op3_push_recovery_demo.mp4` - 112-second demo, forward walking +
  push recovery.
- `demo/op3_push_recovery_short_demo.mp4` - 66-second shorter take of the
  same.
- `demo/op3_backward_demo.mp4` - 35-second backward command behavior demo.
- `robot/` - tracked robot config used by the Isaac track. Generated USD
  files are reproducible and ignored.
- `ROBOTIS-OP3-Common/` - provided OP3 xacro/URDF/mesh source model.
- `third_party/mujoco_menagerie/robotis_op3/` - vendored MuJoCo OP3 model
  and license.

## Do not submit these

These are local/generated/heavy files and are already ignored by `.gitignore`
or should stay untracked:

- `.venv/`
- `IsaacLab/`
- `logs/`
- `outputs/`
- `local_research/`
- `.claude/`
- `.vscode/`
- `STAGE3_PPO_SMOKE_TEST.md`
- `STAGE4_PPO_TRAINING.md`
- root-level ad-hoc recordings such as
  `Screencast from 06-09-26 10_53_05 PM IST (online-video-cutter.com).mp4`
- generated USD/cache files under `robot/op3/usd/configuration/`

## Final checks

Run these before sending:

```bash
git status --short
python3 scripts/replay_validate_mujoco.py \
    --checkpoint checkpoints/op3_walk_mujoco_best.pt \
    --num_envs 8 --steps 1000 \
    --lin_vel_x_range -0.2 0.2 \
    --ang_vel_z_range -0.3 0.3 \
    --obs_normalization
```

Suggested staging command:

```bash
git add README.md FINAL_REPORT.md TECHNICAL_SUMMARY.md SUBMISSION_FILES.md \
    RL_BIPEDAL_LOCOMOTION_CANDIDATE_TASK.md SETUP.md Dockerfile \
    CONCEPTS.md \
    op3_rl scripts tests benchmarks checkpoints demo robot \
    ROBOTIS-OP3-Common third_party/mujoco_menagerie/robotis_op3
```

The safest final submission is the git repository itself. If sending a zip,
create it from tracked files plus the `demo/*.mp4` clips and the checkpoint
files under `checkpoints/`.
