# Benchmarks

This directory defines replay benchmarks for OP3 checkpoints. These are not
ordinary unit tests: they run Isaac Lab physics and saved PPO checkpoints, so
they are slower and should be launched deliberately.

The benchmark goal is to stop choosing checkpoints from TensorBoard curves
alone. A checkpoint only improves the project if it beats the current baseline
under replay metrics.

## Benchmark Config

- `op3_checkpoint_benchmarks.json` -- named benchmark suites and thresholds.

Each case defines:

- stage/environment
- checkpoint path
- number of envs
- max replay steps
- pass/fail thresholds

## Run

From the repo root:

```bash
scripts/run_checkpoint_benchmarks.sh
```

Run only selected cases:

```bash
scripts/run_checkpoint_benchmarks.sh walk_best_smoke rebalanced_candidate
```

The runner writes JSON reports to `outputs/benchmarks/`.

Summarize completed JSON reports without launching Isaac Sim:

```bash
python3 scripts/summarize_checkpoint_benchmarks.py outputs/benchmarks
```

Write a markdown report:

```bash
python3 scripts/summarize_checkpoint_benchmarks.py outputs/benchmarks --out outputs/benchmarks/SUMMARY.md
```

## Interpreting Results

Primary metrics:

- `fallen_by_step40`
- `fallen_by_step50`
- `fallen_by_step100`
- `alive_at_end`
- `env0_max_abs_roll`
- `env0_max_abs_pitch`
- `max_abs_joint_vel`
- `max_abs_applied_torque`
- `torque_clip_fraction`

Important rule: do not promote a checkpoint into `checkpoints/` unless it beats
the current best benchmark under replay, not just training reward.

Use the summary table to rank candidates, but still inspect failed thresholds
and raw JSON before replacing a promoted checkpoint.
