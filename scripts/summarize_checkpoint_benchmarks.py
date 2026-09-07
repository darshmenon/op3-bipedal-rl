"""Summarize checkpoint replay benchmark JSON files.

This is intentionally Isaac-free: it reads outputs produced by
``evaluate_checkpoint_performance.py`` and prints a compact markdown table
for comparing checkpoints after benchmark runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BenchmarkRow:
    case: str
    checkpoint: str
    stage: str
    passed: bool
    fallen_by_step40: int
    fallen_by_step50: int
    fallen_by_step100: int
    alive_at_end: int
    median_fall_step: float | None
    mean_fall_step: float | None
    torque_clip_fraction: float
    env0_max_abs_roll: float
    env0_max_abs_pitch: float
    failures: tuple[str, ...]


def _case_name(path: Path) -> str:
    return path.stem


def _short_checkpoint(path: str) -> str:
    parts = Path(path).parts
    if len(parts) <= 3:
        return path
    return str(Path(*parts[-3:]))


def load_rows(paths: Iterable[Path]) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for path in sorted(paths):
        data = json.loads(path.read_text())
        case = str(data.get("case") or _case_name(path))
        for summary in data.get("summaries", []):
            rows.append(
                BenchmarkRow(
                    case=case,
                    checkpoint=str(summary["checkpoint"]),
                    stage=str(summary["stage"]),
                    passed=bool(summary["passed"]),
                    fallen_by_step40=int(summary["fallen_by_step40"]),
                    fallen_by_step50=int(summary["fallen_by_step50"]),
                    fallen_by_step100=int(summary["fallen_by_step100"]),
                    alive_at_end=int(summary["alive_at_end"]),
                    median_fall_step=summary.get("median_fall_step"),
                    mean_fall_step=summary.get("mean_fall_step"),
                    torque_clip_fraction=float(summary.get("torque_clip_fraction", 0.0)),
                    env0_max_abs_roll=float(summary["env0_max_abs_roll"]),
                    env0_max_abs_pitch=float(summary["env0_max_abs_pitch"]),
                    failures=tuple(str(failure) for failure in summary.get("failures", [])),
                )
            )
    return rows


def rank_rows(rows: Iterable[BenchmarkRow]) -> list[BenchmarkRow]:
    """Rank rows by pass status, survival, later falls, and lower torque clipping."""
    return sorted(
        rows,
        key=lambda row: (
            not row.passed,
            row.fallen_by_step50,
            row.fallen_by_step100,
            -row.alive_at_end,
            -(row.median_fall_step if row.median_fall_step is not None else 10**9),
            row.torque_clip_fraction,
        ),
    )


def _fmt_number(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def render_markdown(rows: Iterable[BenchmarkRow]) -> str:
    ranked = rank_rows(rows)
    lines = [
        "| rank | case | pass | checkpoint | stage | fall40 | fall50 | fall100 | alive_end | median_fall | torque_clip | max_roll | max_pitch |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row.case,
                    "yes" if row.passed else "no",
                    _short_checkpoint(row.checkpoint),
                    row.stage,
                    str(row.fallen_by_step40),
                    str(row.fallen_by_step50),
                    str(row.fallen_by_step100),
                    str(row.alive_at_end),
                    _fmt_number(row.median_fall_step),
                    _fmt_number(row.torque_clip_fraction, 3),
                    _fmt_number(row.env0_max_abs_roll, 3),
                    _fmt_number(row.env0_max_abs_pitch, 3),
                ]
            )
            + " |"
        )

    failed = [row for row in ranked if row.failures]
    if failed:
        lines.extend(["", "Failures:"])
        for row in failed:
            lines.append(f"- `{row.case}`: {'; '.join(row.failures)}")
    return "\n".join(lines)


def _json_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.json"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="?",
        default="outputs/benchmarks",
        help="Benchmark JSON file or directory. Defaults to outputs/benchmarks.",
    )
    parser.add_argument("--out", default=None, help="Optional markdown output path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = _json_paths(Path(args.input))
    if not paths:
        raise SystemExit(f"No benchmark JSON files found under {args.input}")

    text = render_markdown(load_rows(paths))
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
