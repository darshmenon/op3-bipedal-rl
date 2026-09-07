#!/usr/bin/env python3
"""Summarize fallback CPG sweep JSON files by survival metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("sweep_dir", type=Path)
args = parser.parse_args()

rows = []
for path in sorted(args.sweep_dir.glob("*.json")):
    data = json.loads(path.read_text())
    rows.append(
        {
            "trial": path.stem,
            "alive": data.get("alive_at_end", 0),
            "median": data.get("median_fall_step"),
            "mean": data.get("mean_fall_step"),
            "first": data.get("first_fall_step"),
            "fall50": data.get("fallen_by_step50"),
            "fall100": data.get("fallen_by_step100"),
            "freq": data.get("step_freq_hz"),
            "hip": data.get("hip_amp"),
            "knee": data.get("knee_amp"),
            "roll_kp": data.get("roll_kp"),
            "pitch_kp": data.get("pitch_kp"),
            "lat": data.get("lateral_shift_amp"),
            "sign": data.get("lateral_shift_sign"),
        }
    )

rows.sort(
    key=lambda r: (
        r["alive"],
        -1 if r["median"] is None else r["median"],
        -1 if r["mean"] is None else r["mean"],
        -999 if r["first"] is None else r["first"],
    ),
    reverse=True,
)

print(
    "trial alive median mean first fall50 fall100 freq hip knee roll_kp pitch_kp lat sign"
)
for r in rows:
    print(
        f"{r['trial']} {r['alive']} {r['median']} {r['mean']} {r['first']} "
        f"{r['fall50']} {r['fall100']} {r['freq']} {r['hip']} {r['knee']} "
        f"{r['roll_kp']} {r['pitch_kp']} {r['lat']} {r['sign']}"
    )
