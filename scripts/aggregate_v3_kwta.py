#!/usr/bin/env python3
"""Aggregate mnist_v2 sweep results across seeds for the V2 vs V3 comparison.

Reads every mnist_v2_*.json in the given directory that contains a v3_acc field
(i.e. was produced by the v4.0.0+ binary that runs both variants). Computes
per-mode mean/std on accuracy + wall time and prints a side-by-side table plus
the per-seed paired deltas.

Usage:
    python3 scripts/aggregate_v3_kwta.py docs/benchmark_results/v4.0.0/
"""
import json
import statistics
import sys
from pathlib import Path


def load_runs(directory: Path):
    runs = []
    for path in sorted(directory.glob("mnist_v2_*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "v3_acc" not in data:
            continue
        runs.append((path.name, data))
    return runs


def fmt_stat(values, fmt="{:.2f}"):
    if not values:
        return "—"
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(statistics.mean(values))} ± {fmt.format(statistics.stdev(values))}"


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    directory = Path(sys.argv[1])
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(directory)
    if not runs:
        print(f"no v3-aware runs found in {directory}", file=sys.stderr)
        sys.exit(1)

    by_profile = {}
    for name, data in runs:
        profile = data.get("profile", "unknown")
        by_profile.setdefault(profile, []).append((name, data))

    for profile, group in by_profile.items():
        print(f"\n=== profile: {profile} ({len(group)} runs) ===")
        v2_acc, v3_acc = [], []
        v2_wall, v3_wall = [], []
        v2_morph, v3_morph = [], []
        v2_syn, v3_syn = [], []
        per_seed = []
        for name, data in group:
            v2a = data.get("v2_acc", 0) or 0
            v3a = data.get("v3_acc", 0) or 0
            v2w = data.get("v2_duration_s", 0) or 0
            v3w = data.get("v3_duration_s", 0) or 0
            v2_acc.append(v2a)
            v3_acc.append(v3a)
            v2_wall.append(v2w)
            v3_wall.append(v3w)
            v2_morph.append(data.get("v2_morphons", 0))
            v3_morph.append(data.get("v3_morphons", 0))
            v2_syn.append(data.get("v2_synapses", 0))
            v3_syn.append(data.get("v3_synapses", 0))
            per_seed.append({
                "seed": data.get("seed", "?"),
                "v2_acc": v2a,
                "v3_acc": v3a,
                "delta_acc": v3a - v2a,
                "v2_wall": v2w,
                "v3_wall": v3w,
                "speedup": (v2w / v3w) if v3w else float("nan"),
                "file": name,
            })

        print()
        print(f"  {'seed':<6}{'V2 acc':>9}{'V3 acc':>9}{'Δacc':>8}"
              f"{'V2 wall':>10}{'V3 wall':>10}{'speedup':>10}")
        print("  " + "-" * 62)
        for row in per_seed:
            print(f"  {row['seed']:<6}{row['v2_acc']:>8.1f}%{row['v3_acc']:>8.1f}%"
                  f"{row['delta_acc']:>+7.1f} {row['v2_wall']:>9}s{row['v3_wall']:>9}s"
                  f"{row['speedup']:>9.2f}x")

        print()
        print(f"  V2 accuracy:   {fmt_stat(v2_acc, '{:.1f}')}%")
        print(f"  V3 accuracy:   {fmt_stat(v3_acc, '{:.1f}')}%")
        if len(v2_acc) >= 2:
            deltas = [v3 - v2 for v2, v3 in zip(v2_acc, v3_acc)]
            print(f"  Delta paired:  {fmt_stat(deltas, '{:+.1f}')}pp")
        print(f"  V2 wall:       {fmt_stat(v2_wall, '{:.0f}')}s")
        print(f"  V3 wall:       {fmt_stat(v3_wall, '{:.0f}')}s")
        if all(w > 0 for w in v3_wall):
            speedups = [v2 / v3 for v2, v3 in zip(v2_wall, v3_wall)]
            print(f"  Speedup:       {fmt_stat(speedups, '{:.2f}')}x")
        print(f"  V2 morphons:   {fmt_stat(v2_morph, '{:.0f}')}")
        print(f"  V3 morphons:   {fmt_stat(v3_morph, '{:.0f}')}")
        print(f"  V2 synapses:   {fmt_stat(v2_syn, '{:.0f}')}")
        print(f"  V3 synapses:   {fmt_stat(v3_syn, '{:.0f}')}")


if __name__ == "__main__":
    main()
