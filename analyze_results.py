"""Aggregate trials without merging batches; recompute Section 5 Gamma."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from artifact_io import ROOT, output_directory, write_csv
from sparse_lwe import theorem_predictor

SWEEPS = ("m", "n", "k", "q", "sigma")
PARAMETERS = ("n", "m", "k", "q", "sigma")


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= positive trials")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    return (0.0 if successes == 0 else max(0.0, center - radius),
            1.0 if successes == trials else min(1.0, center + radius))


def read_trials(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"sweep", "point_index", "trial", "full_recovery", "B", *PARAMETERS}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("trial CSV is missing required columns")
        rows = list(reader)
    if not rows:
        raise ValueError("trial CSV has no observations")
    seen = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("incomplete or malformed trial row")
        key = (row["sweep"], int(row["point_index"]), int(row["trial"]))
        if key in seen or key[0] not in SWEEPS or min(key[1:]) < 0:
            raise ValueError("duplicate or invalid trial key")
        if row["full_recovery"] not in ("0", "1"):
            raise ValueError("full_recovery must be 0 or 1")
        seen.add(key)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["sweep"], int(row["point_index"])), []).append(row)
    result = []
    for (sweep, point), items in sorted(groups.items(), key=lambda x: x[0][1]):
        first = items[0]
        parameters = {name: (float(first[name]) if name == "sigma" else int(first[name]))
                      for name in PARAMETERS}
        for item in items:
            if any(float(item[name]) != float(first[name]) for name in (*PARAMETERS, "B")):
                raise ValueError("inconsistent parameters within a batch")
        successes = sum(int(item["full_recovery"]) for item in items)
        low, high = wilson_interval(successes, len(items))
        metric = theorem_predictor(**parameters)
        result.append(dict(sweep=sweep, point_index=point, value=parameters[sweep],
                           **parameters, B=float(first["B"]), trials=len(items),
                           successes=successes, success_rate=successes / len(items),
                           wilson95_low=low, wilson95_high=high, **metric))
    return result


def analyze(raw: Path, destination: Path) -> list[dict]:
    destination = output_directory(destination)
    rows = summarize(read_trials(raw))
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "summary_results.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=ROOT / "data/paper/trial_results.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper")
    args = parser.parse_args()
    rows = analyze(args.trials, args.output_dir)
    print(f"Aggregated {sum(row['trials'] for row in rows)} trials in {len(rows)} batches.")


if __name__ == "__main__":
    main()
