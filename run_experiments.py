"""Run or resume CMR sweeps; the default paper preset is the archived grid."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from analyze_results import read_trials, summarize
from artifact_io import ROOT, output_directory, write_csv, write_json
from sparse_lwe import is_prime, recover, sample_instance, theorem_predictor

SWEEP_ORDER = ("m", "n", "k", "q", "sigma")
FIELDS = ("sweep", "value", "point_index", "trial", "seed", "n", "m", "k", "q",
          "sigma", "B", "engine", "cache_dtype", "theorem_hypotheses", "theta", "xi",
          "attenuation", "theorem_lhs", "gamma", "full_recovery", "coordinate_accuracy",
          "first_error", "runtime_seconds")


def make_config(preset: str = "paper", *, trials: int | None = None,
                seed: int | None = None, B: float = 8.0, engine: str = "cached",
                cache_dtype: str = "float32") -> dict:
    if preset not in ("paper", "smoke"):
        raise ValueError("preset must be paper or smoke")
    config = json.loads((ROOT / "data/paper/experiment_config.json").read_text())
    if preset == "smoke":
        config["base_parameters"] = dict(n=12, m=40, k=2, q=31, sigma=1.5)
        config["sweeps"] = dict(m=[20, 40], n=[10, 14], k=[1, 2], q=[17, 31], sigma=[1., 2.])
    config.update(preset=preset, trials_per_point=trials if trials is not None else
                  (100 if preset == "paper" else 2),
                  master_seed=seed if seed is not None else 820260828,
                  B=B, engine=engine, cache_dtype=cache_dtype,
                  schema_version=1, implementation="CMR-1",
                  numpy_version=np.__version__, python_version=platform.python_version())
    if config["trials_per_point"] < 1 or config["master_seed"] < 0:
        raise ValueError("trials must be positive and seed nonnegative")
    if not math.isfinite(B) or B <= 4:
        raise ValueError("require finite B > 4")
    if engine not in ("cached", "reference") or cache_dtype not in ("float32", "float64"):
        raise ValueError("unsupported engine or cache dtype")
    return config


def points(config: dict) -> list[tuple[str, dict]]:
    result = []
    for sweep in SWEEP_ORDER:
        for value in config["sweeps"][sweep]:
            parameters = dict(config["base_parameters"])
            parameters[sweep] = value
            parameters = {name: float(v) if name == "sigma" else int(v)
                          for name, v in parameters.items()}
            result.append((sweep, parameters))
    return result


def trial_rng(master_seed: int, point_index: int, trial: int) -> tuple[np.random.Generator, int]:
    sequence = np.random.SeedSequence([master_seed, point_index, trial])
    # This recorded identifier is NOT a replacement seed for default_rng.
    identifier = int(sequence.generate_state(1, dtype=np.uint64)[0])
    return np.random.default_rng(sequence), identifier


def _validate_existing(rows: list[dict], config: dict, grid: list[tuple[str, dict]]) -> set:
    completed = set()
    for row in rows:
        point, trial = int(row["point_index"]), int(row["trial"])
        if not 0 <= point < len(grid) or not 0 <= trial < config["trials_per_point"]:
            raise ValueError("stored trial outside the requested grid")
        sweep, parameters = grid[point]
        if row["sweep"] != sweep or any(float(row[k]) != v for k, v in parameters.items()):
            raise ValueError("stored trial parameters do not match its batch")
        if (float(row["B"]) != config["B"] or row["engine"] != config["engine"]
                or row["cache_dtype"] != config["cache_dtype"]):
            raise ValueError("stored trial uses different recovery settings")
        _, identifier = trial_rng(config["master_seed"], point, trial)
        if int(row["seed"]) != identifier:
            raise ValueError("stored trial seed does not match the run")
        completed.add((sweep, point, trial))
    return completed


def run(config: dict, destination: Path) -> int:
    destination = output_directory(destination)
    config_path = destination / "experiment_config.json"
    raw_path = destination / "trial_results.csv"
    grid = points(config)
    rows = []
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise ValueError("run configuration differs; choose a new results subdirectory")
        if raw_path.exists():
            # A header-only file can result from interruption before the first trial.
            with raw_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != list(FIELDS):
                    raise ValueError("stored trial CSV has an incompatible schema")
                has_rows = next(reader, None) is not None
            rows = read_trials(raw_path) if has_rows else []
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError("nonempty output directory has no matching run configuration")
    completed = _validate_existing(rows, config, grid)
    destination.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        write_json(config_path, config)
    new_count = 0
    for point_index, (sweep, parameters) in enumerate(grid):
        metric = theorem_predictor(**parameters)
        hypotheses = int(parameters["sigma"] >= 1 and is_prime(parameters["q"])
                         and parameters["q"] >= max(11, 4 * parameters["sigma"]))
        for trial in range(config["trials_per_point"]):
            if (sweep, point_index, trial) in completed:
                continue
            rng, identifier = trial_rng(config["master_seed"], point_index, trial)
            instance = sample_instance(**parameters, rng=rng,
                                       tail_tolerance=config["discrete_gaussian_tail_tolerance"])
            start = time.perf_counter()
            recovered = recover(instance, B=config["B"], engine=config["engine"],
                                cache_dtype=config["cache_dtype"],
                                tail_tolerance=config["discrete_gaussian_tail_tolerance"])
            elapsed = time.perf_counter() - start
            correct = recovered.estimate == instance.secret
            errors = np.flatnonzero(~correct)
            row = dict(sweep=sweep, value=float(parameters[sweep]), point_index=point_index,
                       trial=trial, seed=identifier, **parameters, B=config["B"],
                       engine=config["engine"], cache_dtype=config["cache_dtype"],
                       theorem_hypotheses=hypotheses, **metric, xi=recovered.xi,
                       full_recovery=int(correct.all()), coordinate_accuracy=float(correct.mean()),
                       first_error=int(errors[0] + 1) if errors.size else 0,
                       runtime_seconds=elapsed)
            needs_header = not raw_path.exists()
            with raw_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
            rows.append(row)
            new_count += 1
        write_csv(destination / "summary_results.csv", summarize(rows))
    return new_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("paper", "smoke"), default="paper")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--B", type=float, default=8.)
    parser.add_argument("--engine", choices=("cached", "reference"), default="cached")
    parser.add_argument("--cache-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/run")
    args = parser.parse_args()
    config = make_config(args.preset, trials=args.trials, seed=args.seed, B=args.B,
                         engine=args.engine, cache_dtype=args.cache_dtype)
    count = run(config, args.output_dir)
    print(f"Completed {count} new trials in {args.output_dir}; no log file is written.")


if __name__ == "__main__":
    main()
