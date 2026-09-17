"""Output isolation: generated files belong only below results/."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"


def output_directory(path: Path) -> Path:
    path = path.resolve()
    allowed = RESULTS_ROOT.resolve()
    # Do not follow a results/ symlink out of the repository.
    if RESULTS_ROOT.is_symlink() or path == allowed or not path.is_relative_to(allowed):
        raise ValueError("choose an output subdirectory below results/ (not data/paper/)")
    return path


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty summary")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
