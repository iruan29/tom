#!/usr/bin/env python3
"""Validate the packaged datasets and the 4 x 4 x 3 experiment matrix."""

import csv
import json
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXPECTED_COUNTS = {
    "culture": 70,
    "education": 100,
    "pref": 120,
    "swe": 101,
}
SETTINGS = (
    "joint",
    "direct_solution",
    "cot_solution",
    "direct_belief_profile",
)
TURNS = (0, 5, 10)


def trajectory_path(reference):
    path = Path(reference)
    if path.parts[:1] == ("trajectories",):
        path = Path(*path.parts[1:])
    return DATA_DIR / "trajectories" / path


def validate_data():
    total = 0
    referenced = set()
    for name, expected_count in EXPECTED_COUNTS.items():
        path = DATA_DIR / "benchmarks" / f"{name}-benchmark.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        assert len(records) == expected_count, (
            f"{path}: expected {expected_count} records, got {len(records)}"
        )
        ids = [record["id"] for record in records]
        assert len(ids) == len(set(ids)), f"{path}: duplicate instance IDs"
        for record in records:
            resolved = trajectory_path(record["trajectory"])
            assert resolved.is_file(), f"Missing trajectory: {resolved}"
            json.loads(resolved.read_text(encoding="utf-8"))
            referenced.add(resolved.resolve())
        total += len(records)
    trajectory_files = set(
        path.resolve() for path in (DATA_DIR / "trajectories").rglob("*.json")
    )
    return total, len(trajectory_files), len(referenced)


def validate_matrix():
    manifest = ROOT / "scripts" / "gpt56_sol" / "manifest.tsv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = set(product(EXPECTED_COUNTS, SETTINGS, TURNS))
    actual = {
        (row["benchmark"], row["setting"], int(row["turn"])) for row in rows
    }
    assert actual == expected, "Manifest does not exactly cover the 4 x 4 x 3 matrix"
    assert len(rows) == len(expected), "Manifest contains duplicate rows"
    for row in rows:
        script = ROOT / row["script"]
        assert script.is_file(), f"Missing experiment script: {script}"
    return len(rows)


def main():
    instances, trajectories, referenced = validate_data()
    experiments = validate_matrix()
    print(
        f"OK: {instances} benchmark instances, {trajectories} trajectory files "
        f"({referenced} referenced), and {experiments} experiment scripts."
    )


if __name__ == "__main__":
    main()
