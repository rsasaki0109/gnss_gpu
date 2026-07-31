#!/usr/bin/env python3
"""Audit RTK satellite-PAR surplus holdout monitor/active A/B runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_safe_union import _quantile, _tow  # noqa: E402
from experiments.analyze_wp175_library_fix_integrity import (  # noqa: E402
    read_library_pos,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truth(path: Path) -> dict[float, tuple[float, float, float]]:
    return {
        _tow(row["GPS TOW (s)"]): tuple(
            float(row[f"ECEF {axis} (m)"]) for axis in "XYZ"
        )
        for row in _read_csv(path)
    }


def _position_error(
    row: dict[str, str], truth: dict[float, tuple[float, float, float]]
) -> float | None:
    tow = _tow(float(row["tow"]))
    names = tuple(f"satellite_par_candidate_ecef_{axis}" for axis in "xyz")
    if tow not in truth or not all(row.get(name) for name in names):
        return None
    candidate = tuple(float(row[name]) for name in names)
    if not all(math.isfinite(value) for value in candidate):
        return None
    return math.dist(candidate, truth[tow])


def _route_metrics(
    positions: dict[float, dict[str, float | int]],
    truth: dict[float, tuple[float, float, float]],
) -> dict[str, object]:
    fixed_errors: list[float] = []
    for raw_tow, position in positions.items():
        tow = _tow(raw_tow)
        if int(position["status"]) != 4 or tow not in truth:
            continue
        estimate = tuple(float(position[f"ecef_{axis}"]) for axis in "xyz")
        fixed_errors.append(math.dist(estimate, truth[tow]))
    total = len(positions)
    return {
        "epochs": total,
        "fixed_epochs": len(fixed_errors),
        "fix_rate": len(fixed_errors) / total if total else 0.0,
        "false_fixed_epochs": sum(error >= 0.5 for error in fixed_errors),
        "over_1m_false_fixed_epochs": sum(error > 1.0 for error in fixed_errors),
        "fixed_error_p95_m": _quantile(fixed_errors, 0.95),
    }


def analyze(
    city: str,
    monitor_rows: list[dict[str, str]],
    monitor_positions: dict[float, dict[str, float | int]],
    active_rows: list[dict[str, str]],
    active_positions: dict[float, dict[str, float | int]],
    truth: dict[float, tuple[float, float, float]],
    acquisition_streak_epochs: int = 3,
) -> dict[str, object]:
    def is_strict(row: dict[str, str]) -> bool:
        return (
            float(row["satellite_par_surplus_max_distance_cycles"]) <= 0.10
            and int(row["satellite_par_subset_size"]) >= 8
            and float(row["satellite_par_ratio"]) >= 1.4
            and float(row["float_update_nis_per_observation"]) <= 3.0
            and float(row["float_update_prefit_residual_rms_m"]) <= 50.0
        )

    candidate_errors: list[tuple[dict[str, str], float]] = []
    for row in monitor_rows:
        if row.get("satellite_par_surplus_evaluated") != "1":
            continue
        error = _position_error(row, truth)
        if error is not None:
            candidate_errors.append((row, error))

    passing = [
        (row, error)
        for row, error in candidate_errors
        if row.get("satellite_par_surplus_passed") == "1"
    ]
    strict = [
        (row, error)
        for row, error in candidate_errors
        if is_strict(row)
    ]
    tow_to_block = {
        row["tow"]: min(4, index * 5 // max(1, len(monitor_rows)))
        for index, row in enumerate(monitor_rows)
    }
    block_metrics = []
    for block in range(5):
        selected = [
            error
            for row, error in strict
            if tow_to_block.get(row["tow"]) == block
        ]
        block_metrics.append(
            {
                "block": block,
                "strict_candidates": len(selected),
                "strict_correct_candidates": sum(
                    error < 0.5 for error in selected
                ),
                "strict_wrong_candidates": sum(
                    error >= 0.5 for error in selected
                ),
            }
        )
    runtime = [
        float(row["processing_runtime_ms"])
        for row in active_rows
        if row.get("processing_runtime_ms")
        and math.isfinite(float(row["processing_runtime_ms"]))
    ]
    monitor_route = _route_metrics(monitor_positions, truth)
    active_route = _route_metrics(active_positions, truth)
    monitor_fixed = int(monitor_route["fixed_epochs"])
    active_fixed = int(active_route["fixed_epochs"])

    return {
        "schema": "gnss_gpu_wp176_satellite_par_surplus_validation_v1",
        "city": city,
        "truth_usage": "post_selection_audit_only",
        "selection_policy": {
            "maximum_integer_distance_cycles": 0.10,
            "minimum_fixed_subset_pairs": 8,
            "minimum_ratio": 1.4,
            "maximum_nis_per_observation": 3.0,
            "maximum_prefit_residual_rms_m": 50.0,
            "acquisition_streak_epochs": acquisition_streak_epochs,
        },
        "monitor": monitor_route,
        "active": active_route,
        "fixed_epoch_delta": active_fixed - monitor_fixed,
        "fix_rate_delta": (
            float(active_route["fix_rate"]) - float(monitor_route["fix_rate"])
        ),
        "active_runtime_p95_ms": _quantile(runtime, 0.95),
        "active_runtime_p95_100ms_pass": bool(runtime)
        and _quantile(runtime, 0.95) <= 100.0,
        "surplus_evaluated_candidates": len(candidate_errors),
        "surplus_passing_candidates": len(passing),
        "surplus_passing_wrong_candidates": sum(
            error >= 0.5 for _, error in passing
        ),
        "strict_candidates": len(strict),
        "strict_correct_candidates": sum(error < 0.5 for _, error in strict),
        "strict_wrong_candidates": sum(error >= 0.5 for _, error in strict),
        "strict_over_1m_wrong_candidates": sum(
            error > 1.0 for _, error in strict
        ),
        "contiguous_time_blocks": block_metrics,
        "satellite_par_promoted_epochs": sum(
            row.get("quality_gate_satellite_par_promoted") == "1"
            for row in active_rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("tokyo", "nagoya"), required=True)
    parser.add_argument("--monitor-integrity", type=Path, required=True)
    parser.add_argument("--monitor-positions", type=Path, required=True)
    parser.add_argument("--active-integrity", type=Path, required=True)
    parser.add_argument("--active-positions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--acquisition-streak", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        args.city,
        _read_csv(args.monitor_integrity),
        read_library_pos(args.monitor_positions),
        _read_csv(args.active_integrity),
        read_library_pos(args.active_positions),
        _truth(args.reference),
        args.acquisition_streak,
    )
    payload["input_hashes"] = {
        name: _sha256(path)
        for name, path in {
            "monitor_integrity": args.monitor_integrity,
            "monitor_positions": args.monitor_positions,
            "active_integrity": args.active_integrity,
            "active_positions": args.active_positions,
            "reference": args.reference,
        }.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
