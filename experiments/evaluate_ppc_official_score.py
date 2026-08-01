#!/usr/bin/env python3
"""Score complete PPC trajectories with the official distance-weighted metric.

Estimates are aligned to reference epochs by GPS TOW.  Missing, duplicate, or
non-finite estimates never reduce the reference distance denominator.  Ground
truth is read only by this post-estimator evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from gnss_gpu.ppc_score import score_ppc2024


def _column(fieldnames: list[str], *names: str) -> str:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in names:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(f"missing required column; expected one of {names}")


def read_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        tow_key = _column(fields, "GPS TOW (s)", "tow")
        xyz_keys = (
            _column(fields, "ECEF X (m)", "x"),
            _column(fields, "ECEF Y (m)", "y"),
            _column(fields, "ECEF Z (m)", "z"),
        )
        rows = list(reader)
    tow = np.asarray([float(row[tow_key]) for row in rows], dtype=np.float64)
    xyz = np.asarray(
        [[float(row[key]) for key in xyz_keys] for row in rows], dtype=np.float64
    )
    if tow.size == 0 or not np.all(np.isfinite(tow)) or not np.all(np.isfinite(xyz)):
        raise ValueError("reference must contain finite TOW and ECEF positions")
    rounded = np.round(tow, 3)
    if np.unique(rounded).size != rounded.size:
        raise ValueError("reference contains duplicate TOW values")
    return rounded, xyz


def read_estimates(path: Path) -> tuple[dict[float, np.ndarray], dict[float, int]]:
    if path.suffix.lower() == ".pos":
        return read_pos_estimates(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        tow_key = _column(fields, "tow", "GPS TOW (s)")
        xyz_keys = (
            _column(fields, "x", "x_ecef_m", "ECEF X (m)"),
            _column(fields, "y", "y_ecef_m", "ECEF Y (m)"),
            _column(fields, "z", "z_ecef_m", "ECEF Z (m)"),
        )
        status_key = next((key for key in ("status", "shadow_fixed") if key in fields), None)
        estimates: dict[float, np.ndarray] = {}
        statuses: dict[float, int] = {}
        for line_number, row in enumerate(reader, start=2):
            key = round(float(row[tow_key]), 3)
            if key in estimates:
                raise ValueError(f"duplicate estimate TOW {key} on line {line_number}")
            try:
                estimates[key] = np.asarray([float(row[k]) for k in xyz_keys], dtype=np.float64)
            except (TypeError, ValueError):
                estimates[key] = np.full(3, np.nan, dtype=np.float64)
            if status_key is not None:
                try:
                    status_text = row[status_key].strip().upper()
                    if status_text in {"FIXED", "FLOAT"}:
                        statuses[key] = int(status_text == "FIXED")
                    else:
                        raw_status = int(status_text)
                        statuses[key] = (
                            int(raw_status == 4)
                            if status_key == "status"
                            else int(raw_status != 0)
                        )
                except (TypeError, ValueError):
                    statuses[key] = 0
    return estimates, statuses


def read_pos_estimates(path: Path) -> tuple[dict[float, np.ndarray], dict[float, int]]:
    """Read the LibGNSS++ whitespace POS format used by PPC runs."""
    estimates: dict[float, np.ndarray] = {}
    statuses: dict[float, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.startswith("%"):
                continue
            fields = line.split()
            if len(fields) < 9:
                raise ValueError(f"malformed POS row on line {line_number}")
            key = round(float(fields[1]), 3)
            if key in estimates:
                raise ValueError(f"duplicate estimate TOW {key} on line {line_number}")
            estimates[key] = np.asarray([float(value) for value in fields[2:5]], dtype=np.float64)
            # LibGNSS++ places Status at column 8.  Several legacy PF/FGO POS
            # exporters use a shorter RTKLIB-like layout where column 8 is a
            # floating sigma instead.  Such trajectories remain scoreable but
            # are conservatively treated as FLOAT rather than guessing a FIX.
            try:
                statuses[key] = int(int(fields[8]) == 4)
            except ValueError:
                statuses[key] = 0
    return estimates, statuses


def evaluate_route(estimate_path: Path, reference_path: Path, threshold_m: float = 0.5) -> dict[str, Any]:
    tow, reference = read_reference(reference_path)
    estimates, statuses = read_estimates(estimate_path)
    aligned = np.full_like(reference, np.nan)
    fixed = np.zeros(tow.size, dtype=bool)
    matched = np.zeros(tow.size, dtype=bool)
    for index, epoch_tow in enumerate(tow):
        key = float(epoch_tow)
        if key in estimates:
            aligned[index] = estimates[key]
            matched[index] = bool(np.all(np.isfinite(estimates[key])))
        fixed[index] = statuses.get(key, 0) != 0

    score = score_ppc2024(aligned, reference, threshold_m=threshold_m)
    false_fix = fixed & ~score.pass_mask
    severe_false_fix = fixed & np.isfinite(score.errors_3d) & (score.errors_3d > 1.0)
    finite_errors = score.errors_3d[np.isfinite(score.errors_3d)]
    return {
        "schema": "gnss_gpu_ppc_official_route_score_v1",
        "truth_contract": {"production_input_truth": False, "truth_usage": "post_estimator_scoring_only"},
        "estimate": str(estimate_path),
        "reference": str(reference_path),
        "threshold_m": float(threshold_m),
        "reference_epochs": int(tow.size),
        "matched_finite_epochs": int(np.sum(matched)),
        "missing_or_nonfinite_epochs": int(tow.size - np.sum(matched)),
        "ppc_score_pct": score.score_pct,
        "pass_distance_m": score.pass_distance_m,
        "total_distance_m": score.total_distance_m,
        "epoch_pass_pct": score.epoch_pass_pct,
        "fixed_epochs": int(np.sum(fixed)),
        "correct_fix_epochs": int(np.sum(fixed & score.pass_mask)),
        "false_fix_epochs": int(np.sum(false_fix)),
        "false_fix_above_1m_epochs": int(np.sum(severe_false_fix)),
        "error_rms_m": float(np.sqrt(np.mean(np.square(finite_errors)))) if finite_errors.size else None,
        "error_p95_m": float(np.percentile(finite_errors, 95.0)) if finite_errors.size else None,
        "error_max_m": float(np.max(finite_errors)) if finite_errors.size else None,
        "forward_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_route(args.estimate, args.reference, args.threshold_m)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
