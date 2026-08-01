#!/usr/bin/env python3
"""Compose a high-coverage FLOAT trajectory with an independent safe FIX stream."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.evaluate_ppc_official_score import read_estimates
except ModuleNotFoundError:
    from evaluate_ppc_official_score import read_estimates  # type: ignore[no-redef]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_safe_tracker(path: Path) -> dict[float, dict[str, Any]]:
    output: dict[float, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            try:
                tow = round(float(row["tow"]), 3)
                position = tuple(float(row[axis]) for axis in "xyz")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid tracker row {line_number}") from exc
            if tow in output:
                raise ValueError(f"duplicate tracker TOW {tow}")
            fixed = row.get("shadow_fixed") == "1"
            if fixed and not all(math.isfinite(value) for value in position):
                raise ValueError(f"non-finite fixed tracker position at TOW {tow}")
            output[tow] = {"fixed": fixed, "position": position}
    return output


def compose(
    primary_path: Path,
    tracker_path: Path,
    *,
    minimum_ecef_norm_m: float = 6.0e6,
    maximum_ecef_norm_m: float = 7.0e6,
) -> list[dict[str, Any]]:
    primary, _ = read_estimates(primary_path)
    tracker = read_safe_tracker(tracker_path)
    rows: list[dict[str, Any]] = []
    last_valid_primary: np.ndarray | None = None
    last_tracker_at_valid_primary: np.ndarray | None = None
    for index, tow in enumerate(sorted(set(primary) | set(tracker))):
        tracker_row = tracker.get(tow)
        primary_position = primary.get(tow)
        primary_array = (
            np.asarray(primary_position, dtype=np.float64)
            if primary_position is not None
            else np.full(3, np.nan)
        )
        tracker_array = (
            np.asarray(tracker_row["position"], dtype=np.float64)
            if tracker_row is not None
            else np.full(3, np.nan)
        )
        primary_finite = bool(np.all(np.isfinite(primary_array)))
        primary_norm = float(np.linalg.norm(primary_array)) if primary_finite else math.nan
        primary_integrity_ok = (
            primary_finite
            and minimum_ecef_norm_m <= primary_norm <= maximum_ecef_norm_m
        )
        tracker_finite = bool(np.all(np.isfinite(tracker_array)))
        if primary_integrity_ok and tracker_finite:
            last_valid_primary = primary_array.copy()
            last_tracker_at_valid_primary = tracker_array.copy()
        if tracker_row is not None and tracker_row["fixed"]:
            position = tracker_row["position"]
            status = 4
            source = "safe_imu_pf_fgo_fixed"
        elif primary_integrity_ok:
            position = tuple(float(value) for value in primary_array)
            status = 3
            source = "primary_float"
        elif (
            tracker_finite
            and last_valid_primary is not None
            and last_tracker_at_valid_primary is not None
        ):
            bridged = tracker_array + last_valid_primary - last_tracker_at_valid_primary
            position = tuple(float(value) for value in bridged)
            status = 3
            source = "causal_tracker_integrity_bridge"
        elif tracker_finite:
            position = tuple(float(value) for value in tracker_array)
            status = 3
            source = "tracker_float_fallback"
        else:
            continue
        rows.append(
            {
                "epoch_index": index,
                "tow": tow,
                "status": status,
                "shadow_fixed": int(status == 4),
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "source": source,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--safe-tracker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = compose(args.primary, args.safe_tracker)
    if not rows:
        parser.error("composed trajectory is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_counts = {
        source: sum(row["source"] == source for row in rows)
        for source in (
            "safe_imu_pf_fgo_fixed",
            "primary_float",
            "causal_tracker_integrity_bridge",
            "tracker_float_fallback",
        )
    }
    summary = {
        "schema": "gnss_gpu_ppc_safe_trajectory_v1",
        "truth_contract": {"production_input_truth": False, "truth_usage": "none"},
        "fix_authority": "safe_imu_pf_fgo_tracker_only",
        "primary_status_inherited": False,
        "primary_integrity_gate": {
            "minimum_ecef_norm_m": 6.0e6,
            "maximum_ecef_norm_m": 7.0e6,
            "bridge": "causal_tracker_displacement_reanchored_at_last_valid_primary",
        },
        "epochs": len(rows),
        "fixed_epochs": source_counts["safe_imu_pf_fgo_fixed"],
        "source_counts": source_counts,
        "input_sha256": {
            "primary": _sha256(args.primary),
            "safe_tracker": _sha256(args.safe_tracker),
        },
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
