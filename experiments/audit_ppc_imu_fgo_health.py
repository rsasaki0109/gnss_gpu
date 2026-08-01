#!/usr/bin/env python3
"""Audit truth-free IMU-FGO health telemetry across PPC routes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(math.floor(location))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "maximum": max(values, default=None),
    }


def _finite_field(rows: list[dict[str, str]], name: str) -> list[float]:
    output: list[float] = []
    for row in rows:
        text = row.get(name, "")
        if not text:
            continue
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"non-finite {name}")
        output.append(value)
    return output


def _clean_nis_segments(rows: list[dict[str, str]]) -> list[list[float]]:
    """Split NIS telemetry at every unavailable or faulted epoch."""

    segments: list[list[float]] = []
    current: list[float] = []
    for row in rows:
        text = row.get("imu_fgo_factor_nis_per_dof", "")
        clean = (
            row.get("imu_fgo_available") == "1"
            and row.get("imu_fgo_fault_reason") == "ok"
            and row.get("imu_fgo_recovery_epochs") == "0"
            and bool(text)
        )
        if clean:
            value = float(text)
            if not math.isfinite(value):
                raise ValueError("non-finite imu_fgo_factor_nis_per_dof")
            current.append(value)
            continue
        if current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def audit_health(
    shadows: dict[str, Path], *, rolling_window: int = 25
) -> dict[str, object]:
    if not shadows:
        raise ValueError("at least one shadow CSV is required")
    if rolling_window < 2:
        raise ValueError("rolling window must be >= 2")
    route_results: dict[str, object] = {}
    all_nis: list[float] = []
    all_rolling: list[float] = []
    all_pose: list[float] = []
    clean_complete = True
    for name, path in shadows.items():
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        required = {
            "gps_week",
            "tow",
            "imu_fgo_available",
            "imu_fgo_fault_reason",
            "imu_fgo_recovery_epochs",
            "imu_fgo_factor_nis_per_dof",
            "imu_fgo_pose_correction_m",
            "imu_fgo_accel_bias_step_mps2",
            "imu_fgo_gyro_bias_step_radps",
        }
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"{name}: missing IMU health columns")
        times = [(int(row["gps_week"]), float(row["tow"])) for row in rows]
        if any(current <= previous for previous, current in zip(times, times[1:])):
            raise ValueError(f"{name}: non-increasing time axis")
        available = sum(row["imu_fgo_available"] == "1" for row in rows)
        reasons = Counter(row["imu_fgo_fault_reason"] for row in rows)
        recovery_values = [int(row["imu_fgo_recovery_epochs"]) for row in rows]
        if any(value < 0 for value in recovery_values):
            raise ValueError(f"{name}: negative IMU recovery count")
        recovery_rows = sum(value > 0 for value in recovery_values)
        clean_complete &= (
            available == len(rows)
            and set(reasons) == {"ok"}
            and recovery_rows == 0
        )
        nis_segments = _clean_nis_segments(rows)
        nis = [value for segment in nis_segments for value in segment]
        rolling = [
            statistics.median(segment[index - rolling_window + 1 : index + 1])
            for segment in nis_segments
            for index in range(rolling_window - 1, len(segment))
        ]
        healthy_rows = [
            row
            for row in rows
            if row["imu_fgo_available"] == "1"
            and row["imu_fgo_fault_reason"] == "ok"
            and row["imu_fgo_recovery_epochs"] == "0"
        ]
        pose = _finite_field(healthy_rows, "imu_fgo_pose_correction_m")
        accel_step = _finite_field(healthy_rows, "imu_fgo_accel_bias_step_mps2")
        gyro_step = _finite_field(healthy_rows, "imu_fgo_gyro_bias_step_radps")
        all_nis.extend(nis)
        all_rolling.extend(rolling)
        all_pose.extend(pose)
        route_results[name] = {
            "rows": len(rows),
            "available": available,
            "fault_reasons": dict(sorted(reasons.items())),
            "recovery_rows": recovery_rows,
            "recovery_epochs": sum(recovery_values),
            "clean_nis_segments": len(nis_segments),
            "nis_per_dof": _distribution(nis),
            "rolling_median_nis_per_dof": _distribution(rolling),
            "pose_correction_m": _distribution(pose),
            "accel_bias_step_mps2": _distribution(accel_step),
            "gyro_bias_step_radps": _distribution(gyro_step),
            "sha256": _sha256(path),
        }
    observed_max = max(all_rolling, default=0.0)
    provisional_threshold = math.ceil(observed_max * 1.2 * 2.0) / 2.0
    return {
        "schema": "gnss_gpu_ppc_imu_fgo_health_audit_v1",
        "truth_usage": "none",
        "rolling_window_epochs": rolling_window,
        "routes": route_results,
        "combined": {
            "nis_per_dof": _distribution(all_nis),
            "rolling_median_nis_per_dof": _distribution(all_rolling),
            "pose_correction_m": _distribution(all_pose),
            "clean_complete": clean_complete,
        },
        "provisional_monitor": {
            "rolling_window_epochs": rolling_window,
            "threshold_nis_per_dof": provisional_threshold,
            "basis": "1.2x clean observed maximum, rounded up to 0.5",
            "estimator_action": "telemetry_only",
            "promotion_ready": False,
            "reason": (
                "clean empirical NIS overlaps injected faults; no safe "
                "fail-closed threshold is pre-registered"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shadow",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="route label and shadow CSV; repeat for each route",
    )
    parser.add_argument("--rolling-window", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    shadows: dict[str, Path] = {}
    for item in args.shadow:
        if "=" not in item:
            parser.error("--shadow must be NAME=CSV")
        name, raw_path = item.split("=", 1)
        if not name or name in shadows:
            parser.error("shadow names must be non-empty and unique")
        shadows[name] = Path(raw_path)
    result = audit_health(shadows, rolling_window=args.rolling_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
