#!/usr/bin/env python3
"""Audit the PPC IMU timing, axis, gravity, and lever-arm contract.

The reference trajectory is opened only by this offline audit.  Its output is
diagnostic evidence; the production estimator must consume only the fixed
dataset contract and the IMU/GNSS measurement streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


RUNS = tuple(
    (city, f"run{index}")
    for city in ("tokyo", "nagoya")
    for index in range(1, 4)
)

LEVER_ARM_FRD_M = {
    "tokyo": np.array([0.31, 0.0, -0.55], dtype=np.float64),
    "nagoya": np.array([0.593, -0.670, -1.216], dtype=np.float64),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_csv(path: Path, expected_columns: int) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != expected_columns or not np.all(np.isfinite(values)):
        raise ValueError(f"malformed numeric CSV: {path}")
    return values


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 20 or float(np.std(x)) < 1.0e-9 or float(np.std(y)) < 1.0e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _linear_fit(measured: np.ndarray, expected: np.ndarray) -> dict[str, float] | None:
    x = np.asarray(expected, dtype=np.float64).reshape(-1)
    y = np.asarray(measured, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 20 or float(np.std(x)) < 1.0e-9:
        return None
    design = np.column_stack((x, np.ones_like(x)))
    scale, bias = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (scale * x + bias)
    return {
        "scale": float(scale),
        "bias": float(bias),
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "samples": int(x.size),
    }


def _best_time_shift(
    imu_tow: np.ndarray,
    gyro_z_degps: np.ndarray,
    reference_tow: np.ndarray,
    yaw_rate_ccw_degps: np.ndarray,
    mask: np.ndarray,
    *,
    maximum_shift_s: float = 0.5,
    step_s: float = 0.01,
) -> dict[str, float | int | None]:
    shifts = np.arange(-maximum_shift_s, maximum_shift_s + 0.5 * step_s, step_s)
    scores: list[float] = []
    for shift in shifts:
        # Positive shift means the IMU timestamps are moved later.  A sample
        # observed at raw time t is therefore compared at reference t+shift.
        sampled = np.interp(
            reference_tow - shift,
            imu_tow,
            gyro_z_degps,
            left=np.nan,
            right=np.nan,
        )
        score = _correlation(sampled[mask], yaw_rate_ccw_degps[mask])
        scores.append(float("nan") if score is None else score)
    score_array = np.asarray(scores)
    if not np.any(np.isfinite(score_array)):
        return {
            "best_shift_s": None,
            "best_correlation": None,
            "zero_shift_correlation": None,
            "evaluated_samples": int(np.count_nonzero(mask)),
        }
    best_index = int(np.nanargmax(score_array))
    zero_index = int(np.argmin(np.abs(shifts)))
    return {
        "best_shift_s": float(np.round(shifts[best_index], 6)),
        "best_correlation": float(score_array[best_index]),
        "zero_shift_correlation": float(score_array[zero_index]),
        "correlation_gain_over_zero": float(score_array[best_index] - score_array[zero_index]),
        "evaluated_samples": int(np.count_nonzero(mask)),
    }


def audit_run(dataset_root: Path, city: str, run: str) -> dict[str, Any]:
    run_root = dataset_root / city / run
    imu_path = run_root / "imu.csv"
    reference_path = run_root / "reference.csv"
    imu = _load_csv(imu_path, 8)
    reference = _load_csv(reference_path, 14)

    imu_tow = imu[:, 0]
    imu_week = imu[:, 1].astype(np.int64)
    accel = imu[:, 2:5]
    gyro_degps = imu[:, 5:8]
    ref_tow = reference[:, 0]
    ref_week = reference[:, 1].astype(np.int64)
    heading_rad = np.unwrap(np.deg2rad(reference[:, 10]))
    velocity_en = reference[:, 11:13]
    speed = np.linalg.norm(velocity_en, axis=1)
    dt_ref = np.gradient(ref_tow)
    yaw_rate_ccw_radps = -np.gradient(heading_rad) / dt_ref
    yaw_rate_ccw_degps = np.rad2deg(yaw_rate_ccw_radps)
    longitudinal_expected = np.gradient(speed) / dt_ref
    lateral_left_expected = speed * yaw_rate_ccw_radps

    accel_at_ref = np.column_stack(
        [np.interp(ref_tow, imu_tow, accel[:, axis]) for axis in range(3)]
    )
    gyro_at_ref = np.column_stack(
        [np.interp(ref_tow, imu_tow, gyro_degps[:, axis]) for axis in range(3)]
    )

    moving = speed >= 2.0
    turning = moving & (np.abs(yaw_rate_ccw_degps) >= 0.5)
    accelerating = moving & (np.abs(longitudinal_expected) >= 0.1)
    lateral_dynamic = moving & (np.abs(lateral_left_expected) >= 0.1)
    stationary = speed <= 0.15

    axis_correlations = {
        "forward_accel_vs_speed_derivative": [
            _correlation(accel_at_ref[accelerating, axis], longitudinal_expected[accelerating])
            for axis in range(3)
        ],
        "left_accel_vs_centripetal_acceleration": [
            _correlation(accel_at_ref[lateral_dynamic, axis], lateral_left_expected[lateral_dynamic])
            for axis in range(3)
        ],
        "up_gyro_vs_ccw_yaw_rate": [
            _correlation(gyro_at_ref[turning, axis], yaw_rate_ccw_degps[turning])
            for axis in range(3)
        ],
    }

    stationary_accel = accel_at_ref[stationary]
    stationary_gyro = gyro_at_ref[stationary]
    gravity_mean = (
        np.mean(stationary_accel, axis=0) if stationary_accel.size else np.full(3, np.nan)
    )
    gyro_bias_mean = (
        np.mean(stationary_gyro, axis=0) if stationary_gyro.size else np.full(3, np.nan)
    )

    dt_imu = np.diff(imu_tow)
    positive_dt = dt_imu[dt_imu > 0.0]
    nominal_dt = float(np.median(positive_dt)) if positive_dt.size else math.nan
    time_shift = _best_time_shift(
        imu_tow,
        gyro_degps[:, 2],
        ref_tow,
        yaw_rate_ccw_degps,
        turning,
    )

    lever_frd = LEVER_ARM_FRD_M[city]
    lever_flu = lever_frd * np.array([1.0, -1.0, -1.0])
    return {
        "key": f"{city}_{run}",
        "city": city,
        "run": run,
        "truth_usage": "offline_axis_and_timing_audit_only",
        "production_estimator_reads_reference": False,
        "inputs": {
            "imu": str(imu_path),
            "imu_sha256": _sha256(imu_path),
            "reference": str(reference_path),
            "reference_sha256": _sha256(reference_path),
        },
        "timing": {
            "imu_samples": int(imu.shape[0]),
            "reference_epochs": int(reference.shape[0]),
            "gps_week_match": bool(np.array_equal(np.unique(imu_week), np.unique(ref_week))),
            "start_offset_s": float(imu_tow[0] - ref_tow[0]),
            "end_offset_s": float(imu_tow[-1] - ref_tow[-1]),
            "median_dt_s": nominal_dt,
            "rate_hz": float(1.0 / nominal_dt) if nominal_dt > 0.0 else None,
            "non_monotonic_count": int(np.count_nonzero(dt_imu <= 0.0)),
            "gap_count_above_1_5x": int(np.count_nonzero(dt_imu > 1.5 * nominal_dt)),
            "maximum_gap_s": float(np.max(positive_dt)) if positive_dt.size else None,
            "gyro_reference_shift_audit": time_shift,
        },
        "axis_contract": {
            "candidate_body_frame": "FLU",
            "axis_correlations": axis_correlations,
            "stationary_reference_epochs": int(np.count_nonzero(stationary)),
            "stationary_accel_mean_mps2": gravity_mean.tolist(),
            "stationary_accel_norm_mps2": float(np.linalg.norm(gravity_mean)),
            "stationary_gyro_mean_degps": gyro_bias_mean.tolist(),
            "forward_scale_fit": _linear_fit(
                accel_at_ref[accelerating, 0], longitudinal_expected[accelerating]
            ),
            "left_scale_fit": _linear_fit(
                accel_at_ref[lateral_dynamic, 1], lateral_left_expected[lateral_dynamic]
            ),
            "yaw_scale_fit": _linear_fit(
                gyro_at_ref[turning, 2], yaw_rate_ccw_degps[turning]
            ),
        },
        "lever_arm": {
            "dataset_readme_frd_m": lever_frd.tolist(),
            "fgo_body_flu_m": lever_flu.tolist(),
        },
    }


def _contract_passed(run: dict[str, Any]) -> bool:
    timing = run["timing"]
    correlations = run["axis_contract"]["axis_correlations"]
    forward = correlations["forward_accel_vs_speed_derivative"][0]
    left = correlations["left_accel_vs_centripetal_acceleration"][1]
    yaw = correlations["up_gyro_vs_ccw_yaw_rate"][2]
    shift = timing["gyro_reference_shift_audit"]["best_shift_s"]
    return bool(
        timing["gps_week_match"]
        and abs(timing["start_offset_s"]) <= 0.01
        and abs(timing["end_offset_s"]) <= 0.01
        and timing["non_monotonic_count"] == 0
        and timing["gap_count_above_1_5x"] == 0
        and forward is not None
        and forward >= 0.35
        and left is not None
        and left >= 0.35
        and yaw is not None
        and yaw >= 0.85
        and shift is not None
        and abs(shift) <= 0.05
    )


def build_audit(dataset_root: Path) -> dict[str, Any]:
    runs = [audit_run(dataset_root, city, run) for city, run in RUNS]
    for run in runs:
        run["contract_passed"] = _contract_passed(run)
    return {
        "schema": "gnss_gpu_ppc_imu_contract_audit_v1",
        "truth_usage": "reference_opened_only_by_offline_audit",
        "production_estimator_reads_reference": False,
        "dataset_root": str(dataset_root),
        "runs": runs,
        "all_runs_passed": all(run["contract_passed"] for run in runs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_audit(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_runs_passed": payload["all_runs_passed"]}))
    return 0 if payload["all_runs_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
