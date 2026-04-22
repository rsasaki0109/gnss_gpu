#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate causal PPC TDCP velocity from rover carrier phase.

This is a realtime diagnostic, not a smoother: velocity at epoch i only uses
epochs i - 1 and i plus broadcast ephemeris.  The WLS-reset trajectory is kept
as a practical causal baseline; the truth-anchor trajectory is diagnostic only
and isolates TDCP velocity drift from absolute initialization error.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from evaluate import compute_metrics, ecef_errors_2d_3d
from exp_urbannav_baseline import run_wls
from gnss_gpu.io.ppc import PPCDatasetLoader
from gnss_gpu.ppc_score import ppc_score_dict
from gnss_gpu.tdcp_velocity import L1_WAVELENGTH, estimate_velocity_from_tdcp_with_metrics

RESULTS_DIR = _SCRIPT_DIR / "results"
_SYSTEM_ID_MAP = {"G": 0, "R": 1, "E": 2, "C": 3, "J": 4}


@dataclass(frozen=True)
class _TDCPMeasurement:
    system_id: int
    prn: int
    satellite_ecef: np.ndarray
    carrier_phase: float
    satellite_velocity: np.ndarray
    clock_drift: float
    weight: float
    snr: float
    elevation: float = float("nan")


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sat_prn(sat_id: str) -> int:
    try:
        return int(sat_id[1:])
    except (ValueError, IndexError):
        return 0


def _epoch_measurements(data: dict, epoch_idx: int) -> list[_TDCPMeasurement]:
    sat_ids = data["used_prns"][epoch_idx]
    sat_ecef = np.asarray(data["sat_ecef"][epoch_idx], dtype=np.float64)
    sat_velocity = np.asarray(data["sat_velocity"][epoch_idx], dtype=np.float64)
    carrier_phase = np.asarray(data["carrier_phase"][epoch_idx], dtype=np.float64)
    clock_drift = np.asarray(data["clock_drift"][epoch_idx], dtype=np.float64)
    weights = np.asarray(data["weights"][epoch_idx], dtype=np.float64)

    rows: list[_TDCPMeasurement] = []
    for i, sat_id in enumerate(sat_ids):
        if i >= len(carrier_phase) or i >= sat_velocity.shape[0]:
            continue
        if not (
            np.isfinite(carrier_phase[i])
            and np.all(np.isfinite(sat_velocity[i]))
            and np.isfinite(clock_drift[i])
        ):
            continue
        sys_char = sat_id[0]
        if sys_char not in _SYSTEM_ID_MAP:
            continue
        weight = float(weights[i]) if i < len(weights) and np.isfinite(weights[i]) else 1.0
        rows.append(
            _TDCPMeasurement(
                system_id=_SYSTEM_ID_MAP[sys_char],
                prn=_sat_prn(sat_id),
                satellite_ecef=sat_ecef[i],
                carrier_phase=float(carrier_phase[i]),
                satellite_velocity=sat_velocity[i],
                clock_drift=float(clock_drift[i]),
                weight=max(weight, 1.0),
                snr=max(weight, 1.0),
            )
        )
    return rows


def _velocity_truth(ground_truth: np.ndarray, times: np.ndarray) -> np.ndarray:
    truth = np.asarray(ground_truth, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    vel = np.full_like(truth, np.nan)
    for i in range(1, len(truth)):
        dt = float(times[i] - times[i - 1])
        if dt > 0.0:
            vel[i] = (truth[i] - truth[i - 1]) / dt
    return vel


def run_tdcp_eval(
    data: dict,
    min_sats: int,
    max_postfit_rms_m: float,
    max_cycle_jump: float,
    carrier_phase_sign: float,
    receiver_motion_sign: float,
    max_velocity_mps: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    wls_pos, wls_ms = run_wls(data)
    times = np.asarray(data["times"], dtype=np.float64)
    truth = np.asarray(data["ground_truth"], dtype=np.float64)
    truth_vel = _velocity_truth(truth, times)

    tdcp_reset = np.zeros_like(truth)
    tdcp_truth_anchor = np.zeros_like(truth)
    tdcp_reset[0] = wls_pos[0, :3]
    tdcp_truth_anchor[0] = truth[0]

    per_epoch: list[dict[str, object]] = []
    n_used = 0
    postfit_rms: list[float] = []
    velocity_errors: list[float] = []

    for i in range(1, len(times)):
        dt = float(times[i] - times[i - 1])
        prev_meas = _epoch_measurements(data, i - 1)
        cur_meas = _epoch_measurements(data, i)
        velocity, rms = estimate_velocity_from_tdcp_with_metrics(
            wls_pos[i - 1, :3],
            prev_meas,
            cur_meas,
            dt=dt,
            wavelength=L1_WAVELENGTH,
            carrier_phase_sign=carrier_phase_sign,
            receiver_motion_sign=receiver_motion_sign,
            min_sats=min_sats,
            max_cycle_jump=max_cycle_jump,
            max_postfit_rms_m=max_postfit_rms_m,
            max_velocity_mps=max_velocity_mps,
        )
        used = velocity is not None and dt > 0.0
        if used:
            n_used += 1
            postfit_rms.append(float(rms))
            tdcp_reset[i] = tdcp_reset[i - 1] + velocity * dt
            tdcp_truth_anchor[i] = tdcp_truth_anchor[i - 1] + velocity * dt
            if np.all(np.isfinite(truth_vel[i])):
                velocity_errors.append(float(np.linalg.norm(velocity - truth_vel[i])))
        else:
            tdcp_reset[i] = wls_pos[i, :3]
            tdcp_truth_anchor[i] = tdcp_truth_anchor[i - 1]

        per_epoch.append(
            {
                "epoch": i,
                "tow": float(times[i]),
                "dt": float(dt),
                "tdcp_used": bool(used),
                "n_prev_carrier": len(prev_meas),
                "n_cur_carrier": len(cur_meas),
                "tdcp_postfit_rms_m": float(rms) if np.isfinite(rms) else "",
                "tdcp_velocity_error_mps": (
                    float(np.linalg.norm(velocity - truth_vel[i]))
                    if used and np.all(np.isfinite(truth_vel[i]))
                    else ""
                ),
            }
        )

    wls_metrics = compute_metrics(wls_pos[:, :3], truth)
    tdcp_metrics = compute_metrics(tdcp_reset, truth)
    truth_anchor_metrics = compute_metrics(tdcp_truth_anchor, truth)
    wls_errors_2d, _ = ecef_errors_2d_3d(wls_pos[:, :3], truth)
    tdcp_errors_2d, _ = ecef_errors_2d_3d(tdcp_reset, truth)
    for row in per_epoch:
        idx = int(row["epoch"])
        row["wls_error_2d_m"] = float(wls_errors_2d[idx])
        row["tdcp_reset_error_2d_m"] = float(tdcp_errors_2d[idx])

    use_rate = 100.0 * n_used / max(len(times) - 1, 1)
    vel_rmse = float(np.sqrt(np.mean(np.square(velocity_errors)))) if velocity_errors else float("nan")
    median_postfit = float(np.median(postfit_rms)) if postfit_rms else float("nan")
    summary_rows = [
        {
            "method": "WLS",
            "n_epochs": len(times),
            "time_ms_per_epoch": float(wls_ms),
            **ppc_score_dict(wls_pos[:, :3], truth),
            "rms_2d": float(wls_metrics["rms_2d"]),
            "p50": float(wls_metrics["p50"]),
            "p95": float(wls_metrics["p95"]),
            "max_2d": float(wls_metrics["max_2d"]),
        },
        {
            "method": "TDCP reset-to-WLS fallback",
            "n_epochs": len(times),
            "tdcp_used_epochs": n_used,
            "tdcp_use_rate_pct": float(use_rate),
            "tdcp_velocity_rmse_mps": vel_rmse,
            "tdcp_postfit_rms_median": median_postfit,
            **ppc_score_dict(tdcp_reset, truth),
            "rms_2d": float(tdcp_metrics["rms_2d"]),
            "p50": float(tdcp_metrics["p50"]),
            "p95": float(tdcp_metrics["p95"]),
            "max_2d": float(tdcp_metrics["max_2d"]),
        },
        {
            "method": "TDCP truth-anchor diagnostic",
            "n_epochs": len(times),
            "tdcp_used_epochs": n_used,
            "tdcp_use_rate_pct": float(use_rate),
            "tdcp_velocity_rmse_mps": vel_rmse,
            "tdcp_postfit_rms_median": median_postfit,
            **ppc_score_dict(tdcp_truth_anchor, truth),
            "rms_2d": float(truth_anchor_metrics["rms_2d"]),
            "p50": float(truth_anchor_metrics["p50"]),
            "p95": float(truth_anchor_metrics["p95"]),
            "max_2d": float(truth_anchor_metrics["max_2d"]),
        },
    ]
    arrays = {
        "wls_pos": wls_pos[:, :3],
        "tdcp_reset": tdcp_reset,
        "tdcp_truth_anchor": tdcp_truth_anchor,
    }
    return summary_rows, per_epoch, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal PPC TDCP velocity baseline")
    parser.add_argument("--data-dir", type=Path, required=True, help="PPC run directory")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--systems", type=str, default="G,E,J")
    parser.add_argument("--min-sats", type=int, default=5)
    parser.add_argument("--max-postfit-rms-m", type=float, default=0.5)
    parser.add_argument("--max-cycle-jump", type=float, default=20000.0)
    parser.add_argument(
        "--carrier-phase-sign",
        type=float,
        default=1.0,
        help="Sign applied to RINEX L_cur-L_prev carrier phase before TDCP",
    )
    parser.add_argument(
        "--receiver-motion-sign",
        type=float,
        default=-1.0,
        help="Sign applied to LOS receiver displacement columns before TDCP WLS",
    )
    parser.add_argument("--max-velocity-mps", type=float, default=50.0)
    parser.add_argument("--results-prefix", type=str, default="ppc_tdcp_velocity")
    args = parser.parse_args()

    if not PPCDatasetLoader.is_run_directory(args.data_dir):
        raise FileNotFoundError(f"not a PPC run directory: {args.data_dir}")
    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  PPC TDCP Velocity Baseline")
    print("=" * 72)
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Systems   : {','.join(systems)}")
    print(f"  Start     : {args.start_epoch}")
    print(f"  Max epochs: {args.max_epochs}")

    loader = PPCDatasetLoader(args.data_dir)
    data = loader.load_experiment_data(
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        systems=systems,
        include_sat_velocity=True,
    )
    print(f"  Loaded    : {data['dataset_name']} ({data['n_epochs']} epochs)")

    summary_rows, per_epoch_rows, _arrays = run_tdcp_eval(
        data,
        min_sats=args.min_sats,
        max_postfit_rms_m=args.max_postfit_rms_m,
        max_cycle_jump=args.max_cycle_jump,
        carrier_phase_sign=args.carrier_phase_sign,
        receiver_motion_sign=args.receiver_motion_sign,
        max_velocity_mps=args.max_velocity_mps,
    )

    summary_path = RESULTS_DIR / f"{args.results_prefix}_summary.csv"
    epoch_path = RESULTS_DIR / f"{args.results_prefix}_epochs.csv"
    _write_rows(summary_rows, summary_path)
    _write_rows(per_epoch_rows, epoch_path)

    print()
    for row in summary_rows:
        print(
            f"  {row['method']:<28} "
            f"ppc={row['ppc_score_pct']:.2f}% "
            f"rms={row['rms_2d']:.2f}m p95={row['p95']:.2f}m"
        )
    print(f"  Saved: {summary_path}")
    print(f"  Saved: {epoch_path}")


if __name__ == "__main__":
    main()
