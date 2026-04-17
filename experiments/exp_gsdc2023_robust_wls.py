#!/usr/bin/env python3
"""Evaluate a Huber IRLS WLS prototype on GSDC 2023 train data.

This intentionally stays in Python: the goal is to test whether robustifying the
Google-provided WLS seed can beat the Android WLS baseline before adding CUDA.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for p in [
    str(_PROJECT_ROOT / "python"),
    str(_PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "python"),
    str(_PROJECT_ROOT / "third_party" / "gnssplusplus" / "python"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluate import compute_metrics, ecef_to_lla
from exp_gsdc2023_pf import (
    ELEV_THRESHOLD,
    SIGNAL_TYPES,
    TRAIN_DIR,
    discover_runs,
    elevation_weights,
    match_gt_to_epochs,
    parse_gnss_data,
    parse_ground_truth,
)
from exp_gsdc2023_submission import SAMPLE_SUB, TEST_DIR, epochs_to_unix_ms


DEFAULT_RESULTS_CSV = _SCRIPT_DIR / "results" / "gsdc2023_robust_wls_eval.csv"
DEFAULT_SUBMISSION_CSV = _SCRIPT_DIR / "results" / "gsdc2023_submission_robust_wls.csv"


@dataclass(frozen=True)
class RobustWLSConfig:
    huber_k_m: float = 20.0
    max_iter: int = 6
    max_shift_m: float = 30.0
    min_residual_rms_gain_m: float = 0.0
    prior_sigma_m: float | None = 0.1
    elevation_threshold_deg: float = ELEV_THRESHOLD
    min_weight: float = 1.0e-3
    tol_m: float = 1.0e-3


def _finite_ecef(pos: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(pos)) and np.linalg.norm(pos) > 1.0e6)


def pseudorange_residual_rms(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    rx_ecef: np.ndarray,
) -> float:
    if sat_ecef.shape[0] < 4 or not _finite_ecef(rx_ecef):
        return float("inf")
    ranges = np.linalg.norm(sat_ecef - rx_ecef, axis=1)
    if not np.all(np.isfinite(ranges)):
        return float("inf")
    clock_bias = float(np.median(pseudoranges - ranges))
    residuals = pseudoranges - ranges - clock_bias
    if not np.all(np.isfinite(residuals)):
        return float("inf")
    return float(np.sqrt(np.mean(residuals * residuals)))


def robust_wls_epoch(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    base_weights: np.ndarray,
    init_ecef: np.ndarray,
    config: RobustWLSConfig,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Solve one epoch with Huber IRLS initialized from Android WLS."""
    pos0 = np.asarray(init_ecef, dtype=np.float64).reshape(3)
    stats: dict[str, float | int | bool] = {
        "accepted": False,
        "n_iter": 0,
        "shift_m": float("inf"),
        "wls_residual_rms_m": float("inf"),
        "robust_residual_rms_m": float("inf"),
        "min_huber_weight": 1.0,
        "mean_huber_weight": 1.0,
        "prior_sigma_m": config.prior_sigma_m if config.prior_sigma_m is not None else -1.0,
    }
    if sat_ecef.shape[0] < 4 or not _finite_ecef(pos0):
        return pos0.copy(), stats

    pos = pos0.copy()
    ranges0 = np.linalg.norm(sat_ecef - pos, axis=1)
    clock_bias = float(np.median(pseudoranges - ranges0))
    base_w = np.clip(np.asarray(base_weights, dtype=np.float64), config.min_weight, None)
    stats["wls_residual_rms_m"] = pseudorange_residual_rms(sat_ecef, pseudoranges, pos0)

    for iteration in range(int(config.max_iter)):
        ranges = np.linalg.norm(sat_ecef - pos, axis=1)
        ranges = np.maximum(ranges, 1.0)
        residuals = pseudoranges - ranges - clock_bias
        abs_res = np.abs(residuals)
        huber = np.ones_like(abs_res)
        large = abs_res > float(config.huber_k_m)
        huber[large] = float(config.huber_k_m) / np.maximum(abs_res[large], 1.0e-12)
        weights = np.clip(base_w * huber, config.min_weight, None)

        los = (sat_ecef - pos) / ranges[:, None]
        H = np.column_stack([-los, np.ones(sat_ecef.shape[0])])
        sqrt_w = np.sqrt(weights)
        rhs = residuals * sqrt_w
        lhs = H * sqrt_w[:, None]
        if config.prior_sigma_m is not None and config.prior_sigma_m > 0.0:
            prior_residual = pos0 - pos
            prior_h = np.zeros((3, 4), dtype=np.float64)
            prior_h[:, :3] = np.eye(3)
            prior_weight = max(
                sat_ecef.shape[0] / float(config.prior_sigma_m * config.prior_sigma_m),
                config.min_weight,
            )
            prior_sqrt_w = math.sqrt(prior_weight)
            lhs = np.vstack([lhs, prior_h * prior_sqrt_w])
            rhs = np.concatenate([rhs, prior_residual * prior_sqrt_w])
        try:
            delta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return pos0.copy(), stats
        if not np.all(np.isfinite(delta)):
            return pos0.copy(), stats

        pos += delta[:3]
        clock_bias += float(delta[3])
        stats["n_iter"] = iteration + 1
        stats["min_huber_weight"] = float(np.min(huber))
        stats["mean_huber_weight"] = float(np.mean(huber))
        if float(np.linalg.norm(delta[:3])) < float(config.tol_m):
            break

    shift = float(np.linalg.norm(pos - pos0))
    robust_rms = pseudorange_residual_rms(sat_ecef, pseudoranges, pos)
    stats["shift_m"] = shift
    stats["robust_residual_rms_m"] = robust_rms

    residual_gain = float(stats["wls_residual_rms_m"]) - robust_rms
    if (
        _finite_ecef(pos)
        and shift <= float(config.max_shift_m)
        and residual_gain >= float(config.min_residual_rms_gain_m)
    ):
        stats["accepted"] = True
        return pos, stats
    return pos0.copy(), stats


def run_single_robust_wls(
    data_dir: Path,
    run_name: str,
    phone_name: str,
    config: RobustWLSConfig,
    *,
    verbose: bool = True,
) -> dict[str, object] | None:
    label = f"{run_name}/{phone_name}"
    gnss_path = data_dir / "device_gnss.csv"
    gt_path = data_dir / "ground_truth.csv"
    if not gnss_path.exists() or not gt_path.exists():
        if verbose:
            print(f"  [SKIP] {label}: missing data files")
        return None

    try:
        gt_ecef, gt_times_ms = parse_ground_truth(gt_path)
        epochs = parse_gnss_data(gnss_path)
    except Exception as exc:
        if verbose:
            print(f"  [ERROR] {label}: parse failed: {exc}")
        return None
    if len(epochs) < 10:
        if verbose:
            print(f"  [SKIP] {label}: too few epochs ({len(epochs)})")
        return None

    gt_indices = match_gt_to_epochs(epochs, gt_ecef, gt_times_ms)
    n_epochs = len(epochs)
    wls_positions = np.zeros((n_epochs, 3), dtype=np.float64)
    robust_positions = np.zeros((n_epochs, 3), dtype=np.float64)
    gt_matched = np.zeros((n_epochs, 3), dtype=np.float64)
    accepted = 0
    shifts: list[float] = []
    residual_gains: list[float] = []
    min_huber_weights: list[float] = []

    t0 = time.perf_counter()
    for i, ep in enumerate(epochs):
        wls = np.asarray(ep["wls_ecef"], dtype=np.float64)
        weights = elevation_weights(ep["elevations"], ep["cn0"], config.elevation_threshold_deg)
        robust, stats = robust_wls_epoch(
            ep["sat_ecef"],
            ep["pseudoranges"],
            weights,
            wls,
            config,
        )
        wls_positions[i] = wls
        robust_positions[i] = robust
        gt_matched[i] = gt_ecef[gt_indices[i]]
        if bool(stats["accepted"]):
            accepted += 1
            shifts.append(float(stats["shift_m"]))
            residual_gains.append(
                float(stats["wls_residual_rms_m"]) - float(stats["robust_residual_rms_m"])
            )
            min_huber_weights.append(float(stats["min_huber_weight"]))

    elapsed_s = time.perf_counter() - t0
    wls_metrics = compute_metrics(wls_positions, gt_matched)
    robust_metrics = compute_metrics(robust_positions, gt_matched)
    robust_win = robust_metrics["p50"] < wls_metrics["p50"]
    row: dict[str, object] = {
        "run": run_name,
        "phone": phone_name,
        "n_epochs": n_epochs,
        "n_sv_mean": float(np.mean([ep["n_sv"] for ep in epochs])),
        "huber_k_m": config.huber_k_m,
        "max_shift_m": config.max_shift_m,
        "min_residual_rms_gain_m": config.min_residual_rms_gain_m,
        "prior_sigma_m": config.prior_sigma_m if config.prior_sigma_m is not None else "",
        "accepted_epochs": accepted,
        "accepted_frac": accepted / float(n_epochs),
        "mean_shift_m": float(np.mean(shifts)) if shifts else 0.0,
        "mean_residual_rms_gain_m": float(np.mean(residual_gains)) if residual_gains else 0.0,
        "min_huber_weight_p50": float(np.median(min_huber_weights)) if min_huber_weights else 1.0,
        "wls_p50": wls_metrics["p50"],
        "wls_p95": wls_metrics["p95"],
        "wls_rms": wls_metrics["rms_2d"],
        "robust_p50": robust_metrics["p50"],
        "robust_p95": robust_metrics["p95"],
        "robust_rms": robust_metrics["rms_2d"],
        "delta_p50": robust_metrics["p50"] - wls_metrics["p50"],
        "delta_rms": robust_metrics["rms_2d"] - wls_metrics["rms_2d"],
        "elapsed_s": elapsed_s,
        "ms_per_epoch": elapsed_s * 1000.0 / float(n_epochs),
        "robust_win": robust_win,
    }
    if verbose:
        win = "WIN" if robust_win else "LOSE"
        print(
            f"  {label:65s} WLS P50={wls_metrics['p50']:6.2f}  "
            f"Robust P50={robust_metrics['p50']:6.2f}  "
            f"accepted={accepted / float(n_epochs):5.1%}  [{win}]"
        )
    return row


def discover_test_runs(test_dir: Path) -> list[tuple[str, str, Path]]:
    runs = []
    for run_dir in sorted(test_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for phone_dir in sorted(run_dir.iterdir()):
            if phone_dir.is_dir() and (phone_dir / "device_gnss.csv").exists():
                runs.append((run_dir.name, phone_dir.name, phone_dir))
    return runs


def run_test_robust_wls(data_dir: Path, config: RobustWLSConfig) -> tuple[np.ndarray, np.ndarray] | None:
    gnss_path = data_dir / "device_gnss.csv"
    if not gnss_path.exists():
        return None
    epochs = parse_gnss_data(gnss_path)
    if len(epochs) < 4:
        return None
    positions = np.zeros((len(epochs), 3), dtype=np.float64)
    for i, ep in enumerate(epochs):
        wls = np.asarray(ep["wls_ecef"], dtype=np.float64)
        weights = elevation_weights(ep["elevations"], ep["cn0"], config.elevation_threshold_deg)
        robust, _stats = robust_wls_epoch(
            ep["sat_ecef"],
            ep["pseudoranges"],
            weights,
            wls,
            config,
        )
        positions[i] = robust
    return positions, epochs_to_unix_ms(epochs)


def match_submission_times(
    sub_times_ms: np.ndarray,
    ecef: np.ndarray,
    times_ms: np.ndarray,
) -> np.ndarray:
    out = np.zeros((len(sub_times_ms), 2), dtype=np.float64)
    for i, sub_time in enumerate(sub_times_ms):
        idx = int(np.argmin(np.abs(times_ms - sub_time)))
        lat, lon, _alt = ecef_to_lla(*ecef[idx])
        out[i, 0] = math.degrees(lat)
        out[i, 1] = math.degrees(lon)
    return out


def generate_test_submission(
    config: RobustWLSConfig,
    *,
    test_dir: Path,
    sample_submission: Path,
    output_csv: Path,
) -> None:
    sub_df = pd.read_csv(sample_submission)
    out_rows: list[dict[str, object]] = []
    for trip_idx, trip_id in enumerate(sub_df["tripId"].unique()):
        run_name, phone_name = trip_id.split("/")
        data_dir = test_dir / run_name / phone_name
        result = run_test_robust_wls(data_dir, config)
        sub_trip = sub_df[sub_df["tripId"] == trip_id]
        if result is None:
            for _, row in sub_trip.iterrows():
                out_rows.append({
                    "tripId": trip_id,
                    "UnixTimeMillis": int(row["UnixTimeMillis"]),
                    "LatitudeDegrees": float(row["LatitudeDegrees"]),
                    "LongitudeDegrees": float(row["LongitudeDegrees"]),
                })
            print(f"  [{trip_idx + 1:2d}] {trip_id}: fallback sample rows")
            continue
        ecef, times_ms = result
        latlon = match_submission_times(
            sub_trip["UnixTimeMillis"].values.astype(np.float64),
            ecef,
            times_ms,
        )
        for j, (_, row) in enumerate(sub_trip.iterrows()):
            out_rows.append({
                "tripId": trip_id,
                "UnixTimeMillis": int(row["UnixTimeMillis"]),
                "LatitudeDegrees": float(latlon[j, 0]),
                "LongitudeDegrees": float(latlon[j, 1]),
            })
        print(f"  [{trip_idx + 1:2d}] {trip_id}: {len(sub_trip)} rows")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(output_csv, index=False)
    print(f"  Test submission saved: {output_csv}")


def _parse_config(args: argparse.Namespace) -> RobustWLSConfig:
    return RobustWLSConfig(
        huber_k_m=float(args.huber_k_m),
        max_iter=int(args.max_iter),
        max_shift_m=float(args.max_shift_m),
        min_residual_rms_gain_m=float(args.min_residual_rms_gain_m),
        prior_sigma_m=None if float(args.prior_sigma_m) <= 0.0 else float(args.prior_sigma_m),
        elevation_threshold_deg=float(args.elevation_threshold_deg),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="GSDC 2023 Robust WLS evaluation")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUB)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--submission-output", type=Path, default=DEFAULT_SUBMISSION_CSV)
    parser.add_argument("--single", type=str, default=None, help="Single RUN/PHONE")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit discovered train run/phone pairs")
    parser.add_argument("--huber-k-m", type=float, default=20.0)
    parser.add_argument("--max-iter", type=int, default=6)
    parser.add_argument("--max-shift-m", type=float, default=30.0)
    parser.add_argument("--min-residual-rms-gain-m", type=float, default=0.0)
    parser.add_argument(
        "--prior-sigma-m",
        type=float,
        default=0.1,
        help="Optional Gaussian position prior around Android WLS; negative disables it.",
    )
    parser.add_argument("--elevation-threshold-deg", type=float, default=ELEV_THRESHOLD)
    parser.add_argument("--win-threshold-p50", type=float, default=2.62)
    parser.add_argument("--no-test-submission-on-win", action="store_true")
    args = parser.parse_args()
    config = _parse_config(args)

    print("=" * 80)
    print("  GSDC 2023 Robust WLS (Huber IRLS)")
    print(f"  Signals: {', '.join(SIGNAL_TYPES)}")
    print(
        f"  huber_k={config.huber_k_m:g}m max_shift={config.max_shift_m:g}m "
        f"min_residual_gain={config.min_residual_rms_gain_m:g}m "
        f"prior_sigma={config.prior_sigma_m}"
    )
    print("=" * 80)

    if args.single:
        parts = args.single.split("/")
        if len(parts) != 2:
            raise SystemExit(f"--single expects RUN/PHONE, got: {args.single}")
        run_name, phone_name = parts
        row = run_single_robust_wls(args.train_dir / run_name / phone_name, run_name, phone_name, config)
        if row is None:
            return 1
        print(
            f"\n  WLS:    P50={row['wls_p50']:.2f}m P95={row['wls_p95']:.2f}m "
            f"RMS={row['wls_rms']:.2f}m"
        )
        print(
            f"  Robust: P50={row['robust_p50']:.2f}m P95={row['robust_p95']:.2f}m "
            f"RMS={row['robust_rms']:.2f}m"
        )
        return 0

    runs = discover_runs(args.train_dir)
    if args.max_runs > 0:
        runs = runs[: int(args.max_runs)]
    print(f"\n  Found {len(runs)} train run/phone combinations\n")
    if not runs:
        raise SystemExit(f"No GSDC train runs found under {args.train_dir}")

    rows = []
    for run_name, phone_name, data_dir in runs:
        row = run_single_robust_wls(data_dir, run_name, phone_name, config)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("No valid robust WLS results")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False, float_format="%.4f")

    mean_wls_p50 = float(df["wls_p50"].mean())
    mean_robust_p50 = float(df["robust_p50"].mean())
    mean_wls_rms = float(df["wls_rms"].mean())
    mean_robust_rms = float(df["robust_rms"].mean())
    median_wls_p50 = float(df["wls_p50"].median())
    median_robust_p50 = float(df["robust_p50"].median())
    n_win = int(df["robust_win"].sum())
    n_total = len(df)

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  Runs evaluated:      {n_total}")
    print(f"  Robust wins (P50):   {n_win}/{n_total} ({100.0 * n_win / n_total:.1f}%)")
    print(f"  Mean WLS:            P50={mean_wls_p50:.2f}m RMS={mean_wls_rms:.2f}m")
    print(f"  Mean Robust WLS:     P50={mean_robust_p50:.2f}m RMS={mean_robust_rms:.2f}m")
    print(f"  Median WLS P50:      {median_wls_p50:.2f}m")
    print(f"  Median Robust P50:   {median_robust_p50:.2f}m")
    print(f"  Mean delta:          P50={mean_robust_p50 - mean_wls_p50:+.2f}m RMS={mean_robust_rms - mean_wls_rms:+.2f}m")
    print(f"  Results saved:       {args.output}")
    print("=" * 80)

    if mean_robust_p50 < float(args.win_threshold_p50):
        print(
            f"\n  Robust WLS beats threshold {args.win_threshold_p50:.2f}m; "
            "generating test submission."
        )
        if not args.no_test_submission_on_win:
            generate_test_submission(
                config,
                test_dir=args.test_dir,
                sample_submission=args.sample_submission,
                output_csv=args.submission_output,
            )
    else:
        print(
            f"\n  Robust WLS does not beat threshold {args.win_threshold_p50:.2f}m; "
            "no test submission generated."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
