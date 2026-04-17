#!/usr/bin/env python3
"""Evaluate NOAA CORS DD-pseudorange updates on GSDC 2023 train data.

The experiment keeps the first pass conservative: use raw L1/E1 GSDC
pseudorange rows, match them to the nearest available NOAA CORS daily RINEX
epoch through ``DDPseudorangeComputer``, and apply a bounded DD WLS update
around Android WLS.  Daily public CORS files are often 30 s, so coverage is
reported explicitly.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
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

from evaluate import compute_metrics
from exp_gsdc2023_pf import TRAIN_DIR, discover_runs, match_gt_to_epochs, parse_ground_truth
from gnss_gpu.gsdc_dgnss import (
    DDWLSConfig,
    build_dd_computer,
    cors_station_candidates,
    dd_pseudorange_position_update,
    fetch_first_available_cors_rinex,
    iter_gsdc_rover_epochs,
    run_date_from_gsdc_run,
)


DEFAULT_RESULTS_CSV = _SCRIPT_DIR / "results" / "gsdc2023_dgnss_eval.csv"
PF_100K_MEAN_P50_M = 2.83
WLS_MEAN_P50_M = 2.62


def _load_wls_trajectory(gnss_csv: Path) -> np.ndarray | None:
    try:
        df = pd.read_csv(
            gnss_csv,
            usecols=[
                "ArrivalTimeNanosSinceGpsEpoch",
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ],
            low_memory=False,
        )
    except (FileNotFoundError, ValueError):
        return None
    grouped = df.dropna().groupby("ArrivalTimeNanosSinceGpsEpoch", sort=True).first()
    if grouped.empty:
        return None
    return grouped[
        [
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]
    ].to_numpy(dtype=np.float64)


def run_single_dgnss(
    data_dir: Path,
    run_name: str,
    phone_name: str,
    *,
    cache_dir: Path,
    stations: list[str] | None,
    config: DDWLSConfig,
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
        trajectory = _load_wls_trajectory(gnss_path)
        candidates = stations or cors_station_candidates(
            run_name=run_name,
            trajectory_ecef=trajectory,
        )
        download = fetch_first_available_cors_rinex(
            candidates,
            run_date_from_gsdc_run(run_name),
            cache_dir=cache_dir,
        )
        dd_computer = build_dd_computer(download)
        epochs = list(iter_gsdc_rover_epochs(gnss_path))
    except Exception as exc:
        if verbose:
            print(f"  [ERROR] {label}: setup failed: {exc}")
        return None
    if len(epochs) < 10:
        if verbose:
            print(f"  [SKIP] {label}: too few rover epochs ({len(epochs)})")
        return None

    gt_indices = match_gt_to_epochs(epochs, gt_ecef, gt_times_ms)
    n_epochs = len(epochs)
    wls_positions = np.zeros((n_epochs, 3), dtype=np.float64)
    dgnss_positions = np.zeros((n_epochs, 3), dtype=np.float64)
    gt_matched = np.zeros((n_epochs, 3), dtype=np.float64)
    dd_epochs = 0
    accepted_epochs = 0
    dd_pair_counts: list[int] = []
    shifts: list[float] = []
    residual_gain: list[float] = []

    t0 = time.perf_counter()
    for i, ep in enumerate(epochs):
        wls = np.asarray(ep["wls_ecef"], dtype=np.float64)
        pos = wls
        dd = dd_computer.compute_dd(
            ep["tow"],
            ep["measurements"],
            rover_position_approx=wls,
            min_common_sats=4,
        )
        if dd is not None:
            dd_epochs += 1
            dd_pair_counts.append(int(dd.n_dd))
            pos_candidate, stats = dd_pseudorange_position_update(wls, dd, config)
            if bool(stats["accepted"]):
                accepted_epochs += 1
                pos = pos_candidate
                shifts.append(float(stats["shift_m"]))
                residual_gain.append(float(stats["initial_rms_m"]) - float(stats["final_rms_m"]))
        wls_positions[i] = wls
        dgnss_positions[i] = pos
        gt_matched[i] = gt_ecef[gt_indices[i]]

    elapsed_s = time.perf_counter() - t0
    wls_metrics = compute_metrics(wls_positions, gt_matched)
    dgnss_metrics = compute_metrics(dgnss_positions, gt_matched)
    row: dict[str, object] = {
        "run": run_name,
        "phone": phone_name,
        "station": download.station,
        "source_url": download.source_url,
        "n_epochs": n_epochs,
        "dd_epochs": dd_epochs,
        "dd_coverage_frac": dd_epochs / float(n_epochs),
        "accepted_epochs": accepted_epochs,
        "accepted_frac": accepted_epochs / float(n_epochs),
        "dd_pairs_mean": float(np.mean(dd_pair_counts)) if dd_pair_counts else 0.0,
        "mean_shift_m": float(np.mean(shifts)) if shifts else 0.0,
        "mean_dd_residual_gain_m": float(np.mean(residual_gain)) if residual_gain else 0.0,
        "wls_p50": wls_metrics["p50"],
        "wls_p95": wls_metrics["p95"],
        "wls_rms": wls_metrics["rms_2d"],
        "dgnss_p50": dgnss_metrics["p50"],
        "dgnss_p95": dgnss_metrics["p95"],
        "dgnss_rms": dgnss_metrics["rms_2d"],
        "delta_p50": dgnss_metrics["p50"] - wls_metrics["p50"],
        "delta_rms": dgnss_metrics["rms_2d"] - wls_metrics["rms_2d"],
        "elapsed_s": elapsed_s,
        "ms_per_epoch": elapsed_s * 1000.0 / float(n_epochs),
    }
    if verbose:
        print(
            f"  {label:65s} {download.station:4s} "
            f"cov={row['dd_coverage_frac']:5.1%} acc={row['accepted_frac']:5.1%} "
            f"WLS P50={wls_metrics['p50']:6.2f} DGNSS P50={dgnss_metrics['p50']:6.2f} "
            f"dP50={row['delta_p50']:+6.2f}"
        )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GSDC2023 NOAA CORS DGNSS evaluation")
    parser.add_argument("--single", type=str, default=None, help="Single run: RUN_NAME/PHONE_NAME")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of run/phone pairs")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/gsdc_cors"))
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument(
        "--station",
        action="append",
        default=None,
        help="Override station candidate list; repeat for ordered fallback.",
    )
    parser.add_argument("--dd-sigma-m", type=float, default=8.0)
    parser.add_argument("--prior-sigma-m", type=float, default=3.0)
    parser.add_argument("--max-shift-m", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DDWLSConfig(
        dd_sigma_m=float(args.dd_sigma_m),
        prior_sigma_m=float(args.prior_sigma_m),
        max_shift_m=float(args.max_shift_m),
    )
    print("=" * 80)
    print("  GSDC 2023 NOAA CORS DD-Pseudorange Evaluation")
    print(f"  cache={args.cache_dir}  dd_sigma={config.dd_sigma_m:.1f}m")
    print("=" * 80)

    if args.single:
        parts = args.single.split("/")
        if len(parts) != 2:
            print(f"Error: --single expects RUN/PHONE, got: {args.single}")
            sys.exit(1)
        run_name, phone_name = parts
        row = run_single_dgnss(
            args.train_dir / run_name / phone_name,
            run_name,
            phone_name,
            cache_dir=args.cache_dir,
            stations=args.station,
            config=config,
        )
        if row:
            print(
                f"\n  WLS:   P50={row['wls_p50']:.2f}m P95={row['wls_p95']:.2f}m "
                f"RMS={row['wls_rms']:.2f}m"
            )
            print(
                f"  DGNSS: P50={row['dgnss_p50']:.2f}m P95={row['dgnss_p95']:.2f}m "
                f"RMS={row['dgnss_rms']:.2f}m"
            )
        return

    runs = discover_runs(args.train_dir)
    if args.max_runs is not None:
        runs = runs[: max(args.max_runs, 0)]
    print(f"\n  Found {len(runs)} run/phone combinations\n")

    results = []
    for run_name, phone_name, data_dir in runs:
        row = run_single_dgnss(
            data_dir,
            run_name,
            phone_name,
            cache_dir=args.cache_dir,
            stations=args.station,
            config=config,
        )
        if row is not None:
            results.append(row)
    if not results:
        print("\nNo valid results!")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, float_format="%.4f")
    print(f"\n  Results saved: {args.output}")

    mean_wls_p50 = float(df["wls_p50"].mean())
    mean_dgnss_p50 = float(df["dgnss_p50"].mean())
    mean_wls_rms = float(df["wls_rms"].mean())
    mean_dgnss_rms = float(df["dgnss_rms"].mean())
    mean_cov = float(df["dd_coverage_frac"].mean())
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  Runs evaluated: {len(df)}")
    print(f"  Mean DD coverage: {mean_cov:.1%}")
    print(f"  Mean WLS:   P50={mean_wls_p50:.2f} m   RMS={mean_wls_rms:.2f} m")
    print(f"  Mean DGNSS: P50={mean_dgnss_p50:.2f} m   RMS={mean_dgnss_rms:.2f} m")
    print(f"  Delta:      P50={mean_dgnss_p50 - mean_wls_p50:+.2f} m   RMS={mean_dgnss_rms - mean_wls_rms:+.2f} m")
    if mean_dgnss_p50 < PF_100K_MEAN_P50_M:
        print(f"  RESULT: beats PF-100K mean P50 target ({PF_100K_MEAN_P50_M:.2f} m)")
        if mean_dgnss_p50 < WLS_MEAN_P50_M:
            print(f"  RESULT: also beats WLS mean P50 target ({WLS_MEAN_P50_M:.2f} m)")
    else:
        print(f"  RESULT: negative versus PF-100K mean P50 target ({PF_100K_MEAN_P50_M:.2f} m)")
    print("=" * 80)


if __name__ == "__main__":
    main()
