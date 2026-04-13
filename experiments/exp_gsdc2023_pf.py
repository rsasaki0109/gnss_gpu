#!/usr/bin/env python3
"""Experiment: GPU Particle Filter on Google Smartphone Decimeter Challenge 2023 data.

Evaluates ParticleFilterDevice across all GSDC2023 train runs and phones.
Uses GPS L1/L5 + Galileo E1/E5A (with ISRB correction).
Doppler velocity corrected by satellite velocity (gnssplusplus convention).

Usage
-----
  cd /media/sasaki/aiueo/ai_coding_ws/gnss_gpu
  PYTHONPATH=python python3 experiments/exp_gsdc2023_pf.py
  PYTHONPATH=python python3 experiments/exp_gsdc2023_pf.py --single 2020-06-25-00-34-us-ca-mtv-sb-101/pixel4
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
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

from evaluate import compute_metrics, lla_to_ecef, print_comparison_table

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_DIR = Path("/tmp/gsdc_data/gsdc2023/sdc2023/train")
N_PARTICLES = 100_000
SIGMA_POS = 10.0       # position random-walk sigma [m] — large enough to maintain particle diversity
SIGMA_CB = 300.0
SIGMA_PR = 15.0        # pseudorange sigma [m] — tuned for smartphone noise (~15-20m std)
POS_UPDATE_SIGMA = 3.0 # WLS soft-constraint sigma [m]
ELEV_THRESHOLD = 15.0

# Signal types to use (skip GLONASS — FDMA requires per-satellite frequency handling)
SIGNAL_TYPES = ["GPS_L1_CA", "GPS_L5_Q", "GAL_E1_C_P", "GAL_E5A_Q"]


def discover_runs(train_dir: Path) -> list[tuple[str, str, Path]]:
    """Find all (run_name, phone_name, data_dir) triples."""
    runs = []
    for run_dir in sorted(train_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for phone_dir in sorted(run_dir.iterdir()):
            if not phone_dir.is_dir():
                continue
            gnss_file = phone_dir / "device_gnss.csv"
            gt_file = phone_dir / "ground_truth.csv"
            if gnss_file.exists() and gt_file.exists():
                runs.append((run_dir.name, phone_dir.name, phone_dir))
    return runs


def parse_ground_truth(gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse ground_truth.csv -> (ecef array (N,3), unix_time_ms array (N,))."""
    df = pd.read_csv(gt_path)
    n = len(df)
    ecef = np.zeros((n, 3))
    for i in range(n):
        lat_rad = math.radians(df["LatitudeDegrees"].iloc[i])
        lon_rad = math.radians(df["LongitudeDegrees"].iloc[i])
        alt = df["AltitudeMeters"].iloc[i]
        ecef[i] = lla_to_ecef(lat_rad, lon_rad, alt)
    times_ms = df["UnixTimeMillis"].values.astype(np.float64)
    return ecef, times_ms


def parse_gnss_data(gnss_path: Path):
    """Parse device_gnss.csv, group by epoch, return structured data.

    Uses GPS L1/L5 + Galileo E1/E5A.  Applies ISRB correction to align
    all signals to a common receiver clock reference.
    """
    df = pd.read_csv(gnss_path, low_memory=False)

    # Filter to supported signal types (skip GLONASS FDMA)
    df = df[df["SignalType"].isin(SIGNAL_TYPES)].copy()

    # Drop rows with NaN in critical columns
    critical_cols = [
        "RawPseudorangeMeters", "SvClockBiasMeters",
        "SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters",
    ]
    df = df.dropna(subset=critical_cols)

    # Group by arrival time (epoch)
    grouped = df.groupby("ArrivalTimeNanosSinceGpsEpoch")

    epochs = []
    for arrival_ns, group in sorted(grouped):
        n_sv = len(group)
        if n_sv < 4:
            continue

        sat_ecef = np.column_stack([
            group["SvPositionXEcefMeters"].values,
            group["SvPositionYEcefMeters"].values,
            group["SvPositionZEcefMeters"].values,
        ]).astype(np.float64)

        sv_vel = np.column_stack([
            group["SvVelocityXEcefMetersPerSecond"].values,
            group["SvVelocityYEcefMetersPerSecond"].values,
            group["SvVelocityZEcefMetersPerSecond"].values,
        ]).astype(np.float64)

        raw_pr = group["RawPseudorangeMeters"].values.astype(np.float64)
        sv_clock_bias = group["SvClockBiasMeters"].values.astype(np.float64)
        iono_delay = group["IonosphericDelayMeters"].values.astype(np.float64)
        tropo_delay = group["TroposphericDelayMeters"].values.astype(np.float64)
        isrb = group["IsrbMeters"].values.astype(np.float64)

        # Fill NaN delays with 0
        iono_delay = np.nan_to_num(iono_delay, nan=0.0)
        tropo_delay = np.nan_to_num(tropo_delay, nan=0.0)
        isrb = np.nan_to_num(isrb, nan=0.0)

        # Corrected pseudorange: align all signals to common clock reference
        # ISRB correction ensures L5 / Galileo pseudoranges are compatible with L1
        corrected_pr = raw_pr + sv_clock_bias - iono_delay - tropo_delay - isrb

        elevations = group["SvElevationDegrees"].values.astype(np.float64)
        pr_rate = group["PseudorangeRateMetersPerSecond"].values.astype(np.float64)
        cn0 = group["Cn0DbHz"].values.astype(np.float64)

        wls_ecef = np.array([
            group["WlsPositionXEcefMeters"].iloc[0],
            group["WlsPositionYEcefMeters"].iloc[0],
            group["WlsPositionZEcefMeters"].iloc[0],
        ], dtype=np.float64)

        # Skip epochs with invalid WLS
        if np.any(np.isnan(wls_ecef)) or np.linalg.norm(wls_ecef) < 1e6:
            continue

        epochs.append({
            "arrival_ns": arrival_ns,
            "sat_ecef": sat_ecef,
            "pseudoranges": corrected_pr,
            "elevations": elevations,
            "sv_vel": sv_vel,
            "pr_rate": pr_rate,
            "cn0": cn0,
            "wls_ecef": wls_ecef,
            "n_sv": n_sv,
        })

    return epochs


def compute_doppler_velocity(epoch: dict, rx_ecef: np.ndarray) -> np.ndarray:
    """Compute receiver velocity from Doppler (pseudorange rate) + SV velocity.

    gnssplusplus convention: range_rate = dot(sv_vel, los) - pr_rate
    Then solve for rx_vel via least squares (4-state: vx, vy, vz, clock_drift).
    """
    sat_ecef = epoch["sat_ecef"]
    sv_vel = epoch["sv_vel"]
    pr_rate = epoch["pr_rate"]
    n_sv = len(pr_rate)

    if n_sv < 4:
        return np.zeros(3)

    # Line-of-sight unit vectors (receiver -> satellite)
    los = sat_ecef - rx_ecef
    ranges = np.linalg.norm(los, axis=1, keepdims=True)
    ranges = np.maximum(ranges, 1.0)
    los_unit = los / ranges

    # Design matrix: H = [-los_unit, 1]
    H = np.column_stack([-los_unit, np.ones(n_sv)])

    # Satellite radial velocity: dot(sv_vel, los_unit)
    sv_radial = np.sum(sv_vel * los_unit, axis=1)

    # Observation: pr_rate - sv_radial = -dot(rx_vel, los_unit) + clock_drift
    obs = pr_rate - sv_radial

    try:
        x, _, _, _ = np.linalg.lstsq(H, obs, rcond=None)
        return x[:3]
    except np.linalg.LinAlgError:
        return np.zeros(3)


def elevation_weights(elevations: np.ndarray, cn0: np.ndarray | None = None,
                      threshold: float = 20.0) -> np.ndarray:
    """Compute elevation + Cn0 based weights."""
    weights = np.ones(len(elevations))
    for i, el in enumerate(elevations):
        if el < threshold:
            weights[i] = max(0.1, math.sin(math.radians(max(el, 1.0))) ** 2)
    if cn0 is not None:
        cn0_w = np.clip((cn0 - 20.0) / 25.0, 0.3, 1.0)
        weights *= cn0_w
    return weights


def match_gt_to_epochs(epochs: list[dict], gt_ecef: np.ndarray,
                       gt_times_ms: np.ndarray) -> list[int]:
    """For each epoch, find closest GT by time."""
    GPS_UNIX_OFFSET_NS = 315964800 * 1_000_000_000
    LEAP_SECONDS_NS = 18 * 1_000_000_000

    gt_indices = []
    for ep in epochs:
        unix_ns = ep["arrival_ns"] + GPS_UNIX_OFFSET_NS - LEAP_SECONDS_NS
        unix_ms = unix_ns / 1_000_000
        diffs = np.abs(gt_times_ms - unix_ms)
        idx = int(np.argmin(diffs))
        gt_indices.append(idx)
    return gt_indices


def run_single(data_dir: Path, run_name: str, phone_name: str,
               verbose: bool = True) -> dict | None:
    """Run PF evaluation on a single run/phone. Returns metrics dict or None."""
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
    except Exception as e:
        if verbose:
            print(f"  [ERROR] {label}: parse failed: {e}")
        return None

    if len(epochs) < 10:
        if verbose:
            print(f"  [SKIP] {label}: too few epochs ({len(epochs)})")
        return None

    n_epochs = len(epochs)
    gt_indices = match_gt_to_epochs(epochs, gt_ecef, gt_times_ms)

    # WLS baseline
    wls_positions = np.zeros((n_epochs, 3))
    gt_matched = np.zeros((n_epochs, 3))
    for i, ep in enumerate(epochs):
        wls_positions[i] = ep["wls_ecef"]
        gt_matched[i] = gt_ecef[gt_indices[i]]

    wls_metrics = compute_metrics(wls_positions, gt_matched)

    # Run PF
    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    pf = ParticleFilterDevice(
        n_particles=N_PARTICLES,
        sigma_pos=SIGMA_POS,
        sigma_cb=SIGMA_CB,
        sigma_pr=SIGMA_PR,
        resampling="megopolis",
        ess_threshold=0.5,
        seed=42,
    )

    init_pos = epochs[0]["wls_ecef"]
    init_ranges = np.linalg.norm(epochs[0]["sat_ecef"] - init_pos, axis=1)
    init_cb = float(np.median(epochs[0]["pseudoranges"] - init_ranges))
    pf.initialize(init_pos, clock_bias=init_cb, spread_pos=50.0, spread_cb=500.0)

    pf_positions = np.zeros((n_epochs, 3))
    t0 = time.perf_counter()
    prev_arrival_ns = epochs[0]["arrival_ns"]

    for i, ep in enumerate(epochs):
        dt = (ep["arrival_ns"] - prev_arrival_ns) / 1e9 if i > 0 else 1.0
        dt = max(dt, 0.01)
        prev_arrival_ns = ep["arrival_ns"]

        rx_ecef = pf_positions[i - 1] if i > 0 else init_pos
        velocity = compute_doppler_velocity(ep, rx_ecef)

        pf.predict(velocity=velocity, dt=dt)
        pf.correct_clock_bias(ep["sat_ecef"], ep["pseudoranges"])

        weights = elevation_weights(ep["elevations"], ep["cn0"], ELEV_THRESHOLD)
        pf.update_gmm(ep["sat_ecef"], ep["pseudoranges"], weights=weights,
                       sigma_pr=SIGMA_PR, w_los=0.7, mu_nlos=15.0, sigma_nlos=30.0)
        pf.position_update(ep["wls_ecef"], sigma_pos=POS_UPDATE_SIGMA)

        est = pf.estimate()
        pf_positions[i] = est[:3]

        # Divergence reset: if PF drifts > 500m from WLS, re-initialize
        if np.linalg.norm(est[:3] - ep["wls_ecef"]) > 500.0:
            cb = float(np.median(ep["pseudoranges"] -
                       np.linalg.norm(ep["sat_ecef"] - ep["wls_ecef"], axis=1)))
            pf.initialize(ep["wls_ecef"], clock_bias=cb,
                          spread_pos=50.0, spread_cb=500.0)
            pf_positions[i] = ep["wls_ecef"]

    elapsed = time.perf_counter() - t0
    pf_metrics = compute_metrics(pf_positions, gt_matched)

    n_sv_stats = [ep["n_sv"] for ep in epochs]

    result = {
        "run": run_name,
        "phone": phone_name,
        "n_epochs": n_epochs,
        "n_sv_mean": float(np.mean(n_sv_stats)),
        "wls_p50": wls_metrics["p50"],
        "wls_p95": wls_metrics["p95"],
        "wls_rms": wls_metrics["rms_2d"],
        "pf_p50": pf_metrics["p50"],
        "pf_p95": pf_metrics["p95"],
        "pf_rms": pf_metrics["rms_2d"],
        "elapsed_s": elapsed,
        "ms_per_epoch": elapsed * 1000.0 / n_epochs,
        "pf_win": pf_metrics["p50"] < wls_metrics["p50"],
    }

    if verbose:
        win = "WIN" if result["pf_win"] else "LOSE"
        print(f"  {label:65s} WLS P50={wls_metrics['p50']:6.2f}  "
              f"PF P50={pf_metrics['p50']:6.2f}  [{win}]")

    return result


def main():
    parser = argparse.ArgumentParser(description="GSDC2023 PF evaluation")
    parser.add_argument("--single", type=str, default=None,
                        help="Single run: RUN_NAME/PHONE_NAME")
    args = parser.parse_args()

    print("=" * 80)
    print("  GSDC 2023 GPU Particle Filter Multi-Run Evaluation")
    print(f"  Particles: {N_PARTICLES:,}  Signals: {', '.join(SIGNAL_TYPES)}")
    print("=" * 80)

    if args.single:
        parts = args.single.split("/")
        if len(parts) != 2:
            print(f"Error: --single expects RUN/PHONE, got: {args.single}")
            sys.exit(1)
        run_name, phone_name = parts
        data_dir = TRAIN_DIR / run_name / phone_name
        print(f"\n  Single run: {args.single}\n")
        result = run_single(data_dir, run_name, phone_name, verbose=True)
        if result:
            print(f"\n  WLS: P50={result['wls_p50']:.2f}m  P95={result['wls_p95']:.2f}m  "
                  f"RMS={result['wls_rms']:.2f}m")
            print(f"  PF:  P50={result['pf_p50']:.2f}m  P95={result['pf_p95']:.2f}m  "
                  f"RMS={result['pf_rms']:.2f}m")
            print(f"  Runtime: {result['elapsed_s']:.2f}s ({result['ms_per_epoch']:.3f} ms/epoch)")
        return

    # Full evaluation
    runs = discover_runs(TRAIN_DIR)
    print(f"\n  Found {len(runs)} run/phone combinations\n")

    results = []
    for idx, (run_name, phone_name, data_dir) in enumerate(runs):
        result = run_single(data_dir, run_name, phone_name)
        if result:
            results.append(result)

    if not results:
        print("\nNo valid results!")
        return

    # Save CSV
    csv_path = _SCRIPT_DIR / "results" / "gsdc2023_eval.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Results saved: {csv_path}")

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    n_total = len(results)
    n_win = sum(1 for r in results if r["pf_win"])
    mean_wls_p50 = np.mean([r["wls_p50"] for r in results])
    mean_wls_rms = np.mean([r["wls_rms"] for r in results])
    mean_pf_p50 = np.mean([r["pf_p50"] for r in results])
    mean_pf_rms = np.mean([r["pf_rms"] for r in results])
    med_wls_p50 = np.median([r["wls_p50"] for r in results])
    med_pf_p50 = np.median([r["pf_p50"] for r in results])

    print(f"  Runs evaluated:  {n_total}")
    print(f"  PF wins (P50):   {n_win}/{n_total} ({100*n_win/n_total:.0f}%)")
    print()
    print(f"  Mean WLS:  P50={mean_wls_p50:.2f} m   RMS={mean_wls_rms:.2f} m")
    print(f"  Mean PF:   P50={mean_pf_p50:.2f} m   RMS={mean_pf_rms:.2f} m")
    print(f"  Median WLS P50: {med_wls_p50:.2f} m")
    print(f"  Median PF  P50: {med_pf_p50:.2f} m")
    print()
    print(f"  Mean improvement: P50 {mean_wls_p50 - mean_pf_p50:+.2f} m  "
          f"RMS {mean_wls_rms - mean_pf_rms:+.2f} m")
    print("=" * 80)


if __name__ == "__main__":
    main()
