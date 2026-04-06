#!/usr/bin/env python3
"""Experiment: GPU Particle Filter on Google Smartphone Decimeter Challenge 2023 data.

Parses GSDC2023 device_gnss.csv and ground_truth.csv, runs ParticleFilterDevice
with 100K particles, and compares against the Android WLS baseline.

Usage
-----
  cd /workspace/ai_coding_ws/gnss_gpu
  PYTHONPATH=python:third_party/gnssplusplus/build/python:third_party/gnssplusplus/python \
      python3 experiments/exp_gsdc2023_pf.py
"""

from __future__ import annotations

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
DATA_DIR = Path("/tmp/gsdc_data/gsdc2023/sdc2023/train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4")
N_PARTICLES = 100_000
SIGMA_POS = 1.0
SIGMA_CB = 300.0
SIGMA_PR = 3.0
POS_UPDATE_SIGMA = 1.0
ELEV_THRESHOLD = 15.0  # degrees, below this we down-weight


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

    Returns dict with keys:
        epochs: list of dicts, each with:
            arrival_ns, sat_ecef (M,3), pseudoranges (M,), elevations (M,),
            sv_vel (M,3), pr_rate (M,), wls_ecef (3,)
    """
    df = pd.read_csv(gnss_path)

    # Filter to GPS L1 only for cleaner results
    df = df[df["SignalType"] == "GPS_L1_CA"].copy()

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

        # Corrected pseudorange: add satellite clock correction, subtract atmospheric delays
        corrected_pr = raw_pr + sv_clock_bias - iono_delay - tropo_delay

        elevations = group["SvElevationDegrees"].values.astype(np.float64)
        pr_rate = group["PseudorangeRateMetersPerSecond"].values.astype(np.float64)

        wls_ecef = np.array([
            group["WlsPositionXEcefMeters"].iloc[0],
            group["WlsPositionYEcefMeters"].iloc[0],
            group["WlsPositionZEcefMeters"].iloc[0],
        ], dtype=np.float64)

        epochs.append({
            "arrival_ns": arrival_ns,
            "sat_ecef": sat_ecef,
            "pseudoranges": corrected_pr,
            "elevations": elevations,
            "sv_vel": sv_vel,
            "pr_rate": pr_rate,
            "wls_ecef": wls_ecef,
            "n_sv": n_sv,
        })

    return epochs


def compute_doppler_velocity(epoch: dict, rx_ecef: np.ndarray) -> np.ndarray:
    """Compute receiver velocity from Doppler (pseudorange rate) + SV velocity.

    PseudorangeRateMetersPerSecond = dot(sv_vel - rx_vel, unit_los) + clock_drift
    We solve for rx_vel using least squares (ignoring clock drift term for simplicity,
    or solving 4-state [vx, vy, vz, clock_drift]).
    """
    sat_ecef = epoch["sat_ecef"]
    sv_vel = epoch["sv_vel"]
    pr_rate = epoch["pr_rate"]
    n_sv = len(pr_rate)

    if n_sv < 4:
        return np.zeros(3)

    # Line-of-sight unit vectors
    los = sat_ecef - rx_ecef
    ranges = np.linalg.norm(los, axis=1, keepdims=True)
    ranges = np.maximum(ranges, 1.0)
    los_unit = los / ranges  # (n_sv, 3)

    # Design matrix: H = [-los_unit, 1]  (4-state: vx, vy, vz, clock_drift)
    H = np.column_stack([-los_unit, np.ones(n_sv)])

    # Observation: pr_rate = dot(sv_vel - rx_vel, los_unit) + clock_drift
    # => pr_rate - dot(sv_vel, los_unit) = dot(-rx_vel, los_unit) + clock_drift
    # => pr_rate - dot(sv_vel, los_unit) = H @ [vx, vy, vz, cd]
    sv_radial = np.sum(sv_vel * los_unit, axis=1)
    obs = pr_rate - sv_radial

    try:
        x, _, _, _ = np.linalg.lstsq(H, obs, rcond=None)
        return x[:3]  # receiver velocity ECEF
    except np.linalg.LinAlgError:
        return np.zeros(3)


def elevation_weights(elevations: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    """Compute elevation-based weights. Down-weight low-elevation SVs."""
    weights = np.ones(len(elevations))
    for i, el in enumerate(elevations):
        if el < threshold:
            # Smooth down-weight: sin^2(el) for low elevation
            weights[i] = max(0.1, math.sin(math.radians(el)) ** 2)
    return weights


def match_gt_to_epochs(epochs: list[dict], gt_ecef: np.ndarray, gt_times_ms: np.ndarray) -> list[int]:
    """For each epoch, find closest GT by time. Returns list of GT indices."""
    # Convert epoch arrival_ns to Unix milliseconds
    # GPS epoch offset: GPS time starts 1980-01-06, Unix starts 1970-01-01
    # Difference: 315964800 seconds = 315964800000 ms
    # But arrival_ns is nanoseconds since GPS epoch
    GPS_UNIX_OFFSET_NS = 315964800 * 1_000_000_000  # GPS to Unix offset in ns
    LEAP_SECONDS_NS = 18 * 1_000_000_000  # approximate leap seconds

    gt_indices = []
    for ep in epochs:
        # Convert arrival time from GPS nanos to Unix millis
        unix_ns = ep["arrival_ns"] + GPS_UNIX_OFFSET_NS - LEAP_SECONDS_NS
        unix_ms = unix_ns / 1_000_000

        # Find closest GT
        diffs = np.abs(gt_times_ms - unix_ms)
        idx = int(np.argmin(diffs))
        gt_indices.append(idx)

    return gt_indices


def main():
    print("=" * 72)
    print("  GSDC 2023 GPU Particle Filter Experiment")
    print("  Data: 2020-06-25-00-34-us-ca-mtv-sb-101/pixel4")
    print("=" * 72)

    # ------------------------------------------------------------------
    # [1] Parse data
    # ------------------------------------------------------------------
    print("\n[1] Parsing data ...")

    gt_ecef, gt_times_ms = parse_ground_truth(DATA_DIR / "ground_truth.csv")
    print(f"    Ground truth: {len(gt_ecef)} epochs")

    epochs = parse_gnss_data(DATA_DIR / "device_gnss.csv")
    print(f"    GNSS epochs (GPS L1 CA, >=4 SVs): {len(epochs)}")

    # Match GT to epochs
    gt_indices = match_gt_to_epochs(epochs, gt_ecef, gt_times_ms)
    n_epochs = len(epochs)
    n_sv_stats = [ep["n_sv"] for ep in epochs]
    print(f"    SVs per epoch: min={min(n_sv_stats)}, max={max(n_sv_stats)}, "
          f"mean={np.mean(n_sv_stats):.1f}")

    # ------------------------------------------------------------------
    # [2] WLS baseline (from Android)
    # ------------------------------------------------------------------
    print("\n[2] Computing WLS baseline (Android solution) ...")

    wls_positions = np.zeros((n_epochs, 3))
    gt_matched = np.zeros((n_epochs, 3))
    for i, ep in enumerate(epochs):
        wls_positions[i] = ep["wls_ecef"]
        gt_matched[i] = gt_ecef[gt_indices[i]]

    wls_metrics = compute_metrics(wls_positions, gt_matched)
    print(f"    WLS: P50={wls_metrics['p50']:.2f} m, P95={wls_metrics['p95']:.2f} m, "
          f"RMS={wls_metrics['rms_2d']:.2f} m")

    # ------------------------------------------------------------------
    # [3] Run Particle Filter
    # ------------------------------------------------------------------
    print(f"\n[3] Running ParticleFilterDevice ({N_PARTICLES:,} particles) ...")

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

    # Initialize from first WLS solution
    init_pos = epochs[0]["wls_ecef"]
    # Estimate initial clock bias from pseudoranges
    init_ranges = np.linalg.norm(epochs[0]["sat_ecef"] - init_pos, axis=1)
    init_cb = float(np.median(epochs[0]["pseudoranges"] - init_ranges))

    pf.initialize(init_pos, clock_bias=init_cb, spread_pos=50.0, spread_cb=500.0)
    print(f"    Init pos: [{init_pos[0]:.1f}, {init_pos[1]:.1f}, {init_pos[2]:.1f}]")
    print(f"    Init cb: {init_cb:.1f} m")

    pf_positions = np.zeros((n_epochs, 3))
    t0 = time.perf_counter()

    prev_arrival_ns = epochs[0]["arrival_ns"]

    for i, ep in enumerate(epochs):
        # Time step
        dt = (ep["arrival_ns"] - prev_arrival_ns) / 1e9 if i > 0 else 1.0
        dt = max(dt, 0.01)  # safety clamp
        prev_arrival_ns = ep["arrival_ns"]

        # Doppler velocity from previous estimate
        if i > 0:
            rx_ecef = pf_positions[i - 1]
        else:
            rx_ecef = init_pos
        velocity = compute_doppler_velocity(ep, rx_ecef)

        # Predict
        pf.predict(velocity=velocity, dt=dt)

        # Clock bias correction
        pf.correct_clock_bias(ep["sat_ecef"], ep["pseudoranges"])

        # Elevation weights
        weights = elevation_weights(ep["elevations"], ELEV_THRESHOLD)

        # Pseudorange update
        pf.update(ep["sat_ecef"], ep["pseudoranges"], weights=weights)

        # Position update from WLS
        pf.position_update(ep["wls_ecef"], sigma_pos=POS_UPDATE_SIGMA)

        est = pf.estimate()
        pf_positions[i] = est[:3]

    elapsed = time.perf_counter() - t0
    ms_per_epoch = elapsed * 1000.0 / n_epochs
    print(f"    Done: {elapsed:.2f} s total, {ms_per_epoch:.3f} ms/epoch")

    # ------------------------------------------------------------------
    # [4] Compute metrics
    # ------------------------------------------------------------------
    print("\n[4] Computing metrics ...")

    pf_metrics = compute_metrics(pf_positions, gt_matched)

    all_metrics = {
        "WLS (Android)": wls_metrics,
        f"PF-{N_PARTICLES//1000}K": pf_metrics,
    }

    # ------------------------------------------------------------------
    # [5] Results
    # ------------------------------------------------------------------
    print("\n[5] Results:")
    print_comparison_table(all_metrics)

    print("\n  Detailed breakdown:")
    for label, m in all_metrics.items():
        print(f"\n  {label}:")
        print(f"    P50  = {m['p50']:.2f} m")
        print(f"    P95  = {m['p95']:.2f} m")
        print(f"    RMS  = {m['rms_2d']:.2f} m")
        print(f"    Mean = {m['mean_2d']:.2f} m")
        print(f"    Max  = {m['max_2d']:.2f} m")

    print(f"\n  PF runtime: {elapsed:.2f} s ({ms_per_epoch:.3f} ms/epoch)")
    print(f"  Epochs: {n_epochs}")
    print("=" * 72)


if __name__ == "__main__":
    main()
