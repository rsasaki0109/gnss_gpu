#!/usr/bin/env python3
"""Generate Kaggle GSDC 2023 submission CSV using GPU Particle Filter.

Reads sample_submission.csv for tripId/UnixTimeMillis, processes each
test run through the PF pipeline, and outputs lat/lon predictions.

Usage
-----
  cd /workspace/ai_coding_ws/gnss_gpu
  PYTHONPATH=python python3 experiments/exp_gsdc2023_submission.py
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

from evaluate import ecef_to_lla
from exp_gsdc2023_pf import (
    compute_doppler_velocity,
    elevation_weights,
    parse_gnss_data,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_DIR = Path("/tmp/gsdc_data/gsdc2023/sdc2023/test")
SAMPLE_SUB = Path("/tmp/gsdc_data/gsdc2023/sdc2023/sample_submission.csv")
OUTPUT_CSV = _SCRIPT_DIR / "results" / "gsdc2023_submission.csv"

N_PARTICLES = 100_000
SIGMA_POS = 10.0
SIGMA_CB = 300.0
SIGMA_PR = 15.0
POS_UPDATE_SIGMA = 3.0
ELEV_THRESHOLD = 15.0

# GPS time → Unix time conversion
GPS_UNIX_OFFSET_NS = 315964800 * 1_000_000_000
LEAP_SECONDS_NS = 18 * 1_000_000_000


def epochs_to_unix_ms(epochs: list[dict]) -> np.ndarray:
    """Convert epoch arrival_ns (GPS epoch) to Unix milliseconds."""
    arr = np.array([ep["arrival_ns"] for ep in epochs], dtype=np.float64)
    unix_ns = arr + GPS_UNIX_OFFSET_NS - LEAP_SECONDS_NS
    return unix_ns / 1_000_000


def run_pf_test(data_dir: Path, label: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Run PF on a test run. Returns (pf_ecef (N,3), unix_ms (N,))."""
    gnss_path = data_dir / "device_gnss.csv"
    if not gnss_path.exists():
        print(f"  [SKIP] {label}: missing device_gnss.csv")
        return None

    try:
        epochs = parse_gnss_data(gnss_path)
    except Exception as e:
        print(f"  [ERROR] {label}: parse failed: {e}")
        return None

    if len(epochs) < 4:
        print(f"  [SKIP] {label}: too few epochs ({len(epochs)})")
        return None

    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    n_epochs = len(epochs)

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

        # Divergence reset
        if np.linalg.norm(est[:3] - ep["wls_ecef"]) > 500.0:
            cb = float(np.median(ep["pseudoranges"] -
                       np.linalg.norm(ep["sat_ecef"] - ep["wls_ecef"], axis=1)))
            pf.initialize(ep["wls_ecef"], clock_bias=cb,
                          spread_pos=50.0, spread_cb=500.0)
            pf_positions[i] = ep["wls_ecef"]

    unix_ms = epochs_to_unix_ms(epochs)
    return pf_positions, unix_ms


def match_submission_times(sub_times_ms: np.ndarray,
                           pf_ecef: np.ndarray,
                           pf_times_ms: np.ndarray) -> np.ndarray:
    """For each submission time, find nearest PF epoch and return lat/lon."""
    n = len(sub_times_ms)
    result = np.zeros((n, 2))  # lat, lon in degrees

    for i in range(n):
        diffs = np.abs(pf_times_ms - sub_times_ms[i])
        idx = int(np.argmin(diffs))
        x, y, z = pf_ecef[idx]
        lat_rad, lon_rad, _ = ecef_to_lla(x, y, z)
        result[i, 0] = math.degrees(lat_rad)
        result[i, 1] = math.degrees(lon_rad)

    return result


def main():
    print("=" * 80)
    print("  GSDC 2023 Submission Generation (GPU Particle Filter)")
    print(f"  Particles: {N_PARTICLES:,}")
    print("=" * 80)

    # Read sample submission
    sub_df = pd.read_csv(SAMPLE_SUB)
    trip_ids = sub_df["tripId"].unique()
    print(f"\n  {len(trip_ids)} tripIds, {len(sub_df)} total rows\n")

    # Process each tripId
    out_rows = []
    t0_total = time.perf_counter()

    for trip_idx, trip_id in enumerate(trip_ids):
        run_name, phone_name = trip_id.split("/")
        data_dir = TEST_DIR / run_name / phone_name
        label = f"[{trip_idx+1:2d}/{len(trip_ids)}] {trip_id}"

        t0 = time.perf_counter()
        result = run_pf_test(data_dir, label)

        if result is None:
            # Fallback: use sample_submission values (dummy)
            sub_trip = sub_df[sub_df["tripId"] == trip_id]
            for _, row in sub_trip.iterrows():
                out_rows.append({
                    "tripId": trip_id,
                    "UnixTimeMillis": int(row["UnixTimeMillis"]),
                    "LatitudeDegrees": row["LatitudeDegrees"],
                    "LongitudeDegrees": row["LongitudeDegrees"],
                })
            print(f"  {label}: FALLBACK (sample_submission values)")
            continue

        pf_ecef, pf_times_ms = result
        sub_trip = sub_df[sub_df["tripId"] == trip_id]
        sub_times = sub_trip["UnixTimeMillis"].values.astype(np.float64)

        latlon = match_submission_times(sub_times, pf_ecef, pf_times_ms)
        elapsed = time.perf_counter() - t0

        for j, (_, row) in enumerate(sub_trip.iterrows()):
            out_rows.append({
                "tripId": trip_id,
                "UnixTimeMillis": int(row["UnixTimeMillis"]),
                "LatitudeDegrees": latlon[j, 0],
                "LongitudeDegrees": latlon[j, 1],
            })

        print(f"  {label}: {len(sub_trip)} rows, "
              f"{len(pf_ecef)} PF epochs, {elapsed:.1f}s")

    # Write output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    total_elapsed = time.perf_counter() - t0_total
    print(f"\n  Output: {OUTPUT_CSV}")
    print(f"  Rows: {len(out_df)}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
