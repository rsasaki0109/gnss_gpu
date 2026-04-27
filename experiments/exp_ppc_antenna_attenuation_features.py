#!/usr/bin/env python3
"""Simplified RHCP antenna gain + NLOS attenuation features for one PPC run.

Physical motivation:
    Effective received signal power per satellite ≈
        P_tx + free_space_loss + RHCP_gain(elev) + NLOS_attenuation
    where:
    - RHCP_gain(elev) = G_max * sin²(elev)   (peaks at zenith,
      0 at horizon; simplified GNSS-receiver-antenna model)
    - NLOS_attenuation ≈ -25 dB if blocked by buildings, 0 dB otherwise
    - Free-space loss is roughly constant across satellites and so
      contributes nothing to discriminative scoring.

Per-satellite "effective dB" is therefore computed in dB as:
    eff_db = 10 * log10(sin²(elev) + ε)  +  (0 if LoS else -25)

Per-window features:
    eff_db_p10 / p25 / p50 / p75 / p90: quantiles across all sats
        and all epochs in the window.
    eff_db_max: best-receivable signal proxy.
    usable_count: count of sats with eff_db ≥ -3 (good signal).
    marginal_count: count of sats with eff_db ≥ -10 (marginal).
    nlos_at_high_elev_count: count of sats with elev > 30° AND NLOS
        (these are the multipath-prone NLOS cases; low-elev NLOS is
        expected anywhere in urban environments).

This is the deliberately simple version of Furukawa 2019's signal-
quality estimation; UTD diffraction and full antenna pattern are
omitted.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_pf3d_residual_analysis import _reference_state_at_truth
from exp_urbannav_baseline import load_real_data
from fetch_plateau_subset import PRESET_URLS, expand_meshes
from gnss_gpu.io.plateau import load_plateau
from scan_ppc_plateau_segments import derive_mesh_codes, ensure_subset


RESULTS_DIR = _SCRIPT_DIR / "results"
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _ecef_to_enu_basis(ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = ecef
    p = np.sqrt(x * x + y * y)
    if p < 1e-3:
        return np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(5):
        sinlat = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sinlat * sinlat)
        lat = np.arctan2(z + WGS84_E2 * N * sinlat, p)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    sinlon = np.sin(lon)
    coslon = np.cos(lon)
    east = np.array([-sinlon, coslon, 0.0])
    north = np.array([-sinlat * coslon, -sinlat * sinlon, coslat])
    up = np.array([coslat * coslon, coslat * sinlon, sinlat])
    return east, north, up


def _elevations(rx: np.ndarray, sat_ecef: np.ndarray) -> np.ndarray:
    east, north, up = _ecef_to_enu_basis(rx)
    rel = sat_ecef - rx
    e = rel @ east
    n = rel @ north
    u = rel @ up
    horiz = np.sqrt(e * e + n * n)
    return np.arctan2(u, horiz)  # radians


def _per_sat_features(
    elev_rad: np.ndarray,
    is_los: np.ndarray,
    nlos_attenuation_db: float,
    epsilon: float,
) -> dict[str, np.ndarray]:
    sin_elev = np.sin(np.maximum(elev_rad, 0.0))
    gain_linear = sin_elev * sin_elev
    gain_db = 10.0 * np.log10(gain_linear + epsilon)
    nlos_term = np.where(is_los, 0.0, nlos_attenuation_db)
    eff_db = gain_db + nlos_term
    return {
        "elev_rad": elev_rad,
        "elev_deg": np.degrees(elev_rad),
        "gain_db": gain_db,
        "eff_db": eff_db,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Antenna gain + NLOS attenuation features")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESET_URLS), default="tokyo23")
    parser.add_argument("--systems", default="G,R,J,E")
    parser.add_argument("--plateau-zone", type=int, default=9)
    parser.add_argument("--mesh-radius", type=int, default=1)
    parser.add_argument("--epoch-stride", type=int, default=1)
    parser.add_argument("--subset-root", type=Path, default=Path("/tmp/plateau_segment_cache"))
    parser.add_argument("--window-duration-s", type=float, default=30.0)
    parser.add_argument("--nlos-attenuation-db", type=float, default=-25.0)
    parser.add_argument("--gain-epsilon", type=float, default=1e-3)
    parser.add_argument("--results-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = tuple(s.strip().upper() for s in args.systems.split(","))
    city = args.run_dir.parent.name
    run = args.run_dir.name
    prefix = args.results_prefix or f"ppc_antenna_features_{city}_{run}"

    print(f"loading {city}/{run}...", flush=True)
    t0 = time.monotonic()
    data = load_real_data(args.run_dir, max_epochs=None, systems=systems)
    print(f"  loaded in {time.monotonic() - t0:.1f}s; epochs={len(data['times'])}", flush=True)

    ground_truth = np.asarray(data["ground_truth"], dtype=np.float64)
    mesh_codes = derive_mesh_codes(ground_truth)
    expanded = expand_meshes(sorted(set(mesh_codes)), args.mesh_radius)

    print("loading PLATEAU + BVH...", flush=True)
    t0 = time.monotonic()
    subset_dir = ensure_subset(PRESET_URLS[args.preset], expanded, args.subset_root)
    model = load_plateau(subset_dir, zone=args.plateau_zone)
    from gnss_gpu.bvh import BVHAccelerator
    accelerator = BVHAccelerator.from_building_model(model)
    print(f"  BVH built in {time.monotonic() - t0:.1f}s ({accelerator.n_triangles} triangles)", flush=True)

    epoch_rows: list[dict[str, object]] = []
    n_epochs = len(data["times"])
    print("per-epoch antenna features...", flush=True)
    t0 = time.monotonic()
    for i in range(0, n_epochs, args.epoch_stride):
        sat_ecef = np.asarray(data["sat_ecef"][i], dtype=np.float64)
        if sat_ecef.size == 0:
            continue
        pseudoranges = np.asarray(data["pseudoranges"][i], dtype=np.float64)
        weights = np.asarray(data["weights"][i], dtype=np.float64)
        truth_state = _reference_state_at_truth(sat_ecef, pseudoranges, weights, ground_truth[i])
        rx = truth_state[:3]
        is_los = np.asarray(accelerator.check_los(rx, sat_ecef), dtype=bool)
        elev_rad = _elevations(rx, sat_ecef)
        feats = _per_sat_features(elev_rad, is_los, args.nlos_attenuation_db, args.gain_epsilon)
        eff = feats["eff_db"]
        elev_deg = feats["elev_deg"]
        sat_count = int(eff.size)
        if sat_count == 0:
            continue
        usable = int(np.count_nonzero(eff >= -3.0))
        marginal = int(np.count_nonzero(eff >= -10.0))
        nlos_high_elev = int(np.count_nonzero((elev_deg > 30.0) & (~is_los)))
        epoch_rows.append({
            "city": city, "run": run, "epoch": i,
            "gps_tow": float(data["times"][i]),
            "sat_count": sat_count,
            "los_count": int(np.count_nonzero(is_los)),
            "nlos_count": int(np.count_nonzero(~is_los)),
            "nlos_at_high_elev_count": nlos_high_elev,
            "elev_deg_p50": float(np.percentile(elev_deg, 50)),
            "elev_deg_max": float(elev_deg.max()),
            "eff_db_p10": float(np.percentile(eff, 10)),
            "eff_db_p25": float(np.percentile(eff, 25)),
            "eff_db_p50": float(np.percentile(eff, 50)),
            "eff_db_p75": float(np.percentile(eff, 75)),
            "eff_db_p90": float(np.percentile(eff, 90)),
            "eff_db_max": float(eff.max()),
            "eff_db_mean": float(eff.mean()),
            "usable_count": usable,
            "marginal_count": marginal,
            "gain_db_mean": float(feats["gain_db"].mean()),
        })
        if i > 0 and (i // max(args.epoch_stride, 1)) % 1000 == 0:
            elapsed = time.monotonic() - t0
            rate = (i / args.epoch_stride) / max(elapsed, 1e-3)
            remaining = (n_epochs / args.epoch_stride - i / args.epoch_stride) / max(rate, 1e-3)
            print(f"  epoch {i}/{n_epochs}  rate={rate:.0f}/s  eta={remaining:.0f}s", flush=True)
    print(f"  done in {time.monotonic() - t0:.1f}s", flush=True)

    epoch_df = pd.DataFrame(epoch_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epoch_path = RESULTS_DIR / f"{prefix}_per_epoch.csv"
    epoch_df.to_csv(epoch_path, index=False)
    print(f"saved: {epoch_path} ({len(epoch_df)} rows)")

    if epoch_df.empty:
        return

    t_start = epoch_df["gps_tow"].min()
    epoch_df["window_index"] = ((epoch_df["gps_tow"] - t_start) // args.window_duration_s).astype(int)
    win = epoch_df.groupby("window_index").agg(
        epoch_count=("gps_tow", "size"),
        eff_db_p10_mean=("eff_db_p10", "mean"),
        eff_db_p10_min=("eff_db_p10", "min"),
        eff_db_p50_mean=("eff_db_p50", "mean"),
        eff_db_p90_mean=("eff_db_p90", "mean"),
        eff_db_max_mean=("eff_db_max", "mean"),
        eff_db_max_max=("eff_db_max", "max"),
        eff_db_mean_mean=("eff_db_mean", "mean"),
        usable_count_mean=("usable_count", "mean"),
        usable_count_min=("usable_count", "min"),
        marginal_count_mean=("marginal_count", "mean"),
        nlos_at_high_elev_count_mean=("nlos_at_high_elev_count", "mean"),
        nlos_at_high_elev_count_max=("nlos_at_high_elev_count", "max"),
        gain_db_mean_mean=("gain_db_mean", "mean"),
        elev_deg_p50_mean=("elev_deg_p50", "mean"),
    ).reset_index()
    win.insert(0, "run", run)
    win.insert(0, "city", city)
    win_path = RESULTS_DIR / f"{prefix}_per_window.csv"
    win.to_csv(win_path, index=False)
    print(f"saved: {win_path} ({len(win)} rows)")


if __name__ == "__main__":
    main()
