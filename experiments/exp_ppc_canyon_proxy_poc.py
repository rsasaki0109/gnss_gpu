#!/usr/bin/env python3
"""Heuristic multipath-risk proxy for one PPC run.

Since BVH multipath is not compiled and brute-force compute_multipath
OOMs on the full 2.2M-triangle PLATEAU set, this script computes a
deployable proxy that uses only the BVH `check_los` primitive (fast):

(1) Azimuthal canyon score: cast 16 low-elevation rays in a circle from
    the receiver and count how many are blocked by buildings within
    ~150 m.  0 = open sky everywhere; 16 = completely surrounded.

(2) Per-satellite near-edge risk: for each tracked satellite, also
    check LoS at +/- 3 deg azimuth perturbations.  If any of the three
    rays disagrees with the others, the satellite is near a building
    edge.  Count = number of edge-near sats per epoch.

These two features stress-test the hypothesis that "urban canyon
density at the receiver" + "satellites grazing building edges"
explain demo5 FIX failures that the existing sim_los_* features miss.

Outputs under `experiments/results/`:

- `ppc_canyon_proxy_<city>_<run>_per_epoch.csv`
- `ppc_canyon_proxy_<city>_<run>_per_window.csv`
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
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    lon = np.arctan2(y, x)
    # Compute geodetic latitude iteratively
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


def _probe_azimuth_canyon(
    accelerator: object,
    rx: np.ndarray,
    elevation_deg: float,
    n_bins: int,
    probe_distance_m: float,
) -> int:
    """Return number of azimuth bins where a low-elevation ray is blocked
    by buildings within `probe_distance_m`.
    """
    east, north, up = _ecef_to_enu_basis(rx)
    azimuths = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False)
    el = np.radians(elevation_deg)
    cos_el = np.cos(el)
    sin_el = np.sin(el)
    # Synthetic "satellite" points at probe_distance_m along (az, el)
    sats = np.zeros((n_bins, 3), dtype=np.float64)
    for i, az in enumerate(azimuths):
        dir_enu = (np.cos(az) * cos_el, np.sin(az) * cos_el, sin_el)
        dir_ecef = dir_enu[0] * east + dir_enu[1] * north + dir_enu[2] * up
        sats[i] = rx + probe_distance_m * dir_ecef
    los = np.asarray(accelerator.check_los(rx, sats), dtype=bool)
    return int(np.count_nonzero(~los))


def _per_sat_edge_risk(
    accelerator: object,
    rx: np.ndarray,
    sat_ecef: np.ndarray,
    az_perturbation_deg: float,
) -> tuple[int, int]:
    """For each satellite check LoS at the original azimuth + two
    perturbations; return (edge_near_count, total_count).  An edge-near
    satellite has at least one disagreement among the three samples.
    """
    if len(sat_ecef) == 0:
        return 0, 0
    east, north, up = _ecef_to_enu_basis(rx)
    # Compute az/el for each sat
    rel = sat_ecef - rx
    e = rel @ east
    n = rel @ north
    u = rel @ up
    az = np.arctan2(e, n)  # 0=north
    horiz = np.sqrt(e * e + n * n)
    el = np.arctan2(u, horiz)
    perturb = np.radians(az_perturbation_deg)
    edge_count = 0
    los_main = np.asarray(accelerator.check_los(rx, sat_ecef), dtype=bool)
    # Build perturbed sats (keep distance, shift az)
    dist = np.linalg.norm(rel, axis=1)
    sats_plus = np.zeros_like(sat_ecef)
    sats_minus = np.zeros_like(sat_ecef)
    for i in range(len(sat_ecef)):
        for sign, dst_arr in [(+1, sats_plus), (-1, sats_minus)]:
            new_az = az[i] + sign * perturb
            new_e = np.sin(new_az) * np.cos(el[i])
            new_n = np.cos(new_az) * np.cos(el[i])
            new_u = np.sin(el[i])
            d_ecef = new_e * east + new_n * north + new_u * up
            dst_arr[i] = rx + dist[i] * d_ecef
    los_plus = np.asarray(accelerator.check_los(rx, sats_plus), dtype=bool)
    los_minus = np.asarray(accelerator.check_los(rx, sats_minus), dtype=bool)
    disagreement = (los_main != los_plus) | (los_main != los_minus)
    edge_count = int(disagreement.sum())
    return edge_count, len(sat_ecef)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic multipath proxy via canyon scoring")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESET_URLS), default="tokyo23")
    parser.add_argument("--systems", default="G,R,J,E")
    parser.add_argument("--plateau-zone", type=int, default=9)
    parser.add_argument("--mesh-radius", type=int, default=1)
    parser.add_argument("--epoch-stride", type=int, default=1)
    parser.add_argument("--n-azimuth-bins", type=int, default=16)
    parser.add_argument("--canyon-elevation-deg", type=float, default=5.0)
    parser.add_argument("--canyon-probe-distance-m", type=float, default=150.0)
    parser.add_argument("--edge-az-perturb-deg", type=float, default=3.0)
    parser.add_argument("--subset-root", type=Path, default=Path("/tmp/plateau_segment_cache"))
    parser.add_argument("--window-duration-s", type=float, default=30.0)
    parser.add_argument("--results-prefix", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = tuple(s.strip().upper() for s in args.systems.split(","))
    city = args.run_dir.parent.name
    run = args.run_dir.name
    prefix = args.results_prefix or f"ppc_canyon_proxy_{city}_{run}"
    print(f"loading real data for {city}/{run}...", flush=True)
    t0 = time.monotonic()
    data = load_real_data(args.run_dir, max_epochs=None, systems=systems)
    print(f"  loaded in {time.monotonic() - t0:.1f}s; epochs={len(data['times'])}", flush=True)
    ground_truth = np.asarray(data["ground_truth"], dtype=np.float64)
    mesh_codes = derive_mesh_codes(ground_truth)
    expanded = expand_meshes(sorted(set(mesh_codes)), args.mesh_radius)
    print(f"  meshes: {len(set(mesh_codes))} unique, {len(expanded)} expanded", flush=True)

    print("loading PLATEAU + building BVH...", flush=True)
    t0 = time.monotonic()
    subset_dir = ensure_subset(PRESET_URLS[args.preset], expanded, args.subset_root)
    model = load_plateau(subset_dir, zone=args.plateau_zone)
    from gnss_gpu.bvh import BVHAccelerator
    accelerator = BVHAccelerator.from_building_model(model)
    print(f"  BVH built in {time.monotonic() - t0:.1f}s ({accelerator.n_triangles} triangles)", flush=True)

    epoch_rows: list[dict[str, object]] = []
    n_epochs = len(data["times"])
    print("per-epoch canyon + edge probing...", flush=True)
    t0 = time.monotonic()
    for i in range(0, n_epochs, args.epoch_stride):
        sat_ecef = np.asarray(data["sat_ecef"][i], dtype=np.float64)
        if sat_ecef.size == 0:
            continue
        pseudoranges = np.asarray(data["pseudoranges"][i], dtype=np.float64)
        weights = np.asarray(data["weights"][i], dtype=np.float64)
        truth_state = _reference_state_at_truth(sat_ecef, pseudoranges, weights, ground_truth[i])
        rx = truth_state[:3]
        canyon = _probe_azimuth_canyon(
            accelerator, rx,
            elevation_deg=args.canyon_elevation_deg,
            n_bins=args.n_azimuth_bins,
            probe_distance_m=args.canyon_probe_distance_m,
        )
        edge_near, sat_count = _per_sat_edge_risk(
            accelerator, rx, sat_ecef, args.edge_az_perturb_deg
        )
        epoch_rows.append({
            "city": city, "run": run, "epoch": i,
            "gps_tow": float(data["times"][i]),
            "canyon_blocked_count": canyon,
            "canyon_blocked_fraction": canyon / max(args.n_azimuth_bins, 1),
            "edge_near_sat_count": edge_near,
            "sat_count": sat_count,
            "edge_near_sat_fraction": edge_near / max(sat_count, 1),
        })
        if i > 0 and (i // args.epoch_stride) % 500 == 0:
            elapsed = time.monotonic() - t0
            rate = (i / args.epoch_stride) / elapsed
            remaining = (n_epochs / args.epoch_stride - i / args.epoch_stride) / max(rate, 1e-3)
            print(f"  epoch {i}/{n_epochs}  rate={rate:.0f}/s  eta={remaining:.0f}s", flush=True)
    print(f"  done in {time.monotonic() - t0:.1f}s", flush=True)

    epoch_df = pd.DataFrame(epoch_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epoch_path = RESULTS_DIR / f"{prefix}_per_epoch.csv"
    epoch_df.to_csv(epoch_path, index=False)
    print(f"saved: {epoch_path} ({len(epoch_df)} rows)")

    if not epoch_df.empty:
        t_run_start = epoch_df["gps_tow"].min()
        epoch_df["window_index"] = ((epoch_df["gps_tow"] - t_run_start) // args.window_duration_s).astype(int)
        win = epoch_df.groupby("window_index").agg(
            epoch_count=("gps_tow", "size"),
            canyon_blocked_count_mean=("canyon_blocked_count", "mean"),
            canyon_blocked_count_max=("canyon_blocked_count", "max"),
            canyon_blocked_fraction_mean=("canyon_blocked_fraction", "mean"),
            canyon_blocked_fraction_max=("canyon_blocked_fraction", "max"),
            edge_near_sat_count_mean=("edge_near_sat_count", "mean"),
            edge_near_sat_count_max=("edge_near_sat_count", "max"),
            edge_near_sat_fraction_mean=("edge_near_sat_fraction", "mean"),
            edge_near_sat_fraction_max=("edge_near_sat_fraction", "max"),
        ).reset_index()
        win.insert(0, "run", run)
        win.insert(0, "city", city)
        win_path = RESULTS_DIR / f"{prefix}_per_window.csv"
        win.to_csv(win_path, index=False)
        print(f"saved: {win_path} ({len(win)} rows)")


if __name__ == "__main__":
    main()
