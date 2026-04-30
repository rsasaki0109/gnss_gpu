#!/usr/bin/env python3
"""Phase 2 early-abort check for PR #38 (PLATEAU bridge geometry).

For Tokyo run2 epochs in the hidden-high cluster w23-w27, build two BVHs:
- bldg only
- bldg + brid (PLATEAU bridges, requires the load_plateau patch)

Call check_los on both and report per-epoch satellite flips
(LoS->NLoS due to bridge mesh).  If at least one satellite per epoch
flips, Phase 2 (full retrain) is justified.  If 0 flips across all
epochs, Phase 2 should be aborted.

Self-contained: uses only gnss_gpu.io.rinex / .io.nav_rinex /
.ephemeris / .bvh / .io.plateau.  Does NOT use the experiments/
helpers because those drag in modules missing from sister tree.
"""

from __future__ import annotations

import csv
import glob
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.plateau import load_plateau
from gnss_gpu.io.rinex import read_rinex_obs


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_WEEK_SEC = 604800.0


def datetime_to_gps_tow(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = (dt - GPS_EPOCH).total_seconds()
    week = int(delta // GPS_WEEK_SEC)
    return delta - week * GPS_WEEK_SEC


def lla_to_ecef(lat_deg: float, lon_deg: float, alt: float) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sinl = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sinl * sinl)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - WGS84_E2) + alt) * sinl
    return np.array([x, y, z])


def ecef_to_enu_basis(ecef: np.ndarray):
    x, y, z = ecef
    p = np.sqrt(x * x + y * y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(5):
        sl = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
        lat = np.arctan2(z + WGS84_E2 * N * sl, p)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    east = np.array([-so, co, 0.0])
    north = np.array([-sl * co, -sl * so, cl])
    up = np.array([cl * co, cl * so, sl])
    return east, north, up


def elevations_deg(rx: np.ndarray, sat_ecef: np.ndarray) -> np.ndarray:
    e, n, u = ecef_to_enu_basis(rx)
    rel = sat_ecef - rx
    eu = rel @ e
    nu = rel @ n
    uu = rel @ u
    horiz = np.sqrt(eu * eu + nu * nu)
    return np.degrees(np.arctan2(uu, horiz))


def load_reference_csv(path: Path):
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(
                (
                    float(r["GPS TOW (s)"]),
                    float(r["Latitude (deg)"]),
                    float(r["Longitude (deg)"]),
                    float(r["Ellipsoid Height (m)"]),
                )
            )
    return rows


def find_rx_at_tow(ref_rows, target_tow):
    best = min(ref_rows, key=lambda r: abs(r[0] - target_tow))
    actual_tow, lat, lon, alt = best
    rx = lla_to_ecef(lat, lon, alt)
    return rx, actual_tow, (lat, lon, alt)


def build_combined_mesh_dir(out_dir: Path, bldg_files, brid_files):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for f in bldg_files:
        os.symlink(f, out_dir / os.path.basename(f))
    for f in brid_files:
        os.symlink(f, out_dir / os.path.basename(f))


def main():
    run_dir = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data/tokyo/run2")
    rover_obs = run_dir / "rover.obs"
    base_nav = run_dir / "base.nav"
    reference_csv = run_dir / "reference.csv"

    # w7/w9 (false-high cluster, expected to gain occlusion from bridges)
    # and w23-w27 (hidden-high cluster, original Phase 2 target)
    target_tows = [
        177210.0, 177230.0,            # w7
        177270.0, 177290.0,            # w9
        177700.0, 177750.0,            # w23
        177800.0, 177810.0, 177830.0,  # w26-w27
    ]

    print("loading reference.csv...")
    ref = load_reference_csv(reference_csv)
    print(f"  {len(ref)} rows; TOW [{ref[0][0]:.1f}, {ref[-1][0]:.1f}]")

    print("\nresolving target rx positions...")
    rx_at_tow = {}
    for t in target_tows:
        rx, actual, lla = find_rx_at_tow(ref, t)
        rx_at_tow[t] = (rx, actual, lla)
        print(f"  TOW {t:.1f} -> nearest {actual:.1f}, "
              f"lat {lla[0]:.5f} lon {lla[1]:.5f} alt {lla[2]:.2f}")

    print("\nreading rover.obs (this may take a minute)...")
    t0 = time.monotonic()
    obs = read_rinex_obs(rover_obs)
    print(f"  {len(obs.epochs)} epochs in {time.monotonic()-t0:.1f}s")

    print("\nreading base.nav...")
    t0 = time.monotonic()
    nav = read_nav_rinex_multi(str(base_nav), systems=("G", "E", "J", "C", "R"))
    print(f"  {sum(len(v) for v in nav.values())} nav records, {len(nav)} PRNs, "
          f"in {time.monotonic()-t0:.1f}s")
    eph = Ephemeris(nav)
    avail_prns = eph.available_prns
    print(f"  ephemeris available: {len(avail_prns)} PRNs (first: {avail_prns[:10]})")

    # Index obs epochs by TOW for fast lookup
    obs_index = {}
    for ep in obs.epochs:
        ep_tow = datetime_to_gps_tow(ep.time)
        obs_index[ep_tow] = ep

    sorted_obs_tows = sorted(obs_index)

    def nearest_obs(target):
        i = int(np.argmin([abs(t - target) for t in sorted_obs_tows]))
        return obs_index[sorted_obs_tows[i]], sorted_obs_tows[i]

    print("\nfinding observed satellites at target epochs...")
    sat_at_tow = {}
    for t in target_tows:
        ep, ep_tow = nearest_obs(t)
        sats = list(ep.satellites)
        sat_at_tow[t] = (ep_tow, sats)
        print(f"  TOW {t:.1f} -> obs epoch {ep_tow:.1f}, {len(sats)} sats: {sats[:8]}{'...' if len(sats)>8 else ''}")

    # Locate cached PLATEAU bldg files for the trajectory area
    print("\nlocating PLATEAU bldg cache for tokyo run2 area...")
    bldg_caches = {}
    for c in glob.glob("/tmp/plateau_segment_cache/*"):
        files = glob.glob(os.path.join(c, "53393683_bldg_*.gml"))
        if files:
            bldg_caches[c] = sorted(glob.glob(os.path.join(c, "*_bldg_*.gml")))
    if not bldg_caches:
        raise SystemExit("no bldg cache containing mesh 53393683 found - run "
                         "fetch_plateau_subset.py first")
    cache, bldg_files = next(iter(bldg_caches.items()))
    print(f"  cache: {cache}  ({len(bldg_files)} bldg gml)")

    # Filter bldg files to the trajectory neighborhood (3x3 around 53393683)
    target_meshes = {"53393683", "53393682", "53393684",
                     "53393773", "53393773", "53393593",
                     "53393691", "53393681", "53393685"}
    near_bldg = [f for f in bldg_files
                 if any(os.path.basename(f).startswith(m + "_") for m in target_meshes)]
    print(f"  near-trajectory bldg gml: {len(near_bldg)}")

    # Pick up all brid GML in the staging dir (w23-w27 mesh + w7/w9 mesh)
    brid_dir = Path("/tmp/tokyo_w23w27_brid_tran")
    brid_paths = sorted(str(p) for p in brid_dir.glob("*_brid_*.gml"))
    if not brid_paths:
        raise SystemExit(f"no brid GML found in {brid_dir}")
    print(f"  brid gmls: {len(brid_paths)}")
    for p in brid_paths:
        print(f"    {os.path.basename(p)}  ({os.path.getsize(p)/1024:.1f} KB)")

    # Also expand near-trajectory bldg files to include w7/w9 area
    extra_meshes = {"53393692", "53393691", "53393693"}
    extra_bldg = [f for f in bldg_files
                  if any(os.path.basename(f).startswith(m + "_") for m in extra_meshes)]
    near_bldg = sorted(set(near_bldg) | set(extra_bldg))
    print(f"  expanded near-trajectory bldg gml: {len(near_bldg)}")

    # Build merged dirs
    bldg_only_dir = Path("/tmp/phase2_check_bldg_only")
    bldg_brid_dir = Path("/tmp/phase2_check_bldg_brid")
    print(f"\nbuilding mesh dirs...")
    build_combined_mesh_dir(bldg_only_dir, near_bldg, [])
    build_combined_mesh_dir(bldg_brid_dir, near_bldg, brid_paths)

    print("loading bldg-only mesh + BVH...")
    t0 = time.monotonic()
    m_bldg = load_plateau(bldg_only_dir, zone=9, include_bridges=False)
    bvh_bldg = BVHAccelerator.from_building_model(m_bldg)
    print(f"  {m_bldg.triangles.shape[0]} tris in {time.monotonic()-t0:.1f}s")

    print("loading bldg+brid mesh + BVH...")
    t0 = time.monotonic()
    m_both = load_plateau(bldg_brid_dir, zone=9, include_bridges=True)
    bvh_both = BVHAccelerator.from_building_model(m_both)
    print(f"  {m_both.triangles.shape[0]} tris ({m_both.triangles.shape[0]-m_bldg.triangles.shape[0]} new) "
          f"in {time.monotonic()-t0:.1f}s")

    # For each target epoch: compute sat ECEF, run check_los on both
    print("\n" + "=" * 80)
    print("LoS comparison at target epochs (bldg vs bldg+brid)")
    print("=" * 80)

    grand_flips = 0
    rows = []
    for t in target_tows:
        rx, actual_ref_tow, lla = rx_at_tow[t]
        obs_tow, sats = sat_at_tow[t]
        # Filter to sats with available ephemeris
        usable = [s for s in sats if s in avail_prns]
        if not usable:
            print(f"\n  TOW {t:.1f}: no sats with ephemeris; skipping")
            continue
        sat_ecef, _, used = eph.compute(obs_tow, usable)
        sat_ecef = np.asarray(sat_ecef)
        # Filter elevation > 5 deg
        elev = elevations_deg(rx, sat_ecef)
        keep = elev > 5.0
        sat_ecef = sat_ecef[keep]
        used = [u for u, k in zip(used, keep) if k]
        elev = elev[keep]

        if sat_ecef.shape[0] == 0:
            print(f"\n  TOW {t:.1f}: no sats above 5 deg")
            continue

        los_b = np.asarray(bvh_bldg.check_los(rx, sat_ecef), dtype=bool)
        los_bb = np.asarray(bvh_both.check_los(rx, sat_ecef), dtype=bool)
        flipped = los_b & ~los_bb  # was LoS, now NLoS due to brid

        n_flip = int(flipped.sum())
        grand_flips += n_flip
        print(f"\n  TOW {t:.1f} (obs {obs_tow:.1f}, ref {actual_ref_tow:.1f})")
        print(f"    sats above 5deg with eph: {sat_ecef.shape[0]}")
        print(f"    bldg-only LoS:    {los_b.sum():2d} / {sat_ecef.shape[0]}  "
              f"({100*los_b.mean():.1f}%)")
        print(f"    bldg+brid LoS:    {los_bb.sum():2d} / {sat_ecef.shape[0]}  "
              f"({100*los_bb.mean():.1f}%)")
        print(f"    flipped LoS->NLoS by brid: {n_flip}")
        if n_flip > 0:
            for j in np.where(flipped)[0]:
                print(f"      - {used[j]} (elev {elev[j]:.1f} deg)")
        rows.append((t, sat_ecef.shape[0], int(los_b.sum()),
                    int(los_bb.sum()), n_flip))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'TOW':>10} {'nsat':>6} {'LoS_b':>7} {'LoS_bb':>8} {'flips':>7}")
    for t, n, lb, lbb, fl in rows:
        print(f"{t:10.1f} {n:6d} {lb:7d} {lbb:8d} {fl:7d}")
    print(f"\n{'GRAND TOTAL FLIPS':>50} = {grand_flips}")

    if grand_flips == 0:
        print("\n>>> EARLY ABORT: bridge mesh blocks NO satellites that the building "
              "mesh did not already block.\n    Phase 2 retrain is unlikely to help; "
              "abandon this direction.")
    else:
        print(f"\n>>> PROCEED: bridge mesh contributes {grand_flips} new NLoS flips "
              "across {len(rows)} epochs.\n    Phase 2 full 6-run feature extraction + retrain is justified.")


if __name__ == "__main__":
    main()
