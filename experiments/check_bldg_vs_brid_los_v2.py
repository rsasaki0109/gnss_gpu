#!/usr/bin/env python3
"""Phase 2 early-abort check (v2) with geoid correction.

Same as check_bldg_vs_brid_los.py but applies a constant +36.7 m geoid
undulation (GSI Geoid 2011, Tokyo Hamamatsucho area) to PLATEAU mesh
alt values before converting to ECEF.

Background: PLATEAU CityGML (EPSG:6697) carries orthometric heights
(above Tokyo Bay mean sea level / TP), but PlateauLoader._lla_to_ecef
treats the value as ellipsoidal alt. The rover reference.csv is
explicitly ellipsoidal. Without geoid correction the mesh sits ~37 m
below the rover and rays trivially clear the geometry, producing a
spurious 0-flip result.

Strategy: monkey-patch PlateauLoader._lla_to_ecef to add a constant
N before delegating to the original. This is sufficient for a small
trajectory (the geoid varies <0.1 m / km in this area).
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

import gnss_gpu.io.plateau as plateau_mod
from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.plateau import load_plateau
from gnss_gpu.io.rinex import read_rinex_obs


# GSI Geoid 2011, Tokyo Minato-ku Hamamatsucho approx (35.65 N, 139.78 E).
# Geoid is +36.7 m above WGS84 ellipsoid here, so:
#   h_ellipsoidal = h_orthometric (TP) + N
GEOID_N_TOKYO = 36.7

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


def patch_loader_with_geoid_offset(N_metres: float):
    """Monkey-patch PlateauLoader._lla_to_ecef to add +N to alt.

    Returns the original callable so the caller can restore it.
    """
    Loader = plateau_mod.PlateauLoader
    original = Loader._lla_to_ecef  # already the unwrapped staticmethod fn

    @staticmethod
    def _patched(lat, lon, alt):
        return original(lat, lon, alt + N_metres)

    Loader._lla_to_ecef = _patched
    return original


def main():
    run_dir = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data/tokyo/run2")
    rover_obs = run_dir / "rover.obs"
    base_nav = run_dir / "base.nav"
    reference_csv = run_dir / "reference.csv"

    target_tows = [
        177210.0, 177230.0,            # w7
        177270.0, 177290.0,            # w9
        177700.0, 177750.0,            # w23
        177800.0, 177810.0, 177830.0,  # w26-w27
    ]

    print(f"[geoid correction] applying +{GEOID_N_TOKYO:.2f} m to PLATEAU alt "
          "(TP -> ellipsoidal)")
    patch_loader_with_geoid_offset(GEOID_N_TOKYO)

    print("loading reference.csv...")
    ref = load_reference_csv(reference_csv)
    print(f"  {len(ref)} rows; TOW [{ref[0][0]:.1f}, {ref[-1][0]:.1f}]")

    print("\nresolving target rx positions...")
    rx_at_tow = {}
    for t in target_tows:
        rx, actual, lla = find_rx_at_tow(ref, t)
        rx_at_tow[t] = (rx, actual, lla)
        print(f"  TOW {t:.1f} -> nearest {actual:.1f}, "
              f"lat {lla[0]:.5f} lon {lla[1]:.5f} alt {lla[2]:.2f} (ellipsoidal)")

    print("\nreading rover.obs...")
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

    obs_index = {}
    for ep in obs.epochs:
        ep_tow = datetime_to_gps_tow(ep.time)
        obs_index[ep_tow] = ep
    sorted_obs_tows = sorted(obs_index)

    def nearest_obs(target):
        i = int(np.argmin([abs(t - target) for t in sorted_obs_tows]))
        return obs_index[sorted_obs_tows[i]], sorted_obs_tows[i]

    sat_at_tow = {}
    for t in target_tows:
        ep, ep_tow = nearest_obs(t)
        sats = list(ep.satellites)
        sat_at_tow[t] = (ep_tow, sats)

    print("\nlocating PLATEAU bldg+brid cache for tokyo run2 area...")
    cache_dir = Path("/tmp/plateau_segment_cache_tokyo_run2_bldg_brid")
    bldg_files = sorted(str(p) for p in cache_dir.glob("*_bldg_*.gml"))
    brid_paths = sorted(str(p) for p in cache_dir.glob("*_brid_*.gml"))
    if not bldg_files:
        raise SystemExit(f"no bldg GML in {cache_dir}; run fetch_plateau_subset.py first")
    if not brid_paths:
        raise SystemExit(f"no brid GML in {cache_dir}; rerun fetch with --include-bridges")
    print(f"  bldg gmls: {len(bldg_files)}")
    print(f"  brid gmls: {len(brid_paths)}")
    near_bldg = bldg_files

    bldg_only_dir = Path("/tmp/phase2_check_bldg_only_v2")
    bldg_brid_dir = Path("/tmp/phase2_check_bldg_brid_v2")
    build_combined_mesh_dir(bldg_only_dir, near_bldg, [])
    build_combined_mesh_dir(bldg_brid_dir, near_bldg, brid_paths)

    print("loading bldg-only mesh + BVH (geoid corrected)...")
    t0 = time.monotonic()
    m_bldg = load_plateau(bldg_only_dir, zone=9, kinds=("bldg",))
    bvh_bldg = BVHAccelerator.from_building_model(m_bldg)
    print(f"  {m_bldg.triangles.shape[0]} tris in {time.monotonic()-t0:.1f}s")

    print("loading bldg+brid mesh + BVH (geoid corrected)...")
    t0 = time.monotonic()
    m_both = load_plateau(bldg_brid_dir, zone=9, kinds=("bldg", "brid"))
    bvh_both = BVHAccelerator.from_building_model(m_both)
    print(f"  {m_both.triangles.shape[0]} tris ({m_both.triangles.shape[0]-m_bldg.triangles.shape[0]} new)")

    # Sanity check: report the vertical span of mesh tris vs rover
    rx_sample, _, lla_sample = rx_at_tow[177800.0]
    tri_z = m_both.triangles.reshape(-1, 3)[:, 2]
    print(f"\n[sanity] rover ECEF z = {rx_sample[2]:.1f}, mesh tri z range = "
          f"[{tri_z.min():.1f}, {tri_z.max():.1f}] (delta {tri_z.max()-rx_sample[2]:+.1f} m above rover)")

    print("\n" + "=" * 80)
    print("LoS comparison (geoid-corrected) at target epochs (bldg vs bldg+brid)")
    print("=" * 80)

    grand_flips = 0
    bldg_only_blocks = 0
    rows = []
    for t in target_tows:
        rx, actual_ref_tow, lla = rx_at_tow[t]
        obs_tow, sats = sat_at_tow[t]
        usable = [s for s in sats if s in avail_prns]
        if not usable:
            continue
        sat_ecef, _, used = eph.compute(obs_tow, usable)
        sat_ecef = np.asarray(sat_ecef)
        elev = elevations_deg(rx, sat_ecef)
        keep = elev > 5.0
        sat_ecef = sat_ecef[keep]
        used = [u for u, k in zip(used, keep) if k]
        elev = elev[keep]

        if sat_ecef.shape[0] == 0:
            continue

        los_b = np.asarray(bvh_bldg.check_los(rx, sat_ecef), dtype=bool)
        los_bb = np.asarray(bvh_both.check_los(rx, sat_ecef), dtype=bool)
        flipped = los_b & ~los_bb

        n_flip = int(flipped.sum())
        n_block_b = int((~los_b).sum())
        grand_flips += n_flip
        bldg_only_blocks += n_block_b
        print(f"\n  TOW {t:.1f} (obs {obs_tow:.1f}, ref {actual_ref_tow:.1f})")
        print(f"    sats above 5deg: {sat_ecef.shape[0]}")
        print(f"    bldg-only LoS:    {los_b.sum():2d} / {sat_ecef.shape[0]}  "
              f"({100*los_b.mean():.1f}%)  blocks={n_block_b}")
        print(f"    bldg+brid LoS:    {los_bb.sum():2d} / {sat_ecef.shape[0]}  "
              f"({100*los_bb.mean():.1f}%)")
        print(f"    flipped LoS->NLoS by brid: {n_flip}")
        if n_flip > 0:
            for j in np.where(flipped)[0]:
                print(f"      - {used[j]} (elev {elev[j]:.1f} deg)")
        rows.append((t, sat_ecef.shape[0], int(los_b.sum()),
                    int(los_bb.sum()), n_flip))

    print("\n" + "=" * 80)
    print("SUMMARY (with geoid correction)")
    print("=" * 80)
    print(f"{'TOW':>10} {'nsat':>6} {'LoS_b':>7} {'LoS_bb':>8} {'flips':>7}")
    for t, n, lb, lbb, fl in rows:
        print(f"{t:10.1f} {n:6d} {lb:7d} {lbb:8d} {fl:7d}")
    print(f"\n  bldg-only blocks (NLoS) total: {bldg_only_blocks}")
    print(f"  GRAND TOTAL FLIPS (bldg+brid extra NLoS): {grand_flips}")

    if grand_flips == 0:
        print("\n>>> EARLY ABORT (geoid-corrected): bridge mesh blocks no extra sats.")
    else:
        print(f"\n>>> PROCEED: bridge mesh contributes {grand_flips} new NLoS flips.")


if __name__ == "__main__":
    main()
