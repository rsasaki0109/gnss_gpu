#!/usr/bin/env python3
"""Dense-sampled Phase 2 check (v3) with geoid correction.

Same geoid-corrected loader as v2 but iterates every second across the
full hidden-high / false-high windows instead of 9 hand-picked epochs.

Window definitions match `internal_docs/product_deliverable/README.md`
section 5 (Tokyo run2 false-high + hidden-high clusters):
  - w7  : TOW 177200..177240
  - w9  : TOW 177260..177300
  - w23 : TOW 177680..177760
  - w26-w27 : TOW 177790..177840

Step = 1 sec ⇒ ~210 epochs, sufficient to estimate a real per-second
flip rate from bridge geometry.
"""

from __future__ import annotations

import csv
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.plateau import load_plateau
from gnss_gpu.io.rinex import read_rinex_obs


# Use pyproj+EGM96 (~0.5 m off the official GSI Geoid 2011 in Japan,
# sufficient for LoS work).  The script will hard-fail if pyproj or
# its EGM96 grid is missing -- this is intentional, see the loader's
# module docstring on EPSG:6697.  If you need to run on a host
# without pyproj, set this to a constant float (e.g. 36.7 for Tokyo).
GEOID_CORRECTION = "egm96"

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_WEEK_SEC = 604800.0

WINDOWS = [
    ("w7",     177200, 177240),
    ("w9",     177260, 177300),
    ("w23",    177680, 177760),
    ("w26-27", 177790, 177840),
]


def datetime_to_gps_tow(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = (dt - GPS_EPOCH).total_seconds()
    week = int(delta // GPS_WEEK_SEC)
    return delta - week * GPS_WEEK_SEC


def lla_to_ecef(lat_deg, lon_deg, alt):
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    sinl = np.sin(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sinl * sinl)
    return np.array([(N + alt) * np.cos(lat) * np.cos(lon),
                     (N + alt) * np.cos(lat) * np.sin(lon),
                     (N * (1.0 - WGS84_E2) + alt) * sinl])


def ecef_to_enu_basis(ecef):
    x, y, z = ecef
    p = np.sqrt(x * x + y * y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(5):
        sl = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
        lat = np.arctan2(z + WGS84_E2 * N * sl, p)
    sl, cl = np.sin(lat), np.cos(lat); so, co = np.sin(lon), np.cos(lon)
    return (np.array([-so, co, 0.0]),
            np.array([-sl * co, -sl * so, cl]),
            np.array([cl * co, cl * so, sl]))


def elevations_deg(rx, sat_ecef):
    e, n, u = ecef_to_enu_basis(rx)
    rel = sat_ecef - rx
    return np.degrees(np.arctan2(rel @ u, np.sqrt((rel @ e) ** 2 + (rel @ n) ** 2)))


def load_reference_csv(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["GPS TOW (s)"]),
                         float(r["Latitude (deg)"]),
                         float(r["Longitude (deg)"]),
                         float(r["Ellipsoid Height (m)"])))
    return rows


def build_combined_mesh_dir(out_dir, bldg_files, brid_files):
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

    print(f"[geoid correction] using PlateauLoader(geoid_correction={GEOID_CORRECTION!r})")

    target_tows = []
    for name, lo, hi in WINDOWS:
        target_tows.extend(range(lo, hi + 1))
    print(f"sampling {len(target_tows)} TOWs across {len(WINDOWS)} windows")

    print("\nloading reference.csv...")
    ref = load_reference_csv(reference_csv)
    ref_by_tow = {}
    for tow, lat, lon, alt in ref:
        # 5 Hz -> snap each integer-sec target to the closest sub-sec row
        key = round(tow * 5) / 5
        ref_by_tow[key] = (lat, lon, alt)
    print(f"  {len(ref)} reference rows")

    print("\nreading rover.obs...")
    t0 = time.monotonic()
    obs = read_rinex_obs(rover_obs)
    print(f"  {len(obs.epochs)} obs epochs in {time.monotonic()-t0:.1f}s")

    print("\nreading base.nav...")
    t0 = time.monotonic()
    nav = read_nav_rinex_multi(str(base_nav), systems=("G", "E", "J", "C", "R"))
    eph = Ephemeris(nav)
    avail_prns = eph.available_prns
    print(f"  ephemeris: {len(avail_prns)} PRNs in {time.monotonic()-t0:.1f}s")

    obs_index = {datetime_to_gps_tow(ep.time): ep for ep in obs.epochs}
    sorted_obs_tows = sorted(obs_index)
    obs_tows_arr = np.asarray(sorted_obs_tows)

    def nearest_obs(target):
        i = int(np.argmin(np.abs(obs_tows_arr - target)))
        t = sorted_obs_tows[i]
        return obs_index[t], t

    cache_dir = Path("/tmp/plateau_segment_cache_tokyo_run2_bldg_brid")
    bldg_files = sorted(str(p) for p in cache_dir.glob("*_bldg_*.gml"))
    brid_paths = sorted(str(p) for p in cache_dir.glob("*_brid_*.gml"))
    print(f"\nbldg gml: {len(bldg_files)}, brid gml: {len(brid_paths)}")

    bldg_only_dir = Path("/tmp/phase2_check_bldg_only_v3")
    bldg_brid_dir = Path("/tmp/phase2_check_bldg_brid_v3")
    build_combined_mesh_dir(bldg_only_dir, bldg_files, [])
    build_combined_mesh_dir(bldg_brid_dir, bldg_files, brid_paths)

    print("loading bldg-only mesh...")
    t0 = time.monotonic()
    m_bldg = load_plateau(bldg_only_dir, zone=9, kinds=("bldg",),
                          geoid_correction=GEOID_CORRECTION)
    bvh_bldg = BVHAccelerator.from_building_model(m_bldg)
    print(f"  {m_bldg.triangles.shape[0]} tris in {time.monotonic()-t0:.1f}s")

    print("loading bldg+brid mesh...")
    t0 = time.monotonic()
    m_both = load_plateau(bldg_brid_dir, zone=9, kinds=("bldg", "brid"),
                          geoid_correction=GEOID_CORRECTION)
    bvh_both = BVHAccelerator.from_building_model(m_both)
    print(f"  {m_both.triangles.shape[0]} tris in {time.monotonic()-t0:.1f}s")

    print("\n" + "=" * 80)
    print("DENSE LoS comparison (per-second across windows)")
    print("=" * 80)

    per_window_stats = {}
    flip_prn_counter = Counter()
    flip_epochs = []  # (window, tow, prn, elev)

    for win_name, lo, hi in WINDOWS:
        n_epochs_used = 0
        n_sat_total = 0
        n_block_b_total = 0
        n_block_bb_total = 0
        n_flip_total = 0
        for tow in range(lo, hi + 1):
            # rover position at TOW (5 Hz reference)
            key = round(tow * 5) / 5
            if key not in ref_by_tow:
                continue
            lat, lon, alt = ref_by_tow[key]
            rx = lla_to_ecef(lat, lon, alt)

            ep, ep_tow = nearest_obs(tow)
            if abs(ep_tow - tow) > 0.5:
                continue
            sats = [s for s in ep.satellites if s in avail_prns]
            if not sats:
                continue
            sat_ecef, _, used = eph.compute(ep_tow, sats)
            sat_ecef = np.asarray(sat_ecef)
            elev = elevations_deg(rx, sat_ecef)
            keep = elev > 5.0
            sat_ecef = sat_ecef[keep]; used = [u for u, k in zip(used, keep) if k]
            elev = elev[keep]
            if sat_ecef.shape[0] == 0:
                continue

            los_b = np.asarray(bvh_bldg.check_los(rx, sat_ecef), dtype=bool)
            los_bb = np.asarray(bvh_both.check_los(rx, sat_ecef), dtype=bool)
            flipped = los_b & ~los_bb

            n_epochs_used += 1
            n_sat_total += sat_ecef.shape[0]
            n_block_b_total += int((~los_b).sum())
            n_block_bb_total += int((~los_bb).sum())
            n_flip_total += int(flipped.sum())
            for j in np.where(flipped)[0]:
                flip_prn_counter[used[j]] += 1
                flip_epochs.append((win_name, tow, used[j], float(elev[j])))

        per_window_stats[win_name] = (n_epochs_used, n_sat_total,
                                      n_block_b_total, n_block_bb_total,
                                      n_flip_total)
        rate = n_flip_total / n_epochs_used if n_epochs_used else 0.0
        print(f"  {win_name:>7s} epochs={n_epochs_used:4d}  sat-slots={n_sat_total:5d}  "
              f"bldg_NLoS={n_block_b_total:4d}  bldg+brid_NLoS={n_block_bb_total:4d}  "
              f"flips={n_flip_total:3d}  flip/epoch={rate:.3f}")

    grand_epochs = sum(s[0] for s in per_window_stats.values())
    grand_slots = sum(s[1] for s in per_window_stats.values())
    grand_block_b = sum(s[2] for s in per_window_stats.values())
    grand_block_bb = sum(s[3] for s in per_window_stats.values())
    grand_flips = sum(s[4] for s in per_window_stats.values())
    print("\n" + "=" * 80)
    print("AGGREGATE")
    print("=" * 80)
    print(f"epochs analysed:     {grand_epochs}")
    print(f"sat-slots (elev>5):  {grand_slots}")
    print(f"bldg-only NLoS:      {grand_block_b}  ({100*grand_block_b/grand_slots:.1f}% of slots)")
    print(f"bldg+brid NLoS:      {grand_block_bb}  ({100*grand_block_bb/grand_slots:.1f}% of slots)")
    print(f"flips (LoS->NLoS):   {grand_flips}  "
          f"({100*grand_flips/grand_slots:.2f}% of slots, "
          f"{100*grand_flips/max(grand_block_bb,1):.1f}% of bldg+brid blocks)")
    print(f"flips per epoch:     {grand_flips/max(grand_epochs,1):.3f}")
    print("\nflip PRN distribution:")
    for prn, cnt in flip_prn_counter.most_common():
        print(f"  {prn}: {cnt}")
    print(f"\nfirst 20 flip events (window, TOW, PRN, elev_deg):")
    for ev in flip_epochs[:20]:
        print(f"  {ev[0]:>7s}  TOW {ev[1]}  {ev[2]}  elev {ev[3]:.1f}°")

    # Decision rule
    print("\n" + "=" * 80)
    flip_rate = grand_flips / max(grand_epochs, 1)
    print(f"DECISION (per-epoch flip rate = {flip_rate:.3f})")
    if flip_rate < 0.05:
        print(">>> ABORT: brid mesh contributes <0.05 flips/epoch. The 8.13pp")
        print("    Tokyo run2 residual is not explained by bridge occlusion.")
    elif flip_rate < 0.5:
        print(">>> WEAK: 0.05..0.5 flips/epoch. Plausibly worth trying retrain")
        print("    if compute is cheap, but expected route-MAE gain is small.")
    else:
        print(">>> PROCEED: >0.5 flips/epoch. Bridge geometry is a meaningful")
        print("    occluder; full Phase 2 retrain is justified.")


if __name__ == "__main__":
    main()
