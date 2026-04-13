#!/usr/bin/env python3
"""E2E trajectory evaluation: Open Sky vs Odaiba vs Shinjuku.

Runs signal generation → acquisition → WLS positioning across a full
UrbanNav trajectory for each area, then produces CDF curves of
positioning error.
"""

import csv
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.signal_sim import SignalSimulator
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator, ecef_to_lla
from gnss_gpu.acquisition import Acquisition
from gnss_gpu._gnss_gpu import wls_position

C_LIGHT = 299792458.0
CA_CHIP_RATE = 1.023e6


def load_trajectory(csv_path, step=100):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    positions, times = [], []
    for i in range(0, len(rows), step):
        r = rows[i]
        positions.append([float(r[" ECEF X (m)"]),
                          float(r[" ECEF Y (m)"]),
                          float(r[" ECEF Z (m)"])])
        times.append(float(r["GPS TOW (s)"]))
    return np.array(positions), np.array(times)


def visible_filter(rx, sat_ecef, el_mask_rad=0.175):
    """Return indices of satellites above elevation mask."""
    lat, lon, _ = ecef_to_lla(*rx)
    s_lat, c_lat = math.sin(lat), math.cos(lat)
    s_lon, c_lon = math.sin(lon), math.cos(lon)
    R = np.array([[-s_lon, c_lon, 0],
                  [-s_lat * c_lon, -s_lat * s_lon, c_lat],
                  [c_lat * c_lon, c_lat * s_lon, s_lat]])
    diff = sat_ecef - rx
    enu = (R @ diff.T).T
    el = np.arctan2(enu[:, 2], np.sqrt(enu[:, 0]**2 + enu[:, 1]**2))
    return np.where(el > el_mask_rad)[0]


def run_epoch(rx_true, sat_ecef, prn_ints, usim, acq, fs):
    """Run one E2E epoch, return position error."""
    n_sat = len(prn_ints)
    ranges_true = np.linalg.norm(sat_ecef - rx_true, axis=1)

    # Generate signal
    if usim is not None:
        result = usim.compute_epoch(
            rx_ecef=rx_true, sat_ecef=sat_ecef, prn_list=prn_ints)
        iq = result["iq"]
        excess_delays = result["excess_delays"]
        n_los = result["n_los"]
        n_nlos = result["n_nlos"]
    else:
        # Open sky
        channels = []
        for i in range(n_sat):
            cp = (ranges_true[i] / C_LIGHT * CA_CHIP_RATE) % 1023.0
            channels.append({
                "prn": int(prn_ints[i]), "code_phase": float(cp),
                "carrier_phase": 0.0, "doppler_hz": 0.0,
                "amplitude": 1.0, "nav_bit": 1,
            })
        sim = SignalSimulator(sampling_freq=fs, noise_floor_db=-30)
        iq = sim.generate_epoch(channels)
        excess_delays = np.zeros(n_sat)
        n_los = n_sat
        n_nlos = 0

    signal_i = iq[0::2].copy()
    acq_results = acq.acquire(signal_i, prn_list=prn_ints)

    acquired_idx = []
    pseudoranges = []
    sat_acquired = []
    for i, r in enumerate(acq_results):
        if not r["acquired"]:
            continue
        pr = ranges_true[i]
        if excess_delays[i] > 0.1:
            pr += excess_delays[i]
        acquired_idx.append(i)
        pseudoranges.append(pr)
        sat_acquired.append(sat_ecef[i])

    n_acq = len(acquired_idx)
    if n_acq < 4:
        return float("nan"), n_acq, n_los, n_nlos

    try:
        res, _ = wls_position(
            np.array(sat_acquired).flatten(),
            np.array(pseudoranges), np.ones(n_acq))
        err = float(np.linalg.norm(res[:3] - rx_true))
    except Exception:
        err = float("nan")

    return err, n_acq, n_los, n_nlos


def run_trajectory(name, positions, times, eph, gps_prns,
                   building_model=None, fs=2.6e6, max_epochs=50):
    """Run E2E over a trajectory."""
    if building_model is not None:
        usim = UrbanSignalSimulator(
            building_model=building_model, sampling_freq=fs,
            noise_floor_db=-30, nlos_attenuation_db=8.0, fresnel_coeff=0.5)
    else:
        usim = None

    acq = Acquisition(sampling_freq=fs, intermediate_freq=0, threshold=2.0)

    n_epochs = min(max_epochs, len(positions))
    indices = np.linspace(0, len(positions) - 1, n_epochs, dtype=int)

    errors = []
    stats = {"n_acq": [], "n_los": [], "n_nlos": []}

    for fi, ei in enumerate(indices):
        rx = positions[ei]
        tow = times[ei]

        sat_ecef, _, used_prns = eph.compute(tow, prn_list=gps_prns)
        prn_ints = []
        for p in used_prns:
            if isinstance(p, str) and p.startswith("G"):
                prn_ints.append(int(p[1:]))
            else:
                prn_ints.append(int(p))

        vis = visible_filter(rx, sat_ecef)
        if len(vis) < 4:
            continue
        sat_vis = sat_ecef[vis]
        prn_vis = [prn_ints[j] for j in vis]

        err, n_acq, n_los, n_nlos = run_epoch(rx, sat_vis, prn_vis, usim, acq, fs)
        errors.append(err)
        stats["n_acq"].append(n_acq)
        stats["n_los"].append(n_los)
        stats["n_nlos"].append(n_nlos)

        if (fi + 1) % 10 == 0 or fi == 0:
            err_str = f"{err:.1f}m" if not math.isnan(err) else "N/A"
            print(f"  [{name}] {fi+1}/{n_epochs} err={err_str} acq={n_acq} LOS={n_los} NLOS={n_nlos}")

    return np.array(errors), stats


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "e2e_positioning")
    os.makedirs(out_dir, exist_ok=True)

    # Load ephemeris (Odaiba nav file, valid for both areas at same time)
    nav_path = "experiments/data/urbannav/Odaiba/base.nav"
    print(f"Loading ephemeris: {nav_path}")
    nav_msgs = read_nav_rinex_multi(nav_path)
    eph = Ephemeris(nav_msgs)
    gps_prns = [p for p in eph.available_prns
                if (isinstance(p, str) and p.startswith("G")) or isinstance(p, int)]

    loader = PlateauLoader(zone=9)
    results = {}

    # --- 1. Open Sky (Odaiba trajectory, no buildings) ---
    print("\n=== Open Sky (Odaiba trajectory, no buildings) ===")
    pos_od, times_od = load_trajectory(
        "experiments/data/urbannav/Odaiba/reference.csv", step=500)
    errors, stats = run_trajectory("OpenSky", pos_od, times_od, eph, gps_prns,
                                   building_model=None, max_epochs=30)
    results["Open Sky"] = (errors, stats)

    # --- 2. Odaiba (with PLATEAU buildings) ---
    print("\n=== Odaiba (PLATEAU 249K triangles) ===")
    bldg_od = loader.load_directory("experiments/data/plateau_odaiba")
    bvh_od = BVHAccelerator.from_building_model(bldg_od)
    errors, stats = run_trajectory("Odaiba", pos_od, times_od, eph, gps_prns,
                                   building_model=bvh_od, max_epochs=30)
    results["Odaiba"] = (errors, stats)

    # --- 3. Shinjuku (with PLATEAU buildings) ---
    print("\n=== Shinjuku (PLATEAU 1.23M triangles) ===")
    pos_sj, times_sj = load_trajectory(
        "experiments/data/urbannav/Shinjuku/reference.csv", step=500)
    # Shinjuku nav file
    nav_sj_path = "experiments/data/urbannav/Shinjuku/base.nav"
    if os.path.exists(nav_sj_path):
        nav_sj = read_nav_rinex_multi(nav_sj_path)
        eph_sj = Ephemeris(nav_sj)
        gps_prns_sj = [p for p in eph_sj.available_prns
                       if (isinstance(p, str) and p.startswith("G")) or isinstance(p, int)]
    else:
        eph_sj = eph
        gps_prns_sj = gps_prns

    bldg_sj = loader.load_directory("experiments/data/plateau_shinjuku")
    bvh_sj = BVHAccelerator.from_building_model(bldg_sj)
    errors, stats = run_trajectory("Shinjuku", pos_sj, times_sj, eph_sj, gps_prns_sj,
                                   building_model=bvh_sj, max_epochs=30)
    results["Shinjuku"] = (errors, stats)

    # --- CDF plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#1a1a2e")

    colors = {"Open Sky": "#00d4aa", "Odaiba": "#ffd93d", "Shinjuku": "#ff6b6b"}

    # CDF
    ax = axes[0]
    ax.set_facecolor("#16213e")
    for label, (errs, _) in results.items():
        valid = errs[~np.isnan(errs)]
        if len(valid) == 0:
            continue
        sorted_e = np.sort(valid)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e) * 100
        ax.plot(sorted_e, cdf, color=colors[label], linewidth=2, label=label)

    ax.axhline(50, color="#555577", linestyle=":", alpha=0.5)
    ax.axhline(95, color="#555577", linestyle=":", alpha=0.5)
    ax.text(1, 51, "50%", color="#888", fontsize=8)
    ax.text(1, 96, "95%", color="#888", fontsize=8)
    ax.set_xlabel("Position Error [m]", color="#e0e0e0", fontsize=11)
    ax.set_ylabel("CDF [%]", color="#e0e0e0", fontsize=11)
    ax.set_title("Positioning Error CDF", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#e0e0e0", labelsize=9)
    ax.legend(fontsize=10, facecolor="#16213e", edgecolor="#333355", labelcolor="#e0e0e0")
    ax.grid(True, color="#333355", alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    for s in ax.spines.values():
        s.set_color("#333355")

    # Summary stats
    ax = axes[1]
    ax.set_facecolor("#16213e")
    ax.axis("off")

    rows = []
    for label, (errs, stats) in results.items():
        valid = errs[~np.isnan(errs)]
        if len(valid) == 0:
            rows.append([label, "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue
        p50 = np.percentile(valid, 50)
        p95 = np.percentile(valid, 95)
        rms = np.sqrt(np.mean(valid ** 2))
        avg_nlos = np.mean(stats["n_nlos"])
        avg_acq = np.mean(stats["n_acq"])
        rows.append([label, f"{rms:.1f}", f"{p50:.1f}", f"{p95:.1f}",
                     f"{avg_nlos:.1f}", f"{avg_acq:.0f}"])

    table = ax.table(
        cellText=rows,
        colLabels=["Scenario", "RMS [m]", "P50 [m]", "P95 [m]", "Avg NLOS", "Avg Acq"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for key, cell in table.get_celld().items():
        cell.set_facecolor("#16213e")
        cell.set_edgecolor("#333355")
        cell.set_text_props(color="#e0e0e0")
        if key[0] == 0:
            cell.set_facecolor("#0f3460")
            cell.set_text_props(color="white", fontweight="bold")
    table.scale(1, 1.8)
    ax.set_title("Summary Statistics", color="white", fontsize=13, fontweight="bold")

    fig.suptitle("End-to-End: GPU Signal Sim → Acquisition → WLS Positioning\n"
                 "Open Sky vs Urban (PLATEAU 3D Buildings)",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "e2e_trajectory_cdf.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Scenario':<15} {'RMS':>8} {'P50':>8} {'P95':>8} {'NLOS':>6} {'Acq':>5}")
    print("-" * 55)
    for label, (errs, stats) in results.items():
        valid = errs[~np.isnan(errs)]
        if len(valid) == 0:
            continue
        print(f"{label:<15} {np.sqrt(np.mean(valid**2)):>7.1f}m "
              f"{np.percentile(valid, 50):>7.1f}m "
              f"{np.percentile(valid, 95):>7.1f}m "
              f"{np.mean(stats['n_nlos']):>5.1f} "
              f"{np.mean(stats['n_acq']):>5.0f}")


if __name__ == "__main__":
    main()
