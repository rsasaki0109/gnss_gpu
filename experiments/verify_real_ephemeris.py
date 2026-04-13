#!/usr/bin/env python3
"""Verify End-to-End pipeline with real broadcast ephemeris + PLATEAU.

Uses UrbanNav Odaiba's navigation file to compute actual GPS satellite
positions via IS-GPS-200, then runs LOS/NLOS classification against
PLATEAU 3D buildings. This replaces the synthetic satellite geometry
used in previous verification scripts.
"""

import csv
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator, ecef_to_lla


def load_trajectory(csv_path, step=500):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    positions, times, weeks = [], [], []
    for i in range(0, len(rows), step):
        r = rows[i]
        positions.append([float(r[" ECEF X (m)"]), float(r[" ECEF Y (m)"]), float(r[" ECEF Z (m)"])])
        times.append(float(r["GPS TOW (s)"]))
        weeks.append(int(float(r[" GPS Week"])))
    return np.array(positions), np.array(times), weeks


def ecef_to_enu(rx, pt):
    lat, lon, _ = ecef_to_lla(*rx)
    s_lat, c_lat = math.sin(lat), math.cos(lat)
    s_lon, c_lon = math.sin(lon), math.cos(lon)
    R = np.array([[-s_lon, c_lon, 0], [-s_lat*c_lon, -s_lat*s_lon, c_lat], [c_lat*c_lon, c_lat*s_lon, s_lat]])
    return R @ (np.asarray(pt) - np.asarray(rx))


def render_epoch(rx, sat_ecef, prn_list, result, epoch_idx, traj_enu, cur_enu, gps_tow):
    fig = plt.figure(figsize=(16, 7), facecolor="#1a1a2e")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 0.7], wspace=0.3,
                          left=0.05, right=0.97, top=0.88, bottom=0.08)
    tc, lc, nc, bg = "#e0e0e0", "#00d4aa", "#ff6b6b", "#16213e"
    lat, lon, _ = ecef_to_lla(*rx)
    n_sat = len(prn_list)

    # Top-down trajectory + rays
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(bg); ax1.set_aspect("equal")
    ax1.plot(traj_enu[:, 0], traj_enu[:, 1], "-", color="#444466", lw=1, alpha=0.5)
    ax1.plot(cur_enu[0], cur_enu[1], "o", color="#fff", ms=8, zorder=10)
    for i in range(n_sat):
        if not result["visible"][i]: continue
        d = ecef_to_enu(rx, sat_ecef[i])[:2]
        norm = np.linalg.norm(d)
        if norm < 1: continue
        d = d / norm * 120
        c = lc if result["is_los"][i] else nc
        s = "-" if result["is_los"][i] else "--"
        ax1.plot([cur_enu[0], cur_enu[0]+d[0]], [cur_enu[1], cur_enu[1]+d[1]], s, color=c, lw=1.2, alpha=0.7)
        label = prn_list[i] if isinstance(prn_list[i], str) else f"G{prn_list[i]:02d}"
        ax1.annotate(label, (cur_enu[0]+d[0], cur_enu[1]+d[1]), color=c, fontsize=6, fontweight="bold", ha="center")
    ax1.set_xlabel("East [m]", color=tc, fontsize=9); ax1.set_ylabel("North [m]", color=tc, fontsize=9)
    ax1.set_title(f"TOW={gps_tow:.0f}s  {math.degrees(lat):.4f}°N {math.degrees(lon):.4f}°E", color=tc, fontsize=10)
    ax1.tick_params(colors=tc, labelsize=7); ax1.grid(True, color="#333355", alpha=0.3)
    for s in ax1.spines.values(): s.set_color("#333355")

    # Skyplot
    ax2 = fig.add_subplot(gs[1], projection="polar")
    ax2.set_facecolor(bg); ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
    ax2.set_ylim(0, 90); ax2.set_yticks([0,30,60,90])
    ax2.set_yticklabels(["90°","60°","30°","0°"], color=tc, fontsize=7)
    ax2.tick_params(colors=tc, labelsize=7); ax2.grid(True, color="#333355", alpha=0.4)
    for i in range(n_sat):
        if not result["visible"][i]: continue
        az = result["azimuths"][i]; el = np.degrees(result["elevations"][i])
        c = lc if result["is_los"][i] else nc
        m = "o" if result["is_los"][i] else "s"
        ax2.scatter(az, 90-el, c=c, s=80, marker=m, zorder=5, edgecolors="white", linewidths=0.5)
        label = prn_list[i] if isinstance(prn_list[i], str) else f"G{prn_list[i]:02d}"
        ax2.annotate(label, (az, 90-el), fontsize=6, color=c, ha="center", va="bottom",
                     xytext=(0,6), textcoords="offset points", fontweight="bold")
    ax2.set_title("Skyplot (real ephemeris)", color=tc, fontsize=10, pad=12)
    handles = [mpatches.Patch(color=lc, label=f"LOS ({result['n_los']})"),
               mpatches.Patch(color=nc, label=f"NLOS ({result['n_nlos']})")]
    ax2.legend(handles=handles, loc="lower left", fontsize=7, facecolor=bg, edgecolor="#333355", labelcolor=tc)

    # Stats table
    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor(bg); ax3.axis("off")
    rows = []
    for i in range(n_sat):
        if not result["visible"][i]: continue
        st = "LOS" if result["is_los"][i] else "NLOS"
        el = f"{np.degrees(result['elevations'][i]):.0f}°"
        mp = f"{result['excess_delays'][i]:.1f}m" if result["excess_delays"][i] > 0.1 else "-"
        label = prn_list[i] if isinstance(prn_list[i], str) else f"G{prn_list[i]:02d}"
        rows.append([label, st, el, mp])
    if rows:
        t = ax3.table(cellText=rows, colLabels=["SV","St","El","MP"], cellLoc="center", loc="center")
        t.auto_set_font_size(False); t.set_fontsize(8)
        for k, cell in t.get_celld().items():
            cell.set_facecolor(bg); cell.set_edgecolor("#333355"); cell.set_text_props(color=tc)
            if k[0]==0: cell.set_facecolor("#0f3460"); cell.set_text_props(color="white", fontweight="bold")
            elif len(rows)>k[0]-1 and rows[k[0]-1][1]=="NLOS": cell.set_facecolor("#2d1f1f")
    ax3.set_title("Real Satellites", color=tc, fontsize=10)
    fig.suptitle("LOS/NLOS — Real Ephemeris + PLATEAU Odaiba", color="white", fontsize=13, fontweight="bold")
    return fig


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "los_nlos_verification")
    os.makedirs(out_dir, exist_ok=True)

    # Load ephemeris
    nav_path = "experiments/data/urbannav/Odaiba/base.nav"
    print(f"Loading ephemeris: {nav_path}")
    nav_messages = read_nav_rinex_multi(nav_path)
    eph = Ephemeris(nav_messages)
    gps_prns = [p for p in eph.available_prns if (isinstance(p, str) and p.startswith("G")) or isinstance(p, int)]
    print(f"  GPS PRNs: {len(gps_prns)} ({gps_prns[:5]}...)")

    # Load PLATEAU
    print("Loading PLATEAU Odaiba...")
    loader = PlateauLoader(zone=9)
    building = loader.load_directory("experiments/data/plateau_odaiba")
    bvh = BVHAccelerator.from_building_model(building)
    print(f"  {len(building.triangles)} triangles, {bvh.n_nodes} BVH nodes")

    # Load trajectory
    print("Loading UrbanNav Odaiba trajectory...")
    positions, times, weeks = load_trajectory(
        "experiments/data/urbannav/Odaiba/reference.csv", step=500)
    print(f"  {len(positions)} sampled epochs, TOW range: {times[0]:.0f}-{times[-1]:.0f}")

    traj_enu = np.array([ecef_to_enu(positions[0], p) for p in positions])

    # Urban signal simulator with BVH
    usim = UrbanSignalSimulator(building_model=bvh, noise_floor_db=-35)

    n_epochs = min(15, len(positions))
    indices = np.linspace(0, len(positions)-1, n_epochs, dtype=int)
    frames = []

    for fi, ei in enumerate(indices):
        rx = positions[ei]
        gps_tow = times[ei]

        # Compute REAL satellite positions from broadcast ephemeris
        sat_ecef, sat_clk, used_prns = eph.compute(gps_tow, prn_list=gps_prns)
        if len(used_prns) == 0:
            print(f"  [{fi+1}/{n_epochs}] No satellites — skip")
            continue

        # Convert PRN labels to ints for signal sim
        prn_ints = []
        for p in used_prns:
            if isinstance(p, str) and p.startswith("G"):
                prn_ints.append(int(p[1:]))
            elif isinstance(p, int):
                prn_ints.append(p)
            else:
                prn_ints.append(0)

        print(f"  [{fi+1}/{n_epochs}] TOW={gps_tow:.0f} sats={len(used_prns)}", end="")
        t0 = time.time()
        result = usim.compute_epoch(
            rx_ecef=rx, sat_ecef=sat_ecef, prn_list=prn_ints)
        dt = time.time() - t0
        print(f"  LOS={result['n_los']} NLOS={result['n_nlos']} MP={result['n_multipath']} [{dt*1000:.0f}ms]")

        cur_enu = ecef_to_enu(positions[0], rx)
        fig = render_epoch(rx, sat_ecef, used_prns, result, ei, traj_enu, cur_enu, gps_tow)
        png = os.path.join(out_dir, f"real_eph_{fi:03d}.png")
        fig.savefig(png, dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        frames.append(Image.open(png))

    if frames:
        gif_path = os.path.join(out_dir, "los_nlos_real_ephemeris.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=800, loop=0, optimize=True)
        print(f"\nGIF: {gif_path}")


if __name__ == "__main__":
    main()
