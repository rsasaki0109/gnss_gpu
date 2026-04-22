#!/usr/bin/env python3
"""Verify LOS/NLOS geometry with PLATEAU Shinjuku + UrbanNav + KML export.

Shinjuku has dense high-rise buildings — expects significantly more
NLOS satellites than Odaiba for the same synthetic sky geometry.
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

from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator, ecef_to_lla
from gnss_gpu.viz.kml_export import export_kml


def load_trajectory(csv_path, step=200):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    positions, times = [], []
    for i in range(0, len(rows), step):
        r = rows[i]
        positions.append([float(r[" ECEF X (m)"]), float(r[" ECEF Y (m)"]), float(r[" ECEF Z (m)"])])
        times.append(float(r["GPS TOW (s)"]))
    return np.array(positions), np.array(times)


def generate_sats(rx_ecef, n_sat=10, time_offset=0.0):
    """Generate synthetic satellite positions for visualization."""
    lat, lon, _ = ecef_to_lla(*rx_ecef)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    sat_ecef = np.zeros((n_sat, 3))
    for i in range(n_sat):
        el_deg = 10 + 70 * (i / max(n_sat - 1, 1))
        az_deg = (i * 36 + time_offset * 2) % 360
        el, az = math.radians(el_deg), math.radians(az_deg)
        r = 26600e3
        e = math.sin(az) * math.cos(el)
        n = math.cos(az) * math.cos(el)
        u = math.sin(el)
        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u
        sat_ecef[i] = rx_ecef + r * np.array([dx, dy, dz])
    return sat_ecef


def ecef_to_enu(rx, pt):
    lat, lon, _ = ecef_to_lla(*rx)
    s_lat, c_lat = math.sin(lat), math.cos(lat)
    s_lon, c_lon = math.sin(lon), math.cos(lon)
    R = np.array([[-s_lon, c_lon, 0], [-s_lat*c_lon, -s_lat*s_lon, c_lat], [c_lat*c_lon, c_lat*s_lon, s_lat]])
    return R @ (np.asarray(pt) - np.asarray(rx))


def render_epoch(rx, sats, prns, result, idx, traj_enu, cur_enu, area_name):
    fig = plt.figure(figsize=(16, 7), facecolor="#1a1a2e")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 0.7], wspace=0.3,
                          left=0.05, right=0.97, top=0.88, bottom=0.08)
    tc, lc, nc, bg = "#e0e0e0", "#00d4aa", "#ff6b6b", "#16213e"
    lat, lon, _ = ecef_to_lla(*rx)
    n_sat = len(prns)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(bg); ax1.set_aspect("equal")
    ax1.plot(traj_enu[:, 0], traj_enu[:, 1], "-", color="#444466", lw=1, alpha=0.5)
    ax1.plot(cur_enu[0], cur_enu[1], "o", color="#fff", ms=8, zorder=10)
    for i in range(n_sat):
        if not result["visible"][i]: continue
        d = ecef_to_enu(rx, sats[i])[:2]
        d = d / np.linalg.norm(d) * 120
        c = lc if result["is_los"][i] else nc
        s = "-" if result["is_los"][i] else "--"
        ax1.plot([cur_enu[0], cur_enu[0]+d[0]], [cur_enu[1], cur_enu[1]+d[1]], s, color=c, lw=1.2, alpha=0.7)
        ax1.annotate(f"{prns[i]}", (cur_enu[0]+d[0], cur_enu[1]+d[1]), color=c, fontsize=7, fontweight="bold", ha="center")
    ax1.set_xlabel("East [m]", color=tc, fontsize=9); ax1.set_ylabel("North [m]", color=tc, fontsize=9)
    ax1.set_title(f"Epoch {idx} — {math.degrees(lat):.4f}°N {math.degrees(lon):.4f}°E", color=tc, fontsize=10)
    ax1.tick_params(colors=tc, labelsize=7); ax1.grid(True, color="#333355", alpha=0.3)
    for s in ax1.spines.values(): s.set_color("#333355")

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
        ax2.annotate(f"{prns[i]}", (az, 90-el), fontsize=7, color=c, ha="center", va="bottom",
                     xytext=(0,6), textcoords="offset points", fontweight="bold")
    ax2.set_title("Skyplot", color=tc, fontsize=10, pad=12)
    handles = [mpatches.Patch(color=lc, label=f"LOS ({result['n_los']})"),
               mpatches.Patch(color=nc, label=f"NLOS ({result['n_nlos']})")]
    ax2.legend(handles=handles, loc="lower left", fontsize=7, facecolor=bg, edgecolor="#333355", labelcolor=tc)

    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor(bg); ax3.axis("off")
    rows = []
    for i in range(n_sat):
        if not result["visible"][i]: continue
        st = "LOS" if result["is_los"][i] else "NLOS"
        el = f"{np.degrees(result['elevations'][i]):.0f}°"
        mp = f"{result['excess_delays'][i]:.1f}m" if result["excess_delays"][i] > 0.1 else "-"
        rows.append([str(prns[i]), st, el, mp])
    if rows:
        t = ax3.table(cellText=rows, colLabels=["PRN","St","El","MP"], cellLoc="center", loc="center")
        t.auto_set_font_size(False); t.set_fontsize(8)
        for k, cell in t.get_celld().items():
            cell.set_facecolor(bg); cell.set_edgecolor("#333355"); cell.set_text_props(color=tc)
            if k[0]==0: cell.set_facecolor("#0f3460"); cell.set_text_props(color="white", fontweight="bold")
            elif len(rows)>k[0]-1 and rows[k[0]-1][1]=="NLOS": cell.set_facecolor("#2d1f1f")
    ax3.set_title("Detail", color=tc, fontsize=10)
    fig.suptitle(f"LOS/NLOS — PLATEAU {area_name} + UrbanNav", color="white", fontsize=13, fontweight="bold")
    return fig


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "los_nlos_verification")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading PLATEAU Shinjuku...")
    t0 = time.time()
    loader = PlateauLoader(zone=9)
    building = loader.load_directory("experiments/data/plateau_shinjuku")
    print(f"  {len(building.triangles)} triangles [{time.time()-t0:.1f}s]")

    print("Building BVH...")
    t0 = time.time()
    bvh = BVHAccelerator.from_building_model(building)
    print(f"  BVH: {bvh.n_nodes} nodes [{time.time()-t0:.1f}s]")

    print("Loading UrbanNav Shinjuku...")
    positions, times = load_trajectory("experiments/data/urbannav/Shinjuku/reference.csv", step=200)
    print(f"  {len(positions)} epochs")

    traj_enu = np.array([ecef_to_enu(positions[0], p) for p in positions])

    n_sat = 10
    prn_list = list(range(1, n_sat + 1))
    usim = UrbanSignalSimulator(building_model=bvh, noise_floor_db=-35)

    n_frames = min(20, len(positions))
    indices = np.linspace(0, len(positions)-1, n_frames, dtype=int)

    frames = []
    kml_results = []
    nlos_counts = []

    for fi, ei in enumerate(indices):
        rx = positions[ei]
        t = times[ei] - times[0]
        sats = generate_sats(rx, n_sat, time_offset=t)

        print(f"  [{fi+1}/{n_frames}] t={t:.0f}s", end="")
        t0 = time.time()
        result = usim.compute_epoch(rx_ecef=rx, sat_ecef=sats, prn_list=prn_list)
        dt = time.time() - t0
        print(f"  LOS={result['n_los']} NLOS={result['n_nlos']} [{dt*1000:.0f}ms]")
        nlos_counts.append(result['n_nlos'])

        cur_enu = ecef_to_enu(positions[0], rx)
        fig = render_epoch(rx, sats, prn_list, result, ei, traj_enu, cur_enu, "Shinjuku")
        png = os.path.join(out_dir, f"shinjuku_{fi:03d}.png")
        fig.savefig(png, dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        frames.append(Image.open(png))

        kml_results.append({
            "rx_ecef": rx, "sat_ecef": sats, "prn_list": prn_list,
            "is_los": result["is_los"], "visible": result["visible"],
            "elevations": result["elevations"],
            "excess_delays": result["excess_delays"],
            "epoch_label": f"t={t:.0f}s",
        })

    # Save GIF
    gif_path = os.path.join(out_dir, "los_nlos_shinjuku.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=600, loop=0, optimize=True)
    print(f"\nGIF: {gif_path}")

    # Save KML
    kml_path = os.path.join(out_dir, "los_nlos_shinjuku.kml")
    export_kml(kml_results, kml_path, name="LOS/NLOS Shinjuku — PLATEAU + UrbanNav")
    print(f"KML: {kml_path}")

    # Also export Odaiba KML (rerun quickly)
    print("\n--- Odaiba KML export ---")
    building_od = loader.load_directory("experiments/data/plateau_odaiba")
    bvh_od = BVHAccelerator.from_building_model(building_od)
    usim_od = UrbanSignalSimulator(building_model=bvh_od, noise_floor_db=-35)
    pos_od, times_od = load_trajectory("experiments/data/urbannav/Odaiba/reference.csv", step=200)
    idx_od = np.linspace(0, len(pos_od)-1, min(20, len(pos_od)), dtype=int)
    kml_od = []
    for fi, ei in enumerate(idx_od):
        rx = pos_od[ei]
        t = times_od[ei] - times_od[0]
        sats = generate_sats(rx, n_sat, time_offset=t)
        result = usim_od.compute_epoch(rx_ecef=rx, sat_ecef=sats, prn_list=prn_list)
        kml_od.append({"rx_ecef": rx, "sat_ecef": sats, "prn_list": prn_list,
                       "is_los": result["is_los"], "visible": result["visible"],
                       "elevations": result["elevations"],
                       "excess_delays": result["excess_delays"],
                       "epoch_label": f"t={t:.0f}s"})
    kml_od_path = os.path.join(out_dir, "los_nlos_odaiba.kml")
    export_kml(kml_od, kml_od_path, name="LOS/NLOS Odaiba — PLATEAU + UrbanNav")
    print(f"KML: {kml_od_path}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Shinjuku: avg NLOS={np.mean(nlos_counts):.1f}, max={max(nlos_counts)}")
    print(f"  -> Dense urban = more NLOS (expected)")


if __name__ == "__main__":
    main()
