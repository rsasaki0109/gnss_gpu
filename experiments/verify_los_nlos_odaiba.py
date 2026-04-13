#!/usr/bin/env python3
"""Verify LOS/NLOS geometry with PLATEAU Odaiba + UrbanNav trajectory.

Loads real 3D building data (249K triangles) and an UrbanNav receiver
trajectory, then evaluates LOS/NLOS against a synthetic satellite sky
geometry at sampled epochs to generate verification images.
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

from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator, ecef_to_lla, _sat_elevation_azimuth


def load_urbannav_trajectory(csv_path, step=100):
    """Load UrbanNav reference trajectory, subsampled."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    positions = []
    times = []
    for i in range(0, len(rows), step):
        r = rows[i]
        x = float(r[" ECEF X (m)"])
        y = float(r[" ECEF Y (m)"])
        z = float(r[" ECEF Z (m)"])
        t = float(r["GPS TOW (s)"])
        positions.append([x, y, z])
        times.append(t)
    return np.array(positions), np.array(times)


def generate_sats(rx_ecef, n_sat=10, time_offset=0.0):
    """Generate synthetic satellite positions for a given receiver."""
    lat, lon, _ = ecef_to_lla(*rx_ecef)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    sat_ecef = np.zeros((n_sat, 3))
    for i in range(n_sat):
        el_deg = 10 + 70 * (i / max(n_sat - 1, 1))
        az_deg = (i * 36 + time_offset * 2) % 360
        el = math.radians(el_deg)
        az = math.radians(az_deg)
        r = 26600e3

        e = math.sin(az) * math.cos(el)
        n = math.cos(az) * math.cos(el)
        u = math.sin(el)

        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u
        sat_ecef[i] = rx_ecef + r * np.array([dx, dy, dz])
    return sat_ecef


def ecef_to_enu(rx_ecef, point_ecef):
    lat, lon, _ = ecef_to_lla(*rx_ecef)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ])
    diff = np.asarray(point_ecef) - np.asarray(rx_ecef)
    return R @ diff


def render_epoch(rx_ecef, sat_ecef, prn_list, bvh, result, epoch_idx, traj_enu, current_enu):
    """Render verification figure for one epoch."""
    fig = plt.figure(figsize=(16, 7), facecolor="#1a1a2e")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 0.7],
                          wspace=0.3, left=0.05, right=0.97, top=0.88, bottom=0.08)

    text_color = "#e0e0e0"
    los_color = "#00d4aa"
    nlos_color = "#ff6b6b"
    mp_color = "#ffd93d"
    bg_color = "#16213e"
    n_sat = len(prn_list)

    lat, lon, alt = ecef_to_lla(*rx_ecef)

    # --- Panel 1: Top-down map with trajectory + rays ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(bg_color)
    ax1.set_aspect("equal")

    # Trajectory
    ax1.plot(traj_enu[:, 0], traj_enu[:, 1], "-", color="#444466", linewidth=1, alpha=0.5)
    ax1.plot(current_enu[0], current_enu[1], "o", color="#ffffff", markersize=8, zorder=10)

    # Satellite rays
    for i in range(n_sat):
        if not result["visible"][i]:
            continue
        sat_enu = ecef_to_enu(rx_ecef, sat_ecef[i])
        direction = sat_enu[:2] / np.linalg.norm(sat_enu[:2]) * 120
        color = los_color if result["is_los"][i] else nlos_color
        style = "-" if result["is_los"][i] else "--"
        ax1.plot([current_enu[0], current_enu[0] + direction[0]],
                 [current_enu[1], current_enu[1] + direction[1]],
                 style, color=color, linewidth=1.2, alpha=0.7)
        ax1.annotate(f"{prn_list[i]}", (current_enu[0] + direction[0], current_enu[1] + direction[1]),
                     color=color, fontsize=7, fontweight="bold", ha="center")

    ax1.set_xlabel("East [m]", color=text_color, fontsize=9)
    ax1.set_ylabel("North [m]", color=text_color, fontsize=9)
    ax1.set_title(f"Epoch {epoch_idx} — lat={math.degrees(lat):.4f}° lon={math.degrees(lon):.4f}°",
                  color=text_color, fontsize=10)
    ax1.tick_params(colors=text_color, labelsize=7)
    ax1.grid(True, color="#333355", alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_color("#333355")

    # --- Panel 2: Skyplot ---
    ax2 = fig.add_subplot(gs[1], projection="polar")
    ax2.set_facecolor(bg_color)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_ylim(0, 90)
    ax2.set_yticks([0, 30, 60, 90])
    ax2.set_yticklabels(["90°", "60°", "30°", "0°"], color=text_color, fontsize=7)
    ax2.tick_params(colors=text_color, labelsize=7)
    ax2.grid(True, color="#333355", alpha=0.4)

    for i in range(n_sat):
        if not result["visible"][i]:
            continue
        az = result["azimuths"][i]
        el_deg = np.degrees(result["elevations"][i])
        r = 90 - el_deg
        is_los = result["is_los"][i]
        color = los_color if is_los else nlos_color
        marker = "o" if is_los else "s"
        ax2.scatter(az, r, c=color, s=80, marker=marker, zorder=5,
                    edgecolors="white", linewidths=0.5)
        ax2.annotate(f"{prn_list[i]}", (az, r), fontsize=7, color=color,
                     ha="center", va="bottom", xytext=(0, 6),
                     textcoords="offset points", fontweight="bold")

    ax2.set_title("Skyplot", color=text_color, fontsize=10, pad=12)
    handles = [
        mpatches.Patch(color=los_color, label=f"LOS ({result['n_los']})"),
        mpatches.Patch(color=nlos_color, label=f"NLOS ({result['n_nlos']})"),
    ]
    ax2.legend(handles=handles, loc="lower left", fontsize=7,
               facecolor=bg_color, edgecolor="#333355", labelcolor=text_color)

    # --- Panel 3: Stats ---
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(bg_color)
    ax3.axis("off")

    rows = []
    for i in range(n_sat):
        if not result["visible"][i]:
            continue
        status = "LOS" if result["is_los"][i] else "NLOS"
        el = f"{np.degrees(result['elevations'][i]):.0f}°"
        mp = f"{result['excess_delays'][i]:.1f}m" if result["excess_delays"][i] > 0.1 else "-"
        rows.append([str(prn_list[i]), status, el, mp])

    if rows:
        table = ax3.table(cellText=rows, colLabels=["PRN", "St", "El", "MP"],
                          cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for key, cell in table.get_celld().items():
            cell.set_facecolor(bg_color)
            cell.set_edgecolor("#333355")
            cell.set_text_props(color=text_color)
            if key[0] == 0:
                cell.set_facecolor("#0f3460")
                cell.set_text_props(color="white", fontweight="bold")
            elif len(rows) > key[0] - 1 and rows[key[0] - 1][1] == "NLOS":
                cell.set_facecolor("#2d1f1f")

    summary = f"249K triangles\nBVH ray-trace"
    ax3.text(0.5, 0.02, summary, transform=ax3.transAxes, fontsize=8,
             color=text_color, ha="center", va="bottom", fontfamily="monospace")

    fig.suptitle("LOS/NLOS Verification — PLATEAU Odaiba + UrbanNav",
                 color="white", fontsize=13, fontweight="bold")
    return fig


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "los_nlos_verification")
    os.makedirs(out_dir, exist_ok=True)

    # Load PLATEAU
    print("Loading PLATEAU Odaiba (249K triangles)...")
    t0 = time.time()
    loader = PlateauLoader(zone=9)
    building = loader.load_directory("experiments/data/plateau_odaiba")
    print(f"  Loaded {len(building.triangles)} triangles in {time.time()-t0:.1f}s")

    # Build BVH
    print("Building BVH...")
    t0 = time.time()
    bvh = BVHAccelerator.from_building_model(building)
    print(f"  BVH built in {time.time()-t0:.1f}s")

    # Load trajectory
    print("Loading UrbanNav Odaiba trajectory...")
    positions, times = load_urbannav_trajectory(
        "experiments/data/urbannav/Odaiba/reference.csv", step=200
    )
    print(f"  {len(positions)} sampled epochs")

    # Compute trajectory in ENU relative to first position
    traj_enu = np.array([ecef_to_enu(positions[0], p) for p in positions])

    # Run LOS/NLOS for sampled epochs
    n_sat = 10
    prn_list = list(range(1, n_sat + 1))
    usim = UrbanSignalSimulator(building_model=bvh, noise_floor_db=-35)

    n_epochs = min(20, len(positions))
    epoch_indices = np.linspace(0, len(positions) - 1, n_epochs, dtype=int)

    from PIL import Image
    frames = []

    for frame_idx, ep_idx in enumerate(epoch_indices):
        rx = positions[ep_idx]
        t = times[ep_idx] - times[0]
        sats = generate_sats(rx, n_sat, time_offset=t)

        print(f"  Epoch {frame_idx+1}/{n_epochs}: t={t:.0f}s", end="")
        t0 = time.time()
        result = usim.compute_epoch(rx_ecef=rx, sat_ecef=sats, prn_list=prn_list)
        dt = time.time() - t0
        print(f"  LOS={result['n_los']} NLOS={result['n_nlos']} MP={result['n_multipath']} [{dt*1000:.0f}ms]")

        current_enu = ecef_to_enu(positions[0], rx)
        fig = render_epoch(rx, sats, prn_list, bvh, result, ep_idx, traj_enu, current_enu)

        png_path = os.path.join(out_dir, f"odaiba_{frame_idx:03d}.png")
        fig.savefig(png_path, dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        frames.append(Image.open(png_path))

    # Save GIF
    gif_path = os.path.join(out_dir, "los_nlos_odaiba.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=600, loop=0, optimize=True)
    print(f"\nSaved GIF: {gif_path}")


if __name__ == "__main__":
    main()
