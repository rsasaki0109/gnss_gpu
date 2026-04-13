#!/usr/bin/env python3
"""Verify LOS/NLOS classification with PLATEAU 3D city model.

Generates a visualization showing:
  - Top-down building footprint with receiver position
  - Satellite ray traces (green=LOS, red=NLOS)
  - Skyplot with LOS/NLOS markers
  - Per-satellite stats table

Uses sample_plateau.gml or fetched PLATEAU data.
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator, ecef_to_lla, _sat_elevation_azimuth


def ecef_to_enu(rx_ecef, point_ecef):
    """Convert ECEF to ENU relative to receiver."""
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


def render_verification(rx_ecef, sat_ecef, prn_list, building_model,
                        result, title="LOS/NLOS Verification"):
    """Render a 3-panel verification figure."""
    fig = plt.figure(figsize=(16, 7), facecolor="#1a1a2e")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 0.8],
                          wspace=0.3, left=0.05, right=0.97, top=0.88, bottom=0.08)

    text_color = "#e0e0e0"
    los_color = "#00d4aa"
    nlos_color = "#ff6b6b"
    mp_color = "#ffd93d"
    bg_color = "#16213e"

    n_sat = len(prn_list)

    # --- Panel 1: Top-down building + ray view ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(bg_color)
    ax1.set_aspect("equal")

    # Plot building triangles in ENU
    tris = building_model.triangles
    for tri in tris:
        enu_verts = np.array([ecef_to_enu(rx_ecef, v) for v in tri])
        # Top-down: E-N plane
        xs = list(enu_verts[:, 0]) + [enu_verts[0, 0]]
        ys = list(enu_verts[:, 1]) + [enu_verts[0, 1]]
        ax1.fill(xs, ys, color="#334466", alpha=0.3, edgecolor="#556688", linewidth=0.3)

    # Receiver
    ax1.plot(0, 0, "o", color="#ffffff", markersize=8, zorder=10)
    ax1.annotate("RX", (0, 0), color="white", fontsize=8, ha="center",
                 va="bottom", xytext=(0, 8), textcoords="offset points")

    # Satellite rays (projected to ENU)
    for i in range(n_sat):
        if not result["visible"][i]:
            continue
        sat_enu = ecef_to_enu(rx_ecef, sat_ecef[i])
        # Normalize to ~100m for display
        direction = sat_enu[:2] / np.linalg.norm(sat_enu[:2]) * 80
        color = los_color if result["is_los"][i] else nlos_color
        style = "-" if result["is_los"][i] else "--"
        ax1.plot([0, direction[0]], [0, direction[1]], style,
                 color=color, linewidth=1.5, alpha=0.7)
        ax1.annotate(f"{prn_list[i]}", (direction[0], direction[1]),
                     color=color, fontsize=7, fontweight="bold",
                     ha="center", va="center")

    ax1.set_xlabel("East [m]", color=text_color, fontsize=9)
    ax1.set_ylabel("North [m]", color=text_color, fontsize=9)
    ax1.set_title("Top-Down View: Building + Satellite Rays", color=text_color, fontsize=10)
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

        if result["excess_delays"][i] > 0.1:
            ax2.scatter(az, r, c="none", s=200, marker="o", zorder=4,
                        edgecolors=mp_color, linewidths=1.5, linestyle="--")

    ax2.set_title("Skyplot", color=text_color, fontsize=10, pad=12)

    # Legend
    handles = [
        mpatches.Patch(color=los_color, label="LOS"),
        mpatches.Patch(color=nlos_color, label="NLOS"),
        mpatches.Patch(facecolor="none", edgecolor=mp_color, label="Multipath"),
    ]
    ax2.legend(handles=handles, loc="lower left", fontsize=7,
               facecolor=bg_color, edgecolor="#333355", labelcolor=text_color)

    # --- Panel 3: Stats table ---
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
        table = ax3.table(
            cellText=rows,
            colLabels=["PRN", "Status", "Elev", "MP delay"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for key, cell in table.get_celld().items():
            cell.set_facecolor(bg_color)
            cell.set_edgecolor("#333355")
            cell.set_text_props(color=text_color)
            if key[0] == 0:  # header
                cell.set_facecolor("#0f3460")
                cell.set_text_props(color="white", fontweight="bold")
            elif len(rows) > key[0] - 1:
                row = rows[key[0] - 1]
                if row[1] == "NLOS":
                    cell.set_facecolor("#2d1f1f")

    summary = (
        f"Visible: {int(np.sum(result['visible']))}\n"
        f"LOS: {result['n_los']}  NLOS: {result['n_nlos']}\n"
        f"Multipath: {result['n_multipath']}\n"
        f"Triangles: {len(building_model.triangles)}"
    )
    ax3.text(0.5, 0.02, summary, transform=ax3.transAxes, fontsize=9,
             color=text_color, ha="center", va="bottom", fontfamily="monospace")
    ax3.set_title("Per-Satellite Detail", color=text_color, fontsize=10)

    fig.suptitle(title, color="white", fontsize=13, fontweight="bold")
    return fig


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results", "los_nlos_verification")
    os.makedirs(out_dir, exist_ok=True)

    # --- Load sample PLATEAU data ---
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    gml_path = os.path.join(data_dir, "sample_plateau.gml")

    print(f"Loading PLATEAU model: {gml_path}")
    loader = PlateauLoader(zone=9)
    building = loader.load_citygml(gml_path)
    print(f"  Loaded {len(building.triangles)} triangles")

    # Receiver position: center of the sample buildings (Tokyo Station area)
    # Compute centroid of all building vertices
    all_verts = building.triangles.reshape(-1, 3)
    centroid = all_verts.mean(axis=0)
    rx_ecef = centroid.copy()
    lat, lon, alt = ecef_to_lla(*rx_ecef)
    print(f"  Receiver: lat={math.degrees(lat):.6f}, lon={math.degrees(lon):.6f}, alt={alt:.1f}m")

    # Generate satellite positions at various elevations/azimuths
    n_sat = 12
    sat_ecef = np.zeros((n_sat, 3))
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    for i in range(n_sat):
        el_deg = 10 + 70 * (i / max(n_sat - 1, 1))
        az_deg = i * 30
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

    prn_list = list(range(1, n_sat + 1))

    # Run simulation
    print("Running LOS/NLOS classification...")
    usim = UrbanSignalSimulator(building_model=building, noise_floor_db=-35)
    result = usim.compute_epoch(rx_ecef=rx_ecef, sat_ecef=sat_ecef, prn_list=prn_list)

    print(f"  Visible: {int(np.sum(result['visible']))}, "
          f"LOS: {result['n_los']}, NLOS: {result['n_nlos']}, "
          f"Multipath: {result['n_multipath']}")

    # Render verification
    fig = render_verification(rx_ecef, sat_ecef, prn_list, building, result,
                              title="LOS/NLOS Verification — PLATEAU sample (Tokyo Station)")
    out_path = os.path.join(out_dir, "los_nlos_sample_plateau.png")
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print detailed results
    print("\nPer-satellite results:")
    print(f"{'PRN':>4} {'Status':>6} {'Elev':>6} {'Azim':>6} {'MP delay':>10}")
    print("-" * 40)
    for i in range(n_sat):
        if result["visible"][i]:
            status = "LOS" if result["is_los"][i] else "NLOS"
            el_deg = math.degrees(result["elevations"][i])
            az_deg = math.degrees(result["azimuths"][i])
            mp = result["excess_delays"][i]
            mp_str = f"{mp:.1f}m" if mp > 0.1 else "-"
            print(f"{prn_list[i]:>4} {status:>6} {el_deg:>5.0f}° {az_deg:>5.0f}° {mp_str:>10}")


if __name__ == "__main__":
    main()
