"""Generate demo visualization for GPU urban GNSS signal simulator.

Produces an animated GIF showing:
  1. Skyplot with LOS/NLOS classification over time
  2. IQ constellation diagram
  3. Acquisition correlation map
  4. Time-domain IQ waveform
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image

from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator, ecef_to_lla
from gnss_gpu.acquisition import Acquisition


def make_urban_canyon(rx_ecef):
    """Create a realistic urban canyon around receiver."""
    triangles = []
    offsets = [
        (-30, 0), (30, 0), (0, -30), (0, 30),
        (-20, -20), (20, 20), (-20, 20), (20, -20),
    ]
    heights = [50, 70, 40, 60, 80, 45, 55, 65]
    for (dx, dy), h in zip(offsets, heights):
        b = BuildingModel.create_box(
            center=[rx_ecef[0] + dx, rx_ecef[1] + dy, rx_ecef[2] + h / 2],
            width=18, depth=18, height=h,
        )
        triangles.append(b.triangles)
    return BuildingModel(np.concatenate(triangles))


def sat_positions(rx_ecef, lat, n_sat=10, time_offset=0.0):
    """Generate satellite positions spread across the sky."""
    sat_ecef = np.zeros((n_sat, 3))
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    lon = math.atan2(rx_ecef[1], rx_ecef[0])
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    for i in range(n_sat):
        el_deg = 12 + 68 * (i / max(n_sat - 1, 1))
        az_deg = (i * 36 + time_offset * 3) % 360
        el = math.radians(el_deg)
        az = math.radians(az_deg)
        r = 26600e3

        e_enu = math.sin(az) * math.cos(el)
        n_enu = math.cos(az) * math.cos(el)
        u_enu = math.sin(el)

        dx = -sin_lon * e_enu - sin_lat * cos_lon * n_enu + cos_lat * cos_lon * u_enu
        dy = cos_lon * e_enu - sin_lat * sin_lon * n_enu + cos_lat * sin_lon * u_enu
        dz = cos_lat * n_enu + sin_lat * u_enu

        sat_ecef[i] = rx_ecef + r * np.array([dx, dy, dz])
    return sat_ecef


def render_frame(rx_ecef, canyon, usim, acq, t, n_sat=10):
    """Render one frame of the demo."""
    lat, lon, alt = ecef_to_lla(*rx_ecef)
    sats = sat_positions(rx_ecef, lat, n_sat, time_offset=t)
    prn_list = list(range(1, n_sat + 1))

    result = usim.compute_epoch(rx_ecef=rx_ecef, sat_ecef=sats, prn_list=prn_list)
    iq = result["iq"]
    signal_i = iq[0::2].copy()

    acq_results = acq.acquire(signal_i, prn_list=prn_list)

    fig = plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                           left=0.06, right=0.96, top=0.92, bottom=0.08)

    text_color = "#e0e0e0"
    los_color = "#00d4aa"
    nlos_color = "#ff6b6b"
    mp_color = "#ffd93d"
    grid_color = "#333355"

    # --- Skyplot ---
    ax_sky = fig.add_subplot(gs[0, 0], projection="polar")
    ax_sky.set_facecolor("#16213e")
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_ylim(0, 90)
    ax_sky.set_yticks([0, 30, 60, 90])
    ax_sky.set_yticklabels(["90°", "60°", "30°", "0°"], color=text_color, fontsize=7)
    ax_sky.tick_params(colors=text_color, labelsize=7)
    ax_sky.grid(True, color=grid_color, alpha=0.4)

    for i in range(n_sat):
        if not result["visible"][i]:
            continue
        az = result["azimuths"][i]
        el_deg = np.degrees(result["elevations"][i])
        r = 90 - el_deg

        color = los_color if result["is_los"][i] else nlos_color
        marker = "o" if result["is_los"][i] else "x"
        size = 80 if result["is_los"][i] else 60

        ax_sky.scatter(az, r, c=color, s=size, marker=marker, zorder=5, edgecolors="white", linewidths=0.5)
        ax_sky.annotate(f"{prn_list[i]}", (az, r), fontsize=7, color=color,
                        ha="center", va="bottom", xytext=(0, 6),
                        textcoords="offset points", fontweight="bold")

    ax_sky.set_title("Skyplot (LOS/NLOS)", color=text_color, fontsize=10, pad=12)

    # --- IQ Scatter ---
    ax_iq = fig.add_subplot(gs[0, 1])
    ax_iq.set_facecolor("#16213e")
    i_data = iq[0::2]
    q_data = iq[1::2]
    n_pts = min(500, len(i_data))
    idx = np.random.choice(len(i_data), n_pts, replace=False)
    ax_iq.scatter(i_data[idx], q_data[idx], c="#4ecdc4", s=2, alpha=0.6)
    ax_iq.set_xlabel("I", color=text_color, fontsize=9)
    ax_iq.set_ylabel("Q", color=text_color, fontsize=9)
    ax_iq.set_title("IQ Constellation", color=text_color, fontsize=10)
    ax_iq.tick_params(colors=text_color, labelsize=7)
    ax_iq.grid(True, color=grid_color, alpha=0.3)
    ax_iq.set_aspect("equal")
    lim = max(abs(i_data).max(), abs(q_data).max()) * 1.1
    ax_iq.set_xlim(-lim, lim)
    ax_iq.set_ylim(-lim, lim)
    for spine in ax_iq.spines.values():
        spine.set_color(grid_color)

    # --- Acquisition Results ---
    ax_acq = fig.add_subplot(gs[0, 2])
    ax_acq.set_facecolor("#16213e")
    prns = [r["prn"] for r in acq_results]
    snrs = [r["snr"] for r in acq_results]
    acquired_mask = [r["acquired"] for r in acq_results]
    colors = [los_color if a else "#555577" for a in acquired_mask]
    bars = ax_acq.bar(prns, snrs, color=colors, width=0.7, edgecolor="none")
    ax_acq.axhline(y=2.5, color=nlos_color, linestyle="--", alpha=0.7, linewidth=1)
    ax_acq.text(max(prns) + 0.5, 2.7, "threshold", color=nlos_color, fontsize=7, ha="right")
    ax_acq.set_xlabel("PRN", color=text_color, fontsize=9)
    ax_acq.set_ylabel("SNR", color=text_color, fontsize=9)
    ax_acq.set_title("Acquisition Results", color=text_color, fontsize=10)
    ax_acq.tick_params(colors=text_color, labelsize=7)
    ax_acq.grid(True, axis="y", color=grid_color, alpha=0.3)
    for spine in ax_acq.spines.values():
        spine.set_color(grid_color)

    # --- Time-domain I signal ---
    ax_td = fig.add_subplot(gs[1, :2])
    ax_td.set_facecolor("#16213e")
    n_show = min(500, len(i_data))
    t_us = np.arange(n_show) / 2.6  # microseconds at 2.6 MHz
    ax_td.plot(t_us, i_data[:n_show], color="#6c5ce7", linewidth=0.5, alpha=0.8)
    ax_td.plot(t_us, q_data[:n_show], color="#fd79a8", linewidth=0.5, alpha=0.5)
    ax_td.set_xlabel("Time [μs]", color=text_color, fontsize=9)
    ax_td.set_ylabel("Amplitude", color=text_color, fontsize=9)
    ax_td.set_title("IF Signal (I: purple, Q: pink)", color=text_color, fontsize=10)
    ax_td.tick_params(colors=text_color, labelsize=7)
    ax_td.grid(True, color=grid_color, alpha=0.3)
    for spine in ax_td.spines.values():
        spine.set_color(grid_color)

    # --- Stats panel ---
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.set_facecolor("#16213e")
    ax_stats.axis("off")

    stats_text = (
        f"Epoch t = {t:.0f}s\n\n"
        f"Visible:    {int(np.sum(result['visible']))}\n"
        f"LOS:        {result['n_los']}\n"
        f"NLOS:       {result['n_nlos']}\n"
        f"Multipath:  {result['n_multipath']}\n"
        f"Channels:   {len(result['channels'])}\n"
        f"Acquired:   {sum(acquired_mask)}\n\n"
        f"Fs = 2.6 MHz\n"
        f"GPU: CUDA kernel\n"
        f"3D model: BVH ray-trace"
    )
    ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, color=text_color, verticalalignment="top",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", edgecolor=grid_color))
    ax_stats.set_title("Simulation Stats", color=text_color, fontsize=10)

    fig.suptitle("GPU Urban GNSS Signal Simulator — End-to-End Pipeline",
                 color="#ffffff", fontsize=13, fontweight="bold", y=0.97)

    return fig


def main():
    rx_ecef = np.array([-3959340.0, 3352854.0, 3697471.0])
    canyon = make_urban_canyon(rx_ecef)
    usim = UrbanSignalSimulator(building_model=canyon, noise_floor_db=-35,
                                 nlos_attenuation_db=8.0, fresnel_coeff=0.3)
    acq = Acquisition(sampling_freq=2.6e6, intermediate_freq=0)

    out_dir = os.path.join(os.path.dirname(__file__), "results", "signal_sim_demo")
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    n_frames = 30
    print(f"Generating {n_frames} frames...")
    for frame_idx in range(n_frames):
        t = frame_idx * 10.0  # 10s steps
        fig = render_frame(rx_ecef, canyon, usim, acq, t, n_sat=10)

        png_path = os.path.join(out_dir, f"frame_{frame_idx:03d}.png")
        fig.savefig(png_path, dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)

        frames.append(Image.open(png_path))
        print(f"  frame {frame_idx+1}/{n_frames}")

    gif_path = os.path.join(out_dir, "urban_signal_sim_demo.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=400, loop=0, optimize=True)
    print(f"\nSaved GIF: {gif_path}")

    # Also save a static poster frame
    poster_fig = render_frame(rx_ecef, canyon, usim, acq, 50.0, n_sat=10)
    poster_path = os.path.join(out_dir, "urban_signal_sim_poster.png")
    poster_fig.savefig(poster_path, dpi=150, facecolor=poster_fig.get_facecolor())
    plt.close(poster_fig)
    print(f"Saved poster: {poster_path}")


if __name__ == "__main__":
    main()
