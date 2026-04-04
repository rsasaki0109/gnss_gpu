#!/usr/bin/env python3
"""Visualize particle cloud on OpenStreetMap for UrbanNav sequences.

Creates an mp4 animation showing particles moving on a real map,
with ground truth and EKF estimate for comparison.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    """ECEF to geodetic (lat, lon, alt) using iterative method."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2 * f - f * f
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1 - e2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + e2 * N * sin_lat, p)
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), alt


def particles_ecef_to_lonlat(particles: np.ndarray) -> np.ndarray:
    """Convert (N, 4) ECEF particles to (N, 2) lon/lat."""
    result = np.zeros((len(particles), 2))
    for i in range(len(particles)):
        lat, lon, _ = ecef_to_lla(particles[i, 0], particles[i, 1], particles[i, 2])
        result[i] = [lon, lat]
    return result


def run_pf_with_particle_dumps(
    data: dict,
    n_particles: int,
    wls_init: np.ndarray,
    sigma_pr: float = 10.0,
    dump_every: int = 10,
    max_dump_particles: int = 5000,
) -> dict:
    """Run PF and dump particle clouds at regular intervals."""
    from gnss_gpu import ParticleFilterDevice
    from exp_urbannav_pf3d import PF_SIGMA_CB, PF_SIGMA_POS, _epoch_dt

    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    gt_ecef = data["ground_truth"]

    pf = ParticleFilterDevice(
        n_particles=n_particles,
        sigma_pos=PF_SIGMA_POS,
        sigma_cb=PF_SIGMA_CB,
        sigma_pr=sigma_pr,
        resampling="megopolis",
        seed=42,
    )

    init_pos = np.asarray(wls_init[0, :3], dtype=np.float64)
    init_cb = float(wls_init[0, 3])
    pf.initialize(init_pos, clock_bias=init_cb, spread_pos=50.0, spread_cb=500.0)

    estimates = np.zeros((n_epochs, 3))
    particle_frames = []  # list of (epoch, particles_lonlat, estimate_lonlat, gt_lonlat)

    t0 = time.perf_counter()
    for i in range(n_epochs):
        dt = _epoch_dt(data, i)
        sat_i = np.asarray(sat_ecef[i], dtype=np.float64).reshape(-1, 3)
        pr_i = np.asarray(pseudoranges[i], dtype=np.float64).ravel()
        w_i = np.asarray(weights[i], dtype=np.float64).ravel()
        mask = np.isfinite(pr_i) & (pr_i > 0)
        sat_i, pr_i, w_i = sat_i[mask], pr_i[mask], w_i[mask]

        pf.predict(dt=dt)
        if len(pr_i) >= 4:
            pf.update(sat_i, pr_i, weights=w_i)

        estimate = np.asarray(pf.estimate(), dtype=np.float64)
        estimates[i] = estimate[:3]

        if i % dump_every == 0:
            particles = pf.get_particles()  # (N, 4)
            # Subsample for visualization
            if len(particles) > max_dump_particles:
                idx = np.random.default_rng(42).choice(
                    len(particles), max_dump_particles, replace=False
                )
                particles = particles[idx]

            p_lonlat = particles_ecef_to_lonlat(particles)
            gt = np.asarray(gt_ecef[i], dtype=np.float64)
            gt_lat, gt_lon, _ = ecef_to_lla(gt[0], gt[1], gt[2])
            est_lat, est_lon, _ = ecef_to_lla(estimate[0], estimate[1], estimate[2])

            particle_frames.append({
                "epoch": i,
                "particles_lonlat": p_lonlat,
                "estimate_lonlat": np.array([est_lon, est_lat]),
                "gt_lonlat": np.array([gt_lon, gt_lat]),
            })

    elapsed = time.perf_counter() - t0
    print(f"    PF run: {n_epochs} epochs, {n_particles} particles, {elapsed:.1f}s")
    print(f"    Dumped {len(particle_frames)} frames")
    return {"frames": particle_frames, "estimates": estimates}


def create_animation(
    frames: list[dict],
    output_path: Path,
    title: str = "MegaParticle GNSS",
    fps: int = 10,
    trail_length: int = 50,
    zoom_radius_m: float = 300.0,
) -> None:
    """Create mp4 animation with particles on OpenStreetMap (full + zoom)."""
    import contextily as cx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    gt_lons = np.array([f["gt_lonlat"][0] for f in frames])
    gt_lats = np.array([f["gt_lonlat"][1] for f in frames])

    # Full view extent from ground truth
    lon_min, lon_max = gt_lons.min() - 0.003, gt_lons.max() + 0.003
    lat_min, lat_max = gt_lats.min() - 0.002, gt_lats.max() + 0.002

    def lonlat_to_webmerc(lon, lat):
        x = lon * 20037508.34 / 180.0
        y_val = np.clip(lat, -85, 85)
        y = np.log(np.tan((90 + y_val) * np.pi / 360.0)) * 20037508.34 / np.pi
        return x, y

    # Zoom radius in Web Mercator meters (approximate)
    zoom_r = zoom_radius_m * 1.5  # scale factor for Web Mercator at mid-lat

    # Pre-convert all coordinates
    for f in frames:
        px, py = lonlat_to_webmerc(f["particles_lonlat"][:, 0], f["particles_lonlat"][:, 1])
        f["particles_wm"] = np.column_stack([px, py])
        ex, ey = lonlat_to_webmerc(f["estimate_lonlat"][0], f["estimate_lonlat"][1])
        f["estimate_wm"] = np.array([ex, ey])
        gx, gy = lonlat_to_webmerc(f["gt_lonlat"][0], f["gt_lonlat"][1])
        f["gt_wm"] = np.array([gx, gy])

    xmin_full, ymin_full = lonlat_to_webmerc(lon_min, lat_min)
    xmax_full, ymax_full = lonlat_to_webmerc(lon_max, lat_max)

    # --- Render frames manually (no FuncAnimation for reliability) ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=4000)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(18, 8), dpi=100)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Setup full view
    print("    Downloading OSM tiles (full view)...")
    ax_full.set_xlim(xmin_full, xmax_full)
    ax_full.set_ylim(ymin_full, ymax_full)
    cx.add_basemap(ax_full, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")
    ax_full.set_title("Full trajectory", fontsize=12)
    ax_full.set_axis_off()

    # Setup zoom view (will be re-tiled per frame)
    print("    Downloading OSM tiles (zoom view)...")
    mid_x = (xmin_full + xmax_full) / 2
    mid_y = (ymin_full + ymax_full) / 2
    ax_zoom.set_xlim(mid_x - zoom_r, mid_x + zoom_r)
    ax_zoom.set_ylim(mid_y - zoom_r, mid_y + zoom_r)
    cx.add_basemap(ax_zoom, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")
    ax_zoom.set_title("Zoom (around current position)", fontsize=12)
    ax_zoom.set_axis_off()

    gt_trail_x, gt_trail_y = [], []
    est_trail_x, est_trail_y = [], []

    print(f"    Rendering {len(frames)} frames...")
    with writer.saving(fig, str(output_path), dpi=100):
        for frame_idx, f in enumerate(frames):
            # Clear dynamic elements
            for collection in list(ax_full.collections):
                collection.remove()
            for collection in list(ax_zoom.collections):
                collection.remove()

            # Trails
            gt_trail_x.append(f["gt_wm"][0])
            gt_trail_y.append(f["gt_wm"][1])
            est_trail_x.append(f["estimate_wm"][0])
            est_trail_y.append(f["estimate_wm"][1])
            start = max(0, len(gt_trail_x) - trail_length)

            particles = f["particles_wm"]
            est = f["estimate_wm"]
            gt = f["gt_wm"]

            # --- Full view ---
            # Remove old lines
            while len(ax_full.lines) > 0:
                ax_full.lines[0].remove()
            while len(ax_full.texts) > 0:
                ax_full.texts[0].remove()

            ax_full.scatter(particles[:, 0], particles[:, 1],
                           s=2, c="#22c55e", alpha=0.4, zorder=3, edgecolors="none")
            ax_full.plot(gt_trail_x, gt_trail_y, "-", color="#3b82f6",
                        linewidth=2, alpha=0.7, zorder=4)
            ax_full.plot(est_trail_x, est_trail_y, "-", color="#ef4444",
                        linewidth=2, alpha=0.7, zorder=4)
            ax_full.plot(est[0], est[1], "o", color="#ef4444", markersize=12,
                        markeredgecolor="white", markeredgewidth=2, zorder=6)
            ax_full.plot(gt[0], gt[1], "s", color="#3b82f6", markersize=9,
                        markeredgecolor="white", markeredgewidth=2, zorder=6)
            ax_full.text(0.02, 0.98,
                        f"Epoch {f['epoch']} / {frames[-1]['epoch']}",
                        transform=ax_full.transAxes, fontsize=11, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

            # --- Zoom view (centered on GT) ---
            while len(ax_zoom.lines) > 0:
                ax_zoom.lines[0].remove()
            while len(ax_zoom.texts) > 0:
                ax_zoom.texts[0].remove()

            ax_zoom.set_xlim(gt[0] - zoom_r, gt[0] + zoom_r)
            ax_zoom.set_ylim(gt[1] - zoom_r, gt[1] + zoom_r)

            ax_zoom.scatter(particles[:, 0], particles[:, 1],
                           s=8, c="#22c55e", alpha=0.5, zorder=3, edgecolors="none")
            ax_zoom.plot(gt_trail_x[start:], gt_trail_y[start:], "-",
                        color="#3b82f6", linewidth=3, alpha=0.8, zorder=4)
            ax_zoom.plot(est_trail_x[start:], est_trail_y[start:], "-",
                        color="#ef4444", linewidth=3, alpha=0.8, zorder=4)
            ax_zoom.plot(est[0], est[1], "o", color="#ef4444", markersize=16,
                        markeredgecolor="white", markeredgewidth=2.5, zorder=6,
                        label="PF estimate" if frame_idx == 0 else "")
            ax_zoom.plot(gt[0], gt[1], "s", color="#3b82f6", markersize=12,
                        markeredgecolor="white", markeredgewidth=2.5, zorder=6,
                        label="Ground truth" if frame_idx == 0 else "")

            # Error text
            err_m = np.sqrt((est[0] - gt[0])**2 + (est[1] - gt[1])**2)
            ax_zoom.text(0.02, 0.98,
                        f"Error: {err_m:.0f} m (Web Mercator)\n{len(particles)} particles shown",
                        transform=ax_zoom.transAxes, fontsize=10, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

            if frame_idx == 0:
                ax_zoom.legend(loc="lower right", fontsize=9)

            writer.grab_frame()

    plt.close(fig)
    print(f"    Saved {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Particle cloud visualization on OSM")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=str, default="Odaiba")
    parser.add_argument("--systems", type=str, default="G,E,J")
    parser.add_argument("--urban-rover", type=str, default="trimble")
    parser.add_argument("--n-particles", type=int, default=10000)
    parser.add_argument("--dump-every", type=int, default=10)
    parser.add_argument("--max-dump-particles", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from exp_urbannav_baseline import load_or_generate_data
    from gnss_gpu import wls_position

    systems = tuple(s.strip().upper() for s in args.systems.split(","))
    run_dir = args.data_root / args.run
    data = load_or_generate_data(run_dir, systems=systems, urban_rover=args.urban_rover)

    # WLS init
    n_epochs = data["n_epochs"]
    wls_pos = np.zeros((n_epochs, 4))
    for i in range(n_epochs):
        sat_i = np.asarray(data["sat_ecef"][i], dtype=np.float64).reshape(-1, 3)
        pr_i = np.asarray(data["pseudoranges"][i], dtype=np.float64).ravel()
        mask = np.isfinite(pr_i) & (pr_i > 0)
        if mask.sum() >= 4:
            try:
                wls_pos[i] = wls_position(sat_i[mask], pr_i[mask])
            except Exception:
                if i > 0:
                    wls_pos[i] = wls_pos[i - 1]
        elif i > 0:
            wls_pos[i] = wls_pos[i - 1]

    result = run_pf_with_particle_dumps(
        data, args.n_particles, wls_pos,
        dump_every=args.dump_every,
        max_dump_particles=args.max_dump_particles,
    )

    output = args.output or (RESULTS_DIR / "paper_assets" / f"particle_viz_{args.run.lower()}_{args.n_particles}.mp4")
    create_animation(
        result["frames"], output,
        title=f"MegaParticle GNSS — {args.run} ({args.n_particles:,} particles)",
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
