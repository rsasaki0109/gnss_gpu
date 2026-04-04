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
) -> None:
    """Create mp4 animation with particles on OpenStreetMap."""
    import contextily as cx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    # Compute bounds from all frames
    all_lons = np.concatenate([f["particles_lonlat"][:, 0] for f in frames])
    all_lats = np.concatenate([f["particles_lonlat"][:, 1] for f in frames])
    gt_lons = np.array([f["gt_lonlat"][0] for f in frames])
    gt_lats = np.array([f["gt_lonlat"][1] for f in frames])

    # Use ground truth extent with padding
    lon_min, lon_max = gt_lons.min() - 0.003, gt_lons.max() + 0.003
    lat_min, lat_max = gt_lats.min() - 0.002, gt_lats.max() + 0.002

    # Convert to Web Mercator for contextily
    def lonlat_to_webmerc(lon, lat):
        x = lon * 20037508.34 / 180.0
        y = np.log(np.tan((90 + lat) * np.pi / 360.0)) * 20037508.34 / np.pi
        return x, y

    xmin, ymin = lonlat_to_webmerc(lon_min, lat_min)
    xmax, ymax = lonlat_to_webmerc(lon_max, lat_max)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

    # Download OSM basemap
    print("    Downloading OSM tiles...")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")
    # Save basemap as background
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(ax.bbox)

    # Pre-convert all coordinates to Web Mercator
    for f in frames:
        px, py = lonlat_to_webmerc(f["particles_lonlat"][:, 0], f["particles_lonlat"][:, 1])
        f["particles_wm"] = np.column_stack([px, py])
        ex, ey = lonlat_to_webmerc(f["estimate_lonlat"][0], f["estimate_lonlat"][1])
        f["estimate_wm"] = np.array([ex, ey])
        gx, gy = lonlat_to_webmerc(f["gt_lonlat"][0], f["gt_lonlat"][1])
        f["gt_wm"] = np.array([gx, gy])

    # Animation elements
    particles_scatter = ax.scatter([], [], s=0.3, c="#059669", alpha=0.15, zorder=3)
    est_trail, = ax.plot([], [], "-", color="#ef4444", linewidth=2.5, alpha=0.8, zorder=4)
    gt_trail, = ax.plot([], [], "-", color="#3b82f6", linewidth=2.5, alpha=0.8, zorder=4)
    estimate_dot, = ax.plot([], [], "o", color="#ef4444", markersize=14,
                            markeredgecolor="white", markeredgewidth=2, zorder=6, label="PF estimate")
    gt_dot, = ax.plot([], [], "s", color="#3b82f6", markersize=10,
                      markeredgecolor="white", markeredgewidth=2, zorder=6, label="Ground truth")
    epoch_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
                         va="top", ha="left", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_axis_off()

    gt_trail_x, gt_trail_y = [], []
    est_trail_x, est_trail_y = [], []

    def update(frame_idx):
        f = frames[frame_idx]
        # Particles
        particles_scatter.set_offsets(f["particles_wm"])
        # Estimate
        estimate_dot.set_data([f["estimate_wm"][0]], [f["estimate_wm"][1]])
        # Ground truth
        gt_dot.set_data([f["gt_wm"][0]], [f["gt_wm"][1]])
        # Trails
        gt_trail_x.append(f["gt_wm"][0])
        gt_trail_y.append(f["gt_wm"][1])
        est_trail_x.append(f["estimate_wm"][0])
        est_trail_y.append(f["estimate_wm"][1])
        start = max(0, len(gt_trail_x) - trail_length)
        gt_trail.set_data(gt_trail_x[start:], gt_trail_y[start:])
        est_trail.set_data(est_trail_x[start:], est_trail_y[start:])
        # Text
        epoch_text.set_text(f"Epoch {f['epoch']} / {frames[-1]['epoch']}  |  {len(f['particles_wm'])} particles shown")
        return particles_scatter, estimate_dot, gt_dot, gt_trail, est_trail, epoch_text

    print(f"    Rendering {len(frames)} frames...")
    anim = FuncAnimation(fig, update, frames=len(frames), blit=True, interval=1000 // fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(str(output_path), writer=writer)
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
