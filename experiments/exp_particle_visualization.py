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


def _run_ekf(data: dict, wls_init: np.ndarray) -> np.ndarray:
    """Run EKF to get reference positions for guide velocity."""
    from exp_urbannav_baseline import run_ekf
    ekf_pos, _ = run_ekf(data, wls_init)
    return ekf_pos


def run_pf_with_particle_dumps(
    data: dict,
    n_particles: int,
    wls_init: np.ndarray,
    sigma_pr: float = 10.0,
    dump_every: int = 10,
    max_dump_particles: int = 5000,
) -> dict:
    """Run PF with RobustClear likelihood (same as paper) and dump particles."""
    from exp_urbannav_pf3d import (
        PF_SIGMA_CB, PF_SIGMA_POS, PF_SIGMA_LOS,
        PF_SIGMA_NLOS, PF_NLOS_BIAS,
        _epoch_dt, _select_pf_epoch_measurements, _select_guide_velocity,
        _should_rescue_pf_epoch,
    )
    from gnss_gpu.multi_gnss import MultiGNSSSolver
    from gnss_gpu.multi_gnss_quality import MultiGNSSQualityVetoConfig

    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    gt_ecef = data["ground_truth"]
    system_ids = data.get("system_ids")

    # Run EKF first for guide velocity (same as evaluation pipeline)
    print("    Running EKF for guide reference...")
    ekf_pos = _run_ekf(data, wls_init)

    # Quality veto config (same as evaluation default)
    multi_solver = None
    quality_veto_config = None
    if system_ids is not None and len(data.get("constellations", ())) > 1:
        systems = sorted({int(sid) for epoch_ids in system_ids for sid in epoch_ids})
        multi_solver = MultiGNSSSolver(systems=systems)
        quality_veto_config = MultiGNSSQualityVetoConfig()

    # Use PF3D with RobustClear likelihood (same as paper headline)
    try:
        from gnss_gpu import ParticleFilter3D, BuildingModel
        empty_model = BuildingModel(np.zeros((0, 3, 3), dtype=np.float64))
        pf = ParticleFilter3D(
            n_particles=n_particles,
            building_model=empty_model,  # no 3D model, but clear_nlos_prob > 0
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_los=PF_SIGMA_LOS,
            sigma_nlos=PF_SIGMA_NLOS,
            nlos_bias=PF_NLOS_BIAS,
            blocked_nlos_prob=1.0,
            clear_nlos_prob=0.01,  # RobustClear: 1% NLOS mixture
            resampling="megopolis",
            seed=42,
        )
        print(f"    Using PF3D+RobustClear (clear_nlos_prob=0.01)")
    except (ImportError, RuntimeError):
        from gnss_gpu import ParticleFilterDevice
        pf = ParticleFilterDevice(
            n_particles=n_particles,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=PF_SIGMA_LOS,
            resampling="megopolis",
            seed=42,
        )
        print(f"    Fallback to ParticleFilterDevice (Gaussian)")

    init_pos = np.asarray(wls_init[0, :3], dtype=np.float64)
    init_cb = float(wls_init[0, 3])
    pf.initialize(init_pos, clock_bias=init_cb, spread_pos=50.0, spread_cb=500.0)

    estimates = np.zeros((n_epochs, 3))
    particle_frames = []

    t0 = time.perf_counter()
    for i in range(n_epochs):
        dt = _epoch_dt(data, i)

        # Use same measurement selection as evaluation pipeline
        sat_i, pr_i, w_i, use_multi = _select_pf_epoch_measurements(
            sat_ecef[i], pseudoranges[i], weights[i],
            None if system_ids is None else system_ids[i],
            multi_solver, quality_veto_config,
        )

        # Guide velocity from EKF (same as PF+EKFGuide)
        velocity = _select_guide_velocity(
            ekf_pos, i, dt, "always", use_multi, sat_i.shape[0],
        )

        pf.predict(velocity=velocity, dt=dt)

        # Rescue if too far from EKF
        if ekf_pos is not None and _should_rescue_pf_epoch(
            pf.estimate(), ekf_pos[i], 500.0,
        ):
            pf.initialize(
                ekf_pos[i, :3],
                clock_bias=float(ekf_pos[i, 3]) if ekf_pos.shape[1] > 3 else 0.0,
                spread_pos=50.0, spread_cb=500.0,
            )

        if sat_i.shape[0] >= 4:
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

    # Compute proper ENU-based 2D errors for all epochs
    from evaluate import ecef_errors_2d_3d
    pf_errors_2d, _ = ecef_errors_2d_3d(estimates, gt_ecef)
    ekf_errors_2d, _ = ecef_errors_2d_3d(ekf_pos[:, :3], gt_ecef)

    # Attach per-frame metrics
    for f in particle_frames:
        i = f["epoch"]
        f["error_2d"] = float(pf_errors_2d[i])
        f["ekf_error_2d"] = float(ekf_errors_2d[i])
        # Running RMS up to this epoch
        pf_rms = float(np.sqrt(np.mean(pf_errors_2d[:i+1] ** 2)))
        ekf_rms = float(np.sqrt(np.mean(ekf_errors_2d[:i+1] ** 2)))
        f["pf_rms"] = pf_rms
        f["ekf_rms"] = ekf_rms
    print(f"    PF run: {n_epochs} epochs, {n_particles} particles, {elapsed:.1f}s")
    print(f"    Dumped {len(particle_frames)} frames")
    return {"frames": particle_frames, "estimates": estimates}


def create_animation(
    frames: list[dict],
    output_path: Path,
    title: str = "MegaParticle GNSS",
    fps: int = 10,
    trail_length: int = 50,
    zoom_radius_m: float = 80.0,
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

    # --- Render frames manually ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=5000)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6), dpi=80)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Setup full view
    print("    Downloading OSM tiles (full view)...")
    ax_full.set_xlim(xmin_full, xmax_full)
    ax_full.set_ylim(ymin_full, ymax_full)
    cx.add_basemap(ax_full, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")
    ax_full.set_title("Full trajectory", fontsize=12)
    ax_full.set_axis_off()

    # Setup zoom view — download tiles for the FULL possible extent
    print("    Downloading OSM tiles (zoom view, full coverage)...")
    all_gt_wm = np.array([f["gt_wm"] for f in frames])
    all_est_wm = np.array([f["estimate_wm"] for f in frames])
    all_pts = np.vstack([all_gt_wm, all_est_wm])
    zoom_xmin = all_pts[:, 0].min() - zoom_r * 2
    zoom_xmax = all_pts[:, 0].max() + zoom_r * 2
    zoom_ymin = all_pts[:, 1].min() - zoom_r * 2
    zoom_ymax = all_pts[:, 1].max() + zoom_r * 2
    ax_zoom.set_xlim(zoom_xmin, zoom_xmax)
    ax_zoom.set_ylim(zoom_ymin, zoom_ymax)
    cx.add_basemap(ax_zoom, source=cx.providers.OpenStreetMap.Mapnik, zoom=18)
    ax_zoom.set_title("Zoom (around current position)", fontsize=12)
    ax_zoom.set_axis_off()

    gt_trail_x, gt_trail_y = [], []
    est_trail_x, est_trail_y = [], []

    print(f"    Rendering {len(frames)} frames...")
    with writer.saving(fig, str(output_path), dpi=80):
        for frame_idx, f in enumerate(frames):
            # Clear dynamic elements
            for collection in list(ax_full.collections):
                collection.remove()
            for collection in list(ax_zoom.collections):
                collection.remove()
            for patch in list(ax_full.patches):
                patch.remove()
            for patch in list(ax_zoom.patches):
                patch.remove()

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
                           s=3, c="#ff6600", alpha=0.3, zorder=3, edgecolors="none")
            ax_full.plot(gt_trail_x, gt_trail_y, "-", color="#3b82f6",
                        linewidth=2, alpha=0.7, zorder=4)
            ax_full.plot(est_trail_x, est_trail_y, "-", color="#ef4444",
                        linewidth=2, alpha=0.7, zorder=4)
            ax_full.plot(est[0], est[1], "o", color="#ef4444", markersize=12,
                        markeredgecolor="white", markeredgewidth=2, zorder=6)
            ax_full.plot(gt[0], gt[1], "s", color="#3b82f6", markersize=9,
                        markeredgecolor="white", markeredgewidth=2, zorder=6)
            pf_rms = f.get("pf_rms", 0)
            ekf_rms = f.get("ekf_rms", 0)
            ax_full.text(0.02, 0.98,
                        f"Epoch {f['epoch']} / {frames[-1]['epoch']}\n"
                        f"PF  RMS: {pf_rms:.1f} m\n"
                        f"EKF RMS: {ekf_rms:.1f} m",
                        transform=ax_full.transAxes, fontsize=10, va="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

            # --- Zoom view (centered on GT) ---
            while len(ax_zoom.lines) > 0:
                ax_zoom.lines[0].remove()
            while len(ax_zoom.texts) > 0:
                ax_zoom.texts[0].remove()

            # Center zoom on midpoint of estimate and GT
            cx_mid = (est[0] + gt[0]) / 2
            cy_mid = (est[1] + gt[1]) / 2
            ax_zoom.set_xlim(cx_mid - zoom_r, cx_mid + zoom_r)
            ax_zoom.set_ylim(cy_mid - zoom_r, cy_mid + zoom_r)

            # Particles — semi-transparent to show map underneath
            ax_zoom.scatter(particles[:, 0], particles[:, 1],
                           s=15, c="#ff6600", alpha=0.3, zorder=3,
                           edgecolors="none")
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

            # Metrics overlay (ENU-based)
            err_2d = f.get("error_2d", 0)
            ekf_err = f.get("ekf_error_2d", 0)
            pf_rms = f.get("pf_rms", 0)
            ekf_rms = f.get("ekf_rms", 0)
            ax_zoom.text(0.02, 0.98,
                        f"PF error:  {err_2d:.1f} m  (RMS: {pf_rms:.1f} m)\n"
                        f"EKF error: {ekf_err:.1f} m  (RMS: {ekf_rms:.1f} m)\n"
                        f"{len(particles)} particles",
                        transform=ax_zoom.transAxes, fontsize=9, va="top",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

            if frame_idx == 0:
                ax_zoom.legend(loc="lower right", fontsize=9)

            writer.grab_frame()

    plt.close(fig)
    print(f"    Saved {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


def _run_pf_gnssplusplus(
    run_dir: Path,
    run_name: str,
    n_particles: int,
    dump_every: int,
    max_dump_particles: int,
    rover_source: str = "trimble",
) -> dict:
    """Run PF with gnssplusplus corrected pseudoranges and dump particles."""
    from libgnsspp import preprocess_spp_file, solve_spp_file
    from gnss_gpu import ParticleFilterDevice
    from exp_urbannav_pf3d import PF_SIGMA_POS, PF_SIGMA_CB
    from exp_urbannav_baseline import load_or_generate_data
    from evaluate import ecef_errors_2d_3d

    obs_path = str(run_dir / f"rover_{rover_source}.obs")
    nav_path = str(run_dir / "base.nav")

    print("    Preprocessing with gnssplusplus...")
    epochs = preprocess_spp_file(obs_path, nav_path)
    print(f"    {len(epochs)} epochs preprocessed")

    # SPP positions for guide and comparison
    sol = solve_spp_file(obs_path, nav_path)
    spp_records = [r for r in sol.records() if r.is_valid()]
    spp_lookup = {round(r.time.tow, 1): np.array(r.position_ecef_m) for r in spp_records}

    # GT
    data = load_or_generate_data(run_dir, systems=("G", "E", "J"), urban_rover=rover_source)
    gt = data["ground_truth"]
    our_times = data["times"]

    pf = ParticleFilterDevice(
        n_particles=n_particles, sigma_pos=PF_SIGMA_POS, sigma_cb=PF_SIGMA_CB,
        sigma_pr=3.0, resampling="megopolis", seed=42,
    )

    first_pos = spp_records[0].position_ecef_m if spp_records else gt[0]
    pf.initialize(np.array(first_pos[:3]), clock_bias=0.0, spread_pos=10.0, spread_cb=100.0)

    all_pf_pos = []
    all_spp_pos = []
    all_gt = []
    all_tow = []
    particle_frames = []

    prev_tow = None
    frame_count = 0
    for sol_epoch, measurements in epochs:
        if not sol_epoch.is_valid() or len(measurements) < 4:
            continue
        tow = sol_epoch.time.tow
        tow_key = round(tow, 1)
        dt = tow - prev_tow if prev_tow else 0.1

        # Guide from SPP
        velocity = None
        if prev_tow is not None and tow_key in spp_lookup:
            prev_key = round(prev_tow, 1)
            if prev_key in spp_lookup and dt > 0:
                vel = (spp_lookup[tow_key][:3] - spp_lookup[prev_key][:3]) / dt
                if np.all(np.isfinite(vel)) and np.linalg.norm(vel) < 50:
                    velocity = vel

        pf.predict(velocity=velocity, dt=dt)

        sat_ecef = np.array([m.satellite_ecef for m in measurements])
        pr = np.array([m.corrected_pseudorange for m in measurements])
        w = np.array([m.weight for m in measurements])
        pf.update(sat_ecef, pr, weights=w)

        est = pf.estimate()

        # Match GT
        gt_idx = np.argmin(np.abs(our_times - tow))
        if abs(our_times[gt_idx] - tow) < 0.05:
            all_pf_pos.append(est[:3])
            all_spp_pos.append(np.array(sol_epoch.position_ecef_m[:3]))
            all_gt.append(gt[gt_idx])
            all_tow.append(tow)

            # Dump particles
            if frame_count % dump_every == 0:
                particles = pf.get_particles()
                if len(particles) > max_dump_particles:
                    idx = np.random.default_rng(42).choice(len(particles), max_dump_particles, replace=False)
                    particles = particles[idx]
                p_lonlat = particles_ecef_to_lonlat(particles)
                gt_lat, gt_lon, _ = ecef_to_lla(gt[gt_idx][0], gt[gt_idx][1], gt[gt_idx][2])
                est_lat, est_lon, _ = ecef_to_lla(est[0], est[1], est[2])
                particle_frames.append({
                    "epoch": frame_count,
                    "particles_lonlat": p_lonlat,
                    "estimate_lonlat": np.array([est_lon, est_lat]),
                    "gt_lonlat": np.array([gt_lon, gt_lat]),
                })
            frame_count += 1

        prev_tow = tow

    # Compute ENU errors for metrics
    pf_arr = np.array(all_pf_pos)
    spp_arr = np.array(all_spp_pos)
    gt_arr = np.array(all_gt)
    pf_err, _ = ecef_errors_2d_3d(pf_arr, gt_arr)
    spp_err, _ = ecef_errors_2d_3d(spp_arr, gt_arr)

    # Attach metrics to frames
    epoch_to_pf_err = {}
    epoch_to_spp_err = {}
    cum_pf_sq, cum_spp_sq = 0.0, 0.0
    for i in range(len(pf_err)):
        cum_pf_sq += pf_err[i] ** 2
        cum_spp_sq += spp_err[i] ** 2
        epoch_to_pf_err[i] = pf_err[i]
        epoch_to_spp_err[i] = spp_err[i]

    for f in particle_frames:
        ep = f["epoch"]
        f["error_2d"] = epoch_to_pf_err.get(ep, 0)
        f["ekf_error_2d"] = epoch_to_spp_err.get(ep, 0)  # SPP as "EKF" for display
        n_so_far = ep + 1
        f["pf_rms"] = float(np.sqrt(cum_pf_sq / max(n_so_far, 1))) if ep < len(pf_err) else 0
        f["ekf_rms"] = float(np.sqrt(cum_spp_sq / max(n_so_far, 1))) if ep < len(spp_err) else 0

    print(f"    PF+gpp: P50={np.median(pf_err):.2f}m  RMS={np.sqrt(np.mean(pf_err**2)):.2f}m  >100m={np.mean(pf_err>100)*100:.3f}%")
    print(f"    SPP:    P50={np.median(spp_err):.2f}m  RMS={np.sqrt(np.mean(spp_err**2)):.2f}m")
    return {"frames": particle_frames, "particle_dumps": {}}


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
    parser.add_argument("--use-gnssplusplus", action="store_true",
                        help="Use gnssplusplus corrected pseudoranges")
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
        w_i = np.asarray(data["weights"][i], dtype=np.float64).ravel()
        mask = np.isfinite(pr_i) & (pr_i > 0)
        if mask.sum() >= 4:
            try:
                res = wls_position(sat_i[mask], pr_i[mask], w_i[mask])
                wls_pos[i] = np.array(res[0])
            except Exception:
                if i > 0:
                    wls_pos[i] = wls_pos[i - 1]
        elif i > 0:
            wls_pos[i] = wls_pos[i - 1]

    if args.use_gnssplusplus:
        result = _run_pf_gnssplusplus(
            args.data_root / args.run, args.run, args.n_particles,
            args.dump_every, args.max_dump_particles, args.urban_rover,
        )
    else:
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
