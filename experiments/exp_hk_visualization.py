#!/usr/bin/env python3
"""Visualize PF + cb_correct + position_update on HK-20190428."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "python"))

from exp_particle_visualization import (
    create_animation,
    ecef_to_lla,
    particles_ecef_to_lonlat,
)
from evaluate import compute_metrics, ecef_errors_2d_3d


def main():
    from exp_urbannav_baseline import load_or_generate_data
    from libgnsspp import preprocess_spp_file, solve_spp_file
    from gnss_gpu import ParticleFilterDevice
    from exp_urbannav_pf3d import PF_SIGMA_CB

    run_dir = Path("/tmp/UrbanNav-HK/HK_20190428")
    obs = str(run_dir / "rover_ublox.obs")
    nav = str(run_dir / "base_mixed.nav")

    n_particles = 100000
    sigma_pos = 3.0   # predict noise (best config)
    sigma_pr = 3.0
    pos_update_sigma = 3.0
    dump_every = 5
    max_dump = 3000

    print("Loading gnssplusplus data...")
    epochs = preprocess_spp_file(obs, nav)
    sol = solve_spp_file(obs, nav)
    spp_records = [r for r in sol.records() if r.is_valid()]
    spp_lookup = {round(r.time.tow, 1): np.array(r.position_ecef_m) for r in spp_records}

    data = load_or_generate_data(run_dir, systems=("G",), urban_rover="ublox")
    gt = data["ground_truth"]
    our_times = data["times"]

    # Init
    sol_ep0, meas0 = epochs[0]
    pos0 = np.array(sol_ep0.position_ecef_m[:3])
    init_cb = float(np.median([
        m.corrected_pseudorange - np.linalg.norm(np.array(m.satellite_ecef) - pos0)
        for m in meas0
    ]))

    pf = ParticleFilterDevice(
        n_particles=n_particles, sigma_pos=sigma_pos, sigma_cb=PF_SIGMA_CB,
        sigma_pr=sigma_pr, resampling="megopolis", seed=42,
    )
    pf.initialize(pos0, clock_bias=init_cb, spread_pos=10.0, spread_cb=100.0)

    frames = []
    all_pf_pos, all_spp_pos, all_gt = [], [], []
    prev_tow = None
    frame_count = 0
    cum_pf_sq, cum_spp_sq = 0.0, 0.0
    pf_count, spp_count = 0, 0

    print(f"Running PF ({n_particles:,} particles, sp={sigma_pos}, spr={sigma_pr}, PU={pos_update_sigma})...")
    for sol_epoch, measurements in epochs:
        if not sol_epoch.is_valid() or len(measurements) < 4:
            continue
        tow = sol_epoch.time.tow
        tow_key = round(tow, 1)
        dt = tow - prev_tow if prev_tow else 0.1

        velocity = None
        if prev_tow and tow_key in spp_lookup:
            pk = round(prev_tow, 1)
            if pk in spp_lookup and dt > 0:
                v = (spp_lookup[tow_key][:3] - spp_lookup[pk][:3]) / dt
                if np.all(np.isfinite(v)) and np.linalg.norm(v) < 50:
                    velocity = v

        pf.predict(velocity=velocity, dt=dt)
        sat_ecef = np.array([m.satellite_ecef for m in measurements])
        pr = np.array([m.corrected_pseudorange for m in measurements])
        w = np.array([m.weight for m in measurements])

        pf.correct_clock_bias(sat_ecef, pr)
        pf.update(sat_ecef, pr, weights=w)

        spp_pos = np.array(sol_epoch.position_ecef_m[:3])
        if np.isfinite(spp_pos).all() and np.linalg.norm(spp_pos) > 1e6:
            pf.position_update(spp_pos, sigma_pos=pos_update_sigma)

        pf_est = pf.estimate()[:3]

        gt_idx = np.argmin(np.abs(our_times - tow))
        if abs(our_times[gt_idx] - tow) < 0.55:
            gt_pos = gt[gt_idx]
            pf_err, _ = ecef_errors_2d_3d(pf_est.reshape(1, 3), gt_pos.reshape(1, 3))
            spp_err, _ = ecef_errors_2d_3d(spp_pos.reshape(1, 3), gt_pos.reshape(1, 3))
            cum_pf_sq += float(pf_err[0]) ** 2
            cum_spp_sq += float(spp_err[0]) ** 2
            pf_count += 1
            spp_count += 1
            pf_rms = float(np.sqrt(cum_pf_sq / pf_count))
            spp_rms = float(np.sqrt(cum_spp_sq / spp_count))

            all_pf_pos.append(pf_est)
            all_spp_pos.append(spp_pos)
            all_gt.append(gt_pos)

            if frame_count % dump_every == 0:
                particles = pf.get_particles()
                idx = np.random.default_rng(42).choice(len(particles), min(max_dump, len(particles)), replace=False)
                particles = particles[idx]
                p_lonlat = particles_ecef_to_lonlat(particles)
                gt_lat, gt_lon, _ = ecef_to_lla(gt_pos[0], gt_pos[1], gt_pos[2])
                est_lat, est_lon, _ = ecef_to_lla(pf_est[0], pf_est[1], pf_est[2])
                spp_lat, spp_lon, _ = ecef_to_lla(spp_pos[0], spp_pos[1], spp_pos[2])

                frames.append({
                    "epoch": frame_count,
                    "particles_lonlat": p_lonlat,
                    "estimate_lonlat": np.array([est_lon, est_lat]),
                    "gt_lonlat": np.array([gt_lon, gt_lat]),
                    "spp_lonlat": np.array([spp_lon, spp_lat]),
                    "error_2d": float(pf_err[0]),
                    "ekf_error_2d": float(spp_err[0]),
                    "pf_rms": pf_rms,
                    "ekf_rms": spp_rms,
                    "rtklib_error_2d": float(spp_err[0]),
                    "rtklib_rms": spp_rms,
                })
            frame_count += 1
        prev_tow = tow

    # Final metrics
    pf_arr = np.array(all_pf_pos)
    gt_arr = np.array(all_gt)
    m = compute_metrics(pf_arr, gt_arr)
    print(f"PF: P50={m['p50']:.2f}m  P95={m['p95']:.2f}m  RMS={m['rms_2d']:.2f}m")
    spp_m = compute_metrics(np.array(all_spp_pos), gt_arr)
    print(f"SPP: P50={spp_m['p50']:.2f}m  P95={spp_m['p95']:.2f}m  RMS={spp_m['rms_2d']:.2f}m")

    print(f"\n{len(frames)} frames collected")
    output = PROJECT_ROOT / "experiments" / "results" / "paper_assets" / "particle_viz_hk20190428.mp4"
    create_animation(
        frames, output,
        title=f"GPU Particle Filter GNSS — HK-20190428 ({n_particles:,} particles, cb correct + PU)",
        fps=8,
        zoom_radius_m=120.0,
    )

    # Also create GIF
    gif_output = output.with_suffix(".gif")
    print(f"    Converting to GIF...")
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", str(output),
        "-vf", "fps=8,scale=700:-1:flags=lanczos",
        "-loop", "0", str(gif_output),
    ], capture_output=True)
    if gif_output.exists():
        print(f"    GIF saved: {gif_output} ({gif_output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
