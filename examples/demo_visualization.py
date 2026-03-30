#!/usr/bin/env python3
"""Demo script that generates all gnss_gpu visualization types.

Saves each plot as PNG to /tmp/gnss_gpu_viz/.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnss_gpu.viz.plots import (
    plot_particles,
    plot_skyplot,
    plot_vulnerability_map,
    plot_trajectory,
    plot_dop_timeline,
    plot_positioning_error,
    plot_spectrogram,
    plot_acquisition_grid,
    plot_multipath_scenario,
)

OUTPUT_DIR = "/tmp/gnss_gpu_viz"


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def demo_particles():
    print("Generating particle distribution plot...")
    rng = np.random.default_rng(0)
    # Bimodal cluster
    p1 = rng.normal(loc=[0, 0, 10, 100], scale=[5, 5, 1, 50], size=(3000, 4))
    p2 = rng.normal(loc=[15, 8, 12, 120], scale=[3, 3, 1, 50], size=(2000, 4))
    particles = np.vstack([p1, p2])
    true_pos = [0, 0, 10]
    estimate = [5, 3, 11]
    fig, _ = plot_particles(particles, true_pos=true_pos, estimate=estimate,
                            colorby="height")
    save(fig, "particles.png")


def demo_skyplot():
    print("Generating sky plot...")
    rng = np.random.default_rng(1)
    n_sat = 12
    az = rng.uniform(0, 360, n_sat)
    el = rng.uniform(5, 85, n_sat)
    prns = [f"G{i+1:02d}" for i in range(n_sat)]
    cn0 = rng.uniform(20, 50, n_sat)
    fig, _ = plot_skyplot(az, el, prn_list=prns, cn0=cn0)
    save(fig, "skyplot.png")


def demo_vulnerability_map():
    print("Generating vulnerability map...")
    grid_e = np.linspace(-200, 200, 80)
    grid_n = np.linspace(-200, 200, 80)
    ee, nn = np.meshgrid(grid_e, grid_n)
    # Synthetic HDOP: high near buildings
    hdop = 1.5 + 3.0 * np.exp(-((ee - 50)**2 + nn**2) / 2000)
    hdop += 2.5 * np.exp(-((ee + 80)**2 + (nn - 60)**2) / 1500)
    buildings = [(50, 0, 40, 30), (-80, 60, 25, 35)]
    traj = np.column_stack([
        np.linspace(-150, 150, 60),
        30 * np.sin(np.linspace(0, 2 * np.pi, 60)),
    ])
    fig, _ = plot_vulnerability_map(grid_e, grid_n, hdop,
                                    metric_name="HDOP",
                                    buildings=buildings,
                                    trajectory=traj)
    save(fig, "vulnerability_map.png")


def demo_trajectory():
    print("Generating trajectory plot...")
    t = np.linspace(0, 2 * np.pi, 200)
    gt = np.column_stack([100 * np.cos(t), 60 * np.sin(t)])
    rng = np.random.default_rng(2)
    wls = gt + rng.normal(0, 4, gt.shape)
    pf = gt + rng.normal(0, 1.5, gt.shape)
    fig, _ = plot_trajectory({"WLS": wls, "Particle Filter": pf},
                             true_trajectory=gt)
    save(fig, "trajectory.png")


def demo_dop_timeline():
    print("Generating DOP timeline...")
    t = np.arange(0, 3600, 30)  # 1 hour, 30s intervals
    base = 2.0 + 0.8 * np.sin(2 * np.pi * t / 3600)
    hdop = base
    pdop = base * 1.4
    vdop = base * 0.9
    n_vis = np.clip(8 + (2 * np.sin(2 * np.pi * t / 3600)).astype(int), 4, 12)
    fig, _ = plot_dop_timeline(t, pdop=pdop, hdop=hdop, vdop=vdop,
                               n_visible=n_vis)
    save(fig, "dop_timeline.png")


def demo_positioning_error():
    print("Generating positioning error plot...")
    rng = np.random.default_rng(3)
    t = np.arange(300)
    err_wls = np.abs(rng.normal(5, 3, 300))
    err_pf = np.abs(rng.normal(2, 1, 300))
    fig, _ = plot_positioning_error(t, {"WLS": err_wls, "PF": err_pf})
    save(fig, "positioning_error.png")


def demo_spectrogram():
    print("Generating spectrogram...")
    rng = np.random.default_rng(4)
    n_frames, n_bins = 128, 256
    spec = rng.normal(-120, 5, (n_frames, n_bins))
    # Add a CW interference band
    spec[30:80, 100:110] += 40
    detections = [{"frame_start": 30, "frame_end": 80,
                   "bin_start": 100, "bin_end": 110}]
    fig, _ = plot_spectrogram(spec, sampling_freq=20e6,
                              fft_size=512, hop_size=128,
                              detections=detections)
    save(fig, "spectrogram.png")


def demo_acquisition_grid():
    print("Generating acquisition grid...")
    rng = np.random.default_rng(5)
    n_doppler, n_code = 41, 2046
    corr = rng.rayleigh(0.5, (n_doppler, n_code))
    # Add a peak
    corr[20, 800] = 15.0
    # Add some spread around peak
    for di in range(-2, 3):
        for ci in range(-5, 6):
            if 0 <= 20 + di < n_doppler and 0 <= 800 + ci < n_code:
                corr[20 + di, 800 + ci] += 5 * np.exp(-(di**2 + ci**2) / 4)
    fig, _ = plot_acquisition_grid(corr, sampling_freq=4.092e6,
                                   doppler_range=5000, doppler_step=250)
    save(fig, "acquisition_grid.png")


def demo_multipath_scenario():
    print("Generating multipath scenario...")
    buildings = [
        (60, 0, 40, 30, 50),
        (-40, 40, 25, 25, 35),
    ]
    receiver = np.array([0.0, 0.0, 1.5])
    sats = np.array([
        [200, 150, 300],
        [-200, 100, 400],
        [50, -200, 350],
        [150, -50, 250],
    ])
    los_mask = [True, True, False, False]
    reflections = np.array([
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [60, -15, 30],
        [40, -10, 25],
    ])
    fig, _ = plot_multipath_scenario(buildings, receiver, sats,
                                     los_mask=los_mask,
                                     reflections=reflections)
    save(fig, "multipath_scenario.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    demo_particles()
    demo_skyplot()
    demo_vulnerability_map()
    demo_trajectory()
    demo_dop_timeline()
    demo_positioning_error()
    demo_spectrogram()
    demo_acquisition_grid()
    demo_multipath_scenario()

    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
