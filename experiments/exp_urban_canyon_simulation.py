#!/usr/bin/env python3
"""Urban canyon GNSS simulation with 3D building models.

Combines:
1. Urban canyon GNSS simulator — generate pseudoranges from 3D models
2. NLOS detection evaluation — compare ray-trace vs ground truth
3. Multipath delay simulation — ray-traced reflection delays

Uses local ENU coordinates for the simulation, then evaluates PF vs WLS.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


C_LIGHT = 299792458.0
GPS_L1_FREQ = 1575.42e6


def create_urban_canyon(
    length: float = 300.0,
    building_width: float = 30.0,
    building_height: float = 50.0,
    street_width: float = 20.0,
    resolution: float = 10.0,
) -> np.ndarray:
    """Create two parallel buildings forming an urban canyon (ENU coordinates)."""
    tris = []
    nx = int(length / resolution)
    nz = int(building_height / resolution)

    for side in [-1, 1]:
        x = side * street_width / 2
        x_far = x + side * building_width
        for iz in range(nz):
            z0, z1 = iz * resolution, (iz + 1) * resolution
            for iy in range(nx):
                y0, y1 = iy * resolution, (iy + 1) * resolution
                # Inner wall face (facing street)
                tris.append([[x, y0, z0], [x, y1, z0], [x, y0, z1]])
                tris.append([[x, y1, z0], [x, y1, z1], [x, y0, z1]])
        # Roof
        for iy in range(nx):
            y0, y1 = iy * resolution, (iy + 1) * resolution
            z = building_height
            tris.append([[x, y0, z], [x, y1, z], [x_far, y0, z]])
            tris.append([[x, y1, z], [x_far, y1, z], [x_far, y0, z]])

    return np.array(tris, dtype=np.float64)


def generate_satellite_constellation(
    n_satellites: int = 12,
    elevation_range: tuple[float, float] = (10.0, 85.0),
    seed: int = 42,
) -> np.ndarray:
    """Generate satellite positions in ENU (far away, ~20,200 km)."""
    rng = np.random.default_rng(seed)
    R = 20_200_000.0  # GPS orbit radius in meters

    sats = []
    for i in range(n_satellites):
        el = rng.uniform(*[math.radians(e) for e in elevation_range])
        az = rng.uniform(0, 2 * math.pi)
        e = R * math.cos(el) * math.sin(az)
        n = R * math.cos(el) * math.cos(az)
        u = R * math.sin(el)
        sats.append([e, n, u])
    return np.array(sats, dtype=np.float64)


def simulate_pseudoranges(
    rx_pos: np.ndarray,
    sat_pos: np.ndarray,
    building_model,
    clock_bias: float = 0.0,
    sigma_los: float = 3.0,
    sigma_nlos: float = 30.0,
    nlos_bias: float = 15.0,
    seed: int = 42,
) -> dict:
    """Generate simulated pseudoranges with LOS/NLOS effects."""
    rng = np.random.default_rng(seed)
    n_sat = len(sat_pos)

    true_ranges = np.linalg.norm(sat_pos - rx_pos, axis=1)
    los_flags = building_model.check_los(rx_pos, sat_pos)

    pseudoranges = np.zeros(n_sat)
    for s in range(n_sat):
        if los_flags[s]:
            # LOS: small Gaussian noise
            pseudoranges[s] = true_ranges[s] + clock_bias + rng.normal(0, sigma_los)
        else:
            # NLOS: positive bias + large noise
            pseudoranges[s] = true_ranges[s] + clock_bias + nlos_bias + rng.normal(0, sigma_nlos)

    return {
        "pseudoranges": pseudoranges,
        "true_ranges": true_ranges,
        "los_flags": los_flags,
        "n_los": int(los_flags.sum()),
        "n_nlos": int((~los_flags).sum()),
    }


def run_simulation(
    n_epochs: int = 300,
    n_satellites: int = 12,
    building_height: float = 50.0,
    street_width: float = 20.0,
    n_particles: int = 10000,
    speed: float = 5.0,  # m/s (~18 km/h walking/slow vehicle)
    seed: int = 42,
) -> dict:
    """Run full urban canyon simulation: generate data + evaluate PF vs WLS."""
    from gnss_gpu import BuildingModel, wls_position
    from gnss_gpu import ParticleFilterDevice

    print(f"  Canyon: height={building_height}m, street={street_width}m")
    print(f"  Satellites: {n_satellites}, Epochs: {n_epochs}, Speed: {speed} m/s")

    # Create canyon
    canyon_tris = create_urban_canyon(
        length=n_epochs * speed + 50,
        building_height=building_height,
        street_width=street_width,
    )
    model = BuildingModel(canyon_tris)
    print(f"  Canyon model: {len(canyon_tris)} triangles")

    # Satellite constellation (fixed for simplicity)
    sats = generate_satellite_constellation(n_satellites, seed=seed)

    # True trajectory: straight line through canyon center
    dt = 1.0  # 1 Hz
    true_positions = np.zeros((n_epochs, 3))
    for i in range(n_epochs):
        true_positions[i] = [0, 25 + i * speed * dt, 1.5]  # center of street, 1.5m height

    # Generate pseudoranges
    all_pr = []
    all_los = []
    clock_bias = 50000.0  # 50 km clock bias (typical)

    for i in range(n_epochs):
        result = simulate_pseudoranges(
            true_positions[i], sats, model,
            clock_bias=clock_bias,
            seed=seed + i,
        )
        all_pr.append(result["pseudoranges"])
        all_los.append(result["los_flags"])

    # === 1. NLOS Detection Evaluation ===
    total_los = sum(r.sum() for r in all_los)
    total_nlos = sum((~r).sum() for r in all_los)
    print(f"  NLOS stats: {total_los} LOS, {total_nlos} NLOS ({100*total_nlos/(total_los+total_nlos):.1f}%)")

    # === 2. WLS Positioning ===
    wls_positions = np.zeros((n_epochs, 3))
    wls_errors = []
    for i in range(n_epochs):
        w = np.ones(n_satellites)
        try:
            res = wls_position(sats, all_pr[i], w)
            wls_positions[i] = np.array(res[0])[:3]
        except Exception:
            if i > 0:
                wls_positions[i] = wls_positions[i - 1]
        err = np.linalg.norm(wls_positions[i] - true_positions[i])
        wls_errors.append(err)

    wls_errors = np.array(wls_errors)
    print(f"  WLS: RMS={np.sqrt(np.mean(wls_errors**2)):.2f}m  P50={np.median(wls_errors):.2f}m  P95={np.percentile(wls_errors, 95):.2f}m")

    # === 3. PF Positioning ===
    pf = ParticleFilterDevice(
        n_particles=n_particles,
        sigma_pos=1.0,
        sigma_cb=300.0,
        sigma_pr=5.0,
        resampling="megopolis",
        seed=42,
    )
    pf.initialize(
        true_positions[0] + np.array([5, 5, 0]),  # slightly off init
        clock_bias=clock_bias,
        spread_pos=20.0,
        spread_cb=1000.0,
    )

    pf_positions = np.zeros((n_epochs, 3))
    pf_errors = []
    for i in range(n_epochs):
        velocity = np.array([0, speed, 0]) if i > 0 else None
        pf.predict(velocity=velocity, dt=dt)
        w = np.ones(n_satellites)
        pf.update(sats, all_pr[i], weights=w)
        est = pf.estimate()
        pf_positions[i] = est[:3]
        err = np.linalg.norm(pf_positions[i] - true_positions[i])
        pf_errors.append(err)

    pf_errors = np.array(pf_errors)
    print(f"  PF:  RMS={np.sqrt(np.mean(pf_errors**2)):.2f}m  P50={np.median(pf_errors):.2f}m  P95={np.percentile(pf_errors, 95):.2f}m")
    print(f"  PF improvement: {(1 - np.sqrt(np.mean(pf_errors**2)) / np.sqrt(np.mean(wls_errors**2))) * 100:.1f}%")

    return {
        "wls_errors": wls_errors,
        "pf_errors": pf_errors,
        "true_positions": true_positions,
        "wls_positions": wls_positions,
        "pf_positions": pf_positions,
        "los_flags": all_los,
        "n_satellites": n_satellites,
        "building_height": building_height,
        "street_width": street_width,
    }


def plot_results(results: list[dict], output_path: Path) -> None:
    """Plot simulation results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Error time series (first scenario)
    ax = axes[0, 0]
    r = results[0]
    ax.plot(r["wls_errors"], alpha=0.7, label=f'WLS (RMS={np.sqrt(np.mean(r["wls_errors"]**2)):.1f}m)')
    ax.plot(r["pf_errors"], alpha=0.7, label=f'PF (RMS={np.sqrt(np.mean(r["pf_errors"]**2)):.1f}m)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error [m]")
    ax.set_title(f'Error Timeline (H={r["building_height"]}m, W={r["street_width"]}m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Canyon height sweep
    ax = axes[0, 1]
    heights = [r["building_height"] for r in results]
    wls_rms = [np.sqrt(np.mean(r["wls_errors"]**2)) for r in results]
    pf_rms = [np.sqrt(np.mean(r["pf_errors"]**2)) for r in results]
    ax.plot(heights, wls_rms, "o-", label="WLS", color="#3b82f6")
    ax.plot(heights, pf_rms, "s-", label="PF", color="#ef4444")
    ax.set_xlabel("Building Height [m]")
    ax.set_ylabel("RMS Error [m]")
    ax.set_title("Effect of Canyon Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. NLOS percentage vs error
    ax = axes[1, 0]
    nlos_pcts = []
    for r in results:
        total = sum(len(f) for f in r["los_flags"])
        nlos = sum(int((~f).sum()) for f in r["los_flags"])
        nlos_pcts.append(100 * nlos / total)
    ax.scatter(nlos_pcts, wls_rms, s=80, label="WLS", color="#3b82f6", marker="o")
    ax.scatter(nlos_pcts, pf_rms, s=80, label="PF", color="#ef4444", marker="s")
    ax.set_xlabel("NLOS Rate [%]")
    ax.set_ylabel("RMS Error [m]")
    ax.set_title("NLOS Rate vs Positioning Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Trajectory top-view (first scenario)
    ax = axes[1, 1]
    r = results[0]
    ax.plot(r["true_positions"][:, 0], r["true_positions"][:, 1], "k-", linewidth=2, label="Ground Truth")
    ax.plot(r["wls_positions"][:, 0], r["wls_positions"][:, 1], ".-", alpha=0.5, markersize=2, label="WLS")
    ax.plot(r["pf_positions"][:, 0], r["pf_positions"][:, 1], ".-", alpha=0.5, markersize=2, label="PF")
    # Draw canyon walls
    sw = r["street_width"]
    ax.axvline(-sw/2, color="gray", linewidth=3, alpha=0.5)
    ax.axvline(sw/2, color="gray", linewidth=3, alpha=0.5)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Top View (canyon walls in gray)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Urban Canyon GNSS Simulation: PF vs WLS", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def main() -> None:
    print("Urban Canyon GNSS Simulation")
    print("=" * 50)

    results = []
    for height in [20, 40, 60, 80, 100]:
        print(f"\n--- Building height = {height}m ---")
        r = run_simulation(
            n_epochs=300,
            n_satellites=12,
            building_height=height,
            street_width=20.0,
            n_particles=10000,
        )
        results.append(r)

    output = RESULTS_DIR / "paper_assets" / "sim_urban_canyon.png"
    plot_results(results, output)

    print("\n" + "=" * 50)
    print("Summary:")
    print(f"{'Height':>8s} {'NLOS%':>6s} {'WLS RMS':>9s} {'PF RMS':>8s} {'PF gain':>8s}")
    for r in results:
        total = sum(len(f) for f in r["los_flags"])
        nlos = sum(int((~f).sum()) for f in r["los_flags"])
        nlos_pct = 100 * nlos / total
        wls_rms = np.sqrt(np.mean(r["wls_errors"]**2))
        pf_rms = np.sqrt(np.mean(r["pf_errors"]**2))
        gain = (1 - pf_rms / wls_rms) * 100
        print(f'{r["building_height"]:>7.0f}m {nlos_pct:>5.1f}% {wls_rms:>8.2f}m {pf_rms:>7.2f}m {gain:>7.1f}%')


if __name__ == "__main__":
    main()
