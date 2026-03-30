"""Benchmark GPU-accelerated vulnerability map computation."""

import time
import numpy as np


def _make_satellite_constellation(n_sat=24):
    """Generate a synthetic GPS-like constellation in ECEF."""
    rng = np.random.default_rng(42)
    orbit_radius = 26559.7e3  # GPS orbital radius in meters

    sat_ecef = []
    for i in range(n_sat):
        # Distribute satellites across orbital planes
        plane = i % 6
        slot = i // 6
        raan = np.radians(plane * 60)
        mean_anomaly = np.radians(slot * 90 + plane * 15)

        inclination = np.radians(55.0)

        # Position in orbital plane
        x_orb = orbit_radius * np.cos(mean_anomaly)
        y_orb = orbit_radius * np.sin(mean_anomaly)

        # Rotate by inclination and RAAN
        x = (x_orb * np.cos(raan) -
             y_orb * np.cos(inclination) * np.sin(raan))
        y = (x_orb * np.sin(raan) +
             y_orb * np.cos(inclination) * np.cos(raan))
        z = y_orb * np.sin(inclination)

        sat_ecef.append([x, y, z])

    return np.array(sat_ecef, dtype=np.float64)


def benchmark_skyplot(grid_configs=None, n_iter=5):
    """Benchmark vulnerability map at various grid sizes.

    Parameters
    ----------
    grid_configs : list of tuples
        Each tuple is (grid_size_m, resolution_m) giving effective n_side.
    n_iter : int
        Number of iterations for timing.
    """
    if grid_configs is None:
        # (grid_size_m, resolution_m) -> approximate n_side
        # 10x10:   grid_size=45, resolution=5
        # 50x50:   grid_size=245, resolution=5
        # 100x100: grid_size=495, resolution=5
        # 200x200: grid_size=995, resolution=5
        grid_configs = [
            (45, 5, "10x10"),
            (245, 5, "50x50"),
            (495, 5, "100x100"),
            (995, 5, "200x200"),
        ]

    try:
        from gnss_gpu import VulnerabilityMap
    except (ImportError, RuntimeError) as e:
        print(f"[SKIP] VulnerabilityMap not available: {e}")
        return {}

    # Tokyo Station origin
    origin_lla = (35.6812, 139.7671, 5.0)
    sat_ecef = _make_satellite_constellation(24)
    results = {}

    print("=" * 80)
    print("Vulnerability Map (Skyplot) Benchmark")
    print("=" * 80)
    header = (f"{'Grid':>10} | {'N Points':>10} | {'Mean (ms)':>12} | "
              f"{'Std (ms)':>10} | {'Throughput':>18}")
    print(header)
    print("-" * 80)

    for grid_size, resolution, label in grid_configs:
        # Warm-up
        vm = VulnerabilityMap(origin_lla, grid_size_m=grid_size,
                              resolution_m=resolution)
        n_points = vm.n_grid

        try:
            vm.evaluate(sat_ecef, elevation_mask_deg=10.0)
        except RuntimeError as e:
            print(f"{label:>10} | [SKIP] {e}")
            continue

        times = []
        for _ in range(n_iter):
            vm2 = VulnerabilityMap(origin_lla, grid_size_m=grid_size,
                                   resolution_m=resolution)
            t0 = time.perf_counter()
            vm2.evaluate(sat_ecef, elevation_mask_deg=10.0)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        mean_t = np.mean(times)
        std_t = np.std(times)
        throughput = n_points / mean_t

        if throughput >= 1e6:
            tp_str = f"{throughput / 1e6:.2f} M pts/s"
        elif throughput >= 1e3:
            tp_str = f"{throughput / 1e3:.2f} K pts/s"
        else:
            tp_str = f"{throughput:.1f} pts/s"

        print(f"{label:>10} | {n_points:>10,} | {mean_t * 1e3:>12.3f} | "
              f"{std_t * 1e3:>10.3f} | {tp_str:>18}")

        results[label] = {
            "n_points": n_points,
            "mean_s": mean_t,
            "std_s": std_t,
            "throughput_pts_s": throughput,
        }

    print("-" * 80)
    return results


if __name__ == "__main__":
    benchmark_skyplot()
