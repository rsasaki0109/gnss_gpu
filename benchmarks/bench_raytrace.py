"""Benchmark GPU-accelerated ray tracing for NLOS detection."""

import time
import numpy as np


def _make_building_mesh(n_buildings):
    """Create a mesh of box buildings around a receiver location.

    Each box building has 12 triangles. Returns total triangle array.

    Parameters
    ----------
    n_buildings : int
        Number of box buildings to create.

    Returns
    -------
    triangles : ndarray, shape (n_buildings * 12, 3, 3)
    """
    rng = np.random.default_rng(42)
    all_triangles = []

    for i in range(n_buildings):
        # Place buildings in a grid pattern around origin
        angle = 2 * np.pi * i / n_buildings
        dist = 50.0 + rng.uniform(0, 200)
        cx = dist * np.cos(angle)
        cy = dist * np.sin(angle)
        cz = rng.uniform(10, 50)

        width = rng.uniform(10, 30)
        depth = rng.uniform(10, 30)
        height = rng.uniform(20, 100)

        hw, hd, hh = width / 2, depth / 2, height / 2

        v = np.array([
            [cx - hw, cy - hd, cz - hh],
            [cx + hw, cy - hd, cz - hh],
            [cx + hw, cy + hd, cz - hh],
            [cx - hw, cy + hd, cz - hh],
            [cx - hw, cy - hd, cz + hh],
            [cx + hw, cy - hd, cz + hh],
            [cx + hw, cy + hd, cz + hh],
            [cx - hw, cy + hd, cz + hh],
        ], dtype=np.float64)

        faces = [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [2, 7, 3], [2, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ]

        for f in faces:
            all_triangles.append([v[f[0]], v[f[1]], v[f[2]]])

    return np.array(all_triangles, dtype=np.float64)


def _make_satellites(n_sat):
    """Generate satellite positions in local ENU-like coordinates above the receiver."""
    rng = np.random.default_rng(123)
    sat = []
    for i in range(n_sat):
        el = rng.uniform(10, 80)  # elevation degrees
        az = rng.uniform(0, 360)  # azimuth degrees
        dist = 20200e3  # GPS orbit distance

        el_rad = np.radians(el)
        az_rad = np.radians(az)
        x = dist * np.cos(el_rad) * np.sin(az_rad)
        y = dist * np.cos(el_rad) * np.cos(az_rad)
        z = dist * np.sin(el_rad)
        sat.append([x, y, z])

    return np.array(sat, dtype=np.float64)


def benchmark_raytrace(n_triangle_configs=None, n_sat_configs=None, n_iter=10):
    """Benchmark ray tracing LOS check.

    Parameters
    ----------
    n_triangle_configs : list of int
        Number of triangles (in multiples of 12, i.e., number of buildings).
    n_sat_configs : list of int
        Number of satellites.
    n_iter : int
        Number of iterations for timing.
    """
    if n_triangle_configs is None:
        n_triangle_configs = [12, 100, 1000, 10000]
    if n_sat_configs is None:
        n_sat_configs = [8, 16, 32]

    try:
        from gnss_gpu import BuildingModel
    except (ImportError, RuntimeError) as e:
        print(f"[SKIP] BuildingModel not available: {e}")
        return {}

    rx_ecef = np.array([0.0, 0.0, 1.5])  # Receiver at local origin
    results = {}

    print("=" * 90)
    print("Ray Tracing (LOS Check) Benchmark")
    print("=" * 90)
    header = (f"{'N Triangles':>12} | {'N Sats':>8} | {'Mean (ms)':>12} | "
              f"{'Std (ms)':>10} | {'Checks/s':>16}")
    print(header)
    print("-" * 90)

    for n_tri in n_triangle_configs:
        n_buildings = max(1, n_tri // 12)
        triangles = _make_building_mesh(n_buildings)
        actual_n_tri = len(triangles)

        try:
            model = BuildingModel(triangles)
        except Exception as e:
            print(f"{actual_n_tri:>12} | [SKIP] {e}")
            continue

        for n_sat in n_sat_configs:
            sat_ecef = _make_satellites(n_sat)

            # Warm-up
            try:
                model.check_los(rx_ecef, sat_ecef)
            except Exception as e:
                print(f"{actual_n_tri:>12} | {n_sat:>8} | [SKIP] {e}")
                continue

            times = []
            for _ in range(n_iter):
                t0 = time.perf_counter()
                model.check_los(rx_ecef, sat_ecef)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            mean_t = np.mean(times)
            std_t = np.std(times)
            # Total checks = n_satellites * n_triangles (each ray checked against each triangle)
            total_checks = n_sat * actual_n_tri
            checks_per_s = total_checks / mean_t

            if checks_per_s >= 1e9:
                tp_str = f"{checks_per_s / 1e9:.2f} G checks/s"
            elif checks_per_s >= 1e6:
                tp_str = f"{checks_per_s / 1e6:.2f} M checks/s"
            elif checks_per_s >= 1e3:
                tp_str = f"{checks_per_s / 1e3:.2f} K checks/s"
            else:
                tp_str = f"{checks_per_s:.1f} checks/s"

            print(f"{actual_n_tri:>12,} | {n_sat:>8} | {mean_t * 1e3:>12.3f} | "
                  f"{std_t * 1e3:>10.3f} | {tp_str:>16}")

            results[(actual_n_tri, n_sat)] = {
                "mean_s": mean_t,
                "std_s": std_t,
                "total_checks": total_checks,
                "checks_per_s": checks_per_s,
            }

        print("-" * 90)

    return results


if __name__ == "__main__":
    benchmark_raytrace()
