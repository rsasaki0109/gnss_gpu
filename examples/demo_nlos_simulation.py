#!/usr/bin/env python3
"""CPU-only NLOS simulation demo with a 3D urban canyon model.

This is a measurement-level simulator, not an RF/IQ generator. It follows the
same practical split used by 3D-mapping-aided GNSS papers:

    synthetic 3D city + receiver trajectory + satellite sky
        -> LOS/NLOS ray classification
        -> NLOS pseudorange bias, C/N0 attenuation, and noisier code tracking
        -> compare naive SPP, robust SPP, and geometry-aware SPP

Run from the repo root:

    PYTHONPATH=python python3 examples/demo_nlos_simulation.py

The demo is deliberately CPU-only so it works before CUDA kernels are built.
The geometry and measurement model are small, deterministic, and shaped so the
same logic can later be swapped onto PLATEAU + BVH ray tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from gnss_gpu.robust_spp import robust_spp


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F

PLAIN_LS_THRESHOLD_M = 1.0e12
ROBUST_THRESHOLD_M = 12.0


@dataclass(frozen=True)
class BoxBuilding:
    """Axis-aligned building prism in local ENU coordinates."""

    center_e_m: float
    center_n_m: float
    width_e_m: float
    depth_n_m: float
    height_m: float

    @property
    def bounds_min(self) -> np.ndarray:
        return np.array(
            [
                self.center_e_m - 0.5 * self.width_e_m,
                self.center_n_m - 0.5 * self.depth_n_m,
                0.0,
            ],
            dtype=np.float64,
        )

    @property
    def bounds_max(self) -> np.ndarray:
        return np.array(
            [
                self.center_e_m + 0.5 * self.width_e_m,
                self.center_n_m + 0.5 * self.depth_n_m,
                self.height_m,
            ],
            dtype=np.float64,
        )


def llh_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * math.cos(lat) * math.cos(lon)
    y = (n + alt_m) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def enu_to_ecef_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    s_lat, c_lat = math.sin(lat), math.cos(lat)
    s_lon, c_lon = math.sin(lon), math.cos(lon)
    east = np.array([-s_lon, c_lon, 0.0], dtype=np.float64)
    north = np.array([-s_lat * c_lon, -s_lat * s_lon, c_lat], dtype=np.float64)
    up = np.array([c_lat * c_lon, c_lat * s_lon, s_lat], dtype=np.float64)
    return np.column_stack([east, north, up])


def los_unit_enu(az_deg: float, el_deg: float) -> np.ndarray:
    """Line-of-sight unit vector in ENU; azimuth is degrees clockwise from north."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return np.array(
        [
            math.cos(el) * math.sin(az),
            math.cos(el) * math.cos(az),
            math.sin(el),
        ],
        dtype=np.float64,
    )


def build_constellation(
    rx_ecef: np.ndarray,
    enu_to_ecef: np.ndarray,
    sats_azel_deg: list[tuple[float, float]],
) -> np.ndarray:
    slant_range_m = 21_000_000.0
    sats = np.empty((len(sats_azel_deg), 3), dtype=np.float64)
    for i, (az_deg, el_deg) in enumerate(sats_azel_deg):
        sats[i] = rx_ecef + slant_range_m * (enu_to_ecef @ los_unit_enu(az_deg, el_deg))
    return sats


def build_canyon() -> list[BoxBuilding]:
    """Synthetic street canyon with two building rows and a few height changes."""
    buildings: list[BoxBuilding] = []
    x_centers = [-100.0, -62.0, -24.0, 14.0, 52.0, 90.0]
    north_heights = [42.0, 58.0, 64.0, 52.0, 68.0, 46.0]
    south_heights = [50.0, 44.0, 62.0, 70.0, 55.0, 60.0]
    for x, h_n, h_s in zip(x_centers, north_heights, south_heights):
        buildings.append(BoxBuilding(x, 24.0, 30.0, 18.0, h_n))
        buildings.append(BoxBuilding(x, -24.0, 30.0, 18.0, h_s))
    return buildings


def _ray_intersects_aabb(
    origin: np.ndarray,
    direction: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> bool:
    """Return True when a forward ray intersects an axis-aligned box."""
    t_min = 0.0
    t_max = float("inf")
    eps = 1e-12

    for axis in range(3):
        d = float(direction[axis])
        o = float(origin[axis])
        if abs(d) < eps:
            if o < bounds_min[axis] or o > bounds_max[axis]:
                return False
            continue

        t1 = (bounds_min[axis] - o) / d
        t2 = (bounds_max[axis] - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_max < max(t_min, 1.0e-6):
            return False

    return t_max > 1.0e-6


def classify_los_nlos(
    rx_enu: np.ndarray,
    sats_azel_deg: list[tuple[float, float]],
    buildings: list[BoxBuilding],
) -> np.ndarray:
    """Classify each satellite as LOS using local building ray casting."""
    rx = np.asarray(rx_enu, dtype=np.float64)
    is_los = np.ones(len(sats_azel_deg), dtype=bool)
    for i, (az_deg, el_deg) in enumerate(sats_azel_deg):
        direction = los_unit_enu(az_deg, el_deg)
        for building in buildings:
            if _ray_intersects_aabb(rx, direction, building.bounds_min, building.bounds_max):
                is_los[i] = False
                break
    return is_los


def nlos_expected_bias_m(el_deg: np.ndarray, is_los: np.ndarray) -> np.ndarray:
    """Simple positive NLOS excess-delay model in metres."""
    el = np.asarray(el_deg, dtype=np.float64)
    bias = 16.0 + 1.35 * np.maximum(0.0, 35.0 - el)
    return np.where(is_los, 0.0, bias)


def simulate_observations(
    rng: np.random.Generator,
    true_ranges: np.ndarray,
    clock_bias_m: float,
    elevations_deg: np.ndarray,
    is_los: np.ndarray,
) -> dict[str, np.ndarray]:
    """Generate pseudorange, C/N0, and model metadata for one epoch."""
    expected_bias = nlos_expected_bias_m(elevations_deg, is_los)
    bias_error = rng.normal(0.0, 3.5, size=len(true_ranges))
    true_nlos_bias = np.where(is_los, 0.0, np.maximum(5.0, expected_bias + bias_error))

    los_sigma = 1.1 + 10.0 / np.maximum(elevations_deg, 12.0)
    nlos_sigma = 5.0 + 0.22 * np.maximum(0.0, 35.0 - elevations_deg)
    sigma = np.where(is_los, los_sigma, nlos_sigma)
    noise = rng.normal(0.0, sigma)

    cn0_los = 41.0 + 0.07 * elevations_deg + rng.normal(0.0, 0.8, len(true_ranges))
    cn0_nlos = 27.0 - 0.20 * np.maximum(0.0, 30.0 - elevations_deg)
    cn0_nlos += rng.normal(0.0, 1.1, len(true_ranges))
    cn0_dbhz = np.where(is_los, cn0_los, cn0_nlos)

    pseudorange = true_ranges + clock_bias_m + true_nlos_bias + noise
    return {
        "pseudorange_m": pseudorange,
        "true_nlos_bias_m": true_nlos_bias,
        "expected_nlos_bias_m": expected_bias,
        "sigma_m": sigma,
        "cn0_dbhz": cn0_dbhz,
    }


def horizontal_error_m(est_ecef: np.ndarray, true_ecef: np.ndarray, ecef_to_enu: np.ndarray) -> float:
    d_enu = ecef_to_enu @ (est_ecef - true_ecef)
    return float(math.hypot(d_enu[0], d_enu[1]))


def p50(values: np.ndarray) -> float:
    return float(np.median(values))


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def run_demo() -> dict[str, object]:
    rng = np.random.default_rng(20260606)

    origin_lat, origin_lon, origin_alt = 35.6812, 139.7671, 42.0
    origin_ecef = llh_to_ecef(origin_lat, origin_lon, origin_alt)
    enu_to_ecef = enu_to_ecef_matrix(origin_lat, origin_lon)
    ecef_to_enu = enu_to_ecef.T
    buildings = build_canyon()

    sats_azel = [
        (90.0, 72.0),
        (270.0, 70.0),
        (82.0, 64.0),
        (262.0, 62.0),
        (20.0, 82.0),
        (200.0, 80.0),
        (0.0, 15.0),
        (180.0, 16.0),
        (30.0, 18.0),
        (150.0, 17.0),
        (210.0, 18.0),
        (330.0, 16.0),
    ]
    elevations_deg = np.array([el for _, el in sats_azel], dtype=np.float64)
    sat_ecef = build_constellation(origin_ecef, enu_to_ecef, sats_azel)

    n_epochs = 80
    clock_bias_m = 1432.0
    rx_enu = np.zeros((n_epochs, 3), dtype=np.float64)
    rx_enu[:, 0] = np.linspace(-75.0, 75.0, n_epochs)
    rx_enu[:, 1] = 1.5 * np.sin(np.linspace(0.0, 2.0 * math.pi, n_epochs))
    rx_enu[:, 2] = 1.8
    rx_ecef = origin_ecef + rx_enu @ enu_to_ecef.T

    init_guess = origin_ecef + enu_to_ecef @ np.array([22.0, -18.0, 8.0])
    naive_errors: list[float] = []
    robust_errors: list[float] = []
    geom_errors: list[float] = []
    all_cn0: list[np.ndarray] = []
    all_true_bias: list[np.ndarray] = []
    all_los: list[np.ndarray] = []

    for k in range(n_epochs):
        is_los = classify_los_nlos(rx_enu[k], sats_azel, buildings)
        true_ranges = np.linalg.norm(sat_ecef - rx_ecef[k], axis=1)
        obs = simulate_observations(rng, true_ranges, clock_bias_m, elevations_deg, is_los)
        pr = obs["pseudorange_m"]

        naive = robust_spp(
            sat_ecef,
            pr,
            init_pos=init_guess,
            weight_func="huber",
            threshold=PLAIN_LS_THRESHOLD_M,
            min_satellites=5,
        )
        robust = robust_spp(
            sat_ecef,
            pr,
            init_pos=init_guess,
            weight_func="cauchy",
            threshold=ROBUST_THRESHOLD_M,
            min_satellites=5,
        )

        corrected_pr = pr - obs["expected_nlos_bias_m"]
        geom_weights = np.where(is_los, 1.0, 0.14)
        geometry_aware = robust_spp(
            sat_ecef,
            corrected_pr,
            weights=geom_weights,
            init_pos=init_guess,
            weight_func="cauchy",
            threshold=ROBUST_THRESHOLD_M,
            min_satellites=5,
        )

        if naive is not None and robust is not None and geometry_aware is not None:
            naive_errors.append(horizontal_error_m(naive, rx_ecef[k], ecef_to_enu))
            robust_errors.append(horizontal_error_m(robust, rx_ecef[k], ecef_to_enu))
            geom_errors.append(horizontal_error_m(geometry_aware, rx_ecef[k], ecef_to_enu))

        all_cn0.append(obs["cn0_dbhz"])
        all_true_bias.append(obs["true_nlos_bias_m"])
        all_los.append(is_los)

    naive_arr = np.array(naive_errors, dtype=np.float64)
    robust_arr = np.array(robust_errors, dtype=np.float64)
    geom_arr = np.array(geom_errors, dtype=np.float64)
    los_mask = np.vstack(all_los)
    true_bias = np.vstack(all_true_bias)
    cn0 = np.vstack(all_cn0)

    n_compared = int(len(geom_arr))
    robust_wins = int(np.sum(robust_arr[:n_compared] < naive_arr[:n_compared]))
    geom_wins = int(np.sum(geom_arr[:n_compared] < naive_arr[:n_compared]))

    return {
        "n_epochs": n_compared,
        "n_satellites": len(sats_azel),
        "n_buildings": len(buildings),
        "nlos_fraction": float(np.mean(~los_mask)),
        "nlos_per_epoch_mean": float(np.mean(np.sum(~los_mask, axis=1))),
        "nlos_bias_mean_m": float(np.mean(true_bias[~los_mask])),
        "los_cn0_mean_dbhz": float(np.mean(cn0[los_mask])),
        "nlos_cn0_mean_dbhz": float(np.mean(cn0[~los_mask])),
        "naive_p50_m": p50(naive_arr),
        "naive_rms_m": rms(naive_arr),
        "robust_p50_m": p50(robust_arr),
        "robust_rms_m": rms(robust_arr),
        "geometry_p50_m": p50(geom_arr),
        "geometry_rms_m": rms(geom_arr),
        "robust_wins": robust_wins,
        "geometry_wins": geom_wins,
        "model_basis": [
            "3D ray classification of direct LOS vs blocked/NLOS signals",
            "positive NLOS pseudorange excess delay",
            "lower C/N0 and larger code tracking noise for blocked signals",
            "geometry-aware down-weighting/correction as a 3DMA-GNSS proxy",
        ],
    }


def main() -> dict[str, object]:
    result = run_demo()

    print("NLOS simulation demo (CPU-only measurement-level model)")
    print("=" * 70)
    print(
        f"Scene: {result['n_epochs']} epochs, {result['n_satellites']} satellites, "
        f"{result['n_buildings']} box buildings"
    )
    print(
        f"NLOS: {100.0 * result['nlos_fraction']:.1f}% of signals "
        f"({result['nlos_per_epoch_mean']:.1f}/epoch), "
        f"mean excess delay {result['nlos_bias_mean_m']:.1f} m"
    )
    print(
        f"C/N0: LOS {result['los_cn0_mean_dbhz']:.1f} dB-Hz, "
        f"NLOS {result['nlos_cn0_mean_dbhz']:.1f} dB-Hz\n"
    )

    print(f"{'method':<30}{'P50 err':>12}{'RMS err':>12}")
    print("-" * 54)
    print(f"{'naive SPP (all signals)':<30}{result['naive_p50_m']:>10.2f} m{result['naive_rms_m']:>10.2f} m")
    print(f"{'robust SPP (Cauchy)':<30}{result['robust_p50_m']:>10.2f} m{result['robust_rms_m']:>10.2f} m")
    print(f"{'3D-aware SPP (ray mask)':<30}{result['geometry_p50_m']:>10.2f} m{result['geometry_rms_m']:>10.2f} m")
    print("-" * 54)

    robust_gain = 100.0 * (1.0 - result["robust_rms_m"] / result["naive_rms_m"])
    geom_gain = 100.0 * (1.0 - result["geometry_rms_m"] / result["naive_rms_m"])
    print(
        f"robust wins {result['robust_wins']}/{result['n_epochs']} epochs; "
        f"3D-aware wins {result['geometry_wins']}/{result['n_epochs']} epochs"
    )
    print(f"RMS gain vs naive: robust {robust_gain:.0f}%, 3D-aware {geom_gain:.0f}%")
    print("\nFor the real-mesh version, run examples/demo_plateau_nlos_simulation.py.")
    return result


if __name__ == "__main__":
    main()
