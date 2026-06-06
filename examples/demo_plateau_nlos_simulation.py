#!/usr/bin/env python3
"""PLATEAU-backed NLOS measurement simulation demo.

This is the real-mesh counterpart to ``demo_nlos_simulation.py``. It loads the
small PLATEAU CityGML sample shipped in ``data/sample_plateau.gml``, ray-casts
synthetic satellite paths against the building triangles, and compares:

* naive SPP, which trusts every pseudorange;
* robust SPP, which down-weights large residuals;
* PLATEAU-aware SPP, which uses the ray mask to correct and down-weight NLOS.

Run from the repo root:

    PYTHONPATH=python python3 examples/demo_plateau_nlos_simulation.py

If the native BVH extension is available, the demo uses it. Otherwise it falls
back to a pure-Python triangle ray-cast so the measurement model stays testable
without a CUDA build.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from gnss_gpu.io.plateau import PlateauLoader
from gnss_gpu.robust_spp import robust_spp


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F

PLAIN_LS_THRESHOLD_M = 1.0e12
ROBUST_THRESHOLD_M = 12.0
TOKYO_GEOID_N_M = 36.7


def ecef_to_llh(ecef: np.ndarray) -> tuple[float, float, float]:
    x, y, z = np.asarray(ecef, dtype=np.float64).ravel()
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        lat = math.atan2(z + WGS84_E2 * n * sin_lat, p)
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), float(alt)


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


def _ray_triangle_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> bool:
    eps = 1.0e-9
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = float(np.dot(edge1, h))
    if -eps < a < eps:
        return False
    f = 1.0 / a
    s = origin - v0
    u = f * float(np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    v = f * float(np.dot(direction, q))
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * float(np.dot(edge2, q))
    return t > eps and t < 1.0


def _check_los_cpu(rx_ecef: np.ndarray, sat_ecef: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    is_los = np.ones(sat_ecef.shape[0], dtype=bool)
    for j, sat in enumerate(sat_ecef):
        direction = sat - rx_ecef
        for tri in triangles:
            if _ray_triangle_intersect(rx_ecef, direction, tri[0], tri[1], tri[2]):
                is_los[j] = False
                break
    return is_los


def load_plateau_triangles(gml_path: Path) -> np.ndarray:
    """Load the sample PLATEAU mesh with an explicit Tokyo geoid correction."""
    loader = PlateauLoader(zone=9, geoid_correction=TOKYO_GEOID_N_M)
    model = loader.load_citygml(gml_path)
    triangles = np.asarray(model.triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or triangles.size == 0:
        raise ValueError(f"no triangles loaded from {gml_path}")
    return triangles


def build_local_frame(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local origin ECEF, ENU-to-ECEF rotation, and vertices in local ENU."""
    verts = triangles.reshape(-1, 3)
    origin = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    lat_deg, lon_deg, _alt_m = ecef_to_llh(origin)
    enu_to_ecef = enu_to_ecef_matrix(lat_deg, lon_deg)
    verts_enu = (verts - origin) @ enu_to_ecef
    return origin, enu_to_ecef, verts_enu


def build_satellites(
    origin_ecef: np.ndarray,
    enu_to_ecef: np.ndarray,
    sats_azel_deg: list[tuple[float, float]],
) -> np.ndarray:
    slant_range_m = 21_000_000.0
    return np.array(
        [
            origin_ecef + slant_range_m * (enu_to_ecef @ los_unit_enu(az, el))
            for az, el in sats_azel_deg
        ],
        dtype=np.float64,
    )


def default_satellite_az_el_deg() -> list[tuple[float, float]]:
    return [
        (90.0, 72.0),
        (270.0, 70.0),
        (0.0, 70.0),
        (180.0, 68.0),
        (45.0, 38.0),
        (135.0, 35.0),
        (225.0, 36.0),
        (315.0, 34.0),
        (80.0, 18.0),
        (100.0, 15.0),
        (260.0, 17.0),
        (280.0, 15.0),
        (0.0, 20.0),
        (180.0, 22.0),
    ]


def classify_los_plateau(
    rx_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Classify LOS/NLOS with native BVH when available, else CPU triangles."""
    try:
        from gnss_gpu.bvh import BVHAccelerator

        bvh = BVHAccelerator(triangles)
        masks = np.empty((rx_ecef.shape[0], sat_ecef.shape[0]), dtype=bool)
        for i, rx in enumerate(rx_ecef):
            masks[i] = np.asarray(bvh.check_los(rx, sat_ecef), dtype=bool)
        return masks, "native BVH"
    except Exception:
        masks = np.empty((rx_ecef.shape[0], sat_ecef.shape[0]), dtype=bool)
        for i, rx in enumerate(rx_ecef):
            masks[i] = _check_los_cpu(rx, sat_ecef, triangles)
        return masks, "CPU triangle ray-cast"


def nlos_expected_bias_m(elevations_deg: np.ndarray, is_los: np.ndarray) -> np.ndarray:
    bias = 18.0 + 1.25 * np.maximum(0.0, 35.0 - elevations_deg)
    return np.where(is_los, 0.0, bias)


def simulate_pseudorange_epoch(
    rng: np.random.Generator,
    rx_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    elevations_deg: np.ndarray,
    is_los: np.ndarray,
    clock_bias_m: float,
) -> dict[str, np.ndarray]:
    ranges = np.linalg.norm(sat_ecef - rx_ecef, axis=1)
    expected_bias = nlos_expected_bias_m(elevations_deg, is_los)
    true_bias = np.where(
        is_los,
        0.0,
        np.maximum(5.0, expected_bias + rng.normal(0.0, 3.5, len(elevations_deg))),
    )
    sigma = np.where(
        is_los,
        1.4 + 8.0 / np.maximum(elevations_deg, 12.0),
        5.0 + 0.20 * np.maximum(0.0, 35.0 - elevations_deg),
    )
    cn0_los = 41.0 + 0.06 * elevations_deg + rng.normal(0.0, 0.8, len(elevations_deg))
    cn0_nlos = 27.0 - 0.20 * np.maximum(0.0, 30.0 - elevations_deg)
    cn0_nlos += rng.normal(0.0, 1.1, len(elevations_deg))
    pseudorange = ranges + clock_bias_m + true_bias + rng.normal(0.0, sigma)
    return {
        "pseudorange_m": pseudorange,
        "expected_nlos_bias_m": expected_bias,
        "true_nlos_bias_m": true_bias,
        "cn0_dbhz": np.where(is_los, cn0_los, cn0_nlos),
    }


def horizontal_error_m(
    est_ecef: np.ndarray,
    true_ecef: np.ndarray,
    ecef_to_enu: np.ndarray,
) -> float:
    d_enu = ecef_to_enu @ (est_ecef - true_ecef)
    return float(math.hypot(d_enu[0], d_enu[1]))


def _p50(values: np.ndarray) -> float:
    return float(np.median(values))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def run_demo(
    gml_path: Path | None = None,
    *,
    include_details: bool = False,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parent.parent
    if gml_path is None:
        gml_path = repo_root / "data" / "sample_plateau.gml"

    rng = np.random.default_rng(20260606)
    triangles = load_plateau_triangles(gml_path)
    origin_ecef, enu_to_ecef, verts_enu = build_local_frame(triangles)
    ecef_to_enu = enu_to_ecef.T
    ground_z_m = float(verts_enu[:, 2].min() + 1.8)

    sats_azel = default_satellite_az_el_deg()
    elevations_deg = np.array([el for _az, el in sats_azel], dtype=np.float64)
    sat_ecef = build_satellites(origin_ecef, enu_to_ecef, sats_azel)

    n_requested_epochs = 70
    rx_enu = np.zeros((n_requested_epochs, 3), dtype=np.float64)
    rx_enu[:, 0] = np.linspace(-55.0, 55.0, n_requested_epochs)
    rx_enu[:, 1] = -10.0
    rx_enu[:, 2] = ground_z_m
    rx_ecef = origin_ecef + rx_enu @ enu_to_ecef.T

    los_mask, ray_source = classify_los_plateau(rx_ecef, sat_ecef, triangles)
    clock_bias_m = 1432.0
    init_guess = origin_ecef + enu_to_ecef @ np.array([15.0, -10.0, 5.0])

    naive_errors: list[float] = []
    robust_errors: list[float] = []
    plateau_errors: list[float] = []
    true_biases: list[np.ndarray] = []
    cn0_values: list[np.ndarray] = []
    solved_los_masks: list[np.ndarray] = []
    solved_epoch_indices: list[int] = []

    for k in range(n_requested_epochs):
        is_los = los_mask[k]
        obs = simulate_pseudorange_epoch(
            rng,
            rx_ecef[k],
            sat_ecef,
            elevations_deg,
            is_los,
            clock_bias_m,
        )
        pr = obs["pseudorange_m"]
        expected_bias = obs["expected_nlos_bias_m"]

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
        plateau = robust_spp(
            sat_ecef,
            pr - expected_bias,
            weights=np.where(is_los, 1.0, 0.12),
            init_pos=init_guess,
            weight_func="cauchy",
            threshold=ROBUST_THRESHOLD_M,
            min_satellites=5,
        )
        if naive is None or robust is None or plateau is None:
            continue

        naive_errors.append(horizontal_error_m(naive, rx_ecef[k], ecef_to_enu))
        robust_errors.append(horizontal_error_m(robust, rx_ecef[k], ecef_to_enu))
        plateau_errors.append(horizontal_error_m(plateau, rx_ecef[k], ecef_to_enu))
        true_biases.append(obs["true_nlos_bias_m"])
        cn0_values.append(obs["cn0_dbhz"])
        solved_los_masks.append(is_los)
        solved_epoch_indices.append(k)

    naive_arr = np.asarray(naive_errors, dtype=np.float64)
    robust_arr = np.asarray(robust_errors, dtype=np.float64)
    plateau_arr = np.asarray(plateau_errors, dtype=np.float64)
    used_los = np.vstack(solved_los_masks)
    true_bias = np.vstack(true_biases)
    cn0 = np.vstack(cn0_values)
    result: dict[str, object] = {
        "gml_path": str(gml_path),
        "ray_source": ray_source,
        "n_requested_epochs": n_requested_epochs,
        "n_epochs": int(len(naive_arr)),
        "n_satellites": int(len(sats_azel)),
        "n_triangles": int(triangles.shape[0]),
        "mesh_enu_min_m": verts_enu.min(axis=0).tolist(),
        "mesh_enu_max_m": verts_enu.max(axis=0).tolist(),
        "nlos_fraction": float(np.mean(~used_los)),
        "nlos_per_epoch_mean": float(np.mean(np.sum(~used_los, axis=1))),
        "nlos_bias_mean_m": float(np.mean(true_bias[~used_los])),
        "los_cn0_mean_dbhz": float(np.mean(cn0[used_los])),
        "nlos_cn0_mean_dbhz": float(np.mean(cn0[~used_los])),
        "naive_p50_m": _p50(naive_arr),
        "naive_rms_m": _rms(naive_arr),
        "robust_p50_m": _p50(robust_arr),
        "robust_rms_m": _rms(robust_arr),
        "plateau_p50_m": _p50(plateau_arr),
        "plateau_rms_m": _rms(plateau_arr),
        "robust_wins": int(np.sum(robust_arr < naive_arr)),
        "plateau_wins": int(np.sum(plateau_arr < naive_arr)),
    }
    if include_details:
        triangles_enu = ((triangles.reshape(-1, 3) - origin_ecef) @ enu_to_ecef).reshape(
            triangles.shape
        )
        solved_idx = np.asarray(solved_epoch_indices, dtype=np.int64)
        result["details"] = {
            "triangles_enu": triangles_enu.tolist(),
            "all_rx_enu": rx_enu.tolist(),
            "all_epoch_index": list(range(n_requested_epochs)),
            "all_los_mask": los_mask.tolist(),
            "all_nlos_count": np.sum(~los_mask, axis=1).astype(int).tolist(),
            "rx_enu": rx_enu[solved_idx].tolist(),
            "epoch_index": solved_idx.tolist(),
            "satellite_az_el_deg": [list(azel) for azel in sats_azel],
            "los_mask": used_los.tolist(),
            "nlos_count": np.sum(~used_los, axis=1).astype(int).tolist(),
            "naive_error_m": naive_arr.tolist(),
            "robust_error_m": robust_arr.tolist(),
            "plateau_error_m": plateau_arr.tolist(),
        }
    return result


def main() -> dict[str, object]:
    result = run_demo()

    print("PLATEAU NLOS simulation demo")
    print("=" * 70)
    print(f"Mesh: {result['n_triangles']} triangles from {result['gml_path']}")
    print(f"Ray source: {result['ray_source']}")
    print(
        f"Scene: {result['n_epochs']}/{result['n_requested_epochs']} solved epochs, "
        f"{result['n_satellites']} satellites"
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

    print(f"{'method':<32}{'P50 err':>12}{'RMS err':>12}")
    print("-" * 56)
    print(f"{'naive SPP (all signals)':<32}{result['naive_p50_m']:>10.2f} m{result['naive_rms_m']:>10.2f} m")
    print(f"{'robust SPP (Cauchy)':<32}{result['robust_p50_m']:>10.2f} m{result['robust_rms_m']:>10.2f} m")
    print(f"{'PLATEAU-aware SPP':<32}{result['plateau_p50_m']:>10.2f} m{result['plateau_rms_m']:>10.2f} m")
    print("-" * 56)

    plateau_gain = 100.0 * (1.0 - result["plateau_rms_m"] / result["naive_rms_m"])
    print(
        f"PLATEAU-aware wins {result['plateau_wins']}/{result['n_epochs']} epochs; "
        f"RMS gain vs naive {plateau_gain:.0f}%"
    )
    return result


if __name__ == "__main__":
    main()
