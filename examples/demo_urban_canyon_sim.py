"""Zero-setup demo: why robust GNSS positioning wins in the urban canyon.

This is the recommended first thing to run after cloning. It needs **no GPU, no
CUDA build, and no downloaded data** — only NumPy and the pure-Python solver that
ships with the package:

    PYTHONPATH=python python3 examples/demo_urban_canyon_sim.py

What it shows
-------------
In a dense city, tall buildings block the direct path to low-elevation
satellites, so the receiver instead locks onto a *reflected* signal. That
reflection travels farther, which inflates the measured range by tens of metres
(NLOS multipath). A plain least-squares fix trusts every satellite equally and
gets dragged off by those biased measurements.

This demo simulates that exact situation along a short driving segment and
solves each epoch two ways with the SAME inputs:

* ``naive WLS``   — ordinary least squares (every satellite trusted equally)
* ``robust SPP``  — IRLS with a Cauchy kernel that down-weights NLOS outliers
                    (``gnss_gpu.robust_spp.robust_spp``)

The robust solver is the real package code; the only thing simulated is the
measurement environment. Robust positioning is the core idea behind the GPU
particle-filter stack that beats RTKLIB demo5 on UrbanNav (see the README).
"""

from __future__ import annotations

import math

import numpy as np

from gnss_gpu.robust_spp import robust_spp


def geometric_ranges(rx_ecef: np.ndarray, sats_ecef: np.ndarray) -> np.ndarray:
    """Euclidean receiver-to-satellite ranges [m].

    This matches the range model inside ``robust_spp`` exactly, so the simulated
    measurements and the solver agree. (Earth-rotation / Sagnac and relativistic
    terms are omitted here to keep the demo self-contained; the production SPP in
    ``gnss_gpu.spp`` applies the full correction chain.)
    """
    return np.linalg.norm(np.asarray(sats_ecef) - np.asarray(rx_ecef), axis=1)

# WGS84 constants
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F * WGS84_F

# A generous threshold makes the Cauchy/Huber kernel inert, recovering plain
# least squares. We reuse the same solver for both methods so the comparison
# isolates the robust weighting, not implementation differences.
PLAIN_LS_THRESHOLD_M = 1.0e12
ROBUST_THRESHOLD_M = 15.0


def llh_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Geodetic latitude/longitude/height -> ECEF position [m]."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * math.cos(lat) * math.cos(lon)
    y = (n + alt_m) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def enu_to_ecef_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Columns are the local East/North/Up axes expressed in ECEF."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    s_lat, c_lat = math.sin(lat), math.cos(lat)
    s_lon, c_lon = math.sin(lon), math.cos(lon)
    east = np.array([-s_lon, c_lon, 0.0])
    north = np.array([-s_lat * c_lon, -s_lat * s_lon, c_lat])
    up = np.array([c_lat * c_lon, c_lat * s_lon, s_lat])
    return np.column_stack([east, north, up])


def build_constellation(
    rx_ecef: np.ndarray, enu: np.ndarray, sats_azel_deg: list[tuple[float, float]]
) -> np.ndarray:
    """Place satellites at a typical slant range along given (azimuth, elevation)."""
    slant_range_m = 21_000_000.0  # ~MEO slant range; exact value is not critical here
    sats = np.empty((len(sats_azel_deg), 3), dtype=np.float64)
    for i, (az_deg, el_deg) in enumerate(sats_azel_deg):
        az, el = math.radians(az_deg), math.radians(el_deg)
        # ENU line-of-sight unit vector toward the satellite.
        los_enu = np.array(
            [math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)]
        )
        sats[i] = rx_ecef + slant_range_m * (enu @ los_enu)
    return sats


def horizontal_error_m(est_ecef: np.ndarray, true_ecef: np.ndarray, enu: np.ndarray) -> float:
    """2D (East/North) error magnitude between two ECEF positions [m]."""
    d_enu = enu.T @ (est_ecef - true_ecef)
    return float(math.hypot(d_enu[0], d_enu[1]))


def main() -> None:
    rng = np.random.default_rng(20260525)

    # --- Scene: a receiver driving east through a dense urban canyon ---------
    start_lat, start_lon, start_alt = 35.6804, 139.7690, 45.0  # central Tokyo
    enu = enu_to_ecef_matrix(start_lat, start_lon)
    east_axis_ecef = enu @ np.array([1.0, 0.0, 0.0])
    start_ecef = llh_to_ecef(start_lat, start_lon, start_alt)

    n_epochs = 60          # 60 s at 1 Hz
    speed_mps = 5.0        # ~18 km/h city driving
    true_clock_bias_m = 1234.5
    code_noise_sigma_m = 1.5

    # A fixed sky geometry with a healthy majority of clean, well-spread
    # satellites and a minority of low-elevation lines of sight that the city
    # blocks. (az_from_north_deg, el_deg)
    sats_azel = [
        (25.0, 70.0), (80.0, 62.0), (150.0, 55.0), (210.0, 68.0), (290.0, 58.0),
        (330.0, 48.0), (45.0, 42.0), (110.0, 38.0), (180.0, 45.0), (260.0, 40.0),
        (70.0, 15.0), (200.0, 18.0), (310.0, 12.0),
    ]
    sats_ecef = build_constellation(start_ecef, enu, sats_azel)
    elevations = np.array([el for _, el in sats_azel])

    # Buildings block low-elevation satellites -> those carry a positive NLOS
    # range bias from reflected paths. High-elevation satellites stay clean.
    nlos_mask = elevations < 30.0
    nlos_bias_m = np.where(nlos_mask, rng.uniform(30.0, 80.0, size=len(elevations)), 0.0)
    init_guess = start_ecef + np.array([40.0, -30.0, 20.0])  # coarse prior, shared by both

    naive_errors, robust_errors = [], []
    for k in range(n_epochs):
        rx_true = start_ecef + speed_mps * k * east_axis_ecef
        true_ranges = geometric_ranges(rx_true, sats_ecef)
        noise = rng.normal(0.0, code_noise_sigma_m, size=len(true_ranges))
        pseudoranges = true_ranges + true_clock_bias_m + noise + nlos_bias_m

        naive = robust_spp(
            sats_ecef, pseudoranges, init_pos=init_guess,
            weight_func="huber", threshold=PLAIN_LS_THRESHOLD_M,
        )
        robust = robust_spp(
            sats_ecef, pseudoranges, init_pos=init_guess,
            weight_func="cauchy", threshold=ROBUST_THRESHOLD_M,
        )
        if naive is not None:
            naive_errors.append(horizontal_error_m(naive, rx_true, enu))
        if robust is not None:
            robust_errors.append(horizontal_error_m(robust, rx_true, enu))

    naive_errors = np.array(naive_errors)
    robust_errors = np.array(robust_errors)

    def p50(a: np.ndarray) -> float:
        return float(np.median(a))

    def rms(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean(a**2)))

    naive_p50, naive_rms = p50(naive_errors), rms(naive_errors)
    robust_p50, robust_rms = p50(robust_errors), rms(robust_errors)

    # --- Report --------------------------------------------------------------
    n_clean = int(np.sum(~nlos_mask))
    n_nlos = int(np.sum(nlos_mask))
    print("Urban-canyon GNSS positioning (simulated, CPU-only, no data needed)")
    print("=" * 68)
    print(
        f"Scene: {n_epochs} epochs driving east at {speed_mps:.0f} m/s, "
        f"{len(sats_azel)} satellites ({n_clean} clean, {n_nlos} NLOS-blocked)."
    )
    print(
        f"NLOS range bias on blocked sats: "
        f"{nlos_bias_m[nlos_mask].min():.0f}-{nlos_bias_m[nlos_mask].max():.0f} m.\n"
    )

    print(f"{'method':<26}{'P50 err':>12}{'RMS err':>12}")
    print("-" * 50)
    print(f"{'naive WLS (L2)':<26}{naive_p50:>10.2f} m{naive_rms:>10.2f} m")
    print(f"{'robust SPP (Cauchy)':<26}{robust_p50:>10.2f} m{robust_rms:>10.2f} m")
    print("-" * 50)
    p50_gain = 100.0 * (1.0 - robust_p50 / max(naive_p50, 1e-9))
    rms_gain = 100.0 * (1.0 - robust_rms / max(naive_rms, 1e-9))
    print(f"robust vs naive: {p50_gain:.0f}% better P50, {rms_gain:.0f}% better RMS\n")

    n_compared = min(len(naive_errors), len(robust_errors))
    wins = int(np.sum(robust_errors[:n_compared] < naive_errors[:n_compared]))
    worst = int(np.argmax(naive_errors[:n_compared]))
    print(
        f"robust is closer to truth in {wins}/{n_compared} epochs; "
        f"worst naive epoch {naive_errors[worst]:.1f} m -> "
        f"robust {robust_errors[worst]:.1f} m.\n"
    )

    print(
        "Takeaway: the Cauchy kernel softly down-weights the large-residual\n"
        "(NLOS-biased) pseudoranges, so they pull the fix less and it stays\n"
        "closer to truth. Scaling this robust-weighting idea up is what lets the\n"
        "GPU particle-filter stack beat RTKLIB demo5 on UrbanNav (see README).\n"
        "Next: explore examples/ for the GPU-accelerated demos."
    )

    return {
        "naive_p50_m": naive_p50,
        "naive_rms_m": naive_rms,
        "robust_p50_m": robust_p50,
        "robust_rms_m": robust_rms,
        "robust_wins": wins,
        "n_epochs": n_compared,
    }


if __name__ == "__main__":
    main()
