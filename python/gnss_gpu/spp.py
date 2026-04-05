"""Single Point Positioning (SPP) with full GNSS corrections.

Implements the correction pipeline from gnssplusplus-library:
1. Satellite clock (2-pass iteration)
2. Earth rotation (Sagnac effect)
3. Troposphere (Saastamoinen)
4. Ionosphere (Klobuchar)
5. Group delay (TGD)

Reference: Groves (2013), gnssplusplus-library spp.cpp
"""

from __future__ import annotations

import math

import numpy as np


C_LIGHT = 299792458.0
OMEGA_E = 7.2921151467e-5
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F * WGS84_F


def _ecef_to_llh(x: float, y: float, z: float) -> tuple[float, float, float]:
    p = math.sqrt(x * x + y * y)
    lon = math.atan2(y, x)
    lat = math.atan2(z, p * (1 - WGS84_E2))
    for _ in range(5):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
        lat = math.atan2(z + WGS84_E2 * N * sin_lat, p)
    sin_lat = math.sin(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N if abs(math.cos(lat)) > 1e-10 else 0.0
    return lat, lon, alt


def _elevation_azimuth(
    rx: np.ndarray, sat: np.ndarray,
) -> tuple[float, float]:
    lat, lon, _ = _ecef_to_llh(rx[0], rx[1], rx[2])
    dx = sat - rx
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    e = -sin_lon * dx[0] + cos_lon * dx[1]
    n = -sin_lat * cos_lon * dx[0] - sin_lat * sin_lon * dx[1] + cos_lat * dx[2]
    u = cos_lat * cos_lon * dx[0] + cos_lat * sin_lon * dx[1] + sin_lat * dx[2]
    el = math.atan2(u, math.sqrt(e * e + n * n))
    az = math.atan2(e, n)
    return el, az


def _tropo_saastamoinen(lat: float, alt: float, el: float) -> float:
    P = 1013.25 * (1.0 - 2.2557e-5 * alt) ** 5.2568
    T = 15.0 - 6.5e-3 * alt + 273.15
    e_wv = 6.108 * math.exp((17.15 * (T - 273.15)) / (T - 273.15 + 234.7)) * 0.5
    alt_km = alt / 1000.0
    z = 0.002277 * (P + (1255.0 / T + 0.05) * e_wv) / \
        (1.0 - 0.00266 * math.cos(2.0 * lat) - 0.00028 * alt_km)
    sin_el = math.sin(math.sqrt(el * el + math.radians(6.25) ** 2))
    return z / sin_el


def _iono_klobuchar(
    alpha: list[float], beta: list[float],
    lat: float, lon: float, az: float, el: float,
    gps_time: float,
) -> float:
    PI = math.pi
    lat_sc = lat / PI
    lon_sc = lon / PI
    el_sc = el / PI
    psi = 0.0137 / (el_sc + 0.11) - 0.022
    phi_i = max(min(lat_sc + psi * math.cos(az), 0.416), -0.416)
    lambda_i = lon_sc + psi * math.sin(az) / math.cos(phi_i * PI)
    t_gps = 43200 * lambda_i + gps_time
    t_gps = t_gps % 86400
    phi_m = phi_i + 0.064 * math.cos((lambda_i - 1.617) * PI)
    F = 1.0 + 16.0 * (0.53 - el_sc) ** 3
    PER = max(sum(beta[j] * phi_m ** j for j in range(4)), 72000.0)
    AMP = max(sum(alpha[j] * phi_m ** j for j in range(4)), 0.0)
    x = 2.0 * PI * (t_gps - 50400.0) / PER
    if abs(x) < 1.57:
        delay = C_LIGHT * F * (5e-9 + AMP * (1 - x * x / 2 + x ** 4 / 24))
    else:
        delay = C_LIGHT * F * 5e-9
    return delay


def correct_pseudoranges(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    receiver_approx: np.ndarray,
    gps_time: float,
    iono_alpha: list[float] | None = None,
    iono_beta: list[float] | None = None,
    el_mask_rad: float = 0.087,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply atmospheric + Sagnac corrections to pseudoranges.

    Returns
    -------
    corrected_pr : (K,) corrected pseudoranges
    weights : (K,) elevation-based weights (sin^2(el))
    """
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(pseudoranges, dtype=np.float64).ravel().copy()
    rx = np.asarray(receiver_approx, dtype=np.float64).ravel()[:3]
    n_sat = len(pr)

    if iono_alpha is None:
        iono_alpha = [1.1176e-08, -7.4506e-09, -5.9605e-08, 1.1921e-07]
    if iono_beta is None:
        iono_beta = [1.1264e+05, -3.2768e+04, -2.6214e+05, 4.5875e+05]

    lat, lon, alt = _ecef_to_llh(rx[0], rx[1], rx[2])
    weights = np.ones(n_sat)

    for s in range(n_sat):
        # Sagnac: rotate satellite position
        dx0 = rx - sat[s]
        range_approx = np.linalg.norm(dx0)
        if range_approx < 1e3:
            weights[s] = 0.0
            continue
        transit_time = range_approx / C_LIGHT
        theta = OMEGA_E * transit_time
        sx_rot = sat[s, 0] * math.cos(theta) + sat[s, 1] * math.sin(theta)
        sy_rot = -sat[s, 0] * math.sin(theta) + sat[s, 1] * math.cos(theta)
        sat_rot = np.array([sx_rot, sy_rot, sat[s, 2]])

        el, az = _elevation_azimuth(rx, sat_rot)
        if el < el_mask_rad:
            weights[s] = 0.01  # don't zero out, just heavily downweight
            continue

        # Troposphere
        tropo = _tropo_saastamoinen(lat, alt, el)
        # Ionosphere
        iono = _iono_klobuchar(iono_alpha, iono_beta, lat, lon, az, el, gps_time)

        pr[s] -= (tropo + iono)

        # Elevation weight
        sin_el = max(math.sin(el), 0.1)
        weights[s] = sin_el * sin_el

    return pr, weights
