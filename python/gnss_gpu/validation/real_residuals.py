from __future__ import annotations

import math
import re
from numbers import Real
from typing import Iterable

import numpy as np

from gnss_gpu.transmission_time import (
    C_LIGHT,
    EARTH_ROTATION_RATE,
    sagnac_rotate,
    transmission_time_sat_positions,
)
from gnss_gpu.validation.residuals import ResidualSample


def prn_to_int(prn) -> int:
    if isinstance(prn, Real) and not isinstance(prn, bool):
        return int(prn)

    if isinstance(prn, str):
        # First run of digits anywhere; handles "G01", "E12", "G 5" (RINEX
        # often left-pads single-digit PRNs with a space, e.g. "G 5").
        match = re.search(r"([0-9]+)", prn)
        if match is None:
            raise ValueError(f"PRN has no numeric component: {prn!r}")
        return int(match.group(1))

    raise ValueError(f"Unsupported PRN type: {type(prn)!r}")


def estimate_clock_bias(raw_residuals, method: str = "median") -> float:
    arr = np.asarray(raw_residuals, dtype=float)
    if arr.size == 0:
        raise ValueError("raw_residuals must not be empty")
    if np.all(np.isnan(arr)):
        raise ValueError("raw_residuals contains only NaN values")

    if method == "median":
        return float(np.nanmedian(arr))
    if method == "mean":
        return float(np.nanmean(arr))
    raise ValueError(f"Unsupported clock bias method: {method!r}")


def epoch_residuals(
    pseudorange,
    sat_ecef,
    rx_ecef,
    sat_clock_m=None,
    clock_bias=None,
    clock_method: str = "median",
    atmo_delay_m=None,
) -> tuple[np.ndarray, float]:
    pr = np.asarray(pseudorange, dtype=float)
    sat = np.asarray(sat_ecef, dtype=float)
    rx = np.asarray(rx_ecef, dtype=float)

    if pr.ndim != 1:
        raise ValueError("pseudorange must be a 1D array")
    if sat.ndim != 2 or sat.shape[1] != 3:
        raise ValueError("sat_ecef must have shape [n, 3]")
    if rx.shape != (3,):
        raise ValueError("rx_ecef must have shape [3]")
    if sat.shape[0] != pr.shape[0]:
        raise ValueError("pseudorange and sat_ecef length mismatch")

    if sat_clock_m is None:
        sat_clk = np.zeros(pr.shape[0], dtype=float)
    else:
        sat_clk = np.asarray(sat_clock_m, dtype=float)
        if sat_clk.shape != pr.shape:
            raise ValueError("sat_clock_m must have shape [n]")

    if atmo_delay_m is None:
        atmo = np.zeros(pr.shape[0], dtype=float)
    else:
        atmo = np.asarray(atmo_delay_m, dtype=float)
        if atmo.shape != pr.shape:
            raise ValueError("atmo_delay_m must have shape [n]")

    geom_range = np.linalg.norm(sat - rx[None, :], axis=1)
    # pr = range - sat_clk + rx_clk + atmo + multipath  =>
    # pre = pr - range + sat_clk - atmo = rx_clk + multipath (+ residual atmo).
    pre = pr - geom_range + sat_clk - atmo

    bias = estimate_clock_bias(pre, clock_method) if clock_bias is None else float(clock_bias)
    return pre - bias, bias


def _ecef_to_geodetic_lat_lon(rx_ecef: np.ndarray) -> tuple[float, float]:
    x, y, z = map(float, rx_ecef)
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p == 0.0:
        return math.copysign(math.pi / 2.0, z), lon

    lat = math.atan2(z, p * (1.0 - e2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + e2 * n * sin_lat, p)

    return lat, lon


def elevation_azimuth(rx_ecef, sat_ecef) -> tuple[np.ndarray, np.ndarray]:
    rx = np.asarray(rx_ecef, dtype=float)
    sat = np.asarray(sat_ecef, dtype=float)

    if rx.shape != (3,):
        raise ValueError("rx_ecef must have shape [3]")
    if sat.ndim != 2 or sat.shape[1] != 3:
        raise ValueError("sat_ecef must have shape [n, 3]")

    lat, lon = _ecef_to_geodetic_lat_lon(rx)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    diff = sat - rx[None, :]
    east = -sin_lon * diff[:, 0] + cos_lon * diff[:, 1]
    north = (
        -sin_lat * cos_lon * diff[:, 0]
        - sin_lat * sin_lon * diff[:, 1]
        + cos_lat * diff[:, 2]
    )
    up = (
        cos_lat * cos_lon * diff[:, 0]
        + cos_lat * sin_lon * diff[:, 1]
        + sin_lat * diff[:, 2]
    )

    horiz = np.hypot(east, north)
    elevation = np.arctan2(up, horiz)
    azimuth = np.mod(np.arctan2(east, north), 2.0 * np.pi)
    return elevation, azimuth


def _geodetic_lat_alt(rx_ecef: np.ndarray) -> tuple[float, float]:
    """Return (lat_rad, alt_m) of an ECEF receiver position (WGS84)."""
    x, y, z = map(float, rx_ecef)
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    p = math.hypot(x, y)
    if p == 0.0:
        lat = math.copysign(math.pi / 2.0, z)
        return lat, abs(z) - a * math.sqrt(1.0 - e2)

    lat = math.atan2(z, p * (1.0 - e2))
    n = a
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        lat = math.atan2(z + e2 * n * sin_lat, p)
    alt = p / math.cos(lat) - n
    return lat, alt


def tropo_delays(rx_ecef, sat_ecef, model: str = "saastamoinen") -> np.ndarray:
    """Per-satellite slant tropospheric delay [m] (Saastamoinen, elevation-mapped).

    Uses the same pure-Python Saastamoinen model as gnss_gpu.atmosphere. The
    delay is positive and grows steeply toward the horizon (~2.3 m at zenith to
    tens of metres at very low elevation).
    """
    if model.strip().lower() != "saastamoinen":
        raise ValueError(f"unsupported tropo model: {model!r}")

    from gnss_gpu.atmosphere import _tropo_saastamoinen_cpu

    rx = np.asarray(rx_ecef, dtype=float)
    lat, alt = _geodetic_lat_alt(rx)
    elevation, _ = elevation_azimuth(rx, sat_ecef)
    return np.asarray(
        _tropo_saastamoinen_cpu(lat, alt, np.asarray(elevation, dtype=float)),
        dtype=float,
    )


def iono_delays(rx_ecef, sat_ecef, gps_time, alpha=None, beta=None) -> np.ndarray:
    """Per-satellite L1 ionospheric delay [m] (Klobuchar broadcast model).

    Uses the same pure-Python Klobuchar model as gnss_gpu.atmosphere. ``alpha``
    and ``beta`` are the 4-term GPS broadcast coefficients; when omitted, the
    atmosphere module's default coefficients are used (the data set's nav header
    may not carry them). ``gps_time`` is GPS seconds of week.
    """
    from gnss_gpu.atmosphere import (
        _DEFAULT_ALPHA,
        _DEFAULT_BETA,
        _iono_klobuchar_cpu,
    )

    alpha = list(_DEFAULT_ALPHA) if alpha is None else list(alpha)
    beta = list(_DEFAULT_BETA) if beta is None else list(beta)

    rx = np.asarray(rx_ecef, dtype=float)
    lat, lon = _ecef_to_geodetic_lat_lon(rx)
    elevation, azimuth = elevation_azimuth(rx, sat_ecef)
    return np.asarray(
        _iono_klobuchar_cpu(
            alpha, beta, lat, lon,
            np.asarray(azimuth, dtype=float),
            np.asarray(elevation, dtype=float),
            float(gps_time),
        ),
        dtype=float,
    )


def residual_samples_from_epoch(
    epoch_time,
    prn_list,
    pseudorange,
    sat_ecef,
    rx_ecef,
    *,
    sat_clock_m=None,
    cn0=None,
    is_los=None,
    elevation_mask_rad=None,
    clock_bias=None,
    clock_method: str = "median",
    atmo_delay_m=None,
) -> list[ResidualSample]:
    residuals, _ = epoch_residuals(
        pseudorange,
        sat_ecef,
        rx_ecef,
        sat_clock_m=sat_clock_m,
        clock_bias=clock_bias,
        clock_method=clock_method,
        atmo_delay_m=atmo_delay_m,
    )
    elevation, azimuth = elevation_azimuth(rx_ecef, sat_ecef)

    n = residuals.shape[0]
    if len(prn_list) != n:
        raise ValueError("prn_list length mismatch")

    cn0_arr = np.full(n, np.nan, dtype=float) if cn0 is None else np.asarray(cn0, dtype=float)
    los_arr = np.full(n, True, dtype=bool) if is_los is None else np.asarray(is_los, dtype=bool)

    if cn0_arr.shape != (n,):
        raise ValueError("cn0 must have shape [n]")
    if los_arr.shape != (n,):
        raise ValueError("is_los must have shape [n]")

    samples: list[ResidualSample] = []
    for i in range(n):
        if np.isnan(residuals[i]):
            continue
        if elevation_mask_rad is not None and elevation[i] < elevation_mask_rad:
            continue

        samples.append(
            ResidualSample(
                epoch=float(epoch_time),
                prn=prn_to_int(prn_list[i]),
                residual_m=float(residuals[i]),
                elevation_rad=float(elevation[i]),
                azimuth_rad=float(azimuth[i]),
                cn0_dbhz=float(cn0_arr[i]),
                is_los=bool(los_arr[i]),
            )
        )

    return samples


def collect_residual_samples(
    epochs: Iterable,
    sat_lookup,
    *,
    cn0_from_obs: bool = True,
    elevation_mask_rad=None,
    clock_method: str = "median",
) -> list[ResidualSample]:
    all_samples: list[ResidualSample] = []

    for gnss_obs, rx_ecef in epochs:
        prns = list(gnss_obs.prn)
        pseudorange = np.asarray(gnss_obs.pseudorange, dtype=float)
        obs_cn0 = getattr(gnss_obs, "cn0", None)

        use_prn = []
        use_pr = []
        use_sat = []
        use_clk = []
        use_cn0 = []

        for i, prn in enumerate(prns):
            looked_up = sat_lookup(float(gnss_obs.time), prn)
            if looked_up is None:
                continue

            sat_pos, sat_clk = looked_up
            use_prn.append(prn)
            use_pr.append(pseudorange[i])
            use_sat.append(np.asarray(sat_pos, dtype=float))
            use_clk.append(float(sat_clk))

            if cn0_from_obs and obs_cn0 is not None:
                use_cn0.append(float(np.asarray(obs_cn0, dtype=float)[i]))
            else:
                use_cn0.append(np.nan)

        if not use_prn:
            continue

        all_samples.extend(
            residual_samples_from_epoch(
                gnss_obs.time,
                use_prn,
                np.asarray(use_pr, dtype=float),
                np.vstack(use_sat),
                rx_ecef,
                sat_clock_m=np.asarray(use_clk, dtype=float),
                cn0=np.asarray(use_cn0, dtype=float),
                elevation_mask_rad=elevation_mask_rad,
                clock_method=clock_method,
            )
        )

    return all_samples


def residual_samples_from_experiment_data(
    data,
    *,
    elevation_mask_rad=None,
    clock_method: str = "median",
    apply_tropo: bool = False,
    apply_iono: bool = False,
    iono_alpha=None,
    iono_beta=None,
) -> list[ResidualSample]:
    """Build ResidualSamples from a UrbanNavLoader.load_experiment_data() dict.

    The loader returns per-epoch lists. ``pseudoranges[i]`` already has the
    satellite-clock correction folded in (``pr + sat_clk * c``), so only the
    receiver clock (estimated as the per-epoch median) is removed here.

    When ``apply_tropo`` is False the residuals contain the receiver-relative
    range errors -- multipath/NLOS plus unmodelled atmospheric delay. When
    ``apply_tropo`` is True, the per-satellite Saastamoinen tropospheric delay
    is subtracted before the receiver-clock estimate, which removes most of the
    elevation-dependent bias and better isolates the multipath/NLOS component
    (ionosphere is still unmodelled).

    Required dict keys: ``sat_ecef`` (list of [n,3]), ``pseudoranges``
    (list of [n]), ``ground_truth`` ([n_epoch,3]), ``times`` ([n_epoch]),
    ``used_prns`` (list of n-length PRN-id lists). Optional ``weights`` (SNR)
    is recorded as ``cn0_dbhz`` when present.
    """
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    ground_truth = np.asarray(data["ground_truth"], dtype=float)
    times = np.asarray(data["times"], dtype=float)
    used_prns = data["used_prns"]
    weights = data.get("weights")

    samples: list[ResidualSample] = []
    for i in range(len(times)):
        sat = np.asarray(sat_ecef[i], dtype=float)
        pr = np.asarray(pseudoranges[i], dtype=float)
        if sat.shape[0] == 0:
            continue
        cn0 = None if weights is None else np.asarray(weights[i], dtype=float)
        atmo = None
        if apply_tropo:
            atmo = tropo_delays(ground_truth[i], sat)
        if apply_iono:
            iono = iono_delays(
                ground_truth[i], sat, float(times[i]),
                alpha=iono_alpha, beta=iono_beta)
            atmo = iono if atmo is None else atmo + iono
        samples.extend(
            residual_samples_from_epoch(
                float(times[i]),
                list(used_prns[i]),
                pr,
                sat,
                ground_truth[i],
                sat_clock_m=None,  # satellite clock already folded into pr
                atmo_delay_m=atmo,
                cn0=cn0,
                elevation_mask_rad=elevation_mask_rad,
                clock_method=clock_method,
            )
        )
    return samples


def residual_array(samples: list[ResidualSample]) -> np.ndarray:
    return np.asarray([sample.residual_m for sample in samples], dtype=float)


__all__ = [
    "C_LIGHT",
    "EARTH_ROTATION_RATE",
    "sagnac_rotate",
    "transmission_time_sat_positions",
    "prn_to_int",
    "estimate_clock_bias",
    "epoch_residuals",
    "elevation_azimuth",
    "tropo_delays",
    "iono_delays",
    "residual_samples_from_epoch",
    "collect_residual_samples",
    "residual_samples_from_experiment_data",
    "residual_array",
]
