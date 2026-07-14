"""Constellation and observation-code aware Doppler wavelengths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

C_LIGHT_MPS = 299_792_458.0
GPS_L1_WAVELENGTH_M = C_LIGHT_MPS / 1_575_420_000.0


@dataclass(frozen=True)
class ConstellationClockDriftFit:
    velocity_ecef_mps: np.ndarray
    group_ids: np.ndarray
    clock_drifts_mps: np.ndarray
    residual_rms_mps: float

# RINEX band digit -> carrier frequency. GLONASS FDMA bands are handled below.
_BAND_FREQUENCY_HZ: dict[str, dict[str, float]] = {
    "G": {"1": 1575.42e6, "2": 1227.60e6, "5": 1176.45e6},
    "E": {"1": 1575.42e6, "5": 1176.45e6, "6": 1278.75e6,
          "7": 1207.14e6, "8": 1191.795e6},
    "J": {"1": 1575.42e6, "2": 1227.60e6, "5": 1176.45e6, "6": 1278.75e6},
    # BeiDou: band 2 is legacy B1I; band 1 is B1C.
    "C": {"1": 1575.42e6, "2": 1561.098e6, "5": 1176.45e6,
          "6": 1268.52e6, "7": 1207.14e6, "8": 1191.795e6},
}


def carrier_frequency_hz(
    satellite_id: str,
    observation_code: str,
    *,
    glonass_frequency_channels: Mapping[str, int] | None = None,
) -> float:
    """Resolve a RINEX Doppler code (e.g. D1C/D2I) to carrier frequency."""

    sat = str(satellite_id).strip().upper()
    code = str(observation_code).strip().upper()
    if not sat or sat[0] not in "GEJCR" or len(code) < 2 or code[0] != "D":
        return float("nan")
    system, band = sat[0], code[1]
    if system == "R":
        channels = glonass_frequency_channels or {}
        if sat not in channels:
            return float("nan")
        channel = int(channels[sat])
        if channel < -7 or channel > 6:
            return float("nan")
        if band == "1":
            return 1602.0e6 + channel * 0.5625e6
        if band == "2":
            return 1246.0e6 + channel * 0.4375e6
        if band == "3":
            return 1202.025e6
        return float("nan")
    return float(_BAND_FREQUENCY_HZ.get(system, {}).get(band, float("nan")))


def doppler_wavelengths_m(
    satellite_ids: Sequence[str],
    observation_codes: Sequence[str],
    *,
    glonass_frequency_channels: Mapping[str, int] | None = None,
) -> np.ndarray:
    if len(satellite_ids) != len(observation_codes):
        raise ValueError("satellite_ids and observation_codes must have equal length")
    frequencies = np.asarray(
        [
            carrier_frequency_hz(
                sat, code, glonass_frequency_channels=glonass_frequency_channels
            )
            for sat, code in zip(satellite_ids, observation_codes, strict=True)
        ],
        dtype=np.float64,
    )
    out = np.full(frequencies.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(frequencies) & (frequencies > 0.0)
    out[valid] = C_LIGHT_MPS / frequencies[valid]
    return out


def normalize_doppler_to_reference(
    doppler_hz: np.ndarray,
    wavelengths_m: np.ndarray,
    *,
    reference_wavelength_m: float = GPS_L1_WAVELENGTH_M,
) -> np.ndarray:
    """Preserve row range rates while adapting to a scalar-wavelength API."""

    doppler = np.asarray(doppler_hz, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    if doppler.shape != wavelengths.shape:
        raise ValueError("doppler_hz and wavelengths_m must have matching shapes")
    reference = float(reference_wavelength_m)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_wavelength_m must be positive")
    return doppler * wavelengths / reference


def fit_constellation_clock_drifts(
    satellite_ecef: np.ndarray,
    satellite_velocity_ecef: np.ndarray,
    doppler_hz: np.ndarray,
    wavelengths_m: np.ndarray,
    receiver_position_ecef: np.ndarray,
    constellation_ids: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    doppler_sign: float = -1.0,
) -> ConstellationClockDriftFit:
    """Fit receiver velocity plus one clock-drift state per constellation."""

    sat = np.asarray(satellite_ecef, dtype=np.float64).reshape(-1, 3)
    sat_vel = np.asarray(satellite_velocity_ecef, dtype=np.float64).reshape(-1, 3)
    doppler = np.asarray(doppler_hz, dtype=np.float64).reshape(-1)
    wavelength = np.asarray(wavelengths_m, dtype=np.float64).reshape(-1)
    groups = np.asarray(constellation_ids).reshape(-1)
    position = np.asarray(receiver_position_ecef, dtype=np.float64).reshape(3)
    n = doppler.size
    if sat.shape != (n, 3) or sat_vel.shape != (n, 3) or wavelength.size != n or groups.size != n:
        raise ValueError("Doppler clock-drift inputs must have matching rows")
    weight = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if weight.size != n:
        raise ValueError("weights must match Doppler rows")
    valid = (
        np.isfinite(sat).all(axis=1)
        & np.isfinite(sat_vel).all(axis=1)
        & np.isfinite(doppler)
        & np.isfinite(wavelength)
        & (wavelength > 0.0)
        & np.isfinite(weight)
        & (weight > 0.0)
    )
    sat, sat_vel, doppler, wavelength, groups, weight = (
        values[valid] for values in (sat, sat_vel, doppler, wavelength, groups, weight)
    )
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    if doppler.size < 3 + unique_groups.size:
        raise ValueError("insufficient rows for velocity plus constellation clock drifts")
    delta = sat - position[None, :]
    ranges = np.linalg.norm(delta, axis=1)
    if np.any(ranges <= 1.0):
        raise ValueError("degenerate receiver/satellite geometry")
    los = delta / ranges[:, None]
    design = np.zeros((doppler.size, 3 + unique_groups.size), dtype=np.float64)
    design[:, :3] = -los
    design[np.arange(doppler.size), 3 + inverse] = 1.0
    observation = float(doppler_sign) * doppler * wavelength - np.sum(sat_vel * los, axis=1)
    sqrt_weight = np.sqrt(weight)
    solution, _residuals, rank, _singular = np.linalg.lstsq(
        design * sqrt_weight[:, None], observation * sqrt_weight, rcond=None
    )
    if rank < design.shape[1] or not np.isfinite(solution).all():
        raise ValueError("constellation Doppler geometry is rank deficient")
    residual = observation - design @ solution
    rms = float(np.sqrt(np.sum(weight * residual * residual) / np.sum(weight)))
    return ConstellationClockDriftFit(
        velocity_ecef_mps=solution[:3],
        group_ids=unique_groups,
        clock_drifts_mps=solution[3:],
        residual_rms_mps=rms,
    )


def normalize_constellation_clock_drifts(
    satellite_ecef: np.ndarray,
    satellite_velocity_ecef: np.ndarray,
    doppler_hz: np.ndarray,
    wavelengths_m: np.ndarray,
    receiver_position_ecef: np.ndarray,
    constellation_ids: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    reference_wavelength_m: float = GPS_L1_WAVELENGTH_M,
    doppler_sign: float = -1.0,
) -> tuple[np.ndarray, ConstellationClockDriftFit]:
    """Map multi-clock Doppler rows to one reference clock and wavelength."""

    groups = np.asarray(constellation_ids).reshape(-1)
    doppler = np.asarray(doppler_hz, dtype=np.float64).reshape(-1)
    wavelengths = np.asarray(wavelengths_m, dtype=np.float64).reshape(-1)
    if groups.size != doppler.size or wavelengths.size != doppler.size:
        raise ValueError("constellation_ids and wavelengths must match Doppler rows")
    fit = fit_constellation_clock_drifts(
        satellite_ecef,
        satellite_velocity_ecef,
        doppler,
        wavelengths,
        receiver_position_ecef,
        groups,
        weights=weights,
        doppler_sign=doppler_sign,
    )
    counts = np.asarray([np.count_nonzero(groups == group) for group in fit.group_ids])
    reference_index = int(np.argmax(counts))
    drift_by_group = {
        group: float(drift) for group, drift in zip(fit.group_ids, fit.clock_drifts_mps, strict=True)
    }
    reference_drift = float(fit.clock_drifts_mps[reference_index])
    drift_delta = np.asarray([drift_by_group[group] - reference_drift for group in groups])
    signed_range_rate = float(doppler_sign) * doppler * wavelengths - drift_delta
    equivalent = signed_range_rate / (float(doppler_sign) * float(reference_wavelength_m))
    return equivalent, fit
