"""DDPR-centered ambiguity proposals for shifted integer basins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_gpu.dd_float_kf import (
    AmbiguityKey,
    _dd_geometry_and_design,
    _pair_keys,
    _symmetrize_psd,
)


@dataclass(frozen=True)
class RespawnAmbiguitySeed:
    keys: tuple[AmbiguityKey, ...]
    ahat_cycles: np.ndarray
    qahat_cycles2: np.ndarray
    position_ecef: np.ndarray
    position_covariance: np.ndarray
    position_ambiguity_cov: np.ndarray


def ddpr_centered_ambiguity_seed(
    dd_carrier_result,
    position_ecef: np.ndarray,
    position_covariance: np.ndarray,
    *,
    sigma_cp_cycles: float = 0.20,
) -> RespawnAmbiguitySeed:
    """Linearize float DD ambiguities about a carrier-independent position."""

    keys = _pair_keys(dd_carrier_result)
    position = np.asarray(position_ecef, dtype=np.float64).reshape(3)
    covariance = _symmetrize_psd(
        np.asarray(position_covariance, dtype=np.float64).reshape(3, 3)
    )
    expected_m, position_design = _dd_geometry_and_design(
        dd_carrier_result, position
    )
    wavelengths = np.asarray(
        dd_carrier_result.wavelengths_m, dtype=np.float64
    ).reshape(-1)
    carrier_cycles = np.asarray(
        dd_carrier_result.dd_carrier_cycles, dtype=np.float64
    ).reshape(-1)
    if len(keys) != wavelengths.size or wavelengths.size != carrier_cycles.size:
        raise ValueError("carrier result ambiguity dimensions do not match")
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("carrier wavelengths must be finite and positive")

    # N = phase_cycles - geometry(position) / wavelength.
    ambiguity_design = -position_design / wavelengths[:, None]
    ahat = carrier_cycles - expected_m / wavelengths
    weights = np.clip(
        np.asarray(dd_carrier_result.dd_weights, dtype=np.float64), 1.0e-6, None
    )
    measurement_variance = float(sigma_cp_cycles) ** 2 / weights
    cross_covariance = covariance @ ambiguity_design.T
    qahat = (
        ambiguity_design @ covariance @ ambiguity_design.T
        + np.diag(measurement_variance)
    )
    return RespawnAmbiguitySeed(
        keys=keys,
        ahat_cycles=ahat,
        qahat_cycles2=_symmetrize_psd(qahat),
        position_ecef=position.copy(),
        position_covariance=covariance,
        position_ambiguity_cov=cross_covariance,
    )


def condition_respawn_position(
    seed: RespawnAmbiguitySeed,
    keys: tuple[AmbiguityKey, ...],
    integers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Condition the DDPR navigation Gaussian on a partial integer proposal."""

    index = {key: i for i, key in enumerate(seed.keys)}
    try:
        selected = np.asarray([index[key] for key in keys], dtype=np.int64)
    except KeyError as exc:
        raise KeyError(f"unknown respawn ambiguity key: {exc.args[0]}") from exc
    fixed = np.asarray(integers, dtype=np.float64).reshape(-1)
    if fixed.size != selected.size:
        raise ValueError("integer dimension must match selected keys")
    qahat = seed.qahat_cycles2[np.ix_(selected, selected)]
    cross = seed.position_ambiguity_cov[:, selected]
    innovation = fixed - seed.ahat_cycles[selected]
    try:
        solved = np.linalg.solve(qahat, innovation)
        gain = np.linalg.solve(qahat, cross.T).T
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("respawn ambiguity covariance is singular") from exc
    position = seed.position_ecef + cross @ solved
    covariance = seed.position_covariance - gain @ cross.T
    return position, _symmetrize_psd(covariance), float(innovation @ solved)
