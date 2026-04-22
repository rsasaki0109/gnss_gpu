"""Undifferenced carrier AFV observation selection for MUPF updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CarrierAfvObservation:
    sat_ecef: np.ndarray
    carrier_phase_cycles: np.ndarray
    weights: np.ndarray
    sigma_sequence_cycles: tuple[float, ...]

    @property
    def n_sat(self) -> int:
        return int(self.carrier_phase_cycles.size)


def build_carrier_afv_observation(
    measurements: list,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    spp_position: np.ndarray,
    *,
    snr_min: float,
    elev_min: float,
    target_sigma_cycles: float,
    min_sats: int = 4,
    residual_max_m: float = 30.0,
) -> CarrierAfvObservation | None:
    sat_all = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
    spp_pos = np.asarray(spp_position, dtype=np.float64).ravel()[:3]
    cb_est_m: float | None = None
    if (
        sat_all.shape[0] == pr.size
        and np.isfinite(spp_pos).all()
        and np.linalg.norm(spp_pos) > 1e6
    ):
        cb_est_m = float(np.median(pr - np.linalg.norm(sat_all - spp_pos, axis=1)))

    cp_cycles = []
    cp_sat_ecef = []
    cp_weights = []
    for measurement in measurements:
        cp = float(getattr(measurement, "carrier_phase", 0.0))
        if cp == 0.0 or not np.isfinite(cp) or abs(cp) < 1e3:
            continue
        snr = float(getattr(measurement, "snr", 0.0))
        elev = float(getattr(measurement, "elevation", 0.0))
        if snr < float(snr_min) and snr > 0:
            continue
        if 0 < elev < float(elev_min):
            continue
        sat_pos = np.asarray(measurement.satellite_ecef, dtype=np.float64).ravel()[:3]
        if cb_est_m is not None:
            rng = np.linalg.norm(sat_pos - spp_pos)
            res = abs(float(measurement.corrected_pseudorange) - rng - cb_est_m)
            if res > float(residual_max_m):
                continue
        cp_cycles.append(cp)
        cp_sat_ecef.append(sat_pos)
        cp_weights.append(float(getattr(measurement, "weight", 1.0)))

    if len(cp_cycles) < int(min_sats):
        return None
    return CarrierAfvObservation(
        sat_ecef=np.asarray(cp_sat_ecef, dtype=np.float64),
        carrier_phase_cycles=np.asarray(cp_cycles, dtype=np.float64),
        weights=np.asarray(cp_weights, dtype=np.float64),
        sigma_sequence_cycles=(2.0, 0.5, float(target_sigma_cycles)),
    )
