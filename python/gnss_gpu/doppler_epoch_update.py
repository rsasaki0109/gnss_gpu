"""Doppler epoch update flow for PF smoother forward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from gnss_gpu.doppler_update import build_doppler_update_decision


@dataclass(frozen=True)
class DopplerEpochUpdateResult:
    used: bool
    skipped: bool
    gate_skipped: bool
    used_kf: bool
    gate_reason: str | None


def apply_doppler_epoch_update(
    pf: Any,
    epoch_state: Any,
    stats: Any,
    *,
    measurements: Iterable[Any],
    rover_weights: np.ndarray,
    doppler_config: Any,
    gate_ess_ratio: float | None,
    gate_spread_m: float | None,
    wavelength_m: float,
) -> DopplerEpochUpdateResult:
    if not (doppler_config.per_particle or doppler_config.rbpf_velocity_kf):
        return DopplerEpochUpdateResult(
            used=False,
            skipped=False,
            gate_skipped=False,
            used_kf=False,
            gate_reason=None,
        )

    doppler_decision = build_doppler_update_decision(
        list(measurements),
        rover_weights,
        min_sats=doppler_config.min_sats,
        doppler_sigma_mps=doppler_config.sigma_mps,
        rbpf_velocity_kf=doppler_config.rbpf_velocity_kf,
        rbpf_doppler_sigma=doppler_config.rbpf_doppler_sigma,
        wavelength_m=wavelength_m,
        dd_gate_stats=epoch_state.dd_gate_stats,
        dd_carrier_result=epoch_state.dd_carrier_result,
        gate_ess_ratio=gate_ess_ratio,
        gate_spread_m=gate_spread_m,
        rbpf_gate_min_dd_pairs=doppler_config.rbpf_gate_min_dd_pairs,
        rbpf_gate_min_ess_ratio=doppler_config.rbpf_gate_min_ess_ratio,
        rbpf_gate_max_spread_m=doppler_config.rbpf_gate_max_spread_m,
    )
    epoch_state.doppler_kf_gate_reason = doppler_decision.gate_reason

    if doppler_decision.skipped:
        if doppler_config.rbpf_velocity_kf:
            stats.n_doppler_kf_skip += 1
            if doppler_decision.gate_skipped:
                stats.n_doppler_kf_gate_skip += 1
        else:
            stats.n_doppler_pp_skip += 1
        return DopplerEpochUpdateResult(
            used=False,
            skipped=True,
            gate_skipped=bool(doppler_decision.gate_skipped),
            used_kf=bool(doppler_decision.use_kf),
            gate_reason=doppler_decision.gate_reason,
        )

    if doppler_decision.update is None:
        return DopplerEpochUpdateResult(
            used=False,
            skipped=False,
            gate_skipped=False,
            used_kf=bool(doppler_decision.use_kf),
            gate_reason=doppler_decision.gate_reason,
        )

    epoch_state.doppler_update_epoch = doppler_decision.update
    epoch_state.doppler_sigma_epoch = doppler_decision.sigma_mps
    if doppler_decision.use_kf:
        pf.update_doppler_kf(
            epoch_state.doppler_update_epoch["sat_ecef"],
            epoch_state.doppler_update_epoch["sat_vel"],
            epoch_state.doppler_update_epoch["doppler_hz"],
            weights=epoch_state.doppler_update_epoch["weights"],
            wavelength=float(epoch_state.doppler_update_epoch["wavelength_m"]),
            sigma_mps=epoch_state.doppler_sigma_epoch,
        )
        stats.n_doppler_kf_used += 1
        return DopplerEpochUpdateResult(
            used=True,
            skipped=False,
            gate_skipped=False,
            used_kf=True,
            gate_reason=doppler_decision.gate_reason,
        )

    pf.update_doppler(
        epoch_state.doppler_update_epoch["sat_ecef"],
        epoch_state.doppler_update_epoch["sat_vel"],
        epoch_state.doppler_update_epoch["doppler_hz"],
        weights=epoch_state.doppler_update_epoch["weights"],
        wavelength=float(epoch_state.doppler_update_epoch["wavelength_m"]),
        sigma_mps=epoch_state.doppler_sigma_epoch,
        velocity_update_gain=doppler_config.velocity_update_gain,
        max_velocity_update_mps=doppler_config.max_velocity_update_mps,
    )
    stats.n_doppler_pp_used += 1
    return DopplerEpochUpdateResult(
        used=True,
        skipped=False,
        gate_skipped=False,
        used_kf=False,
        gate_reason=doppler_decision.gate_reason,
    )
