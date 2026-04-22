"""Doppler update extraction and gating helpers for PF smoother runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DopplerUpdateDecision:
    update: dict[str, np.ndarray | float] | None
    sigma_mps: float | None
    use_kf: bool
    gate_reason: str | None = None
    skipped: bool = False
    gate_skipped: bool = False


def build_doppler_update_decision(
    measurements: list,
    weights: np.ndarray,
    *,
    min_sats: int,
    doppler_sigma_mps: float,
    rbpf_velocity_kf: bool,
    rbpf_doppler_sigma: float | None,
    wavelength_m: float,
    dd_gate_stats: Any | None = None,
    dd_carrier_result: Any | None = None,
    gate_ess_ratio: float | None = None,
    gate_spread_m: float | None = None,
    rbpf_gate_min_dd_pairs: int | None = None,
    rbpf_gate_min_ess_ratio: float | None = None,
    rbpf_gate_max_spread_m: float | None = None,
) -> DopplerUpdateDecision:
    dop_sat = []
    dop_sat_vel = []
    dop_obs = []
    dop_weights = []
    weight_arr = np.asarray(weights, dtype=np.float64).ravel()
    for i_m, measurement in enumerate(measurements):
        dop = float(getattr(measurement, "doppler", 0.0))
        sat_vel_m = np.asarray(
            getattr(measurement, "satellite_velocity", np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).ravel()[:3]
        if (
            not np.isfinite(dop)
            or dop == 0.0
            or sat_vel_m.shape[0] != 3
            or not np.isfinite(sat_vel_m).all()
        ):
            continue
        dop_sat.append(np.asarray(measurement.satellite_ecef, dtype=np.float64).ravel()[:3])
        dop_sat_vel.append(sat_vel_m)
        dop_obs.append(dop)
        dop_weights.append(float(weight_arr[i_m]) if i_m < len(weight_arr) else 1.0)

    if len(dop_obs) < int(min_sats):
        return DopplerUpdateDecision(
            update=None,
            sigma_mps=None,
            use_kf=bool(rbpf_velocity_kf),
            gate_reason="min_sats" if rbpf_velocity_kf else None,
            skipped=True,
            gate_skipped=False,
        )

    if rbpf_velocity_kf:
        dd_pairs_for_kf_gate = (
            int(dd_gate_stats.n_kept_pairs)
            if dd_gate_stats is not None
            else (
                int(dd_carrier_result.n_dd)
                if dd_carrier_result is not None
                else 0
            )
        )
        if (
            rbpf_gate_min_dd_pairs is not None
            and dd_pairs_for_kf_gate < int(rbpf_gate_min_dd_pairs)
        ):
            return DopplerUpdateDecision(
                update=None,
                sigma_mps=None,
                use_kf=True,
                gate_reason="min_dd_pairs",
                skipped=True,
                gate_skipped=True,
            )
        if (
            rbpf_gate_min_ess_ratio is not None
            and (
                gate_ess_ratio is None
                or float(gate_ess_ratio) < float(rbpf_gate_min_ess_ratio)
            )
        ):
            return DopplerUpdateDecision(
                update=None,
                sigma_mps=None,
                use_kf=True,
                gate_reason="min_ess_ratio",
                skipped=True,
                gate_skipped=True,
            )
        if (
            rbpf_gate_max_spread_m is not None
            and (
                gate_spread_m is None
                or float(gate_spread_m) > float(rbpf_gate_max_spread_m)
            )
        ):
            return DopplerUpdateDecision(
                update=None,
                sigma_mps=None,
                use_kf=True,
                gate_reason="max_spread",
                skipped=True,
                gate_skipped=True,
            )

    sigma_mps = (
        float(rbpf_doppler_sigma)
        if rbpf_velocity_kf and rbpf_doppler_sigma is not None
        else float(doppler_sigma_mps)
    )
    return DopplerUpdateDecision(
        update={
            "sat_ecef": np.asarray(dop_sat, dtype=np.float64),
            "sat_vel": np.asarray(dop_sat_vel, dtype=np.float64),
            "doppler_hz": np.asarray(dop_obs, dtype=np.float64),
            "weights": np.asarray(dop_weights, dtype=np.float64),
            "wavelength_m": float(wavelength_m),
        },
        sigma_mps=sigma_mps,
        use_kf=bool(rbpf_velocity_kf),
        gate_reason="ok" if rbpf_velocity_kf else None,
        skipped=False,
        gate_skipped=False,
    )
