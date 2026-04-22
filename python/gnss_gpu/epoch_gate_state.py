"""Shared epoch gate-state calculation for PF smoother observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from gnss_gpu.dd_quality import ess_gate_scale, spread_gate_scale
from gnss_gpu.pf_smoother_config import PfSmootherConfigParts


class _ParticleFilterLike(Protocol):
    n_particles: int

    def estimate(self): ...

    def get_ess(self) -> float: ...

    def get_position_spread(self, *, center: np.ndarray) -> float: ...


@dataclass(frozen=True)
class EpochGateState:
    pf_estimate: np.ndarray | None
    ess_ratio: float | None
    spread_m: float | None
    dd_pr_gate_scale: float
    dd_cp_gate_scale: float


def compute_epoch_gate_state(
    pf: _ParticleFilterLike,
    config: PfSmootherConfigParts,
) -> EpochGateState:
    obs = config.observations
    tdcp = config.tdcp_position_update

    need_gate_est = (
        obs.dd_pseudorange.enabled
        or obs.dd_carrier.enabled
        or obs.widelane.enabled
    )
    need_gate_ess = (
        obs.dd_pseudorange.gate_ess_min_scale != 1.0
        or obs.dd_pseudorange.gate_ess_max_scale != 1.0
        or obs.dd_carrier.gate_ess_min_scale != 1.0
        or obs.dd_carrier.gate_ess_max_scale != 1.0
        or (
            obs.dd_carrier.sigma_ess_low_ratio is not None
            and obs.dd_carrier.sigma_ess_high_ratio is not None
        )
        or obs.carrier_rescue.skip_low_support_ess_ratio is not None
        or obs.dd_carrier.gate_low_ess_max_ratio is not None
        or (tdcp.enabled and tdcp.gate_max_ess_ratio is not None)
        or (tdcp.enabled and tdcp.gate_min_ess_ratio is not None)
    )
    need_gate_spread = (
        obs.dd_pseudorange.gate_spread_min_scale != 1.0
        or obs.dd_pseudorange.gate_spread_max_scale != 1.0
        or obs.dd_carrier.gate_spread_min_scale != 1.0
        or obs.dd_carrier.gate_spread_max_scale != 1.0
        or obs.carrier_rescue.skip_low_support_max_spread_m is not None
        or obs.dd_carrier.gate_low_ess_max_spread_m is not None
        or (obs.widelane.enabled and obs.widelane.gate_min_spread_m is not None)
        or (tdcp.enabled and tdcp.gate_min_spread_m is not None)
        or (tdcp.enabled and tdcp.gate_max_spread_m is not None)
    )

    pf_estimate = (
        np.asarray(pf.estimate()[:3], dtype=np.float64)
        if need_gate_est
        else None
    )
    ess_ratio = pf.get_ess() / float(pf.n_particles) if need_gate_ess else None
    spread_m = (
        pf.get_position_spread(center=pf_estimate)
        if need_gate_spread and pf_estimate is not None
        else None
    )

    dd_pr_gate_scale = 1.0
    if ess_ratio is not None:
        dd_pr_gate_scale *= ess_gate_scale(
            ess_ratio,
            min_scale=obs.dd_pseudorange.gate_ess_min_scale,
            max_scale=obs.dd_pseudorange.gate_ess_max_scale,
        )
    if spread_m is not None:
        dd_pr_gate_scale *= spread_gate_scale(
            spread_m,
            low_spread_m=obs.dd_pseudorange.gate_low_spread_m,
            high_spread_m=obs.dd_pseudorange.gate_high_spread_m,
            min_scale=obs.dd_pseudorange.gate_spread_min_scale,
            max_scale=obs.dd_pseudorange.gate_spread_max_scale,
        )

    dd_cp_gate_scale = 1.0
    if ess_ratio is not None:
        dd_cp_gate_scale *= ess_gate_scale(
            ess_ratio,
            min_scale=obs.dd_carrier.gate_ess_min_scale,
            max_scale=obs.dd_carrier.gate_ess_max_scale,
        )
    if spread_m is not None:
        dd_cp_gate_scale *= spread_gate_scale(
            spread_m,
            low_spread_m=obs.dd_carrier.gate_low_spread_m,
            high_spread_m=obs.dd_carrier.gate_high_spread_m,
            min_scale=obs.dd_carrier.gate_spread_min_scale,
            max_scale=obs.dd_carrier.gate_spread_max_scale,
        )

    return EpochGateState(
        pf_estimate=pf_estimate,
        ess_ratio=ess_ratio,
        spread_m=spread_m,
        dd_pr_gate_scale=float(dd_pr_gate_scale),
        dd_cp_gate_scale=float(dd_cp_gate_scale),
    )
