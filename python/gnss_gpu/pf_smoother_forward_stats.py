"""Forward-pass counters for PF smoother evaluations."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class ForwardRunStats:
    n_imu_used: int = 0
    n_imu_fallback: int = 0
    n_imu_stop_detected: int = 0
    n_dd_pr_used: int = 0
    n_dd_pr_skip: int = 0
    n_dd_pr_gate_pairs_rejected: int = 0
    n_dd_pr_gate_epoch_skip: int = 0
    n_wl_used: int = 0
    n_wl_skip: int = 0
    n_wl_candidate_pairs: int = 0
    n_wl_fixed_pairs: int = 0
    n_wl_low_fix_rate: int = 0
    n_wl_gate_skip: int = 0
    n_wl_gate_pair_rejected: int = 0
    n_dd_used: int = 0
    n_dd_skip: int = 0
    n_dd_skip_support_guard: int = 0
    n_dd_sigma_relaxed: int = 0
    dd_sigma_scale_sum: float = 0.0
    n_carrier_anchor_used: int = 0
    n_carrier_anchor_propagated: int = 0
    n_dd_fallback_undiff_used: int = 0
    n_dd_fallback_tracked_attempted: int = 0
    n_dd_fallback_tracked_used: int = 0
    n_dd_fallback_weak_dd_replaced: int = 0
    n_dd_gate_pairs_rejected: int = 0
    n_dd_gate_epoch_skip: int = 0
    n_tdcp_used: int = 0
    n_tdcp_fallback: int = 0
    n_tdcp_pu_used: int = 0
    n_tdcp_pu_skip: int = 0
    n_tdcp_pu_gate_skip: int = 0
    n_fgo_tdcp_motion_used: int = 0
    n_fgo_tdcp_motion_skip: int = 0
    n_doppler_pp_used: int = 0
    n_doppler_pp_skip: int = 0
    n_doppler_kf_used: int = 0
    n_doppler_kf_skip: int = 0
    n_doppler_kf_gate_skip: int = 0
    n_imu_tight_used: int = 0
    n_imu_tight_skip: int = 0

    def record_dd_sigma_relaxed(self, sigma_scale: float) -> None:
        self.n_dd_sigma_relaxed += 1
        self.dd_sigma_scale_sum += float(sigma_scale)

    def as_result_context(self) -> dict[str, int | float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}
