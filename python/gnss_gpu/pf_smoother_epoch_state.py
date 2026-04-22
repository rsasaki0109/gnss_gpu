"""Per-epoch mutable state for PF smoother forward passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gnss_gpu.carrier_rescue import CarrierAnchorAttempt, CarrierFallbackAttempt


def default_widelane_gate_info() -> dict[str, object]:
    return {
        "reason": None,
        "pair_rejected": 0,
        "raw_abs_res_median_m": None,
        "raw_abs_res_max_m": None,
        "kept_abs_res_median_m": None,
        "kept_abs_res_max_m": None,
    }


@dataclass
class EpochForwardState:
    dd_pr_sigma_epoch: float
    velocity: np.ndarray | None = None
    fgo_tdcp_motion_velocity: np.ndarray | None = None
    tdcp_pu_velocity: np.ndarray | None = None
    tdcp_pu_rms: float = field(default_factory=lambda: float("nan"))
    tdcp_pu_spp_diff_mps: float | None = None
    tdcp_pu_reason: str | None = None
    tdcp_pu_gate_reason: str | None = None
    used_tdcp_pu_epoch: bool = False
    imu_velocity: np.ndarray | None = None
    used_tdcp: bool = False
    tdcp_rms: float = field(default_factory=lambda: float("nan"))
    used_imu: bool = False
    imu_stop_detected: bool = False
    used_imu_tight_epoch: bool = False
    dd_pr_gate_stats: Any | None = None
    dd_gate_stats: Any | None = None
    dd_pr_gate_scale: float | None = None
    dd_cp_gate_scale: float | None = None
    dd_pr_input_pairs: int = 0
    dd_cp_input_pairs: int = 0
    dd_pr_raw_abs_res_median_m: float | None = None
    dd_pr_raw_abs_res_max_m: float | None = None
    wl_stats: Any | None = None
    wl_fix_rate: float | None = None
    wl_input_pairs: int = 0
    wl_fixed_pairs: int = 0
    wl_gate_info: dict[str, object] = field(default_factory=default_widelane_gate_info)
    used_widelane_epoch: bool = False
    dd_cp_raw_abs_afv_median_cycles: float | None = None
    dd_cp_raw_abs_afv_max_cycles: float | None = None
    dd_cp_sigma_support_scale: float = 1.0
    dd_cp_sigma_afv_scale: float = 1.0
    dd_cp_sigma_ess_scale: float = 1.0
    dd_cp_sigma_scale: float = 1.0
    dd_cp_sigma_cycles: float | None = None
    dd_cp_support_skip: bool = False
    carrier_anchor_rows: dict[tuple[int, int], dict[str, object]] = field(default_factory=dict)
    doppler_update_epoch: Any | None = None
    doppler_sigma_epoch: float | None = None
    doppler_kf_gate_reason: str | None = None
    anchor_attempt: CarrierAnchorAttempt = field(default_factory=CarrierAnchorAttempt)
    fallback_attempt: CarrierFallbackAttempt = field(default_factory=CarrierFallbackAttempt)
    dd_pr_result: Any | None = None
    dd_carrier_result: Any | None = None


def create_epoch_forward_state(dd_pseudorange_sigma: float) -> EpochForwardState:
    return EpochForwardState(dd_pr_sigma_epoch=float(dd_pseudorange_sigma))
