"""Ground-truth alignment helpers for PF smoother forward epochs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from gnss_gpu.epoch_diagnostics import _build_epoch_diagnostic_row
from gnss_gpu.pf_smoother_epoch_state import EpochForwardState
from gnss_gpu.pf_smoother_runtime import ForwardRunBuffers


@dataclass(frozen=True)
class ForwardAlignmentResult:
    aligned: bool
    gt_index: int | None = None
    aligned_epoch_index: int | None = None


def append_forward_alignment(
    buffers: ForwardRunBuffers,
    *,
    run_name: str,
    tow: float,
    pf_estimate_now: np.ndarray,
    gt: np.ndarray,
    our_times: np.ndarray,
    measurements: Iterable[object],
    epoch_index: int,
    skip_valid_epochs: int,
    use_smoother: bool,
    collect_epoch_diagnostics: bool,
    epoch_state: EpochForwardState,
    rbpf_velocity_kf: bool,
    gate_ess_ratio: float,
    gate_spread_m: float,
    carrier_anchor_sigma_m: float,
    max_time_delta_s: float = 0.05,
) -> ForwardAlignmentResult:
    if int(epoch_index) < int(skip_valid_epochs):
        return ForwardAlignmentResult(aligned=False)

    gt_idx = int(np.argmin(np.abs(np.asarray(our_times, dtype=np.float64) - float(tow))))
    if abs(float(our_times[gt_idx]) - float(tow)) >= float(max_time_delta_s):
        return ForwardAlignmentResult(aligned=False, gt_index=gt_idx)

    gt_now = np.asarray(gt[gt_idx], dtype=np.float64).copy()
    aligned_epoch_index = buffers.append_aligned_epoch(
        pf_estimate_now,
        gt_now,
        stop_flag=bool(epoch_state.imu_stop_detected),
        use_smoother=use_smoother,
    )
    if collect_epoch_diagnostics:
        buffers.aligned_epoch_diagnostics.append(
            _build_epoch_diagnostic_row(
                run_name=run_name,
                tow=tow,
                aligned_epoch_index=aligned_epoch_index,
                store_epoch_index=int(buffers.n_stored - 1) if use_smoother else None,
                gt_index=gt_idx,
                n_measurements=len(list(measurements)),
                used_imu=epoch_state.used_imu,
                used_tdcp=epoch_state.used_tdcp,
                used_tdcp_pu_epoch=epoch_state.used_tdcp_pu_epoch,
                tdcp_pu_rms=epoch_state.tdcp_pu_rms,
                tdcp_pu_spp_diff_mps=epoch_state.tdcp_pu_spp_diff_mps,
                tdcp_pu_gate_reason=epoch_state.tdcp_pu_gate_reason,
                imu_stop_detected=epoch_state.imu_stop_detected,
                used_imu_tight_epoch=epoch_state.used_imu_tight_epoch,
                rbpf_velocity_kf=rbpf_velocity_kf,
                doppler_update_epoch=epoch_state.doppler_update_epoch,
                doppler_kf_gate_reason=epoch_state.doppler_kf_gate_reason,
                dd_pr_result=epoch_state.dd_pr_result,
                dd_pr_sigma_epoch=epoch_state.dd_pr_sigma_epoch,
                dd_pr_gate_stats=epoch_state.dd_pr_gate_stats,
                dd_pr_input_pairs=epoch_state.dd_pr_input_pairs,
                dd_pr_gate_scale=epoch_state.dd_pr_gate_scale,
                dd_pr_raw_abs_res_median_m=epoch_state.dd_pr_raw_abs_res_median_m,
                dd_pr_raw_abs_res_max_m=epoch_state.dd_pr_raw_abs_res_max_m,
                gate_ess_ratio=gate_ess_ratio,
                gate_spread_m=gate_spread_m,
                wl_stats=epoch_state.wl_stats,
                wl_gate_info=epoch_state.wl_gate_info,
                used_widelane_epoch=epoch_state.used_widelane_epoch,
                wl_input_pairs=epoch_state.wl_input_pairs,
                wl_fixed_pairs=epoch_state.wl_fixed_pairs,
                wl_fix_rate=epoch_state.wl_fix_rate,
                dd_carrier_result=epoch_state.dd_carrier_result,
                dd_gate_stats=epoch_state.dd_gate_stats,
                anchor_attempt=epoch_state.anchor_attempt,
                fallback_attempt=epoch_state.fallback_attempt,
                dd_cp_input_pairs=epoch_state.dd_cp_input_pairs,
                dd_cp_gate_scale=epoch_state.dd_cp_gate_scale,
                dd_cp_raw_abs_afv_median_cycles=epoch_state.dd_cp_raw_abs_afv_median_cycles,
                dd_cp_raw_abs_afv_max_cycles=epoch_state.dd_cp_raw_abs_afv_max_cycles,
                dd_cp_sigma_support_scale=epoch_state.dd_cp_sigma_support_scale,
                dd_cp_sigma_afv_scale=epoch_state.dd_cp_sigma_afv_scale,
                dd_cp_sigma_ess_scale=epoch_state.dd_cp_sigma_ess_scale,
                dd_cp_sigma_scale=epoch_state.dd_cp_sigma_scale,
                dd_cp_sigma_cycles=epoch_state.dd_cp_sigma_cycles,
                dd_cp_support_skip=epoch_state.dd_cp_support_skip,
                carrier_anchor_sigma_m=carrier_anchor_sigma_m,
            )
        )

    return ForwardAlignmentResult(
        aligned=True,
        gt_index=gt_idx,
        aligned_epoch_index=aligned_epoch_index,
    )
