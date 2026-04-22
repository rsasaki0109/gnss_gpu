import math

from gnss_gpu.carrier_rescue import CarrierAnchorAttempt, CarrierFallbackAttempt
from gnss_gpu.pf_smoother_epoch_state import (
    create_epoch_forward_state,
    default_widelane_gate_info,
)


def test_create_epoch_forward_state_sets_legacy_defaults():
    state = create_epoch_forward_state(0.75)

    assert state.dd_pr_sigma_epoch == 0.75
    assert state.velocity is None
    assert state.fgo_tdcp_motion_velocity is None
    assert state.tdcp_pu_velocity is None
    assert math.isnan(state.tdcp_pu_rms)
    assert state.tdcp_pu_spp_diff_mps is None
    assert state.tdcp_pu_reason is None
    assert state.tdcp_pu_gate_reason is None
    assert state.used_tdcp_pu_epoch is False
    assert state.imu_velocity is None
    assert state.used_tdcp is False
    assert math.isnan(state.tdcp_rms)
    assert state.used_imu is False
    assert state.imu_stop_detected is False
    assert state.used_imu_tight_epoch is False
    assert state.dd_pr_gate_stats is None
    assert state.dd_gate_stats is None
    assert state.dd_pr_gate_scale is None
    assert state.dd_cp_gate_scale is None
    assert state.dd_pr_input_pairs == 0
    assert state.dd_cp_input_pairs == 0
    assert state.dd_pr_raw_abs_res_median_m is None
    assert state.dd_pr_raw_abs_res_max_m is None
    assert state.wl_stats is None
    assert state.wl_fix_rate is None
    assert state.wl_input_pairs == 0
    assert state.wl_fixed_pairs == 0
    assert state.wl_gate_info == default_widelane_gate_info()
    assert state.used_widelane_epoch is False
    assert state.dd_cp_raw_abs_afv_median_cycles is None
    assert state.dd_cp_raw_abs_afv_max_cycles is None
    assert state.dd_cp_sigma_support_scale == 1.0
    assert state.dd_cp_sigma_afv_scale == 1.0
    assert state.dd_cp_sigma_ess_scale == 1.0
    assert state.dd_cp_sigma_scale == 1.0
    assert state.dd_cp_sigma_cycles is None
    assert state.dd_cp_support_skip is False
    assert state.carrier_anchor_rows == {}
    assert state.doppler_update_epoch is None
    assert state.doppler_sigma_epoch is None
    assert state.doppler_kf_gate_reason is None
    assert isinstance(state.anchor_attempt, CarrierAnchorAttempt)
    assert isinstance(state.fallback_attempt, CarrierFallbackAttempt)
    assert state.dd_pr_result is None
    assert state.dd_carrier_result is None


def test_create_epoch_forward_state_uses_independent_mutable_defaults():
    first = create_epoch_forward_state(0.75)
    second = create_epoch_forward_state(0.75)

    first.wl_gate_info["reason"] = "gate"
    first.carrier_anchor_rows[(1, 2)] = {"ok": True}

    assert second.wl_gate_info["reason"] is None
    assert second.carrier_anchor_rows == {}
    assert first.anchor_attempt is not second.anchor_attempt
    assert first.fallback_attempt is not second.fallback_attempt
