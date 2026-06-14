from __future__ import annotations

import numpy as np
import pytest

from experiments.gsdc2023_bridge_config import (
    BridgeConfig,
    DEFAULT_CT_RBPF_MOTION_SIGMA_M,
    DEFAULT_MOTION_SIGMA_M,
    FACTOR_DT_MAX_S,
    apply_taroz_fgo_preset,
    apply_taroz_gnss_only_preset,
    taroz_imu_noise_for_phone,
    should_refine_outlier_result,
)
from experiments.gsdc2023_output import TAROZ_FGO_CANDIDATE_SOURCES
from experiments.gsdc2023_tdcp import DEFAULT_TDCP_GEOMETRY_CORRECTION, DEFAULT_TDCP_WEIGHT_SCALE


def test_bridge_config_defaults_match_public_factor_dt() -> None:
    cfg = BridgeConfig()

    assert cfg.factor_dt_max_s == FACTOR_DT_MAX_S
    assert cfg.fgo_tol == pytest.approx(1e-7)
    assert cfg.motion_sigma_m == DEFAULT_MOTION_SIGMA_M
    assert cfg.position_source == "baseline"
    assert cfg.imu_frame == "body"
    assert cfg.imu_sample_dt_mode == "bounded"
    assert cfg.tdcp_weight_scale == DEFAULT_TDCP_WEIGHT_SCALE
    assert cfg.tdcp_l5_weight_scale == 1.0
    assert cfg.tdcp_geometry_correction is DEFAULT_TDCP_GEOMETRY_CORRECTION
    assert cfg.tdcp_cycle_jump_mask_cycles == 0.0
    assert cfg.tdcp_doppler_endpoint_mask is True
    assert cfg.ct_rbpf_fgo_enabled is False
    assert cfg.ct_rbpf_motion_sigma_m == DEFAULT_CT_RBPF_MOTION_SIGMA_M
    assert cfg.fgo_raw_wls_proxy_rescue_enabled is False
    assert cfg.fgo_raw_wls_proxy_rescue_phones == ("pixel4",)
    assert cfg.taroz_fgo_candidate_enabled is False
    assert cfg.taroz_fgo_candidate_sources == TAROZ_FGO_CANDIDATE_SOURCES
    assert cfg.stop_attitude_sigma_rad == 0.0
    assert cfg.taroz_imu_factor_mask_csv is None
    assert cfg.taroz_stop_mask_from_seed_velocity is False
    assert cfg.fgo_extra_constellations is False


def test_bridge_config_rejects_invalid_position_source() -> None:
    with pytest.raises(ValueError):
        BridgeConfig(position_source="unsupported")


def test_bridge_config_rejects_non_bool_fgo_extra_constellations() -> None:
    with pytest.raises(ValueError, match="fgo_extra_constellations must be a bool"):
        BridgeConfig(fgo_extra_constellations=1)  # type: ignore[arg-type]


def test_bridge_config_defaults_disable_glonass_constellation() -> None:
    assert BridgeConfig().fgo_glonass_constellation is False


def test_bridge_config_rejects_non_bool_fgo_glonass_constellation() -> None:
    with pytest.raises(ValueError, match="fgo_glonass_constellation must be a bool"):
        BridgeConfig(fgo_glonass_constellation=1)  # type: ignore[arg-type]


def test_taroz_presets_enable_base_correction_and_unscaled_tdcp_weights() -> None:
    fgo = apply_taroz_fgo_preset(BridgeConfig())
    gnss_only = apply_taroz_gnss_only_preset(BridgeConfig())

    assert fgo.apply_base_correction is True
    assert gnss_only.apply_base_correction is True
    assert fgo.tdcp_weight_scale == 1.0
    assert gnss_only.tdcp_weight_scale == 1.0
    assert fgo.tdcp_cycle_jump_mask_cycles == 0.0
    assert gnss_only.tdcp_cycle_jump_mask_cycles == 0.0
    assert fgo.tdcp_doppler_endpoint_mask is True
    assert gnss_only.tdcp_doppler_endpoint_mask is True


def test_bridge_config_requires_ct_candidate_for_direct_ct_source() -> None:
    with pytest.raises(ValueError, match="requires ct_rbpf_fgo_enabled=True"):
        BridgeConfig(position_source="fgo_ct_rbpf")

    cfg = BridgeConfig(position_source="fgo_ct_rbpf", ct_rbpf_fgo_enabled=True)

    assert cfg.position_source == "fgo_ct_rbpf"


def test_bridge_config_requires_dd_carrier_candidate_for_direct_source() -> None:
    with pytest.raises(ValueError, match="requires dd_carrier_fgo_enabled=True"):
        BridgeConfig(position_source="fgo_dd_carrier")

    cfg = BridgeConfig(position_source="fgo_dd_carrier", dd_carrier_fgo_enabled=True)

    assert cfg.position_source == "fgo_dd_carrier"
    assert cfg.dd_carrier_min_dd_pairs == 4


def test_bridge_config_rejects_invalid_imu_frame() -> None:
    with pytest.raises(ValueError, match="unsupported imu_frame"):
        BridgeConfig(imu_frame="device")


def test_bridge_config_rejects_invalid_imu_sample_dt_mode() -> None:
    with pytest.raises(ValueError, match="unsupported imu_sample_dt_mode"):
        BridgeConfig(imu_sample_dt_mode="centered")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("factor_dt_max_s", np.inf, "factor_dt_max_s must be finite"),
        ("fgo_tol", np.nan, "fgo_tol must be finite"),
        ("ct_rbpf_motion_sigma_m", np.nan, "ct_rbpf_motion_sigma_m must be finite"),
        ("tdcp_scale_candidate_weight_scale", np.nan, "tdcp_scale_candidate_weight_scale must be finite"),
        (
            "fgo_raw_wls_proxy_rescue_quality_delta_max",
            np.nan,
            "fgo_raw_wls_proxy_rescue_quality_delta_max must be finite",
        ),
        ("dd_carrier_sigma_cycles", np.nan, "dd_carrier_sigma_cycles must be finite"),
        ("imu_accel_bias_prior_sigma_mps2", np.nan, "imu_accel_bias_prior_sigma_mps2 must be finite"),
        ("imu_accel_bias_between_sigma_mps2", np.inf, "imu_accel_bias_between_sigma_mps2 must be finite"),
        ("imu_attitude_sigma_rad", np.nan, "imu_attitude_sigma_rad must be finite"),
        ("imu_acc_sigma_mps2_sqrt_hz", np.nan, "imu_acc_sigma_mps2_sqrt_hz must be finite"),
        ("imu_gyro_sigma_radps_sqrt_hz", np.inf, "imu_gyro_sigma_radps_sqrt_hz must be finite"),
    ],
)
def test_bridge_config_rejects_non_finite_numeric_guards(field: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BridgeConfig(**{field: value})


@pytest.mark.parametrize(
    ("phone", "acc_sigma", "gyro_sigma"),
    [
        ("pixel5", 0.05, 0.001),
        ("sm-g988b", 0.05, 0.001),
        ("samsunga325g", 0.05, 0.001),
        ("sm-a217m", 0.1, 0.005),
        ("sm-a325f", 0.1, 0.001),
        ("xiaomimi8", 0.05, 0.001),
    ],
)
def test_taroz_imu_noise_for_phone_matches_parameters_m(phone: str, acc_sigma: float, gyro_sigma: float) -> None:
    noise = taroz_imu_noise_for_phone(phone)

    assert noise.acc_sigma_mps2_sqrt_hz == pytest.approx(acc_sigma)
    assert noise.gyro_sigma_radps_sqrt_hz == pytest.approx(gyro_sigma)
    assert noise.acc_sync_coefficient == pytest.approx(0.5)
    assert noise.effective_acc_sigma_mps2_sqrt_hz == pytest.approx(0.5 * acc_sigma)
    assert noise.effective_gyro_sigma_radps_sqrt_hz == pytest.approx(0.5 * gyro_sigma)


def test_bridge_config_rejects_too_few_dd_pairs() -> None:
    with pytest.raises(ValueError, match="dd_carrier_min_dd_pairs must be >= 2"):
        BridgeConfig(dd_carrier_min_dd_pairs=1)


def test_bridge_config_default_dd_carrier_anchor_coverage_matches_chunk_selection_constant() -> None:
    from experiments.gsdc2023_chunk_selection import DD_CARRIER_ANCHOR_COVERAGE_MIN_DEFAULT

    cfg = BridgeConfig()
    assert cfg.dd_carrier_min_anchor_coverage == DD_CARRIER_ANCHOR_COVERAGE_MIN_DEFAULT


def test_bridge_config_rejects_out_of_range_dd_carrier_anchor_coverage() -> None:
    with pytest.raises(ValueError, match="dd_carrier_min_anchor_coverage must be in"):
        BridgeConfig(dd_carrier_min_anchor_coverage=1.5)
    with pytest.raises(ValueError, match="dd_carrier_min_anchor_coverage must be in"):
        BridgeConfig(dd_carrier_min_anchor_coverage=-0.1)


def test_bridge_config_rejects_non_finite_dd_carrier_anchor_coverage() -> None:
    with pytest.raises(ValueError, match="dd_carrier_min_anchor_coverage must be finite"):
        BridgeConfig(dd_carrier_min_anchor_coverage=float("nan"))


def test_bridge_config_rejects_non_positive_tdcp_scale_candidate() -> None:
    with pytest.raises(ValueError, match="tdcp_scale_candidate_weight_scale must be > 0"):
        BridgeConfig(tdcp_scale_candidate_weight_scale=0.0)


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_bridge_config_rejects_invalid_tdcp_l5_weight_scale(value: float) -> None:
    with pytest.raises(ValueError, match="tdcp_l5_weight_scale must be"):
        BridgeConfig(tdcp_l5_weight_scale=value)


def test_bridge_config_rejects_invalid_fgo_raw_wls_proxy_rescue_thresholds() -> None:
    with pytest.raises(ValueError, match="fgo_raw_wls_proxy_rescue_mse_ratio_max must be > 1"):
        BridgeConfig(fgo_raw_wls_proxy_rescue_mse_ratio_max=1.0)
    with pytest.raises(ValueError, match="fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max must be > 0"):
        BridgeConfig(fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max=0.0)


def test_bridge_config_rejects_invalid_taroz_candidate_sources() -> None:
    with pytest.raises(ValueError, match="unsupported taroz_fgo_candidate_source"):
        BridgeConfig(taroz_fgo_candidate_sources=("fgo_bad",))
    with pytest.raises(ValueError, match="taroz_fgo_candidate_sources must be unique"):
        BridgeConfig(
            taroz_fgo_candidate_sources=(
                TAROZ_FGO_CANDIDATE_SOURCES[0],
                TAROZ_FGO_CANDIDATE_SOURCES[0],
            ),
        )


def test_should_refine_outlier_result_only_for_large_gated_auto_errors() -> None:
    assert should_refine_outlier_result("gated", 200, 1200.0) is True
    assert should_refine_outlier_result("auto", 200, 1000.1) is True
    assert should_refine_outlier_result("raw_wls", 200, 5000.0) is False
    assert should_refine_outlier_result("gated", 30, 5000.0) is False
    assert should_refine_outlier_result("gated", 200, 999.0) is False
