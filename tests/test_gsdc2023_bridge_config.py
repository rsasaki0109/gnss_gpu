from __future__ import annotations

import numpy as np
import pytest

from experiments.gsdc2023_bridge_config import (
    BridgeConfig,
    DEFAULT_CT_RBPF_MOTION_SIGMA_M,
    DEFAULT_MOTION_SIGMA_M,
    FACTOR_DT_MAX_S,
    should_refine_outlier_result,
)
from experiments.gsdc2023_tdcp import DEFAULT_TDCP_GEOMETRY_CORRECTION, DEFAULT_TDCP_WEIGHT_SCALE


def test_bridge_config_defaults_match_public_factor_dt() -> None:
    cfg = BridgeConfig()

    assert cfg.factor_dt_max_s == FACTOR_DT_MAX_S
    assert cfg.motion_sigma_m == DEFAULT_MOTION_SIGMA_M
    assert cfg.position_source == "baseline"
    assert cfg.imu_frame == "body"
    assert cfg.tdcp_weight_scale == DEFAULT_TDCP_WEIGHT_SCALE
    assert cfg.tdcp_geometry_correction is DEFAULT_TDCP_GEOMETRY_CORRECTION
    assert cfg.ct_rbpf_fgo_enabled is False
    assert cfg.ct_rbpf_motion_sigma_m == DEFAULT_CT_RBPF_MOTION_SIGMA_M
    assert cfg.fgo_raw_wls_proxy_rescue_enabled is False
    assert cfg.fgo_raw_wls_proxy_rescue_phones == ("pixel4",)


def test_bridge_config_rejects_invalid_position_source() -> None:
    with pytest.raises(ValueError):
        BridgeConfig(position_source="unsupported")


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("factor_dt_max_s", np.inf, "factor_dt_max_s must be finite"),
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
    ],
)
def test_bridge_config_rejects_non_finite_numeric_guards(field: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BridgeConfig(**{field: value})


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


def test_bridge_config_rejects_invalid_fgo_raw_wls_proxy_rescue_thresholds() -> None:
    with pytest.raises(ValueError, match="fgo_raw_wls_proxy_rescue_mse_ratio_max must be > 1"):
        BridgeConfig(fgo_raw_wls_proxy_rescue_mse_ratio_max=1.0)
    with pytest.raises(ValueError, match="fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max must be > 0"):
        BridgeConfig(fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max=0.0)


def test_should_refine_outlier_result_only_for_large_gated_auto_errors() -> None:
    assert should_refine_outlier_result("gated", 200, 1200.0) is True
    assert should_refine_outlier_result("auto", 200, 1000.1) is True
    assert should_refine_outlier_result("raw_wls", 200, 5000.0) is False
    assert should_refine_outlier_result("gated", 30, 5000.0) is False
    assert should_refine_outlier_result("gated", 200, 999.0) is False
