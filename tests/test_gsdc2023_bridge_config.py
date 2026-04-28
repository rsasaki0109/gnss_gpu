from __future__ import annotations

import numpy as np
import pytest

from experiments.gsdc2023_bridge_config import (
    BridgeConfig,
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


def test_bridge_config_rejects_invalid_position_source() -> None:
    with pytest.raises(ValueError):
        BridgeConfig(position_source="unsupported")


def test_bridge_config_rejects_invalid_imu_frame() -> None:
    with pytest.raises(ValueError, match="unsupported imu_frame"):
        BridgeConfig(imu_frame="device")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("factor_dt_max_s", np.inf, "factor_dt_max_s must be finite"),
        ("imu_accel_bias_prior_sigma_mps2", np.nan, "imu_accel_bias_prior_sigma_mps2 must be finite"),
        ("imu_accel_bias_between_sigma_mps2", np.inf, "imu_accel_bias_between_sigma_mps2 must be finite"),
    ],
)
def test_bridge_config_rejects_non_finite_numeric_guards(field: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BridgeConfig(**{field: value})


def test_should_refine_outlier_result_only_for_large_gated_auto_errors() -> None:
    assert should_refine_outlier_result("gated", 200, 1200.0) is True
    assert should_refine_outlier_result("auto", 200, 1000.1) is True
    assert should_refine_outlier_result("raw_wls", 200, 5000.0) is False
    assert should_refine_outlier_result("gated", 30, 5000.0) is False
    assert should_refine_outlier_result("gated", 200, 999.0) is False
