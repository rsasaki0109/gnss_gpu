"""Configuration helpers for the GSDC2023 raw bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.gsdc2023_chunk_selection import GATED_BASELINE_THRESHOLD_DEFAULT
from experiments.gsdc2023_height_constraints import HEIGHT_ABSOLUTE_DIST_M, HEIGHT_ABSOLUTE_SIGMA_M
from experiments.gsdc2023_imu import (
    IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2,
    IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2,
    IMU_DELTA_FRAMES,
)
from experiments.gsdc2023_observation_matrix import (
    OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    OBS_MASK_MIN_CN0_DBHZ,
    OBS_MASK_MIN_ELEVATION_DEG,
    OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_M,
)
from experiments.gsdc2023_output import FACTOR_DT_MAX_S, validate_position_source
from experiments.gsdc2023_tdcp import (
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
)


OUTLIER_REFINEMENT_MSE_PR_THRESHOLD = 1000.0
OUTLIER_REFINEMENT_CHUNK_EPOCHS = 30
DEFAULT_MOTION_SIGMA_M = 0.2


@dataclass(frozen=True)
class BridgeConfig:
    motion_sigma_m: float = DEFAULT_MOTION_SIGMA_M
    clock_drift_sigma_m: float = 1.0
    factor_dt_max_s: float = FACTOR_DT_MAX_S
    fgo_iters: int = 8
    signal_type: str = "GPS_L1_CA"
    constellation_type: int = 1
    weight_mode: str = "sin2el"
    position_source: str = "baseline"
    chunk_epochs: int = 0
    gated_baseline_threshold: float = GATED_BASELINE_THRESHOLD_DEFAULT
    use_vd: bool = True
    multi_gnss: bool = True
    tdcp_enabled: bool = True
    tdcp_consistency_threshold_m: float = DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M
    tdcp_weight_scale: float = DEFAULT_TDCP_WEIGHT_SCALE
    tdcp_geometry_correction: bool = DEFAULT_TDCP_GEOMETRY_CORRECTION
    stop_velocity_sigma_mps: float = 0.0
    stop_position_sigma_m: float = 0.0
    apply_imu_prior: bool = False
    imu_frame: str = "body"
    imu_position_sigma_m: float = 25.0
    imu_velocity_sigma_mps: float = 5.0
    imu_accel_bias_state: bool = False
    imu_accel_bias_prior_sigma_mps2: float = IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2
    imu_accel_bias_between_sigma_mps2: float = IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2
    apply_absolute_height: bool = False
    absolute_height_sigma_m: float = HEIGHT_ABSOLUTE_SIGMA_M
    absolute_height_dist_m: float = HEIGHT_ABSOLUTE_DIST_M
    apply_relative_height: bool = False
    apply_position_offset: bool = False
    apply_base_correction: bool = False
    graph_relative_height: bool = False
    relative_height_sigma_m: float = 0.5
    apply_observation_mask: bool = False
    observation_min_cn0_dbhz: float = OBS_MASK_MIN_CN0_DBHZ
    observation_min_elevation_deg: float = OBS_MASK_MIN_ELEVATION_DEG
    pseudorange_residual_mask_m: float = OBS_MASK_RESIDUAL_THRESHOLD_M
    pseudorange_residual_mask_l5_m: float | None = None
    doppler_residual_mask_mps: float = OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS
    pseudorange_doppler_mask_m: float = OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M
    matlab_residual_diagnostics_mask_path: Path | None = None
    dual_frequency: bool = False

    def __post_init__(self) -> None:
        validate_position_source(self.position_source)
        if self.imu_frame not in IMU_DELTA_FRAMES:
            raise ValueError(f"unsupported imu_frame: {self.imu_frame}")
        if not np.isfinite(self.factor_dt_max_s):
            raise ValueError(f"factor_dt_max_s must be finite, got {self.factor_dt_max_s!r}")
        if not np.isfinite(self.imu_accel_bias_prior_sigma_mps2):
            raise ValueError("imu_accel_bias_prior_sigma_mps2 must be finite")
        if not np.isfinite(self.imu_accel_bias_between_sigma_mps2):
            raise ValueError("imu_accel_bias_between_sigma_mps2 must be finite")


def should_refine_outlier_result(position_source: str, chunk_epochs: int, selected_mse_pr: float) -> bool:
    if position_source not in {"auto", "gated"}:
        return False
    if chunk_epochs <= OUTLIER_REFINEMENT_CHUNK_EPOCHS:
        return False
    return float(selected_mse_pr) > OUTLIER_REFINEMENT_MSE_PR_THRESHOLD


__all__ = [
    "BridgeConfig",
    "DEFAULT_MOTION_SIGMA_M",
    "FACTOR_DT_MAX_S",
    "OUTLIER_REFINEMENT_CHUNK_EPOCHS",
    "OUTLIER_REFINEMENT_MSE_PR_THRESHOLD",
    "should_refine_outlier_result",
]
