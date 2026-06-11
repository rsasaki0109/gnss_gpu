"""Configuration helpers for the GSDC2023 raw bridge."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from experiments.gsdc2023_chunk_selection import (
    DD_CARRIER_ANCHOR_COVERAGE_MIN_DEFAULT,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    GATED_FGO_RAW_WLS_PROXY_RESCUE_GAP_STEP_P95_RATIO_MAX,
    GATED_FGO_RAW_WLS_PROXY_RESCUE_MSE_DELTA_MAX,
    GATED_FGO_RAW_WLS_PROXY_RESCUE_MSE_RATIO_MAX,
    GATED_FGO_RAW_WLS_PROXY_RESCUE_QUALITY_DELTA_MAX,
)
from experiments.gsdc2023_height_constraints import HEIGHT_ABSOLUTE_DIST_M, HEIGHT_ABSOLUTE_SIGMA_M
from experiments.gsdc2023_imu import (
    IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2,
    IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2,
    IMU_DELTA_FRAMES,
    IMU_SAMPLE_DT_MODES,
    IMU_TAROZ_BODY_DELTA_FRAME,
)
from experiments.gsdc2023_observation_matrix import (
    OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    OBS_MASK_MIN_CN0_DBHZ,
    OBS_MASK_MIN_ELEVATION_DEG,
    OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    OBS_MASK_RESIDUAL_THRESHOLD_M,
)
from experiments.gsdc2023_output import (
    CT_RBPF_FGO_SOURCE,
    DD_CARRIER_FGO_SOURCE,
    FACTOR_DT_MAX_S,
    TAROZ_FGO_CANDIDATE_SOURCES,
    validate_position_source,
)
from experiments.gsdc2023_tdcp import (
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
)


OUTLIER_REFINEMENT_MSE_PR_THRESHOLD = 1000.0
OUTLIER_REFINEMENT_CHUNK_EPOCHS = 30
DEFAULT_MOTION_SIGMA_M = 0.2
DEFAULT_CT_RBPF_MOTION_SIGMA_M = DEFAULT_MOTION_SIGMA_M
TAROZ_FGO_WEIGHT_MODE = "taroz_sn"
TAROZ_CLOCK_DRIFT_SIGMA_M = 0.1
TAROZ_STOP_VELOCITY_SIGMA_MPS = 0.01
TAROZ_STOP_POSITION_SIGMA_M = 0.02
TAROZ_STOP_ATTITUDE_SIGMA_RAD = float(np.deg2rad(0.1))
TAROZ_STOP_VELOCITY_THRESHOLD_MPS = 0.5
TAROZ_HEIGHT_SIGMA_M = 0.1
TAROZ_STOP_HUBER_K = 0.5
TAROZ_HEIGHT_HUBER_K = 0.5
TAROZ_DUAL_FREQUENCY_ELEVATION_DEG = 5.0
TAROZ_IMU_DELTA_FRAME = IMU_TAROZ_BODY_DELTA_FRAME
TAROZ_IMU_SAMPLE_DT_MODE = "taroz"
TAROZ_IMU_INTEGRATION_SIGMA = 0.05
TAROZ_IMU_SYNC_COEFFICIENT = 0.5
TAROZ_IMU_ACCEL_SIGMA_DEFAULT_MPS2_SQRT_HZ = 0.05
TAROZ_IMU_GYRO_SIGMA_DEFAULT_RADPS_SQRT_HZ = 0.001
TAROZ_IMU_ACCEL_BIAS_SIGMA_MPS2 = 0.00025
TAROZ_IMU_GYRO_BIAS_SIGMA_RADPS = 0.0000005


@dataclass(frozen=True)
class TarozImuNoise:
    acc_sigma_mps2_sqrt_hz: float
    gyro_sigma_radps_sqrt_hz: float
    acc_sync_coefficient: float = TAROZ_IMU_SYNC_COEFFICIENT
    gyro_sync_coefficient: float = TAROZ_IMU_SYNC_COEFFICIENT
    integration_sigma: float = TAROZ_IMU_INTEGRATION_SIGMA
    acc_bias_sigma_mps2: float = TAROZ_IMU_ACCEL_BIAS_SIGMA_MPS2
    gyro_bias_sigma_radps: float = TAROZ_IMU_GYRO_BIAS_SIGMA_RADPS

    @property
    def effective_acc_sigma_mps2_sqrt_hz(self) -> float:
        return float(self.acc_sync_coefficient * self.acc_sigma_mps2_sqrt_hz)

    @property
    def effective_gyro_sigma_radps_sqrt_hz(self) -> float:
        return float(self.gyro_sync_coefficient * self.gyro_sigma_radps_sqrt_hz)


def taroz_imu_noise_for_phone(phone: str) -> TarozImuNoise:
    """Return the Taroz ``parameters.m`` IMU noise row for a phone name."""
    phone_l = str(phone).lower()
    if "pixel" in phone_l:
        acc_sigma = 0.05
        gyro_sigma = 0.001
    elif any(needle in phone_l for needle in ("sm-s908", "sm-g988", "sm-g325f", "samsun")):
        acc_sigma = 0.05
        gyro_sigma = 0.001
    elif "sm-a217m" in phone_l:
        acc_sigma = 0.1
        gyro_sigma = 0.005
    elif "sm" in phone_l:
        acc_sigma = 0.1
        gyro_sigma = 0.001
    elif "mi" in phone_l:
        acc_sigma = 0.05
        gyro_sigma = 0.001
    else:
        acc_sigma = TAROZ_IMU_ACCEL_SIGMA_DEFAULT_MPS2_SQRT_HZ
        gyro_sigma = TAROZ_IMU_GYRO_SIGMA_DEFAULT_RADPS_SQRT_HZ
    return TarozImuNoise(
        acc_sigma_mps2_sqrt_hz=float(acc_sigma),
        gyro_sigma_radps_sqrt_hz=float(gyro_sigma),
    )


@dataclass(frozen=True)
class BridgeConfig:
    motion_sigma_m: float = DEFAULT_MOTION_SIGMA_M
    clock_drift_sigma_m: float = 1.0
    # None keeps the existing phone/TDCP-derived solver context.  True uses
    # gtsam_gnss ClockFactor_CCDD parity: average drift between adjacent epochs.
    clock_use_average_drift: bool | None = None
    factor_dt_max_s: float = FACTOR_DT_MAX_S
    fgo_iters: int = 8
    fgo_tol: float = 1e-7
    fgo_line_search: bool = True
    fgo_lm_damping: float = 0.0
    signal_type: str = "GPS_L1_CA"
    constellation_type: int = 1
    weight_mode: str = "sin2el"
    # When set to a different mode (e.g. "taroz_sn"), the FGO solver uses a
    # separate weights array while the gate/WLS keeps using ``weight_mode``.
    # ``None`` (default) shares ``weight_mode`` for both, matching legacy.
    fgo_weight_mode: str | None = None
    # Robust kernel applied inside the FGO solver. "huber" (default) is the
    # legacy in-CUDA Huber IRLS; "cauchy" wraps the native solver in a
    # Python-side Cauchy IRLS loop and is targeted at NLOS-heavy trips.
    fgo_robust_kernel: str = "huber"
    fgo_cauchy_c_m: float = 4.0
    fgo_cauchy_outer_iters: int = 3
    # lever 2: enable the taroz parameters.m per-Type / per-phone huber and
    # motion-sigma overrides.  When enabled, raw_bridge looks up the trip
    # Type from ``settings_{split}.csv`` and overrides ``fgo_huber_k_pr``,
    # ``fgo_huber_k_doppler``, ``fgo_huber_k_tdcp`` and ``motion_sigma_m``
    # based on the taroz parameters.m tables (with mi8 / pixel4 phone
    # overrides).  Note the existing default is ``fgo_huber_k_pr = 0.0``
    # (pure L2) which silently disabled Huber in production; enabling
    # per-Type kernel is the first time PR Huber is active on this codepath.
    per_type_kernel_enabled: bool = False
    # When ``per_type_kernel_enabled`` is True these sub-flags control which
    # field overrides are pulled from the taroz Type lookup.  The motion-sigma
    # override (taroz Highway=0.01 / Street=0.05 / mi8=0.1) is much tighter
    # than the pipeline default (0.2) and breaks several mi8/Highway trips on
    # its own, so it is gated separately and OFF by default.  Huber threshold
    # 0.1-0.2 is a safer drop-in.
    per_type_kernel_huber_enabled: bool = True
    per_type_kernel_motion_enabled: bool = False
    fgo_huber_k_pr: float = 0.0
    fgo_huber_k_doppler: float = 0.0
    fgo_huber_k_tdcp: float = 0.0
    # Taroz gtsam_gnss factors use fixed LOS/origin linearized GNSS factors.
    # Keep this off for legacy native FGO and enable it only for Taroz parity
    # presets/candidates.
    fgo_fixed_linearization: bool = False
    # Optional Taroz-export initial state CSV used only as the FGO seed.
    # Positions/velocities in Taroz GNSS CSVs are local ENU and are converted
    # to ECEF by the raw bridge using the trip preprocessing origin.
    taroz_fgo_seed_state_csv: Path | None = None
    # Optional Taroz IMU state CSV used to seed the full-FGO Pose3 and
    # ConstantBias keys separately from the GNSS x/v/c/d seed.  This is needed
    # for Taroz full-graph parity because x and p are distinct keys.
    taroz_pose_bias_seed_state_csv: Path | None = None
    # Optional Taroz-export GNSS factor mask CSV used to filter FGO-only P/D/L
    # weights to the exact exported factor support.
    taroz_factor_mask_csv: Path | None = None
    # Optional Taroz-export IMU factor mask CSV used to filter IMU factors to
    # the exact exported interval support for full FGO parity runs.
    taroz_imu_factor_mask_csv: Path | None = None
    # Optional Taroz-export IMU preintegration CSV used to replace native
    # Python preintegration for full FGO parity runs.
    taroz_imu_preintegration_csv: Path | None = None
    # Taroz full FGO gates stop factors by the initial velocity norm
    # (parameters.m stop_v_th=0.5 m/s) in addition to the device stop mask.
    taroz_stop_mask_from_seed_velocity: bool = False
    # Gate relaxation knobs.  Defaults (None) preserve legacy constants
    # GATED_LOW_BASELINE_FGO_MSE_PR_MAX=9.3 and GATED_FGO_BASELINE_MSE_PR_MIN=20.
    # Setting these higher lets more FGO candidates through (intended for the
    # Cauchy IRLS pipeline whose FGO mse_pr is ~3x the L2 baseline scale).
    gate_fgo_low_baseline_mse_pr_max: float | None = None
    gate_fgo_baseline_mse_pr_min: float | None = None
    gate_fgo_baseline_gap_p95_floor_m: float | None = None
    # Step 3: TEASER-light pairwise consistency pre-filter on PR observations.
    # When enabled, MAD-based per-system outlier masking is applied to the
    # FGO weights array before the solver runs.  Gate/WLS weights are not
    # touched (keeps the legacy mse_pr scale).
    pairwise_consistency_enabled: bool = False
    pairwise_consistency_mad_threshold_m: float = 3.5
    pairwise_consistency_min_obs_after_filter: int = 5
    # lever 4: TEASER max-clique consensus pre-filter (full pairwise graph).
    # Mutually exclusive with pairwise_consistency_enabled — both apply the
    # same FGO-only weight mask but with different inlier-selection logic.
    max_clique_filter_enabled: bool = False
    max_clique_filter_pair_threshold_m: float = 3.0
    max_clique_filter_min_clique_size: int = 5
    # baseline-improvement lever: Hatch carrier-phase smoothing on the raw
    # pseudorange matrix before WLS/FGO.  Reduces multipath/receiver noise
    # by ~1/sqrt(N) while leaving slow-varying bias intact.  Affects all
    # downstream candidates (baseline kaggle_wls untouched).
    hatch_smoothing_enabled: bool = False
    hatch_smoothing_n: int = 100
    # baseline-improvement lever: swap Android-provided TroposphericDelayMeters
    # for our recomputed Saastamoinen tropo (rtklib_tropo_saastamoinen).  Affects
    # all downstream candidates that use ``batch.pseudorange``; kaggle_wls
    # baseline source is unchanged.
    use_rtklib_tropo: bool = False
    # post-process lever: Hampel filter on the final per-tripId lat/lng
    # trajectory.  Applied at submission-CSV build time, *after* gate-based
    # source selection.  Removes 1-Hz outlier spikes from the baseline
    # trajectory (physically-impossible jumps).  Defaults: W21 k=2.5
    # passes=3 mad_floor=5e-7 deg (~5 cm).  See
    # ``experiments.postprocess_gsdc2023_submission_hampel``.
    hampel_postprocess_enabled: bool = False
    hampel_postprocess_window: int = 21
    hampel_postprocess_k: float = 2.5
    hampel_postprocess_passes: int = 3
    hampel_postprocess_mad_floor_deg: float = 5e-7
    # post-process lever: motion-acceleration outlier smoother.  Independent
    # signal from Hampel (motion-physics vs position-MAD), stacks additively.
    # Applied *after* Hampel.  Defaults: accel_max=3.0 m/s², passes=2 — the
    # train-trip sweet spot (-15 cm aggregate on top of Hampel).  See
    # ``experiments.postprocess_gsdc2023_submission_accel_smooth``.
    accel_smoother_enabled: bool = False
    accel_smoother_accel_max: float = 3.0
    accel_smoother_passes: int = 2
    # post-process lever: stationary-segment median snap.  Independent signal
    # from Hampel and accel-smoother (motion vs stationary, GNSS wobble while
    # parked).  Applied last.  Defaults: move_threshold_m=2.0 (~7.2 km/h)
    # min_run_length=10 (~10 s at 1 Hz) — train-trip sweet spot.
    stop_snap_enabled: bool = False
    stop_snap_move_threshold_m: float = 2.0
    stop_snap_min_run_length: int = 10
    # post-process lever: heading-consistency smoother.  Detects physically
    # impossible yaw rate (>heading_max_dps deg/s).  Independent of Hampel
    # (position MAD), accel-smoother (magnitude) and stop-snap (stationary).
    # Default heading_max_dps=45 (sweet spot on train trips; 30 false-flags
    # urban intersection turns).  Applied last.
    heading_smoother_enabled: bool = False
    heading_smoother_max_dps: float = 45.0
    # post-process lever: 1D RTS Kalman smoother (CV motion model).  Applies
    # uniform sub-metre smoothing over the entire trajectory; orthogonal to
    # the four outlier-replacement layers.  Defaults sigma_a=1.0 (m/s²),
    # sigma_z=1.0 (m) — train-trip sweet spot (-9.6 cm aggregate on top of v7,
    # 39 wins / 1 wash / 1 regression).  Applied last.
    kalman_smoother_enabled: bool = False
    kalman_smoother_sigma_a: float = 1.0
    kalman_smoother_sigma_z: float = 1.0
    position_source: str = "baseline"
    chunk_epochs: int = 0
    gated_baseline_threshold: float = GATED_BASELINE_THRESHOLD_DEFAULT
    use_vd: bool = True
    multi_gnss: bool = True
    tdcp_enabled: bool = True
    tdcp_consistency_threshold_m: float = DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M
    tdcp_weight_scale: float = DEFAULT_TDCP_WEIGHT_SCALE
    tdcp_geometry_correction: bool = DEFAULT_TDCP_GEOMETRY_CORRECTION
    tdcp_scale_candidate_enabled: bool = False
    tdcp_scale_candidate_weight_scale: float = 1.0e-7
    tdcp_scale_candidate_phones: tuple[str, ...] = ("pixel4", "pixel4xl", "mi8")
    fgo_raw_wls_proxy_rescue_enabled: bool = False
    fgo_raw_wls_proxy_rescue_phones: tuple[str, ...] = ("pixel4",)
    fgo_raw_wls_proxy_rescue_mse_ratio_max: float = GATED_FGO_RAW_WLS_PROXY_RESCUE_MSE_RATIO_MAX
    fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max: float = GATED_FGO_RAW_WLS_PROXY_RESCUE_GAP_STEP_P95_RATIO_MAX
    fgo_raw_wls_proxy_rescue_quality_delta_max: float = GATED_FGO_RAW_WLS_PROXY_RESCUE_QUALITY_DELTA_MAX
    fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max: float = GATED_FGO_RAW_WLS_PROXY_RESCUE_MSE_DELTA_MAX
    taroz_fgo_candidate_enabled: bool = False
    taroz_fgo_candidate_sources: tuple[str, ...] = TAROZ_FGO_CANDIDATE_SOURCES
    stop_velocity_sigma_mps: float = 0.0
    stop_position_sigma_m: float = 0.0
    stop_attitude_sigma_rad: float = 0.0
    stop_velocity_huber_k: float = 0.0
    stop_position_huber_k: float = 0.0
    apply_imu_prior: bool = False
    imu_frame: str = "body"
    imu_sample_dt_mode: str = "bounded"
    imu_position_sigma_m: float = 25.0
    imu_velocity_sigma_mps: float = 5.0
    imu_attitude_state: bool = False
    imu_attitude_sigma_rad: float = 0.0
    imu_preintegration_velocity_noise_mps_sqrt_hz: float = 0.0
    imu_preintegration_attitude_noise_rad_sqrt_hz: float = 0.0
    imu_diagonal_covariance: bool = False
    imu_preintegration_covariance: bool = False
    imu_factor_use_next_bias: bool = False
    imu_bias_between_sample_count_scaling: bool = False
    imu_accel_bias_state: bool = False
    imu_accel_bias_prior_sigma_mps2: float = IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2
    imu_accel_bias_between_sigma_mps2: float = IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2
    imu_gyro_bias_state: bool = False
    imu_gyro_bias_prior_sigma_radps: float = 0.0
    taroz_imu_noise_enabled: bool = False
    imu_acc_sigma_mps2_sqrt_hz: float = 0.0
    imu_gyro_sigma_radps_sqrt_hz: float = 0.0
    imu_acc_sync_coefficient: float = 1.0
    imu_gyro_sync_coefficient: float = 1.0
    imu_gyro_bias_between_sigma_radps: float = 0.0
    apply_absolute_height: bool = False
    absolute_height_sigma_m: float = HEIGHT_ABSOLUTE_SIGMA_M
    absolute_height_dist_m: float = HEIGHT_ABSOLUTE_DIST_M
    apply_relative_height: bool = False
    apply_position_offset: bool = False
    apply_base_correction: bool = False
    graph_relative_height: bool = False
    relative_height_sigma_m: float = 0.5
    relative_height_huber_k: float = 0.0
    absolute_height_huber_k: float = 0.0
    apply_observation_mask: bool = False
    observation_min_cn0_dbhz: float = OBS_MASK_MIN_CN0_DBHZ
    observation_min_elevation_deg: float = OBS_MASK_MIN_ELEVATION_DEG
    pseudorange_residual_mask_m: float = OBS_MASK_RESIDUAL_THRESHOLD_M
    pseudorange_residual_mask_l5_m: float | None = None
    doppler_residual_mask_mps: float = OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS
    pseudorange_doppler_mask_m: float = OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M
    matlab_residual_diagnostics_mask_path: Path | None = None
    dual_frequency: bool = False
    ct_rbpf_fgo_enabled: bool = False
    ct_rbpf_motion_sigma_m: float = DEFAULT_CT_RBPF_MOTION_SIGMA_M
    dd_carrier_fgo_enabled: bool = False
    dd_carrier_tow_snap_tolerance_s: float = 0.6
    dd_carrier_min_dd_pairs: int = 4
    dd_carrier_sigma_cycles: float = 0.12
    dd_carrier_prior_sigma_m: float = 1.5
    dd_carrier_max_shift_m: float = 3.0
    dd_carrier_max_initial_rms_m: float = 0.40
    dd_carrier_max_final_rms_m: float = 0.25
    dd_carrier_smooth_corrections: bool = False
    dd_carrier_anchor_correction_sigma_m: float = 0.5
    dd_carrier_correction_smooth_sigma_m: float = 0.25
    dd_carrier_correction_zero_sigma_m: float = 5.0
    dd_carrier_base_obs_template: str | None = None
    dd_carrier_require_base_obs_template: bool = False
    dd_carrier_min_anchor_coverage: float = DD_CARRIER_ANCHOR_COVERAGE_MIN_DEFAULT

    def __post_init__(self) -> None:
        validate_position_source(self.position_source)
        if self.position_source == CT_RBPF_FGO_SOURCE and not self.ct_rbpf_fgo_enabled:
            raise ValueError(f"{CT_RBPF_FGO_SOURCE} position_source requires ct_rbpf_fgo_enabled=True")
        if self.position_source == DD_CARRIER_FGO_SOURCE and not self.dd_carrier_fgo_enabled:
            raise ValueError(f"{DD_CARRIER_FGO_SOURCE} position_source requires dd_carrier_fgo_enabled=True")
        if self.imu_frame not in IMU_DELTA_FRAMES:
            raise ValueError(f"unsupported imu_frame: {self.imu_frame}")
        if self.imu_sample_dt_mode not in IMU_SAMPLE_DT_MODES:
            raise ValueError(f"unsupported imu_sample_dt_mode: {self.imu_sample_dt_mode}")
        if not np.isfinite(self.factor_dt_max_s):
            raise ValueError(f"factor_dt_max_s must be finite, got {self.factor_dt_max_s!r}")
        if not np.isfinite(float(self.fgo_tol)):
            raise ValueError("fgo_tol must be finite")
        if float(self.fgo_tol) <= 0.0:
            raise ValueError("fgo_tol must be > 0")
        if not np.isfinite(float(self.fgo_lm_damping)):
            raise ValueError("fgo_lm_damping must be finite")
        if float(self.fgo_lm_damping) < 0.0:
            raise ValueError("fgo_lm_damping must be >= 0")
        for name in (
            "stop_velocity_huber_k",
            "stop_position_huber_k",
            "relative_height_huber_k",
            "absolute_height_huber_k",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if not np.isfinite(self.tdcp_scale_candidate_weight_scale):
            raise ValueError("tdcp_scale_candidate_weight_scale must be finite")
        if float(self.tdcp_scale_candidate_weight_scale) <= 0.0:
            raise ValueError("tdcp_scale_candidate_weight_scale must be > 0")
        for name in (
            "fgo_raw_wls_proxy_rescue_mse_ratio_max",
            "fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max",
            "fgo_raw_wls_proxy_rescue_quality_delta_max",
            "fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if float(self.fgo_raw_wls_proxy_rescue_mse_ratio_max) <= 1.0:
            raise ValueError("fgo_raw_wls_proxy_rescue_mse_ratio_max must be > 1")
        if float(self.fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max) <= 0.0:
            raise ValueError("fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max must be > 0")
        taroz_candidate_sources = tuple(str(source) for source in self.taroz_fgo_candidate_sources)
        if len(taroz_candidate_sources) != len(set(taroz_candidate_sources)):
            raise ValueError("taroz_fgo_candidate_sources must be unique")
        allowed_taroz_sources = set(TAROZ_FGO_CANDIDATE_SOURCES)
        for source in taroz_candidate_sources:
            if source not in allowed_taroz_sources:
                raise ValueError(f"unsupported taroz_fgo_candidate_source: {source}")
        if not np.isfinite(self.ct_rbpf_motion_sigma_m):
            raise ValueError("ct_rbpf_motion_sigma_m must be finite")
        for name in (
            "dd_carrier_tow_snap_tolerance_s",
            "dd_carrier_sigma_cycles",
            "dd_carrier_prior_sigma_m",
            "dd_carrier_max_shift_m",
            "dd_carrier_max_initial_rms_m",
            "dd_carrier_max_final_rms_m",
            "dd_carrier_anchor_correction_sigma_m",
            "dd_carrier_correction_smooth_sigma_m",
            "dd_carrier_correction_zero_sigma_m",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if int(self.dd_carrier_min_dd_pairs) < 2:
            raise ValueError("dd_carrier_min_dd_pairs must be >= 2")
        if not np.isfinite(float(self.dd_carrier_min_anchor_coverage)):
            raise ValueError("dd_carrier_min_anchor_coverage must be finite")
        if not 0.0 <= float(self.dd_carrier_min_anchor_coverage) <= 1.0:
            raise ValueError("dd_carrier_min_anchor_coverage must be in [0, 1]")
        if not np.isfinite(self.imu_accel_bias_prior_sigma_mps2):
            raise ValueError("imu_accel_bias_prior_sigma_mps2 must be finite")
        if not np.isfinite(self.imu_accel_bias_between_sigma_mps2):
            raise ValueError("imu_accel_bias_between_sigma_mps2 must be finite")
        for name in (
            "stop_attitude_sigma_rad",
            "imu_acc_sigma_mps2_sqrt_hz",
            "imu_gyro_sigma_radps_sqrt_hz",
            "imu_acc_sync_coefficient",
            "imu_gyro_sync_coefficient",
            "imu_attitude_sigma_rad",
            "imu_gyro_bias_prior_sigma_radps",
            "imu_gyro_bias_between_sigma_radps",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")


def apply_taroz_fgo_preset(config: BridgeConfig) -> BridgeConfig:
    """Return a config with taroz ``parameters.m`` FGO levers enabled.

    The preset keeps gate/WLS weighting unchanged and switches only the FGO
    objective to taroz SNR weights.  It also enables per-Type PR Huber and
    motion sigma lookup, plus the clock/stop/height sigmas used by the
    MATLAB graph.
    """
    return replace(
        config,
        dual_frequency=True,
        fgo_weight_mode=TAROZ_FGO_WEIGHT_MODE,
        fgo_robust_kernel="huber",
        clock_drift_sigma_m=TAROZ_CLOCK_DRIFT_SIGMA_M,
        clock_use_average_drift=True,
        stop_velocity_sigma_mps=TAROZ_STOP_VELOCITY_SIGMA_MPS,
        stop_position_sigma_m=TAROZ_STOP_POSITION_SIGMA_M,
        stop_attitude_sigma_rad=TAROZ_STOP_ATTITUDE_SIGMA_RAD,
        stop_velocity_huber_k=TAROZ_STOP_HUBER_K,
        stop_position_huber_k=TAROZ_STOP_HUBER_K,
        per_type_kernel_enabled=True,
        per_type_kernel_huber_enabled=True,
        per_type_kernel_motion_enabled=True,
        fgo_fixed_linearization=True,
        apply_base_correction=True,
        tdcp_weight_scale=1.0,
        graph_relative_height=True,
        relative_height_sigma_m=TAROZ_HEIGHT_SIGMA_M,
        relative_height_huber_k=TAROZ_HEIGHT_HUBER_K,
        apply_absolute_height=True,
        absolute_height_sigma_m=TAROZ_HEIGHT_SIGMA_M,
        absolute_height_huber_k=TAROZ_HEIGHT_HUBER_K,
        apply_observation_mask=True,
        observation_min_cn0_dbhz=OBS_MASK_MIN_CN0_DBHZ,
        observation_min_elevation_deg=TAROZ_DUAL_FREQUENCY_ELEVATION_DEG,
        pseudorange_residual_mask_m=OBS_MASK_RESIDUAL_THRESHOLD_M,
        pseudorange_residual_mask_l5_m=OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
        doppler_residual_mask_mps=OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
        pseudorange_doppler_mask_m=OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    )


def apply_taroz_gnss_only_preset(config: BridgeConfig) -> BridgeConfig:
    """Return a config for the taroz GNSS-only ``fgo_gnss.m`` graph.

    This keeps the FGO objective on taroz SNR weights and enables the Type
    lookup for PR/Doppler/TDCP Huber and MotionFactor sigma.  Stop, height and
    IMU priors are explicitly disabled because those belong to the GNSS+IMU
    graph, not ``fgo_gnss.m``.
    """
    return replace(
        config,
        dual_frequency=True,
        fgo_weight_mode=TAROZ_FGO_WEIGHT_MODE,
        fgo_robust_kernel="huber",
        clock_drift_sigma_m=TAROZ_CLOCK_DRIFT_SIGMA_M,
        clock_use_average_drift=True,
        stop_velocity_sigma_mps=0.0,
        stop_position_sigma_m=0.0,
        stop_attitude_sigma_rad=0.0,
        stop_velocity_huber_k=0.0,
        stop_position_huber_k=0.0,
        apply_imu_prior=False,
        imu_attitude_state=False,
        imu_diagonal_covariance=False,
        imu_preintegration_covariance=False,
        imu_factor_use_next_bias=False,
        imu_bias_between_sample_count_scaling=False,
        imu_accel_bias_state=False,
        imu_gyro_bias_state=False,
        taroz_imu_noise_enabled=False,
        per_type_kernel_enabled=True,
        per_type_kernel_huber_enabled=True,
        per_type_kernel_motion_enabled=True,
        fgo_fixed_linearization=True,
        apply_base_correction=True,
        # fgo_gnss.m always runs add_position_offset on the GNSS-only output
        apply_position_offset=True,
        tdcp_weight_scale=1.0,
        graph_relative_height=False,
        relative_height_huber_k=0.0,
        apply_absolute_height=False,
        absolute_height_huber_k=0.0,
        apply_relative_height=False,
        apply_observation_mask=True,
        observation_min_cn0_dbhz=OBS_MASK_MIN_CN0_DBHZ,
        observation_min_elevation_deg=TAROZ_DUAL_FREQUENCY_ELEVATION_DEG,
        pseudorange_residual_mask_m=OBS_MASK_RESIDUAL_THRESHOLD_M,
        pseudorange_residual_mask_l5_m=OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
        doppler_residual_mask_mps=OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
        pseudorange_doppler_mask_m=OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    )


def apply_taroz_marupaku_preset(config: BridgeConfig) -> BridgeConfig:
    """Return the closest available taroz full-FGO config for parity runs.

    ``apply_taroz_fgo_preset`` ports the parameters.m FGO levers but preserves
    the caller's final source-selection policy.  A Taroz parity run should use
    the graph output directly, use Taroz SNR weights for both WLS seeding and
    FGO, and include the available IMU delta-prior path, so this preset also
    forces ``position_source`` to ``fgo``.
    """
    imu_noise = taroz_imu_noise_for_phone("")
    return replace(
        apply_taroz_fgo_preset(config),
        position_source="fgo",
        weight_mode=TAROZ_FGO_WEIGHT_MODE,
        fgo_weight_mode=TAROZ_FGO_WEIGHT_MODE,
        apply_imu_prior=True,
        imu_frame=TAROZ_IMU_DELTA_FRAME,
        imu_sample_dt_mode=TAROZ_IMU_SAMPLE_DT_MODE,
        imu_position_sigma_m=TAROZ_IMU_INTEGRATION_SIGMA,
        imu_velocity_sigma_mps=imu_noise.effective_acc_sigma_mps2_sqrt_hz,
        imu_attitude_state=True,
        imu_attitude_sigma_rad=imu_noise.effective_gyro_sigma_radps_sqrt_hz,
        # Taroz-exported GobsPhone obs.elapsedns is NaN for the parity data used
        # by fgo_gnss_imu, so imuprocessing() leaves the preintegration noise
        # covariance at the raw parameters.m AccSigma/GyroSigma scale even
        # though the nominal sync coefficient remains part of the metadata.
        imu_preintegration_velocity_noise_mps_sqrt_hz=imu_noise.acc_sigma_mps2_sqrt_hz,
        imu_preintegration_attitude_noise_rad_sqrt_hz=imu_noise.gyro_sigma_radps_sqrt_hz,
        imu_diagonal_covariance=True,
        imu_preintegration_covariance=True,
        imu_factor_use_next_bias=True,
        imu_bias_between_sample_count_scaling=True,
        imu_accel_bias_state=True,
        imu_accel_bias_prior_sigma_mps2=0.0,
        imu_accel_bias_between_sigma_mps2=TAROZ_IMU_ACCEL_BIAS_SIGMA_MPS2,
        imu_gyro_bias_state=True,
        imu_gyro_bias_prior_sigma_radps=0.0,
        taroz_imu_noise_enabled=True,
        taroz_stop_mask_from_seed_velocity=True,
        imu_acc_sigma_mps2_sqrt_hz=imu_noise.acc_sigma_mps2_sqrt_hz,
        imu_gyro_sigma_radps_sqrt_hz=imu_noise.gyro_sigma_radps_sqrt_hz,
        imu_acc_sync_coefficient=imu_noise.acc_sync_coefficient,
        imu_gyro_sync_coefficient=imu_noise.gyro_sync_coefficient,
        imu_gyro_bias_between_sigma_radps=imu_noise.gyro_bias_sigma_radps,
    )


def apply_taroz_full_init_pass_preset(config: BridgeConfig) -> BridgeConfig:
    """Adjust full-FGO settings to match ``fgo_gnss_imu(..., true)``.

    Taroz's first GNSS+IMU pass keeps motion, clock, IMU, TDCP and stop-pose
    factors, but the stop-velocity and height factors live under
    ``if ~initflag`` and are therefore absent.
    """
    return replace(
        config,
        stop_velocity_sigma_mps=0.0,
        stop_velocity_huber_k=0.0,
        graph_relative_height=False,
        relative_height_huber_k=0.0,
        apply_absolute_height=False,
        absolute_height_huber_k=0.0,
        apply_relative_height=False,
    )


def should_refine_outlier_result(position_source: str, chunk_epochs: int, selected_mse_pr: float) -> bool:
    if position_source not in {"auto", "gated"}:
        return False
    if chunk_epochs <= OUTLIER_REFINEMENT_CHUNK_EPOCHS:
        return False
    return float(selected_mse_pr) > OUTLIER_REFINEMENT_MSE_PR_THRESHOLD


__all__ = [
    "BridgeConfig",
    "DEFAULT_CT_RBPF_MOTION_SIGMA_M",
    "DEFAULT_MOTION_SIGMA_M",
    "FACTOR_DT_MAX_S",
    "OUTLIER_REFINEMENT_CHUNK_EPOCHS",
    "OUTLIER_REFINEMENT_MSE_PR_THRESHOLD",
    "TarozImuNoise",
    "apply_taroz_fgo_preset",
    "apply_taroz_full_init_pass_preset",
    "apply_taroz_gnss_only_preset",
    "apply_taroz_marupaku_preset",
    "should_refine_outlier_result",
    "taroz_imu_noise_for_phone",
]
