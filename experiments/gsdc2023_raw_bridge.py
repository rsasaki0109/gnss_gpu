from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace as _replace_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluate import ecef_to_lla, lla_to_ecef
from gnss_gpu import wls_position
from experiments.gsdc2023_fgo_cauchy_irls import fgo_gnss_lm_vd_cauchy
from experiments.gsdc2023_per_type_kernel import (
    per_type_kernel_for as _per_type_kernel_for,
    trip_type_from_data_root as _trip_type_from_data_root,
)
from gnss_gpu.fgo import fgo_gnss_lm, fgo_gnss_lm_vd
from gnss_gpu.io.nav_rinex import read_gps_klobuchar_from_nav_header
from gnss_gpu.multi_gnss import (
    SYSTEM_BEIDOU,
    MultiGNSSSolver,
    SYSTEM_GALILEO,
    SYSTEM_GPS,
    SYSTEM_QZSS,
)
from gnss_gpu.spp import _elevation_azimuth
from experiments.gsdc2023_chunk_selection import (
    CATASTROPHIC_BASELINE_GAP_MAX_M,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    GATED_CANDIDATE_QUALITY_MARGIN,
    GATED_FGO_BASELINE_GAP_P95_FLOOR_M,
    GATED_FGO_BASELINE_MSE_PR_MIN,
    GATED_MI8_BASELINE_JUMP_STEP_P95_M,
    GATED_MI8_RAW_WLS_BASELINE_GAP_MAX_M,
    GATED_RAW_WLS_RESCUE_BASELINE_MSE_PR_MIN,
    GATED_RAW_WLS_RESCUE_MSE_PR_MAX,
    GATED_RAW_WLS_RESCUE_MSE_PR_RATIO_MAX,
    GATED_TDCP_BASELINE_GAP_INCREASE_MARGIN_M,
    GATED_TDCP_OFF_CANDIDATE_MARGIN,
    WINDOW_SELECTION_STEP_P95_MAX_M,
    ChunkCandidateQuality,
    ChunkSelectionRecord,
    add_fgo_candidate_from_records as _add_fgo_candidate_from_records,
    add_tdcp_off_fgo_candidates as _add_tdcp_off_fgo_candidates,
    candidate_passes_gated_quality as _candidate_passes_gated_quality,
    catastrophic_baseline_alternative as _catastrophic_baseline_alternative,
    chunk_candidate_quality as _chunk_candidate_quality,
    chunk_quality_payload as _chunk_quality_payload,
    compute_dd_carrier_anchor_coverage_ratio as _compute_dd_carrier_anchor_coverage_ratio,
    fgo_candidate_passes_baseline_gap_guard as _fgo_candidate_passes_baseline_gap_guard,
    fgo_candidate_passes_raw_wls_mse_guard as _fgo_candidate_passes_raw_wls_mse_guard,
    is_fgo_candidate_source as _is_fgo_candidate_source,
    quality_ratio as _quality_ratio,
    raw_wls_candidate_passes_high_pr_mse_rescue as _raw_wls_candidate_passes_high_pr_mse_rescue,
    raw_wls_candidate_passes_mi8_baseline_jump_guard as _raw_wls_candidate_passes_mi8_baseline_jump_guard,
    select_auto_chunk_source as _select_auto_chunk_source,
    select_gated_chunk_source as _select_gated_chunk_source,
    trajectory_motion_stats as _trajectory_motion_stats,
)
from experiments.gsdc2023_bridge_config import (
    BridgeConfig,
    DEFAULT_CT_RBPF_MOTION_SIGMA_M,
    DEFAULT_MOTION_SIGMA_M,
    FACTOR_DT_MAX_S,
    OUTLIER_REFINEMENT_CHUNK_EPOCHS,
    OUTLIER_REFINEMENT_MSE_PR_THRESHOLD,
    TAROZ_FGO_WEIGHT_MODE,
    TAROZ_STOP_VELOCITY_THRESHOLD_MPS,
    should_refine_outlier_result as _should_refine_outlier_result,
    taroz_imu_noise_for_phone as _taroz_imu_noise_for_phone,
)
from experiments.gsdc2023_gnss_log_bridge import (
    GNSS_LOG_SYNTHETIC_PRODUCT_COLUMNS as _GNSS_LOG_SYNTHETIC_PRODUCT_COLUMNS,
    append_gnss_log_only_gps_rows as _append_gnss_log_only_gps_rows,
    gnss_log_corrected_pseudorange_products,
    gnss_log_matlab_epoch_times_ms as _gnss_log_matlab_epoch_times_ms,
    gnss_log_matlab_epoch_times_ms_cached as _gnss_log_matlab_epoch_times_ms_cached,
    gnss_log_signal_type as _gnss_log_signal_type,
    interpolated_raw_values as _interpolated_raw_values,
)
from experiments.gsdc2023_diagnostics_mask import (
    apply_matlab_residual_diagnostics_mask as _apply_matlab_residual_diagnostics_mask,
    diagnostics_bool as _diagnostics_bool,
)
from experiments.gsdc2023_base_correction import (
    base_metadata_dir as _base_metadata_dir,
    base_setting as _base_setting,
    compute_base_pseudorange_correction_matrix as _compute_base_pseudorange_correction_matrix_impl,
    course_base_obs_path as _course_base_obs_path,
    course_nav_path as _course_nav_path,
    filter_matrtklib_duplicate_gps_nav_messages as _filter_matrtklib_duplicate_gps_nav_messages,
    gps_abs_seconds_from_datetime as _gps_abs_seconds_from_datetime,
    gps_arrival_tow_s_from_row as _gps_arrival_tow_s_from_row,
    gps_matrtklib_nav_messages_cached as _gps_matrtklib_nav_messages_cached,
    gps_matrtklib_nav_messages_for_trip as _gps_matrtklib_nav_messages_for_trip,
    gps_matrtklib_sat_product_adjustment as _gps_matrtklib_sat_product_adjustment,
    gps_sat_clock_bias_adjustment_m as _gps_sat_clock_bias_adjustment_m,
    gps_tgd_m_by_svid_cached as _gps_tgd_m_by_svid_cached,
    gps_tgd_m_by_svid_for_trip as _gps_tgd_m_by_svid_for_trip,
    load_base_residual_series_cached as _load_base_residual_series_cached,
    load_settings_frame_cached as _load_settings_frame_cached,
    matlab_base_time_span_mask as _matlab_base_time_span_mask,
    moving_nanmean as _moving_nanmean,
    read_base_station_xyz as _read_base_station_xyz,
    round_seconds_to_interval_like_matlab as _round_seconds_to_interval_like_matlab,
    rtklib_tropo_saastamoinen as _rtklib_tropo_saastamoinen,
    select_base_pseudorange_observation as _select_base_pseudorange_observation,
    select_gps_nav_message as _select_gps_nav_message,
    signal_type_base_obs_codes as _signal_type_base_obs_codes,
    signal_type_iono_scale as _signal_type_iono_scale,
    slot_sat_id as _slot_sat_id,
    trip_course_phone as _trip_course_phone,
    trip_full_phone_time_span_gps_abs_cached as _trip_full_phone_time_span_gps_abs_cached,
    trip_nav_path as _trip_nav_path,
    trip_phone_time_span_for_base_trim as _trip_phone_time_span_for_base_trim,
    unix_ms_to_gps_abs_seconds as _unix_ms_to_gps_abs_seconds,
)
from experiments.gsdc2023_height_constraints import (
    HEIGHT_ABSOLUTE_DIST_M,
    HEIGHT_ABSOLUTE_SIGMA_M,
    HEIGHT_LOOP_CUMDIST_M,
    HEIGHT_LOOP_DIST_M,
    apply_phone_position_offset,
    apply_phone_position_offset_state,
    apply_relative_height_constraint,
    as_1d_float as _as_1d_float,
    as_n_by_3 as _as_n_by_3,
    build_relative_height_groups as _build_relative_height_groups,
    ecef_to_enu_relative as _ecef_to_enu_relative,
    enu_to_ecef_relative as _enu_to_ecef_relative,
    enu_up_ecef_from_origin,
    llh_to_ecef_array as _llh_to_ecef_array,
    load_absolute_height_reference_ecef,
    load_ref_height_mat as _load_ref_height_mat,
    mat_get_field as _mat_get_field,
    numeric_array_from_mat as _numeric_array_from_mat,
    phone_position_offset as _phone_position_offset,
    relative_height_star_edges_for_reference,
    relative_height_star_edges_from_groups,
)
from experiments.gsdc2023_imu import (
    ACC_TIME_OFFSET_MS,
    DEVICE_IMU_COLUMNS,
    GYRO_TIME_OFFSET_MS,
    IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2,
    IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2,
    IMU_DELTA_FRAMES,
    IMU_GRAVITY_MPS2,
    IMU_MOUNTING_ANGLE_RAD,
    IMU_SYNC_COEFFICIENT,
    IMU_SYNC_MODE,
    STOP_ACC_STD_OFFSET,
    STOP_GYRO_MAX,
    STOP_GYRO_STD_OFFSET,
    STOP_WINDOW_SIZE,
    IMUMeasurements,
    IMUPreintegration,
    ProcessedIMU,
    ecef_delta_from_enu_delta as _ecef_delta_from_enu_delta,
    estimate_rpy_from_velocity,
    eul_xyz_to_rotm as _eul_xyz_to_rotm,
    extract_imu_measurements as _extract_imu_measurements,
    fill_nearest as _fill_nearest,
    gtsam_rzryrx_to_rotm as _gtsam_rzryrx_to_rotm,
    imu_preintegration_gravity_segment as _imu_preintegration_gravity_segment,
    imu_preintegration_segment as _imu_preintegration_segment,
    imu_preintegration_segment_with_angle as _imu_preintegration_segment_with_angle,
    imu_preintegration_segment_with_angle_and_dt as _imu_preintegration_segment_with_angle_and_dt,
    imu_preintegration_segment_with_bias_jacobians as _imu_preintegration_segment_with_bias_jacobians,
    interp_vectors as _interp_vectors,
    load_taroz_imu_preintegration_csv as _load_taroz_imu_preintegration_csv,
    load_device_imu_measurements as _load_device_imu_measurements_impl,
    preintegrate_processed_imu,
    process_device_imu,
    project_stop_to_epochs,
    read_device_imu_frame as _read_device_imu_frame_impl,
    rotm_to_rotvec as _rotm_to_rotvec,
    rolling_std as _rolling_std,
    wrap_to_180_deg as _wrap_to_180_deg,
)
from experiments.gsdc2023_residual_model import (
    estimate_residual_clock_series as _estimate_residual_clock_series,
    fill_clock_design as _fill_clock_design,
    geometric_range_rate_with_sagnac as _geometric_range_rate_with_sagnac,
    geometric_range_with_sagnac as _geometric_range_with_sagnac,
    gradient_with_matlab_interval as _gradient_with_matlab_interval,
    mask_doppler_residual_outliers as _mask_doppler_residual_outliers,
    mask_pseudorange_doppler_consistency as _mask_pseudorange_doppler_consistency,
    mask_pseudorange_residual_outliers as _mask_pseudorange_residual_outliers,
    matlab_epoch_interval_s as _matlab_epoch_interval_s,
    median_clock_prediction as _median_clock_prediction,
    min_pseudorange_keep_count as _min_pseudorange_keep_count,
    pseudorange_global_isb_by_group as _pseudorange_global_isb_by_group,
    receiver_velocity_from_reference as _receiver_velocity_from_reference,
    sagnac_correction_m as _sagnac_correction_m,
    solve_clock_biases as _solve_clock_biases,
    weighted_median as _weighted_median,
)
from experiments.gsdc2023_signal_model import (
    MATLAB_SIGNAL_CLOCK_DIM,
    clock_kind_for_observation as _clock_kind_for_observation,
    constellation_to_matlab_sys as _constellation_to_matlab_sys,
    is_l5_signal as _is_l5_signal,
    multi_gnss_mask as _multi_gnss_mask,
    multi_system_for_clock_kind as _multi_system_for_clock_kind,
    remap_pseudorange_isb_by_group as _remap_pseudorange_isb_by_group,
    signal_sort_rank as _signal_sort_rank,
    signal_types_for_constellation as _signal_types_for_constellation,
    slot_frequency_label as _slot_frequency_label,
    slot_frequency_thresholds as _slot_frequency_thresholds,
    slot_pseudorange_common_bias_group_keys as _slot_pseudorange_common_bias_group_keys,
    slot_pseudorange_common_bias_groups as _slot_pseudorange_common_bias_groups,
    slot_sort_key as _slot_sort_key,
    taroz_clock_kind_for_observation as _taroz_clock_kind_for_observation,
)
from experiments.gsdc2023_result_assembly import (
    AssembledSourceOutputs,
    assemble_source_outputs as _assemble_source_outputs,
    build_bridge_result as _build_bridge_result,
)
from experiments.gsdc2023_dd_carrier_bridge import (
    DD_CARRIER_FGO_SOURCE,
    DDCarrierAnchorConfig,
    DDCarrierBridgeConfig,
    apply_sparse_dd_carrier_anchors as _apply_sparse_dd_carrier_anchors,
)
from experiments.gsdc2023_result_metadata import (
    ImuResultSummary,
    imu_result_summary as _imu_result_summary,
    mean_finite_row_norm as _mean_finite_row_norm,
)
from experiments.gsdc2023_validation_context import (
    RawTripValidationContext,
    build_raw_trip_validation_context as _build_raw_trip_validation_context,
    max_epochs_for_build as _max_epochs_for_build,
    outlier_refinement_config as _outlier_refinement_config,
)
from experiments.gsdc2023_solver_selection import (
    batch_without_tdcp as _batch_without_tdcp,
    build_source_solution_catalog as _build_source_solution_catalog,
    fgo_raw_wls_proxy_rescue_enabled as _fgo_raw_wls_proxy_rescue_enabled,
    mi8_gated_baseline_jump_guard_enabled as _mi8_gated_baseline_jump_guard_enabled,
    raw_wls_max_gap_guard_m as _raw_wls_max_gap_guard_m,
    select_gated_solution as _select_gated_solution,
    taroz_fgo_candidate_sources_enabled as _taroz_fgo_candidate_sources_enabled,
    tdcp_off_candidate_enabled as _tdcp_off_candidate_enabled,
    tdcp_scale_candidate_enabled as _tdcp_scale_candidate_enabled,
    with_fixed_source_solution as _with_fixed_source_solution,
    with_source_solution as _with_source_solution,
)
from experiments.gsdc2023_solver_options import (
    FgoRunOptions,
    fgo_run_options_from_config as _fgo_run_options_from_config,
)
from experiments.gsdc2023_solver_context import (
    SolverExecutionContext,
    build_solver_execution_context as _build_solver_execution_context,
    estimate_speed_mps,
    solver_stop_mask,
)
from experiments.gsdc2023_tdcp import (
    ADR_STATE_CYCLE_SLIP,
    ADR_STATE_RESET,
    ADR_STATE_VALID,
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_SIGMA_M,
    DEFAULT_TDCP_WEIGHT_SCALE,
    TDCP_DISABLE_PHONES,
    TDCP_LOFFSET_M,
    TDCP_LOFFSET_PHONES,
    TDCP_XXDD_PHONES,
    apply_tdcp_geometry_correction as _apply_tdcp_geometry_correction,
    apply_tdcp_weight_scale as _apply_tdcp_weight_scale,
    build_tdcp_arrays as _build_tdcp_arrays,
    tdcp_enabled_for_phone as _tdcp_enabled_for_phone,
    tdcp_loffset_m as _tdcp_loffset_m,
    tdcp_use_drift_for_phone as _tdcp_use_drift_for_phone,
    valid_adr_state as _valid_adr_state,
)
from experiments.gsdc2023_trip_stages import (
    AbsoluteHeightStageProducts,
    ClockResidualStageProducts,
    DopplerResidualStageProducts,
    EpochMetadataContext,
    EpochTimeContext,
    FilledObservationMatrixProducts,
    FullObservationContextProducts,
    GnssLogPseudorangeStageProducts,
    GraphTimeDeltaProducts,
    ImuStageProducts,
    FilledObservationPostprocessProducts,
    ObservationMaskBaseCorrectionStageProducts,
    ObservationMatrixInputProducts,
    ObservationPreparationStageProducts,
    PostObservationStageProducts,
    PostObservationStageConfig,
    PostObservationStageDependencies,
    PreparedObservationProducts,
    PseudorangeDopplerStageProducts,
    PseudorangeResidualStageProducts,
    RawObservationFrameProducts,
    TdcpStageProducts,
    apply_base_correction_to_pseudorange as _apply_base_correction_to_pseudorange,
    apply_gnss_log_pseudorange_stage as _apply_gnss_log_pseudorange_stage,
    assemble_prepared_trip_arrays_stage as _assemble_prepared_trip_arrays_stage,
    assemble_trip_arrays_stage as _assemble_trip_arrays_stage,
    build_absolute_height_stage as _build_absolute_height_stage,
    build_clock_residual_stage as _build_clock_residual_stage,
    build_configured_post_observation_stages as _build_configured_post_observation_stages,
    build_doppler_residual_stage as _build_doppler_residual_stage,
    build_epoch_metadata_context as _build_epoch_metadata_context,
    build_epoch_time_context as _build_epoch_time_context,
    build_filled_observation_matrix_stage as _build_filled_observation_matrix_stage,
    build_full_observation_context_stage as _build_full_observation_context_stage,
    build_graph_time_delta_products as _build_graph_time_delta_products,
    build_imu_stage as _build_imu_stage,
    build_observation_mask_base_correction_stage as _build_observation_mask_base_correction_stage,
    build_observation_matrix_input_stage as _build_observation_matrix_input_stage,
    build_observation_preparation_stages as _build_observation_preparation_stages,
    build_post_observation_stages as _build_post_observation_stages,
    unpack_observation_preparation_stage as _unpack_observation_preparation_stage,
    build_pseudorange_doppler_consistency_stage as _build_pseudorange_doppler_consistency_stage,
    build_pseudorange_residual_stage as _build_pseudorange_residual_stage,
    build_raw_observation_frame as _build_raw_observation_frame,
    build_tdcp_stage as _build_tdcp_stage,
    postprocess_filled_observation_stage as _postprocess_filled_observation_stage,
)
from experiments.gsdc2023_observation_matrix import (
    ANDROID_STATE_CODE_LOCK,
    ANDROID_STATE_TOD_OK,
    ANDROID_STATE_TOW_OK,
    BASELINE_BIAS_UNCERTAINTY_NANOS_MAX,
    BASELINE_OUTLIER_FLOOR_M,
    BASELINE_OUTLIER_THRESHOLD_FACTOR,
    BASELINE_OUTLIER_WINDOW,
    CONSTELLATION_GLONASS,
    EARTH_ROTATION_RATE_RAD_S,
    GPS_EPOCH_UNIX_SECONDS,
    GPS_WEEK_SECONDS,
    LIGHT_SPEED_MPS,
    OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    OBS_MASK_MIN_CN0_DBHZ,
    OBS_MASK_MIN_ELEVATION_DEG,
    OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_L5_M,
    OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    OBS_MASK_PSEUDORANGE_MAX_M,
    OBS_MASK_PSEUDORANGE_MIN_M,
    OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    OBS_MASK_RESIDUAL_THRESHOLD_M,
    RAW_GNSS_COLUMNS,
    RAW_GNSS_OPTIONAL_COLUMNS,
    RAW_GNSS_REQUIRED_COLUMNS,
    TripArrays,
    android_state_tracking_ok as _android_state_tracking_ok,
    apply_matlab_signal_observation_mask as _apply_matlab_signal_observation_mask,
    build_epoch_metadata_frame as _build_epoch_metadata_frame,
    clock_jump_from_epoch_counts as _clock_jump_from_epoch_counts,
    fill_observation_matrices as _fill_observation_matrices,
    legacy_matlab_signal_observation_mask as _legacy_matlab_signal_observation_mask,
    load_raw_gnss_frame as _load_raw_gnss_frame,
    load_raw_gnss_frame_epoch_window as _load_raw_gnss_frame_epoch_window,
    matlab_signal_observation_masks as _matlab_signal_observation_masks,
    read_raw_gnss_csv as _read_raw_gnss_csv,
    receiver_clock_bias_from_nanos as _receiver_clock_bias_from_nanos,
    receiver_clock_bias_lookup_from_epoch_meta as _receiver_clock_bias_lookup_from_epoch_meta,
    recompute_rtklib_iono_matrix as _recompute_rtklib_iono_matrix,
    recompute_rtklib_tropo_matrix as _recompute_rtklib_tropo_matrix,
    repair_baseline_wls as _repair_baseline_wls,
    select_epoch_observations as _select_epoch_observations,
)
from experiments.gsdc2023_clock_state import (
    CLOCK_DRIFT_BLOCKLIST_PHONES,
    clean_clock_drift as _clean_clock_drift,
    clock_aid_enabled as _clock_aid_enabled,
    clock_drift_seed_enabled as _clock_drift_seed_enabled,
    clock_jump_threshold_m as _clock_jump_threshold_m,
    combine_clock_jump_masks as _combine_clock_jump_masks,
    detect_clock_jumps_from_clock_bias as _detect_clock_jumps_from_clock_bias,
    effective_multi_gnss_enabled as _effective_multi_gnss_enabled,
    effective_position_source as _effective_position_source,
    factor_break_mask as _factor_break_mask,
    segment_ranges as _segment_ranges,
)
from experiments.gsdc2023_output import (
    BridgeResult,
    CT_RBPF_FGO_SOURCE,
    POSITION_SOURCES,
    TAROZ_PR_D_L_FGO_SOURCE,
    TAROZ_PR_FGO_SOURCE,
    TAROZ_WEIGHTS_FGO_SOURCE,
    TDCP_SCALE_FGO_SOURCE,
    bridge_position_columns,
    ecef_to_llh_deg,
    export_bridge_outputs,
    format_metrics_line,
    has_valid_bridge_outputs,
    load_bridge_metrics,
    metrics_summary,
    score_from_metrics,
    validate_position_source,
)


def resolve_gsdc2023_data_root() -> Path:
    """Data root for GSDC2023 experiments.

    Resolution order:

    1. ``GSDC2023_DATA_ROOT`` if set and the path exists.
    2. ``ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023`` if present (full Kaggle unzip).
    3. ``ref/gsdc2023/dataset_2023`` (legacy / partial tree).
    """

    repo = Path(__file__).resolve().parents[2]
    env = os.environ.get("GSDC2023_DATA_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    kaggle = repo / "ref" / "gsdc2023" / "kaggle_smartphone_decimeter_2023" / "sdc2023"
    if kaggle.is_dir() and (kaggle / "train").is_dir():
        return kaggle
    return repo / "ref" / "gsdc2023" / "dataset_2023"


DEFAULT_ROOT = resolve_gsdc2023_data_root()
VD_SEED_FACTOR_GUARD_MIN_COUNT = 20
VD_SEED_FACTOR_GUARD_DOPPLER_RMS_MPS = 8.0
VD_SEED_FACTOR_GUARD_TDCP_RMS_M = 50.0

GPS_L1_FREQUENCY_HZ = 1575.42e6
GPS_L5_FREQUENCY_HZ = 1176.45e6
GPS_L5_TGD_SCALE = (GPS_L1_FREQUENCY_HZ / GPS_L5_FREQUENCY_HZ) ** 2
GPS_LEAP_SECONDS = 18.0
BASE_MOVMEAN_N_1S = 151
BASE_MOVMEAN_N_15S = 11
BASE_OBS_TRIM_MARGIN_S = 180.0


def nearest_index(sorted_times: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(sorted_times, t))
    if idx <= 0:
        return 0
    if idx >= len(sorted_times):
        return len(sorted_times) - 1
    prev_idx = idx - 1
    return idx if abs(sorted_times[idx] - t) < abs(sorted_times[prev_idx] - t) else prev_idx


def load_ground_truth_ecef(trip_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    gt_path = trip_dir / "ground_truth.csv"
    if not gt_path.is_file():
        return None, None
    gt_df = pd.read_csv(gt_path)
    gt_times = gt_df["UnixTimeMillis"].to_numpy(dtype=np.float64)
    gt_ecef = np.array(
        [
            lla_to_ecef(np.deg2rad(lat), np.deg2rad(lon), alt)
            for lat, lon, alt in gt_df[["LatitudeDegrees", "LongitudeDegrees", "AltitudeMeters"]].to_numpy(
                dtype=np.float64,
            )
        ],
        dtype=np.float64,
    )
    return gt_times, gt_ecef


def compute_base_pseudorange_correction_matrix(
    data_root: Path,
    trip: str,
    times_ms: np.ndarray,
    slot_keys: list[tuple[int, int] | tuple[int, int, str]],
    signal_type: str,
) -> np.ndarray:
    return _compute_base_pseudorange_correction_matrix_impl(
        data_root,
        trip,
        times_ms,
        slot_keys,
        signal_type,
        base_setting_fn=_base_setting,
        base_residual_loader=_load_base_residual_series_cached,
        phone_span_fn=_trip_phone_time_span_for_base_trim,
    )


def collect_matlab_parity_audit(data_root: Path, trip: str, *, include_imu_sync: bool = True) -> dict:
    data_root = Path(data_root)
    split, course, phone = _trip_course_phone(trip)
    audit_root = data_root
    if split is None and data_root.name in {"train", "test"} and course is not None and phone is not None:
        split = data_root.name
        audit_root = data_root.parent
        trip_dir = data_root / course / phone
    else:
        trip_dir = data_root / trip
    course_dir = audit_root / split / course if split is not None and course is not None else trip_dir.parent
    base_dir = _base_metadata_dir(audit_root)
    settings = _load_settings_frame_cached(str(audit_root), split) if split is not None else None
    settings_csv_present = settings is not None
    setting_row = None
    if settings is not None and course is not None and phone is not None:
        rows = settings[(settings["Course"].astype(str) == course) & (settings["Phone"].astype(str) == phone)]
        if not rows.empty:
            setting_row = rows.iloc[0]

    base_name = None
    rinex_type = None
    if setting_row is not None:
        base_raw = setting_row.get("Base1", np.nan)
        rinex_raw = setting_row.get("RINEX", np.nan)
        if pd.notna(base_raw):
            base_name = str(base_raw).strip() or None
        if pd.notna(rinex_raw):
            rinex_type = str(rinex_raw).strip() or None

    expected_base_obs = None
    if base_name is not None:
        suffix = "rnx3" if rinex_type == "V3" else "rnx2" if rinex_type == "V2" else None
        if suffix is not None and course is not None and split is not None:
            expected_base_obs = audit_root / split / course / f"{base_name}_{suffix}.obs"

    base_position_csv = base_dir / "base_position.csv"
    base_offset_csv = base_dir / "base_offset.csv"
    nav_present = any(course_dir.glob("brdc.*")) if course_dir.is_dir() else False
    has_device_imu = (trip_dir / "device_imu.csv").is_file()
    has_ground_truth = (trip_dir / "ground_truth.csv").is_file()
    has_ref_height = (course_dir / "ref_hight.mat").is_file() if course_dir.is_dir() else False

    acc = gyro = mag = None
    imu_rows_acc = imu_rows_gyro = imu_rows_mag = 0
    imu_sync_ready = False
    stop_epoch_count = 0
    gnss_elapsed_present = False
    if include_imu_sync:
        try:
            acc, gyro, mag = load_device_imu_measurements(trip_dir)
            imu_rows_acc = int(acc.times_ms.size) if acc is not None else 0
            imu_rows_gyro = int(gyro.times_ms.size) if gyro is not None else 0
            imu_rows_mag = int(mag.times_ms.size) if mag is not None else 0
        except Exception:  # noqa: BLE001
            acc = gyro = mag = None
        raw_path = trip_dir / "device_gnss.csv"
        if raw_path.is_file():
            try:
                raw_df = _load_raw_gnss_frame(raw_path)
                epoch_meta = _build_epoch_metadata_frame(raw_df)
                if "ChipsetElapsedRealtimeNanos" in epoch_meta.columns:
                    gnss_elapsed = epoch_meta["ChipsetElapsedRealtimeNanos"].to_numpy(dtype=np.float64)
                    gnss_times = epoch_meta["utcTimeMillis"].to_numpy(dtype=np.float64)
                    gnss_elapsed_present = np.isfinite(gnss_elapsed).any()
                    if acc is not None and gyro is not None:
                        acc_proc, _, idx_stop = process_device_imu(acc, gyro, gnss_times, gnss_elapsed)
                        imu_sync_ready = True
                        stop_epoch_count = int(project_stop_to_epochs(acc_proc.times_ms, idx_stop, gnss_times).sum())
            except Exception:  # noqa: BLE001
                pass

    if not settings_csv_present:
        status = "settings_csv_missing"
    elif setting_row is None:
        status = "setting_row_missing"
    elif base_name is None:
        status = "base1_missing"
    elif not base_position_csv.is_file() or not base_offset_csv.is_file():
        status = "base_metadata_missing"
    elif expected_base_obs is None or not expected_base_obs.is_file():
        status = "base_obs_missing"
    elif not nav_present:
        status = "broadcast_nav_missing"
    else:
        status = "base_correction_ready"

    return {
        "dataset_split": split,
        "course": course,
        "phone": phone,
        "settings_csv_present": settings_csv_present,
        "setting_row_present": setting_row is not None,
        "base_name": base_name,
        "rinex_type": rinex_type,
        "base_dir_present": base_dir.is_dir(),
        "base_position_csv_present": base_position_csv.is_file(),
        "base_offset_csv_present": base_offset_csv.is_file(),
        "expected_base_obs": str(expected_base_obs) if expected_base_obs is not None else None,
        "base_obs_file_present": bool(expected_base_obs is not None and expected_base_obs.is_file()),
        "broadcast_nav_present": bool(nav_present),
        "device_imu_present": bool(has_device_imu),
        "ground_truth_present": bool(has_ground_truth),
        "ref_height_present": bool(has_ref_height),
        "imu_sync_checked": bool(include_imu_sync),
        "gnss_elapsed_present": bool(gnss_elapsed_present),
        "imu_rows_acc": imu_rows_acc,
        "imu_rows_gyro": imu_rows_gyro,
        "imu_rows_mag": imu_rows_mag,
        "imu_sync_ready": bool(imu_sync_ready),
        "stop_epoch_count": int(stop_epoch_count),
        "base_correction_status": status,
        "base_correction_ready": status == "base_correction_ready",
    }


def _read_device_imu_frame(path: Path) -> pd.DataFrame:
    return _read_device_imu_frame_impl(path, read_csv_fn=_read_raw_gnss_csv)


def load_device_imu_measurements(trip_dir: Path) -> tuple[IMUMeasurements | None, IMUMeasurements | None, IMUMeasurements | None]:
    return _load_device_imu_measurements_impl(trip_dir, read_csv_fn=_read_raw_gnss_csv)


def _gps_iono_alpha_beta_for_trip(trip_dir: Path) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    nav_path = _trip_nav_path(trip_dir)
    if nav_path is None:
        return None
    alpha, beta = read_gps_klobuchar_from_nav_header(nav_path)
    if alpha is None or beta is None:
        alpha = [0.1118e-07, -0.7451e-08, -0.5961e-07, 0.1192e-06]
        beta = [0.1167e06, -0.2294e06, -0.1311e06, 0.1049e07]
    return tuple(float(value) for value in alpha), tuple(float(value) for value in beta)


def _gnss_log_corrected_pseudorange_matrix(
    trip_dir: Path,
    raw_frame: pd.DataFrame,
    times_ms: np.ndarray,
    slot_keys: tuple[tuple[int, int, str], ...],
    gps_tgd_m_by_svid: dict[int, float],
    rtklib_tropo_m: np.ndarray | None = None,
    rtklib_iono_m: np.ndarray | None = None,
    sat_clock_bias_m: np.ndarray | None = None,
    *,
    phone_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    products = gnss_log_corrected_pseudorange_products(
        trip_dir,
        raw_frame,
        times_ms,
        slot_keys,
        gps_tgd_m_by_svid,
        phone_name=phone_name,
        rtklib_tropo_m=rtklib_tropo_m,
        rtklib_iono_m=rtklib_iono_m,
        sat_clock_bias_m=sat_clock_bias_m,
        sat_clock_adjustment_m=_gps_sat_clock_bias_adjustment_m,
    )
    if products is None:
        return None
    return products.pseudorange, products.weights, products.observable_pseudorange


def build_trip_arrays(
    trip_dir: Path,
    *,
    max_epochs: int,
    start_epoch: int,
    constellation_type: int,
    signal_type: str,
    weight_mode: str,
    fgo_weight_mode: str | None = None,
    multi_gnss: bool = False,
    use_tdcp: bool = False,
    tdcp_consistency_threshold_m: float = DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    tdcp_weight_scale: float = DEFAULT_TDCP_WEIGHT_SCALE,
    tdcp_geometry_correction: bool = DEFAULT_TDCP_GEOMETRY_CORRECTION,
    apply_base_correction: bool = False,
    data_root: Path | None = None,
    trip: str | None = None,
    apply_observation_mask: bool = False,
    observation_min_cn0_dbhz: float = OBS_MASK_MIN_CN0_DBHZ,
    observation_min_elevation_deg: float = OBS_MASK_MIN_ELEVATION_DEG,
    pseudorange_residual_mask_m: float = OBS_MASK_RESIDUAL_THRESHOLD_M,
    pseudorange_residual_mask_l5_m: float | None = None,
    doppler_residual_mask_mps: float = OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    pseudorange_doppler_mask_m: float = OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    matlab_residual_diagnostics_mask_path: Path | None = None,
    dual_frequency: bool = False,
    apply_absolute_height: bool = False,
    absolute_height_dist_m: float = HEIGHT_ABSOLUTE_DIST_M,
    imu_frame: str = "body",
    imu_sample_dt_mode: str = "bounded",
    factor_dt_max_s: float = FACTOR_DT_MAX_S,
    raw_frame_epoch_window: bool = False,
    use_rtklib_tropo: bool = False,
) -> TripArrays:
    raw_path = trip_dir / "device_gnss.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(f"device_gnss.csv not found: {raw_path}")

    gt_times, gt_ecef = load_ground_truth_ecef(trip_dir)
    phone_name = trip_dir.name
    phone_name_l = phone_name.lower()
    tdcp_enabled = _tdcp_enabled_for_phone(phone_name, use_tdcp)
    tdcp_loffset_m = _tdcp_loffset_m(phone_name) if tdcp_enabled else 0.0
    adr_sign = -1.0 if phone_name_l in TDCP_LOFFSET_PHONES else 1.0
    if raw_frame_epoch_window:
        raw_df = _load_raw_gnss_frame_epoch_window(
            raw_path,
            start_epoch=start_epoch,
            max_epochs=max_epochs,
            extra_epochs=8,
        )
    else:
        raw_df = _load_raw_gnss_frame(raw_path)
    epoch_meta = _build_epoch_metadata_frame(raw_df)
    use_taroz_signal_clock = (
        weight_mode == TAROZ_FGO_WEIGHT_MODE
        and (fgo_weight_mode is None or fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE)
    )
    clock_kind_for_observation_fn = (
        _taroz_clock_kind_for_observation
        if use_taroz_signal_clock
        else _clock_kind_for_observation
    )
    observation_preparation = _build_observation_preparation_stages(
        raw_df,
        epoch_meta=epoch_meta,
        trip_dir=trip_dir,
        phone_name=phone_name,
        constellation_type=constellation_type,
        signal_type=signal_type,
        multi_gnss=multi_gnss,
        dual_frequency=dual_frequency,
        apply_observation_mask=apply_observation_mask,
        observation_min_cn0_dbhz=observation_min_cn0_dbhz,
        observation_min_elevation_deg=observation_min_elevation_deg,
        gt_times=gt_times,
        gt_ecef=gt_ecef,
        start_epoch=start_epoch,
        max_epochs=max_epochs,
        weight_mode=weight_mode,
        fgo_weight_mode=fgo_weight_mode,
        tdcp_enabled=tdcp_enabled,
        adr_sign=adr_sign,
        multi_gnss_mask_fn=_multi_gnss_mask,
        signal_types_for_constellation_fn=_signal_types_for_constellation,
        append_gnss_log_only_gps_rows_fn=_append_gnss_log_only_gps_rows,
        matlab_signal_observation_masks_fn=_matlab_signal_observation_masks,
        repair_baseline_wls_fn=_repair_baseline_wls,
        receiver_clock_bias_lookup_from_epoch_meta_fn=_receiver_clock_bias_lookup_from_epoch_meta,
        light_speed_mps=LIGHT_SPEED_MPS,
        gps_tgd_m_by_svid_for_trip_fn=_gps_tgd_m_by_svid_for_trip,
        gps_matrtklib_nav_messages_for_trip_fn=_gps_matrtklib_nav_messages_for_trip,
        gps_iono_alpha_beta_for_trip_fn=_gps_iono_alpha_beta_for_trip,
        gnss_log_matlab_epoch_times_ms_fn=_gnss_log_matlab_epoch_times_ms,
        clean_clock_drift_fn=_clean_clock_drift,
        select_epoch_observations_fn=_select_epoch_observations,
        fill_observation_matrices_fn=_fill_observation_matrices,
        nearest_index_fn=nearest_index,
        gps_arrival_tow_s_from_row_fn=_gps_arrival_tow_s_from_row,
        gps_sat_clock_bias_adjustment_m_fn=_gps_sat_clock_bias_adjustment_m,
        gps_matrtklib_sat_product_adjustment_fn=_gps_matrtklib_sat_product_adjustment,
        clock_kind_for_observation_fn=clock_kind_for_observation_fn,
        is_l5_signal_fn=_is_l5_signal,
        slot_sort_key_fn=_slot_sort_key,
        ecef_to_lla_fn=ecef_to_lla,
        elevation_azimuth_fn=_elevation_azimuth,
        rtklib_tropo_fn=_rtklib_tropo_saastamoinen,
        matlab_signal_clock_dim=MATLAB_SIGNAL_CLOCK_DIM,
        recompute_rtklib_tropo_matrix_fn=_recompute_rtklib_tropo_matrix,
        recompute_rtklib_iono_matrix_fn=_recompute_rtklib_iono_matrix,
        use_matlab_signal_clock=use_taroz_signal_clock,
        use_rtklib_tropo=use_rtklib_tropo,
        taroz_full_trip_cn0_percentile=(
            weight_mode == TAROZ_FGO_WEIGHT_MODE or fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE
        ),
    )
    observation_products = _unpack_observation_preparation_stage(observation_preparation)

    post_observation_stages = _build_configured_post_observation_stages(
        observation_products=observation_products,
        config=PostObservationStageConfig(
            trip_dir=trip_dir,
            phone_name=phone_name,
            apply_absolute_height=apply_absolute_height,
            absolute_height_dist_m=absolute_height_dist_m,
            clock_drift_blocklist_phones=CLOCK_DRIFT_BLOCKLIST_PHONES,
            apply_observation_mask=apply_observation_mask,
            has_window_subset=(
                start_epoch > 0
                or len(observation_products.epochs) < len(observation_products.epoch_time_context.epoch_time_keys)
            ),
            constellation_type=constellation_type,
            signal_type=signal_type,
            weight_mode=weight_mode,
            multi_gnss=multi_gnss,
            observation_min_cn0_dbhz=observation_min_cn0_dbhz,
            observation_min_elevation_deg=observation_min_elevation_deg,
            dual_frequency=dual_frequency,
            factor_dt_max_s=factor_dt_max_s,
            apply_base_correction=apply_base_correction,
            data_root=data_root,
            trip=trip,
            doppler_residual_mask_mps=doppler_residual_mask_mps,
            pseudorange_doppler_mask_m=pseudorange_doppler_mask_m,
            pseudorange_residual_mask_m=pseudorange_residual_mask_m,
            pseudorange_residual_mask_l5_m=pseudorange_residual_mask_l5_m,
            tdcp_consistency_threshold_m=tdcp_consistency_threshold_m,
            tdcp_loffset_m=tdcp_loffset_m,
            matlab_residual_diagnostics_mask_path=matlab_residual_diagnostics_mask_path,
            tdcp_geometry_correction=tdcp_geometry_correction,
            tdcp_weight_scale=tdcp_weight_scale,
            imu_frame=imu_frame,
            imu_sample_dt_mode=imu_sample_dt_mode,
            default_pd_l1_threshold_m=OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
            default_pd_l5_threshold_m=OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_L5_M,
            default_pr_l1_threshold_m=OBS_MASK_RESIDUAL_THRESHOLD_M,
            default_pr_l5_threshold_m=OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
        ),
        dependencies=PostObservationStageDependencies(
            build_trip_arrays_fn=build_trip_arrays,
            gnss_log_corrected_pseudorange_matrix_fn=_gnss_log_corrected_pseudorange_matrix,
            load_absolute_height_reference_ecef_fn=load_absolute_height_reference_ecef,
            clock_jump_from_epoch_counts_fn=_clock_jump_from_epoch_counts,
            estimate_residual_clock_series_fn=_estimate_residual_clock_series,
            combine_clock_jump_masks_fn=_combine_clock_jump_masks,
            detect_clock_jumps_from_clock_bias_fn=_detect_clock_jumps_from_clock_bias,
            clean_clock_drift_fn=_clean_clock_drift,
            correction_matrix_fn=compute_base_pseudorange_correction_matrix,
            mask_doppler_residual_outliers_fn=_mask_doppler_residual_outliers,
            slot_frequency_thresholds_fn=_slot_frequency_thresholds,
            mask_pseudorange_doppler_consistency_fn=_mask_pseudorange_doppler_consistency,
            slot_pseudorange_common_bias_groups_fn=_slot_pseudorange_common_bias_groups,
            remap_pseudorange_isb_by_group_fn=_remap_pseudorange_isb_by_group,
            pseudorange_global_isb_by_group_fn=_pseudorange_global_isb_by_group,
            is_l5_signal_fn=_is_l5_signal,
            mask_pseudorange_residual_outliers_fn=_mask_pseudorange_residual_outliers,
            build_tdcp_arrays_fn=_build_tdcp_arrays,
            apply_diagnostics_mask_fn=_apply_matlab_residual_diagnostics_mask,
            apply_geometry_correction_fn=_apply_tdcp_geometry_correction,
            apply_weight_scale_fn=_apply_tdcp_weight_scale,
            load_device_imu_measurements_fn=load_device_imu_measurements,
            process_device_imu_fn=process_device_imu,
            project_stop_to_epochs_fn=project_stop_to_epochs,
            preintegrate_processed_imu_fn=preintegrate_processed_imu,
        ),
    )
    trip_arrays = _assemble_prepared_trip_arrays_stage(
        trip_arrays_cls=TripArrays,
        observation_products=observation_products,
        post_observation_stages=post_observation_stages,
        has_truth=(gt_times is not None and gt_ecef is not None),
        dual_frequency=dual_frequency,
    )
    return _replace_dataclass(
        trip_arrays,
        build_start_epoch=int(start_epoch),
        build_max_epochs=int(max_epochs),
    )


def run_wls(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    *,
    sys_kind: np.ndarray | None = None,
    n_clock: int = 1,
    fallback_xyz: np.ndarray | None = None,
) -> np.ndarray:
    n_epoch = sat_ecef.shape[0]
    out = np.zeros((n_epoch, 3 + n_clock), dtype=np.float64)
    if fallback_xyz is not None:
        fallback = np.asarray(fallback_xyz, dtype=np.float64).reshape(n_epoch, 3)
        finite_fallback = np.isfinite(fallback).all(axis=1)
        out[finite_fallback, :3] = fallback[finite_fallback]
    solver_cache: dict[tuple[int, ...], MultiGNSSSolver] = {}

    for i in range(n_epoch):
        idx = np.flatnonzero(weights[i] > 0)
        if sys_kind is not None and n_clock > 1 and idx.size:
            sk_all = np.asarray(sys_kind[i, idx], dtype=np.int32)
            idx = idx[(0 <= sk_all) & (sk_all < n_clock)]
        if idx.size < 4:
            continue

        if sys_kind is not None and n_clock > 1:
            active_kinds = sorted({int(sk) for sk in sys_kind[i, idx] if 0 <= int(sk) < n_clock})
            if len(active_kinds) > 1 and idx.size >= 3 + len(active_kinds):
                systems = tuple(active_kinds)
                solver = solver_cache.get(systems)
                if solver is None:
                    solver = MultiGNSSSolver(systems=list(systems), max_iter=25, tol=1e-9)
                    solver_cache[systems] = solver
                kind_to_system = {sk: sk for sk in active_kinds}
                system_ids = np.array([kind_to_system[int(sk)] for sk in sys_kind[i, idx]], dtype=np.int32)
                pos, biases, n_iter = solver.solve(sat_ecef[i, idx], pseudorange[i, idx], system_ids, weights[i, idx])
                if n_iter >= 0 and np.linalg.norm(pos) > 1e3:
                    out[i, :3] = pos
                    bias_by_kind = {
                        sk: float(biases.get(kind_to_system[sk], 0.0))
                        for sk in active_kinds
                    }
                    ref_bias = float(bias_by_kind.get(0, next(iter(bias_by_kind.values()), 0.0)))
                    out[i, 3] = ref_bias
                    for sk, bias in bias_by_kind.items():
                        if 0 < sk < n_clock:
                            out[i, 3 + sk] = float(bias) - ref_bias
                    continue

        state, _ = wls_position(
            sat_ecef[i, idx].reshape(-1),
            pseudorange[i, idx],
            weights[i, idx],
            25,
            1e-9,
        )
        out[i, :4] = state

    return out


def fit_state_with_clock_bias(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    xyz: np.ndarray,
    *,
    sys_kind: np.ndarray | None = None,
    n_clock: int = 1,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    state = np.zeros((xyz.shape[0], 3 + n_clock), dtype=np.float64)
    state[:, :3] = xyz
    weighted_sse = 0.0
    weight_sum = 0.0
    per_epoch_wmse = np.full(xyz.shape[0], np.nan, dtype=np.float64)

    for i in range(xyz.shape[0]):
        idx = np.flatnonzero(weights[i] > 0)
        if idx.size < 4:
            continue
        rho = _geometric_range_with_sagnac(sat_ecef[i, idx], xyz[i])
        resid0 = pseudorange[i, idx] - rho
        w = weights[i, idx]
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            continue
        sk = sys_kind[i, idx] if sys_kind is not None else np.zeros(idx.size, dtype=np.int32)
        bias = _solve_clock_biases(resid0, w, np.asarray(sk, dtype=np.int32), n_clock)
        pred_bias = _fill_clock_design(np.asarray(sk, dtype=np.int32), n_clock) @ bias
        resid = resid0 - pred_bias
        sse = float(np.sum(w * resid * resid))
        state[i, 3 : 3 + n_clock] = bias
        weighted_sse += sse
        weight_sum += w_sum
        per_epoch_wmse[i] = sse / w_sum

    return state, weighted_sse, weight_sum, per_epoch_wmse


def weighted_mse(weighted_sse: float, weight_sum: float) -> float:
    if weight_sum <= 0.0:
        return float("inf")
    return float(weighted_sse / weight_sum)


def _seed_vd_state(
    raw_state: np.ndarray,
    baseline_state: np.ndarray,
    dt: np.ndarray,
    *,
    n_clock: int,
    clock_drift_mps: np.ndarray | None = None,
    imu_attitude_state: bool = False,
    imu_pose_position_state: bool = False,
    imu_accel_bias_state: bool = False,
    imu_gyro_bias_state: bool = False,
) -> np.ndarray:
    n_epoch = raw_state.shape[0]
    base_width = 7 + n_clock
    if imu_attitude_state:
        extra_width = 12 if imu_pose_position_state else 9
    else:
        extra_width = 6 if imu_gyro_bias_state else (3 if imu_accel_bias_state else 0)
    state_width = base_width + extra_width
    seed = np.zeros((n_epoch, state_width), dtype=np.float64)
    seed[:, 6 + n_clock] = np.nan

    raw_pos = raw_state[:, :3].copy()
    invalid = np.linalg.norm(raw_pos, axis=1) < 1e3
    raw_pos[invalid] = baseline_state[invalid, :3]
    seed[:, :3] = raw_pos
    pose_position_idx = base_width if (imu_attitude_state and imu_pose_position_state) else None
    if pose_position_idx is not None:
        seed[:, pose_position_idx : pose_position_idx + 3] = raw_pos

    if raw_state.shape[1] >= 3 + n_clock:
        seed[:, 6 : 6 + n_clock] = raw_state[:, 3 : 3 + n_clock]
    if np.any(invalid):
        seed[invalid, 6 : 6 + n_clock] = baseline_state[invalid, 3 : 3 + n_clock]

    if clock_drift_mps is not None:
        drift = np.asarray(clock_drift_mps, dtype=np.float64).reshape(-1)
        if drift.size == n_epoch:
            finite = np.isfinite(drift)
            seed[finite, 6 + n_clock] = drift[finite]

    if n_epoch > 1:
        for t in range(n_epoch - 1):
            dt_s = float(dt[t])
            if dt_s > 0.0:
                seed[t, 3:6] = (seed[t + 1, :3] - seed[t, :3]) / dt_s
                if not np.isfinite(seed[t, 6 + n_clock]):
                    seed[t, 6 + n_clock] = (seed[t + 1, 6] - seed[t, 6]) / dt_s
        seed[-1, 3:6] = seed[-2, 3:6]
        if not np.isfinite(seed[-1, 6 + n_clock]):
            seed[-1, 6 + n_clock] = seed[-2, 6 + n_clock]

    if imu_attitude_state:
        attitude_idx = base_width + (3 if imu_pose_position_state else 0)
        vel_enu = np.zeros((n_epoch, 3), dtype=np.float64)
        for t in range(n_epoch):
            if not (np.isfinite(seed[t, :3]).all() and np.isfinite(seed[t, 3:6]).all()):
                continue
            vel_enu[t] = _ecef_to_enu_relative((seed[t, :3] + seed[t, 3:6]).reshape(1, 3), seed[t, :3])[0]
        rpy = estimate_rpy_from_velocity(vel_enu)
        rot_enu_body = _eul_xyz_to_rotm(rpy)
        for t in range(n_epoch):
            if not (np.isfinite(seed[t, :3]).all() and np.isfinite(rot_enu_body[t]).all()):
                continue
            enu_basis_ecef = _enu_to_ecef_relative(np.eye(3, dtype=np.float64), seed[t, :3]) - seed[t, :3]
            rot_ecef_enu = enu_basis_ecef.T
            seed[t, attitude_idx : attitude_idx + 3] = _rotm_to_rotvec(rot_ecef_enu @ rot_enu_body[t])

    return seed


def _apply_external_vd_seed_state(
    seed: np.ndarray,
    external_state: np.ndarray | None,
    dt: np.ndarray,
    *,
    n_clock: int,
) -> np.ndarray:
    """Overlay an external VD seed while keeping existing fallback values."""

    if external_state is None:
        return seed
    ext = np.asarray(external_state, dtype=np.float64)
    if ext.ndim != 2 or ext.shape[0] != seed.shape[0]:
        raise ValueError("external VD seed state must be [T, state_dim]")
    if ext.shape[1] < 3:
        return seed

    out = seed.copy()
    base_width = 7 + int(n_clock)
    out_split_pose = out.shape[1] >= base_width + 12
    ext_split_pose = ext.shape[1] >= base_width + 12
    out_pose_idx = base_width if out_split_pose else None
    out_attitude_idx = base_width + 3 if out_split_pose else base_width
    ext_pose_idx = base_width if ext_split_pose else None
    ext_attitude_idx = base_width + 3 if ext_split_pose else base_width

    pos = ext[:, :3]
    finite_pos = np.isfinite(pos).all(axis=1) & (np.linalg.norm(pos, axis=1) > 1.0e3)
    out[finite_pos, :3] = pos[finite_pos]
    if out_pose_idx is not None:
        out[finite_pos, out_pose_idx : out_pose_idx + 3] = pos[finite_pos]
        if ext_pose_idx is not None and ext.shape[1] >= ext_pose_idx + 3:
            pose_pos = ext[:, ext_pose_idx : ext_pose_idx + 3]
            finite_pose = np.isfinite(pose_pos).all(axis=1) & (np.linalg.norm(pose_pos, axis=1) > 1.0e3)
            out[finite_pose, out_pose_idx : out_pose_idx + 3] = pose_pos[finite_pose]

    finite_vel = np.zeros(ext.shape[0], dtype=bool)
    if ext.shape[1] >= 6:
        vel = ext[:, 3:6]
        finite_vel = np.isfinite(vel).all(axis=1)
        out[finite_vel, 3:6] = vel[finite_vel]

    if ext.shape[1] >= base_width:
        clocks = ext[:, 6 : 6 + n_clock]
        finite_clocks = np.isfinite(clocks).all(axis=1)
        out[finite_clocks, 6 : 6 + n_clock] = clocks[finite_clocks]
        drift = ext[:, 6 + n_clock]
        finite_drift = np.isfinite(drift)
        out[finite_drift, 6 + n_clock] = drift[finite_drift]
    elif ext.shape[1] >= 3 + n_clock:
        clocks = ext[:, 3 : 3 + n_clock]
        finite_clocks = np.isfinite(clocks).all(axis=1)
        out[finite_clocks, 6 : 6 + n_clock] = clocks[finite_clocks]

    def _copy_triplet(src_idx: int | None, dst_idx: int | None) -> None:
        if src_idx is None or dst_idx is None:
            return
        if ext.shape[1] < src_idx + 3 or out.shape[1] < dst_idx + 3:
            return
        values = ext[:, src_idx : src_idx + 3]
        finite_values = np.isfinite(values)
        out[:, dst_idx : dst_idx + 3][finite_values] = values[finite_values]

    if out.shape[1] > base_width and ext.shape[1] > base_width:
        if out_split_pose or ext_split_pose:
            _copy_triplet(ext_attitude_idx, out_attitude_idx)
            _copy_triplet(ext_attitude_idx + 3, out_attitude_idx + 3)
            _copy_triplet(ext_attitude_idx + 6, out_attitude_idx + 6)
        else:
            n_extra = min(ext.shape[1] - base_width, out.shape[1] - base_width)
            extra = ext[:, base_width : base_width + n_extra]
            finite_extra = np.isfinite(extra)
            out[:, base_width : base_width + n_extra][finite_extra] = extra[finite_extra]

    if out.shape[0] > 1:
        drift_col = 6 + n_clock
        for t in range(out.shape[0] - 1):
            dt_s = float(dt[t]) if t < len(dt) else 0.0
            if dt_s <= 0.0:
                continue
            if not finite_vel[t] and np.isfinite(out[t : t + 2, :3]).all():
                out[t, 3:6] = (out[t + 1, :3] - out[t, :3]) / dt_s
            if not np.isfinite(out[t, drift_col]) and np.isfinite(out[t : t + 2, 6]).all():
                out[t, drift_col] = (out[t + 1, 6] - out[t, 6]) / dt_s
        if not finite_vel[-1]:
            out[-1, 3:6] = out[-2, 3:6]
        if not np.isfinite(out[-1, drift_col]):
            out[-1, drift_col] = out[-2, drift_col]

    attitude_idx = out_attitude_idx
    if out.shape[1] >= attitude_idx + 3:
        external_attitude = np.zeros(out.shape[0], dtype=bool)
        if ext.shape[1] >= ext_attitude_idx + 3:
            external_attitude = np.isfinite(ext[:, ext_attitude_idx : ext_attitude_idx + 3]).all(axis=1)
        recompute_attitude = ~external_attitude
        if recompute_attitude.any():
            vel_enu = np.zeros((out.shape[0], 3), dtype=np.float64)
            finite_pose_vel = np.isfinite(out[:, :6]).all(axis=1)
            for t in np.flatnonzero(recompute_attitude & finite_pose_vel):
                vel_enu[t] = _ecef_to_enu_relative((out[t, :3] + out[t, 3:6]).reshape(1, 3), out[t, :3])[0]
            rpy = estimate_rpy_from_velocity(vel_enu)
            finite_rpy = np.isfinite(rpy).all(axis=1)
            rot_enu_body = _eul_xyz_to_rotm(rpy)
            for t in np.flatnonzero(recompute_attitude & finite_pose_vel & finite_rpy):
                enu_basis_ecef = _enu_to_ecef_relative(np.eye(3, dtype=np.float64), out[t, :3]) - out[t, :3]
                rot_ecef_enu = enu_basis_ecef.T
                out[t, attitude_idx : attitude_idx + 3] = _rotm_to_rotvec(rot_ecef_enu @ rot_enu_body[t])

    return out


def _taroz_preprocessing_origin_ecef_from_trip_dir(trip_dir: Path) -> np.ndarray:
    path = Path(trip_dir) / "device_gnss.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    cols = [
        "utcTimeMillis",
        "BiasUncertaintyNanos",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    frame = pd.read_csv(path, usecols=lambda col: col in cols)
    if "BiasUncertaintyNanos" in frame.columns:
        bias_unc = pd.to_numeric(frame["BiasUncertaintyNanos"], errors="coerce").to_numpy(dtype=np.float64)
        frame = frame[~(bias_unc > 1.0e4)]
    frame = frame.sort_values("utcTimeMillis", kind="mergesort").drop_duplicates("utcTimeMillis", keep="first")
    xyz_cols = ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    if not set(xyz_cols).issubset(frame.columns):
        raise ValueError(f"{path} is missing WLS ECEF columns")
    xyz = frame[xyz_cols].to_numpy(dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.any():
        raise ValueError(f"{path} has no finite WLS ECEF origin candidate")
    return xyz[np.flatnonzero(finite)[0]].astype(np.float64)


def _resolve_taroz_fgo_seed_state_csv(path: Path, *, prefer_graph_state: bool = False) -> Path:
    seed_path = Path(path)
    if seed_path.is_dir():
        names = (
            ("phone_data_gnss_graph_state.csv", "phone_data_gnss_initial_state.csv")
            if prefer_graph_state
            else ("phone_data_gnss_initial_state.csv", "phone_data_gnss_graph_state.csv")
        )
        for name in names:
            candidate = seed_path / name
            if candidate.is_file():
                seed_path = candidate
                break
        else:
            seed_path = seed_path / names[0]
    if not seed_path.is_file():
        raise FileNotFoundError(seed_path)
    return seed_path


def _taroz_fgo_seed_prefer_graph_state(config: BridgeConfig) -> bool:
    if getattr(config, "fgo_fixed_linearization", False):
        return False
    if getattr(config, "taroz_pose_bias_seed_state_csv", None) is not None:
        return False
    return bool(
        getattr(config, "apply_imu_prior", False)
        and getattr(config, "taroz_stop_mask_from_seed_velocity", False)
    )


def _resolve_taroz_pose_bias_seed_state_csv(path: Path) -> Path:
    seed_path = Path(path)
    if seed_path.is_dir():
        seed_path = seed_path / "phone_data_imu_state.csv"
    if not seed_path.is_file():
        raise FileNotFoundError(seed_path)
    return seed_path


def _load_taroz_result_mat_clocks(seed_path: Path, epoch_index: np.ndarray, n_clock: int) -> tuple[np.ndarray, np.ndarray] | None:
    mat_path = Path(seed_path).with_name("result_gnss_imu.mat")
    if not mat_path.is_file():
        return None
    try:
        from scipy.io import loadmat
    except Exception:
        return None
    try:
        mat = loadmat(mat_path, variable_names=("clkest", "dclkest"))
    except Exception:
        return None
    if "clkest" not in mat or "dclkest" not in mat:
        return None
    clkest = np.asarray(mat["clkest"], dtype=np.float64)
    dclkest = np.asarray(mat["dclkest"], dtype=np.float64).reshape(-1)
    if clkest.ndim != 2 or clkest.shape[0] == 0 or dclkest.size == 0:
        return None
    n_epoch = int(np.asarray(epoch_index).size)
    clocks = np.full((n_epoch, int(n_clock)), np.nan, dtype=np.float64)
    drift = np.full(n_epoch, np.nan, dtype=np.float64)
    epoch_arr = np.asarray(epoch_index, dtype=np.float64).reshape(-1)
    valid = np.isfinite(epoch_arr)
    epoch_zero = np.rint(epoch_arr[valid]).astype(np.int64) - 1
    valid_rows = np.flatnonzero(valid)
    in_range = (epoch_zero >= 0) & (epoch_zero < clkest.shape[0]) & (epoch_zero < dclkest.size)
    if not np.any(in_range):
        return None
    rows = valid_rows[in_range]
    mat_rows = epoch_zero[in_range]
    n_copy = min(int(n_clock), clkest.shape[1])
    clocks[rows, :n_copy] = clkest[mat_rows, :n_copy]
    drift[rows] = dclkest[mat_rows]
    return clocks, drift


def _load_taroz_fgo_seed_state(
    path: Path,
    batch: TripArrays,
    *,
    trip_dir: Path,
    prefer_graph_state: bool = False,
    pose_bias_path: Path | None = None,
) -> np.ndarray:
    seed_path = _resolve_taroz_fgo_seed_state_csv(path, prefer_graph_state=prefer_graph_state)
    frame = pd.read_csv(seed_path)
    required = {
        "utcTimeMillis",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{seed_path} is missing columns: {missing}")

    n_epoch = int(batch.times_ms.size)
    n_clock = int(batch.n_clock)
    base_width = 7 + n_clock
    has_imu_extra = bool(
        {"roll", "pitch", "yaw"}.issubset(set(frame.columns))
        or {"bias_acc_x", "bias_acc_y", "bias_acc_z"}.issubset(set(frame.columns))
        or {"bias_gyro_x", "bias_gyro_y", "bias_gyro_z"}.issubset(set(frame.columns))
    )
    state = np.full((n_epoch, base_width + (9 if has_imu_extra else 0)), np.nan, dtype=np.float64)
    target = pd.DataFrame(
        {
            "_epoch_idx": np.arange(n_epoch, dtype=np.int64),
            "utcTimeMillis": np.rint(np.asarray(batch.times_ms, dtype=np.float64)).astype(np.int64),
        }
    )
    source = frame.copy()
    source["utcTimeMillis"] = np.rint(pd.to_numeric(source["utcTimeMillis"], errors="coerce")).astype("Int64")
    source = source.dropna(subset=["utcTimeMillis"]).copy()
    source["utcTimeMillis"] = source["utcTimeMillis"].astype(np.int64)
    source = source.sort_values("utcTimeMillis", kind="mergesort").drop_duplicates("utcTimeMillis", keep="first")
    joined = target.merge(source, on="utcTimeMillis", how="left", sort=False)

    origin_ecef = _taroz_preprocessing_origin_ecef_from_trip_dir(trip_dir)
    pos_enu = joined[["position_x", "position_y", "position_z"]].to_numpy(dtype=np.float64)
    vel_enu = joined[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(dtype=np.float64)
    finite_pos = np.isfinite(pos_enu).all(axis=1)
    finite_vel = np.isfinite(vel_enu).all(axis=1)
    if finite_pos.any():
        state[finite_pos, :3] = _enu_to_ecef_relative(pos_enu[finite_pos], origin_ecef)
    if finite_vel.any():
        state[finite_vel, 3:6] = _enu_to_ecef_relative(vel_enu[finite_vel], origin_ecef) - origin_ecef
    for clock_idx in range(n_clock):
        col = f"clock_bias_m_{clock_idx}"
        if col not in joined.columns:
            continue
        state[:, 6 + clock_idx] = pd.to_numeric(joined[col], errors="coerce").to_numpy(dtype=np.float64)
    if "clock_drift_mps" in joined.columns:
        state[:, 6 + n_clock] = pd.to_numeric(joined["clock_drift_mps"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(state[:, 6 : 6 + n_clock]).all(axis=1).any():
        epoch_col = (
            pd.to_numeric(joined["epoch_index"], errors="coerce").to_numpy(dtype=np.float64)
            if "epoch_index" in joined.columns
            else joined["_epoch_idx"].to_numpy(dtype=np.float64) + 1.0
        )
        mat_clocks = _load_taroz_result_mat_clocks(seed_path, epoch_col, n_clock)
        if mat_clocks is not None:
            clocks, drift = mat_clocks
            finite_clocks = np.isfinite(clocks).all(axis=1)
            state[finite_clocks, 6 : 6 + n_clock] = clocks[finite_clocks]
            finite_drift = np.isfinite(drift)
            state[finite_drift, 6 + n_clock] = drift[finite_drift]
    if has_imu_extra:
        attitude_idx = base_width
        accel_bias_idx = attitude_idx + 3
        gyro_bias_idx = attitude_idx + 6
        rpy_cols = ["roll", "pitch", "yaw"]
        if set(rpy_cols).issubset(joined.columns):
            rpy = joined[rpy_cols].to_numpy(dtype=np.float64)
            finite_rpy = np.isfinite(rpy).all(axis=1)
            if finite_rpy.any():
                rot_enu_body = _gtsam_rzryrx_to_rotm(rpy[finite_rpy])
                enu_basis_ecef = _enu_to_ecef_relative(np.eye(3, dtype=np.float64), origin_ecef) - origin_ecef
                rot_ecef_enu = enu_basis_ecef.T
                rotm = np.einsum("ij,njk->nik", rot_ecef_enu, rot_enu_body)
                state[finite_rpy, attitude_idx : attitude_idx + 3] = np.vstack(
                    [_rotm_to_rotvec(rot) for rot in rotm]
                )
        acc_bias_cols = ["bias_acc_x", "bias_acc_y", "bias_acc_z"]
        if set(acc_bias_cols).issubset(joined.columns):
            acc_bias = joined[acc_bias_cols].to_numpy(dtype=np.float64)
            finite_acc_bias = np.isfinite(acc_bias).all(axis=1)
            state[finite_acc_bias, accel_bias_idx : accel_bias_idx + 3] = acc_bias[finite_acc_bias]
        gyro_bias_cols = ["bias_gyro_x", "bias_gyro_y", "bias_gyro_z"]
        if set(gyro_bias_cols).issubset(joined.columns):
            gyro_bias = joined[gyro_bias_cols].to_numpy(dtype=np.float64)
            finite_gyro_bias = np.isfinite(gyro_bias).all(axis=1)
            state[finite_gyro_bias, gyro_bias_idx : gyro_bias_idx + 3] = gyro_bias[finite_gyro_bias]
    if pose_bias_path is not None:
        pose_path = _resolve_taroz_pose_bias_seed_state_csv(pose_bias_path)
        pose_frame = pd.read_csv(pose_path)
        pose_required = {
            "utcTimeMillis",
            "position_x",
            "position_y",
            "position_z",
            "roll",
            "pitch",
            "yaw",
        }
        pose_missing = sorted(pose_required - set(pose_frame.columns))
        if pose_missing:
            raise ValueError(f"{pose_path} is missing columns: {pose_missing}")

        split_state = np.full((n_epoch, base_width + 12), np.nan, dtype=np.float64)
        split_state[:, :base_width] = state[:, :base_width]
        finite_base_pos = np.isfinite(state[:, :3]).all(axis=1)
        split_state[finite_base_pos, base_width : base_width + 3] = state[finite_base_pos, :3]
        if state.shape[1] >= base_width + 9:
            split_state[:, base_width + 3 : base_width + 12] = state[:, base_width : base_width + 9]

        pose_source = pose_frame.copy()
        pose_source["utcTimeMillis"] = np.rint(
            pd.to_numeric(pose_source["utcTimeMillis"], errors="coerce")
        ).astype("Int64")
        pose_source = pose_source.dropna(subset=["utcTimeMillis"]).copy()
        pose_source["utcTimeMillis"] = pose_source["utcTimeMillis"].astype(np.int64)
        pose_source = (
            pose_source.sort_values("utcTimeMillis", kind="mergesort")
            .drop_duplicates("utcTimeMillis", keep="first")
        )
        pose_joined = target.merge(pose_source, on="utcTimeMillis", how="left", sort=False)
        pose_enu = pose_joined[["position_x", "position_y", "position_z"]].to_numpy(dtype=np.float64)
        finite_pose = np.isfinite(pose_enu).all(axis=1)
        if finite_pose.any():
            split_state[finite_pose, base_width : base_width + 3] = _enu_to_ecef_relative(
                pose_enu[finite_pose],
                origin_ecef,
            )
        rpy = pose_joined[["roll", "pitch", "yaw"]].to_numpy(dtype=np.float64)
        finite_rpy = np.isfinite(rpy).all(axis=1)
        if finite_rpy.any():
            rot_enu_body = _gtsam_rzryrx_to_rotm(rpy[finite_rpy])
            enu_basis_ecef = _enu_to_ecef_relative(np.eye(3, dtype=np.float64), origin_ecef) - origin_ecef
            rot_ecef_enu = enu_basis_ecef.T
            rotm = np.einsum("ij,njk->nik", rot_ecef_enu, rot_enu_body)
            split_state[finite_rpy, base_width + 3 : base_width + 6] = np.vstack(
                [_rotm_to_rotvec(rot) for rot in rotm]
            )
        acc_bias_cols = ["bias_acc_x", "bias_acc_y", "bias_acc_z"]
        if set(acc_bias_cols).issubset(pose_joined.columns):
            acc_bias = pose_joined[acc_bias_cols].to_numpy(dtype=np.float64)
            finite_acc_bias = np.isfinite(acc_bias).all(axis=1)
            split_state[finite_acc_bias, base_width + 6 : base_width + 9] = acc_bias[finite_acc_bias]
        gyro_bias_cols = ["bias_gyro_x", "bias_gyro_y", "bias_gyro_z"]
        if set(gyro_bias_cols).issubset(pose_joined.columns):
            gyro_bias = pose_joined[gyro_bias_cols].to_numpy(dtype=np.float64)
            finite_gyro_bias = np.isfinite(gyro_bias).all(axis=1)
            split_state[finite_gyro_bias, base_width + 9 : base_width + 12] = gyro_bias[finite_gyro_bias]
        return split_state
    return state


def _taroz_stop_mask_from_seed_velocity(
    stop_mask: np.ndarray | None,
    fgo_seed_state: np.ndarray | None,
    *,
    threshold_mps: float = TAROZ_STOP_VELOCITY_THRESHOLD_MPS,
) -> np.ndarray | None:
    if stop_mask is None:
        return None
    out = np.asarray(stop_mask, dtype=bool).reshape(-1).copy()
    if fgo_seed_state is None:
        return out
    seed = np.asarray(fgo_seed_state, dtype=np.float64)
    if seed.ndim != 2 or seed.shape[1] < 6:
        raise ValueError("taroz stop seed state must have velocity columns")
    n = min(out.size, seed.shape[0])
    speed_mps = np.linalg.norm(seed[:n, 3:6], axis=1)
    seed_stop = np.isfinite(speed_mps) & (speed_mps < float(threshold_mps))
    out[:n] &= seed_stop
    if n < out.size:
        out[n:] = False
    return out


def _resolve_taroz_factor_mask_csv(path: Path) -> Path:
    mask_path = Path(path)
    if mask_path.is_dir():
        mask_path = mask_path / "phone_data_gnss_factor_mask.csv"
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    return mask_path


def _resolve_taroz_imu_factor_mask_csv(path: Path) -> Path:
    mask_path = Path(path)
    if mask_path.is_dir():
        mask_path = mask_path / "phone_data_imu_factor_mask.csv"
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    return mask_path


def _taroz_imu_interval_support_mask(batch: TripArrays, imu_factor_mask_csv: Path) -> np.ndarray:
    n_interval = max(int(batch.times_ms.size) - 1, 0)
    support = np.zeros(n_interval, dtype=bool)
    if n_interval <= 0:
        return support

    mask_path = _resolve_taroz_imu_factor_mask_csv(imu_factor_mask_csv)
    frame = pd.read_csv(mask_path)
    if frame.empty:
        return support
    if "field" in frame.columns:
        frame = frame[frame["field"].astype(str).str.startswith("IMU_")]
    required = {"utcTimeMillis", "nextUtcTimeMillis"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{mask_path} is missing columns: {missing}")

    interval_by_time: dict[tuple[int, int], int] = {}
    times = np.rint(np.asarray(batch.times_ms, dtype=np.float64)).astype(np.int64)
    for interval_idx in range(n_interval):
        interval_by_time[(int(times[interval_idx]), int(times[interval_idx + 1]))] = int(interval_idx)

    keyed = frame[list(required)].copy()
    keyed["utcTimeMillis"] = pd.to_numeric(keyed["utcTimeMillis"], errors="coerce").round().astype("Int64")
    keyed["nextUtcTimeMillis"] = pd.to_numeric(keyed["nextUtcTimeMillis"], errors="coerce").round().astype("Int64")
    keyed = keyed.dropna(subset=["utcTimeMillis", "nextUtcTimeMillis"])
    for _, row in keyed.drop_duplicates().iterrows():
        interval_idx = interval_by_time.get((int(row["utcTimeMillis"]), int(row["nextUtcTimeMillis"])))
        if interval_idx is not None:
            support[interval_idx] = True
    return support


def _mask_preintegration_array(
    value: np.ndarray | None,
    support: np.ndarray,
    *,
    fill_value: float,
) -> np.ndarray | None:
    if value is None:
        return None
    out = np.asarray(value).copy()
    n = min(out.shape[0], support.size)
    if n <= 0:
        return out
    inactive = ~support[:n]
    if inactive.any():
        out[:n][inactive] = fill_value
    return out


def _apply_taroz_imu_factor_mask_to_batch(batch: TripArrays, imu_factor_mask_csv: Path) -> TripArrays:
    preint = getattr(batch, "imu_preintegration", None)
    if preint is None:
        return batch

    support = _taroz_imu_interval_support_mask(batch, imu_factor_mask_csv)
    sample_count = np.asarray(preint.sample_count, dtype=np.int32).copy()
    n = min(sample_count.size, support.size)
    if n > 0:
        sample_count[:n][~support[:n]] = 0

    masked_preint = _replace_dataclass(
        preint,
        sample_count=sample_count,
        delta_t_s=_mask_preintegration_array(preint.delta_t_s, support, fill_value=np.nan),
        delta_v_body=_mask_preintegration_array(preint.delta_v_body, support, fill_value=np.nan),
        delta_p_body=_mask_preintegration_array(preint.delta_p_body, support, fill_value=np.nan),
        delta_angle_rad=_mask_preintegration_array(preint.delta_angle_rad, support, fill_value=np.nan),
        delta_p_bias_accel_jac=_mask_preintegration_array(preint.delta_p_bias_accel_jac, support, fill_value=0.0),
        delta_v_bias_accel_jac=_mask_preintegration_array(preint.delta_v_bias_accel_jac, support, fill_value=0.0),
        delta_p_bias_gyro_jac=_mask_preintegration_array(preint.delta_p_bias_gyro_jac, support, fill_value=0.0),
        delta_v_bias_gyro_jac=_mask_preintegration_array(preint.delta_v_bias_gyro_jac, support, fill_value=0.0),
        delta_angle_bias_gyro_jac=_mask_preintegration_array(
            preint.delta_angle_bias_gyro_jac,
            support,
            fill_value=0.0,
        ),
        pva_accel_noise_cov=_mask_preintegration_array(preint.pva_accel_noise_cov, support, fill_value=0.0),
        pva_gyro_noise_cov=_mask_preintegration_array(preint.pva_gyro_noise_cov, support, fill_value=0.0),
        pva_integration_noise_cov=_mask_preintegration_array(
            preint.pva_integration_noise_cov,
            support,
            fill_value=0.0,
        ),
        gravity_ecef=_mask_preintegration_array(preint.gravity_ecef, support, fill_value=np.nan),
        preint_meas_cov=_mask_preintegration_array(preint.preint_meas_cov, support, fill_value=0.0),
    )
    return _replace_dataclass(batch, imu_preintegration=masked_preint)


def _taroz_factor_support_masks(
    batch: TripArrays,
    factor_mask_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_path = _resolve_taroz_factor_mask_csv(factor_mask_csv)
    frame = pd.read_csv(mask_path)
    required = {"field", "freq", "utcTimeMillis", "sys", "svid"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{mask_path} is missing columns: {missing}")

    n_epoch, n_slot = batch.weights.shape
    pr_mask = np.zeros((n_epoch, n_slot), dtype=bool)
    doppler_mask = np.zeros((n_epoch, n_slot), dtype=bool)
    tdcp_mask = np.zeros((max(n_epoch - 1, 0), n_slot), dtype=bool)
    epoch_by_time = {
        int(round(float(time_ms))): int(epoch_idx)
        for epoch_idx, time_ms in enumerate(np.asarray(batch.times_ms, dtype=np.float64))
        if np.isfinite(time_ms)
    }
    slot_by_key: dict[tuple[int, int, str], int] = {}
    for slot_idx, slot_key in enumerate(batch.slot_keys):
        constellation_type, svid, signal_type = slot_key
        key = (
            int(_constellation_to_matlab_sys(int(constellation_type))),
            int(svid),
            str(_slot_frequency_label(str(signal_type))),
        )
        slot_by_key[key] = int(slot_idx)

    for row in frame.itertuples(index=False):
        field = str(getattr(row, "field")).upper()
        if field not in {"P", "D", "L"}:
            continue
        time_value = getattr(row, "utcTimeMillis")
        if not pd.notna(time_value):
            continue
        epoch_idx = epoch_by_time.get(int(round(float(time_value))))
        if epoch_idx is None:
            continue
        key = (
            int(getattr(row, "sys")),
            int(getattr(row, "svid")),
            str(getattr(row, "freq")),
        )
        slot_idx = slot_by_key.get(key)
        if slot_idx is None:
            continue
        if field == "P":
            pr_mask[epoch_idx, slot_idx] = True
        elif field == "D":
            doppler_mask[epoch_idx, slot_idx] = True
        elif epoch_idx < tdcp_mask.shape[0]:
            tdcp_mask[epoch_idx, slot_idx] = True
    return pr_mask, doppler_mask, tdcp_mask


def _apply_taroz_factor_mask_to_batch(
    batch: TripArrays,
    factor_mask_csv: Path,
    *,
    trip_dir: Path | None = None,
    rebase_state_csv: Path | None = None,
    use_fixed_values: bool = True,
) -> TripArrays:
    pr_mask, doppler_mask, tdcp_mask = _taroz_factor_support_masks(batch, factor_mask_csv)
    pr_source = batch.weights_fgo if batch.weights_fgo is not None else batch.weights
    weights_fgo = np.asarray(pr_source, dtype=np.float64).copy()
    weights_fgo[~pr_mask] = 0.0

    doppler_weights_fgo = batch.doppler_weights_fgo
    doppler_source = batch.doppler_weights_fgo if batch.doppler_weights_fgo is not None else batch.doppler_weights
    if doppler_source is not None:
        doppler_weights_fgo = np.asarray(doppler_source, dtype=np.float64).copy()
        doppler_weights_fgo[~doppler_mask] = 0.0

    tdcp_weights_fgo = batch.tdcp_weights_fgo
    tdcp_source = batch.tdcp_weights_fgo if batch.tdcp_weights_fgo is not None else batch.tdcp_weights
    if tdcp_source is not None:
        tdcp_weights_fgo = np.asarray(tdcp_source, dtype=np.float64).copy()
        if tdcp_mask.shape == tdcp_weights_fgo.shape:
            tdcp_weights_fgo[~tdcp_mask] = 0.0
        else:
            tdcp_weights_fgo[...] = 0.0

    fixed_kwargs: dict[str, np.ndarray] = {}
    frame = pd.read_csv(_resolve_taroz_factor_mask_csv(factor_mask_csv))
    fixed_columns = {
        "field",
        "freq",
        "utcTimeMillis",
        "nextUtcTimeMillis",
        "sys",
        "svid",
        "sigma",
        "measurement",
        "los_e",
        "los_n",
        "los_u",
        "origin1_e",
        "origin1_n",
        "origin1_u",
        "origin2_e",
        "origin2_n",
        "origin2_u",
    }
    if fixed_columns.issubset(frame.columns):
        origin_ecef = _taroz_preprocessing_origin_ecef_from_trip_dir(trip_dir) if trip_dir is not None else None
        rebase_pos_by_time: dict[int, np.ndarray] = {}
        rebase_vel_by_time: dict[int, np.ndarray] = {}
        if rebase_state_csv is not None:
            rebase_path = _resolve_taroz_fgo_seed_state_csv(Path(rebase_state_csv), prefer_graph_state=True)
            rebase_frame = pd.read_csv(rebase_path)
            required_rebase_cols = {
                "utcTimeMillis",
                "position_x",
                "position_y",
                "position_z",
                "velocity_x",
                "velocity_y",
                "velocity_z",
            }
            missing_rebase = sorted(required_rebase_cols - set(rebase_frame.columns))
            if missing_rebase:
                raise ValueError(f"{rebase_path} is missing columns: {missing_rebase}")
            for rebase_row in rebase_frame.itertuples(index=False):
                time_value = getattr(rebase_row, "utcTimeMillis")
                if not pd.notna(time_value):
                    continue
                key = int(round(float(time_value)))
                pos = np.array(
                    [
                        float(getattr(rebase_row, "position_x")),
                        float(getattr(rebase_row, "position_y")),
                        float(getattr(rebase_row, "position_z")),
                    ],
                    dtype=np.float64,
                )
                vel = np.array(
                    [
                        float(getattr(rebase_row, "velocity_x")),
                        float(getattr(rebase_row, "velocity_y")),
                        float(getattr(rebase_row, "velocity_z")),
                    ],
                    dtype=np.float64,
                )
                if np.isfinite(pos).all():
                    rebase_pos_by_time[key] = pos
                if np.isfinite(vel).all():
                    rebase_vel_by_time[key] = vel

        def _taroz_position_to_solver(value: np.ndarray) -> np.ndarray:
            if origin_ecef is None:
                return value
            return _enu_to_ecef_relative(value.reshape(1, 3), origin_ecef)[0]

        def _taroz_vector_to_solver(value: np.ndarray) -> np.ndarray:
            if origin_ecef is None:
                return value
            return _enu_to_ecef_relative(value.reshape(1, 3), origin_ecef)[0] - origin_ecef

        n_epoch, n_slot = batch.weights.shape
        pr_measurement = np.zeros((n_epoch, n_slot), dtype=np.float64)
        pr_weights = np.zeros((n_epoch, n_slot), dtype=np.float64)
        pr_ref = np.full((n_epoch, 3), np.nan, dtype=np.float64)
        pr_los = np.zeros((n_epoch, n_slot, 3), dtype=np.float64)
        doppler_measurement = np.zeros((n_epoch, n_slot), dtype=np.float64)
        doppler_weights = np.zeros((n_epoch, n_slot), dtype=np.float64)
        doppler_ref = np.full((n_epoch, 3), np.nan, dtype=np.float64)
        doppler_los = np.zeros((n_epoch, n_slot, 3), dtype=np.float64)
        tdcp_measurement = np.zeros((max(n_epoch - 1, 0), n_slot), dtype=np.float64)
        tdcp_weights = np.zeros((max(n_epoch - 1, 0), n_slot), dtype=np.float64)
        tdcp_ref = np.full((n_epoch, 3), np.nan, dtype=np.float64)
        fgo_sat_ecef = np.asarray(batch.sat_ecef, dtype=np.float64).copy()
        epoch_by_time = {
            int(round(float(time_ms))): int(epoch_idx)
            for epoch_idx, time_ms in enumerate(np.asarray(batch.times_ms, dtype=np.float64))
            if np.isfinite(time_ms)
        }
        slot_by_key: dict[tuple[int, int, str], int] = {}
        for slot_idx, slot_key in enumerate(batch.slot_keys):
            constellation_type, svid, signal_type = slot_key
            key = (
                int(_constellation_to_matlab_sys(int(constellation_type))),
                int(svid),
                str(_slot_frequency_label(str(signal_type))),
            )
            slot_by_key[key] = int(slot_idx)

        for row in frame.itertuples(index=False):
            field = str(getattr(row, "field")).upper()
            if field not in {"P", "D", "L"}:
                continue
            time_value = getattr(row, "utcTimeMillis")
            if not pd.notna(time_value):
                continue
            epoch_idx = epoch_by_time.get(int(round(float(time_value))))
            if epoch_idx is None:
                continue
            key = (
                int(getattr(row, "sys")),
                int(getattr(row, "svid")),
                str(getattr(row, "freq")),
            )
            slot_idx = slot_by_key.get(key)
            if slot_idx is None:
                continue
            sigma = float(getattr(row, "sigma"))
            if not np.isfinite(sigma) or sigma <= 0.0:
                continue
            measurement = float(getattr(row, "measurement"))
            los = np.array(
                [float(getattr(row, "los_e")), float(getattr(row, "los_n")), float(getattr(row, "los_u"))],
                dtype=np.float64,
            )
            origin1 = np.array(
                [
                    float(getattr(row, "origin1_e")),
                    float(getattr(row, "origin1_n")),
                    float(getattr(row, "origin1_u")),
                ],
                dtype=np.float64,
            )
            if field == "P":
                rebased_origin1 = rebase_pos_by_time.get(int(round(float(time_value))))
                if rebased_origin1 is not None:
                    measurement += float(np.dot(los, origin1 - rebased_origin1))
                    origin1 = rebased_origin1
                pr_measurement[epoch_idx, slot_idx] = measurement
                pr_weights[epoch_idx, slot_idx] = 1.0 / (sigma * sigma)
                pr_los[epoch_idx, slot_idx] = _taroz_vector_to_solver(los)
                if not np.isfinite(pr_ref[epoch_idx]).all():
                    pr_ref[epoch_idx] = _taroz_position_to_solver(origin1)
            elif field == "D":
                rebased_origin1 = rebase_vel_by_time.get(int(round(float(time_value))))
                if rebased_origin1 is not None:
                    measurement += float(np.dot(los, origin1 - rebased_origin1))
                    origin1 = rebased_origin1
                doppler_measurement[epoch_idx, slot_idx] = measurement
                doppler_weights[epoch_idx, slot_idx] = 1.0 / (sigma * sigma)
                doppler_los[epoch_idx, slot_idx] = _taroz_vector_to_solver(los)
                if not np.isfinite(doppler_ref[epoch_idx]).all():
                    doppler_ref[epoch_idx] = _taroz_vector_to_solver(origin1)
            elif epoch_idx < tdcp_measurement.shape[0]:
                next_time_value = getattr(row, "nextUtcTimeMillis")
                next_idx = epoch_idx + 1
                if pd.notna(next_time_value):
                    next_idx = epoch_by_time.get(int(round(float(next_time_value))), next_idx)
                if next_idx >= n_epoch:
                    continue
                origin2 = np.array(
                    [
                        float(getattr(row, "origin2_e")),
                        float(getattr(row, "origin2_n")),
                        float(getattr(row, "origin2_u")),
                    ],
                    dtype=np.float64,
                )
                rebased_origin1 = rebase_pos_by_time.get(int(round(float(time_value))))
                rebased_origin2 = (
                    rebase_pos_by_time.get(int(round(float(next_time_value))))
                    if pd.notna(next_time_value)
                    else None
                )
                if rebased_origin1 is not None and rebased_origin2 is not None and np.isfinite(origin2).all():
                    measurement += float(
                        np.dot(los, (origin2 - rebased_origin2) + (rebased_origin1 - origin1))
                    )
                    origin1 = rebased_origin1
                    origin2 = rebased_origin2
                tdcp_measurement[epoch_idx, slot_idx] = measurement
                tdcp_weights[epoch_idx, slot_idx] = 1.0 / (sigma * sigma)
                los_solver = _taroz_vector_to_solver(los)
                origin1_solver = _taroz_position_to_solver(origin1)
                if not np.isfinite(tdcp_ref[epoch_idx]).all():
                    tdcp_ref[epoch_idx] = origin1_solver
                if np.isfinite(origin2).all() and not np.isfinite(tdcp_ref[next_idx]).all():
                    tdcp_ref[next_idx] = _taroz_position_to_solver(origin2)
                if np.isfinite(origin2).all():
                    fgo_sat_ecef[next_idx, slot_idx] = _taroz_position_to_solver(origin2) - 1000.0 * los_solver

        pr_ref[~np.isfinite(pr_ref).all(axis=1)] = 0.0
        doppler_ref[~np.isfinite(doppler_ref).all(axis=1)] = 0.0
        tdcp_ref[~np.isfinite(tdcp_ref).all(axis=1)] = 0.0
        weights_fgo = pr_weights
        doppler_weights_fgo = doppler_weights
        tdcp_weights_fgo = tdcp_weights
        if use_fixed_values:
            fixed_kwargs = {
                "fgo_pr_measurement": pr_measurement,
                "fgo_pr_linearization_ref_ecef": pr_ref,
                "fgo_pr_linearization_los_ecef": pr_los,
                "fgo_doppler_measurement": doppler_measurement,
                "fgo_doppler_linearization_ref_vel": doppler_ref,
                "fgo_doppler_linearization_los_ecef": doppler_los,
                "fgo_tdcp_measurement": tdcp_measurement,
                "fgo_tdcp_linearization_ref_ecef": tdcp_ref,
                "fgo_sat_ecef": fgo_sat_ecef,
            }
            if origin_ecef is not None:
                fixed_kwargs["fgo_position_origin_ecef"] = origin_ecef

    return _replace_dataclass(
        batch,
        weights_fgo=weights_fgo,
        doppler_weights_fgo=doppler_weights_fgo,
        tdcp_weights_fgo=tdcp_weights_fgo,
        **fixed_kwargs,
    )


def _imu_diagonal_covariance_weights(
    preintegration: IMUPreintegration | None,
    start: int,
    end: int,
    *,
    position_sigma_floor_m: float,
    velocity_noise_mps_sqrt_hz: float,
    attitude_noise_rad_sqrt_hz: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if preintegration is None or end - start <= 1:
        return None, None, None
    i0 = max(int(start), 0)
    i1 = max(int(end) - 1, i0)
    n_interval = int(preintegration.delta_t_s.size)
    if i0 >= n_interval:
        return None, None, None
    i1 = min(i1, n_interval)
    if i1 <= i0:
        return None, None, None
    dt = np.asarray(preintegration.delta_t_s[i0:i1], dtype=np.float64)
    sample_count = np.asarray(preintegration.sample_count[i0:i1], dtype=np.int32)
    valid = np.isfinite(dt) & (dt > 0.0) & (sample_count > 0)
    if not valid.any():
        return None, None, None

    def weights_from_sigma(sigma: np.ndarray) -> np.ndarray:
        out = np.zeros((dt.size, 3), dtype=np.float64)
        good = valid & np.isfinite(sigma) & (sigma > 0.0)
        out[good, :] = 1.0 / (sigma[good, None] * sigma[good, None])
        return out

    vel_noise = abs(float(velocity_noise_mps_sqrt_hz))
    att_noise = abs(float(attitude_noise_rad_sqrt_hz))
    pos_floor = abs(float(position_sigma_floor_m))
    dt_pos = np.maximum(dt, 0.0)
    pos_from_acc = vel_noise * np.sqrt(dt_pos * dt_pos * dt_pos / 3.0) if vel_noise > 0.0 else np.zeros_like(dt)
    pos_sigma = np.maximum(pos_floor, pos_from_acc)
    vel_sigma = vel_noise * np.sqrt(dt_pos)
    att_sigma = att_noise * np.sqrt(dt_pos)
    return weights_from_sigma(pos_sigma), weights_from_sigma(vel_sigma), weights_from_sigma(att_sigma)


def _imu_preintegration_information_matrices(
    preintegration: IMUPreintegration | None,
    start: int,
    end: int,
    *,
    position_sigma_floor_m: float,
    velocity_noise_mps_sqrt_hz: float,
    attitude_noise_rad_sqrt_hz: float,
) -> np.ndarray | None:
    """Return per-interval information matrices in GTSAM ImuFactor [R, P, V] order."""

    if preintegration is None or end - start <= 1:
        return None
    i0 = max(int(start), 0)
    i1 = max(int(end) - 1, i0)
    n_interval = int(preintegration.delta_t_s.size)
    if i0 >= n_interval:
        return None
    i1 = min(i1, n_interval)
    if i1 <= i0:
        return None

    dt = np.asarray(preintegration.delta_t_s[i0:i1], dtype=np.float64)
    sample_count = np.asarray(preintegration.sample_count[i0:i1], dtype=np.int32)
    valid = np.isfinite(dt) & (dt > 0.0) & (sample_count > 0)
    if not valid.any():
        return None

    direct_cov = getattr(preintegration, "preint_meas_cov", None)
    if (
        direct_cov is not None
        and np.asarray(direct_cov).shape == (n_interval, 9, 9)
        and _imu_preintegration_has_complete_bias_jacobians(preintegration, i0, i1, valid)
    ):
        covariances = np.asarray(direct_cov[i0:i1], dtype=np.float64)
        info = np.zeros((dt.size, 9, 9), dtype=np.float64)
        any_cov = False
        for row in range(dt.size):
            if not bool(valid[row]):
                continue
            cov = 0.5 * (covariances[row] + covariances[row].T)
            if not np.isfinite(cov).all() or not np.any(np.abs(cov) > 0.0):
                continue
            row_info = np.linalg.pinv(cov, hermitian=True, rcond=1e-12)
            info[row] = 0.5 * (row_info + row_info.T)
            any_cov = True
        if any_cov:
            return info

    acc_noise = abs(float(velocity_noise_mps_sqrt_hz))
    gyro_noise = abs(float(attitude_noise_rad_sqrt_hz))
    pos_floor = abs(float(position_sigma_floor_m))
    if acc_noise <= 0.0 and gyro_noise <= 0.0 and pos_floor <= 0.0:
        return None

    accel_unit_cov = getattr(preintegration, "pva_accel_noise_cov", None)
    gyro_unit_cov = getattr(preintegration, "pva_gyro_noise_cov", None)
    integration_unit_cov = getattr(preintegration, "pva_integration_noise_cov", None)
    has_sample_covariance = (
        accel_unit_cov is not None
        and gyro_unit_cov is not None
        and integration_unit_cov is not None
        and np.asarray(accel_unit_cov).shape == (n_interval, 9, 9)
        and np.asarray(gyro_unit_cov).shape == (n_interval, 9, 9)
        and np.asarray(integration_unit_cov).shape == (n_interval, 9, 9)
    )
    if has_sample_covariance:
        accel_cov = np.asarray(accel_unit_cov[i0:i1], dtype=np.float64)
        gyro_cov = np.asarray(gyro_unit_cov[i0:i1], dtype=np.float64)
        integration_cov = np.asarray(integration_unit_cov[i0:i1], dtype=np.float64)
        info = np.zeros((dt.size, 9, 9), dtype=np.float64)
        for row in range(dt.size):
            if not bool(valid[row]):
                continue
            cov = (
                (acc_noise * acc_noise) * accel_cov[row]
                + (gyro_noise * gyro_noise) * gyro_cov[row]
                + (pos_floor * pos_floor) * integration_cov[row]
            )
            # Internal propagated covariances are [P, V, R]; GTSAM ImuFactor
            # evaluateError and noise model use [R, P, V].
            order = np.array([6, 7, 8, 0, 1, 2, 3, 4, 5], dtype=np.int64)
            cov = cov[np.ix_(order, order)]
            cov = 0.5 * (cov + cov.T)
            if not np.isfinite(cov).all() or not np.any(np.abs(cov) > 0.0):
                continue
            row_info = np.linalg.pinv(cov, hermitian=True, rcond=1e-12)
            info[row] = 0.5 * (row_info + row_info.T)
        return info

    info = np.zeros((dt.size, 9, 9), dtype=np.float64)
    for row, dt_s in enumerate(dt):
        if not bool(valid[row]):
            continue
        dt_pos = max(float(dt_s), 0.0)
        if acc_noise > 0.0:
            acc_var = acc_noise * acc_noise
            cov_pp = acc_var * dt_pos * dt_pos * dt_pos / 3.0
            if pos_floor > 0.0:
                cov_pp = max(cov_pp, pos_floor * pos_floor)
            cov_vv = acc_var * dt_pos
            cov_pv = acc_var * dt_pos * dt_pos * 0.5
            det = cov_pp * cov_vv - cov_pv * cov_pv
            if np.isfinite(det) and det > 0.0:
                i_pp = cov_vv / det
                i_pv = -cov_pv / det
                i_vv = cov_pp / det
                for axis in range(3):
                    p = 3 + axis
                    v = 6 + axis
                    info[row, p, p] = i_pp
                    info[row, p, v] = i_pv
                    info[row, v, p] = i_pv
                    info[row, v, v] = i_vv
        if gyro_noise > 0.0:
            att_var = gyro_noise * gyro_noise * dt_pos
            if np.isfinite(att_var) and att_var > 0.0:
                for axis in range(3):
                    a = axis
                    info[row, a, a] = 1.0 / att_var
    return info


def _imu_preintegration_has_complete_bias_jacobians(
    preintegration: IMUPreintegration | None,
    start: int | None = None,
    end: int | None = None,
    valid: np.ndarray | None = None,
) -> bool:
    if preintegration is None:
        return False
    n_interval = int(np.asarray(preintegration.delta_t_s).size)
    i0 = 0 if start is None else max(int(start), 0)
    i1 = n_interval if end is None else min(max(int(end), i0), n_interval)
    if i1 <= i0:
        return False
    if valid is None:
        valid_window = np.ones(i1 - i0, dtype=bool)
    else:
        valid_window = np.asarray(valid, dtype=bool).reshape(-1)
        if valid_window.size != i1 - i0:
            return False
    if not valid_window.any():
        return False
    for jac_name in (
        "delta_p_bias_accel_jac",
        "delta_v_bias_accel_jac",
        "delta_p_bias_gyro_jac",
        "delta_v_bias_gyro_jac",
        "delta_angle_bias_gyro_jac",
    ):
        jac = getattr(preintegration, jac_name, None)
        if jac is None or np.asarray(jac).shape != (n_interval, 3, 3):
            return False
        jac_window = np.asarray(jac[i0:i1], dtype=np.float64)
        if not np.isfinite(jac_window[valid_window]).all():
            return False
    return True


def _taroz_preintegration_with_native_solver_gravity(
    taroz_preintegration: IMUPreintegration,
    native_preintegration: IMUPreintegration | None,
) -> tuple[IMUPreintegration, bool]:
    """Return Taroz deltas/Jacobians with native ECEF gravity for solver use.

    Taroz's CSV exports ``gravity_x/y/z`` as the local GTSAM navigation vector
    ``[0, 0, -g]``. The native VD state is ECEF, so its body-frame IMU residual
    needs the ECEF gravity vector produced by native preintegration.
    """

    native_gravity = getattr(native_preintegration, "gravity_ecef", None)
    if native_gravity is None:
        return taroz_preintegration, False
    gravity = np.asarray(native_gravity, dtype=np.float64)
    if gravity.shape != (int(np.asarray(taroz_preintegration.delta_t_s).size), 3):
        return taroz_preintegration, False
    return _replace_dataclass(taroz_preintegration, gravity_ecef=gravity.copy()), True


def _imu_bias_between_sample_count_weights(
    preintegration: IMUPreintegration | None,
    start: int,
    end: int,
    *,
    accel_bias_sigma_mps2: float,
    gyro_bias_sigma_radps: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if preintegration is None or end - start <= 1:
        return None, None
    i0 = max(int(start), 0)
    i1 = max(int(end) - 1, i0)
    n_interval = int(preintegration.delta_t_s.size)
    if i0 >= n_interval:
        return None, None
    i1 = min(i1, n_interval)
    if i1 <= i0:
        return None, None

    dt = np.asarray(preintegration.delta_t_s[i0:i1], dtype=np.float64)
    sample_count = np.asarray(preintegration.sample_count[i0:i1], dtype=np.float64)
    valid = np.isfinite(dt) & (dt > 0.0) & np.isfinite(sample_count) & (sample_count > 0.0)
    if not valid.any():
        return None, None

    def weights_from_sigma(sigma: float) -> np.ndarray | None:
        sigma_abs = abs(float(sigma))
        if not np.isfinite(sigma_abs) or sigma_abs <= 0.0:
            return None
        out = np.zeros((dt.size, 3), dtype=np.float64)
        out[valid, :] = 1.0 / (sample_count[valid, None] * sigma_abs * sigma_abs)
        return out

    return weights_from_sigma(accel_bias_sigma_mps2), weights_from_sigma(gyro_bias_sigma_radps)


def _vd_seed_weighted_rms(residual: np.ndarray, weights: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(residual) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return float("nan"), 0
    r = residual[valid]
    w = weights[valid]
    weight_sum = float(np.sum(w))
    if weight_sum <= 0.0:
        return float("nan"), 0
    return float(np.sqrt(np.sum(w * r * r) / weight_sum)), int(r.size)


def _vd_seed_doppler_rms(
    sat_ecef: np.ndarray,
    state: np.ndarray,
    sat_vel: np.ndarray | None,
    doppler: np.ndarray | None,
    doppler_weights: np.ndarray | None,
    sat_clock_drift_mps: np.ndarray | None,
    n_clock: int,
) -> tuple[float, int]:
    if sat_vel is None or doppler is None or doppler_weights is None:
        return float("nan"), 0
    geom_rate = _geometric_range_rate_with_sagnac(sat_ecef, state[:, None, :3], sat_vel, state[:, None, 3:6])
    if sat_clock_drift_mps is not None and sat_clock_drift_mps.shape == geom_rate.shape:
        finite = np.isfinite(sat_clock_drift_mps)
        geom_rate[finite] -= sat_clock_drift_mps[finite]
    drift = state[:, 6 + n_clock]
    residual = doppler - (drift[:, None] + geom_rate)
    weights = np.where(np.isfinite(drift)[:, None], doppler_weights, 0.0)
    return _vd_seed_weighted_rms(residual, weights)


def _tdcp_unit_vectors_vd(sat_ecef: np.ndarray, receiver_ecef: np.ndarray) -> np.ndarray:
    return _rtklib_geodist_los(sat_ecef, receiver_ecef)


def _native_pr_range_los_vd(sat_ecef: np.ndarray, receiver_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sat = np.asarray(sat_ecef, dtype=np.float64)
    rx = np.asarray(receiver_ecef, dtype=np.float64)
    dx0 = rx[..., 0] - sat[..., 0]
    dy0 = rx[..., 1] - sat[..., 1]
    dz0 = rx[..., 2] - sat[..., 2]
    r0 = np.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
    theta = EARTH_ROTATION_RATE_RAD_S * (r0 / LIGHT_SPEED_MPS)
    sx_rot = sat[..., 0] * np.cos(theta) + sat[..., 1] * np.sin(theta)
    sy_rot = -sat[..., 0] * np.sin(theta) + sat[..., 1] * np.cos(theta)
    delta = np.stack((rx[..., 0] - sx_rot, rx[..., 1] - sy_rot, rx[..., 2] - sat[..., 2]), axis=-1)
    ranges = np.linalg.norm(delta, axis=-1)
    los = np.full_like(delta, np.nan, dtype=np.float64)
    valid = np.isfinite(ranges) & (ranges > 1.0e-6) & np.isfinite(delta).all(axis=-1)
    los[valid] = delta[valid] / ranges[valid, None]
    return ranges, los


def _rtklib_geodist_los(sat_ecef: np.ndarray, receiver_ecef: np.ndarray) -> np.ndarray:
    sat = np.asarray(sat_ecef, dtype=np.float64)
    rx = np.asarray(receiver_ecef, dtype=np.float64)
    delta = rx - sat
    ranges = np.linalg.norm(delta, axis=-1)
    los = np.full_like(delta, np.nan, dtype=np.float64)
    valid = np.isfinite(ranges) & (ranges > 1.0e-6) & np.isfinite(delta).all(axis=-1)
    los[valid] = delta[valid] / ranges[valid, None]
    return los


def _fixed_pr_linearization_inputs(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref = np.asarray(state[:, :3], dtype=np.float64).copy()
    sat = np.asarray(sat_ecef, dtype=np.float64)
    los = _rtklib_geodist_los(sat, ref[:, None, :])
    ranges = _geometric_range_with_sagnac(sat, ref[:, None, :])
    measurement = np.asarray(pseudorange, dtype=np.float64) - ranges
    solver_weights = np.asarray(weights, dtype=np.float64).copy()
    valid = np.isfinite(measurement) & np.isfinite(ranges) & np.isfinite(los).all(axis=2)
    solver_weights[~valid] = 0.0
    return measurement, solver_weights, ref, los


def _fixed_doppler_linearization_inputs(
    sat_ecef: np.ndarray,
    sat_vel: np.ndarray | None,
    doppler: np.ndarray | None,
    doppler_weights: np.ndarray | None,
    sat_clock_drift_mps: np.ndarray | None,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if sat_vel is None or doppler is None or doppler_weights is None:
        return None
    sat = np.asarray(sat_ecef, dtype=np.float64)
    sv = np.asarray(sat_vel, dtype=np.float64)
    rx = np.asarray(state[:, None, :3], dtype=np.float64)
    rv = np.asarray(state[:, None, 3:6], dtype=np.float64)
    delta = sat - rx
    ranges = np.linalg.norm(delta, axis=2)
    rate_los = np.full_like(delta, np.nan, dtype=np.float64)
    valid_range = np.isfinite(ranges) & (ranges > 1.0e-6) & np.isfinite(delta).all(axis=2)
    rate_los[valid_range] = delta[valid_range] / ranges[valid_range, None]
    taroz_los = -rate_los
    euclidean_rate = np.sum(rate_los * (sv - rv), axis=2)
    sagnac_rate = EARTH_ROTATION_RATE_RAD_S * (
        sv[..., 0] * rx[..., 0, 1][:, None]
        + sat[..., 0] * rv[..., 0, 1][:, None]
        - sv[..., 1] * rx[..., 0, 0][:, None]
        - sat[..., 1] * rv[..., 0, 0][:, None]
    ) / LIGHT_SPEED_MPS
    geom_rate = euclidean_rate - sagnac_rate
    if sat_clock_drift_mps is not None and np.asarray(sat_clock_drift_mps).shape == geom_rate.shape:
        sat_clock_drift = np.asarray(sat_clock_drift_mps, dtype=np.float64)
        finite_clock_drift = np.isfinite(sat_clock_drift)
        geom_rate[finite_clock_drift] -= sat_clock_drift[finite_clock_drift]
    measurement = np.asarray(doppler, dtype=np.float64) - geom_rate
    solver_weights = np.asarray(doppler_weights, dtype=np.float64).copy()
    valid = np.isfinite(measurement) & np.isfinite(taroz_los).all(axis=2)
    solver_weights[~valid] = 0.0
    return measurement, solver_weights, np.asarray(state[:, 3:6], dtype=np.float64).copy(), taroz_los


def _fixed_tdcp_linearization_inputs(
    sat_ecef: np.ndarray,
    tdcp_raw_meas: np.ndarray,
    tdcp_weights: np.ndarray,
    state: np.ndarray,
    sat_clock_bias_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sat = np.asarray(sat_ecef, dtype=np.float64)
    raw = np.asarray(tdcp_raw_meas, dtype=np.float64)
    ref = np.asarray(state[:, :3], dtype=np.float64).copy()
    measurement = raw.copy()
    solver_weights = np.asarray(tdcp_weights, dtype=np.float64).copy()
    n_pair = min(measurement.shape[0], solver_weights.shape[0], sat.shape[0] - 1, ref.shape[0] - 1)
    n_sat = min(measurement.shape[1], solver_weights.shape[1], sat.shape[1])
    valid = np.zeros_like(solver_weights, dtype=bool)
    if n_pair > 0 and n_sat > 0:
        rho0 = _geometric_range_with_sagnac(sat[:n_pair, :n_sat], ref[:n_pair, None, :])
        rho1 = _geometric_range_with_sagnac(sat[1 : n_pair + 1, :n_sat], ref[1 : n_pair + 1, None, :])
        measurement[:n_pair, :n_sat] -= rho1 - rho0
        clock_delta_valid = np.ones((n_pair, n_sat), dtype=bool)
        if sat_clock_bias_m is not None:
            clk = np.asarray(sat_clock_bias_m, dtype=np.float64)
            if clk.ndim >= 2 and clk.shape[0] >= n_pair + 1 and clk.shape[1] >= n_sat:
                clock_delta = clk[1 : n_pair + 1, :n_sat] - clk[:n_pair, :n_sat]
                clock_delta_valid = np.isfinite(clock_delta)
                measurement_block = measurement[:n_pair, :n_sat]
                measurement_block[clock_delta_valid] += clock_delta[clock_delta_valid]
            else:
                clock_delta_valid[:] = False
        valid[:n_pair, :n_sat] = (
            np.isfinite(measurement[:n_pair, :n_sat])
            & np.isfinite(rho0)
            & np.isfinite(rho1)
            & clock_delta_valid
            & np.isfinite(solver_weights[:n_pair, :n_sat])
        )
    solver_weights[~valid] = 0.0
    return measurement, solver_weights, ref


def _vd_seed_tdcp_rms(
    sat_ecef: np.ndarray,
    state: np.ndarray,
    tdcp_meas: np.ndarray | None,
    tdcp_weights: np.ndarray | None,
    sys_kind: np.ndarray | None,
    dt: np.ndarray,
    *,
    n_clock: int,
    tdcp_use_drift: bool,
) -> tuple[float, int]:
    if tdcp_meas is None or tdcp_weights is None or state.shape[0] <= 1:
        return float("nan"), 0
    x0 = state[:-1, :3]
    x1 = state[1:, :3]
    los = _tdcp_unit_vectors_vd(sat_ecef[1:], x1[:, None, :])
    predicted = np.sum(los * (x1 - x0)[:, None, :], axis=2)
    if tdcp_use_drift:
        dt_arr = np.asarray(dt, dtype=np.float64).reshape(-1)[: predicted.shape[0]]
        drift = state[:, 6 + n_clock]
        predicted += 0.5 * dt_arr[:, None] * (drift[:-1, None] + drift[1:, None])
        valid_time = np.isfinite(dt_arr) & (dt_arr > 0.0)
    else:
        sk_arr = sys_kind[1:] if sys_kind is not None else np.zeros_like(tdcp_meas, dtype=np.int32)
        for epoch_idx in range(predicted.shape[0]):
            design = _fill_clock_design(np.asarray(sk_arr[epoch_idx], dtype=np.int32), n_clock)
            clock_delta = state[epoch_idx + 1, 6 : 6 + n_clock] - state[epoch_idx, 6 : 6 + n_clock]
            predicted[epoch_idx] += design @ clock_delta
        valid_time = np.ones(predicted.shape[0], dtype=bool)
    residual = tdcp_meas - predicted
    weights = np.where(valid_time[:, None], tdcp_weights, 0.0)
    return _vd_seed_weighted_rms(residual, weights)


def _finite_float_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _vd_seed_factor_guard_segment_summary(
    sat_ecef: np.ndarray,
    state: np.ndarray,
    *,
    sat_vel: np.ndarray | None,
    doppler: np.ndarray | None,
    doppler_weights: np.ndarray | None,
    sat_clock_drift_mps: np.ndarray | None,
    tdcp_meas: np.ndarray | None,
    tdcp_weights: np.ndarray | None,
    sys_kind: np.ndarray | None,
    dt: np.ndarray,
    n_clock: int,
    tdcp_use_drift: bool,
) -> dict[str, object]:
    doppler_rms, doppler_count = _vd_seed_doppler_rms(
        sat_ecef,
        state,
        sat_vel,
        doppler,
        doppler_weights,
        sat_clock_drift_mps,
        n_clock,
    )
    tdcp_rms, tdcp_count = _vd_seed_tdcp_rms(
        sat_ecef,
        state,
        tdcp_meas,
        tdcp_weights,
        sys_kind,
        dt,
        n_clock=n_clock,
        tdcp_use_drift=tdcp_use_drift,
    )
    reject_reason = ""
    if doppler_count >= VD_SEED_FACTOR_GUARD_MIN_COUNT and doppler_rms > VD_SEED_FACTOR_GUARD_DOPPLER_RMS_MPS:
        reject_reason = "doppler"
    elif tdcp_count >= VD_SEED_FACTOR_GUARD_MIN_COUNT and tdcp_rms > VD_SEED_FACTOR_GUARD_TDCP_RMS_M:
        reject_reason = "tdcp"
    return {
        "doppler_rms_mps": _finite_float_or_none(doppler_rms),
        "doppler_count": int(doppler_count),
        "tdcp_rms_m": _finite_float_or_none(tdcp_rms),
        "tdcp_count": int(tdcp_count),
        "reject_reason": reject_reason,
        "would_reject": bool(reject_reason),
    }


def _vd_seed_factor_guard_rejects_segment(
    sat_ecef: np.ndarray,
    state: np.ndarray,
    *,
    sat_vel: np.ndarray | None,
    doppler: np.ndarray | None,
    doppler_weights: np.ndarray | None,
    sat_clock_drift_mps: np.ndarray | None,
    tdcp_meas: np.ndarray | None,
    tdcp_weights: np.ndarray | None,
    sys_kind: np.ndarray | None,
    dt: np.ndarray,
    n_clock: int,
    tdcp_use_drift: bool,
) -> bool:
    summary = _vd_seed_factor_guard_segment_summary(
        sat_ecef,
        state,
        sat_vel=sat_vel,
        doppler=doppler,
        doppler_weights=doppler_weights,
        sat_clock_drift_mps=sat_clock_drift_mps,
        tdcp_meas=tdcp_meas,
        tdcp_weights=tdcp_weights,
        sys_kind=sys_kind,
        dt=dt,
        n_clock=n_clock,
        tdcp_use_drift=tdcp_use_drift,
    )
    return bool(summary["would_reject"])


def _vd_seed_factor_guard_enabled_for_phone(phone: str) -> bool:
    return phone.lower() == "pixel6pro"


@dataclass
class ChunkedFgoRun:
    """Result of ``run_fgo_chunked``.

    Iterating yields the legacy 11-tuple layout (without
    ``failed_chunk_reasons``) so existing tuple unpacks keep working; new
    fields are attribute-only.
    """

    auto_state: np.ndarray
    fgo_state: np.ndarray
    total_iters: int
    failed_chunks: int
    vd_seed_guard_skipped_segments: int
    vd_seed_guard_skipped_epochs: int
    vd_seed_guard_records: list[dict[str, object]]
    auto_sources: np.ndarray
    auto_source_counts: dict[str, int]
    chunk_records: list[ChunkSelectionRecord]
    fgo_vd_state: np.ndarray | None
    failed_chunk_reasons: dict[str, int] = field(default_factory=dict)

    def __iter__(self):
        return iter(
            (
                self.auto_state,
                self.fgo_state,
                self.total_iters,
                self.failed_chunks,
                self.vd_seed_guard_skipped_segments,
                self.vd_seed_guard_skipped_epochs,
                self.vd_seed_guard_records,
                self.auto_sources,
                self.auto_source_counts,
                self.chunk_records,
                self.fgo_vd_state,
            ),
        )


def _record_chunk_fallback(reasons: dict[str, int], reason: str) -> None:
    """Count a chunk fallback reason and warn the first time it appears."""

    if reason not in reasons:
        warnings.warn(
            f"FGO chunk failed and fell back to raw WLS: {reason}",
            RuntimeWarning,
            stacklevel=3,
        )
    reasons[reason] = reasons.get(reason, 0) + 1


def run_fgo_chunked(
    batch: TripArrays,
    raw_wls: np.ndarray,
    *,
    clock_jump: np.ndarray | None,
    clock_drift_seed_mps: np.ndarray | None,
    clock_use_average_drift: bool,
    tdcp_use_drift: bool,
    stop_mask: np.ndarray | None,
    motion_sigma_m: float,
    clock_drift_sigma_m: float,
    stop_velocity_sigma_mps: float,
    stop_position_sigma_m: float,
    apply_imu_prior: bool,
    imu_position_sigma_m: float,
    imu_velocity_sigma_mps: float,
    fgo_iters: int,
    tol: float,
    chunk_epochs: int,
    use_vd: bool,
    fgo_line_search: bool = True,
    fgo_lm_damping: float = 0.0,
    stop_attitude_sigma_rad: float = 0.0,
    stop_velocity_huber_k: float = 0.0,
    stop_position_huber_k: float = 0.0,
    graph_relative_height: bool = False,
    relative_height_sigma_m: float = 0.5,
    relative_height_huber_k: float = 0.0,
    relative_height_stop_mask: np.ndarray | None = None,
    apply_absolute_height: bool = False,
    absolute_height_sigma_m: float = HEIGHT_ABSOLUTE_SIGMA_M,
    absolute_height_huber_k: float = 0.0,
    imu_attitude_state: bool = False,
    imu_attitude_sigma_rad: float = 0.0,
    imu_preintegration_velocity_noise_mps_sqrt_hz: float = 0.0,
    imu_preintegration_attitude_noise_rad_sqrt_hz: float = 0.0,
    imu_diagonal_covariance: bool = False,
    imu_preintegration_covariance: bool = False,
    imu_factor_use_next_bias: bool = False,
    imu_bias_between_sample_count_scaling: bool = False,
    imu_accel_bias_state: bool = False,
    imu_accel_bias_prior_sigma_mps2: float = IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2,
    imu_accel_bias_between_sigma_mps2: float = IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2,
    imu_gyro_bias_state: bool = False,
    imu_gyro_bias_prior_sigma_radps: float = 0.0,
    imu_gyro_bias_between_sigma_radps: float = 0.0,
    vd_seed_factor_guard: bool = True,
    fgo_robust_kernel: str = "huber",
    fgo_cauchy_c_m: float = 4.0,
    fgo_cauchy_outer_iters: int = 3,
    fgo_huber_k_pr: float = 0.0,
    fgo_huber_k_doppler: float = 0.0,
    fgo_huber_k_tdcp: float = 0.0,
    fgo_fixed_linearization: bool = False,
    fgo_seed_state: np.ndarray | None = None,
) -> "ChunkedFgoRun":
    n_epoch = batch.sat_ecef.shape[0]
    chunk_size = n_epoch if chunk_epochs <= 0 or n_epoch <= chunk_epochs else chunk_epochs
    stitched = raw_wls.copy()
    fgo_stitched = raw_wls.copy()
    use_imu_attitude_state = bool(apply_imu_prior and imu_attitude_state)
    use_imu_gyro_bias_state = bool(apply_imu_prior and (imu_gyro_bias_state or use_imu_attitude_state))
    use_imu_accel_bias_state = bool(apply_imu_prior and (imu_accel_bias_state or use_imu_gyro_bias_state))
    use_imu_pose_position_state = bool(use_imu_attitude_state and fgo_fixed_linearization)
    fgo_vd_state_width = (
        7
        + int(batch.n_clock)
        + (3 if use_imu_pose_position_state else 0)
        + (3 if use_imu_attitude_state else 0)
        + (3 if use_imu_accel_bias_state else 0)
        + (3 if use_imu_gyro_bias_state else 0)
    )
    fgo_vd_stitched = (
        np.full((n_epoch, fgo_vd_state_width), np.nan, dtype=np.float64)
        if use_vd
        else None
    )
    total_iters = 0
    failed_chunks = 0
    failed_chunk_reasons: dict[str, int] = {}
    vd_seed_guard_skipped_segments = 0
    vd_seed_guard_skipped_epochs = 0
    vd_seed_guard_records: list[dict[str, object]] = []
    selected_sources = np.empty(n_epoch, dtype=object)
    selected_source_counts = {"baseline": 0, "raw_wls": 0, "fgo": 0}
    prev_tail_xyz: np.ndarray | None = None
    chunk_records: list[ChunkSelectionRecord] = []
    factor_break_mask = _factor_break_mask(clock_jump, batch.dt, n_epoch)
    seed_state_all = None
    if fgo_seed_state is not None:
        seed_state_all = np.asarray(fgo_seed_state, dtype=np.float64)
        if seed_state_all.ndim != 2 or seed_state_all.shape[0] != n_epoch:
            raise ValueError("fgo_seed_state must be [T, state_dim]")
    rel_height_stop_mask_all = None
    if relative_height_stop_mask is not None:
        rel_height_stop_mask_all = np.asarray(relative_height_stop_mask, dtype=bool).reshape(-1)
        if rel_height_stop_mask_all.size != n_epoch:
            raise ValueError("relative_height_stop_mask must have one entry per epoch")
    effective_graph_relative_height = bool(
        graph_relative_height and not (apply_absolute_height and batch.absolute_height_ref_ecef is not None)
    )

    for start in range(0, n_epoch, chunk_size):
        end = min(start + chunk_size, n_epoch)
        baseline_state, baseline_sse, baseline_weight_sum, _ = fit_state_with_clock_bias(
            batch.sat_ecef[start:end],
            batch.pseudorange[start:end],
            batch.weights[start:end],
            batch.kaggle_wls[start:end],
            sys_kind=(batch.sys_kind[start:end] if batch.sys_kind is not None else None),
            n_clock=batch.n_clock,
        )
        raw_state, raw_sse, raw_weight_sum, _ = fit_state_with_clock_bias(
            batch.sat_ecef[start:end],
            batch.pseudorange[start:end],
            batch.weights[start:end],
            raw_wls[start:end, :3],
            sys_kind=(batch.sys_kind[start:end] if batch.sys_kind is not None else None),
            n_clock=batch.n_clock,
        )

        if use_vd:
            dt_chunk = batch.dt[start:end] if batch.dt is not None else np.zeros(end - start, dtype=np.float64)
            fgo_xyz = raw_state[:, :3].copy()
            chunk_iters = 0
            chunk_success = False
            chunk_failed = False
            for seg_start, seg_end in _segment_ranges(start, end, factor_break_mask):
                local_start = seg_start - start
                local_end = seg_end - start
                if local_end - local_start <= 1:
                    fgo_xyz[local_start:local_end] = raw_state[local_start:local_end, :3]
                    chunk_success = True
                    continue
                seg_dt = dt_chunk[local_start:local_end]
                seg_state = _seed_vd_state(
                    raw_state[local_start:local_end],
                    baseline_state[local_start:local_end],
                    seg_dt,
                    n_clock=batch.n_clock,
                    clock_drift_mps=(
                        clock_drift_seed_mps[seg_start:seg_end] if clock_drift_seed_mps is not None else None
                    ),
                    imu_attitude_state=use_imu_attitude_state,
                    imu_pose_position_state=use_imu_pose_position_state,
                    imu_accel_bias_state=use_imu_accel_bias_state,
                    imu_gyro_bias_state=use_imu_gyro_bias_state,
                )
                if seed_state_all is not None:
                    seg_state = _apply_external_vd_seed_state(
                        seg_state,
                        seed_state_all[seg_start:seg_end],
                        seg_dt,
                        n_clock=batch.n_clock,
                    )
                if seg_start == start and start > 0 and not (
                    factor_break_mask is not None and bool(factor_break_mask[start])
                ):
                    seg_state[0, :3] = stitched[start - 1, :3]
                    seg_state[0, 6 : 6 + batch.n_clock] = stitched[start - 1, 3 : 3 + batch.n_clock]
                doppler_weights_source = (
                    batch.doppler_weights_fgo
                    if getattr(batch, "doppler_weights_fgo", None) is not None
                    else batch.doppler_weights
                )
                tdcp_weights_source = (
                    batch.tdcp_weights_fgo
                    if getattr(batch, "tdcp_weights_fgo", None) is not None
                    else batch.tdcp_weights
                )
                tdcp_meas = None
                tdcp_weights = None
                if batch.tdcp_meas is not None and seg_end - seg_start > 1:
                    tdcp_meas = batch.tdcp_meas[seg_start : seg_end - 1]
                    tdcp_weights = tdcp_weights_source[seg_start : seg_end - 1] if tdcp_weights_source is not None else None
                guard_summary = None
                if vd_seed_factor_guard:
                    guard_summary = _vd_seed_factor_guard_segment_summary(
                        batch.sat_ecef[seg_start:seg_end],
                        seg_state,
                        sat_vel=(batch.sat_vel[seg_start:seg_end] if batch.sat_vel is not None else None),
                        doppler=(batch.doppler[seg_start:seg_end] if batch.doppler is not None else None),
                        doppler_weights=(
                            doppler_weights_source[seg_start:seg_end] if doppler_weights_source is not None else None
                        ),
                        sat_clock_drift_mps=(
                            batch.sat_clock_drift_mps[seg_start:seg_end]
                            if batch.sat_clock_drift_mps is not None
                            else None
                        ),
                        tdcp_meas=tdcp_meas,
                        tdcp_weights=tdcp_weights,
                        sys_kind=(batch.sys_kind[seg_start:seg_end] if batch.sys_kind is not None else None),
                        dt=seg_dt,
                        n_clock=batch.n_clock,
                        tdcp_use_drift=tdcp_use_drift,
                    )
                if guard_summary is not None and bool(guard_summary["would_reject"]):
                    fgo_xyz[local_start:local_end] = raw_state[local_start:local_end, :3]
                    vd_seed_guard_skipped_segments += 1
                    vd_seed_guard_skipped_epochs += local_end - local_start
                    vd_seed_guard_records.append(
                        {
                            "chunk_start_epoch": int(start),
                            "chunk_end_epoch": int(end),
                            "segment_start_epoch": int(seg_start),
                            "segment_end_epoch": int(seg_end),
                            "segment_epochs": int(local_end - local_start),
                            "doppler_count": int(guard_summary["doppler_count"]),
                            "doppler_rms_mps": guard_summary["doppler_rms_mps"],
                            "tdcp_count": int(guard_summary["tdcp_count"]),
                            "tdcp_rms_m": guard_summary["tdcp_rms_m"],
                            "reject_reason": str(guard_summary["reject_reason"]),
                        },
                    )
                    chunk_success = True
                    continue
                seg_stop_mask = stop_mask[seg_start:seg_end] if stop_mask is not None else None
                seg_rel_height_stop_mask = (
                    rel_height_stop_mask_all[seg_start:seg_end]
                    if rel_height_stop_mask_all is not None
                    else seg_stop_mask
                )
                imu_delta_p = None
                imu_delta_v = None
                imu_delta_angle = None
                imu_delta_t = None
                imu_delta_p_bias_accel_jac = None
                imu_delta_v_bias_accel_jac = None
                imu_delta_p_bias_gyro_jac = None
                imu_delta_v_bias_gyro_jac = None
                imu_delta_angle_bias_gyro_jac = None
                imu_gravity = None
                imu_position_weights = None
                imu_velocity_weights = None
                imu_attitude_weights = None
                imu_preintegration_information = None
                imu_accel_bias_between_weights = None
                imu_gyro_bias_between_weights = None
                if apply_imu_prior:
                    (
                        imu_delta_p,
                        imu_delta_v,
                        imu_delta_angle,
                        imu_delta_t,
                        imu_delta_p_bias_accel_jac,
                        imu_delta_v_bias_accel_jac,
                        imu_delta_p_bias_gyro_jac,
                        imu_delta_v_bias_gyro_jac,
                        imu_delta_angle_bias_gyro_jac,
                        _,
                    ) = _imu_preintegration_segment_with_bias_jacobians(
                        batch.imu_preintegration,
                        seg_start,
                        seg_end,
                    )
                    imu_gravity = _imu_preintegration_gravity_segment(batch.imu_preintegration, seg_start, seg_end)
                    if imu_preintegration_covariance:
                        preint_velocity_noise = (
                            float(imu_preintegration_velocity_noise_mps_sqrt_hz)
                            if float(imu_preintegration_velocity_noise_mps_sqrt_hz) > 0.0
                            else float(imu_velocity_sigma_mps)
                        )
                        preint_attitude_noise = (
                            float(imu_preintegration_attitude_noise_rad_sqrt_hz)
                            if float(imu_preintegration_attitude_noise_rad_sqrt_hz) > 0.0
                            else float(imu_attitude_sigma_rad)
                        )
                        imu_preintegration_information = _imu_preintegration_information_matrices(
                            batch.imu_preintegration,
                            seg_start,
                            seg_end,
                            position_sigma_floor_m=imu_position_sigma_m,
                            velocity_noise_mps_sqrt_hz=preint_velocity_noise,
                            attitude_noise_rad_sqrt_hz=preint_attitude_noise,
                        )
                    elif imu_diagonal_covariance:
                        imu_position_weights, imu_velocity_weights, imu_attitude_weights = _imu_diagonal_covariance_weights(
                            batch.imu_preintegration,
                            seg_start,
                            seg_end,
                            position_sigma_floor_m=imu_position_sigma_m,
                            velocity_noise_mps_sqrt_hz=imu_velocity_sigma_mps,
                            attitude_noise_rad_sqrt_hz=imu_attitude_sigma_rad,
                        )
                    if imu_bias_between_sample_count_scaling:
                        imu_accel_bias_between_weights, imu_gyro_bias_between_weights = (
                            _imu_bias_between_sample_count_weights(
                                batch.imu_preintegration,
                                seg_start,
                                seg_end,
                                accel_bias_sigma_mps2=imu_accel_bias_between_sigma_mps2,
                                gyro_bias_sigma_radps=imu_gyro_bias_between_sigma_radps,
                            )
                        )
                    if imu_delta_p is not None and imu_delta_v is not None and seg_dt.size > 1:
                        valid_graph_dt = np.isfinite(seg_dt[:-1]) & (seg_dt[:-1] > 0.0)
                        if valid_graph_dt.size == imu_delta_p.shape[0]:
                            imu_delta_p[~valid_graph_dt, :] = np.nan
                            imu_delta_v[~valid_graph_dt, :] = np.nan
                            if imu_delta_angle is not None:
                                imu_delta_angle[~valid_graph_dt, :] = np.nan
                            if imu_delta_t is not None:
                                imu_delta_t[~valid_graph_dt] = np.nan
                            if imu_gravity is not None:
                                imu_gravity[~valid_graph_dt, :] = np.nan
                            if imu_delta_p_bias_accel_jac is not None:
                                imu_delta_p_bias_accel_jac[~valid_graph_dt, :, :] = 0.0
                            if imu_delta_v_bias_accel_jac is not None:
                                imu_delta_v_bias_accel_jac[~valid_graph_dt, :, :] = 0.0
                            if imu_delta_p_bias_gyro_jac is not None:
                                imu_delta_p_bias_gyro_jac[~valid_graph_dt, :, :] = 0.0
                            if imu_delta_v_bias_gyro_jac is not None:
                                imu_delta_v_bias_gyro_jac[~valid_graph_dt, :, :] = 0.0
                            if imu_delta_angle_bias_gyro_jac is not None:
                                imu_delta_angle_bias_gyro_jac[~valid_graph_dt, :, :] = 0.0
                            if imu_position_weights is not None:
                                imu_position_weights[~valid_graph_dt, :] = 0.0
                            if imu_velocity_weights is not None:
                                imu_velocity_weights[~valid_graph_dt, :] = 0.0
                            if imu_attitude_weights is not None:
                                imu_attitude_weights[~valid_graph_dt, :] = 0.0
                            if imu_preintegration_information is not None:
                                imu_preintegration_information[~valid_graph_dt, :, :] = 0.0
                            if imu_accel_bias_between_weights is not None:
                                imu_accel_bias_between_weights[~valid_graph_dt, :] = 0.0
                            if imu_gyro_bias_between_weights is not None:
                                imu_gyro_bias_between_weights[~valid_graph_dt, :] = 0.0
                rh_sigma = 0.0
                rh_up: np.ndarray | None = None
                rh_ei: np.ndarray | None = None
                rh_ej: np.ndarray | None = None
                abs_height_ref: np.ndarray | None = None
                abs_height_sigma = 0.0
                if effective_graph_relative_height:
                    ref_seg = batch.kaggle_wls[seg_start:seg_end, :3]
                    sm = seg_rel_height_stop_mask
                    groups = _build_relative_height_groups(ref_seg, sm)
                    rh_ei, rh_ej = relative_height_star_edges_from_groups(groups)
                    if rh_ei.size > 0:
                        finite = np.isfinite(ref_seg).all(axis=1)
                        if finite.any():
                            ox = ref_seg[np.flatnonzero(finite)[0]]
                            rh_up = enu_up_ecef_from_origin(ox)
                            rh_sigma = float(relative_height_sigma_m)
                if apply_absolute_height and batch.absolute_height_ref_ecef is not None:
                    abs_height_ref = batch.absolute_height_ref_ecef[seg_start:seg_end]
                    finite_abs = np.isfinite(abs_height_ref).all(axis=1)
                    if finite_abs.any():
                        abs_height_sigma = float(absolute_height_sigma_m)
                        if rh_up is None:
                            ref_seg = batch.kaggle_wls[seg_start:seg_end, :3]
                            finite = np.isfinite(ref_seg).all(axis=1)
                            ox = ref_seg[np.flatnonzero(finite)[0]] if finite.any() else abs_height_ref[np.flatnonzero(finite_abs)[0]]
                            rh_up = enu_up_ecef_from_origin(ox)
                try:
                    sat_ecef_source = getattr(batch, "fgo_sat_ecef", None)
                    sat_ecef_for_solver = (
                        sat_ecef_source[seg_start:seg_end]
                        if sat_ecef_source is not None
                        else batch.sat_ecef[seg_start:seg_end]
                    )
                    position_origin_source = getattr(batch, "fgo_position_origin_ecef", None)
                    position_origin_for_solver = None
                    seg_state_for_solver = seg_state
                    if position_origin_source is not None:
                        origin_arr = np.asarray(position_origin_source, dtype=np.float64).reshape(3)
                        if np.isfinite(origin_arr).all():
                            position_origin_for_solver = origin_arr
                            sat_ecef_for_solver = sat_ecef_for_solver - origin_arr.reshape(1, 1, 3)
                            seg_state_for_solver = seg_state.copy()
                            seg_state_for_solver[:, :3] -= origin_arr
                            if use_imu_pose_position_state:
                                pose_idx = 7 + int(batch.n_clock)
                                if seg_state_for_solver.shape[1] >= pose_idx + 3:
                                    seg_state_for_solver[:, pose_idx : pose_idx + 3] -= origin_arr
                            if abs_height_ref is not None:
                                abs_height_ref = abs_height_ref - origin_arr.reshape(1, 3)
                    debug_state_override = os.getenv("GNSS_GPU_FGO_VD_INITIAL_STATE_OVERRIDE_CSV")
                    if debug_state_override:
                        override_flat = np.loadtxt(Path(debug_state_override), delimiter=",", dtype=np.float64)
                        override_state = np.asarray(override_flat, dtype=np.float64).reshape(
                            -1, seg_state_for_solver.shape[1]
                        )
                        if override_state.shape[0] == end - start:
                            seg_state_for_solver[:, :] = override_state[seg_start - start : seg_end - start]
                        elif override_state.shape[0] == seg_end - seg_start:
                            seg_state_for_solver[:, :] = override_state
                        else:
                            raise RuntimeError(
                                "GNSS_GPU_FGO_VD_INITIAL_STATE_OVERRIDE_CSV row count "
                                f"{override_state.shape[0]} does not match segment {seg_end - seg_start} "
                                f"or window {end - start}"
                            )
                    fgo_weights = (
                        batch.weights_fgo[seg_start:seg_end]
                        if batch.weights_fgo is not None
                        else batch.weights[seg_start:seg_end]
                    )
                    pseudorange_for_solver = batch.pseudorange[seg_start:seg_end]
                    fgo_weights_for_solver = fgo_weights
                    pr_linearization_ref_ecef = None
                    pr_linearization_los_ecef = None
                    doppler_for_solver = batch.doppler[seg_start:seg_end] if batch.doppler is not None else None
                    doppler_weights_for_solver = (
                        doppler_weights_source[seg_start:seg_end] if doppler_weights_source is not None else None
                    )
                    doppler_linearization_ref_vel = None
                    doppler_linearization_los_ecef = None
                    tdcp_linearization_ref_ecef = (
                        seg_state_for_solver[:, :3].copy()
                        if tdcp_meas is not None and int(getattr(batch, "tdcp_geometry_correction_count", 0)) > 0
                        else None
                    )
                    if fgo_fixed_linearization:
                        pr_measurement_override = getattr(batch, "fgo_pr_measurement", None)
                        pr_ref_override = getattr(batch, "fgo_pr_linearization_ref_ecef", None)
                        pr_los_override = getattr(batch, "fgo_pr_linearization_los_ecef", None)
                        if (
                            pr_measurement_override is not None
                            and pr_ref_override is not None
                            and pr_los_override is not None
                        ):
                            pseudorange_for_solver = pr_measurement_override[seg_start:seg_end]
                            fgo_weights_for_solver = fgo_weights
                            pr_linearization_ref_ecef = pr_ref_override[seg_start:seg_end]
                            pr_linearization_los_ecef = pr_los_override[seg_start:seg_end]
                            if position_origin_for_solver is not None:
                                pr_linearization_ref_ecef = (
                                    pr_linearization_ref_ecef - position_origin_for_solver.reshape(1, 3)
                                )
                        else:
                            (
                                pseudorange_for_solver,
                                fgo_weights_for_solver,
                                pr_linearization_ref_ecef,
                                pr_linearization_los_ecef,
                            ) = _fixed_pr_linearization_inputs(
                                sat_ecef_for_solver,
                                batch.pseudorange[seg_start:seg_end],
                                fgo_weights,
                                seg_state_for_solver,
                            )
                        doppler_measurement_override = getattr(batch, "fgo_doppler_measurement", None)
                        doppler_ref_override = getattr(batch, "fgo_doppler_linearization_ref_vel", None)
                        doppler_los_override = getattr(batch, "fgo_doppler_linearization_los_ecef", None)
                        doppler_fixed = None
                        if (
                            doppler_measurement_override is not None
                            and doppler_ref_override is not None
                            and doppler_los_override is not None
                            and doppler_weights_for_solver is not None
                        ):
                            doppler_for_solver = doppler_measurement_override[seg_start:seg_end]
                            doppler_linearization_ref_vel = doppler_ref_override[seg_start:seg_end]
                            doppler_linearization_los_ecef = doppler_los_override[seg_start:seg_end]
                        else:
                            doppler_fixed = _fixed_doppler_linearization_inputs(
                                sat_ecef_for_solver,
                                batch.sat_vel[seg_start:seg_end] if batch.sat_vel is not None else None,
                                batch.doppler[seg_start:seg_end] if batch.doppler is not None else None,
                                (
                                    doppler_weights_source[seg_start:seg_end]
                                    if doppler_weights_source is not None
                                    else None
                                ),
                                (
                                    batch.sat_clock_drift_mps[seg_start:seg_end]
                                    if batch.sat_clock_drift_mps is not None
                                    else None
                                ),
                                seg_state_for_solver,
                            )
                        if doppler_fixed is not None:
                            (
                                doppler_for_solver,
                                doppler_weights_for_solver,
                                doppler_linearization_ref_vel,
                                doppler_linearization_los_ecef,
                            ) = doppler_fixed
                        if (
                            tdcp_meas is not None
                            and tdcp_weights is not None
                            and getattr(batch, "tdcp_raw_meas", None) is not None
                            and seg_end - seg_start > 1
                        ):
                            tdcp_measurement_override = getattr(batch, "fgo_tdcp_measurement", None)
                            tdcp_ref_override = getattr(batch, "fgo_tdcp_linearization_ref_ecef", None)
                            if tdcp_measurement_override is not None and tdcp_ref_override is not None:
                                tdcp_meas = tdcp_measurement_override[seg_start : seg_end - 1]
                                tdcp_linearization_ref_ecef = tdcp_ref_override[seg_start:seg_end]
                                if position_origin_for_solver is not None:
                                    tdcp_linearization_ref_ecef = (
                                        tdcp_linearization_ref_ecef - position_origin_for_solver.reshape(1, 3)
                                    )
                            else:
                                (
                                    tdcp_meas,
                                    tdcp_weights,
                                    tdcp_linearization_ref_ecef,
                                ) = _fixed_tdcp_linearization_inputs(
                                    sat_ecef_for_solver,
                                    batch.tdcp_raw_meas[seg_start : seg_end - 1],
                                    tdcp_weights,
                                    seg_state_for_solver,
                                    (
                                        batch.sat_clock_bias_matrix[seg_start:seg_end]
                                        if getattr(batch, "sat_clock_bias_matrix", None) is not None
                                        else None
                                    ),
                                )
                    vd_kwargs = dict(
                        sys_kind=(batch.sys_kind[seg_start:seg_end] if batch.sys_kind is not None else None),
                        n_clock=batch.n_clock,
                        motion_sigma_m=motion_sigma_m,
                        clock_drift_sigma_m=clock_drift_sigma_m,
                        clock_use_average_drift=clock_use_average_drift,
                        stop_velocity_sigma_mps=stop_velocity_sigma_mps,
                        stop_position_sigma_m=stop_position_sigma_m,
                        stop_attitude_sigma_rad=stop_attitude_sigma_rad,
                        stop_velocity_huber_k=stop_velocity_huber_k,
                        stop_position_huber_k=stop_position_huber_k,
                        huber_k=float(fgo_huber_k_pr),
                        doppler_huber_k=float(fgo_huber_k_doppler),
                        tdcp_huber_k=float(fgo_huber_k_tdcp),
                        max_iter=fgo_iters,
                        tol=tol,
                        line_search=bool(fgo_line_search),
                        lm_damping=float(fgo_lm_damping),
                        sat_vel=(batch.sat_vel[seg_start:seg_end] if batch.sat_vel is not None else None),
                        doppler=doppler_for_solver,
                        doppler_weights=doppler_weights_for_solver,
                        sat_clock_drift=(
                            batch.sat_clock_drift_mps[seg_start:seg_end]
                            if batch.sat_clock_drift_mps is not None
                            else None
                        ),
                        dt=seg_dt,
                        stop_mask=seg_stop_mask,
                        tdcp_meas=tdcp_meas,
                        tdcp_weights=tdcp_weights,
                        tdcp_use_drift=tdcp_use_drift,
                        tdcp_linearization_ref_ecef=tdcp_linearization_ref_ecef,
                        pr_linearization_ref_ecef=pr_linearization_ref_ecef,
                        pr_linearization_los_ecef=pr_linearization_los_ecef,
                        doppler_linearization_ref_vel=doppler_linearization_ref_vel,
                        doppler_linearization_los_ecef=doppler_linearization_los_ecef,
                        relative_height_sigma_m=rh_sigma,
                        relative_height_huber_k=relative_height_huber_k if rh_sigma > 0.0 else 0.0,
                        enu_up_ecef=rh_up,
                        rel_height_edge_i=rh_ei,
                        rel_height_edge_j=rh_ej,
                        absolute_height_ref_ecef=abs_height_ref,
                        absolute_height_sigma_m=abs_height_sigma,
                        absolute_height_huber_k=absolute_height_huber_k if abs_height_sigma > 0.0 else 0.0,
                        imu_delta_p=imu_delta_p,
                        imu_delta_v=imu_delta_v,
                        imu_delta_angle=imu_delta_angle if use_imu_attitude_state else None,
                        imu_delta_t=imu_delta_t,
                        imu_delta_p_bias_accel_jac=imu_delta_p_bias_accel_jac,
                        imu_delta_v_bias_accel_jac=imu_delta_v_bias_accel_jac,
                        imu_delta_p_bias_gyro_jac=imu_delta_p_bias_gyro_jac,
                        imu_delta_v_bias_gyro_jac=imu_delta_v_bias_gyro_jac,
                        imu_delta_angle_bias_gyro_jac=imu_delta_angle_bias_gyro_jac,
                        imu_position_sigma_m=imu_position_sigma_m,
                        imu_velocity_sigma_mps=imu_velocity_sigma_mps,
                        imu_attitude_sigma_rad=imu_attitude_sigma_rad if use_imu_attitude_state else 0.0,
                        imu_position_weights=imu_position_weights,
                        imu_velocity_weights=imu_velocity_weights,
                        imu_attitude_weights=imu_attitude_weights if use_imu_attitude_state else None,
                        imu_preintegration_information=imu_preintegration_information,
                        imu_gravity=imu_gravity if use_imu_attitude_state else None,
                        imu_factor_use_next_bias=bool(imu_factor_use_next_bias),
                        imu_accel_bias_prior_sigma_mps2=(
                            imu_accel_bias_prior_sigma_mps2 if use_imu_accel_bias_state else 0.0
                        ),
                        imu_accel_bias_between_sigma_mps2=(
                            imu_accel_bias_between_sigma_mps2 if use_imu_accel_bias_state else 0.0
                        ),
                        imu_accel_bias_between_weights=(
                            imu_accel_bias_between_weights if use_imu_accel_bias_state else None
                        ),
                        imu_gyro_bias_prior_sigma_radps=(
                            imu_gyro_bias_prior_sigma_radps if use_imu_gyro_bias_state else 0.0
                        ),
                        imu_gyro_bias_between_sigma_radps=(
                            imu_gyro_bias_between_sigma_radps if use_imu_gyro_bias_state else 0.0
                        ),
                        imu_gyro_bias_between_weights=(
                            imu_gyro_bias_between_weights if use_imu_gyro_bias_state else None
                        ),
                        )
                    if str(fgo_robust_kernel).lower() == "cauchy":
                        iters, _mse_unused, _diag = fgo_gnss_lm_vd_cauchy(
                            sat_ecef_for_solver,
                            pseudorange_for_solver,
                            fgo_weights_for_solver,
                            seg_state_for_solver,
                            cauchy_c_m=float(fgo_cauchy_c_m),
                            max_outer_iters=int(fgo_cauchy_outer_iters),
                            huber_k_warmstart=float(fgo_huber_k_pr),
                            **vd_kwargs,
                        )
                    else:
                        iters, _ = fgo_gnss_lm_vd(
                            sat_ecef_for_solver,
                            pseudorange_for_solver,
                            fgo_weights_for_solver,
                            seg_state_for_solver,
                            **vd_kwargs,
                        )
                except RuntimeError as exc:
                    iters = -1
                    fallback_reason = f"{type(exc).__name__}: {exc}"
                else:
                    fallback_reason = "native VD solver returned -1 (rejected inputs, e.g. n_state cap)"
                if int(iters) < 0:
                    _record_chunk_fallback(failed_chunk_reasons, fallback_reason)
                    chunk_failed = True
                    fgo_xyz[local_start:local_end] = raw_state[local_start:local_end, :3]
                    continue
                chunk_success = True
                chunk_iters += int(iters)
                if position_origin_for_solver is not None:
                    seg_state[:, :] = seg_state_for_solver
                    seg_state[:, :3] += position_origin_for_solver
                    if use_imu_pose_position_state:
                        pose_idx = 7 + int(batch.n_clock)
                        if seg_state.shape[1] >= pose_idx + 3:
                            seg_state[:, pose_idx : pose_idx + 3] += position_origin_for_solver
                fgo_xyz[local_start:local_end] = seg_state[:, :3]
                if fgo_vd_stitched is not None:
                    fgo_vd_stitched[seg_start:seg_end, : seg_state.shape[1]] = seg_state
            if chunk_failed:
                failed_chunks += 1
            total_iters += chunk_iters
            iters = chunk_iters if chunk_success else -1
            if int(iters) < 0:
                fgo_state = raw_state.copy()
            else:
                fgo_state, _, _, _ = fit_state_with_clock_bias(
                    batch.sat_ecef[start:end],
                    batch.pseudorange[start:end],
                    batch.weights[start:end],
                    fgo_xyz,
                    sys_kind=(batch.sys_kind[start:end] if batch.sys_kind is not None else None),
                    n_clock=batch.n_clock,
                )
        else:
            fgo_xyz = raw_state[:, :3].copy()
            chunk_iters = 0
            chunk_success = False
            chunk_failed = False
            for seg_start, seg_end in _segment_ranges(start, end, factor_break_mask):
                local_start = seg_start - start
                local_end = seg_end - start
                if local_end - local_start <= 1:
                    fgo_xyz[local_start:local_end] = raw_state[local_start:local_end, :3]
                    chunk_success = True
                    continue
                seg_state = np.zeros((local_end - local_start, 3 + batch.n_clock), dtype=np.float64)
                seg_state[:, :3] = raw_state[local_start:local_end, :3]
                seg_state[:, 3 : 3 + batch.n_clock] = raw_state[local_start:local_end, 3 : 3 + batch.n_clock]
                if seg_start == start and start > 0 and not (
                    factor_break_mask is not None and bool(factor_break_mask[start])
                ):
                    seg_state[0] = stitched[start - 1]
                try:
                    fgo_weights = (
                        batch.weights_fgo[seg_start:seg_end]
                        if batch.weights_fgo is not None
                        else batch.weights[seg_start:seg_end]
                    )
                    iters, _ = fgo_gnss_lm(
                        batch.sat_ecef[seg_start:seg_end],
                        batch.pseudorange[seg_start:seg_end],
                        fgo_weights,
                        seg_state,
                        sys_kind=(batch.sys_kind[seg_start:seg_end] if batch.sys_kind is not None else None),
                        n_clock=batch.n_clock,
                        motion_sigma_m=motion_sigma_m,
                        max_iter=fgo_iters,
                        tol=tol,
                        tdcp_huber_k=float(fgo_huber_k_tdcp),
                    )
                except RuntimeError as exc:
                    iters = -1
                    fallback_reason = f"{type(exc).__name__}: {exc}"
                else:
                    fallback_reason = "native solver returned -1 (rejected inputs, e.g. n_state cap)"
                if int(iters) < 0:
                    _record_chunk_fallback(failed_chunk_reasons, fallback_reason)
                    chunk_failed = True
                    fgo_xyz[local_start:local_end] = raw_state[local_start:local_end, :3]
                    continue
                chunk_success = True
                chunk_iters += int(iters)
                fgo_xyz[local_start:local_end] = seg_state[:, :3]
            if chunk_failed:
                failed_chunks += 1
            total_iters += chunk_iters
            iters = chunk_iters if chunk_success else -1
            if int(iters) < 0:
                fgo_state = raw_state.copy()
            else:
                fgo_state, _, _, _ = fit_state_with_clock_bias(
                    batch.sat_ecef[start:end],
                    batch.pseudorange[start:end],
                    batch.weights[start:end],
                    fgo_xyz,
                    sys_kind=(batch.sys_kind[start:end] if batch.sys_kind is not None else None),
                    n_clock=batch.n_clock,
                )

        candidate_states = {
            "baseline": baseline_state,
            "raw_wls": raw_state,
        }
        candidate_mse = {
            "baseline": weighted_mse(baseline_sse, baseline_weight_sum),
            "raw_wls": weighted_mse(raw_sse, raw_weight_sum),
        }
        if int(iters) >= 0:
            fgo_state, fgo_sse, fgo_weight_sum, _ = fit_state_with_clock_bias(
                batch.sat_ecef[start:end],
                batch.pseudorange[start:end],
                batch.weights[start:end],
                fgo_state[:, :3],
                sys_kind=(batch.sys_kind[start:end] if batch.sys_kind is not None else None),
                n_clock=batch.n_clock,
            )
            candidate_states["fgo"] = fgo_state
            candidate_mse["fgo"] = weighted_mse(fgo_sse, fgo_weight_sum)

        baseline_quality = _chunk_candidate_quality(
            candidate_states["baseline"],
            candidate_mse["baseline"],
            baseline_quality=None,
            prev_tail_xyz=prev_tail_xyz,
            baseline_xyz=candidate_states["baseline"][:, :3],
        )
        candidate_quality = {
            "baseline": baseline_quality,
        }
        for name, state in candidate_states.items():
            if name == "baseline":
                continue
            candidate_quality[name] = _chunk_candidate_quality(
                state,
                candidate_mse[name],
                baseline_quality=baseline_quality,
                prev_tail_xyz=prev_tail_xyz,
                baseline_xyz=candidate_states["baseline"][:, :3],
            )

        source = _select_auto_chunk_source(candidate_quality)
        chunk_records.append(
            ChunkSelectionRecord(
                start_epoch=start,
                end_epoch=end,
                auto_source=source,
                candidates=candidate_quality,
            ),
        )
        stitched[start:end] = candidate_states[source]
        fgo_stitched[start:end] = fgo_state
        selected_sources[start:end] = source
        selected_source_counts[source] += end - start
        prev_tail_xyz = stitched[end - 1, :3].copy()

    return ChunkedFgoRun(
        auto_state=stitched,
        fgo_state=fgo_stitched,
        total_iters=total_iters,
        failed_chunks=failed_chunks,
        vd_seed_guard_skipped_segments=vd_seed_guard_skipped_segments,
        vd_seed_guard_skipped_epochs=vd_seed_guard_skipped_epochs,
        vd_seed_guard_records=vd_seed_guard_records,
        auto_sources=selected_sources,
        auto_source_counts=selected_source_counts,
        chunk_records=chunk_records,
        fgo_vd_state=fgo_vd_stitched,
        failed_chunk_reasons=failed_chunk_reasons,
    )



def _dd_carrier_anchor_coverage(
    config: BridgeConfig,
    dd_carrier_stats: Mapping[str, object],
    *,
    n_epoch: int,
) -> float | None:
    """Per-trip DD-carrier anchor coverage thread for the gated selector.

    Returns ``None`` when the DD-carrier FGO is disabled so the chunk selector
    keeps the legacy behaviour.  When enabled, derives the ratio from the
    ``accepted_anchor_epochs`` stat captured during DD anchor application.
    """

    if not config.dd_carrier_fgo_enabled:
        return None
    accepted = int(dd_carrier_stats.get("accepted_anchor_epochs", 0) or 0)
    return _compute_dd_carrier_anchor_coverage_ratio(accepted, n_epoch)


def _dd_carrier_bridge_config_from_bridge_config(config: BridgeConfig) -> DDCarrierBridgeConfig:
    return DDCarrierBridgeConfig(
        tow_snap_tolerance_s=config.dd_carrier_tow_snap_tolerance_s,
        anchor=DDCarrierAnchorConfig(
            min_dd_pairs=config.dd_carrier_min_dd_pairs,
            sigma_cycles=config.dd_carrier_sigma_cycles,
            prior_sigma_m=config.dd_carrier_prior_sigma_m,
            max_shift_m=config.dd_carrier_max_shift_m,
            max_initial_rms_m=config.dd_carrier_max_initial_rms_m,
            max_final_rms_m=config.dd_carrier_max_final_rms_m,
        ),
        base_obs_template=config.dd_carrier_base_obs_template,
        require_base_obs_template=config.dd_carrier_require_base_obs_template,
        smooth_corrections=config.dd_carrier_smooth_corrections,
        anchor_correction_sigma_m=config.dd_carrier_anchor_correction_sigma_m,
        correction_smooth_sigma_m=config.dd_carrier_correction_smooth_sigma_m,
        correction_zero_sigma_m=config.dd_carrier_correction_zero_sigma_m,
    )


def _add_fixed_fgo_candidate_quality(
    records: list[ChunkSelectionRecord],
    *,
    source_name: str,
    candidate_state: np.ndarray,
    baseline_state: np.ndarray,
    auto_state: np.ndarray,
    batch: TripArrays,
) -> None:
    if not _is_fgo_candidate_source(source_name):
        raise ValueError(f"FGO candidate source must be 'fgo' or start with 'fgo_': {source_name}")
    for record in records:
        start = int(record.start_epoch)
        end = int(record.end_epoch)
        fitted, sse, weight_sum, _ = fit_state_with_clock_bias(
            batch.sat_ecef[start:end],
            batch.pseudorange[start:end],
            batch.weights[start:end],
            candidate_state[start:end, :3],
            sys_kind=(batch.sys_kind[start:end] if batch.sys_kind is not None else None),
            n_clock=batch.n_clock,
        )
        candidate_state[start:end] = fitted
        prev_tail_xyz = auto_state[start - 1, :3] if start > 0 else None
        record.candidates[source_name] = _chunk_candidate_quality(
            candidate_state[start:end],
            weighted_mse(sse, weight_sum),
            baseline_quality=record.candidates["baseline"],
            prev_tail_xyz=prev_tail_xyz,
            baseline_xyz=baseline_state[start:end, :3],
        )


def _apply_tdcp_candidate_weight_scale(
    batch: TripArrays,
    *,
    current_scale: float,
    candidate_scale: float,
) -> TripArrays:
    if batch.tdcp_weights is None:
        return batch
    current = float(current_scale)
    candidate = float(candidate_scale)
    if not np.isfinite(current) or current <= 0.0:
        raise ValueError("current TDCP weight scale must be finite and > 0")
    if not np.isfinite(candidate) or candidate <= 0.0:
        raise ValueError("candidate TDCP weight scale must be finite and > 0")
    scaled = np.asarray(batch.tdcp_weights, dtype=np.float64).copy()
    scaled *= candidate / current
    scaled_fgo = None
    if getattr(batch, "tdcp_weights_fgo", None) is not None:
        scaled_fgo = np.asarray(batch.tdcp_weights_fgo, dtype=np.float64).copy()
        scaled_fgo *= candidate / current
    return _replace_dataclass(batch, tdcp_weights=scaled, tdcp_weights_fgo=scaled_fgo)


def _build_taroz_fgo_candidate_batch(
    trip: str,
    batch: TripArrays,
    config: BridgeConfig,
    *,
    data_root: Path,
) -> TripArrays:
    candidate = build_trip_arrays(
        data_root / trip,
        max_epochs=int(batch.build_max_epochs) if int(batch.build_max_epochs) > 0 else int(batch.times_ms.size),
        start_epoch=int(batch.build_start_epoch),
        constellation_type=config.constellation_type,
        signal_type=config.signal_type,
        weight_mode=config.weight_mode,
        fgo_weight_mode=TAROZ_FGO_WEIGHT_MODE,
        multi_gnss=config.multi_gnss,
        use_tdcp=config.tdcp_enabled,
        tdcp_consistency_threshold_m=config.tdcp_consistency_threshold_m,
        tdcp_weight_scale=config.tdcp_weight_scale,
        tdcp_geometry_correction=config.tdcp_geometry_correction,
        apply_base_correction=config.apply_base_correction,
        data_root=data_root,
        trip=trip,
        apply_observation_mask=config.apply_observation_mask,
        observation_min_cn0_dbhz=config.observation_min_cn0_dbhz,
        observation_min_elevation_deg=config.observation_min_elevation_deg,
        pseudorange_residual_mask_m=config.pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=config.pseudorange_residual_mask_l5_m,
        doppler_residual_mask_mps=config.doppler_residual_mask_mps,
        pseudorange_doppler_mask_m=config.pseudorange_doppler_mask_m,
        matlab_residual_diagnostics_mask_path=config.matlab_residual_diagnostics_mask_path,
        dual_frequency=config.dual_frequency,
        apply_absolute_height=config.apply_absolute_height,
        absolute_height_dist_m=config.absolute_height_dist_m,
        imu_frame=config.imu_frame,
        imu_sample_dt_mode=config.imu_sample_dt_mode,
        factor_dt_max_s=config.factor_dt_max_s,
        use_rtklib_tropo=bool(getattr(config, "use_rtklib_tropo", False)),
    )
    if candidate.times_ms.shape != batch.times_ms.shape or not np.allclose(candidate.times_ms, batch.times_ms):
        raise RuntimeError("taroz FGO candidate batch window does not match the main batch")
    if candidate.weights.shape != batch.weights.shape:
        raise RuntimeError("taroz FGO candidate batch observation shape does not match the main batch")
    if tuple(candidate.slot_keys) != tuple(batch.slot_keys):
        raise RuntimeError("taroz FGO candidate batch satellite slots do not match the main batch")
    return _replace_dataclass(
        candidate,
        pseudorange=batch.pseudorange,
        pseudorange_observable=batch.pseudorange_observable,
        weights=batch.weights,
        pseudorange_bias_weights=batch.pseudorange_bias_weights,
        kaggle_wls=batch.kaggle_wls,
        truth=batch.truth,
        has_truth=batch.has_truth,
        build_start_epoch=batch.build_start_epoch,
        build_max_epochs=batch.build_max_epochs,
    )


def _taroz_fgo_candidate_run_kwargs(
    source_name: str,
    base_run_kwargs: Mapping[str, object],
    kernel: object,
) -> dict[str, object]:
    run_kwargs = dict(base_run_kwargs)
    run_kwargs["fgo_huber_k_pr"] = 0.0
    run_kwargs["fgo_huber_k_doppler"] = 0.0
    run_kwargs["fgo_huber_k_tdcp"] = 0.0
    run_kwargs["fgo_fixed_linearization"] = True
    if source_name == TAROZ_WEIGHTS_FGO_SOURCE:
        return run_kwargs
    if source_name == TAROZ_PR_FGO_SOURCE:
        run_kwargs["fgo_huber_k_pr"] = float(getattr(kernel, "pr_huber_k"))
        return run_kwargs
    if source_name == TAROZ_PR_D_L_FGO_SOURCE:
        run_kwargs["fgo_huber_k_pr"] = float(getattr(kernel, "pr_huber_k"))
        run_kwargs["fgo_huber_k_doppler"] = float(getattr(kernel, "doppler_huber_k"))
        run_kwargs["fgo_huber_k_tdcp"] = float(getattr(kernel, "carrier_huber_k"))
        return run_kwargs
    raise ValueError(f"unsupported taroz FGO candidate source: {source_name}")


def _effective_taroz_imu_noise_config(config: BridgeConfig, phone_name: str) -> BridgeConfig:
    if not getattr(config, "taroz_imu_noise_enabled", False):
        return config
    noise = _taroz_imu_noise_for_phone(phone_name)
    return _replace_dataclass(
        config,
        imu_position_sigma_m=float(noise.integration_sigma),
        imu_velocity_sigma_mps=float(noise.effective_acc_sigma_mps2_sqrt_hz),
        imu_attitude_sigma_rad=float(noise.effective_gyro_sigma_radps_sqrt_hz),
        imu_preintegration_velocity_noise_mps_sqrt_hz=float(noise.acc_sigma_mps2_sqrt_hz),
        imu_preintegration_attitude_noise_rad_sqrt_hz=float(noise.gyro_sigma_radps_sqrt_hz),
        # Taroz adds a ConstantBias prior with Inf sigmas, so the bias prior is
        # intentionally disabled; only the between-bias sigma uses this value.
        imu_accel_bias_prior_sigma_mps2=0.0,
        imu_accel_bias_between_sigma_mps2=float(noise.acc_bias_sigma_mps2),
        imu_gyro_bias_prior_sigma_radps=0.0,
        imu_acc_sigma_mps2_sqrt_hz=float(noise.acc_sigma_mps2_sqrt_hz),
        imu_gyro_sigma_radps_sqrt_hz=float(noise.gyro_sigma_radps_sqrt_hz),
        imu_acc_sync_coefficient=float(noise.acc_sync_coefficient),
        imu_gyro_sync_coefficient=float(noise.gyro_sync_coefficient),
        imu_gyro_bias_between_sigma_radps=float(noise.gyro_bias_sigma_radps),
    )


def solve_trip(
    trip: str,
    batch: TripArrays,
    config: BridgeConfig,
    *,
    data_root: Path | None = None,
) -> BridgeResult:
    phone_name = Path(trip).name
    config = _effective_taroz_imu_noise_config(config, phone_name)
    if getattr(config, "hatch_smoothing_enabled", False):
        from experiments.gsdc2023_hatch_smoothing import apply_hatch_smoothing
        from dataclasses import replace as _dc_replace
        smoothed_pr, hatch_stats = apply_hatch_smoothing(
            batch.pseudorange,
            getattr(batch, "adr", None),
            getattr(batch, "adr_state", None),
            smoothing_n=int(getattr(config, "hatch_smoothing_n", 100)),
        )
        batch = _dc_replace(batch, pseudorange=smoothed_pr)
        print(
            f"[hatch] arcs={hatch_stats.arcs_total} obs_smoothed={hatch_stats.obs_smoothed}/"
            f"{hatch_stats.obs_total} mean_arc_len={hatch_stats.mean_arc_length:.1f} "
            f"mean_n={hatch_stats.mean_smoothing_n:.1f}",
            flush=True,
        )
    kaggle_state, kaggle_sse, kaggle_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        batch.kaggle_wls,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    solver_context = _build_solver_execution_context(phone_name, batch, kaggle_state)
    solver_context_kwargs = solver_context.run_kwargs()
    if config.clock_use_average_drift is not None:
        solver_context_kwargs["clock_use_average_drift"] = bool(config.clock_use_average_drift)
    raw_wls = run_wls(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
        fallback_xyz=batch.kaggle_wls,
    )
    raw_wls[:, :3] = _repair_baseline_wls(batch.times_ms, raw_wls[:, :3])
    fgo_run_options = _fgo_run_options_from_config(config)
    fgo_run_kwargs = fgo_run_options.run_kwargs()
    fgo_run_kwargs["vd_seed_factor_guard"] = _vd_seed_factor_guard_enabled_for_phone(phone_name)
    taroz_candidate_sources = _taroz_fgo_candidate_sources_enabled(config)
    taroz_candidate_base_run_kwargs = dict(fgo_run_kwargs)
    effective_trip_type: str | None = None
    if getattr(config, "per_type_kernel_enabled", False) and data_root is not None:
        trip_type = _trip_type_from_data_root(data_root, trip)
        effective_trip_type = trip_type
        kernel = _per_type_kernel_for(trip_type, phone=phone_name)
        if getattr(config, "per_type_kernel_huber_enabled", True):
            fgo_run_kwargs["fgo_huber_k_pr"] = float(kernel.pr_huber_k)
            fgo_run_kwargs["fgo_huber_k_doppler"] = float(kernel.doppler_huber_k)
            fgo_run_kwargs["fgo_huber_k_tdcp"] = float(kernel.carrier_huber_k)
        if getattr(config, "per_type_kernel_motion_enabled", False):
            fgo_run_kwargs["motion_sigma_m"] = float(kernel.motion_sigma_m)
    if getattr(config, "pairwise_consistency_enabled", False):
        from experiments.gsdc2023_pairwise_consistency import (
            apply_pairwise_consistency_pre_filter,
        )
        source_w = batch.weights_fgo if batch.weights_fgo is not None else batch.weights
        filtered_w, pairwise_stats = apply_pairwise_consistency_pre_filter(
            batch.sat_ecef,
            batch.pseudorange,
            source_w,
            reference_xyz=batch.kaggle_wls,
            sys_kind=batch.sys_kind,
            n_clock=batch.n_clock,
            mad_threshold_m=float(getattr(config, "pairwise_consistency_mad_threshold_m", 3.5)),
            min_obs_after_filter=int(getattr(config, "pairwise_consistency_min_obs_after_filter", 5)),
        )
        # Replace only the FGO weights (gate/WLS keeps original).  Use
        # dataclass.replace via a small shim because TripArrays is frozen.
        from dataclasses import replace as _dc_replace
        batch = _dc_replace(batch, weights_fgo=filtered_w)
        print(
            f"[pairwise-filter] epochs={pairwise_stats.epochs_filtered}/{pairwise_stats.epochs_total} "
            f"obs_masked={pairwise_stats.obs_masked}/{pairwise_stats.obs_before}",
            flush=True,
        )
    if getattr(config, "max_clique_filter_enabled", False):
        from experiments.gsdc2023_max_clique_filter import (
            apply_max_clique_consensus_filter,
        )
        source_w = batch.weights_fgo if batch.weights_fgo is not None else batch.weights
        filtered_w, clique_stats = apply_max_clique_consensus_filter(
            batch.sat_ecef,
            batch.pseudorange,
            source_w,
            reference_xyz=batch.kaggle_wls,
            sys_kind=batch.sys_kind,
            n_clock=batch.n_clock,
            pair_threshold_m=float(getattr(config, "max_clique_filter_pair_threshold_m", 3.0)),
            min_clique_size=int(getattr(config, "max_clique_filter_min_clique_size", 5)),
        )
        from dataclasses import replace as _dc_replace
        batch = _dc_replace(batch, weights_fgo=filtered_w)
        print(
            f"[max-clique-filter] epochs={clique_stats.epochs_filtered}/{clique_stats.epochs_total} "
            f"obs_masked={clique_stats.obs_masked}/{clique_stats.obs_before} "
            f"mean_clique_frac={clique_stats.mean_clique_fraction:.3f}",
            flush=True,
        )
    if getattr(config, "taroz_factor_mask_csv", None) is not None:
        factor_trip_dir = Path(data_root) / trip if data_root is not None else None
        recompute_full_fixed_values = False
        factor_rebase_state_csv = (
            Path(config.taroz_fgo_seed_state_csv)
            if recompute_full_fixed_values
            else None
        )
        batch = _apply_taroz_factor_mask_to_batch(
            batch,
            Path(config.taroz_factor_mask_csv),
            trip_dir=factor_trip_dir,
            rebase_state_csv=factor_rebase_state_csv,
            use_fixed_values=not recompute_full_fixed_values,
        )
    if getattr(config, "taroz_imu_preintegration_csv", None) is not None:
        native_preintegration = getattr(batch, "imu_preintegration", None)
        taroz_preintegration = _load_taroz_imu_preintegration_csv(
            Path(config.taroz_imu_preintegration_csv),
            epoch_times_ms=batch.times_ms,
        )
        taroz_preintegration, has_native_solver_gravity = _taroz_preintegration_with_native_solver_gravity(
            taroz_preintegration,
            native_preintegration,
        )
        dt = np.asarray(taroz_preintegration.delta_t_s, dtype=np.float64)
        sample_count = np.asarray(taroz_preintegration.sample_count, dtype=np.int32)
        valid = np.isfinite(dt) & (dt > 0.0) & (sample_count > 0)
        has_complete_bias_jacobians = _imu_preintegration_has_complete_bias_jacobians(
            taroz_preintegration,
            0,
            dt.size,
            valid,
        )
        if has_complete_bias_jacobians and has_native_solver_gravity:
            batch = _replace_dataclass(batch, imu_preintegration=taroz_preintegration)
        elif not has_complete_bias_jacobians:
            print(
                "[taroz-imu-preintegration] ignored: exported bias Jacobians are incomplete",
                flush=True,
            )
        else:
            print(
                "[taroz-imu-preintegration] ignored: native ECEF gravity is unavailable",
                flush=True,
            )
    if getattr(config, "taroz_imu_factor_mask_csv", None) is not None:
        batch = _apply_taroz_imu_factor_mask_to_batch(
            batch,
            Path(config.taroz_imu_factor_mask_csv),
        )
    taroz_height_stop_mask = None
    taroz_base_stop_mask = None
    if getattr(config, "taroz_stop_mask_from_seed_velocity", False):
        current_stop_mask = solver_context_kwargs.get("stop_mask")
        taroz_base_stop_mask = (
            np.asarray(batch.stop_epochs, dtype=bool).reshape(-1).copy()
            if batch.stop_epochs is not None
            else current_stop_mask
        )
        taroz_height_stop_mask = (
            np.asarray(current_stop_mask, dtype=bool).copy() if current_stop_mask is not None else None
        )
    fgo_seed_state = None
    if getattr(config, "taroz_fgo_seed_state_csv", None) is not None:
        if data_root is None:
            raise RuntimeError("taroz FGO seed state CSV requires data_root")
        fgo_seed_state = _load_taroz_fgo_seed_state(
            Path(config.taroz_fgo_seed_state_csv),
            batch,
            trip_dir=Path(data_root) / trip,
            prefer_graph_state=_taroz_fgo_seed_prefer_graph_state(config),
            pose_bias_path=(
                Path(config.taroz_pose_bias_seed_state_csv)
                if getattr(config, "taroz_pose_bias_seed_state_csv", None) is not None
                else None
            ),
        )
    if getattr(config, "taroz_stop_mask_from_seed_velocity", False):
        if taroz_height_stop_mask is not None:
            fgo_run_kwargs["relative_height_stop_mask"] = taroz_height_stop_mask
        before_stop_count = (
            int(np.count_nonzero(taroz_base_stop_mask))
            if taroz_base_stop_mask is not None
            else 0
        )
        solver_context_kwargs["stop_mask"] = _taroz_stop_mask_from_seed_velocity(
            taroz_base_stop_mask,
            fgo_seed_state,
        )
        after_stop_count = (
            int(np.count_nonzero(solver_context_kwargs["stop_mask"]))
            if solver_context_kwargs.get("stop_mask") is not None
            else 0
        )
        seed_status = "applied" if fgo_seed_state is not None else "skipped:no_seed"
        print(
            f"[taroz-stop-mask] seed_velocity_threshold={TAROZ_STOP_VELOCITY_THRESHOLD_MPS:.3f}m/s "
            f"kept={after_stop_count}/{before_stop_count} status={seed_status}",
            flush=True,
        )
    chunked_run = run_fgo_chunked(
        batch,
        raw_wls,
        **solver_context_kwargs,
        **fgo_run_kwargs,
        fgo_seed_state=fgo_seed_state,
    )
    auto_state = chunked_run.auto_state
    fgo_state = chunked_run.fgo_state
    iters = chunked_run.total_iters
    failed_chunks = chunked_run.failed_chunks
    failed_chunk_reasons = chunked_run.failed_chunk_reasons
    vd_seed_guard_skipped_segments = chunked_run.vd_seed_guard_skipped_segments
    vd_seed_guard_skipped_epochs = chunked_run.vd_seed_guard_skipped_epochs
    vd_seed_guard_records = chunked_run.vd_seed_guard_records
    auto_sources = chunked_run.auto_sources
    auto_source_counts = chunked_run.auto_source_counts
    chunk_records = chunked_run.chunk_records
    fgo_vd_state = chunked_run.fgo_vd_state
    tdcp_off_fgo_state: np.ndarray | None = None
    tdcp_off_chunk_records: list[ChunkSelectionRecord] | None = None
    tdcp_scale_fgo_state: np.ndarray | None = None
    tdcp_scale_chunk_records: list[ChunkSelectionRecord] | None = None
    ct_rbpf_fgo_state: np.ndarray | None = None
    ct_rbpf_chunk_records: list[ChunkSelectionRecord] | None = None
    dd_carrier_fgo_state: np.ndarray | None = None
    dd_carrier_stats: dict[str, object] = {}
    taroz_fgo_candidates: dict[str, tuple[np.ndarray, list[ChunkSelectionRecord]]] = {}
    if _tdcp_off_candidate_enabled(config, batch):
        tdcp_off_run = run_fgo_chunked(
            _batch_without_tdcp(batch),
            raw_wls,
            **solver_context_kwargs,
            **fgo_run_kwargs,
            fgo_seed_state=fgo_seed_state,
        )
        tdcp_off_fgo_state = tdcp_off_run.fgo_state
        tdcp_off_chunk_records = tdcp_off_run.chunk_records
    if _tdcp_scale_candidate_enabled(config, batch, phone_name):
        tdcp_scale_batch = _apply_tdcp_candidate_weight_scale(
            batch,
            current_scale=config.tdcp_weight_scale,
            candidate_scale=config.tdcp_scale_candidate_weight_scale,
        )
        tdcp_scale_run = run_fgo_chunked(
            tdcp_scale_batch,
            raw_wls,
            **solver_context_kwargs,
            **fgo_run_kwargs,
            fgo_seed_state=fgo_seed_state,
        )
        tdcp_scale_fgo_state = tdcp_scale_run.fgo_state
        tdcp_scale_chunk_records = tdcp_scale_run.chunk_records
    if config.ct_rbpf_fgo_enabled:
        ct_rbpf_fgo_run_kwargs = dict(fgo_run_kwargs)
        ct_rbpf_fgo_run_kwargs["motion_sigma_m"] = config.ct_rbpf_motion_sigma_m
        ct_rbpf_run = run_fgo_chunked(
            batch,
            raw_wls,
            **solver_context_kwargs,
            **ct_rbpf_fgo_run_kwargs,
            fgo_seed_state=fgo_seed_state,
        )
        ct_rbpf_fgo_state = ct_rbpf_run.fgo_state
        ct_rbpf_chunk_records = ct_rbpf_run.chunk_records
    if taroz_candidate_sources:
        if data_root is None:
            raise RuntimeError("taroz FGO candidates require data_root")
        taroz_candidate_batch = _build_taroz_fgo_candidate_batch(
            trip,
            batch,
            config,
            data_root=data_root,
        )
        taroz_trip_type = _trip_type_from_data_root(data_root, trip)
        taroz_kernel = _per_type_kernel_for(taroz_trip_type, phone=phone_name)
        for source_name in taroz_candidate_sources:
            taroz_candidate_run_kwargs = _taroz_fgo_candidate_run_kwargs(
                source_name,
                taroz_candidate_base_run_kwargs,
                taroz_kernel,
            )
            taroz_run = run_fgo_chunked(
                taroz_candidate_batch,
                raw_wls,
                **solver_context_kwargs,
                **taroz_candidate_run_kwargs,
                fgo_seed_state=fgo_seed_state,
            )
            taroz_fgo_candidates[str(source_name)] = (taroz_run.fgo_state, taroz_run.chunk_records)
    raw_state, raw_sse, raw_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        raw_wls[:, :3],
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    fgo_state, fgo_sse, fgo_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        fgo_state[:, :3],
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    auto_state, auto_sse, auto_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        auto_state[:, :3],
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    baseline_mse_pr = weighted_mse(kaggle_sse, kaggle_weight_sum)
    raw_wls_mse_pr = weighted_mse(raw_sse, raw_weight_sum)
    fgo_mse_pr = weighted_mse(fgo_sse, fgo_weight_sum)
    auto_mse_pr = weighted_mse(auto_sse, auto_weight_sum)

    source_catalog = _build_source_solution_catalog(
        n_epoch=batch.times_ms.size,
        baseline_state=kaggle_state,
        raw_state=raw_state,
        fgo_state=fgo_state,
        auto_state=auto_state,
        auto_sources=auto_sources,
        auto_source_counts=auto_source_counts,
        baseline_mse_pr=baseline_mse_pr,
        raw_wls_mse_pr=raw_wls_mse_pr,
        fgo_mse_pr=fgo_mse_pr,
        auto_mse_pr=auto_mse_pr,
    )
    if tdcp_off_fgo_state is not None and tdcp_off_chunk_records is not None:
        tdcp_off_fgo_state, tdcp_off_sse, tdcp_off_weight_sum, _ = fit_state_with_clock_bias(
            batch.sat_ecef,
            batch.pseudorange,
            batch.weights,
            tdcp_off_fgo_state[:, :3],
            sys_kind=batch.sys_kind,
            n_clock=batch.n_clock,
        )
        source_catalog = _with_fixed_source_solution(
            source_catalog,
            source="fgo_no_tdcp",
            state=tdcp_off_fgo_state,
            mse_pr=weighted_mse(tdcp_off_sse, tdcp_off_weight_sum),
        )
        _add_tdcp_off_fgo_candidates(
            chunk_records,
            tdcp_off_chunk_records,
            tdcp_off_fgo_state,
            source_catalog.states["baseline"],
            source_catalog.states["auto"],
        )
    if tdcp_scale_fgo_state is not None and tdcp_scale_chunk_records is not None:
        tdcp_scale_fgo_state, tdcp_scale_sse, tdcp_scale_weight_sum, _ = fit_state_with_clock_bias(
            batch.sat_ecef,
            batch.pseudorange,
            batch.weights,
            tdcp_scale_fgo_state[:, :3],
            sys_kind=batch.sys_kind,
            n_clock=batch.n_clock,
        )
        source_catalog = _with_fixed_source_solution(
            source_catalog,
            source=TDCP_SCALE_FGO_SOURCE,
            state=tdcp_scale_fgo_state,
            mse_pr=weighted_mse(tdcp_scale_sse, tdcp_scale_weight_sum),
        )
        _add_fgo_candidate_from_records(
            chunk_records,
            tdcp_scale_chunk_records,
            source_name=TDCP_SCALE_FGO_SOURCE,
            candidate_state=tdcp_scale_fgo_state,
            baseline_state=source_catalog.states["baseline"],
            auto_state=source_catalog.states["auto"],
        )
    if ct_rbpf_fgo_state is not None and ct_rbpf_chunk_records is not None:
        ct_rbpf_fgo_state, ct_rbpf_sse, ct_rbpf_weight_sum, _ = fit_state_with_clock_bias(
            batch.sat_ecef,
            batch.pseudorange,
            batch.weights,
            ct_rbpf_fgo_state[:, :3],
            sys_kind=batch.sys_kind,
            n_clock=batch.n_clock,
        )
        source_catalog = _with_fixed_source_solution(
            source_catalog,
            source=CT_RBPF_FGO_SOURCE,
            state=ct_rbpf_fgo_state,
            mse_pr=weighted_mse(ct_rbpf_sse, ct_rbpf_weight_sum),
        )
        _add_fgo_candidate_from_records(
            chunk_records,
            ct_rbpf_chunk_records,
            source_name=CT_RBPF_FGO_SOURCE,
            candidate_state=ct_rbpf_fgo_state,
            baseline_state=source_catalog.states["baseline"],
            auto_state=source_catalog.states["auto"],
        )
    for source_name, (taroz_fgo_state, taroz_chunk_records) in taroz_fgo_candidates.items():
        taroz_fgo_state, taroz_sse, taroz_weight_sum, _ = fit_state_with_clock_bias(
            batch.sat_ecef,
            batch.pseudorange,
            batch.weights,
            taroz_fgo_state[:, :3],
            sys_kind=batch.sys_kind,
            n_clock=batch.n_clock,
        )
        source_catalog = _with_fixed_source_solution(
            source_catalog,
            source=source_name,
            state=taroz_fgo_state,
            mse_pr=weighted_mse(taroz_sse, taroz_weight_sum),
        )
        _add_fgo_candidate_from_records(
            chunk_records,
            taroz_chunk_records,
            source_name=source_name,
            candidate_state=taroz_fgo_state,
            baseline_state=source_catalog.states["baseline"],
            auto_state=source_catalog.states["auto"],
        )
    if config.dd_carrier_fgo_enabled:
        if data_root is None:
            if config.position_source == DD_CARRIER_FGO_SOURCE:
                raise RuntimeError(f"{DD_CARRIER_FGO_SOURCE} requires data_root")
        else:
            try:
                dd_carrier_fgo_state, dd_carrier_stats = _apply_sparse_dd_carrier_anchors(
                    data_root,
                    trip,
                    batch,
                    fgo_state,
                    _dd_carrier_bridge_config_from_bridge_config(config),
                )
            except FileNotFoundError:
                if config.dd_carrier_require_base_obs_template or config.position_source == DD_CARRIER_FGO_SOURCE:
                    raise
                dd_carrier_fgo_state = fgo_state.copy()
                dd_carrier_stats = {
                    "base_snapped_epochs": 0,
                    "dd_epochs": 0,
                    "accepted_anchor_epochs": 0,
                    "dd_pairs_mean": 0.0,
                }
            if dd_carrier_fgo_state is not None:
                dd_carrier_fgo_state, dd_sse, dd_weight_sum, _ = fit_state_with_clock_bias(
                    batch.sat_ecef,
                    batch.pseudorange,
                    batch.weights,
                    dd_carrier_fgo_state[:, :3],
                    sys_kind=batch.sys_kind,
                    n_clock=batch.n_clock,
                )
                source_catalog = _with_fixed_source_solution(
                    source_catalog,
                    source=DD_CARRIER_FGO_SOURCE,
                    state=dd_carrier_fgo_state,
                    mse_pr=weighted_mse(dd_sse, dd_weight_sum),
                )
                _add_fixed_fgo_candidate_quality(
                    chunk_records,
                    source_name=DD_CARRIER_FGO_SOURCE,
                    candidate_state=dd_carrier_fgo_state,
                    baseline_state=source_catalog.states["baseline"],
                    auto_state=source_catalog.states["auto"],
                    batch=batch,
                )

    allow_mi8_raw_wls_jump = _mi8_gated_baseline_jump_guard_enabled(phone_name, config.position_source)
    raw_wls_max_gap_m = _raw_wls_max_gap_guard_m(phone_name, config.position_source)
    allow_fgo_raw_wls_proxy_rescue = _fgo_raw_wls_proxy_rescue_enabled(config, phone_name)
    dd_carrier_anchor_coverage = _dd_carrier_anchor_coverage(
        config, dd_carrier_stats, n_epoch=int(batch.times_ms.size)
    )
    gated_state, gated_sources, gated_counts = _select_gated_solution(
        source_catalog,
        chunk_records,
        n_epoch=batch.times_ms.size,
        baseline_threshold=config.gated_baseline_threshold,
        allow_raw_wls_on_mi8_baseline_jump=allow_mi8_raw_wls_jump,
        raw_wls_max_gap_m=raw_wls_max_gap_m,
        allow_fgo_raw_wls_proxy_rescue=allow_fgo_raw_wls_proxy_rescue,
        fgo_raw_wls_proxy_rescue_mse_ratio_max=config.fgo_raw_wls_proxy_rescue_mse_ratio_max,
        fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max=config.fgo_raw_wls_proxy_rescue_gap_step_p95_ratio_max,
        fgo_raw_wls_proxy_rescue_quality_delta_max=config.fgo_raw_wls_proxy_rescue_quality_delta_max,
        fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max=config.fgo_raw_wls_proxy_rescue_mse_delta_vs_baseline_max,
        dd_carrier_anchor_coverage=dd_carrier_anchor_coverage,
        dd_carrier_min_anchor_coverage=config.dd_carrier_min_anchor_coverage,
        fgo_low_baseline_mse_pr_max=getattr(config, "gate_fgo_low_baseline_mse_pr_max", None),
        fgo_baseline_mse_pr_min=getattr(config, "gate_fgo_baseline_mse_pr_min", None),
        fgo_baseline_gap_p95_floor_m=getattr(config, "gate_fgo_baseline_gap_p95_floor_m", None),
    )
    gated_state, gated_sse, gated_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        gated_state[:, :3],
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
    )
    source_catalog = _with_source_solution(
        source_catalog,
        source="gated",
        state=gated_state,
        source_array=gated_sources,
        source_counts=gated_counts,
        mse_pr=weighted_mse(gated_sse, gated_weight_sum),
    )

    assembled_outputs = _assemble_source_outputs(
        source_catalog,
        batch,
        config,
        phone_name=phone_name,
    )

    result = _build_bridge_result(
        trip=trip,
        batch=batch,
        config=config,
        assembled_outputs=assembled_outputs,
        fgo_iters=int(iters),
        failed_chunks=int(failed_chunks),
        failed_chunk_reasons=failed_chunk_reasons,
        vd_seed_guard_skipped_segments=int(vd_seed_guard_skipped_segments),
        vd_seed_guard_skipped_epochs=int(vd_seed_guard_skipped_epochs),
        vd_seed_guard_records=vd_seed_guard_records,
        baseline_mse_pr=baseline_mse_pr,
        raw_wls_mse_pr=raw_wls_mse_pr,
        fgo_mse_pr=fgo_mse_pr,
        chunk_records=chunk_records,
        allow_raw_wls_on_mi8_baseline_jump=allow_mi8_raw_wls_jump,
        raw_wls_max_gap_m=raw_wls_max_gap_m,
        allow_fgo_raw_wls_proxy_rescue=allow_fgo_raw_wls_proxy_rescue,
        fgo_vd_state=fgo_vd_state,
    )
    result.effective_trip_type = effective_trip_type
    result.effective_motion_sigma_m = float(fgo_run_kwargs.get("motion_sigma_m", config.motion_sigma_m))
    result.effective_fgo_huber_k_pr = float(fgo_run_kwargs.get("fgo_huber_k_pr", config.fgo_huber_k_pr))
    result.effective_fgo_huber_k_doppler = float(
        fgo_run_kwargs.get("fgo_huber_k_doppler", config.fgo_huber_k_doppler),
    )
    result.effective_fgo_huber_k_tdcp = float(fgo_run_kwargs.get("fgo_huber_k_tdcp", config.fgo_huber_k_tdcp))
    if config.dd_carrier_fgo_enabled:
        result.dd_carrier_accepted_anchor_epochs = int(dd_carrier_stats.get("accepted_anchor_epochs", 0) or 0)
        result.dd_carrier_dd_epochs = int(dd_carrier_stats.get("dd_epochs", 0) or 0)
        result.dd_carrier_base_snapped_epochs = int(dd_carrier_stats.get("base_snapped_epochs", 0) or 0)
        result.dd_carrier_dd_pairs_mean = float(dd_carrier_stats.get("dd_pairs_mean", 0.0) or 0.0)
    return result


def validate_raw_gsdc2023_trip(
    data_root: Path,
    trip: str,
    *,
    max_epochs: int = 200,
    start_epoch: int = 0,
    config: BridgeConfig | None = None,
) -> BridgeResult:
    context = _build_raw_trip_validation_context(
        data_root,
        trip,
        config,
        parity_audit_fn=collect_matlab_parity_audit,
    )
    cfg = context.config
    trip_dir = context.trip_dir
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")
    batch = build_trip_arrays(
        trip_dir,
        max_epochs=_max_epochs_for_build(max_epochs),
        start_epoch=start_epoch,
        constellation_type=cfg.constellation_type,
        signal_type=cfg.signal_type,
        weight_mode=cfg.weight_mode,
        fgo_weight_mode=cfg.fgo_weight_mode,
        multi_gnss=cfg.multi_gnss,
        use_tdcp=cfg.tdcp_enabled,
        tdcp_consistency_threshold_m=cfg.tdcp_consistency_threshold_m,
        tdcp_weight_scale=cfg.tdcp_weight_scale,
        tdcp_geometry_correction=cfg.tdcp_geometry_correction,
        apply_base_correction=cfg.apply_base_correction,
        data_root=data_root,
        trip=trip,
        apply_observation_mask=cfg.apply_observation_mask,
        observation_min_cn0_dbhz=cfg.observation_min_cn0_dbhz,
        observation_min_elevation_deg=cfg.observation_min_elevation_deg,
        pseudorange_residual_mask_m=cfg.pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=cfg.pseudorange_residual_mask_l5_m,
        doppler_residual_mask_mps=cfg.doppler_residual_mask_mps,
        pseudorange_doppler_mask_m=cfg.pseudorange_doppler_mask_m,
        matlab_residual_diagnostics_mask_path=cfg.matlab_residual_diagnostics_mask_path,
        dual_frequency=cfg.dual_frequency,
        apply_absolute_height=cfg.apply_absolute_height,
        absolute_height_dist_m=cfg.absolute_height_dist_m,
        imu_frame=cfg.imu_frame,
        imu_sample_dt_mode=cfg.imu_sample_dt_mode,
        factor_dt_max_s=cfg.factor_dt_max_s,
        use_rtklib_tropo=bool(getattr(cfg, "use_rtklib_tropo", False)),
    )
    result = solve_trip(trip, batch, cfg, data_root=data_root)
    result.parity_audit = context.parity_audit
    refined_cfg = _outlier_refinement_config(cfg, result.selected_mse_pr)
    if refined_cfg is not None:
        refined = solve_trip(trip, batch, refined_cfg, data_root=data_root)
        refined.parity_audit = context.parity_audit
        if refined.selected_mse_pr < result.selected_mse_pr:
            return refined
    return result



_build_trip_arrays = build_trip_arrays
_export_bridge_outputs = export_bridge_outputs
_fit_state_with_clock_bias = fit_state_with_clock_bias

__all__ = [
    "AbsoluteHeightStageProducts",
    "BridgeConfig",
    "BridgeResult",
    "AssembledSourceOutputs",
    "ClockResidualStageProducts",
    "DEFAULT_ROOT",
    "DopplerResidualStageProducts",
    "EpochMetadataContext",
    "EpochTimeContext",
    "FACTOR_DT_MAX_S",
    "FgoRunOptions",
    "FilledObservationMatrixProducts",
    "FilledObservationPostprocessProducts",
    "FullObservationContextProducts",
    "GnssLogPseudorangeStageProducts",
    "GraphTimeDeltaProducts",
    "ImuResultSummary",
    "ImuStageProducts",
    "ObservationMatrixInputProducts",
    "ObservationMaskBaseCorrectionStageProducts",
    "ObservationPreparationStageProducts",
    "PostObservationStageProducts",
    "PostObservationStageConfig",
    "PostObservationStageDependencies",
    "PreparedObservationProducts",
    "PseudorangeDopplerStageProducts",
    "PseudorangeResidualStageProducts",
    "SolverExecutionContext",
    "DEFAULT_MOTION_SIGMA_M",
    "DEFAULT_CT_RBPF_MOTION_SIGMA_M",
    "DEFAULT_TDCP_GEOMETRY_CORRECTION",
    "DEFAULT_TDCP_WEIGHT_SCALE",
    "IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2",
    "IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2",
    "resolve_gsdc2023_data_root",
    "GATED_BASELINE_THRESHOLD_DEFAULT",
    "CT_RBPF_FGO_SOURCE",
    "POSITION_SOURCES",
    "RAW_GNSS_COLUMNS",
    "IMUMeasurements",
    "IMUPreintegration",
    "ProcessedIMU",
    "RawTripValidationContext",
    "RawObservationFrameProducts",
    "TdcpStageProducts",
    "TripArrays",
    "bridge_position_columns",
    "build_trip_arrays",
    "collect_matlab_parity_audit",
    "compute_base_pseudorange_correction_matrix",
    "ecef_to_llh_deg",
    "export_bridge_outputs",
    "fit_state_with_clock_bias",
    "format_metrics_line",
    "has_valid_bridge_outputs",
    "load_absolute_height_reference_ecef",
    "load_bridge_metrics",
    "metrics_summary",
    "preintegrate_processed_imu",
    "enu_up_ecef_from_origin",
    "relative_height_star_edges_from_groups",
    "relative_height_star_edges_for_reference",
    "run_fgo_chunked",
    "run_wls",
    "score_from_metrics",
    "solve_trip",
    "validate_position_source",
    "validate_raw_gsdc2023_trip",
    "weighted_mse",
]
