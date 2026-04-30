from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluate import ecef_to_lla, lla_to_ecef
from gnss_gpu import wls_position
from gnss_gpu.fgo import fgo_gnss_lm, fgo_gnss_lm_vd
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
    add_tdcp_off_fgo_candidates as _add_tdcp_off_fgo_candidates,
    candidate_passes_gated_quality as _candidate_passes_gated_quality,
    catastrophic_baseline_alternative as _catastrophic_baseline_alternative,
    chunk_candidate_quality as _chunk_candidate_quality,
    chunk_quality_payload as _chunk_quality_payload,
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
    DEFAULT_MOTION_SIGMA_M,
    FACTOR_DT_MAX_S,
    OUTLIER_REFINEMENT_CHUNK_EPOCHS,
    OUTLIER_REFINEMENT_MSE_PR_THRESHOLD,
    should_refine_outlier_result as _should_refine_outlier_result,
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
    imu_preintegration_segment as _imu_preintegration_segment,
    interp_vectors as _interp_vectors,
    load_device_imu_measurements as _load_device_imu_measurements_impl,
    preintegrate_processed_imu,
    process_device_imu,
    project_stop_to_epochs,
    read_device_imu_frame as _read_device_imu_frame_impl,
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
)
from experiments.gsdc2023_result_assembly import (
    AssembledSourceOutputs,
    assemble_source_outputs as _assemble_source_outputs,
    build_bridge_result as _build_bridge_result,
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
    mi8_gated_baseline_jump_guard_enabled as _mi8_gated_baseline_jump_guard_enabled,
    raw_wls_max_gap_guard_m as _raw_wls_max_gap_guard_m,
    select_gated_solution as _select_gated_solution,
    tdcp_off_candidate_enabled as _tdcp_off_candidate_enabled,
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
    matlab_signal_observation_masks as _matlab_signal_observation_masks,
    read_raw_gnss_csv as _read_raw_gnss_csv,
    receiver_clock_bias_from_nanos as _receiver_clock_bias_from_nanos,
    receiver_clock_bias_lookup_from_epoch_meta as _receiver_clock_bias_lookup_from_epoch_meta,
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
    POSITION_SOURCES,
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
    split, course, phone = _trip_course_phone(trip)
    trip_dir = data_root / trip
    course_dir = trip_dir.parent if course is not None else trip_dir.parent
    base_dir = _base_metadata_dir(data_root)
    settings = _load_settings_frame_cached(str(data_root), split) if split is not None else None
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
            expected_base_obs = data_root / split / course / f"{base_name}_{suffix}.obs"

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


def _gnss_log_corrected_pseudorange_matrix(
    trip_dir: Path,
    raw_frame: pd.DataFrame,
    times_ms: np.ndarray,
    slot_keys: tuple[tuple[int, int, str], ...],
    gps_tgd_m_by_svid: dict[int, float],
    rtklib_tropo_m: np.ndarray | None = None,
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
    factor_dt_max_s: float = FACTOR_DT_MAX_S,
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
    raw_df = _load_raw_gnss_frame(raw_path)
    epoch_meta = _build_epoch_metadata_frame(raw_df)
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
        gnss_log_matlab_epoch_times_ms_fn=_gnss_log_matlab_epoch_times_ms,
        clean_clock_drift_fn=_clean_clock_drift,
        select_epoch_observations_fn=_select_epoch_observations,
        fill_observation_matrices_fn=_fill_observation_matrices,
        nearest_index_fn=nearest_index,
        gps_arrival_tow_s_from_row_fn=_gps_arrival_tow_s_from_row,
        gps_sat_clock_bias_adjustment_m_fn=_gps_sat_clock_bias_adjustment_m,
        gps_matrtklib_sat_product_adjustment_fn=_gps_matrtklib_sat_product_adjustment,
        clock_kind_for_observation_fn=_clock_kind_for_observation,
        is_l5_signal_fn=_is_l5_signal,
        slot_sort_key_fn=_slot_sort_key,
        ecef_to_lla_fn=ecef_to_lla,
        elevation_azimuth_fn=_elevation_azimuth,
        rtklib_tropo_fn=_rtklib_tropo_saastamoinen,
        matlab_signal_clock_dim=MATLAB_SIGNAL_CLOCK_DIM,
        recompute_rtklib_tropo_matrix_fn=_recompute_rtklib_tropo_matrix,
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
    return _assemble_prepared_trip_arrays_stage(
        trip_arrays_cls=TripArrays,
        observation_products=observation_products,
        post_observation_stages=post_observation_stages,
        has_truth=(gt_times is not None and gt_ecef is not None),
        dual_frequency=dual_frequency,
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
    imu_accel_bias_state: bool = False,
) -> np.ndarray:
    n_epoch = raw_state.shape[0]
    state_width = 7 + n_clock + (3 if imu_accel_bias_state else 0)
    seed = np.zeros((n_epoch, state_width), dtype=np.float64)
    seed[:, 6 + n_clock] = np.nan

    raw_pos = raw_state[:, :3].copy()
    invalid = np.linalg.norm(raw_pos, axis=1) < 1e3
    raw_pos[invalid] = baseline_state[invalid, :3]
    seed[:, :3] = raw_pos

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

    return seed


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
    graph_relative_height: bool = False,
    relative_height_sigma_m: float = 0.5,
    apply_absolute_height: bool = False,
    absolute_height_sigma_m: float = HEIGHT_ABSOLUTE_SIGMA_M,
    imu_accel_bias_state: bool = False,
    imu_accel_bias_prior_sigma_mps2: float = IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2,
    imu_accel_bias_between_sigma_mps2: float = IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray, dict[str, int], list[ChunkSelectionRecord]]:
    n_epoch = batch.sat_ecef.shape[0]
    chunk_size = n_epoch if chunk_epochs <= 0 or n_epoch <= chunk_epochs else chunk_epochs
    stitched = raw_wls.copy()
    fgo_stitched = raw_wls.copy()
    total_iters = 0
    failed_chunks = 0
    selected_sources = np.empty(n_epoch, dtype=object)
    selected_source_counts = {"baseline": 0, "raw_wls": 0, "fgo": 0}
    prev_tail_xyz: np.ndarray | None = None
    chunk_records: list[ChunkSelectionRecord] = []
    factor_break_mask = _factor_break_mask(clock_jump, batch.dt, n_epoch)
    use_imu_accel_bias_state = bool(apply_imu_prior and imu_accel_bias_state)

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
                    imu_accel_bias_state=use_imu_accel_bias_state,
                )
                if seg_start == start and start > 0 and not (
                    factor_break_mask is not None and bool(factor_break_mask[start])
                ):
                    seg_state[0, :3] = stitched[start - 1, :3]
                    seg_state[0, 6 : 6 + batch.n_clock] = stitched[start - 1, 3 : 3 + batch.n_clock]
                tdcp_meas = None
                tdcp_weights = None
                if batch.tdcp_meas is not None and seg_end - seg_start > 1:
                    tdcp_meas = batch.tdcp_meas[seg_start : seg_end - 1]
                    tdcp_weights = batch.tdcp_weights[seg_start : seg_end - 1] if batch.tdcp_weights is not None else None
                seg_stop_mask = stop_mask[seg_start:seg_end] if stop_mask is not None else None
                imu_delta_p = None
                imu_delta_v = None
                if apply_imu_prior:
                    imu_delta_p, imu_delta_v, _ = _imu_preintegration_segment(
                        batch.imu_preintegration,
                        seg_start,
                        seg_end,
                    )
                    if imu_delta_p is not None and imu_delta_v is not None and seg_dt.size > 1:
                        valid_graph_dt = np.isfinite(seg_dt[:-1]) & (seg_dt[:-1] > 0.0)
                        if valid_graph_dt.size == imu_delta_p.shape[0]:
                            imu_delta_p[~valid_graph_dt, :] = np.nan
                            imu_delta_v[~valid_graph_dt, :] = np.nan
                rh_sigma = 0.0
                rh_up: np.ndarray | None = None
                rh_ei: np.ndarray | None = None
                rh_ej: np.ndarray | None = None
                abs_height_ref: np.ndarray | None = None
                abs_height_sigma = 0.0
                if graph_relative_height:
                    ref_seg = batch.kaggle_wls[seg_start:seg_end, :3]
                    sm = seg_stop_mask
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
                    iters, _ = fgo_gnss_lm_vd(
                        batch.sat_ecef[seg_start:seg_end],
                        batch.pseudorange[seg_start:seg_end],
                        batch.weights[seg_start:seg_end],
                        seg_state,
                        sys_kind=(batch.sys_kind[seg_start:seg_end] if batch.sys_kind is not None else None),
                        n_clock=batch.n_clock,
                        motion_sigma_m=motion_sigma_m,
                        clock_drift_sigma_m=clock_drift_sigma_m,
                        clock_use_average_drift=clock_use_average_drift,
                        stop_velocity_sigma_mps=stop_velocity_sigma_mps,
                        stop_position_sigma_m=stop_position_sigma_m,
                        max_iter=fgo_iters,
                        tol=tol,
                        sat_vel=(batch.sat_vel[seg_start:seg_end] if batch.sat_vel is not None else None),
                        doppler=(batch.doppler[seg_start:seg_end] if batch.doppler is not None else None),
                        doppler_weights=(batch.doppler_weights[seg_start:seg_end] if batch.doppler_weights is not None else None),
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
                        relative_height_sigma_m=rh_sigma,
                        enu_up_ecef=rh_up,
                        rel_height_edge_i=rh_ei,
                        rel_height_edge_j=rh_ej,
                        absolute_height_ref_ecef=abs_height_ref,
                        absolute_height_sigma_m=abs_height_sigma,
                        imu_delta_p=imu_delta_p,
                        imu_delta_v=imu_delta_v,
                        imu_position_sigma_m=imu_position_sigma_m,
                        imu_velocity_sigma_mps=imu_velocity_sigma_mps,
                        imu_accel_bias_prior_sigma_mps2=(
                            imu_accel_bias_prior_sigma_mps2 if use_imu_accel_bias_state else 0.0
                        ),
                        imu_accel_bias_between_sigma_mps2=(
                            imu_accel_bias_between_sigma_mps2 if use_imu_accel_bias_state else 0.0
                        ),
                    )
                except RuntimeError:
                    iters = -1
                if int(iters) < 0:
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
                    iters, _ = fgo_gnss_lm(
                        batch.sat_ecef[seg_start:seg_end],
                        batch.pseudorange[seg_start:seg_end],
                        batch.weights[seg_start:seg_end],
                        seg_state,
                        sys_kind=(batch.sys_kind[seg_start:seg_end] if batch.sys_kind is not None else None),
                        n_clock=batch.n_clock,
                        motion_sigma_m=motion_sigma_m,
                        max_iter=fgo_iters,
                        tol=tol,
                    )
                except RuntimeError:
                    iters = -1
                if int(iters) < 0:
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

    return (
        stitched,
        fgo_stitched,
        total_iters,
        failed_chunks,
        selected_sources,
        selected_source_counts,
        chunk_records,
    )



def solve_trip(trip: str, batch: TripArrays, config: BridgeConfig) -> BridgeResult:
    phone_name = Path(trip).name
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
    raw_wls = run_wls(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        sys_kind=batch.sys_kind,
        n_clock=batch.n_clock,
        fallback_xyz=batch.kaggle_wls,
    )
    fgo_run_options = _fgo_run_options_from_config(config)
    fgo_run_kwargs = fgo_run_options.run_kwargs()
    auto_state, fgo_state, iters, failed_chunks, auto_sources, auto_source_counts, chunk_records = run_fgo_chunked(
        batch,
        raw_wls,
        **solver_context_kwargs,
        **fgo_run_kwargs,
    )
    tdcp_off_fgo_state: np.ndarray | None = None
    tdcp_off_chunk_records: list[ChunkSelectionRecord] | None = None
    if _tdcp_off_candidate_enabled(config, batch):
        (
            _tdcp_off_auto_state,
            tdcp_off_fgo_state,
            _tdcp_off_iters,
            _tdcp_off_failed_chunks,
            _tdcp_off_auto_sources,
            _tdcp_off_auto_source_counts,
            tdcp_off_chunk_records,
        ) = run_fgo_chunked(
            _batch_without_tdcp(batch),
            raw_wls,
            **solver_context_kwargs,
            **fgo_run_kwargs,
        )
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

    allow_mi8_raw_wls_jump = _mi8_gated_baseline_jump_guard_enabled(phone_name, config.position_source)
    raw_wls_max_gap_m = _raw_wls_max_gap_guard_m(phone_name, config.position_source)
    gated_state, gated_sources, gated_counts = _select_gated_solution(
        source_catalog,
        chunk_records,
        n_epoch=batch.times_ms.size,
        baseline_threshold=config.gated_baseline_threshold,
        allow_raw_wls_on_mi8_baseline_jump=allow_mi8_raw_wls_jump,
        raw_wls_max_gap_m=raw_wls_max_gap_m,
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

    return _build_bridge_result(
        trip=trip,
        batch=batch,
        config=config,
        assembled_outputs=assembled_outputs,
        fgo_iters=int(iters),
        failed_chunks=int(failed_chunks),
        baseline_mse_pr=baseline_mse_pr,
        raw_wls_mse_pr=raw_wls_mse_pr,
        fgo_mse_pr=fgo_mse_pr,
        chunk_records=chunk_records,
        allow_raw_wls_on_mi8_baseline_jump=allow_mi8_raw_wls_jump,
        raw_wls_max_gap_m=raw_wls_max_gap_m,
    )


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
        factor_dt_max_s=cfg.factor_dt_max_s,
    )
    result = solve_trip(trip, batch, cfg)
    result.parity_audit = context.parity_audit
    refined_cfg = _outlier_refinement_config(cfg, result.selected_mse_pr)
    if refined_cfg is not None:
        refined = solve_trip(trip, batch, refined_cfg)
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
    "DEFAULT_TDCP_GEOMETRY_CORRECTION",
    "DEFAULT_TDCP_WEIGHT_SCALE",
    "IMU_ACCEL_BIAS_PRIOR_SIGMA_MPS2",
    "IMU_ACCEL_BIAS_BETWEEN_SIGMA_MPS2",
    "resolve_gsdc2023_data_root",
    "GATED_BASELINE_THRESHOLD_DEFAULT",
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
