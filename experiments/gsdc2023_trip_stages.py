"""Small build-trip stages for the GSDC2023 raw bridge."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BaseCorrectionMatrixFn = Callable[[Path, str, np.ndarray, Sequence[Any], str], np.ndarray]
BuildTripArraysFn = Callable[..., Any]
TripArraysFactoryFn = Callable[..., Any]
BuildTdcpArraysFn = Callable[..., tuple[np.ndarray | None, np.ndarray | None, int]]
AppendGnssLogOnlyRowsFn = Callable[..., Any]
ApplyDiagnosticsMaskFn = Callable[..., None]
ApplyTdcpGeometryCorrectionFn = Callable[[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray], int]
ApplyTdcpWeightScaleFn = Callable[[np.ndarray | None, float], None]
CleanClockDriftFn = Callable[[np.ndarray, np.ndarray | None, np.ndarray | None, str], np.ndarray | None]
ClockJumpFromEpochCountsFn = Callable[[Any], np.ndarray | None]
CombineClockJumpMasksFn = Callable[[np.ndarray | None, np.ndarray | None], np.ndarray | None]
DetectClockJumpsFromClockBiasFn = Callable[[np.ndarray, str], np.ndarray | None]
EstimateResidualClockSeriesFn = Callable[..., tuple[np.ndarray | None, np.ndarray | None]]
IsL5SignalFn = Callable[[str], bool]
GnssLogPseudorangeMatrixFn = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray] | None]
GnssLogEpochTimesFn = Callable[[Path], Any]
GpsMatrtklibNavMessagesForTripFn = Callable[[Path], Any]
GpsTgdBySvidForTripFn = Callable[[Path], dict[int, float]]
FillObservationMatricesFn = Callable[..., Any]
LoadAbsoluteHeightFn = Callable[..., tuple[np.ndarray | None, int]]
LoadImuMeasurementsFn = Callable[[Path], tuple[Any, Any, Any]]
MaskDopplerResidualFn = Callable[..., int]
MatlabSignalObservationMasksFn = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
MaskPseudorangeDopplerFn = Callable[..., int]
MaskPseudorangeResidualFn = Callable[..., int]
MultiGnssMaskFn = Callable[..., np.ndarray]
PreintegrateImuFn = Callable[..., Any]
ProcessImuFn = Callable[[Any, Any, np.ndarray, Any], tuple[Any, Any, Any]]
PseudorangeGlobalIsbFn = Callable[..., Any]
ProjectStopFn = Callable[[np.ndarray, Any, np.ndarray], np.ndarray]
ReceiverClockBiasLookupFn = Callable[[Any], dict[int, float]]
RecomputeRtklibTropoMatrixFn = Callable[..., np.ndarray]
RemapPseudorangeIsbFn = Callable[[Sequence[Any], Any, Sequence[Any]], Any]
RepairBaselineWlsFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
SelectEpochObservationsFn = Callable[..., Sequence[Any]]
SignalTypesForConstellationFn = Callable[..., Sequence[str]]
SlotFrequencyThresholdsFn = Callable[..., np.ndarray]
SlotPseudorangeGroupsFn = Callable[[Sequence[Any]], np.ndarray]


_RAW_OBSERVATION_FINITE_COLUMNS = (
    "RawPseudorangeMeters",
    "SvPositionXEcefMeters",
    "SvPositionYEcefMeters",
    "SvPositionZEcefMeters",
    "SvElevationDegrees",
    "SvClockBiasMeters",
    "IonosphericDelayMeters",
    "TroposphericDelayMeters",
)


@dataclass(frozen=True)
class GraphTimeDeltaProducts:
    dt: np.ndarray
    tdcp_dt: np.ndarray
    factor_dt_gap_count: int


@dataclass(frozen=True)
class RawObservationFrameProducts:
    frame: Any
    observation_mask_count: int


@dataclass(frozen=True)
class ObservationMatrixInputProducts:
    frame: Any
    gps_tgd_m_by_svid: dict[int, float]
    gps_matrtklib_nav_messages: Any


@dataclass(frozen=True)
class FilledObservationMatrixProducts:
    epochs: Sequence[Any]
    observations: Any


@dataclass(frozen=True)
class FilledObservationPostprocessProducts:
    kaggle_wls: np.ndarray
    rtklib_tropo_m: np.ndarray


@dataclass(frozen=True)
class ObservationPreparationStageProducts:
    raw_observation_frame: RawObservationFrameProducts
    metadata_context: "EpochMetadataContext"
    observation_matrix_input: ObservationMatrixInputProducts
    epoch_time_context: "EpochTimeContext"
    observation_matrix_stage: FilledObservationMatrixProducts
    post_fill_observation: FilledObservationPostprocessProducts


@dataclass(frozen=True)
class PreparedObservationProducts:
    filtered_frame: Any
    gps_tgd_m_by_svid: dict[int, float]
    observation_mask_count: int
    metadata_context: "EpochMetadataContext"
    epoch_time_context: "EpochTimeContext"
    baseline_velocity_times_ms: np.ndarray
    baseline_velocity_xyz: np.ndarray
    clock_drift_context_times_ms: np.ndarray
    clock_drift_context_mps: np.ndarray | None
    epochs: Sequence[Any]
    times_ms: np.ndarray
    sat_ecef: np.ndarray
    pseudorange: np.ndarray
    pseudorange_observable: np.ndarray
    weights: np.ndarray
    pseudorange_bias_weights: np.ndarray
    sat_clock_bias_matrix: np.ndarray
    rtklib_tropo_m: np.ndarray
    kaggle_wls: np.ndarray
    truth: np.ndarray
    visible_max: int
    slot_keys: list[Any]
    n_sat_slots: int
    n_clock: int
    elapsed_ns: np.ndarray | None
    sys_kind: np.ndarray | None
    clock_counts: np.ndarray | None
    clock_bias_m: np.ndarray | None
    clock_drift_mps: np.ndarray | None
    sat_vel: np.ndarray | None
    sat_clock_drift_mps: np.ndarray | None
    doppler: np.ndarray | None
    doppler_weights: np.ndarray | None
    adr: np.ndarray | None
    adr_state: np.ndarray | None
    adr_uncertainty: np.ndarray | None


@dataclass(frozen=True)
class FullObservationContextProducts:
    batch: Any | None
    clock_drift_context_times_ms: np.ndarray
    clock_drift_context_mps: np.ndarray | None
    full_isb_batch: Any | None


@dataclass(frozen=True)
class EpochMetadataContext:
    baseline_lookup: dict[int, np.ndarray]
    baseline_velocity_times_ms: np.ndarray
    baseline_velocity_xyz: np.ndarray
    hcdc_lookup: dict[int, float] | None
    elapsed_ns_lookup: dict[int, float] | None
    clock_bias_lookup: dict[int, float] | None
    clock_drift_lookup: dict[int, float] | None


@dataclass(frozen=True)
class EpochTimeContext:
    grouped_observations: dict[float, Any]
    epoch_time_keys: set[float]
    clock_drift_context_times_ms: np.ndarray
    clock_drift_context_mps: np.ndarray | None


@dataclass(frozen=True)
class ImuStageProducts:
    stop_epochs: np.ndarray | None
    imu_preintegration: Any | None


@dataclass(frozen=True)
class AbsoluteHeightStageProducts:
    absolute_height_ref_ecef: np.ndarray | None
    absolute_height_ref_count: int


@dataclass(frozen=True)
class ClockResidualStageProducts:
    clock_jump: np.ndarray | None
    clock_bias_m: np.ndarray | None
    clock_drift_mps: np.ndarray | None


@dataclass(frozen=True)
class PseudorangeResidualStageProducts:
    pseudorange_isb_by_group: Any | None
    residual_mask_count: int


@dataclass(frozen=True)
class GnssLogPseudorangeStageProducts:
    pseudorange_isb_sample_weights: np.ndarray
    applied_count: int


@dataclass(frozen=True)
class DopplerResidualStageProducts:
    doppler_residual_mask_count: int


@dataclass(frozen=True)
class PseudorangeDopplerStageProducts:
    pseudorange_doppler_mask_count: int


@dataclass(frozen=True)
class ObservationMaskBaseCorrectionStageProducts:
    doppler_residual_stage: DopplerResidualStageProducts
    pseudorange_doppler_stage: PseudorangeDopplerStageProducts
    pseudorange_residual_stage: PseudorangeResidualStageProducts
    base_correction_count: int


@dataclass(frozen=True)
class TdcpStageProducts:
    tdcp_meas: np.ndarray | None
    tdcp_weights: np.ndarray | None
    signal_tdcp_weights: np.ndarray | None
    tdcp_consistency_mask_count: int
    tdcp_geometry_correction_count: int


@dataclass(frozen=True)
class PostObservationStageProducts:
    gnss_log_stage: GnssLogPseudorangeStageProducts
    absolute_height_stage: AbsoluteHeightStageProducts
    clock_residual_stage: ClockResidualStageProducts
    full_context_stage: FullObservationContextProducts
    mask_base_stage: ObservationMaskBaseCorrectionStageProducts
    time_delta: GraphTimeDeltaProducts
    tdcp_stage: TdcpStageProducts
    imu_stage: ImuStageProducts
    signal_weights: np.ndarray
    signal_doppler_weights: np.ndarray | None


@dataclass(frozen=True)
class PostObservationStageConfig:
    trip_dir: Path
    phone_name: str
    apply_absolute_height: bool
    absolute_height_dist_m: float
    clock_drift_blocklist_phones: set[str] | Sequence[str]
    apply_observation_mask: bool
    has_window_subset: bool
    constellation_type: int
    signal_type: str
    weight_mode: str
    multi_gnss: bool
    observation_min_cn0_dbhz: float
    observation_min_elevation_deg: float
    dual_frequency: bool
    factor_dt_max_s: float
    apply_base_correction: bool
    data_root: Path | None
    trip: str | None
    doppler_residual_mask_mps: float
    pseudorange_doppler_mask_m: float
    pseudorange_residual_mask_m: float
    pseudorange_residual_mask_l5_m: float | None
    tdcp_consistency_threshold_m: float
    tdcp_loffset_m: float
    matlab_residual_diagnostics_mask_path: Path | None
    tdcp_geometry_correction: bool
    tdcp_weight_scale: float
    imu_frame: str
    default_pd_l1_threshold_m: float
    default_pd_l5_threshold_m: float
    default_pr_l1_threshold_m: float
    default_pr_l5_threshold_m: float


@dataclass(frozen=True)
class PostObservationStageDependencies:
    build_trip_arrays_fn: BuildTripArraysFn
    gnss_log_corrected_pseudorange_matrix_fn: GnssLogPseudorangeMatrixFn
    load_absolute_height_reference_ecef_fn: LoadAbsoluteHeightFn
    clock_jump_from_epoch_counts_fn: ClockJumpFromEpochCountsFn
    estimate_residual_clock_series_fn: EstimateResidualClockSeriesFn
    combine_clock_jump_masks_fn: CombineClockJumpMasksFn
    detect_clock_jumps_from_clock_bias_fn: DetectClockJumpsFromClockBiasFn
    clean_clock_drift_fn: CleanClockDriftFn
    correction_matrix_fn: BaseCorrectionMatrixFn
    mask_doppler_residual_outliers_fn: MaskDopplerResidualFn
    slot_frequency_thresholds_fn: SlotFrequencyThresholdsFn
    mask_pseudorange_doppler_consistency_fn: MaskPseudorangeDopplerFn
    slot_pseudorange_common_bias_groups_fn: SlotPseudorangeGroupsFn
    remap_pseudorange_isb_by_group_fn: RemapPseudorangeIsbFn
    pseudorange_global_isb_by_group_fn: PseudorangeGlobalIsbFn
    is_l5_signal_fn: IsL5SignalFn
    mask_pseudorange_residual_outliers_fn: MaskPseudorangeResidualFn
    build_tdcp_arrays_fn: BuildTdcpArraysFn
    apply_diagnostics_mask_fn: ApplyDiagnosticsMaskFn
    apply_geometry_correction_fn: ApplyTdcpGeometryCorrectionFn
    apply_weight_scale_fn: ApplyTdcpWeightScaleFn
    load_device_imu_measurements_fn: LoadImuMeasurementsFn
    process_device_imu_fn: ProcessImuFn
    project_stop_to_epochs_fn: ProjectStopFn
    preintegrate_processed_imu_fn: PreintegrateImuFn


def build_clock_residual_stage(
    *,
    phone_name: str,
    clock_drift_blocklist_phones: set[str] | Sequence[str],
    times_ms: np.ndarray,
    kaggle_wls: np.ndarray,
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    sat_vel: np.ndarray | None,
    doppler: np.ndarray | None,
    sat_clock_drift_mps: np.ndarray | None,
    sys_kind: np.ndarray | None,
    clock_counts: np.ndarray | None,
    clock_bias_m: np.ndarray | None,
    clock_drift_mps: np.ndarray | None,
    clock_jump_from_epoch_counts_fn: ClockJumpFromEpochCountsFn,
    estimate_residual_clock_series_fn: EstimateResidualClockSeriesFn,
    combine_clock_jump_masks_fn: CombineClockJumpMasksFn,
    detect_clock_jumps_from_clock_bias_fn: DetectClockJumpsFromClockBiasFn,
    clean_clock_drift_fn: CleanClockDriftFn,
) -> ClockResidualStageProducts:
    clock_jump = clock_jump_from_epoch_counts_fn(clock_counts) if clock_counts is not None else None
    if phone_name.lower() in clock_drift_blocklist_phones:
        residual_clock_bias_m, residual_clock_drift_mps = estimate_residual_clock_series_fn(
            times_ms,
            kaggle_wls,
            sat_ecef,
            pseudorange,
            sat_vel,
            doppler,
            sat_clock_drift_mps=sat_clock_drift_mps,
            sys_kind=sys_kind,
        )
        if residual_clock_bias_m is not None:
            clock_bias_m = residual_clock_bias_m
            residual_clock_jump = detect_clock_jumps_from_clock_bias_fn(clock_bias_m, phone_name)
            clock_jump = combine_clock_jump_masks_fn(clock_jump, residual_clock_jump)
        if residual_clock_drift_mps is not None:
            clock_drift_mps = residual_clock_drift_mps

    clock_drift_mps = clean_clock_drift_fn(times_ms, clock_bias_m, clock_drift_mps, phone_name)
    return ClockResidualStageProducts(
        clock_jump=clock_jump,
        clock_bias_m=clock_bias_m,
        clock_drift_mps=clock_drift_mps,
    )


def assemble_trip_arrays_stage(
    *,
    trip_arrays_cls: TripArraysFactoryFn,
    times_ms: np.ndarray,
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    kaggle_wls: np.ndarray,
    truth: np.ndarray,
    visible_max: int,
    has_truth: bool,
    sys_kind: np.ndarray | None,
    n_clock: int,
    sat_vel: np.ndarray | None,
    sat_clock_drift_mps: np.ndarray | None,
    doppler: np.ndarray | None,
    doppler_weights: np.ndarray | None,
    pseudorange_bias_weights: np.ndarray,
    pseudorange_residual_stage: PseudorangeResidualStageProducts,
    time_delta: GraphTimeDeltaProducts,
    tdcp_stage: TdcpStageProducts,
    n_sat_slots: int,
    slot_keys: Sequence[Any],
    elapsed_ns: np.ndarray | None,
    clock_jump: np.ndarray | None,
    clock_bias_m: np.ndarray | None,
    clock_drift_mps: np.ndarray | None,
    imu_stage: ImuStageProducts,
    absolute_height_stage: AbsoluteHeightStageProducts,
    base_correction_count: int,
    observation_mask_count: int,
    doppler_residual_stage: DopplerResidualStageProducts,
    pseudorange_doppler_stage: PseudorangeDopplerStageProducts,
    dual_frequency: bool,
) -> Any:
    return trip_arrays_cls(
        times_ms=times_ms,
        sat_ecef=sat_ecef,
        pseudorange=pseudorange,
        weights=weights,
        kaggle_wls=kaggle_wls,
        truth=truth,
        max_sats=visible_max,
        has_truth=has_truth,
        sys_kind=sys_kind,
        n_clock=n_clock,
        sat_vel=sat_vel,
        sat_clock_drift_mps=sat_clock_drift_mps,
        doppler=doppler,
        doppler_weights=doppler_weights,
        pseudorange_bias_weights=pseudorange_bias_weights,
        pseudorange_isb_by_group=pseudorange_residual_stage.pseudorange_isb_by_group,
        dt=time_delta.dt,
        tdcp_meas=tdcp_stage.tdcp_meas,
        tdcp_weights=tdcp_stage.tdcp_weights,
        n_sat_slots=n_sat_slots,
        slot_keys=tuple(slot_keys),
        elapsed_ns=elapsed_ns,
        clock_jump=clock_jump,
        clock_bias_m=clock_bias_m,
        clock_drift_mps=clock_drift_mps,
        factor_dt_gap_count=time_delta.factor_dt_gap_count,
        stop_epochs=imu_stage.stop_epochs,
        imu_preintegration=imu_stage.imu_preintegration,
        absolute_height_ref_ecef=absolute_height_stage.absolute_height_ref_ecef,
        absolute_height_ref_count=absolute_height_stage.absolute_height_ref_count,
        base_correction_count=base_correction_count,
        observation_mask_count=observation_mask_count,
        residual_mask_count=pseudorange_residual_stage.residual_mask_count,
        doppler_residual_mask_count=doppler_residual_stage.doppler_residual_mask_count,
        pseudorange_doppler_mask_count=pseudorange_doppler_stage.pseudorange_doppler_mask_count,
        tdcp_consistency_mask_count=tdcp_stage.tdcp_consistency_mask_count,
        tdcp_geometry_correction_count=tdcp_stage.tdcp_geometry_correction_count,
        dual_frequency=dual_frequency,
    )


def apply_base_correction_to_pseudorange(
    *,
    data_root: Path | None,
    trip: str | None,
    times_ms: np.ndarray,
    slot_keys: Sequence[Any],
    signal_type: str,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    correction_matrix_fn: BaseCorrectionMatrixFn,
) -> int:
    if data_root is None or trip is None:
        raise RuntimeError("data_root and trip are required when apply_base_correction=True")
    base_correction = correction_matrix_fn(
        data_root,
        trip,
        times_ms,
        slot_keys,
        signal_type,
    )
    valid_base_correction = np.isfinite(base_correction) & (weights > 0.0)
    pseudorange[valid_base_correction] -= base_correction[valid_base_correction]
    return int(np.count_nonzero(valid_base_correction))


def assemble_prepared_trip_arrays_stage(
    *,
    trip_arrays_cls: TripArraysFactoryFn,
    observation_products: PreparedObservationProducts,
    post_observation_stages: PostObservationStageProducts,
    has_truth: bool,
    dual_frequency: bool,
) -> Any:
    clock_residual_stage = post_observation_stages.clock_residual_stage
    mask_base_stage = post_observation_stages.mask_base_stage
    return assemble_trip_arrays_stage(
        trip_arrays_cls=trip_arrays_cls,
        times_ms=observation_products.times_ms,
        sat_ecef=observation_products.sat_ecef,
        pseudorange=observation_products.pseudorange,
        weights=observation_products.weights,
        kaggle_wls=observation_products.kaggle_wls,
        truth=observation_products.truth,
        has_truth=has_truth,
        visible_max=observation_products.visible_max,
        sys_kind=observation_products.sys_kind,
        n_clock=observation_products.n_clock,
        sat_vel=observation_products.sat_vel,
        sat_clock_drift_mps=observation_products.sat_clock_drift_mps,
        doppler=observation_products.doppler,
        doppler_weights=observation_products.doppler_weights,
        pseudorange_bias_weights=observation_products.pseudorange_bias_weights,
        pseudorange_residual_stage=mask_base_stage.pseudorange_residual_stage,
        time_delta=post_observation_stages.time_delta,
        tdcp_stage=post_observation_stages.tdcp_stage,
        n_sat_slots=observation_products.n_sat_slots,
        slot_keys=observation_products.slot_keys,
        elapsed_ns=observation_products.elapsed_ns,
        clock_jump=clock_residual_stage.clock_jump,
        clock_bias_m=clock_residual_stage.clock_bias_m,
        clock_drift_mps=clock_residual_stage.clock_drift_mps,
        imu_stage=post_observation_stages.imu_stage,
        absolute_height_stage=post_observation_stages.absolute_height_stage,
        base_correction_count=mask_base_stage.base_correction_count,
        observation_mask_count=observation_products.observation_mask_count,
        doppler_residual_stage=mask_base_stage.doppler_residual_stage,
        pseudorange_doppler_stage=mask_base_stage.pseudorange_doppler_stage,
        dual_frequency=dual_frequency,
    )


def build_graph_time_delta_products(times_ms: np.ndarray, factor_dt_max_s: float) -> GraphTimeDeltaProducts:
    times = np.asarray(times_ms, dtype=np.float64).reshape(-1)
    n_epoch = times.size
    dt = np.zeros(n_epoch, dtype=np.float64)
    tdcp_dt = np.zeros(n_epoch, dtype=np.float64)
    factor_dt_gap_count = 0
    if n_epoch > 1:
        raw_dt = np.diff(times) / 1000.0
        dt_view = dt[:-1]
        tdcp_view = tdcp_dt[:-1]
        dt_view[:] = raw_dt
        tdcp_view[:] = raw_dt
        bad_dt = (~np.isfinite(raw_dt)) | (raw_dt <= 0.0) | (raw_dt > 30.0)
        if factor_dt_max_s > 0.0:
            bad_dt |= raw_dt >= float(factor_dt_max_s)
        factor_dt_gap_count = int(np.count_nonzero(bad_dt))
        dt_view[bad_dt] = 0.0
        tdcp_view[(~np.isfinite(raw_dt)) | (raw_dt <= 0.0)] = 0.0
    return GraphTimeDeltaProducts(
        dt=dt,
        tdcp_dt=tdcp_dt,
        factor_dt_gap_count=factor_dt_gap_count,
    )


def build_raw_observation_frame(
    raw_frame: Any,
    *,
    epoch_meta: Any,
    trip_dir: Path,
    phone_name: str,
    constellation_type: int,
    signal_type: str,
    multi_gnss: bool,
    dual_frequency: bool,
    apply_observation_mask: bool,
    observation_min_cn0_dbhz: float,
    observation_min_elevation_deg: float,
    multi_gnss_mask_fn: MultiGnssMaskFn,
    signal_types_for_constellation_fn: SignalTypesForConstellationFn,
    append_gnss_log_only_gps_rows_fn: AppendGnssLogOnlyRowsFn,
    matlab_signal_observation_masks_fn: MatlabSignalObservationMasksFn,
) -> RawObservationFrameProducts:
    frame = raw_frame

    if multi_gnss:
        frame = frame[multi_gnss_mask_fn(frame, dual_frequency=dual_frequency)]
    elif dual_frequency:
        signal_types = signal_types_for_constellation_fn(
            constellation_type,
            signal_type,
            dual_frequency=True,
        )
        frame = frame[
            (frame["ConstellationType"] == constellation_type)
            & (frame["SignalType"].isin(signal_types))
        ]
    else:
        frame = frame[
            (frame["ConstellationType"] == constellation_type)
            & (frame["SignalType"] == signal_type)
        ]

    if apply_observation_mask and constellation_type == 1:
        frame = append_gnss_log_only_gps_rows_fn(
            frame,
            raw_frame,
            epoch_meta,
            trip_dir,
            phone_name=phone_name,
            dual_frequency=dual_frequency,
        )

    finite_mask = np.ones(len(frame), dtype=bool)
    for column in _RAW_OBSERVATION_FINITE_COLUMNS:
        finite_mask &= np.isfinite(frame[column])
    frame = frame[finite_mask]

    observation_mask_count = 0
    if apply_observation_mask:
        p_ok, d_ok, l_ok = matlab_signal_observation_masks_fn(
            frame,
            min_cn0_dbhz=observation_min_cn0_dbhz,
            min_elevation_deg=observation_min_elevation_deg,
        )
        p_bias_ok = p_ok.copy()
        frame = frame.copy()
        frame["bridge_p_ok"] = p_ok
        frame["bridge_d_ok"] = d_ok
        frame["bridge_l_ok"] = l_ok
        frame["bridge_p_bias_ok"] = p_bias_ok
        observation_mask_count = int(np.count_nonzero(~p_ok))
        keep_for_gnss_log_p = np.zeros(len(frame), dtype=bool)
        if (trip_dir / "supplemental" / "gnss_log.txt").is_file():
            keep_for_gnss_log_p = (
                (pd.to_numeric(frame["ConstellationType"], errors="coerce").fillna(0).astype(np.int64).to_numpy() == 1)
                & frame["SignalType"].isin(
                    signal_types_for_constellation_fn(1, "GPS_L1_CA", dual_frequency=True)
                ).to_numpy()
            )
        frame = frame[p_ok | d_ok | l_ok | keep_for_gnss_log_p]

    if frame.empty:
        raise RuntimeError("No usable raw observations after filtering")

    return RawObservationFrameProducts(
        frame=frame,
        observation_mask_count=observation_mask_count,
    )


def build_observation_matrix_input_stage(
    filtered_frame: Any,
    *,
    trip_dir: Path,
    gps_tgd_m_by_svid_for_trip_fn: GpsTgdBySvidForTripFn,
    gps_matrtklib_nav_messages_for_trip_fn: GpsMatrtklibNavMessagesForTripFn,
) -> ObservationMatrixInputProducts:
    frame = filtered_frame.sort_values(["utcTimeMillis", "ConstellationType", "Svid", "Cn0DbHz"]).groupby(
        ["utcTimeMillis", "ConstellationType", "Svid", "SignalType"],
        as_index=False,
    ).tail(1)
    return ObservationMatrixInputProducts(
        frame=frame,
        gps_tgd_m_by_svid=gps_tgd_m_by_svid_for_trip_fn(trip_dir),
        gps_matrtklib_nav_messages=gps_matrtklib_nav_messages_for_trip_fn(trip_dir),
    )


def build_filled_observation_matrix_stage(
    *,
    epoch_time_context: EpochTimeContext,
    metadata_context: EpochMetadataContext,
    observation_matrix_input: ObservationMatrixInputProducts,
    gt_times: np.ndarray,
    gt_ecef: np.ndarray,
    start_epoch: int,
    max_epochs: int,
    weight_mode: str,
    multi_gnss: bool,
    dual_frequency: bool,
    tdcp_enabled: bool,
    adr_sign: float,
    select_epoch_observations_fn: SelectEpochObservationsFn,
    fill_observation_matrices_fn: FillObservationMatricesFn,
    nearest_index_fn: Callable[..., Any],
    gps_arrival_tow_s_from_row_fn: Callable[..., Any],
    gps_sat_clock_bias_adjustment_m_fn: Callable[..., Any],
    gps_matrtklib_sat_product_adjustment_fn: Callable[..., Any],
    clock_kind_for_observation_fn: Callable[..., Any],
    is_l5_signal_fn: Callable[..., Any],
    slot_sort_key_fn: Callable[..., Any],
    ecef_to_lla_fn: Callable[..., Any],
    elevation_azimuth_fn: Callable[..., Any],
    rtklib_tropo_fn: Callable[..., Any],
    matlab_signal_clock_dim: int,
) -> FilledObservationMatrixProducts:
    epochs = select_epoch_observations_fn(
        epoch_time_context.epoch_time_keys,
        epoch_time_context.grouped_observations,
        metadata_context.baseline_lookup,
        gt_times,
        gt_ecef,
        start_epoch=start_epoch,
        max_epochs=max_epochs,
        nearest_index_fn=nearest_index_fn,
    )
    if not epochs:
        raise RuntimeError("No usable epochs found")

    observations = fill_observation_matrices_fn(
        epochs,
        source_columns=observation_matrix_input.frame.columns,
        baseline_lookup=metadata_context.baseline_lookup,
        weight_mode=weight_mode,
        multi_gnss=multi_gnss,
        dual_frequency=dual_frequency,
        tdcp_enabled=tdcp_enabled,
        adr_sign=adr_sign,
        elapsed_ns_lookup=metadata_context.elapsed_ns_lookup,
        hcdc_lookup=metadata_context.hcdc_lookup,
        clock_bias_lookup=metadata_context.clock_bias_lookup,
        clock_drift_lookup=metadata_context.clock_drift_lookup,
        gps_tgd_m_by_svid=observation_matrix_input.gps_tgd_m_by_svid,
        gps_matrtklib_nav_messages=observation_matrix_input.gps_matrtklib_nav_messages,
        gps_arrival_tow_s_from_row_fn=gps_arrival_tow_s_from_row_fn,
        gps_sat_clock_bias_adjustment_m_fn=gps_sat_clock_bias_adjustment_m_fn,
        gps_matrtklib_sat_product_adjustment_fn=gps_matrtklib_sat_product_adjustment_fn,
        clock_kind_for_observation_fn=clock_kind_for_observation_fn,
        is_l5_signal_fn=is_l5_signal_fn,
        slot_sort_key_fn=slot_sort_key_fn,
        ecef_to_lla_fn=ecef_to_lla_fn,
        elevation_azimuth_fn=elevation_azimuth_fn,
        rtklib_tropo_fn=rtklib_tropo_fn,
        matlab_signal_clock_dim=matlab_signal_clock_dim,
    )
    return FilledObservationMatrixProducts(
        epochs=epochs,
        observations=observations,
    )


def postprocess_filled_observation_stage(
    *,
    times_ms: np.ndarray,
    kaggle_wls: np.ndarray,
    sat_ecef: np.ndarray,
    rtklib_tropo_m: np.ndarray,
    repair_baseline_wls_fn: RepairBaselineWlsFn,
    recompute_rtklib_tropo_matrix_fn: RecomputeRtklibTropoMatrixFn,
    ecef_to_lla_fn: Callable[..., Any],
    elevation_azimuth_fn: Callable[..., Any],
    rtklib_tropo_fn: Callable[..., Any],
) -> FilledObservationPostprocessProducts:
    repaired_wls = repair_baseline_wls_fn(times_ms, kaggle_wls)
    recomputed_tropo_m = recompute_rtklib_tropo_matrix_fn(
        repaired_wls,
        sat_ecef,
        ecef_to_lla_fn=ecef_to_lla_fn,
        elevation_azimuth_fn=elevation_azimuth_fn,
        rtklib_tropo_fn=rtklib_tropo_fn,
        initial_tropo_m=rtklib_tropo_m,
    )
    return FilledObservationPostprocessProducts(
        kaggle_wls=repaired_wls,
        rtklib_tropo_m=recomputed_tropo_m,
    )


def build_observation_preparation_stages(
    raw_frame: Any,
    epoch_meta: Any,
    *,
    trip_dir: Path,
    phone_name: str,
    constellation_type: int,
    signal_type: str,
    multi_gnss: bool,
    dual_frequency: bool,
    apply_observation_mask: bool,
    observation_min_cn0_dbhz: float,
    observation_min_elevation_deg: float,
    gt_times: np.ndarray,
    gt_ecef: np.ndarray,
    start_epoch: int,
    max_epochs: int,
    weight_mode: str,
    tdcp_enabled: bool,
    adr_sign: float,
    multi_gnss_mask_fn: MultiGnssMaskFn,
    signal_types_for_constellation_fn: SignalTypesForConstellationFn,
    append_gnss_log_only_gps_rows_fn: AppendGnssLogOnlyRowsFn,
    matlab_signal_observation_masks_fn: MatlabSignalObservationMasksFn,
    repair_baseline_wls_fn: RepairBaselineWlsFn,
    receiver_clock_bias_lookup_from_epoch_meta_fn: ReceiverClockBiasLookupFn,
    light_speed_mps: float,
    gps_tgd_m_by_svid_for_trip_fn: GpsTgdBySvidForTripFn,
    gps_matrtklib_nav_messages_for_trip_fn: GpsMatrtklibNavMessagesForTripFn,
    gnss_log_matlab_epoch_times_ms_fn: GnssLogEpochTimesFn,
    clean_clock_drift_fn: CleanClockDriftFn,
    select_epoch_observations_fn: SelectEpochObservationsFn,
    fill_observation_matrices_fn: FillObservationMatricesFn,
    nearest_index_fn: Callable[..., Any],
    gps_arrival_tow_s_from_row_fn: Callable[..., Any],
    gps_sat_clock_bias_adjustment_m_fn: Callable[..., Any],
    gps_matrtklib_sat_product_adjustment_fn: Callable[..., Any],
    clock_kind_for_observation_fn: Callable[..., Any],
    is_l5_signal_fn: Callable[..., Any],
    slot_sort_key_fn: Callable[..., Any],
    ecef_to_lla_fn: Callable[..., Any],
    elevation_azimuth_fn: Callable[..., Any],
    rtklib_tropo_fn: Callable[..., Any],
    matlab_signal_clock_dim: int,
    recompute_rtklib_tropo_matrix_fn: RecomputeRtklibTropoMatrixFn,
) -> ObservationPreparationStageProducts:
    raw_observation_frame = build_raw_observation_frame(
        raw_frame,
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
        multi_gnss_mask_fn=multi_gnss_mask_fn,
        signal_types_for_constellation_fn=signal_types_for_constellation_fn,
        append_gnss_log_only_gps_rows_fn=append_gnss_log_only_gps_rows_fn,
        matlab_signal_observation_masks_fn=matlab_signal_observation_masks_fn,
    )
    metadata_context = build_epoch_metadata_context(
        epoch_meta,
        repair_baseline_wls_fn=repair_baseline_wls_fn,
        receiver_clock_bias_lookup_from_epoch_meta_fn=receiver_clock_bias_lookup_from_epoch_meta_fn,
        light_speed_mps=light_speed_mps,
    )
    observation_matrix_input = build_observation_matrix_input_stage(
        raw_observation_frame.frame,
        trip_dir=trip_dir,
        gps_tgd_m_by_svid_for_trip_fn=gps_tgd_m_by_svid_for_trip_fn,
        gps_matrtklib_nav_messages_for_trip_fn=gps_matrtklib_nav_messages_for_trip_fn,
    )
    epoch_time_context = build_epoch_time_context(
        observation_matrix_input.frame,
        apply_observation_mask=apply_observation_mask,
        multi_gnss=multi_gnss,
        constellation_type=constellation_type,
        dual_frequency=dual_frequency,
        trip_dir=trip_dir,
        clock_bias_lookup=metadata_context.clock_bias_lookup,
        clock_drift_lookup=metadata_context.clock_drift_lookup,
        phone_name=phone_name,
        gnss_log_matlab_epoch_times_ms_fn=gnss_log_matlab_epoch_times_ms_fn,
        clean_clock_drift_fn=clean_clock_drift_fn,
    )
    observation_matrix_stage = build_filled_observation_matrix_stage(
        epoch_time_context=epoch_time_context,
        metadata_context=metadata_context,
        observation_matrix_input=observation_matrix_input,
        gt_times=gt_times,
        gt_ecef=gt_ecef,
        start_epoch=start_epoch,
        max_epochs=max_epochs,
        weight_mode=weight_mode,
        multi_gnss=multi_gnss,
        dual_frequency=dual_frequency,
        tdcp_enabled=tdcp_enabled,
        adr_sign=adr_sign,
        select_epoch_observations_fn=select_epoch_observations_fn,
        fill_observation_matrices_fn=fill_observation_matrices_fn,
        nearest_index_fn=nearest_index_fn,
        gps_arrival_tow_s_from_row_fn=gps_arrival_tow_s_from_row_fn,
        gps_sat_clock_bias_adjustment_m_fn=gps_sat_clock_bias_adjustment_m_fn,
        gps_matrtklib_sat_product_adjustment_fn=gps_matrtklib_sat_product_adjustment_fn,
        clock_kind_for_observation_fn=clock_kind_for_observation_fn,
        is_l5_signal_fn=is_l5_signal_fn,
        slot_sort_key_fn=slot_sort_key_fn,
        ecef_to_lla_fn=ecef_to_lla_fn,
        elevation_azimuth_fn=elevation_azimuth_fn,
        rtklib_tropo_fn=rtklib_tropo_fn,
        matlab_signal_clock_dim=matlab_signal_clock_dim,
    )
    observations = observation_matrix_stage.observations
    post_fill_observation = postprocess_filled_observation_stage(
        times_ms=observations.times_ms,
        kaggle_wls=observations.kaggle_wls,
        sat_ecef=observations.sat_ecef,
        rtklib_tropo_m=observations.rtklib_tropo_m,
        repair_baseline_wls_fn=repair_baseline_wls_fn,
        recompute_rtklib_tropo_matrix_fn=recompute_rtklib_tropo_matrix_fn,
        ecef_to_lla_fn=ecef_to_lla_fn,
        elevation_azimuth_fn=elevation_azimuth_fn,
        rtklib_tropo_fn=rtklib_tropo_fn,
    )
    return ObservationPreparationStageProducts(
        raw_observation_frame=raw_observation_frame,
        metadata_context=metadata_context,
        observation_matrix_input=observation_matrix_input,
        epoch_time_context=epoch_time_context,
        observation_matrix_stage=observation_matrix_stage,
        post_fill_observation=post_fill_observation,
    )


def unpack_observation_preparation_stage(
    stage: ObservationPreparationStageProducts,
) -> PreparedObservationProducts:
    observations = stage.observation_matrix_stage.observations
    post_fill_observation = stage.post_fill_observation
    metadata_context = stage.metadata_context
    epoch_time_context = stage.epoch_time_context
    slot_keys = list(observations.slot_keys)
    return PreparedObservationProducts(
        filtered_frame=stage.observation_matrix_input.frame,
        gps_tgd_m_by_svid=stage.observation_matrix_input.gps_tgd_m_by_svid,
        observation_mask_count=stage.raw_observation_frame.observation_mask_count,
        metadata_context=metadata_context,
        epoch_time_context=epoch_time_context,
        baseline_velocity_times_ms=metadata_context.baseline_velocity_times_ms,
        baseline_velocity_xyz=metadata_context.baseline_velocity_xyz,
        clock_drift_context_times_ms=epoch_time_context.clock_drift_context_times_ms,
        clock_drift_context_mps=epoch_time_context.clock_drift_context_mps,
        epochs=stage.observation_matrix_stage.epochs,
        times_ms=observations.times_ms,
        sat_ecef=observations.sat_ecef,
        pseudorange=observations.pseudorange,
        pseudorange_observable=observations.pseudorange_observable,
        weights=observations.weights,
        pseudorange_bias_weights=observations.pseudorange_bias_weights,
        sat_clock_bias_matrix=observations.sat_clock_bias_matrix,
        rtklib_tropo_m=post_fill_observation.rtklib_tropo_m,
        kaggle_wls=post_fill_observation.kaggle_wls,
        truth=observations.truth,
        visible_max=observations.visible_max,
        slot_keys=slot_keys,
        n_sat_slots=len(slot_keys),
        n_clock=observations.n_clock,
        elapsed_ns=observations.elapsed_ns,
        sys_kind=observations.sys_kind,
        clock_counts=observations.clock_counts,
        clock_bias_m=observations.clock_bias_m,
        clock_drift_mps=observations.clock_drift_mps,
        sat_vel=observations.sat_vel,
        sat_clock_drift_mps=observations.sat_clock_drift_mps,
        doppler=observations.doppler,
        doppler_weights=observations.doppler_weights,
        adr=observations.adr,
        adr_state=observations.adr_state,
        adr_uncertainty=observations.adr_uncertainty,
    )


def build_epoch_metadata_context(
    epoch_meta: Any,
    *,
    repair_baseline_wls_fn: RepairBaselineWlsFn,
    receiver_clock_bias_lookup_from_epoch_meta_fn: ReceiverClockBiasLookupFn,
    light_speed_mps: float,
) -> EpochMetadataContext:
    baseline_lookup = {
        int(row.utcTimeMillis): np.array(
            [
                float(row.WlsPositionXEcefMeters),
                float(row.WlsPositionYEcefMeters),
                float(row.WlsPositionZEcefMeters),
            ],
            dtype=np.float64,
        )
        for row in epoch_meta.itertuples(index=False)
    }
    baseline_velocity_times_ms = epoch_meta["utcTimeMillis"].to_numpy(dtype=np.float64)
    baseline_velocity_xyz = epoch_meta[
        ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    ].to_numpy(dtype=np.float64)
    baseline_velocity_valid = np.isfinite(baseline_velocity_times_ms) & np.isfinite(baseline_velocity_xyz).all(axis=1)
    baseline_velocity_times_ms = baseline_velocity_times_ms[baseline_velocity_valid]
    baseline_velocity_xyz = baseline_velocity_xyz[baseline_velocity_valid]
    if baseline_velocity_times_ms.size:
        baseline_velocity_order = np.argsort(baseline_velocity_times_ms)
        baseline_velocity_times_ms = baseline_velocity_times_ms[baseline_velocity_order]
        baseline_velocity_xyz = repair_baseline_wls_fn(
            baseline_velocity_times_ms,
            baseline_velocity_xyz[baseline_velocity_order],
        )

    hcdc_lookup = None
    if "HardwareClockDiscontinuityCount" in epoch_meta.columns:
        hcdc_lookup = {
            int(row.utcTimeMillis): float(row.HardwareClockDiscontinuityCount)
            for row in epoch_meta.itertuples(index=False)
        }

    elapsed_ns_lookup = None
    if "ChipsetElapsedRealtimeNanos" in epoch_meta.columns:
        elapsed_ns_lookup = {
            int(row.utcTimeMillis): float(row.ChipsetElapsedRealtimeNanos)
            for row in epoch_meta.itertuples(index=False)
            if _not_missing(row.ChipsetElapsedRealtimeNanos)
        }

    clock_bias_lookup = (
        receiver_clock_bias_lookup_from_epoch_meta_fn(epoch_meta)
        if "FullBiasNanos" in epoch_meta.columns
        else None
    )
    clock_drift_lookup = None
    if "DriftNanosPerSecond" in epoch_meta.columns:
        clock_drift_lookup = {
            int(row.utcTimeMillis): (-float(row.DriftNanosPerSecond) * 1e-9 * float(light_speed_mps))
            for row in epoch_meta.itertuples(index=False)
            if _not_missing(row.DriftNanosPerSecond)
        }

    return EpochMetadataContext(
        baseline_lookup=baseline_lookup,
        baseline_velocity_times_ms=baseline_velocity_times_ms,
        baseline_velocity_xyz=baseline_velocity_xyz,
        hcdc_lookup=hcdc_lookup,
        elapsed_ns_lookup=elapsed_ns_lookup,
        clock_bias_lookup=clock_bias_lookup,
        clock_drift_lookup=clock_drift_lookup,
    )


def build_epoch_time_context(
    filtered_frame: Any,
    *,
    apply_observation_mask: bool,
    multi_gnss: bool,
    constellation_type: int,
    dual_frequency: bool,
    trip_dir: Path,
    clock_bias_lookup: dict[int, float] | None,
    clock_drift_lookup: dict[int, float] | None,
    phone_name: str,
    gnss_log_matlab_epoch_times_ms_fn: GnssLogEpochTimesFn,
    clean_clock_drift_fn: CleanClockDriftFn,
) -> EpochTimeContext:
    grouped_observations = {
        float(tow_ms): group
        for tow_ms, group in filtered_frame.groupby("utcTimeMillis", sort=True)
    }
    epoch_time_keys = set(grouped_observations)
    if apply_observation_mask and constellation_type == 1 and dual_frequency:
        epoch_time_keys.update(float(t) for t in gnss_log_matlab_epoch_times_ms_fn(trip_dir))

    clock_drift_context_times_ms = np.asarray(sorted(epoch_time_keys), dtype=np.float64)
    clock_drift_context_mps = None
    if clock_drift_context_times_ms.size and (clock_bias_lookup is not None or clock_drift_lookup is not None):
        full_clock_bias_m = (
            np.array(
                [clock_bias_lookup.get(int(round(float(t))), np.nan) for t in clock_drift_context_times_ms],
                dtype=np.float64,
            )
            if clock_bias_lookup is not None
            else None
        )
        full_clock_drift_mps = (
            np.array(
                [clock_drift_lookup.get(int(round(float(t))), np.nan) for t in clock_drift_context_times_ms],
                dtype=np.float64,
            )
            if clock_drift_lookup is not None
            else None
        )
        clock_drift_context_mps = clean_clock_drift_fn(
            clock_drift_context_times_ms,
            full_clock_bias_m,
            full_clock_drift_mps,
            phone_name,
        )

    return EpochTimeContext(
        grouped_observations=grouped_observations,
        epoch_time_keys=epoch_time_keys,
        clock_drift_context_times_ms=clock_drift_context_times_ms,
        clock_drift_context_mps=clock_drift_context_mps,
    )


def build_full_observation_context_stage(
    *,
    apply_observation_mask: bool,
    has_window_subset: bool,
    needs_clock_drift_context: bool,
    needs_pseudorange_isb_context: bool,
    trip_dir: Path,
    constellation_type: int,
    signal_type: str,
    weight_mode: str,
    multi_gnss: bool,
    observation_min_cn0_dbhz: float,
    observation_min_elevation_deg: float,
    dual_frequency: bool,
    factor_dt_max_s: float,
    clock_drift_context_times_ms: np.ndarray,
    clock_drift_context_mps: np.ndarray | None,
    build_trip_arrays_fn: BuildTripArraysFn,
) -> FullObservationContextProducts:
    should_build = (
        apply_observation_mask
        and has_window_subset
        and (needs_clock_drift_context or needs_pseudorange_isb_context)
    )
    if not should_build:
        return FullObservationContextProducts(
            batch=None,
            clock_drift_context_times_ms=clock_drift_context_times_ms,
            clock_drift_context_mps=clock_drift_context_mps,
            full_isb_batch=None,
        )

    batch = build_trip_arrays_fn(
        trip_dir,
        max_epochs=1_000_000_000,
        start_epoch=0,
        constellation_type=constellation_type,
        signal_type=signal_type,
        weight_mode=weight_mode,
        multi_gnss=multi_gnss,
        use_tdcp=False,
        apply_observation_mask=True,
        observation_min_cn0_dbhz=observation_min_cn0_dbhz,
        observation_min_elevation_deg=observation_min_elevation_deg,
        pseudorange_residual_mask_m=0.0,
        doppler_residual_mask_mps=0.0,
        pseudorange_doppler_mask_m=0.0,
        dual_frequency=dual_frequency,
        factor_dt_max_s=factor_dt_max_s,
    )
    if needs_clock_drift_context and getattr(batch, "clock_drift_mps", None) is not None:
        clock_drift_context_times_ms = batch.times_ms
        clock_drift_context_mps = batch.clock_drift_mps
    return FullObservationContextProducts(
        batch=batch,
        clock_drift_context_times_ms=clock_drift_context_times_ms,
        clock_drift_context_mps=clock_drift_context_mps,
        full_isb_batch=(batch if needs_pseudorange_isb_context else None),
    )


def build_absolute_height_stage(
    *,
    apply_absolute_height: bool,
    trip_dir: Path,
    kaggle_wls: np.ndarray,
    absolute_height_dist_m: float,
    load_absolute_height_reference_ecef_fn: LoadAbsoluteHeightFn,
) -> AbsoluteHeightStageProducts:
    if not apply_absolute_height:
        return AbsoluteHeightStageProducts(absolute_height_ref_ecef=None, absolute_height_ref_count=0)
    absolute_height_ref_ecef, absolute_height_ref_count = load_absolute_height_reference_ecef_fn(
        trip_dir.parent,
        kaggle_wls,
        dist_m=absolute_height_dist_m,
    )
    return AbsoluteHeightStageProducts(
        absolute_height_ref_ecef=absolute_height_ref_ecef,
        absolute_height_ref_count=absolute_height_ref_count,
    )


def apply_gnss_log_pseudorange_stage(
    *,
    trip_dir: Path,
    filtered_frame: Any,
    times_ms: np.ndarray,
    slot_keys: Sequence[Any],
    gps_tgd_m_by_svid: dict[int, float],
    rtklib_tropo_m: np.ndarray,
    sat_clock_bias_m: np.ndarray,
    phone_name: str,
    pseudorange: np.ndarray,
    pseudorange_observable: np.ndarray,
    pseudorange_bias_weights: np.ndarray,
    gnss_log_corrected_pseudorange_matrix_fn: GnssLogPseudorangeMatrixFn,
) -> GnssLogPseudorangeStageProducts:
    gnss_log_pseudorange = gnss_log_corrected_pseudorange_matrix_fn(
        trip_dir,
        filtered_frame,
        times_ms,
        tuple(slot_keys),
        gps_tgd_m_by_svid,
        rtklib_tropo_m,
        sat_clock_bias_m=sat_clock_bias_m,
        phone_name=phone_name,
    )
    if gnss_log_pseudorange is None:
        return GnssLogPseudorangeStageProducts(
            pseudorange_isb_sample_weights=pseudorange_bias_weights,
            applied_count=0,
        )

    gnss_log_pr, gnss_log_weights, gnss_log_observable = gnss_log_pseudorange
    gnss_log_valid = gnss_log_weights > 0.0
    pseudorange[gnss_log_valid] = gnss_log_pr[gnss_log_valid]
    pseudorange_observable[gnss_log_valid] = gnss_log_observable[gnss_log_valid]
    return GnssLogPseudorangeStageProducts(
        pseudorange_isb_sample_weights=gnss_log_weights,
        applied_count=int(np.count_nonzero(gnss_log_valid)),
    )


def build_pseudorange_residual_stage(
    *,
    apply_observation_mask: bool,
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    kaggle_wls: np.ndarray,
    sys_kind: np.ndarray | None,
    n_clock: int,
    slot_keys: Sequence[Any],
    pseudorange_isb_sample_weights: np.ndarray,
    pseudorange_bias_weights: np.ndarray,
    clock_bias_m: np.ndarray | None,
    pseudorange_residual_mask_m: float,
    pseudorange_residual_mask_l5_m: float | None,
    full_isb_batch: Any | None,
    slot_pseudorange_common_bias_groups_fn: SlotPseudorangeGroupsFn,
    remap_pseudorange_isb_by_group_fn: RemapPseudorangeIsbFn,
    pseudorange_global_isb_by_group_fn: PseudorangeGlobalIsbFn,
    slot_frequency_thresholds_fn: SlotFrequencyThresholdsFn,
    is_l5_signal_fn: IsL5SignalFn,
    mask_pseudorange_residual_outliers_fn: MaskPseudorangeResidualFn,
    default_l1_threshold_m: float,
    default_l5_threshold_m: float,
) -> PseudorangeResidualStageProducts:
    if not apply_observation_mask:
        return PseudorangeResidualStageProducts(pseudorange_isb_by_group=None, residual_mask_count=0)

    pseudorange_isb_by_group = None
    common_bias_group = slot_pseudorange_common_bias_groups_fn(slot_keys)
    if clock_bias_m is not None:
        full_isb = getattr(full_isb_batch, "pseudorange_isb_by_group", None) if full_isb_batch is not None else None
        if not _is_empty_group_mapping(full_isb):
            pseudorange_isb_by_group = remap_pseudorange_isb_by_group_fn(
                getattr(full_isb_batch, "slot_keys"),
                full_isb,
                slot_keys,
            )
        if _is_empty_group_mapping(pseudorange_isb_by_group):
            pseudorange_isb_by_group = pseudorange_global_isb_by_group_fn(
                sat_ecef,
                pseudorange,
                pseudorange_isb_sample_weights,
                kaggle_wls,
                clock_bias_m,
                sys_kind=sys_kind,
                common_bias_group=common_bias_group,
            )

    if pseudorange_residual_mask_l5_m is None:
        pseudorange_residual_threshold = slot_frequency_thresholds_fn(
            slot_keys,
            pseudorange_residual_mask_m,
            default_l1_threshold=default_l1_threshold_m,
            default_l5_threshold=default_l5_threshold_m,
        )
    else:
        pseudorange_residual_threshold = np.full(len(slot_keys), float(pseudorange_residual_mask_m), dtype=np.float64)
        for slot_idx, key in enumerate(slot_keys):
            if is_l5_signal_fn(str(key[2])):
                pseudorange_residual_threshold[slot_idx] = float(pseudorange_residual_mask_l5_m)

    residual_mask_count = mask_pseudorange_residual_outliers_fn(
        sat_ecef,
        pseudorange,
        weights,
        kaggle_wls,
        sys_kind=sys_kind,
        n_clock=n_clock,
        threshold_m=pseudorange_residual_threshold,
        receiver_clock_bias_m=clock_bias_m,
        common_bias_group=common_bias_group,
        common_bias_sample_weights=pseudorange_bias_weights,
        common_bias_by_group=pseudorange_isb_by_group,
    )
    return PseudorangeResidualStageProducts(
        pseudorange_isb_by_group=pseudorange_isb_by_group,
        residual_mask_count=residual_mask_count,
    )


def build_doppler_residual_stage(
    *,
    apply_observation_mask: bool,
    times_ms: np.ndarray,
    sat_ecef: np.ndarray,
    sat_vel: np.ndarray | None,
    doppler: np.ndarray | None,
    doppler_weights: np.ndarray | None,
    kaggle_wls: np.ndarray,
    doppler_residual_mask_mps: float,
    clock_drift_mps: np.ndarray | None,
    sat_clock_drift_mps: np.ndarray | None,
    baseline_velocity_times_ms: np.ndarray,
    baseline_velocity_xyz: np.ndarray,
    clock_drift_context_times_ms: np.ndarray,
    clock_drift_context_mps: np.ndarray | None,
    mask_doppler_residual_outliers_fn: MaskDopplerResidualFn,
) -> DopplerResidualStageProducts:
    if not apply_observation_mask:
        return DopplerResidualStageProducts(doppler_residual_mask_count=0)
    return DopplerResidualStageProducts(
        doppler_residual_mask_count=mask_doppler_residual_outliers_fn(
            times_ms,
            sat_ecef,
            sat_vel,
            doppler,
            doppler_weights,
            kaggle_wls,
            threshold_mps=doppler_residual_mask_mps,
            receiver_clock_drift_mps=clock_drift_mps,
            sat_clock_drift_mps=sat_clock_drift_mps,
            velocity_times_ms=baseline_velocity_times_ms,
            velocity_reference_xyz=baseline_velocity_xyz,
            clock_drift_times_ms=clock_drift_context_times_ms,
            clock_drift_reference_mps=clock_drift_context_mps,
        )
    )


def build_pseudorange_doppler_consistency_stage(
    *,
    apply_observation_mask: bool,
    times_ms: np.ndarray,
    pseudorange_observable: np.ndarray,
    weights: np.ndarray,
    doppler: np.ndarray | None,
    doppler_weights: np.ndarray | None,
    phone_name: str,
    sys_kind: np.ndarray | None,
    n_clock: int,
    slot_keys: Sequence[Any],
    pseudorange_doppler_mask_m: float,
    slot_frequency_thresholds_fn: SlotFrequencyThresholdsFn,
    mask_pseudorange_doppler_consistency_fn: MaskPseudorangeDopplerFn,
    default_l1_threshold_m: float,
    default_l5_threshold_m: float,
) -> PseudorangeDopplerStageProducts:
    if not apply_observation_mask:
        return PseudorangeDopplerStageProducts(pseudorange_doppler_mask_count=0)
    pseudorange_doppler_threshold = slot_frequency_thresholds_fn(
        slot_keys,
        pseudorange_doppler_mask_m,
        default_l1_threshold=default_l1_threshold_m,
        default_l5_threshold=default_l5_threshold_m,
    )
    return PseudorangeDopplerStageProducts(
        pseudorange_doppler_mask_count=mask_pseudorange_doppler_consistency_fn(
            times_ms,
            pseudorange_observable,
            weights,
            doppler,
            doppler_weights,
            phone=phone_name,
            sys_kind=sys_kind,
            n_clock=n_clock,
            threshold_m=pseudorange_doppler_threshold,
        )
    )


def build_observation_mask_base_correction_stage(
    *,
    apply_observation_mask: bool,
    apply_base_correction: bool,
    data_root: Path | None,
    trip: str | None,
    times_ms: np.ndarray,
    sat_ecef: np.ndarray,
    sat_vel: np.ndarray | None,
    doppler: np.ndarray | None,
    doppler_weights: np.ndarray | None,
    kaggle_wls: np.ndarray,
    pseudorange_observable: np.ndarray,
    weights: np.ndarray,
    phone_name: str,
    sys_kind: np.ndarray | None,
    n_clock: int,
    slot_keys: Sequence[Any],
    pseudorange: np.ndarray,
    pseudorange_isb_sample_weights: np.ndarray,
    pseudorange_bias_weights: np.ndarray,
    clock_bias_m: np.ndarray | None,
    clock_drift_mps: np.ndarray | None,
    sat_clock_drift_mps: np.ndarray | None,
    baseline_velocity_times_ms: np.ndarray,
    baseline_velocity_xyz: np.ndarray,
    clock_drift_context_times_ms: np.ndarray,
    clock_drift_context_mps: np.ndarray | None,
    doppler_residual_mask_mps: float,
    pseudorange_doppler_mask_m: float,
    pseudorange_residual_mask_m: float,
    pseudorange_residual_mask_l5_m: float | None,
    full_isb_batch: Any | None,
    signal_type: str,
    correction_matrix_fn: BaseCorrectionMatrixFn,
    mask_doppler_residual_outliers_fn: MaskDopplerResidualFn,
    slot_frequency_thresholds_fn: SlotFrequencyThresholdsFn,
    mask_pseudorange_doppler_consistency_fn: MaskPseudorangeDopplerFn,
    slot_pseudorange_common_bias_groups_fn: SlotPseudorangeGroupsFn,
    remap_pseudorange_isb_by_group_fn: RemapPseudorangeIsbFn,
    pseudorange_global_isb_by_group_fn: PseudorangeGlobalIsbFn,
    is_l5_signal_fn: IsL5SignalFn,
    mask_pseudorange_residual_outliers_fn: MaskPseudorangeResidualFn,
    default_pd_l1_threshold_m: float,
    default_pd_l5_threshold_m: float,
    default_pr_l1_threshold_m: float,
    default_pr_l5_threshold_m: float,
) -> ObservationMaskBaseCorrectionStageProducts:
    doppler_residual_stage = build_doppler_residual_stage(
        apply_observation_mask=apply_observation_mask,
        times_ms=times_ms,
        sat_ecef=sat_ecef,
        sat_vel=sat_vel,
        doppler=doppler,
        doppler_weights=doppler_weights,
        kaggle_wls=kaggle_wls,
        doppler_residual_mask_mps=doppler_residual_mask_mps,
        clock_drift_mps=clock_drift_mps,
        sat_clock_drift_mps=sat_clock_drift_mps,
        baseline_velocity_times_ms=baseline_velocity_times_ms,
        baseline_velocity_xyz=baseline_velocity_xyz,
        clock_drift_context_times_ms=clock_drift_context_times_ms,
        clock_drift_context_mps=clock_drift_context_mps,
        mask_doppler_residual_outliers_fn=mask_doppler_residual_outliers_fn,
    )
    pseudorange_doppler_stage = build_pseudorange_doppler_consistency_stage(
        apply_observation_mask=apply_observation_mask,
        times_ms=times_ms,
        pseudorange_observable=pseudorange_observable,
        weights=weights,
        doppler=doppler,
        doppler_weights=doppler_weights,
        phone_name=phone_name,
        sys_kind=sys_kind,
        n_clock=n_clock,
        slot_keys=slot_keys,
        pseudorange_doppler_mask_m=pseudorange_doppler_mask_m,
        slot_frequency_thresholds_fn=slot_frequency_thresholds_fn,
        mask_pseudorange_doppler_consistency_fn=mask_pseudorange_doppler_consistency_fn,
        default_l1_threshold_m=default_pd_l1_threshold_m,
        default_l5_threshold_m=default_pd_l5_threshold_m,
    )
    pseudorange_residual_stage = build_pseudorange_residual_stage(
        apply_observation_mask=apply_observation_mask,
        sat_ecef=sat_ecef,
        pseudorange=pseudorange,
        weights=weights,
        kaggle_wls=kaggle_wls,
        sys_kind=sys_kind,
        n_clock=n_clock,
        slot_keys=slot_keys,
        pseudorange_isb_sample_weights=pseudorange_isb_sample_weights,
        pseudorange_bias_weights=pseudorange_bias_weights,
        clock_bias_m=clock_bias_m,
        pseudorange_residual_mask_m=pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=pseudorange_residual_mask_l5_m,
        full_isb_batch=full_isb_batch,
        slot_pseudorange_common_bias_groups_fn=slot_pseudorange_common_bias_groups_fn,
        remap_pseudorange_isb_by_group_fn=remap_pseudorange_isb_by_group_fn,
        pseudorange_global_isb_by_group_fn=pseudorange_global_isb_by_group_fn,
        slot_frequency_thresholds_fn=slot_frequency_thresholds_fn,
        is_l5_signal_fn=is_l5_signal_fn,
        mask_pseudorange_residual_outliers_fn=mask_pseudorange_residual_outliers_fn,
        default_l1_threshold_m=default_pr_l1_threshold_m,
        default_l5_threshold_m=default_pr_l5_threshold_m,
    )

    base_correction_count = 0
    if apply_base_correction:
        base_correction_count = apply_base_correction_to_pseudorange(
            data_root=data_root,
            trip=trip,
            times_ms=times_ms,
            slot_keys=slot_keys,
            signal_type=signal_type,
            pseudorange=pseudorange,
            weights=weights,
            correction_matrix_fn=correction_matrix_fn,
        )

    return ObservationMaskBaseCorrectionStageProducts(
        doppler_residual_stage=doppler_residual_stage,
        pseudorange_doppler_stage=pseudorange_doppler_stage,
        pseudorange_residual_stage=pseudorange_residual_stage,
        base_correction_count=base_correction_count,
    )


def build_post_observation_stages(
    *,
    trip_dir: Path,
    filtered_frame: Any,
    times_ms: np.ndarray,
    slot_keys: Sequence[Any],
    gps_tgd_m_by_svid: dict[int, float],
    rtklib_tropo_m: np.ndarray,
    sat_clock_bias_m: np.ndarray,
    phone_name: str,
    pseudorange: np.ndarray,
    pseudorange_observable: np.ndarray,
    pseudorange_bias_weights: np.ndarray,
    weights: np.ndarray,
    doppler_weights: np.ndarray | None,
    apply_absolute_height: bool,
    absolute_height_dist_m: float,
    kaggle_wls: np.ndarray,
    clock_drift_blocklist_phones: set[str] | Sequence[str],
    sat_ecef: np.ndarray,
    sat_vel: np.ndarray | None,
    doppler: np.ndarray | None,
    sat_clock_drift_mps: np.ndarray | None,
    sys_kind: np.ndarray | None,
    clock_counts: np.ndarray | None,
    clock_bias_m: np.ndarray | None,
    clock_drift_mps: np.ndarray | None,
    apply_observation_mask: bool,
    has_window_subset: bool,
    constellation_type: int,
    signal_type: str,
    weight_mode: str,
    multi_gnss: bool,
    observation_min_cn0_dbhz: float,
    observation_min_elevation_deg: float,
    dual_frequency: bool,
    factor_dt_max_s: float,
    clock_drift_context_times_ms: np.ndarray,
    clock_drift_context_mps: np.ndarray | None,
    build_trip_arrays_fn: BuildTripArraysFn,
    apply_base_correction: bool,
    data_root: Path | None,
    trip: str | None,
    n_clock: int,
    baseline_velocity_times_ms: np.ndarray,
    baseline_velocity_xyz: np.ndarray,
    doppler_residual_mask_mps: float,
    pseudorange_doppler_mask_m: float,
    pseudorange_residual_mask_m: float,
    pseudorange_residual_mask_l5_m: float | None,
    tdcp_consistency_threshold_m: float,
    tdcp_loffset_m: float,
    matlab_residual_diagnostics_mask_path: Path | None,
    tdcp_geometry_correction: bool,
    tdcp_weight_scale: float,
    adr: np.ndarray | None,
    adr_state: np.ndarray | None,
    adr_uncertainty: np.ndarray | None,
    elapsed_ns: Any,
    imu_frame: str,
    gnss_log_corrected_pseudorange_matrix_fn: GnssLogPseudorangeMatrixFn,
    load_absolute_height_reference_ecef_fn: LoadAbsoluteHeightFn,
    clock_jump_from_epoch_counts_fn: ClockJumpFromEpochCountsFn,
    estimate_residual_clock_series_fn: EstimateResidualClockSeriesFn,
    combine_clock_jump_masks_fn: CombineClockJumpMasksFn,
    detect_clock_jumps_from_clock_bias_fn: DetectClockJumpsFromClockBiasFn,
    clean_clock_drift_fn: CleanClockDriftFn,
    correction_matrix_fn: BaseCorrectionMatrixFn,
    mask_doppler_residual_outliers_fn: MaskDopplerResidualFn,
    slot_frequency_thresholds_fn: SlotFrequencyThresholdsFn,
    mask_pseudorange_doppler_consistency_fn: MaskPseudorangeDopplerFn,
    slot_pseudorange_common_bias_groups_fn: SlotPseudorangeGroupsFn,
    remap_pseudorange_isb_by_group_fn: RemapPseudorangeIsbFn,
    pseudorange_global_isb_by_group_fn: PseudorangeGlobalIsbFn,
    is_l5_signal_fn: IsL5SignalFn,
    mask_pseudorange_residual_outliers_fn: MaskPseudorangeResidualFn,
    build_tdcp_arrays_fn: BuildTdcpArraysFn,
    apply_diagnostics_mask_fn: ApplyDiagnosticsMaskFn,
    apply_geometry_correction_fn: ApplyTdcpGeometryCorrectionFn,
    apply_weight_scale_fn: ApplyTdcpWeightScaleFn,
    load_device_imu_measurements_fn: LoadImuMeasurementsFn,
    process_device_imu_fn: ProcessImuFn,
    project_stop_to_epochs_fn: ProjectStopFn,
    preintegrate_processed_imu_fn: PreintegrateImuFn,
    default_pd_l1_threshold_m: float,
    default_pd_l5_threshold_m: float,
    default_pr_l1_threshold_m: float,
    default_pr_l5_threshold_m: float,
) -> PostObservationStageProducts:
    gnss_log_stage = apply_gnss_log_pseudorange_stage(
        trip_dir=trip_dir,
        filtered_frame=filtered_frame,
        times_ms=times_ms,
        slot_keys=slot_keys,
        gps_tgd_m_by_svid=gps_tgd_m_by_svid,
        rtklib_tropo_m=rtklib_tropo_m,
        sat_clock_bias_m=sat_clock_bias_m,
        phone_name=phone_name,
        pseudorange=pseudorange,
        pseudorange_observable=pseudorange_observable,
        pseudorange_bias_weights=pseudorange_bias_weights,
        gnss_log_corrected_pseudorange_matrix_fn=gnss_log_corrected_pseudorange_matrix_fn,
    )

    signal_weights = weights.copy()
    signal_doppler_weights = doppler_weights.copy() if doppler_weights is not None else None
    absolute_height_stage = build_absolute_height_stage(
        apply_absolute_height=apply_absolute_height,
        trip_dir=trip_dir,
        kaggle_wls=kaggle_wls,
        absolute_height_dist_m=absolute_height_dist_m,
        load_absolute_height_reference_ecef_fn=load_absolute_height_reference_ecef_fn,
    )
    clock_residual_stage = build_clock_residual_stage(
        phone_name=phone_name,
        clock_drift_blocklist_phones=clock_drift_blocklist_phones,
        times_ms=times_ms,
        kaggle_wls=kaggle_wls,
        sat_ecef=sat_ecef,
        pseudorange=pseudorange,
        sat_vel=sat_vel,
        doppler=doppler,
        sat_clock_drift_mps=sat_clock_drift_mps,
        sys_kind=sys_kind,
        clock_counts=clock_counts,
        clock_bias_m=clock_bias_m,
        clock_drift_mps=clock_drift_mps,
        clock_jump_from_epoch_counts_fn=clock_jump_from_epoch_counts_fn,
        estimate_residual_clock_series_fn=estimate_residual_clock_series_fn,
        combine_clock_jump_masks_fn=combine_clock_jump_masks_fn,
        detect_clock_jumps_from_clock_bias_fn=detect_clock_jumps_from_clock_bias_fn,
        clean_clock_drift_fn=clean_clock_drift_fn,
    )
    needs_clock_drift_context = phone_name.lower() in clock_drift_blocklist_phones
    full_context_stage = build_full_observation_context_stage(
        apply_observation_mask=apply_observation_mask,
        has_window_subset=has_window_subset,
        needs_clock_drift_context=needs_clock_drift_context,
        needs_pseudorange_isb_context=clock_residual_stage.clock_bias_m is not None,
        trip_dir=trip_dir,
        constellation_type=constellation_type,
        signal_type=signal_type,
        weight_mode=weight_mode,
        multi_gnss=multi_gnss,
        observation_min_cn0_dbhz=observation_min_cn0_dbhz,
        observation_min_elevation_deg=observation_min_elevation_deg,
        dual_frequency=dual_frequency,
        factor_dt_max_s=factor_dt_max_s,
        clock_drift_context_times_ms=clock_drift_context_times_ms,
        clock_drift_context_mps=clock_drift_context_mps,
        build_trip_arrays_fn=build_trip_arrays_fn,
    )
    mask_base_stage = build_observation_mask_base_correction_stage(
        apply_observation_mask=apply_observation_mask,
        apply_base_correction=apply_base_correction,
        data_root=data_root,
        trip=trip,
        times_ms=times_ms,
        sat_ecef=sat_ecef,
        sat_vel=sat_vel,
        doppler=doppler,
        doppler_weights=doppler_weights,
        kaggle_wls=kaggle_wls,
        pseudorange_observable=pseudorange_observable,
        weights=weights,
        phone_name=phone_name,
        sys_kind=sys_kind,
        n_clock=n_clock,
        slot_keys=slot_keys,
        pseudorange=pseudorange,
        pseudorange_isb_sample_weights=gnss_log_stage.pseudorange_isb_sample_weights,
        pseudorange_bias_weights=pseudorange_bias_weights,
        clock_bias_m=clock_residual_stage.clock_bias_m,
        clock_drift_mps=clock_residual_stage.clock_drift_mps,
        sat_clock_drift_mps=sat_clock_drift_mps,
        baseline_velocity_times_ms=baseline_velocity_times_ms,
        baseline_velocity_xyz=baseline_velocity_xyz,
        clock_drift_context_times_ms=full_context_stage.clock_drift_context_times_ms,
        clock_drift_context_mps=full_context_stage.clock_drift_context_mps,
        doppler_residual_mask_mps=doppler_residual_mask_mps,
        pseudorange_doppler_mask_m=pseudorange_doppler_mask_m,
        pseudorange_residual_mask_m=pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=pseudorange_residual_mask_l5_m,
        full_isb_batch=full_context_stage.full_isb_batch,
        signal_type=signal_type,
        correction_matrix_fn=correction_matrix_fn,
        mask_doppler_residual_outliers_fn=mask_doppler_residual_outliers_fn,
        slot_frequency_thresholds_fn=slot_frequency_thresholds_fn,
        mask_pseudorange_doppler_consistency_fn=mask_pseudorange_doppler_consistency_fn,
        slot_pseudorange_common_bias_groups_fn=slot_pseudorange_common_bias_groups_fn,
        remap_pseudorange_isb_by_group_fn=remap_pseudorange_isb_by_group_fn,
        pseudorange_global_isb_by_group_fn=pseudorange_global_isb_by_group_fn,
        is_l5_signal_fn=is_l5_signal_fn,
        mask_pseudorange_residual_outliers_fn=mask_pseudorange_residual_outliers_fn,
        default_pd_l1_threshold_m=default_pd_l1_threshold_m,
        default_pd_l5_threshold_m=default_pd_l5_threshold_m,
        default_pr_l1_threshold_m=default_pr_l1_threshold_m,
        default_pr_l5_threshold_m=default_pr_l5_threshold_m,
    )

    time_delta = build_graph_time_delta_products(times_ms, factor_dt_max_s)
    tdcp_stage = build_tdcp_stage(
        adr=adr,
        adr_state=adr_state,
        adr_uncertainty=adr_uncertainty,
        doppler=doppler,
        tdcp_dt=time_delta.tdcp_dt,
        tdcp_consistency_threshold_m=tdcp_consistency_threshold_m,
        doppler_weights=doppler_weights,
        clock_jump=clock_residual_stage.clock_jump,
        tdcp_loffset_m=tdcp_loffset_m,
        matlab_residual_diagnostics_mask_path=matlab_residual_diagnostics_mask_path,
        times_ms=times_ms,
        slot_keys=slot_keys,
        weights=weights,
        signal_weights=signal_weights,
        signal_doppler_weights=signal_doppler_weights,
        sat_ecef=sat_ecef,
        kaggle_wls=kaggle_wls,
        tdcp_geometry_correction=tdcp_geometry_correction,
        tdcp_weight_scale=tdcp_weight_scale,
        build_tdcp_arrays_fn=build_tdcp_arrays_fn,
        apply_diagnostics_mask_fn=apply_diagnostics_mask_fn,
        apply_geometry_correction_fn=apply_geometry_correction_fn,
        apply_weight_scale_fn=apply_weight_scale_fn,
    )
    imu_stage = build_imu_stage(
        trip_dir=trip_dir,
        times_ms=times_ms,
        elapsed_ns=elapsed_ns,
        reference_xyz_ecef=kaggle_wls,
        imu_frame=imu_frame,
        load_device_imu_measurements_fn=load_device_imu_measurements_fn,
        process_device_imu_fn=process_device_imu_fn,
        project_stop_to_epochs_fn=project_stop_to_epochs_fn,
        preintegrate_processed_imu_fn=preintegrate_processed_imu_fn,
    )
    return PostObservationStageProducts(
        gnss_log_stage=gnss_log_stage,
        absolute_height_stage=absolute_height_stage,
        clock_residual_stage=clock_residual_stage,
        full_context_stage=full_context_stage,
        mask_base_stage=mask_base_stage,
        time_delta=time_delta,
        tdcp_stage=tdcp_stage,
        imu_stage=imu_stage,
        signal_weights=signal_weights,
        signal_doppler_weights=signal_doppler_weights,
    )


def build_configured_post_observation_stages(
    *,
    observation_products: PreparedObservationProducts,
    config: PostObservationStageConfig,
    dependencies: PostObservationStageDependencies,
) -> PostObservationStageProducts:
    return build_post_observation_stages(
        trip_dir=config.trip_dir,
        filtered_frame=observation_products.filtered_frame,
        times_ms=observation_products.times_ms,
        slot_keys=observation_products.slot_keys,
        gps_tgd_m_by_svid=observation_products.gps_tgd_m_by_svid,
        rtklib_tropo_m=observation_products.rtklib_tropo_m,
        sat_clock_bias_m=observation_products.sat_clock_bias_matrix,
        phone_name=config.phone_name,
        pseudorange=observation_products.pseudorange,
        pseudorange_observable=observation_products.pseudorange_observable,
        pseudorange_bias_weights=observation_products.pseudorange_bias_weights,
        weights=observation_products.weights,
        doppler_weights=observation_products.doppler_weights,
        apply_absolute_height=config.apply_absolute_height,
        absolute_height_dist_m=config.absolute_height_dist_m,
        kaggle_wls=observation_products.kaggle_wls,
        clock_drift_blocklist_phones=config.clock_drift_blocklist_phones,
        sat_ecef=observation_products.sat_ecef,
        sat_vel=observation_products.sat_vel,
        doppler=observation_products.doppler,
        sat_clock_drift_mps=observation_products.sat_clock_drift_mps,
        sys_kind=observation_products.sys_kind,
        clock_counts=observation_products.clock_counts,
        clock_bias_m=observation_products.clock_bias_m,
        clock_drift_mps=observation_products.clock_drift_mps,
        apply_observation_mask=config.apply_observation_mask,
        has_window_subset=config.has_window_subset,
        constellation_type=config.constellation_type,
        signal_type=config.signal_type,
        weight_mode=config.weight_mode,
        multi_gnss=config.multi_gnss,
        observation_min_cn0_dbhz=config.observation_min_cn0_dbhz,
        observation_min_elevation_deg=config.observation_min_elevation_deg,
        dual_frequency=config.dual_frequency,
        factor_dt_max_s=config.factor_dt_max_s,
        clock_drift_context_times_ms=observation_products.clock_drift_context_times_ms,
        clock_drift_context_mps=observation_products.clock_drift_context_mps,
        build_trip_arrays_fn=dependencies.build_trip_arrays_fn,
        apply_base_correction=config.apply_base_correction,
        data_root=config.data_root,
        trip=config.trip,
        n_clock=observation_products.n_clock,
        baseline_velocity_times_ms=observation_products.baseline_velocity_times_ms,
        baseline_velocity_xyz=observation_products.baseline_velocity_xyz,
        doppler_residual_mask_mps=config.doppler_residual_mask_mps,
        pseudorange_doppler_mask_m=config.pseudorange_doppler_mask_m,
        pseudorange_residual_mask_m=config.pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=config.pseudorange_residual_mask_l5_m,
        tdcp_consistency_threshold_m=config.tdcp_consistency_threshold_m,
        tdcp_loffset_m=config.tdcp_loffset_m,
        matlab_residual_diagnostics_mask_path=config.matlab_residual_diagnostics_mask_path,
        tdcp_geometry_correction=config.tdcp_geometry_correction,
        tdcp_weight_scale=config.tdcp_weight_scale,
        adr=observation_products.adr,
        adr_state=observation_products.adr_state,
        adr_uncertainty=observation_products.adr_uncertainty,
        elapsed_ns=observation_products.elapsed_ns,
        imu_frame=config.imu_frame,
        gnss_log_corrected_pseudorange_matrix_fn=dependencies.gnss_log_corrected_pseudorange_matrix_fn,
        load_absolute_height_reference_ecef_fn=dependencies.load_absolute_height_reference_ecef_fn,
        clock_jump_from_epoch_counts_fn=dependencies.clock_jump_from_epoch_counts_fn,
        estimate_residual_clock_series_fn=dependencies.estimate_residual_clock_series_fn,
        combine_clock_jump_masks_fn=dependencies.combine_clock_jump_masks_fn,
        detect_clock_jumps_from_clock_bias_fn=dependencies.detect_clock_jumps_from_clock_bias_fn,
        clean_clock_drift_fn=dependencies.clean_clock_drift_fn,
        correction_matrix_fn=dependencies.correction_matrix_fn,
        mask_doppler_residual_outliers_fn=dependencies.mask_doppler_residual_outliers_fn,
        slot_frequency_thresholds_fn=dependencies.slot_frequency_thresholds_fn,
        mask_pseudorange_doppler_consistency_fn=dependencies.mask_pseudorange_doppler_consistency_fn,
        slot_pseudorange_common_bias_groups_fn=dependencies.slot_pseudorange_common_bias_groups_fn,
        remap_pseudorange_isb_by_group_fn=dependencies.remap_pseudorange_isb_by_group_fn,
        pseudorange_global_isb_by_group_fn=dependencies.pseudorange_global_isb_by_group_fn,
        is_l5_signal_fn=dependencies.is_l5_signal_fn,
        mask_pseudorange_residual_outliers_fn=dependencies.mask_pseudorange_residual_outliers_fn,
        build_tdcp_arrays_fn=dependencies.build_tdcp_arrays_fn,
        apply_diagnostics_mask_fn=dependencies.apply_diagnostics_mask_fn,
        apply_geometry_correction_fn=dependencies.apply_geometry_correction_fn,
        apply_weight_scale_fn=dependencies.apply_weight_scale_fn,
        load_device_imu_measurements_fn=dependencies.load_device_imu_measurements_fn,
        process_device_imu_fn=dependencies.process_device_imu_fn,
        project_stop_to_epochs_fn=dependencies.project_stop_to_epochs_fn,
        preintegrate_processed_imu_fn=dependencies.preintegrate_processed_imu_fn,
        default_pd_l1_threshold_m=config.default_pd_l1_threshold_m,
        default_pd_l5_threshold_m=config.default_pd_l5_threshold_m,
        default_pr_l1_threshold_m=config.default_pr_l1_threshold_m,
        default_pr_l5_threshold_m=config.default_pr_l5_threshold_m,
    )


def build_tdcp_stage(
    *,
    adr: np.ndarray | None,
    adr_state: np.ndarray | None,
    adr_uncertainty: np.ndarray | None,
    doppler: np.ndarray | None,
    tdcp_dt: np.ndarray,
    tdcp_consistency_threshold_m: float,
    doppler_weights: np.ndarray | None,
    clock_jump: np.ndarray | None,
    tdcp_loffset_m: float,
    matlab_residual_diagnostics_mask_path: Path | None,
    times_ms: np.ndarray,
    slot_keys: Sequence[Any],
    weights: np.ndarray,
    signal_weights: np.ndarray,
    signal_doppler_weights: np.ndarray | None,
    sat_ecef: np.ndarray,
    kaggle_wls: np.ndarray,
    tdcp_geometry_correction: bool,
    tdcp_weight_scale: float,
    build_tdcp_arrays_fn: BuildTdcpArraysFn,
    apply_diagnostics_mask_fn: ApplyDiagnosticsMaskFn,
    apply_geometry_correction_fn: ApplyTdcpGeometryCorrectionFn,
    apply_weight_scale_fn: ApplyTdcpWeightScaleFn,
) -> TdcpStageProducts:
    tdcp_meas = None
    tdcp_weights = None
    tdcp_consistency_mask_count = 0
    if adr is not None and adr_state is not None:
        if adr_uncertainty is None:
            adr_uncertainty = np.full_like(adr, np.nan)
        tdcp_meas, tdcp_weights, tdcp_consistency_mask_count = build_tdcp_arrays_fn(
            adr,
            adr_state,
            adr_uncertainty,
            doppler,
            tdcp_dt,
            consistency_threshold_m=tdcp_consistency_threshold_m,
            doppler_weights=doppler_weights,
            clock_jump=clock_jump,
            loffset_m=tdcp_loffset_m,
        )

    signal_tdcp_weights = tdcp_weights.copy() if tdcp_weights is not None else None
    if matlab_residual_diagnostics_mask_path is not None:
        apply_diagnostics_mask_fn(
            diagnostics_path=Path(matlab_residual_diagnostics_mask_path),
            times_ms=times_ms,
            slot_keys=slot_keys,
            weights=weights,
            signal_weights=signal_weights,
            doppler_weights=doppler_weights,
            signal_doppler_weights=signal_doppler_weights,
            tdcp_meas=tdcp_meas,
            tdcp_weights=tdcp_weights,
            signal_tdcp_weights=signal_tdcp_weights,
        )

    tdcp_geometry_correction_count = 0
    if tdcp_geometry_correction:
        tdcp_geometry_correction_count = apply_geometry_correction_fn(
            tdcp_meas,
            tdcp_weights,
            sat_ecef,
            kaggle_wls,
        )
    apply_weight_scale_fn(tdcp_weights, tdcp_weight_scale)
    return TdcpStageProducts(
        tdcp_meas=tdcp_meas,
        tdcp_weights=tdcp_weights,
        signal_tdcp_weights=signal_tdcp_weights,
        tdcp_consistency_mask_count=tdcp_consistency_mask_count,
        tdcp_geometry_correction_count=tdcp_geometry_correction_count,
    )


def build_imu_stage(
    *,
    trip_dir: Path,
    times_ms: np.ndarray,
    elapsed_ns: Any,
    reference_xyz_ecef: np.ndarray,
    imu_frame: str,
    load_device_imu_measurements_fn: LoadImuMeasurementsFn,
    process_device_imu_fn: ProcessImuFn,
    project_stop_to_epochs_fn: ProjectStopFn,
    preintegrate_processed_imu_fn: PreintegrateImuFn,
) -> ImuStageProducts:
    acc_meas, gyro_meas, _ = load_device_imu_measurements_fn(trip_dir)
    if acc_meas is None or gyro_meas is None:
        return ImuStageProducts(stop_epochs=None, imu_preintegration=None)

    try:
        acc_proc, gyro_proc, idx_stop = process_device_imu_fn(acc_meas, gyro_meas, times_ms, elapsed_ns)
        stop_epochs = project_stop_to_epochs_fn(acc_proc.times_ms, idx_stop, times_ms)
        imu_preintegration = preintegrate_processed_imu_fn(
            acc_proc,
            gyro_proc,
            times_ms,
            delta_frame=imu_frame,
            reference_xyz_ecef=reference_xyz_ecef,
        )
    except Exception:  # noqa: BLE001
        return ImuStageProducts(stop_epochs=None, imu_preintegration=None)
    return ImuStageProducts(stop_epochs=stop_epochs, imu_preintegration=imu_preintegration)


def _is_empty_group_mapping(value: Any | None) -> bool:
    if value is None:
        return True
    try:
        return len(value) == 0
    except TypeError:
        return False


def _not_missing(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(value == value)
    except TypeError:
        return False


__all__ = [
    "AbsoluteHeightStageProducts",
    "AppendGnssLogOnlyRowsFn",
    "BaseCorrectionMatrixFn",
    "ApplyDiagnosticsMaskFn",
    "ApplyTdcpGeometryCorrectionFn",
    "ApplyTdcpWeightScaleFn",
    "BuildTdcpArraysFn",
    "BuildTripArraysFn",
    "CleanClockDriftFn",
    "ClockJumpFromEpochCountsFn",
    "ClockResidualStageProducts",
    "CombineClockJumpMasksFn",
    "DetectClockJumpsFromClockBiasFn",
    "EstimateResidualClockSeriesFn",
    "FilledObservationMatrixProducts",
    "FilledObservationPostprocessProducts",
    "FillObservationMatricesFn",
    "FullObservationContextProducts",
    "EpochMetadataContext",
    "EpochTimeContext",
    "GpsMatrtklibNavMessagesForTripFn",
    "GnssLogPseudorangeMatrixFn",
    "GnssLogEpochTimesFn",
    "GnssLogPseudorangeStageProducts",
    "GpsTgdBySvidForTripFn",
    "DopplerResidualStageProducts",
    "GraphTimeDeltaProducts",
    "ImuStageProducts",
    "IsL5SignalFn",
    "LoadAbsoluteHeightFn",
    "LoadImuMeasurementsFn",
    "MaskDopplerResidualFn",
    "MatlabSignalObservationMasksFn",
    "MaskPseudorangeDopplerFn",
    "MaskPseudorangeResidualFn",
    "MultiGnssMaskFn",
    "ObservationMaskBaseCorrectionStageProducts",
    "ObservationMatrixInputProducts",
    "ObservationPreparationStageProducts",
    "PreparedObservationProducts",
    "PreintegrateImuFn",
    "ProcessImuFn",
    "PostObservationStageProducts",
    "PseudorangeGlobalIsbFn",
    "PseudorangeDopplerStageProducts",
    "PseudorangeResidualStageProducts",
    "ProjectStopFn",
    "RawObservationFrameProducts",
    "ReceiverClockBiasLookupFn",
    "RecomputeRtklibTropoMatrixFn",
    "RemapPseudorangeIsbFn",
    "RepairBaselineWlsFn",
    "SelectEpochObservationsFn",
    "SignalTypesForConstellationFn",
    "SlotFrequencyThresholdsFn",
    "SlotPseudorangeGroupsFn",
    "TdcpStageProducts",
    "TripArraysFactoryFn",
    "apply_base_correction_to_pseudorange",
    "apply_gnss_log_pseudorange_stage",
    "assemble_trip_arrays_stage",
    "assemble_prepared_trip_arrays_stage",
    "build_absolute_height_stage",
    "build_clock_residual_stage",
    "build_configured_post_observation_stages",
    "build_raw_observation_frame",
    "build_epoch_metadata_context",
    "build_epoch_time_context",
    "build_full_observation_context_stage",
    "build_doppler_residual_stage",
    "build_graph_time_delta_products",
    "build_imu_stage",
    "build_filled_observation_matrix_stage",
    "build_observation_mask_base_correction_stage",
    "build_observation_matrix_input_stage",
    "build_observation_preparation_stages",
    "build_post_observation_stages",
    "build_pseudorange_doppler_consistency_stage",
    "unpack_observation_preparation_stage",
    "build_pseudorange_residual_stage",
    "build_tdcp_stage",
    "postprocess_filled_observation_stage",
]
