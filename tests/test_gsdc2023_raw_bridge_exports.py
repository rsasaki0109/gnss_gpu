from __future__ import annotations

import importlib

import experiments.gsdc2023_raw_bridge as raw_bridge
import experiments.gsdc2023_trip_stages as trip_stages


TRIP_STAGE_PRODUCT_NAMES = (
    "AbsoluteHeightStageProducts",
    "ClockResidualStageProducts",
    "DopplerResidualStageProducts",
    "EpochMetadataContext",
    "EpochTimeContext",
    "FilledObservationMatrixProducts",
    "FilledObservationPostprocessProducts",
    "FullObservationContextProducts",
    "GnssLogPseudorangeStageProducts",
    "GraphTimeDeltaProducts",
    "ImuStageProducts",
    "ObservationMaskBaseCorrectionStageProducts",
    "ObservationMatrixInputProducts",
    "ObservationPreparationStageProducts",
    "PostObservationStageProducts",
    "PostObservationStageConfig",
    "PostObservationStageDependencies",
    "PreparedObservationProducts",
    "PseudorangeDopplerStageProducts",
    "PseudorangeResidualStageProducts",
    "RawObservationFrameProducts",
    "TdcpStageProducts",
)


TRIP_STAGE_PRIVATE_ALIASES = {
    "_apply_base_correction_to_pseudorange": "apply_base_correction_to_pseudorange",
    "_apply_gnss_log_pseudorange_stage": "apply_gnss_log_pseudorange_stage",
    "_assemble_prepared_trip_arrays_stage": "assemble_prepared_trip_arrays_stage",
    "_assemble_trip_arrays_stage": "assemble_trip_arrays_stage",
    "_build_absolute_height_stage": "build_absolute_height_stage",
    "_build_clock_residual_stage": "build_clock_residual_stage",
    "_build_configured_post_observation_stages": "build_configured_post_observation_stages",
    "_build_doppler_residual_stage": "build_doppler_residual_stage",
    "_build_epoch_metadata_context": "build_epoch_metadata_context",
    "_build_epoch_time_context": "build_epoch_time_context",
    "_build_filled_observation_matrix_stage": "build_filled_observation_matrix_stage",
    "_build_full_observation_context_stage": "build_full_observation_context_stage",
    "_build_graph_time_delta_products": "build_graph_time_delta_products",
    "_build_imu_stage": "build_imu_stage",
    "_build_observation_mask_base_correction_stage": "build_observation_mask_base_correction_stage",
    "_build_observation_matrix_input_stage": "build_observation_matrix_input_stage",
    "_build_observation_preparation_stages": "build_observation_preparation_stages",
    "_build_post_observation_stages": "build_post_observation_stages",
    "_build_pseudorange_doppler_consistency_stage": "build_pseudorange_doppler_consistency_stage",
    "_unpack_observation_preparation_stage": "unpack_observation_preparation_stage",
    "_build_pseudorange_residual_stage": "build_pseudorange_residual_stage",
    "_build_raw_observation_frame": "build_raw_observation_frame",
    "_build_tdcp_stage": "build_tdcp_stage",
    "_postprocess_filled_observation_stage": "postprocess_filled_observation_stage",
}


RAW_BRIDGE_COMPATIBILITY_ALIASES = {
    "experiments.gsdc2023_base_correction": {
        "_filter_matrtklib_duplicate_gps_nav_messages": "filter_matrtklib_duplicate_gps_nav_messages",
        "_gps_sat_clock_bias_adjustment_m": "gps_sat_clock_bias_adjustment_m",
        "_gps_tgd_m_by_svid_for_trip": "gps_tgd_m_by_svid_for_trip",
        "_matlab_base_time_span_mask": "matlab_base_time_span_mask",
        "_read_base_station_xyz": "read_base_station_xyz",
        "_select_base_pseudorange_observation": "select_base_pseudorange_observation",
        "_select_gps_nav_message": "select_gps_nav_message",
        "_signal_type_iono_scale": "signal_type_iono_scale",
        "_unix_ms_to_gps_abs_seconds": "unix_ms_to_gps_abs_seconds",
    },
    "experiments.gsdc2023_bridge_config": {
        "_should_refine_outlier_result": "should_refine_outlier_result",
    },
    "experiments.gsdc2023_clock_state": {
        "_clock_aid_enabled": "clock_aid_enabled",
        "_clock_drift_seed_enabled": "clock_drift_seed_enabled",
        "_effective_multi_gnss_enabled": "effective_multi_gnss_enabled",
        "_effective_position_source": "effective_position_source",
        "_segment_ranges": "segment_ranges",
    },
    "experiments.gsdc2023_gnss_log_bridge": {
        "_append_gnss_log_only_gps_rows": "append_gnss_log_only_gps_rows",
    },
    "experiments.gsdc2023_height_constraints": {
        "_ecef_to_enu_relative": "ecef_to_enu_relative",
        "_enu_to_ecef_relative": "enu_to_ecef_relative",
    },
    "experiments.gsdc2023_imu": {
        "_eul_xyz_to_rotm": "eul_xyz_to_rotm",
        "_imu_preintegration_segment": "imu_preintegration_segment",
    },
    "experiments.gsdc2023_observation_matrix": {
        "_receiver_clock_bias_lookup_from_epoch_meta": "receiver_clock_bias_lookup_from_epoch_meta",
        "_repair_baseline_wls": "repair_baseline_wls",
    },
    "experiments.gsdc2023_output": {
        "_export_bridge_outputs": "export_bridge_outputs",
    },
    "experiments.gsdc2023_residual_model": {
        "_estimate_residual_clock_series": "estimate_residual_clock_series",
        "_geometric_range_rate_with_sagnac": "geometric_range_rate_with_sagnac",
        "_geometric_range_with_sagnac": "geometric_range_with_sagnac",
        "_mask_doppler_residual_outliers": "mask_doppler_residual_outliers",
        "_mask_pseudorange_doppler_consistency": "mask_pseudorange_doppler_consistency",
        "_mask_pseudorange_residual_outliers": "mask_pseudorange_residual_outliers",
        "_receiver_velocity_from_reference": "receiver_velocity_from_reference",
    },
    "experiments.gsdc2023_signal_model": {
        "_multi_system_for_clock_kind": "multi_system_for_clock_kind",
    },
    "experiments.gsdc2023_tdcp": {
        "_build_tdcp_arrays": "build_tdcp_arrays",
    },
}


def test_raw_bridge_reexports_trip_stage_product_types() -> None:
    for name in TRIP_STAGE_PRODUCT_NAMES:
        assert getattr(raw_bridge, name) is getattr(trip_stages, name)
        assert name in raw_bridge.__all__


def test_raw_bridge_private_trip_stage_aliases_track_helpers() -> None:
    for alias_name, helper_name in TRIP_STAGE_PRIVATE_ALIASES.items():
        assert getattr(raw_bridge, alias_name) is getattr(trip_stages, helper_name)


def test_raw_bridge_all_public_exports_exist() -> None:
    for name in raw_bridge.__all__:
        assert getattr(raw_bridge, name) is not None


def test_raw_bridge_downstream_private_aliases_track_split_modules() -> None:
    for module_name, aliases in RAW_BRIDGE_COMPATIBILITY_ALIASES.items():
        module = importlib.import_module(module_name)
        for alias_name, helper_name in aliases.items():
            assert getattr(raw_bridge, alias_name) is getattr(module, helper_name)
