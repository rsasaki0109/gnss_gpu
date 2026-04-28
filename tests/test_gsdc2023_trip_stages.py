from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from experiments.gsdc2023_trip_stages import (
    AbsoluteHeightStageProducts,
    ClockResidualStageProducts,
    DopplerResidualStageProducts,
    EpochMetadataContext,
    EpochTimeContext,
    FilledObservationMatrixProducts,
    FilledObservationPostprocessProducts,
    FullObservationContextProducts,
    GnssLogPseudorangeStageProducts,
    GraphTimeDeltaProducts,
    ImuStageProducts,
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
    apply_base_correction_to_pseudorange,
    apply_gnss_log_pseudorange_stage,
    assemble_prepared_trip_arrays_stage,
    assemble_trip_arrays_stage,
    build_absolute_height_stage,
    build_clock_residual_stage,
    build_configured_post_observation_stages,
    build_doppler_residual_stage,
    build_epoch_metadata_context,
    build_epoch_time_context,
    build_filled_observation_matrix_stage,
    build_full_observation_context_stage,
    build_graph_time_delta_products,
    build_imu_stage,
    build_observation_mask_base_correction_stage,
    build_observation_matrix_input_stage,
    build_observation_preparation_stages,
    build_post_observation_stages,
    build_pseudorange_doppler_consistency_stage,
    build_pseudorange_residual_stage,
    build_raw_observation_frame,
    build_tdcp_stage,
    postprocess_filled_observation_stage,
    unpack_observation_preparation_stage,
)


def test_assemble_trip_arrays_stage_maps_stage_products_to_factory() -> None:
    time_delta = GraphTimeDeltaProducts(
        dt=np.array([0.0, 1.0], dtype=np.float64),
        tdcp_dt=np.array([0.0, 1.0], dtype=np.float64),
        factor_dt_gap_count=2,
    )
    tdcp_stage = TdcpStageProducts(
        tdcp_meas=np.array([[0.25]], dtype=np.float64),
        tdcp_weights=np.array([[3.0]], dtype=np.float64),
        signal_tdcp_weights=np.array([[4.0]], dtype=np.float64),
        tdcp_consistency_mask_count=5,
        tdcp_geometry_correction_count=6,
    )
    imu_stage = ImuStageProducts(
        stop_epochs=np.array([False, True]),
        imu_preintegration=object(),
    )
    absolute_height_stage = AbsoluteHeightStageProducts(
        absolute_height_ref_ecef=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        absolute_height_ref_count=7,
    )
    pseudorange_residual_stage = PseudorangeResidualStageProducts(
        pseudorange_isb_by_group={("G", "L1"): 1.5},
        residual_mask_count=8,
    )
    doppler_residual_stage = DopplerResidualStageProducts(doppler_residual_mask_count=9)
    pseudorange_doppler_stage = PseudorangeDopplerStageProducts(pseudorange_doppler_mask_count=10)

    result = assemble_trip_arrays_stage(
        trip_arrays_cls=lambda **kwargs: kwargs,
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        sat_ecef=np.ones((2, 1, 3), dtype=np.float64),
        pseudorange=np.ones((2, 1), dtype=np.float64),
        weights=np.ones((2, 1), dtype=np.float64),
        kaggle_wls=np.ones((2, 3), dtype=np.float64),
        truth=np.zeros((2, 3), dtype=np.float64),
        visible_max=4,
        has_truth=True,
        sys_kind=np.zeros((2, 1), dtype=np.int32),
        n_clock=2,
        sat_vel=np.zeros((2, 1, 3), dtype=np.float64),
        sat_clock_drift_mps=np.zeros((2, 1), dtype=np.float64),
        doppler=np.zeros((2, 1), dtype=np.float64),
        doppler_weights=np.ones((2, 1), dtype=np.float64),
        pseudorange_bias_weights=np.ones((2, 1), dtype=np.float64),
        pseudorange_residual_stage=pseudorange_residual_stage,
        time_delta=time_delta,
        tdcp_stage=tdcp_stage,
        n_sat_slots=1,
        slot_keys=["G01"],
        elapsed_ns=np.array([1.0, 2.0], dtype=np.float64),
        clock_jump=np.array([False, True]),
        clock_bias_m=np.array([0.1, 0.2], dtype=np.float64),
        clock_drift_mps=np.array([0.01, 0.02], dtype=np.float64),
        imu_stage=imu_stage,
        absolute_height_stage=absolute_height_stage,
        base_correction_count=11,
        observation_mask_count=12,
        doppler_residual_stage=doppler_residual_stage,
        pseudorange_doppler_stage=pseudorange_doppler_stage,
        dual_frequency=True,
    )

    assert result["max_sats"] == 4
    assert result["has_truth"] is True
    assert result["slot_keys"] == ("G01",)
    assert result["pseudorange_isb_by_group"] == {("G", "L1"): 1.5}
    assert result["factor_dt_gap_count"] == 2
    assert result["tdcp_consistency_mask_count"] == 5
    assert result["tdcp_geometry_correction_count"] == 6
    assert result["absolute_height_ref_count"] == 7
    assert result["residual_mask_count"] == 8
    assert result["doppler_residual_mask_count"] == 9
    assert result["pseudorange_doppler_mask_count"] == 10
    assert result["base_correction_count"] == 11
    assert result["observation_mask_count"] == 12
    assert result["dual_frequency"] is True
    np.testing.assert_array_equal(result["dt"], time_delta.dt)
    np.testing.assert_array_equal(result["tdcp_meas"], tdcp_stage.tdcp_meas)
    np.testing.assert_array_equal(result["stop_epochs"], imu_stage.stop_epochs)
    np.testing.assert_array_equal(result["absolute_height_ref_ecef"], absolute_height_stage.absolute_height_ref_ecef)


def test_assemble_prepared_trip_arrays_stage_uses_stage_products() -> None:
    observations = PreparedObservationProducts(
        filtered_frame=object(),
        gps_tgd_m_by_svid={3: 0.25},
        observation_mask_count=12,
        metadata_context=object(),
        epoch_time_context=object(),
        baseline_velocity_times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        baseline_velocity_xyz=np.zeros((2, 3), dtype=np.float64),
        clock_drift_context_times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        clock_drift_context_mps=np.array([0.1, 0.2], dtype=np.float64),
        epochs=["e0", "e1"],
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        sat_ecef=np.ones((2, 1, 3), dtype=np.float64),
        pseudorange=np.ones((2, 1), dtype=np.float64) * 20.0,
        pseudorange_observable=np.ones((2, 1), dtype=np.float64) * 21.0,
        weights=np.ones((2, 1), dtype=np.float64) * 2.0,
        pseudorange_bias_weights=np.ones((2, 1), dtype=np.float64) * 3.0,
        sat_clock_bias_matrix=np.ones((2, 1), dtype=np.float64) * 0.5,
        rtklib_tropo_m=np.ones((2, 1), dtype=np.float64) * 0.2,
        kaggle_wls=np.ones((2, 3), dtype=np.float64) * 4.0,
        truth=np.zeros((2, 3), dtype=np.float64),
        visible_max=5,
        slot_keys=["G03"],
        n_sat_slots=1,
        n_clock=2,
        elapsed_ns=np.array([10.0, 20.0], dtype=np.float64),
        sys_kind=np.zeros((2, 1), dtype=np.int32),
        clock_counts=np.array([1.0, 2.0], dtype=np.float64),
        clock_bias_m=np.array([99.0, 99.0], dtype=np.float64),
        clock_drift_mps=np.array([9.0, 9.0], dtype=np.float64),
        sat_vel=np.zeros((2, 1, 3), dtype=np.float64),
        sat_clock_drift_mps=np.zeros((2, 1), dtype=np.float64),
        doppler=np.ones((2, 1), dtype=np.float64) * -4.0,
        doppler_weights=np.ones((2, 1), dtype=np.float64) * 5.0,
        adr=np.ones((2, 1), dtype=np.float64) * 6.0,
        adr_state=np.ones((2, 1), dtype=np.int32),
        adr_uncertainty=np.ones((2, 1), dtype=np.float64) * 0.1,
    )
    post_stages = PostObservationStageProducts(
        gnss_log_stage=GnssLogPseudorangeStageProducts(
            pseudorange_isb_sample_weights=np.ones((2, 1), dtype=np.float64),
            applied_count=1,
        ),
        absolute_height_stage=AbsoluteHeightStageProducts(
            absolute_height_ref_ecef=np.ones((2, 3), dtype=np.float64),
            absolute_height_ref_count=7,
        ),
        clock_residual_stage=ClockResidualStageProducts(
            clock_jump=np.array([False, True]),
            clock_bias_m=np.array([7.0, 8.0], dtype=np.float64),
            clock_drift_mps=np.array([0.7, 0.8], dtype=np.float64),
        ),
        full_context_stage=FullObservationContextProducts(
            batch=None,
            clock_drift_context_times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
            clock_drift_context_mps=None,
            full_isb_batch=None,
        ),
        mask_base_stage=ObservationMaskBaseCorrectionStageProducts(
            doppler_residual_stage=DopplerResidualStageProducts(doppler_residual_mask_count=3),
            pseudorange_doppler_stage=PseudorangeDopplerStageProducts(pseudorange_doppler_mask_count=4),
            pseudorange_residual_stage=PseudorangeResidualStageProducts(
                pseudorange_isb_by_group={"G_L1": 1.25},
                residual_mask_count=5,
            ),
            base_correction_count=6,
        ),
        time_delta=GraphTimeDeltaProducts(
            dt=np.array([1.0, 0.0], dtype=np.float64),
            tdcp_dt=np.array([1.0, 0.0], dtype=np.float64),
            factor_dt_gap_count=2,
        ),
        tdcp_stage=TdcpStageProducts(
            tdcp_meas=np.ones((2, 1), dtype=np.float64),
            tdcp_weights=np.ones((2, 1), dtype=np.float64) * 2.0,
            signal_tdcp_weights=np.ones((2, 1), dtype=np.float64) * 3.0,
            tdcp_consistency_mask_count=8,
            tdcp_geometry_correction_count=9,
        ),
        imu_stage=ImuStageProducts(
            stop_epochs=np.array([True, False]),
            imu_preintegration=object(),
        ),
        signal_weights=np.ones((2, 1), dtype=np.float64),
        signal_doppler_weights=np.ones((2, 1), dtype=np.float64),
    )

    result = assemble_prepared_trip_arrays_stage(
        trip_arrays_cls=lambda **kwargs: kwargs,
        observation_products=observations,
        post_observation_stages=post_stages,
        has_truth=True,
        dual_frequency=True,
    )

    assert result["max_sats"] == 5
    assert result["has_truth"] is True
    assert result["slot_keys"] == ("G03",)
    assert result["pseudorange_isb_by_group"] == {"G_L1": 1.25}
    assert result["base_correction_count"] == 6
    assert result["observation_mask_count"] == 12
    assert result["absolute_height_ref_count"] == 7
    assert result["factor_dt_gap_count"] == 2
    assert result["tdcp_consistency_mask_count"] == 8
    assert result["tdcp_geometry_correction_count"] == 9
    assert result["dual_frequency"] is True
    np.testing.assert_array_equal(result["clock_jump"], [False, True])
    np.testing.assert_allclose(result["clock_bias_m"], [7.0, 8.0])
    np.testing.assert_allclose(result["clock_drift_mps"], [0.7, 0.8])
    np.testing.assert_array_equal(result["weights"], observations.weights)
    np.testing.assert_array_equal(result["tdcp_meas"], post_stages.tdcp_stage.tdcp_meas)


def test_build_clock_residual_stage_keeps_non_blocklisted_clock_and_cleans_drift() -> None:
    clock_counts = np.array([1.0, 2.0], dtype=np.float64)
    clock_jump = np.array([False, True])
    clock_bias_m = np.array([10.0, 20.0], dtype=np.float64)
    clock_drift_mps = np.array([0.1, 0.2], dtype=np.float64)

    def estimate_residual_clock_series_fn(*_args: Any, **_kwargs: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        raise AssertionError("residual clock should not be estimated for non-blocklisted phones")

    def clean_clock_drift_fn(
        times_ms: np.ndarray,
        bias_m: np.ndarray | None,
        drift_mps: np.ndarray | None,
        phone_name: str,
    ) -> np.ndarray:
        np.testing.assert_array_equal(times_ms, [1000.0, 2000.0])
        np.testing.assert_array_equal(bias_m, clock_bias_m)
        np.testing.assert_array_equal(drift_mps, clock_drift_mps)
        assert phone_name == "pixel4"
        return np.array([1.1, 1.2], dtype=np.float64)

    result = build_clock_residual_stage(
        phone_name="pixel4",
        clock_drift_blocklist_phones={"mi8"},
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        kaggle_wls=np.zeros((2, 3), dtype=np.float64),
        sat_ecef=np.zeros((2, 1, 3), dtype=np.float64),
        pseudorange=np.zeros((2, 1), dtype=np.float64),
        sat_vel=None,
        doppler=None,
        sat_clock_drift_mps=None,
        sys_kind=None,
        clock_counts=clock_counts,
        clock_bias_m=clock_bias_m,
        clock_drift_mps=clock_drift_mps,
        clock_jump_from_epoch_counts_fn=lambda counts: clock_jump if counts is clock_counts else None,
        estimate_residual_clock_series_fn=estimate_residual_clock_series_fn,
        combine_clock_jump_masks_fn=lambda *_args: np.array([True, True]),
        detect_clock_jumps_from_clock_bias_fn=lambda *_args: np.array([True, True]),
        clean_clock_drift_fn=clean_clock_drift_fn,
    )

    np.testing.assert_array_equal(result.clock_jump, clock_jump)
    np.testing.assert_array_equal(result.clock_bias_m, clock_bias_m)
    np.testing.assert_allclose(result.clock_drift_mps, [1.1, 1.2])


def test_build_clock_residual_stage_uses_residual_clock_for_blocklisted_phone() -> None:
    epoch_jump = np.array([False, False, True])
    residual_jump = np.array([False, True, False])
    residual_bias = np.array([1.0, 4.0, 9.0], dtype=np.float64)
    residual_drift = np.array([0.5, 0.6, 0.7], dtype=np.float64)

    def estimate_residual_clock_series_fn(
        times_ms: np.ndarray,
        kaggle_wls: np.ndarray,
        sat_ecef: np.ndarray,
        pseudorange: np.ndarray,
        sat_vel: np.ndarray,
        doppler: np.ndarray,
        *,
        sat_clock_drift_mps: np.ndarray,
        sys_kind: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        np.testing.assert_array_equal(times_ms, [1000.0, 2000.0, 3000.0])
        assert kaggle_wls.shape == (3, 3)
        assert sat_ecef.shape == (3, 1, 3)
        assert pseudorange.shape == (3, 1)
        assert sat_vel.shape == (3, 1, 3)
        assert doppler.shape == (3, 1)
        assert sat_clock_drift_mps.shape == (3, 1)
        assert sys_kind.shape == (3, 1)
        return residual_bias, residual_drift

    def combine_clock_jump_masks_fn(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray:
        np.testing.assert_array_equal(first, epoch_jump)
        np.testing.assert_array_equal(second, residual_jump)
        return np.asarray(first, dtype=bool) | np.asarray(second, dtype=bool)

    def clean_clock_drift_fn(
        _times_ms: np.ndarray,
        bias_m: np.ndarray | None,
        drift_mps: np.ndarray | None,
        phone_name: str,
    ) -> np.ndarray:
        np.testing.assert_array_equal(bias_m, residual_bias)
        np.testing.assert_array_equal(drift_mps, residual_drift)
        assert phone_name == "Mi8"
        return np.asarray(drift_mps, dtype=np.float64) + 10.0

    result = build_clock_residual_stage(
        phone_name="Mi8",
        clock_drift_blocklist_phones={"mi8"},
        times_ms=np.array([1000.0, 2000.0, 3000.0], dtype=np.float64),
        kaggle_wls=np.zeros((3, 3), dtype=np.float64),
        sat_ecef=np.zeros((3, 1, 3), dtype=np.float64),
        pseudorange=np.zeros((3, 1), dtype=np.float64),
        sat_vel=np.zeros((3, 1, 3), dtype=np.float64),
        doppler=np.zeros((3, 1), dtype=np.float64),
        sat_clock_drift_mps=np.zeros((3, 1), dtype=np.float64),
        sys_kind=np.zeros((3, 1), dtype=np.int32),
        clock_counts=np.array([1.0, 1.0, 2.0], dtype=np.float64),
        clock_bias_m=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        clock_drift_mps=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        clock_jump_from_epoch_counts_fn=lambda _counts: epoch_jump,
        estimate_residual_clock_series_fn=estimate_residual_clock_series_fn,
        combine_clock_jump_masks_fn=combine_clock_jump_masks_fn,
        detect_clock_jumps_from_clock_bias_fn=lambda bias_m, phone_name: residual_jump,
        clean_clock_drift_fn=clean_clock_drift_fn,
    )

    np.testing.assert_array_equal(result.clock_jump, [False, True, True])
    np.testing.assert_array_equal(result.clock_bias_m, residual_bias)
    np.testing.assert_allclose(result.clock_drift_mps, residual_drift + 10.0)


def test_apply_base_correction_to_pseudorange_subtracts_valid_weighted_entries(tmp_path: Path) -> None:
    pseudorange = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    weights = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float64)
    correction = np.array([[1.5, 2.0], [np.nan, -4.0]], dtype=np.float64)
    calls: list[tuple[Path, str, tuple[str, str], str]] = []

    def correction_matrix_fn(
        data_root: Path,
        trip: str,
        times_ms: np.ndarray,
        slot_keys: list[str],
        signal_type: str,
    ) -> np.ndarray:
        assert times_ms.tolist() == [1000.0, 2000.0]
        calls.append((data_root, trip, tuple(slot_keys), signal_type))
        return correction

    count = apply_base_correction_to_pseudorange(
        data_root=tmp_path,
        trip="train/course/phone",
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        slot_keys=["G01", "G02"],
        signal_type="GPS_L1_CA",
        pseudorange=pseudorange,
        weights=weights,
        correction_matrix_fn=correction_matrix_fn,
    )

    assert count == 2
    np.testing.assert_allclose(pseudorange, [[8.5, 20.0], [30.0, 44.0]])
    assert calls == [(tmp_path, "train/course/phone", ("G01", "G02"), "GPS_L1_CA")]


def test_apply_base_correction_to_pseudorange_requires_trip_context() -> None:
    with pytest.raises(RuntimeError, match="data_root and trip"):
        apply_base_correction_to_pseudorange(
            data_root=None,
            trip="train/course/phone",
            times_ms=np.array([1000.0], dtype=np.float64),
            slot_keys=[],
            signal_type="GPS_L1_CA",
            pseudorange=np.zeros((1, 0), dtype=np.float64),
            weights=np.zeros((1, 0), dtype=np.float64),
            correction_matrix_fn=lambda *_args: np.zeros((1, 0), dtype=np.float64),
        )


def test_build_graph_time_delta_products_keeps_tdcp_raw_positive_intervals() -> None:
    products = build_graph_time_delta_products(
        np.array([0.0, 1000.0, 2500.0, 2500.0, 50000.0], dtype=np.float64),
        factor_dt_max_s=1.5,
    )

    np.testing.assert_allclose(products.dt, [1.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(products.tdcp_dt, [1.0, 1.5, 0.0, 47.5, 0.0])
    assert products.factor_dt_gap_count == 3


def test_build_graph_time_delta_products_handles_single_epoch() -> None:
    products = build_graph_time_delta_products(np.array([1000.0], dtype=np.float64), factor_dt_max_s=1.5)

    np.testing.assert_allclose(products.dt, [0.0])
    np.testing.assert_allclose(products.tdcp_dt, [0.0])
    assert products.factor_dt_gap_count == 0


def test_build_raw_observation_frame_filters_signal_and_finite_products(tmp_path: Path) -> None:
    raw_frame = _raw_observation_frame(
        [
            {"ConstellationType": 1, "SignalType": "GPS_L1_CA", "RawPseudorangeMeters": 10.0},
            {"ConstellationType": 1, "SignalType": "GPS_L5_Q", "RawPseudorangeMeters": 11.0},
            {"ConstellationType": 3, "SignalType": "GAL_E1", "RawPseudorangeMeters": 12.0},
            {"ConstellationType": 1, "SignalType": "GPS_L1_CA", "RawPseudorangeMeters": np.nan},
        ]
    )

    products = build_raw_observation_frame(
        raw_frame,
        epoch_meta=pd.DataFrame(),
        trip_dir=tmp_path,
        phone_name="pixel4",
        constellation_type=1,
        signal_type="GPS_L1_CA",
        multi_gnss=False,
        dual_frequency=True,
        apply_observation_mask=False,
        observation_min_cn0_dbhz=18.0,
        observation_min_elevation_deg=10.0,
        multi_gnss_mask_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        signal_types_for_constellation_fn=lambda *_args, **_kwargs: ["GPS_L1_CA", "GPS_L5_Q"],
        append_gnss_log_only_gps_rows_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        matlab_signal_observation_masks_fn=lambda *_args, **_kwargs: _raise_unexpected(),
    )

    assert products.observation_mask_count == 0
    assert products.frame["SignalType"].tolist() == ["GPS_L1_CA", "GPS_L5_Q"]
    assert products.frame["RawPseudorangeMeters"].tolist() == [10.0, 11.0]


def test_build_raw_observation_frame_applies_mask_and_keeps_gnss_log_rows(tmp_path: Path) -> None:
    supplemental = tmp_path / "supplemental"
    supplemental.mkdir()
    (supplemental / "gnss_log.txt").write_text("", encoding="utf-8")
    raw_frame = _raw_observation_frame(
        [
            {"ConstellationType": 1, "SignalType": "GPS_L1_CA", "RawPseudorangeMeters": 10.0},
            {"ConstellationType": 1, "SignalType": "GPS_L5_Q", "RawPseudorangeMeters": 11.0},
        ]
    )
    append_calls: list[str] = []

    def append_fn(frame: pd.DataFrame, raw: pd.DataFrame, epoch_meta: pd.DataFrame, trip_dir: Path, **kwargs: object) -> pd.DataFrame:
        assert frame is not raw
        assert raw is raw_frame
        assert trip_dir == tmp_path
        assert kwargs == {"phone_name": "pixel4", "dual_frequency": True}
        append_calls.append("append")
        return frame

    def masks_fn(frame: pd.DataFrame, **kwargs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert kwargs == {"min_cn0_dbhz": 18.0, "min_elevation_deg": 10.0}
        assert frame["SignalType"].tolist() == ["GPS_L1_CA", "GPS_L5_Q"]
        return (
            np.array([True, False], dtype=bool),
            np.array([False, False], dtype=bool),
            np.array([False, False], dtype=bool),
        )

    products = build_raw_observation_frame(
        raw_frame,
        epoch_meta=pd.DataFrame({"utcTimeMillis": [1000]}),
        trip_dir=tmp_path,
        phone_name="pixel4",
        constellation_type=1,
        signal_type="GPS_L1_CA",
        multi_gnss=False,
        dual_frequency=True,
        apply_observation_mask=True,
        observation_min_cn0_dbhz=18.0,
        observation_min_elevation_deg=10.0,
        multi_gnss_mask_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        signal_types_for_constellation_fn=lambda *_args, **_kwargs: ["GPS_L1_CA", "GPS_L5_Q"],
        append_gnss_log_only_gps_rows_fn=append_fn,
        matlab_signal_observation_masks_fn=masks_fn,
    )

    assert append_calls == ["append"]
    assert products.observation_mask_count == 1
    assert products.frame["SignalType"].tolist() == ["GPS_L1_CA", "GPS_L5_Q"]
    assert products.frame["bridge_p_ok"].tolist() == [True, False]
    assert products.frame["bridge_d_ok"].tolist() == [False, False]
    assert products.frame["bridge_l_ok"].tolist() == [False, False]
    assert products.frame["bridge_p_bias_ok"].tolist() == [True, False]


def test_build_raw_observation_frame_appends_gnss_log_gps_rows_in_multi_gnss(tmp_path: Path) -> None:
    supplemental = tmp_path / "supplemental"
    supplemental.mkdir()
    (supplemental / "gnss_log.txt").write_text("", encoding="utf-8")
    raw_frame = _raw_observation_frame(
        [
            {"ConstellationType": 1, "SignalType": "GPS_L1_CA", "RawPseudorangeMeters": 10.0},
            {"ConstellationType": 6, "SignalType": "GAL_E1_C_P", "RawPseudorangeMeters": 12.0},
        ]
    )
    calls: list[str] = []

    def append_fn(
        frame: pd.DataFrame,
        raw: pd.DataFrame,
        epoch_meta: pd.DataFrame,
        trip_dir: Path,
        **kwargs: object,
    ) -> pd.DataFrame:
        assert frame["ConstellationType"].tolist() == [1, 6]
        assert raw is raw_frame
        assert kwargs == {"phone_name": "pixel5", "dual_frequency": True}
        calls.append("append")
        return frame

    products = build_raw_observation_frame(
        raw_frame,
        epoch_meta=pd.DataFrame({"utcTimeMillis": [1000]}),
        trip_dir=tmp_path,
        phone_name="pixel5",
        constellation_type=1,
        signal_type="GPS_L1_CA",
        multi_gnss=True,
        dual_frequency=True,
        apply_observation_mask=True,
        observation_min_cn0_dbhz=18.0,
        observation_min_elevation_deg=10.0,
        multi_gnss_mask_fn=lambda frame, **_kwargs: np.ones(len(frame), dtype=bool),
        signal_types_for_constellation_fn=lambda *_args, **_kwargs: ["GPS_L1_CA", "GPS_L5_Q"],
        append_gnss_log_only_gps_rows_fn=append_fn,
        matlab_signal_observation_masks_fn=lambda frame, **_kwargs: (
            np.ones(len(frame), dtype=bool),
            np.zeros(len(frame), dtype=bool),
            np.zeros(len(frame), dtype=bool),
        ),
    )

    assert calls == ["append"]
    assert products.frame["SignalType"].tolist() == ["GPS_L1_CA", "GAL_E1_C_P"]


def test_build_raw_observation_frame_raises_when_filtering_removes_all_rows(tmp_path: Path) -> None:
    raw_frame = _raw_observation_frame(
        [{"ConstellationType": 3, "SignalType": "GAL_E1", "RawPseudorangeMeters": 10.0}]
    )

    with pytest.raises(RuntimeError, match="No usable raw observations"):
        build_raw_observation_frame(
            raw_frame,
            epoch_meta=pd.DataFrame(),
            trip_dir=tmp_path,
            phone_name="pixel4",
            constellation_type=1,
            signal_type="GPS_L1_CA",
            multi_gnss=False,
            dual_frequency=False,
            apply_observation_mask=False,
            observation_min_cn0_dbhz=18.0,
            observation_min_elevation_deg=10.0,
            multi_gnss_mask_fn=lambda *_args, **_kwargs: _raise_unexpected(),
            signal_types_for_constellation_fn=lambda *_args, **_kwargs: ["GPS_L1_CA"],
            append_gnss_log_only_gps_rows_fn=lambda *_args, **_kwargs: _raise_unexpected(),
            matlab_signal_observation_masks_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        )


def test_build_observation_matrix_input_stage_deduplicates_and_loads_nav_products(tmp_path: Path) -> None:
    filtered_frame = pd.DataFrame(
        {
            "utcTimeMillis": [1000, 1000, 1000, 2000],
            "ConstellationType": [1, 1, 1, 1],
            "Svid": [3, 3, 3, 4],
            "SignalType": ["GPS_L1_CA", "GPS_L1_CA", "GPS_L5_Q", "GPS_L1_CA"],
            "Cn0DbHz": [20.0, 40.0, 30.0, 25.0],
            "row_id": ["low_l1", "high_l1", "l5", "next"],
        }
    )
    calls: list[tuple[str, Path]] = []

    def tgd_loader(trip_dir: Path) -> dict[int, float]:
        calls.append(("tgd", trip_dir))
        return {3: 0.25}

    def nav_loader(trip_dir: Path) -> list[str]:
        calls.append(("nav", trip_dir))
        return ["nav-message"]

    products = build_observation_matrix_input_stage(
        filtered_frame,
        trip_dir=tmp_path,
        gps_tgd_m_by_svid_for_trip_fn=tgd_loader,
        gps_matrtklib_nav_messages_for_trip_fn=nav_loader,
    )

    assert calls == [("tgd", tmp_path), ("nav", tmp_path)]
    assert products.gps_tgd_m_by_svid == {3: 0.25}
    assert products.gps_matrtklib_nav_messages == ["nav-message"]
    assert products.frame["row_id"].tolist() == ["l5", "high_l1", "next"]
    l1_rows = products.frame[products.frame["SignalType"] == "GPS_L1_CA"]
    assert l1_rows[l1_rows["Svid"] == 3]["row_id"].tolist() == ["high_l1"]


def test_build_filled_observation_matrix_stage_selects_epochs_and_forwards_fill_inputs() -> None:
    frame = pd.DataFrame({"utcTimeMillis": [1000], "ConstellationType": [1]})
    epoch_time_context = EpochTimeContext(
        grouped_observations={1000.0: frame},
        epoch_time_keys={1000.0},
        clock_drift_context_times_ms=np.array([1000.0], dtype=np.float64),
        clock_drift_context_mps=None,
    )
    metadata_context = EpochMetadataContext(
        baseline_lookup={1000: np.array([1.0, 2.0, 3.0], dtype=np.float64)},
        baseline_velocity_times_ms=np.array([1000.0], dtype=np.float64),
        baseline_velocity_xyz=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        hcdc_lookup={1000: 0.0},
        elapsed_ns_lookup={1000: 123.0},
        clock_bias_lookup={1000: 4.0},
        clock_drift_lookup={1000: 5.0},
    )
    observation_input = ObservationMatrixInputProducts(
        frame=frame,
        gps_tgd_m_by_svid={3: 0.25},
        gps_matrtklib_nav_messages=["nav"],
    )
    gt_times = np.array([1000.0], dtype=np.float64)
    gt_ecef = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    selected_epochs = ["epoch-0"]
    observations = object()
    nearest_index_fn = object()
    calls: list[str] = []

    def select_fn(
        epoch_time_keys: set[float],
        grouped_observations: dict[float, pd.DataFrame],
        baseline_lookup: dict[int, np.ndarray],
        got_gt_times: np.ndarray,
        got_gt_ecef: np.ndarray,
        **kwargs: object,
    ) -> list[str]:
        calls.append("select")
        assert epoch_time_keys == {1000.0}
        assert grouped_observations is epoch_time_context.grouped_observations
        assert baseline_lookup is metadata_context.baseline_lookup
        np.testing.assert_allclose(got_gt_times, gt_times)
        np.testing.assert_allclose(got_gt_ecef, gt_ecef)
        assert kwargs == {
            "start_epoch": 2,
            "max_epochs": 3,
            "nearest_index_fn": nearest_index_fn,
        }
        return selected_epochs

    def fill_fn(epochs: list[str], **kwargs: object) -> object:
        calls.append("fill")
        assert epochs is selected_epochs
        assert kwargs["source_columns"].tolist() == frame.columns.tolist()
        assert kwargs["baseline_lookup"] is metadata_context.baseline_lookup
        assert kwargs["weight_mode"] == "sin2el"
        assert kwargs["multi_gnss"] is True
        assert kwargs["dual_frequency"] is True
        assert kwargs["tdcp_enabled"] is True
        assert kwargs["adr_sign"] == -1.0
        assert kwargs["elapsed_ns_lookup"] is metadata_context.elapsed_ns_lookup
        assert kwargs["hcdc_lookup"] is metadata_context.hcdc_lookup
        assert kwargs["clock_bias_lookup"] is metadata_context.clock_bias_lookup
        assert kwargs["clock_drift_lookup"] is metadata_context.clock_drift_lookup
        assert kwargs["gps_tgd_m_by_svid"] == {3: 0.25}
        assert kwargs["gps_matrtklib_nav_messages"] == ["nav"]
        assert kwargs["matlab_signal_clock_dim"] == 4
        for name in (
            "gps_arrival_tow_s_from_row_fn",
            "gps_sat_clock_bias_adjustment_m_fn",
            "gps_matrtklib_sat_product_adjustment_fn",
            "clock_kind_for_observation_fn",
            "is_l5_signal_fn",
            "slot_sort_key_fn",
            "ecef_to_lla_fn",
            "elevation_azimuth_fn",
            "rtklib_tropo_fn",
        ):
            assert kwargs[name] is not None
        return observations

    products = build_filled_observation_matrix_stage(
        epoch_time_context=epoch_time_context,
        metadata_context=metadata_context,
        observation_matrix_input=observation_input,
        gt_times=gt_times,
        gt_ecef=gt_ecef,
        start_epoch=2,
        max_epochs=3,
        weight_mode="sin2el",
        multi_gnss=True,
        dual_frequency=True,
        tdcp_enabled=True,
        adr_sign=-1.0,
        select_epoch_observations_fn=select_fn,
        fill_observation_matrices_fn=fill_fn,
        nearest_index_fn=nearest_index_fn,
        gps_arrival_tow_s_from_row_fn=lambda *_args: None,
        gps_sat_clock_bias_adjustment_m_fn=lambda *_args: None,
        gps_matrtklib_sat_product_adjustment_fn=lambda *_args: None,
        clock_kind_for_observation_fn=lambda *_args: None,
        is_l5_signal_fn=lambda *_args: None,
        slot_sort_key_fn=lambda *_args: None,
        ecef_to_lla_fn=lambda *_args: None,
        elevation_azimuth_fn=lambda *_args: None,
        rtklib_tropo_fn=lambda *_args: None,
        matlab_signal_clock_dim=4,
    )

    assert calls == ["select", "fill"]
    assert products.epochs is selected_epochs
    assert products.observations is observations


def test_build_filled_observation_matrix_stage_raises_before_fill_when_no_epochs() -> None:
    frame = pd.DataFrame({"utcTimeMillis": [1000]})
    epoch_time_context = EpochTimeContext(
        grouped_observations={},
        epoch_time_keys=set(),
        clock_drift_context_times_ms=np.array([], dtype=np.float64),
        clock_drift_context_mps=None,
    )
    metadata_context = EpochMetadataContext(
        baseline_lookup={},
        baseline_velocity_times_ms=np.array([], dtype=np.float64),
        baseline_velocity_xyz=np.zeros((0, 3), dtype=np.float64),
        hcdc_lookup=None,
        elapsed_ns_lookup=None,
        clock_bias_lookup=None,
        clock_drift_lookup=None,
    )

    with pytest.raises(RuntimeError, match="No usable epochs"):
        build_filled_observation_matrix_stage(
            epoch_time_context=epoch_time_context,
            metadata_context=metadata_context,
            observation_matrix_input=ObservationMatrixInputProducts(frame=frame, gps_tgd_m_by_svid={}, gps_matrtklib_nav_messages=None),
            gt_times=np.array([], dtype=np.float64),
            gt_ecef=np.zeros((0, 3), dtype=np.float64),
            start_epoch=0,
            max_epochs=1,
            weight_mode="sin2el",
            multi_gnss=False,
            dual_frequency=False,
            tdcp_enabled=False,
            adr_sign=1.0,
            select_epoch_observations_fn=lambda *_args, **_kwargs: [],
            fill_observation_matrices_fn=lambda *_args, **_kwargs: _raise_unexpected(),
            nearest_index_fn=lambda *_args: 0,
            gps_arrival_tow_s_from_row_fn=lambda *_args: None,
            gps_sat_clock_bias_adjustment_m_fn=lambda *_args: None,
            gps_matrtklib_sat_product_adjustment_fn=lambda *_args: None,
            clock_kind_for_observation_fn=lambda *_args: None,
            is_l5_signal_fn=lambda *_args: None,
            slot_sort_key_fn=lambda *_args: None,
            ecef_to_lla_fn=lambda *_args: None,
            elevation_azimuth_fn=lambda *_args: None,
            rtklib_tropo_fn=lambda *_args: None,
            matlab_signal_clock_dim=1,
        )


def test_postprocess_filled_observation_stage_repairs_wls_then_recomputes_tropo() -> None:
    times_ms = np.array([1000.0, 2000.0], dtype=np.float64)
    kaggle_wls = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    sat_ecef = np.arange(12.0, dtype=np.float64).reshape(2, 2, 3)
    rtklib_tropo_m = np.ones((2, 2), dtype=np.float64)
    repaired_wls = kaggle_wls + 10.0
    recomputed_tropo = rtklib_tropo_m + 2.0
    ecef_to_lla_fn = object()
    elevation_azimuth_fn = object()
    rtklib_tropo_fn = object()
    calls: list[str] = []

    def repair_fn(got_times_ms: np.ndarray, got_kaggle_wls: np.ndarray) -> np.ndarray:
        calls.append("repair")
        np.testing.assert_allclose(got_times_ms, times_ms)
        np.testing.assert_allclose(got_kaggle_wls, kaggle_wls)
        return repaired_wls

    def recompute_fn(got_kaggle_wls: np.ndarray, got_sat_ecef: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append("recompute")
        np.testing.assert_allclose(got_kaggle_wls, repaired_wls)
        np.testing.assert_allclose(got_sat_ecef, sat_ecef)
        assert kwargs["ecef_to_lla_fn"] is ecef_to_lla_fn
        assert kwargs["elevation_azimuth_fn"] is elevation_azimuth_fn
        assert kwargs["rtklib_tropo_fn"] is rtklib_tropo_fn
        assert kwargs["initial_tropo_m"] is rtklib_tropo_m
        return recomputed_tropo

    products = postprocess_filled_observation_stage(
        times_ms=times_ms,
        kaggle_wls=kaggle_wls,
        sat_ecef=sat_ecef,
        rtklib_tropo_m=rtklib_tropo_m,
        repair_baseline_wls_fn=repair_fn,
        recompute_rtklib_tropo_matrix_fn=recompute_fn,
        ecef_to_lla_fn=ecef_to_lla_fn,
        elevation_azimuth_fn=elevation_azimuth_fn,
        rtklib_tropo_fn=rtklib_tropo_fn,
    )

    assert calls == ["repair", "recompute"]
    np.testing.assert_allclose(products.kaggle_wls, repaired_wls)
    np.testing.assert_allclose(products.rtklib_tropo_m, recomputed_tropo)


def test_build_observation_preparation_stages_wires_front_pipeline(tmp_path: Path) -> None:
    raw_frame = _raw_observation_frame(
        [
            {"utcTimeMillis": 2000, "Svid": 4, "Cn0DbHz": 20.0},
            {"utcTimeMillis": 1000, "Svid": 3, "Cn0DbHz": 35.0},
            {"utcTimeMillis": 1000, "Svid": 3, "Cn0DbHz": 10.0},
            {"utcTimeMillis": 1000, "ConstellationType": 3, "Svid": 7, "SignalType": "GAL_E1"},
        ]
    )
    epoch_meta = pd.DataFrame(
        {
            "utcTimeMillis": [2000, 1000],
            "WlsPositionXEcefMeters": [2.0, 1.0],
            "WlsPositionYEcefMeters": [20.0, 10.0],
            "WlsPositionZEcefMeters": [200.0, 100.0],
            "FullBiasNanos": [-1.0, -2.0],
            "DriftNanosPerSecond": [0.5, 0.25],
        }
    )
    selected_epochs = ["epoch-1000", "epoch-2000"]
    gt_times = np.array([1000.0, 2000.0], dtype=np.float64)
    gt_ecef = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    obs_times = np.array([1000.0, 2000.0], dtype=np.float64)
    obs_wls = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)
    obs_sat_ecef = np.arange(6.0, dtype=np.float64).reshape(2, 1, 3)
    obs_tropo = np.ones((2, 1), dtype=np.float64)
    ecef_to_lla_fn = object()
    elevation_azimuth_fn = object()
    rtklib_tropo_fn = object()
    nearest_index_fn = object()
    calls: list[str] = []

    @dataclass(frozen=True)
    class FakeObservations:
        times_ms: np.ndarray
        kaggle_wls: np.ndarray
        sat_ecef: np.ndarray
        rtklib_tropo_m: np.ndarray

    def repair_fn(times_ms: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        calls.append("repair")
        if len(calls) == 1:
            np.testing.assert_allclose(times_ms, [1000.0, 2000.0])
            np.testing.assert_allclose(xyz, [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
        else:
            np.testing.assert_allclose(times_ms, obs_times)
            np.testing.assert_allclose(xyz, obs_wls)
        return xyz + 100.0

    def receiver_clock_bias_fn(frame: pd.DataFrame) -> dict[int, float]:
        calls.append("clock-bias")
        assert frame is epoch_meta
        return {1000: 1.5, 2000: 2.5}

    def clean_clock_drift_fn(
        times_ms: np.ndarray,
        clock_bias_m: np.ndarray | None,
        clock_drift_mps: np.ndarray | None,
        phone_name: str,
    ) -> np.ndarray:
        calls.append("clean")
        np.testing.assert_allclose(times_ms, [1000.0, 2000.0])
        np.testing.assert_allclose(clock_bias_m, [1.5, 2.5])
        np.testing.assert_allclose(clock_drift_mps, [-0.5e-9, -1.0e-9])
        assert phone_name == "pixel4"
        return np.array([0.1, 0.2], dtype=np.float64)

    def select_fn(
        epoch_time_keys: set[float],
        grouped_observations: dict[float, pd.DataFrame],
        baseline_lookup: dict[int, np.ndarray],
        got_gt_times: np.ndarray,
        got_gt_ecef: np.ndarray,
        **kwargs: object,
    ) -> list[str]:
        calls.append("select")
        assert epoch_time_keys == {1000.0, 2000.0}
        assert sorted(grouped_observations) == [1000.0, 2000.0]
        assert sorted(baseline_lookup) == [1000, 2000]
        np.testing.assert_allclose(got_gt_times, gt_times)
        np.testing.assert_allclose(got_gt_ecef, gt_ecef)
        assert kwargs == {"start_epoch": 1, "max_epochs": 2, "nearest_index_fn": nearest_index_fn}
        return selected_epochs

    def fill_fn(epochs: list[str], **kwargs: object) -> FakeObservations:
        calls.append("fill")
        assert epochs is selected_epochs
        assert kwargs["weight_mode"] == "sin2el"
        assert kwargs["multi_gnss"] is False
        assert kwargs["dual_frequency"] is False
        assert kwargs["tdcp_enabled"] is True
        assert kwargs["adr_sign"] == -1.0
        assert kwargs["gps_tgd_m_by_svid"] == {3: 0.25}
        assert kwargs["gps_matrtklib_nav_messages"] == ["nav"]
        assert kwargs["matlab_signal_clock_dim"] == 4
        assert list(kwargs["source_columns"]).count("Cn0DbHz") == 1
        return FakeObservations(
            times_ms=obs_times,
            kaggle_wls=obs_wls,
            sat_ecef=obs_sat_ecef,
            rtklib_tropo_m=obs_tropo,
        )

    def recompute_fn(got_wls: np.ndarray, got_sat_ecef: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append("recompute")
        np.testing.assert_allclose(got_wls, obs_wls + 100.0)
        np.testing.assert_allclose(got_sat_ecef, obs_sat_ecef)
        assert kwargs["ecef_to_lla_fn"] is ecef_to_lla_fn
        assert kwargs["elevation_azimuth_fn"] is elevation_azimuth_fn
        assert kwargs["rtklib_tropo_fn"] is rtklib_tropo_fn
        assert kwargs["initial_tropo_m"] is obs_tropo
        return obs_tropo + 2.0

    products = build_observation_preparation_stages(
        raw_frame,
        epoch_meta,
        trip_dir=tmp_path,
        phone_name="pixel4",
        constellation_type=1,
        signal_type="GPS_L1_CA",
        multi_gnss=False,
        dual_frequency=False,
        apply_observation_mask=False,
        observation_min_cn0_dbhz=18.0,
        observation_min_elevation_deg=10.0,
        gt_times=gt_times,
        gt_ecef=gt_ecef,
        start_epoch=1,
        max_epochs=2,
        weight_mode="sin2el",
        tdcp_enabled=True,
        adr_sign=-1.0,
        multi_gnss_mask_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        signal_types_for_constellation_fn=lambda *_args, **_kwargs: ["GPS_L1_CA"],
        append_gnss_log_only_gps_rows_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        matlab_signal_observation_masks_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        repair_baseline_wls_fn=repair_fn,
        receiver_clock_bias_lookup_from_epoch_meta_fn=receiver_clock_bias_fn,
        light_speed_mps=2.0,
        gps_tgd_m_by_svid_for_trip_fn=lambda trip_dir: calls.append("tgd") or {3: 0.25},
        gps_matrtklib_nav_messages_for_trip_fn=lambda trip_dir: calls.append("nav") or ["nav"],
        gnss_log_matlab_epoch_times_ms_fn=lambda _trip_dir: _raise_unexpected(),
        clean_clock_drift_fn=clean_clock_drift_fn,
        select_epoch_observations_fn=select_fn,
        fill_observation_matrices_fn=fill_fn,
        nearest_index_fn=nearest_index_fn,
        gps_arrival_tow_s_from_row_fn=lambda *_args: None,
        gps_sat_clock_bias_adjustment_m_fn=lambda *_args: None,
        gps_matrtklib_sat_product_adjustment_fn=lambda *_args: None,
        clock_kind_for_observation_fn=lambda *_args: None,
        is_l5_signal_fn=lambda *_args: None,
        slot_sort_key_fn=lambda *_args: None,
        ecef_to_lla_fn=ecef_to_lla_fn,
        elevation_azimuth_fn=elevation_azimuth_fn,
        rtklib_tropo_fn=rtklib_tropo_fn,
        matlab_signal_clock_dim=4,
        recompute_rtklib_tropo_matrix_fn=recompute_fn,
    )

    assert isinstance(products, ObservationPreparationStageProducts)
    assert calls == ["repair", "clock-bias", "tgd", "nav", "clean", "select", "fill", "repair", "recompute"]
    assert products.raw_observation_frame.observation_mask_count == 0
    assert products.observation_matrix_input.frame["Cn0DbHz"].tolist() == [35.0, 20.0]
    assert products.observation_matrix_stage.epochs is selected_epochs
    np.testing.assert_allclose(products.metadata_context.baseline_velocity_xyz, [[101.0, 110.0, 200.0], [102.0, 120.0, 300.0]])
    np.testing.assert_allclose(products.epoch_time_context.clock_drift_context_mps, [0.1, 0.2])
    np.testing.assert_allclose(products.post_fill_observation.kaggle_wls, obs_wls + 100.0)
    np.testing.assert_allclose(products.post_fill_observation.rtklib_tropo_m, obs_tropo + 2.0)


def test_unpack_observation_preparation_stage_materializes_repaired_products() -> None:
    frame = pd.DataFrame({"utcTimeMillis": [1000], "SignalType": ["GPS_L1_CA"]})
    metadata_context = EpochMetadataContext(
        baseline_lookup={1000: np.array([1.0, 2.0, 3.0], dtype=np.float64)},
        baseline_velocity_times_ms=np.array([1000.0], dtype=np.float64),
        baseline_velocity_xyz=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        hcdc_lookup={1000: 0.0},
        elapsed_ns_lookup={1000: 5.0},
        clock_bias_lookup={1000: 6.0},
        clock_drift_lookup={1000: 7.0},
    )
    epoch_time_context = EpochTimeContext(
        grouped_observations={1000.0: frame},
        epoch_time_keys={1000.0},
        clock_drift_context_times_ms=np.array([1000.0], dtype=np.float64),
        clock_drift_context_mps=np.array([0.1], dtype=np.float64),
    )
    original_wls = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    repaired_wls = original_wls + 10.0
    original_tropo = np.array([[0.25]], dtype=np.float64)
    recomputed_tropo = original_tropo + 1.0
    slot_keys = ((1, 3, "GPS_L1_CA"),)

    @dataclass(frozen=True)
    class FakeObservations:
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
        slot_keys: tuple[tuple[int, int, str], ...]
        n_clock: int
        elapsed_ns: np.ndarray
        sys_kind: np.ndarray
        clock_counts: np.ndarray
        clock_bias_m: np.ndarray
        clock_drift_mps: np.ndarray
        sat_vel: np.ndarray
        sat_clock_drift_mps: np.ndarray
        doppler: np.ndarray
        doppler_weights: np.ndarray
        adr: np.ndarray
        adr_state: np.ndarray
        adr_uncertainty: np.ndarray

    observations = FakeObservations(
        times_ms=np.array([1000.0], dtype=np.float64),
        sat_ecef=np.ones((1, 1, 3), dtype=np.float64),
        pseudorange=np.array([[20.0]], dtype=np.float64),
        pseudorange_observable=np.array([[21.0]], dtype=np.float64),
        weights=np.array([[1.0]], dtype=np.float64),
        pseudorange_bias_weights=np.array([[2.0]], dtype=np.float64),
        sat_clock_bias_matrix=np.array([[0.5]], dtype=np.float64),
        rtklib_tropo_m=original_tropo,
        kaggle_wls=original_wls,
        truth=np.array([[3.0, 4.0, 5.0]], dtype=np.float64),
        visible_max=4,
        slot_keys=slot_keys,
        n_clock=2,
        elapsed_ns=np.array([9.0], dtype=np.float64),
        sys_kind=np.array([[0]], dtype=np.int32),
        clock_counts=np.array([1.0], dtype=np.float64),
        clock_bias_m=np.array([2.0], dtype=np.float64),
        clock_drift_mps=np.array([0.2], dtype=np.float64),
        sat_vel=np.ones((1, 1, 3), dtype=np.float64) * 3.0,
        sat_clock_drift_mps=np.array([[0.03]], dtype=np.float64),
        doppler=np.array([[-4.0]], dtype=np.float64),
        doppler_weights=np.array([[5.0]], dtype=np.float64),
        adr=np.array([[6.0]], dtype=np.float64),
        adr_state=np.array([[1]], dtype=np.int32),
        adr_uncertainty=np.array([[0.1]], dtype=np.float64),
    )
    stage = ObservationPreparationStageProducts(
        raw_observation_frame=RawObservationFrameProducts(frame=frame, observation_mask_count=2),
        metadata_context=metadata_context,
        observation_matrix_input=ObservationMatrixInputProducts(
            frame=frame,
            gps_tgd_m_by_svid={3: 0.25},
            gps_matrtklib_nav_messages=["nav"],
        ),
        epoch_time_context=epoch_time_context,
        observation_matrix_stage=FilledObservationMatrixProducts(epochs=["epoch"], observations=observations),
        post_fill_observation=FilledObservationPostprocessProducts(
            kaggle_wls=repaired_wls,
            rtklib_tropo_m=recomputed_tropo,
        ),
    )

    products = unpack_observation_preparation_stage(stage)

    assert isinstance(products, PreparedObservationProducts)
    assert products.filtered_frame is frame
    assert products.metadata_context is metadata_context
    assert products.epoch_time_context is epoch_time_context
    assert products.gps_tgd_m_by_svid == {3: 0.25}
    assert products.observation_mask_count == 2
    assert products.epochs == ["epoch"]
    assert products.slot_keys == list(slot_keys)
    assert products.n_sat_slots == 1
    np.testing.assert_array_equal(products.times_ms, observations.times_ms)
    np.testing.assert_array_equal(products.pseudorange, observations.pseudorange)
    np.testing.assert_array_equal(products.kaggle_wls, repaired_wls)
    np.testing.assert_array_equal(products.rtklib_tropo_m, recomputed_tropo)
    np.testing.assert_array_equal(products.baseline_velocity_times_ms, metadata_context.baseline_velocity_times_ms)
    np.testing.assert_array_equal(products.clock_drift_context_mps, epoch_time_context.clock_drift_context_mps)


def test_build_epoch_metadata_context_builds_lookups_and_repairs_sorted_velocity() -> None:
    epoch_meta = pd.DataFrame(
        {
            "utcTimeMillis": [2000, 1000, 3000],
            "WlsPositionXEcefMeters": [2.0, 1.0, np.nan],
            "WlsPositionYEcefMeters": [20.0, 10.0, 30.0],
            "WlsPositionZEcefMeters": [200.0, 100.0, 300.0],
            "HardwareClockDiscontinuityCount": [1.0, 0.0, 2.0],
            "ChipsetElapsedRealtimeNanos": [20.0, np.nan, 30.0],
            "FullBiasNanos": [-1.0, -2.0, -3.0],
            "DriftNanosPerSecond": [0.1, np.nan, -0.2],
        }
    )
    repair_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def repair_fn(times_ms: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        repair_calls.append((times_ms.copy(), xyz.copy()))
        return xyz + 1.0

    context = build_epoch_metadata_context(
        epoch_meta,
        repair_baseline_wls_fn=repair_fn,
        receiver_clock_bias_lookup_from_epoch_meta_fn=lambda _frame: {1000: 7.0, 2000: 8.0},
        light_speed_mps=10.0,
    )

    np.testing.assert_allclose(context.baseline_lookup[1000], [1.0, 10.0, 100.0])
    np.testing.assert_allclose(context.baseline_lookup[2000], [2.0, 20.0, 200.0])
    assert len(repair_calls) == 1
    np.testing.assert_allclose(repair_calls[0][0], [1000.0, 2000.0])
    np.testing.assert_allclose(repair_calls[0][1], [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
    np.testing.assert_allclose(context.baseline_velocity_times_ms, [1000.0, 2000.0])
    np.testing.assert_allclose(context.baseline_velocity_xyz, [[2.0, 11.0, 101.0], [3.0, 21.0, 201.0]])
    assert context.hcdc_lookup == {2000: 1.0, 1000: 0.0, 3000: 2.0}
    assert context.elapsed_ns_lookup == {2000: 20.0, 3000: 30.0}
    assert context.clock_bias_lookup == {1000: 7.0, 2000: 8.0}
    assert context.clock_drift_lookup is not None
    np.testing.assert_allclose(context.clock_drift_lookup[2000], -1e-9)
    np.testing.assert_allclose(context.clock_drift_lookup[3000], 2e-9)


def test_build_epoch_metadata_context_handles_missing_optional_columns() -> None:
    epoch_meta = pd.DataFrame(
        {
            "utcTimeMillis": [1000],
            "WlsPositionXEcefMeters": [1.0],
            "WlsPositionYEcefMeters": [2.0],
            "WlsPositionZEcefMeters": [3.0],
        }
    )

    context = build_epoch_metadata_context(
        epoch_meta,
        repair_baseline_wls_fn=lambda _times, xyz: xyz,
        receiver_clock_bias_lookup_from_epoch_meta_fn=lambda _frame: _raise_unexpected(),
        light_speed_mps=10.0,
    )

    assert context.hcdc_lookup is None
    assert context.elapsed_ns_lookup is None
    assert context.clock_bias_lookup is None
    assert context.clock_drift_lookup is None


def test_build_epoch_time_context_groups_epochs_and_cleans_clock_drift(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "utcTimeMillis": [2000, 1000, 2000],
            "value": [2, 1, 3],
        }
    )
    calls: list[str] = []

    def clean_fn(
        times_ms: np.ndarray,
        clock_bias_m: np.ndarray | None,
        clock_drift_mps: np.ndarray | None,
        phone_name: str,
    ) -> np.ndarray:
        np.testing.assert_allclose(times_ms, [1000.0, 1500.0, 2000.0])
        np.testing.assert_allclose(clock_bias_m, [5.0, np.nan, 6.0])
        np.testing.assert_allclose(clock_drift_mps, [np.nan, 0.2, 0.1])
        assert phone_name == "pixel4"
        calls.append("clean")
        return np.array([0.5, 0.6, 0.7], dtype=np.float64)

    context = build_epoch_time_context(
        frame,
        apply_observation_mask=True,
        multi_gnss=False,
        constellation_type=1,
        dual_frequency=True,
        trip_dir=tmp_path,
        clock_bias_lookup={1000: 5.0, 2000: 6.0},
        clock_drift_lookup={1500: 0.2, 2000: 0.1},
        phone_name="pixel4",
        gnss_log_matlab_epoch_times_ms_fn=lambda _trip_dir: np.array([1500.0], dtype=np.float64),
        clean_clock_drift_fn=clean_fn,
    )

    assert sorted(context.grouped_observations) == [1000.0, 2000.0]
    assert context.epoch_time_keys == {1000.0, 1500.0, 2000.0}
    np.testing.assert_allclose(context.clock_drift_context_times_ms, [1000.0, 1500.0, 2000.0])
    np.testing.assert_allclose(context.clock_drift_context_mps, [0.5, 0.6, 0.7])
    assert calls == ["clean"]


def test_build_epoch_time_context_merges_gnss_log_epochs_in_multi_gnss(tmp_path: Path) -> None:
    context = build_epoch_time_context(
        pd.DataFrame({"utcTimeMillis": [1000], "value": [1]}),
        apply_observation_mask=True,
        multi_gnss=True,
        constellation_type=1,
        dual_frequency=True,
        trip_dir=tmp_path,
        clock_bias_lookup=None,
        clock_drift_lookup=None,
        phone_name="pixel5",
        gnss_log_matlab_epoch_times_ms_fn=lambda _trip_dir: np.array([1500.0], dtype=np.float64),
        clean_clock_drift_fn=lambda *_args: _raise_unexpected(),
    )

    assert context.epoch_time_keys == {1000.0, 1500.0}
    np.testing.assert_allclose(context.clock_drift_context_times_ms, [1000.0, 1500.0])


def test_build_epoch_time_context_skips_clock_clean_without_lookup(tmp_path: Path) -> None:
    context = build_epoch_time_context(
        pd.DataFrame({"utcTimeMillis": [1000], "value": [1]}),
        apply_observation_mask=False,
        multi_gnss=True,
        constellation_type=1,
        dual_frequency=True,
        trip_dir=tmp_path,
        clock_bias_lookup=None,
        clock_drift_lookup=None,
        phone_name="pixel4",
        gnss_log_matlab_epoch_times_ms_fn=lambda _trip_dir: _raise_unexpected(),
        clean_clock_drift_fn=lambda *_args: _raise_unexpected(),
    )

    assert context.epoch_time_keys == {1000.0}
    np.testing.assert_allclose(context.clock_drift_context_times_ms, [1000.0])
    assert context.clock_drift_context_mps is None


def test_build_full_observation_context_stage_skips_when_not_needed(tmp_path: Path) -> None:
    fallback_times = np.array([1000.0], dtype=np.float64)
    fallback_drift = np.array([0.1], dtype=np.float64)

    products = build_full_observation_context_stage(
        apply_observation_mask=True,
        has_window_subset=False,
        needs_clock_drift_context=True,
        needs_pseudorange_isb_context=True,
        trip_dir=tmp_path,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=False,
        observation_min_cn0_dbhz=18.0,
        observation_min_elevation_deg=10.0,
        dual_frequency=False,
        factor_dt_max_s=1.5,
        clock_drift_context_times_ms=fallback_times,
        clock_drift_context_mps=fallback_drift,
        build_trip_arrays_fn=lambda *_args, **_kwargs: _raise_unexpected(),
    )

    assert products.batch is None
    assert products.full_isb_batch is None
    assert products.clock_drift_context_times_ms is fallback_times
    assert products.clock_drift_context_mps is fallback_drift


def test_build_full_observation_context_stage_builds_once_and_updates_context(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Batch:
        times_ms: np.ndarray
        clock_drift_mps: np.ndarray
        pseudorange_isb_by_group: dict[int, float]
        slot_keys: tuple[str, ...]

    batch = Batch(
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        clock_drift_mps=np.array([0.2, 0.3], dtype=np.float64),
        pseudorange_isb_by_group={0: 5.0},
        slot_keys=("G01",),
    )
    calls: list[tuple[Path, dict[str, Any]]] = []

    def build_fn(trip_dir: Path, **kwargs: Any) -> Batch:
        calls.append((trip_dir, kwargs))
        return batch

    products = build_full_observation_context_stage(
        apply_observation_mask=True,
        has_window_subset=True,
        needs_clock_drift_context=True,
        needs_pseudorange_isb_context=True,
        trip_dir=tmp_path,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=False,
        observation_min_cn0_dbhz=18.0,
        observation_min_elevation_deg=10.0,
        dual_frequency=True,
        factor_dt_max_s=1.5,
        clock_drift_context_times_ms=np.array([500.0], dtype=np.float64),
        clock_drift_context_mps=np.array([0.1], dtype=np.float64),
        build_trip_arrays_fn=build_fn,
    )

    assert products.batch is batch
    assert products.full_isb_batch is batch
    assert products.clock_drift_context_times_ms is batch.times_ms
    assert products.clock_drift_context_mps is batch.clock_drift_mps
    assert len(calls) == 1
    trip_dir, kwargs = calls[0]
    assert trip_dir == tmp_path
    assert kwargs["max_epochs"] == 1_000_000_000
    assert kwargs["start_epoch"] == 0
    assert kwargs["constellation_type"] == 1
    assert kwargs["signal_type"] == "GPS_L1_CA"
    assert kwargs["weight_mode"] == "sin2el"
    assert kwargs["multi_gnss"] is False
    assert kwargs["use_tdcp"] is False
    assert kwargs["apply_observation_mask"] is True
    assert kwargs["pseudorange_residual_mask_m"] == 0.0
    assert kwargs["doppler_residual_mask_mps"] == 0.0
    assert kwargs["pseudorange_doppler_mask_m"] == 0.0
    assert kwargs["dual_frequency"] is True
    assert kwargs["factor_dt_max_s"] == 1.5


def test_build_full_observation_context_stage_keeps_fallback_drift_when_batch_has_none(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Batch:
        times_ms: np.ndarray
        clock_drift_mps: None

    fallback_times = np.array([500.0], dtype=np.float64)
    fallback_drift = np.array([0.1], dtype=np.float64)
    batch = Batch(times_ms=np.array([1000.0], dtype=np.float64), clock_drift_mps=None)

    products = build_full_observation_context_stage(
        apply_observation_mask=True,
        has_window_subset=True,
        needs_clock_drift_context=True,
        needs_pseudorange_isb_context=False,
        trip_dir=tmp_path,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=False,
        observation_min_cn0_dbhz=18.0,
        observation_min_elevation_deg=10.0,
        dual_frequency=False,
        factor_dt_max_s=1.5,
        clock_drift_context_times_ms=fallback_times,
        clock_drift_context_mps=fallback_drift,
        build_trip_arrays_fn=lambda *_args, **_kwargs: batch,
    )

    assert products.batch is batch
    assert products.full_isb_batch is None
    assert products.clock_drift_context_times_ms is fallback_times
    assert products.clock_drift_context_mps is fallback_drift


def test_build_absolute_height_stage_uses_loader_when_enabled(tmp_path: Path) -> None:
    ref = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    kaggle_wls = np.zeros((1, 3), dtype=np.float64)
    calls: list[tuple[Path, float]] = []

    def loader(path: Path, got_kaggle_wls: np.ndarray, *, dist_m: float) -> tuple[np.ndarray, int]:
        assert path == tmp_path
        np.testing.assert_allclose(got_kaggle_wls, kaggle_wls)
        calls.append((path, dist_m))
        return ref, 1

    products = build_absolute_height_stage(
        apply_absolute_height=True,
        trip_dir=tmp_path / "pixel4",
        kaggle_wls=kaggle_wls,
        absolute_height_dist_m=15.0,
        load_absolute_height_reference_ecef_fn=loader,
    )

    assert products.absolute_height_ref_ecef is ref
    assert products.absolute_height_ref_count == 1
    assert calls == [(tmp_path, 15.0)]


def test_build_absolute_height_stage_skips_loader_when_disabled(tmp_path: Path) -> None:
    products = build_absolute_height_stage(
        apply_absolute_height=False,
        trip_dir=tmp_path / "pixel4",
        kaggle_wls=np.zeros((1, 3), dtype=np.float64),
        absolute_height_dist_m=15.0,
        load_absolute_height_reference_ecef_fn=lambda *_args, **_kwargs: _raise_unexpected(),
    )

    assert products.absolute_height_ref_ecef is None
    assert products.absolute_height_ref_count == 0


def test_apply_gnss_log_pseudorange_stage_overlays_valid_products(tmp_path: Path) -> None:
    pseudorange = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    pseudorange_observable = np.array([[11.0, 21.0], [31.0, 41.0]], dtype=np.float64)
    pseudorange_bias_weights = np.ones((2, 2), dtype=np.float64)
    gnss_weights = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float64)
    calls: list[tuple[Path, tuple[str, str], str]] = []

    def gnss_log_fn(
        trip_dir: Path,
        filtered_frame: str,
        times_ms: np.ndarray,
        slot_keys: tuple[str, str],
        gps_tgd_m_by_svid: dict[int, float],
        rtklib_tropo_m: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert filtered_frame == "frame"
        np.testing.assert_allclose(times_ms, [1000.0, 2000.0])
        assert gps_tgd_m_by_svid == {1: 0.5}
        np.testing.assert_allclose(rtklib_tropo_m, [[0.1, 0.2], [0.3, 0.4]])
        np.testing.assert_allclose(kwargs["sat_clock_bias_m"], [[1.0, 1.0], [1.0, 1.0]])
        assert kwargs["phone_name"] == "pixel4"
        calls.append((trip_dir, slot_keys, kwargs["phone_name"]))
        return (
            np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64),
            gnss_weights,
            np.array([[101.0, 201.0], [301.0, 401.0]], dtype=np.float64),
        )

    products = apply_gnss_log_pseudorange_stage(
        trip_dir=tmp_path,
        filtered_frame="frame",
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        slot_keys=["G01", "G02"],
        gps_tgd_m_by_svid={1: 0.5},
        rtklib_tropo_m=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        sat_clock_bias_m=np.ones((2, 2), dtype=np.float64),
        phone_name="pixel4",
        pseudorange=pseudorange,
        pseudorange_observable=pseudorange_observable,
        pseudorange_bias_weights=pseudorange_bias_weights,
        gnss_log_corrected_pseudorange_matrix_fn=gnss_log_fn,
    )

    np.testing.assert_allclose(pseudorange, [[10.0, 200.0], [300.0, 40.0]])
    np.testing.assert_allclose(pseudorange_observable, [[11.0, 201.0], [301.0, 41.0]])
    assert products.pseudorange_isb_sample_weights is gnss_weights
    assert products.applied_count == 2
    assert calls == [(tmp_path, ("G01", "G02"), "pixel4")]


def test_apply_gnss_log_pseudorange_stage_keeps_bias_weights_without_products(tmp_path: Path) -> None:
    pseudorange = np.array([[10.0]], dtype=np.float64)
    pseudorange_observable = np.array([[11.0]], dtype=np.float64)
    pseudorange_bias_weights = np.ones((1, 1), dtype=np.float64)

    products = apply_gnss_log_pseudorange_stage(
        trip_dir=tmp_path,
        filtered_frame=None,
        times_ms=np.array([1000.0], dtype=np.float64),
        slot_keys=["G01"],
        gps_tgd_m_by_svid={},
        rtklib_tropo_m=np.zeros((1, 1), dtype=np.float64),
        sat_clock_bias_m=np.zeros((1, 1), dtype=np.float64),
        phone_name="pixel4",
        pseudorange=pseudorange,
        pseudorange_observable=pseudorange_observable,
        pseudorange_bias_weights=pseudorange_bias_weights,
        gnss_log_corrected_pseudorange_matrix_fn=lambda *_args, **_kwargs: None,
    )

    np.testing.assert_allclose(pseudorange, [[10.0]])
    np.testing.assert_allclose(pseudorange_observable, [[11.0]])
    assert products.pseudorange_isb_sample_weights is pseudorange_bias_weights
    assert products.applied_count == 0


def test_build_pseudorange_residual_stage_uses_full_context_and_custom_l5_threshold() -> None:
    @dataclass(frozen=True)
    class FullBatch:
        slot_keys: tuple[tuple[str, int, str], ...]
        pseudorange_isb_by_group: dict[str, float]

    slot_keys = [("G", 1, "L1"), ("G", 2, "L5")]
    full_batch = FullBatch(
        slot_keys=(("G", 1, "L1"), ("G", 2, "L5")),
        pseudorange_isb_by_group={"old": 4.0},
    )
    calls: list[str] = []

    def groups_fn(keys: list[tuple[str, int, str]]) -> np.ndarray:
        assert keys == slot_keys
        calls.append("groups")
        return np.array([0, 1], dtype=np.int32)

    def remap_fn(old_keys: tuple[tuple[str, int, str], ...], old_isb: dict[str, float], new_keys: list[tuple[str, int, str]]) -> dict[str, float]:
        assert old_keys == full_batch.slot_keys
        assert old_isb == {"old": 4.0}
        assert new_keys == slot_keys
        calls.append("remap")
        return {"new": 5.0}

    def global_isb_fn(*_args: object, **_kwargs: object) -> dict[str, float]:
        calls.append("global")
        return {"global": 1.0}

    def mask_fn(*_args: object, **kwargs: Any) -> int:
        np.testing.assert_allclose(kwargs["threshold_m"], [20.0, 15.0])
        np.testing.assert_array_equal(kwargs["common_bias_group"], [0, 1])
        assert kwargs["common_bias_by_group"] == {"new": 5.0}
        calls.append("mask")
        return 9

    products = build_pseudorange_residual_stage(
        apply_observation_mask=True,
        sat_ecef=np.zeros((1, 2, 3), dtype=np.float64),
        pseudorange=np.zeros((1, 2), dtype=np.float64),
        weights=np.ones((1, 2), dtype=np.float64),
        kaggle_wls=np.zeros((1, 3), dtype=np.float64),
        sys_kind=None,
        n_clock=1,
        slot_keys=slot_keys,
        pseudorange_isb_sample_weights=np.ones((1, 2), dtype=np.float64),
        pseudorange_bias_weights=np.ones((1, 2), dtype=np.float64),
        clock_bias_m=np.array([3.0], dtype=np.float64),
        pseudorange_residual_mask_m=20.0,
        pseudorange_residual_mask_l5_m=15.0,
        full_isb_batch=full_batch,
        slot_pseudorange_common_bias_groups_fn=groups_fn,
        remap_pseudorange_isb_by_group_fn=remap_fn,
        pseudorange_global_isb_by_group_fn=global_isb_fn,
        slot_frequency_thresholds_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        is_l5_signal_fn=lambda signal: signal == "L5",
        mask_pseudorange_residual_outliers_fn=mask_fn,
        default_l1_threshold_m=20.0,
        default_l5_threshold_m=15.0,
    )

    assert products.pseudorange_isb_by_group == {"new": 5.0}
    assert products.residual_mask_count == 9
    assert calls == ["groups", "remap", "mask"]


def test_build_pseudorange_residual_stage_uses_global_isb_and_default_thresholds() -> None:
    calls: list[str] = []

    def threshold_fn(slot_keys: list[str], threshold: float, **kwargs: float) -> np.ndarray:
        assert slot_keys == ["G01", "G02"]
        assert threshold == 25.0
        assert kwargs == {"default_l1_threshold": 20.0, "default_l5_threshold": 15.0}
        calls.append("threshold")
        return np.array([20.0, 15.0], dtype=np.float64)

    def global_isb_fn(*_args: object, **kwargs: object) -> dict[int, float]:
        np.testing.assert_array_equal(kwargs["common_bias_group"], [0, 0])
        calls.append("global")
        return {0: 2.0}

    def mask_fn(*_args: object, **kwargs: Any) -> int:
        np.testing.assert_allclose(kwargs["threshold_m"], [20.0, 15.0])
        assert kwargs["common_bias_by_group"] == {0: 2.0}
        calls.append("mask")
        return 4

    products = build_pseudorange_residual_stage(
        apply_observation_mask=True,
        sat_ecef=np.zeros((1, 2, 3), dtype=np.float64),
        pseudorange=np.zeros((1, 2), dtype=np.float64),
        weights=np.ones((1, 2), dtype=np.float64),
        kaggle_wls=np.zeros((1, 3), dtype=np.float64),
        sys_kind=None,
        n_clock=1,
        slot_keys=["G01", "G02"],
        pseudorange_isb_sample_weights=np.ones((1, 2), dtype=np.float64),
        pseudorange_bias_weights=np.ones((1, 2), dtype=np.float64),
        clock_bias_m=np.array([3.0], dtype=np.float64),
        pseudorange_residual_mask_m=25.0,
        pseudorange_residual_mask_l5_m=None,
        full_isb_batch=None,
        slot_pseudorange_common_bias_groups_fn=lambda _keys: np.array([0, 0], dtype=np.int32),
        remap_pseudorange_isb_by_group_fn=lambda *_args: _raise_unexpected(),
        pseudorange_global_isb_by_group_fn=global_isb_fn,
        slot_frequency_thresholds_fn=threshold_fn,
        is_l5_signal_fn=lambda _signal: _raise_unexpected(),
        mask_pseudorange_residual_outliers_fn=mask_fn,
        default_l1_threshold_m=20.0,
        default_l5_threshold_m=15.0,
    )

    assert products.pseudorange_isb_by_group == {0: 2.0}
    assert products.residual_mask_count == 4
    assert calls == ["global", "threshold", "mask"]


def test_build_pseudorange_residual_stage_skips_when_observation_mask_disabled() -> None:
    products = build_pseudorange_residual_stage(
        apply_observation_mask=False,
        sat_ecef=np.zeros((1, 0, 3), dtype=np.float64),
        pseudorange=np.zeros((1, 0), dtype=np.float64),
        weights=np.zeros((1, 0), dtype=np.float64),
        kaggle_wls=np.zeros((1, 3), dtype=np.float64),
        sys_kind=None,
        n_clock=1,
        slot_keys=[],
        pseudorange_isb_sample_weights=np.zeros((1, 0), dtype=np.float64),
        pseudorange_bias_weights=np.zeros((1, 0), dtype=np.float64),
        clock_bias_m=None,
        pseudorange_residual_mask_m=20.0,
        pseudorange_residual_mask_l5_m=None,
        full_isb_batch=None,
        slot_pseudorange_common_bias_groups_fn=lambda *_args: _raise_unexpected(),
        remap_pseudorange_isb_by_group_fn=lambda *_args: _raise_unexpected(),
        pseudorange_global_isb_by_group_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        slot_frequency_thresholds_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        is_l5_signal_fn=lambda *_args: _raise_unexpected(),
        mask_pseudorange_residual_outliers_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        default_l1_threshold_m=20.0,
        default_l5_threshold_m=15.0,
    )

    assert products.pseudorange_isb_by_group is None
    assert products.residual_mask_count == 0


def test_build_doppler_residual_stage_forwards_context() -> None:
    times_ms = np.array([1000.0, 2000.0], dtype=np.float64)
    sat_ecef = np.zeros((2, 1, 3), dtype=np.float64)
    sat_vel = np.ones((2, 1, 3), dtype=np.float64)
    doppler = np.array([[1.0], [2.0]], dtype=np.float64)
    doppler_weights = np.ones((2, 1), dtype=np.float64)
    kaggle_wls = np.zeros((2, 3), dtype=np.float64)
    clock_drift_mps = np.array([0.1, 0.2], dtype=np.float64)
    sat_clock_drift_mps = np.array([[0.3], [0.4]], dtype=np.float64)
    velocity_times_ms = np.array([900.0, 2100.0], dtype=np.float64)
    velocity_xyz = np.ones((2, 3), dtype=np.float64)
    context_times_ms = np.array([800.0, 2200.0], dtype=np.float64)
    context_drift = np.array([0.5, 0.6], dtype=np.float64)

    def mask_fn(*args: object, **kwargs: Any) -> int:
        assert args[0] is times_ms
        assert args[1] is sat_ecef
        assert args[2] is sat_vel
        assert args[3] is doppler
        assert args[4] is doppler_weights
        assert args[5] is kaggle_wls
        assert kwargs["threshold_mps"] == 3.0
        assert kwargs["receiver_clock_drift_mps"] is clock_drift_mps
        assert kwargs["sat_clock_drift_mps"] is sat_clock_drift_mps
        assert kwargs["velocity_times_ms"] is velocity_times_ms
        assert kwargs["velocity_reference_xyz"] is velocity_xyz
        assert kwargs["clock_drift_times_ms"] is context_times_ms
        assert kwargs["clock_drift_reference_mps"] is context_drift
        return 6

    products = build_doppler_residual_stage(
        apply_observation_mask=True,
        times_ms=times_ms,
        sat_ecef=sat_ecef,
        sat_vel=sat_vel,
        doppler=doppler,
        doppler_weights=doppler_weights,
        kaggle_wls=kaggle_wls,
        doppler_residual_mask_mps=3.0,
        clock_drift_mps=clock_drift_mps,
        sat_clock_drift_mps=sat_clock_drift_mps,
        baseline_velocity_times_ms=velocity_times_ms,
        baseline_velocity_xyz=velocity_xyz,
        clock_drift_context_times_ms=context_times_ms,
        clock_drift_context_mps=context_drift,
        mask_doppler_residual_outliers_fn=mask_fn,
    )

    assert products.doppler_residual_mask_count == 6


def test_build_doppler_residual_stage_skips_when_observation_mask_disabled() -> None:
    products = build_doppler_residual_stage(
        apply_observation_mask=False,
        times_ms=np.array([1000.0], dtype=np.float64),
        sat_ecef=np.zeros((1, 0, 3), dtype=np.float64),
        sat_vel=None,
        doppler=None,
        doppler_weights=None,
        kaggle_wls=np.zeros((1, 3), dtype=np.float64),
        doppler_residual_mask_mps=3.0,
        clock_drift_mps=None,
        sat_clock_drift_mps=None,
        baseline_velocity_times_ms=np.array([], dtype=np.float64),
        baseline_velocity_xyz=np.zeros((0, 3), dtype=np.float64),
        clock_drift_context_times_ms=np.array([], dtype=np.float64),
        clock_drift_context_mps=None,
        mask_doppler_residual_outliers_fn=lambda *_args, **_kwargs: _raise_unexpected(),
    )

    assert products.doppler_residual_mask_count == 0


def test_build_pseudorange_doppler_consistency_stage_builds_frequency_thresholds() -> None:
    times_ms = np.array([1000.0, 2000.0], dtype=np.float64)
    pseudorange_observable = np.ones((2, 2), dtype=np.float64)
    weights = np.ones((2, 2), dtype=np.float64)
    doppler = np.ones((2, 2), dtype=np.float64)
    doppler_weights = np.ones((2, 2), dtype=np.float64)
    sys_kind = np.array([[0, 4], [0, 4]], dtype=np.int32)
    slot_keys = ["G01_L1", "G01_L5"]
    calls: list[str] = []

    def threshold_fn(keys: list[str], threshold: float, **kwargs: float) -> np.ndarray:
        assert keys == slot_keys
        assert threshold == 40.0
        assert kwargs == {"default_l1_threshold": 40.0, "default_l5_threshold": 25.0}
        calls.append("threshold")
        return np.array([40.0, 25.0], dtype=np.float64)

    def mask_fn(*args: object, **kwargs: Any) -> int:
        assert args[0] is times_ms
        assert args[1] is pseudorange_observable
        assert args[2] is weights
        assert args[3] is doppler
        assert args[4] is doppler_weights
        assert kwargs["phone"] == "pixel4"
        assert kwargs["sys_kind"] is sys_kind
        assert kwargs["n_clock"] == 7
        np.testing.assert_allclose(kwargs["threshold_m"], [40.0, 25.0])
        calls.append("mask")
        return 8

    products = build_pseudorange_doppler_consistency_stage(
        apply_observation_mask=True,
        times_ms=times_ms,
        pseudorange_observable=pseudorange_observable,
        weights=weights,
        doppler=doppler,
        doppler_weights=doppler_weights,
        phone_name="pixel4",
        sys_kind=sys_kind,
        n_clock=7,
        slot_keys=slot_keys,
        pseudorange_doppler_mask_m=40.0,
        slot_frequency_thresholds_fn=threshold_fn,
        mask_pseudorange_doppler_consistency_fn=mask_fn,
        default_l1_threshold_m=40.0,
        default_l5_threshold_m=25.0,
    )

    assert products.pseudorange_doppler_mask_count == 8
    assert calls == ["threshold", "mask"]


def test_build_pseudorange_doppler_consistency_stage_skips_when_observation_mask_disabled() -> None:
    products = build_pseudorange_doppler_consistency_stage(
        apply_observation_mask=False,
        times_ms=np.array([1000.0], dtype=np.float64),
        pseudorange_observable=np.zeros((1, 0), dtype=np.float64),
        weights=np.zeros((1, 0), dtype=np.float64),
        doppler=None,
        doppler_weights=None,
        phone_name="pixel4",
        sys_kind=None,
        n_clock=1,
        slot_keys=[],
        pseudorange_doppler_mask_m=40.0,
        slot_frequency_thresholds_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        mask_pseudorange_doppler_consistency_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        default_l1_threshold_m=40.0,
        default_l5_threshold_m=25.0,
    )

    assert products.pseudorange_doppler_mask_count == 0


def test_build_observation_mask_base_correction_stage_preserves_mask_then_base_order(tmp_path: Path) -> None:
    calls: list[str] = []
    times_ms = np.array([1000.0], dtype=np.float64)
    slot_keys = ["G01", "G02", "G03"]
    pseudorange = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)
    weights = np.ones((1, 3), dtype=np.float64)
    doppler_weights = np.ones((1, 3), dtype=np.float64)

    def doppler_mask_fn(*_args: object, **_kwargs: object) -> int:
        calls.append("doppler")
        return 1

    def threshold_fn(keys: list[str], threshold: float, **_kwargs: float) -> np.ndarray:
        assert keys == slot_keys
        calls.append(f"threshold:{threshold:g}")
        return np.full(len(keys), threshold, dtype=np.float64)

    def pd_mask_fn(*_args: object, **_kwargs: object) -> int:
        calls.append("pd")
        weights[0, 0] = 0.0
        return 2

    def pr_mask_fn(*_args: object, **kwargs: object) -> int:
        calls.append("pr")
        np.testing.assert_allclose(weights, [[0.0, 1.0, 1.0]])
        assert kwargs["common_bias_by_group"] is None
        weights[0, 1] = 0.0
        return 3

    def correction_matrix_fn(
        data_root: Path,
        trip: str,
        correction_times_ms: np.ndarray,
        correction_slot_keys: list[str],
        signal_type: str,
    ) -> np.ndarray:
        calls.append("base")
        assert data_root == tmp_path
        assert trip == "train/course/phone"
        assert signal_type == "GPS_L1_CA"
        np.testing.assert_array_equal(correction_times_ms, times_ms)
        assert correction_slot_keys == slot_keys
        np.testing.assert_allclose(weights, [[0.0, 0.0, 1.0]])
        return np.array([[10.0, 20.0, 30.0]], dtype=np.float64)

    products = build_observation_mask_base_correction_stage(
        apply_observation_mask=True,
        apply_base_correction=True,
        data_root=tmp_path,
        trip="train/course/phone",
        times_ms=times_ms,
        sat_ecef=np.zeros((1, 3, 3), dtype=np.float64),
        sat_vel=np.zeros((1, 3, 3), dtype=np.float64),
        doppler=np.ones((1, 3), dtype=np.float64),
        doppler_weights=doppler_weights,
        kaggle_wls=np.zeros((1, 3), dtype=np.float64),
        pseudorange_observable=np.ones((1, 3), dtype=np.float64),
        weights=weights,
        phone_name="pixel4",
        sys_kind=np.zeros((1, 3), dtype=np.int32),
        n_clock=1,
        slot_keys=slot_keys,
        pseudorange=pseudorange,
        pseudorange_isb_sample_weights=np.ones((1, 3), dtype=np.float64),
        pseudorange_bias_weights=np.ones((1, 3), dtype=np.float64),
        clock_bias_m=None,
        clock_drift_mps=np.zeros(1, dtype=np.float64),
        sat_clock_drift_mps=np.zeros((1, 3), dtype=np.float64),
        baseline_velocity_times_ms=times_ms,
        baseline_velocity_xyz=np.zeros((1, 3), dtype=np.float64),
        clock_drift_context_times_ms=times_ms,
        clock_drift_context_mps=np.zeros(1, dtype=np.float64),
        doppler_residual_mask_mps=3.0,
        pseudorange_doppler_mask_m=40.0,
        pseudorange_residual_mask_m=20.0,
        pseudorange_residual_mask_l5_m=None,
        full_isb_batch=None,
        signal_type="GPS_L1_CA",
        correction_matrix_fn=correction_matrix_fn,
        mask_doppler_residual_outliers_fn=doppler_mask_fn,
        slot_frequency_thresholds_fn=threshold_fn,
        mask_pseudorange_doppler_consistency_fn=pd_mask_fn,
        slot_pseudorange_common_bias_groups_fn=lambda keys: np.zeros(len(keys), dtype=np.int32),
        remap_pseudorange_isb_by_group_fn=lambda *_args: _raise_unexpected(),
        pseudorange_global_isb_by_group_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        is_l5_signal_fn=lambda _signal: _raise_unexpected(),
        mask_pseudorange_residual_outliers_fn=pr_mask_fn,
        default_pd_l1_threshold_m=40.0,
        default_pd_l5_threshold_m=25.0,
        default_pr_l1_threshold_m=20.0,
        default_pr_l5_threshold_m=15.0,
    )

    assert isinstance(products, ObservationMaskBaseCorrectionStageProducts)
    assert products.doppler_residual_stage.doppler_residual_mask_count == 1
    assert products.pseudorange_doppler_stage.pseudorange_doppler_mask_count == 2
    assert products.pseudorange_residual_stage.residual_mask_count == 3
    assert products.base_correction_count == 1
    assert calls == ["doppler", "threshold:40", "pd", "threshold:20", "pr", "base"]
    np.testing.assert_allclose(pseudorange, [[100.0, 100.0, 70.0]])


def test_build_configured_post_observation_stages_uses_bundled_config_and_dependencies(tmp_path: Path) -> None:
    calls: list[str] = []
    times_ms = np.array([1000.0], dtype=np.float64)
    empty_matrix = np.zeros((1, 0), dtype=np.float64)
    observation_products = PreparedObservationProducts(
        filtered_frame=pd.DataFrame({"utcTimeMillis": [1000]}),
        gps_tgd_m_by_svid={},
        observation_mask_count=0,
        metadata_context=object(),
        epoch_time_context=object(),
        baseline_velocity_times_ms=times_ms,
        baseline_velocity_xyz=np.zeros((1, 3), dtype=np.float64),
        clock_drift_context_times_ms=times_ms,
        clock_drift_context_mps=np.array([0.1], dtype=np.float64),
        epochs=["epoch"],
        times_ms=times_ms,
        sat_ecef=np.zeros((1, 0, 3), dtype=np.float64),
        pseudorange=empty_matrix.copy(),
        pseudorange_observable=empty_matrix.copy(),
        weights=empty_matrix.copy(),
        pseudorange_bias_weights=empty_matrix.copy(),
        sat_clock_bias_matrix=empty_matrix.copy(),
        rtklib_tropo_m=empty_matrix.copy(),
        kaggle_wls=np.zeros((1, 3), dtype=np.float64),
        truth=np.zeros((1, 3), dtype=np.float64),
        visible_max=0,
        slot_keys=[],
        n_sat_slots=0,
        n_clock=1,
        elapsed_ns=np.array([0.0], dtype=np.float64),
        sys_kind=None,
        clock_counts=np.array([1.0], dtype=np.float64),
        clock_bias_m=np.array([3.0], dtype=np.float64),
        clock_drift_mps=np.array([0.4], dtype=np.float64),
        sat_vel=None,
        sat_clock_drift_mps=None,
        doppler=None,
        doppler_weights=None,
        adr=None,
        adr_state=None,
        adr_uncertainty=None,
    )

    config = PostObservationStageConfig(
        trip_dir=tmp_path / "pixel4",
        phone_name="pixel4",
        apply_absolute_height=False,
        absolute_height_dist_m=7.0,
        clock_drift_blocklist_phones={"mi8"},
        apply_observation_mask=False,
        has_window_subset=False,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=False,
        observation_min_cn0_dbhz=18.0,
        observation_min_elevation_deg=10.0,
        dual_frequency=False,
        factor_dt_max_s=1.5,
        apply_base_correction=False,
        data_root=None,
        trip=None,
        doppler_residual_mask_mps=3.0,
        pseudorange_doppler_mask_m=40.0,
        pseudorange_residual_mask_m=20.0,
        pseudorange_residual_mask_l5_m=None,
        tdcp_consistency_threshold_m=1.5,
        tdcp_loffset_m=0.0,
        matlab_residual_diagnostics_mask_path=None,
        tdcp_geometry_correction=False,
        tdcp_weight_scale=2.0,
        imu_frame="body",
        default_pd_l1_threshold_m=40.0,
        default_pd_l5_threshold_m=25.0,
        default_pr_l1_threshold_m=20.0,
        default_pr_l5_threshold_m=15.0,
    )

    def gnss_log_fn(*_args: Any, **_kwargs: Any) -> None:
        calls.append("gnss")
        return None

    def clean_clock_drift_fn(
        got_times_ms: np.ndarray,
        bias_m: np.ndarray | None,
        drift_mps: np.ndarray | None,
        phone_name: str,
    ) -> np.ndarray:
        calls.append("clock")
        np.testing.assert_array_equal(got_times_ms, times_ms)
        np.testing.assert_array_equal(bias_m, [3.0])
        np.testing.assert_array_equal(drift_mps, [0.4])
        assert phone_name == "pixel4"
        return np.asarray(drift_mps, dtype=np.float64) + 1.0

    def scale_fn(tdcp_weights: np.ndarray | None, scale: float) -> None:
        calls.append("scale")
        assert tdcp_weights is None
        assert scale == 2.0

    dependencies = PostObservationStageDependencies(
        build_trip_arrays_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        gnss_log_corrected_pseudorange_matrix_fn=gnss_log_fn,
        load_absolute_height_reference_ecef_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        clock_jump_from_epoch_counts_fn=lambda _counts: np.array([False]),
        estimate_residual_clock_series_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        combine_clock_jump_masks_fn=lambda *_args: _raise_unexpected(),
        detect_clock_jumps_from_clock_bias_fn=lambda *_args: _raise_unexpected(),
        clean_clock_drift_fn=clean_clock_drift_fn,
        correction_matrix_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        mask_doppler_residual_outliers_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        slot_frequency_thresholds_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        mask_pseudorange_doppler_consistency_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        slot_pseudorange_common_bias_groups_fn=lambda *_args: _raise_unexpected(),
        remap_pseudorange_isb_by_group_fn=lambda *_args: _raise_unexpected(),
        pseudorange_global_isb_by_group_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        is_l5_signal_fn=lambda _signal: _raise_unexpected(),
        mask_pseudorange_residual_outliers_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        build_tdcp_arrays_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        apply_diagnostics_mask_fn=lambda **_kwargs: _raise_unexpected(),
        apply_geometry_correction_fn=lambda *_args: _raise_unexpected(),
        apply_weight_scale_fn=scale_fn,
        load_device_imu_measurements_fn=lambda _trip_dir: (None, None, None),
        process_device_imu_fn=lambda *_args: _raise_unexpected(),
        project_stop_to_epochs_fn=lambda *_args: _raise_unexpected(),
        preintegrate_processed_imu_fn=lambda *_args, **_kwargs: _raise_unexpected(),
    )

    products = build_configured_post_observation_stages(
        observation_products=observation_products,
        config=config,
        dependencies=dependencies,
    )

    assert isinstance(products, PostObservationStageProducts)
    assert products.gnss_log_stage.applied_count == 0
    assert products.absolute_height_stage.absolute_height_ref_count == 0
    np.testing.assert_array_equal(products.clock_residual_stage.clock_jump, [False])
    np.testing.assert_allclose(products.clock_residual_stage.clock_drift_mps, [1.4])
    assert products.mask_base_stage.base_correction_count == 0
    assert products.tdcp_stage.tdcp_meas is None
    assert products.imu_stage.stop_epochs is None
    assert calls == ["gnss", "clock", "scale"]


def test_build_post_observation_stages_wires_pipeline_and_preserves_signal_weights(tmp_path: Path) -> None:
    calls: list[str] = []
    times_ms = np.array([1000.0, 2000.0], dtype=np.float64)
    slot_keys = [("G", 1, "L1"), ("G", 2, "L5")]
    pseudorange = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64)
    pseudorange_observable = pseudorange + 1.0
    weights = np.ones((2, 2), dtype=np.float64)
    doppler_weights = np.ones((2, 2), dtype=np.float64)
    kaggle_wls = np.zeros((2, 3), dtype=np.float64)
    sat_ecef = np.zeros((2, 2, 3), dtype=np.float64)
    sat_vel = np.ones((2, 2, 3), dtype=np.float64)
    doppler = np.ones((2, 2), dtype=np.float64)
    clock_jump = np.array([False, True])
    clock_drift = np.array([0.1, 0.2], dtype=np.float64)
    diagnostics_path = tmp_path / "diagnostics.csv"

    def gnss_log_fn(*_args: Any, **_kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls.append("gnss")
        gnss_weights = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        return pseudorange + 5.0, gnss_weights, pseudorange_observable + 5.0

    def abs_height_fn(parent: Path, reference_xyz: np.ndarray, *, dist_m: float) -> tuple[np.ndarray, int]:
        calls.append("height")
        assert parent == tmp_path
        assert reference_xyz is kaggle_wls
        assert dist_m == 7.0
        return reference_xyz + 1.0, 2

    def clean_clock_drift_fn(
        _times_ms: np.ndarray,
        bias_m: np.ndarray | None,
        drift_mps: np.ndarray | None,
        phone_name: str,
    ) -> np.ndarray:
        calls.append("clock")
        assert bias_m is None
        np.testing.assert_array_equal(drift_mps, clock_drift)
        assert phone_name == "pixel4"
        return np.asarray(drift_mps, dtype=np.float64) + 1.0

    def doppler_mask_fn(*_args: Any, **_kwargs: Any) -> int:
        calls.append("doppler-mask")
        return 3

    def threshold_fn(keys: list[tuple[str, int, str]], threshold: float, **_kwargs: float) -> np.ndarray:
        assert keys == slot_keys
        calls.append(f"threshold:{threshold:g}")
        return np.full(len(keys), threshold, dtype=np.float64)

    def pd_mask_fn(*_args: Any, **_kwargs: Any) -> int:
        calls.append("pd-mask")
        weights[0, 1] = 0.0
        return 4

    def pr_mask_fn(*_args: Any, **kwargs: Any) -> int:
        calls.append("pr-mask")
        np.testing.assert_array_equal(kwargs["common_bias_sample_weights"], np.ones((2, 2), dtype=np.float64))
        weights[1, 0] = 0.0
        return 5

    def correction_matrix_fn(
        data_root: Path,
        trip: str,
        correction_times_ms: np.ndarray,
        correction_slot_keys: list[tuple[str, int, str]],
        signal_type: str,
    ) -> np.ndarray:
        calls.append("base")
        assert data_root == tmp_path / "data"
        assert trip == "train/course/pixel4"
        assert signal_type == "GPS_L1_CA"
        np.testing.assert_array_equal(correction_times_ms, times_ms)
        assert correction_slot_keys == slot_keys
        np.testing.assert_allclose(weights, [[1.0, 0.0], [0.0, 1.0]])
        return np.full((2, 2), 10.0, dtype=np.float64)

    def build_tdcp_arrays_fn(*_args: Any, **_kwargs: Any) -> tuple[np.ndarray, np.ndarray, int]:
        calls.append("tdcp")
        return np.ones((1, 2), dtype=np.float64), np.ones((1, 2), dtype=np.float64), 6

    def diagnostics_fn(**kwargs: Any) -> None:
        calls.append("diagnostics")
        assert kwargs["diagnostics_path"] == diagnostics_path
        np.testing.assert_allclose(kwargs["weights"], [[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(kwargs["signal_weights"], np.ones((2, 2), dtype=np.float64))
        assert kwargs["signal_weights"] is not weights
        assert kwargs["signal_doppler_weights"] is not doppler_weights

    def geometry_fn(*_args: Any) -> int:
        calls.append("geometry")
        return 7

    def scale_fn(tdcp_weights: np.ndarray | None, scale: float) -> None:
        calls.append("scale")
        assert scale == 2.0
        if tdcp_weights is not None:
            tdcp_weights *= scale

    products = build_post_observation_stages(
        trip_dir=tmp_path / "pixel4",
        filtered_frame=pd.DataFrame(),
        times_ms=times_ms,
        slot_keys=slot_keys,
        gps_tgd_m_by_svid={},
        rtklib_tropo_m=np.zeros((2, 2), dtype=np.float64),
        sat_clock_bias_m=np.zeros((2, 2), dtype=np.float64),
        phone_name="pixel4",
        pseudorange=pseudorange,
        pseudorange_observable=pseudorange_observable,
        pseudorange_bias_weights=np.ones((2, 2), dtype=np.float64),
        weights=weights,
        doppler_weights=doppler_weights,
        apply_absolute_height=True,
        absolute_height_dist_m=7.0,
        kaggle_wls=kaggle_wls,
        clock_drift_blocklist_phones={"mi8"},
        sat_ecef=sat_ecef,
        sat_vel=sat_vel,
        doppler=doppler,
        sat_clock_drift_mps=np.zeros((2, 2), dtype=np.float64),
        sys_kind=np.zeros((2, 2), dtype=np.int32),
        clock_counts=np.array([1.0, 2.0], dtype=np.float64),
        clock_bias_m=None,
        clock_drift_mps=clock_drift,
        apply_observation_mask=True,
        has_window_subset=False,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=False,
        observation_min_cn0_dbhz=18.0,
        observation_min_elevation_deg=10.0,
        dual_frequency=True,
        factor_dt_max_s=1.5,
        clock_drift_context_times_ms=times_ms,
        clock_drift_context_mps=clock_drift,
        build_trip_arrays_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        apply_base_correction=True,
        data_root=tmp_path / "data",
        trip="train/course/pixel4",
        n_clock=1,
        baseline_velocity_times_ms=times_ms,
        baseline_velocity_xyz=np.zeros((2, 3), dtype=np.float64),
        doppler_residual_mask_mps=3.0,
        pseudorange_doppler_mask_m=40.0,
        pseudorange_residual_mask_m=20.0,
        pseudorange_residual_mask_l5_m=None,
        tdcp_consistency_threshold_m=1.5,
        tdcp_loffset_m=0.0,
        matlab_residual_diagnostics_mask_path=diagnostics_path,
        tdcp_geometry_correction=True,
        tdcp_weight_scale=2.0,
        adr=np.ones((2, 2), dtype=np.float64),
        adr_state=np.ones((2, 2), dtype=np.float64),
        adr_uncertainty=np.ones((2, 2), dtype=np.float64),
        elapsed_ns=np.array([0.0, 1.0], dtype=np.float64),
        imu_frame="body",
        gnss_log_corrected_pseudorange_matrix_fn=gnss_log_fn,
        load_absolute_height_reference_ecef_fn=abs_height_fn,
        clock_jump_from_epoch_counts_fn=lambda _counts: clock_jump,
        estimate_residual_clock_series_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        combine_clock_jump_masks_fn=lambda *_args: _raise_unexpected(),
        detect_clock_jumps_from_clock_bias_fn=lambda *_args: _raise_unexpected(),
        clean_clock_drift_fn=clean_clock_drift_fn,
        correction_matrix_fn=correction_matrix_fn,
        mask_doppler_residual_outliers_fn=doppler_mask_fn,
        slot_frequency_thresholds_fn=threshold_fn,
        mask_pseudorange_doppler_consistency_fn=pd_mask_fn,
        slot_pseudorange_common_bias_groups_fn=lambda keys: np.zeros(len(keys), dtype=np.int32),
        remap_pseudorange_isb_by_group_fn=lambda *_args: _raise_unexpected(),
        pseudorange_global_isb_by_group_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        is_l5_signal_fn=lambda _signal: _raise_unexpected(),
        mask_pseudorange_residual_outliers_fn=pr_mask_fn,
        build_tdcp_arrays_fn=build_tdcp_arrays_fn,
        apply_diagnostics_mask_fn=diagnostics_fn,
        apply_geometry_correction_fn=geometry_fn,
        apply_weight_scale_fn=scale_fn,
        load_device_imu_measurements_fn=lambda _trip_dir: (None, None, None),
        process_device_imu_fn=lambda *_args: _raise_unexpected(),
        project_stop_to_epochs_fn=lambda *_args: _raise_unexpected(),
        preintegrate_processed_imu_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        default_pd_l1_threshold_m=40.0,
        default_pd_l5_threshold_m=25.0,
        default_pr_l1_threshold_m=20.0,
        default_pr_l5_threshold_m=15.0,
    )

    assert isinstance(products, PostObservationStageProducts)
    assert products.gnss_log_stage.applied_count == 1
    assert products.absolute_height_stage.absolute_height_ref_count == 2
    np.testing.assert_allclose(products.clock_residual_stage.clock_drift_mps, clock_drift + 1.0)
    assert products.full_context_stage.batch is None
    assert products.mask_base_stage.base_correction_count == 2
    assert products.mask_base_stage.doppler_residual_stage.doppler_residual_mask_count == 3
    assert products.mask_base_stage.pseudorange_doppler_stage.pseudorange_doppler_mask_count == 4
    assert products.mask_base_stage.pseudorange_residual_stage.residual_mask_count == 5
    assert products.time_delta.factor_dt_gap_count == 0
    assert products.tdcp_stage.tdcp_consistency_mask_count == 6
    assert products.tdcp_stage.tdcp_geometry_correction_count == 7
    assert products.imu_stage.stop_epochs is None
    assert products.signal_weights is not weights
    np.testing.assert_allclose(products.tdcp_stage.tdcp_weights, [[2.0, 2.0]])
    np.testing.assert_allclose(pseudorange, [[95.0, 200.0], [300.0, 390.0]])
    assert calls == [
        "gnss",
        "height",
        "clock",
        "doppler-mask",
        "threshold:40",
        "pd-mask",
        "threshold:20",
        "pr-mask",
        "base",
        "tdcp",
        "diagnostics",
        "geometry",
        "scale",
    ]


def test_build_tdcp_stage_applies_diagnostics_geometry_and_scale(tmp_path: Path) -> None:
    calls: list[str] = []
    adr = np.array([[1.0, 2.0]], dtype=np.float64)
    tdcp_dt = np.array([1.0], dtype=np.float64)
    weights = np.ones((1, 2), dtype=np.float64)
    signal_weights = weights.copy()
    doppler_weights = np.ones((1, 2), dtype=np.float64)
    sat_ecef = np.zeros((1, 2, 3), dtype=np.float64)
    kaggle_wls = np.zeros((1, 3), dtype=np.float64)

    def build_fn(
        got_adr: np.ndarray,
        adr_state: np.ndarray,
        adr_uncertainty: np.ndarray,
        doppler: np.ndarray,
        got_tdcp_dt: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        np.testing.assert_allclose(got_adr, adr)
        np.testing.assert_allclose(adr_state, [[1.0, 1.0]])
        assert np.isnan(adr_uncertainty).all()
        np.testing.assert_allclose(doppler, [[0.5, 0.6]])
        np.testing.assert_allclose(got_tdcp_dt, tdcp_dt)
        assert kwargs["consistency_threshold_m"] == 1.5
        assert kwargs["loffset_m"] == 0.25
        calls.append("build")
        return np.array([[10.0, 20.0]], dtype=np.float64), np.array([[2.0, 4.0]], dtype=np.float64), 3

    def diagnostics_fn(**kwargs: Any) -> None:
        assert kwargs["diagnostics_path"] == tmp_path / "diag.csv"
        np.testing.assert_allclose(kwargs["times_ms"], [1000.0])
        assert kwargs["slot_keys"] == ["G01", "G02"]
        np.testing.assert_allclose(kwargs["signal_tdcp_weights"], [[2.0, 4.0]])
        kwargs["tdcp_weights"][0, 0] = 0.0
        calls.append("diagnostics")

    def geometry_fn(tdcp_meas: np.ndarray, tdcp_weights: np.ndarray, got_sat_ecef: np.ndarray, got_kaggle_wls: np.ndarray) -> int:
        np.testing.assert_allclose(tdcp_meas, [[10.0, 20.0]])
        np.testing.assert_allclose(tdcp_weights, [[0.0, 4.0]])
        np.testing.assert_allclose(got_sat_ecef, sat_ecef)
        np.testing.assert_allclose(got_kaggle_wls, kaggle_wls)
        calls.append("geometry")
        return 7

    def scale_fn(tdcp_weights: np.ndarray, scale: float) -> None:
        assert scale == 0.5
        tdcp_weights *= scale
        calls.append("scale")

    products = build_tdcp_stage(
        adr=adr,
        adr_state=np.array([[1.0, 1.0]], dtype=np.float64),
        adr_uncertainty=None,
        doppler=np.array([[0.5, 0.6]], dtype=np.float64),
        tdcp_dt=tdcp_dt,
        tdcp_consistency_threshold_m=1.5,
        doppler_weights=doppler_weights,
        clock_jump=np.array([False]),
        tdcp_loffset_m=0.25,
        matlab_residual_diagnostics_mask_path=tmp_path / "diag.csv",
        times_ms=np.array([1000.0], dtype=np.float64),
        slot_keys=["G01", "G02"],
        weights=weights,
        signal_weights=signal_weights,
        signal_doppler_weights=doppler_weights.copy(),
        sat_ecef=sat_ecef,
        kaggle_wls=kaggle_wls,
        tdcp_geometry_correction=True,
        tdcp_weight_scale=0.5,
        build_tdcp_arrays_fn=build_fn,
        apply_diagnostics_mask_fn=diagnostics_fn,
        apply_geometry_correction_fn=geometry_fn,
        apply_weight_scale_fn=scale_fn,
    )

    np.testing.assert_allclose(products.tdcp_meas, [[10.0, 20.0]])
    np.testing.assert_allclose(products.tdcp_weights, [[0.0, 2.0]])
    np.testing.assert_allclose(products.signal_tdcp_weights, [[2.0, 4.0]])
    assert products.tdcp_consistency_mask_count == 3
    assert products.tdcp_geometry_correction_count == 7
    assert calls == ["build", "diagnostics", "geometry", "scale"]


def test_build_tdcp_stage_allows_missing_adr_and_still_runs_scale() -> None:
    calls: list[str] = []

    products = build_tdcp_stage(
        adr=None,
        adr_state=np.array([[1.0]], dtype=np.float64),
        adr_uncertainty=None,
        doppler=None,
        tdcp_dt=np.array([1.0], dtype=np.float64),
        tdcp_consistency_threshold_m=1.5,
        doppler_weights=None,
        clock_jump=None,
        tdcp_loffset_m=0.0,
        matlab_residual_diagnostics_mask_path=None,
        times_ms=np.array([1000.0], dtype=np.float64),
        slot_keys=[],
        weights=np.zeros((1, 0), dtype=np.float64),
        signal_weights=np.zeros((1, 0), dtype=np.float64),
        signal_doppler_weights=None,
        sat_ecef=np.zeros((1, 0, 3), dtype=np.float64),
        kaggle_wls=np.zeros((1, 3), dtype=np.float64),
        tdcp_geometry_correction=False,
        tdcp_weight_scale=2.0,
        build_tdcp_arrays_fn=lambda *_args, **_kwargs: _raise_unexpected(),
        apply_diagnostics_mask_fn=lambda **_kwargs: _raise_unexpected(),
        apply_geometry_correction_fn=lambda *_args: _raise_unexpected(),
        apply_weight_scale_fn=lambda weights, scale: calls.append(f"scale:{weights}:{scale}"),
    )

    assert products.tdcp_meas is None
    assert products.tdcp_weights is None
    assert products.signal_tdcp_weights is None
    assert products.tdcp_consistency_mask_count == 0
    assert products.tdcp_geometry_correction_count == 0
    assert calls == ["scale:None:2.0"]


def test_build_imu_stage_returns_empty_when_measurements_missing(tmp_path: Path) -> None:
    products = build_imu_stage(
        trip_dir=tmp_path,
        times_ms=np.array([1000.0], dtype=np.float64),
        elapsed_ns=None,
        reference_xyz_ecef=np.zeros((1, 3), dtype=np.float64),
        imu_frame="body",
        load_device_imu_measurements_fn=lambda _trip_dir: (None, object(), None),
        process_device_imu_fn=lambda *_args: (_raise_unexpected()),
        project_stop_to_epochs_fn=lambda *_args: _raise_unexpected(),
        preintegrate_processed_imu_fn=lambda *_args, **_kwargs: _raise_unexpected(),
    )

    assert products.stop_epochs is None
    assert products.imu_preintegration is None


def test_build_imu_stage_uses_injected_processing_pipeline(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Processed:
        times_ms: np.ndarray

    calls: list[str] = []
    times_ms = np.array([1000.0, 2000.0], dtype=np.float64)
    elapsed_ns = np.array([10.0, 20.0], dtype=np.float64)
    reference = np.zeros((2, 3), dtype=np.float64)

    def load_fn(trip_dir: Path) -> tuple[str, str, None]:
        assert trip_dir == tmp_path
        calls.append("load")
        return "acc", "gyro", None

    def process_fn(acc: str, gyro: str, got_times_ms: np.ndarray, got_elapsed_ns: np.ndarray) -> tuple[Processed, str, np.ndarray]:
        assert (acc, gyro) == ("acc", "gyro")
        np.testing.assert_allclose(got_times_ms, times_ms)
        np.testing.assert_allclose(got_elapsed_ns, elapsed_ns)
        calls.append("process")
        return Processed(np.array([900.0, 1900.0], dtype=np.float64)), "gyro_proc", np.array([False, True])

    def project_fn(imu_times_ms: np.ndarray, idx_stop: np.ndarray, epoch_times_ms: np.ndarray) -> np.ndarray:
        np.testing.assert_allclose(imu_times_ms, [900.0, 1900.0])
        np.testing.assert_array_equal(idx_stop, [False, True])
        np.testing.assert_allclose(epoch_times_ms, times_ms)
        calls.append("project")
        return np.array([False, True])

    def preintegrate_fn(
        acc_proc: Processed,
        gyro_proc: str,
        got_times_ms: np.ndarray,
        *,
        delta_frame: str,
        reference_xyz_ecef: np.ndarray,
    ) -> dict[str, Any]:
        assert gyro_proc == "gyro_proc"
        np.testing.assert_allclose(acc_proc.times_ms, [900.0, 1900.0])
        np.testing.assert_allclose(got_times_ms, times_ms)
        np.testing.assert_allclose(reference_xyz_ecef, reference)
        assert delta_frame == "ecef"
        calls.append("preintegrate")
        return {"ok": True}

    products = build_imu_stage(
        trip_dir=tmp_path,
        times_ms=times_ms,
        elapsed_ns=elapsed_ns,
        reference_xyz_ecef=reference,
        imu_frame="ecef",
        load_device_imu_measurements_fn=load_fn,
        process_device_imu_fn=process_fn,
        project_stop_to_epochs_fn=project_fn,
        preintegrate_processed_imu_fn=preintegrate_fn,
    )

    np.testing.assert_array_equal(products.stop_epochs, [False, True])
    assert products.imu_preintegration == {"ok": True}
    assert calls == ["load", "process", "project", "preintegrate"]


def test_build_imu_stage_suppresses_processing_failures(tmp_path: Path) -> None:
    def process_fn(*_args: object) -> tuple[object, object, object]:
        raise ValueError("bad imu")

    products = build_imu_stage(
        trip_dir=tmp_path,
        times_ms=np.array([1000.0], dtype=np.float64),
        elapsed_ns=None,
        reference_xyz_ecef=np.zeros((1, 3), dtype=np.float64),
        imu_frame="body",
        load_device_imu_measurements_fn=lambda _trip_dir: (object(), object(), None),
        process_device_imu_fn=process_fn,
        project_stop_to_epochs_fn=lambda *_args: _raise_unexpected(),
        preintegrate_processed_imu_fn=lambda *_args, **_kwargs: _raise_unexpected(),
    )

    assert products.stop_epochs is None
    assert products.imu_preintegration is None


def _raw_observation_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults: dict[str, object] = {
        "utcTimeMillis": 1000,
        "ConstellationType": 1,
        "Svid": 1,
        "SignalType": "GPS_L1_CA",
        "Cn0DbHz": 30.0,
        "RawPseudorangeMeters": 10.0,
        "SvPositionXEcefMeters": 1.0,
        "SvPositionYEcefMeters": 2.0,
        "SvPositionZEcefMeters": 3.0,
        "SvElevationDegrees": 45.0,
        "SvClockBiasMeters": 0.1,
        "IonosphericDelayMeters": 0.2,
        "TroposphericDelayMeters": 0.3,
    }
    materialized = []
    for idx, row in enumerate(rows):
        item = defaults.copy()
        item["utcTimeMillis"] = int(defaults["utcTimeMillis"]) + idx
        item["Svid"] = idx + 1
        item.update(row)
        materialized.append(item)
    return pd.DataFrame(materialized)


def _raise_unexpected() -> None:
    raise AssertionError("unexpected call")
