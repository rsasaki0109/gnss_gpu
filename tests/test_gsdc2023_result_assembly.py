from __future__ import annotations

import numpy as np

from experiments.evaluate import lla_to_ecef
from experiments.gsdc2023_bridge_config import BridgeConfig
from experiments.gsdc2023_observation_matrix import TripArrays
from experiments.gsdc2023_result_assembly import (
    assemble_source_outputs,
    build_bridge_result,
    postprocess_output_state,
)
from experiments.gsdc2023_solver_selection import build_source_solution_catalog


def _ecef_track() -> np.ndarray:
    return np.array(
        [
            lla_to_ecef(np.deg2rad(37.0), np.deg2rad(-122.0), 10.0),
            lla_to_ecef(np.deg2rad(37.00001), np.deg2rad(-122.00001), 10.2),
        ],
        dtype=np.float64,
    )


def _state(xyz: np.ndarray, clock_bias_m: float = 0.0) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
            np.full(xyz.shape[0], float(clock_bias_m), dtype=np.float64),
        ],
    )


def _batch(*, truth: np.ndarray, has_truth: bool) -> TripArrays:
    n_epoch = truth.shape[0]
    n_sat = 2
    return TripArrays(
        times_ms=np.arange(n_epoch, dtype=np.float64) * 1000.0,
        sat_ecef=np.zeros((n_epoch, n_sat, 3), dtype=np.float64),
        pseudorange=np.zeros((n_epoch, n_sat), dtype=np.float64),
        weights=np.ones((n_epoch, n_sat), dtype=np.float64),
        kaggle_wls=truth + np.array([1.0, 0.0, 0.0], dtype=np.float64),
        truth=truth,
        max_sats=n_sat,
        has_truth=has_truth,
    )


def _catalog(truth: np.ndarray):
    baseline_state = _state(truth + np.array([1.0, 0.0, 0.0], dtype=np.float64))
    raw_state = _state(truth + np.array([0.0, 2.0, 0.0], dtype=np.float64), clock_bias_m=1.0)
    fgo_state = _state(truth + np.array([0.0, 0.0, 3.0], dtype=np.float64), clock_bias_m=2.0)
    auto_state = raw_state.copy()
    return build_source_solution_catalog(
        n_epoch=truth.shape[0],
        baseline_state=baseline_state,
        raw_state=raw_state,
        fgo_state=fgo_state,
        auto_state=auto_state,
        auto_sources=np.array(["raw_wls", "raw_wls"], dtype=object),
        auto_source_counts={"baseline": 0, "raw_wls": truth.shape[0], "fgo": 0},
        baseline_mse_pr=10.0,
        raw_wls_mse_pr=2.0,
        fgo_mse_pr=5.0,
        auto_mse_pr=2.0,
    )


def test_assemble_source_outputs_selects_source_and_computes_truth_metrics() -> None:
    truth = _ecef_track()
    catalog = _catalog(truth)
    batch = _batch(truth=truth, has_truth=True)

    outputs = assemble_source_outputs(
        catalog,
        batch,
        BridgeConfig(position_source="auto"),
        phone_name="unknown",
    )

    np.testing.assert_allclose(outputs.selected_state, catalog.states["auto"])
    np.testing.assert_array_equal(outputs.selected_sources.astype(str), np.array(["raw_wls", "raw_wls"], dtype=object))
    assert outputs.selected_source_counts == {"baseline": 0, "raw_wls": 2, "fgo": 0}
    assert outputs.selected_mse_pr == 2.0
    assert outputs.truth is batch.truth
    assert outputs.metrics_selected is not None
    assert outputs.metrics_kaggle is not None
    assert outputs.metrics_raw_wls is not None
    assert outputs.metrics_fgo is not None
    assert not np.shares_memory(outputs.output_states["auto"], catalog.states["auto"])


def test_assemble_source_outputs_omits_metrics_without_truth() -> None:
    truth = _ecef_track()
    outputs = assemble_source_outputs(
        _catalog(truth),
        _batch(truth=truth, has_truth=False),
        BridgeConfig(position_source="raw_wls"),
        phone_name="unknown",
    )

    assert outputs.truth is None
    assert outputs.metrics_selected is None
    assert outputs.metrics_kaggle is None
    assert outputs.metrics_raw_wls is None
    assert outputs.metrics_fgo is None
    assert outputs.selected_mse_pr == 2.0


def test_build_bridge_result_maps_assembled_outputs_and_metadata() -> None:
    truth = _ecef_track()
    batch = _batch(truth=truth, has_truth=True)
    config = BridgeConfig(
        position_source="auto",
        apply_observation_mask=True,
        observation_min_cn0_dbhz=20.0,
        factor_dt_max_s=1.25,
    )
    outputs = assemble_source_outputs(
        _catalog(truth),
        batch,
        config,
        phone_name="unknown",
    )

    result = build_bridge_result(
        trip="train/course/phone",
        batch=batch,
        config=config,
        assembled_outputs=outputs,
        fgo_iters=7,
        failed_chunks=1,
        baseline_mse_pr=10.0,
        raw_wls_mse_pr=2.0,
        fgo_mse_pr=5.0,
        chunk_records=[],
        allow_raw_wls_on_mi8_baseline_jump=False,
    )

    assert result.trip == "train/course/phone"
    assert result.signal_type == config.signal_type
    assert result.weight_mode == config.weight_mode
    assert result.selected_source_mode == "auto"
    assert result.fgo_iters == 7
    assert result.failed_chunks == 1
    assert result.selected_mse_pr == outputs.selected_mse_pr
    assert result.baseline_mse_pr == 10.0
    assert result.raw_wls_mse_pr == 2.0
    assert result.fgo_mse_pr == 5.0
    assert result.selected_source_counts == outputs.selected_source_counts
    assert result.chunk_selection_records == []
    assert result.factor_dt_max_s == 1.25
    assert result.observation_mask_applied is True
    np.testing.assert_allclose(result.times_ms, batch.times_ms)
    np.testing.assert_allclose(result.kaggle_wls, outputs.output_states["baseline"][:, :3])
    np.testing.assert_allclose(result.raw_wls, outputs.output_states["raw_wls"])
    np.testing.assert_allclose(result.fgo_state, outputs.output_states["fgo"])
    np.testing.assert_allclose(result.selected_state, outputs.selected_state)
    np.testing.assert_array_equal(result.selected_sources, outputs.selected_sources)
    assert result.truth is outputs.truth
    assert result.metrics_selected is outputs.metrics_selected


def test_postprocess_output_state_returns_copy_when_no_postprocess_enabled() -> None:
    truth = _ecef_track()
    state = _state(truth)

    out = postprocess_output_state(
        state,
        phone_name="unknown",
        apply_relative_height=False,
        apply_position_offset=False,
        reference_wls=truth,
        stop_epochs=None,
    )

    np.testing.assert_allclose(out, state)
    assert not np.shares_memory(out, state)
