from __future__ import annotations

import numpy as np

from experiments.gsdc2023_bridge_config import BridgeConfig
from experiments.gsdc2023_chunk_selection import ChunkCandidateQuality, ChunkSelectionRecord
from experiments.gsdc2023_observation_matrix import TripArrays
from experiments.gsdc2023_solver_selection import (
    batch_without_tdcp,
    build_source_solution_catalog,
    mi8_gated_baseline_jump_guard_enabled,
    raw_wls_max_gap_guard_m,
    select_gated_solution,
    tdcp_off_candidate_enabled,
    with_fixed_source_solution,
    with_source_solution,
)


def _batch(*, tdcp_weights: np.ndarray | None = None) -> TripArrays:
    n_epoch = 3
    n_sat = 2
    return TripArrays(
        times_ms=np.arange(n_epoch, dtype=np.float64) * 1000.0,
        sat_ecef=np.zeros((n_epoch, n_sat, 3), dtype=np.float64),
        pseudorange=np.zeros((n_epoch, n_sat), dtype=np.float64),
        weights=np.ones((n_epoch, n_sat), dtype=np.float64),
        kaggle_wls=np.zeros((n_epoch, 3), dtype=np.float64),
        truth=np.zeros((n_epoch, 3), dtype=np.float64),
        max_sats=n_sat,
        has_truth=False,
        tdcp_meas=np.ones((n_epoch - 1, n_sat), dtype=np.float64) if tdcp_weights is not None else None,
        tdcp_weights=tdcp_weights,
        tdcp_consistency_mask_count=4,
        tdcp_geometry_correction_count=5,
    )


def _state(offset: float, n_epoch: int = 4) -> np.ndarray:
    state = np.zeros((n_epoch, 4), dtype=np.float64)
    state[:, 0] = offset
    return state


def _quality(mse_pr: float, quality_score: float, *, gap_max: float = 0.0) -> ChunkCandidateQuality:
    return ChunkCandidateQuality(
        mse_pr=mse_pr,
        step_mean_m=1.0,
        step_p95_m=1.0,
        accel_mean_m=0.0,
        accel_p95_m=0.0,
        bridge_jump_m=0.0,
        baseline_gap_mean_m=0.0,
        baseline_gap_p95_m=0.0,
        baseline_gap_max_m=gap_max,
        quality_score=quality_score,
    )


def test_tdcp_off_candidate_enabled_requires_gated_vd_tdcp_and_weights() -> None:
    weights = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)

    assert tdcp_off_candidate_enabled(BridgeConfig(position_source="gated"), _batch(tdcp_weights=weights))
    assert not tdcp_off_candidate_enabled(BridgeConfig(position_source="auto"), _batch(tdcp_weights=weights))
    assert not tdcp_off_candidate_enabled(BridgeConfig(position_source="gated", use_vd=False), _batch(tdcp_weights=weights))
    assert not tdcp_off_candidate_enabled(
        BridgeConfig(position_source="gated", tdcp_enabled=False),
        _batch(tdcp_weights=weights),
    )
    assert not tdcp_off_candidate_enabled(
        BridgeConfig(position_source="gated"),
        _batch(tdcp_weights=np.zeros_like(weights)),
    )
    assert not tdcp_off_candidate_enabled(BridgeConfig(position_source="gated"), _batch(tdcp_weights=None))


def test_batch_without_tdcp_clears_tdcp_arrays_and_counts() -> None:
    batch = _batch(tdcp_weights=np.ones((2, 2), dtype=np.float64))

    stripped = batch_without_tdcp(batch)

    assert stripped.tdcp_meas is None
    assert stripped.tdcp_weights is None
    assert stripped.tdcp_consistency_mask_count == 0
    assert stripped.tdcp_geometry_correction_count == 0
    assert batch.tdcp_weights is not None
    assert batch.tdcp_consistency_mask_count == 4


def test_mi8_gated_baseline_jump_guard_only_for_gated_mi8_family() -> None:
    assert mi8_gated_baseline_jump_guard_enabled("mi8", "gated")
    assert mi8_gated_baseline_jump_guard_enabled("xiaomimi8", "gated")
    assert not mi8_gated_baseline_jump_guard_enabled("pixel5", "gated")
    assert not mi8_gated_baseline_jump_guard_enabled("mi8", "auto")


def test_raw_wls_max_gap_guard_only_for_gated_pixel5() -> None:
    assert raw_wls_max_gap_guard_m("pixel5", "gated") == 0.0
    assert raw_wls_max_gap_guard_m("pixel5", "auto") is None
    assert raw_wls_max_gap_guard_m("pixel4xl", "gated") is None
    assert raw_wls_max_gap_guard_m("mi8", "gated") is None


def test_select_gated_solution_splices_selected_chunk_and_counts_sources() -> None:
    catalog = build_source_solution_catalog(
        n_epoch=4,
        baseline_state=_state(0.0),
        raw_state=_state(7.0),
        fgo_state=_state(11.0),
        auto_state=_state(7.0),
        auto_sources=np.array(["baseline", "raw_wls", "raw_wls", "baseline"], dtype=object),
        auto_source_counts={"baseline": 2, "raw_wls": 2, "fgo": 0},
        baseline_mse_pr=1000.0,
        raw_wls_mse_pr=10.0,
        fgo_mse_pr=20.0,
        auto_mse_pr=10.0,
    )
    records = [
        ChunkSelectionRecord(
            start_epoch=1,
            end_epoch=3,
            auto_source="raw_wls",
            candidates={
                "baseline": _quality(1000.0, 1.0),
                "raw_wls": _quality(10.0, 0.1),
                "fgo": _quality(20.0, 0.2),
            },
        ),
    ]

    gated_state, gated_sources, gated_counts = select_gated_solution(
        catalog,
        records,
        n_epoch=4,
        baseline_threshold=500.0,
    )

    np.testing.assert_allclose(gated_state[:, 0], np.array([0.0, 7.0, 7.0, 0.0], dtype=np.float64))
    np.testing.assert_array_equal(
        gated_sources.astype(str),
        np.array(["baseline", "raw_wls", "raw_wls", "baseline"], dtype=object),
    )
    assert gated_counts == {"baseline": 2, "raw_wls": 2, "fgo": 0}


def test_select_gated_solution_applies_optional_raw_wls_gap_guard() -> None:
    catalog = build_source_solution_catalog(
        n_epoch=4,
        baseline_state=_state(0.0),
        raw_state=_state(7.0),
        fgo_state=_state(11.0),
        auto_state=_state(7.0),
        auto_sources=np.array(["baseline", "raw_wls", "raw_wls", "baseline"], dtype=object),
        auto_source_counts={"baseline": 2, "raw_wls": 2, "fgo": 0},
        baseline_mse_pr=1000.0,
        raw_wls_mse_pr=10.0,
        fgo_mse_pr=20.0,
        auto_mse_pr=10.0,
    )
    records = [
        ChunkSelectionRecord(
            start_epoch=1,
            end_epoch=3,
            auto_source="raw_wls",
            candidates={
                "baseline": _quality(1000.0, 1.0),
                "raw_wls": _quality(10.0, 0.1, gap_max=250.0),
            },
        ),
    ]

    gated_state, gated_sources, gated_counts = select_gated_solution(
        catalog,
        records,
        n_epoch=4,
        baseline_threshold=500.0,
        raw_wls_max_gap_m=200.0,
    )

    np.testing.assert_allclose(gated_state[:, 0], np.zeros(4, dtype=np.float64))
    np.testing.assert_array_equal(
        gated_sources.astype(str),
        np.array(["baseline", "baseline", "baseline", "baseline"], dtype=object),
    )
    assert gated_counts == {"baseline": 4, "raw_wls": 0, "fgo": 0}


def test_source_catalog_adds_fixed_and_custom_source_entries() -> None:
    catalog = build_source_solution_catalog(
        n_epoch=2,
        baseline_state=_state(0.0, n_epoch=2),
        raw_state=_state(1.0, n_epoch=2),
        fgo_state=_state(2.0, n_epoch=2),
        auto_state=_state(1.0, n_epoch=2),
        auto_sources=np.array(["raw_wls", "raw_wls"], dtype=object),
        auto_source_counts={"baseline": 0, "raw_wls": 2, "fgo": 0},
        baseline_mse_pr=3.0,
        raw_wls_mse_pr=2.0,
        fgo_mse_pr=4.0,
        auto_mse_pr=2.0,
    )

    catalog = with_fixed_source_solution(catalog, source="fgo_no_tdcp", state=_state(5.0, n_epoch=2), mse_pr=1.5)
    catalog = with_source_solution(
        catalog,
        source="gated",
        state=_state(5.0, n_epoch=2),
        source_array=np.array(["baseline", "fgo_no_tdcp"], dtype=object),
        source_counts={"baseline": 1, "raw_wls": 0, "fgo": 0, "fgo_no_tdcp": 1},
        mse_pr=1.7,
    )

    state, sources, counts, mse = catalog.selected("gated")
    np.testing.assert_allclose(state[:, 0], np.array([5.0, 5.0], dtype=np.float64))
    np.testing.assert_array_equal(sources.astype(str), np.array(["baseline", "fgo_no_tdcp"], dtype=object))
    assert counts == {"baseline": 1, "raw_wls": 0, "fgo": 0, "fgo_no_tdcp": 1}
    assert mse == 1.7
