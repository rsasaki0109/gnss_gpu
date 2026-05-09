from __future__ import annotations

import numpy as np

from experiments.evaluate import lla_to_ecef
from experiments.gsdc2023_height_constraints import enu_to_ecef_relative
from experiments.gsdc2023_observation_matrix import TripArrays
from experiments.gsdc2023_solver_context import (
    build_solver_execution_context,
    estimate_speed_mps,
    solver_stop_mask,
)


def _batch(
    *,
    clock_jump: np.ndarray | None = None,
    clock_bias_m: np.ndarray | None = None,
    clock_drift_mps: np.ndarray | None = None,
    stop_epochs: np.ndarray | None = None,
    kaggle_wls: np.ndarray | None = None,
) -> TripArrays:
    n_epoch = 4
    n_sat = 2
    if kaggle_wls is None:
        kaggle_wls = np.zeros((n_epoch, 3), dtype=np.float64)
    return TripArrays(
        times_ms=np.arange(n_epoch, dtype=np.float64) * 1000.0,
        sat_ecef=np.zeros((n_epoch, n_sat, 3), dtype=np.float64),
        pseudorange=np.zeros((n_epoch, n_sat), dtype=np.float64),
        weights=np.ones((n_epoch, n_sat), dtype=np.float64),
        kaggle_wls=kaggle_wls,
        truth=np.zeros((n_epoch, 3), dtype=np.float64),
        max_sats=n_sat,
        has_truth=False,
        clock_jump=clock_jump,
        clock_bias_m=clock_bias_m,
        clock_drift_mps=clock_drift_mps,
        stop_epochs=stop_epochs,
    )


def _baseline_state(clock_bias: list[float]) -> np.ndarray:
    state = np.zeros((len(clock_bias), 4), dtype=np.float64)
    state[:, 3] = np.asarray(clock_bias, dtype=np.float64)
    return state


def test_estimate_speed_and_solver_stop_mask_filter_fast_epochs() -> None:
    origin_xyz = np.asarray(lla_to_ecef(np.deg2rad(35.0), np.deg2rad(139.0), 10.0), dtype=np.float64)
    enu = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    xyz = enu_to_ecef_relative(enu, origin_xyz)
    times_ms = np.array([0, 1000, 2000, 3000, 4000], dtype=np.float64)

    speed_mps = estimate_speed_mps(xyz, times_ms)
    assert np.all(speed_mps[:3] < 0.1)
    assert np.all(speed_mps[3:] >= 0.5)

    mask = solver_stop_mask(np.ones(5, dtype=bool), xyz, times_ms)
    assert mask is not None
    assert mask.tolist() == [True, True, True, False, False]


def test_solver_execution_context_uses_baseline_clock_for_pixel4_and_keeps_drift_seed() -> None:
    drift = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    batch = _batch(clock_jump=np.array([False, False, True, False]), clock_drift_mps=drift)

    context = build_solver_execution_context(
        "pixel4",
        batch,
        _baseline_state([0.0, 120.0, 121.0, 122.0]),
    )

    assert context.tdcp_use_drift is False
    assert context.clock_use_average_drift is False
    assert context.clock_drift_seed_mps is drift
    assert context.clock_jump is not None
    assert context.clock_jump.tolist() == [False, True, True, False]


def test_solver_execution_context_uses_raw_clock_bias_for_clock_drift_blocklist() -> None:
    drift = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    batch = _batch(
        clock_jump=np.array([False, False, False, False]),
        clock_bias_m=np.array([0.0, 3000.0, 3001.0, 3002.0], dtype=np.float64),
        clock_drift_mps=drift,
    )

    context = build_solver_execution_context(
        "sm-a205u",
        batch,
        _baseline_state([0.0, 1.0, 2.0, 3.0]),
    )

    assert context.tdcp_use_drift is True
    assert context.clock_use_average_drift is True
    assert context.clock_drift_seed_mps is drift
    assert context.clock_jump is not None
    assert context.clock_jump.tolist() == [False, True, False, False]


def test_solver_execution_context_skips_drift_seed_for_sm_a505u() -> None:
    batch = _batch(
        clock_bias_m=np.array([0.0, 3000.0, 3001.0, 3002.0], dtype=np.float64),
        clock_drift_mps=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )

    context = build_solver_execution_context(
        "sm-a505u",
        batch,
        _baseline_state([0.0, 1.0, 2.0, 3.0]),
    )

    assert context.tdcp_use_drift is True
    assert context.clock_drift_seed_mps is None
    assert context.clock_jump is not None
    assert context.clock_jump.tolist() == [False, True, False, False]


def test_solver_execution_context_run_kwargs_match_run_fgo_context_arguments() -> None:
    context = build_solver_execution_context("pixel5", _batch(), _baseline_state([0.0, 1.0, 2.0, 3.0]))

    assert context.run_kwargs() == {
        "clock_jump": None,
        "clock_drift_seed_mps": None,
        "clock_use_average_drift": False,
        "tdcp_use_drift": False,
        "stop_mask": None,
    }
