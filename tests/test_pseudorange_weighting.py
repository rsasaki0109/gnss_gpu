from types import SimpleNamespace

import numpy as np

from gnss_gpu.pseudorange_weighting import apply_pseudorange_weighting


def test_residual_downweight_reduces_large_residual_weight():
    spp = np.array([2_000_000.0, 0.0, 0.0])
    sat = np.array(
        [
            [20_000_000.0, 0.0, 0.0],
            [21_000_000.0, 0.0, 0.0],
            [22_000_000.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    ranges = np.linalg.norm(sat - spp, axis=1)
    weights = apply_pseudorange_weighting(
        [SimpleNamespace(prn=i) for i in range(3)],
        sat,
        ranges + np.array([0.0, 0.0, 20.0]),
        np.ones(3),
        spp,
        {},
        residual_downweight=True,
        residual_threshold=10.0,
        pr_accel_downweight=False,
        pr_accel_threshold=5.0,
    )

    assert weights[0] == 1.0
    assert weights[1] == 1.0
    assert weights[2] < 1.0


def test_residual_downweight_ignores_invalid_spp_position():
    weights = apply_pseudorange_weighting(
        [SimpleNamespace(prn=1)],
        np.array([[20_000_000.0, 0.0, 0.0]]),
        np.array([20_000_000.0]),
        np.array([0.7]),
        np.array([np.nan, 0.0, 0.0]),
        {},
        residual_downweight=True,
        residual_threshold=10.0,
        pr_accel_downweight=False,
        pr_accel_threshold=5.0,
    )

    np.testing.assert_allclose(weights, [0.7])


def test_pr_accel_downweight_updates_history_and_keeps_two_samples():
    history = {7: [100.0, 101.0]}
    weights = apply_pseudorange_weighting(
        [SimpleNamespace(prn=7)],
        np.array([[20_000_000.0, 0.0, 0.0]]),
        np.array([120.0]),
        np.ones(1),
        np.array([2_000_000.0, 0.0, 0.0]),
        history,
        residual_downweight=False,
        residual_threshold=10.0,
        pr_accel_downweight=True,
        pr_accel_threshold=5.0,
    )

    assert weights[0] < 1.0
    assert history[7] == [101.0, 120.0]


def test_pr_accel_downweight_without_history_preserves_weight_first_epoch():
    history = {}
    weights = apply_pseudorange_weighting(
        [SimpleNamespace(prn=9)],
        np.array([[20_000_000.0, 0.0, 0.0]]),
        np.array([120.0]),
        np.array([0.8]),
        np.array([2_000_000.0, 0.0, 0.0]),
        history,
        residual_downweight=False,
        residual_threshold=10.0,
        pr_accel_downweight=True,
        pr_accel_threshold=5.0,
    )

    np.testing.assert_allclose(weights, [0.8])
    assert history[9] == [120.0]
