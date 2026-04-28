import numpy as np

from experiments import gsdc2023_residual_model as residual_model


def test_median_clock_prediction_is_group_local():
    residual = np.array([10.0, 11.0, 100.0, 102.0], dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)
    system_kind = np.array([0, 0, 2, 2], dtype=np.int32)

    pred = residual_model.median_clock_prediction(residual, weights, system_kind, n_clock=3)

    np.testing.assert_allclose(pred, np.array([10.0, 10.0, 100.0, 100.0]))


def test_pseudorange_global_isb_by_group_uses_receiver_clock_bias():
    rx = np.array([[1.0e6, 2.0e6, 3.0e6], [1.0e6, 2.0e6, 3.0e6]], dtype=np.float64)
    sat = np.array(
        [
            [[2.1e7, 0.0, 0.0], [0.0, 2.2e7, 0.0]],
            [[2.1e7, 0.0, 0.0], [0.0, 2.2e7, 0.0]],
        ],
        dtype=np.float64,
    )
    groups = np.array([0, 1], dtype=np.int32)
    clock = np.array([100.0, 200.0], dtype=np.float64)
    ranges = residual_model.geometric_range_with_sagnac(sat, rx[:, None, :])
    pseudorange = ranges + clock[:, None] + np.array([[5.0, -7.0], [5.0, -7.0]], dtype=np.float64)
    weights = np.ones((2, 2), dtype=np.float64)

    isb = residual_model.pseudorange_global_isb_by_group(
        sat,
        pseudorange,
        weights,
        rx,
        clock,
        common_bias_group=groups,
    )

    assert isb == {0: 5.0, 1: -7.0}


def test_pseudorange_residual_mask_uses_supplied_common_bias_groups():
    rx = np.array([[1.0e6, 2.0e6, 3.0e6]], dtype=np.float64)
    sat = np.array(
        [
            [
                [2.1e7, 0.0, 0.0],
                [0.0, 2.2e7, 0.0],
                [0.0, 0.0, 2.3e7],
            ],
        ],
        dtype=np.float64,
    )
    groups = np.array([0, 0, 1], dtype=np.int32)
    ranges = residual_model.geometric_range_with_sagnac(sat[0], rx[0])
    pseudorange = np.array([[ranges[0] + 105.0, ranges[1] + 155.0, ranges[2] + 93.0]], dtype=np.float64)
    weights = np.ones((1, 3), dtype=np.float64)

    masked = residual_model.mask_pseudorange_residual_outliers(
        sat,
        pseudorange,
        weights,
        rx,
        threshold_m=20.0,
        receiver_clock_bias_m=np.array([100.0], dtype=np.float64),
        common_bias_group=groups,
        common_bias_by_group={0: 5.0, 1: -7.0},
    )

    assert masked == 1
    np.testing.assert_array_equal(weights, np.array([[1.0, 0.0, 1.0]]))


def test_pseudorange_doppler_consistency_masks_both_endpoints():
    times_ms = np.array([1000.0, 2000.0], dtype=np.float64)
    pseudorange = np.array([[100.0, 100.0], [110.0, 300.0]], dtype=np.float64)
    weights = np.ones((2, 2), dtype=np.float64)
    doppler = np.array([[-10.0, -10.0], [-10.0, -10.0]], dtype=np.float64)
    doppler_weights = np.ones((2, 2), dtype=np.float64)

    masked = residual_model.mask_pseudorange_doppler_consistency(
        times_ms,
        pseudorange,
        weights,
        doppler,
        doppler_weights,
        phone="pixel4",
        threshold_m=40.0,
    )

    assert masked == 2
    np.testing.assert_array_equal(weights, np.array([[1.0, 0.0], [1.0, 0.0]]))
