from __future__ import annotations

import numpy as np
import pytest

from gnss_gpu.ppc_score import (
    ppc_3d_errors,
    ppc_score_dict,
    ppc_segment_distances,
    score_ppc2024,
)


def test_ppc_score_is_distance_weighted():
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )
    estimated = reference.copy()
    estimated[1, 2] += 0.4
    estimated[2, 2] += 0.6

    score = score_ppc2024(estimated, reference, threshold_m=0.5)

    np.testing.assert_allclose(score.errors_3d, [0.0, 0.4, 0.6])
    np.testing.assert_allclose(score.segment_distances_m, [0.0, 3.0, 4.0])
    assert score.pass_mask.tolist() == [True, True, False]
    assert score.pass_distance_m == pytest.approx(3.0)
    assert score.total_distance_m == pytest.approx(7.0)
    assert score.score_pct == pytest.approx(100.0 * 3.0 / 7.0)
    assert score.epoch_pass_pct == pytest.approx(100.0 * 2.0 / 3.0)
    assert not score.fallback_epoch_weighted


def test_ppc_score_falls_back_to_epoch_weighting_for_static_fixture():
    reference = np.zeros((3, 3), dtype=np.float64)
    estimated = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    score = score_ppc2024(estimated, reference)

    assert score.fallback_epoch_weighted
    assert score.pass_distance_m == pytest.approx(2.0)
    assert score.total_distance_m == pytest.approx(3.0)
    assert score.score_pct == pytest.approx(100.0 * 2.0 / 3.0)


def test_ppc_score_accepts_external_distance_weights():
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    estimated = reference.copy()
    estimated[1, 2] += 0.4
    estimated[2, 2] += 0.6

    score = score_ppc2024(
        estimated,
        reference,
        segment_distances_m=np.array([0.0, 10.0, 1.0], dtype=np.float64),
    )

    np.testing.assert_allclose(score.segment_distances_m, [0.0, 10.0, 1.0])
    assert score.pass_distance_m == pytest.approx(10.0)
    assert score.total_distance_m == pytest.approx(11.0)
    assert score.score_pct == pytest.approx(100.0 * 10.0 / 11.0)


def test_ppc_score_dict_is_csv_ready():
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    estimated = reference.copy()

    row = ppc_score_dict(estimated, reference)

    assert row["ppc_score_pct"] == pytest.approx(100.0)
    assert row["ppc_pass_distance_m"] == pytest.approx(1.0)
    assert row["ppc_total_distance_m"] == pytest.approx(1.0)
    assert row["ppc_threshold_m"] == pytest.approx(0.5)
    assert row["ppc_n_epochs"] == 2
    assert row["ppc_fallback_epoch_weighted"] is False


def test_ppc_helpers_validate_shapes():
    with pytest.raises(ValueError):
        ppc_3d_errors(np.zeros((2, 3)), np.zeros((3, 3)))

    with pytest.raises(ValueError):
        ppc_segment_distances(np.zeros((3, 2)))

    with pytest.raises(ValueError):
        score_ppc2024(np.zeros((3, 3)), np.zeros((3, 3)), segment_distances_m=np.zeros(2))

    with pytest.raises(ValueError):
        score_ppc2024(np.zeros((3, 3)), np.zeros((3, 3)), segment_distances_m=[0.0, -1.0, 0.0])
