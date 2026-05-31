from __future__ import annotations

import numpy as np

from experiments.eval_gsdc2023_dd_carrier_anchor import (
    DDCarrierAnchorConfig,
    _dd_expected_m,
    dd_carrier_fixed_ambiguity_update,
    smooth_anchor_corrections,
)
from gnss_gpu.dd_carrier import DDResult


def _dd_result_for_position(position: np.ndarray, wavelength: float = 0.19) -> DDResult:
    sat_ref = np.array(
        [
            [20_200_000.0, 0.0, 1_000_000.0],
            [20_200_000.0, 0.0, 1_000_000.0],
            [20_200_000.0, 0.0, 1_000_000.0],
            [20_200_000.0, 0.0, 1_000_000.0],
        ],
        dtype=np.float64,
    )
    sat_k = np.array(
        [
            [21_000_000.0, 2_000_000.0, 1_500_000.0],
            [19_000_000.0, -3_000_000.0, 2_000_000.0],
            [20_500_000.0, 4_000_000.0, -2_000_000.0],
            [22_000_000.0, -1_000_000.0, -3_000_000.0],
        ],
        dtype=np.float64,
    )
    template = DDResult(
        dd_carrier_cycles=np.zeros(4, dtype=np.float64),
        sat_ecef_k=sat_k,
        sat_ecef_ref=sat_ref,
        base_range_k=np.zeros(4, dtype=np.float64),
        base_range_ref=np.zeros(4, dtype=np.float64),
        dd_weights=np.ones(4, dtype=np.float64),
        wavelengths_m=np.full(4, wavelength, dtype=np.float64),
        ref_sat_ids=("G01",) * 4,
        n_dd=4,
        sat_ids=("G02", "G03", "G04", "G05"),
    )
    expected_m, _ = _dd_expected_m(position, template)
    template.dd_carrier_cycles = expected_m / wavelength + np.array([101, 102, 103, 104], dtype=np.float64)
    return template


def test_dd_carrier_fixed_ambiguity_update_moves_seed_toward_consistent_position() -> None:
    truth = np.array([6_378_137.0, 10.0, 20.0], dtype=np.float64)
    seed = truth + np.array([0.35, -0.25, 0.15], dtype=np.float64)
    dd = _dd_result_for_position(truth, wavelength=10.0)

    updated, stats = dd_carrier_fixed_ambiguity_update(
        seed,
        dd,
        DDCarrierAnchorConfig(
            min_dd_pairs=4,
            sigma_cycles=0.20,
            prior_sigma_m=10.0,
            max_shift_m=2.0,
            max_initial_rms_m=2.0,
            max_final_rms_m=0.05,
        ),
    )

    assert stats["accepted"] is True
    assert np.linalg.norm(updated - truth) < np.linalg.norm(seed - truth)


def test_dd_carrier_fixed_ambiguity_update_rejects_too_few_pairs() -> None:
    dd = _dd_result_for_position(np.array([6_378_137.0, 0.0, 0.0], dtype=np.float64))
    dd.n_dd = 2

    updated, stats = dd_carrier_fixed_ambiguity_update(
        np.array([6_378_137.0, 0.0, 0.0], dtype=np.float64),
        dd,
        DDCarrierAnchorConfig(min_dd_pairs=4),
    )

    assert stats["accepted"] is False
    assert stats["reason"] == "few_pairs"
    np.testing.assert_allclose(updated, np.array([6_378_137.0, 0.0, 0.0], dtype=np.float64))


def test_smooth_anchor_corrections_spreads_sparse_anchor_deltas() -> None:
    corrections = smooth_anchor_corrections(
        5,
        np.array([0, 4], dtype=np.int64),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        anchor_sigma_m=0.1,
        smooth_sigma_m=0.2,
        zero_sigma_m=10.0,
    )

    assert corrections.shape == (5, 3)
    assert corrections[2, 0] > 0.8
    assert abs(corrections[2, 1]) < 1.0e-12


def test_smooth_anchor_corrections_returns_zero_without_anchors() -> None:
    corrections = smooth_anchor_corrections(
        3,
        np.array([], dtype=np.int64),
        np.zeros((0, 3), dtype=np.float64),
        anchor_sigma_m=0.1,
        smooth_sigma_m=0.2,
        zero_sigma_m=10.0,
    )

    np.testing.assert_allclose(corrections, np.zeros((3, 3), dtype=np.float64))
