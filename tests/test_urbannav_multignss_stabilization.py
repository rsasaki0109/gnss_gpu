from __future__ import annotations

import numpy as np

from experiments.exp_urbannav_multignss_stabilization import (
    ComparativeResidualVariant,
    EpochFeature,
    ResidualBiasVetoVariant,
    SolutionGapVetoVariant,
)


def _feature(**overrides) -> EpochFeature:
    base = dict(
        run="Odaiba",
        epoch_index=0,
        gps_time_s=1.0,
        gps_satellite_count=5,
        multi_satellite_count=9,
        extra_satellite_count=4,
        gps_position=np.zeros(3),
        multi_position=np.ones(3),
        ground_truth=np.zeros(3),
        gps_residual_p95_abs_m=60.0,
        gps_residual_max_abs_m=120.0,
        multi_residual_p95_abs_m=55.0,
        multi_residual_max_abs_m=140.0,
        multi_bias_range_m=40.0,
        solution_gap_2d_m=20.0,
    )
    base.update(overrides)
    return EpochFeature(**base)


def test_residual_bias_veto_accepts_clean_multi():
    variant = ResidualBiasVetoVariant(
        residual_p95_max_m=80.0,
        residual_max_abs_m=200.0,
        bias_delta_max_m=60.0,
        extra_sat_min=2,
    )
    assert variant.use_multi(_feature())


def test_residual_bias_veto_rejects_large_bias():
    variant = ResidualBiasVetoVariant(
        residual_p95_max_m=80.0,
        residual_max_abs_m=200.0,
        bias_delta_max_m=60.0,
        extra_sat_min=2,
    )
    assert not variant.use_multi(_feature(multi_bias_range_m=90.0))


def test_comparative_residual_requires_multi_not_worse_than_gps_margin():
    variant = ComparativeResidualVariant(
        residual_margin_m=10.0,
        residual_max_abs_m=250.0,
        bias_delta_max_m=60.0,
        extra_sat_min=2,
    )
    assert variant.use_multi(_feature(multi_residual_p95_abs_m=69.0, gps_residual_p95_abs_m=60.0))
    assert not variant.use_multi(_feature(multi_residual_p95_abs_m=71.0, gps_residual_p95_abs_m=60.0))


def test_solution_gap_veto_rejects_large_disagreement():
    variant = SolutionGapVetoVariant(
        solution_gap_max_m=50.0,
        bias_delta_max_m=60.0,
        extra_sat_min=2,
    )
    assert variant.use_multi(_feature(solution_gap_2d_m=45.0))
    assert not variant.use_multi(_feature(solution_gap_2d_m=55.0))
