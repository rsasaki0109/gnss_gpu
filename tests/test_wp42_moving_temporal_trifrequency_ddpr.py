from __future__ import annotations

from experiments.analyze_wp42_moving_temporal_trifrequency_ddpr import (
    pair_centered_metrics,
    select_rank_consensus,
)


def test_pair_centered_metrics_removes_constant_pair_bias() -> None:
    rows = [
        (epoch, "G01", "G02", 100.0 + value)
        for epoch, value in enumerate((-1.0, 0.0, 1.0))
    ]
    metrics = pair_centered_metrics(rows, min_pair_epochs=3)
    assert metrics["temporal_rows"] == 3
    assert metrics["temporal_pairs"] == 1
    assert metrics["temporal_epochs"] == 3
    assert abs(float(metrics["temporal_median_abs_m"]) - 1.0) < 1e-12


def test_rank_consensus_accepts_unique_three_family_winner() -> None:
    families = {}
    for family, order in {
        "primary": (1, 2, 3),
        "secondary": (1, 3, 2),
        "tertiary": (1, 2, 3),
    }.items():
        families[family] = [
            {
                "candidate_id": candidate_id,
                    "offset_ecef_m": [float(candidate_id), 0.0, 0.0],
                    "temporal_rms_m": float(rank),
                    "temporal_median_abs_m": float(rank),
                "temporal_epochs": 10,
                "temporal_pairs": 3,
                "temporal_rows": 30,
            }
            for rank, candidate_id in enumerate(order, start=1)
        ]
    selected = select_rank_consensus(
        families, max_family_rank_fraction=0.34, min_runner_margin=0.2
    )
    assert selected["selected_candidate_id"] == 1
    assert (
        selected["reason"] == "unique_moving_temporal_trifrequency_ddpr_rank_consensus"
    )


def test_rank_consensus_rejects_missing_family_supply() -> None:
    families = {
        family: [
            {
                "candidate_id": candidate_id,
                "offset_ecef_m": [float(candidate_id), 0.0, 0.0],
                "temporal_rms_m": float(candidate_id),
                "temporal_median_abs_m": float(candidate_id),
                "temporal_epochs": 9 if family == "tertiary" else 10,
                "temporal_pairs": 3,
                "temporal_rows": 30,
            }
            for candidate_id in (1, 2, 3)
        ]
        for family in ("primary", "secondary", "tertiary")
    }
    selected = select_rank_consensus(families, max_family_rank_fraction=0.34)
    assert selected["selected_candidate_id"] is None
