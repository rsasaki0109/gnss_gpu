import copy

import pytest

from experiments.select_wp38_trifrequency_ddpr_rank import (
    select_trifrequency_ddpr_rank,
)


def _family(name: str, ranks: list[int], *, segment=(10, 40)):
    return {
        "schema": (
            "wp31_static_ddpr_integrity_v1"
            if name == "primary"
            else f"test_{name}_ddpr"
        ),
        "segment": list(segment),
        "production_input_truth": False,
        "pseudorange_family": name,
        "calibration": None,
        "evidence_epochs": 12,
        "candidate_source_sha256": "abc",
        "candidates": [
            {
                "candidate_id": candidate_id,
                "position_ecef": [float(candidate_id), 0.0, 0.0],
                "ddpr_median_abs_m": 0.5 + 0.1 * rank,
                "ddpr_median_abs_m_rank": rank,
            }
            for candidate_id, rank in enumerate(ranks)
        ],
    }


def test_selects_unique_trifrequency_rank_winner():
    result = select_trifrequency_ddpr_rank(
        _family("primary", [1, 2, 3, 4, 5]),
        _family("secondary", [1, 3, 2, 4, 5]),
        _family("tertiary", [1, 3, 2, 4, 5]),
        max_family_rank_fraction=0.4,
    )

    assert result["reason"] == "unique_trifrequency_ddpr_rank_consensus"
    assert result["selected_candidate_id"] == 0
    assert result["winner"]["rank_sum"] == 3
    assert result["runner_margin"] == pytest.approx(4 / 3)


def test_fails_closed_on_small_runner_margin():
    result = select_trifrequency_ddpr_rank(
        _family("primary", [1, 2, 3, 4, 5]),
        _family("secondary", [2, 1, 3, 4, 5]),
        _family("tertiary", [1, 1, 3, 4, 5]),
        max_family_rank_fraction=0.4,
    )

    assert result["selected_candidate_id"] is None
    assert result["runner_margin_pass"] is False


def test_fails_closed_when_one_family_rank_is_outside_top_fraction():
    result = select_trifrequency_ddpr_rank(
        _family("primary", [1, 2, 3, 4, 5]),
        _family("secondary", [1, 2, 3, 4, 5]),
        _family("tertiary", [3, 1, 2, 4, 5]),
        max_family_rank_fraction=0.4,
    )

    assert result["selected_candidate_id"] is None
    assert result["family_rank_pass"] is False


def test_rejects_truth_tainted_or_mismatched_provenance():
    documents = [
        _family("primary", [1, 2, 3]),
        _family("secondary", [1, 2, 3]),
        _family("tertiary", [1, 2, 3]),
    ]
    tainted = copy.deepcopy(documents)
    tainted[1]["production_input_truth"] = True
    with pytest.raises(ValueError, match="truth-free"):
        select_trifrequency_ddpr_rank(*tainted)

    mismatched = copy.deepcopy(documents)
    mismatched[2]["candidate_source_sha256"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        select_trifrequency_ddpr_rank(*mismatched)
