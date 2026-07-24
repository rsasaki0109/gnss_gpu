import copy

from experiments.select_wp34_relative_secondary_parent import (
    select_relative_secondary_parent,
)


def _parent(seed: int, secondary_base: float, x: float = 0.0):
    candidates = {
        "segment": [10, 40],
        "seed_parent_candidate_id": seed,
        "candidates": [
            {
                "candidate_id": index,
                "proposal_kind": "offset_seed",
                "position_ecef": [x + 0.1 * index, 0.0, 0.0],
                "final_norm_rms": 1.0 + 0.01 * index,
                "final_cost": 10.0 + index,
            }
            for index in range(4)
        ],
    }
    secondary = {
        "segment": [10, 40],
        "production_input_truth": False,
        "pseudorange_family": "secondary",
        "calibration": None,
        "evidence_epochs": 12,
        "candidates": [
            {"candidate_id": index, "ddpr_median_abs_m": secondary_base + 0.01 * index}
            for index in range(4)
        ],
    }
    return candidates, secondary


def test_selects_relative_winner_then_compact_primary_top3():
    result = select_relative_secondary_parent(
        [_parent(95, 0.8), _parent(94, 1.0), _parent(96, 1.2)]
    )

    assert result["reason"] == "unique_relative_secondary_parent_primary_compact"
    assert result["selected_seed_parent_candidate_id"] == 95
    assert result["primary_top_ids"] == [0, 1, 2]
    assert result["primary_spread_m"] == 0.2


def test_fails_closed_on_small_relative_margin():
    result = select_relative_secondary_parent(
        [_parent(95, 0.8), _parent(94, 0.84), _parent(96, 1.2)]
    )

    assert result["selected_candidate_id"] is None
    assert result["reason"] == "relative_secondary_parent_gate_failed"


def test_rejects_truth_tainted_secondary_artifact():
    tainted = _parent(95, 0.8)
    tainted = (tainted[0], copy.deepcopy(tainted[1]))
    tainted[1]["production_input_truth"] = True

    try:
        select_relative_secondary_parent(
            [tainted, _parent(94, 1.0), _parent(96, 1.2)]
        )
    except ValueError as error:
        assert "truth-free" in str(error)
    else:
        raise AssertionError("truth-tainted secondary evidence was accepted")
