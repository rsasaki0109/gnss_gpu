from __future__ import annotations

from experiments.select_wp54_cppr_rank import select


def _row(seed_id: int, median: float, p95: float, bad: int) -> dict:
    return {
        "seed_id": seed_id,
        "offset_ecef_m": [float(seed_id), 0.0, 0.0],
        "block_offsets_ecef_m": [[float(seed_id), 0.0, 0.0]] * 2,
        "carrier_rms_cycles": 0.2,
        "block_spread_m": 0.1,
        "cp_pr_consistency": {
            "checked_pairs": 100,
            "median_abs_innovation_m": median,
            "p95_abs_innovation_m": p95,
            "bad_pairs": bad,
        },
    }


def test_select_cppr_rank_requires_top_fraction_and_margin() -> None:
    source = {
        "production_input_truth": False,
        "segment": [10, 20],
        "hypotheses": [
            _row(0, 0.2, 1.0, 2),
            _row(1, 0.4, 2.0, 3),
            _row(2, 0.5, 2.2, 5),
            _row(3, 0.6, 2.5, 6),
            _row(4, 0.7, 3.0, 7),
        ],
    }

    result = select(source)

    assert result["selected_candidate_id"] == 0
    assert result["winner"]["rank_sum"] == 3
    assert result["family_rank_limit"] == 1
    assert result["runner_margin"] >= 0.2
    assert result["absolute_gate_pass"] is True
    assert "audit" not in str(result)


def test_select_cppr_rank_fails_closed_when_winner_is_block_unstable() -> None:
    winner = _row(0, 0.2, 1.0, 0)
    winner["block_spread_m"] = 0.51
    source = {
        "production_input_truth": False,
        "segment": [10, 20],
        "hypotheses": [winner, _row(1, 1.0, 3.0, 4)],
    }

    result = select(source)

    assert result["selected_candidate_id"] is None
    assert result["reason"] == "cppr_rank_gate_failed"
    assert result["absolute_gate"]["block_spread_pass"] is False


def test_select_cppr_rank_does_not_break_metric_ties_by_seed_id() -> None:
    first = _row(0, 0.2, 2.0, 0)
    second = _row(9, 0.3, 1.0, 0)
    source = {
        "production_input_truth": False,
        "segment": [10, 20],
        "hypotheses": [first, second],
    }

    result = select(source)

    rows = {row["candidate_id"]: row for row in result["candidates"]}
    assert rows[0]["family_ranks"]["bad_pairs"] == 1
    assert rows[9]["family_ranks"]["bad_pairs"] == 1
    assert result["selected_candidate_id"] is None
    assert result["runner_margin"] == 0.0
