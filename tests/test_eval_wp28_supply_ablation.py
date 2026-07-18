import csv

from experiments.eval_wp28_supply_ablation import _summarize, _true_spans


def test_true_spans_handles_edges_and_gaps():
    assert _true_spans([True, True, False, True, False, True, True, True]) == [2, 1, 3]
    assert _true_spans([]) == []


def test_summarize_separates_generated_then_pruned_candidate(tmp_path):
    path = tmp_path / "epochs.csv"
    fieldnames = [
        "basin_oracle_min_error_m",
        "respawn_triggered",
        "respawn_oracle_min_error_m",
        "respawn_oracle_rank",
        "n_basins",
        "n_respawn_candidates_born",
        "n_respawn_position_seeds",
        "n_respawn_history_seeds",
        "fix",
        "output_error_m",
        "integrity_map_error_m",
    ]
    rows = [
        ["0.8", "1", "0.2", "9", "8", "10", "2", "1", "0", "3", "nan"],
        ["0.3", "1", "0.1", "2", "8", "10", "2", "1", "0", "2", "0.4"],
    ]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    summary = _summarize(path)
    assert summary["proposal_correct_anchor_epochs"] == 2
    assert summary["proposal_correct_anchor_pct"] == 100.0
    assert summary["proposal_correct_but_not_live_anchor_epochs"] == 1
