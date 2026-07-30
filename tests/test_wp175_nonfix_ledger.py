from experiments.analyze_wp175_nonfix_ledger import analyze, classify


def test_nonfix_ledger_orders_pipeline_causes_and_is_truth_free() -> None:
    rows = [
        {"library_status": "4"},
        {
            "library_status": "3",
            "primary_ffrt_passed": "0",
        },
        {
            "library_status": "3",
            "primary_ffrt_passed": "1",
            "failure_budget_passed": "0",
            "disjoint_a_ffrt_passed": "0",
            "disjoint_b_ffrt_passed": "1",
        },
        {
            "library_status": "3",
            "primary_ffrt_passed": "1",
            "failure_budget_passed": "1",
            "disjoint_passed": "1",
            "disjoint_consensus_declared_fixed": "0",
        },
    ]
    assert classify(rows[1]) == "primary_ffrt_unavailable"
    result = analyze(rows, "tokyo")
    assert result["truth_usage"] == "none"
    assert result["library_fixed_epochs"] == 1
    assert result["nonfixed_epochs"] == 3
    assert result["exclusive_pipeline_causes"] == {
        "causal_consensus_not_declared": 1,
        "disjoint_partition_ffrt_unavailable": 1,
        "primary_ffrt_unavailable": 1,
    }
    assert (
        result[
            "maximum_recoverable_rate_if_all_budgeted_nonfix_passed_quality"
        ]
        == 0.5
    )
