from experiments.eval_wp28_supply_ablation import _true_spans


def test_true_spans_handles_edges_and_gaps():
    assert _true_spans([True, True, False, True, False, True, True, True]) == [2, 1, 3]
    assert _true_spans([]) == []
