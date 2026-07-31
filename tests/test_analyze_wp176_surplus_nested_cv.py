from experiments.analyze_wp176_surplus_nested_cv import (
    Example,
    Policy,
    _counts,
    analyze,
)


def _row(
    *,
    distance: float = 0.04,
    pairs: int = 12,
    ratio: float = 2.0,
) -> dict[str, str]:
    return {
        "satellite_par_surplus_max_distance_cycles": str(distance),
        "satellite_par_subset_size": str(pairs),
        "satellite_par_ratio": str(ratio),
        "float_update_nis_per_observation": "1.0",
        "float_update_prefit_residual_rms_m": "1.0",
    }


def test_counts_separates_correct_and_wrong_selected_candidates() -> None:
    examples = [
        Example("tokyo", 0, _row(), 0.1),
        Example("tokyo", 0, _row(), 0.8),
        Example("tokyo", 0, _row(distance=0.2), 0.1),
    ]

    assert _counts(examples, Policy(0.05, 12, 2.0)) == (1, 1)


def test_analyze_holds_out_every_city_time_block() -> None:
    examples = [
        Example(city, block, _row(), 0.1)
        for city in ("tokyo", "nagoya")
        for block in range(5)
    ]

    payload = analyze(examples)

    assert len(payload["folds"]) == 10
    assert payload["aggregate_holdout_correct"] == 10
    assert payload["aggregate_holdout_wrong"] == 0
    assert all(
        fold["policy"]
        == {
            "maximum_distance_cycles": 0.05,
            "minimum_subset_pairs": 12,
            "minimum_ratio": 2.0,
        }
        for fold in payload["folds"]
    )
