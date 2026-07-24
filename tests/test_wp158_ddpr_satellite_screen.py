from __future__ import annotations

from experiments.build_wp158_ddpr_satellite_screen import (
    aggregate_flags,
    epoch_outliers,
)


def _group(present: dict[str, float], outliers: set[str]) -> dict:
    return {"present": set(present.keys()), "outliers": outliers}


def test_epoch_outliers_clean_case_has_no_outliers() -> None:
    # All satellites agree within the edge threshold -> single cluster.
    residual_map = {"G01": 0.2, "G03": -0.1, "G07": 0.3, "G13": 0.0}
    assert epoch_outliers(residual_map, edge_m=5.0) == set()


def test_epoch_outliers_persistent_bias_satellite_is_flagged() -> None:
    # G07 is far from the consistent cluster in every epoch it appears.
    edge_m = 5.0
    residual_map = {"G01": 0.1, "G03": -0.2, "G13": 0.0, "G07": 25.0}
    assert epoch_outliers(residual_map, edge_m) == {"G07"}


def test_epoch_outliers_below_threshold_case_not_flagged() -> None:
    # G07 differs from the cluster by less than the edge threshold everywhere.
    residual_map = {"G01": 0.0, "G03": 0.1, "G13": -0.1, "G07": 3.0}
    assert epoch_outliers(residual_map, edge_m=5.0) == set()


def test_aggregate_flags_clean_case_no_flags() -> None:
    per_epoch_outliers = [
        _group({"G01": 0.1, "G03": -0.2, "G13": 0.0}, set()) for _ in range(8)
    ]
    flagged, stats = aggregate_flags(per_epoch_outliers, frac_thresh=0.2)
    assert flagged == set()
    for sat_stats in stats.values():
        assert sat_stats["outlier_fraction"] == 0.0


def test_aggregate_flags_persistent_bias_satellite_flagged() -> None:
    per_epoch_outliers = [
        _group({"G01": 0.1, "G03": -0.2, "G07": 25.0}, {"G07"}) for _ in range(8)
    ]
    flagged, stats = aggregate_flags(per_epoch_outliers, frac_thresh=0.2)
    assert flagged == {"G07"}
    assert stats["G07"]["epochs_present"] == 8
    assert stats["G07"]["epochs_outlier"] == 8
    assert stats["G07"]["outlier_fraction"] == 1.0


def test_aggregate_flags_intermittent_2_of_8_flagged_at_frac_0p2() -> None:
    # G07 appears in all 8 epochs but is only an outlier in 2 of them:
    # outlier fraction 0.25 >= frac_thresh 0.2 -> flagged.
    per_epoch_outliers = []
    for i in range(8):
        outliers = {"G07"} if i < 2 else set()
        per_epoch_outliers.append(
            _group({"G01": 0.1, "G03": -0.2, "G07": 0.0}, outliers)
        )
    flagged, stats = aggregate_flags(per_epoch_outliers, frac_thresh=0.2)
    assert flagged == {"G07"}
    assert stats["G07"]["epochs_present"] == 8
    assert stats["G07"]["epochs_outlier"] == 2
    assert stats["G07"]["outlier_fraction"] == 0.25


def test_aggregate_flags_below_threshold_case_not_flagged() -> None:
    # G07 is an outlier in only 1 of 8 epochs: fraction 0.125 < frac_thresh 0.2.
    per_epoch_outliers = []
    for i in range(8):
        outliers = {"G07"} if i < 1 else set()
        per_epoch_outliers.append(
            _group({"G01": 0.1, "G03": -0.2, "G07": 0.0}, outliers)
        )
    flagged, stats = aggregate_flags(per_epoch_outliers, frac_thresh=0.2)
    assert flagged == set()
    assert stats["G07"]["epochs_present"] == 8
    assert stats["G07"]["epochs_outlier"] == 1
    assert abs(stats["G07"]["outlier_fraction"] - 0.125) < 1e-9
