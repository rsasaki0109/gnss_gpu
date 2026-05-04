from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.audit_gsdc2023_residual_value_parity import (
    DEFAULT_RESIDUAL_PARITY_TRIPS,
    residual_value_parity_audit,
)


def test_residual_value_parity_audit_summarizes_trips_and_thresholds(tmp_path: Path) -> None:
    def fake_compare(
        trip_dir: Path,
        *,
        max_epochs: int,
        multi_gnss: bool,
        apply_observation_mask: bool,
        include_inactive_observations: bool,
    ):
        assert apply_observation_mask is False
        assert include_inactive_observations is True
        trip = trip_dir.name
        delta = 2.5e-5 if trip == "phone-a" else -7.0e-5
        merged = pd.DataFrame(
            [
                {
                    "side": "both",
                    "field": "D",
                    "freq": "L1",
                    "epoch_index": 3,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 7,
                    "delta": delta,
                    "pre_residual_delta": delta,
                    "model_delta": -delta,
                    "sat_velocity_delta_norm": abs(delta) * 2.0,
                },
                {
                    "side": "both",
                    "field": "P",
                    "freq": "L1",
                    "epoch_index": 3,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 7,
                    "delta": 1.0e-8,
                    "pre_residual_delta": 1.0e-8,
                },
            ],
        )
        payload = {
            "total_matched_count": 2,
            "total_matlab_only": 0,
            "total_bridge_only": 0,
            "median_abs_delta": 1.0e-6,
            "p95_abs_delta": abs(delta),
            "max_abs_delta": abs(delta),
            "median_abs_pre_residual_delta": 1.0e-6,
            "p95_abs_pre_residual_delta": abs(delta),
            "max_abs_pre_residual_delta": abs(delta),
        }
        return merged, pd.DataFrame(), payload

    summary, max_rows, payload = residual_value_parity_audit(
        tmp_path,
        ["train/course/phone-a", "train/course/phone-b"],
        max_epochs=100,
        multi_gnss=True,
        apply_observation_mask=False,
        include_inactive_observations=True,
        max_abs_delta_threshold_m=1.0e-4,
        p95_abs_delta_threshold_m=8.0e-5,
        verbose=True,
        compare_fn=fake_compare,
    )

    assert summary["trip"].tolist() == ["train/course/phone-a", "train/course/phone-b"]
    assert summary["matched_count"].tolist() == [2, 2]
    assert max_rows.loc[max_rows["trip"].eq("train/course/phone-b"), "svid"].iloc[0] == 7
    assert payload["passed"] is True
    assert payload["total_matlab_only"] == 0
    assert payload["total_bridge_only"] == 0
    assert payload["completed_trip_count"] == 2
    assert payload["overall_max_abs_delta"] == 7.0e-5
    assert payload["worst_trip"] == "train/course/phone-b"
    assert payload["worst_field"] == "D"


def test_residual_value_parity_audit_fails_when_threshold_exceeded(tmp_path: Path) -> None:
    def fake_compare(
        _trip_dir: Path,
        *,
        max_epochs: int,
        multi_gnss: bool,
        apply_observation_mask: bool,
        include_inactive_observations: bool,
    ):
        merged = pd.DataFrame(
            [
                {
                    "side": "both",
                    "field": "D",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 2,
                    "delta": 2.0e-4,
                },
            ],
        )
        return merged, pd.DataFrame(), {"max_abs_delta": 2.0e-4, "p95_abs_delta": 2.0e-4}

    _summary, _max_rows, payload = residual_value_parity_audit(
        tmp_path,
        ["train/course/phone"],
        max_epochs=0,
        multi_gnss=False,
        max_abs_delta_threshold_m=1.0e-4,
        compare_fn=fake_compare,
    )

    assert payload["passed"] is False
    assert payload["overall_max_abs_delta"] == 2.0e-4


def test_residual_value_parity_audit_fails_on_side_only_rows(tmp_path: Path) -> None:
    def fake_compare(
        _trip_dir: Path,
        *,
        max_epochs: int,
        multi_gnss: bool,
        apply_observation_mask: bool,
        include_inactive_observations: bool,
    ):
        merged = pd.DataFrame(
            [
                {
                    "side": "both",
                    "field": "P",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 2,
                    "delta": 1.0e-6,
                },
            ],
        )
        return (
            merged,
            pd.DataFrame(),
            {
                "total_matched_count": 1,
                "total_matlab_only": 1,
                "total_bridge_only": 0,
                "max_abs_delta": 1.0e-6,
                "p95_abs_delta": 1.0e-6,
            },
        )

    _summary, _max_rows, payload = residual_value_parity_audit(
        tmp_path,
        ["train/course/phone"],
        max_epochs=0,
        multi_gnss=False,
        max_abs_delta_threshold_m=1.0e-4,
        compare_fn=fake_compare,
    )

    assert payload["passed"] is False
    assert payload["total_matlab_only"] == 1
    assert payload["total_bridge_only"] == 0


def test_default_residual_parity_trip_set_is_nonempty() -> None:
    assert "train/2020-07-08-22-28-us-ca/pixel4xl" in DEFAULT_RESIDUAL_PARITY_TRIPS
