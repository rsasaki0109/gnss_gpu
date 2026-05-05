from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.audit_gsdc2023_factor_mask_parity import (
    DEFAULT_FACTOR_MASK_PARITY_TRIPS,
    factor_mask_parity_audit,
)


def test_factor_mask_parity_audit_summarizes_trips_and_field_rows(tmp_path: Path) -> None:
    def fake_compare(
        trip_dir: Path,
        *,
        max_epochs: int,
        multi_gnss: bool,
        pseudorange_residual_mask_m: float,
        pseudorange_residual_mask_l5_m: float,
        doppler_residual_mask_mps: float,
        tdcp_consistency_threshold_m: float,
        pseudorange_doppler_mask_m: float,
    ):
        assert max_epochs == 100
        assert multi_gnss is False
        trip = trip_dir.name
        summary = pd.DataFrame(
            [
                {
                    "field": "P",
                    "freq": "L1",
                    "matlab_count": 10,
                    "bridge_count": 10,
                    "matched_count": 10,
                    "matlab_only": 0,
                    "bridge_only": 0,
                    "symmetric_parity": 1.0,
                    "jaccard": 1.0,
                },
            ],
        )
        payload = {
            "total_matlab_count": 10,
            "total_bridge_count": 10,
            "total_matched_count": 10,
            "total_matlab_only": 0,
            "total_bridge_only": 0,
            "side_only_failure_count": 0,
            "top_matlab_only": [],
            "top_bridge_only": [],
            "symmetric_parity": 1.0,
            "jaccard": 1.0,
            "start_epoch": 0,
            "max_epochs": 100,
        }
        return pd.DataFrame(), summary, payload | {"trip_name": trip}

    trip_summary, field_summary, payload = factor_mask_parity_audit(
        tmp_path,
        ["train/course/phone-a", "train/course/phone-b"],
        max_epochs=100,
        multi_gnss=False,
        compare_fn=fake_compare,
    )

    assert trip_summary["trip"].tolist() == ["train/course/phone-a", "train/course/phone-b"]
    assert trip_summary["symmetric_parity"].tolist() == [1.0, 1.0]
    assert field_summary["trip"].tolist() == ["train/course/phone-a", "train/course/phone-b"]
    assert payload["passed"] is True
    assert payload["completed_trip_count"] == 2
    assert payload["overall_min_symmetric_parity"] == 1.0
    assert payload["side_only_failure_count"] == 0
    assert payload["top_matlab_only"] == []
    assert payload["top_bridge_only"] == []


def test_factor_mask_parity_audit_fails_on_side_only_rows(tmp_path: Path) -> None:
    def fake_compare(_trip_dir: Path, **_kwargs):
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "total_matlab_count": 10,
                "total_bridge_count": 11,
                "total_matched_count": 10,
                "total_matlab_only": 0,
                "total_bridge_only": 1,
                "side_only_failure_count": 1,
                "top_bridge_only": [
                    {
                        "field": "P",
                        "freq": "L1",
                        "epoch_index": 1,
                        "utcTimeMillis": 1000,
                        "next_epoch_index": 0,
                        "nextUtcTimeMillis": 0,
                        "sys": 1,
                        "svid": 3,
                        "side": "bridge_only",
                    },
                ],
                "symmetric_parity": 10.0 / 11.0,
            },
        )

    _trip_summary, _field_summary, payload = factor_mask_parity_audit(
        tmp_path,
        ["train/course/phone"],
        max_epochs=0,
        multi_gnss=False,
        compare_fn=fake_compare,
    )

    assert payload["passed"] is False
    assert payload["total_bridge_only"] == 1
    assert payload["side_only_failure_count"] == 1
    assert payload["top_bridge_only"][0]["trip"] == "train/course/phone"
    assert payload["top_bridge_only"][0]["svid"] == 3
    assert payload["overall_min_symmetric_parity"] == 10.0 / 11.0


def test_factor_mask_parity_audit_sums_side_only_by_field_freq(tmp_path: Path) -> None:
    def fake_compare(trip_dir: Path, **_kwargs):
        summary = pd.DataFrame(
            [
                {
                    "field": "P",
                    "freq": "L1",
                    "matlab_count": 2,
                    "bridge_count": 1,
                    "matched_count": 1,
                    "matlab_only": 1,
                    "bridge_only": 0,
                    "symmetric_parity": 0.5,
                    "jaccard": 0.5,
                },
            ],
        )
        payload = {
            "total_matlab_count": 2,
            "total_bridge_count": 1,
            "total_matched_count": 1,
            "total_matlab_only": 1,
            "total_bridge_only": 0,
            "side_only_failure_count": 1,
            "top_matlab_only": [
                {
                    "field": "P",
                    "freq": "L1",
                    "epoch_index": 2,
                    "utcTimeMillis": 2000,
                    "next_epoch_index": 0,
                    "nextUtcTimeMillis": 0,
                    "sys": 1,
                    "svid": 5,
                    "side": "matlab_only",
                },
            ],
            "symmetric_parity": 0.5,
            "jaccard": 0.5,
            "start_epoch": 0,
            "max_epochs": 10,
        }
        return pd.DataFrame(), summary, payload | {"trip_name": trip_dir.name}

    _trip_summary, _field_summary, payload = factor_mask_parity_audit(
        tmp_path,
        ["train/course/phone-a", "train/course/phone-b"],
        max_epochs=10,
        multi_gnss=False,
        compare_fn=fake_compare,
    )

    assert payload["passed"] is False
    assert payload["total_matlab_only"] == 2
    assert payload["side_only_failure_count"] == 2
    assert payload["side_only_by_field_freq"] == {"P": {"L1": {"matlab_only": 2, "bridge_only": 0}}}
    assert [row["trip"] for row in payload["top_matlab_only"]] == [
        "train/course/phone-a",
        "train/course/phone-b",
    ]


def test_default_factor_mask_parity_trip_set_matches_available_matlab_exports() -> None:
    assert "train/2020-07-08-22-28-us-ca/pixel4xl" in DEFAULT_FACTOR_MASK_PARITY_TRIPS
    assert "train/2022-10-06-21-51-us-ca-mtv-n/sm-a205u" in DEFAULT_FACTOR_MASK_PARITY_TRIPS
