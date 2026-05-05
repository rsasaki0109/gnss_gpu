from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.audit_gsdc2023_matlab_equivalence_gate import (
    DEFAULT_EQUIVALENCE_TRIPS,
    _asset_gate,
    _count_gate,
    _factor_gate,
    _residual_gate,
)


def test_asset_gate_requires_base_and_ground_truth_for_all_trips(tmp_path: Path) -> None:
    def fake_asset(_data_root: Path, split: str, *, include_imu_sync: bool) -> pd.DataFrame:
        assert split == "train"
        assert include_imu_sync is False
        return pd.DataFrame(
            [
                {
                    "base_correction_ready": True,
                    "ground_truth_present": True,
                    "ref_height_present": False,
                },
                {
                    "base_correction_ready": True,
                    "ground_truth_present": False,
                    "ref_height_present": False,
                },
            ],
        )

    audit, result = _asset_gate(
        tmp_path,
        ["train"],
        include_imu_sync=False,
        strict_ref_height=False,
        asset_audit_fn=fake_asset,
    )

    assert audit.shape[0] == 2
    assert result.passed is False
    assert result.summary["base_correction_ready"] == 2
    assert result.summary["ground_truth_present"] == 1


def test_factor_gate_passes_through_strict_side_only_failure(tmp_path: Path) -> None:
    def fake_factor(
        _data_root: Path,
        trips,
        *,
        max_epochs: int,
        multi_gnss: bool,
        min_symmetric_parity: float,
        verbose: bool,
    ):
        assert list(trips) == ["train/course/phone"]
        assert max_epochs == 0
        assert multi_gnss is False
        assert min_symmetric_parity == 1.0
        assert verbose is False
        payload = {
            "trip_count": 1,
            "completed_trip_count": 1,
            "error_count": 0,
            "overall_min_symmetric_parity": 0.98,
            "total_matlab_only": 0,
            "total_bridge_only": 1,
            "side_only_failure_count": 1,
            "side_only_by_field_freq": {"P": {"L1": {"matlab_only": 0, "bridge_only": 1}}},
            "top_bridge_only": [{"trip": "train/course/phone", "field": "P", "freq": "L1"}],
            "passed": False,
        }
        return pd.DataFrame(), pd.DataFrame(), payload

    _trip_summary, _field_summary, result = _factor_gate(
        tmp_path,
        ["train/course/phone"],
        max_epochs=0,
        multi_gnss=False,
        min_symmetric_parity=1.0,
        factor_audit_fn=fake_factor,
    )

    assert result.passed is False
    assert result.summary["total_bridge_only"] == 1
    assert result.summary["side_only_failure_count"] == 1
    assert result.summary["side_only_by_field_freq"]["P"]["L1"]["bridge_only"] == 1
    assert result.summary["top_bridge_only"][0]["field"] == "P"
    assert result.summary["overall_min_symmetric_parity"] == 0.98


def test_residual_gate_reports_worst_delta(tmp_path: Path) -> None:
    def fake_residual(
        _data_root: Path,
        trips,
        *,
        max_epochs: int,
        multi_gnss: bool,
        apply_observation_mask: bool,
        include_inactive_observations: bool,
        max_abs_delta_threshold_m: float,
        p95_abs_delta_threshold_m: float | None,
        verbose: bool,
    ):
        assert list(trips) == ["train/course/phone"]
        assert max_epochs == 25
        assert multi_gnss is True
        assert apply_observation_mask is True
        assert include_inactive_observations is True
        assert max_abs_delta_threshold_m == 1.0e-4
        assert p95_abs_delta_threshold_m is None
        assert verbose is False
        payload = {
            "trip_count": 1,
            "completed_trip_count": 1,
            "error_count": 0,
            "overall_max_abs_delta": 9.0e-5,
            "overall_p95_abs_delta_max": 7.0e-5,
            "internal_delta_failure_count": 0,
            "internal_delta_failures": [],
            "worst_trip": "train/course/phone",
            "worst_field": "P",
            "passed": True,
        }
        return pd.DataFrame(), pd.DataFrame(), payload

    _trip_summary, _max_rows, result = _residual_gate(
        tmp_path,
        ["train/course/phone"],
        max_epochs=25,
        multi_gnss=True,
        apply_observation_mask=True,
        include_inactive_observations=True,
        max_abs_delta_threshold_m=1.0e-4,
        p95_abs_delta_threshold_m=None,
        residual_audit_fn=fake_residual,
    )

    assert result.passed is True
    assert result.summary["overall_max_abs_delta"] == 9.0e-5
    assert result.summary["worst_field"] == "P"
    assert result.summary["internal_delta_failure_count"] == 0


def test_residual_gate_fails_on_side_only_rows_even_when_deltas_pass(tmp_path: Path) -> None:
    def fake_residual(_data_root: Path, _trips, **_kwargs):
        trip_summary = pd.DataFrame(
            [
                {
                    "trip": "train/course/phone",
                    "matlab_only_count": 0,
                    "bridge_only_count": 2,
                    "max_abs_delta": 1.0e-6,
                },
            ],
        )
        payload = {
            "trip_count": 1,
            "completed_trip_count": 1,
            "error_count": 0,
            "overall_max_abs_delta": 1.0e-6,
            "overall_p95_abs_delta_max": 1.0e-6,
            "passed": True,
        }
        return trip_summary, pd.DataFrame(), payload

    _trip_summary, _max_rows, result = _residual_gate(
        tmp_path,
        ["train/course/phone"],
        max_epochs=0,
        multi_gnss=True,
        apply_observation_mask=True,
        include_inactive_observations=True,
        max_abs_delta_threshold_m=1.0e-4,
        p95_abs_delta_threshold_m=None,
        residual_audit_fn=fake_residual,
    )

    assert result.passed is False
    assert result.summary["total_bridge_only"] == 2


def test_residual_gate_reports_internal_delta_failures(tmp_path: Path) -> None:
    def fake_residual(_data_root: Path, _trips, **_kwargs):
        payload = {
            "trip_count": 1,
            "completed_trip_count": 1,
            "error_count": 0,
            "overall_max_abs_delta": 1.0e-6,
            "overall_p95_abs_delta_max": 1.0e-6,
            "internal_delta_failure_count": 1,
            "internal_delta_failures": [
                {
                    "trip": "train/course/phone",
                    "component": "model_delta",
                    "max_abs_delta": 2.0e-4,
                    "threshold": 1.0e-4,
                },
            ],
            "passed": False,
        }
        return pd.DataFrame(), pd.DataFrame(), payload

    _trip_summary, _max_rows, result = _residual_gate(
        tmp_path,
        ["train/course/phone"],
        max_epochs=0,
        multi_gnss=True,
        apply_observation_mask=True,
        include_inactive_observations=True,
        max_abs_delta_threshold_m=1.0e-4,
        p95_abs_delta_threshold_m=None,
        residual_audit_fn=fake_residual,
    )

    assert result.passed is False
    assert result.summary["internal_delta_failure_count"] == 1
    assert result.summary["internal_delta_failures"][0]["component"] == "model_delta"


def test_count_gate_requires_exact_count_parity(tmp_path: Path) -> None:
    def fake_count(_data_root: Path, datasets, *, trips, max_epochs: int, multi_gnss: bool):
        assert datasets == ["train"]
        assert list(trips) == ["train/course/phone"]
        assert max_epochs == 0
        assert multi_gnss is False
        payload = {
            "trip_count": 1,
            "trips_with_phone_data": 1,
            "bridge_errors": 0,
            "phone_errors": 0,
            "matched_rows": 12,
            "matched_abs_delta_total": 1,
            "count_delta_failure_count": 1,
            "worst_count_delta": {
                "trip": "train/course/phone",
                "freq": "L1",
                "field": "P",
                "phone_count": 9,
                "bridge_count": 10,
                "count_delta": 1,
                "abs_count_delta": 1,
            },
            "top_count_delta_failures": [
                {
                    "trip": "train/course/phone",
                    "freq": "L1",
                    "field": "P",
                    "phone_count": 9,
                    "bridge_count": 10,
                    "count_delta": 1,
                    "abs_count_delta": 1,
                },
            ],
            "abs_delta_sums": {"P": {"L1": 1, "L5": 0}},
            "count_parity_ratio": 0.999,
        }
        return pd.DataFrame(), pd.DataFrame(), payload

    _comparison, _trip_summary, result = _count_gate(
        tmp_path,
        ["train/course/phone"],
        max_epochs=0,
        multi_gnss=False,
        count_audit_fn=fake_count,
    )

    assert result.passed is False
    assert result.summary["matched_abs_delta_total"] == 1
    assert result.summary["count_delta_failure_count"] == 1
    assert result.summary["worst_count_delta"]["field"] == "P"
    assert result.summary["abs_delta_sums"]["P"]["L1"] == 1
    assert result.summary["count_parity_ratio"] == 0.999


def test_default_equivalence_trip_set_covers_factor_and_residual_exports() -> None:
    assert "train/2020-07-08-22-28-us-ca/pixel4xl" in DEFAULT_EQUIVALENCE_TRIPS
    assert "train/2021-12-08-20-28-us-ca-lax-c/pixel5" in DEFAULT_EQUIVALENCE_TRIPS
