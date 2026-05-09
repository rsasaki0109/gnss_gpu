from __future__ import annotations

import json

import pandas as pd

from experiments.audit_gsdc2023_matlab_equivalence_gate import DEFAULT_EQUIVALENCE_TRIPS
from experiments.build_gsdc2023_pre_submit_manifest import (
    DELTA_CHANGED_THRESHOLD_M,
    build_pre_submit_manifest,
    main,
)
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT as DEFAULT_GSDC2023_DATA_ROOT


CANDIDATE = "pixel5phone_3p375_sjc_r0p84375_p6p0"
PREVIOUS = "pixel5phone_3p375_sjc_r0p84375"
RISKY_TRIP = "2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro"
OTHER_RISKY_TRIP = "2023-05-25-17-32-us-ca-pao-j/pixel6pro"
OTHER_TRIP = "2022-10-06-20-46-us-ca-sjc-r/pixel5"


def _base_submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tripId": [
                RISKY_TRIP,
                RISKY_TRIP,
                OTHER_RISKY_TRIP,
                OTHER_RISKY_TRIP,
                OTHER_TRIP,
                OTHER_TRIP,
            ],
            "UnixTimeMillis": [1000, 2000, 1000, 2000, 1000, 2000],
            "LatitudeDegrees": [37.1, 37.10001, 37.2, 37.20001, 37.3, 37.30001],
            "LongitudeDegrees": [-122.1, -122.09999, -122.2, -122.19999, -122.3, -122.29999],
        },
    )


def test_build_pre_submit_manifest_records_clean_p6p0_risky_trips(tmp_path) -> None:
    base = _base_submission()
    candidate = base.copy()
    candidate.loc[candidate["tripId"] == RISKY_TRIP, "LatitudeDegrees"] += 1.0e-12
    candidate.loc[candidate["tripId"] == OTHER_TRIP, "LatitudeDegrees"] += 0.00001
    previous = base.copy()
    previous.loc[previous["tripId"].isin([RISKY_TRIP, OTHER_RISKY_TRIP]), "LatitudeDegrees"] += 0.00001

    base_path = tmp_path / "base.csv"
    output_dir = tmp_path / "out"
    candidate_dir = output_dir / CANDIDATE
    previous_dir = tmp_path / "previous" / "basecorr_posoffset_pixel5_patch_scripted" / PREVIOUS
    candidate_path = candidate_dir / f"submission_best_basecorr_posoffset_{CANDIDATE}_plus_pixel5_patch_test.csv"
    previous_path = previous_dir / f"submission_best_basecorr_posoffset_{PREVIOUS}_plus_pixel5_patch_20260501.csv"
    build_summary_path = output_dir / "build_summary.json"
    base.to_csv(base_path, index=False)
    candidate_dir.mkdir(parents=True)
    candidate.to_csv(candidate_path, index=False)
    previous_dir.mkdir(parents=True)
    previous.to_csv(previous_path, index=False)
    build_summary_path.write_text(
        json.dumps(
            {
                "input": str(base_path),
                "pr_proxy_risk_report": {
                    "enabled": True,
                    "risky_chunks": 5,
                    "risky_rows": 15,
                    "vd_guard_rows": 6,
                    "candidate_actionable_risky_chunks": 0,
                    "candidate_actionable_risky_rows": 0,
                    "candidate_actionable_by_candidate": {CANDIDATE: 0},
                },
                "candidates": [
                    {
                        "candidate": CANDIDATE,
                        "output": str(candidate_path),
                        "output_sha256": "summary-sha",
                        "effective_phone_scales": {"pixel5": 3.375, "pixel6pro": 0.0},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    manifest = build_pre_submit_manifest(
        build_summary_path,
        previous_output_dir=tmp_path / "previous",
        risky_trips=(RISKY_TRIP, OTHER_RISKY_TRIP),
    )

    assert manifest["candidate_count"] == 1
    assert manifest["risk_report"]["risky_chunks"] == 5
    assert manifest["risk_report"]["candidate_actionable_risky_chunks"] == 0
    assert manifest["candidates"][0]["pixel6pro_scale"] == 0.0
    assert manifest["candidates"][0]["risk_candidate_actionable_chunks"] == 0
    assert (output_dir / "pre_submit_manifest.json").is_file()
    assert (output_dir / "pre_submit_candidate_manifest.csv").is_file()
    assert (output_dir / "pre_submit_trip_delta_checks.csv").is_file()

    trip_rows = pd.read_csv(output_dir / "pre_submit_trip_delta_checks.csv")
    assert trip_rows["input_changed_rows"].tolist() == [0, 0]
    assert (trip_rows["input_max_m"] < DELTA_CHANGED_THRESHOLD_M).all()
    assert trip_rows["previous_changed_rows"].tolist() == [2, 2]
    assert (trip_rows["previous_max_m"] > 1.0).all()


def test_main_writes_manifest(tmp_path) -> None:
    base = _base_submission()
    candidate_dir = tmp_path / "out" / CANDIDATE
    candidate_path = candidate_dir / f"submission_best_basecorr_posoffset_{CANDIDATE}_plus_pixel5_patch_test.csv"
    base_path = tmp_path / "base.csv"
    build_summary_path = tmp_path / "out" / "build_summary.json"
    base.to_csv(base_path, index=False)
    candidate_dir.mkdir(parents=True)
    base.to_csv(candidate_path, index=False)
    build_summary_path.write_text(
        json.dumps(
            {
                "input": str(base_path),
                "pr_proxy_risk_report": {"enabled": True, "candidate_actionable_risky_chunks": 0},
                "candidates": [
                    {
                        "candidate": CANDIDATE,
                        "output": str(candidate_path),
                        "effective_phone_scales": {"pixel6pro": 0.0},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    assert main(["--build-summary", str(build_summary_path), "--risky-trip", RISKY_TRIP]) == 0
    payload = json.loads((tmp_path / "out" / "pre_submit_manifest.json").read_text(encoding="utf-8"))
    assert payload["risky_trips"] == [RISKY_TRIP]


def test_build_pre_submit_manifest_records_matlab_equivalence_gate(tmp_path) -> None:
    base = _base_submission()
    candidate_dir = tmp_path / "out" / CANDIDATE
    candidate_path = candidate_dir / f"submission_best_basecorr_posoffset_{CANDIDATE}_plus_pixel5_patch_test.csv"
    base_path = tmp_path / "base.csv"
    build_summary_path = tmp_path / "out" / "build_summary.json"
    summary_path = tmp_path / "matlab_equivalence_summary.json"
    base.to_csv(base_path, index=False)
    candidate_dir.mkdir(parents=True)
    base.to_csv(candidate_path, index=False)
    build_summary_path.write_text(
        json.dumps(
            {
                "input": str(base_path),
                "pr_proxy_risk_report": {"enabled": True, "candidate_actionable_risky_chunks": 0},
                "candidates": [
                    {
                        "candidate": CANDIDATE,
                        "output": str(candidate_path),
                        "effective_phone_scales": {"pixel6pro": 0.0},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "passed": True,
                "equivalence_claim": "matlab_equivalent",
                "data_root": str(DEFAULT_GSDC2023_DATA_ROOT),
                "trips": list(DEFAULT_EQUIVALENCE_TRIPS),
                "trip_count": len(DEFAULT_EQUIVALENCE_TRIPS),
                "max_epochs": 200,
                "count_max_epochs": 0,
                "factor_multi_gnss": False,
                "residual_multi_gnss": False,
                "residual_observation_mask": True,
                "residual_include_inactive_observations": True,
                "count_multi_gnss": False,
                "asset_datasets": ["train"],
                "quick_assets": True,
                "strict_ref_height": False,
                "gates": {
                    "assets": {
                        "passed": True,
                    },
                    "factor_mask": {
                        "passed": True,
                        "total_matlab_only": 0,
                        "total_bridge_only": 0,
                        "side_only_failure_count": 0,
                        "side_only_by_field_freq": {"P": {"L1": {"matlab_only": 0, "bridge_only": 0}}},
                        "top_matlab_only": [],
                        "top_bridge_only": [],
                    },
                    "raw_bridge_counts": {
                        "passed": True,
                        "matched_abs_delta_total": 0,
                        "count_delta_failure_count": 0,
                        "missing_phone_count_rows": 6,
                        "missing_bridge_count_rows": 0,
                        "abs_delta_sums": {"P": {"L1": 0, "L5": 0}},
                        "top_count_delta_failures": [],
                    },
                    "residual_values": {
                        "passed": True,
                        "total_matlab_only": 0,
                        "total_bridge_only": 0,
                        "overall_max_abs_delta": 5.9e-5,
                        "max_abs_delta_threshold_m": 1.0e-4,
                        "internal_delta_failure_count": 0,
                        "internal_delta_failures": [],
                        "internal_delta_thresholds": {"model_delta": 1.0e-4},
                    },
                    "residual_diagnostics_writer": {
                        "passed": True,
                        "pd_value_passed": True,
                        "wide_passed": True,
                        "total_matlab_only": 0,
                        "total_bridge_only": 0,
                        "wide_total_matlab_only": 0,
                        "wide_total_bridge_only": 0,
                        "wide_sat_col_mismatch_count": 0,
                        "bridge_residual_diagnostics_export_enabled": True,
                        "bridge_residual_diagnostics_export_count": 12,
                        "bridge_residual_diagnostics_export_total_rows": 258537,
                        "bridge_residual_diagnostics_export_expected_columns": 44,
                        "bridge_residual_diagnostics_export_column_count_min": 44,
                        "bridge_residual_diagnostics_export_column_count_max": 44,
                        "bridge_residual_diagnostics_export_column_mismatch_count": 0,
                        "bridge_residual_diagnostics_export_byte_equivalent_count": 0,
                        "bridge_residual_diagnostics_export_byte_difference_count": 12,
                        "writer_regression_manifest": "data/writer_manifest.json",
                        "writer_regression_checked": True,
                        "writer_regression_passed": True,
                        "writer_regression_mismatch_count": 0,
                        "inactive_key_source": "gnss_log_signal_mask",
                    },
                },
            },
        ),
        encoding="utf-8",
    )

    manifest = build_pre_submit_manifest(
        build_summary_path,
        risky_trips=(RISKY_TRIP,),
        matlab_equivalence_summary=summary_path,
    )

    gate = manifest["matlab_equivalence_gate"]
    assert gate["passed"] is True
    assert gate["equivalence_claim"] == "matlab_equivalent"
    assert gate["trip_count"] == len(DEFAULT_EQUIVALENCE_TRIPS)
    assert gate["factor_side_only_failure_count"] == 0
    assert gate["factor_total_matlab_only"] == 0
    assert gate["factor_total_bridge_only"] == 0
    assert gate["factor_side_only_by_field_freq"]["P"]["L1"]["matlab_only"] == 0
    assert gate["raw_bridge_count_delta_failure_count"] == 0
    assert gate["raw_bridge_matched_abs_delta_total"] == 0
    assert gate["raw_bridge_missing_phone_count_rows"] == 6
    assert gate["raw_bridge_abs_delta_sums"]["P"]["L5"] == 0
    assert gate["residual_total_matlab_only"] == 0
    assert gate["residual_internal_delta_failure_count"] == 0
    assert gate["residual_internal_delta_thresholds"]["model_delta"] == 1.0e-4
    assert gate["residual_diagnostics_writer_passed"] is True
    assert gate["residual_diagnostics_writer_total_bridge_only"] == 0
    assert gate["residual_diagnostics_writer_wide_total_matlab_only"] == 0
    assert gate["residual_diagnostics_writer_export_count"] == 12
    assert gate["residual_diagnostics_writer_export_total_rows"] == 258537
    assert gate["residual_diagnostics_writer_export_column_count_min"] == 44
    assert gate["residual_diagnostics_writer_export_column_count_max"] == 44
    assert gate["residual_diagnostics_writer_export_column_mismatch_count"] == 0
    assert gate["residual_diagnostics_writer_export_byte_difference_count"] == 12
    assert gate["residual_diagnostics_writer_regression_manifest"] == "data/writer_manifest.json"
    assert gate["residual_diagnostics_writer_regression_checked"] is True
    assert gate["residual_diagnostics_writer_regression_passed"] is True
    assert gate["residual_diagnostics_writer_regression_mismatch_count"] == 0
    assert gate["residual_diagnostics_writer_inactive_key_source"] == "gnss_log_signal_mask"
    assert gate["cached_summary_validation_checked"] is True
    assert gate["cached_summary_validation_passed"] is True
    assert gate["cached_summary_validation_mismatch_count"] == 0
    assert gate["cached_summary_validation_mismatches"] == []
    assert gate["summary_sha256"]


def test_build_pre_submit_manifest_records_matlab_final_reproduction_gate(tmp_path) -> None:
    base = _base_submission()
    candidate_dir = tmp_path / "out" / CANDIDATE
    candidate_path = candidate_dir / f"submission_best_basecorr_posoffset_{CANDIDATE}_plus_pixel5_patch_test.csv"
    base_path = tmp_path / "base.csv"
    build_summary_path = tmp_path / "out" / "build_summary.json"
    summary_path = tmp_path / "matlab_final_reproduction_summary.json"
    base.to_csv(base_path, index=False)
    candidate_dir.mkdir(parents=True)
    base.to_csv(candidate_path, index=False)
    build_summary_path.write_text(
        json.dumps(
            {
                "input": str(base_path),
                "pr_proxy_risk_report": {"enabled": True, "candidate_actionable_risky_chunks": 0},
                "candidates": [
                    {
                        "candidate": CANDIDATE,
                        "output": str(candidate_path),
                        "effective_phone_scales": {"pixel6pro": 0.0},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "reference_submission": "ref.csv",
                "candidate_submission": "candidate.csv",
                "bridge_root": "bridge",
                "missing_bridge_timestamp_summary": {
                    "rows": 24,
                    "trips": 12,
                    "materialized_source_counts": {"bridge": 24},
                },
                "reconstruction_summary": {
                    "delta_vs_reference": {
                        "rows": 71936,
                        "changed_rows_gt_1e_9m": 0,
                        "changed_rows_gt_0p01m": 0,
                        "mean_delta_m": 0.0,
                        "p50_delta_m": 0.0,
                        "p95_delta_m": 0.0,
                        "max_delta_m": 0.0,
                    },
                },
                "missing_bridge_timestamp_rows_csv": "missing.csv",
                "reconstructed_submission_csv": "reconstructed.csv",
                "reconstruction_summary_json": "reconstruction_summary.json",
            },
        ),
        encoding="utf-8",
    )

    manifest = build_pre_submit_manifest(
        build_summary_path,
        risky_trips=(RISKY_TRIP,),
        matlab_final_reproduction_summary=summary_path,
    )

    gate = manifest["matlab_final_reproduction_gate"]
    assert gate["passed"] is True
    assert gate["max_delta_threshold_m"] == 1e-6
    assert gate["rows"] == 71936
    assert gate["changed_rows_gt_1e_9m"] == 0
    assert gate["changed_rows_gt_0p01m"] == 0
    assert gate["max_delta_m"] == 0.0
    assert gate["missing_bridge_timestamp_rows"] == 24
    assert gate["missing_bridge_timestamp_trips"] == 12
    assert gate["missing_bridge_timestamp_materialized_source_counts"] == {"bridge": 24}
    assert gate["summary_sha256"]
