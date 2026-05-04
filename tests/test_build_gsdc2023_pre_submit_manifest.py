from __future__ import annotations

import json

import pandas as pd

from experiments.build_gsdc2023_pre_submit_manifest import build_pre_submit_manifest, main


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
    assert trip_rows["input_max_m"].tolist() == [0.0, 0.0]
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
                "trip_count": 12,
                "max_epochs": 200,
                "count_max_epochs": 0,
                "gates": {
                    "factor_mask": {"passed": True},
                    "raw_bridge_counts": {"passed": True},
                    "residual_values": {
                        "passed": True,
                        "total_matlab_only": 0,
                        "total_bridge_only": 0,
                        "overall_max_abs_delta": 5.9e-5,
                        "max_abs_delta_threshold_m": 1.0e-4,
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
    assert gate["trip_count"] == 12
    assert gate["residual_total_matlab_only"] == 0
    assert gate["summary_sha256"]
