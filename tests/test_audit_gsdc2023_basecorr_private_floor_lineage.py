from __future__ import annotations

import csv

from experiments.audit_gsdc2023_basecorr_private_floor_lineage import (
    audit_basecorr_private_floor_lineage,
)


def _write_score_history(path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["fileName", "date", "description", "status", "publicScore", "privateScore"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_basecorr_lineage_matches_score_backed_candidate_and_exact_body(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    patch_path = tmp_path / "pixel5_patch.csv"
    input_path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    patch_path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    scored_name = (
        "submission_best_basecorr_posoffset_pixel5phone_3p375_sjc_r0p84375_"
        "plus_pixel5_patch_20260501.csv"
    )
    (tmp_path / scored_name).write_text("dummy\n", encoding="utf-8")
    score_history = tmp_path / "history.csv"
    _write_score_history(
        score_history,
        [
            {
                "fileName": scored_name,
                "date": "2026-05-01",
                "description": "pixel5 3.375 sjc r scale 0.84375",
                "status": "complete",
                "publicScore": "3.725",
                "privateScore": "4.711",
            },
        ],
    )

    summary = audit_basecorr_private_floor_lineage(
        score_history_csv=score_history,
        output_dir=tmp_path / "audit",
        input_path=input_path,
        pixel5_patch_path=patch_path,
        default_output_dir=tmp_path / "out",
        search_roots=(tmp_path,),
    )

    assert summary["build_possible_locally"] is True
    assert summary["score_backed_private_floor_candidate_count"] >= 1
    assert summary["exact_local_private_floor_candidate_count"] >= 1
    assert summary["top_recovery_targets"][0]["candidate"] == "pixel5phone_3p375_sjc_r0p84375"
    assert (tmp_path / "audit" / "basecorr_candidate_lineage.csv").is_file()


def test_basecorr_lineage_reports_missing_inputs_and_unmatched_private_floor(tmp_path) -> None:
    score_history = tmp_path / "history.csv"
    _write_score_history(
        score_history,
        [
            {
                "fileName": "submission_private_floor_weighted_best_p3p25_a0p0625_20260505.csv",
                "date": "2026-05-05",
                "description": "private floor weighted best",
                "status": "complete",
                "publicScore": "3.687",
                "privateScore": "4.710",
            },
        ],
    )

    summary = audit_basecorr_private_floor_lineage(
        score_history_csv=score_history,
        output_dir=tmp_path / "audit",
        input_path=tmp_path / "missing_input.csv",
        pixel5_patch_path=tmp_path / "missing_patch.csv",
        default_output_dir=tmp_path / "out",
        search_roots=(tmp_path,),
    )

    assert summary["build_possible_locally"] is False
    assert summary["build_prerequisite_missing_count"] == 2
    assert summary["unmatched_private_floor_score_rows"] == 1
    assert summary["score_backed_private_floor_candidate_count"] == 0
    assert (tmp_path / "audit" / "unmatched_private_floor_score_rows.csv").is_file()


def test_basecorr_lineage_does_not_match_longer_prefixed_variant(tmp_path) -> None:
    score_history = tmp_path / "history.csv"
    _write_score_history(
        score_history,
        [
            {
                "fileName": (
                    "submission_best_basecorr_posoffset_"
                    "pixel5phone_3p375_sjcr0_combo_sjcq_ebfxx_ebfzz_laxp_laxi_ebfz_"
                    "plus_pixel5_patch_20260502.csv"
                ),
                "date": "2026-05-02",
                "description": "longer variant",
                "status": "complete",
                "publicScore": "3.725",
                "privateScore": "4.710",
            },
        ],
    )

    summary = audit_basecorr_private_floor_lineage(
        score_history_csv=score_history,
        output_dir=tmp_path / "audit",
        input_path=tmp_path / "missing_input.csv",
        pixel5_patch_path=tmp_path / "missing_patch.csv",
        default_output_dir=tmp_path / "out",
        search_roots=(tmp_path,),
    )

    assert summary["score_backed_private_floor_candidate_count"] == 0
    assert summary["unmatched_private_floor_score_rows"] == 1
