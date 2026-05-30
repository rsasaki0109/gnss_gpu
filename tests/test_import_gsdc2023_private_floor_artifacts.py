from __future__ import annotations

import csv

from experiments.import_gsdc2023_private_floor_artifacts import import_private_floor_artifacts


def _csv(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n"
        "trip/phone,1000,37.0,-122.0\n",
        encoding="utf-8",
    )


def _rows(path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_import_plan_finds_unambiguous_builder_inputs(tmp_path) -> None:
    root = tmp_path / "backup"
    input_backup = root / "nested" / "basecorr_pixel4xl_base_on_submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv"
    patch_backup = root / "nested" / "pixel5_fgo_early_raw_late_final_trip_rows.csv"
    _csv(input_backup)
    _csv(patch_backup)
    targets = {
        "basecorr_builder_input": tmp_path / "restore" / input_backup.name,
        "pixel5_patch": tmp_path / "restore" / patch_backup.name,
        "base_0555_submission": tmp_path / "restore" / "submission_20260421_0555.csv",
        "current_1450_submission": tmp_path / "restore" / "submission_20260423_1450.csv",
    }

    summary = import_private_floor_artifacts(
        roots=(root,),
        output_dir=tmp_path / "out",
        artifact_overrides=targets,
    )

    assert summary["all_core_artifacts_ready"] is False
    assert summary["ready_artifact_count"] == 2
    assert summary["missing_artifacts"] == ["base_0555_submission", "current_1450_submission"]
    rows = _rows(tmp_path / "out" / "artifact_import_plan.csv")
    input_row = next(row for row in rows if row["artifact"] == "basecorr_builder_input")
    assert input_row["status"] == "ready"
    assert input_row["selected_candidate"] == str(input_backup)


def test_import_apply_copies_ready_builder_inputs(tmp_path) -> None:
    root = tmp_path / "backup"
    input_backup = root / "basecorr_pixel4xl_base_on_submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv"
    patch_backup = root / "pixel5_fgo_early_raw_late_final_trip_rows.csv"
    _csv(input_backup)
    _csv(patch_backup)
    targets = {
        "basecorr_builder_input": tmp_path / "restore" / input_backup.name,
        "pixel5_patch": tmp_path / "restore" / patch_backup.name,
        "base_0555_submission": tmp_path / "restore" / "submission_20260421_0555.csv",
        "current_1450_submission": tmp_path / "restore" / "submission_20260423_1450.csv",
    }

    summary = import_private_floor_artifacts(
        roots=(root,),
        output_dir=tmp_path / "out",
        apply=True,
        artifact_overrides=targets,
    )

    assert targets["basecorr_builder_input"].is_file()
    assert targets["pixel5_patch"].is_file()
    assert summary["copied_artifact_count"] == 2
    assert set(summary["copied_artifacts"]) == {"basecorr_builder_input", "pixel5_patch"}


def test_import_plan_rejects_reference_submission_sha_mismatch(tmp_path) -> None:
    root = tmp_path / "backup"
    wrong_reference = root / "submission_20260421_0555.csv"
    _csv(wrong_reference)
    targets = {
        "basecorr_builder_input": tmp_path / "restore" / "basecorr_pixel4xl_base_on_submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv",
        "pixel5_patch": tmp_path / "restore" / "pixel5_fgo_early_raw_late_final_trip_rows.csv",
        "base_0555_submission": tmp_path / "restore" / wrong_reference.name,
        "current_1450_submission": tmp_path / "restore" / "submission_20260423_1450.csv",
    }

    summary = import_private_floor_artifacts(
        roots=(root,),
        output_dir=tmp_path / "out",
        artifact_overrides=targets,
    )

    assert "base_0555_submission" in summary["blocked_artifacts"]
    rows = _rows(tmp_path / "out" / "artifact_import_plan.csv")
    base_row = next(row for row in rows if row["artifact"] == "base_0555_submission")
    assert base_row["status"] == "sha_mismatch"
    candidates = _rows(tmp_path / "out" / "artifact_candidates.csv")
    assert candidates[0]["sha256_matches"] == "False"
