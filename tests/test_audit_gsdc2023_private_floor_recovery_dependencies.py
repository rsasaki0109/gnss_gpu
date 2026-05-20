from __future__ import annotations

import csv

from experiments.audit_gsdc2023_private_floor_recovery_dependencies import (
    audit_recovery_dependencies,
)


def _submission(path) -> None:
    path.write_text(
        "tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n"
        "trip/pixel5,1,37.0,-122.0\n",
        encoding="utf-8",
    )


def _routes(path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_recovery_dependency_audit_blocks_when_core_artifacts_missing(tmp_path) -> None:
    summary = audit_recovery_dependencies(output_dir=tmp_path / "audit")

    assert summary["direct_private_floor_builder_ready"] is False
    assert "basecorr_builder_input" in summary["missing_core_artifacts"]
    assert "pixel5_patch" in summary["missing_core_artifacts"]
    routes = _routes(tmp_path / "audit" / "recovery_routes.csv")
    direct = next(row for row in routes if row["route"] == "direct_basecorr_private_floor_builder")
    assert direct["available"] == "False"
    assert direct["blockers"] == "basecorr_builder_input|pixel5_patch"


def test_recovery_dependency_audit_opens_direct_builder_with_overrides(tmp_path) -> None:
    input_path = tmp_path / "basecorr_input.csv"
    patch_path = tmp_path / "pixel5_patch.csv"
    _submission(input_path)
    _submission(patch_path)

    summary = audit_recovery_dependencies(
        output_dir=tmp_path / "audit",
        artifact_overrides={
            "basecorr_builder_input": input_path,
            "pixel5_patch": patch_path,
        },
    )

    assert summary["direct_private_floor_builder_ready"] is True
    assert summary["recoverable_from_current_workspace"] is True
    assert "direct_basecorr_private_floor_builder" in summary["available_routes"]


def test_recovery_dependency_audit_detects_reference_sha_mismatch(tmp_path) -> None:
    wrong_base = tmp_path / "submission_20260421_0555.csv"
    _submission(wrong_base)

    summary = audit_recovery_dependencies(
        output_dir=tmp_path / "audit",
        artifact_overrides={"base_0555_submission": wrong_base},
    )

    assert summary["direct_private_floor_builder_ready"] is False
    artifacts = list(csv.DictReader((tmp_path / "audit" / "recovery_artifacts.csv").open()))
    base = next(row for row in artifacts if row["name"] == "base_0555_submission")
    assert base["exists"] == "True"
    assert base["sha256_matches"] == "False"
    routes = _routes(tmp_path / "audit" / "recovery_routes.csv")
    historical = next(row for row in routes if row["route"] == "rebuild_20260424_best_historical")
    assert "base_0555_submission" in historical["blockers"]
