from pathlib import Path

from experiments.check_gsdc2023_pivot_readiness import (
    ArtifactSpec,
    check_artifacts,
    summarize_readiness,
)
from experiments.reproduce_gsdc2023_best_submission import sha256_file


def test_check_artifacts_reports_missing_required(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    rows = check_artifacts(
        (
            ArtifactSpec("required", missing, required_for_reproduction=True),
        ),
    )

    assert rows[0]["name"] == "required"
    assert rows[0]["exists"] is False
    assert rows[0]["is_file"] is False


def test_summarize_readiness_passes_with_matching_required_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "submission.csv"
    artifact.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    expected_sha = sha256_file(artifact)

    payload = summarize_readiness(
        data_roots=(tmp_path,),
        artifacts=(
            ArtifactSpec(
                "required",
                artifact,
                required_for_reproduction=True,
                expected_sha256=expected_sha,
            ),
        ),
    )

    assert payload["ready_for_exact_reproduction"] is True
    assert payload["missing_required_artifacts"] == []
    assert payload["required_sha256_mismatches"] == []
    assert payload["artifacts"][0]["sha256_matches"] is True


def test_summarize_readiness_fails_on_required_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "submission.csv"
    artifact.write_text("changed\n", encoding="utf-8")

    payload = summarize_readiness(
        data_roots=(tmp_path,),
        artifacts=(
            ArtifactSpec(
                "required",
                artifact,
                required_for_reproduction=True,
                expected_sha256="0" * 64,
            ),
        ),
    )

    assert payload["ready_for_exact_reproduction"] is False
    assert payload["missing_required_artifacts"] == []
    assert payload["required_sha256_mismatches"] == ["required"]
