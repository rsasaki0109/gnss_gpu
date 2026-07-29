from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tools.build_release_bundle import (
    VERSION,
    build_bundle,
    verify_bundle,
    write_deterministic_zip,
)


REPO_ROOT = Path(__file__).parents[1]


def test_release_bundle_builds_and_verifies(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    manifest = build_bundle(REPO_ROOT, output)
    result = verify_bundle(output)
    assert manifest["version"] == VERSION == "0.3.0"
    assert result["passed"] is True
    assert result["file_count"] >= 38
    benchmark = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["runtime"]["normal_latency_max_ms"] <= 100.0
    assert benchmark["runtime"]["search_latency_max_ms"] <= 1_000.0
    assert (
        benchmark["runtime"]["wp172_candidate_supply"][
            "conservative_sequential_average_ms_per_epoch"
        ]
        <= 100.0
    )
    assert benchmark["lambda_fix"]["fix_percent"] > 10.0
    assert benchmark["lambda_fix"]["false_fix_epochs"] == 0
    assert benchmark["cross_domain"]["coverage"]["cities"] == [
        "hong-kong",
        "nagoya",
        "tokyo",
    ]


def test_release_archive_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    build_bundle(REPO_ROOT, output)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    write_deterministic_zip(output, first)
    write_deterministic_zip(output, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
    assert names == sorted(names)
    assert f"gnss_gpu-v{VERSION}/MANIFEST.json" in names


def test_bundle_verification_detects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    build_bundle(REPO_ROOT, output)
    (output / "benchmark.json").write_text("{}\n", encoding="utf-8")
    assert verify_bundle(output)["passed"] is False
