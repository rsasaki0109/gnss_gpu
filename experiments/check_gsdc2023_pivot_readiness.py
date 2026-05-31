#!/usr/bin/env python3
"""Check whether local GSDC2023 artifacts are present for leaderboard work."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.reproduce_gsdc2023_best_submission import (
    BASE_0555_SUBMISSION,
    CURRENT_1450_SUBMISSION,
    EXPECTED_BASE_0555_SHA256,
    EXPECTED_CURRENT_1450_SHA256,
    EXPECTED_FINAL,
    EXPECTED_SOURCE_CANDIDATE,
    PIXEL4XL_BRIDGE_POSITIONS,
    SM_A505U_BRIDGE_POSITIONS,
    SM_A505U_EXCEPTION_ROWS,
    EXPECTED_PIXEL4XL_BRIDGE_SHA256,
    EXPECTED_SM_A505U_BRIDGE_SHA256,
    EXPECTED_SM_A505U_EXCEPTION_SHA256,
    sha256_file,
)


DEFAULT_DATA_ROOTS = (
    Path("/tmp/gsdc_data/gsdc2023"),
    (REPO_ROOT / "../ref/gsdc2023").resolve(),
)


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    path: Path
    required_for_reproduction: bool
    expected_sha256: str | None = None


DEFAULT_ARTIFACTS = (
    ArtifactSpec("base_0555_submission", BASE_0555_SUBMISSION, True, EXPECTED_BASE_0555_SHA256),
    ArtifactSpec("current_1450_submission", CURRENT_1450_SUBMISSION, True, EXPECTED_CURRENT_1450_SHA256),
    ArtifactSpec("expected_source_candidate", EXPECTED_SOURCE_CANDIDATE, False, None),
    ArtifactSpec("expected_final_submission", EXPECTED_FINAL, False, None),
    ArtifactSpec("pixel4xl_bridge_positions", PIXEL4XL_BRIDGE_POSITIONS, False, EXPECTED_PIXEL4XL_BRIDGE_SHA256),
    ArtifactSpec("sm_a505u_bridge_positions", SM_A505U_BRIDGE_POSITIONS, False, EXPECTED_SM_A505U_BRIDGE_SHA256),
    ArtifactSpec("sm_a505u_exception_rows", SM_A505U_EXCEPTION_ROWS, False, EXPECTED_SM_A505U_EXCEPTION_SHA256),
)


def _path_status(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "is_dir": resolved.is_dir(),
        "is_file": resolved.is_file(),
    }


def check_artifacts(artifacts: tuple[ArtifactSpec, ...] = DEFAULT_ARTIFACTS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in artifacts:
        resolved = spec.path.expanduser().resolve()
        row: dict[str, Any] = {
            "name": spec.name,
            "path": str(resolved),
            "required_for_reproduction": spec.required_for_reproduction,
            "exists": resolved.exists(),
            "is_file": resolved.is_file(),
            "expected_sha256": spec.expected_sha256,
            "sha256": None,
            "sha256_matches": None,
        }
        if resolved.is_file():
            row["sha256"] = sha256_file(resolved)
            if spec.expected_sha256 is not None:
                row["sha256_matches"] = row["sha256"] == spec.expected_sha256
        rows.append(row)
    return rows


def summarize_readiness(
    *,
    data_roots: tuple[Path, ...] = DEFAULT_DATA_ROOTS,
    artifacts: tuple[ArtifactSpec, ...] = DEFAULT_ARTIFACTS,
) -> dict[str, Any]:
    data_root_rows = [_path_status(path) for path in data_roots]
    artifact_rows = check_artifacts(artifacts)
    missing_required = [
        row["name"]
        for row in artifact_rows
        if row["required_for_reproduction"] and not row["is_file"]
    ]
    sha_mismatches = [
        row["name"]
        for row in artifact_rows
        if row["required_for_reproduction"] and row["sha256_matches"] is False
    ]
    return {
        "data_roots": data_root_rows,
        "artifacts": artifact_rows,
        "ready_for_exact_reproduction": not missing_required and not sha_mismatches,
        "missing_required_artifacts": missing_required,
        "required_sha256_mismatches": sha_mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit non-zero when required reproduction artifacts are missing or have unexpected hashes",
    )
    args = parser.parse_args(argv)

    payload = summarize_readiness()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
        print(f"saved: {args.output_json}")
    else:
        print(text, end="")
    if args.require_ready and not payload["ready_for_exact_reproduction"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
