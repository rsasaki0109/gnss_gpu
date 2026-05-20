#!/usr/bin/env python3
"""Audit the artifact dependency chain for GSDC2023 private-floor recovery."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.build_gsdc2023_basecorr_posoffset_pixel5_patch_candidates import (
    DEFAULT_INPUT as BASECORR_BUILDER_INPUT,
    DEFAULT_OUTPUT_DIR as BASECORR_OUTPUT_DIR,
    DEFAULT_PIXEL5_PATCH,
)
from experiments.reproduce_gsdc2023_best_submission import (
    BASE_0555_SUBMISSION,
    CURRENT_1450_SUBMISSION,
    EXPECTED_BASE_0555_SHA256,
    EXPECTED_CURRENT_1450_SHA256,
    EXPECTED_FINAL,
    EXPECTED_PIXEL4XL_BRIDGE_SHA256,
    EXPECTED_SM_A505U_BRIDGE_SHA256,
    EXPECTED_SM_A505U_EXCEPTION_SHA256,
    EXPECTED_SOURCE_CANDIDATE,
    PIXEL4XL_BRIDGE_POSITIONS,
    SM_A505U_BRIDGE_POSITIONS,
    SM_A505U_EXCEPTION_ROWS,
)


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    path: Path
    kind: str = "file"
    expected_sha256: str | None = None
    required_for: str = ""


@dataclass(frozen=True)
class RecoveryRoute:
    name: str
    artifact_names: tuple[str, ...]
    command_hint: str
    note: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_artifacts() -> list[ArtifactSpec]:
    return [
        ArtifactSpec(
            "base_0555_submission",
            BASE_0555_SUBMISSION,
            expected_sha256=EXPECTED_BASE_0555_SHA256,
            required_for="20260424 historical rebuild",
        ),
        ArtifactSpec(
            "current_1450_submission",
            CURRENT_1450_SUBMISSION,
            expected_sha256=EXPECTED_CURRENT_1450_SHA256,
            required_for="20260424 historical rebuild",
        ),
        ArtifactSpec(
            "best_20260424_source_candidate",
            EXPECTED_SOURCE_CANDIDATE,
            required_for="optional direct 20260424 source verification",
        ),
        ArtifactSpec(
            "best_20260424_final_submission",
            EXPECTED_FINAL,
            required_for="optional direct 20260424 final body",
        ),
        ArtifactSpec(
            "pixel4xl_bridge_positions",
            PIXEL4XL_BRIDGE_POSITIONS,
            expected_sha256=EXPECTED_PIXEL4XL_BRIDGE_SHA256,
            required_for="20260424 bridge-exception rebuild",
        ),
        ArtifactSpec(
            "sm_a505u_bridge_positions",
            SM_A505U_BRIDGE_POSITIONS,
            expected_sha256=EXPECTED_SM_A505U_BRIDGE_SHA256,
            required_for="20260424 bridge-exception rebuild",
        ),
        ArtifactSpec(
            "sm_a505u_exception_rows",
            SM_A505U_EXCEPTION_ROWS,
            expected_sha256=EXPECTED_SM_A505U_EXCEPTION_SHA256,
            required_for="20260424 bridge-exception rebuild",
        ),
        ArtifactSpec(
            "basecorr_builder_input",
            BASECORR_BUILDER_INPUT,
            required_for="basecorr private-floor candidate builder",
        ),
        ArtifactSpec(
            "pixel5_patch",
            DEFAULT_PIXEL5_PATCH,
            required_for="basecorr private-floor candidate builder",
        ),
        ArtifactSpec(
            "basecorr_output_dir",
            BASECORR_OUTPUT_DIR,
            kind="dir",
            required_for="historical local duplicate/provenance checks",
        ),
    ]


def default_routes() -> list[RecoveryRoute]:
    return [
        RecoveryRoute(
            "rebuild_20260424_best_historical",
            ("base_0555_submission", "current_1450_submission"),
            "experiments/reproduce_gsdc2023_best_submission.py --patch-source historical-submission",
            "Known rebuild path for the 20260424 best artifact when both reference submissions exist.",
        ),
        RecoveryRoute(
            "rebuild_20260424_best_bridge_exception",
            (
                "base_0555_submission",
                "pixel4xl_bridge_positions",
                "sm_a505u_bridge_positions",
                "sm_a505u_exception_rows",
            ),
            "experiments/reproduce_gsdc2023_best_submission.py --patch-source bridge-exception",
            "Known alternate rebuild path using regenerated bridge exception artifacts.",
        ),
        RecoveryRoute(
            "direct_basecorr_private_floor_builder",
            ("basecorr_builder_input", "pixel5_patch"),
            "experiments/build_gsdc2023_basecorr_posoffset_pixel5_patch_candidates.py",
            "Only submit-relevant basecorr private-floor rebuild path currently encoded in this workspace.",
        ),
        RecoveryRoute(
            "historical_basecorr_output_reuse",
            ("basecorr_output_dir",),
            "experiments/submit_gsdc2023_pixel5_candidate_queue.py --previous-output-dir ...",
            "Useful for provenance/duplicate checks only; not enough without exact scored CSV bodies.",
        ),
    ]


def _artifact_available(row: dict[str, Any]) -> bool:
    if row["kind"] == "dir":
        return bool(row["exists"] and row["is_dir"])
    return bool(row["exists"] and row["is_file"] and row["sha256_matches"] is not False)


def audit_recovery_dependencies(
    *,
    output_dir: Path,
    artifact_overrides: dict[str, Path] | None = None,
) -> dict[str, Any]:
    artifact_overrides = artifact_overrides or {}
    rows: list[dict[str, Any]] = []
    for spec in default_artifacts():
        path = artifact_overrides.get(spec.name, spec.path)
        exists = path.exists()
        is_file = path.is_file()
        is_dir = path.is_dir()
        sha256 = _sha256_file(path) if is_file else None
        sha_matches = None if spec.expected_sha256 is None or sha256 is None else sha256 == spec.expected_sha256
        rows.append(
            {
                "name": spec.name,
                "path": str(path),
                "kind": spec.kind,
                "exists": exists,
                "is_file": is_file,
                "is_dir": is_dir,
                "sha256": sha256,
                "expected_sha256": spec.expected_sha256,
                "sha256_matches": sha_matches,
                "required_for": spec.required_for,
            },
        )
    by_name = {row["name"]: row for row in rows}
    route_rows: list[dict[str, Any]] = []
    for route in default_routes():
        blockers = [
            name
            for name in route.artifact_names
            if name not in by_name or not _artifact_available(by_name[name])
        ]
        route_rows.append(
            {
                "route": route.name,
                "available": not blockers,
                "blocker_count": len(blockers),
                "blockers": "|".join(blockers),
                "artifacts": "|".join(route.artifact_names),
                "command_hint": route.command_hint,
                "note": route.note,
            },
        )
    direct_private_floor_ready = not [
        name
        for name in ("basecorr_builder_input", "pixel5_patch")
        if not _artifact_available(by_name[name])
    ]
    available_routes = [row for row in route_rows if row["available"]]
    missing_required = [
        row
        for row in rows
        if row["name"]
        in {
            "base_0555_submission",
            "current_1450_submission",
            "basecorr_builder_input",
            "pixel5_patch",
        }
        and not _artifact_available(row)
    ]
    summary = {
        "artifact_count": len(rows),
        "available_artifact_count": int(sum(1 for row in rows if _artifact_available(row))),
        "route_count": len(route_rows),
        "available_route_count": len(available_routes),
        "direct_private_floor_builder_ready": direct_private_floor_ready,
        "recoverable_from_current_workspace": direct_private_floor_ready,
        "missing_core_artifacts": [row["name"] for row in missing_required],
        "available_routes": [row["route"] for row in available_routes],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "recovery_artifacts.csv", rows)
    _write_csv(output_dir / "recovery_routes.csv", route_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "private_floor_recovery_dependencies.md").write_text(
        "\n".join(
            [
                "# GSDC2023 Private-Floor Recovery Dependencies",
                "",
                f"- Direct private-floor builder ready: `{direct_private_floor_ready}`",
                f"- Recoverable from current workspace: `{summary['recoverable_from_current_workspace']}`",
                f"- Available routes: `{summary['available_route_count']}` / `{summary['route_count']}`",
                f"- Missing core artifacts: `{', '.join(summary['missing_core_artifacts']) or 'none'}`",
                "",
                "## Blocked Routes",
                "",
                "| route | blockers |",
                "|---|---|",
                *[
                    f"| `{row['route']}` | `{row['blockers']}` |"
                    for row in route_rows
                    if not row["available"]
                ],
                "",
                "## Read",
                "",
                (
                    "The current workspace can rebuild basecorr private-floor candidates."
                    if direct_private_floor_ready
                    else "The current workspace cannot rebuild or safely extend the basecorr private-floor line until the missing core artifacts are restored."
                ),
                "",
            ],
        ),
        encoding="utf-8",
    )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_override(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ARTIFACT=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("artifact name must not be empty")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        type=_parse_override,
        help="override an artifact path as ARTIFACT=PATH; repeatable",
    )
    args = parser.parse_args(argv)
    summary = audit_recovery_dependencies(
        output_dir=args.output_dir,
        artifact_overrides=dict(args.artifact),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
