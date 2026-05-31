#!/usr/bin/env python3
"""Locate and optionally restore GSDC2023 private-floor recovery artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from experiments.audit_gsdc2023_private_floor_recovery_dependencies import default_artifacts


CORE_ARTIFACT_NAMES = (
    "base_0555_submission",
    "current_1450_submission",
    "basecorr_builder_input",
    "pixel5_patch",
)


@dataclass(frozen=True)
class ImportArtifact:
    name: str
    target: Path
    expected_sha256: str | None = None

    @property
    def filename(self) -> str:
        return self.target.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_artifacts(
    *,
    artifact_overrides: dict[str, Path] | None = None,
) -> list[ImportArtifact]:
    overrides = artifact_overrides or {}
    specs = {spec.name: spec for spec in default_artifacts()}
    return [
        ImportArtifact(
            name=name,
            target=overrides.get(name, specs[name].path),
            expected_sha256=specs[name].expected_sha256,
        )
        for name in CORE_ARTIFACT_NAMES
    ]


def _candidate_paths(root: Path, filename: str) -> list[Path]:
    if root.is_file():
        return [root] if root.name == filename else []
    if not root.is_dir():
        return []
    return [path for path in root.rglob(filename) if path.is_file()]


def _find_candidates(artifact: ImportArtifact, roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for path in _candidate_paths(root, artifact.filename):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            digest = sha256_file(path)
            rows.append(
                {
                    "artifact": artifact.name,
                    "target": str(artifact.target),
                    "expected_filename": artifact.filename,
                    "candidate": str(path),
                    "size": path.stat().st_size,
                    "sha256": digest,
                    "expected_sha256": artifact.expected_sha256 or "",
                    "sha256_matches": (
                        "" if artifact.expected_sha256 is None else str(digest == artifact.expected_sha256)
                    ),
                },
            )
    return rows


def _select_candidate(artifact: ImportArtifact, candidates: list[dict[str, Any]]) -> tuple[str, str]:
    if not candidates:
        return "", "missing"
    if artifact.expected_sha256 is not None:
        matches = [row for row in candidates if row["sha256"] == artifact.expected_sha256]
        if len(matches) == 1:
            return str(matches[0]["candidate"]), "ready"
        if len(matches) > 1:
            return "", "ambiguous_sha_match"
        return "", "sha_mismatch"
    if len(candidates) == 1:
        return str(candidates[0]["candidate"]), "ready"
    return "", "ambiguous"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def import_private_floor_artifacts(
    *,
    roots: tuple[Path, ...],
    output_dir: Path,
    apply: bool = False,
    overwrite: bool = False,
    artifact_overrides: dict[str, Path] | None = None,
) -> dict[str, Any]:
    artifacts = import_artifacts(artifact_overrides=artifact_overrides)
    candidate_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    copied: list[str] = []

    for artifact in artifacts:
        candidates = _find_candidates(artifact, roots)
        candidate_rows.extend(candidates)
        selected, status = _select_candidate(artifact, candidates)
        target_exists = artifact.target.is_file()
        target_sha = sha256_file(artifact.target) if target_exists else ""
        target_sha_matches = (
            "" if artifact.expected_sha256 is None or not target_exists else str(target_sha == artifact.expected_sha256)
        )
        copied_to = ""
        if apply and selected and (overwrite or not target_exists):
            artifact.target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected, artifact.target)
            copied_to = str(artifact.target)
            copied.append(artifact.name)
            target_exists = True
            target_sha = sha256_file(artifact.target)
            target_sha_matches = (
                ""
                if artifact.expected_sha256 is None
                else str(target_sha == artifact.expected_sha256)
            )
        artifact_rows.append(
            {
                "artifact": artifact.name,
                "target": str(artifact.target),
                "expected_filename": artifact.filename,
                "expected_sha256": artifact.expected_sha256 or "",
                "target_exists": str(target_exists),
                "target_sha256": target_sha,
                "target_sha256_matches": target_sha_matches,
                "candidate_count": len(candidates),
                "selected_candidate": selected,
                "status": "already_present" if target_exists and not copied_to else status,
                "copied_to": copied_to,
            },
        )

    ready_count = sum(
        1
        for row in artifact_rows
        if row["status"] in {"ready", "already_present"}
    )
    summary = {
        "roots": [str(root) for root in roots],
        "output_dir": str(output_dir),
        "apply": apply,
        "overwrite": overwrite,
        "artifact_count": len(artifact_rows),
        "ready_artifact_count": ready_count,
        "copied_artifacts": copied,
        "copied_artifact_count": len(copied),
        "missing_artifacts": [row["artifact"] for row in artifact_rows if row["status"] == "missing"],
        "blocked_artifacts": [
            row["artifact"]
            for row in artifact_rows
            if row["status"] not in {"ready", "already_present"}
        ],
        "all_core_artifacts_ready": ready_count == len(artifact_rows),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "artifact_candidates.csv",
        candidate_rows,
        [
            "artifact",
            "target",
            "expected_filename",
            "candidate",
            "size",
            "sha256",
            "expected_sha256",
            "sha256_matches",
        ],
    )
    _write_csv(
        output_dir / "artifact_import_plan.csv",
        artifact_rows,
        [
            "artifact",
            "target",
            "expected_filename",
            "expected_sha256",
            "target_exists",
            "target_sha256",
            "target_sha256_matches",
            "candidate_count",
            "selected_candidate",
            "status",
            "copied_to",
        ],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True, help="search root; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="copy unambiguous matches into expected locations")
    parser.add_argument("--overwrite", action="store_true", help="allow --apply to replace existing target files")
    args = parser.parse_args(argv)

    summary = import_private_floor_artifacts(
        roots=tuple(args.root),
        output_dir=args.output_dir,
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_core_artifacts_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
