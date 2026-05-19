#!/usr/bin/env python3
"""Run the GSDC2023 private-floor recovery pipeline after artifacts are restored."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.audit_gsdc2023_basecorr_private_floor_lineage import (
    audit_basecorr_private_floor_lineage,
)
from experiments.audit_gsdc2023_private_floor_recovery_dependencies import (
    audit_recovery_dependencies,
)
from experiments.build_gsdc2023_basecorr_posoffset_pixel5_patch_candidates import (
    CANDIDATES,
    DEFAULT_INPUT,
    DEFAULT_PIXEL5_PATCH,
    CandidateConfig,
    build_candidates,
)
from experiments.guard_gsdc2023_private_floor_submit import guard_private_floor_submit
from experiments.screen_gsdc2023_local_submissions import screen_local_submissions


DEFAULT_SCORE_HISTORY = Path("experiments/results/gsdc2023_kaggle_submissions_20260520_bridge.csv")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _score_backed_candidate_names(lineage_csv: Path) -> list[str]:
    rows = _read_csv_rows(lineage_csv)
    names = [
        row["candidate"]
        for row in rows
        if _as_bool(row.get("score_backed_private_floor"))
    ]
    return sorted(name for name in names if name in CANDIDATES)


def _selected_configs(candidate_names: list[str] | None, lineage_csv: Path) -> list[CandidateConfig]:
    names = candidate_names if candidate_names else _score_backed_candidate_names(lineage_csv)
    unknown = sorted(set(names).difference(CANDIDATES))
    if unknown:
        raise SystemExit(f"unknown candidate(s): {', '.join(unknown)}")
    if not names:
        raise SystemExit("no score-backed private-floor candidates selected")
    return [CANDIDATES[name] for name in names]


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_private_floor_recovery(
    *,
    output_dir: Path,
    score_history_csv: Path,
    input_path: Path = DEFAULT_INPUT,
    pixel5_patch_path: Path = DEFAULT_PIXEL5_PATCH,
    candidate_names: list[str] | None = None,
    tag: str = "recovered",
    submitted_csv: Path | None = None,
    reference_best: Path | None = None,
    previous_safe: Path | None = None,
    private_floor_audit_summary: Path | None = None,
    source_family_summaries: tuple[Path, ...] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dependency_dir = output_dir / "dependency_audit"
    dependency_summary = audit_recovery_dependencies(
        output_dir=dependency_dir,
        artifact_overrides={
            "basecorr_builder_input": input_path,
            "pixel5_patch": pixel5_patch_path,
        },
    )
    dependency_summary_path = dependency_dir / "summary.json"
    direct_ready = bool(dependency_summary.get("direct_private_floor_builder_ready", False))

    payload: dict[str, Any] = {
        "output_dir": str(output_dir),
        "score_history_csv": str(score_history_csv),
        "input_path": str(input_path),
        "pixel5_patch_path": str(pixel5_patch_path),
        "tag": tag,
        "dependency_summary_json": str(dependency_summary_path),
        "direct_private_floor_builder_ready": direct_ready,
        "dry_run": dry_run,
    }
    if not direct_ready:
        payload.update(
            {
                "status": "blocked",
                "build_skipped_reason": "direct_private_floor_builder_not_ready",
                "missing_core_artifacts": dependency_summary.get("missing_core_artifacts", []),
                "selected_candidates": [],
            },
        )
        _write_summary(output_dir / "summary.json", payload)
        return payload

    lineage_dir = output_dir / "lineage_audit"
    lineage_summary = audit_basecorr_private_floor_lineage(
        score_history_csv=score_history_csv,
        output_dir=lineage_dir,
        input_path=input_path,
        pixel5_patch_path=pixel5_patch_path,
        search_roots=(output_dir, input_path.parent, pixel5_patch_path.parent),
    )
    lineage_csv = lineage_dir / "basecorr_candidate_lineage.csv"
    configs = _selected_configs(candidate_names, lineage_csv)
    selected_names = [config.name for config in configs]
    payload.update(
        {
            "status": "ready",
            "lineage_summary_json": str(lineage_dir / "summary.json"),
            "lineage_score_backed_private_floor_candidate_count": lineage_summary.get(
                "score_backed_private_floor_candidate_count",
            ),
            "selected_candidates": selected_names,
            "selected_candidate_count": len(selected_names),
        },
    )
    if dry_run:
        payload["status"] = "ready_dry_run"
        _write_summary(output_dir / "summary.json", payload)
        return payload

    build_dir = output_dir / "basecorr_candidates"
    build_summary = build_candidates(
        input_path,
        pixel5_patch_path,
        build_dir,
        configs,
        tag=tag,
    )
    screen_csv = output_dir / "local_submission_screen.csv"
    screen_rows = screen_local_submissions(
        root=build_dir,
        output_csv=screen_csv,
        submitted_csv=submitted_csv,
        reference_best=reference_best,
        previous_safe=previous_safe,
    )
    payload.update(
        {
            "build_summary_json": str(build_dir / "build_summary.json"),
            "built_candidate_count": int(build_summary["candidate_count"]),
            "screen_csv": str(screen_csv),
            "screen_candidate_count": len(screen_rows),
        },
    )
    if private_floor_audit_summary is not None:
        guard_dir = output_dir / "submit_guard"
        guard_summary = guard_private_floor_submit(
            private_floor_audit_summary=private_floor_audit_summary,
            recovery_dependencies_summary=dependency_summary_path,
            screen_csvs=(screen_csv,),
            source_family_summaries=source_family_summaries,
            output_dir=guard_dir,
        )
        payload.update(
            {
                "guard_summary_json": str(guard_dir / "summary.json"),
                "guard_submit_allowed": bool(guard_summary["submit_allowed"]),
                "guard_blocked_count": int(guard_summary["blocked_count"]),
            },
        )
    _write_summary(output_dir / "summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-history-csv", type=Path, default=DEFAULT_SCORE_HISTORY)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pixel5-patch", type=Path, default=DEFAULT_PIXEL5_PATCH)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--tag", default="recovered")
    parser.add_argument("--submitted-csv", type=Path)
    parser.add_argument("--reference-best", type=Path)
    parser.add_argument("--previous-safe", type=Path)
    parser.add_argument("--private-floor-audit-summary", type=Path)
    parser.add_argument("--source-family-summary", action="append", type=Path, default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unready", action="store_true")
    args = parser.parse_args(argv)

    payload = run_private_floor_recovery(
        output_dir=args.output_dir,
        score_history_csv=args.score_history_csv,
        input_path=args.input,
        pixel5_patch_path=args.pixel5_patch,
        candidate_names=list(args.candidate) or None,
        tag=args.tag,
        submitted_csv=args.submitted_csv,
        reference_best=args.reference_best,
        previous_safe=args.previous_safe,
        private_floor_audit_summary=args.private_floor_audit_summary,
        source_family_summaries=tuple(args.source_family_summary),
        dry_run=args.dry_run,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("status") == "blocked" and not args.allow_unready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
