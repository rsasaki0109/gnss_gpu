#!/usr/bin/env python3
"""Audit Tokyo candidate supply without permitting truth in production."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _good(row: dict[str, str]) -> bool:
    return float(row["error_m"]) < 0.5


def audit_native_candidates(
    production_path: Path,
    candidate_paths: Iterable[Path],
    *,
    target_percent: float = 45.0,
) -> dict[str, Any]:
    """Measure the post-hoc oracle union of archived PF-native trajectories."""

    production = _read_rows(production_path)
    production_good = [_good(row) for row in production]
    union_good = production_good.copy()
    candidates: list[dict[str, Any]] = []
    for path in sorted(candidate_paths):
        rows = _read_rows(path)
        if len(rows) != len(production) or not rows or "error_m" not in rows[0]:
            continue
        candidate_good = [_good(row) for row in rows]
        gain = sum(not base and other for base, other in zip(production_good, candidate_good))
        loss = sum(base and not other for base, other in zip(production_good, candidate_good))
        union_good = [
            prior or other for prior, other in zip(union_good, candidate_good)
        ]
        candidates.append(
            {
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "oracle_gain_over_production": gain,
                "replacement_loss": loss,
            }
        )
    denominator = len(production)
    production_epochs = sum(production_good)
    union_epochs = sum(union_good)
    target_epochs = int((target_percent * denominator + 99.999999) // 100)
    return {
        "production_path": production_path.as_posix(),
        "production_sha256": _sha256(production_path),
        "denominator_epochs": denominator,
        "production_sub50cm_epochs": production_epochs,
        "production_sub50cm_percent": 100.0 * production_epochs / denominator,
        "candidate_count": len(candidates),
        "oracle_union_sub50cm_epochs": union_epochs,
        "oracle_union_sub50cm_percent": 100.0 * union_epochs / denominator,
        "target_sub50cm_percent": target_percent,
        "target_sub50cm_epochs": target_epochs,
        "novel_epochs_needed_beyond_archive": max(0, target_epochs - union_epochs),
        "candidates": sorted(
            candidates,
            key=lambda item: (-item["oracle_gain_over_production"], item["path"]),
        ),
    }


def audit_external_research_candidate(
    production_path: Path,
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    """Measure diagnostic supply from a prohibited, FGO-derived candidate."""

    production = _read_rows(production_path)
    reference = _read_rows(reference_path)
    candidate = _read_rows(candidate_path)
    reference_by_tow = {
        round(float(row["GPS TOW (s)"]), 3): (index, row)
        for index, row in enumerate(reference)
    }
    gain = loss = good = matched = fix = false_fix = 0
    for row in candidate:
        match = reference_by_tow.get(round(float(row["tow"]), 3))
        if match is None:
            continue
        index, truth = match
        if index >= len(production):
            continue
        error_sq = sum(
            (
                float(row[f"ecef_{axis}"])
                - float(truth[f"ECEF {axis.upper()} (m)"])
            )
            ** 2
            for axis in "xyz"
        )
        candidate_good = error_sq < 0.25
        production_good = _good(production[index])
        matched += 1
        good += candidate_good
        gain += not production_good and candidate_good
        loss += production_good and not candidate_good
        declared_fix = int(row.get("fix", "0")) == 1
        fix += declared_fix
        false_fix += declared_fix and not candidate_good
    return {
        "path": candidate_path.as_posix(),
        "sha256": _sha256(candidate_path),
        "source_class": "research_diagnostic_runtime_fgo",
        "production_eligible": False,
        "prohibition_reason": "immutable_m4_requires_pf_only_and_runtime_fgo_false",
        "truth_usage": "post_hoc_supply_diagnosis_only",
        "matched_epochs": matched,
        "candidate_sub50cm_epochs": good,
        "oracle_gain_over_production": gain,
        "replacement_loss": loss,
        "declared_fix_epochs": fix,
        "false_fix_epochs": false_fix,
        "declared_false_fix_percent": 100.0 * false_fix / fix if fix else 0.0,
    }


def build_audit(
    production_path: Path,
    native_paths: Iterable[Path],
    *,
    external_path: Path | None = None,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    native = audit_native_candidates(production_path, native_paths)
    external = None
    if external_path is not None:
        if reference_path is None:
            raise ValueError("reference_path is required for the external diagnostic")
        external = audit_external_research_candidate(
            production_path, external_path, reference_path
        )
    return {
        "schema": "gnss_gpu_tokyo_candidate_supply_audit_v1",
        "production_selection_truth": False,
        "audit_truth_usage": "post_hoc_inventory_only",
        "native_archive": native,
        "external_research_diagnostic": external,
        "conclusion": {
            "archived_native_candidates_reach_target": (
                native["novel_epochs_needed_beyond_archive"] == 0
            ),
            "external_candidate_is_production_eligible": False,
            "required_next_mechanism": (
                "new_pf_native_absolute_position_supply_and_truth_free_selection"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production",
        type=Path,
        default=Path(
            "results/wp31/"
            "tokyo_run1_wp160_screened_stability_full_trajectory.csv"
        ),
    )
    parser.add_argument(
        "--native-glob",
        default="results/wp31/tokyo_run1_*full_trajectory.csv",
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("results/wp14/tokyo_run1_fgo_gtsam.csv"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("E:/datasets/PPC-Dataset-data/tokyo/run1/reference.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native_paths = [
        path
        for path in Path().glob(args.native_glob)
        if path.resolve() != args.production.resolve()
    ]
    result = build_audit(
        args.production,
        native_paths,
        external_path=args.external,
        reference_path=args.reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result["conclusion"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
