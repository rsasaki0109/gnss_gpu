#!/usr/bin/env python3
"""Inventory surviving PPC POS trajectories under the official score contract.

This is an offline audit: reference data is used only to score and rank already
materialized trajectories.  Its output must never be consumed by the runtime
estimator or a supposedly blind selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from experiments.evaluate_ppc_official_score import read_estimates, read_reference
except ModuleNotFoundError:
    from evaluate_ppc_official_score import read_estimates, read_reference  # type: ignore[no-redef]
from gnss_gpu.ppc_score import score_ppc2024


ROUTES = tuple(
    (city, run, f"{city}_{run}")
    for city in ("tokyo", "nagoya")
    for run in ("run1", "run2", "run3")
)
_FORBIDDEN_RUNTIME_CANDIDATE_TOKENS = ("oracle", "reference", "ground_truth", "groundtruth")


def is_truth_derived_path(path: Path) -> bool:
    normalized = str(path).replace("-", "_").lower()
    return any(token in normalized for token in _FORBIDDEN_RUNTIME_CANDIDATE_TOKENS)


def load_references(dataset_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        route: read_reference(dataset_root / city / run / "reference.csv")
        for city, run, route in ROUTES
    }


def score_candidate(
    path: Path,
    references: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    minimum_coverage: float,
) -> dict[str, Any] | None:
    estimates, statuses = read_estimates(path)
    if not estimates:
        return None
    estimate_tows = set(estimates)
    overlaps = {
        route: sum(float(tow) in estimate_tows for tow in reference_tow)
        for route, (reference_tow, _) in references.items()
    }
    route = max(overlaps, key=overlaps.get)  # type: ignore[arg-type]
    reference_tow, reference_xyz = references[route]
    overlap = int(overlaps[route])
    coverage = overlap / reference_tow.size
    if coverage < minimum_coverage:
        return None

    aligned = np.full_like(reference_xyz, np.nan)
    fixed = np.zeros(reference_tow.size, dtype=bool)
    for index, tow in enumerate(reference_tow):
        key = float(tow)
        if key in estimates:
            aligned[index] = estimates[key]
        fixed[index] = statuses.get(key, 0) != 0
    score = score_ppc2024(aligned, reference_xyz)
    false_fix = fixed & ~score.pass_mask
    severe_false_fix = fixed & np.isfinite(score.errors_3d) & (score.errors_3d > 1.0)
    return {
        "path": str(path),
        "route": route,
        "coverage_pct": 100.0 * coverage,
        "matched_epochs": overlap,
        "reference_epochs": int(reference_tow.size),
        "ppc_score_pct": score.score_pct,
        "pass_distance_m": score.pass_distance_m,
        "total_distance_m": score.total_distance_m,
        "fixed_epochs": int(np.sum(fixed)),
        "false_fix_epochs": int(np.sum(false_fix)),
        "false_fix_above_1m_epochs": int(np.sum(severe_false_fix)),
    }


def inventory(
    roots: Iterable[Path],
    dataset_root: Path,
    *,
    minimum_coverage: float = 0.95,
    top_per_route: int = 20,
) -> dict[str, Any]:
    references = load_references(dataset_root)
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    discovered = sorted({path.resolve() for root in roots for path in root.rglob("*.pos")})
    excluded_truth_derived = [path for path in discovered if is_truth_derived_path(path)]
    paths = [path for path in discovered if not is_truth_derived_path(path)]
    for path in paths:
        try:
            row = score_candidate(path, references, minimum_coverage=minimum_coverage)
        except (OSError, TypeError, ValueError) as exc:
            failures.append({"path": str(path), "error": str(exc)})
            continue
        if row is not None:
            candidates.append(row)
    by_route = {
        route: sorted(
            (row for row in candidates if row["route"] == route),
            key=lambda row: (-float(row["ppc_score_pct"]), str(row["path"])),
        )[:top_per_route]
        for _, _, route in ROUTES
    }
    return {
        "schema": "gnss_gpu_ppc_pos_candidate_inventory_v1",
        "truth_contract": {
            "production_input_truth": False,
            "truth_usage": "offline_candidate_audit_only",
            "runtime_selector_input_permitted": False,
        },
        "minimum_coverage": float(minimum_coverage),
        "files_discovered": len(discovered),
        "files_scanned": len(paths),
        "truth_derived_files_excluded": len(excluded_truth_derived),
        "truth_derived_paths": [str(path) for path in excluded_truth_derived],
        "full_coverage_candidates": len(candidates),
        "parse_failures": len(failures),
        "failures": failures,
        "top_by_route": by_route,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--top-per-route", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0.0 < args.minimum_coverage <= 1.0:
        parser.error("--minimum-coverage must be in (0, 1]")
    if args.top_per_route < 1:
        parser.error("--top-per-route must be positive")
    result = inventory(
        args.root,
        args.dataset_root,
        minimum_coverage=args.minimum_coverage,
        top_per_route=args.top_per_route,
    )
    encoded = json.dumps(result, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
