#!/usr/bin/env python3
"""Audit quality-diverse satellite-PAR against the prior combined ranking."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_safe_union import _quantile, _tow  # noqa: E402


def _candidate_error(
    row: dict[str, str],
    truth: dict[float, tuple[float, float, float]],
    prefix: str,
) -> float | None:
    tow = _tow(row["tow"])
    if tow not in truth or row.get(f"{prefix}_ffrt_passed") != "1":
        return None
    try:
        position = tuple(
            float(row[f"{prefix}_best_ecef_{axis}"])
            for axis in "xyz"
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in position):
        return None
    return math.dist(position, truth[tow])


def analyze(
    baseline: list[dict[str, str]],
    diverse: list[dict[str, str]],
    reference: list[dict[str, str]],
    domain: str,
) -> dict:
    truth = {
        _tow(row["GPS TOW (s)"]): tuple(
            float(row[f"ECEF {axis.upper()} (m)"]) for axis in "xyz"
        )
        for row in reference
    }
    baseline_by_tow = {_tow(row["tow"]): row for row in baseline}
    diverse_by_tow = {_tow(row["tow"]): row for row in diverse}
    common_tows = sorted(baseline_by_tow.keys() & diverse_by_tow.keys())

    def summarize(rows: list[dict[str, str]]) -> dict:
        satellite_errors = [
            error
            for row in rows
            if (
                error := _candidate_error(
                    row,
                    truth,
                    "lambda_satellite_par_shadow",
                )
            )
            is not None
        ]
        safe_errors = []
        runtimes = []
        for row in rows:
            tow = _tow(row["tow"])
            if (
                row.get("safe_fix_shadow_declared_fixed") == "1"
                and tow in truth
            ):
                try:
                    position = tuple(
                        float(row[f"lambda_shadow_best_ecef_{axis}"])
                        for axis in "xyz"
                    )
                except (KeyError, TypeError, ValueError):
                    position = ()
                if len(position) == 3 and all(
                    math.isfinite(value) for value in position
                ):
                    safe_errors.append(math.dist(position, truth[tow]))
            runtime = row.get("lambda_shadow_runtime_ms", "")
            if runtime:
                value = float(runtime)
                if math.isfinite(value):
                    runtimes.append(value)
        return {
            "epochs": len(rows),
            "satellite_par_ffrt_epochs": len(satellite_errors),
            "satellite_par_sub50cm_epochs": sum(
                error < 0.5 for error in satellite_errors
            ),
            "satellite_par_not_sub50cm_epochs": sum(
                error >= 0.5 for error in satellite_errors
            ),
            "safe_fix_epochs": len(safe_errors),
            "safe_fix_false_epochs": sum(
                error >= 0.5 for error in safe_errors
            ),
            "safe_fix_rate": (
                len(safe_errors) / len(rows) if rows else 0.0
            ),
            "runtime_p95_ms": _quantile(runtimes, 0.95),
        }

    baseline_common = [baseline_by_tow[tow] for tow in common_tows]
    diverse_common = [diverse_by_tow[tow] for tow in common_tows]
    baseline_summary = summarize(baseline_common)
    diverse_summary = summarize(diverse_common)
    runtime_p95 = diverse_summary["runtime_p95_ms"]
    return {
        "schema": "gnss_gpu_wp174_quality_diverse_satellite_par_audit_v1",
        "domain": domain,
        "common_epochs": len(common_tows),
        "runtime_fgo": False,
        "truth_usage": "post_selection_audit_only",
        "baseline_combined_ranking": baseline_summary,
        "quality_diverse_ranking": diverse_summary,
        "additional_satellite_par_ffrt_epochs": (
            diverse_summary["satellite_par_ffrt_epochs"]
            - baseline_summary["satellite_par_ffrt_epochs"]
        ),
        "pass_safe_fix_false_zero": (
            diverse_summary["safe_fix_false_epochs"] == 0
        ),
        "pass_runtime_p95_100ms": (
            runtime_p95 is not None and runtime_p95 <= 100.0
        ),
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--diverse", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))

    payload = analyze(
        read_csv(args.baseline),
        read_csv(args.diverse),
        read_csv(args.reference),
        args.domain,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
