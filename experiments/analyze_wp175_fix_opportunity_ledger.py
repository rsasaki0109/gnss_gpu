#!/usr/bin/env python3
"""Build a truth-separated safe-FIX opportunity ledger for WP175."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.wp174_ffrt import passes_ffrt  # noqa: E402
from experiments.promote_wp172_pf_seeded_rtk_consensus import (  # noqa: E402
    read_gnssplusplus_pos,
)


TARGETS = {"tokyo": 0.50, "nagoya": 0.60}


def _finite(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _tow(value: str) -> float:
    return round(float(value), 3)


def _position(
    row: dict[str, str], prefix: str
) -> tuple[float, float, float] | None:
    values = tuple(_finite(row, f"{prefix}_{axis}") for axis in "xyz")
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[arg-type]


def _current_blocker(row: dict[str, str]) -> str:
    if row.get("lambda_shadow_attempted") != "1":
        return "lambda_not_attempted"
    if row.get("lambda_shadow_solved") != "1":
        return "lambda_not_solved"
    pairs = _finite(row, "pair_count")
    if pairs is None or pairs < 12:
        return "pairs_below_12"
    ratio = _finite(row, "full_ratio")
    bsr = _finite(row, "lambda_shadow_bsr_qscale16")
    if (
        ratio is None
        or bsr is None
        or not passes_ffrt(int(pairs), bsr, ratio)
    ):
        return "ffrt_failed"
    second_delta = _finite(row, "lambda_shadow_second_position_delta_m")
    if second_delta is None or second_delta > 0.25:
        return "second_position_delta_above_25cm"
    nis = _finite(row, "float_update_nis_per_observation")
    if nis is None or nis > 3.0:
        return "nis_above_3"
    prefit = _finite(row, "float_update_prefit_residual_rms_m")
    if prefit is None or prefit > 50.0:
        return "prefit_above_50m"
    if ratio < 1.4:
        return "absolute_ratio_below_1p4"
    consensus = _finite(
        row, "safe_fix_shadow_independent_consensus_delta_m"
    )
    if consensus is None:
        return "independent_consensus_missing"
    if consensus > 0.10:
        return "independent_consensus_above_10cm"
    return "eligible_but_temporally_unconfirmed"


def _candidate_positions(
    row: dict[str, str],
) -> dict[str, list[tuple[float, float, float]]]:
    sources: dict[str, list[tuple[float, float, float]]] = {}
    if row.get("lambda_shadow_solved") == "1":
        best = _position(row, "lambda_shadow_best_ecef")
        if best is not None:
            sources["lambda_best"] = [best]
        topk = [
            candidate
            for ordinal in range(1, 9)
            if (
                candidate := _position(
                    row, f"lambda_shadow_candidate_{ordinal}_ecef"
                )
            )
            is not None
        ]
        if topk:
            sources["lambda_topk"] = topk
    source_specs = {
        "src_par": (
            "lambda_src_par_shadow_ffrt_passed",
            "lambda_src_par_shadow_best_ecef",
        ),
        "satellite_par": (
            "lambda_satellite_par_shadow_ffrt_passed",
            "lambda_satellite_par_shadow_best_ecef",
        ),
        "l1_l5_wlnl": (
            "lambda_l1_l5_wlnl_shadow_nl_ffrt_passed",
            "lambda_l1_l5_wlnl_shadow_best_ecef",
        ),
    }
    for source, (passed_field, position_prefix) in source_specs.items():
        candidate = _position(row, position_prefix)
        if row.get(passed_field) == "1" and candidate is not None:
            sources[source] = [candidate]
    return sources


def _run_lengths(rows: list[dict[str, str]]) -> list[float]:
    runs: list[list[float]] = []
    current: list[float] = []
    for row in sorted(rows, key=lambda item: _tow(item["tow"])):
        tow = _tow(row["tow"])
        if row.get("safe_fix_shadow_declared_fixed") == "1":
            if current:
                runs.append(current)
                current = []
        else:
            current.append(tow)
    if current:
        runs.append(current)
    return [run[-1] - run[0] + 0.2 for run in runs]


def analyze(
    rows: list[dict[str, str]],
    truth: dict[float, tuple[float, float, float]],
    positions: dict[float, dict[str, float | int]],
    domain: str,
    block_count: int = 5,
) -> dict[str, Any]:
    if domain not in TARGETS:
        raise ValueError(f"unsupported domain: {domain}")
    positions = {_tow(str(tow)): position for tow, position in positions.items()}
    fixed_tows = {
        tow
        for tow, position in positions.items()
        if int(position.get("status", -1)) == 4
    }
    fixed_errors = {
        tow: math.dist(
            tuple(float(positions[tow][f"ecef_{axis}"]) for axis in "xyz"),
            truth[tow],
        )
        for tow in fixed_tows & truth.keys()
    }
    correct_fixed_tows = {
        tow for tow, error in fixed_errors.items() if error < 0.5
    }
    false_fixed_tows = fixed_errors.keys() - correct_fixed_tows
    fixed_rows = [row for row in rows if _tow(row["tow"]) in fixed_tows]
    nonfixed_rows = [
        row for row in rows if _tow(row["tow"]) not in fixed_tows
    ]
    blockers = Counter(_current_blocker(row) for row in nonfixed_rows)
    source_metrics: dict[str, Counter[str]] = {}
    union_safe_tows: set[float] = set()
    for row in nonfixed_rows:
        tow = _tow(row["tow"])
        reference = truth.get(tow)
        for source, positions in _candidate_positions(row).items():
            metrics = source_metrics.setdefault(source, Counter())
            metrics["candidate_epochs"] += 1
            if reference is None:
                continue
            metrics["truth_labeled_epochs"] += 1
            safe = any(math.dist(position, reference) < 0.5 for position in positions)
            metrics["sub50cm_oracle_epochs"] += safe
            metrics["not_sub50cm_oracle_epochs"] += not safe
            if safe:
                union_safe_tows.add(tow)
    target_epochs = math.ceil(TARGETS[domain] * len(rows))
    required = max(0, target_epochs - len(correct_fixed_tows))
    if block_count < 2:
        raise ValueError("block_count must be at least 2")
    ordered_rows = sorted(rows, key=lambda item: _tow(item["tow"]))
    blocks: dict[str, dict[str, Any]] = {}
    for block_index in range(block_count):
        start = block_index * len(ordered_rows) // block_count
        stop = (block_index + 1) * len(ordered_rows) // block_count
        block_rows = ordered_rows[start:stop]
        block_fixed = sum(
            _tow(row["tow"]) in fixed_tows
            for row in block_rows
        )
        block_correct = sum(
            _tow(row["tow"]) in correct_fixed_tows
            for row in block_rows
        )
        block_false = block_fixed - block_correct
        blocks[str(block_index)] = {
            "epochs": len(block_rows),
            "start_tow": _tow(block_rows[0]["tow"]),
            "end_tow": _tow(block_rows[-1]["tow"]),
            "safe_fix_epochs": block_fixed,
            "safe_fix_rate": block_fixed / len(block_rows),
            "correct_fix_epochs": block_correct,
            "false_fix_epochs": block_false,
            "correct_fix_rate": block_correct / len(block_rows),
            "nonfix_epochs": len(block_rows) - block_fixed,
        }
    durations = _run_lengths(rows)
    return {
        "domain": domain,
        "epochs": len(rows),
        "library_fixed_epochs": len(fixed_rows),
        "library_fix_rate": len(fixed_rows) / len(rows),
        "truth_labeled_library_fixed_epochs": len(fixed_errors),
        "correct_library_fixed_epochs": len(correct_fixed_tows),
        "false_library_fixed_epochs": len(false_fixed_tows),
        "correct_library_fix_rate": len(correct_fixed_tows) / len(rows),
        "target_safe_fix_rate": TARGETS[domain],
        "target_safe_fix_epochs": target_epochs,
        "additional_correct_fix_epochs_required_after_false_demotion": required,
        "nonfix_epochs": len(nonfixed_rows),
        "truth_free_primary_blockers": dict(sorted(blockers.items())),
        "post_selection_candidate_oracles": {
            source: dict(sorted(metrics.items()))
            for source, metrics in sorted(source_metrics.items())
        },
        "post_selection_union_sub50cm_oracle_epochs": len(union_safe_tows),
        "post_selection_union_covers_required_increment": (
            len(union_safe_tows) >= required
        ),
        "nonfix_run_count": len(durations),
        "nonfix_run_max_s": max(durations) if durations else 0.0,
        "contiguous_nested_cv_blocks": blocks,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _read_truth(
    path: Path,
) -> dict[float, tuple[float, float, float]]:
    return {
        _tow(row["GPS TOW (s)"]): (
            float(row["ECEF X (m)"]),
            float(row["ECEF Y (m)"]),
            float(row["ECEF Z (m)"]),
        )
        for row in _read_rows(path)
    }


def _parse_audit(
    specification: str,
) -> tuple[str, Path, Path, Path]:
    domain, debug, positions, reference = specification.split("=", 3)
    return domain, Path(debug), Path(positions), Path(reference)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        action="append",
        required=True,
        help="DOMAIN=DEBUG_CSV=POSITIONS=REFERENCE_CSV",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    domains = {
        domain: analyze(
            _read_rows(debug),
            _read_truth(reference),
            read_gnssplusplus_pos(positions),
            domain,
        )
        for domain, debug, positions, reference in map(
            _parse_audit, args.audit
        )
    }
    payload = {
        "schema": "gnss_gpu_wp175_library_fix_opportunity_ledger_v2",
        "runtime_fgo": False,
        "selection_truth_usage": "none",
        "truth_usage": "post_selection_candidate_oracle_audit_only",
        "runtime_policy": {
            "minimum_pairs": 12,
            "ffrt_failure_rate": 0.001,
            "covariance_scale": 16,
            "maximum_second_position_delta_m": 0.25,
            "maximum_nis_per_observation": 3.0,
            "maximum_prefit_residual_rms_m": 50.0,
            "minimum_absolute_ratio": 1.4,
            "maximum_independent_consensus_delta_m": 0.10,
        },
        "domains": domains,
    }
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
