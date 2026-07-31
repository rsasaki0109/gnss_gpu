#!/usr/bin/env python3
"""Build one canonical, truth-separated FIX-rate and AR blocker audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp175_library_fix_integrity import read_library_pos  # noqa: E402


def _tow(value: str | float) -> float:
    return round(float(value), 3)


def _flag(row: dict[str, str], field: str) -> bool:
    return row.get(field) == "1"


def _finite(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _position(
    row: dict[str, str], prefix: str
) -> tuple[float, float, float] | None:
    values = tuple(_finite(row, f"{prefix}_ecef_{axis}") for axis in "xyz")
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[arg-type]


def _quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _primary_blocker(row: dict[str, str], minimum_pairs: int) -> str:
    pairs = _finite(row, "primary_pair_count")
    if pairs is None or pairs < minimum_pairs:
        return "primary_pairs_below_minimum"
    if not _flag(row, "primary_ffrt_passed"):
        return "primary_ffrt_rejected"
    if not _flag(row, "disjoint_a_ffrt_passed") or not _flag(
        row, "disjoint_b_ffrt_passed"
    ):
        return "disjoint_partition_ffrt_unavailable"
    if not _flag(row, "disjoint_hard_separation_passed"):
        return "disjoint_hard_separation_rejected"
    if not _flag(row, "disjoint_statistical_separation_passed"):
        return "disjoint_statistical_separation_rejected"
    if not _flag(row, "disjoint_passed"):
        return "disjoint_other_rejection"
    if not _flag(row, "failure_budget_passed"):
        return "independent_failure_budget_rejected"
    if not _flag(row, "quality_gate_passed"):
        return "quality_gate_rejected"
    return "eligible_not_promoted"


SOURCE_SPECS = {
    "primary": {
        "available": "primary_ffrt_passed",
        "candidate_prefix": None,
        "pair_field": "primary_pair_count",
    },
    "causal_arc": {
        "available": "causal_arc_subset_ffrt_passed",
        "candidate_prefix": "causal_arc_candidate",
        "pair_field": "causal_arc_subset_pair_count",
    },
    "satellite_par": {
        "available": "satellite_par_ffrt_passed",
        "candidate_prefix": "satellite_par_candidate",
        "pair_field": "satellite_par_subset_size",
    },
    "source_par": {
        "available": "src_par_ffrt_passed",
        "candidate_prefix": "src_par_candidate",
        "pair_field": "src_par_subset_size",
    },
    "l1_l5_wlnl": {
        "available": "multifrequency_nl_ffrt_passed",
        "candidate_prefix": "multifrequency_candidate",
        "pair_field": "multifrequency_candidate_pair_count",
    },
    "l1_l2_wlnl": {
        "available": "l1_l2_multifrequency_nl_ffrt_passed",
        "candidate_prefix": "l1_l2_multifrequency_candidate",
        "pair_field": "l1_l2_multifrequency_candidate_pair_count",
    },
    "l2_l5_wlnl": {
        "available": "l2_l5_multifrequency_nl_ffrt_passed",
        "candidate_prefix": "l2_l5_multifrequency_candidate",
        "pair_field": "l2_l5_multifrequency_candidate_pair_count",
    },
}


def _source_metrics(
    rows: list[dict[str, str]],
    nonfixed_tows: set[float],
    truth: dict[float, tuple[float, float, float]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source, spec in SOURCE_SPECS.items():
        available = 0
        nonfixed_available = 0
        pair_counts: list[float] = []
        oracle_correct = 0
        oracle_wrong = 0
        prefix = spec["candidate_prefix"]
        for row in rows:
            if not _flag(row, str(spec["available"])):
                continue
            available += 1
            tow = _tow(row["tow"])
            if tow not in nonfixed_tows:
                continue
            nonfixed_available += 1
            pairs = _finite(row, str(spec["pair_field"]))
            if pairs is not None:
                pair_counts.append(pairs)
            if prefix is None:
                continue
            candidate = _position(row, str(prefix))
            reference = truth.get(tow)
            if candidate is None or reference is None:
                continue
            if math.dist(candidate, reference) < 0.5:
                oracle_correct += 1
            else:
                oracle_wrong += 1
        output[source] = {
            "ffrt_passed_epochs": available,
            "nonfixed_ffrt_passed_epochs": nonfixed_available,
            "nonfixed_pair_count_p50": _quantile(pair_counts, 0.5),
            "nonfixed_candidate_oracle_correct_epochs": oracle_correct,
            "nonfixed_candidate_oracle_wrong_epochs": oracle_wrong,
        }
    return output


def _block_metrics(
    records: list[dict[str, Any]], block_count: int
) -> dict[str, dict[str, Any]]:
    ordered = sorted(records, key=lambda record: record["tow"])
    output: dict[str, dict[str, Any]] = {}
    for index in range(block_count):
        start = index * len(ordered) // block_count
        stop = (index + 1) * len(ordered) // block_count
        block = ordered[start:stop]
        fixed = [record for record in block if record["fixed"]]
        correct = [record for record in fixed if record["correct"]]
        false = [record for record in fixed if record["error_m"] >= 0.5]
        output[str(index)] = {
            "start_tow": block[0]["tow"] if block else None,
            "end_tow": block[-1]["tow"] if block else None,
            "epochs": len(block),
            "fixed_epochs": len(fixed),
            "correct_fixed_epochs": len(correct),
            "false_fixed_epochs": len(false),
            "correct_fix_rate": len(correct) / len(block) if block else 0.0,
        }
    return output


def audit_route(
    name: str,
    rows: list[dict[str, str]],
    positions: dict[float, dict[str, float | int]],
    truth: dict[float, tuple[float, float, float]],
    *,
    minimum_pairs: int = 12,
    block_count: int = 5,
) -> dict[str, Any]:
    telemetry = {_tow(row["tow"]): row for row in rows}
    normalized_positions = {
        _tow(tow): position for tow, position in positions.items()
    }
    records: list[dict[str, Any]] = []
    for tow, position in sorted(normalized_positions.items()):
        fixed = int(position.get("status", -1)) == 4
        error_m = math.inf
        if tow in truth:
            estimate = tuple(
                float(position[f"ecef_{axis}"]) for axis in "xyz"
            )
            error_m = math.dist(estimate, truth[tow])
        records.append(
            {
                "tow": tow,
                "fixed": fixed,
                "correct": fixed and error_m < 0.5,
                "error_m": error_m,
            }
        )
    fixed = [record for record in records if record["fixed"]]
    correct = [record for record in fixed if record["correct"]]
    false = [record for record in fixed if record["error_m"] >= 0.5]
    nonfixed_tows = {
        record["tow"] for record in records if not record["fixed"]
    }
    blockers = Counter(
        _primary_blocker(telemetry[tow], minimum_pairs)
        if tow in telemetry
        else "telemetry_missing"
        for tow in nonfixed_tows
    )
    runtimes = [
        value
        for row in rows
        if (value := _finite(row, "processing_runtime_ms")) is not None
    ]
    arc_resets = [
        value
        for row in rows
        if (value := _finite(row, "causal_arc_resets")) is not None
    ]
    total = len(records)
    return {
        "route": name,
        "epochs": total,
        "fixed_epochs": len(fixed),
        "correct_fixed_epochs": len(correct),
        "false_fixed_epochs": len(false),
        "fixed_rate": len(fixed) / total if total else 0.0,
        "correct_fix_rate": len(correct) / total if total else 0.0,
        "false_per_fixed": len(false) / len(fixed) if fixed else 0.0,
        "false_fixed_above_1m_epochs": sum(
            record["error_m"] > 1.0 for record in false
        ),
        "fixed_error_p95_m": _quantile(
            [record["error_m"] for record in fixed], 0.95
        ),
        "runtime_p50_ms": _quantile(runtimes, 0.5),
        "runtime_p95_ms": _quantile(runtimes, 0.95),
        "runtime_max_ms": max(runtimes) if runtimes else None,
        "nonfixed_primary_blockers": dict(sorted(blockers.items())),
        "candidate_sources": _source_metrics(rows, nonfixed_tows, truth),
        "causal_arc": {
            "epochs_with_ready_pairs": sum(
                (_finite(row, "causal_arc_ready_pairs") or 0.0) > 0
                for row in rows
            ),
            "subset_ffrt_passed_epochs": sum(
                _flag(row, "causal_arc_subset_ffrt_passed") for row in rows
            ),
            "reset_events": max(arc_resets) if arc_resets else 0,
        },
        "contiguous_time_blocks": _block_metrics(records, block_count),
    }


def build_audit(
    specifications: list[
        tuple[
            str,
            list[dict[str, str]],
            dict[float, dict[str, float | int]],
            dict[float, tuple[float, float, float]],
        ]
    ],
    *,
    minimum_pairs: int = 12,
    block_count: int = 5,
) -> dict[str, Any]:
    routes = {
        name: audit_route(
            name,
            rows,
            positions,
            truth,
            minimum_pairs=minimum_pairs,
            block_count=block_count,
        )
        for name, rows, positions, truth in specifications
    }
    epochs = sum(route["epochs"] for route in routes.values())
    fixed = sum(route["fixed_epochs"] for route in routes.values())
    correct = sum(route["correct_fixed_epochs"] for route in routes.values())
    false = sum(route["false_fixed_epochs"] for route in routes.values())
    return {
        "schema": "gnss_gpu_fix_rate_canonical_audit_v1",
        "fix_definition": "gnssplusplus final .pos Status == 4",
        "correct_fix_definition": "Status == 4 and 3D error < 0.5 m",
        "selection_truth_usage": "none",
        "truth_usage": "post-decision audit and candidate oracle only",
        "minimum_pairs": minimum_pairs,
        "routes": routes,
        "aggregate": {
            "epochs": epochs,
            "fixed_epochs": fixed,
            "correct_fixed_epochs": correct,
            "false_fixed_epochs": false,
            "fixed_rate": fixed / epochs if epochs else 0.0,
            "correct_fix_rate": correct / epochs if epochs else 0.0,
            "false_per_fixed": false / fixed if fixed else 0.0,
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _read_truth(path: Path) -> dict[float, tuple[float, float, float]]:
    return {
        _tow(row["GPS TOW (s)"]): tuple(
            float(row[f"ECEF {axis.upper()} (m)"]) for axis in "xyz"
        )
        for row in _read_csv(path)
    }


def _parse_route(
    specification: str,
) -> tuple[str, Path, Path, Path]:
    name, integrity, positions, reference = specification.split("=", 3)
    return name, Path(integrity), Path(positions), Path(reference)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        action="append",
        required=True,
        help="NAME=INTEGRITY_CSV=POSITIONS=REFERENCE_CSV",
    )
    parser.add_argument("--minimum-pairs", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.minimum_pairs < 4:
        parser.error("--minimum-pairs must be at least 4")
    if args.blocks < 2:
        parser.error("--blocks must be at least 2")

    inputs = []
    input_hashes: dict[str, dict[str, str]] = {}
    for name, integrity, positions, reference in map(
        _parse_route, args.route
    ):
        inputs.append(
            (
                name,
                _read_csv(integrity),
                read_library_pos(positions),
                _read_truth(reference),
            )
        )
        input_hashes[name] = {
            "integrity_sha256": _sha256(integrity),
            "positions_sha256": _sha256(positions),
            "reference_sha256": _sha256(reference),
        }
    payload = build_audit(
        inputs,
        minimum_pairs=args.minimum_pairs,
        block_count=args.blocks,
    )
    payload["input_hashes"] = input_hashes
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
