#!/usr/bin/env python3
"""Audit a truth-free quality gate on gnssplusplus-library FIX output."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.promote_wp172_pf_seeded_rtk_consensus import (  # noqa: E402
    read_gnssplusplus_pos,
)


TARGETS = {"tokyo": 0.50, "nagoya": 0.60}


@dataclass(frozen=True)
class QualityGatePolicy:
    maximum_float_position_covariance_trace_m2: float = 0.00025
    maximum_covariance_branch_nis_per_observation: float = 10.0
    minimum_strong_innovation_observations: int = 28
    maximum_strong_innovation_nis_per_observation: float = 1.0
    maximum_strong_innovation_suppressed_fraction: float = 0.5


def _tow(value: str | float) -> float:
    return round(float(value), 3)


def _finite(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def gate_branches(
    row: dict[str, str], policy: QualityGatePolicy
) -> tuple[bool, bool, bool]:
    """Return (pass, safe-shadow branch, structural branch).

    The function deliberately has no truth input.  Truth is used only by the
    caller after this decision to audit retained library FIX epochs.
    """

    safe_shadow = row.get("safe_fix_shadow_declared_fixed") == "1"
    covariance = _finite(row, "float_position_covariance_trace_m2")
    nis = _finite(row, "float_update_nis_per_observation")
    observations = _finite(row, "float_update_observation_count")
    suppressed = _finite(row, "float_update_suppressed_outliers")
    covariance_branch = (
        covariance is not None
        and covariance
        <= policy.maximum_float_position_covariance_trace_m2
        and nis is not None
        and nis <= policy.maximum_covariance_branch_nis_per_observation
    )
    strong_innovation_branch = (
        observations is not None
        and observations >= policy.minimum_strong_innovation_observations
        and suppressed is not None
        and suppressed >= 0
        and suppressed / observations
        <= policy.maximum_strong_innovation_suppressed_fraction
        and nis is not None
        and nis
        <= policy.maximum_strong_innovation_nis_per_observation
    )
    structural = covariance_branch or strong_innovation_branch
    return safe_shadow or structural, safe_shadow, structural


def _block_metrics(
    records: list[dict[str, Any]], block_count: int
) -> dict[str, dict[str, Any]]:
    ordered = sorted(records, key=lambda record: record["tow"])
    output: dict[str, dict[str, Any]] = {}
    for block in range(block_count):
        start = len(ordered) * block // block_count
        end = len(ordered) * (block + 1) // block_count
        subset = ordered[start:end]
        retained = [record for record in subset if record["retained"]]
        output[str(block)] = {
            "start_tow": subset[0]["tow"] if subset else None,
            "end_tow": subset[-1]["tow"] if subset else None,
            "epochs": len(subset),
            "original_library_fixed_epochs": sum(
                record["fixed"] for record in subset
            ),
            "retained_library_fixed_epochs": len(retained),
            "retained_library_fix_rate": (
                len(retained) / len(subset) if subset else 0.0
            ),
            "retained_correct_fixed_epochs": sum(
                record["correct"] for record in retained
            ),
            "retained_false_fixed_epochs": sum(
                not record["correct"] for record in retained
            ),
        }
    return output


def analyze(
    rows: list[dict[str, str]],
    truth: dict[float, tuple[float, float, float]],
    positions: dict[float, dict[str, float | int]],
    domain: str,
    policy: QualityGatePolicy = QualityGatePolicy(),
    block_count: int = 5,
) -> dict[str, Any]:
    domain = domain.lower()
    if domain not in TARGETS:
        raise ValueError(f"unsupported domain: {domain}")
    debug = {_tow(row["tow"]): row for row in rows}
    positions = {_tow(tow): position for tow, position in positions.items()}
    records: list[dict[str, Any]] = []
    branch_counts = {
        "safe_shadow": 0,
        "structural": 0,
        "safe_shadow_and_structural": 0,
    }
    for tow, position in sorted(positions.items()):
        fixed = int(position.get("status", -1)) == 4
        retained = False
        safe_shadow = False
        structural = False
        if fixed and tow in debug:
            retained, safe_shadow, structural = gate_branches(
                debug[tow], policy
            )
            branch_counts["safe_shadow"] += int(safe_shadow)
            branch_counts["structural"] += int(structural)
            branch_counts["safe_shadow_and_structural"] += int(
                safe_shadow and structural
            )
        correct = False
        error_m = None
        if fixed and tow in truth:
            estimate = tuple(
                float(position[f"ecef_{axis}"]) for axis in "xyz"
            )
            error_m = math.dist(estimate, truth[tow])
            correct = error_m < 0.5
        records.append(
            {
                "tow": tow,
                "fixed": fixed,
                "retained": retained,
                "correct": correct,
                "error_m": error_m,
            }
        )

    fixed_records = [record for record in records if record["fixed"]]
    retained = [record for record in records if record["retained"]]
    false_original = [
        record
        for record in fixed_records
        if record["error_m"] is not None and not record["correct"]
    ]
    false_retained = [
        record
        for record in retained
        if record["error_m"] is not None and not record["correct"]
    ]
    correct_retained = sum(record["correct"] for record in retained)
    target = TARGETS[domain]
    return {
        "domain": domain,
        "epochs": len(records),
        "target_library_fix_rate": target,
        "original_library_fixed_epochs": len(fixed_records),
        "original_library_fix_rate": (
            len(fixed_records) / len(records) if records else 0.0
        ),
        "original_false_fixed_epochs": len(false_original),
        "retained_library_fixed_epochs": len(retained),
        "retained_library_fix_rate": (
            len(retained) / len(records) if records else 0.0
        ),
        "retained_correct_fixed_epochs": correct_retained,
        "retained_false_fixed_epochs": len(false_retained),
        "demoted_library_fixed_epochs": len(fixed_records) - len(retained),
        "passes_observed_false_fix_zero": not false_retained,
        "passes_library_fix_rate_target": (
            len(retained) / len(records) >= target if records else False
        ),
        "branch_counts_before_overlap_removal": branch_counts,
        "contiguous_blocks": _block_metrics(records, block_count),
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


def _parse_audit(specification: str) -> tuple[str, Path, Path, Path]:
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
    policy = QualityGatePolicy()
    domains = {
        domain: analyze(
            _read_rows(debug),
            _read_truth(reference),
            read_gnssplusplus_pos(positions),
            domain,
            policy,
        )
        for domain, debug, positions, reference in map(
            _parse_audit, args.audit
        )
    }
    payload = {
        "schema": "gnss_gpu_wp175_library_fix_quality_gate_audit_v1",
        "policy": asdict(policy),
        "runtime_fgo": False,
        "selection_truth_usage": (
            "retrospective development calibration; not an independent test"
        ),
        "runtime_truth_usage": "none",
        "audit_truth_usage": "post-decision error labeling only",
        "domains": domains,
        "promotion_ready": all(
            result["passes_observed_false_fix_zero"]
            and result["passes_library_fix_rate_target"]
            for result in domains.values()
        ),
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
