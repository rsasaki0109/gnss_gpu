#!/usr/bin/env python3
"""Declare guarded LAMBDA FIX states on a locked WP172 trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.promote_wp172_pf_seeded_rtk_consensus import (  # noqa: E402
    read_gnssplusplus_pos,
    select_consensus_candidates,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_text_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest().upper()


def _tow(value: str | float) -> float:
    return round(float(value), 3)


def declare_lambda_fix_epochs(
    candidates: dict[float, dict[str, float | int]],
    *,
    minimum_ratio: float,
    minimum_satellites: int,
    minimum_contiguous_epochs: int,
    maximum_epoch_gap_s: float,
) -> set[float]:
    """Return causal FIX declarations without loading position truth."""

    if minimum_contiguous_epochs < 1:
        raise ValueError("minimum_contiguous_epochs must be positive")
    declared: set[float] = set()
    streak = 0
    previous_tow: float | None = None
    for tow, candidate in sorted(candidates.items()):
        eligible = (
            float(candidate["ratio"]) >= minimum_ratio
            and int(candidate["num_satellites"]) >= minimum_satellites
        )
        contiguous = (
            previous_tow is not None
            and 0.0 < tow - previous_tow <= maximum_epoch_gap_s
        )
        streak = streak + 1 if eligible and contiguous else (1 if eligible else 0)
        if streak >= minimum_contiguous_epochs:
            declared.add(tow)
        previous_tow = tow
    return declared


def promote(
    wp172_path: Path,
    wp172_summary_path: Path,
    seeded_path: Path,
    independent_path: Path,
    contract_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    upstream_result = json.loads(wp172_summary_path.read_text(encoding="utf-8"))
    upstream_path = Path(contract["upstream_contract"])
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    upstream_gate = upstream["candidate_gate"]
    policy = contract["integer_ambiguity_method"]

    seeded = read_gnssplusplus_pos(seeded_path)
    independent = read_gnssplusplus_pos(independent_path)
    consensus = select_consensus_candidates(
        seeded,
        independent,
        required_status=int(upstream_gate["required_seeded_status"]),
        max_disagreement_m=float(
            upstream_gate["max_independent_position_disagreement_m"]
        ),
        max_prefit_residual_rms_m=float(
            upstream_gate["max_prefit_residual_rms_m"]
        ),
    )
    declared = declare_lambda_fix_epochs(
        consensus,
        minimum_ratio=float(policy["minimum_lambda_ratio"]),
        minimum_satellites=int(policy["minimum_satellites"]),
        minimum_contiguous_epochs=int(policy["minimum_contiguous_epochs"]),
        maximum_epoch_gap_s=float(policy["maximum_epoch_gap_s"]),
    )

    rows = _read_csv(wp172_path)
    if len(rows) != int(upstream_result["full_denominator_epochs"]):
        raise ValueError("WP172 trajectory and summary denominators differ")
    if sum(int(row["sub50cm"]) for row in rows) != int(
        upstream_result["after_sub50cm_epochs"]
    ):
        raise ValueError("WP172 trajectory and summary sub-50 cm counts differ")
    output: list[dict[str, Any]] = []
    false_fix = 0
    for row in rows:
        is_fix = _tow(row["tow"]) in declared
        is_false_fix = is_fix and int(row["sub50cm"]) == 0
        false_fix += is_false_fix
        output.append(
            {
                **row,
                "fix": int(is_fix),
                "false_fix": int(is_false_fix),
            }
        )

    m4 = {
        path: _sha256(Path(path))
        for path in contract["m4_expected_sha256"]
    }
    summary = {
        "schema": "gnss_gpu_wp173_lambda_fix_declaration_result_v1",
        "contract": contract_path.as_posix(),
        "upstream_trajectory": wp172_path.as_posix(),
        "production_input_truth": contract["production_input_truth"],
        "truth_usage": "post_selection_full_denominator_audit_only",
        "pf_only": contract["pf_only"],
        "runtime_fgo": contract["runtime_fgo"],
        "integer_ambiguity_method": policy,
        "input_hashes": {
            "contract": _sha256(contract_path),
            "upstream_contract": _sha256(upstream_path),
            "upstream_trajectory": _sha256(wp172_path),
            "upstream_summary": _sha256(wp172_summary_path),
            "seeded_rtk": _sha256(seeded_path),
            "independent_rtk": _sha256(independent_path),
        },
        "mandatory_negative_holdouts": {
            holdout: {
                "accepted": False,
                "fix_epochs": 0,
                "reason": contract["holdout_policy"][
                    "mandatory_negative_holdout_disposition"
                ],
            }
            for holdout in (
                "nagoya_wp53",
                "tokyo_wp129",
                "tokyo_wp156",
                "tokyo_wp168",
            )
        },
        "m4_preserved_sha256": m4,
        "selected_consensus_epochs": len(consensus),
        "full_denominator_epochs": len(output),
        "before_sub50cm_epochs": upstream_result["before_sub50cm_epochs"],
        "after_sub50cm_epochs": sum(int(row["sub50cm"]) for row in output),
        "after_sub50cm_percent": (
            100.0 * sum(int(row["sub50cm"]) for row in output) / len(output)
        ),
        "gained_epochs": upstream_result["gained_epochs"],
        "lost_epochs": upstream_result["lost_epochs"],
        "fix_epochs": len(declared),
        "fix_percent": 100.0 * len(declared) / len(output),
        "false_fix_epochs": false_fix,
        "declared_false_fix_percent": (
            100.0 * false_fix / len(declared) if declared else 0.0
        ),
        "promotion_allowed": (
            bool(declared)
            and false_fix == 0
            and all(
                m4[path] == expected
                for path, expected in contract["m4_expected_sha256"].items()
            )
        ),
    }
    return output, summary


def _write_trajectory(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp172", type=Path, required=True)
    parser.add_argument("--wp172-summary", type=Path, required=True)
    parser.add_argument("--seeded-rtk", type=Path, required=True)
    parser.add_argument("--independent-rtk", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/evaluation/wp173_lambda_fix_declaration.json"),
    )
    parser.add_argument("--output-trajectory", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = promote(
        args.wp172,
        args.wp172_summary,
        args.seeded_rtk,
        args.independent_rtk,
        args.contract,
    )
    _write_trajectory(args.output_trajectory, rows)
    summary["output_trajectory"] = args.output_trajectory.as_posix()
    summary["output_trajectory_sha256"] = _sha256(args.output_trajectory)
    summary["output_trajectory_canonical_sha256"] = _canonical_text_sha256(
        args.output_trajectory
    )
    summary["output_trajectory_hash_normalization"] = "CRLF/LF normalized to LF"
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["promotion_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
