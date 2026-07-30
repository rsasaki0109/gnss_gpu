#!/usr/bin/env python3
"""Audit WP173 AR gates without changing positions or FIX declarations."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.promote_wp172_pf_seeded_rtk_consensus import (  # noqa: E402
    read_gnssplusplus_pos,
    select_consensus_candidates,
)
from experiments.promote_wp173_lambda_fix_declarations import (  # noqa: E402
    declare_lambda_fix_epochs,
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


def _tow(value: str | float) -> float:
    return round(float(value), 3)


def trace_declaration_gate(
    candidates: dict[float, dict[str, float | int]],
    *,
    minimum_ratio: float,
    minimum_satellites: int,
    minimum_contiguous_epochs: int,
    maximum_epoch_gap_s: float,
) -> dict[float, dict[str, Any]]:
    """Return the causal WP173 gate state for every consensus candidate."""

    if minimum_contiguous_epochs < 1:
        raise ValueError("minimum_contiguous_epochs must be positive")
    traced: dict[float, dict[str, Any]] = {}
    streak = 0
    previous_tow: float | None = None
    for tow, candidate in sorted(candidates.items()):
        ratio_pass = float(candidate["ratio"]) >= minimum_ratio
        satellites_pass = int(candidate["num_satellites"]) >= minimum_satellites
        eligible = ratio_pass and satellites_pass
        contiguous = (
            previous_tow is not None
            and 0.0 < tow - previous_tow <= maximum_epoch_gap_s
        )
        streak = streak + 1 if eligible and contiguous else (1 if eligible else 0)
        declared = eligible and streak >= minimum_contiguous_epochs
        if declared:
            reason = "declared_fix"
        elif not ratio_pass and not satellites_pass:
            reason = "ratio_and_satellites"
        elif not ratio_pass:
            reason = "ratio"
        elif not satellites_pass:
            reason = "satellites"
        else:
            reason = "streak_warmup_or_gap"
        traced[tow] = {
            "ratio_pass": ratio_pass,
            "satellites_pass": satellites_pass,
            "eligible": eligible,
            "contiguous": contiguous,
            "streak": streak,
            "declared": declared,
            "reason": reason,
        }
        previous_tow = tow
    return traced


def _counterfactual(
    candidates: dict[float, dict[str, float | int]],
    sub50cm_by_tow: dict[float, bool],
    *,
    ratios: Iterable[float],
    streaks: Iterable[int],
    minimum_satellites: int,
    maximum_epoch_gap_s: float,
    locked_declared: set[float],
) -> list[dict[str, Any]]:
    output = []
    for ratio in ratios:
        for streak in streaks:
            declared = declare_lambda_fix_epochs(
                candidates,
                minimum_ratio=float(ratio),
                minimum_satellites=minimum_satellites,
                minimum_contiguous_epochs=int(streak),
                maximum_epoch_gap_s=maximum_epoch_gap_s,
            )
            false_fix = sum(
                not sub50cm_by_tow.get(tow, False)
                for tow in declared
            )
            output.append(
                {
                    "minimum_lambda_ratio": float(ratio),
                    "minimum_contiguous_epochs": int(streak),
                    "fix_epochs": len(declared),
                    "additional_fix_epochs_vs_locked": len(
                        declared - locked_declared
                    ),
                    "false_fix_epochs": false_fix,
                    "declared_false_fix_percent": (
                        100.0 * false_fix / len(declared) if declared else 0.0
                    ),
                }
            )
    return output


def analyze(
    wp172_path: Path,
    seeded_path: Path,
    independent_path: Path,
    contract_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build truth-separated shadow telemetry and a gate-funnel summary."""

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    upstream_path = Path(contract["upstream_contract"])
    declaration_path = Path(contract["declaration_contract"])
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    upstream_gate = upstream["candidate_gate"]
    policy = declaration["integer_ambiguity_method"]

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
    trace = trace_declaration_gate(
        consensus,
        minimum_ratio=float(policy["minimum_lambda_ratio"]),
        minimum_satellites=int(policy["minimum_satellites"]),
        minimum_contiguous_epochs=int(policy["minimum_contiguous_epochs"]),
        maximum_epoch_gap_s=float(policy["maximum_epoch_gap_s"]),
    )
    locked_declared = {
        tow for tow, state in trace.items() if bool(state["declared"])
    }

    wp172_rows = _read_csv(wp172_path)
    sub50cm_by_tow = {
        _tow(row["tow"]): bool(int(row["sub50cm"]))
        for row in wp172_rows
    }
    output: list[dict[str, Any]] = []
    for row in wp172_rows:
        tow = _tow(row["tow"])
        candidate = consensus.get(tow)
        state = trace.get(tow)
        if candidate is None or state is None:
            output.append(
                {
                    "epoch": int(row["epoch"]),
                    "tow": float(row["tow"]),
                    "consensus_candidate": 0,
                    "lambda_ratio": "",
                    "num_satellites": "",
                    "prefit_residual_rms_m": "",
                    "update_nis_per_observation": "",
                    "ratio_pass": 0,
                    "satellites_pass": 0,
                    "eligible": 0,
                    "contiguous": 0,
                    "streak": 0,
                    "locked_declared_fix": 0,
                    "rejection_reason": "no_consensus",
                    "audit_sub50cm": int(row["sub50cm"]),
                }
            )
            continue
        output.append(
            {
                "epoch": int(row["epoch"]),
                "tow": float(row["tow"]),
                "consensus_candidate": 1,
                "lambda_ratio": float(candidate["ratio"]),
                "num_satellites": int(candidate["num_satellites"]),
                "prefit_residual_rms_m": float(
                    candidate["prefit_residual_rms_m"]
                ),
                "update_nis_per_observation": float(
                    candidate["update_nis_per_observation"]
                ),
                "ratio_pass": int(state["ratio_pass"]),
                "satellites_pass": int(state["satellites_pass"]),
                "eligible": int(state["eligible"]),
                "contiguous": int(state["contiguous"]),
                "streak": int(state["streak"]),
                "locked_declared_fix": int(state["declared"]),
                "rejection_reason": state["reason"],
                "audit_sub50cm": int(row["sub50cm"]),
            }
        )

    reasons = Counter(row["rejection_reason"] for row in output)
    reason_audit = {
        reason: {
            "epochs": count,
            "output_sub50cm_epochs": sum(
                int(row["audit_sub50cm"])
                for row in output
                if row["rejection_reason"] == reason
            ),
        }
        for reason, count in sorted(reasons.items())
    }
    counterfactuals = _counterfactual(
        consensus,
        sub50cm_by_tow,
        ratios=contract["counterfactual_grid"]["lambda_ratio_thresholds"],
        streaks=contract["counterfactual_grid"]["minimum_contiguous_epochs"],
        minimum_satellites=int(policy["minimum_satellites"]),
        maximum_epoch_gap_s=float(policy["maximum_epoch_gap_s"]),
        locked_declared=locked_declared,
    )
    locked_rows = [
        row
        for row in counterfactuals
        if math.isclose(
            row["minimum_lambda_ratio"],
            float(policy["minimum_lambda_ratio"]),
        )
        and row["minimum_contiguous_epochs"]
        == int(policy["minimum_contiguous_epochs"])
    ]
    if len(locked_rows) != 1:
        raise ValueError("counterfactual grid must contain the locked WP173 policy")

    summary = {
        "schema": "gnss_gpu_wp174_shadow_ar_diagnostics_result_v1",
        "contract": contract_path.as_posix(),
        "shadow_only": bool(contract["shadow_only"]),
        "production_input_truth": bool(contract["production_input_truth"]),
        "runtime_fgo": bool(contract["runtime_fgo"]),
        "selection_truth_usage": "none",
        "audit_truth_usage": contract["output_policy"]["truth_use"],
        "input_hashes": {
            "contract": _sha256(contract_path),
            "upstream_contract": _sha256(upstream_path),
            "declaration_contract": _sha256(declaration_path),
            "wp172_trajectory": _sha256(wp172_path),
            "seeded_rtk": _sha256(seeded_path),
            "independent_rtk": _sha256(independent_path),
        },
        "locked_policy": policy,
        "gate_funnel": {
            "full_denominator_epochs": len(output),
            "consensus_candidate_epochs": len(consensus),
            "ratio_pass_epochs": sum(
                bool(state["ratio_pass"]) for state in trace.values()
            ),
            "satellites_pass_epochs": sum(
                bool(state["satellites_pass"]) for state in trace.values()
            ),
            "eligible_epochs": sum(
                bool(state["eligible"]) for state in trace.values()
            ),
            "locked_declared_fix_epochs": len(locked_declared),
        },
        "rejection_reasons": reason_audit,
        "counterfactuals_post_selection_audit_only": counterfactuals,
        "locked_counterfactual_matches_production": (
            locked_rows[0]["fix_epochs"] == len(locked_declared)
        ),
        "positions_modified": False,
        "fix_declarations_modified": False,
    }
    return output, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp172", type=Path, required=True)
    parser.add_argument("--seeded-rtk", type=Path, required=True)
    parser.add_argument("--independent-rtk", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/evaluation/wp174_shadow_ar_diagnostics.json"),
    )
    parser.add_argument("--output-diagnostics", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = analyze(
        args.wp172,
        args.seeded_rtk,
        args.independent_rtk,
        args.contract,
    )
    _write_csv(args.output_diagnostics, rows)
    summary["output_diagnostics"] = args.output_diagnostics.as_posix()
    summary["output_diagnostics_sha256"] = _sha256(args.output_diagnostics)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
