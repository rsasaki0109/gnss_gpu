#!/usr/bin/env python3
"""Audit causal IMU ranking of LAMBDA top-K shadow candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _candidate(row: pd.Series, ordinal: int) -> tuple[float, tuple[float, ...]]:
    cost = float(row[f"lambda_shadow_candidate_{ordinal}_cost"])
    position = tuple(
        float(row[f"lambda_shadow_candidate_{ordinal}_ecef_{axis}"])
        for axis in "xyz"
    )
    return cost, position


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def analyze(
    debug: pd.DataFrame,
    bridge: pd.DataFrame,
    reference: pd.DataFrame,
    domain: str,
) -> dict[str, Any]:
    reference = reference.rename(
        columns={
            "GPS TOW (s)": "tow",
            "ECEF X (m)": "truth_x",
            "ECEF Y (m)": "truth_y",
            "ECEF Z (m)": "truth_z",
        }
    )
    joined = debug.merge(bridge, on="tow", how="inner").merge(
        reference[["tow", "truth_x", "truth_y", "truth_z"]],
        on="tow",
        how="inner",
    )
    baseline_good = 0
    oracle_good = 0
    candidate_rows = 0
    eligible_rows = 0
    selected_good = 0
    selected_bad = 0
    alternate_selections = 0
    gains = 0
    harms = 0
    for _, row in joined.iterrows():
        candidates: list[tuple[float, tuple[float, ...]]] = []
        for ordinal in range(1, 9):
            try:
                candidate = _candidate(row, ordinal)
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(candidate[0]) and all(
                math.isfinite(value) for value in candidate[1]
            ):
                candidates.append(candidate)
        if not candidates:
            continue
        candidate_rows += 1
        truth = tuple(float(row[f"truth_{axis}"]) for axis in "xyz")
        candidate_good = [
            _distance(position, truth) < 0.5 for _, position in candidates
        ]
        baseline_good += candidate_good[0]
        oracle_good += any(candidate_good)
        selected = 0
        bridge_position = tuple(
            float(row[f"bridge_ecef_{axis}"]) for axis in "xyz"
        )
        eligible = (
            int(row["anchor"]) == 0
            and int(row["initialized"]) == 1
            and int(row["heading_converged"]) == 1
            and 0.0 < float(row["anchor_age_s"]) <= 0.2
            and math.isfinite(float(row["position_sigma_max_m"]))
            and float(row["position_sigma_max_m"]) <= 0.1
            and all(math.isfinite(value) for value in bridge_position)
        )
        if eligible:
            eligible_rows += 1
            best_cost = candidates[0][0]
            sigma = max(1.0, float(row["position_sigma_max_m"]))
            scores = [
                (cost - best_cost)
                + _distance(position, bridge_position) ** 2 / sigma**2
                for cost, position in candidates
            ]
            selected = min(range(len(scores)), key=scores.__getitem__)
        alternate_selections += selected != 0
        selected_is_good = candidate_good[selected]
        selected_good += selected_is_good
        selected_bad += not selected_is_good
        gains += not candidate_good[0] and selected_is_good
        harms += candidate_good[0] and not selected_is_good
    return {
        "domain": domain,
        "joined_epochs": len(joined),
        "topk_candidate_epochs": candidate_rows,
        "imu_eligible_epochs": eligible_rows,
        "baseline_best_sub50cm_epochs": baseline_good,
        "oracle_topk_sub50cm_epochs": oracle_good,
        "imu_ranked_sub50cm_epochs": selected_good,
        "imu_ranked_not_sub50cm_epochs": selected_bad,
        "imu_alternate_selections": alternate_selections,
        "imu_ranking_gains": gains,
        "imu_ranking_harms": harms,
        "pass_observed_no_harm": harms == 0,
        "improves_over_best": gains > harms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        pd.read_csv(args.debug),
        pd.read_csv(args.bridge),
        pd.read_csv(args.reference, skipinitialspace=True),
        args.domain,
    )
    payload = {
        "schema": "gnss_gpu_wp174_imu_topk_shadow_audit_v1",
        "runtime_fgo": False,
        "truth_usage": "post_selection_audit_only",
        "selection_truth_usage": "none",
        "fixed_policy": {
            "candidate_count": 8,
            "maximum_anchor_age_s": 0.2,
            "maximum_reported_position_sigma_m": 0.1,
            "ranking_sigma_floor_m": 1.0,
            "score": "ils_cost_delta + squared_imu_position_distance/sigma^2",
            "anchor_rows_allowed": False,
            "requires_initialized_and_heading_converged": True,
        },
        "result": result,
        "promotion_ready": (
            result["pass_observed_no_harm"]
            and result["improves_over_best"]
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
