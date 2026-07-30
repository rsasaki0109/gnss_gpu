#!/usr/bin/env python3
"""Audit a top-2 BIE-style soft Lambda shadow without FIX authority."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_wp174_ffrt_calibration import BSR_FIELDS  # noqa: E402
from experiments.analyze_wp174_topk_shadow import (  # noqa: E402
    _finite,
    _quantile,
)
from experiments.wp174_ffrt import passes_ffrt  # noqa: E402


@dataclass(frozen=True)
class SoftGuardPolicy:
    covariance_scale: int
    minimum_pairs: int
    maximum_nis_per_observation: float
    maximum_soft_position_std_m: float


def bie_top2(
    best_cost: float,
    second_cost: float,
    best_position: tuple[float, float, float],
    second_position: tuple[float, float, float],
    covariance_temperature: float,
) -> tuple[tuple[float, float, float], float]:
    if (
        not math.isfinite(covariance_temperature)
        or covariance_temperature <= 0.0
        or second_cost < best_cost
        or not all(
            math.isfinite(value)
            for value in (
                best_cost,
                second_cost,
                *best_position,
                *second_position,
            )
        )
    ):
        raise ValueError("invalid top-2 BIE inputs")
    second_weight = math.exp(
        max(-745.0, -0.5 * (second_cost - best_cost) / covariance_temperature)
    )
    denominator = 1.0 + second_weight
    position = tuple(
        (best_position[index] + second_weight * second_position[index])
        / denominator
        for index in range(3)
    )
    separation = math.dist(best_position, second_position)
    position_std = (
        separation * math.sqrt(second_weight) / denominator
    )
    return position, position_std


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _specifications(values: list[str]) -> dict[str, Path]:
    return {
        domain: Path(raw_path)
        for domain, raw_path in (
            specification.split("=", 1) for specification in values
        )
    }


def _joined_rows(
    domain: str,
    audit_path: Path,
    debug_path: Path,
    reference_path: Path,
) -> list[dict[str, Any]]:
    audit = {round(float(row["tow"]), 3): row for row in _read(audit_path)}
    debug = {round(float(row["tow"]), 3): row for row in _read(debug_path)}
    truth = {
        round(float(row["GPS TOW (s)"]), 3): tuple(
            float(row[f"ECEF {axis} (m)"]) for axis in "XYZ"
        )
        for row in _read(reference_path)
    }
    output = []
    for tow in sorted(set(audit) & set(debug) & set(truth)):
        telemetry = debug[tow]
        best_cost = _finite(telemetry.get("lambda_shadow_best_cost"))
        second_cost = _finite(telemetry.get("lambda_shadow_second_cost"))
        pair_count = _finite(telemetry.get("pair_count"))
        nis = _finite(
            telemetry.get("float_update_nis_per_observation")
        )
        best = tuple(
            _finite(telemetry.get(f"lambda_shadow_best_ecef_{axis}"))
            for axis in "xyz"
        )
        second = tuple(
            _finite(telemetry.get(f"lambda_shadow_second_ecef_{axis}"))
            for axis in "xyz"
        )
        if (
            best_cost is None
            or second_cost is None
            or best_cost <= 0.0
            or pair_count is None
            or nis is None
            or any(value is None for value in (*best, *second))
        ):
            continue
        best_position = tuple(float(value) for value in best)
        second_position = tuple(float(value) for value in second)
        soft_positions = {}
        soft_stds = {}
        for temperature in (1.0, 16.0):
            position, position_std = bie_top2(
                best_cost,
                second_cost,
                best_position,
                second_position,
                temperature,
            )
            soft_positions[temperature] = position
            soft_stds[temperature] = position_std
        output.append(
            {
                "domain": domain,
                "block": int(audit[tow]["block"]),
                "pair_count": pair_count,
                "nis": nis,
                "ratio": second_cost / best_cost,
                "best_good": math.dist(best_position, truth[tow]) < 0.5,
                "soft1_good": (
                    math.dist(soft_positions[1.0], truth[tow]) < 0.5
                ),
                "soft16_good": (
                    math.dist(soft_positions[16.0], truth[tow]) < 0.5
                ),
                "soft1_std": soft_stds[1.0],
                "soft16_std": soft_stds[16.0],
                **{
                    f"bsr{scale}": _finite(telemetry.get(field))
                    for scale, field in BSR_FIELDS.items()
                },
            }
        )
    return output


def _temperature_metrics(
    rows: list[dict[str, Any]],
    temperature: int,
) -> dict[str, Any]:
    label = f"soft{temperature}_good"
    std_label = f"soft{temperature}_std"
    standard_deviations = [float(row[std_label]) for row in rows]
    return {
        "truth_labeled_epochs": len(rows),
        "soft_sub50cm_epochs": sum(bool(row[label]) for row in rows),
        "best_sub50cm_epochs": sum(bool(row["best_good"]) for row in rows),
        "rescued_epochs": sum(
            bool(row[label]) and not bool(row["best_good"]) for row in rows
        ),
        "harmed_epochs": sum(
            bool(row["best_good"]) and not bool(row[label]) for row in rows
        ),
        "soft_position_std_p50_m": _quantile(
            standard_deviations, 0.50
        ),
        "soft_position_std_p95_m": _quantile(
            standard_deviations, 0.95
        ),
    }


def _blocked_cv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policies = [
        SoftGuardPolicy(scale, pairs, nis, position_std)
        for scale in BSR_FIELDS
        for pairs in (4, 6, 8, 10, 12, 16)
        for nis in (1.0, 2.0, 3.0, 5.0, math.inf)
        for position_std in (
            0.001,
            0.003,
            0.01,
            0.03,
            0.05,
            0.10,
            0.25,
            math.inf,
        )
    ]
    pair_count = np.array([row["pair_count"] for row in rows])
    nis = np.array([row["nis"] for row in rows])
    ratio = np.array([row["ratio"] for row in rows])
    position_std = np.array([row["soft16_std"] for row in rows])
    good = np.array([row["soft16_good"] for row in rows], dtype=bool)
    domains = np.array([row["domain"] for row in rows])
    blocks = np.array([row["block"] for row in rows])
    ffrt = {}
    for scale in BSR_FIELDS:
        bsr = np.array(
            [
                (
                    float(row[f"bsr{scale}"])
                    if row[f"bsr{scale}"] is not None
                    else math.nan
                )
                for row in rows
            ]
        )
        ffrt[scale] = np.array(
            [
                math.isfinite(bsr[index])
                and passes_ffrt(
                    int(pair_count[index]),
                    bsr[index],
                    ratio[index],
                )
                for index in range(len(rows))
            ],
            dtype=bool,
        )
    accepted = np.stack(
        [
            ffrt[policy.covariance_scale]
            & (pair_count >= policy.minimum_pairs)
            & (nis <= policy.maximum_nis_per_observation)
            & (position_std <= policy.maximum_soft_position_std_m)
            for policy in policies
        ]
    )
    folds = sorted(set(zip(domains.tolist(), blocks.tolist())))
    fold_results = []
    total_good = 0
    total_bad = 0
    selected_fold_count = 0
    for domain, block in folds:
        test = (domains == domain) & (blocks == block)
        train = ~(
            (domains == domain) & (np.abs(blocks - int(block)) <= 1)
        )
        train_good = (accepted[:, train] & good[train]).sum(axis=1)
        train_bad = (accepted[:, train] & ~good[train]).sum(axis=1)
        zero_bad = np.flatnonzero(train_bad == 0)
        selected_index = (
            int(zero_bad[np.argmax(train_good[zero_bad])])
            if len(zero_bad)
            else None
        )
        selected = (
            policies[selected_index] if selected_index is not None else None
        )
        test_good = (
            int((accepted[selected_index] & test & good).sum())
            if selected_index is not None
            else 0
        )
        test_bad = (
            int((accepted[selected_index] & test & ~good).sum())
            if selected_index is not None
            else 0
        )
        selected_fold_count += selected is not None
        total_good += test_good
        total_bad += test_bad
        fold_results.append(
            {
                "test_domain": domain,
                "test_block": int(block),
                "selected_policy": asdict(selected) if selected else None,
                "test_good_epochs": test_good,
                "test_bad_epochs": test_bad,
            }
        )
    return {
        "candidate_policy_count": len(policies),
        "selected_fold_count": selected_fold_count,
        "out_of_fold_good_epochs": total_good,
        "out_of_fold_bad_epochs": total_bad,
        "folds": fold_results,
    }


def analyze(
    audits: dict[str, Path],
    debug_logs: dict[str, Path],
    references: dict[str, Path],
) -> dict[str, Any]:
    domains = sorted(set(audits) & set(debug_logs) & set(references))
    rows = [
        row
        for domain in domains
        for row in _joined_rows(
            domain,
            audits[domain],
            debug_logs[domain],
            references[domain],
        )
    ]
    return {
        "schema": "gnss_gpu_wp174_bie_soft_shadow_audit_v1",
        "runtime_fgo": False,
        "runtime_integration": False,
        "fix_authority": False,
        "truth_usage": "post_selection_audit_only",
        "selection_status": "rejected",
        "rejection_reason": (
            "top-2 soft position does not improve accuracy and its posterior "
            "spread yields no zero-training-error blocked-CV policy"
        ),
        "domains": {
            domain: {
                "temperature1": _temperature_metrics(
                    [row for row in rows if row["domain"] == domain],
                    1,
                ),
                "temperature16": _temperature_metrics(
                    [row for row in rows if row["domain"] == domain],
                    16,
                ),
            }
            for domain in domains
        },
        "soft_spread_blocked_cv": _blocked_cv(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", required=True)
    parser.add_argument("--debug", action="append", required=True)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        _specifications(args.audit),
        _specifications(args.debug),
        _specifications(args.reference),
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
