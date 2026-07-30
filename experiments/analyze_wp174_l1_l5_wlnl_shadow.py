#!/usr/bin/env python3
"""Audit shadow-only L1/L5 WL->NL candidates against post-selection truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _tow(value: str) -> float:
    return round(float(value), 3)


def _finite(value: str | None) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def analyze(
    debug_path: Path, reference_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    truth = {
        _tow(row["GPS TOW (s)"]): (
            float(row["ECEF X (m)"]),
            float(row["ECEF Y (m)"]),
            float(row["ECEF Z (m)"]),
        )
        for row in _read_csv(reference_path)
    }
    debug = _read_csv(debug_path)
    candidates: list[dict[str, Any]] = []
    runtimes: list[float] = []
    attempted = 0
    pair_ge4 = 0
    wl_passed = 0
    nl_passed = 0
    for row in debug:
        attempted += row.get(
            "lambda_l1_l5_wlnl_shadow_attempted"
        ) == "1"
        pair_count = int(
            row.get("lambda_l1_l5_wlnl_shadow_pair_count") or 0
        )
        pair_ge4 += pair_count >= 4
        wl_pass = row.get(
            "lambda_l1_l5_wlnl_shadow_wl_ffrt_passed"
        ) == "1"
        nl_pass = row.get(
            "lambda_l1_l5_wlnl_shadow_nl_ffrt_passed"
        ) == "1"
        wl_passed += wl_pass
        nl_passed += nl_pass
        runtime = _finite(
            row.get("lambda_l1_l5_wlnl_shadow_runtime_ms")
        )
        if runtime is not None:
            runtimes.append(runtime)
        candidate = tuple(
            _finite(
                row.get(
                    f"lambda_l1_l5_wlnl_shadow_best_ecef_{axis}"
                )
            )
            for axis in "xyz"
        )
        if not nl_pass or any(value is None for value in candidate):
            continue
        tow = _tow(row["tow"])
        reference = truth.get(tow)
        error_m = (
            math.dist(
                tuple(float(value) for value in candidate),
                reference,
            )
            if reference is not None
            else None
        )
        candidates.append(
            {
                "tow": tow,
                "pair_count": pair_count,
                "wl_bsr_qscale16": row.get(
                    "lambda_l1_l5_wlnl_shadow_wl_bsr", ""
                ),
                "wl_ratio": row.get(
                    "lambda_l1_l5_wlnl_shadow_wl_ratio", ""
                ),
                "wl_ffrt_min_ratio": row.get(
                    "lambda_l1_l5_wlnl_shadow_wl_ffrt_min_ratio", ""
                ),
                "mw_disagreements": int(
                    row.get(
                        "lambda_l1_l5_wlnl_shadow_mw_disagreements"
                    )
                    or 0
                ),
                "nl_bsr_qscale16": row.get(
                    "lambda_l1_l5_wlnl_shadow_nl_bsr", ""
                ),
                "nl_ratio": row.get(
                    "lambda_l1_l5_wlnl_shadow_nl_ratio", ""
                ),
                "nl_ffrt_min_ratio": row.get(
                    "lambda_l1_l5_wlnl_shadow_nl_ffrt_min_ratio", ""
                ),
                "error_m": error_m if error_m is not None else "",
                "sub50cm": (
                    int(error_m < 0.5) if error_m is not None else ""
                ),
            }
        )

    labeled = [row for row in candidates if row["sub50cm"] != ""]
    errors = [float(row["error_m"]) for row in labeled]
    summary = {
        "schema": "gnss_gpu_wp174_l1_l5_wlnl_shadow_audit_v1",
        "runtime_fgo": False,
        "truth_usage": "post_selection_audit_only",
        "debug_epochs": len(debug),
        "attempted_epochs": attempted,
        "epochs_with_at_least_four_pairs": pair_ge4,
        "wide_lane_ffrt_passed_epochs": wl_passed,
        "narrow_lane_ffrt_passed_epochs": nl_passed,
        "candidate_epochs": len(candidates),
        "truth_labeled_candidate_epochs": len(labeled),
        "sub50cm_candidate_epochs": sum(
            row["sub50cm"] == 1 for row in labeled
        ),
        "not_sub50cm_candidate_epochs": sum(
            row["sub50cm"] == 0 for row in labeled
        ),
        "candidate_error_m": {
            "p50": _quantile(errors, 0.50),
            "p95": _quantile(errors, 0.95),
            "max": max(errors) if errors else None,
        },
        "runtime_ms": {
            "p50": _quantile(runtimes, 0.50),
            "p95": _quantile(runtimes, 0.95),
            "max": max(runtimes) if runtimes else None,
        },
        "promotion_ready": False,
    }
    return candidates, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-epochs", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    epochs, summary = analyze(args.debug, args.reference)
    args.output_epochs.parent.mkdir(parents=True, exist_ok=True)
    if epochs:
        with args.output_epochs.open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(epochs[0]))
            writer.writeheader()
            writer.writerows(epochs)
    else:
        args.output_epochs.write_text("", encoding="utf-8")
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
