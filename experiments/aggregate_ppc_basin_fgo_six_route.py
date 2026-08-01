#!/usr/bin/env python3
"""Aggregate independently recorded PPC basin-FGO route summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ROUTES = {
    "tokyo/run1",
    "tokyo/run2",
    "tokyo/run3",
    "nagoya/run1",
    "nagoya/run2",
    "nagoya/run3",
}
CONFIG_KEYS = (
    "binary_sha256",
    "max_epochs",
    "skip_epochs",
    "top_k",
    "fix_min_streak",
    "validation_gap_tolerance_epochs",
    "cuda_mode",
    "imu_enabled",
    "native_imu_enabled",
    "native_imu_aperture_m",
    "native_imu_fix_min_streak",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_summaries(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one summary is required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for payload in payloads:
        if payload.get("schema") != "gnss_gpu_ppc_basin_fgo_six_route_v1":
            raise ValueError("invalid route-summary schema")
        if payload.get("production_input_truth") is not False:
            raise ValueError("estimator summary must declare truth-free input")
    config = {key: payloads[0].get(key) for key in CONFIG_KEYS}
    for payload in payloads[1:]:
        mismatch = [key for key, value in config.items() if payload.get(key) != value]
        if mismatch:
            raise ValueError(f"route-summary config mismatch: {', '.join(mismatch)}")

    routes: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for route in payload.get("routes", []):
            name = str(route.get("route", ""))
            if name in routes:
                raise ValueError(f"duplicate route: {name}")
            audit = route.get("audit", {})
            candidate = route.get("candidate_supply_audit", {})
            if audit.get("integrity", {}).get("passed") is not True:
                raise ValueError(f"tracker audit failed integrity: {name}")
            if candidate.get("integrity", {}).get("passed") is not True:
                raise ValueError(f"candidate audit failed integrity: {name}")
            routes[name] = route
    missing = EXPECTED_ROUTES - routes.keys()
    extra = routes.keys() - EXPECTED_ROUTES
    if missing or extra:
        raise ValueError(
            f"route set mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )

    totals = {
        key: sum(int(route["audit"][key]) for route in routes.values())
        for key in (
            "total_epochs",
            "fixed",
            "correct_fix",
            "false_fix",
            "false_fix_above_1m",
        )
    }
    totals["correct_fix_rate_full_denominator"] = (
        totals["correct_fix"] / totals["total_epochs"]
        if totals["total_epochs"]
        else 0.0
    )
    totals["false_per_fixed"] = (
        totals["false_fix"] / totals["fixed"] if totals["fixed"] else 0.0
    )
    candidate_totals = {
        key: sum(int(route["candidate_supply_audit"][key]) for route in routes.values())
        for key in (
            "evaluated_epochs",
            "oracle_correct_epochs",
            "passed_correct_epochs",
            "unique_pass_epochs",
            "unique_pass_correct_epochs",
        )
    }
    return {
        "schema": "gnss_gpu_ppc_basin_fgo_six_route_aggregate_v1",
        "production_input_truth": False,
        "truth_usage": "post_estimator_scoring_only",
        "config": config,
        "route_order": sorted(routes),
        "totals": totals,
        "candidate_supply": candidate_totals,
        "integrity": {
            "complete_six_route_set": True,
            "zero_false_fix": totals["false_fix"] == 0,
            "zero_false_fix_above_1m": totals["false_fix_above_1m"] == 0,
            "passed": (
                totals["false_fix"] == 0
                and totals["false_fix_above_1m"] == 0
            ),
        },
        "inputs": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in paths
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = aggregate_summaries(args.summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
