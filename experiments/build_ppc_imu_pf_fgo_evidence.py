#!/usr/bin/env python3
"""Build reproducible promotion evidence for the PPC native IMU PF/FGO path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


EXPECTED_ROUTES = {
    "nagoya_run1",
    "nagoya_run2",
    "nagoya_run3",
    "tokyo_run1",
    "tokyo_run2",
    "tokyo_run3",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracker(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {int(row["epoch_index"]): row for row in csv.DictReader(stream)}


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _route_evidence(name: str, directory: Path) -> dict:
    paths = {
        "gnss_audit": directory / "gnss_only.audit.json",
        "imu_audit": directory / "imu.audit.json",
        "gnss_summary": directory / "gnss_only.tracker.json",
        "imu_summary": directory / "imu.tracker.json",
        "gnss_tracker": directory / "gnss_only.tracker.csv",
        "imu_tracker": directory / "imu.tracker.csv",
        "shadow": directory / "full.shadow.csv",
        "safe_output": directory / "safe_output.csv",
        "safe_output_summary": directory / "safe_output.json",
        "safe_output_audit": directory / "safe_output.audit.json",
    }
    gnss_audit = _json(paths["gnss_audit"])
    imu_audit = _json(paths["imu_audit"])
    gnss_summary = _json(paths["gnss_summary"])
    imu_summary = _json(paths["imu_summary"])
    safe_summary = _json(paths["safe_output_summary"])
    safe_audit = _json(paths["safe_output_audit"])
    gnss_rows = _tracker(paths["gnss_tracker"])
    imu_rows = _tracker(paths["imu_tracker"])
    if set(gnss_rows) != set(imu_rows):
        raise ValueError(f"{name}: tracker epoch sets differ")

    both = candidate_only = baseline_only = 0
    for epoch, imu_row in imu_rows.items():
        gnss_fixed = gnss_rows[epoch]["shadow_fixed"] == "1"
        imu_fixed = imu_row["shadow_fixed"] == "1"
        both += int(gnss_fixed and imu_fixed)
        candidate_only += int(imu_fixed and not gnss_fixed)
        baseline_only += int(gnss_fixed and not imu_fixed)

    recovery_epochs: set[int] = set()
    runtimes: list[float] = []
    with paths["shadow"].open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            epoch = int(row["epoch_index"])
            if int(row["imu_fgo_recovery_epochs"]) > 0:
                recovery_epochs.add(epoch)
            value = float(row["imu_fgo_runtime_ms"])
            if math.isfinite(value):
                runtimes.append(value)
    recovery_available = sum(
        imu_rows[epoch]["native_imu_fgo_available"] == "1"
        for epoch in recovery_epochs
        if epoch in imu_rows
    )
    recovery_fixed = sum(
        imu_rows[epoch]["shadow_fixed"] == "1"
        for epoch in recovery_epochs
        if epoch in imu_rows
    )
    config_ok = (
        gnss_summary["config"]["native_imu_fgo"] is False
        and imu_summary["config"]["native_imu_fgo"] is True
        and imu_summary["config"]["native_imu_aperture_m"] == 0.3
        and imu_summary["config"]["native_imu_fix_min_streak"] == 2
    )
    candidate_safe = (
        imu_audit["false_fix"] == 0
        and imu_audit["false_fix_above_1m"] == 0
        and imu_audit["integrity"]["passed"] is True
        and imu_audit["truth_usage"] == "post_estimator_scoring_only"
    )
    union = imu_audit["baseline_priority_union"]
    union_rescue_safe = (
        union["tracker_rescue_false_fix"] == 0
        and union["tracker_rescue_false_fix_above_1m"] == 0
    )
    safe_output_ok = (
        safe_summary["fix_authority"] == "imu_pf_fgo_tracker_only"
        and safe_summary["legacy_fixed_status_inherited"] is False
        and safe_audit["fixed"] == imu_audit["fixed"]
        and safe_audit["correct_fix"] == imu_audit["correct_fix"]
        and safe_audit["false_fix"] == 0
        and safe_audit["false_fix_above_1m"] == 0
        and safe_audit["integrity"]["passed"] is True
    )
    p95_runtime = _quantile(runtimes, 0.95)
    passed = (
        config_ok
        and candidate_safe
        and union_rescue_safe
        and safe_output_ok
        and candidate_only > 0
        and baseline_only == 0
        and recovery_available == 0
        and recovery_fixed == 0
        and p95_runtime is not None
        and p95_runtime <= 200.0
    )
    return {
        "route": name,
        "denominator_epochs": imu_audit["total_epochs"],
        "gnss_only_correct_fix": gnss_audit["correct_fix"],
        "imu_correct_fix": imu_audit["correct_fix"],
        "correct_fix_delta": imu_audit["correct_fix"] - gnss_audit["correct_fix"],
        "imu_false_fix": imu_audit["false_fix"],
        "imu_false_fix_above_1m": imu_audit["false_fix_above_1m"],
        "paired": {
            "both_fixed": both,
            "imu_only_fixed": candidate_only,
            "gnss_only_fixed": baseline_only,
        },
        "baseline_priority_union": {
            "gnss_only_correct_fix": gnss_audit["baseline_priority_union"][
                "correct_fix"
            ],
            "imu_correct_fix": union["correct_fix"],
            "correct_fix_delta": union["correct_fix"]
            - gnss_audit["baseline_priority_union"]["correct_fix"],
            "inherited_baseline_false_fix": union["baseline_false_fix"],
            "inherited_baseline_false_fix_above_1m": union[
                "baseline_false_fix_above_1m"
            ],
            "imu_rescue_false_fix": union["tracker_rescue_false_fix"],
            "imu_rescue_false_fix_above_1m": union[
                "tracker_rescue_false_fix_above_1m"
            ],
        },
        "recovery_fail_closed": {
            "recovery_rows": len(recovery_epochs),
            "pf_available_rows": recovery_available,
            "fixed_rows": recovery_fixed,
            "passed": recovery_available == 0 and recovery_fixed == 0,
        },
        "authoritative_safe_output": {
            "fix_authority": safe_summary["fix_authority"],
            "legacy_fixed_status_inherited": safe_summary[
                "legacy_fixed_status_inherited"
            ],
            "fixed": safe_audit["fixed"],
            "correct_fix": safe_audit["correct_fix"],
            "false_fix": safe_audit["false_fix"],
            "false_fix_above_1m": safe_audit["false_fix_above_1m"],
            "passed": safe_output_ok,
        },
        "runtime": {
            "imu_fgo_p95_ms": p95_runtime,
            "budget_ms": 200.0,
            "passed": p95_runtime is not None and p95_runtime <= 200.0,
        },
        "config_ok": config_ok,
        "passed": passed,
        "artifacts": {key: _sha256(path) for key, path in paths.items()},
    }


def build_evidence(
    routes: dict[str, Path],
    blocked_pairs: dict[str, tuple[Path, Path]],
    fault_audits: list[Path],
    parity_path: Path,
    health_path: Path,
) -> dict:
    if set(routes) != EXPECTED_ROUTES:
        raise ValueError("routes must be the six Tokyo/Nagoya PPC routes")
    route_results = {
        name: _route_evidence(name, path) for name, path in sorted(routes.items())
    }
    holdouts = {}
    for name, (gnss_path, imu_path) in sorted(blocked_pairs.items()):
        gnss = _json(gnss_path)
        imu = _json(imu_path)
        holdouts[name] = {
            "gnss_only_correct_fix": gnss["correct_fix"],
            "imu_correct_fix": imu["correct_fix"],
            "correct_fix_delta": imu["correct_fix"] - gnss["correct_fix"],
            "false_fix": imu["false_fix"],
            "false_fix_above_1m": imu["false_fix_above_1m"],
            "passed": (
                imu["correct_fix"] >= gnss["correct_fix"]
                and imu["false_fix"] == 0
                and imu["false_fix_above_1m"] == 0
            ),
            "artifacts": {
                "gnss_audit_sha256": _sha256(gnss_path),
                "imu_audit_sha256": _sha256(imu_path),
            },
        }
    faults = []
    for path in fault_audits:
        audit = _json(path)
        faults.append(
            {
                "name": path.stem,
                "fixed": audit["fixed"],
                "correct_fix": audit["correct_fix"],
                "false_fix": audit["false_fix"],
                "false_fix_above_1m": audit["false_fix_above_1m"],
                "passed": audit["false_fix"] == 0
                and audit["false_fix_above_1m"] == 0,
                "sha256": _sha256(path),
            }
        )
    parity = _json(parity_path)
    health = _json(health_path)

    totals = {
        "denominator_epochs": sum(
            item["denominator_epochs"] for item in route_results.values()
        ),
        "gnss_only_correct_fix": sum(
            item["gnss_only_correct_fix"] for item in route_results.values()
        ),
        "imu_correct_fix": sum(
            item["imu_correct_fix"] for item in route_results.values()
        ),
        "imu_false_fix": sum(
            item["imu_false_fix"] for item in route_results.values()
        ),
        "imu_false_fix_above_1m": sum(
            item["imu_false_fix_above_1m"] for item in route_results.values()
        ),
        "imu_only_fixed": sum(
            item["paired"]["imu_only_fixed"] for item in route_results.values()
        ),
        "gnss_only_fixed": sum(
            item["paired"]["gnss_only_fixed"] for item in route_results.values()
        ),
        "union_correct_delta": sum(
            item["baseline_priority_union"]["correct_fix_delta"]
            for item in route_results.values()
        ),
        "inherited_union_false_fix": sum(
            item["baseline_priority_union"]["inherited_baseline_false_fix"]
            for item in route_results.values()
        ),
        "inherited_union_false_fix_above_1m": sum(
            item["baseline_priority_union"]["inherited_baseline_false_fix_above_1m"]
            for item in route_results.values()
        ),
    }
    totals["correct_fix_delta"] = (
        totals["imu_correct_fix"] - totals["gnss_only_correct_fix"]
    )
    totals["gnss_only_fix_rate"] = (
        totals["gnss_only_correct_fix"] / totals["denominator_epochs"]
    )
    totals["imu_fix_rate"] = totals["imu_correct_fix"] / totals["denominator_epochs"]
    totals["relative_correct_fix_gain"] = (
        totals["correct_fix_delta"] / totals["gnss_only_correct_fix"]
    )
    improved_routes = sum(item["correct_fix_delta"] > 0 for item in route_results.values())
    statistics_result = {
        "improved_routes": improved_routes,
        "route_sign_test_one_sided_p": 0.5**improved_routes,
        "paired_epoch_exact_log10_p_upper_bound": -totals["imu_only_fixed"]
        * math.log10(2.0),
        "note": "route sign test is the conservative unit; epoch decisions are correlated",
    }
    route_pass = all(item["passed"] for item in route_results.values())
    blocked_pass = len(holdouts) >= 2 and all(
        item["passed"] for item in holdouts.values()
    )
    fault_pass = bool(faults) and all(item["passed"] for item in faults)
    parity_pass = (
        parity["passed"] is True
        and parity["acceptance_identity"] is True
        and parity["integer_identity"] is True
    )
    result = {
        "schema": "gnss_gpu_ppc_imu_pf_fgo_promotion_evidence_v1",
        "truth_contract": {
            "production_input_truth": False,
            "truth_usage": "post_estimator_scoring_only",
        },
        "candidate": {
            "native_imu_fgo": True,
            "native_imu_aperture_m": 0.3,
            "native_imu_fix_min_streak": 2,
            "default_enabled": False,
            "gpu_policy": "auto; forced CUDA is parity-safe but slower at this state size",
        },
        "routes": route_results,
        "totals": totals,
        "statistics": statistics_result,
        "blocked_holdouts": holdouts,
        "fault_audits": faults,
        "cpu_gpu_parity": parity,
        "health_monitor": {
            "truth_usage": health["truth_usage"],
            "telemetry_only": health["provisional_monitor"]["estimator_action"]
            == "telemetry_only",
            "sha256": _sha256(health_path),
        },
        "gates": {
            "six_route": route_pass,
            "blocked_holdouts": blocked_pass,
            "faults": fault_pass,
            "cpu_gpu_parity": parity_pass,
            "meaningful_gain": totals["relative_correct_fix_gain"] >= 0.01,
            "false_fix_zero": totals["imu_false_fix"] == 0,
            "false_fix_above_1m_zero": totals["imu_false_fix_above_1m"] == 0,
        },
    }
    result["component_promotion_ready"] = all(result["gates"].values())
    result["default_candidate_ready"] = result["component_promotion_ready"]
    result["default_promotion_ready"] = result["component_promotion_ready"] and all(
        item["authoritative_safe_output"]["passed"]
        for item in route_results.values()
    )
    result["default_enable_policy"] = (
        "ready candidate, intentionally disabled until an explicit release change; "
        "use authoritative safe output, never baseline-priority union"
    )
    result["artifacts"] = {
        "parity_sha256": _sha256(parity_path),
        "health_sha256": _sha256(health_path),
    }
    return result


def _named_path(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError("expected NAME=PATH")
    name, raw_path = item.split("=", 1)
    return name, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", action="append", required=True)
    parser.add_argument(
        "--blocked-pair", action="append", default=[], metavar="NAME=GNSS,IMU"
    )
    parser.add_argument("--fault-audit", action="append", type=Path, default=[])
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        routes = dict(_named_path(item) for item in args.route)
        blocked_pairs = {}
        for item in args.blocked_pair:
            name, raw_paths = item.split("=", 1)
            gnss_path, imu_path = raw_paths.split(",", 1)
            blocked_pairs[name] = (Path(gnss_path), Path(imu_path))
        result = build_evidence(
            routes, blocked_pairs, args.fault_audit, args.parity, args.health
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["component_promotion_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
