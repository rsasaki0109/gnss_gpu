#!/usr/bin/env python3
"""Build and evaluate the final PPC basin PF/FGO promotion evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from gnss_gpu.basin_fgo_promotion import (  # noqa: E402
    SCHEMA,
    evaluate_basin_fgo_promotion,
)
from gnss_gpu.evaluation_contract import build_reproducibility_manifest  # noqa: E402


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_final_evidence(repo_root: Path) -> tuple[dict, dict]:
    tokyo_dir = repo_root / "Testing/basin_fgo_residual_full_tokyo1"
    nagoya_dir = repo_root / "Testing/basin_fgo_residual_full_nagoya1"
    city_audits = {
        "tokyo": _json(tokyo_dir / "safe_union_cv.audit.json"),
        "nagoya": _json(nagoya_dir / "safe_union_cv.audit.json"),
    }
    library_fixed = {"tokyo": 6484, "nagoya": 5274}
    route_summary_path = repo_root / "Testing/basin_fgo_e300_reuse/summary.json"
    route_summary = _json(route_summary_path)
    transfer_dir = repo_root / "Testing/basin_fgo_streak2_gap1_six_e300"
    route_metrics = {}
    manifest_paths: list[Path] = [route_summary_path]
    for route in route_summary["routes"]:
        route_id = route["route"]
        city, run = route_id.split("/")
        audit_path = transfer_dir / f"{city}_{run}.audit.json"
        audit = _json(audit_path)
        route_metrics[route_id] = {
            "latency_p95_ms": route["runtime"]["p95_ms"],
            "false_fix": audit["false_fix"],
            "false_fix_above_1m": audit["false_fix_above_1m"],
        }
        manifest_paths.append(audit_path)

    cv_path = repo_root / "Testing/basin_fgo_safe_union_blocked_cv.json"
    cv = _json(cv_path)
    parity_path = repo_root / "Testing/basin_fgo_cuda_parity/audit.json"
    parity = _json(parity_path)
    manifest_paths.extend((cv_path, parity_path))

    interface_faults = []
    fault_dir = repo_root / "Testing/basin_fgo_faults_e300"
    for fault in ("outage", "ambiguous_holdout", "cycle_slip", "nlos"):
        path = fault_dir / f"{fault}.active.audit.json"
        interface_faults.append(_json(path))
        manifest_paths.append(path)
    raw_faults = []
    for city in ("tokyo", "nagoya"):
        for fault in ("cycle_slip", "nlos", "satellite_loss", "outage"):
            path = (
                repo_root
                / "internal_docs"
                / f"wp176_{city}_surplus_{fault}_fault_v104_2026_07_31.json"
            )
            raw_faults.append(_json(path))
            manifest_paths.append(path)
    fault_false = sum(item["false_fix"] for item in interface_faults) + sum(
        item["false_fixed_epochs"] for item in raw_faults
    )
    raw_fault_above_1m = 0
    for item in raw_faults:
        if "over_1m_false_fixed_epochs" in item:
            raw_fault_above_1m += item["over_1m_false_fixed_epochs"]
        elif item["false_fixed_epochs"] != 0:
            raise ValueError("raw fault evidence lacks >1 m false-FIX accounting")
    fault_above_1m = (
        sum(item["false_fix_above_1m"] for item in interface_faults)
        + raw_fault_above_1m
    )

    for directory in (tokyo_dir, nagoya_dir):
        manifest_paths.extend(
            (
                directory / "safe_union_cv.csv",
                directory / "safe_union_cv.json",
                directory / "safe_union_cv.audit.json",
                directory / "streak2_gap1.tracker.csv",
            )
        )
    city_sources = {
        "tokyo": {
            "directory": tokyo_dir,
            "monitor": repo_root / "results/wp174/tokyo_surplus_rediff_monitor_full_wp176_v96.pos",
            "active": repo_root / "results/wp174/tokyo_surplus_rediff_active_additive_min8_streak1_full_wp176_v104.pos",
            "integrity": repo_root / "results/wp174/tokyo_surplus_rediff_active_additive_min8_streak1_full_wp176_v104_integrity.csv",
        },
        "nagoya": {
            "directory": nagoya_dir,
            "monitor": repo_root / "results/wp174/nagoya_surplus_rediff_monitor_full_wp176_v96.pos",
            "active": repo_root / "results/wp174/nagoya_surplus_rediff_active_additive_min8_streak1_full_wp176_v104.pos",
            "integrity": repo_root / "results/wp174/nagoya_surplus_rediff_active_additive_min8_streak1_full_wp176_v104_integrity.csv",
        },
    }
    for city, sources in city_sources.items():
        route_dir = repo_root / f"datasets/PPC-Dataset-data/{city}/run1"
        manifest_paths.extend(
            (
                sources["monitor"],
                sources["active"],
                sources["integrity"],
                sources["directory"] / f"{city}_run1_basin_fgo_k8_efull.basins.jsonl",
                route_dir / "rover.obs",
                route_dir / "base.obs",
                route_dir / "base.nav",
                route_dir / "imu.csv",
                route_dir / "reference.csv",
            )
        )
    manifest_paths.extend(
        repo_root / relative
        for relative in (
            "python/gnss_gpu/ambiguity_basin_pf.py",
            "python/gnss_gpu/basin_fgo_bridge.py",
            "python/gnss_gpu/basin_fgo_promotion.py",
            "python/gnss_gpu/basin_imu_bridge.py",
            "python/gnss_gpu/basin_ffbsi.py",
            "experiments/run_ppc_basin_fgo_tracker.py",
            "experiments/compose_ppc_safe_basin_union.py",
            "experiments/audit_ppc_safe_basin_union_cv.py",
            "third_party/gnssplusplus/include/libgnss++/algorithms/fgo.hpp",
            "third_party/gnssplusplus/src/algorithms/fgo.cpp",
            "third_party/gnssplusplus/apps/native/gnss_solve.cpp",
            "third_party/gnssplusplus/build-codex-multisd/apps/Release/gnss_solve.exe",
        )
    )
    config = {
        "top_k": 8,
        "fix_min_streak": 2,
        "validation_gap_tolerance_epochs": 1,
        "motion_innovation_limit_m": 0.30,
        "maximum_causal_arc_resets": 2,
        "promotion_streak_epochs": 2,
        "default_enabled": False,
    }
    manifest = build_reproducibility_manifest(
        repo_root=repo_root,
        input_paths=manifest_paths,
        config=config,
        command=["python", "experiments/build_ppc_basin_fgo_final_evidence.py"],
    )
    payload = {
        "schema": SCHEMA,
        "candidate": {
            "id": "ppc-safe-library-basin-pf-fgo-cv-v1",
            "production_input_truth": False,
            "truth_opened_after_estimator_exit": True,
            "estimator_input_kinds": [
                "rover_obs",
                "base_obs",
                "base_nav",
                "ppc_imu",
            ],
            "default_enabled": False,
            "legacy_disabled_parity": True,
            "city_metrics": {
                city: {
                    "correct_fix": audit["correct_fix"],
                    "library_fixed": library_fixed[city],
                    "total_epochs": audit["total_epochs"],
                    "false_fix": audit["false_fix"],
                    "false_fix_above_1m": audit["false_fix_above_1m"],
                }
                for city, audit in city_audits.items()
            },
            "route_metrics": route_metrics,
            "validation": {
                "temporal_blocked_cv": {
                    "passed": cv["passed"],
                    "folds": len(cv["folds"]),
                    "holdout_fixed": cv["holdout_fixed"],
                    "holdout_false": cv["holdout_false"],
                },
                "cross_city_transfer": {
                    "passed": all(
                        metrics["false_fix"] == 0
                        and metrics["false_fix_above_1m"] == 0
                        for metrics in route_metrics.values()
                    )
                },
                "fault_matrix": {
                    "passed": fault_false == 0 and fault_above_1m == 0,
                    "raw_measurement_faults": len(raw_faults),
                    "basin_interface_faults": len(interface_faults),
                    "false_fix": fault_false,
                    "false_fix_above_1m": fault_above_1m,
                },
                "cpu_gpu_parity": {
                    "acceptance_identity": parity["acceptance_identity"],
                    "maximum_ecef_difference_m": parity[
                        "maximum_ecef_difference_m"
                    ],
                },
            },
        },
        "reproducibility_manifest": manifest,
    }
    return payload, evaluate_basin_fgo_promotion(payload, repo_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    payload, result = build_final_evidence(ROOT)
    args.payload.parent.mkdir(parents=True, exist_ok=True)
    args.payload.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["promoted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
