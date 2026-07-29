#!/usr/bin/env python3
"""Fail-closed, requirement-by-requirement v0.3 production promotion audit."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "ros2" / "gnss_gpu_ros"))

from experiments.audit_phase6_ros2_soak import evaluate_soak  # noqa: E402
from gnss_gpu.evaluation_contract import verify_locked_contract  # noqa: E402
from gnss_gpu_ros.replay_contract import evaluate_replay  # noqa: E402
from tools.build_release_bundle import build_bundle, verify_bundle  # noqa: E402


CONTRACT_SCHEMA = "gnss_gpu_v030_production_promotion_contract_v1"
RESULT_SCHEMA = "gnss_gpu_v030_production_promotion_audit_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _gate(
    gate_id: str,
    passed: bool,
    *,
    evidence: str,
    actual: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "passed": bool(passed),
        "evidence": evidence,
        "actual": actual,
        "expected": expected,
    }


def audit_promotion(repo_root: Path, contract_path: Path) -> dict[str, Any]:
    contract = _read_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"expected contract schema {CONTRACT_SCHEMA!r}")
    targets = contract["targets"]
    evidence_paths = contract["evidence"]
    evidence = {
        name: _read_json(repo_root / relative)
        for name, relative in evidence_paths.items()
    }
    holdouts = evidence["negative_holdouts"]
    tokyo = evidence["tokyo_production"]
    runtime = evidence["runtime"]
    wp172_runtime = evidence["wp172_runtime"]
    wp173_replay = evidence["wp173_replay"]
    cross_domain = evidence["cross_domain"]
    soak = evidence["ros2_soak"]
    replay_input = evidence["ros2_replay_input"]
    replay_result = evidence["ros2_replay_result"]

    production = tokyo.get("production_effect", tokyo)
    production_contract = _read_json(repo_root / tokyo["contract"])
    truth_usage = tokyo.get("truth_usage")
    after_sub50cm_percent = production.get(
        "after_sub50cm_pct", production.get("after_sub50cm_percent")
    )
    runtime_assessment = runtime["assessment"]
    holdout_results = holdouts["results"]
    cities = sorted(
        {
            city
            for campaign in cross_domain["campaigns"]
            for city in campaign["coverage"]["cities"]
        }
    )
    replay_recomputed = evaluate_replay(replay_input)
    soak_recomputed = evaluate_soak(
        duration_s=soak["simulated_duration_s"],
        tick_s=soak["tick_s"],
    )
    soak_without_measurement = {
        key: value for key, value in soak.items() if key != "measurement"
    }
    immutable = verify_locked_contract(repo_root)
    with tempfile.TemporaryDirectory(prefix="gnss_gpu-promotion-") as temp:
        bundle_path = Path(temp) / "bundle"
        build_bundle(repo_root, bundle_path)
        bundle_verification = verify_bundle(bundle_path)

    gates = [
        _gate(
            "mandatory_negative_holdouts",
            holdouts.get("passed") is True
            and holdouts.get("rejected_holdouts")
            >= targets["mandatory_negative_holdouts"]
            and all(item.get("accepted") is False for item in holdout_results),
            evidence=evidence_paths["negative_holdouts"],
            actual={
                "rejected": holdouts.get("rejected_holdouts"),
                "accepted": sum(bool(item.get("accepted")) for item in holdout_results),
            },
            expected={
                "rejected_min": targets["mandatory_negative_holdouts"],
                "accepted": 0,
            },
        ),
        _gate(
            "truth_free_production_input",
            tokyo.get("production_input_truth") is False
            and truth_usage
            in {
                "post_application_full_denominator_audit_only",
                "post_selection_full_denominator_audit_only",
            },
            evidence=evidence_paths["tokyo_production"],
            actual={
                "production_input_truth": tokyo.get("production_input_truth"),
                "truth_usage": truth_usage,
            },
            expected={
                "production_input_truth": False,
                "truth_usage": "post_selection_or_application_full_denominator_audit_only",
            },
        ),
        _gate(
            "full_denominator_gain_without_loss",
            production["gained_epochs"] >= targets["gained_epochs_min"]
            and production["lost_epochs"] <= targets["lost_epochs_max"],
            evidence=evidence_paths["tokyo_production"],
            actual={
                "gained_epochs": production["gained_epochs"],
                "lost_epochs": production["lost_epochs"],
            },
            expected={
                "gained_epochs_min": targets["gained_epochs_min"],
                "lost_epochs_max": targets["lost_epochs_max"],
            },
        ),
        _gate(
            "false_fix_zero",
            production["false_fix_epochs"] <= targets["false_fix_epochs_max"],
            evidence=evidence_paths["tokyo_production"],
            actual=production["false_fix_epochs"],
            expected=targets["false_fix_epochs_max"],
        ),
        _gate(
            "lambda_fix_coverage",
            production["fix_percent"] >= targets["tokyo_lambda_fix_full_pct_min"]
            and tokyo.get("integer_ambiguity_method", {}).get("implementation")
            == "libgnss++ MLAMBDA"
            and all(
                item.get("accepted") is False and item.get("fix_epochs") == 0
                for item in tokyo["mandatory_negative_holdouts"].values()
            )
            and wp173_replay.get("passed") is True
            and wp173_replay.get("output_is_canonically_identical") is True
            and wp173_replay.get("replayed_output_canonical_sha256")
            == tokyo.get("output_trajectory_canonical_sha256"),
            evidence=(
                f"{evidence_paths['tokyo_production']}; "
                f"{evidence_paths['wp173_replay']}"
            ),
            actual={
                "method": tokyo.get("integer_ambiguity_method", {}).get(
                    "implementation"
                ),
                "fix_epochs": production["fix_epochs"],
                "fix_percent": production["fix_percent"],
                "negative_holdout_fix_epochs": {
                    name: item.get("fix_epochs")
                    for name, item in tokyo["mandatory_negative_holdouts"].items()
                },
                "replay_canonically_identical": wp173_replay.get(
                    "output_is_canonically_identical"
                ),
            },
            expected={
                "method": "libgnss++ MLAMBDA",
                "fix_percent_min": targets["tokyo_lambda_fix_full_pct_min"],
                "negative_holdout_fix_epochs": 0,
                "replay_canonically_identical": True,
            },
        ),
        _gate(
            "tokyo_sub50cm_target",
            after_sub50cm_percent >= targets["tokyo_sub50cm_full_pct_min"],
            evidence=evidence_paths["tokyo_production"],
            actual=after_sub50cm_percent,
            expected=targets["tokyo_sub50cm_full_pct_min"],
        ),
        _gate(
            "runtime_deadlines",
            runtime.get("passed") is True
            and wp172_runtime.get("passed") is True
            and runtime_assessment["normal_latency_max_ms"]
            <= targets["normal_latency_max_ms"]
            and runtime_assessment["search_latency_max_ms"]
            <= targets["search_latency_max_ms"]
            and wp172_runtime["measurement"][
                "conservative_sequential_average_ms_per_epoch"
            ]
            <= targets["wp172_sequential_average_max_ms"]
            and wp172_runtime["reproducibility"][
                "final_trajectory_is_byte_identical"
            ]
            is True
            and runtime_assessment["deadline_misses"] == 0,
            evidence=(
                f"{evidence_paths['runtime']}; "
                f"{evidence_paths['wp172_runtime']}"
            ),
            actual={
                "normal_latency_max_ms": runtime_assessment["normal_latency_max_ms"],
                "search_latency_max_ms": runtime_assessment["search_latency_max_ms"],
                "wp172_sequential_average_ms": wp172_runtime["measurement"][
                    "conservative_sequential_average_ms_per_epoch"
                ],
                "wp172_final_trajectory_byte_identical": wp172_runtime[
                    "reproducibility"
                ]["final_trajectory_is_byte_identical"],
                "deadline_misses": runtime_assessment["deadline_misses"],
            },
            expected={
                "normal_latency_max_ms": targets["normal_latency_max_ms"],
                "search_latency_max_ms": targets["search_latency_max_ms"],
                "wp172_sequential_average_max_ms": targets[
                    "wp172_sequential_average_max_ms"
                ],
                "wp172_final_trajectory_byte_identical": True,
                "deadline_misses": 0,
            },
        ),
        _gate(
            "multi_city_non_degradation",
            cross_domain.get("passed") is True
            and len(cities) >= targets["minimum_cities"]
            and all(campaign.get("passed") is True for campaign in cross_domain["campaigns"]),
            evidence=evidence_paths["cross_domain"],
            actual={"cities": cities, "campaigns": len(cross_domain["campaigns"])},
            expected={"minimum_cities": targets["minimum_cities"], "all_campaigns_pass": True},
        ),
        _gate(
            "m4_and_immutable_contract",
            immutable.get("passed") is True
            and tokyo.get("m4_preserved_sha256")
            == production_contract.get("m4_expected_sha256"),
            evidence=evidence_paths["tokyo_production"],
            actual={
                "immutable_contract": immutable.get("passed"),
                "m4_preserved_sha256": tokyo.get("m4_preserved_sha256"),
            },
            expected=True,
        ),
        _gate(
            "deterministic_ros2_replay",
            replay_recomputed == replay_result,
            evidence=evidence_paths["ros2_replay_result"],
            actual=replay_recomputed["replay_sha256"],
            expected=replay_result["replay_sha256"],
        ),
        _gate(
            "ros2_continuity_soak",
            soak.get("passed") is True
            and soak_recomputed == soak_without_measurement,
            evidence=evidence_paths["ros2_soak"],
            actual={
                "simulated_duration_s": soak["simulated_duration_s"],
                "passed": soak.get("passed"),
                "deterministic": soak_recomputed == soak_without_measurement,
            },
            expected={"simulated_duration_s_min": 7_200.0, "passed": True},
        ),
        _gate(
            "one_command_reproducibility",
            bundle_verification["passed"] is True
            and bundle_verification["file_count"] >= 38,
            evidence="tools/build_release_bundle.py",
            actual=bundle_verification,
            expected={"passed": True, "file_count_min": 38},
        ),
    ]
    failed = [gate["id"] for gate in gates if not gate["passed"]]
    return {
        "schema": RESULT_SCHEMA,
        "contract": contract_path.relative_to(repo_root).as_posix(),
        "gates": gates,
        "passed_gate_count": len(gates) - len(failed),
        "gate_count": len(gates),
        "failed_gates": failed,
        "promotion_allowed": not failed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/v030_production_promotion.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    result = audit_promotion(repo_root, args.contract.resolve())
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0 if result["promotion_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
