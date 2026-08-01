"""Fail-closed promotion contract for the PPC hybrid basin PF/FGO path."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from gnss_gpu.evaluation_contract import verify_reproducibility_manifest


SCHEMA = "gnss_gpu_basin_fgo_promotion_input_v1"
RESULT_SCHEMA = "gnss_gpu_basin_fgo_promotion_result_v1"
EXPECTED_ROUTES = frozenset(
    f"{city}/run{run}" for city in ("tokyo", "nagoya") for run in range(1, 4)
)
ALLOWED_ESTIMATOR_INPUTS = frozenset(
    {"rover_obs", "base_obs", "base_nav", "ppc_imu"}
)
STRETCH_TARGETS = {"tokyo": 0.70, "nagoya": 0.80}


def _gate(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _city_gate(
    city: str, metrics: Mapping[str, Any]
) -> tuple[bool, str, float | None, float | None]:
    correct = metrics.get("correct_fix")
    total = metrics.get("total_epochs")
    library_fixed = metrics.get("library_fixed")
    false = metrics.get("false_fix")
    above_1m = metrics.get("false_fix_above_1m")
    if (
        not isinstance(correct, int)
        or isinstance(correct, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
        or correct < 0
        or correct > total
        or not isinstance(library_fixed, int)
        or isinstance(library_fixed, bool)
        or library_fixed < 0
        or library_fixed > total
    ):
        return (
            False,
            "correct_fix, library_fixed, and positive total_epochs are required",
            None,
            None,
        )
    rate = correct / total
    library_rate = library_fixed / total
    passed = (
        correct > library_fixed
        and false == 0
        and above_1m == 0
    )
    return (
        passed,
        f"rate={rate:.6%} must exceed library={library_rate:.6%}, "
        f"false={false}, >1m={above_1m}",
        rate,
        library_rate,
    )


def evaluate_basin_fgo_promotion(
    payload: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    """Evaluate all scientific, integrity, compatibility, and runtime gates.

    Missing or malformed evidence is a failed gate, never an implicit pass.
    This function consumes completed estimator artifacts and audit summaries;
    it contains no estimator or reference-truth access itself.
    """

    if payload.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r}")
    candidate = payload.get("candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    gates: list[dict[str, Any]] = []

    declared_inputs = candidate.get("estimator_input_kinds")
    input_set = (
        set(declared_inputs)
        if isinstance(declared_inputs, list)
        and all(isinstance(value, str) for value in declared_inputs)
        else set()
    )
    gates.append(
        _gate(
            "ppc_inputs_only",
            bool(input_set) and input_set <= ALLOWED_ESTIMATOR_INPUTS,
            f"declared={sorted(input_set)}, allowed={sorted(ALLOWED_ESTIMATOR_INPUTS)}",
        )
    )
    gates.append(
        _gate(
            "truth_process_boundary",
            candidate.get("production_input_truth") is False
            and candidate.get("truth_opened_after_estimator_exit") is True,
            "truth must be absent from estimator inputs and opened only after estimator exit",
        )
    )
    gates.append(
        _gate(
            "safe_rollout",
            candidate.get("default_enabled") is False
            and candidate.get("legacy_disabled_parity") is True,
            "candidate must remain default-off and disabled mode must match legacy output",
        )
    )

    city_metrics = candidate.get("city_metrics")
    city_metrics = city_metrics if isinstance(city_metrics, Mapping) else {}
    rates: dict[str, float | None] = {}
    target_rates: dict[str, float | None] = {}
    for city in ("tokyo", "nagoya"):
        metrics = city_metrics.get(city)
        metrics = metrics if isinstance(metrics, Mapping) else {}
        passed, detail, rate, target_rate = _city_gate(city, metrics)
        rates[city] = rate
        target_rates[city] = target_rate
        gates.append(_gate(f"city_target:{city}", passed, detail))

    route_metrics = candidate.get("route_metrics")
    route_metrics = route_metrics if isinstance(route_metrics, Mapping) else {}
    route_ids = set(route_metrics)
    route_runtime_ok = route_ids == EXPECTED_ROUTES
    route_integrity_ok = route_ids == EXPECTED_ROUTES
    for route in EXPECTED_ROUTES:
        metrics = route_metrics.get(route)
        if not isinstance(metrics, Mapping):
            route_runtime_ok = False
            route_integrity_ok = False
            continue
        route_runtime_ok &= _finite_nonnegative(metrics.get("latency_p95_ms")) and float(
            metrics.get("latency_p95_ms", math.inf)
        ) <= 100.0
        route_integrity_ok &= (
            metrics.get("false_fix") == 0
            and metrics.get("false_fix_above_1m") == 0
        )
    gates.append(
        _gate(
            "six_route_runtime",
            route_runtime_ok,
            "all Tokyo/Nagoya run1..3 route p95 values must be <=100 ms",
        )
    )
    gates.append(
        _gate(
            "six_route_integrity",
            route_integrity_ok,
            "all six routes must have zero false FIX and zero >1 m false FIX",
        )
    )

    validation = candidate.get("validation")
    validation = validation if isinstance(validation, Mapping) else {}
    for name in ("temporal_blocked_cv", "cross_city_transfer", "fault_matrix"):
        evidence = validation.get(name)
        passed = isinstance(evidence, Mapping) and evidence.get("passed") is True
        if name == "fault_matrix" and isinstance(evidence, Mapping):
            passed = passed and evidence.get("false_fix") == 0 and evidence.get(
                "false_fix_above_1m"
            ) == 0
        gates.append(_gate(name, passed, f"{name} must provide explicit passing evidence"))

    parity = validation.get("cpu_gpu_parity")
    parity = parity if isinstance(parity, Mapping) else {}
    max_delta = parity.get("maximum_ecef_difference_m")
    gates.append(
        _gate(
            "cpu_gpu_parity",
            parity.get("acceptance_identity") is True
            and _finite_nonnegative(max_delta)
            and float(max_delta) <= 1.0e-5,
            "acceptance must be identical and ECEF difference <=10 micrometres",
        )
    )

    manifest = payload.get("reproducibility_manifest")
    manifest_result = (
        verify_reproducibility_manifest(manifest, repo_root)
        if isinstance(manifest, Mapping)
        else {"passed": False, "content_hash_passed": False, "inputs": []}
    )
    gates.append(
        _gate(
            "reproducibility_manifest",
            manifest_result["passed"],
            "manifest content and every estimator input hash must verify",
        )
    )

    return {
        "schema": RESULT_SCHEMA,
        "candidate_id": candidate.get("id"),
        "promoted": all(gate["passed"] for gate in gates),
        "stretch_achieved": all(
            rates[city] is not None and rates[city] >= STRETCH_TARGETS[city]
            for city in STRETCH_TARGETS
        ),
        "rates": rates,
        "targets": target_rates,
        "stretch_targets": STRETCH_TARGETS,
        "gates": gates,
        "manifest_verification": manifest_result,
    }
