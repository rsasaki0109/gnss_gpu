from __future__ import annotations

from pathlib import Path

from gnss_gpu.basin_fgo_promotion import (
    EXPECTED_ROUTES,
    SCHEMA,
    evaluate_basin_fgo_promotion,
)
from gnss_gpu.evaluation_contract import build_reproducibility_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _payload(tmp_path: Path) -> dict:
    input_path = tmp_path / "rover.obs"
    input_path.write_text("measurement\n", encoding="utf-8")
    manifest = build_reproducibility_manifest(
        repo_root=REPO_ROOT,
        input_paths=[input_path],
        config={"basins": 64, "top_k": 4},
        command=["gnss_solve", "--basin-fgo-shadow"],
    )
    return {
        "schema": SCHEMA,
        "candidate": {
            "id": "hybrid-basin-fgo",
            "production_input_truth": False,
            "truth_opened_after_estimator_exit": True,
            "estimator_input_kinds": ["rover_obs", "base_obs", "base_nav", "ppc_imu"],
            "default_enabled": False,
            "legacy_disabled_parity": True,
            "city_metrics": {
                "tokyo": {
                    "correct_fix": 60,
                    "library_fixed": 59,
                    "total_epochs": 100,
                    "false_fix": 0,
                    "false_fix_above_1m": 0,
                },
                "nagoya": {
                    "correct_fix": 75,
                    "library_fixed": 74,
                    "total_epochs": 100,
                    "false_fix": 0,
                    "false_fix_above_1m": 0,
                },
            },
            "route_metrics": {
                route: {
                    "latency_p95_ms": 50.0,
                    "false_fix": 0,
                    "false_fix_above_1m": 0,
                }
                for route in EXPECTED_ROUTES
            },
            "validation": {
                "temporal_blocked_cv": {"passed": True},
                "cross_city_transfer": {"passed": True},
                "fault_matrix": {
                    "passed": True,
                    "false_fix": 0,
                    "false_fix_above_1m": 0,
                },
                "cpu_gpu_parity": {
                    "acceptance_identity": True,
                    "maximum_ecef_difference_m": 1.0e-6,
                },
            },
        },
        "reproducibility_manifest": manifest,
    }


def test_complete_candidate_passes_all_promotion_gates(tmp_path: Path) -> None:
    result = evaluate_basin_fgo_promotion(_payload(tmp_path), REPO_ROOT)
    assert result["promoted"] is True
    assert result["stretch_achieved"] is False
    assert all(gate["passed"] for gate in result["gates"])


def test_truth_or_extra_sensor_input_fails_closed(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["candidate"]["production_input_truth"] = True
    payload["candidate"]["estimator_input_kinds"].append("camera")
    result = evaluate_basin_fgo_promotion(payload, REPO_ROOT)
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert {"ppc_inputs_only", "truth_process_boundary"} <= failed
    assert result["promoted"] is False


def test_candidate_must_strictly_exceed_library_fix_count(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["candidate"]["city_metrics"]["tokyo"]["correct_fix"] = 59
    result = evaluate_basin_fgo_promotion(payload, REPO_ROOT)
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert "city_target:tokyo" in failed


def test_any_false_fix_or_missing_route_fails_closed(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["candidate"]["city_metrics"]["tokyo"]["false_fix"] = 1
    payload["candidate"]["route_metrics"].pop("tokyo/run3")
    result = evaluate_basin_fgo_promotion(payload, REPO_ROOT)
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert "city_target:tokyo" in failed
    assert "six_route_runtime" in failed
    assert "six_route_integrity" in failed


def test_runtime_parity_and_fault_evidence_are_mandatory(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["candidate"]["route_metrics"]["nagoya/run2"]["latency_p95_ms"] = 100.1
    payload["candidate"]["validation"]["cpu_gpu_parity"][
        "maximum_ecef_difference_m"
    ] = 2.0e-5
    payload["candidate"]["validation"]["fault_matrix"]["false_fix_above_1m"] = 1
    result = evaluate_basin_fgo_promotion(payload, REPO_ROOT)
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert {"six_route_runtime", "cpu_gpu_parity", "fault_matrix"} <= failed
