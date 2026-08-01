from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.build_ppc_imu_pf_fgo_evidence import (
    EXPECTED_ROUTES,
    build_evidence,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _audit(correct: int) -> dict:
    return {
        "truth_usage": "post_estimator_scoring_only",
        "total_epochs": 3,
        "fixed": correct,
        "correct_fix": correct,
        "false_fix": 0,
        "false_fix_above_1m": 0,
        "integrity": {"passed": True},
        "baseline_priority_union": {
            "correct_fix": correct + 1,
            "baseline_false_fix": 2,
            "baseline_false_fix_above_1m": 1,
            "tracker_rescue_false_fix": 0,
            "tracker_rescue_false_fix_above_1m": 0,
        },
    }


def _write_tracker(path: Path, fixed: list[int], *, native_available: bool) -> None:
    fields = ("epoch_index", "shadow_fixed", "native_imu_fgo_available")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(3):
            writer.writerow(
                {
                    "epoch_index": epoch,
                    "shadow_fixed": int(epoch in fixed),
                    "native_imu_fgo_available": int(native_available),
                }
            )


def _write_shadow(path: Path) -> None:
    fields = ("epoch_index", "imu_fgo_recovery_epochs", "imu_fgo_runtime_ms")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(3):
            writer.writerow(
                {
                    "epoch_index": epoch,
                    "imu_fgo_recovery_epochs": 0,
                    "imu_fgo_runtime_ms": 10.0 + epoch,
                }
            )


def _route(path: Path) -> None:
    path.mkdir()
    _write_json(path / "gnss_only.audit.json", _audit(1))
    _write_json(path / "imu.audit.json", _audit(2))
    _write_json(
        path / "gnss_only.tracker.json",
        {"config": {"native_imu_fgo": False}},
    )
    _write_json(
        path / "imu.tracker.json",
        {
            "config": {
                "native_imu_fgo": True,
                "native_imu_aperture_m": 0.3,
                "native_imu_fix_min_streak": 2,
            }
        },
    )
    _write_tracker(path / "gnss_only.tracker.csv", [0], native_available=False)
    _write_tracker(path / "imu.tracker.csv", [0, 1], native_available=True)
    _write_shadow(path / "full.shadow.csv")
    _write_tracker(path / "safe_output.csv", [0, 1], native_available=False)
    _write_json(
        path / "safe_output.json",
        {
            "fix_authority": "imu_pf_fgo_tracker_only",
            "legacy_fixed_status_inherited": False,
        },
    )
    _write_json(path / "safe_output.audit.json", _audit(2))


def test_build_evidence_requires_all_gates_and_reports_gain(tmp_path: Path) -> None:
    routes = {}
    for name in EXPECTED_ROUTES:
        route_path = tmp_path / name
        _route(route_path)
        routes[name] = route_path

    gnss_holdout = tmp_path / "holdout.gnss.json"
    imu_holdout = tmp_path / "holdout.imu.json"
    _write_json(gnss_holdout, _audit(1))
    _write_json(imu_holdout, _audit(2))
    parity = tmp_path / "parity.json"
    _write_json(
        parity,
        {"passed": True, "acceptance_identity": True, "integer_identity": True},
    )
    health = tmp_path / "health.json"
    _write_json(
        health,
        {
            "truth_usage": "none",
            "provisional_monitor": {"estimator_action": "telemetry_only"},
        },
    )
    fault = tmp_path / "fault.json"
    _write_json(fault, _audit(2))

    result = build_evidence(
        routes,
        {"a": (gnss_holdout, imu_holdout), "b": (gnss_holdout, imu_holdout)},
        [fault],
        parity,
        health,
    )
    assert result["component_promotion_ready"] is True
    assert result["default_candidate_ready"] is True
    assert result["default_promotion_ready"] is True
    assert result["totals"]["correct_fix_delta"] == 6
    assert result["totals"]["imu_false_fix"] == 0
    assert result["statistics"]["route_sign_test_one_sided_p"] == 0.015625


def test_build_evidence_fails_if_imu_loses_a_gnss_fix(tmp_path: Path) -> None:
    routes = {}
    for name in EXPECTED_ROUTES:
        route_path = tmp_path / name
        _route(route_path)
        routes[name] = route_path
    bad = routes["tokyo_run1"]
    _write_tracker(bad / "gnss_only.tracker.csv", [0, 2], native_available=False)

    holdout = tmp_path / "holdout.json"
    _write_json(holdout, _audit(1))
    parity = tmp_path / "parity.json"
    _write_json(
        parity,
        {"passed": True, "acceptance_identity": True, "integer_identity": True},
    )
    health = tmp_path / "health.json"
    _write_json(
        health,
        {
            "truth_usage": "none",
            "provisional_monitor": {"estimator_action": "telemetry_only"},
        },
    )
    fault = tmp_path / "fault.json"
    _write_json(fault, _audit(1))

    result = build_evidence(
        routes,
        {"a": (holdout, holdout), "b": (holdout, holdout)},
        [fault],
        parity,
        health,
    )
    assert result["routes"]["tokyo_run1"]["passed"] is False
    assert result["component_promotion_ready"] is False
