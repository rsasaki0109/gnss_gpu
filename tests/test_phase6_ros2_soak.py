from __future__ import annotations

import json
from pathlib import Path

from experiments.audit_phase6_ros2_soak import evaluate_soak


REPO_ROOT = Path(__file__).parents[1]


def test_short_soak_recovers_all_injected_fault_classes() -> None:
    result = evaluate_soak(duration_s=120.0, tick_s=0.05)
    assert result["passed"] is True
    assert result["final"]["navigation_mode"] == "normal"
    assert result["final"]["counters"]["restarts"] == 1
    assert result["dispositions"]["duplicate"] == 1
    assert result["dispositions"]["conflicting_duplicate"] == 1
    assert result["dispositions"]["future_skew"] == 1
    assert result["dispositions"]["out_of_order"] == 1


def test_locked_two_hour_soak_recomputes_deterministically() -> None:
    locked = json.loads(
        (
            REPO_ROOT
            / "internal_docs"
            / "phase6_ros2_soak_result_2026_07_29.json"
        ).read_text(encoding="utf-8")
    )
    recomputed = evaluate_soak(
        duration_s=locked["simulated_duration_s"],
        tick_s=locked["tick_s"],
    )
    locked_without_measurement = {
        key: value for key, value in locked.items() if key != "measurement"
    }
    assert recomputed == locked_without_measurement
    assert recomputed["passed"] is True
