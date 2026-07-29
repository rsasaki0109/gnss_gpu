from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from gnss_gpu_ros.lifecycle_core import (
    EventDisposition,
    LifecycleState,
    NavigationLifecycleCore,
    NavigationMode,
    SensorEvent,
    deterministic_bag_replay,
    replay_sha256,
)
from gnss_gpu_ros.replay_contract import evaluate_replay


def event(sensor, stamp, arrival=None, **payload):
    return SensorEvent.create(
        sensor,
        stamp,
        stamp if arrival is None else arrival,
        payload,
    )


def activated(**parameters):
    core = NavigationLifecycleCore()
    core.configure(parameters)
    core.activate()
    return core


def prime_required_inputs(core, now=1.0):
    core.ingest(event("imu", now, ax=0.0, ay=0.0, az=9.8))
    core.ingest(event("map", now, map_id="tokyo-lod2", ready=True))


def test_lifecycle_transitions_and_restart_are_explicit() -> None:
    core = NavigationLifecycleCore()
    assert core.state == LifecycleState.UNCONFIGURED
    core.configure()
    assert core.state == LifecycleState.INACTIVE
    core.activate()
    core.restart()
    assert core.state == LifecycleState.ACTIVE
    assert core.diagnostic(0.0).counters["restarts"] == 1
    core.deactivate()
    core.cleanup()
    assert core.state == LifecycleState.UNCONFIGURED
    core.shutdown()
    assert core.state == LifecycleState.FINALIZED


def test_error_transition_forces_safe_fallback() -> None:
    core = activated(require_imu=False, require_map=False)
    core.ingest(event("gnss", 1.0, latitude=35.0, longitude=139.0))
    core.fail("runtime_exception")
    assert core.state == LifecycleState.ERROR
    assert core.mode == NavigationMode.SAFE_FALLBACK


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("gnss_timeout_s", 0.0),
        ("imu_timeout_s", float("nan")),
        ("maximum_future_skew_s", -1.0),
        ("fallback_covariance_m2", float("inf")),
        ("require_map", 1),
    ],
)
def test_parameters_fail_closed(name, value) -> None:
    core = NavigationLifecycleCore()
    with pytest.raises(ValueError):
        core.configure({name: value})


def test_unknown_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown"):
        NavigationLifecycleCore().configure({"city": "tokyo"})


def test_all_topics_are_required_before_normal_output() -> None:
    core = activated()
    first = core.ingest(event("gnss", 1.0, latitude=35.0, longitude=139.0))
    assert first.mode == NavigationMode.SAFE_FALLBACK
    assert first.output["covariance_floor_m2"] == 10_000.0
    prime_required_inputs(core, 1.1)
    normal = core.ingest(event("gnss", 1.2, latitude=35.0, longitude=139.0))
    assert normal.mode == NavigationMode.NORMAL
    assert normal.output["covariance_floor_m2"] == 0.0


def test_future_skew_is_rejected_and_activates_safe_fallback() -> None:
    core = activated(maximum_future_skew_s=0.1)
    result = core.ingest(
        event("imu", 10.5, arrival=10.0, ax=0.0, ay=0.0, az=9.8)
    )
    assert result.disposition == EventDisposition.FUTURE_SKEW
    assert result.mode == NavigationMode.SAFE_FALLBACK
    assert core.watchdog(10.0).navigation_mode == NavigationMode.SAFE_FALLBACK
    assert "imu_integrity" in core.watchdog(10.0).reason


def test_duplicate_is_idempotent_and_conflict_fails_safe() -> None:
    core = activated()
    original = event("imu", 1.0, ax=1.0)
    assert core.ingest(original).disposition == EventDisposition.ACCEPTED
    assert core.ingest(original).disposition == EventDisposition.DUPLICATE
    conflict = core.ingest(event("imu", 1.0, ax=2.0))
    assert conflict.disposition == EventDisposition.CONFLICTING_DUPLICATE
    assert conflict.mode == NavigationMode.SAFE_FALLBACK
    assert "imu_integrity" in core.watchdog(1.0).reason
    core.ingest(event("imu", 2.0, ax=2.0))
    assert "imu_integrity" not in core.watchdog(2.0).reason


def test_out_of_order_input_is_dropped_without_rewinding_state() -> None:
    core = activated()
    core.ingest(event("map", 2.0, map_id="new"))
    result = core.ingest(event("map", 1.0, map_id="old"))
    assert result.disposition == EventDisposition.OUT_OF_ORDER
    assert result.mode == NavigationMode.SAFE_FALLBACK
    assert "map_integrity" in core.watchdog(2.0).reason
    assert core.diagnostic(2.0).sensor_age_s["map"] == 0.0


def test_watchdog_detects_missing_and_stale_streams() -> None:
    core = activated(gnss_timeout_s=1.0, imu_timeout_s=0.5, map_timeout_s=5.0)
    prime_required_inputs(core, 1.0)
    core.ingest(event("gnss", 1.0, latitude=35.0, longitude=139.0))
    assert core.watchdog(1.1).navigation_mode == NavigationMode.NORMAL
    stale = core.watchdog(2.1)
    assert stale.navigation_mode == NavigationMode.SAFE_FALLBACK
    assert "gnss_stale" in stale.reason
    assert "imu_stale" in stale.reason
    assert stale.counters["watchdog_trips"] == 1


def test_stale_transition_during_other_sensor_ingest_is_counted_once() -> None:
    core = activated(gnss_timeout_s=1.0, imu_timeout_s=0.5, map_timeout_s=5.0)
    prime_required_inputs(core, 1.0)
    core.ingest(event("gnss", 1.0, latitude=35.0, longitude=139.0))
    assert core.watchdog(1.1).navigation_mode == NavigationMode.NORMAL
    core.ingest(event("imu", 2.1, ax=0.0))
    first = core.watchdog(2.1)
    second = core.watchdog(2.2)
    assert first.navigation_mode == NavigationMode.SAFE_FALLBACK
    assert first.counters["watchdog_trips"] == 1
    assert second.counters["watchdog_trips"] == 1


def test_optional_imu_and_map_allow_gnss_only_operation() -> None:
    core = activated(require_imu=False, require_map=False)
    result = core.ingest(event("gnss", 1.0, latitude=35.0, longitude=139.0))
    assert result.mode == NavigationMode.NORMAL


def test_invalid_fix_does_not_replace_last_safe_fix() -> None:
    core = activated(require_imu=False, require_map=False)
    core.ingest(event("gnss", 1.0, latitude=35.0, longitude=139.0))
    bad = core.ingest(event("gnss", 2.0, latitude=95.0, longitude=139.0))
    assert bad.disposition == EventDisposition.INVALID
    assert core.diagnostic(2.0).last_safe_fix["latitude"] == 35.0


def test_no_fix_status_and_unready_map_are_rejected() -> None:
    core = activated()
    no_fix = core.ingest(
        event("gnss", 1.0, latitude=35.0, longitude=139.0, status=-1)
    )
    assert no_fix.disposition == EventDisposition.INVALID
    unready = core.ingest(event("map", 1.0, map_id="tokyo", ready=False))
    assert unready.disposition == EventDisposition.INVALID


def test_deterministic_replay_covers_missing_duplicate_reverse_and_restart() -> None:
    events = [
        event("imu", 1.0, ax=0.0),
        event("map", 1.0, map_id="map-a"),
        event("gnss", 1.0, latitude=35.0, longitude=139.0),
        event("gnss", 1.0, latitude=35.0, longitude=139.0),
        event("gnss", 0.5, arrival=1.1, latitude=34.0, longitude=138.0),
        event("gnss", 3.0, latitude=35.1, longitude=139.1),
    ]
    first = deterministic_bag_replay(events, restart_before_indices=[5])
    second = deterministic_bag_replay(events, restart_before_indices=[5])
    assert [asdict(step) for step in first] == [asdict(step) for step in second]
    assert replay_sha256(first) == replay_sha256(second)
    assert first[3].disposition == EventDisposition.DUPLICATE.value
    assert first[4].disposition == EventDisposition.OUT_OF_ORDER.value
    assert first[-1].mode == NavigationMode.SAFE_FALLBACK.value
    assert first[-1].diagnostic["counters"]["restarts"] == 1


def test_locked_phase6_replay_recomputes_exactly() -> None:
    repo_root = Path(__file__).parents[3]
    input_path = repo_root / "internal_docs" / "phase6_ros2_replay_input_2026_07_29.json"
    result_path = repo_root / "internal_docs" / "phase6_ros2_replay_result_2026_07_29.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    locked = json.loads(result_path.read_text(encoding="utf-8"))
    recomputed = evaluate_replay(payload)
    assert recomputed == locked
    assert recomputed["dispositions"] == {
        "accepted": 6,
        "conflicting_duplicate": 1,
        "duplicate": 1,
        "future_skew": 1,
        "out_of_order": 1,
    }
    assert recomputed["restart_count"] == 1
