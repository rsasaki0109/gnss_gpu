"""Deterministic safety core for the ROS 2 lifecycle navigation node.

This module has no ROS imports.  It owns lifecycle transitions, timestamp
integrity, watchdogs, and safe-fallback decisions so the same behavior can be
exercised in unit tests and deterministic bag replay.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, Mapping, Sequence


class LifecycleState(str, Enum):
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"
    ERROR = "error"


class NavigationMode(str, Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    SAFE_FALLBACK = "safe_fallback"


class SensorKind(str, Enum):
    GNSS = "gnss"
    IMU = "imu"
    MAP = "map"


class EventDisposition(str, Enum):
    ACCEPTED = "accepted"
    INACTIVE = "inactive"
    INVALID = "invalid"
    FUTURE_SKEW = "future_skew"
    OUT_OF_ORDER = "out_of_order"
    DUPLICATE = "duplicate"
    CONFLICTING_DUPLICATE = "conflicting_duplicate"


@dataclass(frozen=True)
class LifecycleParameters:
    gnss_timeout_s: float = 1.5
    imu_timeout_s: float = 0.25
    map_timeout_s: float = 10.0
    maximum_future_skew_s: float = 0.2
    require_imu: bool = True
    require_map: bool = True
    reject_out_of_order: bool = True
    fallback_covariance_m2: float = 10_000.0
    diagnostics_period_s: float = 0.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> LifecycleParameters:
        known = {item.name for item in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown parameters: {', '.join(sorted(unknown))}")
        instance = cls(**dict(values))
        instance.validate()
        return instance

    def validate(self) -> None:
        positive = (
            "gnss_timeout_s",
            "imu_timeout_s",
            "map_timeout_s",
            "maximum_future_skew_s",
            "fallback_covariance_m2",
            "diagnostics_period_s",
        )
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("require_imu", "require_map", "reject_out_of_order"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class SensorEvent:
    sensor: SensorKind
    stamp_s: float
    arrival_s: float
    payload: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        sensor: SensorKind | str,
        stamp_s: float,
        arrival_s: float,
        payload: Mapping[str, Any],
    ) -> SensorEvent:
        return cls(SensorKind(sensor), float(stamp_s), float(arrival_s), dict(payload))


@dataclass(frozen=True)
class IngestResult:
    sensor: SensorKind
    disposition: EventDisposition
    mode: NavigationMode
    reason: str
    output: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DiagnosticSnapshot:
    lifecycle_state: LifecycleState
    navigation_mode: NavigationMode
    reason: str
    now_s: float
    sensor_age_s: Mapping[str, float | None]
    counters: Mapping[str, int]
    last_safe_fix: Mapping[str, Any] | None


@dataclass(frozen=True)
class ReplayStep:
    event_index: int
    sensor: str
    stamp_s: float
    arrival_s: float
    disposition: str
    mode: str
    reason: str
    output: Mapping[str, Any] | None
    diagnostic: Mapping[str, Any]


@dataclass
class _SensorState:
    stamp_s: float | None = None
    arrival_s: float | None = None
    fingerprint: str | None = None
    payload: Mapping[str, Any] | None = None


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _finite_payload(payload: Mapping[str, Any]) -> bool:
    def valid(value: Any) -> bool:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return True
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if isinstance(value, Mapping):
            return all(isinstance(key, str) and valid(item) for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return all(valid(item) for item in value)
        return False

    return valid(payload)


class NavigationLifecycleCore:
    """Lifecycle state machine plus deterministic input-integrity policy."""

    def __init__(self) -> None:
        self.state = LifecycleState.UNCONFIGURED
        self.parameters: LifecycleParameters | None = None
        self.mode = NavigationMode.SAFE_FALLBACK
        self.reason = "unconfigured"
        self._sensors = {sensor: _SensorState() for sensor in SensorKind}
        self._integrity_faults: dict[SensorKind, str] = {}
        self._last_safe_fix: Mapping[str, Any] | None = None
        self._counters: dict[str, int] = {
            disposition.value: 0 for disposition in EventDisposition
        }
        self._counters.update({"watchdog_trips": 0, "restarts": 0})

    def configure(self, values: Mapping[str, Any] | None = None) -> None:
        if self.state not in {LifecycleState.UNCONFIGURED, LifecycleState.INACTIVE}:
            raise RuntimeError(f"cannot configure from {self.state.value}")
        self.parameters = LifecycleParameters.from_mapping(values or {})
        self._reset_stream_state(clear_fix=True)
        self.state = LifecycleState.INACTIVE
        self.mode = NavigationMode.SAFE_FALLBACK
        self.reason = "configured_inactive"

    def activate(self) -> None:
        if self.state != LifecycleState.INACTIVE:
            raise RuntimeError(f"cannot activate from {self.state.value}")
        self.state = LifecycleState.ACTIVE
        self.mode = NavigationMode.SAFE_FALLBACK
        self.reason = "awaiting_required_inputs"

    def deactivate(self) -> None:
        if self.state != LifecycleState.ACTIVE:
            raise RuntimeError(f"cannot deactivate from {self.state.value}")
        self.state = LifecycleState.INACTIVE
        self.mode = NavigationMode.SAFE_FALLBACK
        self.reason = "deactivated"

    def cleanup(self) -> None:
        if self.state != LifecycleState.INACTIVE:
            raise RuntimeError(f"cannot clean up from {self.state.value}")
        self._reset_stream_state(clear_fix=True)
        self.parameters = None
        self.state = LifecycleState.UNCONFIGURED
        self.reason = "cleaned_up"

    def shutdown(self) -> None:
        self._reset_stream_state(clear_fix=True)
        self.state = LifecycleState.FINALIZED
        self.mode = NavigationMode.SAFE_FALLBACK
        self.reason = "shutdown"

    def fail(self, reason: str) -> None:
        if not reason:
            raise ValueError("failure reason must be non-empty")
        self.state = LifecycleState.ERROR
        self.mode = NavigationMode.SAFE_FALLBACK
        self.reason = reason

    def restart(self) -> None:
        if self.parameters is None:
            raise RuntimeError("cannot restart before configuration")
        was_active = self.state == LifecycleState.ACTIVE
        self._reset_stream_state(clear_fix=True)
        self._counters["restarts"] += 1
        self.state = LifecycleState.ACTIVE if was_active else LifecycleState.INACTIVE
        self.mode = NavigationMode.SAFE_FALLBACK
        self.reason = "restarted_awaiting_inputs" if was_active else "restarted_inactive"

    def _reset_stream_state(self, *, clear_fix: bool) -> None:
        self._sensors = {sensor: _SensorState() for sensor in SensorKind}
        self._integrity_faults = {}
        if clear_fix:
            self._last_safe_fix = None

    def _record_rejection(
        self,
        event: SensorEvent,
        disposition: EventDisposition,
        reason: str,
    ) -> IngestResult:
        self._counters[disposition.value] += 1
        if disposition in {
            EventDisposition.INVALID,
            EventDisposition.FUTURE_SKEW,
            EventDisposition.CONFLICTING_DUPLICATE,
            EventDisposition.OUT_OF_ORDER,
        }:
            self._integrity_faults[event.sensor] = reason
            self.mode = NavigationMode.SAFE_FALLBACK
            self.reason = reason
        return IngestResult(event.sensor, disposition, self.mode, reason, None)

    def ingest(self, event: SensorEvent) -> IngestResult:
        if self.state != LifecycleState.ACTIVE:
            return self._record_rejection(
                event,
                EventDisposition.INACTIVE,
                "node_not_active",
            )
        assert self.parameters is not None
        if (
            not math.isfinite(event.stamp_s)
            or not math.isfinite(event.arrival_s)
            or not _finite_payload(event.payload)
        ):
            return self._record_rejection(
                event,
                EventDisposition.INVALID,
                "non_finite_or_unsupported_input",
            )
        if event.stamp_s > event.arrival_s + self.parameters.maximum_future_skew_s:
            return self._record_rejection(
                event,
                EventDisposition.FUTURE_SKEW,
                "timestamp_exceeds_future_skew",
            )
        if event.sensor == SensorKind.GNSS and not self._valid_fix(event.payload):
            return self._record_rejection(
                event, EventDisposition.INVALID, "invalid_gnss_fix"
            )
        if event.sensor == SensorKind.MAP and not self._valid_map(event.payload):
            return self._record_rejection(
                event, EventDisposition.INVALID, "invalid_map_context"
            )

        sensor_state = self._sensors[event.sensor]
        fingerprint = _canonical_fingerprint(event.payload)
        if sensor_state.stamp_s is not None:
            if event.stamp_s < sensor_state.stamp_s:
                if self.parameters.reject_out_of_order:
                    return self._record_rejection(
                        event,
                        EventDisposition.OUT_OF_ORDER,
                        "timestamp_moved_backwards",
                    )
            elif event.stamp_s == sensor_state.stamp_s:
                disposition = (
                    EventDisposition.DUPLICATE
                    if fingerprint == sensor_state.fingerprint
                    else EventDisposition.CONFLICTING_DUPLICATE
                )
                return self._record_rejection(
                    event,
                    disposition,
                    "duplicate_timestamp",
                )

        sensor_state.stamp_s = event.stamp_s
        sensor_state.arrival_s = event.arrival_s
        sensor_state.fingerprint = fingerprint
        sensor_state.payload = dict(event.payload)
        self._integrity_faults.pop(event.sensor, None)
        self._counters[EventDisposition.ACCEPTED.value] += 1

        output = None
        if event.sensor == SensorKind.GNSS:
            self._last_safe_fix = dict(event.payload)
            output = self._navigation_output(event.arrival_s)
        else:
            self._update_mode(event.arrival_s)
        return IngestResult(
            event.sensor,
            EventDisposition.ACCEPTED,
            self.mode,
            self.reason,
            output,
        )

    @staticmethod
    def _valid_fix(payload: Mapping[str, Any]) -> bool:
        try:
            latitude = float(payload["latitude"])
            longitude = float(payload["longitude"])
        except (KeyError, TypeError, ValueError):
            return False
        status = payload.get("status")
        status_valid = status is None or (
            isinstance(status, (int, float)) and float(status) >= 0.0
        )
        return (
            -90.0 <= latitude <= 90.0
            and -180.0 <= longitude <= 180.0
            and status_valid
        )

    @staticmethod
    def _valid_map(payload: Mapping[str, Any]) -> bool:
        map_id = payload.get("map_id")
        ready = payload.get("ready", True)
        return isinstance(map_id, str) and bool(map_id.strip()) and ready is True

    def _age(self, sensor: SensorKind, now_s: float) -> float | None:
        arrival = self._sensors[sensor].arrival_s
        return None if arrival is None else max(0.0, now_s - arrival)

    def _missing_reasons(self, now_s: float) -> list[str]:
        assert self.parameters is not None
        reasons = []
        limits = {
            SensorKind.GNSS: self.parameters.gnss_timeout_s,
            SensorKind.IMU: self.parameters.imu_timeout_s,
            SensorKind.MAP: self.parameters.map_timeout_s,
        }
        required = {
            SensorKind.GNSS: True,
            SensorKind.IMU: self.parameters.require_imu,
            SensorKind.MAP: self.parameters.require_map,
        }
        for sensor in SensorKind:
            if not required[sensor]:
                continue
            age = self._age(sensor, now_s)
            if age is None:
                reasons.append(f"{sensor.value}_missing")
            elif age > limits[sensor]:
                reasons.append(f"{sensor.value}_stale")
        reasons.extend(
            f"{sensor.value}_integrity:{reason}"
            for sensor, reason in sorted(
                self._integrity_faults.items(),
                key=lambda item: item[0].value,
            )
        )
        return reasons

    def _update_mode(self, now_s: float) -> None:
        before = self.mode
        missing = self._missing_reasons(now_s)
        if missing:
            self.mode = NavigationMode.SAFE_FALLBACK
            self.reason = ",".join(missing)
        else:
            self.mode = NavigationMode.NORMAL
            self.reason = "all_required_inputs_fresh"
        if (
            before != NavigationMode.SAFE_FALLBACK
            and self.mode == NavigationMode.SAFE_FALLBACK
        ):
            self._counters["watchdog_trips"] += 1

    def _navigation_output(self, now_s: float) -> Mapping[str, Any]:
        self._update_mode(now_s)
        assert self.parameters is not None
        fix = dict(self._last_safe_fix or {})
        fix["mode"] = self.mode.value
        fix["reason"] = self.reason
        fix["covariance_floor_m2"] = (
            0.0
            if self.mode == NavigationMode.NORMAL
            else self.parameters.fallback_covariance_m2
        )
        return fix

    def watchdog(self, now_s: float) -> DiagnosticSnapshot:
        if self.state != LifecycleState.ACTIVE:
            return self.diagnostic(now_s)
        self._update_mode(now_s)
        return self.diagnostic(now_s)

    def diagnostic(self, now_s: float) -> DiagnosticSnapshot:
        return DiagnosticSnapshot(
            lifecycle_state=self.state,
            navigation_mode=self.mode,
            reason=self.reason,
            now_s=float(now_s),
            sensor_age_s={
                sensor.value: self._age(sensor, now_s) for sensor in SensorKind
            },
            counters=dict(self._counters),
            last_safe_fix=dict(self._last_safe_fix) if self._last_safe_fix else None,
        )


def deterministic_bag_replay(
    events: Sequence[SensorEvent],
    parameters: Mapping[str, Any] | None = None,
    restart_before_indices: Sequence[int] = (),
) -> list[ReplayStep]:
    """Replay in recorded arrival order and return a canonical audit trail."""

    core = NavigationLifecycleCore()
    core.configure(parameters)
    core.activate()
    steps = []
    restart_set = set(restart_before_indices)
    if any(index < 0 or index >= len(events) for index in restart_set):
        raise ValueError("restart indices must refer to an event")
    for index, event in enumerate(events):
        if index in restart_set:
            core.restart()
        result = core.ingest(event)
        diagnostic = core.watchdog(event.arrival_s)
        steps.append(
            ReplayStep(
                event_index=index,
                sensor=event.sensor.value,
                stamp_s=event.stamp_s,
                arrival_s=event.arrival_s,
                disposition=result.disposition.value,
                mode=result.mode.value,
                reason=result.reason,
                output=dict(result.output) if result.output is not None else None,
                diagnostic=asdict(diagnostic),
            )
        )
    return steps


def replay_sha256(steps: Sequence[ReplayStep]) -> str:
    return hashlib.sha256(
        json.dumps(
            [asdict(step) for step in steps],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest().upper()
