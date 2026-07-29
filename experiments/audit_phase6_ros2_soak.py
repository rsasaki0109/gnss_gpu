#!/usr/bin/env python3
"""Run a deterministic long-duration soak of the ROS 2 lifecycle safety core."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ROS_PACKAGE = REPO_ROOT / "ros2" / "gnss_gpu_ros"
sys.path.insert(0, str(ROS_PACKAGE))

from gnss_gpu_ros.lifecycle_core import (  # noqa: E402
    EventDisposition,
    NavigationLifecycleCore,
    NavigationMode,
    SensorEvent,
)


SCHEMA = "gnss_gpu_ros_soak_audit_v1"


def _event(
    sensor: str,
    stamp_s: float,
    payload: dict[str, Any],
    *,
    arrival_s: float | None = None,
) -> SensorEvent:
    return SensorEvent.create(
        sensor,
        stamp_s,
        stamp_s if arrival_s is None else arrival_s,
        payload,
    )


def evaluate_soak(
    *,
    duration_s: float = 7_200.0,
    tick_s: float = 0.02,
) -> dict[str, Any]:
    if duration_s < 60.0:
        raise ValueError("duration_s must cover at least one minute")
    if tick_s <= 0.0 or tick_s > 0.1:
        raise ValueError("tick_s must be in (0, 0.1]")

    ticks = int(round(duration_s / tick_s))
    core = NavigationLifecycleCore()
    core.configure(
        {
            "gnss_timeout_s": 1.0,
            "imu_timeout_s": 0.25,
            "map_timeout_s": 2.0,
            "maximum_future_skew_s": 0.1,
            "require_imu": True,
            "require_map": True,
            "reject_out_of_order": True,
            "fallback_covariance_m2": 10_000.0,
            "diagnostics_period_s": 0.5,
        }
    )
    core.activate()

    disposition_counts = {item.value: 0 for item in EventDisposition}
    mode_counts = {item.value: 0 for item in NavigationMode}
    normal_recoveries = 0
    previous_mode = core.mode
    digest = hashlib.sha256()
    gnss_outage_start = ticks // 4
    gnss_outage_end = gnss_outage_start + max(2, int(round(2.0 / tick_s)))
    restart_tick = ticks // 2
    injected = {
        "duplicate": max(10, ticks // 16),
        "conflicting_duplicate": max(20, ticks // 12),
        "future_skew": max(30, ticks // 10),
        "out_of_order": max(40, ticks // 8),
    }

    last_imu: SensorEvent | None = None
    last_gnss: SensorEvent | None = None
    for tick in range(ticks + 1):
        now_s = tick * tick_s
        if tick == restart_tick:
            core.restart()

        events: list[SensorEvent] = []
        imu = _event(
            "imu",
            now_s,
            {"ax": 0.01, "ay": -0.02, "az": 9.8, "sequence": tick},
        )
        events.append(imu)
        last_imu = imu

        if tick % max(1, int(round(0.1 / tick_s))) == 0 and not (
            gnss_outage_start <= tick < gnss_outage_end
        ):
            gnss = _event(
                "gnss",
                now_s,
                {
                    "latitude": 35.68 + 1.0e-7 * (tick % 100),
                    "longitude": 139.77 + 1.0e-7 * (tick % 80),
                    "altitude": 40.0,
                    "status": 1,
                },
            )
            events.append(gnss)
            last_gnss = gnss
        if tick % max(1, int(round(1.0 / tick_s))) == 0:
            events.append(
                _event("map", now_s, {"map_id": "tokyo-lod2", "ready": True})
            )

        if tick == injected["duplicate"] and last_imu is not None:
            events.append(last_imu)
        elif tick == injected["conflicting_duplicate"] and last_gnss is not None:
            events.append(
                _event(
                    "gnss",
                    last_gnss.stamp_s,
                    {
                        "latitude": 35.7,
                        "longitude": 139.8,
                        "altitude": 40.0,
                        "status": 1,
                    },
                    arrival_s=now_s,
                )
            )
        elif tick == injected["future_skew"]:
            events.append(
                _event("imu", now_s + 1.0, {"ax": 1.0}, arrival_s=now_s)
            )
        elif tick == injected["out_of_order"] and last_gnss is not None:
            events.append(
                _event(
                    "gnss",
                    last_gnss.stamp_s - 1.0,
                    {
                        "latitude": 35.6,
                        "longitude": 139.7,
                        "altitude": 40.0,
                        "status": 1,
                    },
                    arrival_s=now_s,
                )
            )

        for current in events:
            result = core.ingest(current)
            disposition_counts[result.disposition.value] += 1
            digest.update(
                (
                    f"{tick}|{current.sensor.value}|{result.disposition.value}|"
                    f"{result.mode.value}|{result.reason}\n"
                ).encode("ascii")
            )
        diagnostic = core.watchdog(now_s)
        mode_counts[diagnostic.navigation_mode.value] += 1
        if (
            previous_mode == NavigationMode.SAFE_FALLBACK
            and diagnostic.navigation_mode == NavigationMode.NORMAL
        ):
            normal_recoveries += 1
        previous_mode = diagnostic.navigation_mode

    final = core.diagnostic(duration_s)
    required_dispositions = {
        EventDisposition.DUPLICATE.value: 1,
        EventDisposition.CONFLICTING_DUPLICATE.value: 1,
        EventDisposition.FUTURE_SKEW.value: 1,
        EventDisposition.OUT_OF_ORDER.value: 1,
    }
    checks = {
        "completed_duration": ticks * tick_s >= duration_s,
        "final_state_normal": final.navigation_mode == NavigationMode.NORMAL,
        "faults_injected_once": all(
            disposition_counts[name] == count
            for name, count in required_dispositions.items()
        ),
        "outage_watchdog_tripped": final.counters["watchdog_trips"] >= 1,
        "restarted_once": final.counters["restarts"] == 1,
        "recovered_after_faults": normal_recoveries >= 5,
        "bounded_core_state": (
            len(core._sensors) == 3  # noqa: SLF001 - deliberate state audit
            and len(core._integrity_faults) == 0  # noqa: SLF001
            and len(final.counters) == len(EventDisposition) + 2
        ),
    }
    return {
        "schema": SCHEMA,
        "simulated_duration_s": ticks * tick_s,
        "tick_s": tick_s,
        "ticks": ticks,
        "sensor_rates_hz": {"imu": 1.0 / tick_s, "gnss": 10.0, "map": 1.0},
        "fault_schedule": {
            **injected,
            "gnss_outage": [gnss_outage_start, gnss_outage_end],
            "restart": restart_tick,
        },
        "dispositions": disposition_counts,
        "mode_samples": mode_counts,
        "normal_recoveries": normal_recoveries,
        "final": {
            "lifecycle_state": final.lifecycle_state.value,
            "navigation_mode": final.navigation_mode.value,
            "reason": final.reason,
            "counters": dict(final.counters),
        },
        "state_digest_sha256": digest.hexdigest().upper(),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=7_200.0)
    parser.add_argument("--tick-s", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tracemalloc.start()
    started = time.perf_counter()
    result = evaluate_soak(duration_s=args.duration_s, tick_s=args.tick_s)
    elapsed_s = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["measurement"] = {
        "wall_time_s": elapsed_s,
        "peak_traced_memory_mib": peak_bytes / (1024.0 * 1024.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
