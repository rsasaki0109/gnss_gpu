"""Command-line deterministic replay for lifecycle safety audit fixtures."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from gnss_gpu_ros.lifecycle_core import (
    SensorEvent,
    deterministic_bag_replay,
    replay_sha256,
)


INPUT_SCHEMA = "gnss_gpu_ros_replay_input_v1"
RESULT_SCHEMA = "gnss_gpu_ros_replay_result_v1"


def evaluate_replay(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"expected schema {INPUT_SCHEMA!r}")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("events must be a non-empty array")
    events = [
        SensorEvent.create(
            item["sensor"],
            item["stamp_s"],
            item["arrival_s"],
            item["payload"],
        )
        for item in raw_events
    ]
    restarts = payload.get("restart_before_indices", [])
    if not isinstance(restarts, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in restarts
    ):
        raise ValueError("restart_before_indices must contain integers")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be an object")
    steps = deterministic_bag_replay(events, parameters, restarts)
    dispositions: dict[str, int] = {}
    for step in steps:
        dispositions[step.disposition] = dispositions.get(step.disposition, 0) + 1
    return {
        "schema": RESULT_SCHEMA,
        "input_id": payload.get("id"),
        "event_count": len(steps),
        "restart_count": len(restarts),
        "dispositions": dict(sorted(dispositions.items())),
        "replay_sha256": replay_sha256(steps),
        "steps": [asdict(step) for step in steps],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = evaluate_replay(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
