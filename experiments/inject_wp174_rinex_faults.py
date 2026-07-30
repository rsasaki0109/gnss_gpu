#!/usr/bin/env python3
"""Create deterministic raw-RINEX WP174 fault-injection replays."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable


FAULT_DURATIONS_S = {
    "outage": 5.0,
    "cycle_slip": 0.2,
    "satellite_loss": 3.0,
    "nlos": 5.0,
}
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Epoch:
    header: str
    tow: float
    records: tuple[str, ...]


def _gps_tow(header: str) -> float:
    fields = header[1:].split()
    if len(fields) < 6:
        raise ValueError(f"invalid RINEX epoch header: {header.rstrip()}")
    year, month, day, hour, minute = map(int, fields[:5])
    second = float(fields[5])
    whole = int(math.floor(second))
    stamp = datetime(
        year, month, day, hour, minute, whole, tzinfo=timezone.utc
    )
    gps_seconds = (stamp - GPS_EPOCH).total_seconds() + second - whole
    return gps_seconds % 604800.0


def _observation_types(header: list[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    current_system = ""
    for line in header:
        if "SYS / # / OBS TYPES" not in line:
            continue
        system = line[0].strip()
        if system:
            current_system = system
            output.setdefault(system, [])
        if current_system:
            output[current_system].extend(line[7:60].split())
    return output


def _parse(lines: list[str]) -> tuple[list[str], list[Epoch]]:
    try:
        header_end = next(
            index
            for index, line in enumerate(lines)
            if "END OF HEADER" in line
        )
    except StopIteration as error:
        raise ValueError("RINEX END OF HEADER not found") from error
    header = lines[: header_end + 1]
    epochs: list[Epoch] = []
    cursor = header_end + 1
    while cursor < len(lines):
        line = lines[cursor]
        if not line.startswith(">"):
            cursor += 1
            continue
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"invalid epoch header: {line.rstrip()}")
        count = int(fields[-1])
        records = tuple(lines[cursor + 1 : cursor + 1 + count])
        if len(records) != count:
            raise ValueError("truncated RINEX epoch")
        epochs.append(Epoch(line, _gps_tow(line), records))
        cursor += count + 1
    return header, epochs


def _replace_epoch_count(header: str, count: int) -> str:
    newline = "\n" if header.endswith("\n") else ""
    content = header.rstrip("\r\n")
    replaced = re.sub(r"\s+\d+\s*$", f" {count:2d}", content)
    return replaced + newline


def _selected_satellites(records: Iterable[str]) -> set[str]:
    satellites = sorted(
        record[:3] for record in records if len(record) >= 3
    )
    # Deterministic, constellation-mixed subset. Never use reference truth.
    selected = set(satellites[::2])
    if len(selected) == len(satellites) and len(satellites) > 1:
        selected.remove(satellites[-1])
    return selected


def _modify_observation_record(
    record: str,
    observation_types: list[str],
    fault: str,
) -> str:
    newline = "\n" if record.endswith("\n") else ""
    content = record.rstrip("\r\n")
    prefix = content[:3]
    fields = [
        content[start : start + 16].ljust(16)
        for start in range(3, len(content), 16)
    ]
    while len(fields) < len(observation_types):
        fields.append(" " * 16)
    for index, observation_type in enumerate(observation_types):
        raw = fields[index]
        try:
            value = float(raw[:14])
        except ValueError:
            continue
        if fault == "cycle_slip" and observation_type.startswith("L"):
            value += 100.0
            fields[index] = f"{value:14.3f}1{raw[15:16]}"
        elif fault == "nlos":
            if observation_type.startswith("C"):
                value += 30.0
                fields[index] = f"{value:14.3f}{raw[14:16]}"
            elif observation_type.startswith("L"):
                value += 50.0
                fields[index] = f"{value:14.3f}{raw[14:16]}"
            elif observation_type.startswith("S"):
                fields[index] = f"{15.0:14.3f}{raw[14:16]}"
    return prefix + "".join(fields).rstrip() + newline


def _anchored_starts(
    epochs: list[Epoch],
    fixed_tows: set[float],
    *,
    event_count: int,
    duration_s: float,
    anchor_streak_epochs: int,
    recovery_horizon_s: float,
) -> list[int]:
    if anchor_streak_epochs < 1:
        raise ValueError("anchor_streak_epochs must be positive")
    last_tow = epochs[-1].tow
    eligible = [
        index
        for index in range(anchor_streak_epochs, len(epochs))
        if epochs[index].tow + duration_s + recovery_horizon_s <=
        last_tow + 1e-6
        and all(
            round(epochs[prior].tow, 3) in fixed_tows
            for prior in range(index - anchor_streak_epochs, index)
        )
    ]
    if len(eligible) < event_count:
        raise ValueError(
            "too few baseline library-FIX anchored fault events"
        )
    spacing_s = duration_s + recovery_horizon_s
    # Earliest-finish greedy is maximal for equal-width spacing intervals.
    spaced: list[int] = []
    for index in eligible:
        if (
            not spaced or
            epochs[index].tow - epochs[spaced[-1]].tow >=
            spacing_s - 1e-6
        ):
            spaced.append(index)
    if len(spaced) < event_count:
        raise ValueError(
            "baseline FIX anchors cannot satisfy event spacing"
        )
    if len(spaced) == event_count:
        return spaced
    if event_count == 1:
        return [spaced[len(spaced) // 2]]
    return [
        spaced[
            round(
                event * (len(spaced) - 1) /
                (event_count - 1)
            )
        ]
        for event in range(event_count)
    ]


def inject(
    lines: list[str],
    *,
    fault: str,
    event_count: int = 8,
    duration_s: float | None = None,
    maximum_epochs: int | None = None,
    fixed_anchor_tows: set[float] | None = None,
    anchor_streak_epochs: int = 25,
    recovery_horizon_s: float = 10.0,
) -> tuple[list[str], dict]:
    if fault not in FAULT_DURATIONS_S:
        raise ValueError(f"unsupported fault: {fault}")
    if event_count < 1:
        raise ValueError("event_count must be positive")
    duration = (
        FAULT_DURATIONS_S[fault] if duration_s is None else duration_s
    )
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be positive and finite")
    header, epochs = _parse(lines)
    if maximum_epochs is not None:
        if maximum_epochs < 1:
            raise ValueError("maximum_epochs must be positive")
        epochs = epochs[:maximum_epochs]
    if len(epochs) < event_count * 3:
        raise ValueError("too few epochs for distributed fault events")
    types_by_system = _observation_types(header)
    if fixed_anchor_tows is None:
        starts = sorted(
            {
                min(
                    len(epochs) - 2,
                    max(
                        1,
                        round(
                            (event + 1) * len(epochs) /
                            (event_count + 1)
                        ),
                    ),
                )
                for event in range(event_count)
            }
        )
        selection = "uniform_epoch_index"
    else:
        starts = _anchored_starts(
            epochs,
            fixed_anchor_tows,
            event_count=event_count,
            duration_s=duration,
            anchor_streak_epochs=anchor_streak_epochs,
            recovery_horizon_s=recovery_horizon_s,
        )
        selection = "baseline_library_status4_anchor"
    event_specs = [
        {
            "start_index": index,
            "start_tow": epochs[index].tow,
            "end_tow": epochs[index].tow + duration,
            "selected_satellites": sorted(
                _selected_satellites(epochs[index].records)
            ),
        }
        for index in starts
    ]

    output = list(header)
    for epoch_index, epoch in enumerate(epochs):
        active = [
            event
            for event in event_specs
            if event["start_index"] <= epoch_index
            and epoch.tow <= event["end_tow"] + 1e-6
        ]
        if not active:
            output.append(epoch.header)
            output.extend(epoch.records)
            continue
        selected = set().union(
            *(set(event["selected_satellites"]) for event in active)
        )
        if fault == "outage":
            records: list[str] = []
        elif fault == "satellite_loss":
            records = [
                record
                for record in epoch.records
                if record[:3] not in selected
            ]
        else:
            # A cycle-slip impulse is exactly one epoch; NLOS spans duration.
            modify_now = fault == "nlos" or any(
                event["start_index"] == epoch_index for event in active
            )
            records = [
                (
                    _modify_observation_record(
                        record,
                        types_by_system.get(record[:1], []),
                        fault,
                    )
                    if modify_now and record[:3] in selected
                    else record
                )
                for record in epoch.records
            ]
        output.append(_replace_epoch_count(epoch.header, len(records)))
        output.extend(records)
    manifest = {
        "schema": "gnss_gpu_wp174_raw_rinex_fault_v1",
        "fault": fault,
        "duration_s": duration,
        "event_count": len(event_specs),
        "events": event_specs,
        "event_selection": selection,
        "anchor_streak_epochs": (
            anchor_streak_epochs
            if fixed_anchor_tows is not None
            else None
        ),
        "recovery_horizon_s": (
            recovery_horizon_s
            if fixed_anchor_tows is not None
            else None
        ),
        "truth_used_for_mutation": False,
    }
    return output, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fault", choices=tuple(FAULT_DURATIONS_S), required=True)
    parser.add_argument("--events", type=int, default=8)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument(
        "--anchor-pos",
        type=Path,
        help=(
            "truth-free baseline .pos; schedule only after consecutive "
            "gnssplusplus Status=4 epochs"
        ),
    )
    parser.add_argument("--anchor-streak-epochs", type=int, default=25)
    parser.add_argument("--recovery-horizon-s", type=float, default=10.0)
    parser.add_argument(
        "--maximum-epochs",
        type=int,
        help="optional deterministic smoke prefix; default writes full route",
    )
    args = parser.parse_args()
    source = args.input.read_text(encoding="ascii").splitlines(keepends=True)
    fixed_anchor_tows = None
    if args.anchor_pos is not None:
        fixed_anchor_tows = {
            round(float(fields[1]), 3)
            for line in args.anchor_pos.read_text(
                encoding="utf-8"
            ).splitlines()
            if line and not line.startswith("%")
            for fields in [line.split()]
            if len(fields) > 8 and int(fields[8]) == 4
        }
    mutated, manifest = inject(
        source,
        fault=args.fault,
        event_count=args.events,
        duration_s=args.duration_s,
        maximum_epochs=args.maximum_epochs,
        fixed_anchor_tows=fixed_anchor_tows,
        anchor_streak_epochs=args.anchor_streak_epochs,
        recovery_horizon_s=args.recovery_horizon_s,
    )
    manifest["input_sha256"] = hashlib.sha256(
        "".join(source).encode("ascii")
    ).hexdigest()
    manifest["output_sha256"] = hashlib.sha256(
        "".join(mutated).encode("ascii")
    ).hexdigest()
    if args.anchor_pos is not None:
        manifest["anchor_pos_sha256"] = hashlib.sha256(
            args.anchor_pos.read_bytes()
        ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(mutated), encoding="ascii", newline="")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
